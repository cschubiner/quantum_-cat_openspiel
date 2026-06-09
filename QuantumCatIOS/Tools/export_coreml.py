#!/usr/bin/env python3
"""Best-effort Core ML export path for Quantum Cat PyTorch checkpoints.

This script intentionally lives beside the iOS app because the app bundles the
current best `.pt` artifacts, but native Core ML execution needs converted
`.mlpackage` files. It exports an action-conditioned model wrapper that accepts:

- `observation`: `[1, obs_size]`
- `action_features`: `[1, 1000, ACTION_FEATURE_SIZE]`

The resulting model returns policy logits, value estimates, paradox logits, and
action-value estimates when the checkpoint architecture supports them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

import pyspiel
from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import ACTION_FEATURE_SIZE, AZNet, load_model_payload


class CoreMLExportWrapper(torch.nn.Module):
  def __init__(self, model: AZNet, max_actions: int):
    super().__init__()
    self.model = model
    self.max_actions = max_actions

  def forward(self, observation, action_features):
    if self.model.arch == "action_mlp":
      return self._forward_action_mlp(observation, action_features)
    if self.model.arch == "action_setpool":
      return self._forward_action_setpool(observation, action_features)
    policy, value, _paradox, action_paradox, action_values = (
        self.model.forward_with_all_aux(observation, action_features)
    )
    return policy, value, action_paradox, action_values

  def _legal(self, action_embedding, action_features):
    legal = (action_features[..., 0] > 0.5).unsqueeze(-1)
    return legal

  def _pooled_action_mlp(self, action_embedding, action_features):
    legal = self._legal(action_embedding, action_features)
    legal_f = legal.to(action_embedding.dtype)
    legal_count = legal_f.sum(dim=1).clamp_min(1.0)
    mean_embedding = (action_embedding * legal_f).sum(dim=1) / legal_count
    masked_embedding = action_embedding.masked_fill(~legal, -1e9)
    max_embedding = masked_embedding.max(dim=1).values
    return mean_embedding, max_embedding

  def _setpool_action_embeddings(self, action_embedding, action_features):
    legal = self._legal(action_embedding, action_features)
    legal_f = legal.to(action_embedding.dtype)
    legal_count = legal_f.sum(dim=1).clamp_min(1.0)
    mean_embedding = (action_embedding * legal_f).sum(dim=1) / legal_count
    centered = (action_embedding - mean_embedding.unsqueeze(1)) * legal_f
    variance = (centered * centered).sum(dim=1) / legal_count
    std_embedding = torch.sqrt(variance.clamp_min(1e-8))
    max_embedding = action_embedding.masked_fill(~legal, -1e9).max(dim=1).values
    min_embedding = action_embedding.masked_fill(~legal, 1e9).min(dim=1).values
    return mean_embedding, max_embedding, min_embedding, std_embedding

  def _pair_features_action_mlp(self, state_embedding, action_features):
    action_embedding = self.model.action_encoder(action_features)
    state_expanded = state_embedding.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    state_for_action = self.model.state_action_projection(state_embedding)
    state_for_action = state_for_action.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    mean_embedding, max_embedding = self._pooled_action_mlp(
        action_embedding, action_features
    )
    mean_expanded = mean_embedding.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    max_expanded = max_embedding.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    return torch.cat([
        state_expanded,
        action_embedding,
        state_for_action * action_embedding,
        torch.abs(state_for_action - action_embedding),
        mean_expanded,
        max_expanded,
    ], dim=-1)

  def _forward_action_mlp(self, observation, action_features):
    state_embedding = self.model.body(observation)
    action_embedding = self.model.action_encoder(action_features)
    mean_embedding, max_embedding = self._pooled_action_mlp(
        action_embedding, action_features
    )
    value_features = torch.cat(
        [state_embedding, mean_embedding, max_embedding],
        dim=-1,
    )
    pair_features = self._pair_features_action_mlp(
        state_embedding, action_features
    )
    policy = self.model.policy(pair_features).squeeze(-1)
    value = torch.tanh(self.model.value(value_features))
    action_paradox = self.model.action_paradox(pair_features).squeeze(-1)
    action_values = torch.tanh(self.model.action_value(pair_features).squeeze(-1))
    return policy, value, action_paradox, action_values

  def _pair_features_action_setpool(self, state_embedding, action_features):
    action_embedding = self.model.action_encoder(action_features)
    state_expanded = state_embedding.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    state_for_action = self.model.state_action_projection(state_embedding)
    state_for_action = state_for_action.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    mean_embedding, max_embedding, min_embedding, std_embedding = (
        self._setpool_action_embeddings(action_embedding, action_features)
    )
    mean_expanded = mean_embedding.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    max_expanded = max_embedding.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    min_expanded = min_embedding.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    std_expanded = std_embedding.unsqueeze(1).expand(
        -1, self.max_actions, -1
    )
    centered_embedding = action_embedding - mean_expanded
    return torch.cat([
        state_expanded,
        action_embedding,
        state_for_action * action_embedding,
        torch.abs(state_for_action - action_embedding),
        centered_embedding,
        mean_expanded,
        max_expanded,
        min_expanded,
        std_expanded,
    ], dim=-1)

  def _forward_action_setpool(self, observation, action_features):
    state_embedding = self.model.body(observation)
    action_embedding = self.model.action_encoder(action_features)
    mean_embedding, max_embedding, min_embedding, std_embedding = (
        self._setpool_action_embeddings(action_embedding, action_features)
    )
    value_features = torch.cat(
        [
            state_embedding,
            mean_embedding,
            max_embedding,
            min_embedding,
            std_embedding,
        ],
        dim=-1,
    )
    pair_features = self._pair_features_action_setpool(
        state_embedding, action_features
    )
    policy = self.model.policy(pair_features).squeeze(-1)
    value = torch.tanh(self.model.value(value_features))
    action_paradox = self.model.action_paradox(pair_features).squeeze(-1)
    action_values = torch.tanh(self.model.action_value(pair_features).squeeze(-1))
    return policy, value, action_paradox, action_values


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--max-actions", type=int, default=1000)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  try:
    import coremltools as ct
  except ModuleNotFoundError as exc:
    raise SystemExit(
        "coremltools is not installed. Install it in .venv with "
        "`pip install coremltools` before running this exporter."
    ) from exc

  game = pyspiel.load_game(
      "python_quantum_cat", {"players": args.players, "start_player": 0}
  )
  checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
  checkpoint_args = checkpoint_payload.get("args", {})
  model_args = SimpleNamespace(
      players=int(checkpoint_args.get("players", args.players)),
      arch=checkpoint_args.get("arch", "action_mlp"),
      width=int(checkpoint_args.get("width", 256)),
      depth=int(checkpoint_args.get("depth", 3)),
      value_scale=float(checkpoint_args.get("value_scale", 20.0)),
      separate_action_value_encoder=bool(
          checkpoint_args.get("separate_action_value_encoder", False)
      ),
  )
  model, _, saved_args = load_model_payload(
      args.checkpoint, game, model_args, torch.device("cpu")
  )
  model.eval()
  wrapper = CoreMLExportWrapper(model, args.max_actions).eval()

  first_body_layer = next(
      module for module in model.body.modules() if isinstance(module, torch.nn.Linear)
  )
  obs_size = int(first_body_layer.in_features)
  traced = torch.jit.trace(
      wrapper,
      (
          torch.zeros((1, obs_size), dtype=torch.float32),
          torch.zeros((1, args.max_actions, ACTION_FEATURE_SIZE), dtype=torch.float32),
      ),
      strict=False,
  )

  mlmodel = ct.convert(
      traced,
      convert_to="mlprogram",
      inputs=[
          ct.TensorType(name="observation", shape=(1, obs_size)),
          ct.TensorType(
              name="action_features",
              shape=(1, args.max_actions, ACTION_FEATURE_SIZE),
          ),
      ],
      compute_units=ct.ComputeUnit.ALL,
      minimum_deployment_target=ct.target.iOS17,
  )
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  mlmodel.save(str(out))
  print(f"wrote {out}")
  print(
      f"saved_args_arch={saved_args.get('arch')} obs_size={obs_size} "
      f"action_feature_size={ACTION_FEATURE_SIZE} max_actions={args.max_actions}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
