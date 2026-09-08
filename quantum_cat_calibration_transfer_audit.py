#!/usr/bin/env python3
"""Train/evaluate an abstaining override model on calibration rows.

The calibration corpus is intentionally different from accepted replay: it
contains positives and audited failures. This script asks whether a lightweight
action-conditioned classifier can identify the rows where overriding Run119 is
worthwhile, while abstaining on false positives.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_alphazero_torch import (
    AZNet,
    adapt_action_features,
    adapt_observation,
    initialize_missing_action_paradox_stack_from_policy,
    initialize_missing_action_value_stack_from_policy,
    load_compatible_state_dict,
    load_replay,
    model_input_size,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--calibration-replay", action="append", required=True)
  parser.add_argument(
      "--metadata",
      action="append",
      default=[],
      help=(
          "Optional metadata JSON path. Defaults to each replay path plus "
          "'.metadata.json'."
      ),
  )
  parser.add_argument(
      "--validation-replay",
      action="append",
      default=[],
      help=(
          "Optional held-out calibration replay(s). When provided, training "
          "uses --calibration-replay only and validation uses these replay(s) "
          "only."
      ),
  )
  parser.add_argument(
      "--validation-metadata",
      action="append",
      default=[],
      help=(
          "Optional metadata JSON path(s) for --validation-replay. Defaults "
          "to each validation replay path plus '.metadata.json'."
      ),
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--seed", type=int, default=20262084)
  parser.add_argument("--validation-fraction", type=float, default=0.35)
  parser.add_argument(
      "--validation-split",
      choices=("row", "group", "source_group"),
      default="row",
      help=(
          "row preserves the historical label-stratified row split. group keeps "
          "metadata groups together. source_group also stratifies groups within "
          "each replay source so one source cannot dominate validation."
      ),
  )
  parser.add_argument(
      "--require-explicit-source-disjoint",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Fail closed unless explicit validation replay proposal sources are "
          "disjoint from every calibration replay proposal source."
      ),
  )
  parser.add_argument(
      "--require-each-validation-replay-pass",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Require one common threshold to pass independently on every "
          "explicit validation replay, not only on their aggregate."
      ),
  )
  parser.add_argument("--epochs", type=int, default=400)
  parser.add_argument("--lr", type=float, default=0.003)
  parser.add_argument("--hidden", type=int, default=96)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument(
      "--pos-weight-mode",
      choices=("balanced", "none"),
      default="balanced",
      help=(
          "Positive-class weighting for BCE. balanced preserves the historical "
          "recall-seeking objective; none is usually better for precision-first "
          "abstainers."
      ),
  )
  parser.add_argument(
      "--positive-loss-weight",
      type=float,
      default=1.0,
      help="Multiplier for positive training rows after class weighting.",
  )
  parser.add_argument(
      "--negative-loss-weight",
      type=float,
      default=1.0,
      help="Multiplier for negative training rows after class weighting.",
  )
  parser.add_argument(
      "--false-positive-loss-weight",
      type=float,
      default=1.0,
      help=(
          "Additional multiplier for rows that are not POSITIVE under the "
          "positive gate. This is the main precision-first abstainer knob."
      ),
  )
  parser.add_argument(
      "--near-positive-loss-weight",
      type=float,
      default=1.0,
      help=(
          "Additional multiplier for NEAR_POSITIVE rows. In strict-positive "
          "training these are useful hard negatives; in curriculum training "
          "they can be upweighted positives."
      ),
  )
  parser.add_argument(
      "--focal-gamma",
      type=float,
      default=0.0,
      help=(
          "When >0, apply binary focal loss. This can reduce easy-example "
          "dominance in small hard-negative corpora."
      ),
  )
  parser.add_argument(
      "--label-margin-feature",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Include cheap label margin as an input. Disable for deployment-like "
          "audits because live play does not have teacher rollout margins."
      ),
  )
  parser.add_argument(
      "--observation-feature-mode",
      choices=("none", "summary", "raw", "raw_summary"),
      default="none",
      help=(
          "Append deployable features from the acting player's observation. "
          "This uses the same public/own-info tensor available to the live bot."
      ),
  )
  parser.add_argument(
      "--policy-context-mode",
      choices=("none", "summary"),
      default="none",
      help=(
          "Append deployable features from the current champion policy over "
          "the legal action set. Requires --policy-context-checkpoint for "
          "training-time calibration rows."
      ),
  )
  parser.add_argument(
      "--game-context-mode",
      choices=("none", "onehot"),
      default="none",
      help=(
          "Append deployable trick, match-round, and acting-seat context. "
          "onehot uses fixed-size buckets and normalized scalar copies."
      ),
  )
  parser.add_argument(
      "--policy-context-checkpoint",
      default="",
      help=(
          "Checkpoint used to reproduce the live policy distribution for "
          "--policy-context-mode. Usually az_runs/best_3p_policy_checkpoint.pt."
      ),
  )
  parser.add_argument("--policy-context-device", default="cpu")
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument(
      "--thresholds",
      default="0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95",
  )
  parser.add_argument("--target-coverage-min", type=float, default=0.05)
  parser.add_argument("--target-coverage-max", type=float, default=0.15)
  parser.add_argument(
      "--target-min-overrides",
      type=int,
      default=1,
      help=(
          "Minimum number of threshold overrides required before an offline "
          "gate can pass. Raise this for explicit holdout diagnostics so one "
          "lucky row cannot certify a deployable threshold."
      ),
  )
  parser.add_argument(
      "--target-min-true-margin-positive-overrides",
      type=int,
      default=1,
      help=(
          "Minimum number of overrides with positive audited true margin "
          "required before an offline gate can pass."
      ),
  )
  parser.add_argument("--target-precision-min", type=float, default=0.70)
  parser.add_argument("--target-mean-margin-min", type=float, default=0.50)
  parser.add_argument("--target-false-positive-rate-max", type=float, default=0.15)
  parser.add_argument("--target-paradox-delta-max", type=float, default=0.0)
  parser.add_argument(
      "--positive-label-margin-min",
      type=float,
      default=0.0,
      help=(
          "Only train the override classifier's positive class on audited rows "
          "with at least this true margin. Evaluation still reports precision "
          "against any positive true margin."
      ),
  )
  parser.add_argument(
      "--train-positive-classes",
      default="POSITIVE",
      help=(
          "Comma-separated calibration classes treated as positive training "
          "labels. Default preserves strict deployable POSITIVE-only training; "
          "curriculum diagnostics can use POSITIVE,NEAR_POSITIVE."
      ),
  )
  parser.add_argument(
      "--train-risk-veto",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Train a second deployable p_safe head from audited paradox deltas. "
          "Live improve-graft can use it as a hard override veto."
      ),
  )
  parser.add_argument(
      "--risk-label-mode",
      choices=("paradox_safe", "safe_improve"),
      default="paradox_safe",
      help=(
          "Target for the deployable risk/veto head. paradox_safe preserves "
          "the historical behavior of predicting only whether paradox_delta "
          "is acceptable. safe_improve predicts whether the override is both "
          "a trained positive and paradox-safe, which makes the live risk "
          "threshold a true safe-improvement veto."
      ),
  )
  parser.add_argument(
      "--risk-thresholds",
      default="0.5,0.7,0.85,0.9,0.95,0.97,0.99",
      help="Comma-separated p_safe thresholds for combined offline reports.",
  )
  return parser.parse_args()


def _class_set(spec):
  classes = {
      item.strip()
      for item in str(spec or "").split(",")
      if item.strip()
  }
  if not classes:
    raise ValueError("class set cannot be empty")
  return classes


class ImproveMLP(nn.Module):
  def __init__(self, in_size, hidden):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_size, hidden),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )

  def forward(self, x):
    return self.net(x).squeeze(-1)


def _metadata_path_for(replay_path):
  replay = Path(replay_path)
  return replay.with_suffix(replay.suffix + ".metadata.json")


def _load_metadata(path):
  data = json.loads(Path(path).read_text())
  rows = data if isinstance(data, list) else data.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"Metadata rows not found in {path}")
  return rows


def _metadata_source_namespace(path):
  data = json.loads(Path(path).read_text())
  if isinstance(data, list):
    return ""
  explicit = str(data.get("proposal_source_id", "") or "").strip()
  if explicit:
    return explicit
  if data.get("seed") is not None:
    return f"seed={int(data['seed'])}"
  return ""


def _metadata_group_key(meta, replay_path, row_index):
  group = meta.get("group")
  if isinstance(group, dict):
    source = str(group.get("proposal_source", ""))
    match = group.get("match_index")
    if match is not None:
      return "|".join([
          "match",
          source,
          str(match),
      ])
    pieces = []
    for key in (
        "initial_start_player",
        "round_index",
        "round_start_player",
        "match_round",
    ):
      if key in group:
        pieces.append(f"{key}={group[key]}")
    if pieces:
      return "|".join(["group", source] + pieces)
  return f"row|{Path(replay_path).name}|{row_index}"


def _metadata_source_key(meta, replay_path, source_namespace=""):
  group = meta.get("group")
  if isinstance(group, dict):
    source = str(group.get("proposal_source", "")).strip()
    if source:
      return "|".join(
          part for part in (str(source_namespace).strip(), source) if part
      )
  return f"replay|{Path(replay_path).name}"


def candidate_calibration_metadata_rows(meta):
  """Expand a state-level proposal audit into candidate-level examples."""
  evaluations = list(meta.get("candidate_evaluations") or [])
  if not evaluations:
    return [dict(meta)]
  policy_action = int(meta["policy_best_action"])
  expanded = []
  for candidate_rank, evaluation in enumerate(evaluations):
    candidate_action = int(evaluation["action"])
    if candidate_action == policy_action:
      continue
    candidate_class = str(
        evaluation.get("calibration_class", "UNKNOWN")
    )
    candidate_meta = dict(meta)
    candidate_meta.update({
        "label_best_action": candidate_action,
        "candidate_rank": int(candidate_rank),
        "calibration_class": candidate_class,
        "audit_margin_vs_policy": float(
            evaluation.get("audit_margin_vs_policy") or 0.0
        ),
        "audit_paradox_delta_vs_policy": float(
            evaluation.get("audit_paradox_delta_vs_policy") or 0.0
        ),
        "deployment_target_action": (
            candidate_action
            if candidate_class == "POSITIVE"
            else policy_action
        ),
        "false_positive_for_positive_gate": candidate_class != "POSITIVE",
    })
    for key in (
        "proposal_improve_prob",
        "proposal_safe_prob",
        "proposal_policy_prob",
    ):
      if evaluation.get(key) is not None:
        candidate_meta[key] = float(evaluation[key])
    expanded.append(candidate_meta)
  return expanded or [dict(meta)]


def explicit_source_overlap(examples, train_indices, validation_indices):
  """Returns proposal sources shared by explicit train and validation rows."""
  train_sources = {examples[idx]["source_key"] for idx in train_indices}
  validation_sources = {
      examples[idx]["source_key"] for idx in validation_indices
  }
  return train_sources, validation_sources, sorted(
      train_sources & validation_sources
  )


def observation_feature_vector(observation, mode="none"):
  mode = str(mode or "none")
  if mode == "none" or observation is None:
    return np.zeros(0, dtype=np.float32)
  obs = np.asarray(observation, dtype=np.float32).reshape(-1)
  if obs.size == 0:
    raw = np.zeros(0, dtype=np.float32)
  else:
    raw = obs.astype(np.float32)
  if raw.size == 0:
    summary = np.zeros(8, dtype=np.float32)
  else:
    summary = np.array([
        float(raw.size),
        float(np.mean(raw)),
        float(np.std(raw)),
        float(np.min(raw)),
        float(np.max(raw)),
        float(np.sum(raw > 0.5)),
        float(np.sum(raw < -0.5)),
        float(np.linalg.norm(raw) / max(1.0, np.sqrt(float(raw.size)))),
    ], dtype=np.float32)
  if mode == "summary":
    return summary
  if mode == "raw":
    return raw
  if mode == "raw_summary":
    return np.concatenate([raw, summary]).astype(np.float32)
  raise ValueError(f"Unknown observation feature mode: {mode}")


def game_context_feature_vector(
    trick_number=-1,
    match_round=-1,
    player=-1,
    players=3,
    mode="none",
):
  mode = str(mode or "none")
  if mode == "none":
    return np.zeros(0, dtype=np.float32)
  if mode != "onehot":
    raise ValueError(f"Unknown game context mode: {mode}")
  trick = int(trick_number)
  match = int(match_round)
  seat = int(player)
  player_count = max(2, int(players))
  scalars = np.array([
      float(trick) / 9.0 if trick >= 0 else -1.0,
      float(match) / max(1.0, float(player_count - 1)) if match >= 0 else -1.0,
      float(seat) / max(1.0, float(player_count - 1)) if seat >= 0 else -1.0,
  ], dtype=np.float32)
  onehot = np.zeros(20, dtype=np.float32)
  if 0 <= trick < 10:
    onehot[trick] = 1.0
  if 0 <= match < 5:
    onehot[10 + match] = 1.0
  if 0 <= seat < 5:
    onehot[15 + seat] = 1.0
  return np.concatenate([scalars, onehot]).astype(np.float32)


def _resolve_device(spec):
  spec = str(spec or "cpu")
  if spec == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(spec)


def _checkpoint_obs_size(state_dict):
  for key in ("body.0.weight", "body.input.0.weight"):
    value = state_dict.get(key)
    if value is not None and len(value.shape) == 2:
      return int(value.shape[1])
  for key, value in state_dict.items():
    if key.startswith("body.") and key.endswith(".weight") and len(value.shape) == 2:
      return int(value.shape[1])
  raise ValueError("Could not infer checkpoint observation size")


def _load_policy_context_model(args, sample_row):
  mode = str(getattr(args, "policy_context_mode", "none") or "none")
  if mode == "none":
    return None, None
  checkpoint = str(getattr(args, "policy_context_checkpoint", "") or "")
  if not checkpoint:
    raise ValueError("--policy-context-checkpoint is required for policy context")
  device = _resolve_device(getattr(args, "policy_context_device", "cpu"))
  payload = torch.load(checkpoint, map_location=device, weights_only=False)
  saved_args = payload.get("args", {})
  state_dict = payload["model"]
  obs_size = _checkpoint_obs_size(state_dict)
  num_actions = len(sample_row[1])
  model = AZNet(
      obs_size,
      num_actions,
      int(saved_args.get("players", getattr(args, "players", 3))),
      int(saved_args.get("width", 256)),
      int(saved_args.get("depth", 3)),
      str(saved_args.get("arch", "mlp")),
      bool(saved_args.get("separate_action_value_encoder", False)),
      bool(saved_args.get("separate_action_paradox_encoder", False)),
  ).to(device)
  load_compatible_state_dict(model, state_dict)
  initialize_missing_action_value_stack_from_policy(model, state_dict)
  initialize_missing_action_paradox_stack_from_policy(model, state_dict)
  model.eval()
  return model, device


def _policy_probs_from_row(row, model, device):
  if model is None:
    return None
  legal_mask = np.asarray(row[1], dtype=np.float32) > 0.5
  obs_np = adapt_observation(
      np.asarray(row[0], dtype=np.float32).reshape(-1),
      model_input_size(model),
  )
  action_features = adapt_action_features(
      row[11] if len(row) > 11 else None, len(row[1])
  )
  with torch.no_grad():
    obs_t = torch.tensor(
        obs_np, dtype=torch.float32, device=device
    ).unsqueeze(0)
    action_t = torch.tensor(
        action_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    logits, _ = model(obs_t, action_t)
  logits = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
  probs = np.zeros(len(row[1]), dtype=np.float32)
  legal_indices = np.flatnonzero(legal_mask)
  if len(legal_indices) == 0:
    return probs
  legal_logits = logits[legal_indices]
  max_logit = float(np.max(legal_logits))
  exp = np.exp(legal_logits - max_logit).astype(np.float32)
  total = float(np.sum(exp))
  if total <= 0 or not np.isfinite(total):
    probs[legal_indices] = 1.0 / float(len(legal_indices))
  else:
    probs[legal_indices] = exp / total
  return probs


def policy_context_feature_vector(
    policy,
    legal_mask,
    candidate_action,
    policy_action,
    mode="none",
):
  mode = str(mode or "none")
  if mode == "none" or policy is None:
    return np.zeros(0, dtype=np.float32)
  if mode != "summary":
    raise ValueError(f"Unknown policy context mode: {mode}")
  policy = np.asarray(policy, dtype=np.float32).reshape(-1)
  legal_mask = np.asarray(legal_mask, dtype=np.float32) > 0.5
  legal_indices = np.flatnonzero(legal_mask)
  eps = 1e-8
  if len(legal_indices) == 0:
    return np.zeros(13, dtype=np.float32)
  legal_probs = policy[legal_indices].astype(np.float32)
  legal_total = float(np.sum(legal_probs))
  if legal_total <= 0 or not np.isfinite(legal_total):
    legal_probs = np.full(len(legal_indices), 1.0 / len(legal_indices), dtype=np.float32)
  else:
    legal_probs = legal_probs / legal_total
  normalized_policy = np.zeros_like(policy, dtype=np.float32)
  normalized_policy[legal_indices] = legal_probs
  order = np.argsort(-legal_probs)
  rank_by_action = {
      int(legal_indices[rank_pos]): int(rank_pos)
      for rank_pos in range(len(order))
  }
  # np.argsort returns positions into legal_indices; remap in probability order.
  rank_by_action = {
      int(legal_indices[int(pos)]): int(rank)
      for rank, pos in enumerate(order)
  }
  sorted_probs = legal_probs[order]
  top_prob = float(sorted_probs[0])
  second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
  entropy = float(-np.sum(legal_probs * np.log(np.maximum(legal_probs, eps))))
  entropy_norm = entropy / max(eps, float(np.log(max(2, len(legal_probs)))))
  candidate_action = int(candidate_action)
  policy_action = int(policy_action)
  candidate_prob = float(normalized_policy[candidate_action])
  policy_prob = float(normalized_policy[policy_action])
  rank_den = max(1, len(legal_indices) - 1)
  candidate_rank = float(rank_by_action.get(candidate_action, len(legal_indices))) / rank_den
  policy_rank = float(rank_by_action.get(policy_action, len(legal_indices))) / rank_den
  return np.array([
      candidate_prob,
      policy_prob,
      candidate_prob - policy_prob,
      float(np.log(candidate_prob + eps)),
      float(np.log(policy_prob + eps)),
      float(np.log(candidate_prob + eps) - np.log(policy_prob + eps)),
      entropy_norm,
      top_prob,
      second_prob,
      top_prob - second_prob,
      candidate_rank,
      policy_rank,
      float(candidate_rank == 0.0),
  ], dtype=np.float32)


def pair_feature_vector_from_matrix(
    action_features,
    legal_mask,
    candidate_action,
    policy_action,
    *,
    target_action=None,
    label_margin=0.0,
    legal_count=None,
    candidate_count=0.0,
    include_label_margin=True,
    observation=None,
    observation_feature_mode="none",
    policy_context=None,
    policy_context_mode="none",
    game_context_mode="none",
    trick_number=-1,
    match_round=-1,
    player=-1,
    players=3,
    include_target_match_features=False,
):
  action_features = np.asarray(action_features, dtype=np.float32)
  legal_mask = np.asarray(legal_mask, dtype=np.float32) > 0.5
  candidate_action = int(candidate_action)
  policy_action = int(policy_action)
  target_action = candidate_action if target_action is None else int(target_action)
  if candidate_action < 0 or candidate_action >= action_features.shape[0]:
    raise ValueError(f"candidate action out of range: {candidate_action}")
  if policy_action < 0 or policy_action >= action_features.shape[0]:
    raise ValueError(f"policy action out of range: {policy_action}")
  legal_features = action_features[legal_mask]
  if len(legal_features) == 0:
    legal_mean = np.zeros(action_features.shape[1], dtype=np.float32)
    legal_max = np.zeros(action_features.shape[1], dtype=np.float32)
    legal_min = np.zeros(action_features.shape[1], dtype=np.float32)
    legal_std = np.ones(action_features.shape[1], dtype=np.float32)
  else:
    legal_mean = legal_features.mean(axis=0)
    legal_max = legal_features.max(axis=0)
    legal_min = legal_features.min(axis=0)
    legal_std = legal_features.std(axis=0)
    legal_std = np.where(legal_std < 1e-6, 1.0, legal_std).astype(np.float32)
  label_features = action_features[candidate_action]
  policy_features = action_features[policy_action]
  label_z = (label_features - legal_mean) / legal_std
  policy_z = (policy_features - legal_mean) / legal_std
  scalar_features = np.array([
      float(label_margin) if include_label_margin else 0.0,
      float(legal_count if legal_count is not None else int(np.sum(legal_mask))),
      float(candidate_count),
  ], dtype=np.float32)
  if include_target_match_features:
    scalar_features = np.concatenate([
        scalar_features,
        np.array([
            float(candidate_action == target_action),
            float(policy_action == target_action),
        ], dtype=np.float32),
    ])
  obs_features = observation_feature_vector(
      observation, mode=observation_feature_mode
  )
  policy_features_extra = policy_context_feature_vector(
      policy_context,
      legal_mask,
      candidate_action,
      policy_action,
      mode=policy_context_mode,
  )
  game_context_features = game_context_feature_vector(
      trick_number=trick_number,
      match_round=match_round,
      player=player,
      players=players,
      mode=game_context_mode,
  )
  return np.concatenate([
      scalar_features,
      label_features,
      policy_features,
      label_features - policy_features,
      label_features * policy_features,
      legal_mean,
      legal_max,
      legal_min,
      legal_std,
      label_z,
      policy_z,
      label_z - policy_z,
      obs_features,
      policy_features_extra,
      game_context_features,
  ]).astype(np.float32)


def _pair_feature_vector(
    row,
    meta,
    include_label_margin=True,
    observation_feature_mode="none",
    policy_context=None,
    policy_context_mode="none",
    game_context_mode="none",
    players=3,
):
  action_features = adapt_action_features(row[11] if len(row) > 11 else None, len(row[1]))
  legal_mask = np.array(row[1], dtype=np.float32) > 0.5
  label_action = int(meta["label_best_action"])
  policy_action = int(meta["policy_best_action"])
  target_action = int(row[5])
  return pair_feature_vector_from_matrix(
      action_features,
      legal_mask,
      label_action,
      policy_action,
      target_action=target_action,
      label_margin=float(meta.get("label_margin_vs_policy") or 0.0),
      legal_count=float(meta.get("legal_count") or int(np.sum(legal_mask))),
      candidate_count=float(meta.get("candidate_count") or 0.0),
      include_label_margin=include_label_margin,
      observation=row[0] if len(row) > 0 else None,
      observation_feature_mode=observation_feature_mode,
      policy_context=policy_context,
      policy_context_mode=policy_context_mode,
      game_context_mode=game_context_mode,
      trick_number=int(meta.get("trick_number", -1)),
      match_round=int(meta.get("match_round", -1)),
      player=int(meta.get("player", -1)),
      players=int(players),
      include_target_match_features=False,
  )


def _load_examples(args, replay_paths=None, metadata_paths=None):
  replay_paths = (
      [str(path) for path in args.calibration_replay]
      if replay_paths is None else
      [str(path) for path in replay_paths]
  )
  metadata_paths = (
      [str(path) for path in args.metadata]
      if metadata_paths is None else
      [str(path) for path in metadata_paths]
  )
  if metadata_paths and len(metadata_paths) != len(replay_paths):
    raise ValueError("metadata count must match replay count")
  if not metadata_paths:
    metadata_paths = [str(_metadata_path_for(path)) for path in replay_paths]

  examples = []
  input_counts = {}
  policy_context_model = None
  policy_context_device = None
  train_positive_classes = _class_set(args.train_positive_classes)
  for replay_path, metadata_path in zip(replay_paths, metadata_paths):
    rows = load_replay(replay_path)
    metadata = _load_metadata(metadata_path)
    source_namespace = _metadata_source_namespace(metadata_path)
    if (
        str(args.policy_context_mode) != "none"
        and policy_context_model is None
        and rows
    ):
      policy_context_model, policy_context_device = _load_policy_context_model(
          args, rows[0]
      )
    input_counts[str(replay_path)] = {
        "replay_rows": int(len(rows)),
        "metadata_rows": int(len(metadata)),
        "metadata": str(metadata_path),
    }
    if len(rows) != len(metadata):
      raise ValueError(
          f"Replay/metadata row mismatch for {replay_path}: "
          f"{len(rows)} vs {len(metadata)}"
      )
    candidate_example_count = 0
    for idx, (row, meta) in enumerate(zip(rows, metadata)):
      source_target_action = int(meta["deployment_target_action"])
      replay_action = int(row[5]) if len(row) > 5 else -1
      if replay_action != source_target_action:
        raise ValueError(
            f"Replay target mismatch at {replay_path}:{idx}: "
            f"{replay_action} vs {source_target_action}"
        )
      policy_context = _policy_probs_from_row(
          row, policy_context_model, policy_context_device
      )
      for candidate_meta in candidate_calibration_metadata_rows(meta):
        row_class = str(candidate_meta.get("calibration_class", "UNKNOWN"))
        if row_class == "UNAUDITED":
          continue
        policy_action = int(candidate_meta["policy_best_action"])
        label_action = int(candidate_meta["label_best_action"])
        target_action = int(candidate_meta["deployment_target_action"])
        audit_margin = float(
            candidate_meta.get("audit_margin_vs_policy") or 0.0
        )
        train_positive = (
            row_class in train_positive_classes
            and audit_margin >= float(args.positive_label_margin_min)
        )
        examples.append({
            "input": str(replay_path),
            "row_index": int(idx),
            "candidate_rank": int(candidate_meta.get("candidate_rank", 0)),
            "group_key": _metadata_group_key(meta, replay_path, idx),
            "source_key": _metadata_source_key(
                meta, replay_path, source_namespace
            ),
            "features": _pair_feature_vector(
                row,
                candidate_meta,
                include_label_margin=bool(args.label_margin_feature),
                observation_feature_mode=str(args.observation_feature_mode),
                policy_context=policy_context,
                policy_context_mode=str(args.policy_context_mode),
                game_context_mode=str(args.game_context_mode),
                players=int(args.players),
            ),
            "label": 1.0 if train_positive else 0.0,
            "train_positive_class": bool(
                row_class in train_positive_classes
            ),
            "class": row_class,
            "policy_action": policy_action,
            "label_action": label_action,
            "target_action": target_action,
            "audit_margin": audit_margin,
            "paradox_delta": float(
                candidate_meta.get("audit_paradox_delta_vs_policy") or 0.0
            ),
            "false_positive": bool(
                candidate_meta.get("false_positive_for_positive_gate")
            ),
        })
        candidate_example_count += 1
    input_counts[str(replay_path)]["candidate_examples"] = int(
        candidate_example_count
    )
  return examples, input_counts


def _split_indices(examples, validation_fraction, seed):
  rng = random.Random(seed)
  by_label = {0: [], 1: []}
  for idx, example in enumerate(examples):
    by_label[int(example["label"])].append(idx)
  train = []
  validation = []
  for label, indices in by_label.items():
    rng.shuffle(indices)
    if len(indices) <= 1:
      train.extend(indices)
      continue
    val_count = max(1, int(round(len(indices) * validation_fraction)))
    val_count = min(val_count, len(indices) - 1)
    validation.extend(indices[:val_count])
    train.extend(indices[val_count:])
  rng.shuffle(train)
  rng.shuffle(validation)
  return train, validation


def _split_indices_by_group(examples, validation_fraction, seed, stratify_source):
  rng = random.Random(seed)
  group_rows = {}
  for idx, example in enumerate(examples):
    group_id = example["group_key"]
    if stratify_source:
      group_id = f"{example['input']}::{group_id}"
    group_rows.setdefault(group_id, []).append(idx)

  buckets = {}
  for group_id, rows in group_rows.items():
    has_positive = any(float(examples[idx]["label"]) > 0.5 for idx in rows)
    if stratify_source:
      source = examples[rows[0]]["input"]
    else:
      source = "__all__"
    buckets.setdefault((source, int(has_positive)), []).append(group_id)

  train_groups = set()
  validation_groups = set()
  for _, groups in sorted(buckets.items(), key=lambda item: str(item[0])):
    groups = list(groups)
    rng.shuffle(groups)
    if len(groups) <= 1:
      train_groups.update(groups)
      continue
    val_count = max(1, int(round(len(groups) * validation_fraction)))
    val_count = min(val_count, len(groups) - 1)
    validation_groups.update(groups[:val_count])
    train_groups.update(groups[val_count:])

  train = []
  validation = []
  for group_id, rows in group_rows.items():
    if group_id in validation_groups:
      validation.extend(rows)
    else:
      train.extend(rows)
  rng.shuffle(train)
  rng.shuffle(validation)
  split_report = {
      "split": "source_group" if stratify_source else "group",
      "groups": int(len(group_rows)),
      "train_groups": int(len(train_groups)),
      "validation_groups": int(len(validation_groups)),
      "buckets": {
          f"{source}|label={label}": int(len(groups))
          for (source, label), groups in sorted(
              buckets.items(), key=lambda item: str(item[0])
          )
      },
  }
  return train, validation, split_report


def _standardize(train_x, *arrays):
  mean = train_x.mean(axis=0)
  std = train_x.std(axis=0)
  std = np.where(std < 1e-6, 1.0, std)
  return [(arr - mean) / std for arr in arrays], mean, std


def _train_model(x_train, y_train, args, sample_weights=None):
  torch.manual_seed(int(args.seed))
  model = ImproveMLP(x_train.shape[1], max(1, int(args.hidden)))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  x_t = torch.tensor(x_train, dtype=torch.float32)
  y_t = torch.tensor(y_train, dtype=torch.float32)
  positives = float(torch.sum(y_t).item())
  negatives = float(len(y_train) - positives)
  if (
      str(getattr(args, "pos_weight_mode", "balanced")) == "balanced"
      and positives > 0
      and negatives > 0
  ):
    pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32)
  else:
    pos_weight = torch.tensor([1.0], dtype=torch.float32)
  if sample_weights is None:
    weights_t = torch.ones_like(y_t)
  else:
    weights_t = torch.tensor(sample_weights, dtype=torch.float32)
    if weights_t.shape != y_t.shape:
      raise ValueError(
          f"sample weight shape mismatch: {tuple(weights_t.shape)} "
          f"vs {tuple(y_t.shape)}"
      )
    weight_mean = torch.mean(weights_t)
    if float(weight_mean.item()) > 0:
      weights_t = weights_t / weight_mean
  losses = []
  for _ in range(max(1, int(args.epochs))):
    logits = model(x_t)
    loss_items = F.binary_cross_entropy_with_logits(
        logits, y_t, pos_weight=pos_weight, reduction="none"
    )
    gamma = max(0.0, float(getattr(args, "focal_gamma", 0.0)))
    if gamma > 0:
      probs = torch.sigmoid(logits)
      p_t = torch.where(y_t > 0.5, probs, 1.0 - probs)
      loss_items = loss_items * torch.pow(torch.clamp(1.0 - p_t, min=0.0), gamma)
    loss = torch.mean(loss_items * weights_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
  return model, {
      "initial_loss": round(losses[0], 6),
      "final_loss": round(losses[-1], 6),
      "epochs": int(args.epochs),
      "lr": float(args.lr),
      "hidden": int(args.hidden),
      "pos_weight": round(float(pos_weight.item()), 6),
      "pos_weight_mode": str(getattr(args, "pos_weight_mode", "balanced")),
      "positive_loss_weight": float(getattr(args, "positive_loss_weight", 1.0)),
      "negative_loss_weight": float(getattr(args, "negative_loss_weight", 1.0)),
      "false_positive_loss_weight": float(
          getattr(args, "false_positive_loss_weight", 1.0)
      ),
      "near_positive_loss_weight": float(
          getattr(args, "near_positive_loss_weight", 1.0)
      ),
      "focal_gamma": float(getattr(args, "focal_gamma", 0.0)),
      "sample_weight_min": round(float(torch.min(weights_t).item()), 6),
      "sample_weight_mean": round(float(torch.mean(weights_t).item()), 6),
      "sample_weight_max": round(float(torch.max(weights_t).item()), 6),
  }


def _sample_weights(examples, indices, labels, args):
  weights = []
  for local_idx, example_idx in enumerate(indices):
    example = examples[example_idx]
    label = float(labels[local_idx])
    weight = (
        float(getattr(args, "positive_loss_weight", 1.0))
        if label > 0.5 else
        float(getattr(args, "negative_loss_weight", 1.0))
    )
    if label <= 0.5 and bool(example.get("false_positive", False)):
      weight *= float(getattr(args, "false_positive_loss_weight", 1.0))
    if str(example.get("class", "")) == "NEAR_POSITIVE":
      weight *= float(getattr(args, "near_positive_loss_weight", 1.0))
    weights.append(float(weight))
  return np.asarray(weights, dtype=np.float32)


def _can_train_binary(labels):
  labels = np.array(labels, dtype=np.float32)
  return bool(np.any(labels > 0.5) and np.any(labels <= 0.5))


def _predict(model, x):
  with torch.no_grad():
    logits = model(torch.tensor(x, dtype=torch.float32))
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def _rate(num, den):
  if den <= 0:
    return None
  return round(float(num) / float(den), 6)


def _mean(values):
  if not values:
    return None
  return round(float(np.mean(values)), 6)


def _evaluate_thresholds(
    examples,
    indices,
    probs,
    thresholds,
    args,
    risk_probs=None,
    risk_threshold=None,
):
  rows = [examples[idx] for idx in indices]
  total = len(rows)
  false_positive_rows = [row for row in rows if row["false_positive"]]
  train_positive_classes = _class_set(args.train_positive_classes)
  reports = []
  for threshold in thresholds:
    override_indices = [
        local_idx for local_idx, prob in enumerate(probs)
        if float(prob) >= float(threshold)
        and (
            risk_probs is None
            or risk_threshold is None
            or float(risk_probs[local_idx]) >= float(risk_threshold)
        )
    ]
    overrides = [rows[idx] for idx in override_indices]
    positive_overrides = [
        row for row in overrides if float(row["audit_margin"]) > 0.0
    ]
    margins = [float(row["audit_margin"]) for row in overrides]
    paradox = [float(row["paradox_delta"]) for row in overrides]
    false_positive_overrides = [
        row for row in overrides if row["false_positive"]
    ]
    false_positive_rate = _rate(
        len(false_positive_overrides), len(false_positive_rows)
    )
    coverage = _rate(len(overrides), total)
    precision = _rate(len(positive_overrides), len(overrides))
    mean_margin = _mean(margins)
    mean_paradox = _mean(paradox)
    pass_gate = bool(
        coverage is not None
        and precision is not None
        and false_positive_rate is not None
        and mean_margin is not None
        and mean_paradox is not None
        and len(overrides) >= int(args.target_min_overrides)
        and len(positive_overrides)
        >= int(args.target_min_true_margin_positive_overrides)
        and coverage >= float(args.target_coverage_min)
        and coverage <= float(args.target_coverage_max)
        and precision >= float(args.target_precision_min)
        and mean_margin > float(args.target_mean_margin_min)
        and false_positive_rate <= float(args.target_false_positive_rate_max)
        and mean_paradox <= float(args.target_paradox_delta_max)
    )
    reports.append({
        "threshold": float(threshold),
        "risk_threshold": (
            None if risk_threshold is None else float(risk_threshold)
        ),
        "rows": int(total),
        "overrides": int(len(overrides)),
        "true_margin_positive_overrides": int(len(positive_overrides)),
        "coverage": coverage,
        "precision_true_margin_positive": precision,
        "positive_class_override_rate": _rate(
            sum(1 for row in overrides if row["class"] == "POSITIVE"),
            sum(1 for row in rows if row["class"] == "POSITIVE"),
        ),
        "near_positive_class_override_rate": _rate(
            sum(1 for row in overrides if row["class"] == "NEAR_POSITIVE"),
            sum(1 for row in rows if row["class"] == "NEAR_POSITIVE"),
        ),
        "train_positive_class_override_rate": _rate(
            sum(1 for row in overrides if row["class"] in train_positive_classes),
            sum(1 for row in rows if row["class"] in train_positive_classes),
        ),
        "false_positive_row_override_rate": false_positive_rate,
        "mean_true_margin": mean_margin,
        "mean_paradox_delta": mean_paradox,
        "pass_offline_gate": pass_gate,
    })
  return reports


def _evaluate_combined_thresholds(
    examples, indices, probs, risk_probs, thresholds, risk_thresholds, args
):
  reports = []
  for risk_threshold in risk_thresholds:
    reports.extend(
        _evaluate_thresholds(
            examples,
            indices,
            probs,
            thresholds,
            args,
            risk_probs=risk_probs,
            risk_threshold=risk_threshold,
        )
    )
  return reports


def _passing_across_inputs(aggregate_rows, rows_by_input):
  lookup_by_input = {
      input_name: {
          (row["threshold"], row["risk_threshold"]): row
          for row in rows
      }
      for input_name, rows in rows_by_input.items()
  }
  passing = []
  for row in aggregate_rows:
    key = (row["threshold"], row["risk_threshold"])
    if not row["pass_offline_gate"]:
      continue
    if all(
        lookup.get(key, {}).get("pass_offline_gate", False)
        for lookup in lookup_by_input.values()
    ):
      passing.append(row)
  return passing


def _oracle_report(examples, indices):
  rows = [examples[idx] for idx in indices]
  positives = [row for row in rows if row["class"] == "POSITIVE"]
  false_positive_rows = [row for row in rows if row["false_positive"]]
  return {
      "rows": int(len(rows)),
      "positive_rows": int(len(positives)),
      "oracle_positive_coverage": _rate(len(positives), len(rows)),
      "oracle_positive_mean_margin": _mean([
          float(row["audit_margin"]) for row in positives
      ]),
      "oracle_positive_mean_paradox_delta": _mean([
          float(row["paradox_delta"]) for row in positives
      ]),
      "false_positive_rows": int(len(false_positive_rows)),
  }


def _probability_rows(examples, indices, probs, risk_probs=None):
  rows = []
  for local_index, example_index in enumerate(indices):
    example = examples[example_index]
    rows.append({
        "input": str(example["input"]),
        "row_index": int(example["row_index"]),
        "candidate_rank": int(example.get("candidate_rank", 0)),
        "candidate_action": int(example["label_action"]),
        "source_key": str(example["source_key"]),
        "class": str(example["class"]),
        "audit_margin": float(example["audit_margin"]),
        "false_positive": bool(example["false_positive"]),
        "override_probability": float(probs[local_index]),
        "safe_probability": (
            None if risk_probs is None else float(risk_probs[local_index])
        ),
    })
  return rows


def main():
  args = parse_args()
  thresholds = [
      float(part.strip())
      for part in str(args.thresholds).split(",")
      if part.strip()
  ]
  risk_thresholds = [
      float(part.strip())
      for part in str(args.risk_thresholds).split(",")
      if part.strip()
  ]
  validation_replay_paths = [str(path) for path in args.validation_replay]
  validation_metadata_paths = [str(path) for path in args.validation_metadata]
  if validation_metadata_paths and len(validation_metadata_paths) != len(validation_replay_paths):
    raise ValueError("--validation-metadata count must match --validation-replay count")
  examples, input_counts = _load_examples(args)
  explicit_validation_inputs = {}
  explicit_validation_examples = []
  if validation_replay_paths:
    explicit_validation_examples, explicit_validation_inputs = _load_examples(
        args,
        replay_paths=validation_replay_paths,
        metadata_paths=validation_metadata_paths,
    )
    for example in explicit_validation_examples:
      example["explicit_validation"] = True
    for example in examples:
      example["explicit_validation"] = False
    examples.extend(explicit_validation_examples)
    input_counts.update(explicit_validation_inputs)
  if not examples:
    result = {
        "inputs": input_counts,
        "rows": 0,
        "error": "no audited calibration examples",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return

  labels = np.array([example["label"] for example in examples], dtype=np.float32)
  if args.risk_label_mode == "safe_improve":
    risk_labels = np.array([
        1.0
        if (
            float(example["label"]) > 0.5
            and float(example["paradox_delta"])
            <= float(args.target_paradox_delta_max)
        )
        else 0.0
        for example in examples
    ], dtype=np.float32)
  else:
    risk_labels = np.array([
        1.0
        if float(example["paradox_delta"]) <= float(args.target_paradox_delta_max)
        else 0.0
        for example in examples
    ], dtype=np.float32)
  classes = {}
  for example in examples:
    classes[example["class"]] = classes.get(example["class"], 0) + 1
  if validation_replay_paths:
    train_idx = [
        idx for idx, example in enumerate(examples)
        if not bool(example.get("explicit_validation", False))
    ]
    val_idx = [
        idx for idx, example in enumerate(examples)
        if bool(example.get("explicit_validation", False))
    ]
    split_report = {
        "split": "explicit_replay",
        "groups": None,
        "train_groups": None,
        "validation_groups": None,
        "buckets": {
            "train_replays": int(len(args.calibration_replay)),
            "validation_replays": int(len(validation_replay_paths)),
        },
    }
    if not train_idx:
      raise ValueError("No training examples loaded from --calibration-replay")
    if not val_idx:
      raise ValueError("No validation examples loaded from --validation-replay")
    train_sources, validation_sources, source_overlap = explicit_source_overlap(
        examples, train_idx, val_idx
    )
    split_report.update({
        "train_source_count": int(len(train_sources)),
        "validation_source_count": int(len(validation_sources)),
        "source_overlap_count": int(len(source_overlap)),
        "source_overlap": source_overlap,
        "source_disjoint": not source_overlap,
    })
    if (
        bool(getattr(args, "require_explicit_source_disjoint", False))
        and source_overlap
    ):
      raise ValueError(
          "Explicit validation proposal sources overlap training sources: "
          + ", ".join(source_overlap)
      )
  elif args.validation_split == "row":
    train_idx, val_idx = _split_indices(
        examples, float(args.validation_fraction), int(args.seed)
    )
    split_report = {
        "split": "row",
        "groups": None,
        "train_groups": None,
        "validation_groups": None,
        "buckets": None,
    }
  else:
    train_idx, val_idx, split_report = _split_indices_by_group(
        examples,
        float(args.validation_fraction),
        int(args.seed),
        stratify_source=(args.validation_split == "source_group"),
    )
  if not val_idx:
    val_idx = list(train_idx)
  x_all = np.array([example["features"] for example in examples], dtype=np.float32)
  x_train_raw = x_all[train_idx]
  x_val_raw = x_all[val_idx]
  standardized, mean, std = _standardize(x_train_raw, x_train_raw, x_val_raw, x_all)
  x_train, x_val, x_all_std = standardized
  y_train = labels[train_idx]
  train_weights = _sample_weights(examples, train_idx, y_train, args)
  model, train_report = _train_model(
      x_train, y_train, args, sample_weights=train_weights
  )
  train_probs = _predict(model, x_train)
  val_probs = _predict(model, x_val)
  all_probs = _predict(model, x_all_std)
  risk_model = None
  risk_train_report = None
  train_risk_probs = None
  val_risk_probs = None
  all_risk_probs = None
  if bool(args.train_risk_veto) and _can_train_binary(risk_labels[train_idx]):
    risk_train_weights = _sample_weights(
        examples, train_idx, risk_labels[train_idx], args
    )
    risk_model, risk_train_report = _train_model(
        x_train,
        risk_labels[train_idx],
        args,
        sample_weights=risk_train_weights,
    )
    train_risk_probs = _predict(risk_model, x_train)
    val_risk_probs = _predict(risk_model, x_val)
    all_risk_probs = _predict(risk_model, x_all_std)

  train_thresholds = _evaluate_thresholds(
      examples, train_idx, train_probs, thresholds, args
  )
  validation_thresholds = _evaluate_thresholds(
      examples, val_idx, val_probs, thresholds, args
  )
  all_thresholds = _evaluate_thresholds(
      examples, list(range(len(examples))), all_probs, thresholds, args
  )
  train_risk_thresholds = []
  validation_risk_thresholds = []
  all_risk_thresholds = []
  if risk_model is not None:
    train_risk_thresholds = _evaluate_combined_thresholds(
        examples,
        train_idx,
        train_probs,
        train_risk_probs,
        thresholds,
        risk_thresholds,
        args,
    )
    validation_risk_thresholds = _evaluate_combined_thresholds(
        examples,
        val_idx,
        val_probs,
        val_risk_probs,
        thresholds,
        risk_thresholds,
        args,
    )
    all_risk_thresholds = _evaluate_combined_thresholds(
        examples,
        list(range(len(examples))),
        all_probs,
        all_risk_probs,
        thresholds,
        risk_thresholds,
        args,
    )
  validation_thresholds_by_input = {}
  validation_risk_thresholds_by_input = {}
  if validation_replay_paths:
    for input_name in validation_replay_paths:
      local_positions = [
          local_index
          for local_index, example_index in enumerate(val_idx)
          if examples[example_index]["input"] == input_name
      ]
      input_indices = [val_idx[index] for index in local_positions]
      input_probs = val_probs[local_positions]
      validation_thresholds_by_input[input_name] = _evaluate_thresholds(
          examples, input_indices, input_probs, thresholds, args
      )
      if risk_model is not None:
        input_risk_probs = val_risk_probs[local_positions]
        validation_risk_thresholds_by_input[input_name] = (
            _evaluate_combined_thresholds(
                examples,
                input_indices,
                input_probs,
                input_risk_probs,
                thresholds,
                risk_thresholds,
                args,
            )
        )
  passing_validation = [
      row for row in validation_thresholds if row["pass_offline_gate"]
  ]
  passing_validation_risk = [
      row for row in validation_risk_thresholds if row["pass_offline_gate"]
  ]
  passing_all_validation_inputs = _passing_across_inputs(
      validation_thresholds, validation_thresholds_by_input
  )
  passing_all_validation_inputs_risk = _passing_across_inputs(
      validation_risk_thresholds, validation_risk_thresholds_by_input
  )
  best_validation = None
  if validation_thresholds:
    best_validation = max(
        validation_thresholds,
        key=lambda row: (
            row["pass_offline_gate"],
            row["precision_true_margin_positive"] or -1.0,
            row["mean_true_margin"] or -999.0,
            -(row["false_positive_row_override_rate"] or 999.0),
        ),
    )
  best_validation_risk = None
  if validation_risk_thresholds:
    best_validation_risk = max(
        validation_risk_thresholds,
        key=lambda row: (
            row["pass_offline_gate"],
            row["precision_true_margin_positive"] or -1.0,
            row["mean_true_margin"] or -999.0,
            -(row["false_positive_row_override_rate"] or 999.0),
        ),
    )
  result = {
      "inputs": input_counts,
      "rows": int(len(examples)),
      "classes": {k: int(v) for k, v in sorted(classes.items())},
      "feature_size": int(x_all.shape[1]),
      "feature_schema_version": "deployable_pair_v2",
      "deployable_feature_schema": True,
      "include_target_match_features": False,
      "seed": int(args.seed),
      "validation_fraction": float(args.validation_fraction),
      "validation_split": args.validation_split,
      "validation_replay": [str(path) for path in args.validation_replay],
      "explicit_validation_replay": bool(validation_replay_paths),
      "require_explicit_source_disjoint": bool(
          args.require_explicit_source_disjoint
      ),
      "require_each_validation_replay_pass": bool(
          args.require_each_validation_replay_pass
      ),
      "split_report": split_report,
      "label_margin_feature": bool(args.label_margin_feature),
      "observation_feature_mode": str(args.observation_feature_mode),
      "policy_context_mode": str(args.policy_context_mode),
      "game_context_mode": str(args.game_context_mode),
      "policy_context_checkpoint": str(args.policy_context_checkpoint or ""),
      "positive_label_margin_min": float(args.positive_label_margin_min),
      "train_positive_classes": sorted(_class_set(args.train_positive_classes)),
      "pos_weight_mode": str(args.pos_weight_mode),
      "positive_loss_weight": float(args.positive_loss_weight),
      "negative_loss_weight": float(args.negative_loss_weight),
      "false_positive_loss_weight": float(args.false_positive_loss_weight),
      "near_positive_loss_weight": float(args.near_positive_loss_weight),
      "focal_gamma": float(args.focal_gamma),
      "train_risk_veto": bool(args.train_risk_veto),
      "risk_label_mode": str(args.risk_label_mode),
      "risk_thresholds": risk_thresholds,
      "train_rows": int(len(train_idx)),
      "validation_rows": int(len(val_idx)),
      "train_positive_rows": int(np.sum(labels[train_idx] > 0.5)),
      "validation_positive_rows": int(np.sum(labels[val_idx] > 0.5)),
      "train_report": train_report,
      "risk_train_report": risk_train_report,
      "train_safe_rows": int(np.sum(risk_labels[train_idx] > 0.5)),
      "validation_safe_rows": int(np.sum(risk_labels[val_idx] > 0.5)),
      "gate_targets": {
          "coverage_min": float(args.target_coverage_min),
          "coverage_max": float(args.target_coverage_max),
          "min_overrides": int(args.target_min_overrides),
          "min_true_margin_positive_overrides": int(
              args.target_min_true_margin_positive_overrides
          ),
          "precision_min": float(args.target_precision_min),
          "mean_margin_min": float(args.target_mean_margin_min),
          "false_positive_rate_max": float(args.target_false_positive_rate_max),
          "paradox_delta_max": float(args.target_paradox_delta_max),
      },
      "oracle_train": _oracle_report(examples, train_idx),
      "oracle_validation": _oracle_report(examples, val_idx),
      "validation_probability_rows": _probability_rows(
          examples, val_idx, val_probs, val_risk_probs
      ),
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "validation_thresholds_by_input": validation_thresholds_by_input,
      "all_thresholds": all_thresholds,
      "train_risk_thresholds": train_risk_thresholds,
      "validation_risk_thresholds": validation_risk_thresholds,
      "validation_risk_thresholds_by_input": (
          validation_risk_thresholds_by_input
      ),
      "all_risk_thresholds": all_risk_thresholds,
      "best_validation_threshold": best_validation,
      "best_validation_risk_threshold": best_validation_risk,
      "passing_validation_thresholds": passing_validation,
      "passing_validation_risk_thresholds": passing_validation_risk,
      "passing_all_validation_inputs": passing_all_validation_inputs,
      "passing_all_validation_inputs_risk": (
          passing_all_validation_inputs_risk
      ),
      "offline_gate_pass": bool(
          (
              passing_all_validation_inputs
              or passing_all_validation_inputs_risk
          )
          if args.require_each_validation_replay_pass
          else (passing_validation or passing_validation_risk)
      ),
  }
  if args.model_out:
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "risk_model_state": (
            None if risk_model is None else risk_model.state_dict()
        ),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_size": int(x_all.shape[1]),
        "feature_schema_version": "deployable_pair_v2",
        "deployable_feature_schema": True,
        "include_target_match_features": False,
        "observation_feature_mode": str(args.observation_feature_mode),
        "policy_context_mode": str(args.policy_context_mode),
        "game_context_mode": str(args.game_context_mode),
        "policy_context_checkpoint": str(args.policy_context_checkpoint or ""),
        "train_positive_classes": sorted(_class_set(args.train_positive_classes)),
        "args": vars(args),
        "risk_train_report": risk_train_report,
        "risk_label_mode": str(args.risk_label_mode),
        "risk_target_paradox_delta_max": float(args.target_paradox_delta_max),
    }, model_path)
    result["model_out"] = str(model_path)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  printable = dict(result)
  print(json.dumps(printable, indent=2), flush=True)


if __name__ == "__main__":
  main()
