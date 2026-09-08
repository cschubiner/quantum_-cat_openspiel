#!/usr/bin/env python3
"""Audit a policy checkpoint on fail-closed proposal-distribution replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from quantum_cat_alphazero_torch import (
    adapt_action_features,
    adapt_action_vector,
    adapt_observation,
    load_model_payload,
    load_replay,
    make_game,
    model_input_size,
)
from quantum_cat_proposal_policy_replay_repair import SAFE_TARGET_CONTRACT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--anchor-checkpoint", required=True)
  parser.add_argument("--replay", required=True)
  parser.add_argument("--metadata", default="")
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
  parser.add_argument("--out", required=True)
  parser.add_argument("--min-positive-rows", type=int, default=1)
  parser.add_argument("--min-positive-top1-rate", type=float, default=0.0)
  parser.add_argument("--min-base-keep-top1-rate", type=float, default=0.0)
  parser.add_argument(
      "--min-positive-target-prob-gain", type=float, default=0.0
  )
  parser.add_argument(
      "--max-unsafe-proposal-adoption-rate", type=float, default=1.0
  )
  return parser.parse_args()


def _safe_rate(numerator: int, denominator: int) -> float | None:
  return float(numerator / denominator) if denominator else None


def _rounded(value: float | None) -> float | None:
  return None if value is None else round(float(value), 6)


def _group_report(
    mask: np.ndarray,
    candidate_probs: np.ndarray,
    anchor_probs: np.ndarray,
    targets: np.ndarray,
) -> dict:
  indices = np.flatnonzero(mask)
  if not len(indices):
    return {"count": 0}
  candidate_top = np.argmax(candidate_probs[indices], axis=1)
  anchor_top = np.argmax(anchor_probs[indices], axis=1)
  target = targets[indices]
  row_indices = np.arange(len(indices))
  candidate_target_prob = candidate_probs[indices][row_indices, target]
  anchor_target_prob = anchor_probs[indices][row_indices, target]
  return {
      "count": int(len(indices)),
      "candidate_target_top1_count": int(np.sum(candidate_top == target)),
      "candidate_target_top1_rate": _rounded(np.mean(candidate_top == target)),
      "anchor_target_top1_count": int(np.sum(anchor_top == target)),
      "anchor_target_top1_rate": _rounded(np.mean(anchor_top == target)),
      "candidate_mean_target_prob": _rounded(np.mean(candidate_target_prob)),
      "anchor_mean_target_prob": _rounded(np.mean(anchor_target_prob)),
      "mean_target_prob_gain": _rounded(
          np.mean(candidate_target_prob - anchor_target_prob)
      ),
      "candidate_cross_entropy": _rounded(
          np.mean(-np.log(np.clip(candidate_target_prob, 1e-12, 1.0)))
      ),
      "anchor_cross_entropy": _rounded(
          np.mean(-np.log(np.clip(anchor_target_prob, 1e-12, 1.0)))
      ),
  }


def proposal_policy_audit_report(
    candidate_probs: np.ndarray,
    anchor_probs: np.ndarray,
    policies: np.ndarray,
    metadata_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
  if candidate_probs.shape != anchor_probs.shape:
    raise ValueError("candidate/anchor probability shape mismatch")
  if len(candidate_probs) != len(policies) or len(policies) != len(metadata_rows):
    raise ValueError("probability/replay/metadata row mismatch")
  targets = np.argmax(policies, axis=1).astype(np.int64)
  classes = np.array([
      str(row.get("target_calibration_class", "")) for row in metadata_rows
  ])
  unexpected = sorted(set(classes) - {"POSITIVE", "NEGATIVE"})
  if unexpected:
    raise ValueError(f"unexpected fail-closed target classes: {unexpected}")
  positive_mask = classes == "POSITIVE"
  base_mask = classes == "NEGATIVE"
  for idx, row in enumerate(metadata_rows):
    declared_target = int(row.get("deployment_target_action", -1))
    if declared_target != int(targets[idx]):
      raise ValueError(
          f"row {idx} replay/metadata target mismatch: "
          f"{targets[idx]} != {declared_target}"
      )
    base = int(row.get("policy_best_action", -1))
    if positive_mask[idx] and declared_target == base:
      raise ValueError(f"row {idx} POSITIVE target does not override base")
    if base_mask[idx] and declared_target != base:
      raise ValueError(f"row {idx} NEGATIVE target does not keep base")

  candidate_top = np.argmax(candidate_probs, axis=1)
  unsafe_eligible = np.zeros(len(metadata_rows), dtype=bool)
  unsafe_adopted = np.zeros(len(metadata_rows), dtype=bool)
  for idx, row in enumerate(metadata_rows):
    if not base_mask[idx]:
      continue
    proposed = row.get("primary_proposal_action")
    if proposed is None:
      evaluations = list(row.get("candidate_evaluations") or [])
      proposed = evaluations[0].get("action") if evaluations else None
    if proposed is None:
      continue
    proposed = int(proposed)
    base = int(row["policy_best_action"])
    if proposed == base:
      continue
    unsafe_eligible[idx] = True
    unsafe_adopted[idx] = int(candidate_top[idx]) == proposed

  positive = _group_report(
      positive_mask, candidate_probs, anchor_probs, targets
  )
  base_keep = _group_report(base_mask, candidate_probs, anchor_probs, targets)
  all_rows = _group_report(
      np.ones(len(metadata_rows), dtype=bool),
      candidate_probs,
      anchor_probs,
      targets,
  )
  unsafe_count = int(np.sum(unsafe_adopted))
  unsafe_eligible_count = int(np.sum(unsafe_eligible))
  unsafe_rate = _safe_rate(unsafe_count, unsafe_eligible_count)
  anchor_top = np.argmax(anchor_probs, axis=1)
  row_diagnostics = []
  for idx, row in enumerate(metadata_rows):
    target = int(targets[idx])
    proposed = row.get("primary_proposal_action")
    row_diagnostics.append({
        "row_index": int(idx),
        "target_calibration_class": str(classes[idx]),
        "trick_number": int(row.get("trick_number", -1)),
        "base_action": int(row.get("policy_best_action", -1)),
        "proposed_action": None if proposed is None else int(proposed),
        "target_action": target,
        "candidate_top_action": int(candidate_top[idx]),
        "anchor_top_action": int(anchor_top[idx]),
        "candidate_target_prob": _rounded(candidate_probs[idx, target]),
        "anchor_target_prob": _rounded(anchor_probs[idx, target]),
        "target_prob_gain": _rounded(
            candidate_probs[idx, target] - anchor_probs[idx, target]
        ),
    })
  checks = {
      "minimum_positive_rows": (
          int(positive.get("count", 0)) >= int(args.min_positive_rows)
      ),
      "minimum_positive_top1_rate": (
          float(positive.get("candidate_target_top1_rate") or 0.0)
          >= float(args.min_positive_top1_rate)
      ),
      "minimum_base_keep_top1_rate": (
          float(base_keep.get("candidate_target_top1_rate") or 0.0)
          >= float(args.min_base_keep_top1_rate)
      ),
      "minimum_positive_target_probability_gain": (
          float(positive.get("mean_target_prob_gain") or 0.0)
          >= float(args.min_positive_target_prob_gain)
      ),
      "maximum_unsafe_proposal_adoption_rate": (
          unsafe_rate is not None
          and unsafe_rate <= float(args.max_unsafe_proposal_adoption_rate)
      ),
  }
  return {
      "rows": int(len(metadata_rows)),
      "all": all_rows,
      "positive_override": positive,
      "base_keep": base_keep,
      "unsafe_proposal_adoption": {
          "eligible_count": unsafe_eligible_count,
          "adopted_count": unsafe_count,
          "rate": _rounded(unsafe_rate),
      },
      "row_diagnostics": row_diagnostics,
      "deployment_gate": {
          "thresholds": {
              "min_positive_rows": int(args.min_positive_rows),
              "min_positive_top1_rate": float(args.min_positive_top1_rate),
              "min_base_keep_top1_rate": float(args.min_base_keep_top1_rate),
              "min_positive_target_prob_gain": float(
                  args.min_positive_target_prob_gain
              ),
              "max_unsafe_proposal_adoption_rate": float(
                  args.max_unsafe_proposal_adoption_rate
              ),
          },
          "checks": checks,
          "pass": bool(all(checks.values())),
      },
  }


def checkpoint_probabilities(model, replay, device: torch.device) -> np.ndarray:
  obs_size = model_input_size(model)
  num_actions = int(model.num_actions)
  obs = np.array([
      adapt_observation(row[0], obs_size) for row in replay
  ], dtype=np.float32)
  masks = np.array([
      adapt_action_vector(row[1], num_actions, dtype=np.float32) > 0.5
      for row in replay
  ], dtype=bool)
  features = np.array([
      adapt_action_features(row[11] if len(row) > 11 else None, num_actions)
      for row in replay
  ], dtype=np.float32)
  chunks = []
  model.eval()
  with torch.no_grad():
    for start in range(0, len(replay), 128):
      end = min(len(replay), start + 128)
      obs_t = torch.tensor(obs[start:end], dtype=torch.float32, device=device)
      mask_t = torch.tensor(masks[start:end], dtype=torch.bool, device=device)
      feature_t = torch.tensor(
          features[start:end], dtype=torch.float32, device=device
      )
      logits, _ = model(obs_t, feature_t)
      logits = logits.masked_fill(~mask_t, -1e9)
      chunks.append(F.softmax(logits, dim=1).cpu().numpy())
  return np.concatenate(chunks, axis=0)


def main() -> None:
  args = parse_args()
  use_mps = args.device == "mps" or (
      args.device == "auto" and torch.backends.mps.is_available()
  )
  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device=mps requested, but torch MPS is unavailable")
  device = torch.device("mps" if use_mps else "cpu")
  replay, replay_metadata = load_replay(args.replay, return_metadata=True)
  metadata_path = (
      Path(args.metadata)
      if args.metadata
      else Path(str(args.replay) + ".metadata.json")
  )
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  rows = list(metadata.get("rows") or [])
  if len(rows) != len(replay):
    raise ValueError(f"replay/metadata row mismatch: {len(replay)} != {len(rows)}")
  contract = str(
      metadata.get("safe_policy_target_contract")
      or replay_metadata.get("safe_policy_target_contract")
      or ""
  )
  if contract != SAFE_TARGET_CONTRACT:
    raise ValueError(
        f"expected safe target contract {SAFE_TARGET_CONTRACT}, got "
        f"{contract or 'missing'}"
    )
  scope = str(
      metadata.get("counterfactual_action_paradox_scope")
      or replay_metadata.get("counterfactual_action_paradox_scope")
      or ""
  )
  if scope != "any":
    raise ValueError(f"expected any-player proposal audit scope, got {scope!r}")
  positive_repeat = int(
      metadata.get("positive_repeat", replay_metadata.get("positive_repeat", 1))
  )
  if positive_repeat != 1:
    raise ValueError(
        "checkpoint audit requires an unweighted validation replay "
        f"(positive_repeat={positive_repeat})"
    )

  model_args = SimpleNamespace(
      players=int(args.players),
      arch="action_mlp",
      width=256,
      depth=3,
      separate_action_value_encoder=False,
      separate_action_paradox_encoder=False,
      match_context=True,
  )
  game = make_game(model_args, 0)
  model, _, _ = load_model_payload(args.checkpoint, game, model_args, device)
  anchor, _, _ = load_model_payload(
      args.anchor_checkpoint, game, model_args, device
  )
  policies = np.array([row[2] for row in replay], dtype=np.float32)
  report = proposal_policy_audit_report(
      checkpoint_probabilities(model, replay, device),
      checkpoint_probabilities(anchor, replay, device),
      policies,
      rows,
      args,
  )
  report.update({
      "checkpoint": str(args.checkpoint),
      "anchor_checkpoint": str(args.anchor_checkpoint),
      "replay": str(args.replay),
      "metadata": str(metadata_path),
      "safe_policy_target_contract": contract,
      "counterfactual_action_paradox_scope": scope,
  })
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
