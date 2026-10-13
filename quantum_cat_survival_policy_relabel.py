"""Relabel targeted risk replay rows with survival-aware policy targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument(
      "--policy-target",
      choices=("safest", "soft_survival"),
      default="safest",
  )
  parser.add_argument("--policy-target-temperature", type=float, default=0.15)
  parser.add_argument(
      "--policy-mask",
      choices=("legal", "candidates"),
      default="candidates",
  )
  return parser.parse_args()


def relabel_policy(
    old_policy: np.ndarray,
    old_mask: np.ndarray,
    action_targets: np.ndarray,
    action_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int | None, float | None]:
  num_actions = len(old_policy)
  candidates = [
      int(action)
      for action in np.flatnonzero(np.asarray(action_mask, dtype=np.float32) > 0.0)
  ]
  if not candidates:
    return old_policy, old_mask, None, None
  risks = {action: float(action_targets[action]) for action in candidates}
  policy_target = np.zeros(num_actions, dtype=np.float32)
  if str(args.policy_target) == "safest":
    best_risk = min(risks.values())
    best_actions = [
        action for action, risk in risks.items() if abs(risk - best_risk) <= 1e-6
    ]
    best = int(max(best_actions, key=lambda action: float(old_policy[action])))
    policy_target[best] = 1.0
  else:
    temperature = max(1e-6, float(args.policy_target_temperature))
    logits = np.array(
        [(1.0 - risks[action]) / temperature for action in candidates],
        dtype=np.float32,
    )
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
      weights = np.ones_like(weights, dtype=np.float32)
      total = float(np.sum(weights))
    weights /= total
    for action, weight in zip(candidates, weights):
      policy_target[action] = float(weight)

  if str(args.policy_mask) == "candidates":
    policy_mask = np.zeros(num_actions, dtype=np.float32)
    policy_mask[candidates] = 1.0
  else:
    policy_mask = np.asarray(old_mask, dtype=np.float32).copy()
  best_action = int(np.argmax(policy_target))
  return policy_target, policy_mask, best_action, risks.get(best_action)


def main() -> None:
  args = parse_args()
  rows, metadata = load_replay(args.input, return_metadata=True)
  relabeled = []
  target_counts = {}
  target_risks = []
  labeled_rows = 0
  for row in rows:
    (
        obs,
        mask,
        policy,
        value,
        paradox,
        action,
        player,
        action_targets,
        action_mask,
        action_value_targets,
        action_value_mask,
        action_features,
        terminal_action_paradox_target,
    ) = row
    policy_target, policy_mask, best_action, best_risk = relabel_policy(
        policy, mask, action_targets, action_mask, args
    )
    if best_action is not None:
      labeled_rows += 1
      target_counts[str(best_action)] = target_counts.get(str(best_action), 0) + 1
    if best_risk is not None:
      target_risks.append(float(best_risk))
    relabeled.append((
        obs,
        policy_mask,
        policy_target,
        value,
        paradox,
        action,
        player,
        action_targets,
        action_mask,
        action_value_targets,
        action_value_mask,
        action_features,
        terminal_action_paradox_target,
    ))

  summary = {
      "source": "quantum_cat_survival_policy_relabel",
      "input": str(args.input),
      "policy_target": str(args.policy_target),
      "policy_target_temperature": float(args.policy_target_temperature),
      "policy_mask": str(args.policy_mask),
      "rows": int(len(relabeled)),
      "labeled_rows": int(labeled_rows),
      "mean_target_risk": (
          float(np.mean(np.array(target_risks, dtype=np.float32)))
          if target_risks else 0.0
      ),
      "source_metadata": metadata,
  }
  save_replay(relabeled, args.out, metadata=summary)
  summary_path = (
      Path(args.summary_out)
      if args.summary_out else Path(args.out).with_suffix(Path(args.out).suffix + ".summary.json")
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
