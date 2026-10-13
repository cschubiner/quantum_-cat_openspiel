#!/usr/bin/env python3
"""Build contrast replay from liveness-shield veto rows.

Input rows should come from quantum_cat_counterfactual_liquidity_replay.py with:

  --checkpoint-mode liveness_shield
  --liveness-shield-label-policy raw
  --behavior-policy-target

In that format row[2] is the raw neural policy and row[5] is the deployed
shield action.  This transformer keeps only rows where the shield action differs
from the raw policy top legal action, then labels:

  shield action -> positive target
  raw top action -> negative target
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    load_replay,
    save_replay,
)


def _bool_feature(features: np.ndarray, action: int, index: int) -> bool:
  if features is None or features.ndim != 2:
    return False
  if action < 0 or action >= features.shape[0] or index >= features.shape[1]:
    return False
  return bool(float(features[action, index]) > 0.5)


def _feature_index(features: np.ndarray, name: str) -> int | None:
  """Find an appended action feature by name when feature names are present.

  Older replays only contain numeric features.  The core fixed indices above are
  enough for most summaries; this helper is best-effort for appended columns.
  """
  # Action-feature names are not serialized into replay rows today.  Keep the
  # hook local so adding names later does not change the transform API.
  return None


def _legal_top(policy: np.ndarray, mask: np.ndarray) -> int | None:
  legal = np.flatnonzero(np.asarray(mask, dtype=np.float32) > 0.5)
  if legal.size == 0:
    return None
  policy = np.asarray(policy, dtype=np.float32)
  return int(max((int(a) for a in legal), key=lambda a: (float(policy[a]), -a)))


def _legal_margin(policy: np.ndarray, mask: np.ndarray) -> float:
  legal = np.flatnonzero(np.asarray(mask, dtype=np.float32) > 0.5)
  if legal.size < 2:
    return 0.0
  values = np.asarray(policy, dtype=np.float32)[legal]
  ordered = np.sort(values)
  return float(ordered[-1] - ordered[-2])


def build_veto_contrast_replay(
    replay,
    row_stats,
    *,
    min_raw_top_prob: float,
    min_raw_top_margin: float,
    positive_target: float,
    negative_target: float,
    negative_scope: str,
):
  selected = []
  skipped_no_veto = 0
  skipped_illegal = 0
  skipped_confidence = 0
  raw_top_probs = []
  raw_top_margins = []
  action_counts = Counter()
  shield_feature_counts = Counter()
  raw_feature_counts = Counter()

  if row_stats is not None and len(row_stats) != len(replay):
    raise ValueError(
        f"row_stats ({len(row_stats)}) != replay rows ({len(replay)})"
    )

  for idx, row in enumerate(replay):
    mask = np.asarray(row[1], dtype=np.float32)
    raw_policy = np.asarray(row[2], dtype=np.float32)
    shield_action = int(row[5])
    raw_action = _legal_top(raw_policy, mask)
    if row_stats is not None:
      stat = row_stats[idx]
      shield_action = int(stat.get("best_action", shield_action))
      raw_action = int(stat.get("baseline_action", -1))
    if raw_action is None:
      skipped_illegal += 1
      continue
    if shield_action < 0 or shield_action >= len(mask) or mask[shield_action] <= 0.5:
      skipped_illegal += 1
      continue
    if int(shield_action) == int(raw_action):
      skipped_no_veto += 1
      continue

    raw_top_prob = (
        float(raw_policy[raw_action])
        if row_stats is None and 0 <= raw_action < len(raw_policy)
        else 1.0
    )
    raw_margin = _legal_margin(raw_policy, mask) if row_stats is None else 1.0
    if raw_top_prob < min_raw_top_prob or raw_margin < min_raw_top_margin:
      skipped_confidence += 1
      continue

    legal_actions = np.flatnonzero(mask > 0.5).astype(np.int64)
    policy_target = np.zeros_like(mask, dtype=np.float32)
    policy_target[shield_action] = 1.0
    action_value_targets = np.zeros_like(mask, dtype=np.float32)
    action_value_mask = np.zeros_like(mask, dtype=np.float32)
    action_value_targets[shield_action] = float(positive_target)
    action_value_mask[shield_action] = 1.0
    if negative_scope == "raw":
      negative_actions = [int(raw_action)]
    elif negative_scope == "legal":
      negative_actions = [
          int(action) for action in legal_actions
          if int(action) != int(shield_action)
      ]
    else:
      raise ValueError(f"unknown negative_scope={negative_scope!r}")
    for action in negative_actions:
      if action == int(shield_action):
        continue
      action_value_targets[action] = float(negative_target)
      action_value_mask[action] = 1.0

    items = list(row)
    items[2] = policy_target
    items[5] = shield_action
    items[9] = action_value_targets
    items[10] = action_value_mask
    selected.append(tuple(items))

    raw_top_probs.append(raw_top_prob)
    raw_top_margins.append(raw_margin)
    action_counts["veto_rows"] += 1
    features = np.asarray(row[11], dtype=np.float32) if len(row) > 11 else None
    for prefix, action, counter in (
        ("shield", shield_action, shield_feature_counts),
        ("raw", raw_action, raw_feature_counts),
    ):
      del prefix
      if _bool_feature(features, action, ACTION_FEATURE_FOLLOWS_LED_INDEX):
        counter["follows_led"] += 1
      if _bool_feature(features, action, ACTION_FEATURE_IS_RED_INDEX):
        counter["red"] += 1

  probs = np.asarray(raw_top_probs, dtype=np.float32)
  margins = np.asarray(raw_top_margins, dtype=np.float32)
  summary = {
      "kind": "shield_veto_binary_contrast_replay",
      "source_rows": int(len(replay)),
      "row_stats_used": bool(row_stats is not None),
      "rows": int(len(selected)),
      "skipped_no_veto": int(skipped_no_veto),
      "skipped_illegal": int(skipped_illegal),
      "skipped_confidence": int(skipped_confidence),
      "min_raw_top_prob": float(min_raw_top_prob),
      "min_raw_top_margin": float(min_raw_top_margin),
      "positive_target": float(positive_target),
      "negative_target": float(negative_target),
      "negative_scope": str(negative_scope),
      "labeled_actions_per_row": (
          2 if negative_scope == "raw" else None
      ),
      "labeled_actions_total": int(sum(
          int(np.sum(np.asarray(row[10], dtype=np.float32) > 0.5))
          for row in selected
      )),
      "labeled_actions_per_row_mean": (
          float(np.mean([
              int(np.sum(np.asarray(row[10], dtype=np.float32) > 0.5))
              for row in selected
          ])) if selected else None
      ),
      "raw_top_prob": {
          "min": float(np.min(probs)) if probs.size else None,
          "mean": float(np.mean(probs)) if probs.size else None,
          "max": float(np.max(probs)) if probs.size else None,
      },
      "raw_top_margin": {
          "min": float(np.min(margins)) if margins.size else None,
          "mean": float(np.mean(margins)) if margins.size else None,
          "max": float(np.max(margins)) if margins.size else None,
      },
      "shield_action_feature_counts": dict(sorted(shield_feature_counts.items())),
      "raw_action_feature_counts": dict(sorted(raw_feature_counts.items())),
  }
  return selected, summary


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument(
      "--summary",
      default="",
      help=(
          "Optional generator summary containing row_stats. When present, "
          "baseline_action is treated as raw top and best_action as shielded "
          "teacher action."
      ),
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", required=True)
  parser.add_argument("--min-raw-top-prob", type=float, default=0.0)
  parser.add_argument("--min-raw-top-margin", type=float, default=0.0)
  parser.add_argument("--positive-target", type=float, default=1.0)
  parser.add_argument("--negative-target", type=float, default=-1.0)
  parser.add_argument(
      "--negative-scope",
      choices=("raw", "legal"),
      default="raw",
      help=(
          "Which actions receive negative action-value labels. raw labels only "
          "the raw/base action vetoed by the shield. legal labels every legal "
          "non-shield action, constraining q_policy against unlabeled third "
          "actions."
      ),
  )
  args = parser.parse_args()

  replay, metadata = load_replay(args.input, return_metadata=True)
  row_stats = None
  if args.summary:
    summary_data = json.loads(Path(args.summary).read_text())
    row_stats = summary_data.get("row_stats")
    if not isinstance(row_stats, list):
      raise ValueError(f"{args.summary} does not contain list key row_stats")
  contrast_replay, summary = build_veto_contrast_replay(
      replay,
      row_stats,
      min_raw_top_prob=float(args.min_raw_top_prob),
      min_raw_top_margin=float(args.min_raw_top_margin),
      positive_target=float(args.positive_target),
      negative_target=float(args.negative_target),
      negative_scope=str(args.negative_scope),
  )
  if not contrast_replay:
    raise ValueError("shield-veto contrast replay selected zero rows")
  summary.update({
      "input": str(args.input),
      "summary": str(args.summary),
      "out": str(args.out),
      "source_replay_metadata": metadata,
  })
  Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.summary_out).write_text(json.dumps(summary, indent=2, sort_keys=True))
  save_replay(contrast_replay, args.out, metadata=summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
