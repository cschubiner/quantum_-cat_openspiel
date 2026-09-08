#!/usr/bin/env python3
"""Filter replays by explicit per-action paradox-risk labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--min-labeled-actions", type=int, default=2)
  parser.add_argument("--min-safe-actions", type=int, default=1)
  parser.add_argument("--min-risky-actions", type=int, default=1)
  parser.add_argument("--min-spread", type=float, default=0.0)
  parser.add_argument(
      "--safe-threshold",
      type=float,
      default=0.5,
      help="Labeled actions at or below this target are counted as safe.",
  )
  parser.add_argument(
      "--risky-threshold",
      type=float,
      default=0.5,
      help="Labeled actions above this target are counted as risky.",
  )
  parser.add_argument("--max-rows", type=int, default=0)
  parser.add_argument("--shuffle-seed", type=int, default=0)
  return parser.parse_args()


def action_paradox_stats(example, args):
  if len(example) <= 8 or example[7] is None or example[8] is None:
    return None
  targets = np.asarray(example[7], dtype=np.float32)
  mask = np.asarray(example[8], dtype=np.float32) > 0.0
  labeled = np.flatnonzero(mask)
  if labeled.size < int(args.min_labeled_actions):
    return None
  labeled_targets = targets[labeled]
  safe_count = int(np.sum(labeled_targets <= float(args.safe_threshold)))
  risky_count = int(np.sum(labeled_targets > float(args.risky_threshold)))
  spread = float(np.max(labeled_targets) - np.min(labeled_targets))
  return {
      "labeled_actions": int(labeled.size),
      "safe_actions": safe_count,
      "risky_actions": risky_count,
      "min_target": float(np.min(labeled_targets)),
      "max_target": float(np.max(labeled_targets)),
      "spread": spread,
  }


def keep_stats(stats, args):
  if stats is None:
    return False, "unlabeled"
  if stats["labeled_actions"] < int(args.min_labeled_actions):
    return False, "min_labeled_actions"
  if stats["safe_actions"] < int(args.min_safe_actions):
    return False, "min_safe_actions"
  if stats["risky_actions"] < int(args.min_risky_actions):
    return False, "min_risky_actions"
  if stats["spread"] < float(args.min_spread):
    return False, "min_spread"
  return True, ""


def main():
  args = parse_args()
  replay = []
  input_counts = {}
  input_metadata = []
  for input_path in args.input:
    rows, metadata = load_replay(input_path, return_metadata=True)
    input_counts[str(input_path)] = len(rows)
    input_metadata.append({"path": str(input_path), "metadata": metadata})
    replay.extend(rows)

  candidates = []
  skipped = Counter()
  selected_stats = []
  for idx, example in enumerate(replay):
    stats = action_paradox_stats(example, args)
    keep, reason = keep_stats(stats, args)
    if not keep:
      skipped[reason] += 1
      continue
    candidates.append(idx)
    selected_stats.append(stats)

  if int(args.shuffle_seed) != 0:
    paired = list(zip(candidates, selected_stats))
    random.Random(int(args.shuffle_seed)).shuffle(paired)
    candidates = [idx for idx, _ in paired]
    selected_stats = [stats for _, stats in paired]

  if int(args.max_rows) > 0:
    candidates = candidates[: int(args.max_rows)]
    selected_stats = selected_stats[: int(args.max_rows)]

  selected = [replay[idx] for idx in candidates]
  if selected_stats:
    spread_values = np.asarray([row["spread"] for row in selected_stats])
    safe_counts = np.asarray([row["safe_actions"] for row in selected_stats])
    risky_counts = np.asarray([row["risky_actions"] for row in selected_stats])
    stat_summary = {
        "spread_mean": float(np.mean(spread_values)),
        "spread_min": float(np.min(spread_values)),
        "spread_max": float(np.max(spread_values)),
        "safe_actions_mean": float(np.mean(safe_counts)),
        "risky_actions_mean": float(np.mean(risky_counts)),
    }
  else:
    stat_summary = {}

  summary = {
      "inputs": input_counts,
      "input_metadata": input_metadata,
      "input_rows": len(replay),
      "selected_rows": len(selected),
      "skipped": dict(sorted(skipped.items())),
      "filters": {
          "min_labeled_actions": int(args.min_labeled_actions),
          "min_safe_actions": int(args.min_safe_actions),
          "min_risky_actions": int(args.min_risky_actions),
          "min_spread": float(args.min_spread),
          "safe_threshold": float(args.safe_threshold),
          "risky_threshold": float(args.risky_threshold),
          "max_rows": int(args.max_rows),
          "shuffle_seed": int(args.shuffle_seed),
      },
      "selected_stats": stat_summary,
      "out": args.out,
  }
  if not selected:
    raise SystemExit(json.dumps(summary, indent=2, sort_keys=True))

  save_replay(selected, args.out, metadata=summary)
  text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
  if args.summary_out:
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
