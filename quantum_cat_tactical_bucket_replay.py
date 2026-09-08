#!/usr/bin/env python3
"""Build bucket-balanced policy replays from action-feature targets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from quantum_cat_alphazero_torch import (
    TACTICAL_POLICY_BUCKET_NAMES,
    action_value_label_phase,
    action_value_label_quality,
    adapt_action_features,
    load_replay,
    policy_target_bucket_masks,
    save_replay,
    target_action_feature_rows,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument(
      "--bucket-targets",
      required=True,
      help=(
          "Comma-separated bucket=count caps, for example "
          "token_loss=80,follows_led=80,red=80."
      ),
  )
  parser.add_argument("--phases", default="")
  parser.add_argument("--min-actions", type=int, default=0)
  parser.add_argument("--min-spread", type=float, default=0.0)
  parser.add_argument("--min-top-margin", type=float, default=0.0)
  parser.add_argument("--seed", type=int, default=20260605)
  parser.add_argument(
      "--include-unbucketed",
      type=int,
      default=0,
      help="Optional cap for rows that match none of the requested buckets.",
  )
  return parser.parse_args()


def parse_bucket_targets(spec):
  targets = {}
  for part in str(spec).split(","):
    part = part.strip()
    if not part:
      continue
    if "=" not in part:
      raise ValueError(f"Expected bucket=count in {part!r}")
    name, value = part.split("=", 1)
    name = name.strip()
    if name not in TACTICAL_POLICY_BUCKET_NAMES:
      raise ValueError(
          f"Unknown tactical bucket {name!r}; expected one of "
          f"{','.join(TACTICAL_POLICY_BUCKET_NAMES)}"
      )
    count = int(value)
    if count < 0:
      raise ValueError(f"Bucket target must be non-negative: {part!r}")
    targets[name] = count
  if not targets:
    raise ValueError("--bucket-targets selected no buckets")
  return targets


def _allowed_phases(spec):
  return {part.strip() for part in str(spec).split(",") if part.strip()}


def _policy_target_buckets(example):
  policy = np.asarray(example[2], dtype=np.float32)
  if policy.size == 0 or float(policy.sum()) <= 0.0:
    return set()
  features = adapt_action_features(
      example[11] if len(example) > 11 else None,
      len(policy),
  )
  action_features_t = torch.tensor(features[None, :, :], dtype=torch.float32)
  policy_t = torch.tensor(policy[None, :], dtype=torch.float32)
  action_rows = target_action_feature_rows(action_features_t, policy_t)
  masks = policy_target_bucket_masks(action_rows)
  return {name for name, mask in masks.items() if bool(mask[0].item())}


def _passes_filters(example, args):
  allowed = _allowed_phases(args.phases)
  if allowed:
    phase = action_value_label_phase(example)
    if phase not in allowed:
      return False, "phase"
  mask = np.asarray(example[1], dtype=np.float32) > 0.0
  if int(np.sum(mask)) < int(args.min_actions):
    return False, "min_actions"
  quality = action_value_label_quality(example)
  if quality is not None:
    spread, top_margin = quality
    if spread < float(args.min_spread):
      return False, "min_spread"
    if top_margin < float(args.min_top_margin):
      return False, "min_top_margin"
  elif float(args.min_spread) > 0.0 or float(args.min_top_margin) > 0.0:
    return False, "unlabeled_quality"
  return True, ""


def select_bucket_balanced_replay(replay, args):
  rng = random.Random(int(args.seed))
  targets = parse_bucket_targets(args.bucket_targets)
  candidates = {name: [] for name in targets}
  unbucketed = []
  skipped = Counter()
  available_bucket_counts = Counter()

  indexed = list(enumerate(replay))
  rng.shuffle(indexed)
  for idx, example in indexed:
    keep, reason = _passes_filters(example, args)
    if not keep:
      skipped[reason] += 1
      continue
    buckets = _policy_target_buckets(example)
    matched = sorted(buckets & set(targets))
    if matched:
      for name in matched:
        candidates[name].append(idx)
        available_bucket_counts[name] += 1
    else:
      unbucketed.append(idx)

  selected = []
  selected_set = set()
  selected_bucket_counts = Counter()
  progress = True
  while progress:
    progress = False
    for name in targets:
      if selected_bucket_counts[name] >= targets[name]:
        continue
      while candidates[name]:
        idx = candidates[name].pop(0)
        if idx in selected_set:
          continue
        selected.append(idx)
        selected_set.add(idx)
        for bucket in _policy_target_buckets(replay[idx]) & set(targets):
          selected_bucket_counts[bucket] += 1
        progress = True
        break

  for idx in unbucketed[: max(0, int(args.include_unbucketed))]:
    if idx not in selected_set:
      selected.append(idx)
      selected_set.add(idx)

  selected.sort()
  selected_replay = [replay[idx] for idx in selected]
  selected_row_buckets = [
      (idx, sorted(_policy_target_buckets(replay[idx]))) for idx in selected
  ]
  summary = {
      "input_rows": len(replay),
      "selected_rows": len(selected_replay),
      "bucket_targets": targets,
      "available_bucket_counts": dict(sorted(available_bucket_counts.items())),
      "selected_bucket_counts": dict(sorted(selected_bucket_counts.items())),
      "selected_unbucketed_rows": int(
          sum(1 for _, buckets in selected_row_buckets if not (set(buckets) & set(targets)))
      ),
      "skipped": dict(sorted(skipped.items())),
      "seed": int(args.seed),
      "phases": sorted(_allowed_phases(args.phases)),
      "min_actions": int(args.min_actions),
      "min_spread": float(args.min_spread),
      "min_top_margin": float(args.min_top_margin),
  }
  return selected_replay, summary


def main():
  args = parse_args()
  replay = []
  input_counts = {}
  for input_path in args.input:
    rows = load_replay(input_path)
    input_counts[str(input_path)] = len(rows)
    replay.extend(rows)
  selected, summary = select_bucket_balanced_replay(replay, args)
  summary["inputs"] = input_counts
  summary["out"] = args.out
  if not selected:
    raise SystemExit(json.dumps(summary, indent=2))
  save_replay(selected, args.out, metadata=summary)
  text = json.dumps(summary, indent=2) + "\n"
  summary_path = Path(args.summary_out) if args.summary_out else None
  if summary_path is not None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
