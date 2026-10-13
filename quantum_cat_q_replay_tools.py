#!/usr/bin/env python3
"""Utilities for deriving conservative policy targets from sparse Q labels.

The AlphaZero trainer consumes the main policy target from replay item[2].
Sparse action-value labels live separately in item[9]/item[10], so a cached
Q-label corpus does not affect the policy head unless we deliberately convert
high-confidence rows into policy targets. This tool keeps that conversion
auditable and reversible.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import (
    action_value_label_phase,
    action_value_label_quality,
    load_replay,
    save_replay,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True)
  parser.add_argument(
      "--metadata-input",
      action="append",
      default=[],
      help=(
          "Optional metadata JSON aligned with each --input. When provided, "
          "policy_best_action is used as the original policy top for regret "
          "checks. This is needed for stable replay, whose item[2] policy "
          "target is already rewritten to the teacher/recall-best action."
      ),
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--phases", default="play")
  parser.add_argument("--min-actions", type=int, default=2)
  parser.add_argument("--min-spread", type=float, default=0.0)
  parser.add_argument("--min-top-margin", type=float, default=0.0)
  parser.add_argument(
      "--min-original-top-regret",
      type=float,
      default=-1e9,
      help=(
          "Require Q(best labeled action) - Q(original policy top action) to be "
          "at least this much when the original top action is labeled."
      ),
  )
  parser.add_argument("--temperature", type=float, default=0.05)
  parser.add_argument(
      "--q-policy-weight",
      type=float,
      default=1.0,
      help="Blend weight for the soft Q policy against the original replay policy.",
  )
  parser.add_argument(
      "--keep-action-value-labels",
      action="store_true",
      help="Keep item[9]/item[10] in the derived replay for auxiliary Q training.",
  )
  parser.add_argument(
      "--include-non-overrides",
      action="store_true",
      help=(
          "Keep high-confidence rows even when the Q-soft top action matches "
          "the original replay target top action."
      ),
  )
  return parser.parse_args()


def _allowed_phases(spec):
  return {part.strip() for part in str(spec).split(",") if part.strip()}


def _labeled_mask(mask):
  return np.array(mask, dtype=np.float32) > 0.0


def _softmax_on_mask(values, mask, temperature):
  selected = _labeled_mask(mask)
  if int(np.sum(selected)) == 0:
    return None
  logits = np.array(values, dtype=np.float32)[selected] / max(1e-6, temperature)
  logits -= np.max(logits)
  exp = np.exp(logits)
  total = float(exp.sum())
  if total <= 0.0 or not math.isfinite(total):
    return None
  policy = np.zeros_like(values, dtype=np.float32)
  policy[selected] = exp / total
  return policy


def _clear_action_value_labels(example):
  if len(example) <= 10:
    return example
  items = list(example)
  if items[9] is not None:
    items[9] = np.zeros_like(items[9], dtype=np.float32)
  if items[10] is not None:
    items[10] = np.zeros_like(items[10], dtype=np.float32)
  return tuple(items)


def _one_hot_like(policy, action):
  result = np.zeros_like(np.array(policy, dtype=np.float32), dtype=np.float32)
  if action is None:
    return result
  action = int(action)
  if 0 <= action < result.shape[0]:
    result[action] = 1.0
  return result


def _metadata_original_top(metadata):
  if not metadata:
    return None
  try:
    return int(metadata.get("policy_best_action"))
  except (TypeError, ValueError):
    return None


def derive_policy_replay(replay, args, metadata_rows=None):
  allowed = _allowed_phases(args.phases)
  q_weight = min(1.0, max(0.0, float(args.q_policy_weight)))
  if metadata_rows is None:
    metadata_rows = [None] * len(replay)
  if len(metadata_rows) != len(replay):
    raise ValueError(
        f"metadata row count {len(metadata_rows)} does not match replay "
        f"row count {len(replay)}"
    )
  derived = []
  skipped = {
      "unlabeled": 0,
      "phase": 0,
      "min_actions": 0,
      "quality": 0,
      "original_regret": 0,
      "non_override": 0,
      "bad_softmax": 0,
  }
  margins = []
  spreads = []
  regrets = []
  q_top_changed = 0
  original_top_changed = 0
  rows_seen = 0
  rows_labeled = 0

  metadata_original_top_rows = 0

  for example, metadata in zip(replay, metadata_rows):
    rows_seen += 1
    quality = action_value_label_quality(example)
    if quality is None:
      skipped["unlabeled"] += 1
      continue
    rows_labeled += 1
    phase = action_value_label_phase(example)
    if allowed and phase not in allowed:
      skipped["phase"] += 1
      continue
    targets = np.array(example[9], dtype=np.float32)
    mask = _labeled_mask(example[10])
    if int(np.sum(mask)) < int(args.min_actions):
      skipped["min_actions"] += 1
      continue
    spread, top_margin = quality
    if spread < float(args.min_spread) or top_margin < float(args.min_top_margin):
      skipped["quality"] += 1
      continue
    q_top = int(np.argmax(np.where(mask, targets, -np.inf)))
    replay_policy = np.array(example[2], dtype=np.float32)
    metadata_original_top = _metadata_original_top(metadata)
    if metadata_original_top is not None:
      original_top = metadata_original_top
      original_policy = _one_hot_like(replay_policy, original_top)
      metadata_original_top_rows += 1
    else:
      original_policy = replay_policy
      original_top = int(np.argmax(original_policy))
    if mask[original_top]:
      regret = float(targets[q_top] - targets[original_top])
      if regret < float(args.min_original_top_regret):
        skipped["original_regret"] += 1
        continue
      regrets.append(regret)
    else:
      regret = None
    if q_top != original_top:
      q_top_changed += 1
    elif not args.include_non_overrides:
      skipped["non_override"] += 1
      continue

    q_policy = _softmax_on_mask(targets, mask, float(args.temperature))
    if q_policy is None:
      skipped["bad_softmax"] += 1
      continue
    derived_policy = (1.0 - q_weight) * original_policy + q_weight * q_policy
    total = float(derived_policy.sum())
    if total <= 0.0 or not math.isfinite(total):
      skipped["bad_softmax"] += 1
      continue
    derived_policy = (derived_policy / total).astype(np.float32)
    if int(np.argmax(derived_policy)) != original_top:
      original_top_changed += 1
    items = list(example)
    items[2] = derived_policy
    new_example = tuple(items)
    if not args.keep_action_value_labels:
      new_example = _clear_action_value_labels(new_example)
    derived.append(new_example)
    margins.append(float(top_margin))
    spreads.append(float(spread))

  summary = {
      "input_rows": rows_seen,
      "q_labeled_rows": rows_labeled,
      "derived_rows": len(derived),
      "skipped": skipped,
      "phases": sorted(allowed),
      "min_actions": int(args.min_actions),
      "min_spread": float(args.min_spread),
      "min_top_margin": float(args.min_top_margin),
      "min_original_top_regret": float(args.min_original_top_regret),
      "temperature": float(args.temperature),
      "q_policy_weight": q_weight,
      "include_non_overrides": bool(args.include_non_overrides),
      "keep_action_value_labels": bool(args.keep_action_value_labels),
      "metadata_original_top_rows": metadata_original_top_rows,
      "q_top_changed_rows": q_top_changed,
      "derived_top_changed_rows": original_top_changed,
      "mean_spread": float(np.mean(spreads)) if spreads else None,
      "mean_top_margin": float(np.mean(margins)) if margins else None,
      "mean_original_top_regret": float(np.mean(regrets)) if regrets else None,
  }
  return derived, summary


def _load_metadata_rows(path):
  data = json.loads(Path(path).read_text())
  if isinstance(data, dict) and isinstance(data.get("rows"), list):
    return data["rows"]
  if isinstance(data, list):
    return data
  raise ValueError(f"Metadata file {path} must be a list or contain rows=list")


def main():
  args = parse_args()
  if args.metadata_input and len(args.metadata_input) != len(args.input):
    raise SystemExit(
        "--metadata-input must be supplied once per --input when used"
    )
  replay = []
  metadata_rows = []
  input_counts = {}
  for idx, input_path in enumerate(args.input):
    rows = load_replay(input_path)
    input_counts[str(input_path)] = len(rows)
    replay.extend(rows)
    if args.metadata_input:
      rows_metadata = _load_metadata_rows(args.metadata_input[idx])
      if len(rows_metadata) != len(rows):
        raise SystemExit(
            f"Metadata row count {len(rows_metadata)} for "
            f"{args.metadata_input[idx]} does not match replay row count "
            f"{len(rows)} for {input_path}"
        )
      metadata_rows.extend(rows_metadata)
  derived, summary = derive_policy_replay(
      replay, args, metadata_rows if args.metadata_input else None
  )
  summary["inputs"] = input_counts
  if args.metadata_input:
    summary["metadata_inputs"] = list(args.metadata_input)
  summary["out"] = args.out
  text = json.dumps(summary, indent=2) + "\n"
  if args.summary_out:
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text)
  if not derived:
    raise SystemExit(text)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_replay(derived, out_path)
  print(text, end="")


if __name__ == "__main__":
  main()
