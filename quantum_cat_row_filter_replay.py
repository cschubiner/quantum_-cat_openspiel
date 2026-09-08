#!/usr/bin/env python3
"""Audit and filter policy replays by target-action tactical features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from quantum_cat_alphazero_torch import (
    load_replay,
    policy_target_bucket_masks,
    save_replay,
    target_action_feature_rows,
)
from quantum_cat_validation_family_audit import FEATURE_NAMES


def parse_csv(spec):
  return [part.strip() for part in str(spec or "").split(",") if part.strip()]


def parse_thresholds(spec):
  result = []
  for part in parse_csv(spec):
    if ">=" in part:
      name, value = part.split(">=", 1)
      op = "ge"
    elif "<=" in part:
      name, value = part.split("<=", 1)
      op = "le"
    elif ">" in part:
      name, value = part.split(">", 1)
      op = "gt"
    elif "<" in part:
      name, value = part.split("<", 1)
      op = "lt"
    else:
      raise ValueError(f"Expected feature threshold in {part!r}")
    name = name.strip()
    if name not in FEATURE_NAMES:
      raise ValueError(f"Unknown feature {name!r}")
    result.append((FEATURE_NAMES.index(name), name, op, float(value)))
  return result


def parse_count_targets(spec, allowed_names):
  result = {}
  for part in parse_csv(spec):
    if "=" not in part:
      raise ValueError(f"Expected name=count in {part!r}")
    name, value = part.split("=", 1)
    name = name.strip()
    if allowed_names is not None and name not in allowed_names:
      raise ValueError(f"Unknown target {name!r}")
    result[name] = int(value)
  return result


def row_target_stats(example):
  policy = np.asarray(example[2], dtype=np.float32)
  q_values = np.asarray(example[9], dtype=np.float32)
  q_mask = np.asarray(example[10], dtype=np.float32) > 0.0
  if int(np.sum(q_mask)) < 2:
    return None
  labeled = np.flatnonzero(q_mask)
  target_action = int(np.argmax(policy * q_mask))
  q_best_action = int(labeled[int(np.argmax(q_values[labeled]))])
  sorted_values = np.sort(q_values[labeled])
  top_margin = float(sorted_values[-1] - sorted_values[-2])
  spread = float(sorted_values[-1] - sorted_values[0])
  original_regret = float(q_values[q_best_action] - q_values[target_action])
  return {
      "target_action": target_action,
      "q_best_action": q_best_action,
      "top_margin": top_margin,
      "spread": spread,
      "original_regret": original_regret,
  }


def target_buckets(replay):
  if not replay:
    return []
  policies = torch.tensor(
      np.asarray([row[2] for row in replay], dtype=np.float32)
  )
  features = torch.tensor(
      np.asarray([row[11] for row in replay], dtype=np.float32)
  )
  action_rows = target_action_feature_rows(features, policies)
  masks = policy_target_bucket_masks(action_rows)
  row_buckets = []
  for idx in range(len(replay)):
    row_buckets.append(
        {name for name, mask in masks.items() if bool(mask[idx].item())}
    )
  return row_buckets


def feature_values_for_targets(replay):
  values = []
  for row in replay:
    action = int(np.argmax(np.asarray(row[2]) * (np.asarray(row[10]) > 0.0)))
    values.append(np.asarray(row[11], dtype=np.float32)[action])
  return np.asarray(values, dtype=np.float32)


def keep_row(idx, stats, buckets, features, args, thresholds):
  if stats is None:
    return False
  include_any = set(parse_csv(args.include_any_bucket))
  include_all = set(parse_csv(args.include_all_buckets))
  exclude_any = set(parse_csv(args.exclude_any_bucket))
  row_buckets = buckets[idx]
  if include_any and not (row_buckets & include_any):
    return False
  if include_all and not include_all.issubset(row_buckets):
    return False
  if exclude_any and (row_buckets & exclude_any):
    return False
  if stats["top_margin"] < float(args.min_top_margin):
    return False
  if stats["spread"] < float(args.min_spread):
    return False
  if stats["original_regret"] < float(args.min_original_regret):
    return False
  for feature_idx, _, op, threshold in thresholds:
    value = float(features[idx, feature_idx])
    if op == "ge" and value < threshold:
      return False
    if op == "le" and value > threshold:
      return False
    if op == "gt" and value <= threshold:
      return False
    if op == "lt" and value >= threshold:
      return False
  return True


def row_indicators(idx, buckets, features):
  indicators = {name: 1.0 for name in buckets[idx]}
  for name in [
      "win_aligns_target",
      "hits_prediction",
      "can_still_hit_after",
      "is_red",
      "off_led_loses_token",
      "would_win",
  ]:
    if name in FEATURE_NAMES:
      indicators[name] = (
          1.0 if float(features[idx, FEATURE_NAMES.index(name)]) > 0.5 else 0.0
      )
  return indicators


def matched_score(selected_counts, indicators, targets, selected_len):
  score = 0.0
  next_len = selected_len + 1
  for name, target in targets.items():
    next_count = selected_counts.get(name, 0.0) + indicators.get(name, 0.0)
    # Penalize overshoot heavily, then absolute deviation from desired count.
    overshoot = max(0.0, next_count - target)
    score += abs(next_count - target) + 4.0 * overshoot
  # Favor rows that cover several still-under-target dimensions.
  underfill_gain = sum(
      1.0
      for name, target in targets.items()
      if selected_counts.get(name, 0.0) < target and indicators.get(name, 0.0)
  )
  return score - 0.05 * underfill_gain + 0.0001 * next_len


def select_distribution_matched(candidates, buckets, features, stats_rows, args):
  bucket_targets = parse_count_targets(args.match_bucket_counts, None)
  feature_targets = parse_count_targets(args.match_feature_positive_counts, None)
  targets = dict(bucket_targets)
  targets.update(feature_targets)
  if not targets:
    return candidates
  target_rows = int(args.match_rows)
  if target_rows <= 0:
    raise ValueError("--match-rows must be positive when match targets are set")
  if target_rows > len(candidates):
    raise ValueError(
        f"--match-rows {target_rows} exceeds candidate rows {len(candidates)}"
    )
  selected = []
  selected_set = set()
  selected_counts = Counter()
  while len(selected) < target_rows:
    best = None
    for idx in candidates:
      if idx in selected_set:
        continue
      indicators = row_indicators(idx, buckets, features)
      score = matched_score(
          selected_counts, indicators, targets, len(selected)
      )
      # Prefer higher top-margin rows for otherwise similar target matches.
      score -= 0.01 * float(stats_rows[idx]["top_margin"])
      key = (score, idx)
      if best is None or key < best[0]:
        best = (key, idx, indicators)
    if best is None:
      break
    _, idx, indicators = best
    selected.append(idx)
    selected_set.add(idx)
    for name, value in indicators.items():
      if name in targets:
        selected_counts[name] += value
  return selected


def summarize(replay, selected, buckets, stats_rows, features):
  bucket_counts = Counter()
  for idx in selected:
    bucket_counts.update(buckets[idx])
  feature_summary = {}
  for name in [
      "rank_norm",
      "would_win",
      "win_aligns_target",
      "follows_led",
      "off_led_loses_token",
      "adjacency_gain_norm",
      "largest_after_norm",
      "is_red",
      "hits_prediction",
      "can_still_hit_after",
      "hit_with_future_tricks",
      "token_loss_newly_loses_led",
      "token_loss_legal_lead_damage",
      "cluster_frontier_gain",
      "cluster_connects_components",
  ]:
    if name not in FEATURE_NAMES or not selected:
      continue
    values = features[selected, FEATURE_NAMES.index(name)]
    feature_summary[name] = {
        "mean": round(float(np.mean(values)), 4),
        "positive_rate": round(float(np.mean(values > 0.5)), 4),
    }
  stat_summary = {}
  for name in ["top_margin", "spread", "original_regret"]:
    values = [stats_rows[idx][name] for idx in selected if stats_rows[idx]]
    if values:
      stat_summary[name] = {
          "mean": round(float(np.mean(values)), 4),
          "min": round(float(np.min(values)), 4),
          "max": round(float(np.max(values)), 4),
      }
  return {
      "rows": len(replay),
      "selected_rows": len(selected),
      "selected_indices": selected,
      "selected_bucket_counts": dict(sorted(bucket_counts.items())),
      "selected_stats": stat_summary,
      "selected_target_features": feature_summary,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--out")
  parser.add_argument("--summary-out", required=True)
  parser.add_argument("--include-any-bucket", default="")
  parser.add_argument("--include-all-buckets", default="")
  parser.add_argument("--exclude-any-bucket", default="")
  parser.add_argument("--feature-thresholds", default="")
  parser.add_argument("--min-top-margin", type=float, default=0.0)
  parser.add_argument("--min-spread", type=float, default=0.0)
  parser.add_argument("--min-original-regret", type=float, default=0.0)
  parser.add_argument("--match-rows", type=int, default=0)
  parser.add_argument("--match-bucket-counts", default="")
  parser.add_argument("--match-feature-positive-counts", default="")
  args = parser.parse_args()

  replay, source_metadata = load_replay(args.input, return_metadata=True)
  buckets = target_buckets(replay)
  features = feature_values_for_targets(replay)
  stats_rows = [row_target_stats(row) for row in replay]
  thresholds = parse_thresholds(args.feature_thresholds)
  candidates = [
      idx for idx in range(len(replay))
      if keep_row(idx, stats_rows[idx], buckets, features, args, thresholds)
  ]
  selected = select_distribution_matched(
      candidates, buckets, features, stats_rows, args
  )
  summary = summarize(replay, selected, buckets, stats_rows, features)
  summary.update({
      "input": args.input,
      "out": args.out,
      "criteria": {
          "include_any_bucket": parse_csv(args.include_any_bucket),
          "include_all_buckets": parse_csv(args.include_all_buckets),
          "exclude_any_bucket": parse_csv(args.exclude_any_bucket),
          "feature_thresholds": [
              {"feature": name, "op": op, "value": value}
              for _, name, op, value in thresholds
          ],
          "min_top_margin": args.min_top_margin,
          "min_spread": args.min_spread,
          "min_original_regret": args.min_original_regret,
          "match_rows": args.match_rows,
          "match_bucket_counts": parse_count_targets(
              args.match_bucket_counts, None
          ),
          "match_feature_positive_counts": parse_count_targets(
              args.match_feature_positive_counts, None
          ),
      },
      "candidate_rows": len(candidates),
  })
  source_group_ids = []
  if isinstance(source_metadata, dict):
    maybe_group_ids = source_metadata.get("row_group_ids")
    if isinstance(maybe_group_ids, list) and len(maybe_group_ids) == len(replay):
      source_group_ids = [str(item) for item in maybe_group_ids]
  if source_group_ids:
    selected_group_ids = [source_group_ids[int(idx)] for idx in selected]
    summary.update({
        "row_group_ids": selected_group_ids,
        "row_group_id_kind": "source_match_round",
        "row_group_count": int(len(set(selected_group_ids))),
    })
  Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.summary_out).write_text(json.dumps(summary, indent=2, sort_keys=True))
  if args.out:
    if not selected:
      raise ValueError("filter selected zero rows")
    save_replay([replay[idx] for idx in selected], args.out, metadata=summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
