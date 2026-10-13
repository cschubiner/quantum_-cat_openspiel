#!/usr/bin/env python3
"""Build survival-grammar policy targets from legal action features.

This is a curriculum-pretraining tool.  It deliberately avoids terminal score as
the main teacher and instead constructs dense, local policy targets from public
legality/liquidity features: follow led color, avoid newly burning led-color
tokens, preserve lane surplus, and avoid future forced-card bottlenecks.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    APPENDED_ACTION_FEATURE_INDEX,
    ACTION_FEATURE_PREFIX_SIZE,
    load_replay,
    save_replay,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument(
      "--row-filter",
      choices=("all", "lane_contrast", "bottleneck", "lane_or_bottleneck"),
      default="lane_or_bottleneck",
  )
  parser.add_argument("--max-rows", type=int, default=0)
  parser.add_argument("--seed", type=int, default=20260609)
  parser.add_argument("--temperature", type=float, default=0.85)
  parser.add_argument(
      "--prior-weight",
      type=float,
      default=0.04,
      help="Small log-policy prior weight to keep structural targets sane.",
  )
  parser.add_argument(
      "--action-value-weight",
      type=float,
      default=0.20,
      help=(
          "Optional bounded dense-safety action-value contribution. This is a "
          "secondary signal; structural legality/liquidity remains primary."
      ),
  )
  parser.add_argument("--follow-led-bonus", type=float, default=1.6)
  parser.add_argument("--token-loss-penalty", type=float, default=4.2)
  parser.add_argument(
      "--red-switch-bonus",
      type=float,
      default=0.75,
      help=(
          "Bonus for an off-led lane-burning action that switches to red. "
          "This models the main acceptable reason to leave the led color."
      ),
  )
  parser.add_argument(
      "--red-switch-win-bonus",
      type=float,
      default=1.75,
      help=(
          "Extra bonus when the red switch is predicted to win the current "
          "trick."
      ),
  )
  parser.add_argument(
      "--nonred-switch-penalty",
      type=float,
      default=0.80,
      help="Extra penalty for off-led lane-burning actions that are not red.",
  )
  parser.add_argument(
      "--nonwinning-switch-penalty",
      type=float,
      default=0.65,
      help=(
          "Extra penalty for off-led lane-burning actions that do not win the "
          "current trick."
      ),
  )
  parser.add_argument("--min-lane-surplus-weight", type=float, default=1.25)
  parser.add_argument("--lane-damage-penalty", type=float, default=1.10)
  parser.add_argument("--lane-pressure-penalty", type=float, default=0.45)
  parser.add_argument("--own-min-colors-weight", type=float, default=0.85)
  parser.add_argument("--own-zero-exit-penalty", type=float, default=2.2)
  parser.add_argument("--own-one-exit-penalty", type=float, default=1.25)
  parser.add_argument("--own-two-less-exit-penalty", type=float, default=0.45)
  parser.add_argument("--future-no-exit-penalty", type=float, default=1.7)
  parser.add_argument("--forced-pressure-penalty", type=float, default=0.70)
  parser.add_argument("--paradox-penalty", type=float, default=8.0)
  parser.add_argument("--discard-relief-weight", type=float, default=1.0)
  parser.add_argument("--discard-no-exit-penalty", type=float, default=1.5)
  parser.add_argument("--prediction-abs-gap-penalty", type=float, default=0.35)
  parser.add_argument("--prediction-over-penalty", type=float, default=0.15)
  parser.add_argument("--bottleneck-zero-threshold", type=float, default=0.01)
  parser.add_argument("--bottleneck-one-threshold", type=float, default=0.12)
  parser.add_argument("--min-structural-spread", type=float, default=0.10)
  return parser.parse_args()


def appended(name):
  return APPENDED_ACTION_FEATURE_INDEX[name]


def feature(features, name):
  return features[:, appended(name)]


def has_feature(name):
  return name in APPENDED_ACTION_FEATURE_INDEX


def safe_feature(features, name):
  if not has_feature(name):
    return np.zeros(features.shape[0], dtype=np.float32)
  return feature(features, name)


def phase_name(features, legal):
  if not np.any(legal):
    return "unknown"
  idx = int(np.flatnonzero(legal)[0])
  phase_idx = int(np.argmax(features[idx, 1:6]))
  names = ["chance", "discard", "prediction", "play", "terminal"]
  return names[phase_idx] if 0 <= phase_idx < len(names) else "unknown"


def action_kind_name(features, action):
  idx = int(np.argmax(features[action, 6:11]))
  names = ["discard", "prediction", "play", "paradox", "other"]
  return names[idx] if 0 <= idx < len(names) else "unknown"


def legal_entropy(policy):
  probs = policy[policy > 0.0]
  if probs.size == 0:
    return 0.0
  return float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))


def lane_score(features, args):
  score = np.zeros(features.shape[0], dtype=np.float32)
  off_led = features[:, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX] > 0.5
  red = features[:, ACTION_FEATURE_IS_RED_INDEX] > 0.5
  # Numeric slot 19 is would_win_after_play encoded as 1.0 win, -1.0 loss,
  # and 0.0 when the outcome is not currently knowable.
  would_win = features[:, ACTION_FEATURE_PREFIX_SIZE + 19] > 0.5
  score += float(args.follow_led_bonus) * features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX]
  score -= float(args.token_loss_penalty) * safe_feature(
      features, "token_loss_newly_loses_led"
  )
  score += float(getattr(args, "red_switch_bonus", 0.75)) * (
      off_led & red
  ).astype(np.float32)
  score += float(getattr(args, "red_switch_win_bonus", 1.75)) * (
      off_led & red & would_win
  ).astype(np.float32)
  score -= float(getattr(args, "nonred_switch_penalty", 0.80)) * (
      off_led & ~red
  ).astype(np.float32)
  score -= float(getattr(args, "nonwinning_switch_penalty", 0.65)) * (
      off_led & ~would_win
  ).astype(np.float32)
  score += float(args.min_lane_surplus_weight) * safe_feature(
      features, "legal_z_exit_min_player_lane_surplus_after"
  )
  score -= float(args.lane_damage_penalty) * safe_feature(
      features, "legal_z_exit_lane_surplus_damage"
  )
  score -= float(args.lane_pressure_penalty) * safe_feature(
      features, "legal_z_exit_lane_pressure_player_count_after"
  )
  score += float(args.own_min_colors_weight) * safe_feature(
      features, "legal_z_own_future_min_colors_after"
  )
  score -= float(args.own_zero_exit_penalty) * safe_feature(
      features, "legal_z_own_future_zero_exit_frac_after"
  )
  score -= float(args.own_one_exit_penalty) * safe_feature(
      features, "legal_z_own_future_one_exit_frac_after"
  )
  score -= float(args.own_two_less_exit_penalty) * safe_feature(
      features, "own_future_two_or_less_exit_frac_after"
  )
  score -= float(args.future_no_exit_penalty) * safe_feature(
      features, "future_no_exit_after"
  )
  score -= float(args.forced_pressure_penalty) * safe_feature(
      features, "legal_z_post_hit_forced_pressure"
  )
  score -= float(args.paradox_penalty) * features[:, ACTION_FEATURE_PREFIX_SIZE + 31]
  return score


def discard_score(features, args):
  score = np.zeros(features.shape[0], dtype=np.float32)
  relief_weight = float(args.discard_relief_weight)
  score += relief_weight * safe_feature(features, "legal_z_discard_rank_deficit_relief")
  score += 0.7 * relief_weight * safe_feature(
      features, "legal_z_discard_rank_buffer_relief"
  )
  score += 0.8 * relief_weight * safe_feature(features, "legal_z_discard_safe_flex_delta")
  score += 0.35 * relief_weight * safe_feature(
      features, "legal_z_discard_rank_slot_surplus_after"
  )
  score -= float(args.discard_no_exit_penalty) * safe_feature(
      features, "discard_no_exit_after"
  )
  return score


def prediction_score(features, args):
  score = np.zeros(features.shape[0], dtype=np.float32)
  score -= float(args.prediction_abs_gap_penalty) * safe_feature(
      features, "prediction_action_abs_gap_to_expected"
  )
  score -= float(args.prediction_over_penalty) * safe_feature(
      features, "prediction_action_over_expected"
  )
  return score


def structural_scores(features, policy, value_targets, value_mask, args):
  scores = (
      lane_score(features, args)
      + discard_score(features, args)
      + prediction_score(features, args)
  )
  prior_weight = float(args.prior_weight)
  if prior_weight > 0.0:
    scores += prior_weight * np.log(np.maximum(policy, 1e-8))
  q_weight = float(args.action_value_weight)
  if q_weight > 0.0 and value_targets is not None and value_mask is not None:
    valid_q = np.asarray(value_mask, dtype=np.float32) > 0.0
    q = np.asarray(value_targets, dtype=np.float32)
    scores = np.where(valid_q, scores + q_weight * q, scores)
  return scores.astype(np.float32)


def softmax_target(scores, legal, temperature):
  target = np.zeros_like(scores, dtype=np.float32)
  legal_indices = np.flatnonzero(legal)
  if legal_indices.size == 0:
    return target
  temp = max(1e-6, float(temperature))
  legal_scores = scores[legal_indices] / temp
  legal_scores = legal_scores - float(np.max(legal_scores))
  probs = np.exp(legal_scores)
  probs_sum = float(np.sum(probs))
  if not math.isfinite(probs_sum) or probs_sum <= 0.0:
    probs = np.ones_like(probs, dtype=np.float32)
    probs_sum = float(np.sum(probs))
  target[legal_indices] = probs / probs_sum
  return target


def row_kinds(features, legal, scores, args):
  legal_features = features[legal]
  has_lane = bool(
      legal_features.size
      and np.any(legal_features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5)
      and np.any(safe_feature(legal_features, "token_loss_newly_loses_led") > 0.5)
  )
  zero_exit = safe_feature(legal_features, "own_future_zero_exit_frac_after")
  one_exit = safe_feature(legal_features, "own_future_one_exit_frac_after")
  bottleneck = bool(
      legal_features.size
      and (
          float(np.max(zero_exit, initial=0.0)) >= float(args.bottleneck_zero_threshold)
          or float(np.max(one_exit, initial=0.0)) >=
          float(args.bottleneck_one_threshold)
      )
  )
  legal_scores = scores[legal]
  spread = (
      float(np.max(legal_scores) - np.min(legal_scores))
      if legal_scores.size
      else 0.0
  )
  return has_lane, bottleneck, spread


def keep_row(features, legal, scores, args):
  has_lane, bottleneck, spread = row_kinds(features, legal, scores, args)
  if spread < float(args.min_structural_spread):
    return False
  mode = str(args.row_filter)
  if mode == "all":
    return True
  if mode == "lane_contrast":
    return has_lane
  if mode == "bottleneck":
    return bottleneck
  if mode == "lane_or_bottleneck":
    return has_lane or bottleneck
  raise ValueError(f"unknown row filter {mode!r}")


def relabel_row(row, args):
  mask = np.asarray(row[1], dtype=np.float32)
  policy = np.asarray(row[2], dtype=np.float32)
  features = np.asarray(row[11], dtype=np.float32)
  value_targets = row[9] if len(row) > 10 else None
  value_mask = row[10] if len(row) > 10 else None
  legal = mask > 0.5
  scores = structural_scores(features, policy, value_targets, value_mask, args)
  if not keep_row(features, legal, scores, args):
    return None, None
  target = softmax_target(scores, legal, args.temperature)
  top = int(np.argmax(target)) if float(np.sum(target)) > 0.0 else -1
  has_lane, bottleneck, spread = row_kinds(features, legal, scores, args)
  stats = {
      "phase": phase_name(features, legal),
      "target_action_kind": action_kind_name(features, top) if top >= 0 else "none",
      "top_action": top,
      "target_entropy": legal_entropy(target),
      "top_prob": float(target[top]) if top >= 0 else 0.0,
      "structural_spread": spread,
      "lane_contrast": has_lane,
      "bottleneck": bottleneck,
      "top_follows_led": (
          bool(features[top, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5)
          if top >= 0 else False
      ),
      "top_token_loss": (
          bool(safe_feature(features, "token_loss_newly_loses_led")[top] > 0.5)
          if top >= 0 else False
      ),
      "top_red_switch": (
          bool(
              features[top, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX] > 0.5
              and features[top, ACTION_FEATURE_IS_RED_INDEX] > 0.5
          )
          if top >= 0 else False
      ),
      "top_red_win_switch": (
          bool(
              features[top, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX] > 0.5
              and features[top, ACTION_FEATURE_IS_RED_INDEX] > 0.5
              and features[top, ACTION_FEATURE_PREFIX_SIZE + 19] > 0.5
          )
          if top >= 0 else False
      ),
  }
  new_row = (
      row[0],
      row[1],
      target,
      row[3],
      row[4],
      row[5],
      row[6],
      row[7],
      row[8],
      row[9],
      row[10],
      row[11],
      row[12] if len(row) > 12 else None,
  )
  return new_row, stats


def main():
  args = parse_args()
  rows = []
  row_group_ids = []
  input_counts = {}
  input_metadata = []
  for input_idx, path in enumerate(args.input):
    replay, metadata = load_replay(path, return_metadata=True)
    input_counts[str(path)] = len(replay)
    input_metadata.append({"path": str(path), "metadata": metadata})
    rows.extend(replay)
    source_group_ids = list((metadata or {}).get("row_group_ids") or [])
    if len(source_group_ids) != len(replay):
      source_group_ids = [
          f"input{input_idx:02d}:{Path(path).name}:row{row_idx:06d}"
          for row_idx in range(len(replay))
      ]
    row_group_ids.extend(str(group_id) for group_id in source_group_ids)

  relabeled = []
  relabeled_group_ids = []
  stats_rows = []
  for row, group_id in zip(rows, row_group_ids):
    new_row, stats = relabel_row(row, args)
    if new_row is None:
      continue
    relabeled.append(new_row)
    relabeled_group_ids.append(str(group_id))
    stats_rows.append(stats)

  if int(args.max_rows) > 0 and len(relabeled) > int(args.max_rows):
    rng = np.random.default_rng(int(args.seed))
    indices = np.sort(rng.choice(len(relabeled), size=int(args.max_rows), replace=False))
    relabeled = [relabeled[int(idx)] for idx in indices]
    relabeled_group_ids = [relabeled_group_ids[int(idx)] for idx in indices]
    stats_rows = [stats_rows[int(idx)] for idx in indices]

  phase_counts = Counter(row["phase"] for row in stats_rows)
  kind_counts = Counter(row["target_action_kind"] for row in stats_rows)
  summary = {
      "source": "quantum_cat_survival_grammar_replay",
      "inputs": input_counts,
      "input_metadata": input_metadata,
      "out": args.out,
      "rows": len(relabeled),
      "row_filter": str(args.row_filter),
      "max_rows": int(args.max_rows),
      "seed": int(args.seed),
      "temperature": float(args.temperature),
      "prior_weight": float(args.prior_weight),
      "action_value_weight": float(args.action_value_weight),
      "follow_led_bonus": float(args.follow_led_bonus),
      "token_loss_penalty": float(args.token_loss_penalty),
      "red_switch_bonus": float(args.red_switch_bonus),
      "red_switch_win_bonus": float(args.red_switch_win_bonus),
      "nonred_switch_penalty": float(args.nonred_switch_penalty),
      "nonwinning_switch_penalty": float(args.nonwinning_switch_penalty),
      "phase_counts": dict(sorted(phase_counts.items())),
      "target_action_kind_counts": dict(sorted(kind_counts.items())),
      "lane_contrast_rows": int(sum(1 for row in stats_rows if row["lane_contrast"])),
      "bottleneck_rows": int(sum(1 for row in stats_rows if row["bottleneck"])),
      "target_top_follows_led_rows": int(
          sum(1 for row in stats_rows if row["top_follows_led"])
      ),
      "target_top_token_loss_rows": int(
          sum(1 for row in stats_rows if row["top_token_loss"])
      ),
      "target_top_red_switch_rows": int(
          sum(1 for row in stats_rows if row["top_red_switch"])
      ),
      "target_top_red_win_switch_rows": int(
          sum(1 for row in stats_rows if row["top_red_win_switch"])
      ),
      "mean_target_entropy": (
          float(np.mean([row["target_entropy"] for row in stats_rows]))
          if stats_rows else 0.0
      ),
      "mean_top_prob": (
          float(np.mean([row["top_prob"] for row in stats_rows]))
          if stats_rows else 0.0
      ),
      "mean_structural_spread": (
          float(np.mean([row["structural_spread"] for row in stats_rows]))
          if stats_rows else 0.0
      ),
      "row_group_ids": relabeled_group_ids,
      "row_group_id_kind": "source_match_round_or_row",
      "row_group_count": int(len(set(relabeled_group_ids))),
  }
  if not relabeled:
    raise SystemExit(json.dumps(summary, indent=2, sort_keys=True))
  save_replay(relabeled, args.out, metadata=summary)
  text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
  if args.summary_out:
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
