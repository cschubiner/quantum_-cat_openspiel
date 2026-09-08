#!/usr/bin/env python3
"""Derive broad policy targets from table-liveness value/risk labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    APPENDED_ACTION_FEATURE_INDEX,
    load_replay,
    save_replay,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--value-weight", type=float, default=1.0)
  parser.add_argument("--risk-weight", type=float, default=1.0)
  parser.add_argument("--temperature", type=float, default=0.55)
  parser.add_argument("--min-labeled-actions", type=int, default=2)
  parser.add_argument("--min-score-spread", type=float, default=0.25)
  parser.add_argument("--top-k", type=int, default=0)
  parser.add_argument(
      "--min-target-mass",
      type=float,
      default=0.0,
      help=(
          "When >0, keep the smallest best-score prefix whose softmax mass "
          "reaches this value before renormalizing. This broadens targets "
          "relative to a one-hot teacher while excluding clear tail actions."
      ),
  )
  parser.add_argument(
      "--anchor-mix",
      type=float,
      default=0.0,
      help="Mix this fraction of the original replay policy into the target.",
  )
  parser.add_argument("--lane-score-weight", type=float, default=0.0)
  parser.add_argument("--follow-led-bonus", type=float, default=0.4)
  parser.add_argument("--red-switch-bonus", type=float, default=0.2)
  parser.add_argument("--token-loss-penalty", type=float, default=0.7)
  parser.add_argument(
      "--keep-action-labels",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Keep dense action value/risk labels for auxiliary reports/training.",
  )
  return parser.parse_args()


def _zscore(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float32)
  scale = float(np.std(values))
  if not math.isfinite(scale) or scale <= 1e-6:
    return values - float(np.mean(values))
  return (values - float(np.mean(values))) / scale


def _top_prefix(indices: np.ndarray, weights: np.ndarray, mass: float) -> np.ndarray:
  if mass <= 0.0 or mass >= 0.999999:
    return indices
  order = np.argsort(-weights)
  cumulative = 0.0
  keep = []
  for pos in order:
    keep.append(int(pos))
    cumulative += float(weights[pos])
    if cumulative >= mass:
      break
  return indices[np.array(sorted(keep), dtype=np.int64)]


def _soft_policy(
    scores: np.ndarray,
    indices: np.ndarray,
    num_actions: int,
    temperature: float,
) -> np.ndarray | None:
  if len(indices) == 0:
    return None
  logits = np.asarray(scores, dtype=np.float32) / max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  weights = np.exp(logits)
  total = float(np.sum(weights))
  if not math.isfinite(total) or total <= 0.0:
    return None
  policy = np.zeros(num_actions, dtype=np.float32)
  policy[indices] = weights / total
  return policy


def _lane_score(action_features: np.ndarray, actions: np.ndarray, args) -> np.ndarray:
  if float(args.lane_score_weight) == 0.0:
    return np.zeros(len(actions), dtype=np.float32)
  rows = np.asarray(action_features, dtype=np.float32)[actions]
  token_loss_idx = APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"]
  follows = rows[:, ACTION_FEATURE_FOLLOWS_LED_INDEX]
  is_red = rows[:, ACTION_FEATURE_IS_RED_INDEX]
  token_loss = rows[:, token_loss_idx]
  red_switch = is_red * token_loss
  score = (
      float(args.follow_led_bonus) * follows
      + float(args.red_switch_bonus) * red_switch
      - float(args.token_loss_penalty) * token_loss
  )
  return float(args.lane_score_weight) * score.astype(np.float32)


def _clear_action_labels(row):
  items = list(row)
  if len(items) > 7 and items[7] is not None:
    items[7] = np.zeros_like(items[7], dtype=np.float32)
  if len(items) > 8 and items[8] is not None:
    items[8] = np.zeros_like(items[8], dtype=np.float32)
  if len(items) > 9 and items[9] is not None:
    items[9] = np.zeros_like(items[9], dtype=np.float32)
  if len(items) > 10 and items[10] is not None:
    items[10] = np.zeros_like(items[10], dtype=np.float32)
  return tuple(items)


def main() -> None:
  args = parse_args()
  rows, metadata = load_replay(args.input, return_metadata=True)
  relabeled = []
  skipped = Counter()
  target_nonzeros = []
  target_entropies = []
  target_max_probs = []
  top_changed = 0
  top_features = Counter()
  score_spreads = []

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
    legal = np.asarray(mask, dtype=np.float32) > 0.0
    risk_labeled = np.asarray(action_mask, dtype=np.float32) > 0.0
    value_labeled = np.asarray(action_value_mask, dtype=np.float32) > 0.0
    labeled = np.flatnonzero(legal & risk_labeled & value_labeled)
    if len(labeled) < int(args.min_labeled_actions):
      skipped["too_few_labeled"] += 1
      relabeled.append(row if args.keep_action_labels else _clear_action_labels(row))
      continue

    values = np.asarray(action_value_targets, dtype=np.float32)[labeled]
    risks = np.asarray(action_targets, dtype=np.float32)[labeled]
    score = (
        float(args.value_weight) * _zscore(values)
        - float(args.risk_weight) * _zscore(risks)
        + _lane_score(action_features, labeled, args)
    )
    spread = float(np.max(score) - np.min(score))
    score_spreads.append(spread)
    if spread < float(args.min_score_spread):
      skipped["low_spread"] += 1
      relabeled.append(row if args.keep_action_labels else _clear_action_labels(row))
      continue

    selected = labeled
    selected_score = score
    if int(args.top_k) > 0 and len(selected) > int(args.top_k):
      order = np.argsort(-selected_score)[:int(args.top_k)]
      order = np.array(sorted(order), dtype=np.int64)
      selected = selected[order]
      selected_score = selected_score[order]
    initial = _soft_policy(
        selected_score, selected, len(policy), float(args.temperature)
    )
    if initial is None:
      skipped["bad_softmax"] += 1
      relabeled.append(row if args.keep_action_labels else _clear_action_labels(row))
      continue
    if float(args.min_target_mass) > 0.0:
      selected = _top_prefix(
          selected,
          initial[selected],
          float(args.min_target_mass),
      )
      # Preserve original selected order while recomputing scores.
      score_lookup = {int(a): float(s) for a, s in zip(labeled, score)}
      selected_score = np.array([score_lookup[int(a)] for a in selected], dtype=np.float32)
    new_policy = _soft_policy(
        selected_score, selected, len(policy), float(args.temperature)
    )
    if new_policy is None:
      skipped["bad_softmax"] += 1
      relabeled.append(row if args.keep_action_labels else _clear_action_labels(row))
      continue

    anchor_mix = min(1.0, max(0.0, float(args.anchor_mix)))
    if anchor_mix > 0.0:
      old = np.asarray(policy, dtype=np.float32)
      old_total = float(np.sum(old))
      if old_total > 0:
        new_policy = (1.0 - anchor_mix) * new_policy + anchor_mix * (old / old_total)
        new_policy /= float(np.sum(new_policy))

    old_top = int(np.argmax(policy))
    new_top = int(np.argmax(new_policy))
    if new_top != old_top:
      top_changed += 1
    if action_features is not None:
      rows_features = np.asarray(action_features, dtype=np.float32)
      top_row = rows_features[new_top]
      if float(top_row[ACTION_FEATURE_FOLLOWS_LED_INDEX]) > 0.5:
        top_features["follows_led"] += 1
      if float(top_row[ACTION_FEATURE_IS_RED_INDEX]) > 0.5:
        top_features["is_red"] += 1
      token_loss_idx = APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"]
      if float(top_row[token_loss_idx]) > 0.5:
        top_features["token_loss"] += 1

    p = new_policy[new_policy > 0]
    target_nonzeros.append(int(len(p)))
    target_max_probs.append(float(np.max(p)))
    target_entropies.append(
        float(-(p * np.log(np.maximum(p, 1e-12))).sum() / max(1e-12, np.log(len(p))))
        if len(p) > 1 else 0.0
    )
    new_mask = np.zeros_like(np.asarray(mask, dtype=np.float32))
    new_mask[new_policy > 0.0] = 1.0
    relabeled.append((
        obs,
        new_mask,
        new_policy.astype(np.float32),
        value,
        paradox,
        action,
        player,
        action_targets if args.keep_action_labels else None,
        action_mask if args.keep_action_labels else None,
        action_value_targets if args.keep_action_labels else None,
        action_value_mask if args.keep_action_labels else None,
        action_features,
        terminal_action_paradox_target,
    ))

  count = max(1, len(target_nonzeros))
  source_group_ids = []
  if isinstance(metadata, dict):
    maybe_group_ids = metadata.get("row_group_ids")
    if isinstance(maybe_group_ids, list) and len(maybe_group_ids) == len(rows):
      source_group_ids = [str(item) for item in maybe_group_ids]
  summary = {
      "kind": "table_liveness_policy_replay",
      "input": str(args.input),
      "out": str(args.out),
      "rows": int(len(rows)),
      "relabeled_rows": int(len(target_nonzeros)),
      "skipped": dict(sorted(skipped.items())),
      "top_changed": int(top_changed),
      "top_changed_rate": round(float(top_changed / count), 6),
      "target_nonzero_mean": round(float(np.mean(target_nonzeros)), 6)
      if target_nonzeros else 0.0,
      "target_entropy_mean": round(float(np.mean(target_entropies)), 6)
      if target_entropies else 0.0,
      "target_max_prob_mean": round(float(np.mean(target_max_probs)), 6)
      if target_max_probs else 0.0,
      "score_spread_mean": round(float(np.mean(score_spreads)), 6)
      if score_spreads else 0.0,
      "score_spread_p50": round(float(np.median(score_spreads)), 6)
      if score_spreads else 0.0,
      "top_feature_rates": {
          key: round(float(value / count), 6)
          for key, value in sorted(top_features.items())
      },
      "settings": vars(args),
      "source_metadata": metadata,
      "row_group_ids": source_group_ids,
      "row_group_id_kind": (
          "source_match_round" if source_group_ids else "unavailable"
      ),
      "row_group_count": (
          int(len(set(source_group_ids))) if source_group_ids else 0
      ),
  }
  save_replay(relabeled, args.out, metadata=summary)
  summary_path = (
      Path(args.summary_out)
      if args.summary_out else
      Path(args.out).with_suffix(Path(args.out).suffix + ".summary.json")
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
