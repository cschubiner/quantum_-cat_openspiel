#!/usr/bin/env python3
"""Train a focused filter for follow-led/no-token-loss rank allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_action_set_transformer_trainer import (
    _append_round_context_features,
    _source_states,
)
from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    ACTION_FEATURE_PREFIX_SIZE,
    ACTION_FEATURE_RANK_NORM_INDEX,
    APPENDED_ACTION_FEATURE_INDEX,
)
from quantum_cat_legal_set_selector_trainer import (
    _build_states,
    _load_replays_and_metadata,
    _parse_int_set,
    _state_summary,
    _state_summary_by_source,
)
from quantum_cat_table_liveness_improve_trainer import _parse_float_list


class IntraLaneRankFilter(nn.Module):
  def __init__(self, feature_size: int, hidden: int, dropout: float):
    super().__init__()
    self.body = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
    )
    self.positive = nn.Linear(hidden, 1)
    self.safe = nn.Linear(hidden, 1)
    self.value = nn.Linear(hidden, 1)
    self.risk = nn.Linear(hidden, 1)

  def forward(self, x: torch.Tensor):
    h = self.body(x)
    return (
        self.positive(h).squeeze(-1),
        self.safe(h).squeeze(-1),
        self.value(h).squeeze(-1),
        self.risk(h).squeeze(-1),
    )


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20261291)
  parser.add_argument("--train-source-indices", default="")
  parser.add_argument("--validation-source-indices", default="3")
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--hidden", type=int, default=128)
  parser.add_argument("--dropout", type=float, default=0.15)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--weight-decay", type=float, default=0.001)
  parser.add_argument(
      "--observation-feature-mode",
      choices=("none", "summary", "raw", "raw_summary"),
      default="summary",
  )
  parser.add_argument(
      "--policy-context-mode",
      choices=("none", "summary"),
      default="summary",
  )
  parser.add_argument(
      "--round-context-mode",
      choices=("none", "summary"),
      default="none",
  )
  parser.add_argument(
      "--base-action-source",
      choices=("metadata_policy_best", "policy_context_argmax"),
      default="policy_context_argmax",
  )
  parser.add_argument(
      "--target-mode",
      choices=("best_positive_else_base", "best_value_positive_else_base"),
      default="best_value_positive_else_base",
  )
  parser.add_argument("--positive-value-margin", type=float, default=0.01)
  parser.add_argument("--safe-risk-delta-max", type=float, default=0.0)
  parser.add_argument("--force-led-lane-burn-negative", action="store_true")
  parser.add_argument(
      "--led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="urgent_win",
  )
  parser.add_argument(
      "--candidate-filter",
      choices=(
          "follow_no_token_loss",
          "follow_led",
          "no_token_loss",
          "switch",
          "red_switch",
          "red_win_switch",
          "all",
      ),
      default="follow_no_token_loss",
  )
  parser.add_argument(
      "--intralane-context-mode",
      choices=("none", "rank_scarcity"),
      default="none",
      help=(
          "Append same-state context over the candidate-filtered action set. "
          "rank_scarcity adds percentiles/z-scores for rank and exit-scarcity "
          "metrics among follow-led/no-token-loss alternatives."
      ),
  )
  parser.add_argument("--positive-loss-weight", type=float, default=1.0)
  parser.add_argument("--safe-loss-weight", type=float, default=0.5)
  parser.add_argument("--value-loss-weight", type=float, default=0.7)
  parser.add_argument("--risk-loss-weight", type=float, default=0.7)
  parser.add_argument("--hard-negative-weight", type=float, default=3.0)
  parser.add_argument("--easy-negative-weight", type=float, default=0.5)
  parser.add_argument("--positive-sample-weight", type=float, default=2.0)
  parser.add_argument("--near-value-margin", type=float, default=0.02)
  parser.add_argument("--near-risk-margin", type=float, default=0.03)
  parser.add_argument(
      "--predicted-value-delta-thresholds",
      default="0.00,0.02,0.04,0.08,0.12,0.16",
  )
  parser.add_argument(
      "--predicted-risk-delta-max-thresholds",
      default="-0.12,-0.08,-0.05,-0.03,-0.01,0.00,0.02",
  )
  parser.add_argument(
      "--positive-prob-thresholds",
      default="0.50,0.70,0.85,0.90,0.95,0.97,0.99",
  )
  parser.add_argument(
      "--safe-prob-thresholds",
      default="0.50,0.70,0.85,0.90,0.95,0.97,0.99",
  )
  parser.add_argument("--selection-risk-penalty", type=float, default=0.5)
  parser.add_argument("--selection-value-bonus", type=float, default=0.05)
  parser.add_argument("--selection-policy-tiebreak", type=float, default=0.01)
  parser.add_argument("--min-validation-overrides-for-live", type=int, default=8)
  parser.add_argument(
      "--min-validation-positive-precision-for-live", type=float, default=0.85
  )
  parser.add_argument("--min-validation-safe-rate-for-live", type=float, default=0.90)
  parser.add_argument(
      "--min-validation-mean-value-delta-for-live", type=float, default=0.0
  )
  parser.add_argument(
      "--max-validation-mean-risk-delta-for-live", type=float, default=0.0
  )
  return parser.parse_args()


def _feature_bool(action_features: np.ndarray, action: int, feature_idx: int) -> bool:
  if int(action) < 0 or int(action) >= int(action_features.shape[0]):
    return False
  if int(feature_idx) < 0 or int(feature_idx) >= int(action_features.shape[1]):
    return False
  return bool(float(action_features[int(action), int(feature_idx)]) > 0.5)


def _token_loss(action_features: np.ndarray, action: int) -> bool:
  return bool(
      _feature_bool(action_features, action, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX)
      or _feature_bool(
          action_features,
          action,
          APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"],
      )
  )


def _candidate_allowed(args, action_features: np.ndarray, action: int) -> bool:
  follows = _feature_bool(action_features, action, ACTION_FEATURE_FOLLOWS_LED_INDEX)
  loses = _token_loss(action_features, action)
  red = _feature_bool(action_features, action, ACTION_FEATURE_IS_RED_INDEX)
  would_win = _feature_bool(
      action_features,
      action,
      ACTION_FEATURE_PREFIX_SIZE + 19,
  )
  mode = str(args.candidate_filter)
  if mode == "follow_no_token_loss":
    return bool(follows and not loses)
  if mode == "follow_led":
    return bool(follows)
  if mode == "no_token_loss":
    return bool(not loses)
  if mode == "switch":
    return bool(loses)
  if mode == "red_switch":
    return bool(loses and red)
  if mode == "red_win_switch":
    return bool(loses and red and would_win)
  return True


INTRALANE_SCARCITY_METRICS = (
    ("rank_norm", ACTION_FEATURE_RANK_NORM_INDEX, False),
    ("exit_min_player_open_slots_after",
     APPENDED_ACTION_FEATURE_INDEX["exit_min_player_open_slots_after"], True),
    ("exit_min_player_lane_surplus_after",
     APPENDED_ACTION_FEATURE_INDEX["exit_min_player_lane_surplus_after"], True),
    ("exit_lane_surplus_damage",
     APPENDED_ACTION_FEATURE_INDEX["exit_lane_surplus_damage"], False),
    ("exit_min_lane_surplus_damage",
     APPENDED_ACTION_FEATURE_INDEX["exit_min_lane_surplus_damage"], False),
    ("exit_lane_pressure_player_count_after",
     APPENDED_ACTION_FEATURE_INDEX["exit_lane_pressure_player_count_after"], False),
    ("own_future_min_colors_after",
     APPENDED_ACTION_FEATURE_INDEX["own_future_min_colors_after"], True),
    ("own_future_zero_exit_frac_after",
     APPENDED_ACTION_FEATURE_INDEX["own_future_zero_exit_frac_after"], False),
    ("own_future_one_exit_frac_after",
     APPENDED_ACTION_FEATURE_INDEX["own_future_one_exit_frac_after"], False),
    ("future_rank_deficit_after",
     APPENDED_ACTION_FEATURE_INDEX["future_rank_deficit_after"], False),
    ("future_rank_deficit_delta",
     APPENDED_ACTION_FEATURE_INDEX["future_rank_deficit_delta"], False),
    ("future_safe_flex_score_after",
     APPENDED_ACTION_FEATURE_INDEX["future_safe_flex_score_after"], True),
)


INTRALANE_CONTEXT_SCHEMA = (
    ["intralane_pool_size_norm", "intralane_pool_rank_span"]
    + [
        f"intralane_{name}_{suffix}"
        for name, _, _ in INTRALANE_SCARCITY_METRICS
        for suffix in ("z", "pct", "good_pct", "best_good", "good_delta_vs_base")
    ]
)


def _percentile(values: np.ndarray, value: float) -> float:
  values = np.asarray(values, dtype=np.float32)
  if values.size <= 0:
    return 0.5
  less = float(np.sum(values < float(value)))
  equal = float(np.sum(values == float(value)))
  return float((less + 0.5 * equal) / float(values.size))


def _intralane_context_vector(
    action_features: np.ndarray,
    action: int,
    base_action: int,
    pool: list[int],
) -> np.ndarray:
  if not pool:
    return np.zeros(len(INTRALANE_CONTEXT_SCHEMA), dtype=np.float32)
  pool_arr = np.asarray(pool, dtype=np.int64)
  rank_values = action_features[pool_arr, ACTION_FEATURE_RANK_NORM_INDEX]
  context = [
      min(1.0, float(len(pool)) / 20.0),
      float(np.max(rank_values) - np.min(rank_values)) if len(rank_values) else 0.0,
  ]
  for _, feature_idx, high_good in INTRALANE_SCARCITY_METRICS:
    values = action_features[pool_arr, int(feature_idx)].astype(np.float32)
    value = float(action_features[int(action), int(feature_idx)])
    base_value = (
        float(action_features[int(base_action), int(feature_idx)])
        if 0 <= int(base_action) < int(action_features.shape[0]) else
        0.0
    )
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values)) if values.size else 1.0
    z = 0.0 if std < 1e-6 else (value - mean) / std
    pct = _percentile(values, value)
    good_values = values if bool(high_good) else -values
    good_value = value if bool(high_good) else -value
    good_pct = _percentile(good_values, good_value)
    if bool(high_good):
      best_good = float(value >= float(np.max(values)) - 1e-6)
      good_delta = value - base_value
    else:
      best_good = float(value <= float(np.min(values)) + 1e-6)
      good_delta = base_value - value
    context.extend([z, pct, good_pct, best_good, good_delta])
  return np.asarray(context, dtype=np.float32)


def _state_row(rows, state: dict):
  return rows[int(state["state_index"])]


def _example_weight(args, action: dict) -> float:
  if bool(action["positive"]):
    return float(args.positive_sample_weight)
  near_value = float(action["value_delta"]) >= -float(args.near_value_margin)
  near_risk = float(action["risk_delta"]) <= float(args.near_risk_margin)
  if near_value and near_risk:
    return float(args.hard_negative_weight)
  return float(args.easy_negative_weight)


def _build_examples(args, rows, states: list[dict]) -> tuple[list[dict], dict]:
  examples = []
  skipped_base = 0
  skipped_filter = 0
  class_counts = {"positive": 0, "safe": 0, "negative": 0, "hard_negative": 0}
  for state in states:
    row = _state_row(rows, state)
    action_features = np.asarray(row[11], dtype=np.float32)
    pool = [
        int(action["action"])
        for action in state["actions"]
        if _candidate_allowed(args, action_features, int(action["action"]))
    ]
    for action in state["actions"]:
      action_idx = int(action["action"])
      if bool(action["is_base"]):
        skipped_base += 1
        continue
      if not _candidate_allowed(args, action_features, action_idx):
        skipped_filter += 1
        continue
      feature = np.asarray(action["feature"], dtype=np.float32)
      if str(args.intralane_context_mode) == "rank_scarcity":
        feature = np.concatenate([
            feature,
            _intralane_context_vector(
                action_features,
                action_idx,
                int(state["base_action"]),
                pool,
            ),
        ]).astype(np.float32)
      elif str(args.intralane_context_mode) != "none":
        raise ValueError(
            f"unknown intralane_context_mode={args.intralane_context_mode}"
        )
      follows = _feature_bool(action_features, action_idx, ACTION_FEATURE_FOLLOWS_LED_INDEX)
      red = _feature_bool(action_features, action_idx, ACTION_FEATURE_IS_RED_INDEX)
      loses = _token_loss(action_features, action_idx)
      hard_negative = (
          not bool(action["positive"])
          and float(action["value_delta"]) >= -float(args.near_value_margin)
          and float(action["risk_delta"]) <= float(args.near_risk_margin)
      )
      class_counts["positive"] += int(bool(action["positive"]))
      class_counts["safe"] += int(bool(action["safe"]))
      class_counts["negative"] += int(not bool(action["positive"]))
      class_counts["hard_negative"] += int(bool(hard_negative))
      examples.append({
          "state_index": int(state["state_index"]),
          "group_key": str(state.get("group_key", "")),
          "source_replay_index": int(state.get("source_replay_index", -1)),
          "action_index_in_state": int(state["actions"].index(action)),
          "action": action_idx,
          "feature": feature,
          "policy_prob": float(action["policy_prob"]),
          "value_delta": float(action["value_delta"]),
          "risk_delta": float(action["risk_delta"]),
          "positive": bool(action["positive"]),
          "safe": bool(action["safe"]),
          "hard_negative": bool(hard_negative),
          "follows_led": bool(follows),
          "red": bool(red),
          "token_loss": bool(loses),
          "nonred_token_loss": bool(loses and not red),
          "sample_weight": float(_example_weight(args, action)),
      })
  return examples, {
      "examples": int(len(examples)),
      "skipped_base_actions": int(skipped_base),
      "skipped_candidate_filter": int(skipped_filter),
      "class_counts": class_counts,
  }


def _standardize_examples(train_examples: list[dict], all_examples: list[dict]):
  train_features = np.asarray([ex["feature"] for ex in train_examples], dtype=np.float32)
  all_features = np.asarray([ex["feature"] for ex in all_examples], dtype=np.float32)
  mean = train_features.mean(axis=0)
  std = train_features.std(axis=0)
  std = np.where(std < 1e-6, 1.0, std)
  all_std = (all_features - mean) / std
  for ex, feature in zip(all_examples, all_std):
    ex["feature_std"] = feature.astype(np.float32)
  return mean.astype(np.float32), std.astype(np.float32)


def _pos_weight(examples: list[dict], key: str) -> torch.Tensor:
  positives = sum(1 for ex in examples if bool(ex[key]))
  total = len(examples)
  negatives = total - positives
  if positives <= 0:
    return torch.tensor(1.0, dtype=torch.float32)
  return torch.tensor(max(1.0, negatives / positives), dtype=torch.float32)


def _train(args, train_examples: list[dict], feature_size: int):
  torch.manual_seed(int(args.seed))
  model = IntraLaneRankFilter(feature_size, int(args.hidden), float(args.dropout))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  positive_pos_weight = _pos_weight(train_examples, "positive")
  safe_pos_weight = _pos_weight(train_examples, "safe")
  rng = random.Random(int(args.seed))
  indices = list(range(len(train_examples)))
  losses = []
  positive_losses = []
  safe_losses = []
  value_losses = []
  risk_losses = []
  batch_size = max(1, int(args.batch_size))
  for _ in range(max(1, int(args.epochs))):
    rng.shuffle(indices)
    totals = [0.0] * 5
    seen = 0
    for start in range(0, len(indices), batch_size):
      batch = [train_examples[idx] for idx in indices[start:start + batch_size]]
      x = torch.tensor(
          np.asarray([ex["feature_std"] for ex in batch], dtype=np.float32),
          dtype=torch.float32,
      )
      positive_t = torch.tensor(
          [float(ex["positive"]) for ex in batch], dtype=torch.float32
      )
      safe_t = torch.tensor([float(ex["safe"]) for ex in batch], dtype=torch.float32)
      value_t = torch.tensor(
          [float(ex["value_delta"]) for ex in batch], dtype=torch.float32
      )
      risk_t = torch.tensor(
          [float(ex["risk_delta"]) for ex in batch], dtype=torch.float32
      )
      weights = torch.tensor(
          [float(ex["sample_weight"]) for ex in batch], dtype=torch.float32
      )
      positive_logit, safe_logit, value_p, risk_p = model(x)
      positive_loss = torch.mean(
          F.binary_cross_entropy_with_logits(
              positive_logit,
              positive_t,
              pos_weight=positive_pos_weight,
              reduction="none",
          )
          * weights
      )
      safe_loss = torch.mean(
          F.binary_cross_entropy_with_logits(
              safe_logit,
              safe_t,
              pos_weight=safe_pos_weight,
              reduction="none",
          )
          * weights
      )
      value_loss = torch.mean(
          F.smooth_l1_loss(value_p, value_t, reduction="none") * weights
      )
      risk_loss = torch.mean(
          F.smooth_l1_loss(risk_p, risk_t, reduction="none") * weights
      )
      loss = (
          float(args.positive_loss_weight) * positive_loss
          + float(args.safe_loss_weight) * safe_loss
          + float(args.value_loss_weight) * value_loss
          + float(args.risk_loss_weight) * risk_loss
      )
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      count = len(batch)
      for idx_total, item in enumerate((
          loss, positive_loss, safe_loss, value_loss, risk_loss
      )):
        totals[idx_total] += float(item.detach()) * count
      seen += count
    denom = max(1, seen)
    losses.append(totals[0] / denom)
    positive_losses.append(totals[1] / denom)
    safe_losses.append(totals[2] / denom)
    value_losses.append(totals[3] / denom)
    risk_losses.append(totals[4] / denom)
  return model, {
      "epochs": int(args.epochs),
      "batch_size": int(args.batch_size),
      "hidden": int(args.hidden),
      "dropout": float(args.dropout),
      "lr": float(args.lr),
      "weight_decay": float(args.weight_decay),
      "positive_pos_weight": round(float(positive_pos_weight.item()), 6),
      "safe_pos_weight": round(float(safe_pos_weight.item()), 6),
      "initial_loss": round(float(losses[0]), 6),
      "final_loss": round(float(losses[-1]), 6),
      "initial_positive_loss": round(float(positive_losses[0]), 6),
      "final_positive_loss": round(float(positive_losses[-1]), 6),
      "initial_safe_loss": round(float(safe_losses[0]), 6),
      "final_safe_loss": round(float(safe_losses[-1]), 6),
      "initial_value_loss": round(float(value_losses[0]), 6),
      "final_value_loss": round(float(value_losses[-1]), 6),
      "initial_risk_loss": round(float(risk_losses[0]), 6),
      "final_risk_loss": round(float(risk_losses[-1]), 6),
  }


def _predict_examples(model, examples: list[dict], batch_size: int) -> None:
  model.eval()
  with torch.no_grad():
    for start in range(0, len(examples), max(1, int(batch_size))):
      batch = examples[start:start + max(1, int(batch_size))]
      x = torch.tensor(
          np.asarray([ex["feature_std"] for ex in batch], dtype=np.float32),
          dtype=torch.float32,
      )
      positive_logit, safe_logit, value_p, risk_p = model(x)
      positive_p = torch.sigmoid(positive_logit).cpu().numpy()
      safe_p = torch.sigmoid(safe_logit).cpu().numpy()
      value_np = value_p.cpu().numpy()
      risk_np = risk_p.cpu().numpy()
      for idx, ex in enumerate(batch):
        ex["pred_positive_prob"] = float(positive_p[idx])
        ex["pred_safe_prob"] = float(safe_p[idx])
        ex["pred_value_delta"] = float(value_np[idx])
        ex["pred_risk_delta"] = float(risk_np[idx])


def _pearson(xs, ys):
  xs = np.asarray(xs, dtype=np.float64)
  ys = np.asarray(ys, dtype=np.float64)
  if xs.size < 2 or float(np.std(xs)) <= 1e-12 or float(np.std(ys)) <= 1e-12:
    return None
  return round(float(np.corrcoef(xs, ys)[0, 1]), 6)


def _example_summary(examples: list[dict]) -> dict:
  if not examples:
    return {
        "examples": 0,
        "positive_examples": 0,
        "safe_examples": 0,
        "hard_negatives": 0,
    }
  return {
      "examples": int(len(examples)),
      "positive_examples": int(sum(1 for ex in examples if bool(ex["positive"]))),
      "safe_examples": int(sum(1 for ex in examples if bool(ex["safe"]))),
      "hard_negatives": int(sum(1 for ex in examples if bool(ex["hard_negative"]))),
      "mean_value_delta": round(float(np.mean([ex["value_delta"] for ex in examples])), 6),
      "mean_risk_delta": round(float(np.mean([ex["risk_delta"] for ex in examples])), 6),
  }


def _prediction_summary(examples: list[dict]) -> dict:
  if not examples or "pred_value_delta" not in examples[0]:
    return {}
  value_true = [float(ex["value_delta"]) for ex in examples]
  value_pred = [float(ex["pred_value_delta"]) for ex in examples]
  risk_true = [float(ex["risk_delta"]) for ex in examples]
  risk_pred = [float(ex["pred_risk_delta"]) for ex in examples]
  return {
      "examples": int(len(examples)),
      "value_corr": _pearson(value_true, value_pred),
      "risk_corr": _pearson(risk_true, risk_pred),
      "positive_mean_prob_on_positive": round(float(np.mean([
          ex["pred_positive_prob"] for ex in examples if bool(ex["positive"])
      ] or [0.0])), 6),
      "positive_mean_prob_on_negative": round(float(np.mean([
          ex["pred_positive_prob"] for ex in examples if not bool(ex["positive"])
      ] or [0.0])), 6),
      "safe_mean_prob_on_safe": round(float(np.mean([
          ex["pred_safe_prob"] for ex in examples if bool(ex["safe"])
      ] or [0.0])), 6),
      "safe_mean_prob_on_unsafe": round(float(np.mean([
          ex["pred_safe_prob"] for ex in examples if not bool(ex["safe"])
      ] or [0.0])), 6),
  }


def _selected_by_state(
    examples: list[dict],
    *,
    value_threshold: float,
    risk_threshold: float,
    positive_threshold: float,
    safe_threshold: float,
    risk_penalty: float,
    value_bonus: float,
    policy_tiebreak: float,
) -> list[dict]:
  by_state: dict[int, list[dict]] = {}
  for ex in examples:
    if float(ex.get("pred_value_delta", 0.0)) < float(value_threshold):
      continue
    if float(ex.get("pred_risk_delta", 0.0)) > float(risk_threshold):
      continue
    if float(ex.get("pred_positive_prob", 0.0)) < float(positive_threshold):
      continue
    if float(ex.get("pred_safe_prob", 0.0)) < float(safe_threshold):
      continue
    by_state.setdefault(int(ex["state_index"]), []).append(ex)
  selected = []
  for state_examples in by_state.values():
    best = max(
        state_examples,
        key=lambda ex: (
            float(ex.get("pred_positive_prob", 0.0))
            + float(value_bonus) * float(ex.get("pred_value_delta", 0.0))
            - float(risk_penalty) * max(0.0, float(ex.get("pred_risk_delta", 0.0)))
            + float(policy_tiebreak) * float(ex["policy_prob"])
        ),
    )
    selected.append(best)
  return selected


def _threshold_report(args, examples: list[dict]) -> list[dict]:
  reports = []
  for value_threshold in _parse_float_list(args.predicted_value_delta_thresholds):
    for risk_threshold in _parse_float_list(args.predicted_risk_delta_max_thresholds):
      for positive_threshold in _parse_float_list(args.positive_prob_thresholds):
        for safe_threshold in _parse_float_list(args.safe_prob_thresholds):
          selected = _selected_by_state(
              examples,
              value_threshold=float(value_threshold),
              risk_threshold=float(risk_threshold),
              positive_threshold=float(positive_threshold),
              safe_threshold=float(safe_threshold),
              risk_penalty=float(args.selection_risk_penalty),
              value_bonus=float(args.selection_value_bonus),
              policy_tiebreak=float(args.selection_policy_tiebreak),
          )
          overrides = len(selected)
          positives = sum(1 for ex in selected if bool(ex["positive"]))
          safe = sum(1 for ex in selected if bool(ex["safe"]))
          reports.append({
              "value_threshold": float(value_threshold),
              "risk_threshold": float(risk_threshold),
              "positive_threshold": float(positive_threshold),
              "safe_threshold": float(safe_threshold),
              "states": int(len({int(ex["state_index"]) for ex in examples})),
              "candidate_examples": int(len(examples)),
              "overrides": int(overrides),
              "coverage": (
                  0.0 if not examples else
                  round(float(overrides / max(1, len({int(ex["state_index"]) for ex in examples}))), 6)
              ),
              "positive_precision": (
                  None if overrides == 0 else round(float(positives / overrides), 6)
              ),
              "safe_rate": None if overrides == 0 else round(float(safe / overrides), 6),
              "mean_value_delta": (
                  None if not selected else
                  round(float(np.mean([ex["value_delta"] for ex in selected])), 6)
              ),
              "mean_risk_delta": (
                  None if not selected else
                  round(float(np.mean([ex["risk_delta"] for ex in selected])), 6)
              ),
              "mean_pred_value_delta": (
                  None if not selected else
                  round(float(np.mean([ex["pred_value_delta"] for ex in selected])), 6)
              ),
              "mean_pred_risk_delta": (
                  None if not selected else
                  round(float(np.mean([ex["pred_risk_delta"] for ex in selected])), 6)
              ),
              "mean_positive_prob": (
                  None if not selected else
                  round(float(np.mean([ex["pred_positive_prob"] for ex in selected])), 6)
              ),
              "mean_safe_prob": (
                  None if not selected else
                  round(float(np.mean([ex["pred_safe_prob"] for ex in selected])), 6)
              ),
          })
  return reports


def _best_report(reports: list[dict]) -> dict | None:
  if not reports:
    return None
  return max(
      reports,
      key=lambda row: (
          row["positive_precision"] is not None,
          row["positive_precision"] or -1.0,
          row["safe_rate"] or -1.0,
          row["overrides"] or -1,
          row["mean_value_delta"] or -999.0,
          -(row["mean_risk_delta"] or 999.0),
      ),
  )


def _eligible_reports(args, reports: list[dict]) -> list[dict]:
  eligible = []
  for row in reports:
    if int(row.get("overrides") or 0) < int(args.min_validation_overrides_for_live):
      continue
    if row.get("positive_precision") is None or float(
        row["positive_precision"]
    ) < float(args.min_validation_positive_precision_for_live):
      continue
    if row.get("safe_rate") is None or float(row["safe_rate"]) < float(
        args.min_validation_safe_rate_for_live
    ):
      continue
    if row.get("mean_value_delta") is None or float(
        row["mean_value_delta"]
    ) < float(args.min_validation_mean_value_delta_for_live):
      continue
    if row.get("mean_risk_delta") is None or float(
        row["mean_risk_delta"]
    ) > float(args.max_validation_mean_risk_delta_for_live):
      continue
    eligible.append(row)
  return eligible


def main():
  args = parse_args()
  rows, metadata_rows, source_reports = _load_replays_and_metadata(args)
  states, skipped_report = _build_states(args, rows, metadata_rows)
  round_context_schema = _append_round_context_features(args, rows, states)
  validation_sources = _parse_int_set(str(args.validation_source_indices))
  all_sources = {
      int(state.get("source_replay_index", -1)) for state in states
  }
  train_sources = (
      _parse_int_set(str(args.train_source_indices))
      if str(args.train_source_indices or "").strip() else
      set(all_sources) - set(validation_sources)
  )
  train_states = _source_states(states, train_sources)
  validation_states = _source_states(states, validation_sources)
  if not train_states or not validation_states:
    raise ValueError("empty train/validation source split")
  train_examples, train_build = _build_examples(args, rows, train_states)
  validation_examples, validation_build = _build_examples(args, rows, validation_states)
  all_examples = train_examples + validation_examples
  if not train_examples or not validation_examples:
    raise ValueError("empty train/validation example split")
  mean, std = _standardize_examples(train_examples, all_examples)
  feature_size = len(all_examples[0]["feature_std"])
  model, train_report = _train(args, train_examples, feature_size)
  _predict_examples(model, all_examples, int(args.batch_size))

  train_thresholds = _threshold_report(args, train_examples)
  validation_thresholds = _threshold_report(args, validation_examples)
  gate_eligible = _eligible_reports(args, validation_thresholds)
  best_gate = _best_report(gate_eligible)

  model_out = str(args.model_out or "")
  if model_out:
    path = Path(model_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "feature_size": int(feature_size),
        "model_class": "IntraLaneRankFilter",
        "args": vars(args),
        "train_report": train_report,
        "replay_sources": source_reports,
        "round_context_schema": round_context_schema,
        "intralane_context_schema": (
            list(INTRALANE_CONTEXT_SCHEMA)
            if str(args.intralane_context_mode) == "rank_scarcity" else
            []
        ),
    }, path)

  result = {
      "replay": list(args.replay),
      "metadata": list(args.metadata or []),
      "replay_sources": source_reports,
      "model_out": model_out,
      "source_split": {
          "train_source_indices": sorted(int(x) for x in train_sources),
          "validation_source_indices": sorted(int(x) for x in validation_sources),
      },
      "round_context_schema": round_context_schema,
      "intralane_context_schema": (
          list(INTRALANE_CONTEXT_SCHEMA)
          if str(args.intralane_context_mode) == "rank_scarcity" else
          []
      ),
      "skipped": skipped_report,
      "train_state_summary": _state_summary(train_states),
      "validation_state_summary": _state_summary(validation_states),
      "train_state_summary_by_source": _state_summary_by_source(train_states),
      "validation_state_summary_by_source": _state_summary_by_source(validation_states),
      "train_example_build": train_build,
      "validation_example_build": validation_build,
      "train_example_summary": _example_summary(train_examples),
      "validation_example_summary": _example_summary(validation_examples),
      "train_prediction_summary": _prediction_summary(train_examples),
      "validation_prediction_summary": _prediction_summary(validation_examples),
      "train_report": train_report,
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "best_validation_threshold": _best_report(validation_thresholds),
      "live_screen_gate": {
          "passed": bool(best_gate is not None),
          "best_threshold": best_gate,
          "eligible_thresholds": int(len(gate_eligible)),
          "min_validation_overrides": int(args.min_validation_overrides_for_live),
          "min_positive_precision": float(
              args.min_validation_positive_precision_for_live
          ),
          "min_safe_rate": float(args.min_validation_safe_rate_for_live),
          "min_mean_value_delta": float(
              args.min_validation_mean_value_delta_for_live
          ),
          "max_mean_risk_delta": float(
              args.max_validation_mean_risk_delta_for_live
          ),
      },
      "feature_size": int(feature_size),
      "args": vars(args),
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
