#!/usr/bin/env python3
"""Train an approval boundary from source-holdout selected false positives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_delta_calibration_trainer import (
    _predict_states as _predict_delta_states,
    _prediction_summary as _delta_prediction_summary,
    _selected_action as _delta_selected_action,
    _train as _train_delta,
)
from quantum_cat_legal_set_selector_trainer import (
    _build_states,
    _load_replays_and_metadata,
    _parse_int_set,
    _standardize_state_features,
    _state_summary,
    _state_summary_by_source,
)
from quantum_cat_table_liveness_improve_trainer import _parse_float_list


class ApprovalBoundary(nn.Module):
  """Approves one candidate action relative to the state's base action."""

  def __init__(self, feature_size: int, hidden: int):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(feature_size * 3, hidden),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )

  def forward(self, candidate: torch.Tensor, base: torch.Tensor):
    x = torch.cat([candidate, base, candidate - base], dim=-1)
    return self.net(x).squeeze(-1)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20261231)
  parser.add_argument("--stage1-train-source-indices", default="0")
  parser.add_argument("--mine-source-indices", default="1")
  parser.add_argument("--validation-source-indices", default="2")
  parser.add_argument(
      "--crossfit-bidirectional-mining",
      action="store_true",
      help=(
          "Mine selected false positives in both directions: stage1 sources -> "
          "mine sources and mine sources -> stage1 sources. Then train the "
          "final stage1 model on both non-validation source sets."
      ),
  )
  parser.add_argument("--epochs", type=int, default=120)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--hidden", type=int, default=128)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--positive-value-margin", type=float, default=0.01)
  parser.add_argument("--safe-risk-delta-max", type=float, default=0.0)
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
      "--base-action-source",
      choices=("metadata_policy_best", "policy_context_argmax"),
      default="policy_context_argmax",
  )
  parser.add_argument(
      "--target-mode",
      choices=("best_positive_else_base", "best_value_positive_else_base"),
      default="best_value_positive_else_base",
  )
  parser.add_argument("--force-led-lane-burn-negative", action="store_true")
  parser.add_argument(
      "--led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="urgent_win",
  )
  parser.add_argument("--value-loss-weight", type=float, default=1.0)
  parser.add_argument("--risk-loss-weight", type=float, default=1.0)
  parser.add_argument("--positive-loss-weight", type=float, default=0.7)
  parser.add_argument("--safe-loss-weight", type=float, default=0.4)
  parser.add_argument(
      "--predicted-value-delta-thresholds",
      default="0.00,0.01,0.02,0.04,0.06,0.08,0.10,0.14",
  )
  parser.add_argument(
      "--predicted-risk-delta-max-thresholds",
      default="-0.05,-0.03,-0.02,-0.01,0.00,0.01,0.02",
  )
  parser.add_argument(
      "--positive-prob-thresholds",
      default="0.50,0.70,0.85,0.90,0.95,0.97,0.99",
  )
  parser.add_argument("--selection-risk-penalty", type=float, default=1.0)
  parser.add_argument("--selection-policy-tiebreak", type=float, default=0.01)
  parser.add_argument(
      "--exclude-forced-led-lane-negatives-at-selection",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--approval-epochs", type=int, default=160)
  parser.add_argument("--approval-batch-size", type=int, default=1024)
  parser.add_argument("--approval-hidden", type=int, default=128)
  parser.add_argument("--approval-lr", type=float, default=0.001)
  parser.add_argument("--approval-weight-decay", type=float, default=1e-4)
  parser.add_argument("--approval-positive-weight", type=float, default=1.0)
  parser.add_argument("--approval-base-negative-weight", type=float, default=0.15)
  parser.add_argument("--approval-forced-negative-weight", type=float, default=0.5)
  parser.add_argument(
      "--approval-false-positive-negative-weight",
      type=float,
      default=2.0,
  )
  parser.add_argument("--approval-max-selection-count-weight", type=float, default=5.0)
  parser.add_argument(
      "--approval-prob-thresholds",
      default="0.50,0.70,0.85,0.90,0.95,0.97,0.99",
  )
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


def _source_states(states: list[dict], sources: set[int]) -> list[dict]:
  return [
      state for state in states
      if int(state.get("source_replay_index", -1)) in sources
  ]


def _base_action(state: dict) -> dict:
  for action in state["actions"]:
    if bool(action["is_base"]):
      return action
  raise ValueError("base action missing")


def _threshold_grid(args):
  for value_threshold in _parse_float_list(args.predicted_value_delta_thresholds):
    for risk_threshold in _parse_float_list(
        args.predicted_risk_delta_max_thresholds
    ):
      for positive_threshold in _parse_float_list(args.positive_prob_thresholds):
        yield (
            float(value_threshold),
            float(risk_threshold),
            float(positive_threshold),
        )


def _select_with_stage1(args, state, value_threshold, risk_threshold, positive_threshold):
  return _delta_selected_action(
      state,
      value_threshold=float(value_threshold),
      risk_threshold=float(risk_threshold),
      positive_threshold=float(positive_threshold),
      risk_penalty=float(args.selection_risk_penalty),
      policy_tiebreak=float(args.selection_policy_tiebreak),
      exclude_forced=bool(args.exclude_forced_led_lane_negatives_at_selection),
  )


def _mine_stage1_selections(args, states: list[dict]):
  selected: dict[tuple[int, int], dict] = {}
  total_selections = 0
  positive_selections = 0
  safe_selections = 0
  for value_threshold, risk_threshold, positive_threshold in _threshold_grid(args):
    for state in states:
      action = _select_with_stage1(
          args, state, value_threshold, risk_threshold, positive_threshold
      )
      if action is None:
        continue
      total_selections += 1
      positive_selections += int(bool(action["positive"]))
      safe_selections += int(bool(action["safe"]))
      key = (int(state["state_index"]), int(action["action"]))
      row = selected.setdefault(key, {
          "state_index": int(state["state_index"]),
          "source_replay_index": int(state.get("source_replay_index", -1)),
          "action": int(action["action"]),
          "positive": bool(action["positive"]),
          "safe": bool(action["safe"]),
          "forced_led_lane_negative": bool(
              action.get("forced_led_lane_negative", False)
          ),
          "value_delta": float(action["value_delta"]),
          "risk_delta": float(action["risk_delta"]),
          "count": 0,
          "min_value_threshold": float(value_threshold),
          "max_risk_threshold": float(risk_threshold),
          "min_positive_threshold": float(positive_threshold),
      })
      row["count"] = int(row["count"]) + 1
      row["min_value_threshold"] = min(
          float(row["min_value_threshold"]), float(value_threshold)
      )
      row["max_risk_threshold"] = max(
          float(row["max_risk_threshold"]), float(risk_threshold)
      )
      row["min_positive_threshold"] = min(
          float(row["min_positive_threshold"]), float(positive_threshold)
      )
  false_positive = {
      key: row for key, row in selected.items() if not bool(row["positive"])
  }
  true_positive = {
      key: row for key, row in selected.items() if bool(row["positive"])
  }
  return {
      "selected": selected,
      "false_positive": false_positive,
      "true_positive": true_positive,
      "summary": {
          "states": int(len(states)),
          "threshold_combinations": int(len(list(_threshold_grid(args)))),
          "total_stage1_selections": int(total_selections),
          "stage1_selection_positive_rate": (
              None if total_selections == 0 else
              round(float(positive_selections / total_selections), 6)
          ),
          "stage1_selection_safe_rate": (
              None if total_selections == 0 else
              round(float(safe_selections / total_selections), 6)
          ),
          "unique_selected_actions": int(len(selected)),
          "unique_selected_true_positives": int(len(true_positive)),
          "unique_selected_false_positives": int(len(false_positive)),
      },
  }


def _merge_mined(mined_items: list[dict]):
  selected: dict[tuple[int, int], dict] = {}
  for mined in mined_items:
    for key, row in mined["selected"].items():
      if key not in selected:
        selected[key] = dict(row)
        continue
      merged = selected[key]
      merged["count"] = int(merged["count"]) + int(row["count"])
      merged["positive"] = bool(merged["positive"]) or bool(row["positive"])
      merged["safe"] = bool(merged["safe"]) or bool(row["safe"])
      merged["min_value_threshold"] = min(
          float(merged["min_value_threshold"]), float(row["min_value_threshold"])
      )
      merged["max_risk_threshold"] = max(
          float(merged["max_risk_threshold"]), float(row["max_risk_threshold"])
      )
      merged["min_positive_threshold"] = min(
          float(merged["min_positive_threshold"]),
          float(row["min_positive_threshold"]),
      )
  false_positive = {
      key: row for key, row in selected.items() if not bool(row["positive"])
  }
  true_positive = {
      key: row for key, row in selected.items() if bool(row["positive"])
  }
  total_selections = sum(
      int(item["summary"]["total_stage1_selections"]) for item in mined_items
  )
  positive_selections = 0
  safe_selections = 0
  for item in mined_items:
    summary = item["summary"]
    if summary["stage1_selection_positive_rate"] is not None:
      positive_selections += int(round(
          float(summary["stage1_selection_positive_rate"])
          * int(summary["total_stage1_selections"])
      ))
    if summary["stage1_selection_safe_rate"] is not None:
      safe_selections += int(round(
          float(summary["stage1_selection_safe_rate"])
          * int(summary["total_stage1_selections"])
      ))
  return {
      "selected": selected,
      "false_positive": false_positive,
      "true_positive": true_positive,
      "summary": {
          "states": int(sum(int(item["summary"]["states"]) for item in mined_items)),
          "threshold_combinations": int(
              mined_items[0]["summary"]["threshold_combinations"]
              if mined_items else 0
          ),
          "directions": int(len(mined_items)),
          "total_stage1_selections": int(total_selections),
          "stage1_selection_positive_rate": (
              None if total_selections == 0 else
              round(float(positive_selections / total_selections), 6)
          ),
          "stage1_selection_safe_rate": (
              None if total_selections == 0 else
              round(float(safe_selections / total_selections), 6)
          ),
          "unique_selected_actions": int(len(selected)),
          "unique_selected_true_positives": int(len(true_positive)),
          "unique_selected_false_positives": int(len(false_positive)),
      },
  }


def _build_approval_examples(args, states: list[dict], mined: dict):
  false_positive = mined["false_positive"]
  true_positive = mined["true_positive"]
  features = []
  bases = []
  labels = []
  weights = []
  class_counts = {
      "positive": 0,
      "selected_true_positive": 0,
      "selected_false_positive": 0,
      "base_negative": 0,
      "forced_negative": 0,
  }
  seen_examples: set[tuple[int, int, str]] = set()
  for state in states:
    base = _base_action(state)
    base_feature = np.asarray(base["feature_std"], dtype=np.float32)
    state_index = int(state["state_index"])
    if float(args.approval_base_negative_weight) > 0.0:
      key = (state_index, int(base["action"]), "base")
      seen_examples.add(key)
      features.append(base["feature_std"])
      bases.append(base_feature)
      labels.append(0.0)
      weights.append(float(args.approval_base_negative_weight))
      class_counts["base_negative"] += 1
    for action in state["actions"]:
      action_key = (state_index, int(action["action"]))
      if bool(action["positive"]):
        key = (state_index, int(action["action"]), "positive")
        if key not in seen_examples:
          seen_examples.add(key)
          features.append(action["feature_std"])
          bases.append(base_feature)
          labels.append(1.0)
          boost = 1.0 + (1.0 if action_key in true_positive else 0.0)
          weights.append(float(args.approval_positive_weight) * boost)
          class_counts["positive"] += 1
          class_counts["selected_true_positive"] += int(action_key in true_positive)
      if action_key in false_positive:
        key = (state_index, int(action["action"]), "selected_false_positive")
        if key not in seen_examples:
          seen_examples.add(key)
          count_weight = min(
              float(args.approval_max_selection_count_weight),
              float(false_positive[action_key]["count"]),
          )
          features.append(action["feature_std"])
          bases.append(base_feature)
          labels.append(0.0)
          weights.append(
              float(args.approval_false_positive_negative_weight) * count_weight
          )
          class_counts["selected_false_positive"] += 1
      elif bool(action.get("forced_led_lane_negative", False)) and float(
          args.approval_forced_negative_weight
      ) > 0.0:
        key = (state_index, int(action["action"]), "forced")
        if key not in seen_examples:
          seen_examples.add(key)
          features.append(action["feature_std"])
          bases.append(base_feature)
          labels.append(0.0)
          weights.append(float(args.approval_forced_negative_weight))
          class_counts["forced_negative"] += 1
  if not features:
    raise ValueError("no approval examples built")
  label_array = np.asarray(labels, dtype=np.float32)
  return {
      "x": torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32),
      "base": torch.tensor(np.asarray(bases, dtype=np.float32), dtype=torch.float32),
      "y": torch.tensor(label_array, dtype=torch.float32),
      "weight": torch.tensor(np.asarray(weights, dtype=np.float32), dtype=torch.float32),
      "summary": {
          **class_counts,
          "examples": int(len(features)),
          "positive_examples": int(np.sum(label_array > 0.5)),
          "negative_examples": int(np.sum(label_array <= 0.5)),
          "mean_weight": round(float(np.mean(weights)), 6),
          "max_weight": round(float(np.max(weights)), 6),
      },
  }


def _train_approval(args, states: list[dict], mined: dict, feature_size: int):
  torch.manual_seed(int(args.seed) + 1009)
  model = ApprovalBoundary(feature_size, int(args.approval_hidden))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.approval_lr),
      weight_decay=float(args.approval_weight_decay),
  )
  tensors = _build_approval_examples(args, states, mined)
  x = tensors["x"]
  base = tensors["base"]
  y = tensors["y"]
  weights = tensors["weight"]
  positives = float(torch.sum(y).item())
  negatives = float(y.numel() - positives)
  pos_weight = torch.tensor(
      max(1.0, negatives / max(1.0, positives)),
      dtype=torch.float32,
  )
  rng = torch.Generator().manual_seed(int(args.seed) + 1013)
  losses = []
  batch_size = max(1, int(args.approval_batch_size))
  for _ in range(max(1, int(args.approval_epochs))):
    order = torch.randperm(x.shape[0], generator=rng)
    total = 0.0
    seen = 0
    for start in range(0, x.shape[0], batch_size):
      idx = order[start:start + batch_size]
      logits = model(x[idx], base[idx])
      loss_per = F.binary_cross_entropy_with_logits(
          logits,
          y[idx],
          pos_weight=pos_weight,
          reduction="none",
      )
      loss = torch.mean(loss_per * weights[idx])
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      count = int(idx.numel())
      total += float(loss.detach()) * count
      seen += count
    losses.append(total / max(1, seen))
  return model, {
      **tensors["summary"],
      "epochs": int(args.approval_epochs),
      "batch_size": int(args.approval_batch_size),
      "hidden": int(args.approval_hidden),
      "lr": float(args.approval_lr),
      "weight_decay": float(args.approval_weight_decay),
      "pos_weight": round(float(pos_weight.item()), 6),
      "initial_loss": round(float(losses[0]), 6),
      "final_loss": round(float(losses[-1]), 6),
  }


def _predict_approval_states(model, states: list[dict], batch_size: int):
  refs = []
  features = []
  bases = []
  for state in states:
    base = _base_action(state)
    base_feature = np.asarray(base["feature_std"], dtype=np.float32)
    for action in state["actions"]:
      refs.append(action)
      features.append(action["feature_std"])
      bases.append(base_feature)
  x = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
  b = torch.tensor(np.asarray(bases, dtype=np.float32), dtype=torch.float32)
  model.eval()
  with torch.no_grad():
    for start in range(0, len(refs), max(1, int(batch_size))):
      end = min(len(refs), start + max(1, int(batch_size)))
      logits = model(x[start:end], b[start:end])
      probs = torch.sigmoid(logits).cpu().numpy()
      for offset, action in enumerate(refs[start:end]):
        action["approval_prob"] = float(probs[offset])


def _stage1_threshold_report(args, states: list[dict]):
  reports = []
  for value_threshold, risk_threshold, positive_threshold in _threshold_grid(args):
    overrides = 0
    positives = 0
    safe = 0
    value_deltas = []
    risk_deltas = []
    for state in states:
      selected = _select_with_stage1(
          args, state, value_threshold, risk_threshold, positive_threshold
      )
      if selected is None:
        continue
      overrides += 1
      positives += int(bool(selected["positive"]))
      safe += int(bool(selected["safe"]))
      value_deltas.append(float(selected["value_delta"]))
      risk_deltas.append(float(selected["risk_delta"]))
    reports.append({
        "value_threshold": float(value_threshold),
        "risk_threshold": float(risk_threshold),
        "positive_threshold": float(positive_threshold),
        "states": int(len(states)),
        "overrides": int(overrides),
        "coverage": round(float(overrides / max(1, len(states))), 6),
        "positive_precision": (
            None if overrides == 0 else round(float(positives / overrides), 6)
        ),
        "safe_rate": None if overrides == 0 else round(float(safe / overrides), 6),
        "mean_value_delta": (
            None if not value_deltas else round(float(np.mean(value_deltas)), 6)
        ),
        "mean_risk_delta": (
            None if not risk_deltas else round(float(np.mean(risk_deltas)), 6)
        ),
    })
  return reports


def _approval_threshold_report(args, states: list[dict]):
  reports = []
  for value_threshold, risk_threshold, positive_threshold in _threshold_grid(args):
    for approval_threshold in _parse_float_list(args.approval_prob_thresholds):
      stage1_overrides = 0
      overrides = 0
      positives = 0
      safe = 0
      value_deltas = []
      risk_deltas = []
      approval_probs = []
      pred_values = []
      pred_risks = []
      for state in states:
        selected = _select_with_stage1(
            args, state, value_threshold, risk_threshold, positive_threshold
        )
        if selected is None:
          continue
        stage1_overrides += 1
        if float(selected.get("approval_prob", 0.0)) < float(approval_threshold):
          continue
        overrides += 1
        positives += int(bool(selected["positive"]))
        safe += int(bool(selected["safe"]))
        value_deltas.append(float(selected["value_delta"]))
        risk_deltas.append(float(selected["risk_delta"]))
        approval_probs.append(float(selected["approval_prob"]))
        pred_values.append(float(selected["pred_value_delta"]))
        pred_risks.append(float(selected["pred_risk_delta"]))
      reports.append({
          "value_threshold": float(value_threshold),
          "risk_threshold": float(risk_threshold),
          "positive_threshold": float(positive_threshold),
          "approval_threshold": float(approval_threshold),
          "states": int(len(states)),
          "stage1_overrides": int(stage1_overrides),
          "overrides": int(overrides),
          "coverage": round(float(overrides / max(1, len(states))), 6),
          "positive_precision": (
              None if overrides == 0 else round(float(positives / overrides), 6)
          ),
          "safe_rate": None if overrides == 0 else round(float(safe / overrides), 6),
          "mean_value_delta": (
              None if not value_deltas else round(float(np.mean(value_deltas)), 6)
          ),
          "mean_risk_delta": (
              None if not risk_deltas else round(float(np.mean(risk_deltas)), 6)
          ),
          "mean_approval_prob": (
              None if not approval_probs else round(float(np.mean(approval_probs)), 6)
          ),
          "mean_pred_value_delta": (
              None if not pred_values else round(float(np.mean(pred_values)), 6)
          ),
          "mean_pred_risk_delta": (
              None if not pred_risks else round(float(np.mean(pred_risks)), 6)
          ),
      })
  return reports


def _best_report(reports):
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


def _eligible_reports(args, reports):
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


def _source_report(states: list[dict]):
  return {
      "summary": _state_summary(states),
      "by_source": _state_summary_by_source(states),
  }


def main():
  args = parse_args()
  stage1_sources = _parse_int_set(str(args.stage1_train_source_indices))
  mine_sources = _parse_int_set(str(args.mine_source_indices))
  validation_sources = _parse_int_set(str(args.validation_source_indices))
  if not stage1_sources or not mine_sources or not validation_sources:
    raise ValueError("stage1, mine, and validation source sets are required")

  rows, metadata_rows, source_reports = _load_replays_and_metadata(args)
  states, skipped_report = _build_states(args, rows, metadata_rows)
  stage1_train_states = _source_states(states, stage1_sources)
  mine_states = _source_states(states, mine_sources)
  validation_states = _source_states(states, validation_sources)
  if not stage1_train_states or not mine_states or not validation_states:
    raise ValueError("empty source split")

  final_stage1_train_states = (
      stage1_train_states + mine_states
      if bool(args.crossfit_bidirectional_mining) else
      stage1_train_states
  )
  mean, std = _standardize_state_features(
      final_stage1_train_states,
      validation_states,
      states,
  )
  feature_size = len(states[0]["actions"][0]["feature_std"])
  mining_reports = []
  if bool(args.crossfit_bidirectional_mining):
    mining_model_a, mining_report_a = _train_delta(
        args, stage1_train_states, feature_size
    )
    _predict_delta_states(mining_model_a, mine_states, int(args.batch_size))
    mined_a = _mine_stage1_selections(args, mine_states)
    mining_reports.append({
        "train_sources": sorted(stage1_sources),
        "mine_sources": sorted(mine_sources),
        "delta_train_report": mining_report_a,
        "mined_summary": mined_a["summary"],
    })
    mining_model_b, mining_report_b = _train_delta(args, mine_states, feature_size)
    _predict_delta_states(
        mining_model_b, stage1_train_states, int(args.batch_size)
    )
    mined_b = _mine_stage1_selections(args, stage1_train_states)
    mining_reports.append({
        "train_sources": sorted(mine_sources),
        "mine_sources": sorted(stage1_sources),
        "delta_train_report": mining_report_b,
        "mined_summary": mined_b["summary"],
    })
    mined = _merge_mined([mined_a, mined_b])
  else:
    mining_model, mining_report = _train_delta(
        args, stage1_train_states, feature_size
    )
    _predict_delta_states(mining_model, mine_states, int(args.batch_size))
    mined = _mine_stage1_selections(args, mine_states)
    mining_reports.append({
        "train_sources": sorted(stage1_sources),
        "mine_sources": sorted(mine_sources),
        "delta_train_report": mining_report,
        "mined_summary": mined["summary"],
    })

  delta_model, delta_train_report = _train_delta(
      args, final_stage1_train_states, feature_size
  )
  _predict_delta_states(delta_model, states, int(args.batch_size))
  approval_train_states = stage1_train_states + mine_states
  approval_model, approval_train_report = _train_approval(
      args, approval_train_states, mined, feature_size
  )
  _predict_approval_states(approval_model, states, int(args.approval_batch_size))

  train_stage1_thresholds = _stage1_threshold_report(args, stage1_train_states)
  mine_stage1_thresholds = _stage1_threshold_report(args, mine_states)
  validation_stage1_thresholds = _stage1_threshold_report(args, validation_states)
  train_thresholds = _approval_threshold_report(args, stage1_train_states)
  mine_thresholds = _approval_threshold_report(args, mine_states)
  validation_thresholds = _approval_threshold_report(args, validation_states)
  gate = _eligible_reports(args, validation_thresholds)
  best_gate = _best_report(gate)

  model_out = str(args.model_out or "")
  if model_out:
    path = Path(model_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "delta_model_state": delta_model.state_dict(),
        "approval_model_state": approval_model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "feature_size": int(feature_size),
        "model_class": "DeltaCalibratorPlusApprovalBoundary",
        "args": vars(args),
        "delta_train_report": delta_train_report,
        "approval_train_report": approval_train_report,
        "mined_summary": mined["summary"],
        "replay_sources": source_reports,
    }, path)

  result = {
      "replay": list(args.replay),
      "metadata": list(args.metadata or []),
      "replay_sources": source_reports,
      "model_out": model_out,
      "source_split": {
          "stage1_train_source_indices": sorted(stage1_sources),
          "mine_source_indices": sorted(mine_sources),
          "validation_source_indices": sorted(validation_sources),
          "crossfit_bidirectional_mining": bool(args.crossfit_bidirectional_mining),
          "final_stage1_train_source_indices": sorted(
              stage1_sources | mine_sources
              if bool(args.crossfit_bidirectional_mining) else
              stage1_sources
          ),
      },
      "skipped": skipped_report,
      "feature_size": int(feature_size),
      "stage1_train": _source_report(stage1_train_states),
      "mine": _source_report(mine_states),
      "validation": _source_report(validation_states),
      "all": _source_report(states),
      "delta_train_report": delta_train_report,
      "mining_reports": mining_reports,
      "delta_train_prediction_summary": _delta_prediction_summary(stage1_train_states),
      "delta_mine_prediction_summary": _delta_prediction_summary(mine_states),
      "delta_validation_prediction_summary": _delta_prediction_summary(
          validation_states
      ),
      "mined_summary": mined["summary"],
      "approval_train_report": approval_train_report,
      "train_stage1_thresholds": train_stage1_thresholds,
      "mine_stage1_thresholds": mine_stage1_thresholds,
      "validation_stage1_thresholds": validation_stage1_thresholds,
      "best_validation_stage1_threshold": _best_report(validation_stage1_thresholds),
      "train_thresholds": train_thresholds,
      "mine_thresholds": mine_thresholds,
      "validation_thresholds": validation_thresholds,
      "best_validation_threshold": _best_report(validation_thresholds),
      "live_screen_gate": {
          "passed": bool(best_gate is not None),
          "min_validation_overrides": int(args.min_validation_overrides_for_live),
          "min_positive_precision": float(
              args.min_validation_positive_precision_for_live
          ),
          "min_safe_rate": float(args.min_validation_safe_rate_for_live),
          "min_mean_value_delta": float(
              args.min_validation_mean_value_delta_for_live
          ),
          "max_mean_risk_delta": float(args.max_validation_mean_risk_delta_for_live),
          "eligible_thresholds": int(len(gate)),
          "best_threshold": best_gate,
      },
      "args": vars(args),
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
