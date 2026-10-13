#!/usr/bin/env python3
"""Train action-level value/risk delta heads and audit abstention rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_legal_set_selector_trainer import (
    _build_states,
    _load_replays_and_metadata,
    _parse_int_set,
    _standardize_state_features,
    _state_summary,
    _state_summary_by_source,
)
from quantum_cat_table_liveness_improve_trainer import _parse_float_list


class DeltaCalibrator(nn.Module):
  """Independent per-action predictor for value/risk deltas and gates."""

  def __init__(self, feature_size: int, hidden: int):
    super().__init__()
    self.body = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )
    self.value = nn.Linear(hidden, 1)
    self.risk = nn.Linear(hidden, 1)
    self.positive = nn.Linear(hidden, 1)
    self.safe = nn.Linear(hidden, 1)

  def forward(self, x: torch.Tensor):
    h = self.body(x)
    return (
        self.value(h).squeeze(-1),
        self.risk(h).squeeze(-1),
        self.positive(h).squeeze(-1),
        self.safe(h).squeeze(-1),
    )


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20261211)
  parser.add_argument(
      "--validation-split",
      choices=("state", "metadata_group", "source"),
      default="source",
  )
  parser.add_argument("--validation-source-indices", default="1")
  parser.add_argument("--validation-fraction", type=float, default=0.25)
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


def _split_states(args, states):
  split = str(args.validation_split)
  if split == "source":
    validation_sources = _parse_int_set(str(args.validation_source_indices or ""))
    train_states = [
        state for state in states
        if int(state.get("source_replay_index", -1)) not in validation_sources
    ]
    validation_states = [
        state for state in states
        if int(state.get("source_replay_index", -1)) in validation_sources
    ]
    report = {
        "split": "source",
        "validation_source_indices": sorted(int(x) for x in validation_sources),
        "train_source_indices": sorted({
            int(state.get("source_replay_index", -1)) for state in train_states
        }),
    }
  elif split == "metadata_group":
    from quantum_cat_legal_set_selector_trainer import _split_states as split_states
    train_states, validation_states, report = split_states(args, states)
  else:
    rng = random.Random(int(args.seed))
    shuffled = list(states)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(args.validation_fraction))))
    val_count = min(val_count, len(shuffled) - 1)
    validation_states = shuffled[:val_count]
    train_states = shuffled[val_count:]
    report = {"split": "state", "validation_states": int(len(validation_states))}
  if not train_states or not validation_states:
    raise ValueError("empty train/validation split")
  return train_states, validation_states, report


def _flatten_actions(states):
  features = []
  value_targets = []
  risk_targets = []
  positive_targets = []
  safe_targets = []
  for state in states:
    for action in state["actions"]:
      features.append(action["feature_std"])
      value_targets.append(float(action["value_delta"]))
      risk_targets.append(float(action["risk_delta"]))
      positive_targets.append(float(action["positive"]))
      safe_targets.append(float(action["safe"]))
  return (
      torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32),
      torch.tensor(value_targets, dtype=torch.float32),
      torch.tensor(risk_targets, dtype=torch.float32),
      torch.tensor(positive_targets, dtype=torch.float32),
      torch.tensor(safe_targets, dtype=torch.float32),
  )


def _bce_pos_weight(targets: torch.Tensor) -> torch.Tensor:
  positives = float(torch.sum(targets).item())
  negatives = float(targets.numel() - positives)
  if positives <= 0.0:
    return torch.tensor(1.0, dtype=torch.float32)
  return torch.tensor(max(1.0, negatives / positives), dtype=torch.float32)


def _train(args, train_states, feature_size):
  torch.manual_seed(int(args.seed))
  model = DeltaCalibrator(feature_size, int(args.hidden))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  x, value_t, risk_t, positive_t, safe_t = _flatten_actions(train_states)
  positive_weight = _bce_pos_weight(positive_t)
  safe_weight = _bce_pos_weight(safe_t)
  rng = torch.Generator().manual_seed(int(args.seed))
  losses = []
  value_losses = []
  risk_losses = []
  positive_losses = []
  safe_losses = []
  batch_size = max(1, int(args.batch_size))
  for _ in range(max(1, int(args.epochs))):
    order = torch.randperm(x.shape[0], generator=rng)
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    seen = 0
    for start in range(0, x.shape[0], batch_size):
      idx = order[start:start + batch_size]
      value_p, risk_p, positive_logit, safe_logit = model(x[idx])
      value_loss = F.smooth_l1_loss(value_p, value_t[idx])
      risk_loss = F.smooth_l1_loss(risk_p, risk_t[idx])
      positive_loss = F.binary_cross_entropy_with_logits(
          positive_logit, positive_t[idx], pos_weight=positive_weight
      )
      safe_loss = F.binary_cross_entropy_with_logits(
          safe_logit, safe_t[idx], pos_weight=safe_weight
      )
      loss = (
          float(args.value_loss_weight) * value_loss
          + float(args.risk_loss_weight) * risk_loss
          + float(args.positive_loss_weight) * positive_loss
          + float(args.safe_loss_weight) * safe_loss
      )
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      count = int(idx.numel())
      for idx_total, item in enumerate(
          (loss, value_loss, risk_loss, positive_loss, safe_loss)
      ):
        totals[idx_total] += float(item.detach()) * count
      seen += count
    denom = max(1, seen)
    losses.append(totals[0] / denom)
    value_losses.append(totals[1] / denom)
    risk_losses.append(totals[2] / denom)
    positive_losses.append(totals[3] / denom)
    safe_losses.append(totals[4] / denom)
  return model, {
      "epochs": int(args.epochs),
      "batch_size": int(args.batch_size),
      "hidden": int(args.hidden),
      "lr": float(args.lr),
      "weight_decay": float(args.weight_decay),
      "positive_pos_weight": round(float(positive_weight.item()), 6),
      "safe_pos_weight": round(float(safe_weight.item()), 6),
      "initial_loss": round(losses[0], 6),
      "final_loss": round(losses[-1], 6),
      "initial_value_loss": round(value_losses[0], 6),
      "final_value_loss": round(value_losses[-1], 6),
      "initial_risk_loss": round(risk_losses[0], 6),
      "final_risk_loss": round(risk_losses[-1], 6),
      "initial_positive_loss": round(positive_losses[0], 6),
      "final_positive_loss": round(positive_losses[-1], 6),
      "initial_safe_loss": round(safe_losses[0], 6),
      "final_safe_loss": round(safe_losses[-1], 6),
  }


def _predict_states(model, states, batch_size: int):
  model.eval()
  flat = []
  refs = []
  for state in states:
    for action in state["actions"]:
      flat.append(action["feature_std"])
      refs.append(action)
  x = torch.tensor(np.asarray(flat, dtype=np.float32), dtype=torch.float32)
  with torch.no_grad():
    for start in range(0, len(refs), max(1, batch_size)):
      end = min(len(refs), start + max(1, batch_size))
      value_p, risk_p, positive_logit, safe_logit = model(x[start:end])
      positive_p = torch.sigmoid(positive_logit).cpu().numpy()
      safe_p = torch.sigmoid(safe_logit).cpu().numpy()
      for offset, action in enumerate(refs[start:end]):
        action["pred_value_delta"] = float(value_p[offset].item())
        action["pred_risk_delta"] = float(risk_p[offset].item())
        action["pred_positive_prob"] = float(positive_p[offset])
        action["pred_safe_prob"] = float(safe_p[offset])


def _pearson(xs, ys):
  xs = np.asarray(xs, dtype=np.float64)
  ys = np.asarray(ys, dtype=np.float64)
  if xs.size < 2 or float(np.std(xs)) <= 1e-12 or float(np.std(ys)) <= 1e-12:
    return None
  return round(float(np.corrcoef(xs, ys)[0, 1]), 6)


def _prediction_summary(states):
  value_true = []
  value_pred = []
  risk_true = []
  risk_pred = []
  positive_probs = []
  positive_targets = []
  safe_probs = []
  safe_targets = []
  for state in states:
    for action in state["actions"]:
      value_true.append(float(action["value_delta"]))
      value_pred.append(float(action["pred_value_delta"]))
      risk_true.append(float(action["risk_delta"]))
      risk_pred.append(float(action["pred_risk_delta"]))
      positive_probs.append(float(action["pred_positive_prob"]))
      positive_targets.append(float(action["positive"]))
      safe_probs.append(float(action["pred_safe_prob"]))
      safe_targets.append(float(action["safe"]))
  value_err = np.asarray(value_pred) - np.asarray(value_true)
  risk_err = np.asarray(risk_pred) - np.asarray(risk_true)
  return {
      "actions": int(len(value_true)),
      "value_corr": _pearson(value_true, value_pred),
      "risk_corr": _pearson(risk_true, risk_pred),
      "value_mae": round(float(np.mean(np.abs(value_err))), 6),
      "risk_mae": round(float(np.mean(np.abs(risk_err))), 6),
      "positive_mean_prob_on_positive": round(float(np.mean([
          p for p, y in zip(positive_probs, positive_targets) if y > 0.5
      ] or [0.0])), 6),
      "positive_mean_prob_on_negative": round(float(np.mean([
          p for p, y in zip(positive_probs, positive_targets) if y <= 0.5
      ] or [0.0])), 6),
      "safe_mean_prob_on_safe": round(float(np.mean([
          p for p, y in zip(safe_probs, safe_targets) if y > 0.5
      ] or [0.0])), 6),
      "safe_mean_prob_on_unsafe": round(float(np.mean([
          p for p, y in zip(safe_probs, safe_targets) if y <= 0.5
      ] or [0.0])), 6),
  }


def _selected_action(
    state,
    *,
    value_threshold: float,
    risk_threshold: float,
    positive_threshold: float,
    risk_penalty: float,
    policy_tiebreak: float,
    exclude_forced: bool,
):
  best = None
  best_score = None
  for idx, action in enumerate(state["actions"]):
    if bool(action["is_base"]):
      continue
    if exclude_forced and bool(action["forced_led_lane_negative"]):
      continue
    if float(action["pred_value_delta"]) < value_threshold:
      continue
    if float(action["pred_risk_delta"]) > risk_threshold:
      continue
    if float(action["pred_positive_prob"]) < positive_threshold:
      continue
    score = (
        float(action["pred_value_delta"])
        - float(risk_penalty) * max(0.0, float(action["pred_risk_delta"]))
        + float(policy_tiebreak) * float(action["policy_prob"])
    )
    if best is None or score > float(best_score):
      best = dict(action)
      best["selected_index"] = int(idx)
      best_score = float(score)
  return best


def _threshold_report(args, states):
  reports = []
  value_thresholds = _parse_float_list(args.predicted_value_delta_thresholds)
  risk_thresholds = _parse_float_list(args.predicted_risk_delta_max_thresholds)
  positive_thresholds = _parse_float_list(args.positive_prob_thresholds)
  for value_threshold in value_thresholds:
    for risk_threshold in risk_thresholds:
      for positive_threshold in positive_thresholds:
        overrides = 0
        positives = 0
        safe = 0
        forced = 0
        value_deltas = []
        risk_deltas = []
        pred_values = []
        pred_risks = []
        for state in states:
          selected = _selected_action(
              state,
              value_threshold=float(value_threshold),
              risk_threshold=float(risk_threshold),
              positive_threshold=float(positive_threshold),
              risk_penalty=float(args.selection_risk_penalty),
              policy_tiebreak=float(args.selection_policy_tiebreak),
              exclude_forced=bool(args.exclude_forced_led_lane_negatives_at_selection),
          )
          if selected is None:
            continue
          overrides += 1
          positives += int(bool(selected["positive"]))
          safe += int(bool(selected["safe"]))
          forced += int(bool(selected["forced_led_lane_negative"]))
          value_deltas.append(float(selected["value_delta"]))
          risk_deltas.append(float(selected["risk_delta"]))
          pred_values.append(float(selected["pred_value_delta"]))
          pred_risks.append(float(selected["pred_risk_delta"]))
        precision = positives / overrides if overrides else None
        safe_rate = safe / overrides if overrides else None
        reports.append({
            "value_threshold": float(value_threshold),
            "risk_threshold": float(risk_threshold),
            "positive_threshold": float(positive_threshold),
            "states": int(len(states)),
            "overrides": int(overrides),
            "coverage": round(float(overrides / max(1, len(states))), 6),
            "positive_precision": (
                None if precision is None else round(float(precision), 6)
            ),
            "safe_rate": None if safe_rate is None else round(float(safe_rate), 6),
            "forced_led_lane_negative_selected": int(forced),
            "mean_value_delta": (
                None if not value_deltas else round(float(np.mean(value_deltas)), 6)
            ),
            "mean_risk_delta": (
                None if not risk_deltas else round(float(np.mean(risk_deltas)), 6)
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


def main():
  args = parse_args()
  rows, metadata_rows, source_reports = _load_replays_and_metadata(args)
  states, skipped_report = _build_states(args, rows, metadata_rows)
  train_states, validation_states, split_report = _split_states(args, states)
  mean, std = _standardize_state_features(train_states, validation_states, states)
  feature_size = len(states[0]["actions"][0]["feature_std"])
  model, train_report = _train(args, train_states, feature_size)
  _predict_states(model, states, int(args.batch_size))

  train_thresholds = _threshold_report(args, train_states)
  validation_thresholds = _threshold_report(args, validation_states)
  all_thresholds = _threshold_report(args, states)
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
        "model_class": "DeltaCalibrator",
        "feature_schema_version": "legal_set_pair_v1",
        "args": vars(args),
        "train_report": train_report,
        "replay_sources": source_reports,
    }, path)

  result = {
      "replay": list(args.replay),
      "metadata": list(args.metadata or []),
      "replay_sources": source_reports,
      "model_out": model_out,
      "split_report": split_report,
      "skipped": skipped_report,
      "feature_size": int(feature_size),
      "train_summary": _state_summary(train_states),
      "validation_summary": _state_summary(validation_states),
      "all_summary": _state_summary(states),
      "train_summary_by_source": _state_summary_by_source(train_states),
      "validation_summary_by_source": _state_summary_by_source(validation_states),
      "all_summary_by_source": _state_summary_by_source(states),
      "train_report": train_report,
      "train_prediction_summary": _prediction_summary(train_states),
      "validation_prediction_summary": _prediction_summary(validation_states),
      "all_prediction_summary": _prediction_summary(states),
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "all_thresholds": all_thresholds,
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
          "eligible_thresholds": int(len(gate_eligible)),
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
