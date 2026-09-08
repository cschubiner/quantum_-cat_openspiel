#!/usr/bin/env python3
"""Train/evaluate a red-switch-only selector on repeat-validated banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_legal_set_selector_trainer import (
    _build_states,
    _load_replays_and_metadata,
)


class RedSwitchRanker(nn.Module):
  def __init__(self, feature_size: int, hidden: int, dropout: float):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
    )
    self.positive = nn.Linear(hidden, 1)
    self.value = nn.Linear(hidden, 1)
    self.risk = nn.Linear(hidden, 1)

  def forward(self, x: torch.Tensor):
    h = self.net(x)
    return (
        self.positive(h).squeeze(-1),
        self.value(h).squeeze(-1),
        self.risk(h).squeeze(-1),
    )


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--summary", required=True)
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20262590)
  parser.add_argument("--epochs", type=int, default=180)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--hidden", type=int, default=128)
  parser.add_argument("--dropout", type=float, default=0.10)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--positive-weight", type=float, default=2.0)
  parser.add_argument("--value-weight", type=float, default=0.5)
  parser.add_argument("--risk-weight", type=float, default=0.5)
  parser.add_argument("--thresholds", default="0.00,0.25,0.40,0.50,0.60,0.70,0.80,0.90")
  return parser.parse_args()


def _metadata_pairs(summary_path: Path):
  summary = json.load(open(summary_path))
  paths = []
  for key in ("existing_metadata", "batch_metadata"):
    paths.extend(summary.get(key, []))
  pairs = []
  for metadata in paths:
    data = json.load(open(metadata))
    if data.get("rows"):
      pairs.append((str(metadata).removesuffix(".metadata.json"), str(metadata)))
  return pairs


def _trainer_args(args, pairs):
  return SimpleNamespace(
      replay=[replay for replay, _ in pairs],
      metadata=[metadata for _, metadata in pairs],
      policy_context_checkpoint=str(args.policy_context_checkpoint),
      players=int(args.players),
      base_action_source="metadata_policy_best",
      positive_value_margin=0.01,
      safe_risk_delta_max=0.0,
      force_led_lane_burn_negative=False,
      led_lane_red_exception="urgent_win",
      observation_feature_mode="summary",
      policy_context_mode="summary",
      target_mode="best_value_positive_else_base",
      soft_target_temperature=0.35,
      soft_target_risk_penalty=1.0,
      soft_target_unsafe_penalty=1.0,
  )


def _outcome_map(metadata_rows, state):
  meta = metadata_rows[int(state["state_index"])]
  return {int(row["action"]): row for row in meta.get("action_outcomes", [])}


def _red_examples(states, metadata_rows):
  examples = []
  for state in states:
    outcomes = _outcome_map(metadata_rows, state)
    for action in state["actions"]:
      if bool(action["is_base"]):
        continue
      outcome = outcomes.get(int(action["action"]), {})
      if not bool(outcome.get("is_red_switch")):
        continue
      examples.append({
          "source": int(state["source_replay_index"]),
          "state_index": int(state["state_index"]),
          "action": int(action["action"]),
          "feature": np.asarray(action["feature"], dtype=np.float32),
          "positive": float(bool(action["positive"])),
          "safe": float(bool(action["safe"])),
          "value_delta": float(action["value_delta"]),
          "risk_delta": float(action["risk_delta"]),
          "policy_prob": float(action["policy_prob"]),
          "passed": float(bool(outcome.get("repeat_validation_passed"))),
      })
  if not examples:
    raise ValueError("no red-switch examples")
  return examples


def _standardize(train_examples, all_examples):
  train = np.stack([row["feature"] for row in train_examples]).astype(np.float32)
  mean = train.mean(axis=0)
  std = train.std(axis=0)
  std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
  for row in all_examples:
    row["feature_std"] = ((row["feature"] - mean) / std).astype(np.float32)
  return mean, std


def _train(args, train_examples, feature_size):
  random.seed(int(args.seed))
  np.random.seed(int(args.seed))
  torch.manual_seed(int(args.seed))
  model = RedSwitchRanker(feature_size, int(args.hidden), float(args.dropout))
  opt = torch.optim.AdamW(
      model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
  )
  positives = sum(float(row["positive"]) for row in train_examples)
  negatives = max(1.0, float(len(train_examples)) - positives)
  pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))])
  x = torch.tensor(np.stack([row["feature_std"] for row in train_examples]), dtype=torch.float32)
  y_pos = torch.tensor([row["positive"] for row in train_examples], dtype=torch.float32)
  y_val = torch.tensor([row["value_delta"] for row in train_examples], dtype=torch.float32)
  y_risk = torch.tensor([row["risk_delta"] for row in train_examples], dtype=torch.float32)
  indices = np.arange(len(train_examples))
  history = []
  for epoch in range(int(args.epochs)):
    np.random.shuffle(indices)
    total = 0.0
    batches = 0
    for start in range(0, len(indices), int(args.batch_size)):
      idx = torch.tensor(indices[start:start + int(args.batch_size)], dtype=torch.long)
      logits, pred_val, pred_risk = model(x[idx])
      loss = (
          float(args.positive_weight)
          * F.binary_cross_entropy_with_logits(logits, y_pos[idx], pos_weight=pos_weight)
          + float(args.value_weight) * F.mse_loss(pred_val, y_val[idx])
          + float(args.risk_weight) * F.mse_loss(pred_risk, y_risk[idx])
      )
      opt.zero_grad()
      loss.backward()
      opt.step()
      total += float(loss.detach())
      batches += 1
    if epoch in {0, int(args.epochs) - 1} or (epoch + 1) % 30 == 0:
      history.append({"epoch": int(epoch + 1), "loss": round(total / max(1, batches), 6)})
  return model, history


def _predict(model, examples):
  x = torch.tensor(np.stack([row["feature_std"] for row in examples]), dtype=torch.float32)
  with torch.no_grad():
    logits, pred_value, pred_risk = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()
    pred_value = pred_value.cpu().numpy()
    pred_risk = pred_risk.cpu().numpy()
  for row, prob, value, risk in zip(examples, probs, pred_value, pred_risk):
    row["pred_positive_prob"] = float(prob)
    row["pred_value_delta"] = float(value)
    row["pred_risk_delta"] = float(risk)


def _threshold_report(examples, states, thresholds):
  by_state = {}
  for row in examples:
    by_state.setdefault(int(row["state_index"]), []).append(row)
  reports = []
  for threshold in thresholds:
    overrides = positives = safe = passed = 0
    value_deltas = []
    risk_deltas = []
    pred_probs = []
    for state in states:
      candidates = by_state.get(int(state["state_index"]), [])
      candidates = [
          row for row in candidates
          if float(row.get("pred_positive_prob", 0.0)) >= float(threshold)
      ]
      if not candidates:
        continue
      selected = max(
          candidates,
          key=lambda row: (
              float(row.get("pred_positive_prob", 0.0)),
              float(row.get("pred_value_delta", 0.0)),
              -float(row.get("pred_risk_delta", 0.0)),
              float(row.get("policy_prob", 0.0)),
          ),
      )
      overrides += 1
      positives += int(bool(selected["positive"]))
      safe += int(bool(selected["safe"]))
      passed += int(bool(selected["passed"]))
      value_deltas.append(float(selected["value_delta"]))
      risk_deltas.append(float(selected["risk_delta"]))
      pred_probs.append(float(selected["pred_positive_prob"]))
    reports.append({
        "threshold": float(threshold),
        "states": int(len(states)),
        "overrides": int(overrides),
        "coverage": round(overrides / max(1, len(states)), 6),
        "positive_precision": None if overrides == 0 else round(positives / overrides, 6),
        "safe_rate": None if overrides == 0 else round(safe / overrides, 6),
        "passed_rate": None if overrides == 0 else round(passed / overrides, 6),
        "mean_value_delta": None if not value_deltas else round(float(np.mean(value_deltas)), 6),
        "mean_risk_delta": None if not risk_deltas else round(float(np.mean(risk_deltas)), 6),
        "mean_pred_positive_prob": None if not pred_probs else round(float(np.mean(pred_probs)), 6),
  })
  return reports


def _best_report(reports):
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


def _eligible(report):
  return (
      int(report.get("overrides") or 0) >= 8
      and report.get("positive_precision") is not None
      and float(report["positive_precision"]) >= 0.85
      and report.get("safe_rate") is not None
      and float(report["safe_rate"]) >= 0.90
      and report.get("mean_value_delta") is not None
      and float(report["mean_value_delta"]) >= 0.0
      and report.get("mean_risk_delta") is not None
      and float(report["mean_risk_delta"]) <= 0.0
  )


def _parse_thresholds(text: str):
  return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def main():
  args = parse_args()
  pairs = _metadata_pairs(Path(args.summary))
  load_args = _trainer_args(args, pairs)
  rows, metadata_rows, source_reports = _load_replays_and_metadata(load_args)
  states, skipped = _build_states(load_args, rows, metadata_rows)
  examples = _red_examples(states, metadata_rows)
  folds = [(0, 39), (39, 79), (79, 119)]
  thresholds = _parse_thresholds(args.thresholds)
  fold_reports = []
  for fold_index, (start, stop) in enumerate(folds):
    validation_states = [
        state for state in states
        if start <= int(state["source_replay_index"]) < stop
    ]
    train_examples = [
        row for row in examples
        if not (start <= int(row["source"]) < stop)
    ]
    validation_examples = [
        row for row in examples
        if start <= int(row["source"]) < stop
    ]
    _standardize(train_examples, train_examples + validation_examples)
    model, history = _train(args, train_examples, len(train_examples[0]["feature"]))
    _predict(model, train_examples)
    _predict(model, validation_examples)
    train_states = [
        state for state in states
        if not (start <= int(state["source_replay_index"]) < stop)
    ]
    train_thresholds = _threshold_report(train_examples, train_states, thresholds)
    validation_thresholds = _threshold_report(
        validation_examples, validation_states, thresholds
    )
    best_validation = _best_report(validation_thresholds)
    fold_reports.append({
        "fold_index": int(fold_index),
        "validation_source_range": f"{start}-{stop - 1}",
        "train_examples": int(len(train_examples)),
        "validation_examples": int(len(validation_examples)),
        "train_states": int(len(train_states)),
        "validation_states": int(len(validation_states)),
        "history": history,
        "best_train_threshold": _best_report(train_thresholds),
        "best_validation_threshold": best_validation,
        "live_gate_eligible": bool(_eligible(best_validation)),
        "validation_thresholds": validation_thresholds,
    })
  result = {
      "summary": str(args.summary),
      "source_count": int(len(source_reports)),
      "states": int(len(states)),
      "red_examples": int(len(examples)),
      "skipped": skipped,
      "args": vars(args),
      "fold_reports": fold_reports,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
      "out": str(out),
      "states": result["states"],
      "red_examples": result["red_examples"],
      "best_validation": [
          row["best_validation_threshold"] for row in fold_reports
      ],
      "eligible": [row["live_gate_eligible"] for row in fold_reports],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
