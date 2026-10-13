#!/usr/bin/env python3
"""Train a direct true-red versus false-red boundary classifier."""

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
)
from quantum_cat_table_liveness_improve_trainer import _parse_float_list


class RedBoundaryMLP(nn.Module):
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
    self.value = nn.Linear(hidden, 1)

  def forward(self, x: torch.Tensor):
    h = self.body(x)
    return self.positive(h).squeeze(-1), self.value(h).squeeze(-1)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20261278)
  parser.add_argument("--train-source-indices", default="")
  parser.add_argument("--validation-source-indices", default="7,8,9")
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--hidden", type=int, default=64)
  parser.add_argument("--dropout", type=float, default=0.15)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--weight-decay", type=float, default=1e-3)
  parser.add_argument("--positive-value-margin", type=float, default=0.01)
  parser.add_argument("--safe-risk-delta-max", type=float, default=0.0)
  parser.add_argument(
      "--allow-value-only-action-labels",
      action="store_true",
      default=True,
  )
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
      default="metadata_policy_best",
  )
  parser.add_argument(
      "--target-mode",
      choices=(
          "best_positive_else_base",
          "best_value_positive_else_base",
          "group_relative_soft",
      ),
      default="group_relative_soft",
  )
  parser.add_argument("--soft-target-temperature", type=float, default=0.35)
  parser.add_argument("--soft-target-risk-penalty", type=float, default=1.0)
  parser.add_argument("--soft-target-unsafe-penalty", type=float, default=1.0)
  parser.add_argument("--force-led-lane-burn-negative", action="store_true")
  parser.add_argument(
      "--led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="urgent_win",
  )
  parser.add_argument("--positive-loss-weight", type=float, default=1.0)
  parser.add_argument("--value-loss-weight", type=float, default=0.5)
  parser.add_argument("--positive-example-weight", type=float, default=1.0)
  parser.add_argument("--base-example-weight", type=float, default=1.0)
  parser.add_argument(
      "--source-balance-weight",
      type=float,
      default=0.0,
      help="0 disables inverse-source weighting; 1 fully balances source rows.",
  )
  parser.add_argument(
      "--class-balance",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--positive-prob-thresholds",
      default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99",
  )
  parser.add_argument(
      "--predicted-value-delta-thresholds",
      default="-1.00,-0.50,0.00,0.25,0.50,0.75,1.00,1.25,1.50",
  )
  parser.add_argument("--min-validation-overrides", type=int, default=8)
  parser.add_argument("--min-positive-precision", type=float, default=0.85)
  parser.add_argument("--min-positive-recall", type=float, default=0.25)
  parser.add_argument("--min-decision-accuracy", type=float, default=0.75)
  parser.add_argument("--min-mean-value-delta", type=float, default=0.0)
  return parser.parse_args()


def _source_states(states: list[dict], sources: set[int]) -> list[dict]:
  return [
      state for state in states
      if int(state.get("source_replay_index", -1)) in sources
  ]


def _red_candidate(state: dict) -> dict | None:
  candidates = [action for action in state["actions"] if not bool(action["is_base"])]
  if len(candidates) != 1:
    return None
  return candidates[0]


def _build_boundary_examples(states: list[dict], metadata_rows: list[dict]):
  examples = []
  skipped = 0
  for state in states:
    action = _red_candidate(state)
    if action is None:
      skipped += 1
      continue
    meta = metadata_rows[int(state["state_index"])]
    flags = meta.get("candidate_flags", {}).get(str(int(action["action"])), {})
    base_flags = meta.get("base_flags", {})
    if not (
        bool(base_flags.get("follows_led"))
        and bool(flags.get("red_switch"))
        and bool(flags.get("token_loss"))
    ):
      skipped += 1
      continue
    examples.append({
        "state_index": int(state["state_index"]),
        "source_replay_index": int(state.get("source_replay_index", -1)),
        "feature_std": np.asarray(action["feature_std"], dtype=np.float32),
        "positive": float(bool(action["positive"])),
        "value_delta": float(action["value_delta"]),
        "policy_prob": float(action["policy_prob"]),
        "action": int(action["action"]),
        "base_action": int(state["base_action"]),
        "action_string": flags.get("action_string"),
        "base_action_string": base_flags.get("action_string"),
        "trick_number": meta.get("trick_number"),
        "led_color": meta.get("led_color"),
        "legal_count": meta.get("legal_count"),
        "row_group_id": meta.get("row_group_id"),
    })
  return examples, skipped


def _summary(examples: list[dict]):
  positives = int(sum(int(bool(row["positive"])) for row in examples))
  by_source: dict[str, dict[str, int]] = {}
  for row in examples:
    source = str(int(row["source_replay_index"]))
    bucket = by_source.setdefault(source, {"rows": 0, "positives": 0, "base": 0})
    bucket["rows"] += 1
    if bool(row["positive"]):
      bucket["positives"] += 1
    else:
      bucket["base"] += 1
  return {
      "rows": int(len(examples)),
      "positives": int(positives),
      "base": int(len(examples) - positives),
      "by_source": by_source,
  }


def _example_weights(args, examples: list[dict]) -> torch.Tensor:
  source_counts: dict[int, int] = {}
  for row in examples:
    source = int(row["source_replay_index"])
    source_counts[source] = int(source_counts.get(source, 0)) + 1
  mean_source_count = (
      float(np.mean(list(source_counts.values()))) if source_counts else 1.0
  )
  weights = []
  for row in examples:
    label_weight = (
        float(args.positive_example_weight)
        if bool(row["positive"]) else
        float(args.base_example_weight)
    )
    source_count = max(1.0, float(source_counts.get(int(row["source_replay_index"]), 1)))
    source_weight = mean_source_count / source_count
    mixed_source_weight = (
        (1.0 - float(args.source_balance_weight))
        + float(args.source_balance_weight) * source_weight
    )
    weights.append(label_weight * mixed_source_weight)
  return torch.tensor(weights, dtype=torch.float32)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-6)


def _train(args, train_examples: list[dict], feature_size: int):
  torch.manual_seed(int(args.seed))
  model = RedBoundaryMLP(feature_size, int(args.hidden), float(args.dropout))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  x = torch.tensor(
      np.asarray([row["feature_std"] for row in train_examples], dtype=np.float32),
      dtype=torch.float32,
  )
  positive_t = torch.tensor(
      [float(row["positive"]) for row in train_examples],
      dtype=torch.float32,
  )
  value_t = torch.tensor(
      [float(row["value_delta"]) for row in train_examples],
      dtype=torch.float32,
  )
  example_w = _example_weights(args, train_examples)
  if bool(args.class_balance):
    positives = float(torch.sum(positive_t).item())
    negatives = float(positive_t.numel() - positives)
    pos_weight = torch.tensor(
        1.0 if positives <= 0.0 else max(1.0, negatives / positives),
        dtype=torch.float32,
    )
  else:
    pos_weight = torch.tensor(1.0, dtype=torch.float32)
  rng = torch.Generator().manual_seed(int(args.seed))
  batch_size = max(1, int(args.batch_size))
  losses = []
  positive_losses = []
  value_losses = []
  for _ in range(max(1, int(args.epochs))):
    order = torch.randperm(x.shape[0], generator=rng)
    totals = [0.0, 0.0, 0.0]
    seen = 0
    for start in range(0, int(x.shape[0]), batch_size):
      idx = order[start:start + batch_size]
      positive_logit, value_p = model(x[idx])
      positive_loss = F.binary_cross_entropy_with_logits(
          positive_logit,
          positive_t[idx],
          pos_weight=pos_weight,
          reduction="none",
      )
      positive_loss = _weighted_mean(positive_loss, example_w[idx])
      value_loss = _weighted_mean(
          F.smooth_l1_loss(value_p, value_t[idx], reduction="none"),
          example_w[idx],
      )
      loss = (
          float(args.positive_loss_weight) * positive_loss
          + float(args.value_loss_weight) * value_loss
      )
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      count = int(idx.numel())
      totals[0] += float(loss.detach()) * count
      totals[1] += float(positive_loss.detach()) * count
      totals[2] += float(value_loss.detach()) * count
      seen += count
    denom = max(1, seen)
    losses.append(totals[0] / denom)
    positive_losses.append(totals[1] / denom)
    value_losses.append(totals[2] / denom)
  return model, {
      "epochs": int(args.epochs),
      "batch_size": int(args.batch_size),
      "hidden": int(args.hidden),
      "dropout": float(args.dropout),
      "lr": float(args.lr),
      "weight_decay": float(args.weight_decay),
      "pos_weight": round(float(pos_weight.item()), 6),
      "positive_example_weight": float(args.positive_example_weight),
      "base_example_weight": float(args.base_example_weight),
      "source_balance_weight": float(args.source_balance_weight),
      "initial_loss": round(float(losses[0]), 6),
      "final_loss": round(float(losses[-1]), 6),
      "initial_positive_loss": round(float(positive_losses[0]), 6),
      "final_positive_loss": round(float(positive_losses[-1]), 6),
      "initial_value_loss": round(float(value_losses[0]), 6),
      "final_value_loss": round(float(value_losses[-1]), 6),
  }


def _predict(model, examples: list[dict], batch_size: int):
  model.eval()
  if not examples:
    return
  x = torch.tensor(
      np.asarray([row["feature_std"] for row in examples], dtype=np.float32),
      dtype=torch.float32,
  )
  with torch.no_grad():
    for start in range(0, len(examples), max(1, int(batch_size))):
      end = min(len(examples), start + max(1, int(batch_size)))
      positive_logit, value_p = model(x[start:end])
      positive_p = torch.sigmoid(positive_logit).cpu().numpy()
      for offset, row in enumerate(examples[start:end]):
        row["pred_positive_prob"] = float(positive_p[offset])
        row["pred_value_delta"] = float(value_p[offset].item())


def _threshold_report(args, examples: list[dict]):
  reports = []
  total_positive = int(sum(int(bool(row["positive"])) for row in examples))
  total_base = int(len(examples) - total_positive)
  for positive_threshold in _parse_float_list(args.positive_prob_thresholds):
    for value_threshold in _parse_float_list(args.predicted_value_delta_thresholds):
      selected = [
          row for row in examples
          if float(row["pred_positive_prob"]) >= float(positive_threshold)
          and float(row["pred_value_delta"]) >= float(value_threshold)
      ]
      selected_ids = {id(row) for row in selected}
      positives = int(sum(int(bool(row["positive"])) for row in selected))
      false_base = int(len(selected) - positives)
      base_abstains = int(sum(
          int(not bool(row["positive"])) for row in examples
          if id(row) not in selected_ids
      ))
      correct = positives + base_abstains
      reports.append({
          "positive_threshold": float(positive_threshold),
          "value_threshold": float(value_threshold),
          "states": int(len(examples)),
          "positive_states": int(total_positive),
          "base_target_states": int(total_base),
          "overrides": int(len(selected)),
          "coverage": round(float(len(selected) / max(1, len(examples))), 6),
          "positive_precision": (
              None if not selected else round(float(positives / len(selected)), 6)
          ),
          "positive_recall": round(float(positives / max(1, total_positive)), 6),
          "base_target_abstain_rate": round(
              float(base_abstains / max(1, total_base)), 6
          ),
          "false_base_overrides": int(false_base),
          "decision_accuracy": round(float(correct / max(1, len(examples))), 6),
          "mean_value_delta": (
              None if not selected
              else round(float(np.mean([row["value_delta"] for row in selected])), 6)
          ),
          "mean_pred_value_delta": (
              None if not selected
              else round(
                  float(np.mean([row["pred_value_delta"] for row in selected])),
                  6,
              )
          ),
          "mean_positive_prob": (
              None if not selected
              else round(
                  float(np.mean([row["pred_positive_prob"] for row in selected])),
                  6,
              )
          ),
      })
  return reports


def _eligible_reports(args, reports: list[dict]):
  eligible = []
  for row in reports:
    if int(row.get("overrides") or 0) < int(args.min_validation_overrides):
      continue
    if row.get("positive_precision") is None or float(
        row["positive_precision"]
    ) < float(args.min_positive_precision):
      continue
    if float(row.get("positive_recall") or 0.0) < float(args.min_positive_recall):
      continue
    if float(row.get("decision_accuracy") or 0.0) < float(args.min_decision_accuracy):
      continue
    if row.get("mean_value_delta") is None or float(
        row["mean_value_delta"]
    ) < float(args.min_mean_value_delta):
      continue
    eligible.append(row)
  return eligible


def _best_report(reports: list[dict]):
  if not reports:
    return None
  return max(
      reports,
      key=lambda row: (
          row["positive_precision"] is not None,
          row["positive_precision"] or -1.0,
          row["decision_accuracy"] or -1.0,
          row["positive_recall"] or -1.0,
          row["overrides"] or -1,
          row["mean_value_delta"] or -999.0,
      ),
  )


def _prediction_summary(examples: list[dict]):
  if not examples:
    return {"rows": 0}
  targets = np.asarray([float(row["positive"]) for row in examples], dtype=np.float64)
  probs = np.asarray(
      [float(row["pred_positive_prob"]) for row in examples],
      dtype=np.float64,
  )
  values = np.asarray([float(row["value_delta"]) for row in examples], dtype=np.float64)
  pred_values = np.asarray(
      [float(row["pred_value_delta"]) for row in examples],
      dtype=np.float64,
  )
  corr = None
  if len(values) >= 2 and float(np.std(values)) > 1e-12 and float(np.std(pred_values)) > 1e-12:
    corr = round(float(np.corrcoef(values, pred_values)[0, 1]), 6)
  return {
      "rows": int(len(examples)),
      "positive_mean_prob_on_positive": round(
          float(np.mean(probs[targets > 0.5])) if bool(np.any(targets > 0.5)) else 0.0,
          6,
      ),
      "positive_mean_prob_on_base": round(
          float(np.mean(probs[targets <= 0.5])) if bool(np.any(targets <= 0.5)) else 0.0,
          6,
      ),
      "value_corr": corr,
      "value_mae": round(float(np.mean(np.abs(pred_values - values))), 6),
  }


def main():
  args = parse_args()
  rows, metadata_rows, source_reports = _load_replays_and_metadata(args)
  states, skipped_report = _build_states(args, rows, metadata_rows)
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
  mean, std = _standardize_state_features(train_states, validation_states, states)
  feature_size = len(states[0]["actions"][0]["feature_std"])
  train_examples, train_skipped = _build_boundary_examples(train_states, metadata_rows)
  validation_examples, validation_skipped = _build_boundary_examples(
      validation_states, metadata_rows
  )
  all_examples, all_skipped = _build_boundary_examples(states, metadata_rows)
  if not train_examples or not validation_examples:
    raise ValueError("empty train/validation boundary examples")
  rng = random.Random(int(args.seed))
  rng.shuffle(train_examples)
  model, train_report = _train(args, train_examples, feature_size)
  _predict(model, train_examples, int(args.batch_size))
  _predict(model, validation_examples, int(args.batch_size))
  _predict(model, all_examples, int(args.batch_size))
  train_thresholds = _threshold_report(args, train_examples)
  validation_thresholds = _threshold_report(args, validation_examples)
  all_thresholds = _threshold_report(args, all_examples)
  eligible = _eligible_reports(args, validation_thresholds)
  best_gate = _best_report(eligible)
  model_out = str(args.model_out or "")
  if model_out:
    path = Path(model_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "feature_size": int(feature_size),
        "model_class": "RedBoundaryMLP",
        "feature_schema_version": "red_switch_boundary_pair_v1",
        "args": vars(args),
        "train_report": train_report,
        "replay_sources": source_reports,
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
      "skipped_states": skipped_report,
      "boundary_skipped": {
          "train": int(train_skipped),
          "validation": int(validation_skipped),
          "all": int(all_skipped),
      },
      "feature_size": int(feature_size),
      "train_summary": _summary(train_examples),
      "validation_summary": _summary(validation_examples),
      "all_summary": _summary(all_examples),
      "train_report": train_report,
      "train_prediction_summary": _prediction_summary(train_examples),
      "validation_prediction_summary": _prediction_summary(validation_examples),
      "all_prediction_summary": _prediction_summary(all_examples),
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "all_thresholds": all_thresholds,
      "best_validation_threshold": _best_report(validation_thresholds),
      "boundary_gate": {
          "passed": bool(best_gate is not None),
          "min_validation_overrides": int(args.min_validation_overrides),
          "min_positive_precision": float(args.min_positive_precision),
          "min_positive_recall": float(args.min_positive_recall),
          "min_decision_accuracy": float(args.min_decision_accuracy),
          "min_mean_value_delta": float(args.min_mean_value_delta),
          "eligible_thresholds": int(len(eligible)),
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
