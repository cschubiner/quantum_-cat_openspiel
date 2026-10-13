#!/usr/bin/env python3
"""Train source-balanced delta heads with an adversarial source classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_delta_calibration_trainer import (
    _best_report,
    _eligible_reports,
    _prediction_summary,
    _predict_states,
    _threshold_report,
)
from quantum_cat_legal_set_selector_trainer import (
    _build_states,
    _load_replays_and_metadata,
    _parse_int_set,
    _standardize_state_features,
    _state_summary,
    _state_summary_by_source,
)


class _GradReverse(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x: torch.Tensor, scale: float):
    ctx.scale = float(scale)
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor):
    return -float(ctx.scale) * grad_output, None


def _grad_reverse(x: torch.Tensor, scale: float):
  return _GradReverse.apply(x, float(scale))


class SourceInvariantDelta(nn.Module):
  def __init__(self, feature_size: int, hidden: int, source_count: int):
    super().__init__()
    self.body = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.ReLU(),
        nn.Dropout(0.08),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )
    self.value = nn.Linear(hidden, 1)
    self.risk = nn.Linear(hidden, 1)
    self.positive = nn.Linear(hidden, 1)
    self.safe = nn.Linear(hidden, 1)
    self.source = nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, source_count),
    )

  def forward(self, x: torch.Tensor):
    h = self.body(x)
    return (
        self.value(h).squeeze(-1),
        self.risk(h).squeeze(-1),
        self.positive(h).squeeze(-1),
        self.safe(h).squeeze(-1),
    )

  def source_logits(self, x: torch.Tensor, reverse_scale: float):
    h = self.body(x)
    return self.source(_grad_reverse(h, reverse_scale))


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20261251)
  parser.add_argument("--train-source-indices", default="")
  parser.add_argument("--validation-source-indices", default="2")
  parser.add_argument("--epochs", type=int, default=160)
  parser.add_argument("--batch-size", type=int, default=2048)
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
  parser.add_argument("--source-adversary-weight", type=float, default=0.1)
  parser.add_argument(
      "--source-balance-weight",
      type=float,
      default=1.0,
      help="0 disables inverse-source sample weighting; 1 fully balances sources.",
  )
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


def _source_states(states: list[dict], sources: set[int]) -> list[dict]:
  return [
      state for state in states
      if int(state.get("source_replay_index", -1)) in sources
  ]


def _flatten_actions(states: list[dict], source_to_label: dict[int, int], balance: float):
  features = []
  value_targets = []
  risk_targets = []
  positive_targets = []
  safe_targets = []
  source_targets = []
  source_counts: dict[int, int] = {}
  for state in states:
    source_idx = int(state.get("source_replay_index", -1))
    for action in state["actions"]:
      features.append(action["feature_std"])
      value_targets.append(float(action["value_delta"]))
      risk_targets.append(float(action["risk_delta"]))
      positive_targets.append(float(action["positive"]))
      safe_targets.append(float(action["safe"]))
      source_targets.append(int(source_to_label[source_idx]))
      source_counts[source_idx] = int(source_counts.get(source_idx, 0)) + 1
  weights = []
  if source_counts:
    mean_count = float(np.mean(list(source_counts.values())))
  else:
    mean_count = 1.0
  for state in states:
    source_idx = int(state.get("source_replay_index", -1))
    count = max(1.0, float(source_counts.get(source_idx, 1)))
    source_weight = mean_count / count
    break
  del state, source_idx, count
  for label in source_targets:
    source_idx = next(
        key for key, value in source_to_label.items() if int(value) == int(label)
    )
    source_weight = mean_count / max(1.0, float(source_counts.get(source_idx, 1)))
    weights.append((1.0 - float(balance)) + float(balance) * source_weight)
  return (
      torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32),
      torch.tensor(value_targets, dtype=torch.float32),
      torch.tensor(risk_targets, dtype=torch.float32),
      torch.tensor(positive_targets, dtype=torch.float32),
      torch.tensor(safe_targets, dtype=torch.float32),
      torch.tensor(source_targets, dtype=torch.long),
      torch.tensor(weights, dtype=torch.float32),
      {str(k): int(v) for k, v in sorted(source_counts.items())},
  )


def _bce_pos_weight(targets: torch.Tensor) -> torch.Tensor:
  positives = float(torch.sum(targets).item())
  negatives = float(targets.numel() - positives)
  if positives <= 0.0:
    return torch.tensor(1.0, dtype=torch.float32)
  return torch.tensor(max(1.0, negatives / positives), dtype=torch.float32)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-6)


def _train(args, train_states: list[dict], feature_size: int, train_sources: list[int]):
  torch.manual_seed(int(args.seed))
  source_to_label = {int(source): idx for idx, source in enumerate(train_sources)}
  model = SourceInvariantDelta(feature_size, int(args.hidden), len(source_to_label))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  (
      x,
      value_t,
      risk_t,
      positive_t,
      safe_t,
      source_t,
      sample_w,
      source_counts,
  ) = _flatten_actions(
      train_states,
      source_to_label,
      float(args.source_balance_weight),
  )
  positive_weight = _bce_pos_weight(positive_t)
  safe_weight = _bce_pos_weight(safe_t)
  rng = torch.Generator().manual_seed(int(args.seed))
  losses = []
  source_losses = []
  supervised_losses = []
  source_accs = []
  batch_size = max(1, int(args.batch_size))
  for _ in range(max(1, int(args.epochs))):
    order = torch.randperm(x.shape[0], generator=rng)
    totals = [0.0, 0.0, 0.0, 0.0]
    seen = 0
    for start in range(0, x.shape[0], batch_size):
      idx = order[start:start + batch_size]
      value_p, risk_p, positive_logit, safe_logit = model(x[idx])
      weights = sample_w[idx]
      value_loss = _weighted_mean(
          F.smooth_l1_loss(value_p, value_t[idx], reduction="none"),
          weights,
      )
      risk_loss = _weighted_mean(
          F.smooth_l1_loss(risk_p, risk_t[idx], reduction="none"),
          weights,
      )
      positive_loss = _weighted_mean(
          F.binary_cross_entropy_with_logits(
              positive_logit,
              positive_t[idx],
              pos_weight=positive_weight,
              reduction="none",
          ),
          weights,
      )
      safe_loss = _weighted_mean(
          F.binary_cross_entropy_with_logits(
              safe_logit,
              safe_t[idx],
              pos_weight=safe_weight,
              reduction="none",
          ),
          weights,
      )
      supervised_loss = (
          float(args.value_loss_weight) * value_loss
          + float(args.risk_loss_weight) * risk_loss
          + float(args.positive_loss_weight) * positive_loss
          + float(args.safe_loss_weight) * safe_loss
      )
      source_logits = model.source_logits(
          x[idx],
          reverse_scale=float(args.source_adversary_weight),
      )
      source_loss = F.cross_entropy(source_logits, source_t[idx])
      loss = supervised_loss + float(args.source_adversary_weight) * source_loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      count = int(idx.numel())
      source_pred = torch.argmax(source_logits.detach(), dim=1)
      source_acc = float(torch.mean((source_pred == source_t[idx]).float()).item())
      totals[0] += float(loss.detach()) * count
      totals[1] += float(supervised_loss.detach()) * count
      totals[2] += float(source_loss.detach()) * count
      totals[3] += source_acc * count
      seen += count
    denom = max(1, seen)
    losses.append(totals[0] / denom)
    supervised_losses.append(totals[1] / denom)
    source_losses.append(totals[2] / denom)
    source_accs.append(totals[3] / denom)
  return model, {
      "epochs": int(args.epochs),
      "batch_size": int(args.batch_size),
      "hidden": int(args.hidden),
      "lr": float(args.lr),
      "weight_decay": float(args.weight_decay),
      "source_adversary_weight": float(args.source_adversary_weight),
      "source_balance_weight": float(args.source_balance_weight),
      "source_counts": source_counts,
      "source_to_label": {str(k): int(v) for k, v in source_to_label.items()},
      "positive_pos_weight": round(float(positive_weight.item()), 6),
      "safe_pos_weight": round(float(safe_weight.item()), 6),
      "initial_loss": round(losses[0], 6),
      "final_loss": round(losses[-1], 6),
      "initial_supervised_loss": round(supervised_losses[0], 6),
      "final_supervised_loss": round(supervised_losses[-1], 6),
      "initial_source_loss": round(source_losses[0], 6),
      "final_source_loss": round(source_losses[-1], 6),
      "initial_source_acc": round(source_accs[0], 6),
      "final_source_acc": round(source_accs[-1], 6),
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
  model, train_report = _train(args, train_states, feature_size, sorted(train_sources))
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
        "model_class": "SourceInvariantDelta",
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
      "source_split": {
          "train_source_indices": sorted(int(x) for x in train_sources),
          "validation_source_indices": sorted(int(x) for x in validation_sources),
      },
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
