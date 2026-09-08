#!/usr/bin/env python3
"""Train a pairwise hard-negative boundary for table-liveness overrides."""

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


class PairwiseBoundary(nn.Module):
  """Scores one candidate action relative to the state's base action."""

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
  parser.add_argument("--seed", type=int, default=20261221)
  parser.add_argument(
      "--validation-split",
      choices=("source", "state"),
      default="source",
  )
  parser.add_argument("--validation-source-indices", default="1")
  parser.add_argument("--validation-fraction", type=float, default=0.25)
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
  parser.add_argument(
      "--max-hard-negatives-per-state",
      type=int,
      default=6,
      help="Hard negatives kept per state, in addition to the base action.",
  )
  parser.add_argument(
      "--negative-min-value-delta",
      type=float,
      default=-0.02,
      help="Keep non-positive candidates above this value delta as lookalikes.",
  )
  parser.add_argument(
      "--negative-max-risk-delta",
      type=float,
      default=0.10,
      help="Keep non-positive candidates below this risk delta as lookalikes.",
  )
  parser.add_argument("--margin-loss-weight", type=float, default=1.0)
  parser.add_argument("--classification-loss-weight", type=float, default=0.25)
  parser.add_argument(
      "--score-margin-thresholds",
      default="-1.0,-0.5,0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0",
  )
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
  if str(args.validation_split) == "source":
    validation_sources = _parse_int_set(str(args.validation_source_indices or ""))
    train_states = [
        state for state in states
        if int(state.get("source_replay_index", -1)) not in validation_sources
    ]
    validation_states = [
        state for state in states
        if int(state.get("source_replay_index", -1)) in validation_sources
    ]
    split_report = {
        "split": "source",
        "validation_source_indices": sorted(int(x) for x in validation_sources),
        "train_source_indices": sorted({
            int(state.get("source_replay_index", -1)) for state in train_states
        }),
    }
  else:
    rng = random.Random(int(args.seed))
    shuffled = list(states)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(args.validation_fraction))))
    val_count = min(val_count, len(shuffled) - 1)
    validation_states = shuffled[:val_count]
    train_states = shuffled[val_count:]
    split_report = {"split": "state", "validation_states": int(len(validation_states))}
  if not train_states or not validation_states:
    raise ValueError("empty train/validation split")
  return train_states, validation_states, split_report


def _base_action(state: dict) -> dict:
  for action in state["actions"]:
    if bool(action["is_base"]):
      return action
  raise ValueError("base action missing")


def _hard_negative_candidates(state: dict, args) -> list[dict]:
  candidates = []
  for action in state["actions"]:
    if bool(action["is_base"]) or bool(action["positive"]):
      continue
    if (
        float(action["value_delta"]) >= float(args.negative_min_value_delta)
        or float(action["risk_delta"]) <= float(args.negative_max_risk_delta)
        or bool(action["safe"])
        or float(action["policy_prob"]) > 0.05
    ):
      candidates.append(action)
  candidates.sort(
      key=lambda action: (
          float(action["value_delta"]),
          -float(action["risk_delta"]),
          float(action["policy_prob"]),
          -float(action.get("forced_led_lane_negative", False)),
      ),
      reverse=True,
  )
  return candidates[:max(0, int(args.max_hard_negatives_per_state))]


def _build_pairs(states: list[dict], args):
  pos_features = []
  neg_features = []
  pair_base_features = []
  cls_features = []
  cls_base_features = []
  cls_labels = []
  pair_states = 0
  skipped_no_positive = 0
  hard_negative_count = 0
  base_pairs = 0
  for state in states:
    base = _base_action(state)
    base_feature = base["feature_std"]
    positives = [action for action in state["actions"] if bool(action["positive"])]
    if not positives:
      skipped_no_positive += 1
      continue
    negatives = [base] + _hard_negative_candidates(state, args)
    pair_states += 1
    hard_negative_count += len(negatives) - 1
    base_pairs += len(positives)
    for positive in positives:
      for negative in negatives:
        pos_features.append(positive["feature_std"])
        neg_features.append(negative["feature_std"])
        pair_base_features.append(base_feature)
      cls_features.append(positive["feature_std"])
      cls_base_features.append(base_feature)
      cls_labels.append(1.0)
    for negative in negatives:
      cls_features.append(negative["feature_std"])
      cls_base_features.append(base_feature)
      cls_labels.append(0.0)
  if not pos_features:
    raise ValueError("no pairwise training pairs built")
  return {
      "pos": torch.tensor(np.asarray(pos_features, dtype=np.float32), dtype=torch.float32),
      "neg": torch.tensor(np.asarray(neg_features, dtype=np.float32), dtype=torch.float32),
      "pair_base": torch.tensor(
          np.asarray(pair_base_features, dtype=np.float32), dtype=torch.float32
      ),
      "cls_x": torch.tensor(np.asarray(cls_features, dtype=np.float32), dtype=torch.float32),
      "cls_base": torch.tensor(
          np.asarray(cls_base_features, dtype=np.float32), dtype=torch.float32
      ),
      "cls_y": torch.tensor(cls_labels, dtype=torch.float32),
      "summary": {
          "pair_states": int(pair_states),
          "skipped_no_positive_states": int(skipped_no_positive),
          "pairs": int(len(pos_features)),
          "classification_examples": int(len(cls_features)),
          "hard_negative_actions": int(hard_negative_count),
          "base_pairs": int(base_pairs),
      },
  }


def _train(args, train_states, feature_size):
  torch.manual_seed(int(args.seed))
  model = PairwiseBoundary(feature_size, int(args.hidden))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  tensors = _build_pairs(train_states, args)
  pos = tensors["pos"]
  neg = tensors["neg"]
  pair_base = tensors["pair_base"]
  cls_x = tensors["cls_x"]
  cls_base = tensors["cls_base"]
  cls_y = tensors["cls_y"]
  pos_weight = torch.tensor(
      max(1.0, float(torch.sum(1.0 - cls_y).item()) / max(1.0, float(torch.sum(cls_y).item()))),
      dtype=torch.float32,
  )
  rng = torch.Generator().manual_seed(int(args.seed))
  losses = []
  margin_losses = []
  cls_losses = []
  batch_size = max(1, int(args.batch_size))
  for _ in range(max(1, int(args.epochs))):
    pair_order = torch.randperm(pos.shape[0], generator=rng)
    cls_order = torch.randperm(cls_x.shape[0], generator=rng)
    total_loss = 0.0
    total_margin = 0.0
    total_cls = 0.0
    seen = 0
    for start in range(0, pos.shape[0], batch_size):
      pair_idx = pair_order[start:start + batch_size]
      cls_idx = cls_order[start % cls_x.shape[0]:(start % cls_x.shape[0]) + len(pair_idx)]
      if int(cls_idx.numel()) < int(pair_idx.numel()):
        cls_idx = cls_order[:len(pair_idx)]
      pos_score = model(pos[pair_idx], pair_base[pair_idx])
      neg_score = model(neg[pair_idx], pair_base[pair_idx])
      margin_loss = F.softplus(-(pos_score - neg_score)).mean()
      cls_score = model(cls_x[cls_idx], cls_base[cls_idx])
      cls_loss = F.binary_cross_entropy_with_logits(
          cls_score, cls_y[cls_idx], pos_weight=pos_weight
      )
      loss = (
          float(args.margin_loss_weight) * margin_loss
          + float(args.classification_loss_weight) * cls_loss
      )
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      count = int(pair_idx.numel())
      total_loss += float(loss.detach()) * count
      total_margin += float(margin_loss.detach()) * count
      total_cls += float(cls_loss.detach()) * count
      seen += count
    denom = max(1, seen)
    losses.append(total_loss / denom)
    margin_losses.append(total_margin / denom)
    cls_losses.append(total_cls / denom)
  report = {
      **tensors["summary"],
      "epochs": int(args.epochs),
      "batch_size": int(args.batch_size),
      "hidden": int(args.hidden),
      "lr": float(args.lr),
      "weight_decay": float(args.weight_decay),
      "classification_pos_weight": round(float(pos_weight.item()), 6),
      "initial_loss": round(losses[0], 6),
      "final_loss": round(losses[-1], 6),
      "initial_margin_loss": round(margin_losses[0], 6),
      "final_margin_loss": round(margin_losses[-1], 6),
      "initial_classification_loss": round(cls_losses[0], 6),
      "final_classification_loss": round(cls_losses[-1], 6),
  }
  return model, report


def _predict_states(model, states, batch_size: int):
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
      scores = model(x[start:end], b[start:end]).cpu().numpy()
      for offset, action in enumerate(refs[start:end]):
        action["pair_score"] = float(scores[offset])


def _selected_action(state: dict, threshold: float, exclude_forced: bool):
  base = _base_action(state)
  base_score = float(base.get("pair_score", 0.0))
  best = None
  best_margin = None
  for action in state["actions"]:
    if bool(action["is_base"]):
      continue
    if exclude_forced and bool(action.get("forced_led_lane_negative", False)):
      continue
    margin = float(action.get("pair_score", 0.0)) - base_score
    if margin < threshold:
      continue
    if best is None or margin > float(best_margin):
      best = dict(action)
      best["score_margin"] = float(margin)
      best_margin = float(margin)
  return best


def _threshold_report(states: list[dict], thresholds: list[float], exclude_forced: bool):
  reports = []
  for threshold in thresholds:
    overrides = 0
    positives = 0
    safe = 0
    forced = 0
    margins = []
    value_deltas = []
    risk_deltas = []
    for state in states:
      selected = _selected_action(state, float(threshold), exclude_forced)
      if selected is None:
        continue
      overrides += 1
      positives += int(bool(selected["positive"]))
      safe += int(bool(selected["safe"]))
      forced += int(bool(selected.get("forced_led_lane_negative", False)))
      margins.append(float(selected["score_margin"]))
      value_deltas.append(float(selected["value_delta"]))
      risk_deltas.append(float(selected["risk_delta"]))
    reports.append({
        "score_margin_threshold": float(threshold),
        "states": int(len(states)),
        "overrides": int(overrides),
        "coverage": round(float(overrides / max(1, len(states))), 6),
        "positive_precision": (
            None if overrides == 0 else round(float(positives / overrides), 6)
        ),
        "safe_rate": None if overrides == 0 else round(float(safe / overrides), 6),
        "forced_led_lane_negative_selected": int(forced),
        "mean_score_margin": (
            None if not margins else round(float(np.mean(margins)), 6)
        ),
        "mean_value_delta": (
            None if not value_deltas else round(float(np.mean(value_deltas)), 6)
        ),
        "mean_risk_delta": (
            None if not risk_deltas else round(float(np.mean(risk_deltas)), 6)
        ),
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
  ) if reports else None


def _eligible_reports(args, reports):
  out = []
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
    out.append(row)
  return out


def main():
  args = parse_args()
  rows, metadata_rows, source_reports = _load_replays_and_metadata(args)
  states, skipped_report = _build_states(args, rows, metadata_rows)
  train_states, validation_states, split_report = _split_states(args, states)
  mean, std = _standardize_state_features(train_states, validation_states, states)
  feature_size = len(states[0]["actions"][0]["feature_std"])
  model, train_report = _train(args, train_states, feature_size)
  _predict_states(model, states, int(args.batch_size))

  thresholds = _parse_float_list(args.score_margin_thresholds)
  train_thresholds = _threshold_report(
      train_states,
      thresholds,
      bool(args.exclude_forced_led_lane_negatives_at_selection),
  )
  validation_thresholds = _threshold_report(
      validation_states,
      thresholds,
      bool(args.exclude_forced_led_lane_negatives_at_selection),
  )
  all_thresholds = _threshold_report(
      states,
      thresholds,
      bool(args.exclude_forced_led_lane_negatives_at_selection),
  )
  gate = _eligible_reports(args, validation_thresholds)
  best_gate = _best_report(gate)

  if str(args.model_out or ""):
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "feature_size": int(feature_size),
        "model_class": "PairwiseBoundary",
        "args": vars(args),
        "train_report": train_report,
        "replay_sources": source_reports,
    }, model_path)

  result = {
      "replay": list(args.replay),
      "metadata": list(args.metadata or []),
      "replay_sources": source_reports,
      "model_out": str(args.model_out or ""),
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
