#!/usr/bin/env python3
"""Calibrate an action-set model on one source and test on another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quantum_cat_action_set_transformer_trainer import (
    ActionSetTransformer,
    _append_round_context_features,
    _best_report,
    _best_value_only_report,
    _eligible_reports,
    _predict_states,
    _source_states,
    _threshold_report,
    _value_only_eligible_reports,
    _value_only_threshold_report,
)
from quantum_cat_legal_set_selector_trainer import (
    _build_states,
    _load_replays_and_metadata,
    _parse_int_set,
    _state_summary,
    _state_summary_by_source,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True)
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument("--calibration-source-indices", required=True)
  parser.add_argument("--test-source-indices", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument(
      "--gate-mode",
      choices=("risk", "value_only"),
      default="risk",
      help=(
          "risk preserves the historical risk/safe threshold gate. value_only "
          "uses action-value preference labels only, for belief-rollout RFT "
          "banks that intentionally do not contain action-risk labels."
      ),
  )
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--score-margin-thresholds", default="")
  parser.add_argument("--predicted-value-delta-thresholds", default="")
  parser.add_argument("--predicted-risk-delta-max-thresholds", default="")
  parser.add_argument("--positive-prob-thresholds", default="")
  parser.add_argument("--safe-prob-thresholds", default="")
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
  parser.add_argument("--min-validation-value-only-overrides", type=int, default=8)
  parser.add_argument(
      "--min-validation-value-only-positive-precision", type=float, default=0.85
  )
  parser.add_argument(
      "--min-validation-value-only-positive-recall", type=float, default=0.25
  )
  parser.add_argument(
      "--min-validation-value-only-decision-accuracy", type=float, default=0.75
  )
  parser.add_argument(
      "--min-validation-value-only-mean-value-delta", type=float, default=0.0
  )
  return parser.parse_args()


def _namespace_from_checkpoint(checkpoint: dict, args):
  saved_args = dict(checkpoint.get("args") or {})
  saved_args["replay"] = list(args.replay)
  saved_args["metadata"] = list(args.metadata or [])
  for key in (
      "score_margin_thresholds",
      "predicted_value_delta_thresholds",
      "predicted_risk_delta_max_thresholds",
      "positive_prob_thresholds",
      "safe_prob_thresholds",
  ):
    override = str(getattr(args, key) or "").strip()
    if override:
      saved_args[key] = override
  for key in (
      "min_validation_overrides_for_live",
      "min_validation_positive_precision_for_live",
      "min_validation_safe_rate_for_live",
      "min_validation_mean_value_delta_for_live",
      "max_validation_mean_risk_delta_for_live",
      "min_validation_value_only_overrides",
      "min_validation_value_only_positive_precision",
      "min_validation_value_only_positive_recall",
      "min_validation_value_only_decision_accuracy",
      "min_validation_value_only_mean_value_delta",
  ):
    saved_args[key] = getattr(args, key)
  return SimpleNamespace(**saved_args)


def _apply_saved_standardization(states: list[dict], checkpoint: dict):
  mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
  std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
  for state in states:
    for action in state["actions"]:
      feature = np.asarray(action["feature"], dtype=np.float32)
      if feature.shape != mean.shape:
        raise ValueError(
            f"feature shape mismatch: got {feature.shape}, expected {mean.shape}"
        )
      action["feature_std"] = ((feature - mean) / std).astype(np.float32)


def _make_model(checkpoint: dict):
  model_args = checkpoint.get("args") or {}
  model = ActionSetTransformer(
      int(checkpoint["feature_size"]),
      int(model_args.get("hidden", 64)),
      int(model_args.get("layers", 1)),
      int(model_args.get("heads", 4)),
      float(model_args.get("dropout", 0.08)),
  )
  model.load_state_dict(checkpoint["model_state"])
  model.eval()
  return model


def _risk_threshold_key(row: dict):
  return (
      float(row["score_margin_threshold"]),
      float(row["value_threshold"]),
      float(row["risk_threshold"]),
      float(row["positive_threshold"]),
      float(row["safe_threshold"]),
  )


def _value_only_threshold_key(row: dict):
  return (
      float(row["score_margin_threshold"]),
      float(row["value_threshold"]),
      float(row["positive_threshold"]),
  )


def _row_by_key(rows: list[dict], key, key_fn):
  for row in rows:
    if key_fn(row) == key:
      return row
  return None


def _passes_risk(args, row: dict | None):
  if row is None:
    return False
  if int(row.get("overrides") or 0) < int(args.min_validation_overrides_for_live):
    return False
  if row.get("positive_precision") is None or float(
      row["positive_precision"]
  ) < float(args.min_validation_positive_precision_for_live):
    return False
  if row.get("safe_rate") is None or float(row["safe_rate"]) < float(
      args.min_validation_safe_rate_for_live
  ):
    return False
  if row.get("mean_value_delta") is None or float(
      row["mean_value_delta"]
  ) < float(args.min_validation_mean_value_delta_for_live):
    return False
  if row.get("mean_risk_delta") is None or float(
      row["mean_risk_delta"]
  ) > float(args.max_validation_mean_risk_delta_for_live):
    return False
  return True


def _passes_value_only(args, row: dict | None):
  if row is None:
    return False
  if int(row.get("overrides") or 0) < int(
      args.min_validation_value_only_overrides
  ):
    return False
  if row.get("positive_precision") is None or float(
      row["positive_precision"]
  ) < float(args.min_validation_value_only_positive_precision):
    return False
  if row.get("positive_recall") is None or float(
      row["positive_recall"]
  ) < float(args.min_validation_value_only_positive_recall):
    return False
  if float(row.get("decision_accuracy") or 0.0) < float(
      args.min_validation_value_only_decision_accuracy
  ):
    return False
  if row.get("mean_value_delta") is None or float(
      row["mean_value_delta"]
  ) < float(args.min_validation_value_only_mean_value_delta):
    return False
  return True


def main():
  args = parse_args()
  checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
  eval_args = _namespace_from_checkpoint(checkpoint, args)
  rows, metadata_rows, source_reports = _load_replays_and_metadata(eval_args)
  states, skipped_report = _build_states(eval_args, rows, metadata_rows)
  round_context_schema = _append_round_context_features(eval_args, rows, states)
  _apply_saved_standardization(states, checkpoint)
  model = _make_model(checkpoint)
  _predict_states(model, states, int(args.batch_size))

  calibration_sources = _parse_int_set(str(args.calibration_source_indices))
  test_sources = _parse_int_set(str(args.test_source_indices))
  calibration_states = _source_states(states, calibration_sources)
  test_states = _source_states(states, test_sources)
  if not calibration_states or not test_states:
    raise ValueError("empty calibration or test source split")

  gate_mode = str(args.gate_mode)
  if gate_mode == "value_only":
    calibration_thresholds = _value_only_threshold_report(
        eval_args, calibration_states
    )
    test_thresholds = _value_only_threshold_report(eval_args, test_states)
    calibration_eligible = _value_only_eligible_reports(
        eval_args, calibration_thresholds
    )
    best_calibration = _best_value_only_report(calibration_thresholds)
    best_eligible_calibration = _best_value_only_report(calibration_eligible)
    key_fn = _value_only_threshold_key
    pass_fn = _passes_value_only
  else:
    calibration_thresholds = _threshold_report(eval_args, calibration_states)
    test_thresholds = _threshold_report(eval_args, test_states)
    calibration_eligible = _eligible_reports(eval_args, calibration_thresholds)
    best_calibration = _best_report(calibration_thresholds)
    best_eligible_calibration = _best_report(calibration_eligible)
    key_fn = _risk_threshold_key
    pass_fn = _passes_risk
  calibrated_key = (
      key_fn(best_eligible_calibration)
      if best_eligible_calibration is not None else
      (key_fn(best_calibration) if best_calibration is not None else None)
  )
  calibrated_test = (
      _row_by_key(test_thresholds, calibrated_key, key_fn)
      if calibrated_key is not None else None
  )
  if gate_mode == "value_only":
    test_eligible = _value_only_eligible_reports(eval_args, test_thresholds)
    best_test = _best_value_only_report(test_thresholds)
  else:
    test_eligible = _eligible_reports(eval_args, test_thresholds)
    best_test = _best_report(test_thresholds)

  result = {
      "model": str(args.model),
      "replay": list(args.replay),
      "metadata": list(args.metadata or []),
      "replay_sources": source_reports,
      "source_split": {
          "calibration_source_indices": sorted(int(x) for x in calibration_sources),
          "test_source_indices": sorted(int(x) for x in test_sources),
      },
      "gate_mode": gate_mode,
      "skipped": skipped_report,
      "feature_size": int(checkpoint["feature_size"]),
      "feature_schema_version": checkpoint.get("feature_schema_version"),
      "round_context_schema": round_context_schema,
      "calibration_summary": _state_summary(calibration_states),
      "test_summary": _state_summary(test_states),
      "all_summary": _state_summary(states),
      "calibration_summary_by_source": _state_summary_by_source(calibration_states),
      "test_summary_by_source": _state_summary_by_source(test_states),
      "all_summary_by_source": _state_summary_by_source(states),
      "calibration_thresholds": calibration_thresholds,
      "test_thresholds": test_thresholds,
      "best_calibration_threshold": best_calibration,
      "best_eligible_calibration_threshold": best_eligible_calibration,
      "calibrated_test_threshold": calibrated_test,
      "best_test_threshold": best_test,
      "calibrated_gate": {
          "passed": bool(
              best_eligible_calibration is not None
              and pass_fn(eval_args, calibrated_test)
          ),
          "calibration_passed": bool(best_eligible_calibration is not None),
          "test_passed_at_calibrated_threshold": bool(
              pass_fn(eval_args, calibrated_test)
          ),
          "test_oracle_eligible_thresholds": int(len(test_eligible)),
          "min_validation_overrides": int(eval_args.min_validation_overrides_for_live),
          "min_positive_precision": float(
              eval_args.min_validation_positive_precision_for_live
          ),
          "min_safe_rate": float(eval_args.min_validation_safe_rate_for_live),
          "min_mean_value_delta": float(
              eval_args.min_validation_mean_value_delta_for_live
          ),
          "max_mean_risk_delta": float(
              eval_args.max_validation_mean_risk_delta_for_live
          ),
          "min_value_only_overrides": int(
              eval_args.min_validation_value_only_overrides
          ),
          "min_value_only_positive_precision": float(
              eval_args.min_validation_value_only_positive_precision
          ),
          "min_value_only_positive_recall": float(
              eval_args.min_validation_value_only_positive_recall
          ),
          "min_value_only_decision_accuracy": float(
              eval_args.min_validation_value_only_decision_accuracy
          ),
          "min_value_only_mean_value_delta": float(
              eval_args.min_validation_value_only_mean_value_delta
          ),
      },
      "args": vars(args),
      "model_args": checkpoint.get("args") or {},
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
