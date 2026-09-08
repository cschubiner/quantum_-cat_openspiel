#!/usr/bin/env python3
"""Audit value-only action-set gate decisions for belief-RFT reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import quantum_cat_action_set_transformer_trainer as trainer


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--report", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument(
      "--source-indices",
      default="",
      help=(
          "Comma-separated source indices to audit. Defaults to the report's "
          "validation_source_indices."
      ),
  )
  return parser.parse_args()


def _round_float(value, digits=6):
  if value is None:
    return None
  return round(float(value), int(digits))


def _action_string(meta: dict, action: int):
  flags = meta.get("candidate_flags", {}).get(str(int(action)), {})
  return flags.get("action_string") or f"action={int(action)}"


def _action_record(meta: dict, action: dict):
  flags = meta.get("candidate_flags", {}).get(str(int(action["action"])), {})
  return {
      "action": int(action["action"]),
      "action_string": flags.get("action_string"),
      "value_delta": _round_float(action.get("value_delta")),
      "policy_prob": _round_float(action.get("policy_prob")),
      "pred_value_delta": _round_float(action.get("pred_value_delta")),
      "pred_positive_prob": _round_float(action.get("pred_positive_prob")),
      "set_score": _round_float(action.get("set_score")),
      "follows_led": flags.get("follows_led"),
      "red": flags.get("red"),
      "red_switch": flags.get("red_switch"),
      "token_loss": flags.get("token_loss"),
      "actual_score": flags.get("actual_score"),
      "belief_score_a": flags.get("belief_score_a"),
      "belief_score_b": flags.get("belief_score_b"),
  }


def _load_predicted_states(report: dict):
  args = argparse.Namespace(**report["args"])
  rows, metadata_rows, _ = trainer._load_replays_and_metadata(args)
  states, skipped = trainer._build_states(args, rows, metadata_rows)
  trainer._append_round_context_features(args, rows, states)
  checkpoint = torch.load(report["model_out"], map_location="cpu", weights_only=False)
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
  model = trainer.ActionSetTransformer(
      int(checkpoint["feature_size"]),
      int(args.hidden),
      int(args.layers),
      int(args.heads),
      float(args.dropout),
  )
  model.load_state_dict(checkpoint["model_state"])
  trainer._predict_states(model, states, int(args.batch_size))
  return args, states, metadata_rows, skipped


def _decision_record(args, state: dict, meta: dict, threshold: dict):
  selected = trainer._value_only_selected_action(
      state,
      score_margin_threshold=float(threshold["score_margin_threshold"]),
      value_threshold=float(threshold["value_threshold"]),
      positive_threshold=float(threshold["positive_threshold"]),
      value_bonus=float(args.selection_value_bonus),
      policy_tiebreak=float(args.selection_policy_tiebreak),
      exclude_forced=bool(args.exclude_forced_led_lane_negatives_at_selection),
  )
  base = trainer._base_action(state)
  positives = [action for action in state["actions"] if bool(action["positive"])]
  record = {
      "state_index": int(state["state_index"]),
      "source": int(state.get("source_replay_index", -1)),
      "row_group_id": meta.get("row_group_id"),
      "round_seed": meta.get("round_seed"),
      "match_index": meta.get("match_index"),
      "round_index": meta.get("round_index"),
      "decision_index": meta.get("decision_index"),
      "trick_number": meta.get("trick_number"),
      "player": meta.get("player"),
      "led_color": meta.get("led_color"),
      "legal_count": meta.get("legal_count"),
      "base_action": int(base["action"]),
      "base_action_string": (
          meta.get("base_flags", {}).get("action_string")
          or meta.get("policy_best_action_string")
          or _action_string(meta, int(base["action"]))
      ),
      "target_action": int(state["target_action"]),
      "target_action_string": (
          meta.get("label_best_action_string")
          or _action_string(meta, int(state["target_action"]))
      ),
      "target_is_base": bool(state.get("target_is_base", False)),
      "has_positive": bool(state.get("has_positive", False)),
      "positive_actions": [_action_record(meta, action) for action in positives],
      "selected": None,
  }
  if selected is None:
    record["outcome"] = (
        "miss_positive" if bool(state.get("has_positive", False)) else "base_abstain"
    )
    return record
  selected_record = _action_record(meta, selected)
  selected_record["score_margin"] = _round_float(selected.get("score_margin"))
  selected_record["selected_positive"] = bool(selected.get("positive", False))
  record["selected"] = selected_record
  record["outcome"] = (
      "true_override" if bool(selected.get("positive", False))
      else "false_base_override"
  )
  return record


def main():
  args = parse_args()
  report_path = Path(args.report)
  report = json.loads(report_path.read_text())
  threshold = report.get("best_validation_value_only_threshold")
  if threshold is None:
    raise ValueError(f"report has no best_validation_value_only_threshold: {report_path}")
  train_args, states, metadata_rows, skipped = _load_predicted_states(report)
  source_spec = str(args.source_indices or "").strip()
  if not source_spec:
    source_spec = str(train_args.validation_source_indices)
  audit_sources = trainer._parse_int_set(source_spec)
  audit_states = trainer._source_states(states, audit_sources)
  if not audit_states:
    raise ValueError(f"no states found for sources: {sorted(audit_sources)}")
  decisions = [
      _decision_record(train_args, state, metadata_rows[int(state["state_index"])], threshold)
      for state in audit_states
  ]
  outcome_counts: dict[str, int] = {}
  for row in decisions:
    key = str(row["outcome"])
    outcome_counts[key] = int(outcome_counts.get(key, 0)) + 1
  false_base = [row for row in decisions if row["outcome"] == "false_base_override"]
  missed = [row for row in decisions if row["outcome"] == "miss_positive"]
  true_overrides = [row for row in decisions if row["outcome"] == "true_override"]
  result = {
      "report": str(report_path),
      "model": report.get("model_out"),
      "audited_sources": sorted(int(x) for x in audit_sources),
      "threshold": threshold,
      "skipped": skipped,
      "summary": {
          "states": int(len(decisions)),
          "outcome_counts": outcome_counts,
          "true_overrides": int(len(true_overrides)),
          "false_base_overrides": int(len(false_base)),
          "missed_positives": int(len(missed)),
      },
      "false_base_overrides": false_base,
      "missed_positives": missed,
      "true_overrides": true_overrides,
      "all_decisions": decisions,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
