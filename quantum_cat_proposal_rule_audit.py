"""Audit deterministic approval rules for live proposal-mined actions.

This is a cheap check before training another selector: can proposal scalars
and public action predicates separate useful best-of-N proposals from false
positives in a source-held-out way?
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


NUMERIC_FEATURES = [
    "score_margin",
    "candidate_score",
    "base_score",
    "candidate_pred_value_delta",
    "candidate_pred_positive_prob",
    "candidate_policy_prob",
    "base_policy_prob",
    "policy_prob_delta",
    "policy_log_ratio",
    "candidate_count",
    "scored_candidate_count",
    "candidate_liveness_failure_count",
    "candidate_own_lane_surplus_after",
    "candidate_lane_surplus_damage",
    "candidate_min_player_lane_surplus_after",
    "candidate_own_public_slots_after",
    "candidate_public_slot_damage",
    "candidate_min_player_open_slots_after",
    "base_liveness_failure_count",
    "base_own_lane_surplus_after",
    "base_lane_surplus_damage",
    "base_min_player_lane_surplus_after",
    "base_own_public_slots_after",
    "base_public_slot_damage",
    "base_min_player_open_slots_after",
    "candidate_minus_base_lane_surplus",
    "candidate_minus_base_public_slots",
    "candidate_minus_base_liveness_failures",
]

BOOL_FEATURES = [
    "candidate_follows_led",
    "candidate_switches_off_led",
    "candidate_is_red_switch",
    "candidate_lost_led_token",
    "candidate_would_win_now",
    "candidate_responsible_red_win_switch",
    "base_follows_led",
    "base_switches_off_led",
    "base_is_red_switch",
    "base_lost_led_token",
    "base_would_win_now",
    "base_responsible_red_win_switch",
]

PAIR_GE_RULES = [
    ("candidate_pred_positive_prob", "candidate_pred_value_delta"),
    ("score_margin", "candidate_pred_value_delta"),
    ("score_margin", "candidate_pred_positive_prob"),
    ("candidate_policy_prob", "candidate_pred_positive_prob"),
    ("policy_prob_delta", "candidate_pred_positive_prob"),
    ("policy_log_ratio", "candidate_pred_positive_prob"),
]

BOOL_NUMERIC_RULES = [
    "candidate_is_red_switch",
    "candidate_responsible_red_win_switch",
    "candidate_would_win_now",
    "candidate_follows_led",
    "candidate_switches_off_led",
    "candidate_lost_led_token",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--metadata",
      nargs="+",
      required=True,
      help="Proposal metadata JSON sidecars. Each file is treated as one source.",
  )
  parser.add_argument("--out", required=True)
  parser.add_argument(
      "--targets",
      default="positive_safe,safe",
      help=(
          "Comma-separated targets: positive_safe, positive, safe, "
          "no_candidate_any, positive_no_candidate_any, "
          "positive_safe_no_candidate_any."
      ),
  )
  parser.add_argument("--threshold-grid-size", type=int, default=13)
  parser.add_argument("--top-k", type=int, default=8)
  parser.add_argument("--min-train-approvals", type=int, default=12)
  parser.add_argument("--min-holdout-approvals", type=int, default=4)
  parser.add_argument("--min-train-precision", type=float, default=0.75)
  parser.add_argument("--min-holdout-precision", type=float, default=0.75)
  parser.add_argument("--min-train-recall", type=float, default=0.20)
  parser.add_argument("--min-holdout-recall", type=float, default=0.20)
  parser.add_argument("--max-false-approval-rate", type=float, default=0.50)
  parser.add_argument("--min-safe-rate", type=float, default=0.90)
  parser.add_argument("--min-mean-value-delta", type=float, default=0.0)
  parser.add_argument("--max-mean-risk-delta", type=float, default=0.0)
  return parser.parse_args()


def _sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def _as_rows(blob: Any) -> list[dict]:
  if isinstance(blob, list):
    return [row for row in blob if isinstance(row, dict)]
  if isinstance(blob, dict):
    rows = blob.get("rows")
    if isinstance(rows, list):
      return [row for row in rows if isinstance(row, dict)]
    if isinstance(rows, dict):
      return [
          row for _, row in sorted(rows.items(), key=lambda item: int(item[0]))
          if isinstance(row, dict)
      ]
  return []


def _load_source(path: Path, source_index: int) -> tuple[list[dict], dict]:
  with path.open() as f:
    blob = json.load(f)
  rows = _as_rows(blob)
  source_info = {
      "source_index": int(source_index),
      "path": str(path),
      "sha256": _sha256_file(path),
      "rows": int(len(rows)),
  }
  if isinstance(blob, dict):
    summary = blob.get("summary")
    if isinstance(summary, dict):
      source_info["summary"] = summary
  return rows, source_info


def _float(value: Any, default: float = 0.0) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return float(default)
  if not math.isfinite(result):
    return float(default)
  return result


def _bool(value: Any) -> bool:
  return bool(value)


def _find_action(row: dict, action_id: int) -> dict:
  for action in row.get("action_outcomes", []):
    if not isinstance(action, dict):
      continue
    if int(action.get("action", -1)) == int(action_id):
      return action
  return {}


def _liveness(action: dict) -> dict:
  live = action.get("liveness")
  return live if isinstance(live, dict) else {}


def _action_float(action: dict, key: str) -> float:
  return _float(_liveness(action).get(key, 0.0))


def _action_bool(action: dict, key: str) -> bool:
  if key in action:
    return _bool(action.get(key))
  return _bool(_liveness(action).get(key, False))


def _build_features(proposal: dict, candidate: dict, base: dict) -> dict[str, float]:
  eps = 1e-9
  candidate_policy = _float(proposal.get("candidate_policy_prob", 0.0))
  base_policy = _float(proposal.get("base_policy_prob", 0.0))
  features = {
      "score_margin": _float(proposal.get("score_margin", 0.0)),
      "candidate_score": _float(proposal.get("candidate_score", 0.0)),
      "base_score": _float(proposal.get("base_score", 0.0)),
      "candidate_pred_value_delta": _float(
          proposal.get("candidate_pred_value_delta", 0.0)
      ),
      "candidate_pred_positive_prob": _float(
          proposal.get("candidate_pred_positive_prob", 0.0)
      ),
      "candidate_policy_prob": candidate_policy,
      "base_policy_prob": base_policy,
      "policy_prob_delta": candidate_policy - base_policy,
      "policy_log_ratio": math.log((candidate_policy + eps) / (base_policy + eps)),
      "candidate_count": _float(proposal.get("candidate_count", 0.0)),
      "scored_candidate_count": _float(proposal.get("scored_candidate_count", 0.0)),
  }
  for prefix, action in (("candidate", candidate), ("base", base)):
    features[f"{prefix}_liveness_failure_count"] = _action_float(
        action, "liveness_failure_count"
    )
    features[f"{prefix}_own_lane_surplus_after"] = _action_float(
        action, "own_lane_surplus_after"
    )
    features[f"{prefix}_lane_surplus_damage"] = _action_float(
        action, "lane_surplus_damage"
    )
    features[f"{prefix}_min_player_lane_surplus_after"] = _action_float(
        action, "min_player_lane_surplus_after"
    )
    features[f"{prefix}_own_public_slots_after"] = _action_float(
        action, "own_public_slots_after"
    )
    features[f"{prefix}_public_slot_damage"] = _action_float(
        action, "public_slot_damage"
    )
    features[f"{prefix}_min_player_open_slots_after"] = _action_float(
        action, "min_player_open_slots_after"
    )
    features[f"{prefix}_follows_led"] = 1.0 if _action_bool(action, "follows_led") else 0.0
    features[f"{prefix}_switches_off_led"] = (
        1.0 if _action_bool(action, "switches_off_led") else 0.0
    )
    features[f"{prefix}_is_red_switch"] = (
        1.0 if _action_bool(action, "is_red_switch") else 0.0
    )
    features[f"{prefix}_lost_led_token"] = (
        1.0 if _action_bool(action, "lost_led_token") else 0.0
    )
    features[f"{prefix}_would_win_now"] = (
        1.0 if _action_bool(action, "would_win_now") else 0.0
    )
    features[f"{prefix}_responsible_red_win_switch"] = (
        1.0 if _action_bool(action, "responsible_red_win_switch") else 0.0
    )
  features["candidate_minus_base_lane_surplus"] = (
      features["candidate_own_lane_surplus_after"]
      - features["base_own_lane_surplus_after"]
  )
  features["candidate_minus_base_public_slots"] = (
      features["candidate_own_public_slots_after"]
      - features["base_own_public_slots_after"]
  )
  features["candidate_minus_base_liveness_failures"] = (
      features["candidate_liveness_failure_count"]
      - features["base_liveness_failure_count"]
  )
  return features


def _target(proposal: dict, target_name: str) -> bool:
  positive = _bool(proposal.get("positive_by_dense_delta", False))
  safe = _bool(proposal.get("safe_by_paradox_delta", False))
  no_candidate_any = not _bool(proposal.get("candidate_any_paradox", False))
  if target_name == "positive":
    return positive
  if target_name == "safe":
    return safe
  if target_name == "no_candidate_any":
    return no_candidate_any
  if target_name == "positive_no_candidate_any":
    return bool(positive and no_candidate_any)
  if target_name == "positive_safe":
    return bool(positive and safe)
  if target_name == "positive_safe_no_candidate_any":
    return bool(positive and safe and no_candidate_any)
  raise ValueError(f"unknown target {target_name!r}")


def _examples(metadata_paths: list[str]) -> tuple[list[dict], list[dict], dict]:
  examples = []
  source_infos = []
  skipped = {
      "missing_proposal": 0,
      "below_threshold": 0,
      "missing_candidate_action": 0,
      "missing_base_action": 0,
  }
  for source_index, path_text in enumerate(metadata_paths):
    path = Path(path_text)
    rows, source_info = _load_source(path, source_index)
    source_infos.append(source_info)
    for row_index, row in enumerate(rows):
      proposal = row.get("proposal")
      if not isinstance(proposal, dict):
        skipped["missing_proposal"] += 1
        continue
      if not _bool(proposal.get("above_threshold", False)):
        skipped["below_threshold"] += 1
        continue
      candidate_action = int(proposal.get("candidate_action", -1))
      base_action = int(proposal.get("base_action", row.get("policy_best_action", -1)))
      candidate = _find_action(row, candidate_action)
      base = _find_action(row, base_action)
      if not candidate:
        skipped["missing_candidate_action"] += 1
        continue
      if not base:
        skipped["missing_base_action"] += 1
        continue
      features = _build_features(proposal, candidate, base)
      examples.append({
          "source_index": int(source_index),
          "source_path": str(path),
          "row_index": int(row_index),
          "phase": row.get("phase"),
          "trick_number": row.get("trick_number"),
          "player": row.get("player"),
          "base_action": int(base_action),
          "candidate_action": int(candidate_action),
          "base_action_string": proposal.get("base_action_string"),
          "candidate_action_string": proposal.get("candidate_action_string"),
          "positive_by_dense_delta": _bool(
              proposal.get("positive_by_dense_delta", False)
          ),
          "safe_by_paradox_delta": _bool(proposal.get("safe_by_paradox_delta", False)),
          "candidate_any_paradox": _bool(proposal.get("candidate_any_paradox", False)),
          "base_any_paradox": _bool(proposal.get("base_any_paradox", False)),
          "dense_value_delta": _float(
              proposal.get("dense_action_value_delta_vs_base", 0.0)
          ),
          "paradox_delta": _float(proposal.get("paradox_delta_vs_base", 0.0)),
          "features": features,
      })
  return examples, source_infos, skipped


def _thresholds(values: list[float], grid_size: int) -> list[float]:
  finite = sorted(v for v in values if math.isfinite(v))
  if not finite:
    return []
  unique = sorted(set(round(v, 10) for v in finite))
  if len(unique) <= max(2, grid_size):
    return [float(v) for v in unique]
  qs = np.linspace(0.05, 0.95, max(3, int(grid_size)))
  return [float(np.quantile(np.asarray(finite, dtype=np.float64), q)) for q in qs]


def _rule_name(conditions: list[tuple[str, str, float]]) -> str:
  parts = []
  for feature, op, threshold in conditions:
    if op == "==":
      parts.append(f"{feature} == {int(threshold)}")
    else:
      parts.append(f"{feature} {op} {threshold:.6g}")
  return " AND ".join(parts)


def _make_rule(conditions: list[tuple[str, str, float]]) -> dict:
  return {
      "conditions": [
          {"feature": f, "op": op, "threshold": float(t)}
          for f, op, t in conditions
      ],
      "name": _rule_name(conditions),
  }


def _candidate_rules(train_examples: list[dict], grid_size: int) -> list[dict]:
  rules = []
  numeric_thresholds = {
      feature: _thresholds(
          [float(ex["features"].get(feature, 0.0)) for ex in train_examples],
          grid_size,
      )
      for feature in NUMERIC_FEATURES
  }
  for feature, thresholds in numeric_thresholds.items():
    for threshold in thresholds:
      rules.append(_make_rule([(feature, ">=", threshold)]))
      rules.append(_make_rule([(feature, "<=", threshold)]))
  for feature in BOOL_FEATURES:
    rules.append(_make_rule([(feature, "==", 1.0)]))
    rules.append(_make_rule([(feature, "==", 0.0)]))
  for bool_feature in BOOL_NUMERIC_RULES:
    for numeric_feature, thresholds in numeric_thresholds.items():
      for threshold in thresholds:
        rules.append(_make_rule([
            (bool_feature, "==", 1.0),
            (numeric_feature, ">=", threshold),
        ]))
        rules.append(_make_rule([
            (bool_feature, "==", 1.0),
            (numeric_feature, "<=", threshold),
        ]))
  for left, right in PAIR_GE_RULES:
    for left_threshold in numeric_thresholds.get(left, []):
      for right_threshold in numeric_thresholds.get(right, []):
        rules.append(_make_rule([
            (left, ">=", left_threshold),
            (right, ">=", right_threshold),
        ]))
  seen = set()
  unique_rules = []
  for rule in rules:
    key = rule["name"]
    if key in seen:
      continue
    seen.add(key)
    unique_rules.append(rule)
  return unique_rules


def _selects(rule: dict, ex: dict) -> bool:
  features = ex["features"]
  for condition in rule["conditions"]:
    value = float(features.get(condition["feature"], 0.0))
    threshold = float(condition["threshold"])
    op = condition["op"]
    if op == ">=":
      if value < threshold:
        return False
    elif op == "<=":
      if value > threshold:
        return False
    elif op == "==":
      if int(round(value)) != int(round(threshold)):
        return False
    else:
      raise ValueError(f"unknown op {op!r}")
  return True


def _metrics(examples: list[dict], rule: dict, target_name: str) -> dict:
  selected = [ex for ex in examples if _selects(rule, ex)]
  total_targets = int(sum(_target(ex, target_name) for ex in examples))
  total_rejected = int(len(examples) - total_targets)
  approvals = int(len(selected))
  true_approvals = int(sum(_target(ex, target_name) for ex in selected))
  false_approvals = int(approvals - true_approvals)
  safe = int(sum(bool(ex["safe_by_paradox_delta"]) for ex in selected))
  positive = int(sum(bool(ex["positive_by_dense_delta"]) for ex in selected))
  candidate_any = int(sum(bool(ex["candidate_any_paradox"]) for ex in selected))
  dense_values = [float(ex["dense_value_delta"]) for ex in selected]
  paradox_deltas = [float(ex["paradox_delta"]) for ex in selected]
  return {
      "examples": int(len(examples)),
      "target": target_name,
      "target_positives": int(total_targets),
      "target_negatives": int(total_rejected),
      "approvals": approvals,
      "approval_rate": round(float(approvals / max(1, len(examples))), 6),
      "approval_precision": (
          None if approvals == 0 else round(float(true_approvals / approvals), 6)
      ),
      "approval_recall": (
          None if total_targets == 0 else round(float(true_approvals / total_targets), 6)
      ),
      "false_approvals": false_approvals,
      "false_approval_rate": (
          None if total_rejected == 0 else round(float(false_approvals / total_rejected), 6)
      ),
      "safe_rate": None if approvals == 0 else round(float(safe / approvals), 6),
      "positive_rate": (
          None if approvals == 0 else round(float(positive / approvals), 6)
      ),
      "candidate_any_paradox_rate": (
          None if approvals == 0 else round(float(candidate_any / approvals), 6)
      ),
      "mean_dense_value_delta": (
          None if not dense_values else round(float(np.mean(dense_values)), 6)
      ),
      "mean_paradox_delta": (
          None if not paradox_deltas else round(float(np.mean(paradox_deltas)), 6)
      ),
  }


def _passes(metrics: dict, args: argparse.Namespace, *, train: bool) -> bool:
  min_approvals = args.min_train_approvals if train else args.min_holdout_approvals
  min_precision = args.min_train_precision if train else args.min_holdout_precision
  min_recall = args.min_train_recall if train else args.min_holdout_recall
  if int(metrics["target_positives"]) <= 0:
    return False
  if int(metrics["approvals"]) < int(min_approvals):
    return False
  precision = metrics.get("approval_precision")
  if precision is None or float(precision) < float(min_precision):
    return False
  recall = metrics.get("approval_recall")
  if recall is None or float(recall) < float(min_recall):
    return False
  false_rate = metrics.get("false_approval_rate")
  if false_rate is not None and float(false_rate) > args.max_false_approval_rate:
    return False
  safe_rate = metrics.get("safe_rate")
  if safe_rate is None or float(safe_rate) < args.min_safe_rate:
    return False
  mean_value = metrics.get("mean_dense_value_delta")
  if mean_value is None or float(mean_value) < args.min_mean_value_delta:
    return False
  mean_risk = metrics.get("mean_paradox_delta")
  if mean_risk is None or float(mean_risk) > args.max_mean_risk_delta:
    return False
  return True


def _sort_key(item: dict) -> tuple:
  metrics = item["metrics"]
  precision = -1.0 if metrics["approval_precision"] is None else metrics["approval_precision"]
  safe = -1.0 if metrics["safe_rate"] is None else metrics["safe_rate"]
  value = -999.0 if metrics["mean_dense_value_delta"] is None else metrics["mean_dense_value_delta"]
  recall = -1.0 if metrics["approval_recall"] is None else metrics["approval_recall"]
  false_rate = 999.0 if metrics["false_approval_rate"] is None else metrics["false_approval_rate"]
  approvals = int(metrics["approvals"])
  risk = 999.0 if metrics["mean_paradox_delta"] is None else metrics["mean_paradox_delta"]
  return (precision, safe, value, recall, -risk, -false_rate, approvals)


def _rule_record(rule: dict, metrics: dict) -> dict:
  return {
      "rule": rule,
      "metrics": metrics,
  }


def _summarize_examples(examples: list[dict]) -> dict:
  if not examples:
    return {}
  return {
      "examples": int(len(examples)),
      "positive_by_dense_delta": int(
          sum(bool(ex["positive_by_dense_delta"]) for ex in examples)
      ),
      "safe_by_paradox_delta": int(
          sum(bool(ex["safe_by_paradox_delta"]) for ex in examples)
      ),
      "positive_safe": int(
          sum(
              bool(ex["positive_by_dense_delta"]) and bool(ex["safe_by_paradox_delta"])
              for ex in examples
          )
      ),
      "candidate_any_paradox": int(
          sum(bool(ex["candidate_any_paradox"]) for ex in examples)
      ),
      "base_any_paradox": int(sum(bool(ex["base_any_paradox"]) for ex in examples)),
      "mean_dense_value_delta": round(
          float(np.mean([float(ex["dense_value_delta"]) for ex in examples])), 6
      ),
      "mean_paradox_delta": round(
          float(np.mean([float(ex["paradox_delta"]) for ex in examples])), 6
      ),
  }


def _audit_target(
    examples: list[dict],
    source_indices: list[int],
    target_name: str,
    args: argparse.Namespace,
) -> dict:
  holdouts = []
  selected_holdout_passes = 0
  any_upper_bound_passes = 0
  for holdout in source_indices:
    train_examples = [ex for ex in examples if int(ex["source_index"]) != int(holdout)]
    holdout_examples = [ex for ex in examples if int(ex["source_index"]) == int(holdout)]
    candidate_rules = _candidate_rules(train_examples, int(args.threshold_grid_size))
    train_records = []
    train_passing = []
    for rule in candidate_rules:
      metrics = _metrics(train_examples, rule, target_name)
      record = _rule_record(rule, metrics)
      train_records.append(record)
      if _passes(metrics, args, train=True):
        train_passing.append(record)
    ranked_train = sorted(train_records, key=_sort_key, reverse=True)
    ranked_train_passing = sorted(train_passing, key=_sort_key, reverse=True)
    selected_rule_record = (
        ranked_train_passing[0] if ranked_train_passing else ranked_train[0]
    )
    selected_rule = selected_rule_record["rule"]
    selected_holdout_metrics = _metrics(holdout_examples, selected_rule, target_name)
    selected_passed = _passes(selected_holdout_metrics, args, train=False)
    if selected_passed:
      selected_holdout_passes += 1
    holdout_records_from_train_passing = []
    for train_record in train_passing:
      holdout_metrics = _metrics(holdout_examples, train_record["rule"], target_name)
      holdout_records_from_train_passing.append(
          _rule_record(train_record["rule"], holdout_metrics)
      )
    holdout_passing = [
        record for record in holdout_records_from_train_passing
        if _passes(record["metrics"], args, train=False)
    ]
    if holdout_passing:
      any_upper_bound_passes += 1
    holdouts.append({
        "holdout_source_index": int(holdout),
        "train_summary": _summarize_examples(train_examples),
        "holdout_summary": _summarize_examples(holdout_examples),
        "candidate_rules": int(len(candidate_rules)),
        "train_passing_rules": int(len(train_passing)),
        "selected_by_train": {
            "train": selected_rule_record,
            "holdout": _rule_record(selected_rule, selected_holdout_metrics),
            "passed_holdout_gate": bool(selected_passed),
        },
        "best_holdout_among_train_passing": (
            None if not holdout_records_from_train_passing else
            sorted(holdout_records_from_train_passing, key=_sort_key, reverse=True)[0]
        ),
        "holdout_passing_rules_among_train_passing": int(len(holdout_passing)),
        "top_train_rules": ranked_train[: int(args.top_k)],
        "top_train_passing_rules": ranked_train_passing[: int(args.top_k)],
      })
  return {
      "target": target_name,
      "source_holdouts": holdouts,
      "selected_holdout_passes": int(selected_holdout_passes),
      "any_train_passing_rule_holdout_passes": int(any_upper_bound_passes),
      "sources": int(len(source_indices)),
      "all_selected_holdouts_passed": bool(
          selected_holdout_passes == len(source_indices)
      ),
      "all_holdouts_have_some_passing_train_rule": bool(
          any_upper_bound_passes == len(source_indices)
      ),
      "gate": {
          "min_train_approvals": int(args.min_train_approvals),
          "min_holdout_approvals": int(args.min_holdout_approvals),
          "min_train_precision": float(args.min_train_precision),
          "min_holdout_precision": float(args.min_holdout_precision),
          "min_train_recall": float(args.min_train_recall),
          "min_holdout_recall": float(args.min_holdout_recall),
          "max_false_approval_rate": float(args.max_false_approval_rate),
          "min_safe_rate": float(args.min_safe_rate),
          "min_mean_value_delta": float(args.min_mean_value_delta),
          "max_mean_risk_delta": float(args.max_mean_risk_delta),
      },
  }


def main() -> None:
  args = parse_args()
  targets = [item.strip() for item in str(args.targets).split(",") if item.strip()]
  examples, source_infos, skipped = _examples(args.metadata)
  if not examples:
    raise ValueError("no above-threshold proposal examples found")
  source_indices = sorted(set(int(ex["source_index"]) for ex in examples))
  report = {
      "kind": "proposal_rule_audit",
      "created_at_utc": _datetime.datetime.now(
          _datetime.timezone.utc
      ).isoformat(),
      "metadata_sources": source_infos,
      "skipped": skipped,
      "overall_summary": _summarize_examples(examples),
      "targets": {
          target: _audit_target(examples, source_indices, target, args)
          for target in targets
      },
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w") as f:
    json.dump(report, f, indent=2, sort_keys=True)
  print(json.dumps({
      "out": str(out),
      "examples": int(len(examples)),
      "sources": int(len(source_indices)),
      "targets": {
          target: {
              "selected_holdout_passes": report["targets"][target][
                  "selected_holdout_passes"
              ],
              "sources": report["targets"][target]["sources"],
              "all_selected_holdouts_passed": report["targets"][target][
                  "all_selected_holdouts_passed"
              ],
              "any_train_passing_rule_holdout_passes": report["targets"][target][
                  "any_train_passing_rule_holdout_passes"
              ],
          }
          for target in targets
      },
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
