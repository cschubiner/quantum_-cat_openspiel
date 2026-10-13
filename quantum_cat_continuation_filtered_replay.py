#!/usr/bin/env python3
"""Export continuation-filtered candidate-vs-base replay rows.

This converts `quantum_cat_cell_exit_continuation_audit.py` output into a
small replay whose action labels come from continuation outcomes, not from the
one-step cell-exit heuristic. It is intended for selector/abstainer training:
supported overrides become positives, rejected one-step positives become hard
negatives, and the base action is always labeled for abstention.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-replay", required=True)
  parser.add_argument("--source-metadata", required=True)
  parser.add_argument(
      "--audit",
      action="append",
      required=True,
      help="Continuation audit JSON. May be passed more than once.",
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--metadata-out", default="")
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--support-mode", choices=("risk_or_score", "risk_only"), default="risk_or_score")
  parser.add_argument("--score-weight", type=float, default=0.10)
  parser.add_argument("--min-support-risk-delta", type=float, default=0.0)
  parser.add_argument("--min-support-score-delta", type=float, default=0.0)
  parser.add_argument("--positive-value-target", type=float, default=0.85)
  parser.add_argument("--negative-value-target", type=float, default=-0.85)
  parser.add_argument("--base-value-target", type=float, default=0.0)
  parser.add_argument("--positive-risk-target", type=float, default=0.05)
  parser.add_argument("--negative-risk-target", type=float, default=0.95)
  parser.add_argument("--base-risk-from-continuation", action="store_true")
  return parser.parse_args()


def _load_metadata_rows(path: str) -> list[dict]:
  data = json.loads(Path(path).read_text())
  rows = data if isinstance(data, list) else data.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"metadata rows not found in {path}")
  return rows


def _support_override(row: dict, args: argparse.Namespace) -> bool:
  risk_delta = float(row.get("selected_delta_any_paradox_rate", 0.0))
  score_delta = float(row.get("selected_delta_actor_score", 0.0))
  if str(args.support_mode) == "risk_only":
    return risk_delta > float(args.min_support_risk_delta)
  return bool(
      risk_delta > float(args.min_support_risk_delta)
      or (
          risk_delta == float(args.min_support_risk_delta)
          and score_delta > float(args.min_support_score_delta)
      )
  )


def _selected_value(row: dict, supported: bool, args: argparse.Namespace) -> float:
  risk_delta = float(row.get("selected_delta_any_paradox_rate", 0.0))
  score_delta = float(row.get("selected_delta_actor_score", 0.0))
  shaped = risk_delta + float(args.score_weight) * score_delta
  if supported:
    return float(max(float(args.positive_value_target), shaped))
  return float(min(float(args.negative_value_target), shaped))


def _base_action_risk(row: dict, args: argparse.Namespace) -> float:
  if not bool(args.base_risk_from_continuation):
    return float(args.positive_risk_target)
  base = (row.get("actions") or {}).get("base") or {}
  return float(base.get("any_paradox_rate", 0.0))


def _make_row(source_row: tuple, audit_row: dict, supported: bool, args: argparse.Namespace):
  arrays = [np.array(item, copy=True) if isinstance(item, np.ndarray) else item for item in source_row]
  num_actions = int(np.asarray(arrays[1]).shape[0])
  base_action = int(audit_row["base_action"])
  selected_action = int(audit_row["selected_action"])
  policy_target = np.zeros(num_actions, dtype=np.float32)
  target_action = selected_action if supported else base_action
  policy_target[target_action] = 1.0
  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_risk_targets = np.zeros(num_actions, dtype=np.float32)
  action_risk_mask = np.zeros(num_actions, dtype=np.float32)

  action_value_targets[base_action] = float(args.base_value_target)
  action_value_mask[base_action] = 1.0
  action_risk_targets[base_action] = _base_action_risk(audit_row, args)
  action_risk_mask[base_action] = 1.0

  action_value_targets[selected_action] = _selected_value(audit_row, supported, args)
  action_value_mask[selected_action] = 1.0
  action_risk_targets[selected_action] = (
      float(args.positive_risk_target) if supported else float(args.negative_risk_target)
  )
  action_risk_mask[selected_action] = 1.0

  return (
      np.asarray(arrays[0], dtype=np.float32),
      np.asarray(arrays[1], dtype=np.float32),
      policy_target,
      np.asarray(arrays[3], dtype=np.float32),
      np.asarray(arrays[4], dtype=np.float32),
      int(target_action),
      int(arrays[6]),
      action_risk_targets,
      action_risk_mask,
      action_value_targets,
      action_value_mask,
      np.asarray(arrays[11], dtype=np.float32),
  )


def _metadata_for(
    source_meta: dict,
    audit_row: dict,
    supported: bool,
    args: argparse.Namespace,
) -> dict:
  base_action = int(audit_row["base_action"])
  selected_action = int(audit_row["selected_action"])
  target_action = selected_action if supported else base_action
  base = (audit_row.get("actions") or {}).get("base") or {}
  selected = (audit_row.get("actions") or {}).get("selected") or {}
  row = dict(source_meta)
  row.update({
      "proposal_source": "continuation_filtered_cell_exit",
      "source_state_index": int(audit_row["state_index"]),
      "policy_best_action": int(base_action),
      "label_best_action": int(target_action),
      "deployment_target_action": int(target_action),
      "selected_candidate_action": int(selected_action),
      "policy_target_changed": bool(target_action != base_action),
      "continuation_supported_override": bool(supported),
      "continuation_rejected_override": bool(not supported),
      "continuation_selected_delta_any_paradox_rate": float(
          audit_row.get("selected_delta_any_paradox_rate", 0.0)
      ),
      "continuation_selected_delta_actor_paradox_rate": float(
          audit_row.get("selected_delta_actor_paradox_rate", 0.0)
      ),
      "continuation_selected_delta_actor_score": float(
          audit_row.get("selected_delta_actor_score", 0.0)
      ),
      "continuation_base_any_paradox_rate": float(base.get("any_paradox_rate", 0.0)),
      "continuation_selected_any_paradox_rate": float(
          selected.get("any_paradox_rate", 0.0)
      ),
      "continuation_base_action_string": base.get("action_string"),
      "continuation_selected_action_string": selected.get("action_string"),
      "calibration_class": "POSITIVE" if supported else "HARD_NEGATIVE",
      "false_positive_for_positive_gate": bool(not supported),
      "candidate_actions": [int(base_action), int(selected_action)],
      "candidate_count": 2,
      "candidate_set_contains_label_best": True,
      "label_margin_vs_policy": float(_selected_value(audit_row, supported, args)),
      "audit_margin_vs_policy": float(
          audit_row.get("selected_delta_any_paradox_rate", 0.0)
      ),
      "audit_paradox_delta_vs_policy": -float(
          audit_row.get("selected_delta_any_paradox_rate", 0.0)
      ),
  })
  return row


def main() -> None:
  args = parse_args()
  source_rows = load_replay(args.source_replay)
  source_metadata = _load_metadata_rows(args.source_metadata)
  if len(source_rows) != len(source_metadata):
    raise ValueError(
        f"source replay/metadata mismatch: {len(source_rows)} vs {len(source_metadata)}"
    )

  replay_rows = []
  metadata_rows = []
  summary = Counter()
  input_audits = []
  for audit_path in args.audit:
    audit = json.loads(Path(audit_path).read_text())
    input_audits.append(str(audit_path))
    for audit_row in audit.get("audited_rows", []):
      state_index = int(audit_row["state_index"])
      supported = _support_override(audit_row, args)
      source_row = source_rows[state_index]
      source_meta = source_metadata[state_index]
      replay_rows.append(_make_row(source_row, audit_row, supported, args))
      metadata = _metadata_for(source_meta, audit_row, supported, args)
      metadata["source_index"] = int(len(metadata_rows))
      metadata_rows.append(metadata)
      summary["rows"] += 1
      summary["supported_overrides"] += int(supported)
      summary["rejected_overrides"] += int(not supported)
      summary["one_step_positive"] += int(
          bool(audit_row.get("one_step_selected_positive"))
      )
      summary["continuation_selected_safer"] += int(
          float(audit_row.get("selected_delta_any_paradox_rate", 0.0)) > 0.0
      )
      summary["continuation_selected_riskier"] += int(
          float(audit_row.get("selected_delta_any_paradox_rate", 0.0)) < 0.0
      )

  out = Path(args.out)
  metadata_out = (
      Path(args.metadata_out)
      if str(args.metadata_out or "").strip()
      else out.with_suffix(out.suffix + ".metadata.json")
  )
  summary_out = (
      Path(args.summary_out)
      if str(args.summary_out or "").strip()
      else out.with_suffix(out.suffix + ".summary.json")
  )
  row_group_ids = [
      f"continuation:{row.get('source_name', '')}:{row.get('source_state_index')}"
      for row in metadata_rows
  ]
  summary_json = {
      "kind": "continuation_filtered_replay",
      "source_replay": str(args.source_replay),
      "source_metadata": str(args.source_metadata),
      "input_audits": input_audits,
      "out": str(out),
      "metadata_out": str(metadata_out),
      "support_mode": str(args.support_mode),
      "score_weight": float(args.score_weight),
      "summary": {key: int(value) for key, value in sorted(summary.items())},
      "rates": {
          "support_rate": (
              round(float(summary["supported_overrides"]) / float(summary["rows"]), 6)
              if summary["rows"] else None
          ),
          "rejection_rate": (
              round(float(summary["rejected_overrides"]) / float(summary["rows"]), 6)
              if summary["rows"] else None
          ),
      },
      "row_group_ids": row_group_ids,
      "row_group_id_kind": "continuation_source_state",
      "row_group_count": int(len(set(row_group_ids))),
  }
  save_replay(replay_rows, out, metadata=summary_json)
  metadata_out.parent.mkdir(parents=True, exist_ok=True)
  metadata_out.write_text(json.dumps(metadata_rows, indent=2, sort_keys=True) + "\n")
  summary_out.parent.mkdir(parents=True, exist_ok=True)
  summary_out.write_text(json.dumps(summary_json, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary_json, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
