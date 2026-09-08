#!/usr/bin/env python3
"""Build selector-audit rows for led-lane-compatible cell-exit candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay
from quantum_cat_table_liveness_improve_trainer import (
    _led_lane_burn_label_shape,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", required=True)
  parser.add_argument("--metadata", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--positive-value-margin", type=float, default=0.04)
  parser.add_argument("--safe-risk-delta-max", type=float, default=0.0)
  parser.add_argument("--force-led-lane-burn-negative", action="store_true")
  parser.add_argument(
      "--led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="urgent_win",
  )
  return parser.parse_args()


def _load_metadata_rows(path: str) -> list[dict]:
  data = json.loads(Path(path).read_text())
  rows = data if isinstance(data, list) else data.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"metadata rows not found in {path}")
  return rows


def _source_name(meta: dict) -> str:
  source = meta.get("source_name")
  if source is not None:
    return str(source)
  group = meta.get("group")
  if isinstance(group, dict) and group.get("source_name") is not None:
    return str(group["source_name"])
  return ""


def _trick_number(meta: dict) -> int:
  return int(meta.get("trick_number", -1))


def _legal_actions(row) -> list[int]:
  legal_mask = np.asarray(row[1], dtype=np.float32) > 0.5
  value_mask = np.asarray(row[10], dtype=np.float32) > 0.5
  risk_mask = np.asarray(row[8], dtype=np.float32) > 0.5
  return [int(action) for action in np.flatnonzero(legal_mask & value_mask & risk_mask)]


def _base_action(meta: dict) -> int:
  return int(meta.get("policy_best_action", -1))


def _candidate_rows(args, row, meta: dict, state_index: int) -> list[dict]:
  legal = _legal_actions(row)
  base = _base_action(meta)
  if base not in set(legal):
    return []
  values = np.asarray(row[9], dtype=np.float32)
  risks = np.asarray(row[7], dtype=np.float32)
  action_features = np.asarray(row[11], dtype=np.float32)
  base_value = float(values[base])
  base_risk = float(risks[base])
  candidates = []
  for action in legal:
    if int(action) == base:
      continue
    value_delta = float(values[action]) - base_value
    risk_delta = float(risks[action]) - base_risk
    positive = bool(
        value_delta >= float(args.positive_value_margin)
        and risk_delta <= float(args.safe_risk_delta_max)
    )
    if not positive:
      continue
    forced, kept_exception = _led_lane_burn_label_shape(
        action_features, int(base), int(action), args
    )
    if forced:
      continue
    score = float(value_delta - risk_delta)
    candidates.append({
        "kind": "led_compatible_one_step_positive_candidate",
        "state_index": int(state_index),
        "source_name": _source_name(meta),
        "trick_number": _trick_number(meta),
        "legal_count": int(len(legal)),
        "base_action": int(base),
        "selected_action": int(action),
        "target_action": int(action),
        "selected_positive": True,
        "selected_safe": True,
        "proposal_value_delta": float(value_delta),
        "proposal_risk_delta": float(risk_delta),
        "proposal_score": float(score),
        "kept_by_led_lane_red_exception": bool(kept_exception),
    })
  return candidates


def main() -> None:
  args = parse_args()
  rows = load_replay(args.replay)
  metadata_rows = _load_metadata_rows(args.metadata)
  if len(rows) != len(metadata_rows):
    raise ValueError(
        f"replay/metadata length mismatch: {len(rows)} vs {len(metadata_rows)}"
    )
  best_rows = []
  raw_candidate_count = 0
  candidate_states = 0
  for state_index, (row, meta) in enumerate(zip(rows, metadata_rows)):
    candidates = _candidate_rows(args, row, meta, state_index)
    raw_candidate_count += len(candidates)
    if not candidates:
      continue
    candidate_states += 1
    best_rows.append(max(
        candidates,
        key=lambda item: (
            float(item["proposal_score"]),
            float(item["proposal_value_delta"]),
            -float(item["proposal_risk_delta"]),
            -int(item["selected_action"]),
        ),
    ))

  by_source = Counter(str(row.get("source_name", "")) for row in best_rows)
  by_trick = Counter(str(row.get("trick_number", -1)) for row in best_rows)
  summary = {
      "kind": "led_compatible_best_per_state_selector_audit",
      "replay": str(args.replay),
      "metadata": str(args.metadata),
      "rows": int(len(rows)),
      "raw_candidate_actions": int(raw_candidate_count),
      "candidate_states": int(candidate_states),
      "selected_rows": int(len(best_rows)),
      "positive_value_margin": float(args.positive_value_margin),
      "safe_risk_delta_max": float(args.safe_risk_delta_max),
      "force_led_lane_burn_negative": bool(args.force_led_lane_burn_negative),
      "led_lane_red_exception": str(args.led_lane_red_exception),
      "by_source": dict(sorted(by_source.items())),
      "by_trick": dict(sorted(by_trick.items(), key=lambda item: int(item[0]))),
  }
  payload = {
      "selected": best_rows,
      "false_positives": [],
      "summary": summary,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  summary_path = (
      Path(args.summary_out)
      if str(args.summary_out or "").strip()
      else out.with_suffix(out.suffix + ".summary.json")
  )
  encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
  summary["audit_sha256"] = hashlib.sha256(encoded).hexdigest()
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
