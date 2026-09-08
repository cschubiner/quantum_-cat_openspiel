#!/usr/bin/env python3
"""Relabel dense failure-miner replay action values with bounded safety targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import quantum_cat_failure_miner as failure_miner
from quantum_cat_alphazero_torch import load_replay
from quantum_cat_alphazero_torch import save_replay


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True)
  parser.add_argument(
      "--metadata-input",
      action="append",
      default=[],
      help=(
          "Optional metadata JSON for each --input. If omitted, the tool uses "
          "<input>.metadata.json when present."
      ),
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--value-scale", type=float, default=20.0)
  parser.add_argument(
      "--dense-action-value-objective",
      choices=("safety", "safety_raw", "score"),
      default="safety",
  )
  parser.add_argument("--dense-safety-value-safe-target", type=float, default=0.9)
  parser.add_argument("--dense-safety-value-paradox-target", type=float, default=-0.55)
  parser.add_argument("--dense-safety-value-forced-penalty", type=float, default=0.25)
  parser.add_argument("--dense-safety-value-actor-penalty", type=float, default=0.10)
  parser.add_argument("--dense-safety-value-score-weight", type=float, default=0.05)
  parser.add_argument("--dense-safety-value-min", type=float, default=-0.95)
  parser.add_argument("--dense-safety-value-max", type=float, default=0.95)
  return parser.parse_args()


def _default_metadata_path(replay_path):
  return replay_path.with_suffix(replay_path.suffix + ".metadata.json")


def _load_metadata_rows(path):
  data = json.loads(Path(path).read_text())
  rows = data.get("rows")
  if not isinstance(rows, list):
    raise ValueError(f"{path} does not contain a top-level rows list")
  return rows, data


def _phase(metadata_row):
  metadata_row = metadata_row or {}
  if "phase" in metadata_row:
    return str(metadata_row.get("phase", "unknown"))
  dense = metadata_row.get("dense")
  if isinstance(dense, dict):
    return str(dense.get("phase", "unknown"))
  return "unknown"


def _action_outcomes(metadata_row):
  outcomes = (metadata_row or {}).get("action_outcomes")
  if isinstance(outcomes, list) and outcomes:
    return outcomes
  dense = (metadata_row or {}).get("dense")
  if isinstance(dense, dict):
    outcomes = dense.get("action_outcomes")
    if isinstance(outcomes, list) and outcomes:
      return outcomes
  return None


def relabel_row(row, metadata_row, args):
  if metadata_row is None:
    raise ValueError("Dense safety relabeling requires failure-miner metadata")
  outcomes = _action_outcomes(metadata_row)
  if not outcomes:
    raise ValueError("Metadata row missing non-empty action_outcomes")
  value_targets = np.zeros_like(row[1], dtype=np.float32)
  value_mask = np.zeros_like(row[1], dtype=np.float32)
  for outcome in outcomes:
    action = int(outcome["action"])
    if action < 0 or action >= value_targets.shape[0]:
      continue
    value_targets[action] = failure_miner.dense_action_value_target(
        outcome, args, args.value_scale
    )
    value_mask[action] = 1.0
  return (
      row[0],
      row[1],
      row[2],
      row[3],
      row[4],
      row[5],
      row[6],
      row[7],
      row[8],
      value_targets,
      value_mask,
      row[11],
      row[12] if len(row) > 12 else None,
  )


def main():
  args = parse_args()
  input_paths = [Path(item) for item in args.input]
  metadata_paths = [Path(item) for item in args.metadata_input]
  if metadata_paths and len(metadata_paths) != len(input_paths):
    raise ValueError("--metadata-input must be supplied once per --input")

  relabeled = []
  source_summaries = []
  phase_counts = Counter()
  changed_rows = 0
  labeled_actions = 0
  target_min = None
  target_max = None

  for idx, replay_path in enumerate(input_paths):
    rows, replay_metadata = load_replay(replay_path, return_metadata=True)
    metadata_path = (
        metadata_paths[idx]
        if metadata_paths
        else _default_metadata_path(replay_path)
    )
    metadata_rows, metadata = _load_metadata_rows(metadata_path)
    if len(metadata_rows) != len(rows):
      raise ValueError(
          f"{metadata_path} row count {len(metadata_rows)} does not match "
          f"{replay_path} row count {len(rows)}"
      )
    source_phase_counts = Counter()
    for row, metadata_row in zip(rows, metadata_rows):
      new_row = relabel_row(row, metadata_row, args)
      relabeled.append(new_row)
      phase = _phase(metadata_row)
      phase_counts[phase] += 1
      source_phase_counts[phase] += 1
      if bool(metadata_row.get("policy_target_changed", False)):
        changed_rows += 1
      values = new_row[9][new_row[10] > 0]
      labeled_actions += int(values.shape[0])
      if values.size:
        local_min = float(np.min(values))
        local_max = float(np.max(values))
        target_min = local_min if target_min is None else min(target_min, local_min)
        target_max = local_max if target_max is None else max(target_max, local_max)
    source_summaries.append({
        "replay": str(replay_path),
        "metadata": str(metadata_path),
        "rows": len(rows),
        "phase_counts": dict(source_phase_counts),
        "replay_metadata": replay_metadata,
        "dense_metadata_summary": {
            key: value for key, value in metadata.items() if key != "rows"
        },
    })

  summary = {
      "source": "quantum_cat_relabel_dense_safety_replay",
      "rows": len(relabeled),
      "phase_counts": dict(phase_counts),
      "policy_target_changed_rows": changed_rows,
      "labeled_actions": labeled_actions,
      "target_min": target_min,
      "target_max": target_max,
      "value_scale": float(args.value_scale),
      "dense_action_value_objective": str(args.dense_action_value_objective),
      "dense_safety_value_safe_target": float(args.dense_safety_value_safe_target),
      "dense_safety_value_paradox_target": float(
          args.dense_safety_value_paradox_target
      ),
      "dense_safety_value_forced_penalty": float(
          args.dense_safety_value_forced_penalty
      ),
      "dense_safety_value_actor_penalty": float(
          args.dense_safety_value_actor_penalty
      ),
      "dense_safety_value_score_weight": float(
          args.dense_safety_value_score_weight
      ),
      "dense_safety_value_min": float(args.dense_safety_value_min),
      "dense_safety_value_max": float(args.dense_safety_value_max),
      "sources": source_summaries,
  }
  save_replay(relabeled, args.out, metadata=summary)
  if args.summary_out:
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
