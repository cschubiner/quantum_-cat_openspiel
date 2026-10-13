#!/usr/bin/env python3
"""Derive fail-closed policy replay from independently audited proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay
from quantum_cat_improve_graft_miner import _candidate_target_from_audit


SAFE_TARGET_CONTRACT = "positive_only_any_player_v1"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", required=True)
  parser.add_argument("--audit-json", required=True)
  parser.add_argument("--metadata", default="")
  parser.add_argument("--out", required=True)
  parser.add_argument(
      "--require-paradox-scope",
      choices=("acting", "any"),
      default="any",
  )
  parser.add_argument(
      "--positive-repeat",
      type=int,
      default=1,
      help=(
          "Repeat each uniquely audited POSITIVE row this many times in the "
          "derived training replay. Unique evidence counts remain explicit in "
          "the metadata; leave at 1 for validation replays."
      ),
  )
  return parser.parse_args()


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def safe_target_from_audit_row(row: dict, positive_margin: float, max_delta: float):
  base_action = int(row["policy_best_action"])
  audit = {
      int(action): dict(stats)
      for action, stats in dict(row.get("audit") or {}).items()
  }
  candidate_actions = [
      int(item["action"])
      for item in list(row.get("candidate_evaluations") or [])
  ]
  if not candidate_actions:
    proposed = row.get("primary_proposal_action", row.get("proposed_action"))
    if proposed is not None and int(proposed) != base_action:
      candidate_actions = [int(proposed)]
  return _candidate_target_from_audit(
      audit,
      base_action,
      candidate_actions,
      SimpleNamespace(
          calibration_positive_margin=float(positive_margin),
          calibration_max_positive_paradox_delta=float(max_delta),
      ),
  )


def action_string_for_target(audit_row: dict, target: int) -> str:
  if int(target) == int(audit_row["policy_best_action"]):
    return str(audit_row.get("policy_best_action_string", target))
  for item in list(audit_row.get("candidate_evaluations") or []):
    if int(item.get("action", -1)) == int(target):
      return str(item.get("action_string", target))
  if int(target) == int(audit_row.get("proposed_action", -1)):
    return str(audit_row.get("proposed_action_string", target))
  return str(target)


def preserve_candidate_proposal_context(audit_row: dict, evaluations: list[dict]):
  """Merge recomputed audit classes with the original proposal evidence."""
  source_by_action = {
      int(item["action"]): dict(item)
      for item in list(audit_row.get("candidate_evaluations") or [])
  }
  proposal_by_action = {
      int(item["action"]): dict(item)
      for item in list(audit_row.get("proposal_candidates") or [])
      if item.get("action") is not None
  }
  merged = []
  for evaluation in evaluations:
    action = int(evaluation["action"])
    row = dict(source_by_action.get(action, {}))
    proposal = proposal_by_action.get(action, {})
    if proposal:
      row.setdefault("proposal_improve_prob", proposal.get("improve"))
      row.setdefault("proposal_safe_prob", proposal.get("safe"))
      row.setdefault("proposal_policy_prob", proposal.get("policy_prob"))
    row.update(evaluation)
    row["action"] = action
    merged.append(row)
  return merged


def main() -> None:
  args = parse_args()
  if int(args.positive_repeat) < 1:
    raise ValueError("--positive-repeat must be at least 1")
  replay_path = Path(args.replay)
  audit_path = Path(args.audit_json)
  out_path = Path(args.out)
  replay, replay_metadata = load_replay(replay_path, return_metadata=True)
  audit = json.loads(audit_path.read_text(encoding="utf-8"))
  audit_rows = list(audit.get("rows") or [])
  if len(replay) != len(audit_rows):
    raise ValueError(
        f"replay/audit row mismatch: {len(replay)} != {len(audit_rows)}"
    )
  scope = str(
      audit.get("counterfactual_action_paradox_scope")
      or replay_metadata.get("counterfactual_action_paradox_scope")
      or ""
  )
  if scope != str(args.require_paradox_scope):
    raise ValueError(
        "proposal replay paradox scope mismatch: "
        f"expected {args.require_paradox_scope}, got {scope or 'missing'}"
    )
  metadata_path = (
      Path(args.metadata)
      if args.metadata
      else Path(str(replay_path) + ".metadata.json")
  )
  source_sidecar = (
      json.loads(metadata_path.read_text(encoding="utf-8"))
      if metadata_path.exists()
      else {}
  )
  positive_margin_value = audit.get(
      "calibration_positive_margin",
      source_sidecar.get(
          "positive_margin",
          replay_metadata.get("calibration_positive_margin"),
      ),
  )
  max_delta_value = audit.get(
      "calibration_max_positive_paradox_delta",
      source_sidecar.get(
          "max_positive_paradox_delta",
          replay_metadata.get("calibration_max_positive_paradox_delta"),
      ),
  )
  if positive_margin_value is None or max_delta_value is None:
    raise ValueError("proposal replay is missing audited positive gate thresholds")
  positive_margin = float(positive_margin_value)
  max_delta = float(max_delta_value)
  unique_repaired = []
  target_rows = []
  changed = 0
  positive_targets = 0
  for idx, (example, audit_row) in enumerate(zip(replay, audit_rows)):
    target, target_class, evaluations = safe_target_from_audit_row(
        audit_row, positive_margin, max_delta
    )
    example = list(example)
    legal_mask = np.asarray(example[1], dtype=np.float32)
    if target < 0 or target >= len(legal_mask) or legal_mask[target] <= 0.0:
      raise ValueError(f"row {idx} safe target is not legal: {target}")
    old_target = int(np.argmax(np.asarray(example[2], dtype=np.float32)))
    policy = np.zeros_like(np.asarray(example[2], dtype=np.float32))
    policy[int(target)] = 1.0
    example[2] = policy
    example[5] = int(target)
    unique_repaired.append(tuple(example))
    changed += int(old_target != int(target))
    positive_targets += int(target_class == "POSITIVE")
    target_rows.append({
        "row_index": int(idx),
        "old_target_action": int(old_target),
        "deployment_target_action": int(target),
        "deployment_target_action_string": action_string_for_target(
            audit_row, target
        ),
        "target_calibration_class": str(target_class),
        "candidate_evaluations": preserve_candidate_proposal_context(
            audit_row, evaluations
        ),
    })

  sample_indices = []
  for idx, target_row in enumerate(target_rows):
    repeat = (
        int(args.positive_repeat)
        if target_row["target_calibration_class"] == "POSITIVE"
        else 1
    )
    sample_indices.extend([idx] * repeat)
  repaired = [unique_repaired[idx] for idx in sample_indices]
  weighted_positive_targets = int(positive_targets * args.positive_repeat)
  base_keep_targets = int(len(unique_repaired) - positive_targets)

  derived_metadata = dict(replay_metadata or {})
  derived_metadata.update({
      "safe_policy_target_contract": SAFE_TARGET_CONTRACT,
      "counterfactual_action_paradox_scope": scope,
      "source_replay": str(replay_path),
      "source_replay_sha256": file_sha256(replay_path),
      "changed_policy_targets": int(changed),
      "unique_policy_rows": int(len(unique_repaired)),
      "unique_positive_policy_targets": int(positive_targets),
      "unique_base_keep_policy_targets": int(base_keep_targets),
      "positive_repeat": int(args.positive_repeat),
      "weighted_policy_rows": int(len(repaired)),
      "weighted_positive_policy_targets": int(weighted_positive_targets),
      "weighted_base_keep_policy_targets": int(base_keep_targets),
      "calibration_positive_margin": float(positive_margin),
      "calibration_max_positive_paradox_delta": float(max_delta),
  })
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_replay(repaired, out_path, metadata=derived_metadata)

  sidecar_rows = list(source_sidecar.get("rows") or [])
  if sidecar_rows and len(sidecar_rows) != len(target_rows):
    raise ValueError(
        f"metadata/audit row mismatch: {len(sidecar_rows)} != {len(target_rows)}"
    )
  if not sidecar_rows:
    sidecar_rows = [{} for _ in target_rows]
  for metadata_row, target_row in zip(sidecar_rows, target_rows):
    metadata_row.update(target_row)
    metadata_row["safe_policy_target_contract"] = SAFE_TARGET_CONTRACT
  weighted_sidecar_rows = [dict(sidecar_rows[idx]) for idx in sample_indices]
  for weighted_idx, (metadata_row, source_idx) in enumerate(zip(
      weighted_sidecar_rows, sample_indices
  )):
    metadata_row["weighted_row_index"] = int(weighted_idx)
    metadata_row["unique_source_row_index"] = int(source_idx)
  derived_sidecar = dict(source_sidecar)
  derived_sidecar.update(derived_metadata)
  derived_sidecar["rows"] = weighted_sidecar_rows
  out_sidecar = Path(str(out_path) + ".metadata.json")
  out_sidecar.write_text(
      json.dumps(derived_sidecar, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = {
      "out": str(out_path),
      "out_metadata": str(out_sidecar),
      "rows": int(len(repaired)),
      "unique_rows": int(len(unique_repaired)),
      "changed_policy_targets": int(changed),
      "unique_positive_policy_targets": int(positive_targets),
      "unique_base_keep_policy_targets": int(base_keep_targets),
      "positive_repeat": int(args.positive_repeat),
      "weighted_positive_policy_targets": int(weighted_positive_targets),
      "weighted_base_keep_policy_targets": int(base_keep_targets),
      "counterfactual_action_paradox_scope": scope,
      "safe_policy_target_contract": SAFE_TARGET_CONTRACT,
      "source_replay_sha256": derived_metadata["source_replay_sha256"],
      "out_replay_sha256": file_sha256(out_path),
  }
  summary_path = Path(str(out_path) + ".summary.json")
  summary_path.write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
