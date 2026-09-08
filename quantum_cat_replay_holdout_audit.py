#!/usr/bin/env python3
"""Audit source-disjoint replay train/validation banks and late-exit coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _sidecar_rows(path: Path) -> list[dict]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  rows = payload if isinstance(payload, list) else payload.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"sidecar rows are not a list: {path}")
  return rows


def _base_any_paradox_improvement(row: dict) -> float:
  explicit = row.get("best_vs_base_any_paradox_improvement")
  if explicit is not None:
    return float(explicit)
  records = {
      int(item.get("action")): item
      for item in row.get("rollout_records", [])
      if item.get("action") is not None
  }
  base_action = row.get("decision", {}).get("policy_action")
  best_action = row.get("best_action")
  if base_action is None or best_action is None:
    return 0.0
  base = records.get(int(base_action))
  best = records.get(int(best_action))
  if base is None or best is None:
    return 0.0
  return float(base.get("any_paradox_rate", 0.0)) - float(
      best.get("any_paradox_rate", 0.0)
  )


def _bank(
    path: Path,
    sidecar: Path,
    min_base_improvement: float,
    min_audit_improvement: float,
    bank_kind: str = "correction",
) -> dict:
  replay, metadata = load_replay(path, return_metadata=True)
  groups = list(metadata.get("row_group_ids") or [])
  if len(groups) != len(replay):
    raise ValueError(
        f"row_group_ids count mismatch for {path}: {len(groups)} != {len(replay)}"
    )
  rows = _sidecar_rows(sidecar)
  if len(rows) != len(replay):
    raise ValueError(
        f"sidecar row count mismatch for {path}: {len(rows)} != {len(replay)}"
    )
  late_forced = [
      row for row in rows
      if 5 <= int(row.get("decision", {}).get("trick_number", -1)) <= 7
      and float(row.get("forced_paradox_rate_spread", 0.0)) >= 0.5
  ]
  trick_counts = Counter(
      int(row.get("decision", {}).get("trick_number", -1)) for row in rows
  )
  base_keep_target_rows = 0
  if bank_kind == "audit_rejected_base":
    for example, row in zip(replay, rows):
      base_action = int(row.get("decision", {}).get("policy_action", -1))
      target = np.asarray(example[2], dtype=np.float32)
      selected_action = int(example[5])
      if (
          0 <= base_action < len(target)
          and selected_action == base_action
          and int(np.argmax(target)) == base_action
          and np.isclose(float(target[base_action]), 1.0)
          and np.isclose(float(target.sum()), 1.0)
      ):
        base_keep_target_rows += 1
  return {
      "path": str(path),
      "sha256": _sha256(path),
      "rows": len(replay),
      "source_groups": sorted(set(str(group) for group in groups)),
      "source_group_count": len(set(str(group) for group in groups)),
      "late_forced_exit_contrast_rows": len(late_forced),
      "policy_target_changed_rows": sum(
          bool(row.get("policy_target_changed")) for row in rows
      ),
      "information_safe_teacher_rows": sum(
          bool(row.get("information_safe_rollout_teacher")) for row in rows
      ),
      "base_protected_teacher_rows": sum(
          _base_any_paradox_improvement(row) >= float(min_base_improvement)
          for row in rows
      ),
      "independent_belief_validated_teacher_rows": sum(
          bool(row.get("independent_belief_validation")) for row in rows
      ),
      "audit_base_protected_teacher_rows": sum(
          bool(row.get("independent_belief_validation"))
          and row.get("audit_best_vs_base_any_paradox_improvement") is not None
          and float(row["audit_best_vs_base_any_paradox_improvement"])
          >= float(min_audit_improvement)
          for row in rows
      ),
      "base_keep_target_rows": int(base_keep_target_rows),
      "audit_rejected_base_rows": sum(
          bool(row.get("audit_override_rejected"))
          and bool(row.get("audit_base_keep_validation"))
          for row in rows
      ),
      "selection_proposal_protected_rows": sum(
          row.get(
              "selection_proposed_best_vs_base_any_paradox_improvement"
          ) is not None
          and float(
              row["selection_proposed_best_vs_base_any_paradox_improvement"]
          ) >= float(min_base_improvement)
          for row in rows
      ),
      "fresh_audit_rejection_confirmed_rows": sum(
          row.get("audit_best_vs_base_any_paradox_improvement") is not None
          and float(row["audit_best_vs_base_any_paradox_improvement"])
          < float(min_audit_improvement)
          for row in rows
      ),
      "proposed_action_changed_rows": sum(
          row.get("proposed_best_action") is not None
          and row.get("decision", {}).get("policy_action") is not None
          and int(row["proposed_best_action"])
          != int(row["decision"]["policy_action"])
          for row in rows
      ),
      "min_base_any_paradox_improvement": float(min_base_improvement),
      "min_audit_base_any_paradox_improvement": float(min_audit_improvement),
      "decision_trick_counts": {
          str(key): int(value) for key, value in sorted(trick_counts.items())
      },
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", required=True)
  parser.add_argument("--validation", required=True)
  parser.add_argument("--train-sidecar", required=True)
  parser.add_argument("--validation-sidecar", required=True)
  parser.add_argument("--expected-total-rows", type=int, default=0)
  parser.add_argument("--min-late-forced-rows", type=int, default=1)
  parser.add_argument(
      "--bank-kind",
      choices=("correction", "audit_rejected_base"),
      default="correction",
      help=(
          "Audit strict correction labels, or separately audit hard base-keep "
          "targets emitted when a proposed correction fails its fresh-world "
          "validation."
      ),
  )
  parser.add_argument(
      "--min-base-any-paradox-improvement",
      type=float,
      default=0.0,
      help="Required paired survival improvement over the source policy per row.",
  )
  parser.add_argument(
      "--require-independent-belief-validation",
      action="store_true",
      help=(
          "Require every row's selected teacher action to pass a second audit "
          "on fresh actor-information-state worlds."
      ),
  )
  parser.add_argument(
      "--min-audit-base-any-paradox-improvement",
      type=float,
      default=0.0,
      help="Required paired survival improvement on fresh audit belief worlds.",
  )
  parser.add_argument(
      "--require-trick-coverage",
      default="",
      help=(
          "Optional comma-separated trick numbers that must each appear in "
          "both train and validation, for example 0,1,2,3,4,5,6,7."
      ),
  )
  parser.add_argument(
      "--allow-unsafe-teacher",
      action="store_true",
      help=(
          "Allow legacy rows whose rollout targets used the simulator's true "
          "hidden deal. Never use this relaxation for a promotion bank."
      ),
  )
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  train = _bank(
      Path(args.train), Path(args.train_sidecar),
      args.min_base_any_paradox_improvement,
      args.min_audit_base_any_paradox_improvement,
      args.bank_kind,
  )
  validation = _bank(
      Path(args.validation), Path(args.validation_sidecar),
      args.min_base_any_paradox_improvement,
      args.min_audit_base_any_paradox_improvement,
      args.bank_kind,
  )
  overlap = sorted(set(train["source_groups"]) & set(validation["source_groups"]))
  total_rows = int(train["rows"] + validation["rows"])
  late_rows = int(
      train["late_forced_exit_contrast_rows"]
      + validation["late_forced_exit_contrast_rows"]
  )
  errors = []
  if overlap:
    errors.append(f"source group overlap: {overlap}")
  if args.expected_total_rows and total_rows != args.expected_total_rows:
    errors.append(
        f"total rows {total_rows} != expected {args.expected_total_rows}"
    )
  if args.bank_kind == "correction" and late_rows < args.min_late_forced_rows:
    errors.append(
        f"late forced-exit rows {late_rows} < required {args.min_late_forced_rows}"
    )
  if args.bank_kind == "correction":
    if (
        train["policy_target_changed_rows"] != train["rows"]
        or validation["policy_target_changed_rows"] != validation["rows"]
    ):
      errors.append("not every replay row changes the source policy target")
  else:
    for name, bank in (("train", train), ("validation", validation)):
      if bank["base_keep_target_rows"] != bank["rows"]:
        errors.append(f"{name} has a row that does not hard-target its base action")
      if bank["audit_rejected_base_rows"] != bank["rows"]:
        errors.append(f"{name} has a row without audit-rejection provenance")
      if bank["selection_proposal_protected_rows"] != bank["rows"]:
        errors.append(
            f"{name} has a proposal below selection improvement "
            f"{args.min_base_any_paradox_improvement}"
        )
      if bank["fresh_audit_rejection_confirmed_rows"] != bank["rows"]:
        errors.append(
            f"{name} has a proposal that did not fail fresh audit threshold "
            f"{args.min_audit_base_any_paradox_improvement}"
        )
      if bank["proposed_action_changed_rows"] != bank["rows"]:
        errors.append(f"{name} has a proposal that did not change the base action")
  if not args.allow_unsafe_teacher and (
      train["information_safe_teacher_rows"] != train["rows"]
      or validation["information_safe_teacher_rows"] != validation["rows"]
  ):
    errors.append(
        "not every replay row uses an actor-information-state rollout teacher"
    )
  if args.bank_kind == "correction" and (
      train["base_protected_teacher_rows"] != train["rows"]
      or validation["base_protected_teacher_rows"] != validation["rows"]
  ):
    errors.append(
        "not every replay row clears the required paired base-policy survival "
        f"improvement {args.min_base_any_paradox_improvement}"
    )
  require_audit = bool(args.require_independent_belief_validation) or (
      float(args.min_audit_base_any_paradox_improvement) > 0.0
  )
  if args.bank_kind == "correction" and require_audit and (
      train["independent_belief_validated_teacher_rows"] != train["rows"]
      or validation["independent_belief_validated_teacher_rows"]
      != validation["rows"]
  ):
    errors.append(
        "not every replay row passed independent held-out belief validation"
    )
  if args.bank_kind == "correction" and require_audit and (
      train["audit_base_protected_teacher_rows"] != train["rows"]
      or validation["audit_base_protected_teacher_rows"] != validation["rows"]
  ):
    errors.append(
        "not every replay row clears the required held-out paired base-policy "
        f"survival improvement {args.min_audit_base_any_paradox_improvement}"
    )
  required_tricks = {
      int(item.strip())
      for item in str(args.require_trick_coverage or "").split(",")
      if item.strip()
  }
  for name, bank in (("train", train), ("validation", validation)):
    observed = {int(item) for item in bank["decision_trick_counts"]}
    missing = sorted(required_tricks - observed)
    if missing:
      errors.append(f"{name} missing required trick coverage: {missing}")

  report = {
      "kind": "source_disjoint_replay_holdout_audit",
      "bank_kind": str(args.bank_kind),
      "passed": not errors,
      "errors": errors,
      "total_rows": total_rows,
      "late_forced_exit_contrast_rows": late_rows,
      "min_base_any_paradox_improvement": float(
          args.min_base_any_paradox_improvement
      ),
      "require_independent_belief_validation": bool(
          args.require_independent_belief_validation
      ),
      "min_audit_base_any_paradox_improvement": float(
          args.min_audit_base_any_paradox_improvement
      ),
      "required_trick_coverage": sorted(required_tricks),
      "source_group_overlap": overlap,
      "train": train,
      "validation": validation,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if not errors else 2


if __name__ == "__main__":
  raise SystemExit(main())
