#!/usr/bin/env python3
"""Source-held-out gate for action-value or action-risk checkpoint margins."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay
from quantum_cat_calibration_transfer_audit import _load_policy_context_model
from quantum_cat_legal_set_selector_trainer import (
    _action_aux_predictions_from_row,
)


_TARGET_KINDS = {"audited_correction", "audit_rejected_base_keep"}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _metadata_path(replay_path: Path) -> Path:
  return replay_path.with_suffix(".metadata.json")


def _load_metadata_rows(path: Path) -> list[dict]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(payload, list):
    rows = payload
  elif isinstance(payload, dict):
    rows = payload.get("row_metadata", payload.get("rows", []))
  else:
    rows = []
  if not isinstance(rows, list):
    raise ValueError(f"metadata rows are not a list: {path}")
  return rows


def _wilson(successes: int, total: int, z: float = 1.959963984540054):
  if total <= 0:
    return None
  p = float(successes) / float(total)
  denom = 1.0 + z * z / total
  center = (p + z * z / (2.0 * total)) / denom
  half = z * math.sqrt(
      (p * (1.0 - p) + z * z / (4.0 * total)) / total
  ) / denom
  return [
      round(max(0.0, center - half), 6),
      round(min(1.0, center + half), 6),
  ]


def _load_rows(
    replay_specs: list[str],
    metadata_specs: list[str],
    split: str,
) -> tuple[list[dict], list[dict]]:
  if metadata_specs and len(metadata_specs) != len(replay_specs):
    raise ValueError(
        f"{split} replay/metadata counts differ: "
        f"{len(replay_specs)} vs {len(metadata_specs)}"
    )
  rows = []
  sources = []
  for source_index, replay_spec in enumerate(replay_specs):
    replay_path = Path(replay_spec)
    metadata_path = (
        Path(metadata_specs[source_index])
        if metadata_specs else _metadata_path(replay_path)
    )
    replay = load_replay(replay_path)
    metadata = _load_metadata_rows(metadata_path)
    if len(replay) != len(metadata):
      raise ValueError(
          f"{split} replay/metadata row mismatch for {replay_path}: "
          f"{len(replay)} vs {len(metadata)}"
      )
    source_name = f"{split}:{source_index}:{replay_path.name}"
    sources.append({
        "source": source_name,
        "replay": str(replay_path),
        "replay_sha256": _sha256(replay_path),
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "rows": len(replay),
    })
    for row_index, (example, meta) in enumerate(zip(replay, metadata)):
      kind = str(meta.get("target_kind", ""))
      if kind not in _TARGET_KINDS:
        raise ValueError(
            f"unsupported target_kind={kind!r}: {metadata_path}/row={row_index}"
        )
      base_action = int(
          meta.get("policy_base_action", meta.get("policy_best_action", -1))
      )
      proposed_action = int(
          meta.get("proposed_best_action", meta.get("proposed_action", -1))
      )
      legal_mask = np.asarray(example[1], dtype=np.float32) > 0.5
      if (
          base_action < 0
          or proposed_action < 0
          or base_action == proposed_action
          or base_action >= len(legal_mask)
          or proposed_action >= len(legal_mask)
          or not bool(legal_mask[base_action])
          or not bool(legal_mask[proposed_action])
      ):
        raise ValueError(
            f"invalid base/proposed pair: {replay_path}/row={row_index}"
        )
      rows.append({
          "example": example,
          "kind": kind,
          "source": source_name,
          "source_group_id": str(meta.get("source_group_id", source_name)),
          "base_action": base_action,
          "proposed_action": proposed_action,
          "row_index": row_index,
      })
  return rows, sources


def _score_rows(rows: list[dict], checkpoint: str, players: int, device: str,
                score_mode: str) -> list[dict]:
  if not rows:
    raise ValueError("no rows to score")
  model_args = argparse.Namespace(
      policy_context_mode="summary",
      policy_context_checkpoint=str(checkpoint),
      policy_context_device=str(device),
      players=int(players),
  )
  model, torch_device = _load_policy_context_model(
      model_args, rows[0]["example"]
  )
  scored = []
  for row in rows:
    values, risks = _action_aux_predictions_from_row(
        row["example"], model, torch_device
    )
    base_action = int(row["base_action"])
    proposed_action = int(row["proposed_action"])
    if score_mode == "action_value":
      base_score = float(values[base_action])
      proposed_score = float(values[proposed_action])
      improvement = proposed_score - base_score
    elif score_mode == "action_paradox":
      base_score = float(risks[base_action])
      proposed_score = float(risks[proposed_action])
      improvement = base_score - proposed_score
    else:
      raise ValueError(f"unsupported score mode: {score_mode}")
    scored.append({
        key: value for key, value in row.items() if key != "example"
    } | {
        "base_score": base_score,
        "proposed_score": proposed_score,
        "predicted_improvement": float(improvement),
    })
  return scored


def _threshold_summary(rows: list[dict], threshold: float) -> dict:
  positives = sum(row["kind"] == "audited_correction" for row in rows)
  base_keeps = len(rows) - positives
  overrides = 0
  correct_overrides = 0
  base_abstains = 0
  correction_hits = 0
  for row in rows:
    override = float(row["predicted_improvement"]) + 1e-12 >= float(threshold)
    correction = row["kind"] == "audited_correction"
    overrides += int(override)
    correct_overrides += int(override and correction)
    correction_hits += int(override and correction)
    base_abstains += int((not override) and (not correction))
  precision = None if overrides == 0 else correct_overrides / overrides
  recall = None if positives == 0 else correction_hits / positives
  base_keep_rate = None if base_keeps == 0 else base_abstains / base_keeps
  return {
      "threshold": float(threshold),
      "rows": len(rows),
      "positive_rows": int(positives),
      "base_keep_rows": int(base_keeps),
      "overrides": int(overrides),
      "correct_overrides": int(correct_overrides),
      "false_overrides": int(overrides - correct_overrides),
      "override_precision": (
          None if precision is None else round(float(precision), 6)
      ),
      "override_precision_wilson95": _wilson(correct_overrides, overrides),
      "correction_recall": None if recall is None else round(float(recall), 6),
      "correction_recall_wilson95": _wilson(correction_hits, positives),
      "base_keep_rate": (
          None if base_keep_rate is None else round(float(base_keep_rate), 6)
      ),
      "base_keep_rate_wilson95": _wilson(base_abstains, base_keeps),
  }


def _reports_by_source(rows: list[dict], threshold: float) -> dict[str, dict]:
  grouped = defaultdict(list)
  for row in rows:
    # A replay artifact may combine many independent collection seeds.  The
    # metadata group is the statistical source; the NPZ filename is only a
    # storage container and must not collapse per-source gates.
    grouped[str(row["source_group_id"])].append(row)
  return {
      source: _threshold_summary(source_rows, threshold)
      for source, source_rows in sorted(grouped.items())
  }


def _passes(
    aggregate: dict,
    by_source: dict[str, dict],
    *,
    min_overrides: int,
    min_precision: float,
    min_base_keep_rate: float,
    min_source_overrides: int,
) -> bool:
  if int(aggregate["overrides"]) < int(min_overrides):
    return False
  if (
      aggregate["override_precision"] is None
      or float(aggregate["override_precision"]) < float(min_precision)
      or aggregate["base_keep_rate"] is None
      or float(aggregate["base_keep_rate"]) < float(min_base_keep_rate)
  ):
    return False
  for report in by_source.values():
    if int(report["positive_rows"]) > 0:
      if int(report["overrides"]) < int(min_source_overrides):
        return False
      if (
          report["override_precision"] is None
          or float(report["override_precision"]) < float(min_precision)
      ):
        return False
    if (
        int(report["base_keep_rows"]) > 0
        and (
            report["base_keep_rate"] is None
            or float(report["base_keep_rate"]) < float(min_base_keep_rate)
        )
    ):
      return False
  return True


def audit_report(args) -> dict:
  calibration_rows, calibration_sources = _load_rows(
      args.calibration_replay, args.calibration_metadata, "calibration"
  )
  validation_rows, validation_sources = _load_rows(
      args.validation_replay, args.validation_metadata, "validation"
  )
  calibration_groups = {row["source_group_id"] for row in calibration_rows}
  validation_groups = {row["source_group_id"] for row in validation_rows}
  overlap = sorted(calibration_groups & validation_groups)
  if overlap:
    raise ValueError(f"calibration/validation source-group overlap: {overlap}")
  calibration_scored = _score_rows(
      calibration_rows, args.checkpoint, args.players, args.device, args.score_mode
  )
  validation_scored = _score_rows(
      validation_rows, args.checkpoint, args.players, args.device, args.score_mode
  )
  thresholds = sorted({float(item) for item in args.thresholds.split(",")})
  threshold_reports = []
  for threshold in thresholds:
    calibration = _threshold_summary(calibration_scored, threshold)
    validation = _threshold_summary(validation_scored, threshold)
    calibration_by_source = _reports_by_source(calibration_scored, threshold)
    validation_by_source = _reports_by_source(validation_scored, threshold)
    calibration_passed = _passes(
        calibration,
        calibration_by_source,
        min_overrides=args.min_calibration_overrides,
        min_precision=args.min_precision,
        min_base_keep_rate=args.min_base_keep_rate,
        min_source_overrides=args.min_source_overrides,
    )
    validation_passed = _passes(
        validation,
        validation_by_source,
        min_overrides=args.min_validation_overrides,
        min_precision=args.min_precision,
        min_base_keep_rate=args.min_base_keep_rate,
        min_source_overrides=args.min_source_overrides,
    )
    threshold_reports.append({
        "threshold": threshold,
        "calibration": calibration,
        "validation": validation,
        "calibration_by_source": calibration_by_source,
        "validation_by_source": validation_by_source,
        "calibration_passed": bool(calibration_passed),
        "validation_passed": bool(validation_passed),
        "passed": bool(calibration_passed and validation_passed),
    })
  eligible = [row for row in threshold_reports if row["passed"]]
  best = max(
      threshold_reports,
      key=lambda row: (
          row["validation"]["override_precision"] is not None,
          row["validation"]["override_precision"] or -1.0,
          row["validation"]["correct_overrides"],
          row["validation"]["base_keep_rate"] or -1.0,
      ),
  )
  checkpoint_path = Path(args.checkpoint)
  return {
      "schema_version": 1,
      "checkpoint": str(checkpoint_path),
      "checkpoint_sha256": _sha256(checkpoint_path),
      "players": int(args.players),
      "score_mode": str(args.score_mode),
      "source_group_overlap": overlap,
      "calibration_sources": calibration_sources,
      "validation_sources": validation_sources,
      "gate": {
          "min_calibration_overrides": int(args.min_calibration_overrides),
          "min_validation_overrides": int(args.min_validation_overrides),
          "min_source_overrides": int(args.min_source_overrides),
          "min_precision": float(args.min_precision),
          "min_base_keep_rate": float(args.min_base_keep_rate),
          "eligible_thresholds": len(eligible),
          "passed": bool(eligible),
      },
      "best_validation_threshold": best,
      "threshold_reports": threshold_reports,
  }


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--calibration-replay", action="append", required=True)
  parser.add_argument("--calibration-metadata", action="append", default=[])
  parser.add_argument("--validation-replay", action="append", required=True)
  parser.add_argument("--validation-metadata", action="append", default=[])
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
      "--score-mode", choices=("action_value", "action_paradox"), required=True
  )
  parser.add_argument(
      "--thresholds",
      default="0,0.0025,0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2",
  )
  parser.add_argument("--min-calibration-overrides", type=int, default=8)
  parser.add_argument("--min-validation-overrides", type=int, default=8)
  parser.add_argument("--min-source-overrides", type=int, default=1)
  parser.add_argument("--min-precision", type=float, default=0.9)
  parser.add_argument("--min-base-keep-rate", type=float, default=0.9)
  parser.add_argument("--out", required=True)
  return parser


def main() -> None:
  args = _parser().parse_args()
  report = audit_report(args)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
      "out": str(out_path),
      "checkpoint_sha256": report["checkpoint_sha256"],
      "gate": report["gate"],
      "best_validation_threshold": report["best_validation_threshold"],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
