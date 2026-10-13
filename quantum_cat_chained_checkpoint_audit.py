#!/usr/bin/env python3
"""Audit a checkpoint on source-held-out chained correction/base replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay
from quantum_cat_calibration_transfer_audit import (
    _load_policy_context_model,
    _policy_probs_from_row,
)


def _sidecar_rows(path: Path) -> list[dict]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  rows = payload if isinstance(payload, list) else payload.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"metadata rows are not a list: {path}")
  return rows


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
  if total <= 0:
    return None
  p = float(successes) / float(total)
  denom = 1.0 + z * z / total
  center = (p + z * z / (2.0 * total)) / denom
  half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
  return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def _load_rows(
    replay_paths: list[str],
    sidecar_paths: list[str],
    kind: str,
) -> list[dict]:
  if len(replay_paths) != len(sidecar_paths):
    raise ValueError(f"{kind} replay/sidecar counts differ")
  result = []
  for replay_spec, sidecar_spec in zip(replay_paths, sidecar_paths):
    replay_path = Path(replay_spec)
    sidecar_path = Path(sidecar_spec)
    replay = load_replay(replay_path)
    metadata = _sidecar_rows(sidecar_path)
    if len(replay) != len(metadata):
      raise ValueError(
          f"replay/sidecar row mismatch: {replay_path}: "
          f"{len(replay)} != {len(metadata)}"
      )
    for index, (example, row) in enumerate(zip(replay, metadata)):
      decision = row.get("decision", {})
      base_action = int(
          row.get(
              "policy_base_action",
              row.get("policy_best_action", decision.get("policy_action", -1)),
          )
      )
      target_action = (
          int(row.get("applied_action", -1))
          if kind == "audited_correction"
          else base_action
      )
      proposal_action = int(row.get("proposed_best_action", target_action))
      legal = [int(action) for action in np.flatnonzero(np.asarray(example[1]) > 0.5)]
      if base_action not in legal or target_action not in legal or proposal_action not in legal:
        raise ValueError(f"illegal base/target/proposal: {replay_path}/row={index}")
      result.append({
          "example": example,
          "metadata": row,
          "kind": kind,
          "source_group_id": str(row.get("source_group_id", "")),
          "base_action": base_action,
          "target_action": target_action,
          "proposal_action": proposal_action,
          "legal": legal,
      })
  return result


def _threshold_summary(rows: list[dict], margin_threshold: float) -> dict:
  positives = sum(row["kind"] == "audited_correction" for row in rows)
  base_keeps = len(rows) - positives
  overrides = 0
  correct_overrides = 0
  false_overrides = 0
  correction_hits = 0
  correction_abstains = 0
  correction_wrong_action = 0
  base_abstains = 0
  base_proposal_adoptions = 0
  for row in rows:
    top_action = int(row["top_action"])
    base_action = int(row["base_action"])
    margin = float(row["top_vs_base_prob_margin"])
    selected = (
        top_action
        if top_action != base_action and margin + 1e-12 >= float(margin_threshold)
        else base_action
    )
    override = selected != base_action
    correct = (
        row["kind"] == "audited_correction"
        and selected == int(row["target_action"])
    )
    overrides += int(override)
    correct_overrides += int(correct)
    false_overrides += int(override and not correct)
    if row["kind"] == "audited_correction":
      correction_hits += int(correct)
      correction_abstains += int(selected == base_action)
      correction_wrong_action += int(override and not correct)
    else:
      base_abstains += int(selected == base_action)
      base_proposal_adoptions += int(selected == int(row["proposal_action"]))
  correct_decisions = correction_hits + base_abstains
  precision = None if overrides == 0 else correct_overrides / overrides
  recall = None if positives == 0 else correction_hits / positives
  base_rate = None if base_keeps == 0 else base_abstains / base_keeps
  return {
      "top_vs_base_prob_margin_threshold": float(margin_threshold),
      "rows": len(rows),
      "positive_rows": int(positives),
      "base_keep_rows": int(base_keeps),
      "overrides": int(overrides),
      "correct_overrides": int(correct_overrides),
      "false_overrides": int(false_overrides),
      "override_precision": None if precision is None else round(float(precision), 6),
      "override_precision_wilson95": _wilson(correct_overrides, overrides),
      "correction_recall": None if recall is None else round(float(recall), 6),
      "correction_recall_wilson95": _wilson(correction_hits, positives),
      "correction_hits": int(correction_hits),
      "correction_abstains": int(correction_abstains),
      "correction_wrong_action": int(correction_wrong_action),
      "base_keep_rate": None if base_rate is None else round(float(base_rate), 6),
      "base_keep_rate_wilson95": _wilson(base_abstains, base_keeps),
      "base_abstains": int(base_abstains),
      "base_proposal_adoptions": int(base_proposal_adoptions),
      "decision_accuracy": round(float(correct_decisions / max(1, len(rows))), 6),
  }


def checkpoint_audit_report(
    checkpoint: str,
    positive_paths: list[str],
    positive_sidecars: list[str],
    base_paths: list[str],
    base_sidecars: list[str],
    *,
    players: int,
    device: str,
    thresholds: list[float],
    min_overrides: int,
    min_precision: float,
    min_recall: float,
    min_base_keep_rate: float,
) -> dict:
  rows = [
      *_load_rows(positive_paths, positive_sidecars, "audited_correction"),
      *_load_rows(base_paths, base_sidecars, "audit_rejected_base_keep"),
  ]
  if not rows:
    raise ValueError("no replay rows to audit")
  model_args = argparse.Namespace(
      policy_context_mode="summary",
      policy_context_checkpoint=str(checkpoint),
      policy_context_device=str(device),
      players=int(players),
  )
  model, torch_device = _load_policy_context_model(model_args, rows[0]["example"])
  for row in rows:
    probs = _policy_probs_from_row(row["example"], model, torch_device)
    legal = row["legal"]
    top_action = int(max(legal, key=lambda action: float(probs[action])))
    base_action = int(row["base_action"])
    target_action = int(row["target_action"])
    proposal_action = int(row["proposal_action"])
    row.update({
        "top_action": top_action,
        "top_prob": float(probs[top_action]),
        "base_prob": float(probs[base_action]),
        "target_prob": float(probs[target_action]),
        "proposal_prob": float(probs[proposal_action]),
        "top_vs_base_prob_margin": float(probs[top_action] - probs[base_action]),
        "target_vs_base_prob_margin": float(
            probs[target_action] - probs[base_action]
        ),
    })

  threshold_reports = [
      _threshold_summary(rows, float(threshold)) for threshold in thresholds
  ]
  eligible = [
      report for report in threshold_reports
      if int(report["overrides"]) >= int(min_overrides)
      and report["override_precision"] is not None
      and float(report["override_precision"]) >= float(min_precision)
      and report["correction_recall"] is not None
      and float(report["correction_recall"]) >= float(min_recall)
      and report["base_keep_rate"] is not None
      and float(report["base_keep_rate"]) >= float(min_base_keep_rate)
  ]
  best = max(
      threshold_reports,
      key=lambda item: (
          item["override_precision"] is not None,
          item["override_precision"] or -1.0,
          item["correction_recall"] or -1.0,
          item["base_keep_rate"] or -1.0,
          item["overrides"],
      ),
  )
  source_rows = defaultdict(list)
  for row in rows:
    source_rows[row["source_group_id"]].append(row)
  source_report = {
      source: _threshold_summary(items, 0.0)
      for source, items in sorted(source_rows.items())
  }
  positive_rows = [row for row in rows if row["kind"] == "audited_correction"]
  base_rows = [row for row in rows if row["kind"] == "audit_rejected_base_keep"]
  return {
      "kind": "chained_checkpoint_policy_audit",
      "checkpoint": str(checkpoint),
      "players": int(players),
      "positive_rows": len(positive_rows),
      "base_keep_rows": len(base_rows),
      "positive_target_prob_mean": round(float(np.mean([
          row["target_prob"] for row in positive_rows
      ])), 6),
      "positive_target_vs_base_margin_mean": round(float(np.mean([
          row["target_vs_base_prob_margin"] for row in positive_rows
      ])), 6),
      "base_prob_mean": round(float(np.mean([
          row["base_prob"] for row in base_rows
      ])), 6),
      "threshold_reports": threshold_reports,
      "best_threshold": best,
      "gate": {
          "passed": bool(eligible),
          "min_overrides": int(min_overrides),
          "min_precision": float(min_precision),
          "min_recall": float(min_recall),
          "min_base_keep_rate": float(min_base_keep_rate),
          "eligible_thresholds": len(eligible),
          "best_eligible_threshold": eligible[0] if eligible else None,
      },
      "by_source_at_zero_margin": source_report,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--positive", action="append", required=True)
  parser.add_argument("--positive-sidecar", action="append", required=True)
  parser.add_argument("--base-keep", action="append", required=True)
  parser.add_argument("--base-keep-sidecar", action="append", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--margin-thresholds", default="0,0.01,0.02,0.05,0.1,0.2,0.3")
  parser.add_argument("--min-overrides", type=int, default=8)
  parser.add_argument("--min-precision", type=float, default=0.85)
  parser.add_argument("--min-recall", type=float, default=0.25)
  parser.add_argument("--min-base-keep-rate", type=float, default=0.90)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  thresholds = [
      float(item.strip())
      for item in str(args.margin_thresholds).split(",")
      if item.strip()
  ]
  report = checkpoint_audit_report(
      args.checkpoint,
      args.positive,
      args.positive_sidecar,
      args.base_keep,
      args.base_keep_sidecar,
      players=args.players,
      device=args.device,
      thresholds=thresholds,
      min_overrides=args.min_overrides,
      min_precision=args.min_precision,
      min_recall=args.min_recall,
      min_base_keep_rate=args.min_base_keep_rate,
  )
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
