#!/usr/bin/env python3
"""Convert audited best-of-rollouts rows into improve-graft calibration pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_cat_alphazero_torch import load_replay, save_replay


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--positive-replay", action="append", default=[])
  parser.add_argument("--positive-metadata", action="append", default=[])
  parser.add_argument("--negative-replay", action="append", default=[])
  parser.add_argument("--negative-metadata", action="append", default=[])
  parser.add_argument("--out", required=True)
  parser.add_argument("--metadata-out", default="")
  parser.add_argument("--summary-out", default="")
  return parser.parse_args()


def _sidecar_rows(path: Path) -> list[dict]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  rows = payload if isinstance(payload, list) else payload.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"metadata rows are not a list: {path}")
  return rows


def _metadata_paths(replays: list[str], explicit: list[str]) -> list[Path]:
  if explicit and len(explicit) != len(replays):
    raise ValueError("metadata count must match replay count")
  if explicit:
    return [Path(path) for path in explicit]
  return [
      Path(path).with_suffix(Path(path).suffix + ".metadata.json")
      for path in replays
  ]


def _group(
    meta: dict,
    kind: str,
    source_path: str,
    row_index: int,
    source_group_id: str = "",
) -> dict:
  return {
      "proposal_source": str(
          source_group_id or meta.get("row_group_id") or kind
      ),
      "source_path": str(source_path),
      "source_row_index": int(row_index),
      "match_index": int(meta.get("match_index", -1)),
      "round_index": int(meta.get("round_index", -1)),
  }


def calibration_metadata_row(
    meta: dict,
    replay_action: int,
    kind: str,
    source_path: str,
    row_index: int,
    source_group_id: str = "",
) -> dict:
  """Builds the deployable pair label consumed by calibration_transfer_audit."""
  decision = dict(meta.get("decision") or {})
  policy_action = int(decision["policy_action"])
  if kind == "positive":
    label_action = int(meta["best_action"])
    row_class = "POSITIVE"
    audit_improvement = float(
        meta["audit_best_vs_base_any_paradox_improvement"]
    )
    false_positive = False
  elif kind == "negative":
    label_action = int(meta["proposed_best_action"])
    row_class = "NEGATIVE"
    audit_improvement = float(
        meta["audit_best_vs_base_any_paradox_improvement"]
    )
    false_positive = True
  else:
    raise ValueError(f"unsupported calibration row kind: {kind}")
  if label_action == policy_action:
    raise ValueError("calibration candidate must differ from the source action")
  expected_target = label_action if kind == "positive" else policy_action
  if int(replay_action) != int(expected_target):
    raise ValueError(
        f"replay target {replay_action} did not match {kind} target "
        f"{expected_target}"
    )
  if not bool(meta.get("information_safe_rollout_teacher")):
    raise ValueError("calibration row did not use an information-safe teacher")
  if not bool(meta.get("independent_belief_validation")):
    raise ValueError("calibration row lacks independent belief validation")
  return {
      "group": _group(
          meta, kind, source_path, row_index, source_group_id
      ),
      "policy_best_action": int(policy_action),
      "label_best_action": int(label_action),
      "deployment_target_action": int(replay_action),
      "calibration_class": row_class,
      "label_margin_vs_policy": 0.0,
      "audit_margin_vs_policy": float(audit_improvement),
      "audit_paradox_delta_vs_policy": float(-audit_improvement),
      "false_positive_for_positive_gate": bool(false_positive),
      "legal_count": int(decision.get("legal_count", 0)),
      "candidate_count": max(
          1, int(len(meta.get("candidate_actions") or [])) - 1
      ),
      "phase": str(decision.get("phase", "")),
      "trick_number": int(decision.get("trick_number", -1)),
      "information_safe_rollout_teacher": True,
      "independent_belief_validation": True,
      "source_kind": str(kind),
      "source_metadata": meta,
  }


def _append_inputs(
    replay_paths: list[str],
    metadata_paths: list[Path],
    kind: str,
    output_rows: list,
    output_metadata: list[dict],
    output_groups: list[str],
) -> list[dict]:
  sources = []
  for replay_spec, metadata_path in zip(replay_paths, metadata_paths):
    replay_path = Path(replay_spec)
    rows, replay_metadata = load_replay(replay_path, return_metadata=True)
    sidecar = _sidecar_rows(metadata_path)
    if len(rows) != len(sidecar):
      raise ValueError(
          f"row count mismatch for {replay_path}: {len(rows)} != {len(sidecar)}"
      )
    groups = list((replay_metadata or {}).get("row_group_ids") or [])
    if groups and len(groups) != len(rows):
      raise ValueError(f"row-group count mismatch for {replay_path}")
    for index, (row, meta) in enumerate(zip(rows, sidecar)):
      converted = calibration_metadata_row(
          meta,
          int(row[5]),
          kind,
          str(replay_path),
          index,
          str(groups[index]) if groups else "",
      )
      output_rows.append(row)
      output_metadata.append(converted)
      output_groups.append(
          str(groups[index]) if groups else f"{replay_path.name}:row{index:08d}"
      )
    sources.append({
        "kind": str(kind),
        "replay": str(replay_path),
        "metadata": str(metadata_path),
        "rows": int(len(rows)),
    })
  return sources


def main() -> None:
  args = parse_args()
  if not args.positive_replay or not args.negative_replay:
    raise ValueError("at least one positive and one negative replay are required")
  positive_metadata = _metadata_paths(
      args.positive_replay, args.positive_metadata
  )
  negative_metadata = _metadata_paths(
      args.negative_replay, args.negative_metadata
  )
  rows = []
  metadata_rows = []
  group_ids = []
  sources = _append_inputs(
      args.positive_replay,
      positive_metadata,
      "positive",
      rows,
      metadata_rows,
      group_ids,
  )
  sources.extend(_append_inputs(
      args.negative_replay,
      negative_metadata,
      "negative",
      rows,
      metadata_rows,
      group_ids,
  ))
  summary = {
      "kind": "best_of_rollouts_improve_graft_calibration",
      "rows": int(len(rows)),
      "positive_rows": int(sum(
          row["calibration_class"] == "POSITIVE" for row in metadata_rows
      )),
      "negative_rows": int(sum(
          row["calibration_class"] == "NEGATIVE" for row in metadata_rows
      )),
      "source_groups": int(len(set(group_ids))),
      "sources": sources,
      "row_group_ids": group_ids,
      "row_group_id_kind": "independent_source_shard",
      "row_group_count": int(len(set(group_ids))),
  }
  out = Path(args.out)
  metadata_out = (
      Path(args.metadata_out)
      if str(args.metadata_out).strip()
      else out.with_suffix(out.suffix + ".metadata.json")
  )
  summary_out = (
      Path(args.summary_out)
      if str(args.summary_out).strip()
      else out.with_suffix(out.suffix + ".summary.json")
  )
  save_replay(rows, out, metadata=summary)
  metadata_out.parent.mkdir(parents=True, exist_ok=True)
  metadata_out.write_text(
      json.dumps({**summary, "rows": metadata_rows}, indent=2, sort_keys=True)
      + "\n",
      encoding="utf-8",
  )
  summary_out.parent.mkdir(parents=True, exist_ok=True)
  summary_out.write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
