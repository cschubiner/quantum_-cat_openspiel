#!/usr/bin/env python3
"""Repair legacy improve-graft metadata to label the actual live proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_cat_improve_graft_miner import proposal_calibration_fields


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--audit-json", required=True)
  parser.add_argument("--metadata", required=True)
  parser.add_argument("--out", required=True)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
  payload = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
  audit_rows = list(audit.get("rows") or [])
  metadata_rows = list(payload.get("rows") or [])
  if len(audit_rows) != len(metadata_rows):
    raise ValueError(
        f"audit/metadata row mismatch: {len(audit_rows)} != "
        f"{len(metadata_rows)}"
    )
  seed = int(payload["seed"])
  base_mode = str(payload.get("improve_base_mode", "policy"))
  shadow = bool(audit.get("shadow_run119_actions", False))
  proposal_source_id = (
      f"improve_graft_{base_mode}_{'shadow_' if shadow else ''}seed{seed}"
  )
  for audit_row, metadata_row in zip(audit_rows, metadata_rows):
    proposed = int(audit_row["primary_proposal_action"])
    proposal_class = str(audit_row["primary_proposal_class"])
    target_class = str(
        metadata_row.get(
            "target_calibration_class",
            metadata_row.get("calibration_class", "UNAUDITED"),
        )
    )
    metadata_row.update(proposal_calibration_fields(
        proposed,
        audit_row["primary_proposal_action_string"],
        proposal_class,
        audit_row.get("audit_margin_vs_policy"),
        audit_row.get("audit_paradox_delta_vs_policy"),
    ))
    metadata_row["target_calibration_class"] = target_class
    metadata_row["primary_proposal_class"] = proposal_class
    group = dict(metadata_row.get("group") or {})
    group["proposal_source"] = proposal_source_id
    metadata_row["group"] = group
  payload.update({
      "proposal_source_id": proposal_source_id,
      "metadata_contract": "live_proposal_pair_v2",
      "rows": metadata_rows,
  })
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({
      "out": str(out),
      "rows": int(len(metadata_rows)),
      "proposal_source_id": proposal_source_id,
      "positive_rows": int(sum(
          row["calibration_class"] == "POSITIVE" for row in metadata_rows
      )),
      "false_positive_rows": int(sum(
          row["false_positive_for_positive_gate"] for row in metadata_rows
      )),
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
