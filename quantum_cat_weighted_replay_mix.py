#!/usr/bin/env python3
"""Build a deterministic row-weighted replay mix from existing replay files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def parse_source(spec: str):
  """Parses name=path:rows, path:rows, or name=path."""
  name = ""
  source = spec
  if "=" in source:
    name, source = source.split("=", 1)
    name = name.strip()
  rows = None
  if ":" in source:
    maybe_path, maybe_rows = source.rsplit(":", 1)
    try:
      rows = int(maybe_rows)
      source = maybe_path
    except ValueError:
      rows = None
  path = source.strip()
  if not name:
    name = Path(path).stem
  if rows is not None and rows < 0:
    raise ValueError(f"row count must be non-negative in {spec!r}")
  return name, path, rows


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input",
      action="append",
      required=True,
      help=(
          "Replay source as name=path:rows. If rows is omitted, use every row. "
          "If rows exceeds available rows, sampling uses replacement."
      ),
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument(
      "--sidecar-metadata-out",
      default="",
      help=(
          "Optional JSON path for merged sidecar metadata rows. When set, "
          "each input's PATH + '.metadata.json' sidecar is sampled in the same "
          "order as replay rows and written with source provenance. This is "
          "needed by trainers that use metadata_policy_best."
      ),
  )
  parser.add_argument("--shuffle-seed", type=int, default=0)
  parser.add_argument(
      "--include-source-metadata",
      action="store_true",
      help=(
          "Embed full source replay metadata in the mixed replay metadata. "
          "Default is compact provenance only, because source metadata can "
          "contain large row_stats or nested replay metadata."
      ),
  )
  return parser.parse_args()


def compact_metadata(metadata):
  """Returns stable, small provenance for a source replay metadata object."""
  if not isinstance(metadata, dict):
    return {"type": type(metadata).__name__}
  excluded = {
      "row_stats",
      "rows",
      "samples",
      "anchor_samples",
      "source_replay_metadata",
      "metadata",
      "sources",
  }
  compact = {}
  for key, value in metadata.items():
    if key in excluded:
      continue
    if isinstance(value, (str, int, float, bool)) or value is None:
      compact[key] = value
    elif isinstance(value, list):
      compact[f"{key}_len"] = len(value)
    elif isinstance(value, dict):
      compact[f"{key}_keys"] = sorted(str(item) for item in value.keys())[:50]
      compact[f"{key}_key_count"] = len(value)
    else:
      compact[f"{key}_type"] = type(value).__name__
  encoded = json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
  compact["metadata_sha256"] = hashlib.sha256(encoded).hexdigest()
  compact["metadata_bytes"] = len(encoded)
  return compact


def main():
  args = parse_args()
  rng = np.random.default_rng(args.shuffle_seed)
  combined = []
  combined_group_ids = []
  combined_sidecar_rows = []
  sources = []

  for spec in args.input:
    name, path, requested_rows = parse_source(spec)
    replay, metadata = load_replay(path, return_metadata=True)
    available_rows = len(replay)
    if requested_rows is None:
      requested_rows = available_rows
    replace = requested_rows > available_rows
    if requested_rows == 0:
      indices = np.array([], dtype=np.int64)
    else:
      indices = rng.choice(
          available_rows,
          size=requested_rows,
          replace=replace,
      )
    selected = [replay[int(idx)] for idx in indices]
    combined.extend(selected)
    sidecar_rows = []
    sidecar_path = Path(path).with_suffix(Path(path).suffix + ".metadata.json")
    if args.sidecar_metadata_out and sidecar_path.exists():
      sidecar = json.loads(sidecar_path.read_text())
      raw_rows = sidecar if isinstance(sidecar, list) else sidecar.get("rows", [])
      if not isinstance(raw_rows, list):
        raise ValueError(f"sidecar metadata rows not found in {sidecar_path}")
      if len(raw_rows) != available_rows:
        raise ValueError(
            f"sidecar row count mismatch for {sidecar_path}: "
            f"{len(raw_rows)} vs {available_rows}"
        )
      for idx in indices:
        source_row = dict(raw_rows[int(idx)])
        group = source_row.get("group")
        if isinstance(group, dict):
          group = dict(group)
          group["source_name"] = name
          group["source_path"] = path
          group["source_row_index"] = int(idx)
          source_row["group"] = group
        else:
          source_row["row_group_id"] = (
              f"{name}:{source_row.get('row_group_id', f'row{int(idx):08d}')}"
          )
        source_row["source_name"] = name
        source_row["source_path"] = path
        source_row["source_row_index"] = int(idx)
        sidecar_rows.append(source_row)
    elif args.sidecar_metadata_out:
      raise FileNotFoundError(f"sidecar metadata not found: {sidecar_path}")
    combined_sidecar_rows.extend(sidecar_rows)
    source_group_ids = []
    if isinstance(metadata, dict):
      source_group_ids = list(metadata.get("row_group_ids") or [])
    if source_group_ids and len(source_group_ids) == available_rows:
      combined_group_ids.extend(
          f"{name}:{source_group_ids[int(idx)]}" for idx in indices
      )
    else:
      combined_group_ids.extend(
          f"{name}:row{int(idx):08d}" for idx in indices
      )
    source_row = {
        "name": name,
        "path": path,
        "available_rows": available_rows,
        "requested_rows": requested_rows,
        "selected_rows": len(selected),
        "unique_selected_rows": int(len(set(int(idx) for idx in indices))),
        "sampled_with_replacement": bool(replace),
        "metadata": (
            metadata if bool(args.include_source_metadata)
            else compact_metadata(metadata)
        ),
    }
    sources.append(source_row)

  order = np.arange(len(combined))
  if int(args.shuffle_seed) != 0:
    rng.shuffle(order)
    combined = [combined[int(idx)] for idx in order]
    combined_group_ids = [combined_group_ids[int(idx)] for idx in order]
    if combined_sidecar_rows:
      combined_sidecar_rows = [combined_sidecar_rows[int(idx)] for idx in order]

  total_rows = len(combined)
  summary = {
      "kind": "weighted_replay_mix",
      "out": args.out,
      "shuffle_seed": int(args.shuffle_seed),
      "rows": total_rows,
      "row_group_ids": combined_group_ids,
      "row_group_id_kind": "source_match_round",
      "row_group_count": int(len(set(combined_group_ids))),
      "sources": sources,
      "source_row_fractions": {
          source["name"]: (
              float(source["selected_rows"] / total_rows)
              if total_rows else 0.0
          )
          for source in sources
      },
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_replay(combined, out_path, metadata=summary)
  text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
  if args.summary_out:
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text)
  if args.sidecar_metadata_out:
    if len(combined_sidecar_rows) != len(combined):
      raise ValueError(
          "merged sidecar row count did not match replay rows: "
          f"{len(combined_sidecar_rows)} vs {len(combined)}"
      )
    sidecar_path = Path(args.sidecar_metadata_out)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps({
        **summary,
        "rows": combined_sidecar_rows,
    }, indent=2, sort_keys=True) + "\n")
  print(text, end="")


if __name__ == "__main__":
  main()
