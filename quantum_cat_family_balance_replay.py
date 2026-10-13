#!/usr/bin/env python3
"""Build an auditable family-balanced replay by repeating source replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def parse_source(spec):
  repeat = 1
  source = spec
  if ":" in source:
    maybe_source, maybe_repeat = source.rsplit(":", 1)
    try:
      repeat = int(maybe_repeat)
      source = maybe_source
    except ValueError:
      repeat = 1
      source = spec
  name = Path(source).stem
  path = source
  if "=" in source:
    name, path = source.split("=", 1)
  if repeat <= 0:
    raise ValueError(f"repeat must be positive in {spec!r}")
  return name.strip(), path.strip(), repeat


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--shuffle-seed", type=int, default=0)
  return parser.parse_args()


def main():
  args = parse_args()
  rng = np.random.default_rng(args.shuffle_seed)
  combined = []
  sources = []

  for spec in args.input:
    name, path, repeat = parse_source(spec)
    replay, metadata = load_replay(path, return_metadata=True)
    source_rows = list(replay)
    for _ in range(repeat):
      combined.extend(source_rows)
    sources.append({
        "name": name,
        "path": path,
        "rows": len(source_rows),
        "repeat": repeat,
        "weighted_rows": len(source_rows) * repeat,
        "metadata": metadata,
    })

  order = np.arange(len(combined))
  if int(args.shuffle_seed) != 0:
    rng.shuffle(order)
    combined = [combined[int(idx)] for idx in order]

  summary = {
      "out": args.out,
      "shuffle_seed": int(args.shuffle_seed),
      "sources": sources,
      "rows": len(combined),
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_replay(combined, out_path, metadata=summary)
  text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
  if args.summary_out:
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
