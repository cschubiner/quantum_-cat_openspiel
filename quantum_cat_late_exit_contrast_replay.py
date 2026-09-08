#!/usr/bin/env python3
"""Build binary best-vs-selected late-exit contrast replay rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def _load_summary_rows(path: str):
  data = json.loads(Path(path).read_text())
  rows = data.get("rows")
  if not isinstance(rows, list):
    raise ValueError(f"{path} does not contain a JSON list at key 'rows'")
  return data, rows


def _top_margin(row) -> float:
  values = np.asarray(row[9], dtype=np.float32)
  mask = np.asarray(row[10], dtype=np.float32) > 0.0
  if int(np.sum(mask)) < 2:
    return 0.0
  labeled = values[mask]
  sorted_values = np.sort(labeled)
  return float(sorted_values[-1] - sorted_values[-2])


def _action_color(summary_row, key: str) -> str:
  text = str(summary_row.get(key, ""))
  marker = "color="
  if marker not in text:
    return ""
  return text.split(marker, 1)[1].split(")", 1)[0].strip()


def build_contrast_replay(
    replay,
    summary_rows,
    *,
    min_top_margin: float,
    best_target: float,
    selected_target: float,
):
  if len(replay) != len(summary_rows):
    raise ValueError(
        f"replay rows ({len(replay)}) != summary rows ({len(summary_rows)})"
    )

  selected = []
  skipped_same = 0
  skipped_margin = 0
  led_counts = {}
  trick_counts = {}
  best_color_counts = {}
  selected_color_counts = {}
  margin_values = []

  for idx, (row, summary_row) in enumerate(zip(replay, summary_rows)):
    best_action = int(summary_row["best_action"])
    selected_action = int(summary_row["selected_action"])
    if best_action == selected_action:
      skipped_same += 1
      continue
    margin = _top_margin(row)
    if margin < min_top_margin:
      skipped_margin += 1
      continue

    mask = np.asarray(row[1], dtype=np.float32)
    if best_action >= len(mask) or selected_action >= len(mask):
      raise ValueError(f"row {idx} action id out of range")
    if mask[best_action] <= 0.0 or mask[selected_action] <= 0.0:
      raise ValueError(f"row {idx} contrast action is not legal")

    policy = np.zeros_like(mask, dtype=np.float32)
    policy[best_action] = 1.0
    action_value_targets = np.zeros_like(mask, dtype=np.float32)
    action_value_mask = np.zeros_like(mask, dtype=np.float32)
    action_value_targets[best_action] = float(best_target)
    action_value_targets[selected_action] = float(selected_target)
    action_value_mask[best_action] = 1.0
    action_value_mask[selected_action] = 1.0

    action_targets = (
        np.asarray(row[7], dtype=np.float32).copy()
        if len(row) > 7 and row[7] is not None
        else np.zeros_like(mask, dtype=np.float32)
    )
    action_mask = (
        np.asarray(row[8], dtype=np.float32).copy()
        if len(row) > 8 and row[8] is not None
        else np.zeros_like(mask, dtype=np.float32)
    )

    items = list(row)
    items[2] = policy
    items[5] = selected_action
    items[7] = action_targets
    items[8] = action_mask
    items[9] = action_value_targets
    items[10] = action_value_mask
    selected.append(tuple(items))

    led = str(summary_row.get("led_color"))
    trick = str(summary_row.get("trick_number"))
    best_color = _action_color(summary_row, "best_action_string")
    selected_color = _action_color(summary_row, "selected_action_string")
    led_counts[led] = led_counts.get(led, 0) + 1
    trick_counts[trick] = trick_counts.get(trick, 0) + 1
    best_color_counts[best_color] = best_color_counts.get(best_color, 0) + 1
    selected_color_counts[selected_color] = (
        selected_color_counts.get(selected_color, 0) + 1
    )
    margin_values.append(margin)

  margins = np.asarray(margin_values, dtype=np.float32)
  summary = {
      "kind": "late_exit_binary_best_vs_selected_contrast_replay",
      "source_rows": len(replay),
      "rows": len(selected),
      "skipped_same_best_selected": skipped_same,
      "skipped_below_min_top_margin": skipped_margin,
      "min_top_margin": float(min_top_margin),
      "best_target": float(best_target),
      "selected_target": float(selected_target),
      "labeled_actions_per_row": 2,
      "led_color_counts": dict(sorted(led_counts.items())),
      "trick_counts": dict(sorted(trick_counts.items())),
      "best_color_counts": dict(sorted(best_color_counts.items())),
      "selected_color_counts": dict(sorted(selected_color_counts.items())),
      "top_margin": {
          "min": float(np.min(margins)) if len(margins) else None,
          "mean": float(np.mean(margins)) if len(margins) else None,
          "max": float(np.max(margins)) if len(margins) else None,
      },
  }
  return selected, summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--summary", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", required=True)
  parser.add_argument("--min-top-margin", type=float, default=0.0)
  parser.add_argument("--best-target", type=float, default=1.0)
  parser.add_argument("--selected-target", type=float, default=-1.0)
  args = parser.parse_args()

  replay, replay_meta = load_replay(args.input, return_metadata=True)
  source_summary, summary_rows = _load_summary_rows(args.summary)
  contrast_replay, summary = build_contrast_replay(
      replay,
      summary_rows,
      min_top_margin=float(args.min_top_margin),
      best_target=float(args.best_target),
      selected_target=float(args.selected_target),
  )
  if not contrast_replay:
    raise ValueError("contrast replay selected zero rows")
  summary.update({
      "input": args.input,
      "summary": args.summary,
      "out": args.out,
      "source_replay_metadata": replay_meta,
      "source_summary_header": {
          key: value
          for key, value in source_summary.items()
          if key != "rows"
      },
  })
  if isinstance(summary.get("source_replay_metadata"), dict):
    summary["source_replay_metadata"] = {
        key: value
        for key, value in summary["source_replay_metadata"].items()
        if key != "rows"
    }
  Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.summary_out).write_text(json.dumps(summary, indent=2, sort_keys=True))
  save_replay(contrast_replay, args.out, metadata=summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
