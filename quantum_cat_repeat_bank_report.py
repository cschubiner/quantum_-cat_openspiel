#!/usr/bin/env python3
"""Summarize repeat-validated Quantum Cat replay metadata banks."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "metadata",
      nargs="+",
      help="Metadata JSON paths or glob patterns ending in .npz.metadata.json.",
  )
  parser.add_argument(
      "--json-out",
      default="",
      help="Optional path for the machine-readable report.",
  )
  return parser.parse_args()


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata_rows(data):
  if isinstance(data, list):
    return data
  return data.get("metadata_rows") or data.get("rows") or data.get("metadata") or []


def _expand_patterns(patterns: list[str]) -> list[Path]:
  paths: list[Path] = []
  for pattern in patterns:
    matches = glob.glob(pattern)
    if matches:
      paths.extend(Path(match) for match in matches)
    else:
      paths.append(Path(pattern))
  unique = sorted({path for path in paths})
  missing = [str(path) for path in unique if not path.exists()]
  if missing:
    raise FileNotFoundError(f"metadata paths not found: {missing}")
  return unique


def _npz_path_for_metadata(path: Path, data=None) -> Path:
  if isinstance(data, dict):
    replay = str(data.get("replay") or "").strip()
    if replay:
      return Path(replay)
  text = str(path)
  if text.endswith(".npz.metadata.json"):
    return Path(text[: -len(".metadata.json")])
  if text.endswith(".partial.metadata.json"):
    return Path(text[: -len(".metadata.json")] + ".npz")
  return path.with_suffix("")


def build_report(paths: list[Path]) -> dict:
  totals = Counter()
  led_colors = Counter()
  label_best_colors = Counter()
  selected_colors = Counter()
  candidate_count_hist = Counter()
  candidate_colors = Counter()
  rows_with_candidate_color = Counter()
  passed_candidate_colors = Counter()
  rows_with_passed_candidate_color = Counter()
  file_reports = []

  for source_index, path in enumerate(paths):
    data = json.loads(path.read_text())
    rows = _metadata_rows(data)
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    replay_path = _npz_path_for_metadata(path, data)
    file_report = {
        "source_index": int(source_index),
        "metadata": str(path),
        "partial": bool(data.get("partial", False)) if isinstance(data, dict) else False,
        "rows": int(len(rows)),
        "summary": summary,
        "metadata_sha256": _sha256(path),
    }
    if replay_path.exists():
      file_report["replay"] = str(replay_path)
      file_report["replay_sha256"] = _sha256(replay_path)
    file_reports.append(file_report)

    totals["rows"] += len(rows)
    totals["summary_repeat_validation_attempted"] += int(
        summary.get("repeat_validation_attempted", 0)
    )
    totals["summary_repeat_validation_passed"] += int(
        summary.get("repeat_validation_passed", 0)
    )
    totals["summary_action_targets_written"] += int(
        summary.get("repeat_validation_action_targets_written", 0)
    )
    for row in rows:
      led = row.get("led_contrast") or {}
      led_colors[str(led.get("led_color"))] += 1
      label_best_colors[str(led.get("label_best_color"))] += 1
      selected_colors[
          str(row.get("repeat_validation_selected_candidate_color",
                      led.get("label_best_color")))
      ] += 1
      if bool(row.get("repeat_validation_action_targets_written", False)):
        totals["rows_with_action_targets_written"] += 1
      if bool(row.get("repeat_validation_exclusive_action_targets", False)):
        totals["rows_with_exclusive_action_targets"] += 1
      repeat_validation = row.get("repeat_validation") or {}
      candidates = repeat_validation.get("candidates") or []
      candidate_count_hist[str(len(candidates))] += 1
      seen_colors = set()
      passed_seen_colors = set()
      row_has_passed_red_switch = False
      row_has_passed_follow_led = False
      row_has_red_candidate = False
      row_has_follow_candidate = False
      for candidate in candidates:
        color = str(candidate.get("candidate_color"))
        candidate_colors[color] += 1
        seen_colors.add(color)
        if bool(candidate.get("candidate_is_red_switch", False)):
          totals["candidate_red_switch"] += 1
          row_has_red_candidate = True
        if bool(candidate.get("candidate_follows_led", False)):
          totals["candidate_follows_led"] += 1
          row_has_follow_candidate = True
        if bool(candidate.get("passed", False)):
          passed_candidate_colors[color] += 1
          passed_seen_colors.add(color)
          if bool(candidate.get("candidate_is_red_switch", False)):
            totals["passed_red_switch"] += 1
            row_has_passed_red_switch = True
          if bool(candidate.get("candidate_follows_led", False)):
            totals["passed_follows_led"] += 1
            row_has_passed_follow_led = True
      for color in seen_colors:
        rows_with_candidate_color[color] += 1
      for color in passed_seen_colors:
        rows_with_passed_candidate_color[color] += 1
      totals["rows_with_red_candidate"] += int(row_has_red_candidate)
      totals["rows_with_follow_candidate"] += int(row_has_follow_candidate)
      totals["rows_with_passed_red_switch"] += int(row_has_passed_red_switch)
      totals["rows_with_passed_follow_led"] += int(row_has_passed_follow_led)

  rows = int(totals["rows"])
  return {
      "files": file_reports,
      "totals": dict(totals),
      "rates": {
          "selected_red_rate": (
              float(selected_colors.get("R", 0) / rows) if rows else 0.0
          ),
          "rows_with_passed_red_switch_rate": (
              float(totals["rows_with_passed_red_switch"] / rows) if rows else 0.0
          ),
          "rows_with_passed_follow_led_rate": (
              float(totals["rows_with_passed_follow_led"] / rows) if rows else 0.0
          ),
      },
      "led_color_counts": dict(led_colors),
      "label_best_color_counts": dict(label_best_colors),
      "selected_color_counts": dict(selected_colors),
      "candidate_count_hist": dict(candidate_count_hist),
      "candidate_color_counts": dict(candidate_colors),
      "rows_with_candidate_color_counts": dict(rows_with_candidate_color),
      "passed_candidate_color_counts": dict(passed_candidate_colors),
      "rows_with_passed_candidate_color_counts": dict(rows_with_passed_candidate_color),
  }


def main():
  args = parse_args()
  report = build_report(_expand_patterns(args.metadata))
  print(json.dumps(report, indent=2, sort_keys=True))
  if args.json_out:
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
  main()
