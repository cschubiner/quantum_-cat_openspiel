#!/usr/bin/env python3
"""Report repeat-validation attempt yield under alternative thresholds."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path


DEFAULT_SPECS = [
    "strict:1.0:1.0:0.0:0.5:0.01",
    "three_of_four_clean:0.75:0.75:0.0:0.5:0.0",
    "three_of_four_allow_one_candidate_paradox:0.75:0.75:0.25:0.5:0.0",
    "weak_map:0.50:0.75:0.25:0.25:0.0",
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "metadata",
      nargs="+",
      help="Metadata JSON paths or glob patterns.",
  )
  parser.add_argument(
      "--spec",
      action="append",
      default=[],
      help=(
          "Threshold spec name:min_win:min_nonworse:max_candidate_any:"
          "min_base_any:min_value_delta. Defaults to a small diagnostic grid."
      ),
  )
  parser.add_argument("--json-out", default="")
  return parser.parse_args()


def _expand(patterns: list[str]) -> list[Path]:
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


def _parse_spec(text: str) -> dict:
  parts = text.split(":")
  if len(parts) != 6:
    raise ValueError(
        "--spec must be name:min_win:min_nonworse:max_candidate_any:"
        f"min_base_any:min_value_delta, got {text!r}"
    )
  name, min_win, min_nonworse, max_candidate_any, min_base_any, min_value = parts
  return {
      "name": name,
      "min_win_rate": float(min_win),
      "min_nonworse_rate": float(min_nonworse),
      "max_candidate_any_paradox_rate": float(max_candidate_any),
      "min_base_any_paradox_rate": float(min_base_any),
      "min_value_delta": float(min_value),
  }


def _attempt_rows(data) -> list[dict]:
  if isinstance(data, list):
    return data
  rows = data.get("repeat_validation_attempt_rows") or []
  if rows:
    return rows
  # Backward-compatible fallback: accepted replay metadata rows still contain
  # enough repeat-validation detail to count as admitted attempts.
  fallback = []
  for row in data.get("rows", []) or []:
    repeat = row.get("repeat_validation")
    if repeat:
      fallback.append({
          "admission": "accepted_existing_row",
          "phase": row.get("phase"),
          "player": row.get("player"),
          "trick_number": row.get("trick_number"),
          "legal_count": row.get("legal_count"),
          "led_contrast": row.get("led_contrast") or {},
          "repeat_validation": repeat,
      })
  return fallback


def _candidate_passes(candidate: dict, spec: dict) -> bool:
  return (
      float(candidate.get("win_rate", 0.0)) >= spec["min_win_rate"]
      and float(candidate.get("nonworse_rate", 0.0)) >= spec["min_nonworse_rate"]
      and float(candidate.get("candidate_any_paradox_rate", 1.0))
      <= spec["max_candidate_any_paradox_rate"]
      and float(candidate.get("base_any_paradox_rate", 0.0))
      >= spec["min_base_any_paradox_rate"]
      and float(candidate.get("mean_value_delta", -1e9)) >= spec["min_value_delta"]
  )


def _candidate_sort_key(candidate: dict):
  return (
      -float(candidate.get("candidate_any_paradox_rate", 1.0)),
      float(candidate.get("win_rate", 0.0)),
      float(candidate.get("nonworse_rate", 0.0)),
      float(candidate.get("mean_value_delta", -1e9)),
      -float(candidate.get("mean_risk_delta", 1e9)),
      float(candidate.get("mean_acting_score_delta", -1e9)),
      float(candidate.get("candidate_policy_prob", 0.0)),
  )


def _summarize_for_spec(attempts: list[dict], spec: dict) -> dict:
  totals = Counter()
  admissions = Counter()
  led_colors = Counter()
  selected_colors = Counter()
  qualifying_colors = Counter()
  selected_is_red_switch = 0
  selected_follows_led = 0
  best_value_deltas = []

  for attempt in attempts:
    totals["attempts"] += 1
    admissions[str(attempt.get("admission", "unknown"))] += 1
    led = attempt.get("led_contrast") or {}
    led_colors[str(led.get("led_color"))] += 1
    repeat = attempt.get("repeat_validation") or {}
    candidates = repeat.get("candidates") or []
    totals["candidate_records"] += len(candidates)
    qualifying = [c for c in candidates if _candidate_passes(c, spec)]
    if not qualifying:
      continue
    totals["attempts_with_qualifying_candidate"] += 1
    if any(bool(c.get("candidate_is_red_switch", False)) for c in qualifying):
      totals["attempts_with_qualifying_red_switch"] += 1
    if any(bool(c.get("candidate_follows_led", False)) for c in qualifying):
      totals["attempts_with_qualifying_follow_led"] += 1
    for candidate in qualifying:
      qualifying_colors[str(candidate.get("candidate_color"))] += 1
    selected = max(qualifying, key=_candidate_sort_key)
    selected_colors[str(selected.get("candidate_color"))] += 1
    selected_is_red_switch += int(bool(selected.get("candidate_is_red_switch", False)))
    selected_follows_led += int(bool(selected.get("candidate_follows_led", False)))
    best_value_deltas.append(float(selected.get("mean_value_delta", 0.0)))

  attempts = int(totals["attempts"])
  qualifying_attempts = int(totals["attempts_with_qualifying_candidate"])
  return {
      "spec": spec,
      "totals": dict(totals),
      "rates": {
          "qualifying_attempt_rate": (
              qualifying_attempts / attempts if attempts else 0.0
          ),
          "qualifying_red_switch_attempt_rate": (
              totals["attempts_with_qualifying_red_switch"] / attempts
              if attempts else 0.0
          ),
          "selected_red_rate": (
              selected_colors.get("R", 0) / qualifying_attempts
              if qualifying_attempts else 0.0
          ),
          "selected_red_switch_rate": (
              selected_is_red_switch / qualifying_attempts
              if qualifying_attempts else 0.0
          ),
          "selected_follow_led_rate": (
              selected_follows_led / qualifying_attempts
              if qualifying_attempts else 0.0
          ),
      },
      "admission_counts": dict(admissions),
      "led_color_counts": dict(led_colors),
      "qualifying_candidate_color_counts": dict(qualifying_colors),
      "selected_color_counts": dict(selected_colors),
      "selected_mean_value_delta": (
          sum(best_value_deltas) / len(best_value_deltas)
          if best_value_deltas else None
      ),
  }


def build_report(paths: list[Path], specs: list[dict]) -> dict:
  attempts = []
  files = []
  for path in paths:
    data = json.loads(path.read_text())
    rows = _attempt_rows(data)
    attempts.extend(rows)
    files.append({"path": str(path), "attempt_rows": len(rows)})
  return {
      "files": files,
      "attempt_rows": len(attempts),
      "reports": [
          _summarize_for_spec(attempts, spec)
          for spec in specs
      ],
  }


def main():
  args = parse_args()
  specs = [_parse_spec(text) for text in (args.spec or DEFAULT_SPECS)]
  report = build_report(_expand(args.metadata), specs)
  text = json.dumps(report, indent=2, sort_keys=True)
  print(text)
  if args.json_out:
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
  main()
