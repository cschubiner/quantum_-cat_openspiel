#!/usr/bin/env python3
"""Build the early/pre-collapse red-rescue bank in small restartable batches."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from quantum_cat_repeat_bank_report import build_report as build_bank_report
from quantum_cat_repeat_validation_attempt_report import (
    DEFAULT_SPECS,
    _parse_spec,
    build_report as build_attempt_report,
)


def _expand_patterns(patterns: list[str]) -> list[Path]:
  paths: list[Path] = []
  for pattern in patterns:
    matches = glob.glob(pattern)
    if matches:
      paths.extend(Path(match) for match in matches)
    else:
      candidate = Path(pattern)
      if candidate.exists():
        paths.append(candidate)
  return sorted({path for path in paths if path.exists()})


def _final_for_partial(path: Path) -> Path:
  text = str(path)
  if not text.endswith(".partial.metadata.json"):
    return path
  return Path(text[: -len(".partial.metadata.json")] + ".metadata.json")


def _selected_metadata_under(root: Path) -> list[Path]:
  final_paths = set(root.glob("**/*.npz.metadata.json"))
  selected = set(final_paths)
  for partial_path in root.glob("**/*.partial.metadata.json"):
    if _final_for_partial(partial_path) not in final_paths:
      selected.add(partial_path)
  return sorted(selected)


def _report_rows(report: dict[str, Any]) -> int:
  return int((report.get("totals") or {}).get("rows", 0) or 0)


def _write_json(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _run_collector(args: argparse.Namespace, batch_index: int) -> dict[str, Any]:
  batch_seed_start = int(args.seed_start) + batch_index * int(args.batch_size)
  batch_source_start = int(args.source_start) + batch_index * int(args.batch_size)
  batch_out = Path(args.out_root) / f"batch_{batch_index:03d}"
  batch_prefix = f"{args.prefix}_b{batch_index:03d}"
  command = [
      sys.executable,
      "quantum_cat_exclusive_red_passed_collector.py",
      "--out-root",
      str(batch_out),
      "--prefix",
      batch_prefix,
      "--seed-start",
      str(batch_seed_start),
      "--seed-count",
      str(args.batch_size),
      "--source-start",
      str(batch_source_start),
      "--max-decisions",
      str(args.max_decisions),
      "--max-decisions-per-round",
      str(args.max_decisions_per_round),
      "--min-trick-number",
      str(args.min_trick_number),
      "--max-trick-number",
      str(args.max_trick_number),
      "--progress-interval",
      str(args.progress_interval),
      "--partial-save-interval-decisions",
      str(args.partial_save_interval_decisions),
      "--repeat-validation-rollouts",
      str(args.repeat_validation_rollouts),
      "--repeat-validation-max-candidates",
      str(args.repeat_validation_max_candidates),
      "--repeat-validation-prefilter-candidates",
      "--record-repeat-validation-attempts",
      "--max-workers",
      str(args.max_workers),
      "--command-timeout-sec",
      str(args.command_timeout_sec),
  ]
  if args.dry_run:
    return {
        "batch_index": batch_index,
        "batch_out": str(batch_out),
        "command": command,
        "exit_code": None,
        "status": "dry_run",
      }

  env = os.environ.copy()
  env["PYTHONPATH"] = (
      "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"
  )
  batch_out.mkdir(parents=True, exist_ok=True)
  stdout_path = batch_out / "bank_builder_collector.stdout.log"
  with stdout_path.open("w", encoding="utf-8") as stdout_log:
    stdout_log.write("$ " + shlex.join(command) + "\n")
    stdout_log.flush()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        env=env,
        stdout=stdout_log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    stdout_log.write(f"\nexit_code={completed.returncode}\n")
  status = "completed" if completed.returncode == 0 else "collector_failed"
  return {
      "batch_index": batch_index,
      "batch_out": str(batch_out),
      "command": command,
      "exit_code": int(completed.returncode),
      "status": status,
      "stdout": str(stdout_path),
  }


def _write_combined_reports(
    out_root: Path,
    existing_metadata: list[Path],
    batch_rows: list[dict[str, Any]],
) -> dict[str, Any]:
  batch_metadata = _selected_metadata_under(out_root)
  all_metadata = sorted(set(existing_metadata + batch_metadata))
  bank_report = build_bank_report(all_metadata) if all_metadata else {}
  attempt_specs = [_parse_spec(text) for text in DEFAULT_SPECS]
  attempt_report = (
      build_attempt_report(all_metadata, attempt_specs) if all_metadata else {}
  )
  _write_json(out_root / "combined_bank_report.json", bank_report)
  _write_json(out_root / "combined_attempt_threshold_report.json", attempt_report)
  summary = {
      "existing_metadata": [str(path) for path in existing_metadata],
      "batch_metadata": [str(path) for path in batch_metadata],
      "all_metadata_count": len(all_metadata),
      "combined_rows": _report_rows(bank_report),
      "combined_selected_red_rate": (bank_report.get("rates") or {}).get(
          "selected_red_rate", 0.0
      ),
      "combined_rows_with_passed_red_switch_rate": (
          bank_report.get("rates") or {}
      ).get("rows_with_passed_red_switch_rate", 0.0),
      "batches": batch_rows,
      "combined_bank_report": str(out_root / "combined_bank_report.json"),
      "combined_attempt_threshold_report": str(
          out_root / "combined_attempt_threshold_report.json"
      ),
  }
  _write_json(out_root / "bank_builder_summary.json", summary)
  return summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-root", required=True)
  parser.add_argument("--prefix", default="early_auto")
  parser.add_argument("--existing-metadata", action="append", default=[])
  parser.add_argument("--seed-start", type=int, default=20262400)
  parser.add_argument("--source-start", type=int, default=400)
  parser.add_argument("--batch-size", type=int, default=6)
  parser.add_argument("--max-batches", type=int, default=1)
  parser.add_argument("--target-total-rows", type=int, default=150)
  parser.add_argument("--max-decisions", type=int, default=2)
  parser.add_argument("--max-decisions-per-round", type=int, default=1)
  parser.add_argument("--min-trick-number", type=int, default=0)
  parser.add_argument("--max-trick-number", type=int, default=1)
  parser.add_argument("--progress-interval", type=int, default=1)
  parser.add_argument("--partial-save-interval-decisions", type=int, default=1)
  parser.add_argument("--repeat-validation-rollouts", type=int, default=4)
  parser.add_argument("--repeat-validation-max-candidates", type=int, default=6)
  parser.add_argument("--max-workers", type=int, default=2)
  parser.add_argument("--command-timeout-sec", type=float, default=600.0)
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  existing_metadata = _expand_patterns(args.existing_metadata)
  batch_rows: list[dict[str, Any]] = []
  summary = _write_combined_reports(out_root, existing_metadata, batch_rows)
  if int(summary["combined_rows"]) >= int(args.target_total_rows):
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0

  for batch_index in range(max(0, int(args.max_batches))):
    row = _run_collector(args, batch_index)
    batch_rows.append(row)
    summary = _write_combined_reports(out_root, existing_metadata, batch_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if row["status"] not in {"completed", "dry_run"}:
      return int(row.get("exit_code") or 1)
    if int(summary["combined_rows"]) >= int(args.target_total_rows):
      break
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
