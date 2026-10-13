#!/usr/bin/env python3
"""Build red-augmented repeated proposal-slate banks in restartable shards."""

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
      path = Path(pattern)
      if path.exists():
        paths.append(path)
  return sorted({path for path in paths if path.exists()})


def _final_for_partial(path: Path) -> Path:
  text = str(path)
  if text.endswith(".partial.metadata.json"):
    return Path(text[: -len(".partial.metadata.json")] + ".metadata.json")
  return path


def _metadata_rows(path: Path) -> int:
  try:
    data = json.loads(path.read_text())
  except Exception:
    return 0
  if isinstance(data, list):
    return len(data)
  rows = data.get("rows") or data.get("metadata_rows") or data.get("metadata") or []
  return len(rows) if isinstance(rows, list) else 0


def _selected_metadata_under(root: Path) -> list[Path]:
  final_paths = set(root.glob("**/*.npz.metadata.json"))
  selected = set(final_paths)
  for partial_path in root.glob("**/*.partial.metadata.json"):
    if _final_for_partial(partial_path) not in final_paths:
      selected.add(partial_path)
  return sorted(path for path in selected if _metadata_rows(path) > 0)


def _write_json(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _report_rows(report: dict[str, Any]) -> int:
  return int((report.get("totals") or {}).get("rows", 0) or 0)


def _collector_command(args: argparse.Namespace, shard_index: int, out_path: Path) -> list[str]:
  seed = int(args.seed_start) + shard_index
  return [
      sys.executable,
      "quantum_cat_live_substitution_replay.py",
      "--checkpoint",
      str(args.checkpoint),
      "--players",
      str(args.players),
      "--matches",
      str(args.matches),
      "--seed",
      str(seed),
      "--bot-mode",
      "policy",
      "--decision-phases",
      "play",
      "--max-decisions",
      str(args.max_decisions),
      "--max-decisions-per-round",
      str(args.max_decisions_per_round),
      "--max-accepted-rows",
      str(args.max_accepted_rows),
      "--dense-action-value-objective",
      "safety",
      "--label-base-action-source",
      "frozen_policy",
      "--proposal-mode",
      "action_set_transformer",
      "--require-proposal",
      "--require-proposal-outcome",
      "scored_has_positive_safe_no_candidate_any_paradox",
      "--action-set-transformer-model",
      str(args.action_set_transformer_model),
      "--action-set-transformer-base-mode",
      "policy",
      "--action-set-transformer-phases",
      "play",
      "--action-set-transformer-score-margin-threshold",
      "0.0",
      "--action-set-transformer-value-threshold",
      "0.0",
      "--action-set-transformer-positive-threshold",
      "0.5",
      "--action-set-transformer-max-policy-log-gap",
      "-1.0",
      "--action-set-transformer-max-actions",
      str(args.action_set_transformer_max_actions),
      "--action-set-transformer-top-policy",
      str(args.action_set_transformer_top_policy),
      "--action-set-transformer-include-bots",
      str(args.action_set_transformer_include_bots),
      "--action-set-transformer-feature-candidates",
      "--action-set-transformer-exclude-forced-led-lane-negatives",
      "--action-set-transformer-selection-value-bonus",
      str(args.action_set_transformer_selection_value_bonus),
      "--action-set-transformer-policy-tiebreak",
      str(args.action_set_transformer_policy_tiebreak),
      "--repeat-validate-labels",
      "--repeat-validation-candidate-mode",
      "proposal_scored_plus_red_follow_value",
      "--repeat-validation-max-candidates",
      str(args.repeat_validation_max_candidates),
      "--repeat-validation-rollouts",
      str(args.repeat_validation_rollouts),
      "--repeat-validation-min-win-rate",
      str(args.repeat_validation_min_win_rate),
      "--repeat-validation-min-nonworse-rate",
      str(args.repeat_validation_min_nonworse_rate),
      "--repeat-validation-min-value-delta",
      str(args.repeat_validation_min_value_delta),
      "--repeat-validation-max-risk-delta",
      str(args.repeat_validation_max_risk_delta),
      "--repeat-validation-max-candidate-any-paradox-rate",
      str(args.repeat_validation_max_candidate_any_paradox_rate),
      "--repeat-validation-min-base-any-paradox-rate",
      str(args.repeat_validation_min_base_any_paradox_rate),
      "--repeat-validation-require-passed-red-switch",
      "--repeat-validation-exclusive-action-targets",
      "--record-repeat-validation-attempts",
      "--record-round-history",
      "--enable-behavior-exploration",
      "--behavior-led-contrast-switch-prob",
      str(args.behavior_led_contrast_switch_prob),
      "--behavior-red-switch-prob",
      str(args.behavior_red_switch_prob),
      "--progress-interval",
      str(args.progress_interval),
      "--partial-save-interval-decisions",
      str(args.partial_save_interval_decisions),
      "--device",
      "cpu",
      "--out",
      str(out_path),
      "--summary-out",
      str(Path(str(out_path) + ".summary.json")),
  ] + (["--match-context"] if bool(args.match_context) else [])


def _run_collector(args: argparse.Namespace, shard_index: int) -> dict[str, Any]:
  out_root = Path(args.out_root)
  out_path = out_root / f"{args.prefix}_shard{shard_index:03d}.npz"
  metadata_path = Path(str(out_path) + ".metadata.json")
  partial_metadata_path = out_path.with_suffix(out_path.suffix + ".partial.metadata.json")
  stdout_path = out_root / f"{args.prefix}_shard{shard_index:03d}.stdout.log"
  if metadata_path.exists() or (
      partial_metadata_path.exists() and _metadata_rows(partial_metadata_path) > 0
  ):
    selected = metadata_path if metadata_path.exists() else partial_metadata_path
    return {
        "shard_index": int(shard_index),
        "out": str(out_path),
        "metadata": str(selected),
        "status": "skipped_existing",
        "exit_code": 0,
        "rows": _metadata_rows(selected),
    }
  command = _collector_command(args, shard_index, out_path)
  if bool(args.dry_run):
    return {
        "shard_index": int(shard_index),
        "out": str(out_path),
        "command": command,
        "status": "dry_run",
        "exit_code": None,
        "rows": 0,
    }

  env = os.environ.copy()
  env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"
  out_root.mkdir(parents=True, exist_ok=True)
  with stdout_path.open("w", encoding="utf-8") as stdout_log:
    stdout_log.write("$ " + shlex.join(command) + "\n")
    stdout_log.flush()
    try:
      completed = subprocess.run(
          command,
          cwd=Path(__file__).resolve().parent,
          env=env,
          stdout=stdout_log,
          stderr=subprocess.STDOUT,
          text=True,
          timeout=float(args.command_timeout_sec),
          check=False,
      )
      exit_code = int(completed.returncode)
      status = "completed" if exit_code == 0 else "collector_failed"
    except subprocess.TimeoutExpired:
      exit_code = 124
      status = "timeout"
      stdout_log.write(f"\ntimeout_sec={float(args.command_timeout_sec)}\n")
    stdout_log.write(f"\nexit_code={exit_code}\n")
  selected = metadata_path if metadata_path.exists() else partial_metadata_path
  return {
      "shard_index": int(shard_index),
      "out": str(out_path),
      "metadata": str(selected) if selected.exists() else "",
      "command": command,
      "stdout": str(stdout_path),
      "status": status,
      "exit_code": int(exit_code),
      "rows": _metadata_rows(selected) if selected.exists() else 0,
  }


def _write_combined_reports(
    out_root: Path,
    existing_metadata: list[Path],
    shard_rows: list[dict[str, Any]],
) -> dict[str, Any]:
  collected_metadata = _selected_metadata_under(out_root)
  all_metadata = sorted(set(existing_metadata + collected_metadata))
  bank_report = build_bank_report(all_metadata) if all_metadata else {}
  specs = [_parse_spec(text) for text in DEFAULT_SPECS]
  attempt_report = build_attempt_report(all_metadata, specs) if all_metadata else {}
  _write_json(out_root / "combined_bank_report.json", bank_report)
  _write_json(out_root / "combined_attempt_threshold_report.json", attempt_report)
  summary = {
      "existing_metadata": [str(path) for path in existing_metadata],
      "collected_metadata": [str(path) for path in collected_metadata],
      "all_metadata_count": len(all_metadata),
      "combined_rows": _report_rows(bank_report),
      "combined_selected_red_rate": (bank_report.get("rates") or {}).get(
          "selected_red_rate", 0.0
      ),
      "combined_rows_with_passed_follow_led_rate": (
          bank_report.get("rates") or {}
      ).get("rows_with_passed_follow_led_rate", 0.0),
      "combined_rows_with_passed_red_switch_rate": (
          bank_report.get("rates") or {}
      ).get("rows_with_passed_red_switch_rate", 0.0),
      "shards": shard_rows,
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
  parser.add_argument("--prefix", default="repeat_red_aug")
  parser.add_argument("--existing-metadata", action="append", default=[])
  parser.add_argument("--seed-start", type=int, default=202640000)
  parser.add_argument("--shard-start", type=int, default=0)
  parser.add_argument("--max-shards", type=int, default=4)
  parser.add_argument("--target-source-count", type=int, default=12)
  parser.add_argument("--target-total-rows", type=int, default=48)
  parser.add_argument(
      "--checkpoint",
      default=(
          "az_runs/run779_5p_low_exit_causal_20260613/"
          "action_aux_heads_low_exit_reset_s120_lr1e3_cpu/"
          "checkpoint_0000_loaded_replay_step_0040.pt"
      ),
  )
  parser.add_argument(
      "--action-set-transformer-model",
      default="az_runs/run899_balanced_table_liveness_calibrated_20260614/holdout_cal_test_model.pt",
  )
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--matches", type=int, default=100)
  parser.add_argument("--match-context", action="store_true", default=True)
  parser.add_argument("--max-decisions", type=int, default=250)
  parser.add_argument("--max-decisions-per-round", type=int, default=5)
  parser.add_argument("--max-accepted-rows", type=int, default=4)
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument("--partial-save-interval-decisions", type=int, default=10)
  parser.add_argument("--command-timeout-sec", type=float, default=240.0)
  parser.add_argument("--action-set-transformer-max-actions", type=int, default=6)
  parser.add_argument("--action-set-transformer-top-policy", type=int, default=4)
  parser.add_argument(
      "--action-set-transformer-include-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument("--action-set-transformer-selection-value-bonus", type=float, default=0.05)
  parser.add_argument("--action-set-transformer-policy-tiebreak", type=float, default=0.01)
  parser.add_argument("--repeat-validation-max-candidates", type=int, default=6)
  parser.add_argument("--repeat-validation-rollouts", type=int, default=3)
  parser.add_argument("--repeat-validation-min-win-rate", type=float, default=0.67)
  parser.add_argument("--repeat-validation-min-nonworse-rate", type=float, default=0.67)
  parser.add_argument("--repeat-validation-min-value-delta", type=float, default=0.01)
  parser.add_argument("--repeat-validation-max-risk-delta", type=float, default=0.0)
  parser.add_argument(
      "--repeat-validation-max-candidate-any-paradox-rate",
      type=float,
      default=0.0,
  )
  parser.add_argument("--repeat-validation-min-base-any-paradox-rate", type=float, default=0.0)
  parser.add_argument("--behavior-led-contrast-switch-prob", type=float, default=0.9)
  parser.add_argument("--behavior-red-switch-prob", type=float, default=0.9)
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  existing_metadata = _expand_patterns(args.existing_metadata)
  shard_rows: list[dict[str, Any]] = []
  summary = _write_combined_reports(out_root, existing_metadata, shard_rows)
  if (
      int(summary["all_metadata_count"]) >= int(args.target_source_count)
      or int(summary["combined_rows"]) >= int(args.target_total_rows)
  ):
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0

  for offset in range(max(0, int(args.max_shards))):
    shard_index = int(args.shard_start) + offset
    row = _run_collector(args, shard_index)
    shard_rows.append(row)
    summary = _write_combined_reports(out_root, existing_metadata, shard_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if (
        int(summary["all_metadata_count"]) >= int(args.target_source_count)
        or int(summary["combined_rows"]) >= int(args.target_total_rows)
    ):
      break
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
