#!/usr/bin/env python3
"""Collect restartable exclusive red-passed repeat-validation replay shards.

This is a thin orchestration wrapper around quantum_cat_live_substitution_replay.
It preserves per-source artifacts, skips completed shards by default, limits
worker count, and writes an aggregate repeat-bank report after each shard.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from quantum_cat_repeat_bank_report import build_report


DEFAULT_CHECKPOINT = (
    "az_runs/run779_5p_low_exit_causal_20260613/"
    "action_aux_heads_low_exit_reset_s120_lr1e3_cpu/"
    "checkpoint_0000_loaded_replay_step_0040.pt"
)


def _parse_seeds(raw: str, seed_start: int, seed_count: int) -> list[int]:
  seeds: list[int] = []
  if raw:
    for chunk in raw.split(","):
      chunk = chunk.strip()
      if chunk:
        seeds.append(int(chunk))
  if seed_count > 0:
    seeds.extend(seed_start + offset for offset in range(seed_count))
  deduped: list[int] = []
  seen: set[int] = set()
  for seed in seeds:
    if seed not in seen:
      seen.add(seed)
      deduped.append(seed)
  return deduped


def _json_or_none(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
  if not isinstance(data, dict):
    raise ValueError(f"Expected JSON object in {path}")
  return data


def _metadata_path(out_path: Path) -> Path:
  return Path(str(out_path) + ".metadata.json")


def _summary_path(out_path: Path) -> Path:
  return Path(str(out_path) + ".summary.json")


def _completed(out_path: Path) -> bool:
  return (
      out_path.exists()
      and _metadata_path(out_path).exists()
      and _summary_path(out_path).exists()
  )


def _shard_path(args: argparse.Namespace, seed: int, source_index: int) -> Path:
  return (
      Path(args.out_root)
      / f"{args.prefix}_source{source_index}_seed{seed}.npz"
  )


def _build_command(
    args: argparse.Namespace, seed: int, source_index: int, out_path: Path
) -> list[str]:
  command = [
      sys.executable,
      "quantum_cat_live_substitution_replay.py",
      "--checkpoint",
      args.checkpoint,
      "--players",
      str(args.players),
      "--matches",
      str(args.matches),
      "--seed",
      str(seed),
      "--bot-mode",
      args.bot_mode,
      "--decision-phases",
      args.decision_phases,
      "--max-decisions",
      str(args.max_decisions),
      "--max-decisions-per-round",
      str(args.max_decisions_per_round),
      "--min-trick-number",
      str(args.min_trick_number),
      "--max-trick-number",
      str(args.max_trick_number),
      "--max-led-bucket-rows",
      str(args.max_led_bucket_rows),
      "--repeat-validation-candidate-mode",
      args.repeat_validation_candidate_mode,
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
      "--behavior-exploration-temperature",
      str(args.behavior_exploration_temperature),
      "--behavior-exploration-prob",
      str(args.behavior_exploration_prob),
      "--behavior-led-contrast-switch-prob",
      str(args.behavior_led_contrast_switch_prob),
      "--behavior-red-switch-prob",
      str(args.behavior_red_switch_prob),
      "--label-base-action-source",
      args.label_base_action_source,
      "--progress-interval",
      str(args.progress_interval),
      "--partial-save-interval-decisions",
      str(args.partial_save_interval_decisions),
      "--out",
      str(out_path),
  ]
  if args.match_context:
    command.append("--match-context")
  if args.led_contrast_only:
    command.append("--led-contrast-only")
  if args.require_red_switch_legal:
    command.append("--require-red-switch-legal")
  if args.require_policy_target_changed:
    command.append("--require-policy-target-changed")
  if args.repeat_validate_labels:
    command.append("--repeat-validate-labels")
  if args.repeat_validation_require_passed_red_switch:
    command.append("--repeat-validation-require-passed-red-switch")
  if args.repeat_validation_require_passed_follow_led:
    command.append("--repeat-validation-require-passed-follow-led")
  if args.repeat_validation_prefilter_candidates:
    command.append("--repeat-validation-prefilter-candidates")
  if args.record_repeat_validation_attempts:
    command.append("--record-repeat-validation-attempts")
  if args.repeat_validation_exclusive_action_targets:
    command.append("--repeat-validation-exclusive-action-targets")
  if args.enable_behavior_exploration:
    command.append("--enable-behavior-exploration")
  if args.device:
    command.extend(["--device", args.device])
  if args.max_repeat_validation_label_color_rows > 0:
    command.extend([
        "--max-repeat-validation-label-color-rows",
        str(args.max_repeat_validation_label_color_rows),
    ])
  return command


def _summarize_shard(
    seed: int,
    source_index: int,
    out_path: Path,
    status: str,
    *,
    exit_code: int | None = None,
    timed_out: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
  summary_data = _json_or_none(_summary_path(out_path)) or {}
  metadata_data = _json_or_none(_metadata_path(out_path)) or {}
  summary = summary_data.get("summary", {}) if summary_data else {}
  row_count = int(summary_data.get("rows", 0) or 0)
  if not row_count and isinstance(metadata_data.get("rows"), list):
    row_count = len(metadata_data["rows"])
  return {
      "seed": int(seed),
      "source_index": int(source_index),
      "status": status,
      "out": str(out_path),
      "metadata": str(_metadata_path(out_path)),
      "summary_path": str(_summary_path(out_path)),
      "stdout": str(out_path.with_suffix(out_path.suffix + ".stdout.log")),
      "exit_code": exit_code,
      "timed_out": bool(timed_out),
      "rows": row_count,
      "repeat_validation_attempted": int(
          summary.get("repeat_validation_attempted", 0)
      ),
      "repeat_validation_passed": int(
          summary.get("repeat_validation_passed", 0)
      ),
      "exclusive_targets_written": int(
          summary.get("repeat_validation_exclusive_action_targets_written", 0)
      ),
      "command": command or [],
  }


def _run_shard(
    args: argparse.Namespace, seed: int, source_index: int
) -> dict[str, Any]:
  out_path = _shard_path(args, seed, source_index)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  if _completed(out_path) and not args.rerun:
    return _summarize_shard(seed, source_index, out_path, "skipped_completed")

  command = _build_command(args, seed, source_index, out_path)
  if args.dry_run:
    return _summarize_shard(
        seed, source_index, out_path, "dry_run", command=command
    )

  env = os.environ.copy()
  env["PYTHONPATH"] = (
      "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"
  )
  for name in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    env[name] = str(max(1, int(args.torch_threads)))

  stdout_path = out_path.with_suffix(out_path.suffix + ".stdout.log")
  with stdout_path.open("w", encoding="utf-8") as stdout_log:
    stdout_log.write("$ " + shlex.join(command) + "\n")
    stdout_log.flush()
    timed_out = False
    try:
      completed = subprocess.run(
          command,
          cwd=Path(__file__).resolve().parent,
          env=env,
          stdout=stdout_log,
          stderr=subprocess.STDOUT,
          text=True,
          check=False,
          timeout=(
              float(args.command_timeout_sec)
              if args.command_timeout_sec > 0
              else None
          ),
      )
      exit_code = int(completed.returncode)
    except subprocess.TimeoutExpired:
      timed_out = True
      exit_code = 124
    stdout_log.write(f"\nexit_code={exit_code}\n")
    if timed_out:
      stdout_log.write(f"timeout_sec={args.command_timeout_sec}\n")

  if timed_out:
    status = "command_timeout"
  elif exit_code != 0:
    status = "command_failed"
  elif _completed(out_path):
    status = "completed"
  else:
    status = "missing_artifacts"
  return _summarize_shard(
      seed,
      source_index,
      out_path,
      status,
      exit_code=exit_code,
      timed_out=timed_out,
      command=command,
  )


def _completed_metadata_paths(rows: list[dict[str, Any]]) -> list[Path]:
  paths: list[Path] = []
  for row in rows:
    path = Path(str(row.get("metadata") or ""))
    if path.exists() and row.get("status") in {"completed", "skipped_completed"}:
      paths.append(path)
  return sorted(set(paths))


def _write_summary(out_root: Path, rows: list[dict[str, Any]]) -> None:
  status_counts: dict[str, int] = {}
  for row in rows:
    status = str(row.get("status") or "unknown")
    status_counts[status] = status_counts.get(status, 0) + 1
  metadata_paths = _completed_metadata_paths(rows)
  bank_report = build_report(metadata_paths) if metadata_paths else {}
  total_rows = int((bank_report.get("totals") or {}).get("rows", 0))
  summary = {
      "rows": sorted(rows, key=lambda row: int(row.get("source_index", -1))),
      "status_counts": status_counts,
      "completed_metadata": [str(path) for path in metadata_paths],
      "total_replay_rows": total_rows,
      "bank_report_path": str(out_root / "exclusive_bank_report.json"),
  }
  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "collector_summary.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n"
  )
  if metadata_paths:
    (out_root / "exclusive_bank_report.json").write_text(
        json.dumps(bank_report, indent=2, sort_keys=True) + "\n"
    )


def _run_sweep(
    args: argparse.Namespace, seeds: list[int], out_root: Path
) -> list[dict[str, Any]]:
  indexed_seeds = [
      (seed, int(args.source_start) + offset) for offset, seed in enumerate(seeds)
  ]
  rows: list[dict[str, Any]] = []
  max_workers = max(1, int(args.max_workers))
  if max_workers == 1:
    for seed, source_index in indexed_seeds:
      row = _run_shard(args, seed, source_index)
      rows.append(row)
      print(json.dumps(row, sort_keys=True), flush=True)
      _write_summary(out_root, rows)
      if args.stop_after_rows > 0:
        completed_rows = sum(int(row.get("rows", 0) or 0) for row in rows)
        if completed_rows >= int(args.stop_after_rows):
          break
    return rows

  if args.stop_after_rows > 0:
    raise ValueError("--stop-after-rows requires --max-workers 1")
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_seed = {
        executor.submit(_run_shard, args, seed, source_index): (seed, source_index)
        for seed, source_index in indexed_seeds
    }
    for future in concurrent.futures.as_completed(future_to_seed):
      seed, source_index = future_to_seed[future]
      try:
        row = future.result()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        row = {
            "seed": int(seed),
            "source_index": int(source_index),
            "status": "runner_exception",
            "error": repr(exc),
        }
      rows.append(row)
      print(json.dumps(row, sort_keys=True), flush=True)
      _write_summary(out_root, rows)
  rows.sort(key=lambda row: int(row.get("source_index", -1)))
  return rows


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
  parser.add_argument("--out-root", required=True)
  parser.add_argument("--prefix", default="red_passed_exclusive")
  parser.add_argument("--seeds", default="")
  parser.add_argument("--seed-start", type=int, default=20261800)
  parser.add_argument("--seed-count", type=int, default=0)
  parser.add_argument("--source-start", type=int, default=0)
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--matches", type=int, default=20)
  parser.add_argument("--bot-mode", default="q_policy")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--max-decisions", type=int, default=80)
  parser.add_argument("--max-decisions-per-round", type=int, default=2)
  parser.add_argument("--min-trick-number", type=int, default=0)
  parser.add_argument("--max-trick-number", type=int, default=99)
  parser.add_argument("--max-led-bucket-rows", type=int, default=0)
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument("--label-base-action-source", default="frozen_policy")
  parser.add_argument("--repeat-validation-candidate-mode", default="follow_red_value")
  parser.add_argument("--repeat-validation-max-candidates", type=int, default=6)
  parser.add_argument("--repeat-validation-rollouts", type=int, default=2)
  parser.add_argument("--repeat-validation-min-win-rate", type=float, default=1.0)
  parser.add_argument(
      "--repeat-validation-min-nonworse-rate", type=float, default=1.0
  )
  parser.add_argument("--repeat-validation-min-value-delta", type=float, default=0.01)
  parser.add_argument("--repeat-validation-max-risk-delta", type=float, default=0.0)
  parser.add_argument(
      "--repeat-validation-max-candidate-any-paradox-rate",
      type=float,
      default=0.0,
  )
  parser.add_argument(
      "--repeat-validation-min-base-any-paradox-rate",
      type=float,
      default=0.5,
  )
  parser.add_argument(
      "--max-repeat-validation-label-color-rows", type=int, default=0
  )
  parser.add_argument("--behavior-exploration-prob", type=float, default=1.0)
  parser.add_argument(
      "--behavior-exploration-temperature", type=float, default=0.75
  )
  parser.add_argument("--behavior-led-contrast-switch-prob", type=float, default=0.7)
  parser.add_argument("--behavior-red-switch-prob", type=float, default=0.95)
  parser.add_argument("--torch-threads", type=int, default=1)
  parser.add_argument("--max-workers", type=int, default=1)
  parser.add_argument("--partial-save-interval-decisions", type=int, default=0)
  parser.add_argument("--stop-after-rows", type=int, default=0)
  parser.add_argument("--command-timeout-sec", type=float, default=0.0)
  parser.add_argument("--rerun", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--no-match-context", dest="match_context", action="store_false")
  parser.add_argument("--no-led-contrast-only", dest="led_contrast_only", action="store_false")
  parser.add_argument(
      "--no-require-red-switch-legal",
      dest="require_red_switch_legal",
      action="store_false",
  )
  parser.add_argument(
      "--no-require-policy-target-changed",
      dest="require_policy_target_changed",
      action="store_false",
  )
  parser.add_argument(
      "--no-repeat-validate-labels",
      dest="repeat_validate_labels",
      action="store_false",
  )
  parser.add_argument(
      "--no-repeat-validation-require-passed-red-switch",
      dest="repeat_validation_require_passed_red_switch",
      action="store_false",
  )
  parser.add_argument(
      "--repeat-validation-require-passed-follow-led",
      action="store_true",
  )
  parser.add_argument(
      "--repeat-validation-prefilter-candidates",
      action="store_true",
      help=(
          "Use direct repeat-validation candidate prefiltering to skip dense "
          "all-legal labels for states where no candidate passes."
      ),
  )
  parser.add_argument(
      "--record-repeat-validation-attempts",
      action="store_true",
      help=(
          "Ask shard workers to write diagnostic repeat-validation attempt rows "
          "for accepted and rejected candidate queues."
      ),
  )
  parser.add_argument(
      "--no-repeat-validation-exclusive-action-targets",
      dest="repeat_validation_exclusive_action_targets",
      action="store_false",
  )
  parser.add_argument(
      "--no-enable-behavior-exploration",
      dest="enable_behavior_exploration",
      action="store_false",
  )
  parser.set_defaults(
      match_context=True,
      led_contrast_only=True,
      require_red_switch_legal=True,
      require_policy_target_changed=True,
      repeat_validate_labels=True,
      repeat_validation_require_passed_red_switch=True,
      repeat_validation_exclusive_action_targets=True,
      enable_behavior_exploration=True,
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  seeds = _parse_seeds(args.seeds, args.seed_start, args.seed_count)
  if not seeds:
    raise SystemExit("Provide --seeds or --seed-count with --seed-start.")
  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  rows = _run_sweep(args, seeds, out_root)
  _write_summary(out_root, rows)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
