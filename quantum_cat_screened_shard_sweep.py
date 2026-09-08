"""Run artifact-preserving screened calibration shard sweeps.

This is a thin orchestration wrapper around quantum_cat_belief_mcts_teacher_audit.
It keeps the expensive teacher logic in one place while making multi-seed corpus
mining reproducible.
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


def _summarize_audit(seed: int, seed_dir: Path) -> dict[str, Any]:
  audit_path = seed_dir / "audit.json"
  metadata_path = seed_dir / "calibration.npz.metadata.json"
  audit = _json_or_none(audit_path)
  metadata = _json_or_none(metadata_path)
  if audit is None:
    return {
        "seed": seed,
        "status": "missing_audit",
        "directory": str(seed_dir),
    }
  calibration_summary = audit.get("calibration_summary") or {}
  classes = calibration_summary.get("classes") or {}
  return {
      "seed": seed,
      "status": (
          "accepted"
          if audit.get("early_stop_reason") == "calibration_targets_satisfied"
          else "rejected"
      ),
      "directory": str(seed_dir),
      "early_stop_reason": audit.get("early_stop_reason"),
      "states_evaluated": audit.get("states_evaluated"),
      "calibration_replay_rows": audit.get("calibration_replay_rows"),
      "teacher_stability_pass": audit.get("teacher_stability_pass"),
      "classes": classes,
      "target_counts": audit.get("calibration_target_class_counts") or {},
      "metadata_early_stop_reason": (
          metadata.get("early_stop_reason") if metadata else None
      ),
  }


def _stable_positive_count(row: dict[str, Any]) -> int:
  target_counts = row.get("target_counts") or {}
  classes = row.get("classes") or {}
  return int(target_counts.get("POSITIVE", classes.get("POSITIVE", 0)) or 0)


def _class_signal_count(row: dict[str, Any], class_names: str) -> int:
  target_counts = row.get("target_counts") or {}
  classes = row.get("classes") or {}
  total = 0
  for raw_name in str(class_names or "").split(","):
    name = raw_name.strip()
    if not name:
      continue
    total += int(target_counts.get(name, classes.get(name, 0)) or 0)
  return total


def _write_summary(out_root: Path, rows: list[dict[str, Any]]) -> None:
  accepted = [row for row in rows if row.get("status") == "accepted"]
  status_counts: dict[str, int] = {}
  for row in rows:
    status = str(row.get("status") or "unknown")
    status_counts[status] = status_counts.get(status, 0) + 1
  full_rejected_count = status_counts.get("rejected", 0)
  prescreen_rejected_count = status_counts.get("prescreen_rejected", 0)
  summary = {
      "rows": rows,
      "accepted_count": len(accepted),
      "rejected_count": full_rejected_count + prescreen_rejected_count,
      "full_rejected_count": full_rejected_count,
      "prescreen_rejected_count": prescreen_rejected_count,
      "missing_count": status_counts.get("missing_audit", 0),
      "status_counts": status_counts,
      "accepted_dirs": [
          (row.get("full") or {}).get("directory") or row["directory"]
          for row in accepted
      ],
  }
  with (out_root / "sweep_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")


def _build_command(args: argparse.Namespace, seed: int, seed_dir: Path) -> list[str]:
  return [
      sys.executable,
      "quantum_cat_belief_mcts_teacher_audit.py",
      "--checkpoint",
      args.checkpoint,
      "--players",
      str(args.players),
      "--states",
      str(args.states),
      "--max-matches",
      str(args.max_matches),
      "--max-states-per-match",
      str(args.max_states_per_match),
      "--state-mix",
      args.state_mix,
      "--state-min-legal-count",
      str(args.state_min_legal_count),
      "--state-max-legal-count",
      str(args.state_max_legal_count),
      "--state-min-trick-number",
      str(args.state_min_trick_number),
      "--phases",
      args.phases,
      "--budgets",
      args.budgets,
      "--repeat-label-passes",
      str(args.repeat_label_passes),
      "--audit-rollouts",
      str(args.audit_rollouts),
      "--audit-belief-samples",
      str(args.audit_belief_samples),
      "--audit-belief-candidates",
      str(args.audit_belief_candidates),
      "--teacher-belief-source",
      args.teacher_belief_source,
      "--teacher-belief-policy-temperature",
      str(args.teacher_belief_policy_temperature),
      "--teacher-belief-uniform-mix",
      str(args.teacher_belief_uniform_mix),
      "--pass-min-high-margin-rows",
      str(args.pass_min_high_margin_rows),
      "--calibration-replay-out",
      str(seed_dir / "calibration.npz"),
      "--calibration-include-policy-matches",
      "--calibration-positive-margin",
      str(args.calibration_positive_margin),
      "--calibration-max-positive-paradox-delta",
      str(args.calibration_max_positive_paradox_delta),
      "--calibration-target-positive-rows",
      str(args.calibration_target_positive_rows),
      "--calibration-target-negative-rows",
      str(args.calibration_target_negative_rows),
      "--calibration-target-require-stable",
      "--calibration-stable-min-top1-mode-rate",
      str(args.calibration_stable_min_top1_mode_rate),
      "--calibration-stable-min-pairwise-spearman",
      str(args.calibration_stable_min_pairwise_spearman),
      "--calibration-stable-min-top2-jaccard",
      str(args.calibration_stable_min_top2_jaccard),
      "--calibration-stable-min-consensus-top-margin",
      str(args.calibration_stable_min_consensus_top_margin),
      "--calibration-demote-unstable-positives",
      "--calibration-abort-after-states",
      str(args.calibration_abort_after_states),
      "--calibration-abort-min-positive-rows",
      str(args.calibration_abort_min_positive_rows),
      "--full-match-rollout",
      "--rollout-max-plies",
      str(args.rollout_max_plies),
      "--match-context",
      "--progress-interval",
      str(args.progress_interval),
      "--out",
      str(seed_dir / "audit.json"),
      "--seed",
      str(seed),
  ]


def _args_with_overrides(
    args: argparse.Namespace, **overrides: Any
) -> argparse.Namespace:
  data = vars(args).copy()
  data.update(overrides)
  return argparse.Namespace(**data)


def _run_seed_dir(
    args: argparse.Namespace, seed: int, seed_dir: Path
) -> dict[str, Any]:
  seed_dir.mkdir(parents=True, exist_ok=True)
  audit_path = seed_dir / "audit.json"
  if audit_path.exists() and not args.rerun:
    return _summarize_audit(seed, seed_dir)

  command = _build_command(args, seed, seed_dir)
  if args.dry_run:
    return {
        "seed": seed,
        "status": "dry_run",
        "directory": str(seed_dir),
        "command": command,
    }

  env = os.environ.copy()
  env["PYTHONPATH"] = (
      "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"
  )
  with (seed_dir / "stdout.log").open("w", encoding="utf-8") as stdout_log:
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
  row = _summarize_audit(seed, seed_dir)
  row["exit_code"] = exit_code
  if timed_out:
    row["status"] = "command_timeout"
    row["timeout_sec"] = args.command_timeout_sec
  elif exit_code != 0:
    row["status"] = "command_failed"
  return row


def _run_seed(args: argparse.Namespace, seed: int, out_root: Path) -> dict[str, Any]:
  return _run_seed_dir(args, seed, out_root / f"seed_{seed}")


def _prescreen_args(args: argparse.Namespace) -> argparse.Namespace:
  state_mix = args.prescreen_state_mix
  if not state_mix:
    state_mix = f"near_paradox={args.prescreen_states}"
  return _args_with_overrides(
      args,
      states=args.prescreen_states,
      max_matches=args.prescreen_max_matches,
      max_states_per_match=args.prescreen_max_states_per_match,
      state_mix=state_mix,
      budgets=args.prescreen_budgets,
      repeat_label_passes=args.prescreen_repeat_label_passes,
      audit_rollouts=args.prescreen_audit_rollouts,
      audit_belief_samples=args.prescreen_audit_belief_samples,
      audit_belief_candidates=args.prescreen_audit_belief_candidates,
      pass_min_high_margin_rows=args.prescreen_pass_min_high_margin_rows,
      calibration_abort_after_states=args.prescreen_abort_after_states,
      calibration_abort_min_positive_rows=args.prescreen_abort_min_positive_rows,
      calibration_target_positive_rows=args.prescreen_target_positive_rows,
      calibration_target_negative_rows=args.prescreen_target_negative_rows,
      progress_interval=args.prescreen_progress_interval,
      command_timeout_sec=args.prescreen_timeout_sec,
  )


def _run_prescreened_seed(
    args: argparse.Namespace, seed: int, out_root: Path
) -> dict[str, Any]:
  seed_dir = out_root / f"seed_{seed}"
  prescreen_dir = seed_dir / "prescreen"
  full_dir = seed_dir / "full"
  if args.dry_run:
    return {
        "seed": seed,
        "status": "dry_run",
        "directory": str(seed_dir),
        "prescreen_command": _build_command(_prescreen_args(args), seed, prescreen_dir),
        "full_command": _build_command(args, seed, full_dir),
    }
  prescreen_row = _run_seed_dir(_prescreen_args(args), seed, prescreen_dir)
  prescreen_row["stage"] = "prescreen"
  prescreen_positive_rows = _stable_positive_count(prescreen_row)
  prescreen_signal_rows = _class_signal_count(
      prescreen_row, args.prescreen_signal_classes
  )
  prescreen_stability_pass = bool(prescreen_row.get("teacher_stability_pass"))
  advances = (
      prescreen_row.get("exit_code", 0) == 0
      and prescreen_signal_rows >= int(args.prescreen_min_signal_rows)
      and (
          not args.prescreen_require_teacher_stability
          or prescreen_stability_pass
      )
  )
  if not advances:
    return {
        "seed": seed,
        "status": "prescreen_rejected",
        "directory": str(seed_dir),
        "prescreen": prescreen_row,
        "prescreen_positive_rows": prescreen_positive_rows,
        "prescreen_signal_rows": prescreen_signal_rows,
        "prescreen_signal_classes": args.prescreen_signal_classes,
        "prescreen_min_signal_rows": int(args.prescreen_min_signal_rows),
        "prescreen_stability_pass": prescreen_stability_pass,
        "prescreen_require_teacher_stability": bool(
            args.prescreen_require_teacher_stability
        ),
    }

  if args.dry_run:
    full_command = _build_command(args, seed, full_dir)
    return {
        "seed": seed,
        "status": "dry_run",
        "directory": str(seed_dir),
        "prescreen": prescreen_row,
        "prescreen_positive_rows": prescreen_positive_rows,
        "prescreen_signal_rows": prescreen_signal_rows,
        "prescreen_signal_classes": args.prescreen_signal_classes,
        "prescreen_stability_pass": prescreen_stability_pass,
        "prescreen_require_teacher_stability": bool(
            args.prescreen_require_teacher_stability
        ),
        "full_command": full_command,
    }

  full_row = _run_seed_dir(args, seed, full_dir)
  full_row["stage"] = "full"
  return {
      "seed": seed,
      "status": full_row.get("status"),
      "directory": str(seed_dir),
      "prescreen": prescreen_row,
      "prescreen_positive_rows": prescreen_positive_rows,
      "prescreen_signal_rows": prescreen_signal_rows,
      "prescreen_signal_classes": args.prescreen_signal_classes,
      "prescreen_stability_pass": prescreen_stability_pass,
      "prescreen_require_teacher_stability": bool(
          args.prescreen_require_teacher_stability
      ),
      "full": full_row,
      "early_stop_reason": full_row.get("early_stop_reason"),
      "states_evaluated": full_row.get("states_evaluated"),
      "calibration_replay_rows": full_row.get("calibration_replay_rows"),
      "teacher_stability_pass": full_row.get("teacher_stability_pass"),
      "classes": full_row.get("classes") or {},
      "target_counts": full_row.get("target_counts") or {},
  }


def _run_sweep(
    args: argparse.Namespace, seeds: list[int], out_root: Path
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  accepted_count = 0
  max_workers = max(1, int(args.max_workers))
  run_one = _run_prescreened_seed if args.prescreen else _run_seed
  if max_workers == 1:
    for seed in seeds:
      row = run_one(args, seed, out_root)
      rows.append(row)
      print(json.dumps(row, sort_keys=True), flush=True)
      if row.get("status") == "accepted":
        accepted_count += 1
        if (
            args.stop_after_accepted > 0
            and accepted_count >= args.stop_after_accepted
        ):
          break
      _write_summary(out_root, rows)
    return rows

  if args.stop_after_accepted > 0:
    raise ValueError("--stop-after-accepted requires --max-workers 1")

  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_seed = {
        executor.submit(run_one, args, seed, out_root): seed for seed in seeds
    }
    for future in concurrent.futures.as_completed(future_to_seed):
      seed = future_to_seed[future]
      try:
        row = future.result()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        row = {
            "seed": seed,
            "status": "runner_exception",
            "error": repr(exc),
            "directory": str(out_root / f"seed_{seed}"),
        }
      rows.append(row)
      print(json.dumps(row, sort_keys=True), flush=True)
      _write_summary(out_root, rows)
  rows.sort(key=lambda row: int(row.get("seed", -1)))
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", default="az_runs/best_3p_policy_checkpoint.pt")
  parser.add_argument("--out-root", required=True)
  parser.add_argument("--seeds", default="")
  parser.add_argument("--seed-start", type=int, default=0)
  parser.add_argument("--seed-count", type=int, default=0)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--states", type=int, default=216)
  parser.add_argument("--max-matches", type=int, default=4096)
  parser.add_argument("--max-states-per-match", type=int, default=3)
  parser.add_argument("--state-mix", default="near_paradox=216")
  parser.add_argument("--state-min-legal-count", type=int, default=2)
  parser.add_argument("--state-max-legal-count", type=int, default=6)
  parser.add_argument("--state-min-trick-number", type=int, default=3)
  parser.add_argument("--phases", default="play")
  parser.add_argument("--budgets", default="32x4,64x8")
  parser.add_argument("--repeat-label-passes", type=int, default=5)
  parser.add_argument("--audit-rollouts", type=int, default=32)
  parser.add_argument("--audit-belief-samples", type=int, default=16)
  parser.add_argument("--audit-belief-candidates", type=int, default=64)
  parser.add_argument("--teacher-belief-source", default="policy_weighted")
  parser.add_argument("--teacher-belief-policy-temperature", type=float, default=1.0)
  parser.add_argument("--teacher-belief-uniform-mix", type=float, default=0.05)
  parser.add_argument("--pass-min-high-margin-rows", type=int, default=20)
  parser.add_argument("--calibration-positive-margin", type=float, default=0.25)
  parser.add_argument("--calibration-max-positive-paradox-delta", type=float, default=-0.05)
  parser.add_argument("--calibration-target-positive-rows", type=int, default=12)
  parser.add_argument("--calibration-target-negative-rows", type=int, default=80)
  parser.add_argument("--calibration-stable-min-top1-mode-rate", type=float, default=0.6)
  parser.add_argument("--calibration-stable-min-pairwise-spearman", type=float, default=0.35)
  parser.add_argument("--calibration-stable-min-top2-jaccard", type=float, default=0.4)
  parser.add_argument("--calibration-stable-min-consensus-top-margin", type=float, default=0.05)
  parser.add_argument("--calibration-abort-after-states", type=int, default=48)
  parser.add_argument("--calibration-abort-min-positive-rows", type=int, default=3)
  parser.add_argument("--rollout-max-plies", type=int, default=160)
  parser.add_argument("--progress-interval", type=int, default=24)
  parser.add_argument("--stop-after-accepted", type=int, default=0)
  parser.add_argument("--max-workers", type=int, default=1)
  parser.add_argument(
      "--command-timeout-sec",
      type=float,
      default=0.0,
      help="Per-seed wall-clock timeout. Zero disables the timeout.",
  )
  parser.add_argument("--rerun", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--prescreen", action="store_true")
  parser.add_argument("--prescreen-min-positive-rows", type=int, default=1)
  parser.add_argument(
      "--prescreen-min-signal-rows",
      type=int,
      default=None,
      help=(
          "Minimum prescreen signal rows required to launch full audit. "
          "Defaults to --prescreen-min-positive-rows for compatibility."
      ),
  )
  parser.add_argument(
      "--prescreen-signal-classes",
      default="POSITIVE",
      help=(
          "Comma-separated calibration classes counted as prescreen signal, "
          "for example POSITIVE,NEAR_POSITIVE."
      ),
  )
  parser.add_argument(
      "--prescreen-require-teacher-stability",
      action="store_true",
      help="Launch full audit only if the prescreen teacher stability gate passes.",
  )
  parser.add_argument("--prescreen-states", type=int, default=24)
  parser.add_argument("--prescreen-max-matches", type=int, default=512)
  parser.add_argument("--prescreen-max-states-per-match", type=int, default=3)
  parser.add_argument("--prescreen-state-mix", default="")
  parser.add_argument("--prescreen-budgets", default="8x2,16x4")
  parser.add_argument("--prescreen-repeat-label-passes", type=int, default=2)
  parser.add_argument("--prescreen-audit-rollouts", type=int, default=8)
  parser.add_argument("--prescreen-audit-belief-samples", type=int, default=4)
  parser.add_argument("--prescreen-audit-belief-candidates", type=int, default=16)
  parser.add_argument("--prescreen-pass-min-high-margin-rows", type=int, default=4)
  parser.add_argument("--prescreen-abort-after-states", type=int, default=12)
  parser.add_argument("--prescreen-abort-min-positive-rows", type=int, default=1)
  parser.add_argument("--prescreen-target-positive-rows", type=int, default=1)
  parser.add_argument("--prescreen-target-negative-rows", type=int, default=0)
  parser.add_argument("--prescreen-progress-interval", type=int, default=12)
  parser.add_argument(
      "--prescreen-timeout-sec",
      type=float,
      default=0.0,
      help="Per-seed prescreen timeout. Zero disables the timeout.",
  )
  args = parser.parse_args()
  if args.prescreen_min_signal_rows is None:
    args.prescreen_min_signal_rows = args.prescreen_min_positive_rows

  seeds = _parse_seeds(args.seeds, args.seed_start, args.seed_count)
  if not seeds:
    parser.error("Provide --seeds or --seed-count with --seed-start.")

  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)

  rows = _run_sweep(args, seeds, out_root)
  _write_summary(out_root, rows)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
