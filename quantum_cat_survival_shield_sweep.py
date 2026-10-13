"""Run reproducible same-policy survival-shield screens."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


def _csv_str(value: str) -> list[str]:
  return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _csv_int(value: str) -> list[int]:
  return [int(part) for part in _csv_str(value)]


def _csv_float(value: str) -> list[float]:
  return [float(part) for part in _csv_str(value)]


def _slug(value: Any) -> str:
  text = str(value)
  for old, new in [
      (".", "p"),
      ("-", "m"),
      ("/", "_"),
      (":", "_"),
      (",", "_"),
      ("=", "-"),
  ]:
    text = text.replace(old, new)
  return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _load_json(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
  if not isinstance(data, dict):
    raise ValueError(f"Expected JSON object in {path}")
  return data


def _gate_summary(result: dict[str, Any] | None) -> dict[str, Any]:
  if not result:
    return {}
  gate = result.get("homogeneous_paradox_gate") or {}
  return {
      "hand_rounds": gate.get("hand_rounds"),
      "rounds_with_any_paradox": gate.get("rounds_with_any_paradox"),
      "same_policy_paradox_round_rate": gate.get("same_policy_paradox_round_rate"),
      "passed": gate.get("passed"),
      "ci95": gate.get("same_policy_paradox_round_rate_ci95"),
  }


def _build_baseline_command(
    args: argparse.Namespace,
    bot: str,
    seed: int,
    out_path: Path,
) -> list[str]:
  aliases = [f"h{seat}={bot}" for seat in range(args.players)]
  return [
      sys.executable,
      "quantum_cat_full_match_elo.py",
      "--players",
      str(args.players),
      "--matches",
      str(args.matches),
      "--bots",
      *aliases,
      "--match-context",
      "--homogeneous-paradox-threshold",
      str(args.homogeneous_paradox_threshold),
      "--workers",
      str(args.workers),
      "--seed",
      str(seed),
      "--out",
      str(out_path),
  ]


def _build_shield_command(
    args: argparse.Namespace,
    base_bot: str,
    continuation_bot: str,
    threshold: float,
    score_mode: str,
    lcb_z: float,
    rollouts: int,
    samples: int,
    max_actions: int,
    seed: int,
    out_path: Path,
) -> list[str]:
  aliases = [f"h{seat}=survival_shield" for seat in range(args.players)]
  command = [
      sys.executable,
      "quantum_cat_full_match_elo.py",
      "--players",
      str(args.players),
      "--matches",
      str(args.matches),
      "--bots",
      *aliases,
      "--survival-shield-base-bot",
      base_bot,
      "--survival-shield-continuation-bot",
      continuation_bot,
      "--survival-shield-threshold",
      str(threshold),
      "--survival-shield-score-mode",
      str(score_mode),
      "--survival-shield-lcb-z",
      str(lcb_z),
      "--survival-shield-rollouts",
      str(rollouts),
      "--survival-shield-samples",
      str(samples),
      "--survival-shield-max-actions",
      str(max_actions),
      "--survival-shield-include-bots",
      str(args.include_bots),
      "--survival-shield-feature-candidates",
      str(args.feature_candidates).lower(),
      "--match-context",
      "--homogeneous-paradox-threshold",
      str(args.homogeneous_paradox_threshold),
      "--workers",
      str(args.workers),
      "--seed",
      str(seed),
      "--out",
      str(out_path),
  ]
  if str(args.phases or "").strip():
    command.extend(["--survival-shield-phases", str(args.phases)])
  return command


def _run_command(
    command: list[str],
    out_dir: Path,
    timeout_sec: float,
    dry_run: bool,
) -> dict[str, Any]:
  out_dir.mkdir(parents=True, exist_ok=True)
  stdout_path = out_dir / "stdout.log"
  row = {
      "command": command,
      "command_text": shlex.join(command),
      "stdout": str(stdout_path),
  }
  if dry_run:
    row["status"] = "dry_run"
    return row
  env = os.environ.copy()
  env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"
  started = time.perf_counter()
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
          check=False,
          timeout=timeout_sec if timeout_sec > 0 else None,
      )
      row["exit_code"] = int(completed.returncode)
      row["status"] = "ok" if completed.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
      row["exit_code"] = 124
      row["status"] = "timeout"
      stdout_log.write(f"\ntimeout_sec={timeout_sec}\n")
  row["elapsed_sec"] = round(time.perf_counter() - started, 3)
  return row


def _write_summary(out_root: Path, rows: list[dict[str, Any]]) -> None:
  status_counts: dict[str, int] = {}
  best = None
  for row in rows:
    status = str(row.get("status") or "unknown")
    status_counts[status] = status_counts.get(status, 0) + 1
    rate = row.get("same_policy_paradox_round_rate")
    if rate is None:
      continue
    if best is None or float(rate) < float(best["same_policy_paradox_round_rate"]):
      best = row
  summary = {
      "status_counts": status_counts,
      "rows": rows,
      "best_by_same_policy_paradox_round_rate": best,
  }
  with (out_root / "sweep_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=10)
  parser.add_argument("--base-bots", default="heuristic_safe14")
  parser.add_argument(
      "--continuation-bots",
      default="",
      help="Comma-separated continuation bots. Empty uses each base bot.",
  )
  parser.add_argument("--thresholds", default="0.45,0.55,0.65,0.75")
  parser.add_argument(
      "--score-modes",
      default="wilson_lcb",
      help="Comma-separated survival shield score modes: mean,wilson_lcb.",
  )
  parser.add_argument("--lcb-z-values", default="1.96")
  parser.add_argument("--rollouts", default="16,32,64")
  parser.add_argument("--samples", default="1")
  parser.add_argument("--max-actions", default="6")
  parser.add_argument(
      "--include-bots",
      default="heuristic_safe14,heuristic_safe8,heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument("--feature-candidates", action=argparse.BooleanOptionalAction,
                      default=True)
  parser.add_argument("--phases", default="")
  parser.add_argument("--seed", type=int, default=20266600)
  parser.add_argument(
      "--shared-seed",
      action="store_true",
      help=(
          "Use the same evaluator seed for every config. This is useful for "
          "small paired smoke screens; larger sweeps should usually leave it "
          "off so configs see independent deal samples."
      ),
  )
  parser.add_argument("--workers", type=int, default=1)
  parser.add_argument("--command-timeout-sec", type=float, default=0.0)
  parser.add_argument("--homogeneous-paradox-threshold", type=float, default=0.40)
  parser.add_argument("--out-root", default="az_runs/survival_shield_sweep")
  parser.add_argument("--include-baselines", action="store_true")
  parser.add_argument("--rerun", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  base_bots = _csv_str(args.base_bots)
  continuation_bots = _csv_str(args.continuation_bots)
  thresholds = _csv_float(args.thresholds)
  score_modes = _csv_str(args.score_modes)
  lcb_z_values = _csv_float(args.lcb_z_values)
  rollouts_values = _csv_int(args.rollouts)
  sample_values = _csv_int(args.samples)
  max_action_values = _csv_int(args.max_actions)
  rows: list[dict[str, Any]] = []
  run_index = 0

  if args.include_baselines:
    for base_bot in base_bots:
      seed = args.seed if args.shared_seed else args.seed + run_index
      run_index += 1
      run_name = f"baseline_{_slug(base_bot)}_m{args.matches}_seed{seed}"
      run_dir = out_root / run_name
      out_path = run_dir / "result.json"
      command = _build_baseline_command(args, base_bot, seed, out_path)
      command_row = {
          "kind": "baseline",
          "base_bot": base_bot,
          "seed": seed,
          "output": str(out_path),
      }
      if out_path.exists() and not args.rerun and not args.dry_run:
        command_row.update({"status": "cached"})
      else:
        command_row.update(
            _run_command(command, run_dir, args.command_timeout_sec, args.dry_run)
        )
      result = _load_json(out_path)
      command_row.update(_gate_summary(result))
      rows.append(command_row)
      _write_summary(out_root, rows)

  for (
      base_bot,
      threshold,
      score_mode,
      lcb_z,
      rollouts,
      samples,
      max_actions,
  ) in itertools.product(
      base_bots,
      thresholds,
      score_modes,
      lcb_z_values,
      rollouts_values,
      sample_values,
      max_action_values,
  ):
    continuation_list = continuation_bots or [base_bot]
    for continuation_bot in continuation_list:
      seed = args.seed if args.shared_seed else args.seed + run_index
      run_index += 1
      run_name = (
          f"shield_base-{_slug(base_bot)}_cont-{_slug(continuation_bot)}_"
          f"t{_slug(threshold)}_{_slug(score_mode)}_z{_slug(lcb_z)}_"
          f"r{rollouts}_s{samples}_a{max_actions}_"
          f"m{args.matches}_seed{seed}"
      )
      run_dir = out_root / run_name
      out_path = run_dir / "result.json"
      command = _build_shield_command(
          args,
          base_bot,
          continuation_bot,
          threshold,
          score_mode,
          lcb_z,
          rollouts,
          samples,
          max_actions,
          seed,
          out_path,
      )
      command_row = {
          "kind": "survival_shield",
          "base_bot": base_bot,
          "continuation_bot": continuation_bot,
          "threshold": threshold,
          "score_mode": score_mode,
          "lcb_z": lcb_z,
          "rollouts": rollouts,
          "samples": samples,
          "max_actions": max_actions,
          "include_bots": args.include_bots,
          "feature_candidates": args.feature_candidates,
          "phases": args.phases,
          "seed": seed,
          "output": str(out_path),
      }
      if out_path.exists() and not args.rerun and not args.dry_run:
        command_row.update({"status": "cached"})
      else:
        command_row.update(
            _run_command(command, run_dir, args.command_timeout_sec, args.dry_run)
        )
      result = _load_json(out_path)
      command_row.update(_gate_summary(result))
      rows.append(command_row)
      _write_summary(out_root, rows)

  print(json.dumps({
      "out_root": str(out_root),
      "rows": rows,
  }, indent=2), flush=True)


if __name__ == "__main__":
  main()
