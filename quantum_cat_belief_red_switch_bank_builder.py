#!/usr/bin/env python3
"""Build belief-legal red-switch replay banks in restartable shards.

This wraps quantum_cat_belief_red_switch_probe.py so the expensive
belief-rollout labels can be collected across multiple seeds/sources and then
merged into one replay with aligned sidecar metadata.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _metadata_path_for_replay(replay_path: Path) -> Path:
  return replay_path.with_suffix(replay_path.suffix + ".metadata.json")


def _expand_paths(patterns: list[str]) -> list[Path]:
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


def _discover_replays(out_root: Path) -> list[Path]:
  return sorted(
      path for path in out_root.glob("**/belief_rollout_replay.npz")
      if path.is_file()
  )


def _discover_metadata(out_root: Path) -> list[Path]:
  return sorted(
      path for path in out_root.glob("**/belief_rollout_replay.npz.metadata.json")
      if path.is_file()
  )


def _replay_path_for_metadata(metadata_path: Path) -> Path:
  text = str(metadata_path)
  if text.endswith(".metadata.json"):
    return Path(text[: -len(".metadata.json")])
  return metadata_path.with_suffix("")


def _load_sidecar_rows(metadata_path: Path) -> list[dict[str, Any]]:
  data = json.loads(metadata_path.read_text())
  rows = data if isinstance(data, list) else data.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"metadata rows missing from {metadata_path}")
  return [dict(row) for row in rows]


def _probe_summary(probe_path: Path) -> dict[str, Any]:
  if not str(probe_path) or str(probe_path) == ".":
    return {}
  if not probe_path.exists() or not probe_path.is_file():
    return {}
  try:
    return json.loads(probe_path.read_text())
  except json.JSONDecodeError:
    return {}


def _bank_probe_fallback(metadata: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
  totals = metadata.get("totals") if isinstance(metadata, dict) else None
  if not isinstance(totals, dict):
    return {}, 0, 0
  summary = {
      "states": int(totals.get("audited_states", 0) or 0),
      "stable_belief_red_overrides_base": int(
          totals.get("stable_red_overrides", 0) or 0
      ),
      "belief_stable_rows": int(totals.get("stable_rows", 0) or 0),
      "belief_split_agree": int(totals.get("belief_split_agree", 0) or 0),
  }
  return (
      summary,
      int(totals.get("eligible_decisions", 0) or 0),
      int(totals.get("visited_decisions", 0) or 0),
  )


def _label_mix(metadata_rows: list[dict[str, Any]]) -> dict[str, int]:
  def _int_or(value: Any, fallback: int) -> int:
    if value is None:
      return int(fallback)
    try:
      return int(value)
    except (TypeError, ValueError):
      return int(fallback)

  mix = {
      "red_override": 0,
      "base_target": 0,
      "other_override": 0,
      "unknown": 0,
  }
  for row in metadata_rows:
    belief_flags = row.get("belief_best_flags", {})
    base_flags = row.get("base_flags", {})
    belief_best = _int_or(row.get("belief_best_action"), -1)
    base_action = _int_or(row.get("base_action"), _int_or(
        base_flags.get("action") if isinstance(base_flags, dict) else None,
        -2,
    ))
    is_red_switch = bool(
        row.get("belief_best_is_red_switch", False)
        or (isinstance(belief_flags, dict) and belief_flags.get("red_switch", False))
    )
    if is_red_switch and belief_best != base_action:
      mix["red_override"] += 1
    elif belief_best == base_action:
      mix["base_target"] += 1
    elif belief_best >= 0:
      mix["other_override"] += 1
    else:
      mix["unknown"] += 1
  return mix


def _run_probe(args: argparse.Namespace, batch_index: int) -> dict[str, Any]:
  batch_seed = int(args.seed_start) + int(batch_index) * int(args.seed_stride)
  batch_dir = Path(args.out_root) / f"batch_{batch_index:03d}"
  probe_path = batch_dir / "probe.json"
  replay_path = batch_dir / "belief_rollout_replay.npz"
  metadata_path = _metadata_path_for_replay(replay_path)
  if metadata_path.exists() and probe_path.exists():
    return {
        "batch_index": int(batch_index),
        "seed": int(batch_seed),
        "batch_dir": str(batch_dir),
        "probe": str(probe_path),
        "replay": str(replay_path) if replay_path.exists() else "",
        "metadata": str(metadata_path),
        "status": (
            "skipped_existing" if replay_path.exists()
            else "skipped_existing_zero_rows"
        ),
        "exit_code": 0,
    }

  command = [
      sys.executable,
      "quantum_cat_belief_red_switch_probe.py",
      "--checkpoint",
      str(args.checkpoint),
      "--out",
      str(probe_path),
      "--replay-out",
      str(replay_path),
      "--players",
      str(args.players),
      "--matches",
      str(args.matches),
      "--max-states",
      str(args.max_states),
      "--partial-save-interval-states",
      str(args.partial_save_interval_states),
      "--belief-samples",
      str(args.belief_samples),
      "--belief-candidates",
      str(args.belief_candidates),
      "--belief-source",
      str(args.belief_source),
      "--score-mode",
      str(args.score_mode),
      "--max-red-candidates",
      str(args.max_red_candidates),
      "--max-follow-candidates",
      str(args.max_follow_candidates),
      "--rollout-actual-rollouts",
      str(args.rollout_actual_rollouts),
      "--rollout-paradox-penalty",
      str(args.rollout_paradox_penalty),
      "--min-belief-action-coverage",
      str(args.min_belief_action_coverage),
      "--stable-margin",
      str(args.stable_margin),
      "--replay-row-filter",
      str(args.replay_row_filter),
      "--replay-target-temperature",
      str(args.replay_target_temperature),
      "--source-bot-mode",
      str(args.source_bot_mode),
      "--behavior-exploration-prob",
      str(args.behavior_exploration_prob),
      "--behavior-led-contrast-switch-prob",
      str(args.behavior_led_contrast_switch_prob),
      "--behavior-red-switch-prob",
      str(args.behavior_red_switch_prob),
      "--behavior-exploration-temperature",
      str(args.behavior_exploration_temperature),
      "--seed",
      str(batch_seed),
      "--torch-threads",
      str(args.torch_threads),
      "--device",
      str(args.device),
  ]
  if bool(args.require_actual_best_red_switch):
    command.append("--require-actual-best-red-switch")
  if bool(args.require_actual_best_overrides_base):
    command.append("--require-actual-best-overrides-base")
  if bool(args.match_context):
    command.append("--match-context")
  else:
    command.append("--no-match-context")

  if args.dry_run:
    return {
        "batch_index": int(batch_index),
        "seed": int(batch_seed),
        "batch_dir": str(batch_dir),
        "command": command,
        "status": "dry_run",
        "exit_code": None,
    }

  batch_dir.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env["PYTHONPATH"] = (
      "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"
  )
  stdout_path = batch_dir / "belief_bank_builder_probe.stdout.log"
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
        timeout=(
            None if float(args.command_timeout_sec) <= 0.0
            else float(args.command_timeout_sec)
        ),
    )
    stdout_log.write(f"\nexit_code={completed.returncode}\n")
  status = "completed" if completed.returncode == 0 else "probe_failed"
  return {
      "batch_index": int(batch_index),
      "seed": int(batch_seed),
      "batch_dir": str(batch_dir),
      "probe": str(probe_path),
      "replay": str(replay_path),
      "metadata": str(metadata_path),
      "stdout": str(stdout_path),
      "command": command,
      "status": status,
      "exit_code": int(completed.returncode),
  }


def _combine_sources(
    out_root: Path,
    metadata_paths: list[Path],
    existing_replay_paths: list[Path],
) -> dict[str, Any]:
  all_metadata_paths = {
      *metadata_paths,
      *(_metadata_path_for_replay(path) for path in existing_replay_paths),
  }
  combined_rows = []
  combined_metadata_rows = []
  row_group_ids = []
  sources = []
  totals = {
      "audited_states": 0,
      "eligible_decisions": 0,
      "visited_decisions": 0,
      "stable_red_overrides": 0,
      "stable_rows": 0,
      "belief_split_agree": 0,
  }

  for source_index, metadata_path in enumerate(sorted(all_metadata_paths)):
    if not metadata_path.exists():
      continue
    replay_path = _replay_path_for_metadata(metadata_path)
    replay = []
    replay_meta = {}
    if replay_path.exists():
      replay, replay_meta = load_replay(replay_path, return_metadata=True)
    sidecar_rows = _load_sidecar_rows(metadata_path)
    if len(sidecar_rows) != len(replay):
      raise ValueError(
          f"sidecar row count mismatch for {metadata_path}: "
          f"{len(sidecar_rows)} vs {len(replay)}"
      )
    sidecar_raw = json.loads(metadata_path.read_text())
    sidecar_data = sidecar_raw if isinstance(sidecar_raw, dict) else {}
    probe_path = Path(str(
        replay_meta.get("source_probe_out")
        or sidecar_data.get("source_probe_out")
        or ""
    ))
    probe_data = _probe_summary(probe_path)
    if probe_data:
      probe_summary = probe_data.get("summary", {})
      eligible_decisions = int(probe_data.get("eligible_decisions", 0) or 0)
      visited_decisions = int(probe_data.get("visited_decisions", 0) or 0)
    else:
      probe_summary, eligible_decisions, visited_decisions = (
          _bank_probe_fallback(sidecar_data)
      )
    source_name = metadata_path.parent.name
    source_row = {
        "source_index": int(source_index),
        "source_name": source_name,
        "replay": str(replay_path) if replay_path.exists() else "",
        "metadata": str(metadata_path),
        "rows": int(len(replay)),
        "metadata_sha256": _sha256(metadata_path),
        "probe": str(probe_path) if probe_path else "",
        "probe_summary": probe_summary,
    }
    if replay_path.exists():
      source_row["replay_sha256"] = _sha256(replay_path)
    sources.append(source_row)
    totals["audited_states"] += int(probe_summary.get("states", 0) or 0)
    totals["stable_red_overrides"] += int(
        probe_summary.get("stable_belief_red_overrides_base", 0) or 0
    )
    totals["stable_rows"] += int(probe_summary.get("belief_stable_rows", 0) or 0)
    totals["belief_split_agree"] += int(
        probe_summary.get("belief_split_agree", 0) or 0
    )
    totals["eligible_decisions"] += int(eligible_decisions)
    totals["visited_decisions"] += int(visited_decisions)
    for row_index, (row, metadata_row) in enumerate(zip(replay, sidecar_rows)):
      original_row_group_id = str(
          metadata_row.get(
              "row_group_id",
              (
                  f"m{metadata_row.get('match_index', -1)}:"
                  f"r{metadata_row.get('round_index', -1)}:"
                  f"d{metadata_row.get('decision_index', row_index)}"
              ),
          )
      )
      row_group_id = (
          f"source={source_index}|source_name={source_name}|"
          f"row={row_index}|{original_row_group_id}"
      )
      enriched = dict(metadata_row)
      enriched["source_replay_index"] = int(source_index)
      enriched["source_name"] = source_name
      enriched["source_replay"] = str(replay_path)
      enriched["source_metadata"] = str(metadata_path)
      enriched["source_row_index"] = int(row_index)
      enriched["original_row_group_id"] = original_row_group_id
      enriched["row_group_id"] = row_group_id
      combined_rows.append(row)
      combined_metadata_rows.append(enriched)
      row_group_ids.append(row_group_id)

  combined_replay = out_root / "combined_belief_rollout_replay.npz"
  combined_metadata = _metadata_path_for_replay(combined_replay)
  summary = {
      "kind": "belief_red_switch_bank",
      "out_root": str(out_root),
      "combined_replay": str(combined_replay),
      "combined_metadata": str(combined_metadata),
      "rows": int(len(combined_rows)),
      "sources": sources,
      "source_count": int(len(sources)),
      "row_group_ids": row_group_ids,
      "row_group_count": int(len(set(row_group_ids))),
      "label_mix": _label_mix(combined_metadata_rows),
      "totals": totals,
      "rates": {
          "stable_red_override_rate_per_audited_state": (
              float(totals["stable_red_overrides"] / totals["audited_states"])
              if totals["audited_states"] else 0.0
          ),
          "replay_row_rate_per_audited_state": (
              float(len(combined_rows) / totals["audited_states"])
              if totals["audited_states"] else 0.0
          ),
      },
  }
  if combined_rows:
    save_replay(combined_rows, combined_replay, metadata=summary)
    _write_json(combined_metadata, {**summary, "rows": combined_metadata_rows})
    summary["combined_replay_sha256"] = _sha256(combined_replay)
    summary["combined_metadata_sha256"] = _sha256(combined_metadata)
  _write_json(out_root / "belief_bank_summary.json", summary)
  return summary


def _targets_met(args: argparse.Namespace, summary: dict[str, Any]) -> bool:
  if int(summary.get("rows", 0) or 0) < int(args.target_total_rows):
    return False
  label_mix = summary.get("label_mix", {})
  red_count = int(label_mix.get("red_override", 0) or 0)
  base_count = int(label_mix.get("base_target", 0) or 0)
  if red_count < int(args.target_min_red_overrides):
    return False
  if base_count < int(args.target_min_base_targets):
    return False
  return True


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out-root", required=True)
  parser.add_argument("--existing-replay", action="append", default=[])
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--matches", type=int, default=10)
  parser.add_argument("--max-states", type=int, default=20)
  parser.add_argument("--partial-save-interval-states", type=int, default=10)
  parser.add_argument("--seed-start", type=int, default=20262700)
  parser.add_argument("--seed-stride", type=int, default=1009)
  parser.add_argument("--max-batches", type=int, default=1)
  parser.add_argument("--target-total-rows", type=int, default=120)
  parser.add_argument("--target-min-red-overrides", type=int, default=0)
  parser.add_argument("--target-min-base-targets", type=int, default=0)
  parser.add_argument("--match-context", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--belief-samples", type=int, default=2)
  parser.add_argument("--belief-candidates", type=int, default=4)
  parser.add_argument(
      "--belief-source",
      choices=("infostate", "policy_weighted", "policy_weighted_cap", "mixed", "actual"),
      default="infostate",
  )
  parser.add_argument(
      "--score-mode",
      choices=("table_liveness", "rollout_survival"),
      default="rollout_survival",
  )
  parser.add_argument("--max-red-candidates", type=int, default=1)
  parser.add_argument("--max-follow-candidates", type=int, default=1)
  parser.add_argument("--rollout-actual-rollouts", type=int, default=1)
  parser.add_argument("--rollout-paradox-penalty", type=float, default=5.0)
  parser.add_argument("--min-belief-action-coverage", type=float, default=0.75)
  parser.add_argument("--stable-margin", type=float, default=0.20)
  parser.add_argument("--require-actual-best-red-switch", action="store_true")
  parser.add_argument("--require-actual-best-overrides-base", action="store_true")
  parser.add_argument(
      "--replay-row-filter",
      choices=(
          "stable_red_overrides",
          "stable_overrides",
          "stable_base",
          "stable_red_or_base",
          "all_stable",
          "all",
      ),
      default="stable_red_overrides",
  )
  parser.add_argument("--replay-target-temperature", type=float, default=0.35)
  parser.add_argument("--source-bot-mode", default="policy")
  parser.add_argument("--behavior-exploration-prob", type=float, default=0.0)
  parser.add_argument("--behavior-led-contrast-switch-prob", type=float, default=0.35)
  parser.add_argument("--behavior-red-switch-prob", type=float, default=0.90)
  parser.add_argument("--behavior-exploration-temperature", type=float, default=0.8)
  parser.add_argument("--torch-threads", type=int, default=1)
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
  parser.add_argument("--command-timeout-sec", type=float, default=0.0)
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  existing_replays = _expand_paths(args.existing_replay)
  batch_rows = []
  summary = _combine_sources(out_root, _discover_metadata(out_root), existing_replays)
  if _targets_met(args, summary):
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0

  for batch_index in range(max(0, int(args.max_batches))):
    row = _run_probe(args, batch_index)
    batch_rows.append(row)
    summary = _combine_sources(out_root, _discover_metadata(out_root), existing_replays)
    summary["batches"] = batch_rows
    _write_json(out_root / "belief_bank_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if row["status"] not in {
        "completed",
        "skipped_existing",
        "skipped_existing_zero_rows",
        "dry_run",
    }:
      return int(row.get("exit_code") or 1)
    if _targets_met(args, summary):
      break
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
