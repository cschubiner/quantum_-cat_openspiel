#!/usr/bin/env python3
"""Build balanced Run932-style proposal-history banks in restartable shards."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from quantum_cat_alphazero_torch import load_replay, save_replay


DEFAULT_CHECKPOINT = (
    "az_runs/run779_5p_low_exit_causal_20260613/"
    "action_aux_heads_low_exit_reset_s120_lr1e3_cpu/"
    "checkpoint_0000_loaded_replay_step_0040.pt"
)
DEFAULT_PROPOSAL_MODEL = (
    "az_runs/run899_balanced_table_liveness_calibrated_20260614/"
    "holdout_cal_test_model.pt"
)


def _write_json(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _replay_path_for_metadata(metadata_path: Path, data: dict[str, Any] | None = None) -> Path:
  if isinstance(data, dict):
    replay = str(data.get("replay") or "").strip()
    if replay:
      return Path(replay)
  text = str(metadata_path)
  if text.endswith(".npz.metadata.json"):
    return Path(text[: -len(".metadata.json")])
  if text.endswith(".partial.metadata.json"):
    return Path(text[: -len(".metadata.json")] + ".npz")
  return metadata_path.with_suffix("")


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
  combined = {
      path for path in selected
      if path.name != "combined_balanced_history_replay.npz.metadata.json"
  }
  return sorted(combined)


def _metadata_rows(data: Any) -> list[dict[str, Any]]:
  if isinstance(data, list):
    return data
  if isinstance(data, dict):
    rows = data.get("rows") or data.get("metadata_rows") or data.get("metadata")
    if isinstance(rows, list):
      return rows
  return []


def _source_kind(metadata_path: Path, data: dict[str, Any]) -> str:
  outcome = str(data.get("require_proposal_outcome") or "")
  path_text = str(metadata_path)
  if "positive_safe_no_candidate_any" in outcome or "positive_no_any" in path_text:
    return "positive_no_any"
  if "candidate_any_paradox" in outcome or "avoidable" in path_text:
    return "avoidable_candidate_any"
  return "unknown"


def _candidate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
  counts = Counter()
  for row in rows:
    proposal = row.get("proposal") if isinstance(row, dict) else {}
    if not isinstance(proposal, dict):
      proposal = {}
    counts["scored_candidates"] += int(
        proposal.get("scored_outcome_available_count", 0) or 0
    )
    counts["positive_safe_no_candidate_any"] += int(
        proposal.get("scored_positive_safe_no_candidate_any_paradox_count", 0) or 0
    )
    counts["candidate_any"] += int(
        proposal.get("scored_candidate_any_paradox_count", 0) or 0
    )
    history = row.get("round_history_summary") if isinstance(row, dict) else {}
    if not isinstance(history, dict):
      history = {}
    counts["prior_avoidable_switch_events"] += int(
        history.get("prior_avoidable_switch_events", 0) or 0
    )
    counts["prior_avoidable_nonred_switch_events"] += int(
        history.get("prior_avoidable_nonred_switch_events", 0) or 0
    )
    counts["prior_avoidable_token_loss_events"] += int(
        history.get("prior_avoidable_token_loss_events", 0) or 0
    )
  return dict(counts)


def _metadata_row_group_id(metadata_row: dict[str, Any], row_index: int) -> str:
  explicit = metadata_row.get("row_group_id")
  if explicit is not None:
    return str(explicit)
  group = metadata_row.get("group")
  if not isinstance(group, dict):
    group = {}
  match_index = metadata_row.get("match_index", group.get("match_index", -1))
  round_index = metadata_row.get("round_index", group.get("round_index", -1))
  decision_index = metadata_row.get(
      "decision_index", group.get("decision_index", row_index)
  )
  return f"m{match_index}:r{round_index}:d{decision_index}"


def _collector_command(args: argparse.Namespace, kind: str, shard_index: int, out_path: Path) -> list[str]:
  if kind == "positive_no_any":
    seed = int(args.positive_seed_start) + int(shard_index)
    matches = int(args.positive_matches)
    max_rows = int(args.positive_rows_per_shard)
    max_decisions = int(args.positive_max_decisions)
    max_decisions_per_round = int(args.positive_max_decisions_per_round)
    outcome = "scored_has_positive_safe_no_candidate_any_paradox"
    extras: list[str] = []
  elif kind == "avoidable_candidate_any":
    seed = int(args.hard_negative_seed_start) + int(shard_index)
    matches = int(args.hard_negative_matches)
    max_rows = int(args.hard_negative_rows_per_shard)
    max_decisions = int(args.hard_negative_max_decisions)
    max_decisions_per_round = int(args.hard_negative_max_decisions_per_round)
    outcome = "scored_has_candidate_any_paradox"
    extras = [
        "--enable-behavior-exploration",
        "--behavior-led-contrast-switch-prob",
        str(args.behavior_led_contrast_switch_prob),
        "--behavior-red-switch-prob",
        str(args.behavior_red_switch_prob),
        "--min-round-history-avoidable-switch-events",
        str(args.min_hard_negative_avoidable_switch_events),
    ]
  else:
    raise ValueError(f"unknown source kind: {kind}")

  command = [
      sys.executable,
      "quantum_cat_live_substitution_replay.py",
      "--checkpoint",
      str(args.checkpoint),
      "--players",
      str(args.players),
      "--matches",
      str(matches),
      "--seed",
      str(seed),
      "--bot-mode",
      "policy",
      "--decision-phases",
      "play",
      "--proposal-mode",
      "action_set_transformer",
      "--action-set-transformer-model",
      str(args.proposal_model),
      "--action-set-transformer-base-mode",
      "policy",
      "--action-set-transformer-phases",
      "play",
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
      "--label-base-action-source",
      "frozen_policy",
      "--require-proposal",
      "--require-proposal-outcome",
      outcome,
      "--max-accepted-rows",
      str(max_rows),
      "--max-decisions",
      str(max_decisions),
      "--max-decisions-per-round",
      str(max_decisions_per_round),
      "--record-round-history",
      "--out",
      str(out_path),
  ]
  if bool(args.match_context):
    command.append("--match-context")
  command.extend(extras)
  return command


def _run_collector(args: argparse.Namespace, kind: str, shard_index: int) -> dict[str, Any]:
  kind_dir = Path(args.out_root) / kind
  prefix = "positive_no_any" if kind == "positive_no_any" else "avoidable_candidate_any"
  out_path = kind_dir / f"{prefix}_shard{shard_index:03d}.npz"
  metadata_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
  stdout_path = out_path.with_suffix(out_path.suffix + ".stdout.log")
  if metadata_path.exists() and out_path.exists():
    return {
        "kind": kind,
        "shard_index": int(shard_index),
        "out": str(out_path),
        "metadata": str(metadata_path),
        "status": "skipped_existing",
        "exit_code": 0,
    }
  command = _collector_command(args, kind, shard_index, out_path)
  if bool(args.dry_run):
    return {
        "kind": kind,
        "shard_index": int(shard_index),
        "out": str(out_path),
        "metadata": str(metadata_path),
        "command": command,
        "status": "dry_run",
        "exit_code": None,
    }

  env = os.environ.copy()
  env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".{os.pathsep}{env['PYTHONPATH']}"
  kind_dir.mkdir(parents=True, exist_ok=True)
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
  return {
      "kind": kind,
      "shard_index": int(shard_index),
      "out": str(out_path),
      "metadata": str(metadata_path) if metadata_path.exists() else "",
      "command": command,
      "stdout": str(stdout_path),
      "status": status,
      "exit_code": int(exit_code),
  }


def _combine_sources(
    out_root: Path,
    metadata_paths: list[Path],
    run_rows: list[dict[str, Any]],
) -> dict[str, Any]:
  combined_rows = []
  combined_metadata_rows = []
  row_group_ids = []
  sources = []
  kind_counts = Counter()
  totals = Counter()
  candidate_totals = Counter()

  for source_index, metadata_path in enumerate(sorted(metadata_paths)):
    data = json.loads(metadata_path.read_text())
    if not isinstance(data, dict):
      data = {"rows": _metadata_rows(data)}
    sidecar_rows = _metadata_rows(data)
    replay_path = _replay_path_for_metadata(metadata_path, data)
    replay = []
    replay_meta = {}
    if replay_path.exists():
      replay, replay_meta = load_replay(replay_path, return_metadata=True)
    if len(sidecar_rows) != len(replay):
      raise ValueError(
          f"sidecar row count mismatch for {metadata_path}: "
          f"{len(sidecar_rows)} vs {len(replay)}"
      )
    kind = _source_kind(metadata_path, data)
    counts = _candidate_counts(sidecar_rows)
    source_row = {
        "source_index": int(source_index),
        "source_kind": kind,
        "source_name": metadata_path.parent.name,
        "replay": str(replay_path) if replay_path.exists() else "",
        "metadata": str(metadata_path),
        "rows": int(len(replay)),
        "summary": data.get("summary", {}),
        "candidate_counts": counts,
        "metadata_sha256": _sha256(metadata_path),
    }
    if replay_path.exists():
      source_row["replay_sha256"] = _sha256(replay_path)
    if replay_meta:
      source_row["replay_meta_kind"] = str(replay_meta.get("kind") or "")
    sources.append(source_row)
    kind_counts[kind] += int(len(replay))
    totals["rows"] += int(len(replay))
    totals["sources"] += 1
    candidate_totals.update(counts)
    for row_index, (row, metadata_row) in enumerate(zip(replay, sidecar_rows)):
      original_row_group_id = _metadata_row_group_id(metadata_row, row_index)
      row_group_id = (
          f"source={source_index}|kind={kind}|"
          f"row={row_index}|{original_row_group_id}"
      )
      enriched = dict(metadata_row)
      enriched["source_replay_index"] = int(source_index)
      enriched["source_kind"] = kind
      enriched["source_name"] = metadata_path.parent.name
      enriched["source_replay"] = str(replay_path)
      enriched["source_metadata"] = str(metadata_path)
      enriched["source_row_index"] = int(row_index)
      enriched["original_row_group_id"] = original_row_group_id
      enriched["row_group_id"] = row_group_id
      combined_rows.append(row)
      combined_metadata_rows.append(enriched)
      row_group_ids.append(row_group_id)

  combined_replay = out_root / "combined_balanced_history_replay.npz"
  combined_metadata = combined_replay.with_suffix(combined_replay.suffix + ".metadata.json")
  summary = {
      "kind": "balanced_history_bank",
      "out_root": str(out_root),
      "combined_replay": str(combined_replay),
      "combined_metadata": str(combined_metadata),
      "rows": int(len(combined_rows)),
      "source_count": int(len(sources)),
      "sources": sources,
      "row_group_count": int(len(set(row_group_ids))),
      "row_group_ids": row_group_ids,
      "kind_counts": dict(kind_counts),
      "candidate_totals": dict(candidate_totals),
      "batches": run_rows,
      "rates": {
          "positive_no_any_row_fraction": (
              float(kind_counts["positive_no_any"] / len(combined_rows))
              if combined_rows else 0.0
          ),
          "avoidable_candidate_any_row_fraction": (
              float(kind_counts["avoidable_candidate_any"] / len(combined_rows))
              if combined_rows else 0.0
          ),
      },
  }
  if combined_rows:
    save_replay(combined_rows, combined_replay, metadata=summary)
    _write_json(combined_metadata, {**summary, "rows": combined_metadata_rows})
    summary["combined_replay_sha256"] = _sha256(combined_replay)
    summary["combined_metadata_sha256"] = _sha256(combined_metadata)
  _write_json(out_root / "balanced_history_bank_summary.json", summary)
  return summary


def _targets_met(args: argparse.Namespace, summary: dict[str, Any]) -> bool:
  counts = summary.get("kind_counts") or {}
  positive_rows = int(counts.get("positive_no_any", 0) or 0)
  hard_rows = int(counts.get("avoidable_candidate_any", 0) or 0)
  return (
      positive_rows >= int(args.target_positive_rows)
      and hard_rows >= int(args.target_hard_negative_rows)
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-root", required=True)
  parser.add_argument("--existing-metadata", action="append", default=[])
  parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
  parser.add_argument("--proposal-model", default=DEFAULT_PROPOSAL_MODEL)
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument(
      "--match-context",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--target-positive-rows", type=int, default=20)
  parser.add_argument("--target-hard-negative-rows", type=int, default=20)
  parser.add_argument("--max-batches", type=int, default=1)
  parser.add_argument("--start-shard-index", type=int, default=0)
  parser.add_argument("--positive-seed-start", type=int, default=202633001)
  parser.add_argument("--hard-negative-seed-start", type=int, default=202634001)
  parser.add_argument("--positive-matches", type=int, default=80)
  parser.add_argument("--hard-negative-matches", type=int, default=80)
  parser.add_argument("--positive-rows-per-shard", type=int, default=3)
  parser.add_argument("--hard-negative-rows-per-shard", type=int, default=3)
  parser.add_argument("--positive-max-decisions", type=int, default=240)
  parser.add_argument("--hard-negative-max-decisions", type=int, default=240)
  parser.add_argument("--positive-max-decisions-per-round", type=int, default=4)
  parser.add_argument("--hard-negative-max-decisions-per-round", type=int, default=5)
  parser.add_argument("--behavior-led-contrast-switch-prob", type=float, default=0.9)
  parser.add_argument("--behavior-red-switch-prob", type=float, default=0.75)
  parser.add_argument("--min-hard-negative-avoidable-switch-events", type=int, default=1)
  parser.add_argument("--action-set-transformer-max-actions", type=int, default=4)
  parser.add_argument("--action-set-transformer-top-policy", type=int, default=2)
  parser.add_argument(
      "--action-set-transformer-include-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument("--action-set-transformer-selection-value-bonus", type=float, default=0.05)
  parser.add_argument("--action-set-transformer-policy-tiebreak", type=float, default=0.01)
  parser.add_argument("--command-timeout-sec", type=float, default=900.0)
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  run_rows: list[dict[str, Any]] = []
  existing_metadata = _expand_patterns(args.existing_metadata)
  summary = _combine_sources(
      out_root,
      sorted(set(existing_metadata + _selected_metadata_under(out_root))),
      run_rows,
  )
  if _targets_met(args, summary):
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0

  for batch_offset in range(max(0, int(args.max_batches))):
    shard_index = int(args.start_shard_index) + int(batch_offset)
    if int((summary.get("kind_counts") or {}).get("positive_no_any", 0) or 0) < int(
        args.target_positive_rows
    ):
      row = _run_collector(args, "positive_no_any", shard_index)
      run_rows.append(row)
      if row["status"] not in {"completed", "skipped_existing", "dry_run"}:
        summary = _combine_sources(
            out_root,
            sorted(set(existing_metadata + _selected_metadata_under(out_root))),
            run_rows,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return int(row.get("exit_code") or 1)
    if int((summary.get("kind_counts") or {}).get("avoidable_candidate_any", 0) or 0) < int(
        args.target_hard_negative_rows
    ):
      row = _run_collector(args, "avoidable_candidate_any", shard_index)
      run_rows.append(row)
      if row["status"] not in {"completed", "skipped_existing", "dry_run"}:
        summary = _combine_sources(
            out_root,
            sorted(set(existing_metadata + _selected_metadata_under(out_root))),
            run_rows,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return int(row.get("exit_code") or 1)
    summary = _combine_sources(
        out_root,
        sorted(set(existing_metadata + _selected_metadata_under(out_root))),
        run_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if _targets_met(args, summary):
      break
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
