#!/usr/bin/env python3
"""Run source-held-out screens for red-augmented repeated proposal banks."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import subprocess
import sys


def _expand_paths(patterns: list[str]) -> list[Path]:
  paths: list[Path] = []
  for pattern in patterns:
    matches = sorted(glob.glob(pattern))
    if matches:
      paths.extend(Path(path) for path in matches)
    else:
      paths.append(Path(pattern))
  return paths


def _metadata_rows(path: Path) -> int:
  try:
    data = json.loads(path.read_text())
  except json.JSONDecodeError:
    return 0
  rows = data.get("rows", [])
  return len(rows) if isinstance(rows, list) else 0


def _replay_for_metadata(path: Path) -> Path:
  data = json.loads(path.read_text())
  replay = str(data.get("replay", "") or "")
  if replay:
    return Path(replay)
  name = path.name
  if name.endswith(".metadata.json"):
    return path.with_name(name[: -len(".metadata.json")])
  raise ValueError(f"metadata has no replay path: {path}")


def _collect_sources(patterns: list[str]) -> tuple[list[Path], list[Path]]:
  metadata_paths = []
  for path in _expand_paths(patterns):
    if _metadata_rows(path) > 0:
      metadata_paths.append(path)
  if not metadata_paths:
    raise ValueError("no positive-row metadata sources found")
  replay_paths = [_replay_for_metadata(path) for path in metadata_paths]
  missing = [str(path) for path in replay_paths if not path.exists()]
  if missing:
    raise FileNotFoundError(f"missing replay files: {missing}")
  return replay_paths, metadata_paths


def _trainer_command(
    args,
    replay_paths: list[Path],
    metadata_paths: list[Path],
    fold_dir: Path,
    holdout: int,
) -> list[str]:
  train_sources = ",".join(
      str(idx) for idx in range(len(replay_paths)) if idx != int(holdout)
  )
  cmd = [
      sys.executable,
      "quantum_cat_proposal_veto_trainer.py",
      "--policy-context-checkpoint",
      args.policy_context_checkpoint,
      "--players",
      str(args.players),
      "--seed",
      str(args.seed + int(holdout)),
      "--train-source-indices",
      train_sources,
      "--validation-source-indices",
      str(holdout),
      "--epochs",
      str(args.epochs),
      "--hidden",
      str(args.hidden),
      "--dropout",
      str(args.dropout),
      "--lr",
      str(args.lr),
      "--weight-decay",
      str(args.weight_decay),
      "--loss-mode",
      "listwise",
      "--target-mode",
      "group_relative_soft",
      "--soft-target-temperature",
      str(args.soft_target_temperature),
      "--soft-target-risk-penalty",
      str(args.soft_target_risk_penalty),
      "--soft-target-unsafe-penalty",
      str(args.soft_target_unsafe_penalty),
      "--approve-label",
      "positive_safe_no_candidate_any_paradox",
      "--proposal-example-source",
      "scored_candidates",
      "--proposal-feature-mode",
      "scalar_action_flags_slate_state_round_history_context",
      "--proposal-model-kind",
      "deepset",
      "--observation-feature-mode",
      "summary",
      "--policy-context-mode",
      "summary",
      "--min-validation-approvals-for-live",
      str(args.min_validation_approvals_for_live),
      "--max-validation-candidate-any-paradox-rate-for-live",
      str(args.max_validation_candidate_any_paradox_rate_for_live),
      "--out",
      str(fold_dir / "report.json"),
  ]
  for replay in replay_paths:
    cmd.extend(["--replay", str(replay)])
  for metadata in metadata_paths:
    cmd.extend(["--metadata", str(metadata)])
  return cmd


def _sum_group_metric(folds: list[dict], key: str) -> int:
  return int(sum(int(fold["group_top1"].get(key, 0)) for fold in folds))


def _mean_weighted(folds: list[dict], key: str) -> float:
  total_groups = int(sum(int(fold["group_top1"].get("groups", 0)) for fold in folds))
  if total_groups <= 0:
    return 0.0
  numer = 0.0
  for fold in folds:
    groups = int(fold["group_top1"].get("groups", 0))
    numer += groups * float(fold["group_top1"].get(key, 0.0))
  return numer / float(total_groups)


def _compact_summary(args, replay_paths, metadata_paths, folds) -> dict:
  groups = _sum_group_metric(folds, "groups")
  top1_approved = _sum_group_metric(folds, "top1_approved")
  top1_candidate_any = _sum_group_metric(folds, "top1_candidate_any_paradox")
  top1_safe = _sum_group_metric(folds, "top1_safe")
  validation_examples = int(
      sum(int(fold["validation_summary"].get("examples", 0)) for fold in folds)
  )
  validation_approved = int(
      sum(int(fold["validation_summary"].get("approved", 0)) for fold in folds)
  )
  validation_candidate_any = int(
      sum(
          int(fold["validation_summary"].get("candidate_any_paradox", 0))
          for fold in folds
      )
  )
  return {
      "run": args.run_name,
      "question": (
          "Do the red-augmented repeated proposal slates still transfer under "
          "leave-one-source-out top-1 screening after adding new sources?"
      ),
      "sources": len(replay_paths),
      "replays": [str(path) for path in replay_paths],
      "metadata": [str(path) for path in metadata_paths],
      "aggregate": {
          "folds": len(folds),
          "gates_passed": int(sum(bool(fold.get("gate_passed")) for fold in folds)),
          "groups": groups,
          "top1_approved": top1_approved,
          "top1_approved_rate": top1_approved / max(1, groups),
          "top1_candidate_any_paradox": top1_candidate_any,
          "top1_candidate_any_paradox_rate": top1_candidate_any / max(1, groups),
          "top1_safe": top1_safe,
          "top1_safe_rate": top1_safe / max(1, groups),
          "top1_mean_dense_value_delta": _mean_weighted(
              folds, "top1_mean_dense_value_delta"
          ),
          "top1_mean_group_relative_reward": _mean_weighted(
              folds, "top1_mean_group_relative_reward"
          ),
          "top1_mean_paradox_delta": _mean_weighted(
              folds, "top1_mean_paradox_delta"
          ),
          "oracle_best_group_relative_reward": _mean_weighted(
              folds, "oracle_best_group_relative_reward"
          ),
          "validation_examples": validation_examples,
          "validation_approved_examples": validation_approved,
          "validation_candidate_any_examples": validation_candidate_any,
      },
      "folds": folds,
      "decision": (
          "This is an offline source-held-out screen only. Do not promote, "
          "install, live-screen, or distill from it unless the follow-up live "
          "gate is explicitly run and passes."
      ),
  }


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--metadata", action="append", required=True)
  parser.add_argument("--out-root", required=True)
  parser.add_argument("--run-name", default="red_augmented_repeat_fold_screen")
  parser.add_argument(
      "--policy-context-checkpoint",
      default=(
          "az_runs/run779_5p_low_exit_causal_20260613/"
          "action_aux_heads_low_exit_reset_s120_lr1e3_cpu/"
          "checkpoint_0000_loaded_replay_step_0040.pt"
      ),
  )
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=202640500)
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--hidden", type=int, default=24)
  parser.add_argument("--dropout", type=float, default=0.08)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--soft-target-temperature", type=float, default=0.35)
  parser.add_argument("--soft-target-risk-penalty", type=float, default=1.0)
  parser.add_argument("--soft-target-unsafe-penalty", type=float, default=1.0)
  parser.add_argument("--min-validation-approvals-for-live", type=int, default=1)
  parser.add_argument(
      "--max-validation-candidate-any-paradox-rate-for-live",
      type=float,
      default=0.10,
  )
  parser.add_argument("--max-folds", type=int, default=0)
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_root = Path(args.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  replay_paths, metadata_paths = _collect_sources(args.metadata)
  fold_count = len(replay_paths)
  if int(args.max_folds) > 0:
    fold_count = min(fold_count, int(args.max_folds))
  if args.dry_run:
    print(json.dumps({
        "sources": len(replay_paths),
        "folds": fold_count,
        "replays": [str(path) for path in replay_paths],
        "metadata": [str(path) for path in metadata_paths],
    }, indent=2, sort_keys=True))
    return 0

  folds = []
  for holdout in range(fold_count):
    fold_dir = out_root / f"fold_holdout_{holdout}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    cmd = _trainer_command(args, replay_paths, metadata_paths, fold_dir, holdout)
    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
    )
    (fold_dir / "trainer.stdout.log").write_text(completed.stdout)
    (fold_dir / "trainer.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
      raise RuntimeError(
          f"fold {holdout} failed with exit {completed.returncode}; "
          f"see {fold_dir / 'trainer.stderr.log'}"
      )
    report = json.loads((fold_dir / "report.json").read_text())
    folds.append({
        "fold": f"fold_holdout_{holdout}",
        "validation_source_indices": str(holdout),
        "gate_passed": bool(report.get("validation_gate", {}).get("passed", False)),
        "eligible_thresholds": int(
            report.get("validation_gate", {}).get("eligible_thresholds", 0)
        ),
        "best_threshold": report.get("validation_gate", {}).get("best_threshold"),
        "validation_summary": report.get("validation_summary", {}),
        "group_top1": report.get("validation_group_top1_summary", {}),
        "report": str(fold_dir / "report.json"),
      })
    summary = _compact_summary(args, replay_paths, metadata_paths, folds)
    (out_root / "compact_summary.partial.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

  summary = _compact_summary(args, replay_paths, metadata_paths, folds)
  (out_root / "compact_summary.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
