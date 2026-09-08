#!/usr/bin/env python3
"""Mine first low-exit warning states before a forced paradox.

This is a cheaper trajectory-label path than broad rollout liveness mining. It
plays normal same-policy rounds, finds failed rounds, then labels the earliest
pre-paradox play decision whose legal count is already low. The replay teaches
which legal substitution avoids, delays, or softens the later forced paradox.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyspiel
import torch

from quantum_cat_alphazero_torch import action_feature_matrix, save_replay
from quantum_cat_failure_miner import (
    action_string,
    dense_label_row_for_decision,
    evaluate_decision,
    parse_phase_filter,
    play_round_with_history,
    terminal_round_result,
)
from quantum_cat_full_match_elo import load_neural


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=40)
  parser.add_argument("--seed", type=int, default=20289301)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--min-trick-number", type=int, default=5)
  parser.add_argument("--max-warning-legal-count", type=int, default=3)
  parser.add_argument("--lookback-decisions", type=int, default=24)
  parser.add_argument("--max-failure-rounds", type=int, default=80)
  parser.add_argument("--max-rows", type=int, default=300)
  parser.add_argument("--min-score-delta", type=float, default=-99.0)
  parser.add_argument(
      "--pair-objective",
      choices=("survive", "survive_or_delay"),
      default="survive_or_delay",
  )
  parser.add_argument("--safe-action-value-target", type=float, default=0.9)
  parser.add_argument("--risky-action-value-target", type=float, default=-0.9)
  parser.add_argument("--improved-risk-target", type=float, default=0.35)
  parser.add_argument(
      "--dense-action-labels",
      action="store_true",
      help=(
          "For kept warning states, label every evaluated legal action using "
          "the dense same-policy continuation outcome targets from "
          "quantum_cat_failure_miner. The default keeps the historical sparse "
          "selected-vs-best pair labels."
      ),
  )
  parser.add_argument("--dense-target-temperature", type=float, default=0.05)
  parser.add_argument(
      "--dense-action-value-objective",
      choices=("safety", "safety_raw", "score"),
      default="safety",
  )
  parser.add_argument("--dense-safety-value-safe-target", type=float, default=0.9)
  parser.add_argument("--dense-safety-value-paradox-target", type=float, default=-0.55)
  parser.add_argument("--dense-safety-value-forced-penalty", type=float, default=0.25)
  parser.add_argument("--dense-safety-value-actor-penalty", type=float, default=0.10)
  parser.add_argument("--dense-safety-value-score-weight", type=float, default=0.05)
  parser.add_argument("--dense-safety-value-min", type=float, default=-0.95)
  parser.add_argument("--dense-safety-value-max", type=float, default=0.95)
  parser.add_argument(
      "--bot-mode",
      choices=("policy", "q_policy", "liveness_policy", "q_liveness_policy"),
      default="policy",
  )
  parser.add_argument("--liveness-shield-phases", default="play")
  parser.add_argument("--liveness-shield-min-open-slot-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-min-public-damage-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument(
      "--action-paradox-rerank-mode",
      choices=("additive", "threshold", "relative"),
      default="additive",
  )
  parser.add_argument("--action-paradox-risk-threshold", type=float, default=0.0)
  parser.add_argument("--action-paradox-min-risk-margin", type=float, default=0.0)
  parser.add_argument("--action-feasibility-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-max-policy-log-gap", type=float, default=2.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument("--action-value-rerank-phases", default="")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  return parser.parse_args()


def trigger_trick(record):
  trigger = record.get("first_new_paradox_trigger")
  if not trigger:
    return 99
  return int(trigger.get("trick_number", 99))


def trigger_legal_count(record):
  trigger = record.get("first_new_paradox_trigger")
  if not trigger:
    return 99
  return int(trigger.get("legal_count", 99))


def safety_key(record):
  return (
      int(bool(record.get("any_paradox", False))),
      int(bool(record.get("forced_paradox", False))),
      int(bool(record.get("actor_paradox", False))),
      -trigger_trick(record),
      -trigger_legal_count(record),
      -float(record.get("acting_score_delta", 0.0)),
      -float(record.get("total_score_delta", 0.0)),
  )


def candidate_improves(candidate, baseline_record, args):
  if safety_key(candidate) >= safety_key(baseline_record):
    return False
  if str(args.pair_objective) == "survive":
    return not bool(candidate.get("any_paradox", False))
  return True


def find_warning_decision(decisions, first_paradox_index, allowed_phases, args):
  if first_paradox_index is None:
    return None
  end = max(0, int(first_paradox_index))
  start = max(0, end - int(args.lookback_decisions))
  candidates = []
  for decision in decisions[start:end]:
    if str(decision.get("phase")) not in allowed_phases:
      continue
    if len(decision.get("legal", [])) < 2:
      continue
    if len(decision.get("legal", [])) > int(args.max_warning_legal_count):
      continue
    if int(decision.get("trick_number", -1)) < int(args.min_trick_number):
      continue
    candidates.append(decision)
  if not candidates:
    return None
  return min(
      candidates,
      key=lambda row: (int(row.get("trick_number", 99)), int(row.get("index", 99))),
  )


def replay_row_for_warning(decision, best_candidate, baseline_record, args):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  selected_action = int(decision["action"])
  best_action = int(best_candidate["action"])
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[list(decision["legal"])] = 1.0
  policy_target = np.zeros(num_actions, dtype=np.float32)
  policy_target[best_action] = 1.0

  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  action_paradox_targets[selected_action] = (
      1.0 if bool(baseline_record.get("any_paradox", False)) else 0.0
  )
  action_paradox_targets[best_action] = (
      0.0
      if not bool(best_candidate.get("any_paradox", False))
      else float(args.improved_risk_target)
  )
  action_paradox_mask[selected_action] = 1.0
  action_paradox_mask[best_action] = 1.0

  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_value_targets[selected_action] = float(args.risky_action_value_target)
  action_value_targets[best_action] = float(args.safe_action_value_target)
  action_value_mask[selected_action] = 1.0
  action_value_mask[best_action] = 1.0

  return (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      policy_target,
      np.zeros(int(args.players), dtype=np.float32),
      np.zeros(int(args.players), dtype=np.float32),
      best_action,
      player,
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      action_feature_matrix(state, player, num_actions),
  )


def main():
  args = parse_args()
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model, model_args = load_neural(args.checkpoint, args.players, device, args)
  value_scale = float(model_args.value_scale)
  allowed_phases = parse_phase_filter(args.decision_phases)
  rows = []
  metadata_rows = []
  summary = Counter()

  for match_idx in range(int(args.matches)):
    if int(args.max_rows) > 0 and len(rows) >= int(args.max_rows):
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if int(args.max_rows) > 0 and len(rows) >= int(args.max_rows):
        break
      if (
          int(args.max_failure_rounds) > 0
          and summary["failure_rounds"] >= int(args.max_failure_rounds)
      ):
        break
      start_player = round_idx % int(args.players)
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": int(args.players),
              "start_player": int(start_player),
              "match_context": int(bool(args.match_context)),
          },
      )
      prior_totals = np.copy(totals)
      round_seed = int(args.seed) + match_idx * 1000003 + round_idx * 9973
      terminal, decisions, first_paradox_index = play_round_with_history(
          model,
          game,
          seed=round_seed,
          value_scale=value_scale,
          device=device,
          match_totals=prior_totals if args.match_context else None,
          round_index=round_idx,
          args=args,
      )
      returns = np.array(terminal.returns(), dtype=np.float32)
      totals = returns if args.match_context else totals + returns
      result = terminal_round_result(terminal)
      summary["rounds"] += 1
      if not result["any_paradox"]:
        continue
      summary["failure_rounds"] += 1
      warning = find_warning_decision(
          decisions,
          first_paradox_index,
          allowed_phases,
          args,
      )
      if warning is None:
        summary["no_warning_decision"] += 1
        continue
      summary["warning_decisions"] += 1
      evaluated, _avoiding, alternatives = evaluate_decision(
          model,
          warning,
          value_scale,
          device,
          args,
          rollout_seed=round_seed,
      )
      baseline = evaluated["baseline"]
      baseline_record = {
          "any_paradox": bool(baseline.get("any_paradox", False)),
          "forced_paradox": bool(baseline.get("forced_paradox", False)),
          "actor_paradox": bool(baseline.get("actor_paradox", False)),
          "first_new_paradox_trigger": baseline.get("first_new_paradox_trigger"),
          "acting_score_delta": 0.0,
          "total_score_delta": 0.0,
      }
      improving = [
          candidate
          for candidate in alternatives
          if (
              float(candidate.get("acting_score_delta", 0.0)) >=
              float(args.min_score_delta)
              and candidate_improves(candidate, baseline_record, args)
          )
      ]
      if not improving:
        summary["no_improving_candidate"] += 1
        continue
      best = min(improving, key=safety_key)
      dense_metadata = None
      if bool(args.dense_action_labels):
        row, dense_metadata = dense_label_row_for_decision(
            warning,
            baseline,
            alternatives,
            args,
            value_scale,
        )
        if row is None:
          summary["dense_row_failed"] += 1
          continue
      else:
        row = replay_row_for_warning(warning, best, baseline_record, args)
      rows.append(row)
      summary["rows"] += 1
      if bool(args.dense_action_labels):
        summary["dense_rows"] += 1
        action_value_mask = row[10]
        summary["dense_labeled_actions"] += int(np.sum(action_value_mask))
        if dense_metadata is not None:
          if bool(dense_metadata.get("policy_target_changed", False)):
            summary["dense_policy_target_changed"] += 1
          label_safety = dense_metadata.get("label_best_safety_tuple") or []
          if label_safety:
            if not any(bool(item) for item in label_safety):
              summary["dense_label_best_survives"] += 1
            else:
              summary["dense_label_best_paradoxes"] += 1
      if not bool(best.get("any_paradox", False)):
        summary["best_survives_round"] += 1
      if bool(best.get("forced_paradox", False)):
        summary["best_forced_paradox"] += 1
      summary[f"warning_trick_{int(warning.get('trick_number', -1))}"] += 1
      summary[f"warning_legal_{len(warning.get('legal', []))}"] += 1
      summary[f"trigger_trick_{trigger_trick(baseline_record)}"] += 1
      metadata_row = {
          "match_index": int(match_idx),
          "round_index": int(round_idx),
          "start_player": int(start_player),
          "decision_index": int(warning["index"]),
          "player": int(warning["player"]),
          "trick_number": int(warning["trick_number"]),
          "led_color": getattr(warning["state"], "_led_color", None),
          "legal_count": int(len(warning["legal"])),
          "selected_action": int(warning["action"]),
          "selected_action_string": str(warning["action_string"]),
          "best_action": int(best["action"]),
          "best_action_string": action_string(
              warning["state"], int(warning["player"]), int(best["action"])
          ),
          "baseline_safety_key": list(safety_key(baseline_record)),
          "best_safety_key": list(safety_key(best)),
          "best_any_paradox": bool(best.get("any_paradox", False)),
          "best_actor_paradox": bool(best.get("actor_paradox", False)),
          "best_forced_paradox": bool(best.get("forced_paradox", False)),
          "best_first_new_paradox_trigger": best.get("first_new_paradox_trigger"),
      }
      if dense_metadata is not None:
        metadata_row["dense"] = dense_metadata
      metadata_rows.append(metadata_row)
      if (
          int(args.progress_interval) > 0
          and summary["warning_decisions"] % int(args.progress_interval) == 0
      ):
        print(json.dumps({
            "iteration": "low_exit_progress",
            "matches": int(match_idx + 1),
            "rounds": int(summary["rounds"]),
            "failure_rounds": int(summary["failure_rounds"]),
            "warning_decisions": int(summary["warning_decisions"]),
            "rows": int(len(rows)),
        }), flush=True)

  metadata = {
      "source": "quantum_cat_low_exit_warning_miner",
      "checkpoint": args.checkpoint,
      "players": int(args.players),
      "matches_requested": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "bot_mode": str(args.bot_mode),
      "decision_phases": str(args.decision_phases),
      "min_trick_number": int(args.min_trick_number),
      "max_warning_legal_count": int(args.max_warning_legal_count),
      "lookback_decisions": int(args.lookback_decisions),
      "pair_objective": str(args.pair_objective),
      "dense_action_labels": bool(args.dense_action_labels),
      "dense_target_temperature": float(args.dense_target_temperature),
      "dense_action_value_objective": str(args.dense_action_value_objective),
      "summary": dict(summary),
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_replay(rows, out_path, metadata=metadata)
  out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(
      json.dumps({**metadata, "rows": metadata_rows}, indent=2) + "\n"
  )
  summary_out = Path(args.summary_out) if args.summary_out else out_path.with_suffix(
      out_path.suffix + ".summary.json"
  )
  summary_out.write_text(json.dumps({
      **metadata,
      "out": str(out_path),
      "rows": int(len(rows)),
  }, indent=2) + "\n")
  print(json.dumps({
      "iteration": "low_exit_done",
      "out": str(out_path),
      "rows": int(len(rows)),
      "summary": dict(summary),
  }, indent=2), flush=True)


if __name__ == "__main__":
  main()
