#!/usr/bin/env python3
"""Mine earliest upstream decisions that causally avoid or delay forced paradox.

This is a narrower companion to quantum_cat_failure_miner.py. For each failed
round, it scans the pre-paradox decision window from earliest to latest and
keeps only the first decision where the actually chosen action still leads to a
forced paradox under same-policy continuation, but at least one legal
alternative is better. In strict mode, better means the alternative survives the
round. In relaxed mode, survival still wins, but an alternative that delays or
softens the forced exit is accepted as a denser causal signal.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyspiel
import torch

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    ACTION_FEATURE_PREFIX_SIZE,
    APPENDED_ACTION_FEATURE_INDEX,
    action_feature_matrix,
    save_replay,
)
from quantum_cat_failure_miner import (
    action_string,
    evaluate_decision,
    parse_phase_filter,
    play_round_with_history,
    reached_mining_cap,
    rollout_from_action,
    should_evaluate_decision,
    terminal_round_result,
)
from quantum_cat_full_match_elo import load_neural


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=20)
  parser.add_argument("--seed", type=int, default=20284001)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--lookback-decisions", type=int, default=18)
  parser.add_argument("--max-failure-rounds", type=int, default=40)
  parser.add_argument("--max-decisions", type=int, default=0)
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--min-score-delta", type=float, default=-99.0)
  parser.add_argument("--out", required=True)
  parser.add_argument("--replay-out", required=True)
  parser.add_argument("--metadata-out", default="")
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument(
      "--bot-mode",
      choices=("policy", "q_policy", "liveness_policy", "q_liveness_policy"),
      default="liveness_policy",
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
  parser.add_argument("--safe-action-value-target", type=float, default=0.9)
  parser.add_argument("--risky-action-value-target", type=float, default=-0.9)
  parser.add_argument(
      "--pair-objective",
      choices=("survive", "survive_or_delay"),
      default="survive",
      help=(
          "survive keeps only alternatives with no paradox. survive_or_delay "
          "also accepts alternatives that make the first future paradox less "
          "bad: non-forced, non-actor, later trigger, or better score."
      ),
  )
  parser.add_argument(
      "--repeat-rollouts-per-action",
      type=int,
      default=1,
      help=(
          "Validate accepted base-vs-candidate contrasts across this many "
          "common-seed continuation rollouts. 1 preserves the historical "
          "single-rollout behavior."
      ),
  )
  parser.add_argument(
      "--min-repeat-win-rate",
      type=float,
      default=1.0,
      help="Minimum fraction of repeat rollouts where the candidate beats base.",
  )
  parser.add_argument(
      "--min-repeat-baseline-forced-rate",
      type=float,
      default=1.0,
      help=(
          "Minimum fraction of repeat rollouts where the base action reaches "
          "a forced paradox. Lower this only for relaxed diagnostics."
      ),
  )
  parser.add_argument(
      "--max-repeat-candidate-any-paradox-rate",
      type=float,
      default=0.0,
      help=(
          "Maximum candidate any-paradox rate across repeat rollouts. The "
          "default enforces pure survival labels."
      ),
  )
  parser.add_argument(
      "--min-repeat-acting-score-delta",
      type=float,
      default=-99.0,
      help="Minimum mean acting-score delta for the candidate versus base.",
  )
  parser.add_argument(
      "--improved-action-risk-target",
      type=float,
      default=0.35,
      help=(
          "Action-risk target for relaxed alternatives that improve the "
          "baseline but still end in a paradox."
      ),
  )
  parser.add_argument(
      "--selected-action-filter",
      choices=("all", "follow_led", "token_preserving_follow_led"),
      default="all",
      help=(
          "Optional filter for the actually selected upstream action. "
          "token_preserving_follow_led targets the subtle failure class where "
          "the policy followed the lane and did not burn the led token, but "
          "the continuation still collapses later."
      ),
  )
  parser.add_argument(
      "--safe-action-filter",
      choices=("all", "follow_led", "follow_or_red_switch", "red_switch",
               "red_win_switch"),
      default="all",
      help=(
          "Optional filter for accepted improving alternatives. "
          "follow_or_red_switch focuses labels on either preserving the led "
          "lane or taking a responsible red/trump switch."
      ),
  )
  parser.add_argument(
      "--extra-legal-negative-filter",
      choices=("none", "lane_burn_non_follow_or_red"),
      default="none",
      help=(
          "Also label selected non-pair legal actions as bad. "
          "lane_burn_non_follow_or_red marks off-led token/lane burns that "
          "neither follow led color nor switch to red."
      ),
  )
  parser.add_argument("--extra-legal-negative-risk-target", type=float, default=1.0)
  parser.add_argument("--extra-legal-negative-value-target", type=float, default=-0.9)
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
  """Lower is safer, with score only breaking comparable safety outcomes."""
  return (
      int(bool(record.get("any_paradox", False))),
      int(bool(record.get("forced_paradox", False))),
      int(bool(record.get("actor_paradox", False))),
      -trigger_trick(record),
      -trigger_legal_count(record),
      -float(record.get("acting_score_delta", 0.0)),
      -float(record.get("total_score_delta", 0.0)),
  )


def candidate_improves_baseline(candidate, baseline_record, args):
  if safety_key(candidate) >= safety_key(baseline_record):
    return False
  if str(args.pair_objective) == "survive":
    return not bool(candidate.get("any_paradox", False))
  return True


def improving_candidates(evaluated, alternatives, features, args):
  baseline = evaluated["baseline"]
  baseline_record = {
      "any_paradox": bool(baseline.get("any_paradox", False)),
      "forced_paradox": bool(baseline.get("forced_paradox", False)),
      "actor_paradox": bool(baseline.get("actor_paradox", False)),
      "first_new_paradox_trigger": baseline.get("first_new_paradox_trigger"),
      "acting_score_delta": 0.0,
      "total_score_delta": 0.0,
  }
  candidates = [
      item for item in alternatives
      if (
          float(item.get("acting_score_delta", 0.0)) >= float(args.min_score_delta)
          and action_matches_filter(
              features,
              int(item.get("action")),
              str(args.safe_action_filter),
          )
          and candidate_improves_baseline(item, baseline_record, args)
      )
  ]
  return sorted(candidates, key=safety_key)


def _record_from_rollout(result, player, baseline=None, action=None, state=None):
  if baseline is None:
    acting_score_delta = 0.0
    total_score_delta = 0.0
  else:
    acting_score_delta = float(result["raw_scores"][player] - baseline["raw_scores"][player])
    total_score_delta = float(sum(result["raw_scores"]) - sum(baseline["raw_scores"]))
  record = {
      "any_paradox": bool(result["any_paradox"]),
      "actor_paradox": bool(result["paradoxed"][player]),
      "forced_paradox": bool(result.get("forced_paradox", False)),
      "first_new_paradox_trigger": result.get("first_new_paradox_trigger"),
      "paradoxed": result["paradoxed"],
      "raw_scores": result["raw_scores"],
      "acting_score_delta": acting_score_delta,
      "total_score_delta": total_score_delta,
  }
  if action is not None:
    record["action"] = int(action)
    if state is not None:
      record["action_string"] = action_string(state, player, int(action))
  return record


def repeat_validate_candidate(
    model,
    decision,
    candidate,
    value_scale,
    device,
    args,
    base_seed,
):
  rollouts = max(1, int(getattr(args, "repeat_rollouts_per_action", 1)))
  player = int(decision["player"])
  selected_action = int(decision["action"])
  candidate_action = int(candidate["action"])
  wins = 0
  baseline_forced = 0
  candidate_any = 0
  candidate_forced = 0
  candidate_actor = 0
  acting_deltas = []
  total_deltas = []
  samples = []
  for rollout_idx in range(rollouts):
    seed = (
        int(base_seed)
        + int(decision["index"]) * 1009
        + rollout_idx * 9176
        + 7919
    )
    baseline = rollout_from_action(
        model,
        decision["state"],
        selected_action,
        value_scale,
        device,
        seed,
        args,
    )
    candidate_result = rollout_from_action(
        model,
        decision["state"],
        candidate_action,
        value_scale,
        device,
        seed,
        args,
    )
    baseline_record = _record_from_rollout(baseline, player)
    candidate_record = _record_from_rollout(
        candidate_result,
        player,
        baseline=baseline,
        action=candidate_action,
        state=decision["state"],
    )
    win = candidate_improves_baseline(candidate_record, baseline_record, args)
    wins += int(bool(win))
    baseline_forced += int(bool(baseline_record["forced_paradox"]))
    candidate_any += int(bool(candidate_record["any_paradox"]))
    candidate_forced += int(bool(candidate_record["forced_paradox"]))
    candidate_actor += int(bool(candidate_record["actor_paradox"]))
    acting_deltas.append(float(candidate_record["acting_score_delta"]))
    total_deltas.append(float(candidate_record["total_score_delta"]))
    if len(samples) < 4:
      samples.append({
          "seed": int(seed),
          "win": bool(win),
          "baseline_any_paradox": bool(baseline_record["any_paradox"]),
          "baseline_forced_paradox": bool(baseline_record["forced_paradox"]),
          "baseline_actor_paradox": bool(baseline_record["actor_paradox"]),
          "candidate_any_paradox": bool(candidate_record["any_paradox"]),
          "candidate_forced_paradox": bool(candidate_record["forced_paradox"]),
          "candidate_actor_paradox": bool(candidate_record["actor_paradox"]),
          "acting_score_delta": float(candidate_record["acting_score_delta"]),
          "total_score_delta": float(candidate_record["total_score_delta"]),
          "baseline_trigger": baseline_record.get("first_new_paradox_trigger"),
          "candidate_trigger": candidate_record.get("first_new_paradox_trigger"),
      })
  summary = {
      "rollouts": int(rollouts),
      "wins": int(wins),
      "win_rate": float(wins / rollouts),
      "baseline_forced_rate": float(baseline_forced / rollouts),
      "candidate_any_paradox_rate": float(candidate_any / rollouts),
      "candidate_forced_paradox_rate": float(candidate_forced / rollouts),
      "candidate_actor_paradox_rate": float(candidate_actor / rollouts),
      "mean_acting_score_delta": float(np.mean(acting_deltas)),
      "mean_total_score_delta": float(np.mean(total_deltas)),
      "samples": samples,
  }
  passed = (
      float(summary["win_rate"]) >= float(args.min_repeat_win_rate)
      and float(summary["baseline_forced_rate"])
      >= float(args.min_repeat_baseline_forced_rate)
      and float(summary["candidate_any_paradox_rate"])
      <= float(args.max_repeat_candidate_any_paradox_rate)
      and float(summary["mean_acting_score_delta"])
      >= float(args.min_repeat_acting_score_delta)
  )
  return passed, summary


def _feature_value(features, action, index, default=0.0):
  if index is None or index < 0 or index >= features.shape[1]:
    return default
  return float(features[int(action), int(index)])


def action_flags(features, action):
  action = int(action)
  newly_loses_led = _feature_value(
      features,
      action,
      APPENDED_ACTION_FEATURE_INDEX.get("token_loss_newly_loses_led"),
  ) > 0.5
  token_loss = (
      _feature_value(features, action, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX)
      > 0.5
      or newly_loses_led
  )
  is_red = _feature_value(features, action, ACTION_FEATURE_IS_RED_INDEX) > 0.5
  would_win = _feature_value(
      features,
      action,
      ACTION_FEATURE_PREFIX_SIZE + 19,
  ) > 0.5
  return {
      "follows_led": (
          _feature_value(features, action, ACTION_FEATURE_FOLLOWS_LED_INDEX)
          > 0.5
      ),
      "token_loss": token_loss,
      "newly_loses_led": newly_loses_led,
      "is_red": is_red,
      "red_switch": bool(is_red and token_loss),
      "red_win_switch": bool(is_red and token_loss and would_win),
      "would_win": would_win,
  }


def action_matches_filter(features, action, action_filter):
  flags = action_flags(features, action)
  if action_filter == "all":
    return True
  if action_filter == "follow_led":
    return bool(flags["follows_led"])
  if action_filter == "token_preserving_follow_led":
    return bool(flags["follows_led"]) and not bool(flags["token_loss"])
  if action_filter == "follow_or_red_switch":
    return bool(flags["follows_led"]) or bool(flags["red_switch"])
  if action_filter == "red_switch":
    return bool(flags["red_switch"])
  if action_filter == "red_win_switch":
    return bool(flags["red_win_switch"])
  raise ValueError(f"Unknown action filter: {action_filter}")


def action_matches_extra_negative_filter(features, action, action_filter):
  flags = action_flags(features, action)
  if action_filter == "none":
    return False
  if action_filter == "lane_burn_non_follow_or_red":
    return (
        bool(flags["token_loss"])
        and not bool(flags["follows_led"])
        and not bool(flags["red_switch"])
    )
  raise ValueError(f"Unknown extra negative filter: {action_filter}")


def extra_legal_negative_actions(decision, features, safe_action, args):
  negative_filter = str(getattr(args, "extra_legal_negative_filter", "none"))
  if negative_filter == "none":
    return []
  selected_action = int(decision["action"])
  safe_action = int(safe_action)
  actions = []
  for action in decision["legal"]:
    action = int(action)
    if action in (selected_action, safe_action):
      continue
    if action_matches_extra_negative_filter(features, action, negative_filter):
      actions.append(action)
  return actions


def best_improving_candidate(evaluated, alternatives, features, args):
  candidates = improving_candidates(evaluated, alternatives, features, args)
  if not candidates:
    return None
  return candidates[0]


def replay_row_for_pair(decision, safe_candidate, args,
                        extra_negative_actions=None):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  selected_action = int(decision["action"])
  safe_action = int(safe_candidate["action"])

  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[list(decision["legal"])] = 1.0
  policy_target = np.zeros(num_actions, dtype=np.float32)
  policy_target[safe_action] = 1.0

  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  action_paradox_targets[selected_action] = 1.0
  action_paradox_targets[safe_action] = (
      0.0 if not bool(safe_candidate.get("any_paradox", False))
      else float(args.improved_action_risk_target)
  )
  action_paradox_mask[selected_action] = 1.0
  action_paradox_mask[safe_action] = 1.0

  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_value_targets[selected_action] = float(args.risky_action_value_target)
  action_value_targets[safe_action] = float(args.safe_action_value_target)
  action_value_mask[selected_action] = 1.0
  action_value_mask[safe_action] = 1.0

  for action in extra_negative_actions or []:
    action = int(action)
    action_paradox_targets[action] = float(args.extra_legal_negative_risk_target)
    action_paradox_mask[action] = 1.0
    action_value_targets[action] = float(args.extra_legal_negative_value_target)
    action_value_mask[action] = 1.0

  return (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      policy_target,
      np.zeros(int(args.players), dtype=np.float32),
      np.zeros(int(args.players), dtype=np.float32),
      safe_action,
      player,
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      action_feature_matrix(state, player, num_actions),
  )


def pair_metadata(decision, evaluated, safe_candidate, features, match_idx,
                  round_idx, start_player, extra_negative_actions=None):
  selected_action = int(decision["action"])
  safe_action = int(safe_candidate["action"])
  selected_flags = action_flags(features, selected_action)
  safe_flags = action_flags(features, safe_action)
  extra_negative_rows = []
  for action in extra_negative_actions or []:
    extra_negative_rows.append({
        "action": int(action),
        "action_string": action_string(
            decision["state"], int(decision["player"]), int(action)
        ),
        "flags": action_flags(features, int(action)),
    })
  return {
      "match_index": int(match_idx),
      "round_index": int(round_idx),
      "start_player": int(start_player),
      "decision_index": int(decision["index"]),
      "phase": str(decision["phase"]),
      "player": int(decision["player"]),
      "trick_number": int(decision["trick_number"]),
      "legal_count": int(len(decision["legal"])),
      "policy_best_action": selected_action,
      "policy_best_action_string": str(decision["action_string"]),
      "safe_action": safe_action,
      "safe_action_string": str(safe_candidate["action_string"]),
      "extra_legal_negative_count": int(len(extra_negative_rows)),
      "extra_legal_negative_actions": extra_negative_rows,
      "selected_follows_led": bool(selected_flags["follows_led"]),
      "selected_token_loss": bool(selected_flags["token_loss"]),
      "selected_red_switch": bool(selected_flags["red_switch"]),
      "selected_red_win_switch": bool(selected_flags["red_win_switch"]),
      "safe_follows_led": bool(safe_flags["follows_led"]),
      "safe_token_loss": bool(safe_flags["token_loss"]),
      "safe_red_switch": bool(safe_flags["red_switch"]),
      "safe_red_win_switch": bool(safe_flags["red_win_switch"]),
      "safe_would_win": bool(safe_flags["would_win"]),
      "safe_any_paradox": bool(safe_candidate.get("any_paradox", False)),
      "safe_forced_paradox": bool(safe_candidate.get("forced_paradox", False)),
      "safe_actor_paradox": bool(safe_candidate.get("actor_paradox", False)),
      "safe_acting_score_delta": float(safe_candidate["acting_score_delta"]),
      "safe_total_score_delta": float(safe_candidate["total_score_delta"]),
      "baseline_forced_paradox": bool(evaluated["baseline"].get("forced_paradox")),
      "baseline_trigger": evaluated["baseline"].get("first_new_paradox_trigger"),
      "safe_trigger": safe_candidate.get("first_new_paradox_trigger"),
      "candidate_count": int(max(0, len(decision["legal"]) - 1)),
      "policy_action_probability": float(decision["policy_prob"]),
  }


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
  summary = Counter()
  rows = []
  row_group_ids = []
  metadata_rows = []
  failure_records = []

  for match_idx in range(int(args.matches)):
    if reached_mining_cap(args, summary):
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if reached_mining_cap(args, summary):
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
      terminal, decisions, first_new_paradox = play_round_with_history(
          model,
          game,
          seed=round_seed,
          value_scale=value_scale,
          device=device,
          match_totals=prior_totals if args.match_context else None,
          round_index=round_idx,
          args=args,
      )
      result = terminal_round_result(terminal)
      returns = np.array(terminal.returns(), dtype=np.float32)
      totals = returns if args.match_context else totals + returns
      summary["rounds"] += 1
      if result["any_paradox"]:
        summary["paradox_rounds"] += 1
      if not result["any_paradox"] or first_new_paradox is None:
        continue
      if summary["failure_rounds_mined"] >= int(args.max_failure_rounds):
        continue
      summary["failure_rounds_mined"] += 1

      start = max(0, first_new_paradox - int(args.lookback_decisions) + 1)
      window = decisions[start:first_new_paradox + 1]
      failure_record = {
          "match_index": int(match_idx),
          "round_index": int(round_idx),
          "start_player": int(start_player),
          "terminal": result,
          "first_new_paradox_decision": int(first_new_paradox),
          "selected_pair": None,
          "evaluated_decisions": 0,
      }
      for decision in window:
        if reached_mining_cap(args, summary):
          break
        if len(decision["legal"]) <= 1:
          continue
        if not should_evaluate_decision(decision, allowed_phases):
          continue
        num_actions = int(decision["state"].num_distinct_actions())
        features = action_feature_matrix(
            decision["state"],
            int(decision["player"]),
            num_actions,
        )
        if not action_matches_filter(
            features,
            int(decision["action"]),
            str(args.selected_action_filter),
        ):
          summary["decisions_rejected_by_selected_filter"] += 1
          summary[
              f"decisions_rejected_by_selected_filter_"
              f"{args.selected_action_filter}"
          ] += 1
          continue
        summary["decisions_evaluated"] += 1
        summary[f"decisions_evaluated_phase_{decision['phase']}"] += 1
        failure_record["evaluated_decisions"] += 1
        if (
            int(args.progress_interval) > 0
            and summary["decisions_evaluated"] % int(args.progress_interval) == 0
        ):
          print(json.dumps({
              "iteration": "causal_pairwise_progress",
              "rounds": int(summary["rounds"]),
              "failure_rounds_mined": int(summary["failure_rounds_mined"]),
              "decisions_evaluated": int(summary["decisions_evaluated"]),
              "pairs": int(summary["causal_pairs"]),
              "phase": str(decision["phase"]),
              "legal_count": int(len(decision["legal"])),
          }), flush=True)
        evaluated, _avoiding, alternatives = evaluate_decision(
            model,
            decision,
            value_scale,
            device,
            args,
            rollout_seed=round_seed,
        )
        baseline_forced = bool(evaluated["baseline"].get("forced_paradox"))
        if not baseline_forced:
          continue
        candidate_queue = improving_candidates(
            evaluated,
            alternatives,
            features,
            args,
        )
        if not candidate_queue:
          summary["decisions_without_safe_filtered_candidate"] += 1
          continue
        safe_candidate = None
        repeat_validation = None
        for candidate in candidate_queue:
          if int(args.repeat_rollouts_per_action) <= 1:
            safe_candidate = candidate
            repeat_validation = {
                "rollouts": 1,
                "wins": 1,
                "win_rate": 1.0,
                "baseline_forced_rate": 1.0,
                "candidate_any_paradox_rate": (
                    1.0 if bool(candidate.get("any_paradox", False)) else 0.0
                ),
                "candidate_forced_paradox_rate": (
                    1.0 if bool(candidate.get("forced_paradox", False)) else 0.0
                ),
                "candidate_actor_paradox_rate": (
                    1.0 if bool(candidate.get("actor_paradox", False)) else 0.0
                ),
                "mean_acting_score_delta": float(
                    candidate.get("acting_score_delta", 0.0)
                ),
                "mean_total_score_delta": float(
                    candidate.get("total_score_delta", 0.0)
                ),
                "samples": [],
            }
            break
          summary["repeat_candidates_validated"] += 1
          passed, validation = repeat_validate_candidate(
              model,
              decision,
              candidate,
              value_scale,
              device,
              args,
              base_seed=round_seed,
          )
          if passed:
            safe_candidate = dict(candidate)
            repeat_validation = validation
            break
          summary["repeat_candidates_rejected"] += 1
        if safe_candidate is None:
          summary["decisions_without_repeat_validated_candidate"] += 1
          continue
        safe_candidate["repeat_validation"] = repeat_validation
        extra_negatives = extra_legal_negative_actions(
            decision,
            features,
            int(safe_candidate["action"]),
            args,
        )
        rows.append(replay_row_for_pair(
            decision,
            safe_candidate,
            args,
            extra_negative_actions=extra_negatives,
        ))
        meta = pair_metadata(
            decision,
            evaluated,
            safe_candidate,
            features,
            match_idx,
            round_idx,
            start_player,
            extra_negative_actions=extra_negatives,
        )
        meta["repeat_validation"] = repeat_validation
        meta["source_index"] = int(len(metadata_rows))
        metadata_rows.append(meta)
        row_group_ids.append(
            f"m{int(match_idx):06d}_r{int(round_idx):02d}"
        )
        failure_record["selected_pair"] = meta
        summary["causal_pairs"] += 1
        summary[f"causal_pairs_phase_{decision['phase']}"] += 1
        selected_flags = action_flags(features, int(decision["action"]))
        safe_flags = action_flags(features, int(safe_candidate["action"]))
        for prefix, flags in (("selected", selected_flags), ("safe", safe_flags)):
          for key, value in flags.items():
            if bool(value):
              summary[f"causal_pairs_{prefix}_{key}"] += 1
        summary["causal_pairs_extra_legal_negatives"] += int(len(extra_negatives))
        if extra_negatives:
          summary["causal_pairs_with_extra_legal_negatives"] += 1
        for action in extra_negatives:
          negative_flags = action_flags(features, int(action))
          for key, value in negative_flags.items():
            if bool(value):
              summary[f"causal_pairs_extra_negative_{key}"] += 1
        if bool(safe_candidate.get("any_paradox", False)):
          summary["causal_pairs_relaxed"] += 1
        else:
          summary["causal_pairs_survived"] += 1
        if repeat_validation is not None:
          summary["repeat_validated_pairs"] += 1
          summary["repeat_validation_rollouts"] += int(
              repeat_validation.get("rollouts", 0)
          )
        break
      if failure_record["selected_pair"] is None:
        summary["failure_rounds_without_pair"] += 1
      failure_records.append(failure_record)

  output = {
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "lookback_decisions": int(args.lookback_decisions),
      "repeat_rollouts_per_action": int(args.repeat_rollouts_per_action),
      "min_repeat_win_rate": float(args.min_repeat_win_rate),
      "min_repeat_baseline_forced_rate": float(
          args.min_repeat_baseline_forced_rate
      ),
      "max_repeat_candidate_any_paradox_rate": float(
          args.max_repeat_candidate_any_paradox_rate
      ),
      "min_repeat_acting_score_delta": float(args.min_repeat_acting_score_delta),
      "extra_legal_negative_filter": str(args.extra_legal_negative_filter),
      "extra_legal_negative_risk_target": float(
          args.extra_legal_negative_risk_target
      ),
      "extra_legal_negative_value_target": float(
          args.extra_legal_negative_value_target
      ),
      "summary": dict(summary),
      "replay_rows": len(rows),
      "failure_rounds": failure_records,
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(output, indent=2) + "\n")
  save_replay(
      rows,
      args.replay_out,
      metadata={
          "source": "quantum_cat_causal_pairwise_miner",
          "checkpoint": str(args.checkpoint),
          "summary": dict(summary),
          "row_group_ids": row_group_ids,
          "row_group_count": int(len(set(row_group_ids))),
          "row_group_id_kind": "match_round",
          "pair_objective": str(args.pair_objective),
          "repeat_rollouts_per_action": int(args.repeat_rollouts_per_action),
          "min_repeat_win_rate": float(args.min_repeat_win_rate),
          "min_repeat_baseline_forced_rate": float(
              args.min_repeat_baseline_forced_rate
          ),
          "max_repeat_candidate_any_paradox_rate": float(
              args.max_repeat_candidate_any_paradox_rate
          ),
          "min_repeat_acting_score_delta": float(args.min_repeat_acting_score_delta),
          "safe_action_value_target": float(args.safe_action_value_target),
          "risky_action_value_target": float(args.risky_action_value_target),
          "improved_action_risk_target": float(args.improved_action_risk_target),
          "extra_legal_negative_filter": str(args.extra_legal_negative_filter),
          "extra_legal_negative_risk_target": float(
              args.extra_legal_negative_risk_target
          ),
          "extra_legal_negative_value_target": float(
              args.extra_legal_negative_value_target
          ),
      },
  )
  metadata_path = (
      Path(args.metadata_out)
      if args.metadata_out
      else Path(args.replay_out).with_suffix(Path(args.replay_out).suffix + ".metadata.json")
  )
  metadata_path.parent.mkdir(parents=True, exist_ok=True)
  metadata_path.write_text(json.dumps({
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "seed": int(args.seed),
      "pair_objective": str(args.pair_objective),
      "repeat_rollouts_per_action": int(args.repeat_rollouts_per_action),
      "min_repeat_win_rate": float(args.min_repeat_win_rate),
      "min_repeat_baseline_forced_rate": float(
          args.min_repeat_baseline_forced_rate
      ),
      "max_repeat_candidate_any_paradox_rate": float(
          args.max_repeat_candidate_any_paradox_rate
      ),
      "min_repeat_acting_score_delta": float(args.min_repeat_acting_score_delta),
      "extra_legal_negative_filter": str(args.extra_legal_negative_filter),
      "extra_legal_negative_risk_target": float(
          args.extra_legal_negative_risk_target
      ),
      "extra_legal_negative_value_target": float(
          args.extra_legal_negative_value_target
      ),
      "rows": metadata_rows,
  }, indent=2) + "\n")
  print(json.dumps({**output, "failure_rounds": len(failure_records)}, indent=2))


if __name__ == "__main__":
  main()
