#!/usr/bin/env python3
"""Audit counterfactual action labels before training or reranking.

This script collects real champion-policy decision states, labels candidate
actions with the existing counterfactual machinery, then independently
re-evaluates label-best vs champion-policy-best on fresh common rollout seeds.
It is a go/no-go diagnostic for Q-label quality, not a deployment bot.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import (
    TACTICAL_POLICY_BUCKET_NAMES,
    action_feature_matrix,
    counterfactual_action_value_targets,
    counterfactual_belief_states,
    load_model_payload,
    make_game,
    match_outcome_scores,
    model_policy_value,
    phase_name_for_state,
    play_rollout_to_terminal,
    rollout_match_score_after_action,
    rollout_paradox_after_action,
    rollout_score_after_action,
    save_replay,
    sampled_counterfactual_legal_actions,
    policy_target_bucket_masks,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--states", type=int, default=24)
  parser.add_argument("--max-matches", type=int, default=8)
  parser.add_argument(
      "--max-states-per-match",
      type=int,
      default=0,
      help="When >0, collect at most this many audited states per full match.",
  )
  parser.add_argument(
      "--state-selection",
      choices=("first", "high_entropy", "hard"),
      default="first",
      help=(
          "How to select states from champion self-play. 'first' preserves "
          "the original traversal order. 'high_entropy' collects uncertain "
          "policy states. 'hard' also prioritizes states whose acting player "
          "later has low match score or paradoxes."
      ),
  )
  parser.add_argument(
      "--state-pool-per-match",
      type=int,
      default=0,
      help=(
          "Optional cap on candidate states retained per match before hard "
          "selection. 0 keeps every matching candidate in the match."
      ),
  )
  parser.add_argument(
      "--state-min-legal-count",
      type=int,
      default=0,
      help="When >0, only collect candidate states with at least this many legal actions.",
  )
  parser.add_argument(
      "--state-max-legal-count",
      type=int,
      default=0,
      help="When >0, only collect candidate states with at most this many legal actions.",
  )
  parser.add_argument(
      "--state-min-trick-number",
      type=int,
      default=-1,
      help="When >=0, only collect candidate states at or after this trick number.",
  )
  parser.add_argument(
      "--state-max-trick-number",
      type=int,
      default=-1,
      help="When >=0, only collect candidate states at or before this trick number.",
  )
  parser.add_argument(
      "--state-min-match-round",
      type=int,
      default=-1,
      help="When >=0, only collect candidate states at or after this match round.",
  )
  parser.add_argument(
      "--state-max-match-round",
      type=int,
      default=-1,
      help="When >=0, only collect candidate states at or before this match round.",
  )
  parser.add_argument("--hard-entropy-weight", type=float, default=1.0)
  parser.add_argument("--hard-loss-weight", type=float, default=0.25)
  parser.add_argument("--hard-paradox-weight", type=float, default=0.75)
  parser.add_argument("--seed", type=int, default=20260622)
  parser.add_argument("--phases", default="prediction,play")
  parser.add_argument("--label-rollouts", type=int, default=1)
  parser.add_argument("--audit-rollouts", type=int, default=4)
  parser.add_argument(
      "--stability-audit-passes",
      type=int,
      default=1,
      help=(
          "Repeat independent audits with fresh seeds before trusting a "
          "calibration positive. 1 preserves the historical single-audit path."
      ),
  )
  parser.add_argument(
      "--stability-seed-stride",
      type=int,
      default=1299709,
      help="Seed stride between repeated stability audit passes.",
  )
  parser.add_argument(
      "--stability-min-positive-sign-rate",
      type=float,
      default=1.0,
      help=(
          "Minimum fraction of repeated audits whose label-vs-policy margin "
          "must be positive before a POSITIVE calibration row is trusted."
      ),
  )
  parser.add_argument(
      "--stability-min-safe-risk-rate",
      type=float,
      default=1.0,
      help=(
          "Minimum fraction of repeated audits whose paradox delta is no worse "
          "than --calibration-max-positive-paradox-delta before a POSITIVE row "
          "is trusted."
      ),
  )
  parser.add_argument(
      "--unstable-positive-as-negative",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "When a row is POSITIVE by aggregate margin but fails stability, "
          "export it as a hard no-override calibration row."
      ),
  )
  parser.add_argument("--label-seed-offset", type=int, default=1000003)
  parser.add_argument("--audit-seed-offset", type=int, default=2000003)
  parser.add_argument("--belief-samples", type=int, default=1)
  parser.add_argument("--belief-candidates", type=int, default=8)
  parser.add_argument("--belief-policy-temperature", type=float, default=1.0)
  parser.add_argument("--belief-uniform-mix", type=float, default=0.15)
  parser.add_argument(
      "--belief-ref-policy-mix",
      default="model:0.65,heuristic:0.25,uniform:0.10",
  )
  parser.add_argument("--belief-ref-policy-mix-by-phase", default="")
  parser.add_argument("--label-max-actions", type=int, default=4)
  parser.add_argument("--label-top-policy", type=int, default=2)
  parser.add_argument(
      "--label-include-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument(
      "--label-feature-candidates",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "When label-max-actions caps candidates, include tactical "
          "action-feature extremes before random fill."
      ),
  )
  parser.add_argument("--min-label-margin", type=float, default=0.05)
  parser.add_argument(
      "--label-paradox-penalty",
      type=float,
      default=0.0,
      help=(
          "When >0, choose label-best by mean_score - penalty * "
          "mean_paradox instead of score-only counterfactual labels."
      ),
  )
  parser.add_argument(
      "--label-max-paradox-delta",
      type=float,
      default=None,
      help=(
          "When set, candidate label actions whose label-time mean paradox "
          "exceeds the policy action by more than this amount are rejected."
      ),
  )
  parser.add_argument("--min-audit-margin", type=float, default=0.0)
  parser.add_argument("--pass-min-score-rate", type=float, default=0.54)
  parser.add_argument("--pass-min-avg-diff", type=float, default=0.0)
  parser.add_argument("--pass-max-paradox-delta", type=float, default=0.02)
  parser.add_argument("--full-match-rollout", action="store_true")
  parser.add_argument(
      "--counterfactual-action-paradox-scope",
      choices=("acting", "any"),
      default="any",
      help=(
          "Paradox-risk target used by independent action audits. 'any' "
          "matches the homogeneous any-player paradox-rate promotion gate."
      ),
  )
  parser.add_argument("--rollout-max-plies", type=int, default=0)
  parser.add_argument("--match-context", action="store_true", default=True)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument(
      "--accepted-replay-out",
      default="",
      help=(
          "Optional replay .npz containing only labels that passed the "
          "independent audit acceptance rule."
      ),
  )
  parser.add_argument(
      "--calibration-replay-out",
      default="",
      help=(
          "Optional replay .npz containing all independently audited "
          "label-vs-policy disagreements, including negatives. A JSON sidecar "
          "with row classes and margins is written next to it."
      ),
  )
  parser.add_argument(
      "--calibration-positive-margin",
      type=float,
      default=0.5,
      help="Audit margin required for a POSITIVE calibration row.",
  )
  parser.add_argument(
      "--calibration-max-positive-paradox-delta",
      type=float,
      default=0.0,
      help=(
          "Maximum paradox delta allowed for a POSITIVE calibration row. "
          "Rows with positive score margin but worse risk become NEAR_POSITIVE."
      ),
  )
  parser.add_argument(
      "--calibration-include-policy-matches",
      action="store_true",
      help=(
          "Also export states where the capped label-best action is already "
          "Run119's policy action as no-override NEGATIVE calibration rows. "
          "These are cheap abstention examples."
      ),
  )
  parser.add_argument(
      "--calibration-target-positive-rows",
      type=int,
      default=0,
      help=(
          "When >0, stop evaluating the collected state pool after at least "
          "this many POSITIVE calibration rows have been exported, subject to "
          "--calibration-target-negative-rows also being satisfied."
      ),
  )
  parser.add_argument(
      "--calibration-target-negative-rows",
      type=int,
      default=0,
      help=(
          "When >0, stop evaluating the collected state pool after at least "
          "this many NEGATIVE calibration rows have been exported, subject to "
          "--calibration-target-positive-rows also being satisfied."
      ),
  )
  parser.add_argument(
      "--candidate-recall-audit-every",
      type=int,
      default=0,
      help=(
          "When >0, every Nth labeled state is rescored with all legal actions "
          "to estimate whether the capped candidate set includes the all-legal "
          "label-best action."
      ),
  )
  parser.add_argument(
      "--progress-interval",
      type=int,
      default=0,
      help="Print compact JSON progress every N processed states when >0.",
  )
  parser.add_argument("--out", default="az_runs/label_oracle_audit_latest.json")
  return parser.parse_args()


def model_args_for(players):
  return SimpleNamespace(
      players=players,
      arch="mlp",
      width=256,
      depth=3,
      value_scale=20.0,
      match_context=True,
  )


def audit_namespace(args, saved_args):
  return SimpleNamespace(
      players=args.players,
      states=max(1, int(args.states)),
      max_matches=max(1, int(args.max_matches)),
      max_states_per_match=max(0, int(args.max_states_per_match)),
      state_selection=args.state_selection,
      state_pool_per_match=max(0, int(args.state_pool_per_match)),
      state_min_legal_count=max(0, int(getattr(args, "state_min_legal_count", 0))),
      state_max_legal_count=max(0, int(getattr(args, "state_max_legal_count", 0))),
      state_min_trick_number=int(getattr(args, "state_min_trick_number", -1)),
      state_max_trick_number=int(getattr(args, "state_max_trick_number", -1)),
      state_min_match_round=int(getattr(args, "state_min_match_round", -1)),
      state_max_match_round=int(getattr(args, "state_max_match_round", -1)),
      hard_entropy_weight=float(args.hard_entropy_weight),
      hard_loss_weight=float(args.hard_loss_weight),
      hard_paradox_weight=float(args.hard_paradox_weight),
      seed=int(args.seed),
      match_context=bool(args.match_context),
      value_scale=float(saved_args.get("value_scale", 20.0)),
      self_play_policy_mode="policy",
      self_play_belief_samples=1,
      self_play_belief_sims=1,
      league_opponent_mode="policy",
      eval_belief_samples=max(1, int(args.belief_samples)),
      counterfactual_action_value_rollouts=max(1, int(args.label_rollouts)),
      counterfactual_action_value_advantage=False,
      counterfactual_action_value_min_spread=0.0,
      counterfactual_action_value_max_stderr=0.0,
      counterfactual_action_value_audit_rollouts=0,
      counterfactual_action_value_audit_min_margin=0.0,
      counterfactual_action_value_confidence_weight=False,
      stability_min_positive_sign_rate=float(
          getattr(args, "stability_min_positive_sign_rate", 1.0)
      ),
      stability_min_safe_risk_rate=float(
          getattr(args, "stability_min_safe_risk_rate", 1.0)
      ),
      label_paradox_penalty=max(0.0, float(args.label_paradox_penalty)),
      label_max_paradox_delta=args.label_max_paradox_delta,
      counterfactual_full_match_rollout=bool(args.full_match_rollout),
      counterfactual_action_paradox_scope=str(
          getattr(args, "counterfactual_action_paradox_scope", "any")
      ),
      full_match_training=bool(args.match_context),
      counterfactual_rollout_max_plies=max(0, int(args.rollout_max_plies)),
      counterfactual_rollout_opponent_mode="",
      counterfactual_action_label_phases=args.phases,
      counterfactual_action_max_legal=max(0, int(args.label_max_actions)),
      counterfactual_action_top_policy=max(0, int(args.label_top_policy)),
      counterfactual_action_include_bots=args.label_include_bots,
      counterfactual_action_feature_candidates=bool(args.label_feature_candidates),
      counterfactual_action_min_policy_entropy=0.0,
      counterfactual_action_max_policy_top_prob=1.0,
      counterfactual_belief_source="policy_weighted",
      counterfactual_belief_samples=max(1, int(args.belief_samples)),
      counterfactual_belief_candidates=max(
          int(args.belief_samples), int(args.belief_candidates)
      ),
      counterfactual_belief_policy_temperature=float(
          args.belief_policy_temperature
      ),
      counterfactual_belief_uniform_mix=float(args.belief_uniform_mix),
      counterfactual_belief_ref_policy_mix=args.belief_ref_policy_mix,
      counterfactual_belief_ref_policy_mix_by_phase=(
          args.belief_ref_policy_mix_by_phase
      ),
      counterfactual_belief_logprob_floor=1e-6,
  )


def action_string(state, player, action):
  try:
    return state._action_to_string(player, int(action))  # pylint:disable=protected-access
  except Exception:
    return str(action)


def set_eval_seed(seed):
  seed = int(seed)
  random.seed(seed)
  np.random.seed(seed % (2**32 - 1))


def choose_champion_action(model, args, device, state, player):
  policy, _ = model_policy_value(
      model,
      state,
      player,
      state.num_distinct_actions(),
      args.value_scale,
      device,
  )
  legal = state.legal_actions(player)
  return policy, int(max(legal, key=lambda action: policy[action]))


def policy_uncertainty_stats(policy, legal):
  legal = list(legal)
  if not legal:
    return {
        "policy_top_prob": None,
        "policy_entropy": 0.0,
        "policy_entropy_norm": 0.0,
  }
  probs = np.array([max(0.0, float(policy[action])) for action in legal],
                   dtype=np.float64)
  total = float(np.sum(probs))
  if total <= 0:
    probs = np.ones(len(legal), dtype=np.float64) / float(len(legal))
  else:
    probs = probs / total
  entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
  entropy_norm = entropy / max(1e-12, math.log(max(2, len(legal))))
  return {
      "policy_top_prob": float(np.max(probs)),
      "policy_entropy": entropy,
      "policy_entropy_norm": entropy_norm,
  }


def _state_filter_reason(state, legal, args):
  min_legal = int(getattr(args, "state_min_legal_count", 0))
  max_legal = int(getattr(args, "state_max_legal_count", 0))
  min_trick = int(getattr(args, "state_min_trick_number", -1))
  max_trick = int(getattr(args, "state_max_trick_number", -1))
  min_round = int(getattr(args, "state_min_match_round", -1))
  max_round = int(getattr(args, "state_max_match_round", -1))
  if min_legal > 0 and len(legal) < min_legal:
    return "low_legal_count"
  if max_legal > 0 and len(legal) > max_legal:
    return "high_legal_count"
  if min_trick >= 0 and int(getattr(state, "_trick_number", -1)) < min_trick:
    return "early_trick_number"
  if max_trick >= 0 and int(getattr(state, "_trick_number", -1)) > max_trick:
    return "late_trick_number"
  if min_round >= 0 and int(getattr(state, "_match_round", -1)) < min_round:
    return "early_match_round"
  if max_round >= 0 and int(getattr(state, "_match_round", -1)) > max_round:
    return "late_match_round"
  return None


def _new_collection_report(args):
  return {
      "candidate_states_seen": 0,
      "candidate_states_eligible": 0,
      "candidate_states_pooled": 0,
      "candidate_states_kept": 0,
      "skipped_low_legal_count": 0,
      "skipped_high_legal_count": 0,
      "skipped_early_trick_number": 0,
      "skipped_late_trick_number": 0,
      "skipped_early_match_round": 0,
      "skipped_late_match_round": 0,
      "state_min_legal_count": int(getattr(args, "state_min_legal_count", 0)),
      "state_max_legal_count": int(getattr(args, "state_max_legal_count", 0)),
      "state_min_trick_number": int(getattr(args, "state_min_trick_number", -1)),
      "state_max_trick_number": int(getattr(args, "state_max_trick_number", -1)),
      "state_min_match_round": int(getattr(args, "state_min_match_round", -1)),
      "state_max_match_round": int(getattr(args, "state_max_match_round", -1)),
  }


def _note_collection_skip(report, reason):
  key = f"skipped_{reason}"
  report[key] = int(report.get(key, 0)) + 1


def _parse_tactical_selection(selection):
  text = str(selection or "")
  if not text.startswith("tactical:"):
    return []
  spec = text.split(":", 1)[1]
  names = [part.strip() for part in spec.split("+") if part.strip()]
  unknown = [name for name in names if name not in TACTICAL_POLICY_BUCKET_NAMES]
  if unknown:
    raise ValueError(
        "Unknown tactical state-selection bucket(s) "
        f"{unknown}; expected one of {TACTICAL_POLICY_BUCKET_NAMES}"
    )
  return names


def _legal_action_tactical_buckets(state, player, legal):
  if not legal:
    return {}
  features = action_feature_matrix(state, player, state.num_distinct_actions())
  rows = torch.tensor(features[list(legal)], dtype=torch.float32)
  masks = policy_target_bucket_masks(rows)
  buckets = {}
  for name, mask in masks.items():
    matching = [
        int(action)
        for action, is_match in zip(legal, mask.detach().cpu().numpy())
        if bool(is_match)
    ]
    if matching:
      buckets[name] = matching
  return buckets


def _tactical_match_summary(state, player, legal, target_buckets):
  buckets = _legal_action_tactical_buckets(state, player, legal)
  target_matches = {
      name: buckets.get(name, []) for name in target_buckets
  }
  match_count = sum(len(actions) for actions in target_matches.values())
  return buckets, target_matches, match_count


def collect_states_first(model, args, device, wanted_phases):
  states = []
  state_groups = []
  matches = 0
  collection_report = _new_collection_report(args)
  while len(states) < args.states and matches < args.max_matches:
    initial_start = matches % args.players
    totals = np.zeros(args.players, dtype=np.float32)
    collected_this_match = 0
    for round_idx in range(args.players):
      round_game = make_game(args, (initial_start + round_idx) % args.players)
      state = round_game.new_initial_state()
      if args.match_context:
        state.set_match_context(totals, round_idx)
      while not state.is_terminal() and len(states) < args.states:
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(int(np.random.choice(actions, p=probs)))
          continue
        player = int(state.current_player())
        phase = phase_name_for_state(state)
        legal = state.legal_actions(player)
        skip_reason = _state_filter_reason(state, legal, args)
        cap_open = (
            args.max_states_per_match <= 0
            or collected_this_match < args.max_states_per_match
        )
        if phase in wanted_phases and cap_open:
          collection_report["candidate_states_seen"] += 1
          if skip_reason is not None:
            _note_collection_skip(collection_report, skip_reason)
          else:
            collection_report["candidate_states_eligible"] += 1
        if phase in wanted_phases and cap_open and skip_reason is None:
          states.append(state.clone())
          collection_report["candidate_states_pooled"] += 1
          collection_report["candidate_states_kept"] += 1
          state_groups.append({
              "match_index": int(matches),
              "initial_start_player": int(initial_start),
              "round_index": int(round_idx),
              "collected_index_in_match": int(collected_this_match),
              "legal_count": int(len(legal)),
          })
          collected_this_match += 1
        policy, champion_action = choose_champion_action(
            model, args, device, state, player
        )
        del policy
        state.apply_action(champion_action)
      returns = np.array(state.returns(), dtype=np.float32)
      totals = returns if args.match_context else totals + returns
      if len(states) >= args.states:
        break
    matches += 1
  args._state_collection_report = collection_report
  return states, state_groups, matches


def collect_states_hard(model, args, device, wanted_phases):
  states = []
  state_groups = []
  matches = 0
  per_match_pick = max(1, int(args.max_states_per_match or 1))
  pool_cap = int(getattr(args, "state_pool_per_match", 0))
  collection_report = _new_collection_report(args)
  while len(states) < args.states and matches < args.max_matches:
    initial_start = matches % args.players
    totals = np.zeros(args.players, dtype=np.float32)
    match_paradoxes = np.zeros(args.players, dtype=np.float32)
    candidates = []
    collected_this_match = 0
    for round_idx in range(args.players):
      round_game = make_game(args, (initial_start + round_idx) % args.players)
      state = round_game.new_initial_state()
      if args.match_context:
        state.set_match_context(totals, round_idx)
      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(int(np.random.choice(actions, p=probs)))
          continue
        player = int(state.current_player())
        phase = phase_name_for_state(state)
        policy, champion_action = choose_champion_action(
            model, args, device, state, player
        )
        if phase in wanted_phases:
          legal = state.legal_actions(player)
          collection_report["candidate_states_seen"] += 1
          skip_reason = _state_filter_reason(state, legal, args)
          if skip_reason is not None:
            _note_collection_skip(collection_report, skip_reason)
            state.apply_action(champion_action)
            continue
          collection_report["candidate_states_eligible"] += 1
          uncertainty = policy_uncertainty_stats(policy, legal)
          if pool_cap <= 0 or len(candidates) < pool_cap:
            collection_report["candidate_states_pooled"] += 1
            candidates.append({
                "state": state.clone(),
                "player": int(player),
                "phase": phase,
                "round_index": int(round_idx),
                "collected_index_in_match": int(collected_this_match),
                "legal_count": int(len(legal)),
                "champion_action": int(champion_action),
                "champion_action_string": action_string(
                    state, player, champion_action
                ),
                **uncertainty,
            })
            collected_this_match += 1
        state.apply_action(champion_action)
      raw_scores = (
          state.raw_round_scores()
          if hasattr(state, "raw_round_scores")
          else state.returns()
      )
      round_scores = np.array(raw_scores, dtype=np.float32)
      totals = totals + round_scores
      match_paradoxes += np.array(
          getattr(state, "_has_paradoxed", [False] * args.players),
          dtype=np.float32,
      )
    if not candidates:
      matches += 1
      continue
    for candidate in candidates:
      player = int(candidate["player"])
      player_score = float(totals[player])
      best_score = float(np.max(totals))
      loss_gap = max(0.0, best_score - player_score)
      paradox_count = float(match_paradoxes[player])
      entropy = float(candidate.get("policy_entropy_norm") or 0.0)
      if args.state_selection == "high_entropy":
        hard_score = entropy
      else:
        hard_score = (
            float(args.hard_entropy_weight) * entropy
            + float(args.hard_loss_weight) * loss_gap
            + float(args.hard_paradox_weight) * paradox_count
        )
      candidate["match_final_totals"] = [float(x) for x in totals]
      candidate["acting_player_final_score"] = player_score
      candidate["acting_player_loss_gap"] = float(loss_gap)
      candidate["acting_player_match_paradoxes"] = paradox_count
      candidate["hard_score"] = float(hard_score)
    candidates.sort(key=lambda item: float(item["hard_score"]), reverse=True)
    for rank, candidate in enumerate(candidates[:per_match_pick]):
      if len(states) >= args.states:
        break
      states.append(candidate["state"])
      collection_report["candidate_states_kept"] += 1
      state_groups.append({
          "match_index": int(matches),
          "initial_start_player": int(initial_start),
          "round_index": int(candidate["round_index"]),
          "collected_index_in_match": int(candidate["collected_index_in_match"]),
          "legal_count": int(candidate.get("legal_count", 0)),
          "state_selection": args.state_selection,
          "hard_rank_in_match": int(rank),
          "hard_score": float(candidate["hard_score"]),
          "policy_entropy": float(candidate["policy_entropy"]),
          "policy_entropy_norm": float(candidate["policy_entropy_norm"]),
          "policy_top_prob": candidate["policy_top_prob"],
          "champion_action": int(candidate["champion_action"]),
          "champion_action_string": candidate["champion_action_string"],
          "match_final_totals": candidate["match_final_totals"],
          "acting_player_final_score": float(candidate["acting_player_final_score"]),
          "acting_player_loss_gap": float(candidate["acting_player_loss_gap"]),
          "acting_player_match_paradoxes": float(
              candidate["acting_player_match_paradoxes"]
          ),
      })
    matches += 1
  args._state_collection_report = collection_report
  return states, state_groups, matches


def collect_states_tactical(model, args, device, wanted_phases):
  target_buckets = _parse_tactical_selection(args.state_selection)
  if not target_buckets:
    raise ValueError(f"Bad tactical state selection: {args.state_selection!r}")
  states = []
  state_groups = []
  matches = 0
  per_match_pick = max(1, int(args.max_states_per_match or 1))
  pool_cap = int(getattr(args, "state_pool_per_match", 0))
  collection_report = _new_collection_report(args)
  collection_report["tactical_target_buckets"] = list(target_buckets)
  collection_report["tactical_candidate_matches"] = {
      name: 0 for name in target_buckets
  }
  collection_report["tactical_candidate_actions"] = {
      name: 0 for name in target_buckets
  }
  while len(states) < args.states and matches < args.max_matches:
    initial_start = matches % args.players
    totals = np.zeros(args.players, dtype=np.float32)
    match_paradoxes = np.zeros(args.players, dtype=np.float32)
    candidates = []
    collected_this_match = 0
    for round_idx in range(args.players):
      round_game = make_game(args, (initial_start + round_idx) % args.players)
      state = round_game.new_initial_state()
      if args.match_context:
        state.set_match_context(totals, round_idx)
      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(int(np.random.choice(actions, p=probs)))
          continue
        player = int(state.current_player())
        phase = phase_name_for_state(state)
        policy, champion_action = choose_champion_action(
            model, args, device, state, player
        )
        if phase in wanted_phases:
          legal = state.legal_actions(player)
          collection_report["candidate_states_seen"] += 1
          skip_reason = _state_filter_reason(state, legal, args)
          if skip_reason is not None:
            _note_collection_skip(collection_report, skip_reason)
            state.apply_action(champion_action)
            continue
          collection_report["candidate_states_eligible"] += 1
          all_buckets, target_matches, match_count = _tactical_match_summary(
              state, player, legal, target_buckets
          )
          if match_count <= 0:
            _note_collection_skip(collection_report, "tactical_miss")
            state.apply_action(champion_action)
            continue
          for name, actions in target_matches.items():
            if actions:
              collection_report["tactical_candidate_matches"][name] += 1
              collection_report["tactical_candidate_actions"][name] += len(actions)
          uncertainty = policy_uncertainty_stats(policy, legal)
          if pool_cap <= 0 or len(candidates) < pool_cap:
            collection_report["candidate_states_pooled"] += 1
            candidates.append({
                "state": state.clone(),
                "player": int(player),
                "phase": phase,
                "round_index": int(round_idx),
                "collected_index_in_match": int(collected_this_match),
                "legal_count": int(len(legal)),
                "champion_action": int(champion_action),
                "champion_action_string": action_string(
                    state, player, champion_action
                ),
                "tactical_target_buckets": list(target_buckets),
                "tactical_bucket_matches": {
                    name: [int(action) for action in actions]
                    for name, actions in target_matches.items()
                    if actions
                },
                "tactical_available_buckets": sorted(all_buckets),
                "tactical_match_count": int(match_count),
                **uncertainty,
            })
            collected_this_match += 1
        state.apply_action(champion_action)
      raw_scores = (
          state.raw_round_scores()
          if hasattr(state, "raw_round_scores")
          else state.returns()
      )
      round_scores = np.array(raw_scores, dtype=np.float32)
      totals = totals + round_scores
      match_paradoxes += np.array(
          getattr(state, "_has_paradoxed", [False] * args.players),
          dtype=np.float32,
      )
    if not candidates:
      matches += 1
      continue
    for candidate in candidates:
      player = int(candidate["player"])
      player_score = float(totals[player])
      best_score = float(np.max(totals))
      loss_gap = max(0.0, best_score - player_score)
      paradox_count = float(match_paradoxes[player])
      entropy = float(candidate.get("policy_entropy_norm") or 0.0)
      hard_score = (
          10.0 * float(candidate.get("tactical_match_count", 0))
          + float(args.hard_entropy_weight) * entropy
          + float(args.hard_loss_weight) * loss_gap
          + float(args.hard_paradox_weight) * paradox_count
      )
      candidate["match_final_totals"] = [float(x) for x in totals]
      candidate["acting_player_final_score"] = player_score
      candidate["acting_player_loss_gap"] = float(loss_gap)
      candidate["acting_player_match_paradoxes"] = paradox_count
      candidate["hard_score"] = float(hard_score)
    candidates.sort(key=lambda item: float(item["hard_score"]), reverse=True)
    for rank, candidate in enumerate(candidates[:per_match_pick]):
      if len(states) >= args.states:
        break
      states.append(candidate["state"])
      collection_report["candidate_states_kept"] += 1
      state_groups.append({
          "match_index": int(matches),
          "initial_start_player": int(initial_start),
          "round_index": int(candidate["round_index"]),
          "collected_index_in_match": int(candidate["collected_index_in_match"]),
          "legal_count": int(candidate.get("legal_count", 0)),
          "state_selection": args.state_selection,
          "hard_rank_in_match": int(rank),
          "hard_score": float(candidate["hard_score"]),
          "policy_entropy": float(candidate["policy_entropy"]),
          "policy_entropy_norm": float(candidate["policy_entropy_norm"]),
          "policy_top_prob": candidate["policy_top_prob"],
          "champion_action": int(candidate["champion_action"]),
          "champion_action_string": candidate["champion_action_string"],
          "match_final_totals": candidate["match_final_totals"],
          "acting_player_final_score": float(candidate["acting_player_final_score"]),
          "acting_player_loss_gap": float(candidate["acting_player_loss_gap"]),
          "acting_player_match_paradoxes": float(
              candidate["acting_player_match_paradoxes"]
          ),
          "tactical_target_buckets": list(candidate["tactical_target_buckets"]),
          "tactical_bucket_matches": candidate["tactical_bucket_matches"],
          "tactical_available_buckets": candidate["tactical_available_buckets"],
          "tactical_match_count": int(candidate["tactical_match_count"]),
      })
    matches += 1
  args._state_collection_report = collection_report
  return states, state_groups, matches


def collect_states(model, args, device, wanted_phases):
  selection = str(getattr(args, "state_selection", "first"))
  if selection == "first":
    return collect_states_first(model, args, device, wanted_phases)
  if selection.startswith("tactical:"):
    return collect_states_tactical(model, args, device, wanted_phases)
  return collect_states_hard(model, args, device, wanted_phases)


def label_scores_for_state(state, player, round_game, model, args, device, policy):
  legal = state.legal_actions(player)
  seat_roles = ["learner"] * args.players
  penalty = max(0.0, float(getattr(args, "label_paradox_penalty", 0.0)))
  max_paradox_delta = getattr(args, "label_max_paradox_delta", None)
  if max_paradox_delta is not None:
    max_paradox_delta = float(max_paradox_delta)
  if penalty > 0 or max_paradox_delta is not None:
    candidate_actions = sampled_counterfactual_legal_actions(
        state, player, legal, args, policy=policy
    )
    if not candidate_actions:
      return {}, [], {}
    stats = evaluate_actions_independently(
        state,
        player,
        candidate_actions,
        round_game,
        model,
        args,
        device,
        getattr(args, "counterfactual_action_value_rollouts", 1),
    )
    policy_best = int(max(legal, key=lambda action: policy[action]))
    policy_paradox = stats.get(policy_best, {}).get("mean_paradox")
    if policy_paradox is not None:
      policy_paradox = float(policy_paradox)
    scores = {}
    for action in candidate_actions:
      action = int(action)
      mean_score = stats.get(action, {}).get("mean")
      if mean_score is None:
        continue
      mean_paradox = stats.get(action, {}).get("mean_paradox")
      mean_paradox = 0.0 if mean_paradox is None else float(mean_paradox)
      if (
          max_paradox_delta is not None
          and policy_paradox is not None
          and action != policy_best
          and mean_paradox > policy_paradox + max_paradox_delta
      ):
        continue
      scores[action] = float(mean_score) - penalty * mean_paradox
    return scores, [int(action) for action in candidate_actions], {
        str(k): v for k, v in sorted(stats.items())
    }
  targets, mask = counterfactual_action_value_targets(
      state,
      round_game,
      player,
      legal,
      seat_roles,
      {},
      model,
      None,
      args,
      device,
      policy=policy,
  )
  if targets is None or mask is None:
    return {}, [], {}
  candidate_actions = sampled_counterfactual_legal_actions(
      state, player, legal, args, policy=policy
  )
  scores = {
      int(action): float(targets[action]) * float(args.value_scale)
      for action in candidate_actions
      if action < len(mask) and float(mask[action]) > 0
  }
  return scores, [int(action) for action in candidate_actions], {}


def evaluate_actions_independently(
    state, player, actions, round_game, model, args, device, rollouts
):
  rollout_fn = (
      rollout_match_score_after_action
      if args.counterfactual_full_match_rollout
      else rollout_score_after_action
  )
  action_outcomes = {int(action): [] for action in actions}
  action_paradoxes = {int(action): [] for action in actions}
  belief_states = counterfactual_belief_states(
      state, player, args, model=model, device=device
  )
  seat_roles = ["learner"] * args.players
  for belief_state in belief_states:
    legal = set(belief_state.legal_actions(player))
    if any(action not in legal for action in actions):
      continue
    for _ in range(max(1, int(rollouts))):
      py_state = random.getstate()
      np_state = np.random.get_state()
      for action in actions:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if (
            args.counterfactual_full_match_rollout
            and int(getattr(args, "counterfactual_rollout_max_plies", 0)) == 0
        ):
          detail = rollout_full_match_detail_after_action(
              belief_state,
              player,
              int(action),
              seat_roles,
              model,
              args,
              device,
          )
          action_outcomes[int(action)].append(float(detail["score"]))
          action_paradoxes[int(action)].append(float(detail["paradox_count"]))
        else:
          action_outcomes[int(action)].append(
              float(
                  rollout_fn(
                      belief_state,
                      round_game,
                      player,
                      int(action),
                      seat_roles,
                      {},
                      model,
                      None,
                      args,
                      device,
                  )
              )
          )
          random.setstate(py_state)
          np.random.set_state(np_state)
          action_paradoxes[int(action)].append(
              float(
                  rollout_paradox_after_action(
                      belief_state,
                      round_game,
                      player,
                      int(action),
                      seat_roles,
                      {},
                      model,
                      None,
                      args,
                      device,
                  )
              )
          )
  return {
      action: {
          "mean": float(np.mean(values)) if values else None,
          "mean_paradox": (
              float(np.mean(action_paradoxes[action]))
              if action_paradoxes[action] else None
          ),
          "n": int(len(values)),
      }
      for action, values in action_outcomes.items()
  }


def evaluate_actions_with_stability(
    state,
    player,
    actions,
    round_game,
    model,
    args,
    device,
    rollouts,
    *,
    base_seed,
    passes,
    seed_stride,
    label_action,
    policy_action,
    safe_paradox_delta,
):
  """Run one or more independent action audits and summarize stability."""
  passes = max(1, int(passes))
  if passes == 1:
    set_eval_seed(base_seed)
    audit = evaluate_actions_independently(
        state, player, actions, round_game, model, args, device, rollouts
    )
    label_stats = audit.get(int(label_action), {})
    policy_stats = audit.get(int(policy_action), {})
    label_mean = label_stats.get("mean")
    policy_mean = policy_stats.get("mean")
    label_paradox = label_stats.get("mean_paradox")
    policy_paradox = policy_stats.get("mean_paradox")
    margin = (
        None if label_mean is None or policy_mean is None
        else float(label_mean - policy_mean)
    )
    paradox_delta = (
        None if label_paradox is None or policy_paradox is None
        else float(label_paradox - policy_paradox)
    )
    return audit, {
        "passes": 1,
        "valid_passes": int(margin is not None),
        "margins": [] if margin is None else [float(margin)],
        "paradox_deltas": [] if paradox_delta is None else [float(paradox_delta)],
        "positive_sign_rate": None if margin is None else float(margin > 0.0),
        "safe_risk_rate": (
            None if paradox_delta is None
            else float(paradox_delta <= float(safe_paradox_delta))
        ),
        "stable_positive": True,
        "seeds": [int(base_seed)],
    }

  per_pass = []
  margins = []
  paradox_deltas = []
  seeds = []
  for pass_idx in range(passes):
    seed = int(base_seed) + int(pass_idx) * int(seed_stride)
    seeds.append(int(seed))
    set_eval_seed(seed)
    audit = evaluate_actions_independently(
        state, player, actions, round_game, model, args, device, rollouts
    )
    per_pass.append(audit)
    label_stats = audit.get(int(label_action), {})
    policy_stats = audit.get(int(policy_action), {})
    label_mean = label_stats.get("mean")
    policy_mean = policy_stats.get("mean")
    label_paradox = label_stats.get("mean_paradox")
    policy_paradox = policy_stats.get("mean_paradox")
    if label_mean is not None and policy_mean is not None:
      margins.append(float(label_mean - policy_mean))
    if label_paradox is not None and policy_paradox is not None:
      paradox_deltas.append(float(label_paradox - policy_paradox))

  aggregate = {}
  for action in actions:
    action = int(action)
    means = [
        audit.get(action, {}).get("mean")
        for audit in per_pass
        if audit.get(action, {}).get("mean") is not None
    ]
    paradoxes = [
        audit.get(action, {}).get("mean_paradox")
        for audit in per_pass
        if audit.get(action, {}).get("mean_paradox") is not None
    ]
    counts = [
        int(audit.get(action, {}).get("n") or 0)
        for audit in per_pass
    ]
    aggregate[action] = {
        "mean": None if not means else float(np.mean(means)),
        "mean_paradox": None if not paradoxes else float(np.mean(paradoxes)),
        "n": int(np.sum(counts)),
        "pass_means": [None if value is None else float(value) for value in means],
        "pass_paradoxes": [
            None if value is None else float(value) for value in paradoxes
        ],
    }

  positive_signs = [float(margin > 0.0) for margin in margins]
  safe_risks = [
      float(delta <= float(safe_paradox_delta)) for delta in paradox_deltas
  ]
  positive_sign_rate = None if not positive_signs else float(np.mean(positive_signs))
  safe_risk_rate = None if not safe_risks else float(np.mean(safe_risks))
  return aggregate, {
      "passes": int(passes),
      "valid_passes": int(len(margins)),
      "margins": [float(x) for x in margins],
      "paradox_deltas": [float(x) for x in paradox_deltas],
      "mean_margin": None if not margins else float(np.mean(margins)),
      "mean_paradox_delta": (
          None if not paradox_deltas else float(np.mean(paradox_deltas))
      ),
      "positive_sign_rate": positive_sign_rate,
      "safe_risk_rate": safe_risk_rate,
      "stable_positive": (
          positive_sign_rate is not None
          and safe_risk_rate is not None
          and positive_sign_rate >= float(
              getattr(args, "stability_min_positive_sign_rate", 1.0)
          )
          and safe_risk_rate >= float(
              getattr(args, "stability_min_safe_risk_rate", 1.0)
          )
      ),
      "seeds": seeds,
  }


def scoped_full_match_paradox_count(
    paradox_counts, any_paradox_rounds, acting_player, args
):
  """Return full-match paradox risk in the configured promotion scope."""
  if getattr(args, "counterfactual_action_paradox_scope", "any") == "any":
    return float(any_paradox_rounds)
  paradox_counts = np.asarray(paradox_counts, dtype=np.float32)
  if 0 <= int(acting_player) < paradox_counts.size:
    return float(paradox_counts[int(acting_player)])
  return 0.0


def rollout_full_match_detail_after_action(
    state, acting_player, first_action, seat_roles, model, args, device
):
  """Roll out an action through the remaining match and report score/risk."""
  rollout_state = state.clone()
  current_round = int(getattr(rollout_state, "_match_round", 0))
  current_start = int(getattr(rollout_state, "_round_start_player", 0))
  initial_start = (current_start - current_round) % args.players
  match_totals = np.array(
      getattr(rollout_state, "_match_totals", np.zeros(args.players)),
      dtype=np.float32,
  )
  final_round_scores = np.zeros(args.players, dtype=np.float32)
  paradoxes = np.zeros(args.players, dtype=np.float32)
  any_paradox_rounds = 0.0

  rollout_state.apply_action(first_action)
  play_rollout_to_terminal(
      rollout_state, round_game=make_game(args, current_start),
      seat_roles=seat_roles, fixed_bots={}, model=model, opponent_model=None,
      args=args, device=device,
  )
  raw_scores = (
      rollout_state.raw_round_scores()
      if hasattr(rollout_state, "raw_round_scores")
      else rollout_state.returns()
  )
  final_round_scores = np.array(raw_scores, dtype=np.float32)
  match_totals += final_round_scores
  round_paradoxes = np.array(
      getattr(rollout_state, "_has_paradoxed", [False] * args.players),
      dtype=np.float32,
  )
  paradoxes += round_paradoxes
  any_paradox_rounds += float(np.any(round_paradoxes > 0.0))

  for next_round in range(current_round + 1, args.players):
    start_player = (initial_start + next_round) % args.players
    next_game = make_game(args, start_player)
    next_state = next_game.new_initial_state()
    if getattr(args, "match_context", False):
      next_state.set_match_context(match_totals, next_round)
    play_rollout_to_terminal(
        next_state, next_game, seat_roles, {}, model, None, args, device
    )
    raw_scores = (
        next_state.raw_round_scores()
        if hasattr(next_state, "raw_round_scores")
        else next_state.returns()
    )
    final_round_scores = np.array(raw_scores, dtype=np.float32)
    match_totals += final_round_scores
    round_paradoxes = np.array(
        getattr(next_state, "_has_paradoxed", [False] * args.players),
        dtype=np.float32,
    )
    paradoxes += round_paradoxes
    any_paradox_rounds += float(np.any(round_paradoxes > 0.0))

  official_scores = match_outcome_scores(match_totals, final_round_scores)
  return {
      "score": float(match_totals[acting_player]),
      "official_score": float(official_scores[acting_player]),
      "paradox_count": scoped_full_match_paradox_count(
          paradoxes, any_paradox_rounds, acting_player, args
      ),
      "acting_player_paradox_count": float(paradoxes[acting_player]),
      "any_player_paradox_round_count": float(any_paradox_rounds),
  }


def accepted_replay_example(
    state,
    player,
    best_action,
    policy_action,
    audit,
    args,
    label_stats=None,
    scored_actions=None,
):
  num_actions = state.num_distinct_actions()
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[state.legal_actions(player)] = 1.0
  policy = np.zeros(num_actions, dtype=np.float32)
  policy[int(best_action)] = 1.0
  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  value_scale = max(1e-6, float(getattr(args, "value_scale", 20.0)))
  if bool(getattr(args, "counterfactual_full_match_rollout", False)):
    current_round = int(getattr(state, "_match_round", 0))
    max_future_paradoxes = max(
        1.0,
        float(getattr(args, "players", 3) - current_round),
    )
  else:
    max_future_paradoxes = 1.0
  source_stats = {}
  if label_stats:
    for action, stats in label_stats.items():
      try:
        action_int = int(action)
      except (TypeError, ValueError):
        continue
      if scored_actions is not None and action_int not in scored_actions:
        continue
      source_stats[action_int] = stats
  for action, stats in audit.items():
    source_stats[int(action)] = stats
  for action in (int(best_action), int(policy_action)):
    if action not in source_stats:
      stats = audit.get(action) or audit.get(str(action))
      if stats:
        source_stats[action] = stats
  for action, stats in sorted(source_stats.items()):
    if not stats:
      continue
    mean_score = stats.get("mean")
    if mean_score is not None:
      action_value_targets[action] = float(mean_score) / value_scale
      action_value_mask[action] = 1.0
    mean_paradox = stats.get("mean_paradox")
    if mean_paradox is not None:
      action_paradox_targets[action] = min(
          1.0, max(0.0, float(mean_paradox) / max_future_paradoxes)
      )
      action_paradox_mask[action] = 1.0
  return (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      policy,
      np.zeros(getattr(args, "players", 3), dtype=np.float32),
      np.zeros(getattr(args, "players", 3), dtype=np.float32),
      int(best_action),
      int(player),
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      action_feature_matrix(state, player, num_actions),
  )


def accepted_replay_summary(replay):
  if not replay:
    return {
        "rows": 0,
        "q_label_count_mean": None,
        "risk_label_count_mean": None,
        "legal_count_mean": None,
        "q_label_coverage_mean": None,
        "risk_label_coverage_mean": None,
        "q_spread_mean": None,
    }
  legal_counts = []
  q_counts = []
  risk_counts = []
  q_coverages = []
  risk_coverages = []
  q_spreads = []
  for example in replay:
    legal_mask = np.array(example[1], dtype=np.float32) > 0.0
    q_mask = np.array(example[10], dtype=np.float32) > 0.0
    risk_mask = np.array(example[8], dtype=np.float32) > 0.0
    legal_count = int(np.sum(legal_mask))
    q_count = int(np.sum(q_mask))
    risk_count = int(np.sum(risk_mask))
    legal_counts.append(legal_count)
    q_counts.append(q_count)
    risk_counts.append(risk_count)
    if legal_count > 0:
      q_coverages.append(q_count / float(legal_count))
      risk_coverages.append(risk_count / float(legal_count))
    if q_count >= 2:
      targets = np.array(example[9], dtype=np.float32)
      q_spreads.append(float(np.max(targets[q_mask]) - np.min(targets[q_mask])))

  def summarize(values):
    if not values:
      return None
    return {
        "min": round(float(np.min(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "max": round(float(np.max(values)), 6),
    }

  return {
      "rows": int(len(replay)),
      "q_label_count": summarize(q_counts),
      "risk_label_count": summarize(risk_counts),
      "legal_count": summarize(legal_counts),
      "q_label_coverage": summarize(q_coverages),
      "risk_label_coverage": summarize(risk_coverages),
      "q_spread": summarize(q_spreads),
  }


def calibration_row_class(row, positive_margin, max_positive_paradox_delta):
  """Classify an expensive-audited label-vs-policy disagreement."""
  if not row.get("audited"):
    return "UNAUDITED"
  margin = row.get("audit_margin_vs_policy")
  paradox_delta = row.get("audit_paradox_delta_vs_policy")
  if margin is None:
    return "UNAUDITED"
  margin = float(margin)
  if margin <= 0.0:
    return "NEGATIVE"
  if paradox_delta is None:
    return "NEAR_POSITIVE"
  paradox_delta = float(paradox_delta)
  if (
      margin >= float(positive_margin)
      and paradox_delta <= float(max_positive_paradox_delta)
  ):
    return "POSITIVE"
  return "NEAR_POSITIVE"


def calibration_target_action(row, row_class):
  if row_class in ("POSITIVE", "NEAR_POSITIVE"):
    return int(row["label_best_action"])
  return int(row["policy_best_action"])


def calibration_summary(metadata):
  classes = {}
  false_positive_rows = 0
  margins_by_class = {}
  paradox_by_class = {}
  candidate_recalls = []
  for item in metadata:
    row_class = item.get("calibration_class", "UNKNOWN")
    classes[row_class] = classes.get(row_class, 0) + 1
    if item.get("false_positive_for_positive_gate"):
      false_positive_rows += 1
    margin = item.get("audit_margin_vs_policy")
    if margin is not None:
      margins_by_class.setdefault(row_class, []).append(float(margin))
    paradox_delta = item.get("audit_paradox_delta_vs_policy")
    if paradox_delta is not None:
      paradox_by_class.setdefault(row_class, []).append(float(paradox_delta))
    if item.get("candidate_set_contains_label_best") is not None:
      candidate_recalls.append(bool(item["candidate_set_contains_label_best"]))

  def summarize(values):
    if not values:
      return None
    return {
        "min": round(float(np.min(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "max": round(float(np.max(values)), 6),
    }

  return {
      "rows": int(len(metadata)),
      "classes": {k: int(v) for k, v in sorted(classes.items())},
      "false_positive_for_positive_gate": int(false_positive_rows),
      "margin_by_class": {
          k: summarize(v) for k, v in sorted(margins_by_class.items())
      },
      "paradox_delta_by_class": {
          k: summarize(v) for k, v in sorted(paradox_by_class.items())
      },
      "candidate_set_label_best_recall": (
          None
          if not candidate_recalls
          else float(np.mean(np.array(candidate_recalls, dtype=np.float32)))
      ),
  }


def candidate_recall_summary(rows):
  audits = [
      row.get("candidate_recall_audit")
      for row in rows
      if row.get("candidate_recall_audit")
  ]
  if not audits:
    return {
        "audited_rows": 0,
        "cap_contains_all_legal_best_rate": None,
        "mean_all_legal_best_minus_cap_best": None,
        "all_legal_better_rows": 0,
    }
  contains = [
      bool(audit.get("cap_contains_all_legal_best")) for audit in audits
  ]
  score_gaps = [
      float(audit.get("all_legal_best_minus_cap_best", 0.0))
      for audit in audits
      if audit.get("all_legal_best_minus_cap_best") is not None
  ]
  better_rows = [
      gap for gap in score_gaps if gap > 1e-9
  ]
  return {
      "audited_rows": int(len(audits)),
      "cap_contains_all_legal_best_rate": float(np.mean(contains)),
      "mean_all_legal_best_minus_cap_best": (
          None if not score_gaps else float(np.mean(score_gaps))
      ),
      "max_all_legal_best_minus_cap_best": (
          None if not score_gaps else float(np.max(score_gaps))
      ),
      "all_legal_better_rows": int(len(better_rows)),
      "all_legal_better_rate": float(len(better_rows) / float(len(audits))),
  }


def maybe_candidate_recall_audit(
    cli_args,
    args,
    state,
    player,
    round_game,
    model,
    device,
    policy,
    capped_scores,
    capped_candidates,
    idx,
):
  every = int(getattr(cli_args, "candidate_recall_audit_every", 0))
  if every <= 0 or not capped_scores or (idx % every) != 0:
    return None
  all_args = SimpleNamespace(**vars(args))
  all_args.counterfactual_action_max_legal = 0
  all_args.counterfactual_action_top_policy = 0
  all_scores, all_candidates, _ = label_scores_for_state(
      state, player, round_game, model, all_args, device, policy
  )
  if not all_scores:
    return {
        "all_legal_labeled": False,
        "capped_candidate_count": int(len(capped_candidates)),
        "all_legal_candidate_count": int(len(all_candidates)),
    }
  capped_best = int(max(capped_scores, key=lambda action: capped_scores[action]))
  all_best = int(max(all_scores, key=lambda action: all_scores[action]))
  capped_scored_under_all = [
      int(action) for action in capped_candidates if int(action) in all_scores
  ]
  if capped_scored_under_all:
    capped_best_under_all = int(
        max(capped_scored_under_all, key=lambda action: all_scores[action])
    )
    capped_best_score_under_all = float(all_scores[capped_best_under_all])
  else:
    capped_best_under_all = capped_best
    capped_best_score_under_all = None
  capped_best_score = float(capped_scores[capped_best])
  all_best_score = float(all_scores[all_best])
  return {
      "all_legal_labeled": True,
      "capped_candidate_count": int(len(capped_candidates)),
      "all_legal_candidate_count": int(len(all_candidates)),
      "capped_best_action": capped_best,
      "capped_best_action_string": action_string(state, player, capped_best),
      "all_legal_best_action": all_best,
      "all_legal_best_action_string": action_string(state, player, all_best),
      "cap_contains_all_legal_best": bool(
          int(all_best) in set(int(action) for action in capped_candidates)
      ),
      "capped_best_score": capped_best_score,
      "capped_best_under_all_legal_action": capped_best_under_all,
      "capped_best_under_all_legal_action_string": action_string(
          state, player, capped_best_under_all
      ),
      "capped_best_under_all_legal_score": capped_best_score_under_all,
      "all_legal_best_score": all_best_score,
      "all_legal_best_minus_cap_best": (
          None
          if capped_best_score_under_all is None
          else float(all_best_score - capped_best_score_under_all)
      ),
  }


def summarize(
    rows,
    pass_min_score_rate,
    pass_min_avg_diff,
    pass_max_paradox_delta,
    accept_min_audit_margin,
    accept_max_paradox_delta,
):
  summary = {
      "states": len(rows),
      "distinct_match_groups": 0,
      "labeled_states": 0,
      "disagreements": 0,
      "audited_disagreements": 0,
      "audit_passes": 0,
      "label_best_matches_policy": 0,
      "mean_label_margin_vs_policy": None,
      "mean_audit_margin_vs_policy": None,
      "audit_wins": 0,
      "audit_losses": 0,
      "audit_ties": 0,
      "audit_score_rate": None,
      "mean_audit_paradox_delta_vs_policy": None,
      "pass_min_score_rate": float(pass_min_score_rate),
      "pass_min_avg_diff": float(pass_min_avg_diff),
      "pass_max_paradox_delta": float(pass_max_paradox_delta),
      "label_oracle_pass": False,
      "accept_min_audit_margin": float(accept_min_audit_margin),
      "accept_max_paradox_delta": float(accept_max_paradox_delta),
      "accepted_labels": 0,
      "accepted_rate_per_state": None,
      "accepted_rate_per_labeled_state": None,
      "accepted_mean_audit_margin": None,
      "accepted_mean_paradox_delta": None,
  }
  label_margins = []
  audit_margins = []
  audit_paradox_deltas = []
  accepted_margins = []
  accepted_paradox_deltas = []
  by_phase = {}
  match_groups = set()
  for row in rows:
    group = row.get("group") or {}
    if "match_index" in group:
      match_groups.add(int(group["match_index"]))
    phase = row["phase"]
    by_phase.setdefault(phase, {
        "states": 0,
        "labeled_states": 0,
        "disagreements": 0,
        "audited_disagreements": 0,
        "audit_passes": 0,
        "label_best_matches_policy": 0,
        "audit_wins": 0,
        "audit_losses": 0,
        "audit_ties": 0,
        "accepted_labels": 0,
        "label_margins": [],
        "audit_margins": [],
        "audit_paradox_deltas": [],
        "accepted_margins": [],
        "accepted_paradox_deltas": [],
    })
    bucket = by_phase[phase]
    bucket["states"] += 1
    if not row.get("labeled"):
      continue
    summary["labeled_states"] += 1
    bucket["labeled_states"] += 1
    margin = row.get("label_margin_vs_policy")
    if margin is not None:
      label_margins.append(float(margin))
      bucket["label_margins"].append(float(margin))
    if row["label_best_action"] == row["policy_best_action"]:
      summary["label_best_matches_policy"] += 1
      bucket["label_best_matches_policy"] += 1
    else:
      summary["disagreements"] += 1
      bucket["disagreements"] += 1
      if row.get("audited"):
        summary["audited_disagreements"] += 1
        bucket["audited_disagreements"] += 1
        audit_margin = float(row["audit_margin_vs_policy"])
        audit_margins.append(audit_margin)
        bucket["audit_margins"].append(audit_margin)
        if audit_margin > 0:
          summary["audit_wins"] += 1
          bucket["audit_wins"] += 1
        elif audit_margin < 0:
          summary["audit_losses"] += 1
          bucket["audit_losses"] += 1
        else:
          summary["audit_ties"] += 1
          bucket["audit_ties"] += 1
        paradox_delta = row.get("audit_paradox_delta_vs_policy")
        if paradox_delta is not None:
          paradox_delta = float(paradox_delta)
          audit_paradox_deltas.append(paradox_delta)
          bucket["audit_paradox_deltas"].append(paradox_delta)
        if row.get("audit_pass"):
          summary["audit_passes"] += 1
          bucket["audit_passes"] += 1
        accepted = bool(row.get("accepted_for_replay"))
        if accepted:
          summary["accepted_labels"] += 1
          bucket["accepted_labels"] += 1
          accepted_margins.append(audit_margin)
          bucket["accepted_margins"].append(audit_margin)
          if paradox_delta is not None:
            accepted_paradox_deltas.append(paradox_delta)
            bucket["accepted_paradox_deltas"].append(paradox_delta)
  if label_margins:
    summary["mean_label_margin_vs_policy"] = float(np.mean(label_margins))
  if audit_margins:
    summary["mean_audit_margin_vs_policy"] = float(np.mean(audit_margins))
    summary["audit_score_rate"] = float(
        (summary["audit_wins"] + 0.5 * summary["audit_ties"])
        / float(len(audit_margins))
    )
  if audit_paradox_deltas:
    summary["mean_audit_paradox_delta_vs_policy"] = float(
        np.mean(audit_paradox_deltas)
    )
  avg_diff = summary["mean_audit_margin_vs_policy"]
  score_rate = summary["audit_score_rate"]
  paradox_delta = summary["mean_audit_paradox_delta_vs_policy"]
  summary["label_oracle_pass"] = bool(
      score_rate is not None
      and avg_diff is not None
      and paradox_delta is not None
      and score_rate >= float(pass_min_score_rate)
      and avg_diff > float(pass_min_avg_diff)
      and paradox_delta <= float(pass_max_paradox_delta)
  )
  summary["distinct_match_groups"] = int(len(match_groups))
  if rows:
    summary["accepted_rate_per_state"] = float(
        summary["accepted_labels"] / float(len(rows))
    )
  if summary["labeled_states"]:
    summary["accepted_rate_per_labeled_state"] = float(
        summary["accepted_labels"] / float(summary["labeled_states"])
    )
  if accepted_margins:
    summary["accepted_mean_audit_margin"] = float(np.mean(accepted_margins))
  if accepted_paradox_deltas:
    summary["accepted_mean_paradox_delta"] = float(
        np.mean(accepted_paradox_deltas)
    )
  for phase, bucket in by_phase.items():
    if bucket["label_margins"]:
      bucket["mean_label_margin_vs_policy"] = float(np.mean(bucket["label_margins"]))
    else:
      bucket["mean_label_margin_vs_policy"] = None
    if bucket["audit_margins"]:
      bucket["mean_audit_margin_vs_policy"] = float(np.mean(bucket["audit_margins"]))
      bucket["audit_score_rate"] = float(
          (bucket["audit_wins"] + 0.5 * bucket["audit_ties"])
          / float(len(bucket["audit_margins"]))
      )
    else:
      bucket["mean_audit_margin_vs_policy"] = None
      bucket["audit_score_rate"] = None
    if bucket["audit_paradox_deltas"]:
      bucket["mean_audit_paradox_delta_vs_policy"] = float(
          np.mean(bucket["audit_paradox_deltas"])
      )
    else:
      bucket["mean_audit_paradox_delta_vs_policy"] = None
    if bucket["states"]:
      bucket["accepted_rate_per_state"] = float(
          bucket["accepted_labels"] / float(bucket["states"])
      )
    else:
      bucket["accepted_rate_per_state"] = None
    if bucket["labeled_states"]:
      bucket["accepted_rate_per_labeled_state"] = float(
          bucket["accepted_labels"] / float(bucket["labeled_states"])
      )
    else:
      bucket["accepted_rate_per_labeled_state"] = None
    if bucket["accepted_margins"]:
      bucket["accepted_mean_audit_margin"] = float(
          np.mean(bucket["accepted_margins"])
      )
    else:
      bucket["accepted_mean_audit_margin"] = None
    if bucket["accepted_paradox_deltas"]:
      bucket["accepted_mean_paradox_delta"] = float(
          np.mean(bucket["accepted_paradox_deltas"])
      )
    else:
      bucket["accepted_mean_paradox_delta"] = None
    del bucket["label_margins"]
    del bucket["audit_margins"]
    del bucket["audit_paradox_deltas"]
    del bucket["accepted_margins"]
    del bucket["accepted_paradox_deltas"]
  return summary, by_phase


def main():
  cli_args = parse_args()
  random.seed(cli_args.seed)
  np.random.seed(cli_args.seed % (2**32 - 1))
  torch.manual_seed(cli_args.seed)
  device = torch.device(
      "mps"
      if cli_args.device == "mps" or (
          cli_args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  load_args = model_args_for(cli_args.players)
  load_game = make_game(SimpleNamespace(players=cli_args.players, match_context=True), 0)
  model, _, saved_args = load_model_payload(
      cli_args.checkpoint, load_game, load_args, device
  )
  model.eval()
  args = audit_namespace(cli_args, saved_args)
  wanted = {part.strip() for part in cli_args.phases.split(",") if part.strip()}
  started = time.perf_counter()
  states, state_groups, matches_used = collect_states(model, args, device, wanted)
  rows = []
  accepted_replay = []
  calibration_replay = []
  calibration_metadata = []
  early_stop_reason = None
  for idx, state in enumerate(states):
    player = int(state.current_player())
    phase = phase_name_for_state(state)
    round_game = make_game(args, int(getattr(state, "_round_start_player", 0)))
    policy, policy_best = choose_champion_action(model, args, device, state, player)
    label_seed = int(cli_args.seed) + int(cli_args.label_seed_offset) + idx * 7919
    audit_seed = int(cli_args.seed) + int(cli_args.audit_seed_offset) + idx * 104729
    set_eval_seed(label_seed)
    scores, candidates, label_stats = label_scores_for_state(
        state, player, round_game, model, args, device, policy
    )
    row = {
        "index": idx,
        "phase": phase,
        "player": player,
        "group": state_groups[idx] if idx < len(state_groups) else {},
        "match_round": int(getattr(state, "_match_round", -1)),
        "round_start_player": int(getattr(state, "_round_start_player", -1)),
        "match_totals": [
            float(x)
            for x in np.array(
                getattr(state, "_match_totals", np.zeros(args.players)),
                dtype=np.float32,
            )
        ],
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "legal_count": int(len(state.legal_actions(player))),
        "candidate_actions": candidates,
        "label_seed": int(label_seed),
        "audit_seed": int(audit_seed),
        "policy_best_action": int(policy_best),
        "policy_best_action_string": action_string(state, player, policy_best),
        "labeled": bool(scores),
        "accepted_for_replay": False,
    }
    if scores:
      recall_audit = maybe_candidate_recall_audit(
          cli_args,
          args,
          state,
          player,
          round_game,
          model,
          device,
          policy,
          scores,
          candidates,
          idx,
      )
      if recall_audit is not None:
        row["candidate_recall_audit"] = recall_audit
      label_best = max(scores, key=lambda action: scores[action])
      policy_score = scores.get(policy_best)
      label_margin = (
          float(scores[label_best] - policy_score)
          if policy_score is not None else None
      )
      row.update({
          "label_scores": {str(k): round(v, 6) for k, v in sorted(scores.items())},
          "label_stats": label_stats,
          "label_best_action": int(label_best),
          "label_best_action_string": action_string(state, player, label_best),
          "label_margin_vs_policy": label_margin,
      })
      if (
          label_best == policy_best
          and bool(cli_args.calibration_include_policy_matches)
      ):
        row["calibration_class"] = "NEGATIVE"
        row["false_positive_for_positive_gate"] = True
        calibration_replay.append(
            accepted_replay_example(
                state,
                player,
                policy_best,
                policy_best,
                {},
                args,
                label_stats=label_stats,
                scored_actions=set(int(action) for action in scores),
            )
        )
        calibration_metadata.append({
            "source_index": int(idx),
            "phase": phase,
            "player": int(player),
            "group": row.get("group") or {},
            "policy_best_action": int(policy_best),
            "policy_best_action_string": row["policy_best_action_string"],
            "label_best_action": int(label_best),
            "label_best_action_string": row["label_best_action_string"],
            "deployment_target_action": int(policy_best),
            "deployment_target_action_string": row["policy_best_action_string"],
            "calibration_class": "NEGATIVE",
            "false_positive_for_positive_gate": True,
            "label_margin_vs_policy": (
                None if label_margin is None else float(label_margin)
            ),
            "audit_margin_vs_policy": 0.0,
            "audit_paradox_delta_vs_policy": 0.0,
            "legal_count": int(row["legal_count"]),
            "candidate_count": int(len(candidates)),
            "candidate_set_contains_label_best": bool(
                int(label_best) in set(int(action) for action in candidates)
            ),
            "candidate_actions": [int(action) for action in candidates],
            "label_seed": int(label_seed),
            "audit_seed": None,
            "policy_match_negative": True,
        })
      should_audit = (
          label_best != policy_best
          and label_margin is not None
          and label_margin >= float(cli_args.min_label_margin)
      )
      if should_audit:
        audit, stability = evaluate_actions_with_stability(
            state,
            player,
            [label_best, policy_best],
            round_game,
            model,
            args,
            device,
            cli_args.audit_rollouts,
            base_seed=audit_seed,
            passes=cli_args.stability_audit_passes,
            seed_stride=cli_args.stability_seed_stride,
            label_action=label_best,
            policy_action=policy_best,
            safe_paradox_delta=cli_args.calibration_max_positive_paradox_delta,
        )
        label_mean = audit[int(label_best)]["mean"]
        policy_mean = audit[int(policy_best)]["mean"]
        label_paradox = audit[int(label_best)].get("mean_paradox")
        policy_paradox = audit[int(policy_best)].get("mean_paradox")
        audit_margin = (
            None
            if label_mean is None or policy_mean is None
            else float(label_mean - policy_mean)
        )
        paradox_delta = (
            None
            if label_paradox is None or policy_paradox is None
            else float(label_paradox - policy_paradox)
        )
        row.update({
            "audited": audit_margin is not None,
            "audit": {str(k): v for k, v in sorted(audit.items())},
            "stability": stability,
            "audit_margin_vs_policy": audit_margin,
            "audit_paradox_delta_vs_policy": paradox_delta,
            "audit_pass": (
                audit_margin is not None
                and audit_margin >= float(cli_args.min_audit_margin)
            ),
            "accepted_for_replay": (
                audit_margin is not None
                and paradox_delta is not None
                and audit_margin >= float(cli_args.min_audit_margin)
                and paradox_delta <= float(cli_args.pass_max_paradox_delta)
            ),
        })
        row_class = calibration_row_class(
            row,
            cli_args.calibration_positive_margin,
            cli_args.calibration_max_positive_paradox_delta,
        )
        raw_row_class = row_class
        stability_demoted = False
        if (
            raw_row_class == "POSITIVE"
            and int(cli_args.stability_audit_passes) > 1
            and not bool(stability.get("stable_positive", False))
            and bool(cli_args.unstable_positive_as_negative)
        ):
          row_class = "UNSTABLE_POSITIVE"
          row["accepted_for_replay"] = False
          stability_demoted = True
        row["calibration_class"] = row_class
        row["raw_calibration_class"] = raw_row_class
        row["stability_demoted"] = bool(stability_demoted)
        row["false_positive_for_positive_gate"] = bool(
            row.get("audited") and row_class != "POSITIVE"
        )
        if row.get("audited"):
          target_action = calibration_target_action(row, row_class)
          calibration_replay.append(
              accepted_replay_example(
                  state,
                  player,
                  target_action,
                  policy_best,
                  audit,
                  args,
                  label_stats=label_stats,
                  scored_actions=set(int(action) for action in scores),
              )
          )
          calibration_metadata.append({
              "source_index": int(idx),
              "phase": phase,
              "player": int(player),
              "group": row.get("group") or {},
              "policy_best_action": int(policy_best),
              "policy_best_action_string": row["policy_best_action_string"],
              "label_best_action": int(label_best),
              "label_best_action_string": row["label_best_action_string"],
              "deployment_target_action": int(target_action),
              "deployment_target_action_string": action_string(
                  state, player, target_action
              ),
              "calibration_class": row_class,
              "raw_calibration_class": raw_row_class,
              "stability_demoted": bool(stability_demoted),
              "stability": stability,
              "false_positive_for_positive_gate": bool(
                  row["false_positive_for_positive_gate"]
              ),
              "label_margin_vs_policy": (
                  None if label_margin is None else float(label_margin)
              ),
              "audit_margin_vs_policy": (
                  None if audit_margin is None else float(audit_margin)
              ),
              "audit_paradox_delta_vs_policy": (
                  None if paradox_delta is None else float(paradox_delta)
              ),
              "legal_count": int(row["legal_count"]),
              "candidate_count": int(len(candidates)),
              "candidate_set_contains_label_best": bool(
                  int(label_best) in set(int(action) for action in candidates)
              ),
              "candidate_actions": [int(action) for action in candidates],
              "label_seed": int(label_seed),
              "audit_seed": int(audit_seed),
          })
        if row["accepted_for_replay"]:
          accepted_replay.append(
              accepted_replay_example(
                  state,
                  player,
                  label_best,
                  policy_best,
                  audit,
                  args,
                  label_stats=label_stats,
                  scored_actions=set(int(action) for action in scores),
              )
          )
      else:
        row["audited"] = False
        row["accepted_for_replay"] = False
        row["calibration_class"] = "UNAUDITED"
        row["false_positive_for_positive_gate"] = False
    rows.append(row)
    calibration_class_counts = {}
    for item in calibration_metadata:
      row_class = str(item.get("calibration_class", "UNKNOWN"))
      calibration_class_counts[row_class] = (
          calibration_class_counts.get(row_class, 0) + 1
      )
    target_positive_rows = int(cli_args.calibration_target_positive_rows)
    target_negative_rows = int(cli_args.calibration_target_negative_rows)
    positive_target_met = (
        target_positive_rows <= 0
        or calibration_class_counts.get("POSITIVE", 0) >= target_positive_rows
    )
    negative_target_met = (
        target_negative_rows <= 0
        or calibration_class_counts.get("NEGATIVE", 0) >= target_negative_rows
    )
    if (
        (target_positive_rows > 0 or target_negative_rows > 0)
        and positive_target_met
        and negative_target_met
    ):
      early_stop_reason = "calibration_target_rows"
      if cli_args.progress_interval > 0:
        print(
            json.dumps({
                "processed_states": int(idx + 1),
                "total_states": int(len(states)),
                "accepted_replay_rows": int(len(accepted_replay)),
                "calibration_replay_rows": int(len(calibration_replay)),
                "calibration_classes": {
                    str(k): int(v)
                    for k, v in sorted(calibration_class_counts.items())
                },
                "early_stop_reason": early_stop_reason,
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }),
            flush=True,
        )
      break
    if cli_args.progress_interval > 0 and (
        (idx + 1) % cli_args.progress_interval == 0
        or (idx + 1) == len(states)
    ):
      calibration_class_counts = {}
      for item in calibration_metadata:
        row_class = str(item.get("calibration_class", "UNKNOWN"))
        calibration_class_counts[row_class] = (
            calibration_class_counts.get(row_class, 0) + 1
        )
      print(
          json.dumps({
              "processed_states": int(idx + 1),
              "total_states": int(len(states)),
              "accepted_replay_rows": int(len(accepted_replay)),
              "calibration_replay_rows": int(len(calibration_replay)),
              "calibration_classes": {
                  str(k): int(v)
                  for k, v in sorted(calibration_class_counts.items())
              },
              "elapsed_sec": round(time.perf_counter() - started, 3),
          }),
          flush=True,
      )

  summary, by_phase = summarize(
      rows,
      cli_args.pass_min_score_rate,
      cli_args.pass_min_avg_diff,
      cli_args.pass_max_paradox_delta,
      cli_args.min_audit_margin,
      cli_args.pass_max_paradox_delta,
  )
  result = {
      "checkpoint": cli_args.checkpoint,
      "players": cli_args.players,
      "seed": cli_args.seed,
      "states_requested": cli_args.states,
      "states_collected": len(states),
      "states_evaluated": len(rows),
      "early_stop_reason": early_stop_reason,
      "matches_used": matches_used,
      "state_collection_report": getattr(args, "_state_collection_report", {}),
      "max_states_per_match": cli_args.max_states_per_match,
      "state_selection": cli_args.state_selection,
      "state_pool_per_match": cli_args.state_pool_per_match,
      "state_min_legal_count": cli_args.state_min_legal_count,
      "state_max_legal_count": cli_args.state_max_legal_count,
      "state_min_trick_number": cli_args.state_min_trick_number,
      "state_max_trick_number": cli_args.state_max_trick_number,
      "state_min_match_round": cli_args.state_min_match_round,
      "state_max_match_round": cli_args.state_max_match_round,
      "hard_entropy_weight": cli_args.hard_entropy_weight,
      "hard_loss_weight": cli_args.hard_loss_weight,
      "hard_paradox_weight": cli_args.hard_paradox_weight,
      "elapsed_sec": round(time.perf_counter() - started, 3),
      "phases": sorted(wanted),
      "label_rollouts": cli_args.label_rollouts,
      "audit_rollouts": cli_args.audit_rollouts,
      "stability_audit_passes": int(cli_args.stability_audit_passes),
      "stability_seed_stride": int(cli_args.stability_seed_stride),
      "stability_min_positive_sign_rate": float(
          cli_args.stability_min_positive_sign_rate
      ),
      "stability_min_safe_risk_rate": float(
          cli_args.stability_min_safe_risk_rate
      ),
      "unstable_positive_as_negative": bool(
          cli_args.unstable_positive_as_negative
      ),
      "label_seed_offset": cli_args.label_seed_offset,
      "audit_seed_offset": cli_args.audit_seed_offset,
      "belief_samples": cli_args.belief_samples,
      "belief_candidates": cli_args.belief_candidates,
      "belief_ref_policy_mix": cli_args.belief_ref_policy_mix,
      "label_max_actions": cli_args.label_max_actions,
      "label_top_policy": cli_args.label_top_policy,
      "label_include_bots": cli_args.label_include_bots,
      "min_label_margin": cli_args.min_label_margin,
      "label_paradox_penalty": cli_args.label_paradox_penalty,
      "label_max_paradox_delta": cli_args.label_max_paradox_delta,
      "min_audit_margin": cli_args.min_audit_margin,
      "pass_min_score_rate": cli_args.pass_min_score_rate,
      "pass_min_avg_diff": cli_args.pass_min_avg_diff,
      "pass_max_paradox_delta": cli_args.pass_max_paradox_delta,
      "full_match_rollout": bool(cli_args.full_match_rollout),
      "counterfactual_action_paradox_scope": str(
          cli_args.counterfactual_action_paradox_scope
      ),
      "rollout_max_plies": cli_args.rollout_max_plies,
      "accepted_replay_out": cli_args.accepted_replay_out,
      "accepted_replay_rows": len(accepted_replay),
      "accepted_replay_summary": accepted_replay_summary(accepted_replay),
      "calibration_replay_out": cli_args.calibration_replay_out,
      "calibration_replay_rows": len(calibration_replay),
      "calibration_positive_margin": cli_args.calibration_positive_margin,
      "calibration_max_positive_paradox_delta": (
          cli_args.calibration_max_positive_paradox_delta
      ),
      "calibration_include_policy_matches": bool(
          cli_args.calibration_include_policy_matches
      ),
      "calibration_target_positive_rows": int(
          cli_args.calibration_target_positive_rows
      ),
      "calibration_target_negative_rows": int(
          cli_args.calibration_target_negative_rows
      ),
      "calibration_summary": calibration_summary(calibration_metadata),
      "candidate_recall_audit_every": cli_args.candidate_recall_audit_every,
      "candidate_recall_summary": candidate_recall_summary(rows),
      "summary": summary,
      "by_phase": by_phase,
      "rows": rows,
  }
  if cli_args.accepted_replay_out:
    Path(cli_args.accepted_replay_out).parent.mkdir(parents=True, exist_ok=True)
    save_replay(accepted_replay, cli_args.accepted_replay_out)
  if cli_args.calibration_replay_out:
    calibration_path = Path(cli_args.calibration_replay_out)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    save_replay(calibration_replay, calibration_path)
    metadata_path = calibration_path.with_suffix(
        calibration_path.suffix + ".metadata.json"
    )
    metadata_path.write_text(
        json.dumps({
            "checkpoint": cli_args.checkpoint,
            "seed": cli_args.seed,
            "positive_margin": cli_args.calibration_positive_margin,
            "max_positive_paradox_delta": (
                cli_args.calibration_max_positive_paradox_delta
            ),
            "stability_audit_passes": int(cli_args.stability_audit_passes),
            "stability_seed_stride": int(cli_args.stability_seed_stride),
            "stability_min_positive_sign_rate": float(
                cli_args.stability_min_positive_sign_rate
            ),
            "stability_min_safe_risk_rate": float(
                cli_args.stability_min_safe_risk_rate
            ),
            "unstable_positive_as_negative": bool(
                cli_args.unstable_positive_as_negative
            ),
            "include_policy_matches": bool(
                cli_args.calibration_include_policy_matches
            ),
            "summary": calibration_summary(calibration_metadata),
            "rows": calibration_metadata,
        }, indent=2) + "\n"
    )
  out = Path(cli_args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  printable = {k: v for k, v in result.items() if k != "rows"}
  print(json.dumps(printable, indent=2), flush=True)


if __name__ == "__main__":
  main()
