#!/usr/bin/env python3
"""Audit whether MCTS top actions are good enough to distill.

This is a go/no-go diagnostic for top-action distillation. It collects real
champion-policy decision states, compares raw policy top actions to MCTS top
actions, filters for stable/high-confidence MCTS disagreements, then audits the
branch outcomes with common rollout seeds.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import make_bot
from quantum_cat_alphazero_torch import (
    action_feature_matrix,
    counterfactual_belief_states,
    load_model_payload,
    make_game,
    match_outcome_scores,
    mcts_policy,
    model_policy_value,
    phase_name_for_state,
    play_rollout_to_terminal,
    rollout_match_score_after_action,
    rollout_paradox_after_action,
    rollout_score_after_action,
    save_replay,
)


def _csv_ints(spec: str) -> list[int]:
  values = []
  for part in str(spec or "").split(","):
    part = part.strip()
    if part:
      values.append(int(part))
  return values


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--states", type=int, default=96)
  parser.add_argument("--max-matches", type=int, default=32)
  parser.add_argument(
      "--max-states-per-match",
      type=int,
      default=0,
      help="When >0, collect at most this many audited states per full match.",
  )
  parser.add_argument("--phases", default="play")
  parser.add_argument("--seed", type=int, default=20261910)
  parser.add_argument("--mcts-sims", default="12,32")
  parser.add_argument("--mcts-repeats", type=int, default=2)
  parser.add_argument("--mcts-belief-samples", type=int, default=1)
  parser.add_argument("--mcts-top-min-prob", type=float, default=0.25)
  parser.add_argument("--mcts-top-min-margin", type=float, default=0.03)
  parser.add_argument("--min-seed-stability", type=float, default=0.5)
  parser.add_argument("--min-budget-stability", type=float, default=0.5)
  parser.add_argument("--audit-rollouts", type=int, default=4)
  parser.add_argument("--audit-belief-samples", type=int, default=2)
  parser.add_argument("--full-match-rollout", action="store_true")
  parser.add_argument("--rollout-max-plies", type=int, default=0)
  parser.add_argument("--match-context", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--c-puct", type=float, default=1.5)
  parser.add_argument("--paradox-value-penalty", type=float, default=0.0)
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument("--action-paradox-root-only", action="store_true")
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-root-only", action="store_true")
  parser.add_argument("--audit-candidate-set", action="store_true")
  parser.add_argument("--candidate-top-policy-actions", type=int, default=2)
  parser.add_argument(
      "--candidate-random-actions",
      type=int,
      default=0,
      help=(
          "Add this many uniformly sampled legal actions to the candidate "
          "confirmation set. This reduces candidate-set bias when policy/MCTS/"
          "heuristic proposals miss the real best action."
      ),
  )
  parser.add_argument(
      "--candidate-all-legal-max",
      type=int,
      default=0,
      help=(
          "If positive and the legal action count is at most this value, "
          "evaluate every legal action in the candidate confirmation set."
      ),
  )
  parser.add_argument("--candidate-confirm-rollouts", type=int, default=0)
  parser.add_argument(
      "--candidate-confirm-all",
      action="store_true",
      help=(
          "When confirming candidate labels, independently evaluate every "
          "candidate action instead of only candidate-best and raw-top."
      ),
  )
  parser.add_argument("--candidate-min-confirm-margin", type=float, default=0.0)
  parser.add_argument(
      "--candidate-max-paradox-regression",
      type=float,
      default=0.0,
      help=(
          "Maximum allowed confirmed paradox increase for an admitted "
          "candidate label. Use a negative value to disable this filter."
      ),
  )
  parser.add_argument(
      "--candidate-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument(
      "--admitted-replay-out",
      default="",
      help=(
          "Optional replay .npz path containing only admitted candidate labels. "
          "Each row stores a one-hot policy target plus raw/candidate "
          "action-value targets from the independent confirmation audit."
      ),
  )
  parser.add_argument(
      "--admitted-replay-actions",
      choices=("best_raw", "confirmed"),
      default="best_raw",
      help=(
          "Which confirmed action-value labels to export for admitted replay "
          "rows. best_raw preserves the historical two-action target; "
          "confirmed exports every independently confirmed action."
      ),
  )
  parser.add_argument("--value-scale", type=float, default=20.0)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument(
      "--progress-interval",
      type=int,
      default=0,
      help="Print compact JSON progress every N audited states when >0.",
  )
  parser.add_argument("--out", default="az_runs/run191_mcts_top_audit/audit.json")
  return parser.parse_args()


def model_args_for(players, value_scale):
  return SimpleNamespace(
      players=players,
      arch="mlp",
      width=256,
      depth=3,
      value_scale=value_scale,
      match_context=True,
  )


def audit_namespace(cli_args, saved_args):
  return SimpleNamespace(
      players=int(cli_args.players),
      match_context=bool(cli_args.match_context),
      value_scale=float(saved_args.get("value_scale", cli_args.value_scale)),
      sims=max(_csv_ints(cli_args.mcts_sims) or [16]),
      c_puct=float(cli_args.c_puct),
      paradox_value_penalty=float(cli_args.paradox_value_penalty),
      action_paradox_selection_penalty=float(
          cli_args.action_paradox_selection_penalty
      ),
      action_paradox_root_only=bool(cli_args.action_paradox_root_only),
      action_value_selection_weight=float(cli_args.action_value_selection_weight),
      action_value_root_only=bool(cli_args.action_value_root_only),
      self_play_policy_mode="policy",
      self_play_belief_samples=1,
      self_play_belief_sims=1,
      league_opponent_mode="policy",
      eval_belief_samples=max(1, int(cli_args.audit_belief_samples)),
      counterfactual_full_match_rollout=bool(cli_args.full_match_rollout),
      full_match_training=bool(cli_args.match_context),
      counterfactual_rollout_max_plies=max(0, int(cli_args.rollout_max_plies)),
      counterfactual_rollout_opponent_mode="",
      counterfactual_belief_source="infostate",
      counterfactual_belief_samples=max(1, int(cli_args.audit_belief_samples)),
      counterfactual_belief_candidates=max(1, int(cli_args.audit_belief_samples)),
      counterfactual_belief_policy_temperature=1.0,
      counterfactual_belief_uniform_mix=0.0,
      counterfactual_belief_ref_policy_mix="model:1.0",
      counterfactual_belief_ref_policy_mix_by_phase="",
      counterfactual_belief_logprob_floor=1e-6,
  )


def action_string(state, player, action):
  try:
    return state._action_to_string(player, int(action))  # pylint:disable=protected-access
  except Exception:
    return str(action)


def admitted_replay_example(
    state,
    player,
    raw_action,
    best_action,
    confirm_audit,
    args,
    value_actions=None,
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
  if value_actions is None:
    value_actions = (int(raw_action), int(best_action))
  for action in value_actions:
    action = int(action)
    stats = confirm_audit.get(action)
    if not stats:
      continue
    mean_score = stats.get("mean_score")
    if mean_score is None:
      continue
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


def choose_policy_action(model, args, device, state, player):
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


def _sample_action(policy, legal, temperature):
  legal = list(legal)
  if temperature <= 0:
    return int(max(legal, key=lambda action: float(policy[action])))
  probs = np.array([max(0.0, float(policy[action])) for action in legal])
  total = float(np.sum(probs))
  if total <= 0 or not np.isfinite(total):
    probs = np.ones(len(legal), dtype=np.float64) / float(len(legal))
  else:
    probs = probs.astype(np.float64) / total
  probs = np.power(probs, 1.0 / max(1e-6, float(temperature)))
  probs /= float(np.sum(probs))
  return int(np.random.choice(legal, p=probs))


def collect_states(
    model, args, device, wanted_phases, max_states, max_matches,
    max_states_per_match=0,
):
  states = []
  matches = 0
  while len(states) < max_states and matches < max_matches:
    initial_start = matches % args.players
    totals = np.zeros(args.players, dtype=np.float32)
    collected_this_match = 0
    for round_idx in range(args.players):
      round_game = make_game(args, (initial_start + round_idx) % args.players)
      state = round_game.new_initial_state()
      if args.match_context:
        state.set_match_context(totals, round_idx)
      ply = 0
      while not state.is_terminal() and len(states) < max_states:
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(int(np.random.choice(actions, p=probs)))
          continue
        player = int(state.current_player())
        phase = phase_name_for_state(state)
        cap_open = (
            max_states_per_match <= 0
            or collected_this_match < max_states_per_match
        )
        if phase in wanted_phases and cap_open:
          states.append(state.clone())
          collected_this_match += 1
        policy, _ = choose_policy_action(model, args, device, state, player)
        state.apply_action(_sample_action(policy, state.legal_actions(player), 0.15 if ply < 8 else 0.0))
        ply += 1
      raw_scores = (
          state.raw_round_scores()
          if hasattr(state, "raw_round_scores")
          else state.returns()
      )
      totals += np.array(raw_scores, dtype=np.float32)
      if len(states) >= max_states:
        break
    matches += 1
  return states, matches


def mcts_policy_average(state, player, model, args, device, sims, seed, belief_samples):
  legal = state.legal_actions(player)
  legal_set = set(legal)
  policy = np.zeros(state.num_distinct_actions(), dtype=np.float32)
  samples_used = 0
  for sample_idx in range(max(1, int(belief_samples))):
    sample_seed = int(seed) + 1009 * sample_idx
    random.seed(sample_seed)
    np.random.seed(sample_seed % (2**32 - 1))
    if belief_samples > 1:
      try:
        sampled = state.resample_from_infostate(player, np.random)
      except Exception:
        sampled = state.clone()
    else:
      sampled = state.clone()
    if any(action not in sampled.legal_actions(player) for action in legal_set):
      continue
    random.seed((sample_seed + 17) % (2**32 - 1))
    np.random.seed((sample_seed + 17) % (2**32 - 1))
    policy += np.array(
        mcts_policy(sampled, model, args, device, add_noise=False, sims=sims),
        dtype=np.float32,
    )
    samples_used += 1
  if samples_used <= 0:
    return policy, 0
  policy /= float(samples_used)
  masked = np.zeros_like(policy)
  masked[legal] = np.maximum(policy[legal], 0.0)
  total = float(np.sum(masked[legal]))
  if total <= 0:
    masked[legal] = 1.0 / float(len(legal))
  else:
    masked /= total
  return masked, samples_used


def mcts_top_report(state, player, model, args, device, cli_args):
  sims_values = _csv_ints(cli_args.mcts_sims)
  if not sims_values:
    sims_values = [int(getattr(args, "sims", 16))]
  primary_sims = sims_values[-1]
  per_budget_tops = {}
  per_seed_tops = []
  primary_policy = np.zeros(state.num_distinct_actions(), dtype=np.float32)
  for sims in sims_values:
    budget_policy = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    budget_used = 0
    budget_tops = []
    for repeat in range(max(1, int(cli_args.mcts_repeats))):
      seed = int(cli_args.seed) + 7919 * repeat + 104729 * sims
      policy, samples_used = mcts_policy_average(
          state,
          player,
          model,
          args,
          device,
          sims,
          seed,
          max(1, int(cli_args.mcts_belief_samples)),
      )
      if samples_used <= 0:
        continue
      top = int(np.argmax(policy))
      budget_tops.append(top)
      budget_policy += policy
      budget_used += 1
      if sims == primary_sims:
        per_seed_tops.append(top)
    if budget_used > 0:
      budget_policy /= float(budget_used)
      per_budget_tops[int(sims)] = int(np.argmax(budget_policy))
      if sims == primary_sims:
        primary_policy = budget_policy
  legal = state.legal_actions(player)
  if float(np.sum(primary_policy[legal])) <= 0:
    primary_policy[legal] = 1.0 / float(len(legal))
  primary_top = int(max(legal, key=lambda action: primary_policy[action]))
  legal_probs = sorted((float(primary_policy[action]), int(action)) for action in legal)
  top_prob = legal_probs[-1][0]
  second_prob = legal_probs[-2][0] if len(legal_probs) > 1 else 0.0
  seed_stability = (
      float(np.mean([top == primary_top for top in per_seed_tops]))
      if per_seed_tops else 0.0
  )
  budget_tops = list(per_budget_tops.values())
  budget_stability = (
      float(np.mean([top == primary_top for top in budget_tops]))
      if budget_tops else 0.0
  )
  return {
      "policy": primary_policy,
      "top_action": primary_top,
      "top_prob": top_prob,
      "top_margin": float(top_prob - second_prob),
      "seed_stability": seed_stability,
      "budget_stability": budget_stability,
      "per_budget_tops": {str(k): int(v) for k, v in sorted(per_budget_tops.items())},
      "per_seed_tops": [int(x) for x in per_seed_tops],
      "primary_sims": int(primary_sims),
  }


def evaluate_actions(state, player, actions, round_game, model, args, device, rollouts):
  rollout_fn = (
      rollout_match_score_after_action
      if args.counterfactual_full_match_rollout
      else rollout_score_after_action
  )
  seat_roles = ["learner"] * args.players
  outcomes = {
      int(action): {
          "scores": [],
          "official_scores": [],
          "paradoxes": [],
          "totals": [],
          "final_round_scores": [],
      }
      for action in actions
  }
  belief_states = counterfactual_belief_states(
      state, player, args, model=model, device=device
  )
  for belief_idx, belief_state in enumerate(belief_states):
    legal = set(belief_state.legal_actions(player))
    if any(int(action) not in legal for action in actions):
      continue
    for rollout_idx in range(max(1, int(rollouts))):
      seed = (
          int(getattr(args, "seed", 0))
          + 1000003 * belief_idx
          + 9176 * rollout_idx
      )
      random.seed(seed)
      np.random.seed(seed % (2**32 - 1))
      py_state = random.getstate()
      np_state = np.random.get_state()
      for action in actions:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if (
            args.counterfactual_full_match_rollout
            and int(getattr(args, "counterfactual_rollout_max_plies", 0)) == 0
        ):
          detail = rollout_match_detail_after_action(
              belief_state,
              player,
              int(action),
              seat_roles,
              model,
              args,
              device,
          )
          outcomes[int(action)]["scores"].append(float(detail["score"]))
          outcomes[int(action)]["official_scores"].append(
              float(detail["official_score"])
          )
          outcomes[int(action)]["paradoxes"].append(float(detail["paradox_count"]))
          outcomes[int(action)]["totals"].append(detail["totals"])
          outcomes[int(action)]["final_round_scores"].append(
              detail["final_round_scores"]
          )
        else:
          score = rollout_fn(
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
          random.setstate(py_state)
          np.random.set_state(np_state)
          paradox = rollout_paradox_after_action(
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
          outcomes[int(action)]["scores"].append(float(score))
          outcomes[int(action)]["paradoxes"].append(float(paradox))
  return {
      int(action): {
          "mean_score": (
              float(np.mean(values["scores"])) if values["scores"] else None
          ),
          "mean_official_score": (
              float(np.mean(values["official_scores"]))
              if values["official_scores"] else None
          ),
          "mean_paradox": (
              float(np.mean(values["paradoxes"])) if values["paradoxes"] else None
          ),
          "mean_totals": (
              np.mean(np.array(values["totals"], dtype=np.float32), axis=0).tolist()
              if values["totals"] else None
          ),
          "mean_final_round_scores": (
              np.mean(
                  np.array(values["final_round_scores"], dtype=np.float32),
                  axis=0,
              ).tolist()
              if values["final_round_scores"] else None
          ),
          "n": int(len(values["scores"])),
      }
      for action, values in outcomes.items()
  }


def candidate_action_set(state, player, raw_policy, raw_top, mcts_top, cli_args):
  legal = list(state.legal_actions(player))
  legal_set = set(legal)
  actions = []

  def add(action):
    action = int(action)
    if action in legal_set and action not in actions:
      actions.append(action)

  add(raw_top)
  add(mcts_top)
  all_legal_max = max(0, int(getattr(cli_args, "candidate_all_legal_max", 0)))
  if all_legal_max > 0 and len(legal) <= all_legal_max:
    for action in legal:
      add(action)
  top_count = max(0, int(cli_args.candidate_top_policy_actions))
  if top_count > 0:
    for action in sorted(legal, key=lambda a: float(raw_policy[a]), reverse=True):
      add(action)
      if len([a for a in actions if a in legal_set]) >= top_count + 2:
        break
  for bot_name in [part.strip() for part in str(cli_args.candidate_bots).split(",") if part.strip()]:
    try:
      add(make_bot(bot_name, seed=0).step(state.clone(), player))
    except Exception:
      continue
  random_count = max(0, int(getattr(cli_args, "candidate_random_actions", 0)))
  if random_count > 0:
    unseen = [action for action in legal if int(action) not in set(actions)]
    random.shuffle(unseen)
    for action in unseen[:random_count]:
      add(action)
  return actions


def best_audited_action(audit):
  best_action = None
  best_score = -float("inf")
  for action_text, stats in audit.items():
    score = stats.get("mean_official_score")
    if score is None:
      score = stats.get("mean_score")
    if score is None:
      continue
    if float(score) > best_score:
      best_score = float(score)
      best_action = int(action_text)
  return best_action, best_score


def rollout_match_detail_after_action(
    state, acting_player, first_action, seat_roles, model, args, device
):
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

  rollout_state.apply_action(first_action)
  play_rollout_to_terminal(
      rollout_state,
      make_game(args, current_start),
      seat_roles,
      {},
      model,
      None,
      args,
      device,
  )
  raw_scores = (
      rollout_state.raw_round_scores()
      if hasattr(rollout_state, "raw_round_scores")
      else rollout_state.returns()
  )
  final_round_scores = np.array(raw_scores, dtype=np.float32)
  match_totals += final_round_scores
  paradoxes += np.array(
      getattr(rollout_state, "_has_paradoxed", [False] * args.players),
      dtype=np.float32,
  )

  for next_round in range(current_round + 1, args.players):
    start_player = (initial_start + next_round) % args.players
    next_game = make_game(args, start_player)
    next_state = next_game.new_initial_state()
    if getattr(args, "match_context", False):
      next_state.set_match_context(match_totals, next_round)
    play_rollout_to_terminal(
        next_state,
        next_game,
        seat_roles,
        {},
        model,
        None,
        args,
        device,
    )
    raw_scores = (
        next_state.raw_round_scores()
        if hasattr(next_state, "raw_round_scores")
        else next_state.returns()
    )
    final_round_scores = np.array(raw_scores, dtype=np.float32)
    match_totals += final_round_scores
    paradoxes += np.array(
        getattr(next_state, "_has_paradoxed", [False] * args.players),
        dtype=np.float32,
    )

  official_scores = match_outcome_scores(match_totals, final_round_scores)
  return {
      "score": float(match_totals[acting_player]),
      "official_score": float(official_scores[acting_player]),
      "paradox_count": float(paradoxes[acting_player]),
      "totals": [float(x) for x in match_totals],
      "final_round_scores": [float(x) for x in final_round_scores],
  }


def summarize(rows):
  summary = {
      "num_states": len(rows),
      "num_play_states": sum(1 for row in rows if row["phase"] == "play"),
      "num_disagreements": 0,
      "num_kept_high_confidence": 0,
      "kept_rate": 0.0,
      "mcts_top_vs_raw_wins": 0,
      "mcts_top_vs_raw_losses": 0,
      "mcts_top_vs_raw_ties": 0,
      "mcts_top_vs_raw_score_rate": None,
      "mcts_top_vs_raw_avg_diff": None,
      "mcts_top_vs_raw_official_wins": 0,
      "mcts_top_vs_raw_official_losses": 0,
      "mcts_top_vs_raw_official_ties": 0,
      "mcts_top_vs_raw_official_score_rate": None,
      "mcts_top_vs_raw_avg_official_diff": None,
      "candidate_best_vs_raw_wins": 0,
      "candidate_best_vs_raw_losses": 0,
      "candidate_best_vs_raw_ties": 0,
      "candidate_best_vs_raw_score_rate": None,
      "candidate_best_vs_raw_avg_diff": None,
      "candidate_best_vs_raw_official_wins": 0,
      "candidate_best_vs_raw_official_losses": 0,
      "candidate_best_vs_raw_official_ties": 0,
      "candidate_best_vs_raw_official_score_rate": None,
      "candidate_best_vs_raw_avg_official_diff": None,
      "candidate_best_is_raw": 0,
      "candidate_best_is_mcts": 0,
      "candidate_confirmed_count": 0,
      "candidate_admitted_count": 0,
      "candidate_admitted_wins": 0,
      "candidate_admitted_losses": 0,
      "candidate_admitted_ties": 0,
      "candidate_admitted_score_rate": None,
      "candidate_admitted_avg_diff": None,
      "candidate_admitted_avg_official_diff": None,
      "candidate_admitted_avg_paradox_delta": None,
      "mcts_top_paradox": None,
      "raw_top_paradox": None,
      "mean_top_action_stability_seed": None,
      "mean_top_action_stability_budget": None,
  }
  diffs = []
  official_diffs = []
  candidate_diffs = []
  candidate_official_diffs = []
  admitted_diffs = []
  admitted_official_diffs = []
  admitted_paradox_deltas = []
  mcts_paradoxes = []
  raw_paradoxes = []
  seed_stability = []
  budget_stability = []
  for row in rows:
    seed_stability.append(float(row["mcts_seed_stability"]))
    budget_stability.append(float(row["mcts_budget_stability"]))
    if row["mcts_top_action"] != row["raw_top_action"]:
      summary["num_disagreements"] += 1
    candidate_best = row.get("candidate_best_action")
    candidate_audit = row.get("candidate_confirm_audit") or row.get(
        "candidate_audit"
    )
    if candidate_best is not None and candidate_audit:
      if row.get("candidate_confirm_audit"):
        summary["candidate_confirmed_count"] += 1
      if int(candidate_best) == int(row["raw_top_action"]):
        summary["candidate_best_is_raw"] += 1
      if int(candidate_best) == int(row["mcts_top_action"]):
        summary["candidate_best_is_mcts"] += 1
      best_stats = candidate_audit.get(str(candidate_best))
      raw_stats = candidate_audit.get(str(row["raw_top_action"]))
      if best_stats and raw_stats:
        best_score = best_stats.get("mean_score")
        raw_score = raw_stats.get("mean_score")
        if best_score is not None and raw_score is not None:
          candidate_diff = float(best_score - raw_score)
          candidate_diffs.append(candidate_diff)
          if candidate_diff > 1e-9:
            summary["candidate_best_vs_raw_wins"] += 1
          elif candidate_diff < -1e-9:
            summary["candidate_best_vs_raw_losses"] += 1
          else:
            summary["candidate_best_vs_raw_ties"] += 1
        best_official = best_stats.get("mean_official_score")
        raw_official = raw_stats.get("mean_official_score")
        if best_official is not None and raw_official is not None:
          candidate_official_diff = float(best_official - raw_official)
          candidate_official_diffs.append(candidate_official_diff)
          if candidate_official_diff > 1e-9:
            summary["candidate_best_vs_raw_official_wins"] += 1
          elif candidate_official_diff < -1e-9:
            summary["candidate_best_vs_raw_official_losses"] += 1
          else:
            summary["candidate_best_vs_raw_official_ties"] += 1
      if row.get("candidate_label_admitted"):
        summary["candidate_admitted_count"] += 1
        admitted_margin = row.get("candidate_confirm_margin_vs_raw")
        admitted_official_margin = row.get(
            "candidate_confirm_official_margin_vs_raw"
        )
        admitted_paradox_delta = row.get("candidate_confirm_paradox_delta")
        if admitted_margin is not None:
          admitted_diffs.append(float(admitted_margin))
          if float(admitted_margin) > 1e-9:
            summary["candidate_admitted_wins"] += 1
          elif float(admitted_margin) < -1e-9:
            summary["candidate_admitted_losses"] += 1
          else:
            summary["candidate_admitted_ties"] += 1
        if admitted_official_margin is not None:
          admitted_official_diffs.append(float(admitted_official_margin))
        if admitted_paradox_delta is not None:
          admitted_paradox_deltas.append(float(admitted_paradox_delta))
    if not row.get("kept_high_confidence"):
      continue
    summary["num_kept_high_confidence"] += 1
    audit = row.get("audit", {})
    mcts = audit.get(str(row["mcts_top_action"]))
    raw = audit.get(str(row["raw_top_action"]))
    if not mcts or not raw:
      continue
    mcts_score = mcts.get("mean_score")
    raw_score = raw.get("mean_score")
    if mcts_score is None or raw_score is None:
      continue
    diff = float(mcts_score - raw_score)
    diffs.append(diff)
    if diff > 1e-9:
      summary["mcts_top_vs_raw_wins"] += 1
    elif diff < -1e-9:
      summary["mcts_top_vs_raw_losses"] += 1
    else:
      summary["mcts_top_vs_raw_ties"] += 1
    mcts_official = mcts.get("mean_official_score")
    raw_official = raw.get("mean_official_score")
    if mcts_official is not None and raw_official is not None:
      official_diff = float(mcts_official - raw_official)
      official_diffs.append(official_diff)
      if official_diff > 1e-9:
        summary["mcts_top_vs_raw_official_wins"] += 1
      elif official_diff < -1e-9:
        summary["mcts_top_vs_raw_official_losses"] += 1
      else:
        summary["mcts_top_vs_raw_official_ties"] += 1
    if mcts.get("mean_paradox") is not None:
      mcts_paradoxes.append(float(mcts["mean_paradox"]))
    if raw.get("mean_paradox") is not None:
      raw_paradoxes.append(float(raw["mean_paradox"]))
  if rows:
    summary["kept_rate"] = summary["num_kept_high_confidence"] / float(len(rows))
  comparisons = (
      summary["mcts_top_vs_raw_wins"]
      + summary["mcts_top_vs_raw_losses"]
      + summary["mcts_top_vs_raw_ties"]
  )
  if comparisons:
    summary["mcts_top_vs_raw_score_rate"] = (
        summary["mcts_top_vs_raw_wins"] + 0.5 * summary["mcts_top_vs_raw_ties"]
    ) / float(comparisons)
  if diffs:
    summary["mcts_top_vs_raw_avg_diff"] = float(np.mean(diffs))
  official_comparisons = (
      summary["mcts_top_vs_raw_official_wins"]
      + summary["mcts_top_vs_raw_official_losses"]
      + summary["mcts_top_vs_raw_official_ties"]
  )
  if official_comparisons:
    summary["mcts_top_vs_raw_official_score_rate"] = (
        summary["mcts_top_vs_raw_official_wins"]
        + 0.5 * summary["mcts_top_vs_raw_official_ties"]
    ) / float(official_comparisons)
  if official_diffs:
    summary["mcts_top_vs_raw_avg_official_diff"] = float(np.mean(official_diffs))
  candidate_comparisons = (
      summary["candidate_best_vs_raw_wins"]
      + summary["candidate_best_vs_raw_losses"]
      + summary["candidate_best_vs_raw_ties"]
  )
  if candidate_comparisons:
    summary["candidate_best_vs_raw_score_rate"] = (
        summary["candidate_best_vs_raw_wins"]
        + 0.5 * summary["candidate_best_vs_raw_ties"]
    ) / float(candidate_comparisons)
  if candidate_diffs:
    summary["candidate_best_vs_raw_avg_diff"] = float(np.mean(candidate_diffs))
  candidate_official_comparisons = (
      summary["candidate_best_vs_raw_official_wins"]
      + summary["candidate_best_vs_raw_official_losses"]
      + summary["candidate_best_vs_raw_official_ties"]
  )
  if candidate_official_comparisons:
    summary["candidate_best_vs_raw_official_score_rate"] = (
        summary["candidate_best_vs_raw_official_wins"]
        + 0.5 * summary["candidate_best_vs_raw_official_ties"]
    ) / float(candidate_official_comparisons)
  if candidate_official_diffs:
    summary["candidate_best_vs_raw_avg_official_diff"] = float(
        np.mean(candidate_official_diffs)
    )
  admitted_comparisons = (
      summary["candidate_admitted_wins"]
      + summary["candidate_admitted_losses"]
      + summary["candidate_admitted_ties"]
  )
  if admitted_comparisons:
    summary["candidate_admitted_score_rate"] = (
        summary["candidate_admitted_wins"]
        + 0.5 * summary["candidate_admitted_ties"]
    ) / float(admitted_comparisons)
  if admitted_diffs:
    summary["candidate_admitted_avg_diff"] = float(np.mean(admitted_diffs))
  if admitted_official_diffs:
    summary["candidate_admitted_avg_official_diff"] = float(
        np.mean(admitted_official_diffs)
    )
  if admitted_paradox_deltas:
    summary["candidate_admitted_avg_paradox_delta"] = float(
        np.mean(admitted_paradox_deltas)
    )
  if mcts_paradoxes:
    summary["mcts_top_paradox"] = float(np.mean(mcts_paradoxes))
  if raw_paradoxes:
    summary["raw_top_paradox"] = float(np.mean(raw_paradoxes))
  if seed_stability:
    summary["mean_top_action_stability_seed"] = float(np.mean(seed_stability))
  if budget_stability:
    summary["mean_top_action_stability_budget"] = float(np.mean(budget_stability))
  return summary


def main():
  cli_args = parse_args()
  random.seed(cli_args.seed)
  np.random.seed(cli_args.seed % (2**32 - 1))
  torch.manual_seed(cli_args.seed)
  if cli_args.device == "auto":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  else:
    device = torch.device(cli_args.device)
  load_args = model_args_for(cli_args.players, cli_args.value_scale)
  load_game = pyspiel.load_game(
      "python_quantum_cat",
      {"players": cli_args.players, "match_context": int(cli_args.match_context)},
  )
  model, _, saved_args = load_model_payload(
      cli_args.checkpoint, load_game, load_args, device
  )
  model.eval()
  args = audit_namespace(cli_args, saved_args)
  args.seed = int(cli_args.seed)
  wanted = {part.strip() for part in cli_args.phases.split(",") if part.strip()}
  started = time.perf_counter()
  states, matches_used = collect_states(
      model,
      args,
      device,
      wanted,
      max(1, int(cli_args.states)),
      max(1, int(cli_args.max_matches)),
      max(0, int(cli_args.max_states_per_match)),
  )
  rows = []
  admitted_replay = []
  for idx, state in enumerate(states):
    player = int(state.current_player())
    phase = phase_name_for_state(state)
    round_game = make_game(args, int(getattr(state, "_round_start_player", 0)))
    raw_policy, raw_top = choose_policy_action(model, args, device, state, player)
    mcts = mcts_top_report(state, player, model, args, device, cli_args)
    mcts_top = int(mcts["top_action"])
    kept = (
        mcts_top != raw_top
        and float(mcts["top_prob"]) >= float(cli_args.mcts_top_min_prob)
        and float(mcts["top_margin"]) >= float(cli_args.mcts_top_min_margin)
        and float(mcts["seed_stability"]) >= float(cli_args.min_seed_stability)
        and float(mcts["budget_stability"]) >= float(cli_args.min_budget_stability)
    )
    row = {
        "index": idx,
        "phase": phase,
        "player": player,
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "legal_count": int(len(state.legal_actions(player))),
        "raw_top_action": int(raw_top),
        "raw_top_action_string": action_string(state, player, raw_top),
        "raw_top_prob": float(raw_policy[raw_top]),
        "mcts_top_action": mcts_top,
        "mcts_top_action_string": action_string(state, player, mcts_top),
        "mcts_top_prob": float(mcts["top_prob"]),
        "mcts_top_margin": float(mcts["top_margin"]),
        "mcts_seed_stability": float(mcts["seed_stability"]),
        "mcts_budget_stability": float(mcts["budget_stability"]),
        "mcts_per_budget_tops": mcts["per_budget_tops"],
        "mcts_per_seed_tops": mcts["per_seed_tops"],
        "mcts_primary_sims": int(mcts["primary_sims"]),
        "kept_high_confidence": bool(kept),
    }
    if kept:
      audit = evaluate_actions(
          state,
          player,
          [mcts_top, raw_top],
          round_game,
          model,
          args,
          device,
          cli_args.audit_rollouts,
      )
      row["audit"] = {str(k): v for k, v in sorted(audit.items())}
      mcts_mean = audit[mcts_top]["mean_score"]
      raw_mean = audit[raw_top]["mean_score"]
      row["audit_margin_vs_raw"] = (
          None
          if mcts_mean is None or raw_mean is None
          else float(mcts_mean - raw_mean)
      )
    if cli_args.audit_candidate_set:
      candidates = candidate_action_set(
          state, player, raw_policy, raw_top, mcts_top, cli_args
      )
      if len(candidates) > 1:
        candidate_audit = evaluate_actions(
            state,
            player,
            candidates,
            round_game,
            model,
            args,
            device,
            cli_args.audit_rollouts,
        )
        candidate_audit_text = {
            str(k): v for k, v in sorted(candidate_audit.items())
        }
        best_action, best_score = best_audited_action(candidate_audit_text)
        row["candidate_actions"] = [int(action) for action in candidates]
        row["candidate_action_strings"] = {
            str(action): action_string(state, player, action)
            for action in candidates
        }
        row["candidate_audit"] = candidate_audit_text
        row["candidate_best_action"] = best_action
        row["candidate_best_action_string"] = (
            action_string(state, player, best_action)
            if best_action is not None else None
        )
        row["candidate_best_score"] = best_score
        confirm_rollouts = max(0, int(cli_args.candidate_confirm_rollouts))
        if confirm_rollouts > 0 and best_action is not None:
          original_seed = int(getattr(args, "seed", 0))
          args.seed = original_seed + 424242
          confirm_actions = (
              candidates if cli_args.candidate_confirm_all
              else [best_action, raw_top]
          )
          confirm_audit = evaluate_actions(
              state,
              player,
              confirm_actions,
              round_game,
              model,
              args,
              device,
              confirm_rollouts,
          )
          args.seed = original_seed
          row["candidate_confirm_audit"] = {
              str(k): v for k, v in sorted(confirm_audit.items())
          }
          best_stats = confirm_audit.get(int(best_action))
          raw_stats = confirm_audit.get(int(raw_top))
          if best_stats and raw_stats:
            best_score = best_stats.get("mean_score")
            raw_score = raw_stats.get("mean_score")
            best_official = best_stats.get("mean_official_score")
            raw_official = raw_stats.get("mean_official_score")
            best_paradox = best_stats.get("mean_paradox")
            raw_paradox = raw_stats.get("mean_paradox")
            margin = (
                None
                if best_score is None or raw_score is None
                else float(best_score - raw_score)
            )
            official_margin = (
                None
                if best_official is None or raw_official is None
                else float(best_official - raw_official)
            )
            paradox_delta = (
                None
                if best_paradox is None or raw_paradox is None
                else float(best_paradox - raw_paradox)
            )
            max_paradox_delta = float(cli_args.candidate_max_paradox_regression)
            row["candidate_confirm_margin_vs_raw"] = margin
            row["candidate_confirm_official_margin_vs_raw"] = official_margin
            row["candidate_confirm_paradox_delta"] = paradox_delta
            row["candidate_label_admitted"] = bool(
                int(best_action) != int(raw_top)
                and margin is not None
                and margin >= float(cli_args.candidate_min_confirm_margin)
                and (
                    max_paradox_delta < 0
                    or paradox_delta is not None
                    and paradox_delta <= max_paradox_delta
                )
            )
            if row["candidate_label_admitted"]:
              replay_value_actions = (
                  sorted(confirm_audit)
                  if cli_args.admitted_replay_actions == "confirmed"
                  else [raw_top, best_action]
              )
              admitted_replay.append(
                  admitted_replay_example(
                      state,
                      player,
                      raw_top,
                      best_action,
                      confirm_audit,
                      args,
                      value_actions=replay_value_actions,
                  )
              )
    rows.append(row)
    progress_interval = max(0, int(cli_args.progress_interval))
    if progress_interval and (idx + 1) % progress_interval == 0:
      print(
          json.dumps({
              "progress": idx + 1,
              "states_collected": len(states),
              "elapsed_sec": round(time.perf_counter() - started, 3),
              "admitted_replay_count": len(admitted_replay),
          }),
          flush=True,
      )

  admitted_replay_out = str(cli_args.admitted_replay_out or "")
  if admitted_replay_out and admitted_replay:
    replay_out = Path(admitted_replay_out)
    replay_out.parent.mkdir(parents=True, exist_ok=True)
    save_replay(admitted_replay, replay_out)

  result = {
      "checkpoint": cli_args.checkpoint,
      "players": int(cli_args.players),
      "seed": int(cli_args.seed),
      "states_requested": int(cli_args.states),
      "states_collected": len(states),
      "matches_used": int(matches_used),
      "max_states_per_match": int(cli_args.max_states_per_match),
      "elapsed_sec": round(time.perf_counter() - started, 3),
      "phases": sorted(wanted),
      "mcts_sims": _csv_ints(cli_args.mcts_sims),
      "mcts_repeats": int(cli_args.mcts_repeats),
      "mcts_belief_samples": int(cli_args.mcts_belief_samples),
      "mcts_top_min_prob": float(cli_args.mcts_top_min_prob),
      "mcts_top_min_margin": float(cli_args.mcts_top_min_margin),
      "min_seed_stability": float(cli_args.min_seed_stability),
      "min_budget_stability": float(cli_args.min_budget_stability),
      "audit_rollouts": int(cli_args.audit_rollouts),
      "audit_belief_samples": int(cli_args.audit_belief_samples),
      "full_match_rollout": bool(cli_args.full_match_rollout),
      "rollout_max_plies": int(cli_args.rollout_max_plies),
      "audit_candidate_set": bool(cli_args.audit_candidate_set),
      "candidate_top_policy_actions": int(cli_args.candidate_top_policy_actions),
      "candidate_random_actions": int(cli_args.candidate_random_actions),
      "candidate_all_legal_max": int(cli_args.candidate_all_legal_max),
      "candidate_confirm_rollouts": int(cli_args.candidate_confirm_rollouts),
      "candidate_confirm_all": bool(cli_args.candidate_confirm_all),
      "candidate_min_confirm_margin": float(cli_args.candidate_min_confirm_margin),
      "candidate_max_paradox_regression": float(
          cli_args.candidate_max_paradox_regression
      ),
      "candidate_bots": cli_args.candidate_bots,
      "admitted_replay_out": admitted_replay_out,
      "admitted_replay_actions": str(cli_args.admitted_replay_actions),
      "admitted_replay_count": len(admitted_replay),
      "summary": summarize(rows),
      "rows": rows,
  }
  out = Path(cli_args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  printable = {k: v for k, v in result.items() if k != "rows"}
  print(json.dumps(printable, indent=2), flush=True)


if __name__ == "__main__":
  main()
