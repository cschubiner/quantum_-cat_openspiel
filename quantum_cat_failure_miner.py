#!/usr/bin/env python3
"""Mine avoidable same-policy paradox failures from homogeneous self-play.

This script plays full Cat in the Box matches with the same raw neural policy in
every seat. For rounds that end with any player paradoxing, it replays late
decisions from the exact simulator state and tests every legal replacement
action under the same-policy continuation. A decision is "avoidable" when the
chosen action leads to an any-player paradox but at least one legal replacement
does not.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from quantum_cat_alphazero_torch import (
    action_feasibility_scores,
    action_feature_matrix,
    model_action_risks,
    model_action_values,
    model_policy_probs,
    q_policy_select_action,
    save_replay,
)
from quantum_cat_full_match_elo import (
    action_liveness_certificate,
    action_liveness_key,
    load_neural,
)


PHASE_NAMES = {
    0: "chance",
    1: "discard",
    2: "prediction",
    3: "play",
    4: "terminal",
}


_ACTOR_BOT_MODES = {
    "policy",
    "liveness_policy",
    "q_policy",
    "q_liveness_policy",
    "liveness_shield",
}
_LIVENESS_SHIELD_OVERRIDE_TYPES = {
    "liveness_shield_base_mode": str,
    "liveness_shield_phases": str,
    "liveness_shield_min_trick_number": int,
    "liveness_shield_led_colors": str,
    "liveness_shield_min_open_slot_delta": int,
    "liveness_shield_min_public_damage_delta": int,
    "liveness_shield_max_policy_log_gap": float,
}


def resolve_actor_bot_args(args):
  """Resolve a replay actor spec without silently degrading its policy mode."""
  spec = str(getattr(args, "bot_mode", "policy") or "policy").strip()
  mode, separator, payload = spec.partition("?")
  mode = mode.strip().lower().replace("-", "_")
  if mode not in _ACTOR_BOT_MODES:
    allowed = ", ".join(sorted(_ACTOR_BOT_MODES))
    raise ValueError(f"unsupported actor bot mode {mode!r}; allowed: {allowed}")

  resolved = SimpleNamespace(**vars(args))
  if separator:
    if mode != "liveness_shield":
      raise ValueError(
          "actor bot overrides are only supported for liveness_shield"
      )
    for part in payload.split(","):
      if not part.strip():
        continue
      if "=" not in part:
        raise ValueError(
            f"bad liveness_shield override {part!r}; expected key=value"
        )
      key, value = part.split("=", 1)
      key = key.strip().replace("-", "_")
      if key not in _LIVENESS_SHIELD_OVERRIDE_TYPES:
        allowed = ", ".join(sorted(_LIVENESS_SHIELD_OVERRIDE_TYPES))
        raise ValueError(
            f"unsupported liveness_shield override {key!r}; allowed: {allowed}"
        )
      value_type = _LIVENESS_SHIELD_OVERRIDE_TYPES[key]
      setattr(resolved, key, value_type(value.strip()))

  if mode == "liveness_shield":
    base_mode = str(
        getattr(resolved, "liveness_shield_base_mode", "q_policy")
    ).lower().replace("-", "_")
    if base_mode not in ("policy", "q_policy"):
      raise ValueError(
          "liveness_shield_base_mode must be policy or q_policy"
      )
    mode = "liveness_policy" if base_mode == "policy" else "q_liveness_policy"
  resolved.bot_mode = mode
  return resolved


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=30)
  parser.add_argument("--seed", type=int, default=20263210)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--lookback-decisions", type=int, default=8)
  parser.add_argument("--max-failure-rounds", type=int, default=30)
  parser.add_argument("--max-decisions", type=int, default=0)
  parser.add_argument(
      "--decision-phases",
      default="",
      help=(
          "Optional comma-separated phase filter for evaluated decisions, e.g. "
          "discard,prediction,play. Empty keeps all phases."
      ),
  )
  parser.add_argument(
      "--max-decisions-per-failure-round",
      type=int,
      default=0,
      help=(
          "Maximum evaluated decisions per mined failure round after filtering. "
          "0 disables the per-round cap."
      ),
  )
  parser.add_argument(
      "--progress-interval",
      type=int,
      default=0,
      help="Print a compact progress row every N evaluated decisions.",
  )
  parser.add_argument("--min-score-delta", type=float, default=-99.0)
  parser.add_argument("--target-temperature", type=float, default=0.05)
  parser.add_argument(
      "--bot-mode",
      default="policy",
      help=(
          "Selection rule used for homogeneous self-play and counterfactual "
          "continuations. Accepts policy, q_policy, liveness_policy, "
          "q_liveness_policy, or a liveness_shield?... selector payload. "
          "The selector is validated and resolved before model execution."
      ),
  )
  parser.add_argument("--liveness-shield-phases", default="discard,prediction,play")
  parser.add_argument("--liveness-shield-min-open-slot-delta", type=int, default=1)
  parser.add_argument(
      "--liveness-shield-min-public-damage-delta", type=int, default=2
  )
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
  parser.add_argument("--out", required=True)
  parser.add_argument("--replay-out", default="")
  parser.add_argument(
      "--dense-replay-out",
      default="",
      help=(
          "Optional replay with one row per evaluated decision. All legal "
          "actions receive same-policy counterfactual risk/value labels, and "
          "the policy target is derived lexicographically from any-paradox, "
          "forced-late paradox, actor paradox, then score."
      ),
  )
  parser.add_argument(
      "--dense-target-temperature",
      type=float,
      default=0.05,
      help="Softmax temperature for dense replay policy targets.",
  )
  parser.add_argument(
      "--dense-action-value-objective",
      choices=("safety", "safety_raw", "score"),
      default="safety",
      help=(
          "Per-action value target for dense replay. 'safety' uses bounded "
          "anti-paradox targets compatible with tanh action-value heads; "
          "'safety_raw' keeps the older unbounded safety penalties; 'score' "
          "keeps raw acting score delta."
      ),
  )
  parser.add_argument(
      "--dense-safety-value-safe-target",
      type=float,
      default=0.9,
      help="Bounded dense action-value target for non-paradox outcomes.",
  )
  parser.add_argument(
      "--dense-safety-value-paradox-target",
      type=float,
      default=-0.55,
      help=(
          "Bounded dense action-value baseline for outcomes where any player "
          "paradoxes."
      ),
  )
  parser.add_argument(
      "--dense-safety-value-forced-penalty",
      type=float,
      default=0.25,
      help="Extra bounded penalty for outcomes that end in a forced paradox.",
  )
  parser.add_argument(
      "--dense-safety-value-actor-penalty",
      type=float,
      default=0.10,
      help="Extra bounded penalty when the acting player is one of the paradoxers.",
  )
  parser.add_argument(
      "--dense-safety-value-score-weight",
      type=float,
      default=0.05,
      help=(
          "Small score-delta shaping term added after the safety target is chosen."
      ),
  )
  parser.add_argument(
      "--dense-safety-value-min",
      type=float,
      default=-0.95,
      help="Lower clip for bounded dense safety action-value targets.",
  )
  parser.add_argument(
      "--dense-safety-value-max",
      type=float,
      default=0.95,
      help="Upper clip for bounded dense safety action-value targets.",
  )
  parser.add_argument(
      "--calibration-replay-out",
      default="",
      help=(
          "Optional calibration replay for the improve-graft classifier. "
          "Writes a sibling .metadata.json file."
      ),
  )
  parser.add_argument(
      "--calibration-max-negatives-per-decision",
      type=int,
      default=4,
      help="Cap non-avoiding candidate rows per evaluated decision.",
  )
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  return parser.parse_args()


def phase_name(state):
  return PHASE_NAMES.get(int(getattr(state, "_phase", -1)), "unknown")


def parse_phase_filter(spec):
  return {
      item.strip().lower() for item in str(spec or "").split(",")
      if item.strip()
  }


def should_evaluate_decision(decision, allowed_phases):
  if not allowed_phases:
    return True
  return str(decision.get("phase", "")).lower() in allowed_phases


def reached_mining_cap(args, summary):
  max_decisions = int(getattr(args, "max_decisions", 0))
  if max_decisions > 0 and summary["decisions_evaluated"] >= max_decisions:
    return True
  max_failure_rounds = int(getattr(args, "max_failure_rounds", 0))
  if (
      max_failure_rounds > 0
      and summary["failure_rounds_mined"] >= max_failure_rounds
  ):
    return True
  return False


def action_string(state, player, action):
  try:
    return state._action_to_string(player, int(action))  # pylint: disable=protected-access
  except Exception:
    return str(int(action))


def _new_paradox_players(before_paradox, after_paradox):
  return [
      idx for idx, (before, after) in enumerate(
          zip(before_paradox, after_paradox)
      )
      if bool(after) and not bool(before)
  ]


def _paradox_trigger_row(
    state_before,
    player,
    action,
    legal_count,
    before_paradox,
    after_paradox,
    continuation_decision_index,
):
  new_players = _new_paradox_players(before_paradox, after_paradox)
  if not new_players:
    return None
  return {
      "phase": phase_name(state_before),
      "player": int(player),
      "action": int(action),
      "action_string": action_string(state_before, player, action),
      "legal_count": int(legal_count),
      "trick_number": int(getattr(state_before, "_trick_number", -1)),
      "new_paradox_players": [int(item) for item in new_players],
      "continuation_decision_index": int(continuation_decision_index),
  }


def rerank_enabled_for_phase(args, state):
  spec = str(getattr(args, "action_value_rerank_phases", "") or "")
  if not spec.strip():
    return True
  allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
  return phase_name(state) in allowed


def split_phase_names(spec):
  normalized = str(spec or "").replace("|", ",").replace("+", ",")
  return {part.strip().lower() for part in normalized.split(",") if part.strip()}


def liveness_enabled_for_phase(args, state):
  spec = str(getattr(args, "liveness_shield_phases", "") or "").strip()
  if spec and phase_name(state).lower() not in split_phase_names(spec):
    return False
  min_trick = int(getattr(args, "liveness_shield_min_trick_number", -1))
  if min_trick >= 0 and int(getattr(state, "_trick_number", -1)) < min_trick:
    return False
  led_spec = str(getattr(args, "liveness_shield_led_colors", "") or "")
  if led_spec.strip():
    allowed = {
        part.strip().upper()
        for part in led_spec.replace("|", ",").split(",")
        if part.strip()
    }
    led_color = getattr(state, "_led_color", None)
    led_name = "NONE" if led_color is None else str(led_color).upper()
    if led_name not in allowed:
      return False
  return True


def within_liveness_policy_gap(args, policy, base_action, candidate_action):
  max_gap = float(getattr(args, "liveness_shield_max_policy_log_gap", -1.0))
  if max_gap < 0:
    return True
  gap = math.log(max(float(policy[int(base_action)]), 1e-12)) - math.log(
      max(float(policy[int(candidate_action)]), 1e-12)
  )
  return float(gap) <= max_gap


def liveness_candidate_dominates_base(base_row, candidate_row, args):
  if bool(candidate_row.get("is_paradox", False)):
    return False
  if bool(base_row.get("is_paradox", False)):
    return True
  public_not_worse = (
      int(candidate_row.get("min_player_open_slots_after", 0)) >=
      int(base_row.get("min_player_open_slots_after", 0))
      and int(candidate_row.get("total_player_open_slots_after", 0)) >=
      int(base_row.get("total_player_open_slots_after", 0))
      and int(candidate_row.get("own_public_slots_after", 0)) >=
      int(base_row.get("own_public_slots_after", 0))
      and int(candidate_row.get("min_player_lane_surplus_after", 0)) >=
      int(base_row.get("min_player_lane_surplus_after", 0))
      and int(candidate_row.get("total_player_lane_surplus_after", 0)) >=
      int(base_row.get("total_player_lane_surplus_after", 0))
      and int(candidate_row.get("lane_pressure_player_count_after", 0)) <=
      int(base_row.get("lane_pressure_player_count_after", 0))
      and int(candidate_row.get("public_slot_damage", 0)) <=
      int(base_row.get("public_slot_damage", 0))
      and int(candidate_row.get("own_public_slot_damage", 0)) <=
      int(base_row.get("own_public_slot_damage", 0))
  )
  base_deficit = int(base_row.get("own_total_deficit", 0))
  candidate_deficit = int(candidate_row.get("own_total_deficit", 0))
  if candidate_deficit < base_deficit:
    return True
  if candidate_deficit > base_deficit:
    return False
  if (
      bool(candidate_row.get("own_feasible", False))
      and not bool(base_row.get("own_feasible", False))
  ):
    return True
  if (
      bool(base_row.get("lost_led_token", False))
      and not bool(candidate_row.get("lost_led_token", False))
  ):
    return True
  if (
      int(candidate_row.get("min_player_lane_surplus_after", 0)) >
      int(base_row.get("min_player_lane_surplus_after", 0))
  ):
    return True
  if (
      int(candidate_row.get("lane_pressure_player_count_after", 0)) <
      int(base_row.get("lane_pressure_player_count_after", 0))
  ):
    return True
  if (
      int(candidate_row.get("own_dead_rank_count", 0)) <
      int(base_row.get("own_dead_rank_count", 0))
      and public_not_worse
  ):
    return True
  if (
      int(candidate_row.get("own_buffer_deficit", 0)) <
      int(base_row.get("own_buffer_deficit", 0))
      and public_not_worse
  ):
    return True
  open_delta = int(candidate_row.get("min_player_open_slots_after", 0)) - int(
      base_row.get("min_player_open_slots_after", 0)
  )
  if open_delta >= int(getattr(args, "liveness_shield_min_open_slot_delta", 1)):
    return True
  damage_delta = int(base_row.get("public_slot_damage", 0)) - int(
      candidate_row.get("public_slot_damage", 0)
  )
  if damage_delta >= int(
      getattr(args, "liveness_shield_min_public_damage_delta", 2)
  ):
    return True
  if (
      bool(base_row.get("over_target_would_win", False))
      and not bool(candidate_row.get("over_target_would_win", False))
      and damage_delta >= 0
  ):
    return True
  return False


def liveness_policy_action_from_policy(
    state, player, legal, policy, args, base_action=None
):
  if base_action is None:
    base_action = int(max(legal, key=lambda action: float(policy[action])))
  else:
    base_action = int(base_action)
  if len(legal) <= 1 or not liveness_enabled_for_phase(args, state):
    return base_action
  scored_legal = sorted(int(action) for action in legal)
  non_paradox = [action for action in scored_legal if action != 999]
  if non_paradox:
    scored_legal = non_paradox
  scored = {
      action: action_liveness_certificate(
          state, player, action, base_action=base_action
      )
      for action in scored_legal
  }
  if not scored:
    return base_action
  base_row = scored.get(base_action)
  if base_row is None:
    base_row = action_liveness_certificate(
        state, player, base_action, base_action=base_action
    )
  selected_action, selected_row = max(
      scored.items(),
      key=lambda item: action_liveness_key(
          item[1], item[0], base_action=base_action
      ),
  )
  selected_action = int(selected_action)
  if (
      selected_action != base_action
      and liveness_candidate_dominates_base(base_row, selected_row, args)
      and within_liveness_policy_gap(args, policy, base_action, selected_action)
  ):
    return selected_action
  return base_action


def choose_policy_action(model, state, player, value_scale, device, args):
  args = resolve_actor_bot_args(args)
  policy = model_policy_probs(
      model,
      state,
      player,
      state.num_distinct_actions(),
      device,
  )
  legal = list(state.legal_actions(player))
  action = max(legal, key=lambda legal_action: float(policy[legal_action]))
  bot_mode = str(getattr(args, "bot_mode", "policy"))
  if bot_mode == "liveness_policy":
    action = liveness_policy_action_from_policy(state, player, legal, policy, args)
    return maybe_explore_behavior_action(state, legal, action, policy, args), policy
  use_q_base = bot_mode in ("q_policy", "q_liveness_policy")
  if not use_q_base:
    return maybe_explore_behavior_action(state, legal, action, policy, args), policy
  if len(legal) <= 1 or not rerank_enabled_for_phase(args, state):
    if bot_mode == "q_liveness_policy":
      action = liveness_policy_action_from_policy(
          state, player, legal, policy, args, base_action=action
      )
    return maybe_explore_behavior_action(state, legal, action, policy, args), policy

  value_weight = float(getattr(args, "action_value_selection_weight", 0.0))
  risk_penalty = float(getattr(args, "action_paradox_selection_penalty", 0.0))
  risk_mode = str(
      getattr(args, "action_paradox_rerank_mode", "additive") or "additive"
  ).lower()
  if (
      value_weight <= 0.0
      and risk_penalty <= 0.0
      and float(getattr(args, "action_feasibility_selection_weight", 0.0)) == 0.0
      and risk_mode not in ("threshold", "relative")
  ):
    if bot_mode == "q_liveness_policy":
      action = liveness_policy_action_from_policy(
          state, player, legal, policy, args, base_action=action
      )
    return maybe_explore_behavior_action(state, legal, action, policy, args), policy

  num_actions = int(state.num_distinct_actions())
  action_values = (
      model_action_values(model, state, player, num_actions, device)
      if value_weight > 0.0 else
      np.zeros(num_actions, dtype=np.float32)
  )
  clip = max(0.0, float(getattr(args, "action_value_rerank_clip", 0.5)))
  if clip > 0.0:
    action_values = np.clip(action_values, -clip, clip)
  min_margin = float(getattr(args, "action_value_rerank_min_margin", 0.0))
  if min_margin > 0.0 and len(legal) > 1:
    legal_values = np.array([float(action_values[item]) for item in legal])
    top_two = np.partition(legal_values, -2)[-2:]
    value_margin = float(np.max(top_two) - np.min(top_two))
    if value_margin < min_margin:
      if bot_mode == "q_liveness_policy":
        action = liveness_policy_action_from_policy(
            state, player, legal, policy, args, base_action=action
        )
      return maybe_explore_behavior_action(state, legal, action, policy, args), policy

  action_risks = (
      model_action_risks(model, state, player, num_actions, device)
      if risk_penalty > 0.0 or risk_mode in ("threshold", "relative") else
      None
  )
  feasibility_scores = (
      action_feasibility_scores(action_feature_matrix(state, player, num_actions))
      if float(getattr(args, "action_feasibility_selection_weight", 0.0)) != 0.0
      else None
  )
  action = q_policy_select_action(
      legal,
      policy,
      action_values,
      action_risks,
      args,
      value_scale,
      feasibility_scores=feasibility_scores,
  )
  if bot_mode == "q_liveness_policy":
    action = liveness_policy_action_from_policy(
        state, player, legal, policy, args, base_action=action
    )
  action = maybe_explore_behavior_action(state, legal, action, policy, args)
  return int(action), policy


def _action_color(state, action):
  action = int(action)
  if action == 999:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  if num_card_types <= 0:
    return None
  color_idx = action // num_card_types
  colors = ("R", "B", "Y", "G")
  if 0 <= color_idx < len(colors):
    return colors[color_idx]
  return None


def _sample_from_policy(policy, candidates, temperature):
  if not candidates:
    return None
  temperature = max(1e-6, float(temperature))
  logits = np.array([
      math.log(max(float(policy[int(action)]), 1e-12)) / temperature
      for action in candidates
  ], dtype=np.float64)
  logits -= float(np.max(logits))
  probs = np.exp(logits)
  probs /= probs.sum()
  return int(np.random.choice([int(action) for action in candidates], p=probs))


def maybe_explore_behavior_action(state, legal, action, policy, args):
  """Optional data-collection exploration; disabled by default."""
  if not bool(getattr(args, "enable_behavior_exploration", False)):
    return int(action)
  if len(legal) <= 1:
    return int(action)
  led_switch_prob = float(getattr(args, "behavior_led_contrast_switch_prob", 0.0))
  led_color = getattr(state, "_led_color", None)
  if led_switch_prob > 0 and led_color in ("B", "Y", "G"):
    led_legal = [
        int(item) for item in legal if _action_color(state, item) == led_color
    ]
    switch_legal = [
        int(item) for item in legal
        if item != 999 and _action_color(state, item) not in (None, led_color)
    ]
    if led_legal and switch_legal and np.random.random() < led_switch_prob:
      red_switch_legal = [
          item for item in switch_legal if _action_color(state, item) == "R"
      ]
      red_prob = float(getattr(args, "behavior_red_switch_prob", 0.75))
      pool = (
          red_switch_legal
          if red_switch_legal and np.random.random() < red_prob
          else switch_legal
      )
      sampled = _sample_from_policy(
          policy,
          pool,
          float(getattr(args, "behavior_exploration_temperature", 1.0)),
      )
      if sampled is not None:
        return sampled
  epsilon = float(getattr(args, "behavior_exploration_prob", 0.0))
  if epsilon <= 0 or np.random.random() >= epsilon:
    return int(action)
  sampled = _sample_from_policy(
      policy,
      [int(item) for item in legal],
      float(getattr(args, "behavior_exploration_temperature", 1.0)),
  )
  return int(action if sampled is None else sampled)


def terminal_round_result(state):
  paradoxes = [bool(value) for value in getattr(state, "_has_paradoxed", [])]
  raw_scores = (
      state.raw_round_scores()
      if hasattr(state, "raw_round_scores")
      else state.returns()
  )
  return {
      "any_paradox": bool(any(paradoxes)),
      "paradoxed": paradoxes,
      "raw_scores": [float(value) for value in raw_scores],
  }


def rollout_from_action(
    model, state_before, first_action, value_scale, device, seed, args
):
  if seed is not None:
    np.random.seed(int(seed))
  rollout = state_before.clone()
  first_player = int(state_before.current_player())
  first_legal_count = len(state_before.legal_actions(first_player))
  before_paradox = list(getattr(rollout, "_has_paradoxed", []))
  rollout.apply_action(int(first_action))
  after_paradox = list(getattr(rollout, "_has_paradoxed", []))
  first_new_paradox_trigger = _paradox_trigger_row(
      state_before,
      first_player,
      first_action,
      first_legal_count,
      before_paradox,
      after_paradox,
      continuation_decision_index=-1,
  )
  decisions = 0
  while not rollout.is_terminal():
    if rollout.is_chance_node():
      outcomes = rollout.chance_outcomes()
      actions, probs = zip(*outcomes)
      rollout.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    player = rollout.current_player()
    legal = list(rollout.legal_actions(player))
    continuation_state = rollout.clone()
    before_paradox = list(getattr(rollout, "_has_paradoxed", []))
    action, _policy = choose_policy_action(
        model, rollout, player, value_scale, device, args
    )
    rollout.apply_action(action)
    after_paradox = list(getattr(rollout, "_has_paradoxed", []))
    if first_new_paradox_trigger is None:
      first_new_paradox_trigger = _paradox_trigger_row(
          continuation_state,
          player,
          action,
          len(legal),
          before_paradox,
          after_paradox,
          continuation_decision_index=decisions,
      )
    decisions += 1
  result = terminal_round_result(rollout)
  result["continuation_decisions"] = decisions
  result["first_new_paradox_trigger"] = first_new_paradox_trigger
  result["forced_paradox"] = bool(
      first_new_paradox_trigger is not None
      and int(first_new_paradox_trigger.get("legal_count", 0)) <= 1
  )
  return result


def play_round_with_history(
    model,
    game,
    seed,
    value_scale,
    device,
    match_totals=None,
    round_index=0,
    args=None,
):
  if seed is not None:
    np.random.seed(int(seed))
  state = game.new_initial_state()
  if match_totals is not None:
    state.set_match_context(match_totals, round_index)
  decisions = []
  first_new_paradox_decision = None
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    player = int(state.current_player())
    legal = list(state.legal_actions(player))
    state_before = state.clone()
    before_paradox = list(getattr(state, "_has_paradoxed", []))
    action, policy = choose_policy_action(
        model, state, player, value_scale, device, args
    )
    policy_prob = float(policy[action])
    state.apply_action(action)
    after_paradox = list(getattr(state, "_has_paradoxed", []))
    new_paradox = [
        idx for idx, (before, after) in enumerate(zip(before_paradox, after_paradox))
        if bool(after) and not bool(before)
    ]
    decision = {
        "index": len(decisions),
        "state": state_before,
        "player": player,
        "phase": phase_name(state_before),
        "trick_number": int(getattr(state_before, "_trick_number", -1)),
        "legal": legal,
        "action": int(action),
        "action_string": action_string(state_before, player, action),
        "policy_prob": policy_prob,
        "policy": np.array(policy, dtype=np.float32),
        "new_paradox_players": new_paradox,
    }
    decisions.append(decision)
    if new_paradox and first_new_paradox_decision is None:
      first_new_paradox_decision = len(decisions) - 1
  return state, decisions, first_new_paradox_decision


def candidate_policy_target(num_actions, candidates, temperature):
  if not candidates:
    return None
  scores = np.array(
      [
          float(candidate["acting_score_delta"])
          - 0.1 * float(candidate["total_score_delta"])
          for candidate in candidates
      ],
      dtype=np.float32,
  )
  logits = scores / max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  exp = np.exp(logits)
  total = float(exp.sum())
  if total <= 0.0 or not math.isfinite(total):
    return None
  policy = np.zeros(num_actions, dtype=np.float32)
  for candidate, prob in zip(candidates, exp / total):
    policy[int(candidate["action"])] = float(prob)
  return policy


def evaluate_decision(model, decision, value_scale, device, args, rollout_seed):
  state_before = decision["state"]
  player = int(decision["player"])
  selected_action = int(decision["action"])
  baseline = rollout_from_action(
      model, state_before, selected_action, value_scale, device, rollout_seed, args
  )
  alternatives = []
  avoiding = []
  for action in decision["legal"]:
    action = int(action)
    if action == selected_action:
      continue
    result = rollout_from_action(
        model, state_before, action, value_scale, device, rollout_seed, args
    )
    total_score_delta = float(sum(result["raw_scores"]) - sum(baseline["raw_scores"]))
    acting_score_delta = float(
        result["raw_scores"][player] - baseline["raw_scores"][player]
    )
    record = {
        "action": action,
        "action_string": action_string(state_before, player, action),
        "any_paradox": bool(result["any_paradox"]),
        "actor_paradox": bool(result["paradoxed"][player]),
        "forced_paradox": bool(result.get("forced_paradox", False)),
        "first_new_paradox_trigger": result.get("first_new_paradox_trigger"),
        "paradoxed": result["paradoxed"],
        "raw_scores": result["raw_scores"],
        "acting_score_delta": acting_score_delta,
        "total_score_delta": total_score_delta,
    }
    alternatives.append(record)
    if (
        bool(baseline["any_paradox"])
        and not bool(result["any_paradox"])
        and acting_score_delta >= float(args.min_score_delta)
    ):
      avoiding.append(record)
  alternatives.sort(
      key=lambda item: (
          bool(item["any_paradox"]),
          -float(item["acting_score_delta"]),
          -float(item["total_score_delta"]),
      )
  )
  avoiding.sort(
      key=lambda item: (
          -float(item["acting_score_delta"]),
          -float(item["total_score_delta"]),
      )
  )
  public = {
      key: value for key, value in decision.items()
      if key not in ("state", "policy")
  }
  public["baseline"] = baseline
  public["baseline"]["actor_paradox"] = bool(baseline["paradoxed"][player])
  public["alternatives"] = alternatives[:12]
  public["num_avoiding_alternatives"] = len(avoiding)
  public["best_avoiding_alternative"] = avoiding[0] if avoiding else None
  return public, avoiding, alternatives


def replay_row_for_decision(decision, avoiding, args):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  target = candidate_policy_target(num_actions, avoiding, args.target_temperature)
  if target is None:
    return None
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[list(decision["legal"])] = 1.0
  return (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      target,
      np.zeros(int(args.players), dtype=np.float32),
      np.zeros(int(args.players), dtype=np.float32),
      int(np.argmax(target)),
      player,
      None,
      None,
      None,
      None,
      action_feature_matrix(state, player, num_actions),
  )


def dense_outcome_records(decision, baseline, alternatives):
  player = int(decision["player"])
  selected_action = int(decision["action"])
  selected = {
      "action": selected_action,
      "action_string": str(decision["action_string"]),
      "any_paradox": bool(baseline["any_paradox"]),
      "actor_paradox": bool(baseline["paradoxed"][player]),
      "forced_paradox": bool(baseline.get("forced_paradox", False)),
      "first_new_paradox_trigger": baseline.get("first_new_paradox_trigger"),
      "paradoxed": baseline["paradoxed"],
      "raw_scores": baseline["raw_scores"],
      "acting_score_delta": 0.0,
      "total_score_delta": 0.0,
  }
  records = [selected]
  records.extend(dict(item) for item in alternatives)
  return records


def _dense_outcome_safety_tuple(record):
  return (
      int(bool(record.get("any_paradox", False))),
      int(bool(record.get("forced_paradox", False))),
      int(bool(record.get("actor_paradox", False))),
  )


def _dense_outcome_score(record):
  return (
      float(record.get("acting_score_delta", 0.0))
      - 0.1 * float(record.get("total_score_delta", 0.0))
  )


def dense_policy_target(num_actions, records, temperature):
  if not records:
    return None
  best_safety = min(_dense_outcome_safety_tuple(record) for record in records)
  candidates = [
      record for record in records
      if _dense_outcome_safety_tuple(record) == best_safety
  ]
  scores = np.array(
      [_dense_outcome_score(record) for record in candidates],
      dtype=np.float32,
  )
  logits = scores / max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  exp = np.exp(logits)
  total = float(exp.sum())
  if total <= 0.0 or not math.isfinite(total):
    return None
  target = np.zeros(num_actions, dtype=np.float32)
  for record, prob in zip(candidates, exp / total):
    target[int(record["action"])] = float(prob)
  return target


def dense_action_value_target(record, args, value_scale):
  objective = str(getattr(args, "dense_action_value_objective", "safety"))
  if objective == "score":
    return float(record.get("acting_score_delta", 0.0))
  score_delta = float(record.get("acting_score_delta", 0.0))
  if objective == "safety_raw":
    target = 1.0
    if bool(record.get("any_paradox", False)):
      target -= 2.0
    if bool(record.get("forced_paradox", False)):
      target -= 0.5
    if bool(record.get("actor_paradox", False)):
      target -= 0.25
    target += 0.05 * math.tanh(score_delta / max(1e-6, float(value_scale)))
    return float(target)
  if bool(record.get("any_paradox", False)):
    target = float(getattr(args, "dense_safety_value_paradox_target", -0.55))
  else:
    target = float(getattr(args, "dense_safety_value_safe_target", 0.9))
  if bool(record.get("forced_paradox", False)):
    target -= float(getattr(args, "dense_safety_value_forced_penalty", 0.25))
  if bool(record.get("actor_paradox", False)):
    target -= float(getattr(args, "dense_safety_value_actor_penalty", 0.10))
  score_weight = float(getattr(args, "dense_safety_value_score_weight", 0.05))
  target += score_weight * math.tanh(score_delta / max(1e-6, float(value_scale)))
  target_min = float(getattr(args, "dense_safety_value_min", -0.95))
  target_max = float(getattr(args, "dense_safety_value_max", 0.95))
  if target_min > target_max:
    target_min, target_max = target_max, target_min
  target = min(target_max, max(target_min, target))
  return float(target)


def dense_label_row_for_decision(decision, baseline, alternatives, args,
                                 value_scale):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  records = dense_outcome_records(decision, baseline, alternatives)
  target = dense_policy_target(
      num_actions, records, float(args.dense_target_temperature)
  )
  if target is None:
    return None, None
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[list(decision["legal"])] = 1.0
  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  selected_action = int(decision["action"])
  for record in records:
    action = int(record["action"])
    action_paradox_targets[action] = 1.0 if bool(record["any_paradox"]) else 0.0
    action_paradox_mask[action] = 1.0
    action_value_targets[action] = dense_action_value_target(
        record, args, value_scale
    )
    action_value_mask[action] = 1.0
  row = (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      target,
      np.zeros(int(args.players), dtype=np.float32),
      np.zeros(int(args.players), dtype=np.float32),
      int(np.argmax(target)),
      player,
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      action_feature_matrix(state, player, num_actions),
  )
  label_best_action = int(np.argmax(target))
  metadata_records = []
  for record in records:
    action = int(record["action"])
    meta_record = {
        "action": action,
        "action_string": str(record["action_string"]),
        "any_paradox": bool(record["any_paradox"]),
        "actor_paradox": bool(record.get("actor_paradox", False)),
        "forced_paradox": bool(record.get("forced_paradox", False)),
        "first_new_paradox_trigger": record.get("first_new_paradox_trigger"),
        "acting_score_delta": float(record.get("acting_score_delta", 0.0)),
        "total_score_delta": float(record.get("total_score_delta", 0.0)),
        "dense_action_value_target": float(action_value_targets[action]),
        "dense_policy_target": float(target[action]),
        "liveness": action_liveness_certificate(
            state, player, action, base_action=selected_action
        ),
      }
    metadata_records.append(meta_record)
  metadata = {
      "phase": str(decision["phase"]),
      "player": player,
      "trick_number": int(decision["trick_number"]),
      "legal_count": int(len(decision["legal"])),
      "policy_best_action": selected_action,
      "policy_best_action_string": str(decision["action_string"]),
      "label_best_action": label_best_action,
      "label_best_action_string": action_string(state, player, label_best_action),
      "label_best_safety_tuple": list(
          _dense_outcome_safety_tuple(
              next(
                  record for record in records
                  if int(record["action"]) == label_best_action
              )
          )
      ),
      "policy_target_changed": bool(label_best_action != selected_action),
      "action_outcomes": metadata_records,
  }
  return row, metadata


def calibration_row_for_candidate(decision, candidate, positive, args):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  selected_action = int(decision["action"])
  candidate_action = int(candidate["action"])
  deployment_action = candidate_action if positive else selected_action
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[list(decision["legal"])] = 1.0
  policy_target = np.zeros(num_actions, dtype=np.float32)
  policy_target[deployment_action] = 1.0
  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  action_paradox_targets[selected_action] = 1.0
  action_paradox_targets[candidate_action] = (
      1.0 if bool(candidate["any_paradox"]) else 0.0
  )
  action_paradox_mask[selected_action] = 1.0
  action_paradox_mask[candidate_action] = 1.0
  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_value_targets[selected_action] = 0.0
  action_value_targets[candidate_action] = float(candidate["acting_score_delta"])
  action_value_mask[selected_action] = 1.0
  action_value_mask[candidate_action] = 1.0
  row = (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      policy_target,
      np.zeros(int(args.players), dtype=np.float32),
      np.zeros(int(args.players), dtype=np.float32),
      int(deployment_action),
      player,
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      action_feature_matrix(state, player, num_actions),
  )
  return row


def calibration_rows_for_decision(decision, alternatives, args, group):
  positive_rows = []
  negative_candidates = []
  for candidate in alternatives:
    positive = (
        not bool(candidate["any_paradox"])
        and float(candidate["acting_score_delta"]) >= float(args.min_score_delta)
    )
    if positive:
      positive_rows.append(candidate)
    else:
      negative_candidates.append(candidate)
  max_negatives = int(args.calibration_max_negatives_per_decision)
  negative_candidates.sort(
      key=lambda item: (
          -float(item["acting_score_delta"]),
          bool(item["any_paradox"]),
          -float(item["total_score_delta"]),
      )
  )
  if max_negatives >= 0:
    negative_candidates = negative_candidates[:max_negatives]

  rows = []
  metadata = []
  for positive, candidates in (
      (True, positive_rows),
      (False, negative_candidates),
  ):
    for candidate in candidates:
      row = calibration_row_for_candidate(decision, candidate, positive, args)
      selected_action = int(decision["action"])
      candidate_action = int(candidate["action"])
      deployment_action = candidate_action if positive else selected_action
      rows.append(row)
      metadata.append({
          "source_index": int(len(metadata)),
          "phase": str(decision["phase"]),
          "player": int(decision["player"]),
          "group": dict(group),
          "policy_best_action": selected_action,
          "policy_best_action_string": str(decision["action_string"]),
          "label_best_action": candidate_action,
          "label_best_action_string": str(candidate["action_string"]),
          "deployment_target_action": int(deployment_action),
          "deployment_target_action_string": action_string(
              decision["state"], int(decision["player"]), deployment_action
          ),
          "calibration_class": "POSITIVE" if positive else "NEGATIVE",
          "false_positive_for_positive_gate": not bool(positive),
          "label_margin_vs_policy": float(candidate["acting_score_delta"]),
          "audit_margin_vs_policy": float(candidate["acting_score_delta"]),
          "audit_paradox_delta_vs_policy": (
              -1.0 if not bool(candidate["any_paradox"]) else 0.0
          ),
          "legal_count": int(len(decision["legal"])),
          "candidate_count": int(max(0, len(decision["legal"]) - 1)),
          "candidate_actions": [
              int(action) for action in decision["legal"]
              if int(action) != selected_action
          ],
          "policy_action_probability": float(decision["policy_prob"]),
          "candidate_policy_probability": float(
              decision.get("policy", np.zeros(0, dtype=np.float32))[candidate_action]
          ) if candidate_action < len(decision.get("policy", [])) else None,
          "baseline_any_paradox": True,
          "candidate_any_paradox": bool(candidate["any_paradox"]),
      })
  return rows, metadata


def main():
  args = resolve_actor_bot_args(parse_args())
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model, model_args = load_neural(args.checkpoint, args.players, device, args)
  value_scale = float(model_args.value_scale)
  mined = []
  replay_rows = []
  dense_rows = []
  dense_metadata = []
  calibration_rows = []
  calibration_metadata = []
  summary = Counter()
  allowed_phases = parse_phase_filter(args.decision_phases)

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
      terminal, decisions, first_new_paradox = play_round_with_history(
          model,
          game,
          seed=int(args.seed) + match_idx * 1000003 + round_idx * 9973,
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
      round_record = {
          "match_index": match_idx,
          "round_index": round_idx,
          "start_player": start_player,
          "terminal": result,
          "first_new_paradox_decision": int(first_new_paradox),
          "evaluated_decisions": [],
      }
      decisions_this_failure_round = 0
      for decision in window:
        if int(args.max_decisions) > 0 and summary["decisions_evaluated"] >= int(args.max_decisions):
          break
        if (
            int(args.max_decisions_per_failure_round) > 0
            and decisions_this_failure_round >= int(args.max_decisions_per_failure_round)
        ):
          break
        if len(decision["legal"]) <= 1:
          continue
        if not should_evaluate_decision(decision, allowed_phases):
          continue
        summary["decisions_evaluated"] += 1
        summary[f"decisions_evaluated_phase_{decision['phase']}"] += 1
        decisions_this_failure_round += 1
        if (
            int(args.progress_interval) > 0
            and summary["decisions_evaluated"] % int(args.progress_interval) == 0
        ):
          print(json.dumps({
              "iteration": "failure_miner_progress",
              "rounds": int(summary["rounds"]),
              "failure_rounds_mined": int(summary["failure_rounds_mined"]),
              "decisions_evaluated": int(summary["decisions_evaluated"]),
              "phase": str(decision["phase"]),
              "legal_count": int(len(decision["legal"])),
          }), flush=True)
        evaluated, avoiding, alternatives = evaluate_decision(
            model,
            decision,
            value_scale,
            device,
            args,
            rollout_seed=int(args.seed) + match_idx * 1000003 + round_idx * 9973,
        )
        if args.dense_replay_out:
          dense_row, dense_meta = dense_label_row_for_decision(
              decision,
              evaluated["baseline"],
              alternatives,
              args,
              value_scale,
          )
          if dense_row is not None:
            dense_meta["source_index"] = int(len(dense_metadata))
            dense_meta["group"] = {
                "proposal_source": "failure_miner_dense",
                "match_index": int(match_idx),
                "round_index": int(round_idx),
                "decision_index": int(decision["index"]),
                "start_player": int(start_player),
            }
            dense_rows.append(dense_row)
            dense_metadata.append(dense_meta)
            summary["dense_label_rows"] += 1
            summary[f"dense_label_phase_{decision['phase']}"] += 1
            if dense_meta["policy_target_changed"]:
              summary["dense_policy_target_changed"] += 1
              summary[f"dense_policy_target_changed_phase_{decision['phase']}"] += 1
        if args.calibration_replay_out:
          group = {
              "proposal_source": "failure_miner",
              "match_index": int(match_idx),
              "round_index": int(round_idx),
              "decision_index": int(decision["index"]),
              "start_player": int(start_player),
          }
          rows, metadata = calibration_rows_for_decision(
              decision,
              alternatives,
              args,
              group,
          )
          source_offset = len(calibration_metadata)
          for offset, meta in enumerate(metadata):
            meta["source_index"] = int(source_offset + offset)
          calibration_rows.extend(rows)
          calibration_metadata.extend(metadata)
        if avoiding:
          summary["avoidable_decisions"] += 1
          summary[f"avoidable_phase_{decision['phase']}"] += 1
          row = replay_row_for_decision(decision, avoiding, args)
          if row is not None:
            replay_rows.append(row)
        round_record["evaluated_decisions"].append(evaluated)
      mined.append(round_record)

  output = {
      "checkpoint": args.checkpoint,
      "players": int(args.players),
      "matches": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "lookback_decisions": int(args.lookback_decisions),
      "min_score_delta": float(args.min_score_delta),
      "summary": dict(summary),
      "replay_rows": len(replay_rows),
      "dense_replay_rows": len(dense_rows),
      "dense_policy_target_changed_rows": int(sum(
          1 for item in dense_metadata
          if bool(item.get("policy_target_changed", False))
      )),
      "calibration_replay_rows": len(calibration_rows),
      "calibration_positive_rows": int(sum(
          1 for item in calibration_metadata
          if item.get("calibration_class") == "POSITIVE"
      )),
      "calibration_negative_rows": int(sum(
          1 for item in calibration_metadata
          if item.get("calibration_class") == "NEGATIVE"
      )),
      "failure_rounds": mined,
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(output, indent=2) + "\n")
  if args.replay_out:
    replay_path = Path(args.replay_out)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    save_replay(
        replay_rows,
        replay_path,
        metadata={
            "source": "quantum_cat_failure_miner",
            "checkpoint": args.checkpoint,
            "summary": dict(summary),
            "target_temperature": float(args.target_temperature),
            "min_score_delta": float(args.min_score_delta),
        },
    )
  if args.dense_replay_out:
    dense_path = Path(args.dense_replay_out)
    dense_path.parent.mkdir(parents=True, exist_ok=True)
    save_replay(
        dense_rows,
        dense_path,
        metadata={
            "source": "quantum_cat_failure_miner_dense",
            "checkpoint": args.checkpoint,
            "summary": dict(summary),
            "dense_target_temperature": float(args.dense_target_temperature),
            "dense_action_value_objective": str(
                args.dense_action_value_objective
            ),
            "dense_safety_value_safe_target": float(
                args.dense_safety_value_safe_target
            ),
            "dense_safety_value_paradox_target": float(
                args.dense_safety_value_paradox_target
            ),
            "dense_safety_value_forced_penalty": float(
                args.dense_safety_value_forced_penalty
            ),
            "dense_safety_value_actor_penalty": float(
                args.dense_safety_value_actor_penalty
            ),
            "dense_safety_value_score_weight": float(
                args.dense_safety_value_score_weight
            ),
            "dense_safety_value_min": float(args.dense_safety_value_min),
            "dense_safety_value_max": float(args.dense_safety_value_max),
            "min_score_delta": float(args.min_score_delta),
        },
    )
    dense_metadata_path = dense_path.with_suffix(
        dense_path.suffix + ".metadata.json"
    )
    dense_metadata_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "players": int(args.players),
        "seed": int(args.seed),
        "dense_target_temperature": float(args.dense_target_temperature),
        "dense_action_value_objective": str(args.dense_action_value_objective),
        "dense_safety_value_safe_target": float(
            args.dense_safety_value_safe_target
        ),
        "dense_safety_value_paradox_target": float(
            args.dense_safety_value_paradox_target
        ),
        "dense_safety_value_forced_penalty": float(
            args.dense_safety_value_forced_penalty
        ),
        "dense_safety_value_actor_penalty": float(
            args.dense_safety_value_actor_penalty
        ),
        "dense_safety_value_score_weight": float(
            args.dense_safety_value_score_weight
        ),
        "dense_safety_value_min": float(args.dense_safety_value_min),
        "dense_safety_value_max": float(args.dense_safety_value_max),
        "rows": dense_metadata,
      }, indent=2) + "\n")
  if args.calibration_replay_out:
    calibration_path = Path(args.calibration_replay_out)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    save_replay(
        calibration_rows,
        calibration_path,
        metadata={
            "source": "quantum_cat_failure_miner_calibration",
            "checkpoint": args.checkpoint,
            "summary": dict(summary),
            "min_score_delta": float(args.min_score_delta),
            "max_negatives_per_decision": int(
                args.calibration_max_negatives_per_decision
            ),
        },
    )
    metadata_path = calibration_path.with_suffix(
        calibration_path.suffix + ".metadata.json"
    )
    metadata_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "players": int(args.players),
        "seed": int(args.seed),
        "min_score_delta": float(args.min_score_delta),
        "max_negatives_per_decision": int(
            args.calibration_max_negatives_per_decision
        ),
        "rows": calibration_metadata,
    }, indent=2) + "\n")
  print(json.dumps({**output, "failure_rounds": len(mined)}, indent=2))


if __name__ == "__main__":
  main()
