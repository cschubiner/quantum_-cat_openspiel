#!/usr/bin/env python3
"""Full-match Elo evaluator for Cat in the Box.

A full base-game match has one round per player. The round start player rotates
so every seat starts exactly once, and Elo is updated from cumulative match
scores rather than one-round scores.
"""

from __future__ import annotations

import argparse
from collections import deque
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace
from urllib.parse import unquote

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import (
    base_bot_name,
    make_bot,
    multiplayer_elo_update,
    play_game,
    shared_prediction_action,
)
from quantum_cat_alphazero_torch import (
    AZBeliefPolicyBot,
    AZBeliefSearchBot,
    AZNet,
    AZPolicyBot,
    AZSearchBot,
    APPENDED_ACTION_FEATURE_INDEX,
    action_feasibility_scores,
    action_feature_matrix,
    load_model_payload,
    model_action_risks,
    model_action_values,
    model_policy_value,
    q_policy_select_action,
    sampled_belief_states_for_policy,
    sampled_counterfactual_legal_actions,
)
from quantum_cat_calibration_transfer_audit import (
    ImproveMLP,
    pair_feature_vector_from_matrix,
)


def _phase_name(state):
  return {
      0: "chance",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "terminal",
  }.get(int(getattr(state, "_phase", -1)), "unknown")


def _action_kind(state, action):
  if int(action) == 999:
    return "paradox"
  phase = _phase_name(state)
  if phase == "discard":
    return "discard"
  if phase == "prediction":
    return "prediction"
  if phase == "play":
    return "play"
  return phase


def _empty_decision_stats():
  return {
      "decisions": 0,
      "forced_actions": 0,
      "meaningful_decisions": 0,
      "phase_counts": {},
      "selected_action_kinds": {},
  }


def _inc_counter(mapping, key, amount=1):
  mapping[key] = int(mapping.get(key, 0)) + int(amount)


def _record_decision(stats, state, action, legal_count):
  stats["decisions"] += 1
  phase = _phase_name(state)
  _inc_counter(stats["phase_counts"], phase)
  _inc_counter(stats["selected_action_kinds"], _action_kind(state, action))
  if legal_count <= 1:
    stats["forced_actions"] += 1
  else:
    stats["meaningful_decisions"] += 1


def _compact_decision_row(state, player, action, legal_count, after_paradox=None):
  row = {
      "phase": _phase_name(state),
      "player": int(player),
      "action": int(action),
      "action_kind": _action_kind(state, action),
      "legal_count": int(legal_count),
      "trick_number": int(getattr(state, "_trick_number", -1)),
      "led_color": getattr(state, "_led_color", None),
      "tricks_won": [
          int(value) for value in getattr(state, "_tricks_won", [])
      ],
      "predictions": [
          int(value) for value in getattr(state, "_predictions", [])
      ],
  }
  if after_paradox is not None:
    row["has_paradoxed_after"] = [bool(value) for value in after_paradox]
  return row


def _merge_decision_stats(target, source):
  for key, value in source.items():
    if isinstance(value, dict):
      nested = target.setdefault(key, {})
      _merge_decision_stats(nested, value)
    elif isinstance(value, list):
      target_list = target.setdefault(key, [])
      target_list.extend(value)
      del target_list[100:]
    elif isinstance(value, (int, np.integer)):
      target[key] = int(target.get(key, 0)) + int(value)
    elif isinstance(value, (float, np.floating)):
      if key in {
          "survival_shield_threshold",
          "survival_shield_lcb_z",
          "value_shield_threshold",
          "residual_delta_clip",
          "residual_delta_scale",
          "residual_q_policy_base_margin",
      }:
        target.setdefault(key, float(value))
      else:
        target[key] = float(target.get(key, 0.0)) + float(value)
    else:
      target[key] = value


def _decision_stats_rates(stats):
  meaningful = float(stats.get("meaningful_decisions", 0) or 0)
  decisions = float(stats.get("decisions", 0) or 0)
  considered = float(stats.get("rerank_considered", 0) or 0)
  applied = float(stats.get("rerank_applied", 0) or 0)
  graft_decisions = float(stats.get("graft_phase_decisions", 0) or 0)
  if decisions:
    stats["forced_action_rate"] = round(
        float(stats.get("forced_actions", 0)) / decisions, 4
    )
  if meaningful:
    if "rerank_overrides" in stats:
      stats["rerank_override_rate_per_meaningful_decision"] = round(
          float(stats.get("rerank_overrides", 0)) / meaningful, 4
      )
    if "graft_overrides" in stats:
      stats["graft_override_rate_per_meaningful_decision"] = round(
          float(stats.get("graft_overrides", 0)) / meaningful, 4
      )
  if considered:
    stats["rerank_applied_rate_when_considered"] = round(applied / considered, 4)
  if applied:
    stats["rerank_override_rate_when_applied"] = round(
        float(stats.get("rerank_overrides", 0)) / applied, 4
    )
    stats["rerank_value_margin_avg_when_applied"] = round(
        float(stats.get("rerank_value_margin_sum", 0.0)) / applied, 6
    )
    stats["rerank_policy_log_gap_avg_when_applied"] = round(
        float(stats.get("rerank_policy_log_gap_sum", 0.0)) / applied, 6
    )
  risk_diagnostics = float(stats.get("rerank_risk_diagnostics", 0) or 0)
  if risk_diagnostics:
    stats["rerank_baseline_risk_avg"] = round(
        float(stats.get("rerank_baseline_risk_sum", 0.0)) / risk_diagnostics,
        6,
    )
    stats["rerank_min_risk_avg"] = round(
        float(stats.get("rerank_min_risk_sum", 0.0)) / risk_diagnostics, 6
    )
    stats["rerank_risk_spread_avg"] = round(
        float(stats.get("rerank_risk_spread_sum", 0.0)) / risk_diagnostics, 6
    )
  selected_risk_count = float(stats.get("rerank_selected_risk_count", 0) or 0)
  if selected_risk_count:
    stats["rerank_selected_risk_avg"] = round(
        float(stats.get("rerank_selected_risk_sum", 0.0)) / selected_risk_count,
        6,
    )
    stats["rerank_selected_risk_margin_avg"] = round(
        float(stats.get("rerank_selected_risk_margin_sum", 0.0))
        / selected_risk_count,
        6,
    )
  threshold_considered = float(
      stats.get("rerank_threshold_considered", 0) or 0
  )
  if threshold_considered:
    stats["rerank_threshold_candidate_avg"] = round(
        float(stats.get("rerank_threshold_candidates", 0))
        / threshold_considered,
        4,
    )
  relative_considered = float(stats.get("rerank_relative_considered", 0) or 0)
  if relative_considered:
    stats["rerank_relative_candidate_avg"] = round(
        float(stats.get("rerank_relative_candidates", 0))
        / relative_considered,
        4,
    )
    stats["rerank_relative_gap_filtered_candidate_avg"] = round(
        float(stats.get("rerank_relative_gap_filtered_candidates", 0))
        / relative_considered,
        4,
    )
  if graft_decisions:
    stats["graft_override_rate_in_graft_phases"] = round(
        float(stats.get("graft_overrides", 0)) / graft_decisions, 4
    )
  improve_considered = float(stats.get("improve_considered", 0) or 0)
  improve_scored = float(stats.get("improve_scored_candidates", 0) or 0)
  improve_overrides = float(stats.get("improve_overrides", 0) or 0)
  improve_shadow_overrides = float(stats.get("improve_shadow_overrides", 0) or 0)
  if improve_considered:
    stats["improve_override_rate_when_considered"] = round(
        improve_overrides / improve_considered, 4
    )
    if "improve_shadow_overrides" in stats:
      stats["improve_shadow_override_rate_when_considered"] = round(
          improve_shadow_overrides / improve_considered, 4
      )
    stats["improve_abstain_rate_when_considered"] = round(
        float(stats.get("improve_abstained", 0)) / improve_considered, 4
    )
    if "improve_risk_vetoed" in stats:
      stats["improve_risk_veto_rate_when_considered"] = round(
          float(stats.get("improve_risk_vetoed", 0)) / improve_considered, 4
      )
  if improve_scored:
    stats["improve_override_rate_per_scored_candidate"] = round(
        improve_overrides / improve_scored, 4
    )
    if "improve_shadow_overrides" in stats:
      stats["improve_shadow_override_rate_per_scored_candidate"] = round(
          improve_shadow_overrides / improve_scored, 4
      )
  root_considered = float(stats.get("root_rollout_considered", 0) or 0)
  root_overrides = float(stats.get("root_rollout_overrides", 0) or 0)
  root_scored = float(stats.get("root_rollout_scored_candidates", 0) or 0)
  if root_considered:
    stats["root_rollout_override_rate_when_considered"] = round(
        root_overrides / root_considered, 4
    )
  if root_scored:
    stats["root_rollout_avg_scored_candidates"] = round(
        root_scored / max(root_considered, 1.0), 4
    )
  value_considered = float(stats.get("value_shield_considered", 0) or 0)
  value_scored = float(stats.get("value_shield_scored_candidates", 0) or 0)
  if value_considered:
    stats["value_shield_base_keep_rate"] = round(
        float(stats.get("value_shield_base_kept", 0)) / value_considered,
        4,
    )
    stats["value_shield_override_rate"] = round(
        float(stats.get("value_shield_overrides", 0)) / value_considered,
        4,
    )
    stats["value_shield_fallback_rate"] = round(
        float(stats.get("value_shield_fallback_max_survival", 0))
        / value_considered,
        4,
    )
    stats["value_shield_avg_scored_candidates"] = round(
        value_scored / value_considered,
        4,
    )
    stats["value_shield_base_survival_avg"] = round(
        float(stats.get("value_shield_base_survival_sum", 0.0))
        / value_considered,
        6,
    )
    stats["value_shield_selected_survival_avg"] = round(
        float(stats.get("value_shield_selected_survival_sum", 0.0))
        / value_considered,
        6,
    )
    stats["value_shield_max_survival_avg"] = round(
        float(stats.get("value_shield_max_survival_sum", 0.0))
        / value_considered,
        6,
    )
  shield_considered = float(stats.get("survival_shield_considered", 0) or 0)
  shield_scored = float(stats.get("survival_shield_scored_candidates", 0) or 0)
  if shield_considered:
    stats["survival_shield_base_keep_rate"] = round(
        float(stats.get("survival_shield_base_kept", 0)) / shield_considered,
        4,
    )
    stats["survival_shield_override_rate"] = round(
        float(stats.get("survival_shield_overrides", 0)) / shield_considered,
        4,
    )
    stats["survival_shield_fallback_rate"] = round(
        float(stats.get("survival_shield_fallback_max_survival", 0))
        / shield_considered,
        4,
    )
    stats["survival_shield_avg_scored_candidates"] = round(
        shield_scored / shield_considered,
        4,
    )
    stats["survival_shield_base_survival_avg"] = round(
        float(stats.get("survival_shield_base_survival_sum", 0.0))
        / shield_considered,
        6,
    )
    stats["survival_shield_selected_survival_avg"] = round(
        float(stats.get("survival_shield_selected_survival_sum", 0.0))
        / shield_considered,
        6,
    )
    stats["survival_shield_max_survival_avg"] = round(
        float(stats.get("survival_shield_max_survival_sum", 0.0))
        / shield_considered,
        6,
    )
    if "survival_shield_base_survival_mean_sum" in stats:
      stats["survival_shield_base_survival_mean_avg"] = round(
          float(stats.get("survival_shield_base_survival_mean_sum", 0.0))
          / shield_considered,
          6,
      )
      stats["survival_shield_selected_survival_mean_avg"] = round(
          float(stats.get("survival_shield_selected_survival_mean_sum", 0.0))
          / shield_considered,
          6,
      )
      stats["survival_shield_max_survival_mean_avg"] = round(
          float(stats.get("survival_shield_max_survival_mean_sum", 0.0))
          / shield_considered,
          6,
      )
    if "survival_shield_base_survival_lcb_sum" in stats:
      stats["survival_shield_base_survival_lcb_avg"] = round(
          float(stats.get("survival_shield_base_survival_lcb_sum", 0.0))
          / shield_considered,
          6,
      )
      stats["survival_shield_selected_survival_lcb_avg"] = round(
          float(stats.get("survival_shield_selected_survival_lcb_sum", 0.0))
          / shield_considered,
          6,
      )
      stats["survival_shield_max_survival_lcb_avg"] = round(
          float(stats.get("survival_shield_max_survival_lcb_sum", 0.0))
          / shield_considered,
          6,
      )
  feasibility_considered = float(
      stats.get("feasibility_shield_considered", 0) or 0
  )
  if feasibility_considered:
    stats["feasibility_shield_base_keep_rate"] = round(
        float(stats.get("feasibility_shield_base_kept", 0))
        / feasibility_considered,
        4,
    )
    stats["feasibility_shield_override_rate"] = round(
        float(stats.get("feasibility_shield_overrides", 0))
        / feasibility_considered,
        4,
    )
    stats["feasibility_shield_base_feasible_rate"] = round(
        float(stats.get("feasibility_shield_base_feasible", 0))
        / feasibility_considered,
        4,
    )
    stats["feasibility_shield_legal_infeasible_avg"] = round(
        float(stats.get("feasibility_shield_legal_infeasible_count", 0))
        / feasibility_considered,
        4,
    )
    stats["feasibility_shield_base_slot_surplus_avg"] = round(
        float(stats.get("feasibility_shield_base_slot_surplus_sum", 0.0))
        / feasibility_considered,
        6,
    )
    stats["feasibility_shield_selected_slot_surplus_avg"] = round(
        float(stats.get("feasibility_shield_selected_slot_surplus_sum", 0.0))
        / feasibility_considered,
        6,
    )
    stats["feasibility_shield_selected_deficit_avg"] = round(
        float(stats.get("feasibility_shield_selected_deficit_sum", 0.0))
        / feasibility_considered,
        6,
    )
    stats["feasibility_shield_base_buffer_deficit_avg"] = round(
        float(stats.get("feasibility_shield_base_buffer_deficit_sum", 0.0))
        / feasibility_considered,
        6,
    )
    stats["feasibility_shield_selected_buffer_deficit_avg"] = round(
        float(stats.get("feasibility_shield_selected_buffer_deficit_sum", 0.0))
        / feasibility_considered,
        6,
    )
  exit_considered = float(
      stats.get("exit_liquidity_considered", 0) or 0
  )
  if exit_considered:
    stats["exit_liquidity_base_keep_rate"] = round(
        float(stats.get("exit_liquidity_base_kept", 0)) / exit_considered,
        4,
    )
    stats["exit_liquidity_override_rate"] = round(
        float(stats.get("exit_liquidity_overrides", 0)) / exit_considered,
        4,
    )
    stats["exit_liquidity_shadow_override_rate"] = round(
        float(stats.get("exit_liquidity_shadow_overrides", 0))
        / exit_considered,
        4,
    )
    stats["exit_liquidity_base_public_slot_damage_avg"] = round(
        float(stats.get("exit_liquidity_base_public_slot_damage_sum", 0.0))
        / exit_considered,
        6,
    )
    stats["exit_liquidity_selected_public_slot_damage_avg"] = round(
        float(stats.get("exit_liquidity_selected_public_slot_damage_sum", 0.0))
        / exit_considered,
        6,
    )
    stats["exit_liquidity_base_own_deficit_avg"] = round(
        float(stats.get("exit_liquidity_base_own_deficit_sum", 0.0))
        / exit_considered,
        6,
    )
    stats["exit_liquidity_selected_own_deficit_avg"] = round(
        float(stats.get("exit_liquidity_selected_own_deficit_sum", 0.0))
        / exit_considered,
        6,
    )
  lf_considered = float(stats.get("lf_shield_considered", 0) or 0)
  if lf_considered:
    stats["lf_shield_base_keep_rate"] = round(
        float(stats.get("lf_shield_base_kept", 0)) / lf_considered, 4
    )
    stats["lf_shield_override_rate"] = round(
        float(stats.get("lf_shield_overrides", 0)) / lf_considered, 4
    )
    stats["lf_shield_shadow_override_rate"] = round(
        float(stats.get("lf_shield_shadow_overrides", 0)) / lf_considered, 4
    )
    stats["lf_shield_base_feasible_rate"] = round(
        float(stats.get("lf_shield_base_feasible", 0)) / lf_considered, 4
    )
    stats["lf_shield_base_lost_led_rate"] = round(
        float(stats.get("lf_shield_base_lost_led_token", 0)) / lf_considered, 4
    )
    stats["lf_shield_base_public_slot_damage_avg"] = round(
        float(stats.get("lf_shield_base_public_slot_damage_sum", 0.0))
        / lf_considered,
        6,
    )
    stats["lf_shield_selected_public_slot_damage_avg"] = round(
        float(stats.get("lf_shield_selected_public_slot_damage_sum", 0.0))
        / lf_considered,
        6,
    )
    stats["lf_shield_base_own_deficit_avg"] = round(
        float(stats.get("lf_shield_base_own_deficit_sum", 0.0))
        / lf_considered,
        6,
    )
    stats["lf_shield_selected_own_deficit_avg"] = round(
        float(stats.get("lf_shield_selected_own_deficit_sum", 0.0))
        / lf_considered,
        6,
    )
    stats["lf_shield_base_min_lane_surplus_avg"] = round(
        float(stats.get("lf_shield_base_min_lane_surplus_sum", 0.0))
        / lf_considered,
        6,
    )
    stats["lf_shield_selected_min_lane_surplus_avg"] = round(
        float(stats.get("lf_shield_selected_min_lane_surplus_sum", 0.0))
        / lf_considered,
        6,
    )
  for value in stats.values():
    if isinstance(value, dict):
      _decision_stats_rates(value)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=300)
  parser.add_argument(
      "--paired-deals",
      type=int,
      default=0,
      help=(
          "When >0 in ladder mode, run this many common-deal blocks across "
          "the seating schedule. With --schedule=permutations each block "
          "uses the same chance seed for every ordered seating, giving a "
          "paired seat-swapped direct gate."
      ),
  )
  parser.add_argument(
      "--bots",
      nargs="+",
      default=["random", "heuristic", "heuristic_target2"],
      help=(
          "Built-in bot names. Use alias=bot, e.g. h0=heuristic h1=heuristic, "
          "to seat cloned built-in policies without name collisions."
      ),
  )
  parser.add_argument("--candidate", default=None)
  parser.add_argument("--opponents", default="heuristic,heuristic_target2,random")
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument(
      "--neural-bot",
      action="append",
      default=[],
      help=(
          "Named neural bot spec for ladder mode. Format: name=checkpoint "
          "or name=checkpoint:policy, name=checkpoint:mcts, "
          "name=checkpoint:belief_policy, name=checkpoint:q_policy, "
          "name=checkpoint:root_rollout, name=checkpoint:value_shield, "
          "name=checkpoint:liveness_shield, "
          "name=checkpoint:residual_policy, "
          "name=checkpoint:residual_q_policy, "
          "name=checkpoint:residual_q_risk_policy, name=checkpoint:play_graft, "
          "name=checkpoint:improve_graft, or name=checkpoint:belief. "
          "Append ?key=value,... after the mode to "
          "override per-bot evaluator knobs, for example "
          "name=checkpoint:belief_policy?belief_source=policy_weighted."
      ),
  )
  parser.add_argument("--belief-samples", type=int, default=4)
  parser.add_argument("--belief-sims", type=int, default=8)
  parser.add_argument(
      "--root-rollout-samples",
      type=int,
      default=2,
      help="Belief particles per decision for :root_rollout neural bots.",
  )
  parser.add_argument(
      "--root-rollouts",
      type=int,
      default=1,
      help="Common rollout seeds per belief particle for :root_rollout bots.",
  )
  parser.add_argument(
      "--root-rollout-max-actions",
      type=int,
      default=4,
      help=(
          "Maximum legal candidate actions scored by :root_rollout. 0 scores "
          "all legal actions, which can be expensive in play states."
      ),
  )
  parser.add_argument(
      "--root-rollout-top-policy",
      type=int,
      default=2,
      help="Always include this many top raw-policy legal actions.",
  )
  parser.add_argument(
      "--root-rollout-include-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
      help=(
          "Comma-separated heuristic bot names whose chosen legal actions are "
          "included in :root_rollout candidate sets before random fill."
      ),
  )
  parser.add_argument(
      "--root-rollout-max-plies",
      type=int,
      default=0,
      help=(
          "If >0, stop :root_rollout playouts after this many non-chance "
          "plies and use the neural value head as the leaf estimate."
      ),
  )
  parser.add_argument(
      "--root-rollout-phases",
      default="",
      help=(
          "Optional comma-separated phase names where :root_rollout may apply "
          "common-particle action evaluation. Supported names: chance,discard,"
          "prediction,play,terminal. Empty applies it in every phase."
      ),
  )
  parser.add_argument(
      "--root-rollout-full-match",
      action="store_true",
      help=(
          "Score :root_rollout candidates by rolling out the current round "
          "plus any remaining full-match rounds from the match context."
      ),
  )
  parser.add_argument(
      "--root-rollout-objective",
      choices=("score", "paradox_then_score"),
      default="score",
      help=(
          "How :root_rollout ranks candidate rollouts. score preserves the "
          "legacy expected-score objective. paradox_then_score ranks lower "
          "paradox probability first, then expected score, then raw policy."
      ),
  )
  parser.add_argument(
      "--root-rollout-paradox-scope",
      choices=("acting", "any"),
      default="acting",
      help=(
          "Paradox event used by --root-rollout-objective=paradox_then_score. "
          "acting tracks only the acting player; any tracks whether any player "
          "paradoxed in the rolled-out hand."
      ),
  )
  parser.add_argument(
      "--root-rollout-continuation-bot",
      default="",
      help=(
          "Optional built-in bot used for non-root continuation actions inside "
          ":root_rollout playouts. Empty uses the neural model policy."
      ),
  )
  parser.add_argument(
      "--root-rollout-continuation-mode",
      choices=("", "policy", "q_policy", "liveness_shield"),
      default="",
      help=(
          "Optional neural bot mode used for non-root continuation actions "
          "inside :root_rollout playouts. Empty uses the raw neural policy; "
          "--root-rollout-continuation-bot takes precedence when set."
      ),
  )
  parser.add_argument(
      "--root-rollout-include-continuation-candidate",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Include the configured continuation policy's preferred legal action "
          "in :root_rollout candidate sets before random fill."
      ),
  )
  parser.add_argument(
      "--survival-shield-base-bot",
      default="heuristic_safe14",
      help=(
          "Built-in policy used by survival_shield evaluator bots before the "
          "whole-round survival threshold is applied."
      ),
  )
  parser.add_argument(
      "--survival-shield-continuation-bot",
      default="heuristic_safe14",
      help=(
          "Built-in continuation policy for survival_shield candidate rollouts."
      ),
  )
  parser.add_argument(
      "--survival-shield-threshold",
      type=float,
      default=0.45,
      help=(
          "Minimum estimated no-any-player-paradox probability required to keep "
          "the base action. If no legal action clears it, survival_shield "
          "falls back to the max-survival legal action."
      ),
  )
  parser.add_argument(
      "--survival-shield-score-mode",
      choices=("mean", "wilson_lcb"),
      default="wilson_lcb",
      help=(
          "Candidate survival score used by survival_shield thresholding. "
          "wilson_lcb is conservative for small/noisy rollout counts; mean "
          "preserves the original sampled survival fraction."
      ),
  )
  parser.add_argument(
      "--survival-shield-selection-mode",
      choices=("threshold", "dominance"),
      default="threshold",
      help=(
          "threshold preserves the legacy keep-if-above-threshold behavior. "
          "dominance keeps the base action unless another candidate is "
          "confidently safer by Wilson bounds or mean survival margin."
      ),
  )
  parser.add_argument(
      "--survival-shield-override-margin",
      type=float,
      default=0.05,
      help=(
          "For --survival-shield-selection-mode=dominance, override the base "
          "action when candidate_lcb > base_ucb + this margin."
      ),
  )
  parser.add_argument(
      "--survival-shield-override-mean-delta",
      type=float,
      default=0.20,
      help=(
          "For dominance mode, also allow an override when candidate mean "
          "survival beats base mean by at least this amount and is not less "
          "conservative than the base by LCB."
      ),
  )
  parser.add_argument(
      "--survival-shield-lcb-z",
      type=float,
      default=1.96,
      help=(
          "Z value for --survival-shield-score-mode=wilson_lcb. 1.96 is the "
          "standard 95%% Wilson lower bound."
      ),
  )
  parser.add_argument(
      "--survival-shield-rollouts",
      type=int,
      default=4,
      help="Common-seed whole-round rollouts per legal candidate.",
  )
  parser.add_argument(
      "--survival-shield-samples",
      type=int,
      default=1,
      help=(
          "Public-information belief samples per survival_shield decision. "
          "1 is cheap for threshold sweeps; larger values move toward the "
          "particle safety-search plan."
      ),
  )
  parser.add_argument(
      "--survival-shield-max-actions",
      type=int,
      default=0,
      help=(
          "Maximum legal actions scored by survival_shield. 0 scores every "
          "legal action; positive values keep the base action plus heuristic "
          "and tactical feature candidates to cap cost."
      ),
  )
  parser.add_argument(
      "--survival-shield-include-bots",
      default="heuristic_safe14,heuristic_safe8,heuristic,heuristic_target2,heuristic_adj2",
      help=(
          "Comma-separated built-in bots whose suggested legal actions are "
          "included before feature extremes when survival_shield caps "
          "candidate count."
      ),
  )
  parser.add_argument(
      "--survival-shield-feature-candidates",
      type=_parse_bool,
      default=True,
      help=(
          "When survival_shield caps candidate count, include deterministic "
          "tactical action-feature extremes before even-spaced legal fill."
      ),
  )
  parser.add_argument(
      "--survival-shield-phases",
      default="",
      help=(
          "Optional comma-separated phases where survival_shield is active. "
          "Empty applies it to every non-forced decision."
      ),
  )
  parser.add_argument(
      "--feasibility-shield-base-bot",
      default="heuristic_safe14",
      help=(
          "Built-in base policy used by feasibility_shield evaluator bots. "
          "The shield keeps this action when it preserves own-hand assignment "
          "feasibility, otherwise it selects the best feasible legal action."
      ),
  )
  parser.add_argument(
      "--feasibility-shield-min-slot-surplus",
      type=int,
      default=0,
      help=(
          "Minimum total own-hand open-slot surplus required to keep the base "
          "action. 0 means only exact assignment feasibility is required."
      ),
  )
  parser.add_argument(
      "--feasibility-shield-max-buffer-deficit-to-keep",
      type=int,
      default=-1,
      help=(
          "Optional early-flexibility guard for feasibility_shield. When >=0, "
          "the base action is kept only if the sum of per-rank one-extra-slot "
          "deficits after the action is no larger than this value."
      ),
  )
  parser.add_argument(
      "--feasibility-shield-phases",
      default="",
      help=(
          "Optional comma-separated phases where feasibility_shield is active. "
          "Empty applies it to every non-forced decision."
      ),
  )
  parser.add_argument(
      "--exit-liquidity-base-bot",
      default="heuristic_safe14",
      help=(
          "Built-in base policy used by exit_liquidity_shield evaluator bots. "
          "The shield compares legal actions by own-hand feasibility plus "
          "public exit-space damage from board claims and token loss."
      ),
  )
  parser.add_argument(
      "--exit-liquidity-phases",
      default="play",
      help=(
          "Optional comma-separated phases where exit_liquidity_shield is "
          "active. Empty applies it to every non-forced decision."
      ),
  )
  parser.add_argument(
      "--exit-liquidity-min-trick-number",
      type=int,
      default=0,
      help=(
          "Minimum trick number where exit_liquidity_shield may intervene. "
          "Use 0 to allow all play-phase decisions."
      ),
  )
  parser.add_argument(
      "--exit-liquidity-min-damage-delta",
      type=float,
      default=2.0,
      help=(
          "Minimum public-slot-damage improvement required for a pure "
          "liquidity override when own feasibility is otherwise tied."
      ),
  )
  parser.add_argument(
      "--exit-liquidity-shadow-only",
      type=_parse_bool,
      default=False,
      help=(
          "When true, compute exit_liquidity_shield diagnostics but always "
          "play the base action."
      ),
  )
  parser.add_argument(
      "--lf-shield-base-bot",
      default="heuristic_safe14",
      help="Built-in base policy used by liquidity_feasibility_shield bots.",
  )
  parser.add_argument(
      "--lf-shield-phases",
      default="play",
      help=(
          "Optional comma-separated phases where liquidity_feasibility_shield "
          "is active. Empty applies it to every non-forced decision."
      ),
  )
  parser.add_argument(
      "--lf-shield-min-trick-number",
      type=int,
      default=0,
      help="Minimum trick number where liquidity_feasibility_shield may intervene.",
  )
  parser.add_argument(
      "--lf-shield-min-damage-delta",
      type=float,
      default=0.5,
      help=(
          "Minimum public-slot-damage improvement for a pure liquidity "
          "override when feasibility is otherwise not improved."
      ),
  )
  parser.add_argument(
      "--lf-shield-min-lane-surplus-delta",
      type=int,
      default=1,
      help=(
          "Minimum min-player lane-surplus improvement for a lane-preservation "
          "override."
      ),
  )
  parser.add_argument(
      "--lf-shield-min-slot-surplus",
      type=int,
      default=0,
      help=(
          "Minimum exact own-hand slot surplus required to keep the base "
          "action."
      ),
  )
  parser.add_argument(
      "--lf-shield-max-buffer-deficit-to-keep",
      type=int,
      default=-1,
      help=(
          "When >=0, keep the base action only if its exact own-hand buffer "
          "deficit is no larger than this value."
      ),
  )
  parser.add_argument(
      "--lf-shield-led-token-max-damage-increase",
      type=float,
      default=0.0,
      help=(
          "Allow led-token-preservation overrides whose public-slot damage is "
          "at most this much worse than the base action."
      ),
  )
  parser.add_argument(
      "--lf-shield-shadow-only",
      type=_parse_bool,
      default=False,
      help=(
          "When true, compute liquidity_feasibility_shield diagnostics but "
          "always play the base action."
      ),
  )
  parser.add_argument(
      "--liveness-teacher-base-bot",
      default="heuristic_safe14",
      help=(
          "Built-in base policy used by liveness_key_teacher bots outside the "
          "enabled phases and as the tie-break action inside the liveness key."
      ),
  )
  parser.add_argument(
      "--liveness-teacher-phases",
      default="play",
      help=(
          "Optional comma-separated phases where liveness_key_teacher chooses "
          "the max deterministic liveness action. Empty applies to every "
          "non-forced decision."
      ),
  )
  parser.add_argument(
      "--liveness-teacher-min-trick-number",
      type=int,
      default=0,
      help=(
          "Minimum trick number where liveness_key_teacher may replace its "
          "base action. Use 0 to allow all play-phase decisions."
      ),
  )
  parser.add_argument(
      "--liveness-teacher-exact-legal-pressure",
      type=_parse_bool,
      default=False,
      help=(
          "When true, liveness_key_teacher prepends exact post-action legal-exit "
          "pressure to its key. This is a stronger teacher/audit signal and may "
          "use full simulator state."
      ),
  )
  parser.add_argument(
      "--liveness-teacher-sample-limit",
      type=int,
      default=20,
      help="Maximum liveness_key_teacher override samples stored in artifacts.",
  )
  parser.add_argument(
      "--liveness-shield-base-mode",
      choices=("policy", "q_policy"),
      default="q_policy",
      help=(
          "Neural base action used by :liveness_shield before deterministic "
          "own-hand/public-exit dominance checks."
      ),
  )
  parser.add_argument(
      "--liveness-shield-phases",
      default="discard,prediction,play",
      help=(
          "Optional comma-separated phases where :liveness_shield may "
          "override. Empty applies it to every non-forced decision."
      ),
  )
  parser.add_argument(
      "--liveness-shield-min-trick-number",
      type=int,
      default=-1,
      help=(
          "Optional minimum trick number for :liveness_shield overrides. "
          "-1 disables this filter."
      ),
  )
  parser.add_argument(
      "--liveness-shield-led-colors",
      default="",
      help=(
          "Optional comma-separated led-color filter for :liveness_shield. "
          "Use R/G/B/Y or None; empty disables this filter."
      ),
  )
  parser.add_argument(
      "--liveness-shield-min-open-slot-delta",
      type=int,
      default=1,
      help=(
          "Minimum improvement in min public player open slots needed for a "
          "pure public-liveness override when exact own feasibility is tied."
      ),
  )
  parser.add_argument(
      "--liveness-shield-min-public-damage-delta",
      type=int,
      default=2,
      help=(
          "Minimum public-slot-damage improvement needed for a pure "
          "public-liveness override when exact own feasibility is tied."
      ),
  )
  parser.add_argument(
      "--liveness-shield-max-policy-log-gap",
      type=float,
      default=-1.0,
      help=(
          "Optional policy guard for :liveness_shield overrides. When >=0, "
          "an alternative must be within this log-probability gap of the "
          "base action."
      ),
  )
  parser.add_argument(
      "--liveness-shield-shadow-only",
      type=_parse_bool,
      default=False,
      help=(
          "When true, compute :liveness_shield diagnostics but always play "
          "the neural base action."
      ),
  )
  parser.add_argument(
      "--liveness-shield-sample-limit",
      type=int,
      default=20,
      help="Maximum :liveness_shield override samples stored in artifacts.",
  )
  parser.add_argument(
      "--residual-anchor-checkpoint",
      default="",
      help=(
          "Anchor checkpoint for :residual_policy. The candidate contributes "
          "only bounded log-policy deltas over this anchor."
      ),
  )
  parser.add_argument(
      "--residual-policy-phases",
      default="discard,prediction,play",
      help=(
          "Comma-separated phases where :residual_policy may apply candidate "
          "deltas. Empty enables every non-forced decision."
      ),
  )
  parser.add_argument(
      "--residual-delta-clip",
      type=float,
      default=1.0,
      help="Absolute log-probability delta cap for :residual_policy.",
  )
  parser.add_argument(
      "--residual-delta-scale",
      type=float,
      default=1.0,
      help="Multiplier applied to clipped candidate-anchor deltas.",
  )
  parser.add_argument(
      "--residual-q-policy-base-margin",
      type=float,
      default=0.0,
      help=(
          "For :residual_q_policy, lift the deployed q-policy base action "
          "this many log-score units above the best raw anchor policy action "
          "before applying candidate residual deltas."
      ),
  )
  parser.add_argument(
      "--survival-checkpoint",
      default="",
      help=(
          "Optional state-value checkpoint used by :value_shield neural bots. "
          "Empty uses the policy checkpoint itself."
      ),
  )
  parser.add_argument("--survival-value-threshold", type=float, default=0.55)
  parser.add_argument("--survival-value-max-actions", type=int, default=0)
  parser.add_argument(
      "--survival-value-phases",
      default="",
      help=(
          "Optional comma-separated phases where :value_shield may override. "
          "Empty applies it to every non-forced decision."
      ),
  )
  parser.add_argument(
      "--survival-value-scope",
      choices=("mean", "current", "acting"),
      default="mean",
      help=(
          "Which value output to convert into no-paradox probability when "
          "scoring post-action states."
      ),
  )
  parser.add_argument("--survival-value-chance-depth", type=int, default=3)
  parser.add_argument("--survival-value-max-chance-outcomes", type=int, default=32)
  parser.add_argument(
      "--survival-value-max-policy-log-gap",
      type=float,
      default=-1.0,
      help=(
          "When >=0, :value_shield only uses above-threshold alternatives "
          "whose policy log probability is within this gap of the raw policy "
          "choice. The max-survival fallback is still allowed."
      ),
  )
  parser.add_argument(
      "--belief-source",
      choices=("infostate", "policy_weighted", "ranker_resample"),
      default="infostate",
      help=(
          "Hidden-state sampler for :belief_policy and :belief neural bots. "
          "policy_weighted reweights sampled worlds by public-history policy "
          "likelihood before averaging/searching; ranker_resample uses a "
          "learned contrastive hidden-world scorer."
      ),
  )
  parser.add_argument("--belief-candidates", type=int, default=8)
  parser.add_argument("--belief-policy-temperature", type=float, default=1.0)
  parser.add_argument("--belief-uniform-mix", type=float, default=0.15)
  parser.add_argument("--belief-ranker", default="")
  parser.add_argument("--belief-ranker-candidates", type=int, default=64)
  parser.add_argument("--belief-ranker-temperature", type=float, default=0.7)
  parser.add_argument("--belief-ranker-uniform-mix", type=float, default=0.25)
  parser.add_argument(
      "--belief-ref-policy-mix",
      default="model:1.0",
      help=(
          "Reference mixture for --belief-source=policy_weighted. "
          "Comma-separated weights over model, model_avg, model0/model1/etc, "
          "uniform, heuristic, heuristic_target2, and heuristic_adj2."
      ),
  )
  parser.add_argument(
      "--belief-ref-policy-mix-by-phase",
      default="",
      help=(
          "Optional semicolon-separated phase overrides for "
          "--belief-source=policy_weighted, e.g. "
          "prediction=model:0.65,heuristic:0.25,uniform:0.10;play=model:1.0."
      ),
  )
  parser.add_argument("--belief-logprob-floor", type=float, default=1e-6)
  parser.add_argument(
      "--device",
      choices=("cpu", "mps", "auto"),
      default="cpu",
      help=(
          "Device for single-process evaluation. Worker processes always use "
          "CPU to avoid tiny MCTS inference contention on MPS."
      ),
  )
  parser.add_argument(
      "--mcts-sims",
      type=int,
      default=64,
      help="Plain neural MCTS simulations for --neural-bot name=checkpoint:mcts.",
  )
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-root-only", action="store_true")
  parser.add_argument(
      "--action-paradox-selection-penalty",
      type=float,
      default=0.0,
      help=(
          "For :q_policy bots, subtract this many score units times the "
          "per-action paradox-risk head from each legal action's rerank score. "
          "For :mcts bots, subtract it from risk-aware tree selection."
      ),
  )
  parser.add_argument("--action-paradox-root-only", action="store_true")
  parser.add_argument(
      "--action-paradox-rerank-mode",
      choices=("additive", "threshold", "relative"),
      default="additive",
      help=(
          "How :q_policy uses the action paradox-risk head. additive preserves "
          "the historical log(policy)-penalty*risk score. threshold only "
          "switches away from the raw top policy when risk is high enough and "
          "a lower-risk action is within the configured policy log-gap. "
          "relative first builds a lowest-risk safe set using "
          "--action-paradox-min-risk-margin as slack, then picks the best "
          "policy/value action inside that set."
      ),
  )
  parser.add_argument("--action-paradox-risk-threshold", type=float, default=0.0)
  parser.add_argument("--action-paradox-min-risk-margin", type=float, default=0.0)
  parser.add_argument(
      "--action-feasibility-selection-weight",
      type=float,
      default=0.0,
      help=(
          "Optional :q_policy bonus for actions that preserve future own-hand "
          "legal exits. Defaults to off."
      ),
  )
  parser.add_argument(
      "--action-paradox-max-policy-log-gap",
      type=float,
      default=2.0,
      help=(
          "For --action-paradox-rerank-mode=threshold, reject lower-risk "
          "actions whose raw policy log probability is more than this far "
          "below the raw top action. Negative disables the gap guard."
      ),
  )
  parser.add_argument(
      "--action-value-rerank-clip",
      type=float,
      default=0.5,
      help=(
          "Clip raw action-value head output for fast :q_policy reranking "
          "before multiplying by value_scale and "
          "--action-value-selection-weight."
      ),
  )
  parser.add_argument(
      "--action-value-rerank-phases",
      default="",
      help=(
          "Optional comma-separated phase names where :q_policy may apply "
          "the action-value reranker. Supported names: chance,discard,"
          "prediction,play,terminal. Empty applies it in every phase."
      ),
  )
  parser.add_argument(
      "--action-value-rerank-min-margin",
      type=float,
      default=0.0,
      help=(
          "Only apply :q_policy reranking when the best legal action-value "
          "prediction beats the second-best by at least this raw model margin. "
          "0 preserves the historical always-rerank behavior."
      ),
  )
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument(
      "--schedule",
      choices=("combinations", "permutations"),
      default="combinations",
      help=(
          "Ladder seating schedule. permutations balances every ordered seating "
          "for the selected player count and is preferred for direct gates."
      ),
  )
  parser.add_argument("--k-factor", type=float, default=24.0)
  parser.add_argument(
      "--homogeneous-paradox-threshold",
      type=float,
      default=0.0,
      help=(
          "When >0, add a homogeneous self-play safety gate summary. The gate "
          "is eligible when every participating bot name is the same built-in "
          "bot behind aliases or a neural alias with the same checkpoint/mode/"
          "options. The strict metric is rounds_with_any_player_paradox / "
          "total_rounds, so 0.40 means fewer than 40%% of full-match hand "
          "rounds contain a paradox."
      ),
  )
  parser.add_argument(
      "--workers",
      type=int,
      default=0,
      help=(
          "CPU worker processes for independent full-match simulations. "
          "0 auto-sizes from CPU count, capped at 16."
      ),
  )
  parser.add_argument(
      "--worker-torch-threads",
      type=int,
      default=1,
      help="torch.set_num_threads value inside CPU evaluator workers.",
  )
  parser.add_argument(
      "--auto-worker-min-games",
      type=int,
      default=32,
      help=(
          "Minimum matches before --workers=0 auto-spawns workers. Explicit "
          "positive worker counts ignore this floor."
      ),
  )
  parser.add_argument("--seed", type=int, default=20260602)
  parser.add_argument("--out", default="experiments/full_match_elo_latest.json")
  return parser.parse_args()


def _model_args(players):
  return SimpleNamespace(
      players=players,
      arch="mlp",
      width=256,
      depth=3,
      value_scale=20.0,
      c_puct=1.8,
      sims=64,
      match_context=False,
      action_value_selection_weight=0.0,
      action_value_root_only=False,
      action_paradox_selection_penalty=0.0,
      action_paradox_root_only=False,
      action_paradox_rerank_mode="additive",
      action_paradox_risk_threshold=0.0,
      action_paradox_min_risk_margin=0.0,
      action_feasibility_selection_weight=0.0,
      action_paradox_max_policy_log_gap=2.0,
      action_value_rerank_clip=0.5,
      action_value_rerank_phases="",
      action_value_rerank_min_margin=0.0,
      belief_source="infostate",
      belief_candidates=8,
      belief_policy_temperature=1.0,
      belief_uniform_mix=0.15,
      belief_ref_policy_mix="model:1.0",
      belief_logprob_floor=1e-6,
      belief_ranker="",
      belief_ranker_candidates=64,
      belief_ranker_temperature=0.7,
      belief_ranker_uniform_mix=0.25,
      root_rollout_samples=2,
      root_rollouts=1,
      root_rollout_max_actions=4,
      root_rollout_top_policy=2,
      root_rollout_include_bots="heuristic,heuristic_target2,heuristic_adj2",
      root_rollout_max_plies=0,
      root_rollout_phases="",
      root_rollout_full_match=False,
      root_rollout_objective="score",
      root_rollout_paradox_scope="acting",
      root_rollout_continuation_bot="",
      root_rollout_continuation_mode="",
      root_rollout_include_continuation_candidate=True,
      survival_shield_base_bot="heuristic_safe14",
      survival_shield_continuation_bot="heuristic_safe14",
      survival_shield_threshold=0.45,
      survival_shield_score_mode="wilson_lcb",
      survival_shield_lcb_z=1.96,
      survival_shield_rollouts=4,
      survival_shield_samples=1,
      survival_shield_max_actions=0,
      survival_shield_include_bots=(
          "heuristic_safe14,heuristic_safe8,heuristic,heuristic_target2,"
          "heuristic_adj2"
      ),
      survival_shield_feature_candidates=True,
      survival_shield_phases="",
      exit_liquidity_base_bot="heuristic_safe14",
      exit_liquidity_phases="play",
      exit_liquidity_min_trick_number=0,
      exit_liquidity_min_damage_delta=2.0,
      exit_liquidity_shadow_only=False,
      lf_shield_base_bot="heuristic_safe14",
      lf_shield_phases="play",
      lf_shield_min_trick_number=0,
      lf_shield_min_damage_delta=0.5,
      lf_shield_min_lane_surplus_delta=1,
      lf_shield_min_slot_surplus=0,
      lf_shield_max_buffer_deficit_to_keep=-1,
      lf_shield_led_token_max_damage_increase=0.0,
      lf_shield_shadow_only=False,
      liveness_teacher_base_bot="heuristic_safe14",
      liveness_teacher_phases="play",
      liveness_teacher_min_trick_number=0,
      liveness_teacher_exact_legal_pressure=False,
      liveness_teacher_sample_limit=20,
      liveness_shield_base_mode="q_policy",
      liveness_shield_phases="discard,prediction,play",
      liveness_shield_min_trick_number=-1,
      liveness_shield_led_colors="",
      liveness_shield_min_open_slot_delta=1,
      liveness_shield_min_public_damage_delta=2,
      liveness_shield_max_policy_log_gap=-1.0,
      liveness_shield_shadow_only=False,
      liveness_shield_sample_limit=20,
      residual_anchor_checkpoint="",
      residual_policy_phases="discard,prediction,play",
      residual_delta_clip=1.0,
      residual_delta_scale=1.0,
      residual_q_policy_base_margin=0.0,
      survival_checkpoint="",
      survival_value_threshold=0.55,
      survival_value_max_actions=0,
      survival_value_phases="",
      survival_value_scope="mean",
      survival_value_chance_depth=3,
      survival_value_max_chance_outcomes=32,
      survival_value_max_policy_log_gap=-1.0,
      graft_checkpoint="",
      graft_builtin_bot="",
      base_mode="policy",
      graft_phases="play",
      graft_mode="policy",
      improve_model="",
      improve_threshold=0.95,
      improve_risk_threshold=0.0,
      improve_phases="play",
      improve_max_actions=0,
      improve_top_policy=0,
      improve_include_bots="heuristic,heuristic_target2,heuristic_adj2",
      improve_feature_candidates=True,
      improve_min_legal_count=0,
      improve_max_legal_count=0,
      improve_min_trick_number=-1,
      improve_min_match_round=-1,
      improve_near_paradox_proxy=False,
      improve_near_paradox_min_pressure=0.1,
      improve_shadow_only=False,
      improve_sample_limit=20,
  )


def load_neural(checkpoint, players, device, args=None):
  if not checkpoint:
    return None, None
  model_args = _model_args(players)
  game = pyspiel.load_game(
      "python_quantum_cat", {"players": players, "start_player": 0}
  )
  model, _, saved_args = load_model_payload(checkpoint, game, model_args, device)
  model.eval()
  model_args.value_scale = saved_args.get("value_scale", model_args.value_scale)
  model_args.arch = saved_args.get("arch", model_args.arch)
  model_args.match_context = saved_args.get("match_context", False)
  if args is not None:
    model_args.action_value_selection_weight = float(
        getattr(args, "action_value_selection_weight", 0.0)
    )
    model_args.action_value_root_only = bool(
        getattr(args, "action_value_root_only", False)
    )
    model_args.action_paradox_selection_penalty = float(
        getattr(args, "action_paradox_selection_penalty", 0.0)
    )
    model_args.action_paradox_root_only = bool(
        getattr(args, "action_paradox_root_only", False)
    )
    model_args.action_paradox_rerank_mode = str(
        getattr(args, "action_paradox_rerank_mode", "additive")
    )
    model_args.action_paradox_risk_threshold = float(
        getattr(args, "action_paradox_risk_threshold", 0.0)
    )
    model_args.action_paradox_min_risk_margin = float(
        getattr(args, "action_paradox_min_risk_margin", 0.0)
    )
    model_args.action_feasibility_selection_weight = float(
        getattr(args, "action_feasibility_selection_weight", 0.0)
    )
    model_args.action_paradox_max_policy_log_gap = float(
        getattr(args, "action_paradox_max_policy_log_gap", 2.0)
    )
    model_args.action_value_rerank_clip = float(
        getattr(args, "action_value_rerank_clip", 0.5)
    )
    model_args.action_value_rerank_phases = str(
        getattr(args, "action_value_rerank_phases", "")
    )
    model_args.action_value_rerank_min_margin = float(
        getattr(args, "action_value_rerank_min_margin", 0.0)
    )
    model_args.belief_source = str(getattr(args, "belief_source", "infostate"))
    model_args.belief_candidates = int(getattr(args, "belief_candidates", 8))
    model_args.belief_policy_temperature = float(
        getattr(args, "belief_policy_temperature", 1.0)
    )
    model_args.belief_uniform_mix = float(getattr(args, "belief_uniform_mix", 0.15))
    model_args.belief_ref_policy_mix = str(
        getattr(args, "belief_ref_policy_mix", "model:1.0")
    )
    model_args.belief_ref_policy_mix_by_phase = str(
        getattr(args, "belief_ref_policy_mix_by_phase", "")
    )
    model_args.belief_logprob_floor = float(
        getattr(args, "belief_logprob_floor", 1e-6)
    )
    model_args.belief_ranker = str(getattr(args, "belief_ranker", ""))
    model_args.belief_ranker_candidates = int(
        getattr(args, "belief_ranker_candidates", 64)
    )
    model_args.belief_ranker_temperature = float(
        getattr(args, "belief_ranker_temperature", 0.7)
    )
    model_args.belief_ranker_uniform_mix = float(
        getattr(args, "belief_ranker_uniform_mix", 0.25)
    )
    model_args.root_rollout_samples = int(
        getattr(args, "root_rollout_samples", 2)
    )
    model_args.root_rollouts = int(getattr(args, "root_rollouts", 1))
    model_args.root_rollout_max_actions = int(
        getattr(args, "root_rollout_max_actions", 4)
    )
    model_args.root_rollout_top_policy = int(
        getattr(args, "root_rollout_top_policy", 2)
    )
    model_args.root_rollout_include_bots = str(
        getattr(args, "root_rollout_include_bots", "")
    )
    model_args.root_rollout_max_plies = int(
        getattr(args, "root_rollout_max_plies", 0)
    )
    model_args.root_rollout_phases = str(
        getattr(args, "root_rollout_phases", "")
    )
    model_args.root_rollout_full_match = bool(
        getattr(args, "root_rollout_full_match", False)
    )
    model_args.root_rollout_objective = str(
        getattr(args, "root_rollout_objective", "score")
    )
    model_args.root_rollout_paradox_scope = str(
        getattr(args, "root_rollout_paradox_scope", "acting")
    )
    model_args.root_rollout_continuation_bot = str(
        getattr(args, "root_rollout_continuation_bot", "")
    )
    model_args.root_rollout_continuation_mode = str(
        getattr(args, "root_rollout_continuation_mode", "")
    )
    model_args.root_rollout_include_continuation_candidate = bool(
        getattr(args, "root_rollout_include_continuation_candidate", True)
    )
    model_args.survival_shield_base_bot = str(
        getattr(args, "survival_shield_base_bot", "heuristic_safe14")
    )
    model_args.survival_shield_continuation_bot = str(
        getattr(args, "survival_shield_continuation_bot", "heuristic_safe14")
    )
    model_args.survival_shield_threshold = float(
        getattr(args, "survival_shield_threshold", 0.45)
    )
    model_args.survival_shield_score_mode = str(
        getattr(args, "survival_shield_score_mode", "wilson_lcb")
    )
    model_args.survival_shield_lcb_z = float(
        getattr(args, "survival_shield_lcb_z", 1.96)
    )
    model_args.survival_shield_rollouts = int(
        getattr(args, "survival_shield_rollouts", 4)
    )
    model_args.survival_shield_samples = int(
        getattr(args, "survival_shield_samples", 1)
    )
    model_args.survival_shield_max_actions = int(
        getattr(args, "survival_shield_max_actions", 0)
    )
    model_args.survival_shield_include_bots = str(
        getattr(
            args,
            "survival_shield_include_bots",
            "heuristic_safe14,heuristic_safe8,heuristic,heuristic_target2,heuristic_adj2",
        )
    )
    model_args.survival_shield_feature_candidates = bool(
        getattr(args, "survival_shield_feature_candidates", True)
    )
    model_args.survival_shield_phases = str(
        getattr(args, "survival_shield_phases", "")
    )
    model_args.lf_shield_base_bot = str(
        getattr(args, "lf_shield_base_bot", "heuristic_safe14")
    )
    model_args.lf_shield_phases = str(
        getattr(args, "lf_shield_phases", "play")
    )
    model_args.lf_shield_min_trick_number = int(
        getattr(args, "lf_shield_min_trick_number", 0)
    )
    model_args.lf_shield_min_damage_delta = float(
        getattr(args, "lf_shield_min_damage_delta", 0.5)
    )
    model_args.lf_shield_min_lane_surplus_delta = int(
        getattr(args, "lf_shield_min_lane_surplus_delta", 1)
    )
    model_args.lf_shield_min_slot_surplus = int(
        getattr(args, "lf_shield_min_slot_surplus", 0)
    )
    model_args.lf_shield_max_buffer_deficit_to_keep = int(
        getattr(args, "lf_shield_max_buffer_deficit_to_keep", -1)
    )
    model_args.lf_shield_led_token_max_damage_increase = float(
        getattr(args, "lf_shield_led_token_max_damage_increase", 0.0)
    )
    model_args.lf_shield_shadow_only = bool(
        getattr(args, "lf_shield_shadow_only", False)
    )
    model_args.liveness_shield_base_mode = str(
        getattr(args, "liveness_shield_base_mode", "q_policy")
    )
    model_args.liveness_shield_phases = str(
        getattr(args, "liveness_shield_phases", "discard,prediction,play")
    )
    model_args.liveness_shield_min_trick_number = int(
        getattr(args, "liveness_shield_min_trick_number", -1)
    )
    model_args.liveness_shield_led_colors = str(
        getattr(args, "liveness_shield_led_colors", "")
    )
    model_args.liveness_shield_min_open_slot_delta = int(
        getattr(args, "liveness_shield_min_open_slot_delta", 1)
    )
    model_args.liveness_shield_min_public_damage_delta = int(
        getattr(args, "liveness_shield_min_public_damage_delta", 2)
    )
    model_args.liveness_shield_max_policy_log_gap = float(
        getattr(args, "liveness_shield_max_policy_log_gap", -1.0)
    )
    model_args.liveness_shield_shadow_only = bool(
        getattr(args, "liveness_shield_shadow_only", False)
    )
    model_args.liveness_shield_sample_limit = int(
        getattr(args, "liveness_shield_sample_limit", 20)
    )
    model_args.residual_anchor_checkpoint = str(
        getattr(args, "residual_anchor_checkpoint", "")
    )
    model_args.residual_policy_phases = str(
        getattr(args, "residual_policy_phases", "discard,prediction,play")
    )
    model_args.residual_delta_clip = float(
        getattr(args, "residual_delta_clip", 1.0)
    )
    model_args.residual_delta_scale = float(
        getattr(args, "residual_delta_scale", 1.0)
    )
    model_args.residual_q_policy_base_margin = float(
        getattr(args, "residual_q_policy_base_margin", 0.0)
    )
    model_args.survival_checkpoint = str(
        getattr(args, "survival_checkpoint", "")
    )
    model_args.survival_value_threshold = float(
        getattr(args, "survival_value_threshold", 0.55)
    )
    model_args.survival_value_max_actions = int(
        getattr(args, "survival_value_max_actions", 0)
    )
    model_args.survival_value_phases = str(
        getattr(args, "survival_value_phases", "")
    )
    model_args.survival_value_scope = str(
        getattr(args, "survival_value_scope", "mean")
    )
    model_args.survival_value_chance_depth = int(
        getattr(args, "survival_value_chance_depth", 3)
    )
    model_args.survival_value_max_chance_outcomes = int(
        getattr(args, "survival_value_max_chance_outcomes", 32)
    )
    model_args.survival_value_max_policy_log_gap = float(
        getattr(args, "survival_value_max_policy_log_gap", -1.0)
    )
  return model, model_args


_NEURAL_BOT_OVERRIDE_TYPES = {
    "action_value_selection_weight": float,
    "action_value_root_only": bool,
    "action_paradox_selection_penalty": float,
    "action_paradox_root_only": bool,
    "action_paradox_rerank_mode": str,
    "action_paradox_risk_threshold": float,
    "action_paradox_min_risk_margin": float,
    "action_paradox_phase_risk_checkpoint": str,
    "action_paradox_phase_risk_phases": str,
    "action_feasibility_selection_weight": float,
    "action_paradox_max_policy_log_gap": float,
    "action_value_rerank_clip": float,
    "action_value_rerank_phases": str,
    "action_value_rerank_min_margin": float,
    "belief_source": str,
    "belief_candidates": int,
    "belief_policy_temperature": float,
    "belief_uniform_mix": float,
    "belief_ref_policy_mix": str,
    "belief_ref_policy_mix_by_phase": str,
    "belief_logprob_floor": float,
    "belief_ranker": str,
    "belief_ranker_candidates": int,
    "belief_ranker_temperature": float,
    "belief_ranker_uniform_mix": float,
    "root_rollout_samples": int,
    "root_rollouts": int,
    "root_rollout_max_actions": int,
    "root_rollout_top_policy": int,
    "root_rollout_include_bots": str,
    "root_rollout_max_plies": int,
    "root_rollout_phases": str,
    "root_rollout_full_match": bool,
    "root_rollout_objective": str,
    "root_rollout_paradox_scope": str,
    "root_rollout_continuation_bot": str,
    "root_rollout_continuation_mode": str,
    "root_rollout_include_continuation_candidate": bool,
    "liveness_shield_base_mode": str,
    "liveness_shield_phases": str,
    "liveness_shield_min_trick_number": int,
    "liveness_shield_led_colors": str,
    "liveness_shield_min_open_slot_delta": int,
    "liveness_shield_min_public_damage_delta": int,
    "liveness_shield_max_policy_log_gap": float,
    "liveness_shield_shadow_only": bool,
    "liveness_shield_sample_limit": int,
    "residual_anchor_checkpoint": str,
    "residual_policy_phases": str,
    "residual_delta_clip": float,
    "residual_delta_scale": float,
    "residual_q_policy_base_margin": float,
    "survival_checkpoint": str,
    "survival_value_threshold": float,
    "survival_value_max_actions": int,
    "survival_value_phases": str,
    "survival_value_scope": str,
    "survival_value_chance_depth": int,
    "survival_value_max_chance_outcomes": int,
    "survival_value_max_policy_log_gap": float,
    "graft_checkpoint": str,
    "graft_builtin_bot": str,
    "base_mode": str,
    "graft_phases": str,
    "graft_mode": str,
    "improve_model": str,
    "improve_threshold": float,
    "improve_risk_threshold": float,
    "improve_phases": str,
    "improve_max_actions": int,
    "improve_top_policy": int,
    "improve_include_bots": str,
    "improve_feature_candidates": bool,
    "improve_min_legal_count": int,
    "improve_max_legal_count": int,
    "improve_min_trick_number": int,
    "improve_min_match_round": int,
    "improve_near_paradox_proxy": bool,
    "improve_near_paradox_min_pressure": float,
    "improve_shadow_only": bool,
    "improve_sample_limit": int,
}


def _parse_bool(value):
  text = str(value).strip().lower()
  if text in ("1", "true", "yes", "y", "on"):
    return True
  if text in ("0", "false", "no", "n", "off"):
    return False
  raise ValueError(f"Expected boolean override, got {value!r}")


def _split_payload_options(payload):
  if "?" not in payload:
    return payload, {}
  payload, raw_options = payload.split("?", 1)
  overrides = {}
  for raw_part in raw_options.split(","):
    part = raw_part.strip()
    if not part:
      continue
    if "=" not in part:
      raise ValueError(f"Bad --neural-bot override {part!r}; expected key=value")
    key, value = part.split("=", 1)
    key = key.strip().replace("-", "_")
    if key not in _NEURAL_BOT_OVERRIDE_TYPES:
      allowed = ", ".join(sorted(_NEURAL_BOT_OVERRIDE_TYPES))
      raise ValueError(f"Unknown --neural-bot override {key!r}; allowed: {allowed}")
    value_type = _NEURAL_BOT_OVERRIDE_TYPES[key]
    value = unquote(value.strip())
    overrides[key] = _parse_bool(value) if value_type is bool else value_type(value)
  return payload, overrides


def _apply_neural_bot_overrides(model_args, overrides):
  for key, value in overrides.items():
    setattr(model_args, key, value)


def _split_phase_names(spec):
  normalized = str(spec or "").replace("|", ",").replace("+", ",")
  return {part.strip().lower() for part in normalized.split(",") if part.strip()}


def load_neural_bots(specs, players, device, args=None):
  neural = {}
  for spec in specs:
    if "=" not in spec:
      raise ValueError(f"Bad --neural-bot spec: {spec}")
    name, payload = spec.split("=", 1)
    payload, overrides = _split_payload_options(payload)
    mode = "policy"
    checkpoint = payload
    if payload.endswith(":belief_policy"):
      checkpoint = payload[:-len(":belief_policy")]
      mode = "belief_policy"
    elif payload.endswith(":belief-policy"):
      checkpoint = payload[:-len(":belief-policy")]
      mode = "belief_policy"
    elif payload.endswith(":residual_q_policy"):
      checkpoint = payload[:-len(":residual_q_policy")]
      mode = "residual_q_policy"
    elif payload.endswith(":residual-q-policy"):
      checkpoint = payload[:-len(":residual-q-policy")]
      mode = "residual_q_policy"
    elif payload.endswith(":residual_q_risk_policy"):
      checkpoint = payload[:-len(":residual_q_risk_policy")]
      mode = "residual_q_risk_policy"
    elif payload.endswith(":residual-q-risk-policy"):
      checkpoint = payload[:-len(":residual-q-risk-policy")]
      mode = "residual_q_risk_policy"
    elif payload.endswith(":q_policy"):
      checkpoint = payload[:-len(":q_policy")]
      mode = "q_policy"
    elif payload.endswith(":q-policy"):
      checkpoint = payload[:-len(":q-policy")]
      mode = "q_policy"
    elif payload.endswith(":root_rollout"):
      checkpoint = payload[:-len(":root_rollout")]
      mode = "root_rollout"
    elif payload.endswith(":root-rollout"):
      checkpoint = payload[:-len(":root-rollout")]
      mode = "root_rollout"
    elif payload.endswith(":value_shield"):
      checkpoint = payload[:-len(":value_shield")]
      mode = "value_shield"
    elif payload.endswith(":value-shield"):
      checkpoint = payload[:-len(":value-shield")]
      mode = "value_shield"
    elif payload.endswith(":liveness_shield"):
      checkpoint = payload[:-len(":liveness_shield")]
      mode = "liveness_shield"
    elif payload.endswith(":liveness-shield"):
      checkpoint = payload[:-len(":liveness-shield")]
      mode = "liveness_shield"
    elif payload.endswith(":residual_policy"):
      checkpoint = payload[:-len(":residual_policy")]
      mode = "residual_policy"
    elif payload.endswith(":residual-policy"):
      checkpoint = payload[:-len(":residual-policy")]
      mode = "residual_policy"
    elif payload.endswith(":play_graft"):
      checkpoint = payload[:-len(":play_graft")]
      mode = "play_graft"
    elif payload.endswith(":play-graft"):
      checkpoint = payload[:-len(":play-graft")]
      mode = "play_graft"
    elif payload.endswith(":improve_graft"):
      checkpoint = payload[:-len(":improve_graft")]
      mode = "improve_graft"
    elif payload.endswith(":improve-graft"):
      checkpoint = payload[:-len(":improve-graft")]
      mode = "improve_graft"
    elif payload.endswith(":belief"):
      checkpoint = payload[:-len(":belief")]
      mode = "belief"
    elif payload.endswith(":mcts"):
      checkpoint = payload[:-len(":mcts")]
      mode = "mcts"
    elif payload.endswith(":policy"):
      checkpoint = payload[:-len(":policy")]
    model, model_args = load_neural(checkpoint, players, device, args)
    _apply_neural_bot_overrides(model_args, overrides)
    entry = {
        "model": model,
        "args": model_args,
        "mode": mode,
        "checkpoint": checkpoint,
        "overrides": overrides,
    }
    if mode in (
        "residual_policy",
        "residual_q_policy",
        "residual_q_risk_policy",
    ):
      anchor_checkpoint = str(
          getattr(model_args, "residual_anchor_checkpoint", "") or ""
      )
      if not anchor_checkpoint:
        raise ValueError(
            f"--neural-bot {name}=...:{mode} requires "
            "residual_anchor_checkpoint"
        )
      anchor_model, anchor_args = load_neural(
          anchor_checkpoint, players, device, args
      )
      _apply_neural_bot_overrides(anchor_args, overrides)
      entry["anchor_model"] = anchor_model
      entry["anchor_args"] = anchor_args
    if mode == "play_graft":
      graft_checkpoint = str(getattr(model_args, "graft_checkpoint", "") or "")
      graft_builtin_bot = str(getattr(model_args, "graft_builtin_bot", "") or "")
      if graft_checkpoint and graft_builtin_bot:
        raise ValueError(
            f"--neural-bot {name}=...:play_graft must use only one of "
            "graft_checkpoint=PATH or graft_builtin_bot=NAME"
        )
      if not graft_checkpoint and not graft_builtin_bot:
        raise ValueError(
            f"--neural-bot {name}=...:play_graft requires "
            "graft_checkpoint=PATH or graft_builtin_bot=NAME"
        )
      if graft_checkpoint:
        graft_model, graft_args = load_neural(graft_checkpoint, players, device, args)
        _apply_neural_bot_overrides(graft_args, overrides)
      else:
        graft_model, graft_args = None, None
      entry.update({
          "graft_model": graft_model,
          "graft_args": graft_args,
          "graft_checkpoint": graft_checkpoint,
          "graft_builtin_bot": graft_builtin_bot,
      })
    if mode == "improve_graft":
      improve_model = str(getattr(model_args, "improve_model", "") or "")
      if not improve_model:
        raise ValueError(
            f"--neural-bot {name}=...:improve_graft requires "
            "improve_model=PATH"
        )
      entry["improve_scorer"] = load_improve_scorer(improve_model, device)
      entry["improve_model"] = improve_model
    if mode == "value_shield":
      survival_checkpoint = str(
          getattr(model_args, "survival_checkpoint", "") or checkpoint
      )
      if survival_checkpoint == checkpoint:
        survival_model = model
        survival_args = model_args
      else:
        survival_model, survival_args = load_neural(
            survival_checkpoint, players, device, args
        )
        _apply_neural_bot_overrides(survival_args, overrides)
      entry.update({
          "survival_checkpoint": survival_checkpoint,
          "survival_model": survival_model,
          "survival_args": survival_args,
      })
    if mode in ("q_policy", "residual_q_policy", "residual_q_risk_policy"):
      phase_risk_checkpoint = str(
          getattr(model_args, "action_paradox_phase_risk_checkpoint", "") or ""
      )
      if phase_risk_checkpoint:
        phase_risk_model, phase_risk_args = load_neural(
            phase_risk_checkpoint, players, device, args
        )
        _apply_neural_bot_overrides(phase_risk_args, overrides)
        entry.update({
            "phase_risk_checkpoint": phase_risk_checkpoint,
            "phase_risk_model": phase_risk_model,
            "phase_risk_args": phase_risk_args,
        })
    neural[name] = entry
  return neural


def unique_preserving_order(names):
  seen = set()
  unique = []
  for name in names:
    if name in seen:
      continue
    seen.add(name)
    unique.append(name)
  return unique


class AZQPolicyBot:
  """Fast root policy plus action-value reranker."""

  def __init__(
      self,
      model,
      name,
      device,
      model_args,
      phase_risk_model=None,
      phase_risk_phases="",
  ):
    self.name = name
    self.model = model
    self.device = device
    self.args = model_args
    self.phase_risk_model = phase_risk_model
    self.phase_risk_phases = _split_phase_names(phase_risk_phases)
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "rerank_disabled": 0,
        "rerank_considered": 0,
        "rerank_abstained_margin": 0,
        "rerank_applied": 0,
        "rerank_overrides": 0,
        "rerank_risk_used": 0,
        "rerank_value_margin_sum": 0.0,
        "rerank_policy_log_gap_sum": 0.0,
        "rerank_risk_diagnostics": 0,
        "rerank_baseline_risk_sum": 0.0,
        "rerank_min_risk_sum": 0.0,
        "rerank_risk_spread_sum": 0.0,
        "rerank_selected_risk_count": 0,
        "rerank_selected_risk_sum": 0.0,
        "rerank_selected_risk_margin_sum": 0.0,
        "rerank_threshold_considered": 0,
        "rerank_threshold_candidates": 0,
        "rerank_threshold_applied": 0,
        "rerank_threshold_missing_risk": 0,
        "rerank_threshold_blocked_baseline_risk": 0,
        "rerank_threshold_blocked_risk_margin": 0,
        "rerank_threshold_blocked_policy_gap": 0,
        "rerank_relative_considered": 0,
        "rerank_relative_candidates": 0,
        "rerank_relative_gap_filtered_candidates": 0,
        "rerank_relative_applied": 0,
        "rerank_relative_missing_risk": 0,
        "rerank_relative_blocked_policy_gap": 0,
        "rerank_by_phase": {},
        "rerank_overrides_by_phase": {},
    })

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def _rerank_enabled_for_phase(self, state):
    spec = str(getattr(self.args, "action_value_rerank_phases", "") or "")
    if not spec.strip():
      return True
    allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
    return self._phase_name(state) in allowed

  def _risk_model_for_phase(self, state):
    if self.phase_risk_model is None or not self.phase_risk_phases:
      return self.model
    if self._phase_name(state) in self.phase_risk_phases:
      return self.phase_risk_model
    return self.model

  def step(self, state, player):
    legal = state.legal_actions(player)
    legal_count = len(legal)
    if len(legal) == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return legal[0]
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    baseline_action = max(legal, key=lambda action: policy[action])
    weight = float(getattr(self.args, "action_value_selection_weight", 0.0))
    risk_penalty_weight = float(
        getattr(self.args, "action_paradox_selection_penalty", 0.0)
    )
    risk_mode = str(
        getattr(self.args, "action_paradox_rerank_mode", "additive")
        or "additive"
    ).lower()
    feasibility_weight = float(
        getattr(self.args, "action_feasibility_selection_weight", 0.0)
    )
    if (
        (
            weight <= 0
            and risk_penalty_weight <= 0
            and feasibility_weight == 0.0
            and risk_mode not in ("threshold", "relative")
        )
        or not self._rerank_enabled_for_phase(state)
    ):
      self._decision_stats["rerank_disabled"] += 1
      _record_decision(
          self._decision_stats, state, baseline_action, legal_count
      )
      return baseline_action
    phase = self._phase_name(state)
    self._decision_stats["rerank_considered"] += 1
    _inc_counter(self._decision_stats["rerank_by_phase"], phase)
    action_values = (
        model_action_values(
            self.model, state, player, state.num_distinct_actions(), self.device
        )
        if weight > 0 else
        np.zeros(state.num_distinct_actions(), dtype=np.float32)
    )
    clip = max(0.0, float(getattr(self.args, "action_value_rerank_clip", 0.5)))
    if clip > 0:
      action_values = np.clip(action_values, -clip, clip)
    legal_values = np.array([float(action_values[action]) for action in legal])
    if len(legal_values) > 1:
      top_two = np.partition(legal_values, -2)[-2:]
      value_margin = float(np.max(top_two) - np.min(top_two))
    else:
      value_margin = 0.0
    min_margin = float(getattr(self.args, "action_value_rerank_min_margin", 0.0))
    if min_margin > 0 and len(legal) > 1:
      if value_margin < min_margin:
        self._decision_stats["rerank_abstained_margin"] += 1
        _record_decision(
            self._decision_stats, state, baseline_action, legal_count
        )
        return baseline_action
    if risk_penalty_weight > 0:
      self._decision_stats["rerank_risk_used"] += 1
    elif risk_mode in ("threshold", "relative"):
      self._decision_stats["rerank_risk_used"] += 1
    action_risks = (
        model_action_risks(
            self._risk_model_for_phase(state),
            state,
            player,
            state.num_distinct_actions(),
            self.device,
        )
        if risk_penalty_weight > 0 or risk_mode in ("threshold", "relative") else
        None
    )
    feasibility_scores = (
        action_feasibility_scores(
            action_feature_matrix(
                state, player, state.num_distinct_actions()
            )
        )
        if feasibility_weight != 0.0 else
        None
    )
    selected_action = q_policy_select_action(
        legal,
        policy,
        action_values,
        action_risks,
        self.args,
        getattr(self.args, "value_scale", 1.0),
        stats=self._decision_stats,
        feasibility_scores=feasibility_scores,
    )
    self._decision_stats["rerank_applied"] += 1
    self._decision_stats["rerank_value_margin_sum"] += value_margin
    policy_gap = np.log(max(float(policy[baseline_action]), 1e-12)) - np.log(
        max(float(policy[selected_action]), 1e-12)
    )
    self._decision_stats["rerank_policy_log_gap_sum"] += float(policy_gap)
    if selected_action != baseline_action:
      self._decision_stats["rerank_overrides"] += 1
      _inc_counter(self._decision_stats["rerank_overrides_by_phase"], phase)
    _record_decision(self._decision_stats, state, selected_action, legal_count)
    return selected_action

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class AZResidualPolicyBot:
  """Candidate policy as a bounded residual over an anchor checkpoint."""

  def __init__(self, model, anchor_model, name, device, args, anchor_args=None):
    self.name = name
    self.model = model
    self.anchor_model = anchor_model
    self.device = device
    self.args = args
    self.anchor_args = anchor_args if anchor_args is not None else args
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "residual_disabled": 0,
        "residual_considered": 0,
        "residual_overrides": 0,
        "residual_delta_clip": float(getattr(args, "residual_delta_clip", 1.0)),
        "residual_delta_scale": float(
            getattr(args, "residual_delta_scale", 1.0)
        ),
        "residual_by_phase": {},
        "residual_overrides_by_phase": {},
        "residual_abs_delta_sum": 0.0,
        "residual_selected_abs_delta_sum": 0.0,
        "residual_samples": [],
    })

  def _enabled_phases(self):
    spec = str(getattr(self.args, "residual_policy_phases", "") or "").strip()
    if not spec:
      return None
    return _split_phase_names(spec)

  def _enabled_for_state(self, state):
    phases = self._enabled_phases()
    return phases is None or _phase_name(state) in phases

  def _policies(self, state, player, num_actions):
    anchor_policy, _ = model_policy_value(
        self.anchor_model,
        state,
        player,
        num_actions,
        float(getattr(
            self.anchor_args,
            "value_scale",
            getattr(self.args, "value_scale", 20.0),
        )),
        self.device,
    )
    candidate_policy, _ = model_policy_value(
        self.model,
        state,
        player,
        num_actions,
        float(getattr(self.args, "value_scale", 20.0)),
        self.device,
    )
    return anchor_policy, candidate_policy

  def _record_sample(
      self, state, player, anchor_action, selected_action, selected_delta
  ):
    samples = self._decision_stats["residual_samples"]
    if len(samples) >= 20:
      return
    samples.append({
        "phase": _phase_name(state),
        "player": int(player),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "anchor_action": int(anchor_action),
        "selected_action": int(selected_action),
        "selected_delta": round(float(selected_delta), 6),
        "legal_count": int(len(state.legal_actions(player))),
    })

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    anchor_policy, candidate_policy = self._policies(
        state, player, state.num_distinct_actions()
    )
    anchor_action = int(max(legal, key=lambda action: anchor_policy[action]))
    if not self._enabled_for_state(state):
      self._decision_stats["residual_disabled"] += 1
      _record_decision(self._decision_stats, state, anchor_action, legal_count)
      return anchor_action
    self._decision_stats["residual_considered"] += 1
    phase = _phase_name(state)
    _inc_counter(self._decision_stats["residual_by_phase"], phase)
    eps = 1e-12
    legal_anchor_log = {
        int(action): math.log(max(eps, float(anchor_policy[action])))
        for action in legal
    }
    legal_candidate_log = {
        int(action): math.log(max(eps, float(candidate_policy[action])))
        for action in legal
    }
    clip = float(getattr(self.args, "residual_delta_clip", 1.0))
    scale = float(getattr(self.args, "residual_delta_scale", 1.0))
    scores = {}
    deltas = {}
    for action in legal:
      action = int(action)
      delta = legal_candidate_log[action] - legal_anchor_log[action]
      if clip >= 0:
        delta = max(-clip, min(clip, delta))
      delta *= scale
      deltas[action] = delta
      scores[action] = legal_anchor_log[action] + delta
      self._decision_stats["residual_abs_delta_sum"] += abs(float(delta))
    selected_action = int(max(legal, key=lambda action: scores[int(action)]))
    selected_delta = float(deltas[selected_action])
    self._decision_stats["residual_selected_abs_delta_sum"] += abs(
        selected_delta
    )
    if selected_action != anchor_action:
      self._decision_stats["residual_overrides"] += 1
      _inc_counter(self._decision_stats["residual_overrides_by_phase"], phase)
      self._record_sample(
          state, player, anchor_action, selected_action, selected_delta
      )
    _record_decision(self._decision_stats, state, selected_action, legal_count)
    return selected_action

  def decision_stats(self):
    stats = json.loads(json.dumps(self._decision_stats))
    considered = float(stats.get("residual_considered", 0) or 0)
    if considered:
      stats["residual_override_rate_when_considered"] = round(
          float(stats.get("residual_overrides", 0)) / considered, 4
      )
      stats["residual_selected_abs_delta_avg"] = round(
          float(stats.get("residual_selected_abs_delta_sum", 0.0)) /
          considered,
          6,
      )
    delta_count = sum(
        int(count) for count in stats.get("residual_by_phase", {}).values()
    )
    if delta_count:
      stats["residual_abs_delta_avg_per_decision"] = round(
          float(stats.get("residual_abs_delta_sum", 0.0)) / delta_count,
          6,
      )
    _decision_stats_rates(stats)
    return stats


class AZResidualQPolicyBot(AZResidualPolicyBot):
  """Candidate policy as a bounded residual over the deployed q-policy action."""

  def __init__(
      self,
      model,
      anchor_model,
      name,
      device,
      args,
      anchor_args=None,
      phase_risk_model=None,
      phase_risk_phases="",
  ):
    super().__init__(
        model,
        anchor_model,
        name,
        device,
        args,
        anchor_args=anchor_args,
    )
    self.base_bot = AZQPolicyBot(
        anchor_model,
        f"{name}_q_base",
        device,
        self.anchor_args,
        phase_risk_model=phase_risk_model,
        phase_risk_phases=phase_risk_phases,
    )
    self._decision_stats.update({
        "residual_base_mode": "q_policy",
        "residual_q_policy_base_margin": float(
            getattr(args, "residual_q_policy_base_margin", 0.0)
        ),
        "residual_q_policy_base_over_raw": 0,
    })

  def _base_action_and_policy(self, state, player, num_actions):
    anchor_policy, _ = model_policy_value(
        self.anchor_model,
        state,
        player,
        num_actions,
        float(getattr(
            self.anchor_args,
            "value_scale",
            getattr(self.args, "value_scale", 20.0),
        )),
        self.device,
    )
    return int(self.base_bot.step(state, player)), anchor_policy

  def _candidate_policy(self, state, player, num_actions):
    candidate_policy, _ = model_policy_value(
        self.model,
        state,
        player,
        num_actions,
        float(getattr(self.args, "value_scale", 20.0)),
        self.device,
    )
    return candidate_policy

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    num_actions = state.num_distinct_actions()
    base_action, anchor_policy = self._base_action_and_policy(
        state, player, num_actions
    )
    candidate_policy = self._candidate_policy(state, player, num_actions)
    if not self._enabled_for_state(state):
      self._decision_stats["residual_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return base_action
    self._decision_stats["residual_considered"] += 1
    phase = _phase_name(state)
    _inc_counter(self._decision_stats["residual_by_phase"], phase)
    eps = 1e-12
    legal_anchor_log = {
        int(action): math.log(max(eps, float(anchor_policy[action])))
        for action in legal
    }
    raw_top_action = int(max(legal, key=lambda action: anchor_policy[action]))
    if base_action != raw_top_action:
      self._decision_stats["residual_q_policy_base_over_raw"] += 1
    legal_candidate_log = {
        int(action): math.log(max(eps, float(candidate_policy[action])))
        for action in legal
    }
    base_scores = dict(legal_anchor_log)
    margin = float(getattr(self.args, "residual_q_policy_base_margin", 0.0))
    tie_margin = margin if margin > 0.0 else 1e-9
    best_raw_score = max(base_scores.values())
    base_scores[base_action] = max(
        float(base_scores[base_action]),
        best_raw_score,
    ) + tie_margin
    clip = float(getattr(self.args, "residual_delta_clip", 1.0))
    scale = float(getattr(self.args, "residual_delta_scale", 1.0))
    scores = {}
    deltas = {}
    for action in legal:
      action = int(action)
      delta = legal_candidate_log[action] - legal_anchor_log[action]
      if clip >= 0:
        delta = max(-clip, min(clip, delta))
      delta *= scale
      deltas[action] = delta
      scores[action] = float(base_scores[action]) + delta
      self._decision_stats["residual_abs_delta_sum"] += abs(float(delta))
    selected_action = int(max(legal, key=lambda action: scores[int(action)]))
    selected_delta = float(deltas[selected_action])
    self._decision_stats["residual_selected_abs_delta_sum"] += abs(
        selected_delta
    )
    if selected_action != base_action:
      self._decision_stats["residual_overrides"] += 1
      _inc_counter(self._decision_stats["residual_overrides_by_phase"], phase)
      self._record_sample(
          state, player, base_action, selected_action, selected_delta
      )
    _record_decision(self._decision_stats, state, selected_action, legal_count)
    return selected_action


class AZResidualQRiskPolicyBot(AZResidualQPolicyBot):
  """Protected anchor q-policy reranked by candidate risk/value heads."""

  def __init__(
      self,
      model,
      anchor_model,
      name,
      device,
      args,
      anchor_args=None,
      phase_risk_model=None,
      phase_risk_phases="",
  ):
    super().__init__(
        model,
        anchor_model,
        name,
        device,
        args,
        anchor_args=anchor_args,
        phase_risk_model=phase_risk_model,
        phase_risk_phases=phase_risk_phases,
    )
    self.phase_risk_model = phase_risk_model
    self.phase_risk_phases = _split_phase_names(phase_risk_phases)
    self._decision_stats["residual_signal"] = "risk_value"

  def _risk_model_for_phase(self, state):
    if self.phase_risk_model is None or not self.phase_risk_phases:
      return self.model
    if _phase_name(state) in self.phase_risk_phases:
      return self.phase_risk_model
    return self.model

  def _candidate_action_values(self, state, player, num_actions):
    return model_action_values(self.model, state, player, num_actions, self.device)

  def _candidate_action_risks(self, state, player, num_actions):
    return model_action_risks(
        self._risk_model_for_phase(state),
        state,
        player,
        num_actions,
        self.device,
    )

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    num_actions = state.num_distinct_actions()
    base_action, anchor_policy = self._base_action_and_policy(
        state, player, num_actions
    )
    if not self._enabled_for_state(state):
      self._decision_stats["residual_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return base_action

    weight = float(getattr(self.args, "action_value_selection_weight", 0.0))
    risk_penalty_weight = float(
        getattr(self.args, "action_paradox_selection_penalty", 0.0)
    )
    risk_mode = str(
        getattr(self.args, "action_paradox_rerank_mode", "additive")
        or "additive"
    ).lower()
    feasibility_weight = float(
        getattr(self.args, "action_feasibility_selection_weight", 0.0)
    )
    if (
        weight <= 0
        and risk_penalty_weight <= 0
        and feasibility_weight == 0.0
        and risk_mode not in ("threshold", "relative")
    ):
      self._decision_stats["residual_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return base_action

    self._decision_stats["residual_considered"] += 1
    phase = _phase_name(state)
    _inc_counter(self._decision_stats["residual_by_phase"], phase)
    protected_policy = np.asarray(anchor_policy, dtype=np.float32).copy()
    raw_top_action = int(max(legal, key=lambda action: anchor_policy[action]))
    if base_action != raw_top_action:
      self._decision_stats["residual_q_policy_base_over_raw"] += 1
    best_raw_prob = max(float(protected_policy[action]) for action in legal)
    protected_policy[base_action] = max(
        float(protected_policy[base_action]),
        best_raw_prob,
    ) + 1e-9

    action_values = (
        self._candidate_action_values(state, player, num_actions)
        if weight > 0 else
        np.zeros(num_actions, dtype=np.float32)
    )
    clip = max(0.0, float(getattr(self.args, "action_value_rerank_clip", 0.5)))
    if clip > 0:
      action_values = np.clip(action_values, -clip, clip)
    min_margin = float(getattr(self.args, "action_value_rerank_min_margin", 0.0))
    if min_margin > 0 and len(legal) > 1:
      legal_values = np.array([float(action_values[action]) for action in legal])
      top_two = np.partition(legal_values, -2)[-2:]
      value_margin = float(np.max(top_two) - np.min(top_two))
      if value_margin < min_margin:
        self._decision_stats["rerank_abstained_margin"] = int(
            self._decision_stats.get("rerank_abstained_margin", 0)
        ) + 1
        _record_decision(self._decision_stats, state, base_action, legal_count)
        return base_action
    action_risks = (
        self._candidate_action_risks(state, player, num_actions)
        if risk_penalty_weight > 0 or risk_mode in ("threshold", "relative") else
        None
    )
    feasibility_scores = (
        action_feasibility_scores(action_feature_matrix(state, player, num_actions))
        if feasibility_weight != 0.0 else
        None
    )
    selected_action = int(q_policy_select_action(
        legal,
        protected_policy,
        action_values,
        action_risks,
        self.args,
        getattr(self.args, "value_scale", 1.0),
        stats=self._decision_stats,
        feasibility_scores=feasibility_scores,
    ))
    self._decision_stats["rerank_applied"] = int(
        self._decision_stats.get("rerank_applied", 0)
    ) + 1
    if selected_action != base_action:
      self._decision_stats["residual_overrides"] += 1
      _inc_counter(self._decision_stats["residual_overrides_by_phase"], phase)
      selected_delta = (
          float(action_risks[base_action]) - float(action_risks[selected_action])
          if action_risks is not None else
          float(action_values[selected_action]) - float(action_values[base_action])
      )
      self._record_sample(
          state, player, base_action, selected_action, selected_delta
      )
    _record_decision(self._decision_stats, state, selected_action, legal_count)
    return selected_action


class AZLivenessShieldPolicyBot:
  """Baseline-protected neural policy with deterministic liveness guards."""

  def __init__(self, model, name, device, model_args):
    self.name = name
    self.model = model
    self.device = device
    self.args = model_args
    base_mode = str(getattr(model_args, "liveness_shield_base_mode", "q_policy"))
    if base_mode == "q_policy":
      self.base_bot = AZQPolicyBot(model, f"{name}_base", device, model_args)
    elif base_mode == "policy":
      self.base_bot = AZPolicyBot(
          model, f"{name}_base", device, model_args.value_scale
      )
    else:
      raise ValueError(f"Unsupported liveness shield base mode: {base_mode}")
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "liveness_shield_disabled": 0,
        "liveness_shield_considered": 0,
        "liveness_shield_base_kept": 0,
        "liveness_shield_overrides": 0,
        "liveness_shield_shadow_overrides": 0,
        "liveness_shield_blocked_policy_gap": 0,
        "liveness_shield_no_scores": 0,
        "liveness_shield_base_failure_sum": 0.0,
        "liveness_shield_selected_failure_sum": 0.0,
        "liveness_shield_base_public_damage_sum": 0.0,
        "liveness_shield_selected_public_damage_sum": 0.0,
        "liveness_shield_base_min_open_slots_sum": 0.0,
        "liveness_shield_selected_min_open_slots_sum": 0.0,
        "liveness_shield_by_phase": {},
        "liveness_shield_overrides_by_phase": {},
        "liveness_shield_base_mode": base_mode,
        "liveness_shield_override_samples": [],
    })

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def _enabled_for_phase(self, state):
    spec = str(getattr(self.args, "liveness_shield_phases", "") or "").strip()
    if not spec:
      return True
    return self._phase_name(state).lower() in _split_phase_names(spec)

  def _enabled_for_state(self, state):
    if not self._enabled_for_phase(state):
      return False
    min_trick = int(getattr(self.args, "liveness_shield_min_trick_number", -1))
    if min_trick >= 0 and int(getattr(state, "_trick_number", -1)) < min_trick:
      return False
    led_spec = str(getattr(self.args, "liveness_shield_led_colors", "") or "")
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

  def _policy(self, state, player):
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    return policy

  def _within_policy_gap(self, policy, base_action, candidate_action):
    max_gap = float(
        getattr(self.args, "liveness_shield_max_policy_log_gap", -1.0)
    )
    if max_gap < 0:
      return True
    gap = np.log(max(float(policy[int(base_action)]), 1e-12)) - np.log(
        max(float(policy[int(candidate_action)]), 1e-12)
    )
    return float(gap) <= max_gap

  def _score_actions(self, state, player, legal, base_action):
    legal = sorted(int(action) for action in legal)
    non_paradox = [action for action in legal if action != 999]
    if non_paradox:
      legal = non_paradox
    return {
        int(action): action_liveness_certificate(
            state, player, action, base_action=base_action
        )
        for action in legal
    }

  def _candidate_key(self, item, base_action):
    action, row = item
    return action_liveness_key(row, action, base_action=base_action)

  def _dominates_base(self, base_row, candidate_row):
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
    min_open_delta = int(
        getattr(self.args, "liveness_shield_min_open_slot_delta", 1)
    )
    if open_delta >= min_open_delta:
      return True
    damage_delta = int(base_row.get("public_slot_damage", 0)) - int(
        candidate_row.get("public_slot_damage", 0)
    )
    min_damage_delta = int(
        getattr(self.args, "liveness_shield_min_public_damage_delta", 2)
    )
    if damage_delta >= min_damage_delta:
      return True
    if (
        bool(base_row.get("lost_led_token", False))
        and not bool(candidate_row.get("lost_led_token", False))
        and damage_delta >= 0
    ):
      return True
    if (
        bool(base_row.get("over_target_would_win", False))
        and not bool(candidate_row.get("over_target_would_win", False))
        and damage_delta >= 0
    ):
      return True
    return False

  def _append_override_sample(
      self, state, player, legal, base_action, selected_action, base_row,
      selected_row, shadow
  ):
    samples = self._decision_stats["liveness_shield_override_samples"]
    limit = int(getattr(self.args, "liveness_shield_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": self._phase_name(state).lower(),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "legal_count": int(len(legal)),
        "base_action": int(base_action),
        "selected_action": int(selected_action),
        "shadow": bool(shadow),
        "base_liveness": dict(base_row),
        "selected_liveness": dict(selected_row),
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    base_action = int(self.base_bot.step(state.clone(), player))
    if base_action not in legal:
      policy = self._policy(state, player)
      base_action = int(max(legal, key=lambda action: policy[action]))
    if not self._enabled_for_state(state):
      self._decision_stats["liveness_shield_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = self._phase_name(state).lower()
    self._decision_stats["liveness_shield_considered"] += 1
    _inc_counter(self._decision_stats["liveness_shield_by_phase"], phase)
    policy = self._policy(state, player)
    scored = self._score_actions(state, player, legal, base_action)
    if not scored:
      self._decision_stats["liveness_shield_no_scores"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    base_row = scored.get(int(base_action))
    if base_row is None:
      base_row = action_liveness_certificate(
          state, player, base_action, base_action=base_action
      )
    selected_action, selected_row = max(
        scored.items(), key=lambda item: self._candidate_key(item, base_action)
    )
    selected_action = int(selected_action)
    should_override = (
        selected_action != int(base_action)
        and self._dominates_base(base_row, selected_row)
    )
    if should_override and not self._within_policy_gap(
        policy, base_action, selected_action
    ):
      self._decision_stats["liveness_shield_blocked_policy_gap"] += 1
      should_override = False
    shadow_only = bool(getattr(self.args, "liveness_shield_shadow_only", False))
    final_action = int(base_action)
    final_row = base_row
    if should_override:
      if shadow_only:
        self._decision_stats["liveness_shield_shadow_overrides"] += 1
        self._append_override_sample(
            state, player, legal, base_action, selected_action, base_row,
            selected_row, shadow=True
        )
      else:
        final_action = selected_action
        final_row = selected_row
        self._decision_stats["liveness_shield_overrides"] += 1
        _inc_counter(
            self._decision_stats["liveness_shield_overrides_by_phase"], phase
        )
        self._append_override_sample(
            state, player, legal, base_action, selected_action, base_row,
            selected_row, shadow=False
        )
    else:
      self._decision_stats["liveness_shield_base_kept"] += 1
    self._decision_stats["liveness_shield_base_failure_sum"] += float(
        base_row.get("liveness_failure_count", 0)
    )
    self._decision_stats["liveness_shield_selected_failure_sum"] += float(
        final_row.get("liveness_failure_count", 0)
    )
    self._decision_stats["liveness_shield_base_public_damage_sum"] += float(
        base_row.get("public_slot_damage", 0)
    )
    self._decision_stats["liveness_shield_selected_public_damage_sum"] += float(
        final_row.get("public_slot_damage", 0)
    )
    self._decision_stats["liveness_shield_base_min_open_slots_sum"] += float(
        base_row.get("min_player_open_slots_after", 0)
    )
    self._decision_stats["liveness_shield_selected_min_open_slots_sum"] += float(
        final_row.get("min_player_open_slots_after", 0)
    )
    _record_decision(self._decision_stats, state, final_action, legal_count)
    return int(final_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    considered = float(stats.get("liveness_shield_considered", 0) or 0)
    if considered:
      stats["liveness_shield_base_failure_avg"] = round(
          float(stats.get("liveness_shield_base_failure_sum", 0.0)) /
          considered,
          6,
      )
      stats["liveness_shield_selected_failure_avg"] = round(
          float(stats.get("liveness_shield_selected_failure_sum", 0.0)) /
          considered,
          6,
      )
      stats["liveness_shield_base_public_damage_avg"] = round(
          float(stats.get("liveness_shield_base_public_damage_sum", 0.0)) /
          considered,
          6,
      )
      stats["liveness_shield_selected_public_damage_avg"] = round(
          float(stats.get("liveness_shield_selected_public_damage_sum", 0.0)) /
          considered,
          6,
      )
      stats["liveness_shield_base_min_open_slots_avg"] = round(
          float(stats.get("liveness_shield_base_min_open_slots_sum", 0.0)) /
          considered,
          6,
      )
      stats["liveness_shield_selected_min_open_slots_avg"] = round(
          float(stats.get("liveness_shield_selected_min_open_slots_sum", 0.0)) /
          considered,
          6,
      )
    return stats

  def raw_decision_stats(self):
    stats = json.loads(json.dumps(self._decision_stats))
    if hasattr(self.base_bot, "raw_decision_stats"):
      stats["nested"] = {"base_bot": self.base_bot.raw_decision_stats()}
    elif hasattr(self.base_bot, "decision_stats"):
      stats["nested"] = {"base_bot": self.base_bot.decision_stats()}
    return stats


class AZValueShieldPolicyBot:
  """Raw policy constrained by a post-action round-survival value model."""

  def __init__(self, policy_model, survival_model, name, device, model_args):
    self.name = name
    self.policy_model = policy_model
    self.survival_model = survival_model
    self.device = device
    self.args = model_args
    self.players = int(getattr(model_args, "players", 3))
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "value_shield_disabled": 0,
        "value_shield_considered": 0,
        "value_shield_base_kept": 0,
        "value_shield_overrides": 0,
        "value_shield_fallback_max_survival": 0,
        "value_shield_no_scores": 0,
        "value_shield_paradox_candidate_filtered": 0,
        "value_shield_scored_candidates": 0,
        "value_shield_base_survival_sum": 0.0,
        "value_shield_selected_survival_sum": 0.0,
        "value_shield_max_survival_sum": 0.0,
        "value_shield_by_phase": {},
        "value_shield_overrides_by_phase": {},
        "value_shield_fallback_by_phase": {},
        "value_shield_threshold": float(
            getattr(model_args, "survival_value_threshold", 0.55)
        ),
    })

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  @staticmethod
  def _survival_probability(raw_value):
    return float(np.clip((float(raw_value) + 1.0) / 2.0, 0.0, 1.0))

  @staticmethod
  def _selection_key(action, survival, policy):
    return (float(policy[int(action)]), float(survival), -int(action))

  def _enabled_for_phase(self, state):
    spec = str(getattr(self.args, "survival_value_phases", "") or "").strip()
    if not spec:
      return True
    allowed = _split_phase_names(spec)
    return self._phase_name(state).lower() in allowed

  def _any_paradox(self, state):
    paradoxes = getattr(state, "_has_paradoxed", [False] * self.players)
    return any(bool(value) for value in paradoxes)

  def _candidate_actions(self, legal, policy, base_action):
    legal = sorted(int(action) for action in legal)
    non_paradox_legal = [action for action in legal if action != 999]
    if non_paradox_legal:
      if 999 in legal:
        self._decision_stats["value_shield_paradox_candidate_filtered"] += 1
      legal = non_paradox_legal
    max_actions = int(getattr(self.args, "survival_value_max_actions", 0) or 0)
    if max_actions <= 0 or len(legal) <= max_actions:
      return legal
    selected = []
    if int(base_action) in legal:
      selected.append(int(base_action))
    ranked = sorted(
        legal,
        key=lambda action: (float(policy[int(action)]), -int(action)),
        reverse=True,
    )
    for action in ranked:
      if action not in selected:
        selected.append(action)
      if len(selected) >= max_actions:
        break
    return sorted(selected)

  def _chance_children(self, state):
    outcomes = list(state.chance_outcomes())
    max_outcomes = int(
        getattr(self.args, "survival_value_max_chance_outcomes", 32) or 0
    )
    if max_outcomes > 0 and len(outcomes) > max_outcomes:
      outcomes = sorted(outcomes, key=lambda item: float(item[1]), reverse=True)[
          :max_outcomes
      ]
    total_prob = sum(float(prob) for _action, prob in outcomes)
    if total_prob <= 0:
      return []
    children = []
    for action, prob in outcomes:
      child = state.clone()
      child.apply_action(int(action))
      children.append((child, float(prob) / total_prob))
    return children

  def _state_survival_score(self, state, acting_player, chance_depth=0):
    if self._any_paradox(state):
      return 0.0
    if state.is_terminal():
      return 1.0
    if state.is_chance_node():
      max_depth = max(
          0, int(getattr(self.args, "survival_value_chance_depth", 3) or 0)
      )
      if chance_depth >= max_depth:
        return 0.5
      children = self._chance_children(state)
      if not children:
        return 0.5
      return float(sum(
          prob * self._state_survival_score(child, acting_player, chance_depth + 1)
          for child, prob in children
      ))
    eval_player = int(state.current_player())
    if eval_player < 0 or eval_player >= self.players:
      return 0.5
    try:
      _policy, values = model_policy_value(
          self.survival_model,
          state,
          eval_player,
          state.num_distinct_actions(),
          1.0,
          self.device,
      )
    except Exception:
      return 0.5
    values = np.array(values, dtype=np.float32)
    scope = str(getattr(self.args, "survival_value_scope", "mean") or "mean")
    if scope == "acting" and 0 <= int(acting_player) < len(values):
      raw_value = float(values[int(acting_player)])
    elif scope == "current" and 0 <= eval_player < len(values):
      raw_value = float(values[eval_player])
    else:
      raw_value = float(np.mean(values[:self.players]))
    return self._survival_probability(raw_value)

  def _score_candidates(self, state, player, candidates):
    scores = {}
    for action in candidates:
      rollout = state.clone()
      rollout.apply_action(int(action))
      scores[int(action)] = self._state_survival_score(rollout, player)
    return scores

  def _within_policy_gap(self, policy, base_action, action):
    max_gap = float(
        getattr(self.args, "survival_value_max_policy_log_gap", -1.0)
    )
    if max_gap < 0:
      return True
    base_log = math.log(max(float(policy[int(base_action)]), 1e-12))
    action_log = math.log(max(float(policy[int(action)]), 1e-12))
    return (base_log - action_log) <= max_gap

  def step(self, state, player):
    legal = state.legal_actions(player)
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return legal[0]
    policy, _ = model_policy_value(
        self.policy_model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    base_action = max(legal, key=lambda action: policy[action])
    if not self._enabled_for_phase(state):
      self._decision_stats["value_shield_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = self._phase_name(state).lower()
    self._decision_stats["value_shield_considered"] += 1
    _inc_counter(self._decision_stats["value_shield_by_phase"], phase)
    candidates = self._candidate_actions(legal, policy, base_action)
    scores = self._score_candidates(state, player, candidates)
    self._decision_stats["value_shield_scored_candidates"] += len(scores)
    if not scores:
      self._decision_stats["value_shield_no_scores"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    threshold = float(getattr(self.args, "survival_value_threshold", 0.55))
    max_action, max_survival = max(
        scores.items(),
        key=lambda item: (float(item[1]), float(policy[int(item[0])]), -int(item[0])),
    )
    base_scored = int(base_action) in scores
    base_survival = float(scores.get(int(base_action), 0.0))
    self._decision_stats["value_shield_base_survival_sum"] += base_survival
    self._decision_stats["value_shield_max_survival_sum"] += float(max_survival)
    if base_scored and base_survival >= threshold:
      selected_action = int(base_action)
      selected_survival = base_survival
      self._decision_stats["value_shield_base_kept"] += 1
    else:
      safe_candidates = {
          action: survival
          for action, survival in scores.items()
          if (
              float(survival) >= threshold
              and self._within_policy_gap(policy, base_action, action)
          )
      }
      if safe_candidates:
        selected_action, selected_survival = max(
            safe_candidates.items(),
            key=lambda item: self._selection_key(item[0], item[1], policy),
        )
        self._decision_stats["value_shield_overrides"] += int(
            selected_action != int(base_action)
        )
        if selected_action != int(base_action):
          _inc_counter(self._decision_stats["value_shield_overrides_by_phase"], phase)
      else:
        selected_action, selected_survival = int(max_action), float(max_survival)
        self._decision_stats["value_shield_fallback_max_survival"] += 1
        _inc_counter(self._decision_stats["value_shield_fallback_by_phase"], phase)
        if selected_action != int(base_action):
          self._decision_stats["value_shield_overrides"] += 1
          _inc_counter(self._decision_stats["value_shield_overrides_by_phase"], phase)
    self._decision_stats["value_shield_selected_survival_sum"] += float(
        selected_survival
    )
    _record_decision(self._decision_stats, state, selected_action, legal_count)
    return int(selected_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class AZPhaseGraftBot:
  """Use a graft actor in selected phases and a base actor elsewhere."""

  def __init__(self, base_bot, graft_bot, name, graft_phases="play"):
    self.base_bot = base_bot
    self.graft_bot = graft_bot
    self.name = name
    self.graft_phases = _split_phase_names(graft_phases or "play")
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "base_phase_decisions": 0,
        "graft_phase_decisions": 0,
        "graft_overrides": 0,
        "graft_by_phase": {},
        "graft_overrides_by_phase": {},
    })

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def step(self, state, player):
    legal_count = len(state.legal_actions(player))
    phase = self._phase_name(state).lower()
    if legal_count <= 1:
      action = state.legal_actions(player)[0]
      _record_decision(self._decision_stats, state, action, legal_count)
      return action
    if phase in self.graft_phases:
      base_action = self.base_bot.step(state.clone(), player)
      graft_action = self.graft_bot.step(state, player)
      self._decision_stats["graft_phase_decisions"] += 1
      _inc_counter(self._decision_stats["graft_by_phase"], phase)
      if graft_action != base_action:
        self._decision_stats["graft_overrides"] += 1
        _inc_counter(self._decision_stats["graft_overrides_by_phase"], phase)
      _record_decision(
          self._decision_stats, state, graft_action, legal_count
      )
      return graft_action
    action = self.base_bot.step(state, player)
    self._decision_stats["base_phase_decisions"] += 1
    _record_decision(self._decision_stats, state, action, legal_count)
    return action

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    stats = json.loads(json.dumps(self._decision_stats))
    nested = {}
    if hasattr(self.base_bot, "raw_decision_stats"):
      nested["base_bot"] = self.base_bot.raw_decision_stats()
    elif hasattr(self.base_bot, "decision_stats"):
      nested["base_bot"] = self.base_bot.decision_stats()
    if hasattr(self.graft_bot, "raw_decision_stats"):
      nested["graft_bot"] = self.graft_bot.raw_decision_stats()
    elif hasattr(self.graft_bot, "decision_stats"):
      nested["graft_bot"] = self.graft_bot.decision_stats()
    if nested:
      stats["nested"] = nested
    return stats


class AZRootRolloutBot:
  """Root action evaluator using common belief particles and rollout seeds."""

  def __init__(self, model, name, device, model_args):
    self.name = name
    self.model = model
    self.device = device
    self.args = model_args
    continuation_bot_name = str(
        getattr(model_args, "root_rollout_continuation_bot", "") or ""
    ).strip()
    players = int(getattr(model_args, "players", 3))
    self._continuation_bots = (
        [make_bot(continuation_bot_name, seed=seat) for seat in range(players)]
        if continuation_bot_name else
        []
    )
    continuation_mode = str(
        getattr(model_args, "root_rollout_continuation_mode", "") or ""
    ).strip().lower().replace("-", "_")
    if self._continuation_bots:
      continuation_mode = ""
    if continuation_mode == "root_rollout":
      raise ValueError("root_rollout cannot be used as its own continuation mode")
    if continuation_mode == "policy":
      self._continuation_neural_bot = AZPolicyBot(
          model, f"{name}_continuation_policy", device, model_args.value_scale
      )
    elif continuation_mode == "q_policy":
      self._continuation_neural_bot = AZQPolicyBot(
          model, f"{name}_continuation_q_policy", device, model_args
      )
    elif continuation_mode == "liveness_shield":
      self._continuation_neural_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_continuation_liveness_shield", device, model_args
      )
    elif continuation_mode:
      raise ValueError(f"Unsupported root rollout continuation mode: {continuation_mode}")
    else:
      self._continuation_neural_bot = None
    self._continuation_mode = continuation_mode
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "root_rollout_disabled": 0,
        "root_rollout_considered": 0,
        "root_rollout_no_candidates": 0,
        "root_rollout_no_rollouts": 0,
        "root_rollout_overrides": 0,
        "root_rollout_scored_candidates": 0,
        "root_rollout_by_phase": {},
        "root_rollout_overrides_by_phase": {},
        "root_rollout_continuation_mode": continuation_mode,
        "root_rollout_continuation_bot": continuation_bot_name,
    })

  @staticmethod
  def _select_scored_action(scored, policy, objective):
    objective = str(objective or "score")
    if objective == "paradox_then_score":
      return max(
          scored,
          key=lambda action: (
              -float(scored[action]["paradox"]),
              float(scored[action]["score"]),
              float(policy[action]),
          ),
      )
    return max(
        scored,
        key=lambda action: (
            float(scored[action]["score"]),
            float(policy[action]),
        ),
    )

  def _continuation_action(self, state, player):
    legal = state.legal_actions(player)
    if self._continuation_bots:
      bot = self._continuation_bots[int(player) % len(self._continuation_bots)]
      try:
        action = int(bot.step(state.clone(), player))
        if action in legal:
          return action
      except Exception:
        pass
    if self._continuation_neural_bot is not None:
      try:
        action = int(self._continuation_neural_bot.step(state.clone(), player))
        if action in legal:
          return action
      except Exception:
        pass
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    return max(legal, key=lambda action: policy[action])

  def _has_explicit_continuation_policy(self):
    return bool(self._continuation_bots) or self._continuation_neural_bot is not None

  def _candidate_actions(self, state, player, legal, policy):
    legal = list(legal)
    max_actions = int(getattr(self.args, "root_rollout_max_actions", 4))
    if max_actions <= 0 or len(legal) <= max_actions:
      return sorted(legal)
    selected = []
    selected_set = set()
    top_count = max(0, int(getattr(self.args, "root_rollout_top_policy", 2)))
    if top_count > 0:
      ordered = sorted(legal, key=lambda action: float(policy[action]), reverse=True)
      for action in ordered[:min(top_count, max_actions)]:
        selected.append(action)
        selected_set.add(action)
    for bot_name in _split_csv(
        getattr(self.args, "root_rollout_include_bots", "")
    ):
      if len(selected) >= max_actions:
        break
      try:
        action = make_bot(bot_name, seed=0).step(state.clone(), player)
      except Exception:
        continue
      if action in legal and action not in selected_set:
        selected.append(action)
        selected_set.add(action)
    if (
        len(selected) < max_actions
        and bool(
            getattr(self.args, "root_rollout_include_continuation_candidate", True)
        )
        and self._has_explicit_continuation_policy()
    ):
      try:
        action = int(self._continuation_action(state.clone(), player))
      except Exception:
        action = None
      if action in legal and action not in selected_set:
        selected.append(action)
        selected_set.add(action)
    remaining = [action for action in legal if action not in selected_set]
    fill_count = max_actions - len(selected)
    if fill_count > 0 and remaining:
      if fill_count >= len(remaining):
        selected.extend(remaining)
      else:
        chosen = np.random.choice(
            remaining, size=fill_count, replace=False
        ).tolist()
        selected.extend(int(action) for action in chosen)
    return sorted(selected)

  def _enabled_for_phase(self, state):
    spec = str(getattr(self.args, "root_rollout_phases", "") or "").strip()
    if not spec:
      return True
    allowed = {part.strip() for part in spec.split(",") if part.strip()}
    phase = {
        0: "chance",
        1: "discard",
        2: "prediction",
        3: "play",
        4: "terminal",
    }.get(int(getattr(state, "_phase", -1)), "unknown")
    return phase in allowed

  def _leaf_value(self, state, player):
    while state.is_chance_node() and not state.is_terminal():
      actions, probs = zip(*state.chance_outcomes())
      state.apply_action(int(np.random.choice(actions, p=probs)))
    if state.is_terminal():
      return float(np.array(state.returns(), dtype=np.float32)[player])
    _, value = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    return float(np.array(value, dtype=np.float32)[player])

  def _paradox_indicator(self, state, player):
    players = int(getattr(self.args, "players", 3))
    paradoxes = np.array(
        getattr(state, "_has_paradoxed", [False] * players), dtype=np.float32
    )
    scope = str(getattr(self.args, "root_rollout_paradox_scope", "acting"))
    if scope == "any":
      return float(np.max(paradoxes)) if paradoxes.size else 0.0
    if 0 <= int(player) < len(paradoxes):
      return float(paradoxes[int(player)])
    return 0.0

  def _rollout_round_result(self, belief_state, player, first_action):
    rollout = belief_state.clone()
    rollout.apply_action(first_action)
    max_plies = max(0, int(getattr(self.args, "root_rollout_max_plies", 0)))
    decision_plies = 0
    while not rollout.is_terminal():
      if rollout.is_chance_node():
        actions, probs = zip(*rollout.chance_outcomes())
        rollout.apply_action(int(np.random.choice(actions, p=probs)))
        continue
      if max_plies > 0 and decision_plies >= max_plies:
        return {
            "paradox": self._paradox_indicator(rollout, player),
            "score": self._leaf_value(rollout, player),
        }
      current = int(rollout.current_player())
      rollout.apply_action(self._continuation_action(rollout, current))
      decision_plies += 1
    return {
        "paradox": self._paradox_indicator(rollout, player),
        "score": float(np.array(rollout.returns(), dtype=np.float32)[player]),
    }

  def _play_model_round(self, state):
    while not state.is_terminal():
      if state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state.apply_action(int(np.random.choice(actions, p=probs)))
        continue
      current = int(state.current_player())
      state.apply_action(self._continuation_action(state, current))
    return state

  def _rollout_full_match_result(self, belief_state, player, first_action):
    current_round = int(getattr(belief_state, "_match_round", 0))
    round_start = int(getattr(belief_state, "_round_start_player", 0))
    players = int(getattr(self.args, "players", 3))
    initial_start = (round_start - current_round) % players

    rollout = belief_state.clone()
    rollout.apply_action(first_action)
    max_plies = max(0, int(getattr(self.args, "root_rollout_max_plies", 0)))
    decision_plies = 0
    while not rollout.is_terminal():
      if rollout.is_chance_node():
        actions, probs = zip(*rollout.chance_outcomes())
        rollout.apply_action(int(np.random.choice(actions, p=probs)))
        continue
      if max_plies > 0 and decision_plies >= max_plies:
        return {
            "paradox": self._paradox_indicator(rollout, player),
            "score": self._leaf_value(rollout, player),
        }
      current = int(rollout.current_player())
      rollout.apply_action(self._continuation_action(rollout, current))
      decision_plies += 1

    current_round_paradox = self._paradox_indicator(rollout, player)
    totals = np.array(rollout.returns(), dtype=np.float32)
    for next_round in range(current_round + 1, players):
      start_player = (initial_start + next_round) % players
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": players,
              "start_player": start_player,
              "match_context": int(getattr(self.args, "match_context", False)),
          },
      )
      next_state = game.new_initial_state()
      if getattr(self.args, "match_context", False):
        next_state.set_match_context(totals, next_round)
      next_state = self._play_model_round(next_state)
      next_returns = np.array(next_state.returns(), dtype=np.float32)
      if getattr(self.args, "match_context", False):
        totals = next_returns
      else:
        totals += next_returns
    return {
        "paradox": current_round_paradox,
        "score": float(totals[player]),
    }

  def _rollout_result(self, belief_state, player, first_action):
    if getattr(self.args, "root_rollout_full_match", False):
      return self._rollout_full_match_result(belief_state, player, first_action)
    return self._rollout_round_result(belief_state, player, first_action)

  def _rollout_score(self, belief_state, player, first_action):
    return self._rollout_result(belief_state, player, first_action)["score"]

  def step(self, state, player):
    legal = state.legal_actions(player)
    legal_count = len(legal)
    if len(legal) == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return legal[0]
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    baseline_action = max(legal, key=lambda action: policy[action])
    if not self._enabled_for_phase(state):
      self._decision_stats["root_rollout_disabled"] += 1
      _record_decision(self._decision_stats, state, baseline_action, legal_count)
      return baseline_action
    phase = _phase_name(state)
    self._decision_stats["root_rollout_considered"] += 1
    _inc_counter(self._decision_stats["root_rollout_by_phase"], phase)
    candidates = self._candidate_actions(state, player, legal, policy)
    if not candidates:
      self._decision_stats["root_rollout_no_candidates"] += 1
      _record_decision(self._decision_stats, state, baseline_action, legal_count)
      return baseline_action
    samples = max(1, int(getattr(self.args, "root_rollout_samples", 2)))
    sampled_states = sampled_belief_states_for_policy(
        state,
        player,
        samples,
        self.args,
        self.model,
        self.device,
        self.args.value_scale,
        context="eval",
    )
    rollouts = max(1, int(getattr(self.args, "root_rollouts", 1)))
    action_results = {
        action: {"paradox": [], "score": []} for action in candidates
    }
    for belief_state in sampled_states:
      legal_in_belief = set(belief_state.legal_actions(player))
      legal_candidates = [
          action for action in candidates if action in legal_in_belief
      ]
      if not legal_candidates:
        continue
      for _ in range(rollouts):
        np_state = np.random.get_state()
        for action in legal_candidates:
          np.random.set_state(np_state)
          result = self._rollout_result(belief_state, player, action)
          action_results[action]["paradox"].append(float(result["paradox"]))
          action_results[action]["score"].append(float(result["score"]))
    scored = {
        action: {
            "paradox": float(np.mean(results["paradox"])),
            "score": float(np.mean(results["score"])),
        }
        for action, results in action_results.items()
        if results["score"]
    }
    if not scored:
      self._decision_stats["root_rollout_no_rollouts"] += 1
      _record_decision(self._decision_stats, state, baseline_action, legal_count)
      return baseline_action
    self._decision_stats["root_rollout_scored_candidates"] += len(scored)
    selected_action = self._select_scored_action(
        scored,
        policy,
        getattr(self.args, "root_rollout_objective", "score"),
    )
    if selected_action != baseline_action:
      self._decision_stats["root_rollout_overrides"] += 1
      _inc_counter(self._decision_stats["root_rollout_overrides_by_phase"], phase)
    _record_decision(
        self._decision_stats, state, selected_action, legal_count
    )
    return selected_action

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    stats = json.loads(json.dumps(self._decision_stats))
    if self._continuation_neural_bot is not None:
      if hasattr(self._continuation_neural_bot, "raw_decision_stats"):
        stats["nested"] = {
            "continuation_bot": self._continuation_neural_bot.raw_decision_stats()
        }
      elif hasattr(self._continuation_neural_bot, "decision_stats"):
        stats["nested"] = {
            "continuation_bot": self._continuation_neural_bot.decision_stats()
        }
    return stats


def own_hand_feasibility(state, player):
  """Exact public/own-info feasibility for assigning remaining ranks to colors."""
  hand = np.asarray(getattr(state, "_hands")[int(player)], dtype=np.int32)
  tokens = np.asarray(getattr(state, "_color_tokens")[int(player)], dtype=bool)
  board = np.asarray(getattr(state, "_board_ownership"), dtype=np.int32)
  num_colors = int(getattr(state, "_num_colors", board.shape[0]))
  num_card_types = int(getattr(state, "_num_card_types", len(hand)))
  open_slots_by_rank = []
  total_slots = 0
  matching_size = 0
  total_deficit = 0
  buffer_deficit = 0
  tight_rank_count = 0
  dead_rank_count = 0
  singleton_card_count = 0
  min_choices = None
  min_rank_surplus = None
  remaining_cards = 0
  for rank_idx in range(num_card_types):
    count = max(0, int(hand[rank_idx]))
    if count <= 0:
      continue
    remaining_cards += count
    open_slots = 0
    for color_idx in range(num_colors):
      if bool(tokens[color_idx]) and int(board[color_idx, rank_idx]) == -1:
        open_slots += 1
    open_slots_by_rank.append(int(open_slots))
    total_slots += open_slots
    matching_size += min(count, open_slots)
    deficit = max(0, count - open_slots)
    total_deficit += deficit
    buffer_deficit += max(0, count + 1 - open_slots)
    if open_slots <= count:
      tight_rank_count += 1
    if open_slots <= 0:
      dead_rank_count += 1
    if open_slots == 1:
      singleton_card_count += count
    min_choices = open_slots if min_choices is None else min(min_choices, open_slots)
    rank_surplus = open_slots - count
    min_rank_surplus = (
        rank_surplus if min_rank_surplus is None
        else min(min_rank_surplus, rank_surplus)
    )
  slot_surplus = total_slots - remaining_cards
  feasible = total_deficit == 0
  return {
      "feasible": bool(feasible),
      "matching_size": int(matching_size),
      "remaining_cards": int(remaining_cards),
      "total_open_slots": int(total_slots),
      "slot_surplus": int(slot_surplus),
      "total_deficit": int(total_deficit),
      "buffer_deficit": int(buffer_deficit),
      "tight_rank_count": int(tight_rank_count),
      "dead_rank_count": int(dead_rank_count),
      "singleton_card_count": int(singleton_card_count),
      "min_choices_per_remaining_card": int(min_choices or 0),
      "min_rank_surplus": int(min_rank_surplus or 0),
      "open_slots_by_rank": open_slots_by_rank,
  }


def own_hand_feasibility_after_action(state, player, action):
  action = int(action)
  if action == 999:
    row = own_hand_feasibility(state, player)
    row.update({
        "feasible": False,
        "is_paradox": True,
        "total_deficit": max(1, int(row.get("remaining_cards", 0))),
        "slot_surplus": -max(1, int(row.get("remaining_cards", 0))),
    })
    return row
  clone = state.clone()
  try:
    clone.apply_action(action)
  except Exception:
    return {
        "feasible": False,
        "is_paradox": False,
        "matching_size": 0,
        "remaining_cards": 0,
        "total_open_slots": 0,
        "slot_surplus": -1,
        "total_deficit": 1,
        "buffer_deficit": 1,
        "tight_rank_count": 1,
        "dead_rank_count": 1,
        "singleton_card_count": 0,
        "min_choices_per_remaining_card": 0,
        "min_rank_surplus": -1,
        "open_slots_by_rank": [],
    }
  row = own_hand_feasibility(clone, player)
  paradoxes = getattr(clone, "_has_paradoxed", [])
  row["is_paradox"] = bool(
      int(player) < len(paradoxes) and bool(paradoxes[int(player)])
  )
  if row["is_paradox"]:
    row["feasible"] = False
    row["total_deficit"] = max(1, int(row.get("total_deficit", 0)))
  return row


def _action_would_win_after_play(state, player, action):
  if int(action) == 999 or int(getattr(state, "_phase", -1)) != 3:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  if num_card_types <= 0:
    return None
  color_idx = int(action) // num_card_types
  rank_idx = int(action) % num_card_types
  color_names = ["R", "B", "Y", "G"]
  if color_idx < 0 or color_idx >= len(color_names):
    return None
  plays = list(getattr(state, "_cards_played_this_trick", []))
  if not plays or int(player) >= len(plays):
    return None
  plays[int(player)] = (rank_idx + 1, color_names[color_idx])
  if any(card is None for card in plays):
    return None
  red = [(idx, card[0]) for idx, card in enumerate(plays) if card[1] == "R"]
  if red:
    return max(red, key=lambda item: item[1])[0] == int(player)
  led_color = getattr(state, "_led_color", None) or color_names[color_idx]
  led = [(idx, card[0]) for idx, card in enumerate(plays) if card[1] == led_color]
  if not led:
    return False
  return max(led, key=lambda item: item[1])[0] == int(player)


def public_exit_liquidity(state, player):
  """Public exit-space summary from board occupancy and color tokens.

  This intentionally ignores hidden opponent hands. It asks how many future
  color-rank cells remain reachable to each player under public X/claimed cells
  and public color-token constraints, then combines that with the acting
  player's exact own-hand feasibility.
  """
  del player
  board = np.asarray(getattr(state, "_board_ownership"), dtype=np.int32)
  tokens = np.asarray(getattr(state, "_color_tokens"), dtype=bool)
  num_players = int(getattr(state, "_num_players", tokens.shape[0]))
  num_colors = int(getattr(state, "_num_colors", board.shape[0]))
  num_card_types = int(getattr(state, "_num_card_types", board.shape[1]))
  open_by_color = []
  for color_idx in range(num_colors):
    open_by_color.append(int(np.sum(board[color_idx] == -1)))
  player_open_slots = []
  player_color_open_slots = []
  player_min_rank_slots = []
  player_remaining_cards = []
  player_lane_surplus = []
  for seat in range(num_players):
    color_slots = []
    for color_idx in range(num_colors):
      color_slots.append(
          int(open_by_color[color_idx]) if bool(tokens[seat, color_idx]) else 0
      )
    player_color_open_slots.append(color_slots)
    player_open_slots.append(int(sum(color_slots)))
    rank_slots = []
    for rank_idx in range(num_card_types):
      slots = 0
      for color_idx in range(num_colors):
        if bool(tokens[seat, color_idx]) and int(board[color_idx, rank_idx]) == -1:
          slots += 1
      rank_slots.append(slots)
    player_min_rank_slots.append(int(min(rank_slots) if rank_slots else 0))
    try:
      played_count = int(state._count_cards_played_by(seat))  # pylint: disable=protected-access
    except Exception:
      hands = getattr(state, "_hands", [])
      if seat < len(hands):
        played_count = int(num_card_types - np.sum(np.asarray(hands[seat])))
      else:
        played_count = int(getattr(state, "_trick_number", 0))
    remaining = max(0, int(getattr(state, "_num_tricks", num_card_types)) - played_count)
    player_remaining_cards.append(int(remaining))
    player_lane_surplus.append(int(player_open_slots[-1] - remaining))
  return {
      "open_cells": int(np.sum(board == -1)),
      "blocked_or_claimed_cells": int(np.sum(board != -1)),
      "open_by_color": open_by_color,
      "player_open_slots": player_open_slots,
      "player_color_open_slots": player_color_open_slots,
      "player_min_rank_slots": player_min_rank_slots,
      "player_remaining_cards": player_remaining_cards,
      "player_lane_surplus": player_lane_surplus,
      "total_player_open_slots": int(sum(player_open_slots)),
      "min_player_open_slots": int(min(player_open_slots) if player_open_slots else 0),
      "total_player_remaining_cards": int(sum(player_remaining_cards)),
      "total_player_lane_surplus": int(sum(player_lane_surplus)),
      "min_player_lane_surplus": int(
          min(player_lane_surplus) if player_lane_surplus else 0
      ),
      "lane_pressure_player_count": int(
          sum(1 for surplus in player_lane_surplus if surplus < 0)
      ),
  }


def public_exit_liquidity_after_action(state, player, action):
  action = int(action)
  player = int(player)
  before = public_exit_liquidity(state, player)
  base_own = own_hand_feasibility(state, player)
  if action == 999:
    row = {
        "is_paradox": True,
        "own_feasible": False,
        "own_total_deficit": max(1, int(base_own.get("remaining_cards", 0))),
        "own_buffer_deficit": max(1, int(base_own.get("remaining_cards", 0))),
        "own_dead_rank_count": max(1, int(base_own.get("dead_rank_count", 0))),
        "own_tight_rank_count": int(base_own.get("tight_rank_count", 0)),
        "own_singleton_card_count": int(base_own.get("singleton_card_count", 0)),
        "own_slot_surplus": -max(1, int(base_own.get("remaining_cards", 0))),
        "own_min_choices": 0,
        "public_slot_damage": before["total_player_open_slots"],
        "own_public_slot_damage": (
            before["player_open_slots"][player]
            if player < len(before["player_open_slots"]) else 0
        ),
        "board_open_cell_damage": before["open_cells"],
        "own_public_slots_after": 0,
        "total_player_open_slots_after": 0,
        "min_player_open_slots_after": 0,
        "own_lane_surplus_after": 0,
        "total_player_lane_surplus_after": -before["total_player_remaining_cards"],
        "min_player_lane_surplus_after": -max(
            before["player_remaining_cards"] or [0]
        ),
        "lane_surplus_damage": before["total_player_lane_surplus"],
        "min_lane_surplus_damage": before["min_player_lane_surplus"],
        "lane_pressure_player_count_after": before["lane_pressure_player_count"],
        "own_min_rank_slots_after": 0,
        "min_player_rank_slots_after": 0,
        "lost_led_token": False,
        "would_win_now": None,
        "over_target_would_win": False,
    }
    return row

  clone = state.clone()
  led_color = getattr(state, "_led_color", None)
  color_names = ["R", "B", "Y", "G"]
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  action_color_idx = action // num_card_types if num_card_types else -1
  action_color = (
      color_names[action_color_idx]
      if 0 <= action_color_idx < len(color_names) else None
  )
  would_win_now = _action_would_win_after_play(state, player, action)
  try:
    clone.apply_action(action)
  except Exception:
    row = {
        "is_paradox": False,
        "own_feasible": False,
        "own_total_deficit": 1,
        "own_buffer_deficit": 1,
        "own_dead_rank_count": 1,
        "own_tight_rank_count": 1,
        "own_singleton_card_count": 0,
        "own_slot_surplus": -1,
        "own_min_choices": 0,
        "public_slot_damage": before["total_player_open_slots"],
        "own_public_slot_damage": (
            before["player_open_slots"][player]
            if player < len(before["player_open_slots"]) else 0
        ),
        "board_open_cell_damage": before["open_cells"],
        "own_public_slots_after": 0,
        "total_player_open_slots_after": 0,
        "min_player_open_slots_after": 0,
        "own_lane_surplus_after": 0,
        "total_player_lane_surplus_after": -before["total_player_remaining_cards"],
        "min_player_lane_surplus_after": -max(
            before["player_remaining_cards"] or [0]
        ),
        "lane_surplus_damage": before["total_player_lane_surplus"],
        "min_lane_surplus_damage": before["min_player_lane_surplus"],
        "lane_pressure_player_count_after": before["lane_pressure_player_count"],
        "own_min_rank_slots_after": 0,
        "min_player_rank_slots_after": 0,
        "lost_led_token": False,
        "would_win_now": would_win_now,
        "over_target_would_win": False,
    }
    return row

  after = public_exit_liquidity(clone, player)
  own = own_hand_feasibility(clone, player)
  paradoxes = getattr(clone, "_has_paradoxed", [])
  is_paradox = bool(player < len(paradoxes) and bool(paradoxes[player]))
  before_player_slots = (
      before["player_open_slots"][player]
      if player < len(before["player_open_slots"]) else 0
  )
  after_player_slots = (
      after["player_open_slots"][player]
      if player < len(after["player_open_slots"]) else 0
  )
  after_player_min_rank_slots = (
      after["player_min_rank_slots"][player]
      if player < len(after["player_min_rank_slots"]) else 0
  )
  after_player_lane_surplus = (
      after["player_lane_surplus"][player]
      if player < len(after["player_lane_surplus"]) else 0
  )
  predictions = getattr(state, "_predictions", [])
  tricks_won = getattr(state, "_tricks_won", [])
  prediction = int(predictions[player]) if player < len(predictions) else -1
  tricks = int(tricks_won[player]) if player < len(tricks_won) else 0
  lost_led_token = False
  if led_color is not None and action_color is not None and action_color != led_color:
    if led_color in color_names:
      led_idx = color_names.index(led_color)
      before_tokens = np.asarray(getattr(state, "_color_tokens"), dtype=bool)
      after_tokens = np.asarray(getattr(clone, "_color_tokens"), dtype=bool)
      if (
          player < before_tokens.shape[0]
          and bool(before_tokens[player, led_idx])
          and not bool(after_tokens[player, led_idx])
      ):
        lost_led_token = True
  row = {
      "is_paradox": is_paradox,
      "own_feasible": bool(own.get("feasible", False)) and not is_paradox,
      "own_total_deficit": int(max(1, own.get("total_deficit", 0)) if is_paradox
                               else own.get("total_deficit", 0)),
      "own_buffer_deficit": int(own.get("buffer_deficit", 0)),
      "own_dead_rank_count": int(own.get("dead_rank_count", 0)),
      "own_tight_rank_count": int(own.get("tight_rank_count", 0)),
      "own_singleton_card_count": int(own.get("singleton_card_count", 0)),
      "own_slot_surplus": int(own.get("slot_surplus", 0)),
      "own_min_choices": int(own.get("min_choices_per_remaining_card", 0)),
      "public_slot_damage": int(
          before["total_player_open_slots"] - after["total_player_open_slots"]
      ),
      "own_public_slot_damage": int(before_player_slots - after_player_slots),
      "board_open_cell_damage": int(before["open_cells"] - after["open_cells"]),
      "own_public_slots_after": int(after_player_slots),
      "total_player_open_slots_after": int(after["total_player_open_slots"]),
      "min_player_open_slots_after": int(after["min_player_open_slots"]),
      "own_lane_surplus_after": int(after_player_lane_surplus),
      "total_player_lane_surplus_after": int(after["total_player_lane_surplus"]),
      "min_player_lane_surplus_after": int(after["min_player_lane_surplus"]),
      "lane_surplus_damage": int(
          before["total_player_lane_surplus"] - after["total_player_lane_surplus"]
      ),
      "min_lane_surplus_damage": int(
          before["min_player_lane_surplus"] - after["min_player_lane_surplus"]
      ),
      "lane_pressure_player_count_after": int(after["lane_pressure_player_count"]),
      "own_min_rank_slots_after": int(after_player_min_rank_slots),
      "min_player_rank_slots_after": int(
          min(after["player_min_rank_slots"]) if after["player_min_rank_slots"] else 0
      ),
      "lost_led_token": bool(lost_led_token),
      "would_win_now": would_win_now,
      "over_target_would_win": bool(
          would_win_now is True and prediction >= 0 and tricks >= prediction
      ),
  }
  return row


def action_liveness_key(row, action, base_action=None):
  """Lexicographic deterministic safety key for a candidate action.

  Higher is better. This key is deliberately local and public/own-info only:
  exact own remaining-hand feasibility comes first, then public exit-space
  preservation, then prediction/token-loss tie breakers. It is meant for
  within-state comparisons, not as a calibrated global risk probability.
  """
  base_bonus = (
      1 if base_action is not None and int(action) == int(base_action) else 0
  )
  return (
      0 if bool(row.get("is_paradox", False)) else 1,
      1 if bool(row.get("own_feasible", False)) else 0,
      -int(row.get("own_total_deficit", 0)),
      -int(row.get("own_dead_rank_count", 0)),
      0 if bool(row.get("lost_led_token", False)) else 1,
      int(row.get("min_player_lane_surplus_after", 0)),
      int(row.get("total_player_lane_surplus_after", 0)),
      int(row.get("own_lane_surplus_after", 0)),
      -int(row.get("lane_pressure_player_count_after", 0)),
      int(row.get("min_player_open_slots_after", 0)),
      int(row.get("total_player_open_slots_after", 0)),
      int(row.get("own_public_slots_after", 0)),
      int(row.get("min_player_rank_slots_after", 0)),
      int(row.get("own_min_rank_slots_after", 0)),
      0 if bool(row.get("over_target_would_win", False)) else 1,
      -int(row.get("public_slot_damage", 0)),
      -int(row.get("own_public_slot_damage", 0)),
      -int(row.get("board_open_cell_damage", 0)),
      -int(row.get("own_buffer_deficit", 0)),
      int(row.get("own_min_choices", 0)),
      int(row.get("own_slot_surplus", 0)),
      base_bonus,
      -int(action),
  )


def action_liveness_certificate(state, player, action, base_action=None):
  """Reusable liveness row for mining, audits, and guarded deployment."""
  row = dict(public_exit_liquidity_after_action(state, player, action))
  row.update({
      "action": int(action),
      "phase": _phase_name(state).lower(),
      "trick_number": int(getattr(state, "_trick_number", -1)),
  })
  key = action_liveness_key(row, action, base_action=base_action)
  row["liveness_key"] = [int(value) for value in key]
  row["liveness_failure_count"] = int(bool(row.get("is_paradox", False))) + int(
      not bool(row.get("own_feasible", False))
  ) + int(row.get("min_player_open_slots_after", 0) <= 0)
  return row


def exact_legal_pressure_after_action(state, action):
  """Full-state post-action legal-exit pressure for teacher/audit experiments."""
  clone = state.clone()
  clone.apply_action(int(action))
  num_players = int(getattr(clone, "_num_players", 0) or 0)
  if clone.is_terminal():
    return {
        "exact_terminal_after": True,
        "exact_forced_paradox_players_after": 0,
        "exact_min_nonparadox_legal_after": 99,
        "exact_current_nonparadox_legal_after": 99,
        "exact_total_nonparadox_legal_after": 99 * max(1, num_players),
    }

  counts = []
  forced_count = 0
  for seat in range(num_players):
    if bool(getattr(clone, "_has_paradoxed", [False] * num_players)[seat]):
      continue
    remaining_cards = int(np.sum(getattr(clone, "_hands", [])[seat]))
    if remaining_cards <= 0:
      continue
    legal = [int(value) for value in clone.legal_actions(seat)]
    nonparadox_count = sum(1 for value in legal if value != 999)
    counts.append(nonparadox_count)
    if nonparadox_count <= 0:
      forced_count += 1

  current = int(clone.current_player())
  if 0 <= current < num_players:
    current_legal = [int(value) for value in clone.legal_actions(current)]
    current_count = sum(1 for value in current_legal if value != 999)
  else:
    current_count = 99

  return {
      "exact_terminal_after": False,
      "exact_forced_paradox_players_after": int(forced_count),
      "exact_min_nonparadox_legal_after": int(min(counts)) if counts else 99,
      "exact_current_nonparadox_legal_after": int(current_count),
      "exact_total_nonparadox_legal_after": int(sum(counts)),
  }


class ExitLiquidityShieldBot:
  """Base policy with public exit-liquidity diagnostics and optional overrides."""

  def __init__(self, name, args, seed=0):
    self.name = name
    self.args = args
    self.seed = int(seed)
    base_name = self._base_bot_name(name, args)
    self.base_bot = make_bot(base_name, seed=seed)
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "exit_liquidity_disabled": 0,
        "exit_liquidity_considered": 0,
        "exit_liquidity_base_kept": 0,
        "exit_liquidity_overrides": 0,
        "exit_liquidity_shadow_overrides": 0,
        "exit_liquidity_base_public_slot_damage_sum": 0.0,
        "exit_liquidity_selected_public_slot_damage_sum": 0.0,
        "exit_liquidity_base_own_deficit_sum": 0.0,
        "exit_liquidity_selected_own_deficit_sum": 0.0,
        "exit_liquidity_by_phase": {},
        "exit_liquidity_overrides_by_phase": {},
        "exit_liquidity_base_bot": base_name,
        "exit_liquidity_override_samples": [],
    })

  @staticmethod
  def _base_bot_name(name, args):
    if name == "exit_liquidity_shield_safe14":
      return "heuristic_safe14"
    if name == "exit_liquidity_shield_safe8":
      return "heuristic_safe8"
    return str(getattr(args, "exit_liquidity_base_bot", "heuristic_safe14"))

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def _enabled_for_state(self, state):
    min_trick = int(getattr(self.args, "exit_liquidity_min_trick_number", 0) or 0)
    if int(getattr(state, "_trick_number", 0)) < min_trick:
      return False
    spec = str(getattr(self.args, "exit_liquidity_phases", "play") or "").strip()
    if not spec:
      return True
    allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
    return self._phase_name(state).lower() in allowed

  def _base_action(self, state, player, legal):
    try:
      action = int(self.base_bot.step(state.clone(), player))
    except Exception:
      action = int(legal[0])
    if action in legal:
      return action
    return int(legal[0])

  @staticmethod
  def _liquidity_key(action, row, base_action):
    return (
        0 if bool(row.get("is_paradox", False)) else 1,
        1 if bool(row.get("own_feasible", False)) else 0,
        -int(row.get("own_total_deficit", 0)),
        -int(row.get("own_dead_rank_count", 0)),
        0 if bool(row.get("over_target_would_win", False)) else 1,
        -int(row.get("public_slot_damage", 0)),
        -int(row.get("own_public_slot_damage", 0)),
        0 if bool(row.get("lost_led_token", False)) else 1,
        -int(row.get("own_buffer_deficit", 0)),
        int(row.get("own_min_choices", 0)),
        int(row.get("own_slot_surplus", 0)),
        1 if int(action) == int(base_action) else 0,
        -int(action),
    )

  def _score_actions(self, state, player, legal):
    legal = sorted(int(action) for action in legal)
    non_paradox = [action for action in legal if action != 999]
    if non_paradox:
      legal = non_paradox
    return {
        int(action): public_exit_liquidity_after_action(state, player, action)
        for action in legal
    }

  def _should_override(self, base_row, selected_row):
    if bool(selected_row.get("is_paradox", False)):
      return False
    if bool(base_row.get("is_paradox", False)):
      return True
    base_deficit = int(base_row.get("own_total_deficit", 0))
    selected_deficit = int(selected_row.get("own_total_deficit", 0))
    if selected_deficit < base_deficit:
      return True
    base_damage = float(base_row.get("public_slot_damage", 0.0))
    selected_damage = float(selected_row.get("public_slot_damage", 0.0))
    base_own_damage = float(base_row.get("own_public_slot_damage", 0.0))
    selected_own_damage = float(selected_row.get("own_public_slot_damage", 0.0))
    if (
        bool(base_row.get("over_target_would_win", False))
        and not bool(selected_row.get("over_target_would_win", False))
        and selected_damage <= base_damage
        and selected_own_damage <= base_own_damage
    ):
      return True
    damage_delta = base_damage - selected_damage
    min_delta = float(getattr(self.args, "exit_liquidity_min_damage_delta", 2.0))
    if damage_delta >= min_delta:
      return True
    if (
        bool(base_row.get("lost_led_token", False))
        and not bool(selected_row.get("lost_led_token", False))
        and damage_delta > 0.0
    ):
      return True
    return False

  def _append_override_sample(
      self, state, player, legal, base_action, selected_action, base_row,
      selected_row, shadow
  ):
    samples = self._decision_stats["exit_liquidity_override_samples"]
    limit = int(getattr(self.args, "exit_liquidity_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": self._phase_name(state).lower(),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "led_color": getattr(state, "_led_color", None),
        "legal_count": int(len(legal)),
        "base_action": int(base_action),
        "selected_action": int(selected_action),
        "shadow": bool(shadow),
        "base_exit_liquidity": dict(base_row),
        "selected_exit_liquidity": dict(selected_row),
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    base_action = self._base_action(state, player, legal)
    if not self._enabled_for_state(state):
      self._decision_stats["exit_liquidity_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = self._phase_name(state).lower()
    self._decision_stats["exit_liquidity_considered"] += 1
    _inc_counter(self._decision_stats["exit_liquidity_by_phase"], phase)
    scored = self._score_actions(state, player, legal)
    if not scored:
      self._decision_stats["exit_liquidity_base_kept"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    base_row = scored.get(int(base_action))
    if base_row is None:
      base_row = public_exit_liquidity_after_action(state, player, base_action)
    selected_action, selected_row = max(
        scored.items(),
        key=lambda item: self._liquidity_key(item[0], item[1], base_action),
    )
    selected_action = int(selected_action)
    should_override = (
        selected_action != int(base_action)
        and self._should_override(base_row, selected_row)
    )
    shadow_only = bool(getattr(self.args, "exit_liquidity_shadow_only", False))
    self._decision_stats["exit_liquidity_base_public_slot_damage_sum"] += float(
        base_row.get("public_slot_damage", 0.0)
    )
    self._decision_stats["exit_liquidity_base_own_deficit_sum"] += float(
        base_row.get("own_total_deficit", 0.0)
    )
    final_action = int(base_action)
    final_row = base_row
    if should_override:
      if shadow_only:
        self._decision_stats["exit_liquidity_shadow_overrides"] += 1
        self._append_override_sample(
            state, player, legal, base_action, selected_action, base_row,
            selected_row, shadow=True
        )
      else:
        final_action = selected_action
        final_row = selected_row
        self._decision_stats["exit_liquidity_overrides"] += 1
        _inc_counter(
            self._decision_stats["exit_liquidity_overrides_by_phase"], phase
        )
        self._append_override_sample(
            state, player, legal, base_action, selected_action, base_row,
            selected_row, shadow=False
        )
    else:
      self._decision_stats["exit_liquidity_base_kept"] += 1
    if should_override and shadow_only:
      final_row = base_row
    self._decision_stats["exit_liquidity_selected_public_slot_damage_sum"] += float(
        final_row.get("public_slot_damage", 0.0)
    )
    self._decision_stats["exit_liquidity_selected_own_deficit_sum"] += float(
        final_row.get("own_total_deficit", 0.0)
    )
    _record_decision(self._decision_stats, state, final_action, legal_count)
    return int(final_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class LiquidityFeasibilityShieldBot:
  """Base policy guarded by exact own feasibility and public exit liquidity."""

  def __init__(self, name, args, seed=0):
    self.name = name
    self.args = args
    self.seed = int(seed)
    base_name = self._base_bot_name(name, args)
    self.base_bot = make_bot(base_name, seed=seed)
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "lf_shield_disabled": 0,
        "lf_shield_considered": 0,
        "lf_shield_base_kept": 0,
        "lf_shield_overrides": 0,
        "lf_shield_shadow_overrides": 0,
        "lf_shield_base_feasible": 0,
        "lf_shield_base_lost_led_token": 0,
        "lf_shield_base_public_slot_damage_sum": 0.0,
        "lf_shield_selected_public_slot_damage_sum": 0.0,
        "lf_shield_base_own_deficit_sum": 0.0,
        "lf_shield_selected_own_deficit_sum": 0.0,
        "lf_shield_base_min_lane_surplus_sum": 0.0,
        "lf_shield_selected_min_lane_surplus_sum": 0.0,
        "lf_shield_by_phase": {},
        "lf_shield_overrides_by_phase": {},
        "lf_shield_override_reasons": {},
        "lf_shield_base_bot": base_name,
        "lf_shield_override_samples": [],
    })

  @staticmethod
  def _base_bot_name(name, args):
    if name == "liquidity_feasibility_shield_safe14":
      return "heuristic_safe14"
    if name == "liquidity_feasibility_shield_safe8":
      return "heuristic_safe8"
    return str(getattr(args, "lf_shield_base_bot", "heuristic_safe14"))

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def _enabled_for_state(self, state):
    min_trick = int(getattr(self.args, "lf_shield_min_trick_number", 0) or 0)
    if int(getattr(state, "_trick_number", 0)) < min_trick:
      return False
    spec = str(getattr(self.args, "lf_shield_phases", "play") or "").strip()
    if not spec:
      return True
    allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
    return self._phase_name(state).lower() in allowed

  def _base_action(self, state, player, legal):
    try:
      action = int(self.base_bot.step(state.clone(), player))
    except Exception:
      action = int(legal[0])
    if action in legal:
      return action
    return int(legal[0])

  def _score_actions(self, state, player, legal, base_action):
    scored = {}
    legal = sorted(int(action) for action in legal)
    non_paradox = [action for action in legal if action != 999]
    if non_paradox:
      legal = non_paradox
    for action in legal:
      scored[int(action)] = action_liveness_certificate(
          state, player, action, base_action=base_action
      )
    return scored

  def _should_override(self, base_row, selected_row):
    if bool(selected_row.get("is_paradox", False)):
      return False, ""
    if bool(base_row.get("is_paradox", False)):
      return True, "base_paradox"

    base_deficit = int(base_row.get("own_total_deficit", 0))
    selected_deficit = int(selected_row.get("own_total_deficit", 0))
    base_feasible = bool(base_row.get("own_feasible", False))
    selected_feasible = bool(selected_row.get("own_feasible", False))
    if selected_feasible and not base_feasible:
      return True, "own_feasibility"
    if selected_deficit < base_deficit:
      return True, "own_deficit"

    min_surplus = int(getattr(self.args, "lf_shield_min_slot_surplus", 0) or 0)
    max_buffer = int(
        getattr(self.args, "lf_shield_max_buffer_deficit_to_keep", -1)
    )
    base_slot_surplus = int(base_row.get("own_slot_surplus", 0))
    selected_slot_surplus = int(selected_row.get("own_slot_surplus", 0))
    base_buffer = int(base_row.get("own_buffer_deficit", 0))
    selected_buffer = int(selected_row.get("own_buffer_deficit", 0))
    base_buffer_ok = max_buffer < 0 or base_buffer <= max_buffer
    if (
        base_feasible
        and base_slot_surplus < min_surplus
        and selected_feasible
        and selected_slot_surplus > base_slot_surplus
    ):
      return True, "own_slot_surplus"
    if (
        max_buffer >= 0
        and not base_buffer_ok
        and selected_buffer < base_buffer
        and selected_deficit <= base_deficit
    ):
      return True, "own_buffer"

    base_damage = float(base_row.get("public_slot_damage", 0.0))
    selected_damage = float(selected_row.get("public_slot_damage", 0.0))
    damage_delta = base_damage - selected_damage
    base_min_lane = int(base_row.get("min_player_lane_surplus_after", 0))
    selected_min_lane = int(selected_row.get("min_player_lane_surplus_after", 0))
    lane_delta = selected_min_lane - base_min_lane
    lane_min_delta = int(
        getattr(self.args, "lf_shield_min_lane_surplus_delta", 1) or 0
    )
    if (
        lane_delta >= lane_min_delta
        and selected_deficit <= base_deficit
        and selected_damage <= base_damage
    ):
      return True, "lane_surplus"

    led_damage_allowance = float(
        getattr(self.args, "lf_shield_led_token_max_damage_increase", 0.0)
    )
    if (
        bool(base_row.get("lost_led_token", False))
        and not bool(selected_row.get("lost_led_token", False))
        and selected_deficit <= base_deficit
        and selected_damage <= base_damage + led_damage_allowance
    ):
      return True, "led_token"

    if (
        bool(base_row.get("over_target_would_win", False))
        and not bool(selected_row.get("over_target_would_win", False))
        and selected_deficit <= base_deficit
        and selected_damage <= base_damage
    ):
      return True, "over_target_win"

    min_damage_delta = float(
        getattr(self.args, "lf_shield_min_damage_delta", 0.5)
    )
    if damage_delta >= min_damage_delta and selected_deficit <= base_deficit:
      return True, "public_damage"

    return False, ""

  def _append_override_sample(
      self, state, player, legal, base_action, selected_action, base_row,
      selected_row, reason, shadow
  ):
    samples = self._decision_stats["lf_shield_override_samples"]
    limit = int(getattr(self.args, "lf_shield_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": self._phase_name(state).lower(),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "led_color": getattr(state, "_led_color", None),
        "legal_count": int(len(legal)),
        "base_action": int(base_action),
        "selected_action": int(selected_action),
        "reason": str(reason),
        "shadow": bool(shadow),
        "base_liveness": dict(base_row),
        "selected_liveness": dict(selected_row),
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    base_action = self._base_action(state, player, legal)
    if not self._enabled_for_state(state):
      self._decision_stats["lf_shield_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = self._phase_name(state).lower()
    self._decision_stats["lf_shield_considered"] += 1
    _inc_counter(self._decision_stats["lf_shield_by_phase"], phase)
    scored = self._score_actions(state, player, legal, base_action)
    if not scored:
      self._decision_stats["lf_shield_base_kept"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    base_row = scored.get(int(base_action))
    if base_row is None:
      base_row = action_liveness_certificate(
          state, player, base_action, base_action=base_action
      )
    selected_action, selected_row = max(
        scored.items(),
        key=lambda item: tuple(item[1].get("liveness_key", [])),
    )
    selected_action = int(selected_action)
    should_override, reason = (
        (False, "") if selected_action == int(base_action)
        else self._should_override(base_row, selected_row)
    )
    shadow_only = bool(getattr(self.args, "lf_shield_shadow_only", False))

    if bool(base_row.get("own_feasible", False)):
      self._decision_stats["lf_shield_base_feasible"] += 1
    if bool(base_row.get("lost_led_token", False)):
      self._decision_stats["lf_shield_base_lost_led_token"] += 1
    self._decision_stats["lf_shield_base_public_slot_damage_sum"] += float(
        base_row.get("public_slot_damage", 0.0)
    )
    self._decision_stats["lf_shield_base_own_deficit_sum"] += float(
        base_row.get("own_total_deficit", 0.0)
    )
    self._decision_stats["lf_shield_base_min_lane_surplus_sum"] += float(
        base_row.get("min_player_lane_surplus_after", 0.0)
    )

    final_action = int(base_action)
    final_row = base_row
    if should_override:
      _inc_counter(self._decision_stats["lf_shield_override_reasons"], reason)
      if shadow_only:
        self._decision_stats["lf_shield_shadow_overrides"] += 1
        self._append_override_sample(
            state, player, legal, base_action, selected_action, base_row,
            selected_row, reason, shadow=True
        )
      else:
        final_action = selected_action
        final_row = selected_row
        self._decision_stats["lf_shield_overrides"] += 1
        _inc_counter(self._decision_stats["lf_shield_overrides_by_phase"], phase)
        self._append_override_sample(
            state, player, legal, base_action, selected_action, base_row,
            selected_row, reason, shadow=False
        )
    else:
      self._decision_stats["lf_shield_base_kept"] += 1

    self._decision_stats["lf_shield_selected_public_slot_damage_sum"] += float(
        final_row.get("public_slot_damage", 0.0)
    )
    self._decision_stats["lf_shield_selected_own_deficit_sum"] += float(
        final_row.get("own_total_deficit", 0.0)
    )
    self._decision_stats["lf_shield_selected_min_lane_surplus_sum"] += float(
        final_row.get("min_player_lane_surplus_after", 0.0)
    )
    _record_decision(self._decision_stats, state, final_action, legal_count)
    return int(final_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class LivenessKeyTeacherBot:
  """Deterministic survival teacher that directly maximizes liveness key."""

  def __init__(self, name, args, seed=0):
    self.name = name
    self.args = args
    self.seed = int(seed)
    base_name = self._base_bot_name(name, args)
    self.base_bot = make_bot(base_name, seed=seed)
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "liveness_teacher_disabled": 0,
        "liveness_teacher_considered": 0,
        "liveness_teacher_base_kept": 0,
        "liveness_teacher_overrides": 0,
        "liveness_teacher_base_feasible": 0,
        "liveness_teacher_selected_feasible": 0,
        "liveness_teacher_base_lost_led_token": 0,
        "liveness_teacher_selected_lost_led_token": 0,
        "liveness_teacher_base_public_slot_damage_sum": 0.0,
        "liveness_teacher_selected_public_slot_damage_sum": 0.0,
        "liveness_teacher_base_own_deficit_sum": 0.0,
        "liveness_teacher_selected_own_deficit_sum": 0.0,
        "liveness_teacher_base_min_lane_surplus_sum": 0.0,
        "liveness_teacher_selected_min_lane_surplus_sum": 0.0,
        "liveness_teacher_base_forced_players_sum": 0.0,
        "liveness_teacher_selected_forced_players_sum": 0.0,
        "liveness_teacher_base_min_nonparadox_legal_sum": 0.0,
        "liveness_teacher_selected_min_nonparadox_legal_sum": 0.0,
        "liveness_teacher_by_phase": {},
        "liveness_teacher_overrides_by_phase": {},
        "liveness_teacher_base_bot": base_name,
        "liveness_teacher_exact_legal_pressure": bool(
            getattr(args, "liveness_teacher_exact_legal_pressure", False)
        ),
        "liveness_teacher_override_samples": [],
    })

  @staticmethod
  def _base_bot_name(name, args):
    if name == "liveness_key_teacher_safe14":
      return "heuristic_safe14"
    if name == "liveness_key_teacher_safe8":
      return "heuristic_safe8"
    return str(getattr(args, "liveness_teacher_base_bot", "heuristic_safe14"))

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def _enabled_for_state(self, state):
    min_trick = int(getattr(self.args, "liveness_teacher_min_trick_number", 0) or 0)
    if int(getattr(state, "_trick_number", 0)) < min_trick:
      return False
    spec = str(getattr(self.args, "liveness_teacher_phases", "play") or "").strip()
    if not spec:
      return True
    allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
    return self._phase_name(state).lower() in allowed

  def _base_action(self, state, player, legal):
    try:
      action = int(self.base_bot.step(state.clone(), player))
    except Exception:
      action = int(legal[0])
    if action in legal:
      return action
    return int(legal[0])

  def _score_actions(self, state, player, legal, base_action):
    legal = sorted(int(action) for action in legal)
    non_paradox = [action for action in legal if action != 999]
    if non_paradox:
      legal = non_paradox
    scored = {}
    use_exact = bool(
        getattr(self.args, "liveness_teacher_exact_legal_pressure", False)
    )
    for action in legal:
      row = action_liveness_certificate(
          state, player, action, base_action=base_action
      )
      if use_exact:
        row.update(exact_legal_pressure_after_action(state, action))
      scored[int(action)] = row
    return scored

  def _teacher_key(self, row):
    liveness_key = tuple(int(value) for value in row.get("liveness_key", []))
    if not bool(getattr(self.args, "liveness_teacher_exact_legal_pressure", False)):
      return liveness_key
    return (
        1 if bool(row.get("exact_terminal_after", False)) else 0,
        -int(row.get("exact_forced_paradox_players_after", 0)),
        0 if bool(row.get("lost_led_token", False)) else 1,
        int(row.get("exact_min_nonparadox_legal_after", 0)),
        int(row.get("exact_current_nonparadox_legal_after", 0)),
        int(row.get("exact_total_nonparadox_legal_after", 0)),
    ) + liveness_key

  def _append_override_sample(
      self, state, player, legal, base_action, selected_action, base_row,
      selected_row
  ):
    samples = self._decision_stats["liveness_teacher_override_samples"]
    limit = int(getattr(self.args, "liveness_teacher_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": self._phase_name(state).lower(),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "led_color": getattr(state, "_led_color", None),
        "legal_count": int(len(legal)),
        "base_action": int(base_action),
        "selected_action": int(selected_action),
        "base_liveness": dict(base_row),
        "selected_liveness": dict(selected_row),
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    base_action = self._base_action(state, player, legal)
    if not self._enabled_for_state(state):
      self._decision_stats["liveness_teacher_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)

    phase = self._phase_name(state).lower()
    self._decision_stats["liveness_teacher_considered"] += 1
    _inc_counter(self._decision_stats["liveness_teacher_by_phase"], phase)
    scored = self._score_actions(state, player, legal, base_action)
    if not scored:
      self._decision_stats["liveness_teacher_base_kept"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    base_row = scored.get(int(base_action))
    if base_row is None:
      base_row = action_liveness_certificate(
          state, player, base_action, base_action=base_action
      )
    selected_action, selected_row = max(
        scored.items(),
        key=lambda item: self._teacher_key(item[1]),
    )
    selected_action = int(selected_action)

    if bool(base_row.get("own_feasible", False)):
      self._decision_stats["liveness_teacher_base_feasible"] += 1
    if bool(selected_row.get("own_feasible", False)):
      self._decision_stats["liveness_teacher_selected_feasible"] += 1
    if bool(base_row.get("lost_led_token", False)):
      self._decision_stats["liveness_teacher_base_lost_led_token"] += 1
    if bool(selected_row.get("lost_led_token", False)):
      self._decision_stats["liveness_teacher_selected_lost_led_token"] += 1
    self._decision_stats["liveness_teacher_base_public_slot_damage_sum"] += float(
        base_row.get("public_slot_damage", 0.0)
    )
    self._decision_stats["liveness_teacher_selected_public_slot_damage_sum"] += float(
        selected_row.get("public_slot_damage", 0.0)
    )
    self._decision_stats["liveness_teacher_base_own_deficit_sum"] += float(
        base_row.get("own_total_deficit", 0.0)
    )
    self._decision_stats["liveness_teacher_selected_own_deficit_sum"] += float(
        selected_row.get("own_total_deficit", 0.0)
    )
    self._decision_stats["liveness_teacher_base_min_lane_surplus_sum"] += float(
        base_row.get("min_player_lane_surplus_after", 0.0)
    )
    self._decision_stats["liveness_teacher_selected_min_lane_surplus_sum"] += float(
        selected_row.get("min_player_lane_surplus_after", 0.0)
    )
    self._decision_stats["liveness_teacher_base_forced_players_sum"] += float(
        base_row.get("exact_forced_paradox_players_after", 0.0)
    )
    self._decision_stats["liveness_teacher_selected_forced_players_sum"] += float(
        selected_row.get("exact_forced_paradox_players_after", 0.0)
    )
    self._decision_stats["liveness_teacher_base_min_nonparadox_legal_sum"] += float(
        base_row.get("exact_min_nonparadox_legal_after", 0.0)
    )
    self._decision_stats["liveness_teacher_selected_min_nonparadox_legal_sum"] += float(
        selected_row.get("exact_min_nonparadox_legal_after", 0.0)
    )

    if selected_action == int(base_action):
      self._decision_stats["liveness_teacher_base_kept"] += 1
    else:
      self._decision_stats["liveness_teacher_overrides"] += 1
      _inc_counter(self._decision_stats["liveness_teacher_overrides_by_phase"], phase)
      self._append_override_sample(
          state, player, legal, base_action, selected_action, base_row, selected_row
      )
    _record_decision(self._decision_stats, state, selected_action, legal_count)
    return int(selected_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class OwnHandFeasibilityShieldBot:
  """Base policy constrained by exact own-hand future assignment feasibility."""

  def __init__(self, name, args, seed=0):
    self.name = name
    self.args = args
    self.seed = int(seed)
    base_name = self._base_bot_name(name, args)
    self.base_bot = make_bot(base_name, seed=seed)
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "feasibility_shield_disabled": 0,
        "feasibility_shield_considered": 0,
        "feasibility_shield_base_kept": 0,
        "feasibility_shield_overrides": 0,
        "feasibility_shield_base_feasible": 0,
        "feasibility_shield_no_scores": 0,
        "feasibility_shield_legal_infeasible_count": 0,
        "feasibility_shield_base_slot_surplus_sum": 0.0,
        "feasibility_shield_selected_slot_surplus_sum": 0.0,
        "feasibility_shield_selected_deficit_sum": 0.0,
        "feasibility_shield_base_buffer_deficit_sum": 0.0,
        "feasibility_shield_selected_buffer_deficit_sum": 0.0,
        "feasibility_shield_by_phase": {},
        "feasibility_shield_overrides_by_phase": {},
        "feasibility_shield_base_bot": base_name,
        "feasibility_shield_min_slot_surplus": int(
            getattr(args, "feasibility_shield_min_slot_surplus", 0) or 0
        ),
        "feasibility_shield_max_buffer_deficit_to_keep": int(
            getattr(
                args,
                "feasibility_shield_max_buffer_deficit_to_keep",
                -1,
            )
        ),
        "feasibility_shield_override_samples": [],
    })

  @staticmethod
  def _base_bot_name(name, args):
    if name == "feasibility_shield_safe14":
      return "heuristic_safe14"
    if name == "feasibility_shield_safe8":
      return "heuristic_safe8"
    return str(getattr(args, "feasibility_shield_base_bot", "heuristic_safe14"))

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def _enabled_for_phase(self, state):
    spec = str(getattr(self.args, "feasibility_shield_phases", "") or "").strip()
    if not spec:
      return True
    allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
    return self._phase_name(state).lower() in allowed

  def _base_action(self, state, player, legal):
    try:
      action = int(self.base_bot.step(state.clone(), player))
    except Exception:
      action = int(legal[0])
    if action in legal:
      return action
    return int(legal[0])

  @staticmethod
  def _feasibility_key(action, row, base_action):
    return (
        0 if bool(row.get("is_paradox", False)) else 1,
        1 if bool(row.get("feasible", False)) else 0,
        -int(row.get("total_deficit", 0)),
        -int(row.get("buffer_deficit", 0)),
        int(row.get("min_choices_per_remaining_card", 0)),
        int(row.get("slot_surplus", 0)),
        -int(row.get("tight_rank_count", 0)),
        -int(row.get("singleton_card_count", 0)),
        1 if int(action) == int(base_action) else 0,
        -int(action),
    )

  def _score_actions(self, state, player, legal):
    scored = {}
    legal = sorted(int(action) for action in legal)
    non_paradox = [action for action in legal if action != 999]
    if non_paradox:
      legal = non_paradox
    for action in legal:
      scored[int(action)] = own_hand_feasibility_after_action(
          state, player, action
      )
    return scored

  def _append_override_sample(
      self, state, player, legal, base_action, selected_action, base_row,
      selected_row
  ):
    samples = self._decision_stats["feasibility_shield_override_samples"]
    limit = int(getattr(self.args, "feasibility_shield_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": self._phase_name(state).lower(),
        "legal_count": int(len(legal)),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "base_action": int(base_action),
        "selected_action": int(selected_action),
        "base_feasibility": {
            key: value for key, value in base_row.items()
            if key != "open_slots_by_rank"
        },
        "selected_feasibility": {
            key: value for key, value in selected_row.items()
            if key != "open_slots_by_rank"
        },
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    base_action = self._base_action(state, player, legal)
    if not self._enabled_for_phase(state):
      self._decision_stats["feasibility_shield_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = self._phase_name(state).lower()
    min_surplus = int(
        getattr(self.args, "feasibility_shield_min_slot_surplus", 0) or 0
    )
    max_buffer_deficit = int(
        getattr(
            self.args,
            "feasibility_shield_max_buffer_deficit_to_keep",
            -1,
        )
    )
    self._decision_stats["feasibility_shield_considered"] += 1
    _inc_counter(self._decision_stats["feasibility_shield_by_phase"], phase)
    scored = self._score_actions(state, player, legal)
    if not scored:
      self._decision_stats["feasibility_shield_no_scores"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    legal_infeasible = sum(
        1 for row in scored.values() if not bool(row.get("feasible", False))
    )
    self._decision_stats["feasibility_shield_legal_infeasible_count"] += int(
        legal_infeasible
    )
    base_row = scored.get(int(base_action))
    if base_row is None:
      base_row = own_hand_feasibility_after_action(state, player, base_action)
    base_feasible = bool(base_row.get("feasible", False))
    if base_feasible:
      self._decision_stats["feasibility_shield_base_feasible"] += 1
    self._decision_stats["feasibility_shield_base_slot_surplus_sum"] += float(
        base_row.get("slot_surplus", 0)
    )
    self._decision_stats[
        "feasibility_shield_base_buffer_deficit_sum"
    ] += float(base_row.get("buffer_deficit", 0))
    base_buffer_ok = (
        max_buffer_deficit < 0
        or int(base_row.get("buffer_deficit", 0)) <= max_buffer_deficit
    )
    if (
        base_feasible
        and int(base_row.get("slot_surplus", 0)) >= min_surplus
        and base_buffer_ok
    ):
      selected_action = int(base_action)
      selected_row = base_row
      self._decision_stats["feasibility_shield_base_kept"] += 1
    else:
      selected_action, selected_row = max(
          scored.items(),
          key=lambda item: self._feasibility_key(
              item[0], item[1], base_action
          ),
      )
      selected_action = int(selected_action)
      if selected_action != int(base_action):
        self._decision_stats["feasibility_shield_overrides"] += 1
        _inc_counter(
            self._decision_stats["feasibility_shield_overrides_by_phase"],
            phase,
        )
        self._append_override_sample(
            state,
            player,
            legal,
            base_action,
            selected_action,
            base_row,
            selected_row,
        )
      else:
        self._decision_stats["feasibility_shield_base_kept"] += 1
    self._decision_stats["feasibility_shield_selected_slot_surplus_sum"] += float(
        selected_row.get("slot_surplus", 0)
    )
    self._decision_stats["feasibility_shield_selected_deficit_sum"] += float(
        selected_row.get("total_deficit", 0)
    )
    self._decision_stats[
        "feasibility_shield_selected_buffer_deficit_sum"
    ] += float(selected_row.get("buffer_deficit", 0))
    _record_decision(self._decision_stats, state, selected_action, legal_count)
    return int(selected_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class SurvivalShieldBot:
  """Built-in base policy constrained by same-policy round survival rollouts."""

  def __init__(self, name, args, seed=0):
    self.name = name
    self.args = args
    self.seed = int(seed)
    self.players = int(getattr(args, "players", 3))
    base_name = self._base_bot_name(name, args)
    continuation_name = str(
        getattr(args, "survival_shield_continuation_bot", base_name) or base_name
    )
    self.base_bot = make_bot(base_name, seed=seed)
    self._continuation_bots = [
        make_bot(continuation_name, seed=seed + 1009 + seat)
        for seat in range(max(1, self.players))
    ]
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "survival_shield_disabled": 0,
        "survival_shield_considered": 0,
        "survival_shield_base_kept": 0,
        "survival_shield_overrides": 0,
        "survival_shield_fallback_max_survival": 0,
        "survival_shield_no_scores": 0,
        "survival_shield_base_invalid": 0,
        "survival_shield_paradox_candidate_filtered": 0,
        "survival_shield_early_paradox_rollouts": 0,
        "survival_shield_sampler_fallback": 0,
        "survival_shield_scored_candidates": 0,
        "survival_shield_candidate_bot_additions": 0,
        "survival_shield_candidate_feature_additions": 0,
        "survival_shield_candidate_fill_additions": 0,
        "survival_shield_rollouts": 0,
        "survival_shield_base_survival_sum": 0.0,
        "survival_shield_selected_survival_sum": 0.0,
        "survival_shield_max_survival_sum": 0.0,
        "survival_shield_base_survival_mean_sum": 0.0,
        "survival_shield_selected_survival_mean_sum": 0.0,
        "survival_shield_max_survival_mean_sum": 0.0,
        "survival_shield_base_survival_lcb_sum": 0.0,
        "survival_shield_selected_survival_lcb_sum": 0.0,
        "survival_shield_max_survival_lcb_sum": 0.0,
        "survival_shield_by_phase": {},
        "survival_shield_overrides_by_phase": {},
        "survival_shield_fallback_by_phase": {},
        "survival_shield_base_bot": base_name,
        "survival_shield_continuation_bot": continuation_name,
        "survival_shield_threshold": float(
            getattr(args, "survival_shield_threshold", 0.45)
        ),
        "survival_shield_selection_mode": str(
            getattr(args, "survival_shield_selection_mode", "threshold")
        ),
        "survival_shield_override_margin": float(
            getattr(args, "survival_shield_override_margin", 0.05)
        ),
        "survival_shield_override_mean_delta": float(
            getattr(args, "survival_shield_override_mean_delta", 0.20)
        ),
        "survival_shield_score_mode": str(
            getattr(args, "survival_shield_score_mode", "wilson_lcb")
        ),
        "survival_shield_lcb_z": float(
            getattr(args, "survival_shield_lcb_z", 1.96)
        ),
        "survival_shield_include_bots": str(
            getattr(
                args,
                "survival_shield_include_bots",
                "heuristic_safe14,heuristic_safe8,heuristic,heuristic_target2,heuristic_adj2",
            )
        ),
        "survival_shield_feature_candidates": bool(
            getattr(args, "survival_shield_feature_candidates", True)
        ),
    })

  @staticmethod
  def _base_bot_name(name, args):
    if name == "survival_shield_safe14":
      return "heuristic_safe14"
    if name == "survival_shield_safe8":
      return "heuristic_safe8"
    return str(getattr(args, "survival_shield_base_bot", "heuristic_safe14"))

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def _enabled_for_phase(self, state):
    spec = str(getattr(self.args, "survival_shield_phases", "") or "").strip()
    if not spec:
      return True
    allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
    return self._phase_name(state).lower() in allowed

  def _base_action(self, state, player, legal):
    try:
      action = int(self.base_bot.step(state.clone(), player))
    except Exception:
      action = int(legal[0])
    if action not in legal:
      self._decision_stats["survival_shield_base_invalid"] += 1
      return int(legal[0])
    return action

  def _candidate_actions(self, state, player, legal, base_action):
    legal = sorted(int(action) for action in legal)
    non_paradox_legal = [action for action in legal if action != 999]
    if non_paradox_legal:
      if 999 in legal:
        self._decision_stats["survival_shield_paradox_candidate_filtered"] += 1
      legal = non_paradox_legal
    max_actions = int(getattr(self.args, "survival_shield_max_actions", 0) or 0)
    if max_actions <= 0 or len(legal) <= max_actions:
      return legal
    selected = []
    selected_set = set()

    def add(action, source):
      if len(selected) >= max_actions:
        return False
      action = int(action)
      if action not in legal or action in selected_set:
        return False
      selected.append(action)
      selected_set.add(action)
      if source:
        _inc_counter(
            self._decision_stats,
            f"survival_shield_candidate_{source}_additions",
        )
      return True

    if int(base_action) in legal:
      add(int(base_action), "")
    for bot_name in _split_csv(
        getattr(self.args, "survival_shield_include_bots", "")
    ):
      if len(selected) >= max_actions:
        break
      try:
        action = make_bot(bot_name, seed=self.seed).step(state.clone(), player)
      except Exception:
        continue
      add(action, "bot")

    if bool(getattr(self.args, "survival_shield_feature_candidates", True)):
      try:
        action_features = action_feature_matrix(
            state, player, state.num_distinct_actions()
        )
      except Exception:
        action_features = None
      feature_preferences = [
          ("hits_prediction", True),
          ("can_still_hit_after", True),
          ("post_hit_low_card_survival_margin", True),
          ("post_hit_forced_card_pressure", False),
          ("token_loss_legal_lead_damage", False),
          ("token_loss_dead_card_damage", False),
          ("token_loss_singleton_card_damage", False),
          ("token_loss_rank_damage", False),
          ("token_loss_creates_no_exit", False),
          ("future_safe_flex_score_after", True),
          ("future_max_rank_deficit_after", False),
          ("future_buffer_deficit_after", False),
          ("future_no_exit_after", False),
          ("discard_rank_deficit_relief", True),
          ("discard_rank_buffer_relief", True),
          ("discard_safe_flex_delta", True),
          ("prediction_action_abs_gap_to_expected", False),
          ("prediction_action_under_expected", True),
      ]
      if action_features is not None:
        for feature_name, reverse in feature_preferences:
          if len(selected) >= max_actions:
            break
          feature_idx = APPENDED_ACTION_FEATURE_INDEX.get(feature_name)
          if feature_idx is None or feature_idx >= action_features.shape[1]:
            continue
          candidates = [action for action in legal if action not in selected_set]
          if not candidates:
            break
          key = lambda action: float(action_features[int(action), feature_idx])
          action = max(candidates, key=key) if reverse else min(candidates, key=key)
          add(action, "feature")

    remaining = [action for action in legal if action not in selected_set]
    if remaining:
      fill_count = min(max_actions - len(selected), len(remaining))
      if fill_count >= len(remaining):
        fill_actions = remaining
      else:
        positions = np.linspace(
            0, len(remaining) - 1, num=fill_count, dtype=np.int32
        )
        fill_actions = [remaining[int(pos)] for pos in positions]
      for action in fill_actions:
        if len(selected) >= max_actions:
          break
        add(action, "fill")
    return sorted(selected)

  @staticmethod
  def _non_paradox_fallback(legal, preferred_action):
    preferred_action = int(preferred_action)
    if preferred_action != 999:
      return preferred_action
    for action in sorted(int(action) for action in legal):
      if action != 999:
        return int(action)
    return preferred_action

  def _continuation_action(self, state, player):
    legal = state.legal_actions(player)
    if not legal:
      raise ValueError("Continuation state has no legal actions")
    bot = self._continuation_bots[int(player) % len(self._continuation_bots)]
    try:
      action = int(bot.step(state.clone(), player))
      if action in legal:
        return action
    except Exception:
      pass
    return int(legal[0])

  def _any_paradox(self, state):
    paradoxes = getattr(state, "_has_paradoxed", [False] * self.players)
    return any(bool(value) for value in paradoxes)

  def _rollout_survives(self, belief_state, player, first_action):
    rollout = belief_state.clone()
    rollout.apply_action(int(first_action))
    if self._any_paradox(rollout):
      self._decision_stats["survival_shield_early_paradox_rollouts"] += 1
      return 0.0
    while not rollout.is_terminal():
      if rollout.is_chance_node():
        actions, probs = zip(*rollout.chance_outcomes())
        rollout.apply_action(int(np.random.choice(actions, p=probs)))
        if self._any_paradox(rollout):
          self._decision_stats["survival_shield_early_paradox_rollouts"] += 1
          return 0.0
        continue
      current = int(rollout.current_player())
      rollout.apply_action(self._continuation_action(rollout, current))
      if self._any_paradox(rollout):
        self._decision_stats["survival_shield_early_paradox_rollouts"] += 1
        return 0.0
    return 0.0 if self._any_paradox(rollout) else 1.0

  @staticmethod
  def _wilson_lower_bound(successes, trials, z):
    trials = int(trials)
    if trials <= 0:
      return 0.0
    z = max(0.0, float(z))
    mean_value = float(successes) / float(trials)
    if z == 0.0:
      return float(np.clip(mean_value, 0.0, 1.0))
    z2 = z * z
    denominator = 1.0 + z2 / float(trials)
    center = mean_value + z2 / (2.0 * float(trials))
    radius = z * math.sqrt(
        (mean_value * (1.0 - mean_value) + z2 / (4.0 * float(trials)))
        / float(trials)
    )
    return float(np.clip((center - radius) / denominator, 0.0, 1.0))

  @staticmethod
  def _wilson_upper_bound(successes, trials, z):
    trials = int(trials)
    if trials <= 0:
      return 0.0
    z = max(0.0, float(z))
    mean_value = float(successes) / float(trials)
    if z == 0.0:
      return float(np.clip(mean_value, 0.0, 1.0))
    z2 = z * z
    denominator = 1.0 + z2 / float(trials)
    center = mean_value + z2 / (2.0 * float(trials))
    radius = z * math.sqrt(
        (mean_value * (1.0 - mean_value) + z2 / (4.0 * float(trials)))
        / float(trials)
    )
    return float(np.clip((center + radius) / denominator, 0.0, 1.0))

  def _score_rollout_values(self, values):
    values = [float(value) for value in values]
    if not values:
      return None
    mean_value = float(np.mean(values))
    lcb = self._wilson_lower_bound(
        sum(values),
        len(values),
        float(getattr(self.args, "survival_shield_lcb_z", 1.96)),
    )
    ucb = self._wilson_upper_bound(
        sum(values),
        len(values),
        float(getattr(self.args, "survival_shield_lcb_z", 1.96)),
    )
    mode = str(
        getattr(self.args, "survival_shield_score_mode", "wilson_lcb")
        or "wilson_lcb"
    )
    score = mean_value if mode == "mean" else lcb
    return {
        "score": float(score),
        "mean": float(mean_value),
        "wilson_lcb": float(lcb),
        "wilson_ucb": float(ucb),
        "rollouts": int(len(values)),
    }

  @staticmethod
  def _score_metric(row, key, default_key="score"):
    if isinstance(row, dict):
      if key in row:
        return float(row[key])
      if default_key in row:
        return float(row[default_key])
    return float(row)

  def _sample_belief_states(self, state, player):
    samples = max(1, int(getattr(self.args, "survival_shield_samples", 1) or 1))
    try:
      return sampled_belief_states_for_policy(
          state,
          player,
          samples,
          self.args,
          None,
          torch.device("cpu"),
          1.0,
          context="eval",
      )
    except Exception:
      self._decision_stats["survival_shield_sampler_fallback"] += 1
      return [state.clone()]

  def _score_candidates(self, state, player, candidates):
    action_results = {int(action): [] for action in candidates}
    rollouts = max(1, int(getattr(self.args, "survival_shield_rollouts", 4) or 1))
    for belief_state in self._sample_belief_states(state, player):
      legal_in_belief = set(int(action) for action in belief_state.legal_actions(player))
      legal_candidates = [
          int(action) for action in candidates if int(action) in legal_in_belief
      ]
      if not legal_candidates:
        continue
      for _ in range(rollouts):
        np_state = np.random.get_state()
        for action in legal_candidates:
          np.random.set_state(np_state)
          action_results[action].append(
              float(self._rollout_survives(belief_state, player, action))
          )
          self._decision_stats["survival_shield_rollouts"] += 1
    scores = {}
    for action, values in action_results.items():
      score_row = self._score_rollout_values(values)
      if score_row is not None:
        scores[action] = score_row
    return scores

  @staticmethod
  def _selection_key(action, survival, base_action):
    action = int(action)
    return (
        float(survival),
        action != 999,
        action == int(base_action),
        -action,
    )

  def step(self, state, player):
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    base_action = self._base_action(state, player, legal)
    if not self._enabled_for_phase(state):
      self._decision_stats["survival_shield_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = self._phase_name(state).lower()
    threshold = float(getattr(self.args, "survival_shield_threshold", 0.45))
    self._decision_stats["survival_shield_considered"] += 1
    _inc_counter(self._decision_stats["survival_shield_by_phase"], phase)
    candidates = self._candidate_actions(state, player, legal, base_action)
    score_rows = self._score_candidates(state, player, candidates)
    scores = {
        int(action): self._score_metric(row, "score")
        for action, row in score_rows.items()
    }
    score_means = {
        int(action): self._score_metric(row, "mean")
        for action, row in score_rows.items()
    }
    score_lcbs = {
        int(action): self._score_metric(row, "wilson_lcb")
        for action, row in score_rows.items()
    }
    score_ucbs = {
        int(action): self._score_metric(row, "wilson_ucb", default_key="mean")
        for action, row in score_rows.items()
    }
    self._decision_stats["survival_shield_scored_candidates"] += len(scores)
    if not scores:
      self._decision_stats["survival_shield_no_scores"] += 1
      selected_action = self._non_paradox_fallback(legal, base_action)
      _record_decision(self._decision_stats, state, selected_action, legal_count)
      return int(selected_action)
    max_action, max_survival = max(
        scores.items(),
        key=lambda item: self._selection_key(item[0], item[1], base_action),
    )
    base_action_scored = int(base_action) in scores
    base_survival = float(scores.get(int(base_action), 0.0))
    base_mean = float(score_means.get(int(base_action), base_survival))
    base_lcb = float(score_lcbs.get(int(base_action), base_survival))
    base_ucb = float(score_ucbs.get(int(base_action), base_mean))
    max_mean = float(score_means.get(int(max_action), max_survival))
    max_lcb = float(score_lcbs.get(int(max_action), max_survival))
    max_ucb = float(score_ucbs.get(int(max_action), max_mean))
    self._decision_stats["survival_shield_base_survival_sum"] += base_survival
    self._decision_stats["survival_shield_max_survival_sum"] += float(max_survival)
    self._decision_stats["survival_shield_base_survival_mean_sum"] += base_mean
    self._decision_stats["survival_shield_base_survival_lcb_sum"] += base_lcb
    self._decision_stats["survival_shield_max_survival_mean_sum"] += max_mean
    self._decision_stats["survival_shield_max_survival_lcb_sum"] += max_lcb
    selection_mode = str(
        getattr(self.args, "survival_shield_selection_mode", "threshold")
        or "threshold"
    )
    if selection_mode == "dominance":
      margin = float(getattr(self.args, "survival_shield_override_margin", 0.05))
      mean_delta = float(
          getattr(self.args, "survival_shield_override_mean_delta", 0.20)
      )
      confident_bound_win = (
          max_action != int(base_action)
          and max_lcb > base_ucb + margin
      )
      confident_mean_win = (
          max_action != int(base_action)
          and max_mean - base_mean >= mean_delta
          and max_lcb >= base_lcb
      )
      if not base_action_scored:
        selected_action = int(max_action)
        selected_survival = float(max_survival)
        self._decision_stats["survival_shield_fallback_max_survival"] += 1
        _inc_counter(self._decision_stats["survival_shield_fallback_by_phase"], phase)
        if selected_action != int(base_action):
          self._decision_stats["survival_shield_overrides"] += 1
          _inc_counter(
              self._decision_stats["survival_shield_overrides_by_phase"], phase
          )
      elif confident_bound_win or confident_mean_win:
        selected_action = int(max_action)
        selected_survival = float(max_survival)
        self._decision_stats["survival_shield_overrides"] += 1
        _inc_counter(self._decision_stats["survival_shield_overrides_by_phase"], phase)
      else:
        selected_action = int(base_action)
        selected_survival = base_survival
        self._decision_stats["survival_shield_base_kept"] += 1
    elif base_action_scored and base_survival >= threshold:
      selected_action = int(base_action)
      selected_survival = base_survival
      self._decision_stats["survival_shield_base_kept"] += 1
    else:
      safe_scores = {
          action: survival
          for action, survival in scores.items()
          if float(survival) >= threshold
      }
      if safe_scores:
        selected_action, selected_survival = max(
            safe_scores.items(),
            key=lambda item: self._selection_key(item[0], item[1], base_action),
        )
        self._decision_stats["survival_shield_overrides"] += 1
        _inc_counter(self._decision_stats["survival_shield_overrides_by_phase"], phase)
      else:
        selected_action, selected_survival = int(max_action), float(max_survival)
        self._decision_stats["survival_shield_fallback_max_survival"] += 1
        _inc_counter(self._decision_stats["survival_shield_fallback_by_phase"], phase)
        if selected_action != int(base_action):
          self._decision_stats["survival_shield_overrides"] += 1
          _inc_counter(
              self._decision_stats["survival_shield_overrides_by_phase"], phase
          )
    self._decision_stats["survival_shield_selected_survival_sum"] += float(
        selected_survival
    )
    self._decision_stats["survival_shield_selected_survival_mean_sum"] += float(
        score_means.get(int(selected_action), selected_survival)
    )
    self._decision_stats["survival_shield_selected_survival_lcb_sum"] += float(
        score_lcbs.get(int(selected_action), selected_survival)
    )
    _record_decision(self._decision_stats, state, int(selected_action), legal_count)
    return int(selected_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


def load_improve_scorer(path, device):
  payload = torch.load(path, map_location=device, weights_only=False)
  feature_size = int(payload.get("feature_size", len(payload["feature_mean"])))
  saved_args = payload.get("args", {})
  hidden = int(saved_args.get("hidden", 64))
  model = ImproveMLP(feature_size, hidden).to(device)
  model.load_state_dict(payload["model_state"])
  model.eval()
  risk_model = None
  if payload.get("risk_model_state") is not None:
    risk_model = ImproveMLP(feature_size, hidden).to(device)
    risk_model.load_state_dict(payload["risk_model_state"])
    risk_model.eval()
  mean_arr = np.asarray(payload["feature_mean"], dtype=np.float32)
  std_arr = np.asarray(payload["feature_std"], dtype=np.float32)
  std_arr = np.where(std_arr < 1e-6, 1.0, std_arr)
  return {
      "model": model,
      "risk_model": risk_model,
      "feature_mean": mean_arr,
      "feature_std": std_arr,
      "feature_size": feature_size,
      "feature_schema_version": str(
          payload.get("feature_schema_version")
          or saved_args.get("feature_schema_version")
          or "legacy_target_match_v1"
      ),
      "include_target_match_features": bool(
          payload.get("include_target_match_features")
          if "include_target_match_features" in payload
          else saved_args.get(
              "include_target_match_features",
              not bool(payload.get("deployable_feature_schema", False)),
          )
      ),
      "observation_feature_mode": str(
          payload.get("observation_feature_mode")
          or saved_args.get("observation_feature_mode")
          or "none"
      ),
      "policy_context_mode": str(
          payload.get("policy_context_mode")
          or saved_args.get("policy_context_mode")
          or "none"
      ),
      "args": saved_args,
  }


class AZImproveGraftBot:
  """Run119 policy with an abstaining pairwise improvement override model."""

  def __init__(self, model, improve_scorer, name, device, model_args):
    self.name = name
    self.model = model
    self.improve_scorer = improve_scorer
    self.device = device
    self.args = model_args
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "improve_disabled": 0,
        "improve_disabled_by_reason": {},
        "improve_considered": 0,
        "improve_abstained": 0,
        "improve_overrides": 0,
        "improve_shadow_overrides": 0,
        "improve_scored_candidates": 0,
        "improve_risk_vetoed": 0,
        "improve_risk_model_missing": 0,
        "improve_by_phase": {},
        "improve_overrides_by_phase": {},
        "improve_shadow_overrides_by_phase": {},
        "improve_override_samples": [],
        "improve_shadow_override_samples": [],
        "improve_max_prob_sum": 0.0,
        "improve_max_safe_prob_sum": 0.0,
    })

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  @staticmethod
  def _near_paradox_pressure_score(state, player, legal):
    if not legal:
      return 0.0
    action_features = action_feature_matrix(
        state, player, state.num_distinct_actions()
    )
    legal_features = action_features[list(legal)]

    def feature(name, default=0.0):
      idx = APPENDED_ACTION_FEATURE_INDEX.get(name)
      if idx is None or idx >= legal_features.shape[1]:
        return np.full(len(legal_features), float(default), dtype=np.float32)
      return legal_features[:, idx].astype(np.float32)

    hit_future = feature("hit_with_future_tricks")
    forced_pressure = feature("post_hit_forced_card_pressure")
    low_legal_ratio = feature("post_hit_low_legal_lead_ratio")
    survival_margin = feature("post_hit_low_card_survival_margin")
    token_damage = (
        feature("token_loss_legal_lead_damage")
        + feature("token_loss_dead_card_damage")
        + feature("token_loss_singleton_card_damage")
        + feature("token_loss_rank_damage")
    )
    no_exit = feature("token_loss_creates_no_exit")
    pressure = np.maximum(
        hit_future * (
            forced_pressure
            + low_legal_ratio
            + np.maximum(0.0, -survival_margin)
        ),
        token_damage + no_exit,
    )
    return float(np.max(pressure)) if pressure.size else 0.0

  def _enabled_for_state(self, state, legal_count):
    spec = str(getattr(self.args, "improve_phases", "play") or "").strip()
    if not spec:
      phase_allowed = True
    else:
      allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
      phase_allowed = self._phase_name(state).lower() in allowed
    if not phase_allowed:
      return False, "phase"
    min_legal = int(getattr(self.args, "improve_min_legal_count", 0) or 0)
    max_legal = int(getattr(self.args, "improve_max_legal_count", 0) or 0)
    if min_legal > 0 and legal_count < min_legal:
      return False, "legal_count_low"
    if max_legal > 0 and legal_count > max_legal:
      return False, "legal_count_high"
    min_trick = int(getattr(self.args, "improve_min_trick_number", -1))
    if min_trick >= 0 and int(getattr(state, "_trick_number", -1)) < min_trick:
      return False, "trick_number"
    min_round = int(getattr(self.args, "improve_min_match_round", -1))
    if min_round >= 0 and int(getattr(state, "_match_round", -1)) < min_round:
      return False, "match_round"
    if bool(getattr(self.args, "improve_near_paradox_proxy", False)):
      player = int(state.current_player())
      pressure = self._near_paradox_pressure_score(
          state, player, state.legal_actions(player)
      )
      minimum = float(
          getattr(self.args, "improve_near_paradox_min_pressure", 0.1) or 0.0
      )
      if pressure < minimum:
        return False, "near_paradox_proxy"
    return True, "enabled"

  def _candidate_actions(self, state, player, legal, policy, policy_action):
    legal = list(legal)
    max_actions = int(getattr(self.args, "improve_max_actions", 0) or 0)
    if max_actions > 0:
      sampler_args = SimpleNamespace(
          counterfactual_action_max_legal=max_actions,
          counterfactual_action_top_policy=max(
              0, int(getattr(self.args, "improve_top_policy", 0) or 0)
          ),
          counterfactual_action_include_bots=str(
              getattr(
                  self.args,
                  "improve_include_bots",
                  "heuristic,heuristic_target2,heuristic_adj2",
              )
          ),
          counterfactual_action_feature_candidates=bool(
              getattr(self.args, "improve_feature_candidates", True)
          ),
      )
      sampled = sampled_counterfactual_legal_actions(
          state, player, legal, sampler_args, policy=policy
      )
      return sorted(int(action) for action in sampled if action != policy_action)
    return sorted(int(action) for action in legal if action != policy_action)

  def _score_candidates(self, state, player, legal, policy, policy_action, candidates):
    if not candidates:
      return {}
    action_features = action_feature_matrix(
        state, player, state.num_distinct_actions()
    )
    observation = np.array(state.observation_tensor(player), dtype=np.float32)
    observation_feature_mode = str(
        self.improve_scorer.get("observation_feature_mode", "none") or "none"
    )
    policy_context_mode = str(
        self.improve_scorer.get("policy_context_mode", "none") or "none"
    )
    legal_mask = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    legal_mask[list(legal)] = 1.0
    rows = [
        pair_feature_vector_from_matrix(
            action_features,
            legal_mask,
            candidate,
            policy_action,
            target_action=candidate,
            label_margin=0.0,
            legal_count=len(legal),
            candidate_count=len(candidates),
            include_label_margin=False,
            observation=observation,
            observation_feature_mode=observation_feature_mode,
            policy_context=policy,
            policy_context_mode=policy_context_mode,
            include_target_match_features=bool(
                self.improve_scorer.get("include_target_match_features", False)
            ),
        )
        for candidate in candidates
    ]
    x = np.asarray(rows, dtype=np.float32)
    scorer_feature_size = int(self.improve_scorer["feature_size"])
    if x.shape[1] > scorer_feature_size:
      x = x[:, :scorer_feature_size]
    if x.shape[1] != scorer_feature_size:
      raise ValueError(
          "improve feature size mismatch: "
          f"{x.shape[1]} vs {scorer_feature_size}"
      )
    x = (x - self.improve_scorer["feature_mean"]) / self.improve_scorer["feature_std"]
    with torch.no_grad():
      x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
      logits = self.improve_scorer["model"](x_t)
      probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
      risk_model = self.improve_scorer.get("risk_model")
      safe_probs = (
          torch.sigmoid(risk_model(x_t)).cpu().numpy().astype(np.float32)
          if risk_model is not None else
          np.ones_like(probs, dtype=np.float32)
      )
    return {
        int(action): {
            "improve": float(prob),
            "safe": float(safe_prob),
        }
        for action, prob, safe_prob in zip(candidates, probs, safe_probs)
    }

  def _append_override_sample(
      self,
      key,
      state,
      player,
      legal,
      policy,
      policy_action,
      candidate_action,
      candidate_scores,
      threshold,
      risk_threshold,
  ):
    samples = self._decision_stats.setdefault(key, [])
    limit = int(getattr(self.args, "improve_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    sample = {
        "player": int(player),
        "phase": self._phase_name(state).lower(),
        "legal_count": int(len(legal)),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "match_round": int(getattr(state, "_match_round", -1)),
        "policy_action": int(policy_action),
        "candidate_action": int(candidate_action),
        "policy_prob": float(policy[policy_action]),
        "candidate_policy_prob": float(policy[candidate_action]),
        "candidate_improve_prob": float(candidate_scores["improve"]),
        "candidate_safe_prob": float(candidate_scores["safe"]),
        "improve_threshold": float(threshold),
        "improve_risk_threshold": float(risk_threshold),
        "legal_actions": [int(action) for action in legal],
        "observation": state.observation_string(player),
    }
    samples.append(sample)

  def step(self, state, player):
    legal = state.legal_actions(player)
    legal_count = len(legal)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return legal[0]
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    policy_action = max(legal, key=lambda action: policy[action])
    enabled, disabled_reason = self._enabled_for_state(state, legal_count)
    if not enabled:
      self._decision_stats["improve_disabled"] += 1
      _inc_counter(
          self._decision_stats["improve_disabled_by_reason"],
          disabled_reason,
      )
      _record_decision(self._decision_stats, state, policy_action, legal_count)
      return policy_action
    phase = self._phase_name(state).lower()
    self._decision_stats["improve_considered"] += 1
    _inc_counter(self._decision_stats["improve_by_phase"], phase)
    candidates = self._candidate_actions(state, player, legal, policy, policy_action)
    scores = self._score_candidates(
        state, player, legal, policy, policy_action, candidates
    )
    self._decision_stats["improve_scored_candidates"] += len(scores)
    threshold = float(getattr(self.args, "improve_threshold", 0.95))
    if not scores:
      self._decision_stats["improve_abstained"] += 1
      _record_decision(self._decision_stats, state, policy_action, legal_count)
      return policy_action
    best_action, best_scores = max(
        scores.items(),
        key=lambda item: (float(item[1]["improve"]), float(policy[item[0]])),
    )
    best_prob = float(best_scores["improve"])
    best_safe_prob = float(best_scores["safe"])
    self._decision_stats["improve_max_prob_sum"] += float(best_prob)
    self._decision_stats["improve_max_safe_prob_sum"] += float(best_safe_prob)
    if best_prob >= threshold:
      risk_threshold = float(getattr(self.args, "improve_risk_threshold", 0.0))
      if risk_threshold > 0 and self.improve_scorer.get("risk_model") is None:
        self._decision_stats["improve_risk_model_missing"] += 1
        self._decision_stats["improve_abstained"] += 1
        _record_decision(self._decision_stats, state, policy_action, legal_count)
        return policy_action
      if risk_threshold > 0 and best_safe_prob < risk_threshold:
        self._decision_stats["improve_risk_vetoed"] += 1
        self._decision_stats["improve_abstained"] += 1
        _record_decision(self._decision_stats, state, policy_action, legal_count)
        return policy_action
      if bool(getattr(self.args, "improve_shadow_only", False)):
        self._decision_stats["improve_shadow_overrides"] += 1
        self._decision_stats["improve_abstained"] += 1
        _inc_counter(self._decision_stats["improve_shadow_overrides_by_phase"], phase)
        self._append_override_sample(
            "improve_shadow_override_samples",
            state,
            player,
            legal,
            policy,
            policy_action,
            best_action,
            best_scores,
            threshold,
            risk_threshold,
        )
        _record_decision(self._decision_stats, state, policy_action, legal_count)
        return policy_action
      self._decision_stats["improve_overrides"] += 1
      _inc_counter(self._decision_stats["improve_overrides_by_phase"], phase)
      self._append_override_sample(
          "improve_override_samples",
          state,
          player,
          legal,
          policy,
          policy_action,
          best_action,
          best_scores,
          threshold,
          risk_threshold,
      )
      _record_decision(self._decision_stats, state, best_action, legal_count)
      return int(best_action)
    self._decision_stats["improve_abstained"] += 1
    _record_decision(self._decision_stats, state, policy_action, legal_count)
    return policy_action

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    considered = float(stats.get("improve_considered", 0) or 0)
    if considered:
      stats["improve_max_prob_avg_when_considered"] = round(
          float(stats.get("improve_max_prob_sum", 0.0)) / considered, 6
      )
      stats["improve_max_safe_prob_avg_when_considered"] = round(
          float(stats.get("improve_max_safe_prob_sum", 0.0)) / considered, 6
      )
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


def _split_csv(value):
  return [item.strip() for item in str(value).split(",") if item.strip()]


def make_eval_bot(
    name, seed, model, model_args, device, belief_samples, belief_sims,
    mcts_sims,
    neural_bots=None,
    eval_args=None,
):
  def _make_neural_entry_bot(entry, bot_name, entry_model=None, entry_args=None):
    entry_model = entry["model"] if entry_model is None else entry_model
    entry_args = entry["args"] if entry_args is None else entry_args
    mode = str(entry.get("mode", "policy"))
    if mode == "belief_policy":
      return AZBeliefPolicyBot(
          entry_model, bot_name, device, entry_args.value_scale,
          belief_samples, entry_args
      )
    if mode == "belief":
      return AZBeliefSearchBot(
          entry_model, bot_name, device, entry_args, belief_samples, belief_sims
      )
    if mode == "mcts":
      return AZSearchBot(entry_model, bot_name, device, entry_args, max(1, mcts_sims))
    if mode == "q_policy":
      return AZQPolicyBot(
          entry_model,
          bot_name,
          device,
          entry_args,
          phase_risk_model=entry.get("phase_risk_model"),
          phase_risk_phases=getattr(
              entry_args, "action_paradox_phase_risk_phases", ""
          ),
      )
    if mode == "value_shield":
      return AZValueShieldPolicyBot(
          entry_model,
          entry.get("survival_model", entry_model),
          bot_name,
          device,
          entry_args,
      )
    if mode == "liveness_shield":
      return AZLivenessShieldPolicyBot(entry_model, bot_name, device, entry_args)
    if mode == "residual_policy":
      return AZResidualPolicyBot(
          entry_model,
          entry["anchor_model"],
          bot_name,
          device,
          entry_args,
          anchor_args=entry.get("anchor_args"),
      )
    if mode == "residual_q_policy":
      return AZResidualQPolicyBot(
          entry_model,
          entry["anchor_model"],
          bot_name,
          device,
          entry_args,
          anchor_args=entry.get("anchor_args"),
          phase_risk_model=entry.get("phase_risk_model"),
          phase_risk_phases=getattr(
              entry_args, "action_paradox_phase_risk_phases", ""
          ),
      )
    if mode == "residual_q_risk_policy":
      return AZResidualQRiskPolicyBot(
          entry_model,
          entry["anchor_model"],
          bot_name,
          device,
          entry_args,
          anchor_args=entry.get("anchor_args"),
          phase_risk_model=entry.get("phase_risk_model"),
          phase_risk_phases=getattr(
              entry_args, "action_paradox_phase_risk_phases", ""
          ),
      )
    if mode == "root_rollout":
      return AZRootRolloutBot(entry_model, bot_name, device, entry_args)
    if mode == "improve_graft":
      return AZImproveGraftBot(
          entry_model,
          entry["improve_scorer"],
          bot_name,
          device,
          entry_args,
      )
    return AZPolicyBot(entry_model, bot_name, device, entry_args.value_scale)

  if neural_bots and name in neural_bots:
    entry = neural_bots[name]
    if entry["mode"] == "play_graft":
      base_entry = {**entry, "mode": str(getattr(entry["args"], "base_mode", "policy"))}
      graft_entry = {**entry, "mode": str(getattr(entry["args"], "graft_mode", "policy"))}
      graft_builtin_bot = str(entry.get("graft_builtin_bot", "") or "")
      if graft_builtin_bot:
        graft_bot = make_bot(graft_builtin_bot, seed=seed)
      else:
        graft_bot = _make_neural_entry_bot(
            graft_entry,
            f"{name}_graft",
            entry_model=entry["graft_model"],
            entry_args=entry["graft_args"],
        )
      return AZPhaseGraftBot(
          _make_neural_entry_bot(base_entry, f"{name}_base"),
          graft_bot,
          name,
          getattr(entry["args"], "graft_phases", "play"),
      )
    return _make_neural_entry_bot(entry, name)
  if name == "az_policy":
    if model is None:
      raise ValueError("az_policy requires --checkpoint")
    return AZPolicyBot(model, name, device, model_args.value_scale)
  if name == "az_belief_policy":
    if model is None:
      raise ValueError("az_belief_policy requires --checkpoint")
    return AZBeliefPolicyBot(
        model, name, device, model_args.value_scale, belief_samples, model_args
    )
  if name == "az_belief_search":
    if model is None:
      raise ValueError("az_belief_search requires --checkpoint")
    return AZBeliefSearchBot(
        model, name, device, model_args, belief_samples, belief_sims
    )
  if base_bot_name(name).startswith("survival_shield"):
    shield_args = eval_args or model_args
    if shield_args is None:
      shield_args = _model_args(3)
    return SurvivalShieldBot(base_bot_name(name), shield_args, seed=seed)
  if base_bot_name(name).startswith("exit_liquidity_shield"):
    shield_args = eval_args or model_args
    if shield_args is None:
      shield_args = _model_args(3)
    return ExitLiquidityShieldBot(base_bot_name(name), shield_args, seed=seed)
  if base_bot_name(name).startswith("liquidity_feasibility_shield"):
    shield_args = eval_args or model_args
    if shield_args is None:
      shield_args = _model_args(3)
    return LiquidityFeasibilityShieldBot(
        base_bot_name(name), shield_args, seed=seed
    )
  if base_bot_name(name).startswith("liveness_key_teacher"):
    teacher_args = eval_args or model_args
    if teacher_args is None:
      teacher_args = _model_args(3)
    return LivenessKeyTeacherBot(base_bot_name(name), teacher_args, seed=seed)
  if base_bot_name(name).startswith("feasibility_shield"):
    shield_args = eval_args or model_args
    if shield_args is None:
      shield_args = _model_args(3)
    return OwnHandFeasibilityShieldBot(base_bot_name(name), shield_args, seed=seed)
  return make_bot(name, seed=seed)


def play_round(game, bots, seed, match_totals=None, round_index=0):
  if seed is not None:
    np.random.seed(seed)
  state = game.new_initial_state()
  if match_totals is not None:
    state.set_match_context(match_totals, round_index)
  recent_decisions = deque(maxlen=5)
  first_paradox_trace = None
  while not state.is_terminal():
    decision_row = None
    pre_paradox = any(bool(value) for value in getattr(state, "_has_paradoxed", []))
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      action = int(np.random.choice(actions, p=probs))
    else:
      player = state.current_player()
      legal = state.legal_actions(player)
      action = shared_prediction_action(state, player, legal)
      if action is None:
        action = bots[player].step(state, player)
      decision_row = _compact_decision_row(state, player, action, len(legal))
    state.apply_action(action)
    if decision_row is not None:
      after_paradox = [
          bool(value) for value in getattr(state, "_has_paradoxed", [])
      ]
      decision_row["has_paradoxed_after"] = after_paradox
      recent_decisions.append(decision_row)
      if first_paradox_trace is None and not pre_paradox and any(after_paradox):
        first_paradox_trace = {
            "trigger": decision_row,
            "last_decisions": list(recent_decisions),
        }
  state._first_paradox_trace = first_paradox_trace
  return state.returns(), state


def play_match(players, bot_names, bots, seed, initial_start=0, match_context=False):
  totals = np.zeros(players, dtype=np.float32)
  paradoxes = np.zeros(players, dtype=np.int32)
  rounds = []
  for round_idx in range(players):
    start_player = (initial_start + round_idx) % players
    game = pyspiel.load_game(
        "python_quantum_cat",
        {
            "players": players,
            "start_player": start_player,
            "match_context": int(match_context),
        },
    )
    prior_totals = np.copy(totals)
    returns, state = play_round(
        game,
        bots,
        seed=seed + round_idx * 9973,
        match_totals=prior_totals if match_context else None,
        round_index=round_idx,
    )
    returns = np.array(returns, dtype=np.float32)
    if match_context:
      totals = returns
      round_returns = returns - prior_totals
    else:
      totals += returns
      round_returns = returns
    paradoxes += np.array(state._has_paradoxed, dtype=np.int32)
    rounds.append({
        "start_player": start_player,
        "returns": [float(x) for x in round_returns],
        "totals_after_round": [float(x) for x in totals],
        "paradoxed": list(state._has_paradoxed),
        "first_paradox_trace": getattr(state, "_first_paradox_trace", None),
    })
  bot_decision_stats = {}
  for seat, bot in enumerate(bots):
    if hasattr(bot, "raw_decision_stats"):
      stats = bot.raw_decision_stats()
    elif hasattr(bot, "decision_stats"):
      stats = bot.decision_stats()
    else:
      stats = None
    if stats is not None:
      bot_decision_stats[str(seat)] = {
          "bot_name": bot_names[seat],
          "stats": stats,
      }
  return {
      "bot_names": bot_names,
      "totals": [float(x) for x in totals],
      "final_round_returns": (
          [float(x) for x in rounds[-1]["returns"]]
          if rounds else [0.0] * players
      ),
      "paradoxes": [int(x) for x in paradoxes],
      "rounds": rounds,
      "bot_decision_stats": bot_decision_stats,
  }


_ELO_WORKER = {}


def _elo_worker_init(args_dict):
  args = argparse.Namespace(**args_dict)
  torch.set_num_threads(max(1, int(getattr(args, "worker_torch_threads", 1))))
  device = torch.device("cpu")
  model, model_args = load_neural(args.checkpoint, args.players, device, args)
  neural_bots = load_neural_bots(args.neural_bot, args.players, device, args)
  _ELO_WORKER.clear()
  _ELO_WORKER.update({
      "args": args,
      "device": device,
      "model": model,
      "model_args": model_args,
      "neural_bots": neural_bots,
  })


def _unpack_match_job(job):
  if len(job) == 3:
    match_idx, seated, use_match_context = job
    return match_idx, seated, use_match_context, match_idx, match_idx % len(seated)
  match_idx, seated, use_match_context, seed_idx, initial_start = job
  return match_idx, seated, use_match_context, seed_idx, initial_start


def _elo_worker_match(job):
  match_idx, seated, use_match_context, seed_idx, initial_start = (
      _unpack_match_job(job)
  )
  args = _ELO_WORKER["args"]
  device = _ELO_WORKER["device"]
  model = _ELO_WORKER["model"]
  model_args = _ELO_WORKER["model_args"]
  neural_bots = _ELO_WORKER["neural_bots"]
  seed = args.seed + seed_idx * 101
  np.random.seed(seed % (2**32 - 1))
  torch.manual_seed(seed)
  bots = [
      make_eval_bot(
          name,
          args.seed + seed_idx * 101 + seat,
          model,
          model_args,
          device,
          args.belief_samples,
          args.belief_sims,
          args.mcts_sims,
          neural_bots,
          args,
      )
      for seat, name in enumerate(seated)
  ]
  match = play_match(
      args.players,
      seated,
      bots,
      seed=args.seed + 100000 + seed_idx * 37,
      initial_start=initial_start,
      match_context=use_match_context,
  )
  return match_idx, seated, match


def match_jobs(args, names, lineups, use_match_context):
  jobs = []
  paired_deals = int(getattr(args, "paired_deals", 0))
  if paired_deals > 0:
    if args.candidate:
      raise ValueError("--paired-deals is only supported in ladder mode")
    if args.schedule != "permutations":
      raise ValueError("--paired-deals requires --schedule=permutations")
    match_idx = 0
    for deal_idx in range(paired_deals):
      initial_start = deal_idx % args.players
      for lineup in lineups:
        jobs.append((
            match_idx,
            list(lineup),
            use_match_context,
            deal_idx,
            initial_start,
        ))
        match_idx += 1
    return jobs
  for match_idx in range(args.matches):
    if args.candidate:
      seated = candidate_lineup(args, match_idx)
    else:
      lineup = list(lineups[match_idx % len(lineups)])
      if args.schedule == "combinations":
        shift = match_idx % args.players
        seated = lineup[shift:] + lineup[:shift]
      else:
        seated = lineup
    jobs.append((match_idx, seated, use_match_context))
  return jobs


def run_match_jobs(args, jobs, model, model_args, neural_bots, device):
  requested_workers = int(getattr(args, "workers", 1))
  if requested_workers <= 0:
    if len(jobs) < int(getattr(args, "auto_worker_min_games", 32)):
      return run_match_jobs(
          argparse.Namespace(**{**vars(args), "workers": 1}),
          jobs,
          model,
          model_args,
          neural_bots,
          device,
      )
    requested_workers = min(16, max(1, (os.cpu_count() or 2) - 2))
  workers = max(1, min(requested_workers, len(jobs)))
  if workers <= 1:
    results = []
    started = time.perf_counter()
    for job in jobs:
      match_idx, seated, use_match_context, seed_idx, initial_start = (
          _unpack_match_job(job)
      )
      bots = [
          make_eval_bot(
              name,
              args.seed + seed_idx * 101 + seat,
              model,
              model_args,
              device,
              args.belief_samples,
              args.belief_sims,
              args.mcts_sims,
              neural_bots,
              args,
          )
          for seat, name in enumerate(seated)
      ]
      match = play_match(
          args.players,
          seated,
          bots,
          seed=args.seed + 100000 + seed_idx * 37,
          initial_start=initial_start,
          match_context=use_match_context,
      )
      results.append((match_idx, seated, match))
    return results, workers, time.perf_counter() - started

  started = time.perf_counter()
  ctx = mp.get_context("spawn")
  args_dict = vars(args).copy()
  results = []
  with ctx.Pool(
      processes=workers,
      initializer=_elo_worker_init,
      initargs=(args_dict,),
  ) as pool:
    for result in pool.imap_unordered(_elo_worker_match, jobs):
      results.append(result)
  results.sort(key=lambda item: item[0])
  return results, workers, time.perf_counter() - started


def match_outcome_scores(totals, final_round_returns):
  """Numeric match-outcome scores following the rulebook tiebreak.

  Cat in the Box ranks players by total match score. Ties are broken by the
  tied players' final-round scores; if those also tie, the players share the
  result. The returned numbers are only for ordering, not score-margin stats.
  """
  return [
      float(total) * 1000.0 + float(final_round)
      for total, final_round in zip(totals, final_round_returns)
  ]


def candidate_lineup(args, match_idx):
  opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
  if len(opponents) < args.players - 1:
    raise ValueError("Need at least players-1 opponents")
  seated = [args.candidate] + [
      opponents[(match_idx + offset) % len(opponents)]
      for offset in range(args.players - 1)
  ]
  shift = match_idx % args.players
  return seated[-shift:] + seated[:-shift] if shift else seated


def ladder_lineups(args, names):
  if len(names) < args.players:
    raise ValueError("Need at least as many bot names as players")
  if args.schedule == "permutations":
    return list(itertools.permutations(names, args.players))
  return list(itertools.combinations(names, args.players))


def update_pairwise(pairwise, seated, totals, outcome_scores=None):
  if outcome_scores is None:
    outcome_scores = totals
  for i in range(len(seated)):
    for j in range(len(seated)):
      if i == j:
        continue
      name_i = seated[i]
      name_j = seated[j]
      stats = pairwise[name_i][name_j]
      diff = float(totals[i] - totals[j])
      outcome_diff = float(outcome_scores[i] - outcome_scores[j])
      stats["matches"] += 1
      stats["score_diff_sum"] += diff
      stats["score_diff_sq_sum"] += diff * diff
      if outcome_diff > 0:
        stats["wins"] += 1
      elif outcome_diff < 0:
        stats["losses"] += 1
      else:
        stats["ties"] += 1


def summarize_pairwise(pairwise):
  summary = {}
  for name, opponents in pairwise.items():
    summary[name] = {}
    for opponent, stats in opponents.items():
      matches = stats["matches"]
      if matches <= 0:
        continue
      score_rate = (stats["wins"] + 0.5 * stats["ties"]) / matches
      clipped_score_rate = min(0.999, max(0.001, score_rate))
      elo_diff = -400.0 * np.log10(1.0 / clipped_score_rate - 1.0)
      score_rate_se = np.sqrt(score_rate * (1.0 - score_rate) / matches)
      score_rate_ci = (
          max(0.0, score_rate - 1.96 * score_rate_se),
          min(1.0, score_rate + 1.96 * score_rate_se),
      )
      avg_score_diff = stats["score_diff_sum"] / matches
      if matches > 1:
        variance = (
            stats["score_diff_sq_sum"]
            - stats["score_diff_sum"] * stats["score_diff_sum"] / matches
        ) / (matches - 1)
        score_diff_se = np.sqrt(max(0.0, variance) / matches)
      else:
        score_diff_se = 0.0
      score_diff_ci = (
          avg_score_diff - 1.96 * score_diff_se,
          avg_score_diff + 1.96 * score_diff_se,
      )
      summary[name][opponent] = {
          "matches": matches,
          "wins": stats["wins"],
          "losses": stats["losses"],
          "ties": stats["ties"],
          "win_rate": round(stats["wins"] / matches, 4),
          "non_loss_rate": round((stats["wins"] + stats["ties"]) / matches, 4),
          "score_rate": round(score_rate, 4),
          "score_rate_ci95": [round(float(x), 4) for x in score_rate_ci],
          "head_to_head_elo_diff": round(float(elo_diff), 2),
          "avg_score_diff": round(avg_score_diff, 4),
          "avg_score_diff_ci95": [round(float(x), 4) for x in score_diff_ci],
      }
  return summary


def pairwise_elo_ratings(pairwise, names):
  """Batch Elo estimate from aggregate pairwise match outcomes.

  The online Elo stream is useful as a noisy ladder, but its final numbers
  depend on match order. For promotion gates we also report a Bradley-Terry-like
  least-squares rating from aggregate head-to-head scores so direct comparisons
  are stable under schedule permutation.
  """
  equations = []
  targets = []
  weights = []
  for i, name_i in enumerate(names):
    for j in range(i + 1, len(names)):
      name_j = names[j]
      stats = pairwise[name_i][name_j]
      matches = stats["matches"]
      if matches <= 0:
        continue
      score = (stats["wins"] + 0.5 * stats["ties"]) / matches
      score = min(0.999, max(0.001, score))
      rating_diff = -400.0 * np.log10(1.0 / score - 1.0)
      row = np.zeros(len(names), dtype=np.float64)
      row[i] = 1.0
      row[j] = -1.0
      weight = np.sqrt(matches)
      equations.append(row * weight)
      targets.append(rating_diff * weight)
      weights.append(weight)

  if not equations:
    return {name: 1000.0 for name in names}

  constraint = np.ones(len(names), dtype=np.float64)
  equations.append(constraint)
  targets.append(1000.0 * len(names))
  matrix = np.vstack(equations)
  target = np.array(targets, dtype=np.float64)
  solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
  return {name: round(float(solution[idx]), 2) for idx, name in enumerate(names)}


def _homogeneous_neural_spec(entry):
  if entry is None:
    return None
  return {
      "kind": "neural",
      "checkpoint": entry.get("checkpoint"),
      "mode": entry.get("mode"),
      "survival_checkpoint": entry.get("survival_checkpoint", ""),
      "graft_checkpoint": entry.get("graft_checkpoint", ""),
      "graft_builtin_bot": entry.get("graft_builtin_bot", ""),
      "phase_risk_checkpoint": entry.get("phase_risk_checkpoint", ""),
      "base_mode": str(getattr(entry.get("args"), "base_mode", "policy")),
      "overrides": entry.get("overrides", {}),
  }


def _homogeneous_bot_spec(name, neural_bots):
  if name in neural_bots:
    return _homogeneous_neural_spec(neural_bots[name])
  base_name = base_bot_name(name)
  return {
      "kind": "builtin",
      "name": base_name,
  }


def summarize_homogeneous_paradox_gate(
    names,
    players,
    games_by_bot,
    paradoxes_by_bot,
    neural_bots,
    threshold,
    match_results=None,
):
  threshold = float(threshold)
  result = {
      "threshold": threshold,
      "eligible": False,
      "passed": None,
  }
  if threshold <= 0:
    result["reason"] = "disabled"
    return result
  if len(names) != int(players):
    result["reason"] = "requires_exactly_one_alias_per_seat"
    return result
  specs = []
  for name in names:
    specs.append(_homogeneous_bot_spec(name, neural_bots))
  first_spec = specs[0] if specs else None
  if any(spec != first_spec for spec in specs):
    result["reason"] = "alias_specs_differ"
    return result

  seat_match_appearances = int(
      sum(int(games_by_bot.get(name, 0)) for name in names)
  )
  completed_matches = (
      int(seat_match_appearances // int(players)) if int(players) > 0 else 0
  )
  hand_rounds = int(completed_matches * int(players))
  bot_round_opportunities = int(seat_match_appearances * int(players))
  total_paradoxes = int(
      sum(int(paradoxes_by_bot.get(name, 0)) for name in names)
  )
  rounds_with_any_paradox = total_paradoxes
  if match_results is not None:
    observed_hand_rounds = 0
    observed_any_paradox_rounds = 0
    for _match_idx, _seated, match in match_results:
      for round_row in match.get("rounds", []):
        if "paradoxed" not in round_row:
          continue
        observed_hand_rounds += 1
        if any(bool(value) for value in round_row.get("paradoxed", [])):
          observed_any_paradox_rounds += 1
    if observed_hand_rounds > 0:
      hand_rounds = observed_hand_rounds
      rounds_with_any_paradox = observed_any_paradox_rounds
  if bot_round_opportunities <= 0:
    result["reason"] = "no_completed_bot_rounds"
    return result

  bot_round_rate = total_paradoxes / bot_round_opportunities
  bot_round_se = math.sqrt(
      bot_round_rate * (1.0 - bot_round_rate) / bot_round_opportunities
  )
  bot_round_ci95 = [
      max(0.0, bot_round_rate - 1.96 * bot_round_se),
      min(1.0, bot_round_rate + 1.96 * bot_round_se),
  ]
  hand_round_rate = (
      rounds_with_any_paradox / hand_rounds if hand_rounds > 0 else 0.0
  )
  hand_round_se = math.sqrt(
      hand_round_rate * (1.0 - hand_round_rate) / hand_rounds
  ) if hand_rounds > 0 else 0.0
  hand_round_ci95 = [
      max(0.0, hand_round_rate - 1.96 * hand_round_se),
      min(1.0, hand_round_rate + 1.96 * hand_round_se),
  ]
  return {
      "threshold": threshold,
      "eligible": True,
      "passed": bool(hand_round_rate < threshold),
      "reason": None,
      "bot_spec": first_spec,
      "checkpoint": first_spec.get("checkpoint") if first_spec else None,
      "mode": first_spec.get("mode") if first_spec else None,
      "aliases": list(names),
      "completed_matches": completed_matches,
      "seat_match_appearances": seat_match_appearances,
      "hand_rounds": hand_rounds,
      "rounds_with_any_paradox": rounds_with_any_paradox,
      "bot_round_opportunities": bot_round_opportunities,
      "total_paradoxes": total_paradoxes,
      "same_policy_paradox_round_rate": round(hand_round_rate, 4),
      "same_policy_paradox_round_rate_ci95": [
          round(hand_round_ci95[0], 4),
          round(hand_round_ci95[1], 4),
      ],
      "any_paradox_round_rate": round(hand_round_rate, 4),
      "any_paradox_round_rate_ci95": [
          round(hand_round_ci95[0], 4),
          round(hand_round_ci95[1], 4),
      ],
      "paradoxes_per_seat_match": round(
          total_paradoxes / seat_match_appearances, 4
      ),
      "per_seat_paradox_round_rate": round(bot_round_rate, 4),
      "per_seat_paradox_round_rate_ci95": [
          round(bot_round_ci95[0], 4),
          round(bot_round_ci95[1], 4),
      ],
      "paradoxes_per_match": round(
          total_paradoxes / seat_match_appearances, 4
      ),
      "paradoxes_per_round": round(bot_round_rate, 4),
      "paradox_round_rate": round(bot_round_rate, 4),
      "paradox_round_rate_ci95": [
          round(bot_round_ci95[0], 4),
          round(bot_round_ci95[1], 4),
      ],
      "ci95_upper_below_threshold": bool(hand_round_ci95[1] < threshold),
      "per_seat_ci95_upper_below_threshold": bool(bot_round_ci95[1] < threshold),
  }


def summarize_first_paradox_traces(match_results, sample_limit=12):
  summary = {
      "rounds_with_trace": 0,
      "trigger_by_phase": {},
      "trigger_by_action_kind": {},
      "trigger_by_legal_count": {},
      "trigger_by_trick_number": {},
      "trigger_by_led_color": {},
      "forced_triggers": 0,
      "prediction_gap_sum": 0.0,
      "prediction_gap_count": 0,
      "samples": [],
  }
  for _match_idx, seated, match in match_results:
    for round_row in match.get("rounds", []):
      trace = round_row.get("first_paradox_trace")
      if not trace:
        continue
      trigger = trace.get("trigger", {})
      summary["rounds_with_trace"] += 1
      _inc_counter(summary["trigger_by_phase"], trigger.get("phase", "unknown"))
      _inc_counter(
          summary["trigger_by_action_kind"],
          trigger.get("action_kind", "unknown"),
      )
      legal_count = int(trigger.get("legal_count", -1))
      _inc_counter(summary["trigger_by_legal_count"], str(legal_count))
      _inc_counter(
          summary["trigger_by_trick_number"],
          str(int(trigger.get("trick_number", -1))),
      )
      _inc_counter(
          summary["trigger_by_led_color"],
          str(trigger.get("led_color", None)),
      )
      if legal_count <= 1:
        summary["forced_triggers"] += 1
      player = int(trigger.get("player", -1))
      predictions = trigger.get("predictions", [])
      tricks_won = trigger.get("tricks_won", [])
      if (
          0 <= player < len(predictions)
          and 0 <= player < len(tricks_won)
          and int(predictions[player]) >= 0
      ):
        gap = float(predictions[player]) - float(tricks_won[player])
        summary["prediction_gap_sum"] += gap
        summary["prediction_gap_count"] += 1
      if len(summary["samples"]) < int(sample_limit):
        summary["samples"].append({
            "seated": list(seated),
            "start_player": round_row.get("start_player"),
            "trigger": trigger,
            "last_decisions": trace.get("last_decisions", []),
        })
  count = float(summary.get("prediction_gap_count", 0) or 0)
  if count:
    summary["prediction_gap_avg"] = round(
        float(summary["prediction_gap_sum"]) / count,
        4,
    )
  return summary


def main():
  args = parse_args()
  if args.players < 2 or args.players > 5:
    raise ValueError("--players must be 2..5")

  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device=mps requested, but torch MPS is not available")
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model, model_args = load_neural(args.checkpoint, args.players, device, args)
  neural_bots = load_neural_bots(args.neural_bot, args.players, device, args)
  use_match_context = args.match_context or any(
      entry["args"].match_context for entry in neural_bots.values()
  )
  if model_args is not None:
    use_match_context = use_match_context or getattr(model_args, "match_context", False)
  names = (
      [args.candidate] + [name.strip() for name in args.opponents.split(",")
                          if name.strip()]
      if args.candidate else unique_preserving_order(args.bots + list(neural_bots))
  )
  ratings = {name: 1000.0 for name in names}
  totals_by_bot = {name: [] for name in names}
  games_by_bot = {name: 0 for name in names}
  paradoxes_by_bot = {name: 0 for name in names}
  decision_stats_by_bot = {name: {} for name in names}
  pairwise = {
      name: {
          opponent: {
              "matches": 0,
              "wins": 0,
              "losses": 0,
              "ties": 0,
              "score_diff_sum": 0.0,
              "score_diff_sq_sum": 0.0,
          }
          for opponent in names if opponent != name
      }
      for name in names
  }
  sample_matches = []
  lineups = ladder_lineups(args, names) if not args.candidate else None
  jobs = match_jobs(args, names, lineups, use_match_context)
  match_results, workers_used, simulation_elapsed_sec = run_match_jobs(
      args, jobs, model, model_args, neural_bots, device
  )

  for _match_idx, seated, match in match_results:
    outcome_scores = match_outcome_scores(
        match["totals"], match["final_round_returns"]
    )
    multiplayer_elo_update(ratings, seated, outcome_scores, args.k_factor)
    update_pairwise(pairwise, seated, match["totals"], outcome_scores)
    for seat, name in enumerate(seated):
      totals_by_bot[name].append(float(match["totals"][seat]))
      games_by_bot[name] += 1
      paradoxes_by_bot[name] += int(match["paradoxes"][seat])
      seat_stats = match.get("bot_decision_stats", {}).get(str(seat))
      if seat_stats:
        _merge_decision_stats(
            decision_stats_by_bot.setdefault(name, {}),
            seat_stats.get("stats", {}),
        )
    if len(sample_matches) < 3:
      sample_matches.append(match)

  for stats in decision_stats_by_bot.values():
    _decision_stats_rates(stats)

  result = {
      "players": args.players,
      "matches": len(match_results),
      "requested_matches": args.matches,
      "paired_deals": args.paired_deals,
      "rounds_per_match": args.players,
      "match_context": use_match_context,
      "schedule": args.schedule if not args.candidate else "candidate",
      "workers": workers_used,
      "simulation_elapsed_sec": round(simulation_elapsed_sec, 3),
      "checkpoint": args.checkpoint,
      "neural_bots": {
          name: {
              "checkpoint": entry["checkpoint"],
              "mode": entry["mode"],
              **(
                  {"survival_checkpoint": entry["survival_checkpoint"]}
                  if "survival_checkpoint" in entry else {}
              ),
              **(
                  {"graft_checkpoint": entry["graft_checkpoint"]}
                  if "graft_checkpoint" in entry else {}
              ),
              "overrides": entry.get("overrides", {}),
          }
          for name, entry in neural_bots.items()
      },
      "candidate": args.candidate,
      "ratings": {k: round(v, 2) for k, v in sorted(
          ratings.items(), key=lambda item: item[1], reverse=True
      )},
      "online_ratings": {k: round(v, 2) for k, v in sorted(
          ratings.items(), key=lambda item: item[1], reverse=True
      )},
      "pairwise_elo_ratings": {
          k: v for k, v in sorted(
              pairwise_elo_ratings(pairwise, names).items(),
              key=lambda item: item[1],
              reverse=True,
          )
      },
      "stats": {
          name: {
              "matches": games_by_bot[name],
              "avg_match_total": round(mean(totals_by_bot[name]), 4)
              if totals_by_bot[name] else None,
              "paradoxes_per_match": round(
                  paradoxes_by_bot[name] / games_by_bot[name], 4
              ) if games_by_bot[name] else None,
              "paradoxes_per_round": round(
                  paradoxes_by_bot[name] / (games_by_bot[name] * args.players), 4
              ) if games_by_bot[name] else None,
              "paradox_round_rate": round(
                  paradoxes_by_bot[name] / (games_by_bot[name] * args.players), 4
              ) if games_by_bot[name] else None,
          }
          for name in names
      },
      "decision_stats": {
          name: stats
          for name, stats in decision_stats_by_bot.items()
          if stats.get("decisions", 0) > 0
      },
      "first_paradox_summary": summarize_first_paradox_traces(match_results),
      "pairwise": summarize_pairwise(pairwise),
      "sample_matches": sample_matches,
  }
  if float(getattr(args, "homogeneous_paradox_threshold", 0.0)) > 0.0:
    result["homogeneous_paradox_gate"] = summarize_homogeneous_paradox_gate(
        names,
        args.players,
        games_by_bot,
        paradoxes_by_bot,
        neural_bots,
        args.homogeneous_paradox_threshold,
        match_results=match_results,
    )
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
