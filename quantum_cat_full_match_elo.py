#!/usr/bin/env python3
"""Full-match Elo evaluator for Cat in the Box.

A full base-game match has one round per player. The round start player rotates
so every seat starts exactly once, and Elo is updated from cumulative match
scores rather than one-round scores.
"""

from __future__ import annotations

import argparse
from collections import deque
from functools import lru_cache
import hashlib
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
from scipy import stats as scipy_stats
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
import quantum_cat_liveness_oracle as liveness_oracle
from quantum_cat_ai import (
    base_bot_name,
    make_bot,
    multiplayer_elo_update,
    play_game,
    shared_discard_action,
    shared_prediction_action,
)
from quantum_cat_alphazero_torch import (
    AZBeliefPolicyBot,
    AZBeliefSearchBot,
    AZNet,
    AZPolicyBot,
    AZSearchBot,
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    ACTION_FEATURE_PREFIX_SIZE,
    APPENDED_ACTION_FEATURE_INDEX,
    action_feasibility_scores,
    action_feature_matrix,
    load_model_payload,
    model_action_risks,
    model_action_values,
    model_policy_value,
    model_policy_value_action_aux,
    q_policy_select_action,
    sampled_belief_states_for_policy,
    sampled_counterfactual_legal_actions,
    terminal_search_value,
    value_targets_from_scores,
)
from quantum_cat_calibration_transfer_audit import (
    ImproveMLP,
    pair_feature_vector_from_matrix,
)
from quantum_cat_action_set_transformer_trainer import ActionSetTransformer
from quantum_cat_legal_set_selector_trainer import (
    _selector_pair_feature as selector_pair_feature,
)
from quantum_cat_proposal_veto_trainer import (
    ProposalVetoDeepSet,
    ProposalVetoMLP,
    _proposal_feature,
    _slate_relative_fields,
)
from quantum_cat_pairwise_boundary_trainer import PairwiseBoundary
from quantum_cat_intralane_rank_filter_trainer import (
    IntraLaneRankFilter,
    _candidate_allowed as intralane_candidate_allowed,
    _intralane_context_vector,
)
from quantum_cat_table_liveness_improve_trainer import (
    _led_lane_burn_label_shape,
)
from quantum_cat_public_history import (
    PUBLIC_HISTORY_ACTION_SCHEMA,
    public_history_action_features,
    public_history_features,
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


def _shared_discard_decision(stats, state, player, legal):
  action = shared_discard_action(state, player, legal)
  if action is None:
    return None
  _record_decision(stats, state, int(action), len(legal))
  return int(action)


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


def _trim_root_rollout_replay_samples(samples, base_keep_cap=100):
  """Bounds negative replay rows without ever discarding an accepted row."""
  retained = []
  base_keeps = 0
  for sample in samples:
    accepted = bool(
        isinstance(sample, dict) and sample.get("audit_accepted") is True
    )
    if accepted:
      retained.append(sample)
    elif base_keeps < max(0, int(base_keep_cap)):
      retained.append(sample)
      base_keeps += 1
  return retained


def _merge_decision_stats(target, source):
  for key, value in source.items():
    if isinstance(value, dict):
      nested = target.setdefault(key, {})
      _merge_decision_stats(nested, value)
    elif isinstance(value, list):
      target_list = target.setdefault(key, [])
      target_list.extend(value)
      if key == "root_rollout_replay_samples":
        target_list[:] = _trim_root_rollout_replay_samples(target_list)
      else:
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
  action_set_considered = float(
      stats.get("action_set_transformer_considered", 0) or 0
  )
  action_set_overrides = float(
      stats.get("action_set_transformer_overrides", 0) or 0
  )
  action_set_shadow_overrides = float(
      stats.get("action_set_transformer_shadow_overrides", 0) or 0
  )
  action_set_scored = float(
      stats.get("action_set_transformer_scored_candidates", 0) or 0
  )
  if action_set_considered:
    stats["action_set_transformer_override_rate_when_considered"] = round(
        action_set_overrides / action_set_considered, 4
    )
    stats["action_set_transformer_shadow_override_rate_when_considered"] = round(
        action_set_shadow_overrides / action_set_considered, 4
    )
    stats["action_set_transformer_abstain_rate_when_considered"] = round(
        float(stats.get("action_set_transformer_abstained", 0))
        / action_set_considered,
        4,
    )
    stats["action_set_transformer_scored_candidates_avg"] = round(
        action_set_scored / action_set_considered,
        4,
    )
    stats["action_set_transformer_best_margin_avg"] = round(
        float(stats.get("action_set_transformer_best_margin_sum", 0.0))
        / action_set_considered,
        6,
    )
    stats["action_set_transformer_base_score_avg"] = round(
        float(stats.get("action_set_transformer_base_score_sum", 0.0))
        / action_set_considered,
        6,
    )
    stats["action_set_transformer_selected_score_avg"] = round(
        float(stats.get("action_set_transformer_selected_score_sum", 0.0))
        / action_set_considered,
        6,
    )
    stats["action_set_transformer_selected_value_delta_avg"] = round(
        float(stats.get("action_set_transformer_selected_value_delta_sum", 0.0))
        / action_set_considered,
        6,
    )
    stats["action_set_transformer_selected_positive_prob_avg"] = round(
        float(
            stats.get("action_set_transformer_selected_positive_prob_sum", 0.0)
        )
        / action_set_considered,
        6,
      )
  proposal_considered = float(stats.get("proposal_veto_considered", 0) or 0)
  proposal_overrides = float(stats.get("proposal_veto_overrides", 0) or 0)
  proposal_shadow_overrides = float(
      stats.get("proposal_veto_shadow_overrides", 0) or 0
  )
  proposal_scored = float(stats.get("proposal_veto_scored_candidates", 0) or 0)
  if proposal_considered:
    stats["proposal_veto_override_rate_when_considered"] = round(
        proposal_overrides / proposal_considered, 4
    )
    stats["proposal_veto_shadow_override_rate_when_considered"] = round(
        proposal_shadow_overrides / proposal_considered, 4
    )
    stats["proposal_veto_abstain_rate_when_considered"] = round(
        float(stats.get("proposal_veto_abstained", 0)) / proposal_considered,
        4,
    )
    stats["proposal_veto_scored_candidates_avg"] = round(
        proposal_scored / proposal_considered,
        4,
    )
    stats["proposal_veto_selected_logit_avg"] = round(
        float(stats.get("proposal_veto_selected_logit_sum", 0.0))
        / proposal_considered,
        6,
    )
    stats["proposal_veto_selected_source_margin_avg"] = round(
        float(stats.get("proposal_veto_selected_source_margin_sum", 0.0))
        / proposal_considered,
        6,
    )
    stats["proposal_veto_selected_value_delta_avg"] = round(
        float(stats.get("proposal_veto_selected_value_delta_sum", 0.0))
        / proposal_considered,
        6,
    )
    stats["proposal_veto_source_margin_blocked_rate_when_considered"] = round(
        float(stats.get("proposal_veto_blocked_source_margin_threshold", 0))
        / proposal_considered,
        4,
    )
    stats["proposal_veto_source_value_blocked_rate_when_considered"] = round(
        float(stats.get("proposal_veto_blocked_source_value_threshold", 0))
        / proposal_considered,
        4,
    )
    stats["proposal_veto_source_positive_blocked_rate_when_considered"] = round(
        float(stats.get("proposal_veto_blocked_source_positive_threshold", 0))
        / proposal_considered,
        4,
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
          "uses every ordered seating; --schedule=cyclic uses each distinct "
          "seat rotation once and avoids pseudoreplicating identical opponents."
      ),
  )
  parser.add_argument(
      "--deal-offset",
      type=int,
      default=0,
      help=(
          "First global common-deal index. Use successive non-overlapping "
          "offsets to extend a frozen paired-deal evaluation without replaying "
          "previous chance seeds or starting-seat rotations."
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
          "name=checkpoint:competitive_tactics, "
          "name=checkpoint:residual_policy, "
          "name=checkpoint:residual_q_policy, "
          "name=checkpoint:residual_q_risk_policy, name=checkpoint:play_graft, "
          "name=checkpoint:improve_graft, "
          "name=checkpoint:pairwise_boundary, "
          "name=checkpoint:action_set_transformer, or name=checkpoint:belief. "
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
      "--root-rollout-teacher-candidates",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Fill candidate slots with the same follow-led, red-switch, and "
          "off-led contrast buckets used by best-of-rollouts replay."
      ),
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
      "--root-rollout-led-colors",
      default="",
      help=(
          "Optional comma-separated led-color filter for :root_rollout. "
          "Use R/G/B/Y or None for leading states; empty disables this filter."
      ),
  )
  parser.add_argument(
      "--root-rollout-min-trick-number",
      type=int,
      default=-1,
      help="Optional first trick where :root_rollout may override policy.",
  )
  parser.add_argument(
      "--root-rollout-max-trick-number",
      type=int,
      default=-1,
      help="Optional last trick where :root_rollout may override policy.",
  )
  parser.add_argument(
      "--root-rollout-min-match-round",
      type=int,
      default=-1,
      help="Optional first public match round eligible for root search.",
  )
  parser.add_argument(
      "--root-rollout-max-match-round",
      type=int,
      default=-1,
      help="Optional last public match round eligible for root search.",
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
      "--root-rollout-min-paradox-improvement",
      type=float,
      default=0.0,
      help=(
          "For paradox_then_score, keep the raw-policy action unless the "
          "selected rollout action lowers paired paradox rate by at least "
          "this amount."
      ),
  )
  parser.add_argument(
      "--root-rollout-audit-samples",
      type=int,
      default=0,
      help=(
          "Fresh actor-information-state particles used to independently "
          "audit a proposed :root_rollout override. 0 preserves the legacy "
          "single-pass selector."
      ),
  )
  parser.add_argument(
      "--root-rollout-audit-rollouts",
      type=int,
      default=1,
      help="Common rollout seeds per fresh audit particle.",
  )
  parser.add_argument(
      "--root-rollout-audit-passes",
      type=int,
      default=1,
      help=(
          "Independent fresh audit passes required before a :root_rollout "
          "override is applied."
      ),
  )
  parser.add_argument(
      "--root-rollout-min-audit-paradox-improvement",
      type=float,
      default=0.0,
      help=(
          "Keep the raw-policy action unless the proposed override also "
          "lowers paired paradox rate by this amount on fresh audit worlds."
      ),
  )
  parser.add_argument(
      "--root-rollout-min-audit-score-improvement",
      type=float,
      default=-1e9,
      help=(
          "Optional minimum actor score gain required on every independently "
          "resampled audit before a root-search override is accepted."
      ),
  )
  parser.add_argument(
      "--root-rollout-max-audit-paired-binomial-p",
      type=float,
      default=1.0,
      help=(
          "Maximum one-sided exact-binomial p-value for saved versus harmed "
          "paired information-state worlds in every fresh audit pass. "
          "Multiple continuations from one world are nested replicates and "
          "count as one inference unit."
      ),
  )
  parser.add_argument(
      "--first-paradox-trace-window",
      type=int,
      default=5,
      help=(
          "Number of recent decisions stored before the first paradox trigger. "
          "This is diagnostics-only and does not affect game play."
      ),
  )
  parser.add_argument(
      "--first-paradox-summary-sample-limit",
      type=int,
      default=12,
      help="Maximum first-paradox trace samples retained in the summary.",
  )
  parser.add_argument(
      "--sample-match-limit",
      type=int,
      default=3,
      help="Maximum full match records retained in the output artifact.",
  )
  parser.add_argument(
      "--root-rollout-base-mode",
      choices=("policy", "liveness_shield", "competitive_tactics"),
      default="policy",
      help=(
          "Root action protected by :root_rollout. policy preserves the raw "
          "neural baseline; liveness_shield starts from the deterministic "
          "public-information shield; competitive_tactics additionally "
          "preserves locked actor-relative bidding and tactical decisions. "
          "Only independently audited rollout overrides are applied on top."
      ),
  )
  parser.add_argument(
      "--root-rollout-preserve-tactical-overrides",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "When root search protects a competitive_tactics baseline, preserve "
          "any exact-terminal or bid-regime correction that already changed "
          "the underlying neural action. Search remains enabled on all other "
          "information-state decisions."
      ),
  )
  parser.add_argument(
      "--root-rollout-preserve-own-liveness",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Before actor-information-state search, reject candidate moves "
          "that worsen the incumbent's own-hand exit margin, feasibility, "
          "color-token safety, or exact-bid protection."
      ),
  )
  parser.add_argument(
      "--root-rollout-max-baseline-risk-increase",
      type=float,
      default=-1.0,
      help=(
          "Maximum calibrated acting-player paradox-probability increase "
          "allowed versus the protected incumbent action. A negative value "
          "disables this information-safe prefilter."
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
      choices=("", "policy", "q_policy", "liveness_shield", "competitive_tactics"),
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
      "--liveness-teacher-exact-key-order",
      choices=(
          "lane_first",
          "pressure_first",
          "late_red_pressure_first",
          "table_liveness_first",
          "late_table_liveness_first",
      ),
      default="lane_first",
      help=(
          "Tie-break order used by liveness_key_teacher when exact legal "
          "pressure is enabled. lane_first preserves historical behavior; "
          "pressure_first ranks exact legal-exit capacity before generic "
          "led-token preservation; late_red_pressure_first applies that "
          "pressure-first order only from trick 5 onward when led color is "
          "red or no color has been led yet. table_liveness_first and "
          "late_table_liveness_first additionally rank by exact hidden-hand "
          "table liveness margins for training-teacher diagnostics."
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
      "--liveness-shield-exact-legal-pressure",
      type=_parse_bool,
      default=False,
      help=(
          "When true, :liveness_shield prepends exact post-action legal-exit "
          "pressure to its action key and dominance test. This is a full-state "
          "diagnostic/oracle-style shield."
      ),
  )
  parser.add_argument(
      "--liveness-shield-exact-forced-only",
      type=_parse_bool,
      default=False,
      help=(
          "When true with --liveness-shield-exact-legal-pressure, allow "
          ":liveness_shield overrides only when the candidate immediately "
          "reduces the exact number of post-action forced-paradox players."
      ),
  )
  parser.add_argument(
      "--liveness-shield-keep-responsible-red-win",
      type=_parse_bool,
      default=False,
      help=(
          "When true, :liveness_shield keeps a base action that spends the "
          "led-color token to win a still-needed trick instead of overriding "
          "it only for public led-token preservation. Exact forced-paradox "
          "improvements may still override first."
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
          "Optional state-value checkpoint used by :value_shield or :belief "
          "neural bots. Empty uses the policy checkpoint itself."
      ),
  )
  parser.add_argument(
      "--survival-value-base-mode",
      choices=("policy", "liveness_shield", "belief", "belief_liveness"),
      default="policy",
      help=(
          "Policy whose decisions the one-step survival value shield protects. "
          "belief_liveness uses information-state belief search for the "
          "opening trick window and the public-liveness policy afterward."
      ),
  )
  parser.add_argument("--survival-value-threshold", type=float, default=0.55)
  parser.add_argument("--survival-value-max-actions", type=int, default=0)
  parser.add_argument(
      "--survival-value-selection-mode",
      choices=("threshold", "relative_dominance"),
      default="threshold",
      help=(
          "threshold preserves historical value-shield selection. "
          "relative_dominance keeps the protected base action unless a "
          "common-world alternative clears every configured safety guard."
      ),
  )
  parser.add_argument(
      "--survival-value-min-improvement",
      type=float,
      default=0.0,
      help="Minimum paired predicted survival improvement required to override.",
  )
  parser.add_argument(
      "--survival-value-paired-lcb-z",
      type=float,
      default=0.0,
      help=(
          "Subtract this many paired-world standard errors from predicted "
          "survival improvement before applying the dominance margin."
      ),
  )
  parser.add_argument(
      "--survival-value-min-world-win-share",
      type=float,
      default=0.0,
      help=(
          "Minimum fraction of common information-state worlds on which "
          "a relative-dominance candidate improves on the protected action."
      ),
  )
  parser.add_argument(
      "--survival-value-min-trick-number",
      type=int,
      default=-1,
      help="Earliest play trick on which learned-survival overrides are enabled.",
  )
  parser.add_argument(
      "--survival-value-max-trick-number",
      type=int,
      default=-1,
      help="Latest play trick on which learned-survival overrides are enabled.",
  )
  parser.add_argument(
      "--survival-value-opening-max-trick-number",
      type=int,
      default=1,
      help=(
          "Last trick where the belief_liveness base uses belief search "
          "before handing decisions back to the public-liveness policy."
      ),
  )
  parser.add_argument(
      "--survival-value-opening-min-trick-number",
      type=int,
      default=0,
      help="Earliest trick where the belief_liveness base may use belief search.",
  )
  parser.add_argument(
      "--survival-value-opening-lead-only",
      type=_parse_bool,
      default=False,
      help=(
          "Restrict expensive opening belief search to the information-safe "
          "public states where the current player is leading the trick."
      ),
  )
  parser.add_argument(
      "--survival-value-require-public-safety",
      type=_parse_bool,
      default=False,
      help=(
          "Reject learned-survival alternatives that worsen information-safe "
          "public open slots, public slot damage, or the actor's known exits."
      ),
  )
  parser.add_argument(
      "--survival-value-analytic-posterior",
      type=_parse_bool,
      default=False,
      help=(
          "Apply a guarded opponent-rank-capacity posterior using only the "
          "acting hand/discard and public board, token, and trick information."
      ),
  )
  parser.add_argument(
      "--survival-value-analytic-min-trick-number",
      type=int,
      default=2,
      help="Earliest trick where the analytic public-posterior guard may fire.",
  )
  parser.add_argument(
      "--survival-value-analytic-objective",
      choices=("table_survival", "competitive"),
      default="table_survival",
      help=(
          "Use the historical table-survival posterior or the competitive "
          "posterior that pressures opponents only after the actor's bid is "
          "complete while preserving the actor's own exits."
      ),
  )
  parser.add_argument(
      "--survival-value-analytic-min-advantage",
      type=float,
      default=0.04,
      help=(
          "Minimum reduction in summed opponent rank-overflow probability "
          "required by the guarded analytic public posterior."
      ),
  )
  parser.add_argument(
      "--survival-value-search-penalty",
      type=float,
      default=0.0,
      help=(
          "Subtract this many score units times the predicted table-paradox "
          "probability at every nonterminal information-state search leaf."
      ),
  )
  parser.add_argument(
      "--survival-value-belief-samples",
      type=int,
      default=4,
      help=(
          "Information-state opponent-hand samples shared across one-step "
          "survival-value candidate actions. Never inspect the actual next "
          "player's private hand."
      ),
  )
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
      choices=("additive", "threshold", "relative", "relative_dominance"),
      default="additive",
      help=(
          "How :q_policy uses the action paradox-risk head. additive preserves "
          "the historical log(policy)-penalty*risk score. threshold only "
          "switches away from the raw top policy when risk is high enough and "
          "a lower-risk action is within the configured policy log-gap. "
          "relative first builds a lowest-risk safe set using "
          "--action-paradox-min-risk-margin as slack, then picks the best "
          "policy/value action inside that set. relative_dominance does the "
          "same, but abstains unless the raw top policy action is at least "
          "--action-paradox-min-risk-margin riskier than the minimum-risk "
          "legal action."
      ),
  )
  parser.add_argument("--action-paradox-risk-threshold", type=float, default=0.0)
  parser.add_argument("--action-paradox-min-risk-margin", type=float, default=0.0)
  parser.add_argument(
      "--action-paradox-rerank-led-contrast-only",
      action="store_true",
      help=(
          "For :q_policy risk reranking, only allow overrides in play states "
          "with a non-red led color, a legal led-color follow, and a legal red "
          "switch. Useful for led-contrast-trained risk heads."
      ),
  )
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
  parser.add_argument(
      "--action-value-max-policy-log-gap",
      type=float,
      default=-1.0,
      help=(
          "For :q_policy action-value reranking, reject actions whose raw "
          "policy log probability is more than this far below the raw top "
          "policy action. Negative preserves the historical unguarded "
          "behavior."
      ),
  )
  parser.add_argument(
      "--action-value-rerank-min-trick-number",
      type=int,
      default=-1,
      help=(
          "For play-phase :q_policy action-value reranking, disable reranking "
          "before this trick number. Negative preserves historical behavior."
      ),
  )
  parser.add_argument(
      "--action-value-rerank-max-trick-number",
      type=int,
      default=-1,
      help=(
          "For play-phase :q_policy action-value reranking, disable reranking "
          "after this trick number. Negative preserves historical behavior."
      ),
  )
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument(
      "--schedule",
      choices=("combinations", "permutations", "cyclic"),
      default="combinations",
      help=(
          "Ladder seating schedule. permutations balances every ordered seating; "
          "cyclic rotates each selected lineup once per seat and is preferred "
          "when opponent aliases represent the same policy."
      ),
  )
  parser.add_argument("--k-factor", type=float, default=24.0)
  parser.add_argument(
      "--competitive-gate-candidate",
      default="",
      help=(
          "Named candidate for an official-score, common-deal, seat-balanced "
          "competitive promotion summary. Requires --paired-deals and a "
          "seat-balanced permutations or cyclic schedule."
      ),
  )
  parser.add_argument(
      "--competitive-gate-opponents",
      default="",
      help=(
          "Optional comma-separated opponent aliases used as the competitive "
          "reference. Empty compares the candidate against every other seat."
      ),
  )
  parser.add_argument(
      "--competitive-min-independent-deals",
      type=int,
      default=30,
      help=(
          "Minimum independent common-deal blocks for a competitive "
          "promotion pass. Seat permutations are not independent samples."
      ),
  )
  parser.add_argument(
      "--competitive-max-own-paradox-regression",
      type=float,
      default=0.0,
      help=(
          "Allowed candidate-minus-reference personal paradox-rate increase. "
          "The guard fails only when the independent-deal 95%% lower bound "
          "exceeds this value. Table-wide paradoxes never veto promotion."
      ),
  )
  parser.add_argument(
      "--homogeneous-paradox-threshold",
      type=float,
      default=0.0,
      help=(
          "When >0, add the legacy homogeneous self-play diagnostic summary. It "
          "is not a competitive promotion criterion. The diagnostic "
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
      neural_prediction_mode="shared",
      arch="mlp",
      width=256,
      depth=3,
      value_scale=20.0,
      c_puct=1.8,
      sims=64,
      match_context=False,
      paradox_value_penalty=0.0,
      terminal_paradox_penalty=0.0,
      terminal_any_paradox_penalty=0.0,
      ordinal_value_weight=0.0,
      official_outcome_value_weight=0.0,
      action_value_selection_weight=0.0,
      action_value_root_only=False,
      action_paradox_selection_penalty=0.0,
      action_paradox_root_only=False,
      action_paradox_rerank_mode="additive",
      action_paradox_risk_threshold=0.0,
      action_paradox_min_risk_margin=0.0,
      action_paradox_rerank_led_contrast_only=False,
      action_feasibility_selection_weight=0.0,
      action_paradox_max_policy_log_gap=2.0,
      action_value_rerank_clip=0.5,
      action_value_rerank_phases="",
      action_value_rerank_min_margin=0.0,
      action_value_max_policy_log_gap=-1.0,
      action_value_rerank_min_trick_number=-1,
      action_value_rerank_max_trick_number=-1,
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
      belief_samples=0,
      belief_sims=0,
      root_rollout_samples=2,
      root_rollouts=1,
      root_rollout_seed_salt=0,
      root_rollout_max_actions=4,
      root_rollout_top_policy=2,
      root_rollout_include_bots="heuristic,heuristic_target2,heuristic_adj2",
      root_rollout_max_plies=0,
      root_rollout_phases="",
      root_rollout_led_colors="",
      root_rollout_min_match_round=-1,
      root_rollout_max_match_round=-1,
      root_rollout_full_match=False,
      root_rollout_objective="score",
      root_rollout_paradox_scope="acting",
      root_rollout_min_paradox_improvement=0.0,
      root_rollout_audit_samples=0,
      root_rollout_audit_rollouts=1,
      root_rollout_audit_passes=1,
      root_rollout_min_audit_paradox_improvement=0.0,
      root_rollout_min_audit_score_improvement=-1e9,
      root_rollout_max_audit_paired_binomial_p=1.0,
      root_rollout_base_mode="policy",
      root_rollout_continuation_bot="",
      root_rollout_continuation_mode="",
      root_rollout_include_continuation_candidate=True,
      root_rollout_record_replay_samples=False,
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
      liveness_teacher_exact_key_order="lane_first",
      liveness_teacher_sample_limit=20,
      liveness_shield_base_mode="q_policy",
      liveness_shield_phases="discard,prediction,play",
      liveness_shield_min_trick_number=-1,
      liveness_shield_led_colors="",
      liveness_shield_min_open_slot_delta=1,
      liveness_shield_min_public_damage_delta=2,
      liveness_shield_exact_legal_pressure=False,
      liveness_shield_exact_forced_only=False,
      liveness_shield_keep_responsible_red_win=False,
      liveness_shield_max_policy_log_gap=-1.0,
      liveness_shield_shadow_only=False,
      liveness_shield_sample_limit=20,
      residual_anchor_checkpoint="",
      residual_policy_phases="discard,prediction,play",
      residual_delta_clip=1.0,
      residual_delta_scale=1.0,
      residual_q_policy_base_margin=0.0,
      survival_checkpoint="",
      survival_value_base_mode="policy",
      survival_value_threshold=0.55,
      survival_value_max_actions=0,
      survival_value_selection_mode="threshold",
      survival_value_min_improvement=0.0,
      survival_value_paired_lcb_z=0.0,
      survival_value_min_world_win_share=0.0,
      survival_value_min_trick_number=-1,
      survival_value_max_trick_number=-1,
      survival_value_opening_min_trick_number=0,
      survival_value_opening_max_trick_number=1,
      survival_value_opening_lead_only=False,
      survival_value_require_public_safety=False,
      survival_value_analytic_posterior=False,
      survival_value_analytic_objective="table_survival",
      survival_value_analytic_min_trick_number=2,
      survival_value_analytic_min_advantage=0.04,
      survival_value_search_penalty=0.0,
      survival_value_belief_samples=4,
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
      improve_base_mode="policy",
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
      improve_trick_numbers="",
      improve_led_colors="",
      improve_candidate_colors="",
      improve_require_base_off_led=False,
      improve_require_responsible_red_win=False,
      improve_min_match_round=-1,
      improve_near_paradox_proxy=False,
      improve_near_paradox_min_pressure=0.1,
      improve_guard_led_follow=False,
      improve_guard_led_follow_red_win_must_be_urgent=False,
      improve_max_policy_log_gap=-1.0,
      improve_shadow_only=False,
      improve_sample_limit=20,
      intralane_filter_model="",
      intralane_filter_base_mode="policy",
      intralane_filter_phases="play",
      intralane_filter_min_trick_number=2,
      intralane_filter_max_trick_number=5,
      intralane_filter_candidate_filter="follow_no_token_loss",
      intralane_filter_require_base_allowed=True,
      intralane_filter_require_forced_switch_red_contrast=False,
      intralane_filter_positive_threshold=0.50,
      intralane_filter_safe_threshold=0.85,
      intralane_filter_value_threshold=0.08,
      intralane_filter_risk_threshold=-0.01,
      intralane_filter_selection_value_bonus=0.05,
      intralane_filter_selection_risk_penalty=0.5,
      intralane_filter_policy_tiebreak=0.01,
      intralane_filter_max_policy_log_gap=-1.0,
      intralane_filter_shadow_only=False,
      intralane_filter_sample_limit=20,
      pairwise_boundary_model="",
      pairwise_boundary_base_mode="policy",
      pairwise_boundary_threshold=0.0,
      pairwise_boundary_phases="play",
      pairwise_boundary_min_legal_count=0,
      pairwise_boundary_max_legal_count=0,
      pairwise_boundary_min_trick_number=-1,
      pairwise_boundary_max_trick_number=-1,
      pairwise_boundary_max_policy_log_gap=-1.0,
      pairwise_boundary_max_actions=0,
      pairwise_boundary_top_policy=0,
      pairwise_boundary_include_bots=(
          "heuristic,heuristic_target2,heuristic_adj2"
      ),
      pairwise_boundary_feature_candidates=True,
      pairwise_boundary_exclude_forced_led_lane_negatives=True,
      pairwise_boundary_led_lane_red_exception="win",
      pairwise_boundary_shadow_only=False,
      pairwise_boundary_sample_limit=20,
      action_set_transformer_model="",
      action_set_transformer_base_mode="policy",
      action_set_transformer_phases="play",
      action_set_transformer_min_legal_count=0,
      action_set_transformer_max_legal_count=0,
      action_set_transformer_min_trick_number=-1,
      action_set_transformer_max_trick_number=-1,
      action_set_transformer_score_margin_threshold=0.0,
      action_set_transformer_value_threshold=0.0,
      action_set_transformer_positive_threshold=0.5,
      action_set_transformer_max_policy_log_gap=-1.0,
      action_set_transformer_max_actions=4,
      action_set_transformer_top_policy=2,
      action_set_transformer_include_bots=(
          "heuristic,heuristic_target2,heuristic_adj2"
      ),
      action_set_transformer_feature_candidates=True,
      action_set_transformer_exclude_forced_led_lane_negatives=True,
      action_set_transformer_shadow_only=False,
      action_set_transformer_selection_value_bonus=0.05,
      action_set_transformer_policy_tiebreak=0.01,
      action_set_transformer_sample_limit=20,
      proposal_veto_model="",
      proposal_veto_base_mode="policy",
      proposal_veto_phases="play",
      proposal_veto_min_legal_count=0,
      proposal_veto_max_legal_count=0,
      proposal_veto_min_trick_number=-1,
      proposal_veto_max_trick_number=-1,
      proposal_veto_max_policy_log_gap=-1.0,
      proposal_veto_max_actions=6,
      proposal_veto_top_policy=4,
      proposal_veto_include_bots=(
          "heuristic,heuristic_target2,heuristic_adj2"
      ),
      proposal_veto_candidate_mode="source_plus_red_follow_policy",
      proposal_veto_selection_mode="proposal_model",
      proposal_veto_feature_candidates=True,
      proposal_veto_exclude_forced_led_lane_negatives=True,
      proposal_veto_score_threshold=-1e9,
      proposal_veto_margin_threshold=-1e9,
      proposal_veto_min_source_score=-1e9,
      proposal_veto_min_source_margin=-1e9,
      proposal_veto_min_source_value_delta=-1e9,
      proposal_veto_min_source_positive_prob=-1e9,
      proposal_veto_shadow_only=False,
      proposal_veto_sample_limit=20,
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
  model_args.official_outcome_value_weight = float(
      saved_args.get("official_outcome_value_weight", 0.0)
  )
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
    model_args.action_paradox_rerank_led_contrast_only = bool(
        getattr(args, "action_paradox_rerank_led_contrast_only", False)
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
    model_args.action_value_max_policy_log_gap = float(
        getattr(args, "action_value_max_policy_log_gap", -1.0)
    )
    model_args.action_value_rerank_min_trick_number = int(
        getattr(args, "action_value_rerank_min_trick_number", -1)
    )
    model_args.action_value_rerank_max_trick_number = int(
        getattr(args, "action_value_rerank_max_trick_number", -1)
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
    model_args.root_rollout_teacher_candidates = bool(
        getattr(args, "root_rollout_teacher_candidates", False)
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
    model_args.root_rollout_led_colors = str(
        getattr(args, "root_rollout_led_colors", "")
    )
    model_args.root_rollout_min_trick_number = int(
        getattr(args, "root_rollout_min_trick_number", -1)
    )
    model_args.root_rollout_max_trick_number = int(
        getattr(args, "root_rollout_max_trick_number", -1)
    )
    model_args.root_rollout_min_match_round = int(
        getattr(args, "root_rollout_min_match_round", -1)
    )
    model_args.root_rollout_max_match_round = int(
        getattr(args, "root_rollout_max_match_round", -1)
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
    model_args.root_rollout_min_paradox_improvement = float(
        getattr(args, "root_rollout_min_paradox_improvement", 0.0)
    )
    model_args.root_rollout_audit_samples = int(
        getattr(args, "root_rollout_audit_samples", 0)
    )
    model_args.root_rollout_audit_rollouts = int(
        getattr(args, "root_rollout_audit_rollouts", 1)
    )
    model_args.root_rollout_audit_passes = int(
        getattr(args, "root_rollout_audit_passes", 1)
    )
    model_args.root_rollout_min_audit_paradox_improvement = float(
        getattr(args, "root_rollout_min_audit_paradox_improvement", 0.0)
    )
    model_args.root_rollout_min_audit_score_improvement = float(
        getattr(args, "root_rollout_min_audit_score_improvement", -1e9)
    )
    model_args.root_rollout_max_audit_paired_binomial_p = float(
        getattr(args, "root_rollout_max_audit_paired_binomial_p", 1.0)
    )
    model_args.root_rollout_base_mode = str(
        getattr(args, "root_rollout_base_mode", "policy")
    )
    model_args.root_rollout_preserve_tactical_overrides = bool(
        getattr(args, "root_rollout_preserve_tactical_overrides", False)
    )
    model_args.root_rollout_preserve_own_liveness = bool(
        getattr(args, "root_rollout_preserve_own_liveness", False)
    )
    model_args.root_rollout_max_baseline_risk_increase = float(
        getattr(args, "root_rollout_max_baseline_risk_increase", -1.0)
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
    model_args.liveness_shield_exact_legal_pressure = bool(
        getattr(args, "liveness_shield_exact_legal_pressure", False)
    )
    model_args.liveness_shield_exact_forced_only = bool(
        getattr(args, "liveness_shield_exact_forced_only", False)
    )
    model_args.liveness_shield_keep_responsible_red_win = bool(
        getattr(args, "liveness_shield_keep_responsible_red_win", False)
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
    model_args.survival_value_base_mode = str(
        getattr(args, "survival_value_base_mode", "policy")
    )
    model_args.survival_value_threshold = float(
        getattr(args, "survival_value_threshold", 0.55)
    )
    model_args.survival_value_max_actions = int(
        getattr(args, "survival_value_max_actions", 0)
    )
    model_args.survival_value_selection_mode = str(
        getattr(args, "survival_value_selection_mode", "threshold")
    )
    model_args.survival_value_min_improvement = float(
        getattr(args, "survival_value_min_improvement", 0.0)
    )
    model_args.survival_value_paired_lcb_z = float(
        getattr(args, "survival_value_paired_lcb_z", 0.0)
    )
    model_args.survival_value_min_world_win_share = float(
        getattr(args, "survival_value_min_world_win_share", 0.0)
    )
    model_args.survival_value_min_trick_number = int(
        getattr(args, "survival_value_min_trick_number", -1)
    )
    model_args.survival_value_max_trick_number = int(
        getattr(args, "survival_value_max_trick_number", -1)
    )
    model_args.survival_value_opening_max_trick_number = int(
        getattr(args, "survival_value_opening_max_trick_number", 1)
    )
    model_args.survival_value_opening_min_trick_number = int(
        getattr(args, "survival_value_opening_min_trick_number", 0)
    )
    model_args.survival_value_opening_lead_only = bool(
        getattr(args, "survival_value_opening_lead_only", False)
    )
    model_args.survival_value_require_public_safety = bool(
        getattr(args, "survival_value_require_public_safety", False)
    )
    model_args.survival_value_analytic_posterior = bool(
        getattr(args, "survival_value_analytic_posterior", False)
    )
    model_args.survival_value_analytic_objective = str(
        getattr(args, "survival_value_analytic_objective", "table_survival")
    )
    model_args.survival_value_analytic_min_trick_number = int(
        getattr(args, "survival_value_analytic_min_trick_number", 2)
    )
    model_args.survival_value_analytic_min_advantage = float(
        getattr(args, "survival_value_analytic_min_advantage", 0.04)
    )
    model_args.survival_value_search_penalty = float(
        getattr(args, "survival_value_search_penalty", 0.0)
    )
    model_args.survival_value_belief_samples = int(
        getattr(args, "survival_value_belief_samples", 4)
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
    "neural_prediction_mode": str,
    "competitive_tactics_base_mode": str,
    "competitive_tactics_exact_terminal": bool,
    "competitive_tactics_round_terminal": bool,
    "competitive_tactics_bid_play": bool,
    "competitive_tactics_bid_regime": bool,
    "competitive_tactics_lead_only": bool,
    "competitive_tactics_min_trick_number": int,
    "competitive_tactics_max_trick_number": int,
    "competitive_tactics_min_prediction": int,
    "competitive_tactics_min_needed_tricks": int,
    "competitive_tactics_min_lead_rank_gain": int,
    "competitive_tactics_adapt_negative_opponent": bool,
    "competitive_tactics_prediction_style": str,
    "competitive_tactics_min_own_margin": int,
    "paradox_value_penalty": float,
    "terminal_paradox_penalty": float,
    "terminal_any_paradox_penalty": float,
    "ordinal_value_weight": float,
    "official_outcome_value_weight": float,
    "action_value_selection_weight": float,
    "action_value_root_only": bool,
    "action_paradox_selection_penalty": float,
    "action_paradox_root_only": bool,
    "action_paradox_rerank_mode": str,
    "action_paradox_risk_threshold": float,
    "action_paradox_min_risk_margin": float,
    "action_paradox_rerank_led_contrast_only": bool,
    "action_paradox_phase_risk_checkpoint": str,
    "action_paradox_phase_risk_phases": str,
    "action_feasibility_selection_weight": float,
    "action_paradox_max_policy_log_gap": float,
    "action_value_rerank_clip": float,
    "action_value_rerank_phases": str,
    "action_value_rerank_min_margin": float,
    "action_value_max_policy_log_gap": float,
    "action_value_rerank_min_trick_number": int,
    "action_value_rerank_max_trick_number": int,
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
    "belief_samples": int,
    "belief_sims": int,
    "root_rollout_samples": int,
    "root_rollouts": int,
    "root_rollout_seed_salt": int,
    "root_rollout_max_actions": int,
    "root_rollout_top_policy": int,
    "root_rollout_teacher_candidates": bool,
    "root_rollout_include_bots": str,
    "root_rollout_max_plies": int,
    "root_rollout_phases": str,
    "root_rollout_led_colors": str,
    "root_rollout_min_trick_number": int,
    "root_rollout_max_trick_number": int,
    "root_rollout_min_match_round": int,
    "root_rollout_max_match_round": int,
    "root_rollout_full_match": bool,
    "root_rollout_objective": str,
    "root_rollout_paradox_scope": str,
    "root_rollout_min_paradox_improvement": float,
    "root_rollout_audit_samples": int,
    "root_rollout_audit_rollouts": int,
    "root_rollout_audit_passes": int,
    "root_rollout_min_audit_paradox_improvement": float,
    "root_rollout_min_audit_score_improvement": float,
    "root_rollout_max_audit_paired_binomial_p": float,
    "root_rollout_base_mode": str,
    "root_rollout_preserve_tactical_overrides": bool,
    "root_rollout_preserve_own_liveness": bool,
    "root_rollout_max_baseline_risk_increase": float,
    "root_rollout_continuation_bot": str,
    "root_rollout_continuation_mode": str,
    "root_rollout_include_continuation_candidate": bool,
    "root_rollout_record_replay_samples": bool,
    "liveness_shield_base_mode": str,
    "liveness_shield_phases": str,
    "liveness_shield_min_trick_number": int,
    "liveness_shield_led_colors": str,
    "liveness_shield_min_open_slot_delta": int,
    "liveness_shield_min_public_damage_delta": int,
    "liveness_shield_exact_legal_pressure": bool,
    "liveness_shield_exact_forced_only": bool,
    "liveness_shield_keep_responsible_red_win": bool,
    "liveness_shield_max_policy_log_gap": float,
    "liveness_shield_shadow_only": bool,
    "liveness_shield_sample_limit": int,
    "belief_table_liveness_base_mode": str,
    "belief_table_liveness_samples": int,
    "belief_table_liveness_min_valid_samples": int,
    "belief_table_liveness_seed_salt": int,
    "belief_table_liveness_phases": str,
    "belief_table_liveness_min_trick_number": int,
    "belief_table_liveness_max_trick_number": int,
    "belief_table_liveness_aggregation": str,
    "belief_table_liveness_quantile": float,
    "belief_table_liveness_min_paired_lcb": float,
    "belief_table_liveness_paired_lcb_z": float,
    "belief_table_liveness_min_best_share": float,
    "belief_table_liveness_max_policy_log_gap": float,
    "belief_table_liveness_shadow_only": bool,
    "belief_table_liveness_sample_limit": int,
    "residual_anchor_checkpoint": str,
    "residual_policy_phases": str,
    "residual_delta_clip": float,
    "residual_delta_scale": float,
    "residual_q_policy_base_margin": float,
    "survival_checkpoint": str,
    "survival_value_base_mode": str,
    "survival_value_threshold": float,
    "survival_value_max_actions": int,
    "survival_value_selection_mode": str,
    "survival_value_min_improvement": float,
    "survival_value_paired_lcb_z": float,
    "survival_value_min_world_win_share": float,
    "survival_value_min_trick_number": int,
    "survival_value_max_trick_number": int,
    "survival_value_opening_min_trick_number": int,
    "survival_value_opening_max_trick_number": int,
    "survival_value_opening_lead_only": bool,
    "survival_value_require_public_safety": bool,
    "survival_value_analytic_posterior": bool,
    "survival_value_analytic_objective": str,
    "survival_value_analytic_min_trick_number": int,
    "survival_value_analytic_min_advantage": float,
    "survival_value_search_penalty": float,
    "survival_value_belief_samples": int,
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
    "improve_base_mode": str,
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
    "improve_trick_numbers": str,
    "improve_led_colors": str,
    "improve_candidate_colors": str,
    "improve_require_base_off_led": bool,
    "improve_require_responsible_red_win": bool,
    "improve_min_match_round": int,
    "improve_near_paradox_proxy": bool,
    "improve_near_paradox_min_pressure": float,
    "improve_guard_led_follow": bool,
    "improve_guard_led_follow_red_win_must_be_urgent": bool,
    "improve_max_policy_log_gap": float,
    "improve_shadow_only": bool,
    "improve_sample_limit": int,
    "intralane_filter_model": str,
    "intralane_filter_base_mode": str,
    "intralane_filter_phases": str,
    "intralane_filter_min_trick_number": int,
    "intralane_filter_max_trick_number": int,
    "intralane_filter_candidate_filter": str,
    "intralane_filter_require_base_allowed": bool,
    "intralane_filter_require_forced_switch_red_contrast": bool,
    "intralane_filter_positive_threshold": float,
    "intralane_filter_safe_threshold": float,
    "intralane_filter_value_threshold": float,
    "intralane_filter_risk_threshold": float,
    "intralane_filter_selection_value_bonus": float,
    "intralane_filter_selection_risk_penalty": float,
    "intralane_filter_policy_tiebreak": float,
    "intralane_filter_max_policy_log_gap": float,
    "intralane_filter_shadow_only": bool,
    "intralane_filter_sample_limit": int,
    "pairwise_boundary_model": str,
    "pairwise_boundary_base_mode": str,
    "pairwise_boundary_threshold": float,
    "pairwise_boundary_phases": str,
    "pairwise_boundary_min_legal_count": int,
    "pairwise_boundary_max_legal_count": int,
    "pairwise_boundary_min_trick_number": int,
    "pairwise_boundary_max_trick_number": int,
    "pairwise_boundary_max_policy_log_gap": float,
    "pairwise_boundary_max_actions": int,
    "pairwise_boundary_top_policy": int,
    "pairwise_boundary_include_bots": str,
    "pairwise_boundary_feature_candidates": bool,
    "pairwise_boundary_exclude_forced_led_lane_negatives": bool,
    "pairwise_boundary_led_lane_red_exception": str,
    "pairwise_boundary_shadow_only": bool,
    "pairwise_boundary_sample_limit": int,
    "action_set_transformer_model": str,
    "action_set_transformer_base_mode": str,
    "action_set_transformer_phases": str,
    "action_set_transformer_min_legal_count": int,
    "action_set_transformer_max_legal_count": int,
    "action_set_transformer_min_trick_number": int,
    "action_set_transformer_max_trick_number": int,
    "action_set_transformer_score_margin_threshold": float,
    "action_set_transformer_value_threshold": float,
    "action_set_transformer_positive_threshold": float,
    "action_set_transformer_max_policy_log_gap": float,
    "action_set_transformer_max_actions": int,
    "action_set_transformer_top_policy": int,
    "action_set_transformer_include_bots": str,
    "action_set_transformer_feature_candidates": bool,
    "action_set_transformer_exclude_forced_led_lane_negatives": bool,
    "action_set_transformer_shadow_only": bool,
    "action_set_transformer_selection_value_bonus": float,
    "action_set_transformer_policy_tiebreak": float,
    "action_set_transformer_sample_limit": int,
    "proposal_veto_model": str,
    "proposal_veto_base_mode": str,
    "proposal_veto_phases": str,
    "proposal_veto_min_legal_count": int,
    "proposal_veto_max_legal_count": int,
    "proposal_veto_min_trick_number": int,
    "proposal_veto_max_trick_number": int,
    "proposal_veto_max_policy_log_gap": float,
    "proposal_veto_max_actions": int,
    "proposal_veto_top_policy": int,
    "proposal_veto_include_bots": str,
    "proposal_veto_candidate_mode": str,
    "proposal_veto_selection_mode": str,
    "proposal_veto_feature_candidates": bool,
    "proposal_veto_exclude_forced_led_lane_negatives": bool,
    "proposal_veto_score_threshold": float,
    "proposal_veto_margin_threshold": float,
    "proposal_veto_min_source_score": float,
    "proposal_veto_min_source_margin": float,
    "proposal_veto_min_source_value_delta": float,
    "proposal_veto_min_source_positive_prob": float,
    "proposal_veto_shadow_only": bool,
    "proposal_veto_sample_limit": int,
    "oracle_min_margin": int,
    "oracle_apply_discard": bool,
    "liveness_shield_oracle_min_margin": int,
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


def _uses_neural_prediction(model_args):
  mode = str(
      getattr(model_args, "neural_prediction_mode", "shared") or "shared"
  ).strip().lower().replace("-", "_")
  if mode not in ("shared", "model"):
    raise ValueError(
        "neural_prediction_mode must be 'shared' or 'model', "
        f"got {mode!r}"
    )
  return mode == "model"


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
    if payload.endswith(":competitive_tactics"):
      checkpoint = payload[:-len(":competitive_tactics")]
      mode = "competitive_tactics"
    elif payload.endswith(":competitive-tactics"):
      checkpoint = payload[:-len(":competitive-tactics")]
      mode = "competitive_tactics"
    elif payload.endswith(":belief_policy"):
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
    elif payload.endswith(":belief_table_liveness"):
      checkpoint = payload[:-len(":belief_table_liveness")]
      mode = "belief_table_liveness"
    elif payload.endswith(":belief-table-liveness"):
      checkpoint = payload[:-len(":belief-table-liveness")]
      mode = "belief_table_liveness"
    elif payload.endswith(":oracle_shield"):
      checkpoint = payload[:-len(":oracle_shield")]
      mode = "oracle_shield"
    elif payload.endswith(":oracle-shield"):
      checkpoint = payload[:-len(":oracle-shield")]
      mode = "oracle_shield"
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
    elif payload.endswith(":intralane_filter"):
      checkpoint = payload[:-len(":intralane_filter")]
      mode = "intralane_filter"
    elif payload.endswith(":intralane-filter"):
      checkpoint = payload[:-len(":intralane-filter")]
      mode = "intralane_filter"
    elif payload.endswith(":pairwise_boundary"):
      checkpoint = payload[:-len(":pairwise_boundary")]
      mode = "pairwise_boundary"
    elif payload.endswith(":pairwise-boundary"):
      checkpoint = payload[:-len(":pairwise-boundary")]
      mode = "pairwise_boundary"
    elif payload.endswith(":action_set_transformer"):
      checkpoint = payload[:-len(":action_set_transformer")]
      mode = "action_set_transformer"
    elif payload.endswith(":action-set-transformer"):
      checkpoint = payload[:-len(":action-set-transformer")]
      mode = "action_set_transformer"
    elif payload.endswith(":action_set"):
      checkpoint = payload[:-len(":action_set")]
      mode = "action_set_transformer"
    elif payload.endswith(":action-set"):
      checkpoint = payload[:-len(":action-set")]
      mode = "action_set_transformer"
    elif payload.endswith(":proposal_veto_top1"):
      checkpoint = payload[:-len(":proposal_veto_top1")]
      mode = "proposal_veto_top1"
    elif payload.endswith(":proposal-veto-top1"):
      checkpoint = payload[:-len(":proposal-veto-top1")]
      mode = "proposal_veto_top1"
    elif payload.endswith(":proposal_veto"):
      checkpoint = payload[:-len(":proposal_veto")]
      mode = "proposal_veto_top1"
    elif payload.endswith(":proposal-veto"):
      checkpoint = payload[:-len(":proposal-veto")]
      mode = "proposal_veto_top1"
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
    if mode == "intralane_filter":
      filter_model = str(getattr(model_args, "intralane_filter_model", "") or "")
      if not filter_model:
        raise ValueError(
          f"--neural-bot {name}=...:intralane_filter requires "
          "intralane_filter_model=PATH"
        )
      entry["intralane_filter"] = load_intralane_filter(filter_model, device)
      entry["intralane_filter_model"] = filter_model
    if mode == "pairwise_boundary":
      boundary_model = str(
          getattr(model_args, "pairwise_boundary_model", "") or ""
      )
      if not boundary_model:
        raise ValueError(
            f"--neural-bot {name}=...:pairwise_boundary requires "
            "pairwise_boundary_model=PATH"
        )
      entry["pairwise_boundary"] = load_pairwise_boundary(boundary_model, device)
      entry["pairwise_boundary_model"] = boundary_model
    if mode == "action_set_transformer":
      selector_model = str(
          getattr(model_args, "action_set_transformer_model", "") or ""
      )
      if not selector_model:
        raise ValueError(
            f"--neural-bot {name}=...:action_set_transformer requires "
            "action_set_transformer_model=PATH"
        )
      entry["action_set_transformer"] = load_action_set_transformer(
          selector_model, device
      )
      entry["action_set_transformer_model"] = selector_model
    if mode == "proposal_veto_top1":
      selector_model = str(
          getattr(model_args, "action_set_transformer_model", "") or ""
      )
      if not selector_model:
        raise ValueError(
            f"--neural-bot {name}=...:proposal_veto_top1 requires "
            "action_set_transformer_model=PATH"
        )
      proposal_model = str(getattr(model_args, "proposal_veto_model", "") or "")
      if not proposal_model:
        raise ValueError(
            f"--neural-bot {name}=...:proposal_veto_top1 requires "
            "proposal_veto_model=PATH"
        )
      entry["action_set_transformer"] = load_action_set_transformer(
          selector_model, device
      )
      entry["action_set_transformer_model"] = selector_model
      entry["proposal_veto"] = load_proposal_veto_selector(
          proposal_model, device
      )
      entry["proposal_veto_model"] = proposal_model
    if mode == "value_shield" or (
        mode == "belief"
        and float(getattr(model_args, "survival_value_search_penalty", 0.0)) > 0
    ):
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
      uses_survival_search = mode == "belief" or (
          mode == "value_shield"
          and str(getattr(model_args, "survival_value_base_mode", "policy"))
          in ("belief", "belief_liveness")
          and float(getattr(model_args, "survival_value_search_penalty", 0.0)) > 0
      )
      if uses_survival_search:
        model_args._search_survival_model = survival_model
        model_args._survival_search_counters = {
            "nonterminal_evaluations": 0,
        }
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
    self.uses_neural_prediction = _uses_neural_prediction(model_args)
    self.phase_risk_model = phase_risk_model
    self.phase_risk_phases = _split_phase_names(phase_risk_phases)
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "rerank_disabled": 0,
        "rerank_disabled_led_contrast": 0,
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
        "rerank_relative_blocked_risk_margin": 0,
        "rerank_by_phase": {},
        "rerank_overrides_by_phase": {},
    })

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  def _rerank_enabled_for_phase(self, state):
    spec = str(getattr(self.args, "action_value_rerank_phases", "") or "")
    if not spec.strip():
      phase_allowed = True
    else:
      allowed = {part.strip().lower() for part in spec.split(",") if part.strip()}
      phase_allowed = self._phase_name(state) in allowed
    if not phase_allowed:
      return False
    if self._phase_name(state) != "play":
      return True
    trick_number = int(getattr(state, "_trick_number", 0))
    min_trick = int(getattr(
        self.args, "action_value_rerank_min_trick_number", -1
    ))
    max_trick = int(getattr(
        self.args, "action_value_rerank_max_trick_number", -1
    ))
    if min_trick >= 0 and trick_number < min_trick:
      return False
    if max_trick >= 0 and trick_number > max_trick:
      return False
    return True

  def _risk_model_for_phase(self, state):
    if self.phase_risk_model is None or not self.phase_risk_phases:
      return self.model
    if self._phase_name(state) in self.phase_risk_phases:
      return self.phase_risk_model
    return self.model

  @staticmethod
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

  def _led_contrast_rerank_enabled(self, state, legal):
    if not bool(
        getattr(self.args, "action_paradox_rerank_led_contrast_only", False)
    ):
      return True
    if self._phase_name(state) != "play":
      return False
    led_color = getattr(state, "_led_color", None)
    if led_color not in ("B", "Y", "G"):
      return False
    led_legal = any(self._action_color(state, action) == led_color
                    for action in legal)
    red_legal = any(self._action_color(state, action) == "R" for action in legal)
    return bool(led_legal and red_legal)

  def step(self, state, player):
    legal = state.legal_actions(player)
    legal_count = len(legal)
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
    if len(legal) == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return legal[0]
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
    risk_needed = (
        risk_penalty_weight > 0
        or risk_mode in ("threshold", "relative", "relative_dominance")
    )
    risk_model = self._risk_model_for_phase(state)
    can_share_risk_forward = risk_model is self.model
    policy, _, action_values, action_risks, action_features = (
        model_policy_value_action_aux(
            self.model,
            state,
            player,
            state.num_distinct_actions(),
            self.args.value_scale,
            self.device,
            need_action_values=weight > 0,
            need_action_risks=risk_needed and can_share_risk_forward,
            return_action_features=feasibility_weight != 0.0,
            use_shared_prediction=not self.uses_neural_prediction,
        )
    )
    baseline_action = max(legal, key=lambda action: policy[action])
    if (
        self._phase_name(state) == "prediction"
        and self.uses_neural_prediction
        and getattr(self.model, "separate_prediction_policy_head", False)
    ):
      # Q/risk heads model card-play outcomes; the dedicated bidding head is
      # trained directly on actor-safe full-match bid interventions instead.
      _record_decision(self._decision_stats, state, baseline_action, legal_count)
      return baseline_action
    if (
        (
            weight <= 0
            and risk_penalty_weight <= 0
            and feasibility_weight == 0.0
            and risk_mode not in ("threshold", "relative", "relative_dominance")
        )
        or not self._rerank_enabled_for_phase(state)
    ):
      self._decision_stats["rerank_disabled"] += 1
      _record_decision(
          self._decision_stats, state, baseline_action, legal_count
      )
      return baseline_action
    if not self._led_contrast_rerank_enabled(state, legal):
      self._decision_stats["rerank_disabled"] += 1
      self._decision_stats["rerank_disabled_led_contrast"] += 1
      _record_decision(
          self._decision_stats, state, baseline_action, legal_count
      )
      return baseline_action
    phase = self._phase_name(state)
    self._decision_stats["rerank_considered"] += 1
    _inc_counter(self._decision_stats["rerank_by_phase"], phase)
    if action_values is None:
      action_values = np.zeros(state.num_distinct_actions(), dtype=np.float32)
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
    elif risk_mode in ("threshold", "relative", "relative_dominance"):
      self._decision_stats["rerank_risk_used"] += 1
    if risk_needed and action_risks is None:
      action_risks = model_action_risks(
          risk_model,
          state,
          player,
          state.num_distinct_actions(),
          self.device,
      )
    feasibility_scores = (
        action_feasibility_scores(action_features)
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
        and risk_mode not in ("threshold", "relative", "relative_dominance")
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
        if risk_penalty_weight > 0
        or risk_mode in ("threshold", "relative", "relative_dominance") else
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


class AZOracleShieldPolicyBot:
  """Neural policy guarded by the exact own-hand liveness oracle.

  Play phase: keeps the policy's preferred action unless its post-action
  adversarial liveness margin (quantum_cat_liveness_oracle) drops below
  `oracle_min_margin` while a safer legal alternative exists. Provably dead
  moves are always vetoed when an alive move exists.

  Discard phase (optional, default on): prefers the shared discard heuristic
  when it is margin-safe, otherwise the margin-maximizing discard.

  Prediction phase: shared prediction heuristic, like the other policy bots.
  """

  def __init__(self, model, name, device, model_args):
    self.name = name
    self.model = model
    self.device = device
    self.args = model_args
    self.min_margin = int(getattr(model_args, "oracle_min_margin", 1))
    self.apply_discard = bool(getattr(model_args, "oracle_apply_discard", True))
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "oracle_shield_considered": 0,
        "oracle_shield_base_kept": 0,
        "oracle_shield_overrides": 0,
        "oracle_shield_discard_overrides": 0,
        "oracle_shield_min_margin": self.min_margin,
    })

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

  def step(self, state, player):
    legal = state.legal_actions(player)
    if not legal:
      return None
    phase = int(getattr(state, "_phase", -1))
    if phase == 2:
      action = shared_prediction_action(state, player, legal)
      if action is not None:
        return int(action)
      return int(legal[0])
    if phase == 1:
      shared = shared_discard_action(state, player, legal)
      if not self.apply_discard:
        return int(shared) if shared is not None else int(legal[0])
      view = liveness_oracle.discard_view_from_state(state, player)
      margins = {
          int(a): liveness_oracle.post_discard_margin(view, int(a))
          for a in legal
      }
      best = max(margins.values())
      threshold = min(self.min_margin, best)
      if shared is not None and margins[int(shared)] >= threshold:
        return int(shared)
      self._decision_stats["oracle_shield_discard_overrides"] += 1
      return max(legal, key=lambda a: margins[int(a)])
    if phase != 3:
      return int(legal[0])
    if legal == [999]:
      return 999
    policy = self._policy(state, player)
    self._decision_stats["oracle_shield_considered"] += 1
    base = max(
        (a for a in legal if a != 999),
        key=lambda a: float(policy[a]),
    )
    choice = liveness_oracle.select_action_with_oracle(
        state, player, policy,
        min_margin=self.min_margin,
        apply_to_discard=False,
    )
    if choice is None:
      choice = base
    if int(choice) == int(base):
      self._decision_stats["oracle_shield_base_kept"] += 1
    else:
      self._decision_stats["oracle_shield_overrides"] += 1
    return int(choice)


class AZLivenessShieldPolicyBot:
  """Baseline-protected neural policy with deterministic liveness guards."""

  def __init__(self, model, name, device, model_args):
    self.name = name
    self.model = model
    self.device = device
    self.args = model_args
    self.uses_neural_prediction = _uses_neural_prediction(model_args)
    base_mode = str(getattr(model_args, "liveness_shield_base_mode", "q_policy"))
    if base_mode == "q_policy":
      self.base_bot = AZQPolicyBot(model, f"{name}_base", device, model_args)
    elif base_mode == "policy":
      self.base_bot = AZPolicyBot(
          model,
          f"{name}_base",
          device,
          model_args.value_scale,
          use_shared_prediction=not self.uses_neural_prediction,
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
        "liveness_shield_blocked_oracle_margin": 0,
        "liveness_shield_base_responsible_red_win": 0,
        "liveness_shield_candidate_responsible_red_win": 0,
        "liveness_shield_candidate_leaves_responsible_red_win": 0,
        "liveness_shield_responsible_red_win_guard_candidates": 0,
        "liveness_shield_no_scores": 0,
        "liveness_shield_base_failure_sum": 0.0,
        "liveness_shield_selected_failure_sum": 0.0,
        "liveness_shield_base_public_damage_sum": 0.0,
        "liveness_shield_selected_public_damage_sum": 0.0,
        "liveness_shield_base_min_open_slots_sum": 0.0,
        "liveness_shield_selected_min_open_slots_sum": 0.0,
        "liveness_shield_base_exact_forced_sum": 0.0,
        "liveness_shield_selected_exact_forced_sum": 0.0,
        "liveness_shield_base_exact_min_legal_sum": 0.0,
        "liveness_shield_selected_exact_min_legal_sum": 0.0,
        "liveness_shield_base_exact_current_legal_sum": 0.0,
        "liveness_shield_selected_exact_current_legal_sum": 0.0,
        "liveness_shield_base_exact_total_legal_sum": 0.0,
        "liveness_shield_selected_exact_total_legal_sum": 0.0,
        "liveness_shield_by_phase": {},
        "liveness_shield_overrides_by_phase": {},
        "liveness_shield_base_mode": base_mode,
        "liveness_shield_exact_legal_pressure": bool(
            getattr(model_args, "liveness_shield_exact_legal_pressure", False)
        ),
        "liveness_shield_exact_forced_only": bool(
            getattr(model_args, "liveness_shield_exact_forced_only", False)
        ),
        "liveness_shield_keep_responsible_red_win": bool(
            getattr(
                model_args, "liveness_shield_keep_responsible_red_win", False
            )
        ),
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
        use_shared_prediction=not self.uses_neural_prediction,
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
    use_exact = bool(
        getattr(self.args, "liveness_shield_exact_legal_pressure", False)
    )
    scored = {}
    for action in legal:
      row = action_liveness_certificate(
          state, player, action, base_action=base_action
      )
      if use_exact:
        row.update(exact_legal_pressure_after_action(state, action))
      scored[int(action)] = row
    return scored

  def _candidate_key(self, item, base_action):
    action, row = item
    liveness_key = action_liveness_key(row, action, base_action=base_action)
    if not bool(getattr(self.args, "liveness_shield_exact_legal_pressure", False)):
      return liveness_key
    return (
        1 if bool(row.get("exact_terminal_after", False)) else 0,
        -int(row.get("exact_forced_paradox_players_after", 0)),
        0 if bool(row.get("lost_led_token", False)) else 1,
        int(row.get("exact_min_nonparadox_legal_after", 0)),
        int(row.get("exact_current_nonparadox_legal_after", 0)),
        int(row.get("exact_total_nonparadox_legal_after", 0)),
    ) + liveness_key

  @staticmethod
  def _responsible_token_loss_win(row):
    if "responsible_red_win_switch" in row:
      return bool(row.get("responsible_red_win_switch", False))
    return (
        bool(row.get("lost_led_token", False))
        and bool(row.get("would_win_now", False))
        and not bool(row.get("over_target_would_win", False))
    )

  def _responsible_red_win_guard_blocks(self, base_row, candidate_row):
    return (
        bool(
            getattr(
                self.args, "liveness_shield_keep_responsible_red_win", False
            )
        )
        and self._responsible_token_loss_win(base_row)
        and not self._responsible_token_loss_win(candidate_row)
    )

  def _dominates_base(self, base_row, candidate_row):
    if bool(candidate_row.get("is_paradox", False)):
      return False
    if bool(base_row.get("is_paradox", False)):
      return True
    if bool(getattr(self.args, "liveness_shield_exact_legal_pressure", False)):
      base_forced = int(base_row.get("exact_forced_paradox_players_after", 0))
      candidate_forced = int(
          candidate_row.get("exact_forced_paradox_players_after", 0)
      )
      if candidate_forced < base_forced:
        return True
      if bool(getattr(self.args, "liveness_shield_exact_forced_only", False)):
        return False
      if candidate_forced > base_forced:
        return False
      base_min_legal = int(base_row.get("exact_min_nonparadox_legal_after", 0))
      candidate_min_legal = int(
          candidate_row.get("exact_min_nonparadox_legal_after", 0)
      )
      base_current_legal = int(
          base_row.get("exact_current_nonparadox_legal_after", 0)
      )
      candidate_current_legal = int(
          candidate_row.get("exact_current_nonparadox_legal_after", 0)
      )
      base_total_legal = int(
          base_row.get("exact_total_nonparadox_legal_after", 0)
      )
      candidate_total_legal = int(
          candidate_row.get("exact_total_nonparadox_legal_after", 0)
      )
      min_open_delta = int(
          getattr(self.args, "liveness_shield_min_open_slot_delta", 1)
      )
      exact_not_worse = (
          candidate_min_legal >= base_min_legal
          and candidate_current_legal >= base_current_legal
          and candidate_total_legal >= base_total_legal
      )
      if (
          candidate_min_legal >= base_min_legal + min_open_delta
          and candidate_total_legal >= base_total_legal
      ):
        return True
      if (
          candidate_current_legal >= base_current_legal + min_open_delta
          and exact_not_worse
      ):
        return True
    if self._responsible_red_win_guard_blocks(base_row, candidate_row):
      return False
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
    if (
        bool(base_row.get("lost_led_token", False))
        and not bool(candidate_row.get("lost_led_token", False))
    ):
      if (
          bool(base_row.get("own_feasible", False))
          and not bool(candidate_row.get("own_feasible", False))
      ):
        return False
      return True
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
      if bool(getattr(self.args, "liveness_shield_exact_legal_pressure", False)):
        base_row.update(exact_legal_pressure_after_action(state, base_action))
    selected_action, selected_row = max(
        scored.items(), key=lambda item: self._candidate_key(item, base_action)
    )
    selected_action = int(selected_action)
    base_responsible = self._responsible_token_loss_win(base_row)
    candidate_responsible = self._responsible_token_loss_win(selected_row)
    if base_responsible:
      self._decision_stats["liveness_shield_base_responsible_red_win"] += 1
    if candidate_responsible:
      self._decision_stats["liveness_shield_candidate_responsible_red_win"] += 1
    if selected_action != int(base_action) and base_responsible:
      if not candidate_responsible:
        self._decision_stats[
            "liveness_shield_candidate_leaves_responsible_red_win"
        ] += 1
      if self._responsible_red_win_guard_blocks(base_row, selected_row):
        self._decision_stats[
            "liveness_shield_responsible_red_win_guard_candidates"
        ] += 1
    should_override = (
        selected_action != int(base_action)
        and self._dominates_base(base_row, selected_row)
    )
    if should_override and not self._within_policy_gap(
        policy, base_action, selected_action
    ):
      self._decision_stats["liveness_shield_blocked_policy_gap"] += 1
      should_override = False
    oracle_gate = int(
        getattr(self.args, "liveness_shield_oracle_min_margin", -1)
    )
    if (
        should_override
        and oracle_gate >= 0
        and int(getattr(state, "_phase", -1)) == 3
        and int(base_action) != 999
    ):
      view = liveness_oracle.view_from_state(state, player)
      base_margin = liveness_oracle.post_play_margin(view, int(base_action))
      if base_margin >= oracle_gate:
        self._decision_stats["liveness_shield_blocked_oracle_margin"] += 1
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
    self._decision_stats["liveness_shield_base_exact_forced_sum"] += float(
        base_row.get("exact_forced_paradox_players_after", 0)
    )
    self._decision_stats["liveness_shield_selected_exact_forced_sum"] += float(
        final_row.get("exact_forced_paradox_players_after", 0)
    )
    self._decision_stats["liveness_shield_base_exact_min_legal_sum"] += float(
        base_row.get("exact_min_nonparadox_legal_after", 0)
    )
    self._decision_stats["liveness_shield_selected_exact_min_legal_sum"] += float(
        final_row.get("exact_min_nonparadox_legal_after", 0)
    )
    self._decision_stats["liveness_shield_base_exact_current_legal_sum"] += float(
        base_row.get("exact_current_nonparadox_legal_after", 0)
    )
    self._decision_stats[
        "liveness_shield_selected_exact_current_legal_sum"
    ] += float(final_row.get("exact_current_nonparadox_legal_after", 0))
    self._decision_stats["liveness_shield_base_exact_total_legal_sum"] += float(
        base_row.get("exact_total_nonparadox_legal_after", 0)
    )
    self._decision_stats["liveness_shield_selected_exact_total_legal_sum"] += float(
        final_row.get("exact_total_nonparadox_legal_after", 0)
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
      for key in (
          "exact_forced",
          "exact_min_legal",
          "exact_current_legal",
          "exact_total_legal",
      ):
        base_sum_key = f"liveness_shield_base_{key}_sum"
        selected_sum_key = f"liveness_shield_selected_{key}_sum"
        stats[f"liveness_shield_base_{key}_avg"] = round(
            float(stats.get(base_sum_key, 0.0)) / considered,
            6,
        )
        stats[f"liveness_shield_selected_{key}_avg"] = round(
            float(stats.get(selected_sum_key, 0.0)) / considered,
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


@lru_cache(maxsize=32768)
def _public_rank_count_probabilities(population, rank_copies, hand_size):
  """Exact one-rank hypergeometric marginal from visible card conservation."""
  if not 0 <= rank_copies <= population or not 0 <= hand_size <= population:
    raise ValueError((population, rank_copies, hand_size))
  denominator = math.comb(population, hand_size)
  return tuple(
      (
          count,
          math.comb(rank_copies, count)
          * math.comb(population - rank_copies, hand_size - count)
          / denominator,
      )
      for count in range(
          max(0, hand_size - population + rank_copies),
          min(rank_copies, hand_size) + 1,
      )
  )


def _public_unknown_rank_counts(state, observer):
  """Count unseen ranks without touching opponent hands or hidden discards."""
  counts = np.full(int(state._num_card_types), 5, dtype=np.int32)
  counts -= np.asarray(state._hands[int(observer)], dtype=np.int32)
  own_discard = int(state._discarded_cards[int(observer)])
  if own_discard > 0:
    counts[own_discard - 1] -= 1
  board = np.asarray(state._board_ownership)
  counts -= np.sum(board >= 0, axis=0, dtype=np.int32)
  counts -= np.sum(board == -2, axis=0, dtype=np.int32)
  if np.any(counts < 0):
    raise ValueError(f"Contradictory public rank counts: {counts.tolist()}")
  return counts


def _public_hand_size(state, player):
  played = int(state._count_cards_played_by(int(player)))
  discarded = int(bool(state._has_discarded[int(player)]))
  return int(state._cards_per_player_initial) - discarded - played


def _public_opponent_capacity_posterior(state, player, unknown_counts):
  """Approximate table capacity from exact public one-rank marginals."""
  population = int(np.sum(unknown_counts))
  hand_size = _public_hand_size(state, player)
  remaining_plays = max(0, hand_size - 1)
  tokens = np.asarray(state._color_tokens[int(player)], dtype=bool)
  board = np.asarray(state._board_ownership)
  capacities = np.sum((board == -1) & tokens[:, None], axis=0)
  expected_matching = 0.0
  overflow_probabilities = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  for rank, remaining in enumerate(unknown_counts):
    capacity = int(capacities[rank])
    rank_overflow = np.zeros(3, dtype=np.float64)
    for count, probability in _public_rank_count_probabilities(
        population, int(remaining), hand_size
    ):
      expected_matching += min(count, capacity) * probability
      rank_overflow[min(2, max(0, count - capacity))] += probability
    combined = np.zeros(3, dtype=np.float64)
    for previous in range(3):
      for additional in range(3):
        combined[min(2, previous + additional)] += (
            overflow_probabilities[previous] * rank_overflow[additional]
        )
    overflow_probabilities = combined
  return {
      "seat": int(player),
      "expected_surplus": float(expected_matching - remaining_plays),
      "approx_failure_probability": float(overflow_probabilities[2]),
  }


def public_posterior_action_score(state, observer, action):
  """Score an action using only the acting hand and public game information."""
  candidate = state.clone()
  candidate.apply_action(int(action))
  any_paradox = any(bool(flag) for flag in candidate._has_paradoxed)
  if candidate.is_terminal():
    marker = -100.0 if any_paradox else 100.0
    return {
        "any_paradox": any_paradox,
        "min_expected_surplus": marker,
        "total_expected_surplus": marker,
        "sum_failure_probability": 4.0 if any_paradox else 0.0,
        "max_failure_probability": 1.0 if any_paradox else 0.0,
        "next_expected_surplus": marker,
        "own_margin": marker,
    }
  unknown_counts = _public_unknown_rank_counts(candidate, observer)
  opponent_metrics = [
      _public_opponent_capacity_posterior(candidate, seat, unknown_counts)
      for seat in range(int(candidate._num_players))
      if seat != int(observer) and _public_hand_size(candidate, seat) > 1
  ]
  own_view = liveness_oracle.view_from_state(candidate, observer)
  own_surplus = float(own_view.matching() - own_view.plays_remaining)
  surpluses = [own_surplus] + [
      row["expected_surplus"] for row in opponent_metrics
  ]
  next_row = next(
      (
          row for row in opponent_metrics
          if row["seat"] == int(candidate.current_player())
      ),
      None,
  )
  return {
      "any_paradox": any_paradox,
      "min_expected_surplus": float(min(surpluses)),
      "total_expected_surplus": float(sum(surpluses)),
      "sum_failure_probability": float(sum(
          row["approx_failure_probability"] for row in opponent_metrics
      )),
      "max_failure_probability": float(max([
          row["approx_failure_probability"] for row in opponent_metrics
      ] or [0.0])),
      "next_expected_surplus": float(
          own_surplus if next_row is None else next_row["expected_surplus"]
      ),
      "own_margin": float(own_view.margin()),
  }


class AZValueShieldPolicyBot:
  """Raw policy constrained by a post-action round-survival value model."""

  def __init__(self, policy_model, survival_model, name, device, model_args):
    self.name = name
    self.policy_model = policy_model
    self.survival_model = survival_model
    self.device = device
    self.args = model_args
    self.players = int(getattr(model_args, "players", 3))
    base_mode = str(getattr(model_args, "survival_value_base_mode", "policy"))
    if base_mode not in ("policy", "liveness_shield", "belief", "belief_liveness"):
      raise ValueError(f"Unsupported survival value shield base mode: {base_mode}")
    self.base_mode = base_mode
    self.base_bot = None
    self.opening_belief_bot = None
    if base_mode in ("liveness_shield", "belief_liveness"):
      self.base_bot = AZLivenessShieldPolicyBot(
          policy_model, f"{name}_base", device, model_args
      )
    if base_mode in ("belief", "belief_liveness"):
      belief_bot = AZBeliefSearchBot(
          policy_model,
          f"{name}_belief_base",
          device,
          model_args,
          max(1, int(getattr(model_args, "belief_samples", 4) or 4)),
          max(1, int(getattr(model_args, "belief_sims", 8) or 8)),
      )
      if base_mode == "belief":
        self.base_bot = belief_bot
      else:
        self.opening_belief_bot = belief_bot
    self._candidate_world_scores = {}
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "value_shield_disabled": 0,
        "value_shield_opening_belief_decisions": 0,
        "value_shield_liveness_decisions": 0,
        "value_shield_analytic_considered": 0,
        "value_shield_analytic_overrides": 0,
        "value_shield_analytic_blocked_own_feasibility": 0,
        "value_shield_analytic_blocked_own_margin": 0,
        "value_shield_analytic_blocked_token_loss": 0,
        "value_shield_analytic_blocked_advantage": 0,
        "value_shield_analytic_blocked_bid_incomplete": 0,
        "value_shield_analytic_blocked_over_target": 0,
        "value_shield_analytic_competitive_overrides": 0,
        "value_shield_analytic_overrides_by_trick": {},
        "value_shield_considered": 0,
        "value_shield_base_kept": 0,
        "value_shield_overrides": 0,
        "value_shield_fallback_max_survival": 0,
        "value_shield_no_scores": 0,
        "value_shield_blocked_improvement": 0,
        "value_shield_blocked_paired_lcb": 0,
        "value_shield_blocked_world_agreement": 0,
        "value_shield_blocked_policy_gap": 0,
        "value_shield_blocked_public_safety": 0,
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
        "value_shield_selection_mode": str(
            getattr(model_args, "survival_value_selection_mode", "threshold")
        ),
        "value_shield_base_mode": base_mode,
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
    phase = self._phase_name(state).lower()
    if spec and phase not in _split_phase_names(spec):
      return False
    if phase == "play":
      trick_number = int(getattr(state, "_trick_number", -1))
      min_trick = int(
          getattr(self.args, "survival_value_min_trick_number", -1)
      )
      max_trick = int(
          getattr(self.args, "survival_value_max_trick_number", -1)
      )
      if min_trick >= 0 and trick_number < min_trick:
        return False
      if max_trick >= 0 and trick_number > max_trick:
        return False
    return True

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
    self._candidate_world_scores = {}
    sample_count = max(
        1, int(getattr(self.args, "survival_value_belief_samples", 4) or 1)
    )
    worlds = sampled_belief_states_for_policy(
        state,
        player,
        sample_count,
        self.args,
        self.policy_model,
        self.device,
        float(getattr(self.args, "value_scale", 1.0)),
        context="eval",
    )
    if not worlds:
      raise RuntimeError("Survival value shield produced no information-state worlds")
    expected_observation = np.asarray(
        state.observation_tensor(player), dtype=np.float32
    )
    expected_legal = sorted(int(action) for action in state.legal_actions(player))
    for world in worlds:
      sampled_observation = np.asarray(
          world.observation_tensor(player), dtype=np.float32
      )
      sampled_legal = sorted(
          int(action) for action in world.legal_actions(player)
      )
      if (
          sampled_legal != expected_legal
          or not np.array_equal(sampled_observation, expected_observation)
      ):
        raise RuntimeError(
            "Survival value shield sampled a world outside the acting "
            "player's information state"
        )
    scores = {}
    for action in candidates:
      action_scores = []
      for sampled in worlds:
        rollout = sampled.clone()
        rollout.apply_action(int(action))
        action_scores.append(self._state_survival_score(rollout, player))
      self._candidate_world_scores[int(action)] = tuple(
          float(score) for score in action_scores
      )
      scores[int(action)] = float(np.mean(action_scores))
    return scores

  def _publicly_safe_alternative(self, state, player, base_action, action):
    if not bool(
        getattr(self.args, "survival_value_require_public_safety", False)
    ):
      return True
    base = action_liveness_certificate(
        state, player, int(base_action), base_action=int(base_action)
    )
    candidate = action_liveness_certificate(
        state, player, int(action), base_action=int(base_action)
    )
    return (
        int(candidate.get("min_player_open_slots_after", 0))
        >= int(base.get("min_player_open_slots_after", 0))
        and int(candidate.get("public_slot_damage", 0))
        <= int(base.get("public_slot_damage", 0))
        and int(candidate.get("own_total_deficit", 0))
        <= int(base.get("own_total_deficit", 0))
        and int(candidate.get("own_dead_rank_count", 0))
        <= int(base.get("own_dead_rank_count", 0))
        and (
            not bool(base.get("own_feasible", False))
            or bool(candidate.get("own_feasible", False))
        )
    )

  def _dominant_alternative(
      self, state, player, scores, policy, base_action
  ):
    base_action = int(base_action)
    base_score = float(scores.get(base_action, 0.0))
    base_samples = np.asarray(
        self._candidate_world_scores.get(base_action, (base_score,)),
        dtype=np.float64,
    )
    min_improvement = max(
        0.0, float(getattr(self.args, "survival_value_min_improvement", 0.0))
    )
    lcb_z = max(
        0.0, float(getattr(self.args, "survival_value_paired_lcb_z", 0.0))
    )
    min_win_share = min(1.0, max(
        0.0,
        float(getattr(self.args, "survival_value_min_world_win_share", 0.0)),
    ))
    eligible = []
    for action, score in scores.items():
      action = int(action)
      if action == base_action:
        continue
      candidate_samples = np.asarray(
          self._candidate_world_scores.get(action, (float(score),)),
          dtype=np.float64,
      )
      if candidate_samples.shape != base_samples.shape:
        raise RuntimeError(
            "Survival candidates were not scored on identical paired worlds"
        )
      deltas = candidate_samples - base_samples
      mean_delta = float(np.mean(deltas))
      if mean_delta <= min_improvement:
        self._decision_stats["value_shield_blocked_improvement"] += 1
        continue
      standard_error = (
          float(np.std(deltas, ddof=1)) / math.sqrt(float(len(deltas)))
          if len(deltas) > 1 else 0.0
      )
      paired_lcb = mean_delta - lcb_z * standard_error
      if paired_lcb <= min_improvement:
        self._decision_stats["value_shield_blocked_paired_lcb"] += 1
        continue
      world_win_share = float(np.mean(deltas > 1e-9))
      if world_win_share < min_win_share:
        self._decision_stats["value_shield_blocked_world_agreement"] += 1
        continue
      if not self._within_policy_gap(policy, base_action, action):
        self._decision_stats["value_shield_blocked_policy_gap"] += 1
        continue
      if not self._publicly_safe_alternative(
          state, player, base_action, action
      ):
        self._decision_stats["value_shield_blocked_public_safety"] += 1
        continue
      eligible.append((
          paired_lcb,
          mean_delta,
          float(policy[action]),
          -action,
          action,
          float(score),
      ))
    if not eligible:
      return base_action, base_score
    winner = max(eligible)
    return int(winner[-2]), float(winner[-1])

  def _within_policy_gap(self, policy, base_action, action):
    max_gap = float(
        getattr(self.args, "survival_value_max_policy_log_gap", -1.0)
    )
    if max_gap < 0:
      return True
    base_log = math.log(max(float(policy[int(base_action)]), 1e-12))
    action_log = math.log(max(float(policy[int(action)]), 1e-12))
    return (base_log - action_log) <= max_gap

  def _protected_base_action(self, state, player, legal, policy):
    if self.base_bot is None:
      return int(max(legal, key=lambda action: policy[action]))
    if self.opening_belief_bot is not None:
      min_trick = int(
          getattr(self.args, "survival_value_opening_min_trick_number", 0)
      )
      max_trick = int(
          getattr(self.args, "survival_value_opening_max_trick_number", 1)
      )
      trick_number = int(getattr(state, "_trick_number", -1))
      require_leading = bool(
          getattr(self.args, "survival_value_opening_lead_only", False)
      )
      if (
          self._phase_name(state).lower() == "play"
          and min_trick <= trick_number <= max_trick
          and (not require_leading or getattr(state, "_led_color", None) is None)
      ):
        self._decision_stats["value_shield_opening_belief_decisions"] += 1
        return self._deterministic_opening_belief_action(state, player, legal)
      self._decision_stats["value_shield_liveness_decisions"] += 1
    return int(self.base_bot.step(state.clone(), player))

  @staticmethod
  def _opening_information_state_seed(state, player, legal):
    digest = hashlib.sha256()
    digest.update(b"quantum-cat-opening-belief-v1")
    digest.update(np.asarray(
        state.observation_tensor(int(player)), dtype=np.float32
    ).tobytes())
    digest.update(np.asarray(
        sorted(int(action) for action in legal), dtype=np.int32
    ).tobytes())
    digest.update(str(int(player)).encode("ascii"))
    return int.from_bytes(digest.digest()[:4], byteorder="little", signed=False)

  def _deterministic_opening_belief_action(self, state, player, legal):
    seed = self._opening_information_state_seed(state, player, legal)
    previous_numpy_state = np.random.get_state()
    try:
      np.random.seed(seed)
      return int(self.opening_belief_bot.step(state.clone(), player))
    finally:
      np.random.set_state(previous_numpy_state)

  def _analytic_posterior_action(self, state, player, legal, base_action):
    if not bool(getattr(self.args, "survival_value_analytic_posterior", False)):
      return int(base_action)
    if self._phase_name(state).lower() != "play" or int(base_action) == 999:
      return int(base_action)
    trick_number = int(getattr(state, "_trick_number", -1))
    if trick_number < int(
        getattr(self.args, "survival_value_analytic_min_trick_number", 2)
    ):
      return int(base_action)
    alternatives = [int(action) for action in legal if int(action) != 999]
    if len(alternatives) < 2 or int(base_action) not in alternatives:
      return int(base_action)
    objective = str(
        getattr(
            self.args, "survival_value_analytic_objective", "table_survival"
        ) or "table_survival"
    )
    if objective not in ("table_survival", "competitive"):
      raise ValueError(f"Unsupported analytic posterior objective: {objective}")
    if objective == "competitive":
      predictions = list(getattr(state, "_predictions", []))
      tricks_won = list(getattr(state, "_tricks_won", []))
      if (
          int(player) >= len(predictions)
          or int(player) >= len(tricks_won)
          or int(predictions[int(player)]) < 0
          or int(predictions[int(player)]) != int(tricks_won[int(player)])
      ):
        self._decision_stats[
            "value_shield_analytic_blocked_bid_incomplete"
        ] += 1
        return int(base_action)
    self._decision_stats["value_shield_analytic_considered"] += 1
    scores = {
        action: public_posterior_action_score(state, player, action)
        for action in alternatives
    }
    if objective == "competitive":
      base_row = scores[int(base_action)]
      eligible = []
      minimum_advantage = float(
          getattr(self.args, "survival_value_analytic_min_advantage", 0.04)
      )
      for action in alternatives:
        if int(action) == int(base_action):
          continue
        candidate_row = scores[int(action)]
        certificate = action_liveness_certificate(
            state, player, int(action), base_action=int(base_action)
        )
        if not bool(certificate.get("own_feasible", False)):
          self._decision_stats[
              "value_shield_analytic_blocked_own_feasibility"
          ] += 1
          continue
        if candidate_row["own_margin"] < base_row["own_margin"]:
          self._decision_stats["value_shield_analytic_blocked_own_margin"] += 1
          continue
        if (
            bool(certificate.get("lost_led_token", False))
            and not bool(certificate.get("responsible_red_win_switch", False))
        ):
          self._decision_stats[
              "value_shield_analytic_blocked_token_loss"
          ] += 1
          continue
        if bool(certificate.get("over_target_would_win", False)):
          self._decision_stats[
              "value_shield_analytic_blocked_over_target"
          ] += 1
          continue
        opponent_pressure_gain = (
            float(candidate_row["sum_failure_probability"])
            - float(base_row["sum_failure_probability"])
        )
        if opponent_pressure_gain < minimum_advantage:
          self._decision_stats[
              "value_shield_analytic_blocked_advantage"
          ] += 1
          continue
        eligible.append((
            round(opponent_pressure_gain, 7),
            round(float(candidate_row["max_failure_probability"]), 7),
            round(float(candidate_row["own_margin"]), 7),
            -int(action),
            int(action),
        ))
      if not eligible:
        return int(base_action)
      candidate_action = int(max(eligible)[-1])
      self._decision_stats["value_shield_analytic_overrides"] += 1
      self._decision_stats["value_shield_analytic_competitive_overrides"] += 1
      _inc_counter(
          self._decision_stats["value_shield_analytic_overrides_by_trick"],
          str(trick_number),
      )
      return candidate_action
    candidate_action = max(
        alternatives,
        key=lambda action: (
            -round(scores[action]["sum_failure_probability"], 7),
            -round(scores[action]["max_failure_probability"], 7),
            round(scores[action]["min_expected_surplus"], 7),
            int(action == int(base_action)),
        ),
    )
    if candidate_action == int(base_action):
      return int(base_action)
    base_row = scores[int(base_action)]
    candidate_row = scores[candidate_action]
    certificate = action_liveness_certificate(
        state, player, candidate_action, base_action=int(base_action)
    )
    if not bool(certificate.get("own_feasible", False)):
      self._decision_stats[
          "value_shield_analytic_blocked_own_feasibility"
      ] += 1
      return int(base_action)
    if candidate_row["own_margin"] - base_row["own_margin"] < -1:
      self._decision_stats["value_shield_analytic_blocked_own_margin"] += 1
      return int(base_action)
    if (
        bool(certificate.get("lost_led_token", False))
        and not bool(certificate.get("responsible_red_win_switch", False))
    ):
      self._decision_stats["value_shield_analytic_blocked_token_loss"] += 1
      return int(base_action)
    failure_improvement = (
        base_row["sum_failure_probability"]
        - candidate_row["sum_failure_probability"]
    )
    if failure_improvement < float(
        getattr(self.args, "survival_value_analytic_min_advantage", 0.04)
    ):
      self._decision_stats["value_shield_analytic_blocked_advantage"] += 1
      return int(base_action)
    self._decision_stats["value_shield_analytic_overrides"] += 1
    _inc_counter(
        self._decision_stats["value_shield_analytic_overrides_by_trick"],
        str(trick_number),
    )
    return int(candidate_action)

  def step(self, state, player):
    legal = state.legal_actions(player)
    legal_count = len(legal)
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
    base_action = self._protected_base_action(state, player, legal, policy)
    base_action = self._analytic_posterior_action(
        state, player, legal, base_action
    )
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
    selection_mode = str(
        getattr(self.args, "survival_value_selection_mode", "threshold")
    )
    if selection_mode == "relative_dominance":
      if not base_scored or base_survival >= threshold:
        selected_action, selected_survival = int(base_action), base_survival
      else:
        selected_action, selected_survival = self._dominant_alternative(
            state, player, scores, policy, int(base_action)
        )
      if selected_action == int(base_action):
        self._decision_stats["value_shield_base_kept"] += 1
      else:
        self._decision_stats["value_shield_overrides"] += 1
        _inc_counter(self._decision_stats["value_shield_overrides_by_phase"], phase)
    elif base_scored and base_survival >= threshold:
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
    stats = json.loads(json.dumps(self._decision_stats))
    if self.base_bot is not None and hasattr(self.base_bot, "raw_decision_stats"):
      stats["nested"] = {"base_bot": self.base_bot.raw_decision_stats()}
    return stats


class AZPhaseGraftBot:
  """Use a graft actor in selected phases and a base actor elsewhere."""

  def __init__(self, base_bot, graft_bot, name, graft_phases="play"):
    self.base_bot = base_bot
    self.graft_bot = graft_bot
    self.name = name
    self.graft_phases = _split_phase_names(graft_phases or "play")
    prediction_bot = (
        self.graft_bot if "prediction" in self.graft_phases else self.base_bot
    )
    self.uses_neural_prediction = bool(
        getattr(prediction_bot, "uses_neural_prediction", False)
    )
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
    legal = list(state.legal_actions(player))
    legal_count = len(legal)
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
    phase = self._phase_name(state).lower()
    if legal_count <= 1:
      action = legal[0]
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
    self.uses_neural_prediction = _uses_neural_prediction(model_args)
    self._search_rng = np.random.RandomState(0)
    base_mode = str(
        getattr(model_args, "root_rollout_base_mode", "policy") or "policy"
    ).strip().lower().replace("-", "_")
    if base_mode == "policy":
      self._base_neural_bot = None
    elif base_mode == "liveness_shield":
      self._base_neural_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_base_liveness_shield", device, model_args
      )
    elif base_mode == "competitive_tactics":
      self._base_neural_bot = AZCompetitiveTacticsBot(
          model, f"{name}_base_competitive_tactics", device, model_args
      )
      self.uses_neural_prediction = bool(
          getattr(self._base_neural_bot, "uses_neural_prediction", False)
      )
    else:
      raise ValueError(f"Unsupported root rollout base mode: {base_mode}")
    self._base_mode = base_mode
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
          model,
          f"{name}_continuation_policy",
          device,
          model_args.value_scale,
          use_shared_prediction=not self.uses_neural_prediction,
      )
    elif continuation_mode == "q_policy":
      self._continuation_neural_bot = AZQPolicyBot(
          model, f"{name}_continuation_q_policy", device, model_args
      )
    elif continuation_mode == "liveness_shield":
      self._continuation_neural_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_continuation_liveness_shield", device, model_args
      )
    elif continuation_mode == "competitive_tactics":
      self._continuation_neural_bot = AZCompetitiveTacticsBot(
          model, f"{name}_continuation_competitive_tactics", device, model_args
      )
    elif continuation_mode:
      raise ValueError(f"Unsupported root rollout continuation mode: {continuation_mode}")
    else:
      self._continuation_neural_bot = None
    self._continuation_mode = continuation_mode
    max_audit_p = float(
        getattr(model_args, "root_rollout_max_audit_paired_binomial_p", 1.0)
    )
    if not 0.0 <= max_audit_p <= 1.0:
      raise ValueError(
          "root_rollout_max_audit_paired_binomial_p must be in [0, 1]"
      )
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "root_rollout_disabled": 0,
        "root_rollout_protected_tactical_overrides": 0,
        "root_rollout_blocked_own_liveness": 0,
        "root_rollout_blocked_baseline_risk": 0,
        "root_rollout_considered": 0,
        "root_rollout_no_candidates": 0,
        "root_rollout_no_rollouts": 0,
        "root_rollout_blocked_paradox_margin": 0,
        "root_rollout_belief_sample_failures": 0,
        "root_rollout_audit_considered": 0,
        "root_rollout_audit_no_rollouts": 0,
        "root_rollout_audit_blocked": 0,
        "root_rollout_audit_passed": 0,
        "root_rollout_audit_passes_passed": 0,
        "root_rollout_overrides": 0,
        "root_rollout_scored_candidates": 0,
        "root_rollout_by_phase": {},
        "root_rollout_by_trick": {},
        "root_rollout_overrides_by_phase": {},
        "root_rollout_overrides_by_trick": {},
        "root_rollout_base_mode": base_mode,
        "root_rollout_base_failures": 0,
        "root_rollout_continuation_mode": continuation_mode,
        "root_rollout_continuation_bot": continuation_bot_name,
        "root_rollout_belief_source": str(
            getattr(model_args, "belief_source", "infostate") or "infostate"
        ),
        "root_rollout_audit_inference_unit": "information_state_world",
        "root_rollout_audit_paradox_scope": str(
            getattr(model_args, "root_rollout_paradox_scope", "acting")
        ),
        "root_rollout_rng_scope": "acting_information_state",
        "root_rollout_replay_samples": [],
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
        use_shared_prediction=not self.uses_neural_prediction,
    )
    return max(legal, key=lambda action: policy[action])

  def _reset_search_rng(self, state, player, legal):
    """Seeds search from deployable information, independent of prior calls."""
    digest = hashlib.sha256()
    observation = np.asarray(
        state.observation_tensor(player), dtype=np.float32
    )
    digest.update(observation.tobytes())
    digest.update(np.asarray(
        sorted(int(action) for action in legal), dtype=np.int32
    ).tobytes())
    digest.update(str(int(player)).encode("ascii"))
    digest.update(str(int(
        getattr(self.args, "root_rollout_seed_salt", 0)
    )).encode("ascii"))
    seed = int.from_bytes(digest.digest()[:4], byteorder="little", signed=False)
    self._search_rng.seed(seed)
    return seed

  @classmethod
  def _paired_audit_summary(cls, scored, baseline_action, proposed_action):
    baseline = scored.get(int(baseline_action))
    proposed = scored.get(int(proposed_action))
    if baseline is None or proposed is None:
      return {"valid": False, "reason": "missing_action"}
    baseline_outcomes = baseline.get("paradox_world_rates")
    proposed_outcomes = proposed.get("paradox_world_rates")
    if baseline_outcomes is None and proposed_outcomes is None:
      baseline_outcomes = baseline.get("paradox_outcomes")
      proposed_outcomes = proposed.get("paradox_outcomes")
    if (
        not isinstance(baseline_outcomes, list)
        or not isinstance(proposed_outcomes, list)
        or not baseline_outcomes
        or len(baseline_outcomes) != len(proposed_outcomes)
    ):
      return {"valid": False, "reason": "misaligned_outcomes"}
    saved = 0
    harmed = 0
    for baseline_value, proposed_value in zip(
        baseline_outcomes, proposed_outcomes
    ):
      difference = float(baseline_value) - float(proposed_value)
      saved += int(difference > 1e-12)
      harmed += int(difference < -1e-12)
    return {
        "valid": True,
        "worlds": len(baseline_outcomes),
        "baseline_paradox": float(baseline["paradox"]),
        "proposed_paradox": float(proposed["paradox"]),
        "paradox_improvement": (
            float(baseline["paradox"]) - float(proposed["paradox"])
        ),
        "baseline_score": float(baseline["score"]),
        "proposed_score": float(proposed["score"]),
        "saved": saved,
        "harmed": harmed,
        "paired_binomial_p": cls._one_sided_binomial_tail(saved, harmed),
    }

  def _record_replay_sample(
      self, state, player, legal, policy, baseline_action, proposed_action,
      selected_action, search_seed, selection_scored, audit_summaries,
  ):
    if not bool(getattr(
        self.args, "root_rollout_record_replay_samples", False
    )):
      return
    legal = [int(action) for action in legal]
    all_action_features = action_feature_matrix(
        state, player, state.num_distinct_actions()
    )
    sample = {
        "schema_version": 1,
        "information_safe": True,
        "player": int(player),
        "phase": _phase_name(state),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "search_seed": int(search_seed),
        "observation": np.asarray(
            state.observation_tensor(player), dtype=np.float32
        ).tolist(),
        "legal_actions": legal,
        "legal_action_features": np.asarray(
            all_action_features[legal], dtype=np.float32
        ).tolist(),
        "legal_policy": [float(policy[action]) for action in legal],
        "baseline_action": int(baseline_action),
        "proposed_action": int(proposed_action),
        "selected_action": int(selected_action),
        "audit_accepted": int(selected_action) == int(proposed_action),
        "audit_contract": {
            "baseline_mode": self._base_mode,
            "samples": int(getattr(self.args, "root_rollout_audit_samples", 0)),
            "rollouts_per_world": int(getattr(
                self.args, "root_rollout_audit_rollouts", 1
            )),
            "required_passes": int(getattr(
                self.args, "root_rollout_audit_passes", 1
            )),
            "min_paradox_improvement": float(getattr(
                self.args, "root_rollout_min_audit_paradox_improvement", 0.0
            )),
            "min_score_improvement": float(getattr(
                self.args, "root_rollout_min_audit_score_improvement", -1e9
            )),
            "max_paired_binomial_p": float(getattr(
                self.args, "root_rollout_max_audit_paired_binomial_p", 1.0
            )),
            "paradox_scope": str(getattr(
                self.args, "root_rollout_paradox_scope", "acting"
            )),
            "inference_unit": "information_state_world",
        },
        "selection": self._paired_audit_summary(
            selection_scored, baseline_action, proposed_action
        ),
        "audit_passes": audit_summaries,
    }
    self._decision_stats["root_rollout_replay_samples"].append(sample)
    self._decision_stats["root_rollout_replay_samples"][:] = (
        _trim_root_rollout_replay_samples(
            self._decision_stats["root_rollout_replay_samples"]
        )
    )

  def _has_explicit_continuation_policy(self):
    return bool(self._continuation_bots) or self._continuation_neural_bot is not None

  def _candidate_actions(
      self, state, player, legal, policy, baseline_action=None
  ):
    legal = list(legal)
    max_actions = int(getattr(self.args, "root_rollout_max_actions", 4))
    if max_actions <= 0 or len(legal) <= max_actions:
      return sorted(legal)
    selected = []
    selected_set = set()
    if baseline_action in legal:
      selected.append(int(baseline_action))
      selected_set.add(int(baseline_action))
    top_count = max(0, int(getattr(self.args, "root_rollout_top_policy", 2)))
    if top_count > 0:
      ordered = sorted(legal, key=lambda action: float(policy[action]), reverse=True)
      for action in ordered[:min(top_count, max_actions)]:
        if action in selected_set:
          continue
        selected.append(action)
        selected_set.add(action)
        if len(selected) >= max_actions:
          break
    if bool(getattr(self.args, "root_rollout_teacher_candidates", False)):
      led_color = getattr(state, "_led_color", None)
      if led_color in ("B", "Y", "G"):
        buckets = (
            [action for action in legal if _action_color(state, action) == led_color],
            [action for action in legal if _action_color(state, action) == "R"],
            [
                action for action in legal
                if _action_color(state, action) not in (None, led_color)
            ],
        )
        for bucket in buckets:
          ordered = sorted(
              bucket, key=lambda action: float(policy[action]), reverse=True
          )
          for action in ordered:
            if action not in selected_set:
              selected.append(action)
              selected_set.add(action)
              break
          if len(selected) >= max_actions:
            return sorted(selected[:max_actions])
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
        chosen = self._search_rng.choice(
            remaining, size=fill_count, replace=False
        ).tolist()
        selected.extend(int(action) for action in chosen)
    return sorted(selected)

  def _base_action(self, state, player, legal, policy):
    if self._base_neural_bot is not None:
      try:
        action = int(self._base_neural_bot.step(state.clone(), player))
        if action in legal:
          return action
      except Exception:
        self._decision_stats["root_rollout_base_failures"] += 1
    return int(max(legal, key=lambda action: policy[action]))

  def _safe_candidate_actions(self, state, player, candidates, baseline_action):
    """Filter search proposals using only actor-public incumbent safeguards."""
    preserve_liveness = bool(
        getattr(self.args, "root_rollout_preserve_own_liveness", False)
    )
    risk_limit = float(
        getattr(self.args, "root_rollout_max_baseline_risk_increase", -1.0)
    )
    if not preserve_liveness and risk_limit < 0.0:
      return list(candidates)

    baseline_action = int(baseline_action)
    baseline_row = None
    baseline_margin = None
    own_view = None
    if preserve_liveness:
      try:
        baseline_row = action_liveness_certificate(
            state, player, baseline_action, base_action=baseline_action
        )
        own_view = liveness_oracle.view_from_state(state, player)
        baseline_margin = int(
            liveness_oracle.post_play_margin(own_view, baseline_action)
        )
      except Exception:
        self._decision_stats["root_rollout_blocked_own_liveness"] += max(
            0, len(candidates) - 1
        )
        return [baseline_action]

    risks = None
    if risk_limit >= 0.0:
      try:
        _, _, _, risks, _ = model_policy_value_action_aux(
            self.model,
            state,
            player,
            state.num_distinct_actions(),
            self.args.value_scale,
            self.device,
            need_action_risks=True,
            use_shared_prediction=not self.uses_neural_prediction,
        )
      except Exception:
        risks = None
      if risks is None or not np.isfinite(float(risks[baseline_action])):
        self._decision_stats["root_rollout_blocked_baseline_risk"] += max(
            0, len(candidates) - 1
        )
        return [baseline_action]

    allowed = []
    for candidate in candidates:
      action = int(candidate)
      if action == baseline_action:
        allowed.append(action)
        continue
      if preserve_liveness:
        try:
          candidate_row = action_liveness_certificate(
              state, player, action, base_action=baseline_action
          )
          candidate_margin = int(
              liveness_oracle.post_play_margin(own_view, action)
          )
        except Exception:
          self._decision_stats["root_rollout_blocked_own_liveness"] += 1
          continue
        worsens_feasibility = (
            bool(baseline_row.get("own_feasible", False))
            and not bool(candidate_row.get("own_feasible", False))
        )
        loses_irresponsible_token = (
            bool(candidate_row.get("lost_led_token", False))
            and not bool(baseline_row.get("lost_led_token", False))
            and not bool(candidate_row.get("responsible_red_win_switch", False))
        )
        breaks_exact_bid = (
            bool(candidate_row.get("over_target_would_win", False))
            and not bool(baseline_row.get("over_target_would_win", False))
        )
        if (
            worsens_feasibility
            or candidate_margin < baseline_margin
            or int(candidate_row.get("own_buffer_deficit", 0))
            > int(baseline_row.get("own_buffer_deficit", 0))
            or loses_irresponsible_token
            or breaks_exact_bid
        ):
          self._decision_stats["root_rollout_blocked_own_liveness"] += 1
          continue
      if risk_limit >= 0.0 and (
          not np.isfinite(float(risks[action]))
          or float(risks[action])
          > float(risks[baseline_action]) + risk_limit + 1e-9
      ):
        self._decision_stats["root_rollout_blocked_baseline_risk"] += 1
        continue
      allowed.append(action)
    return allowed if baseline_action in allowed else [baseline_action]

  def _enabled_for_phase(self, state):
    spec = str(getattr(self.args, "root_rollout_phases", "") or "").strip()
    phase = {
        0: "chance",
        1: "discard",
        2: "prediction",
        3: "play",
        4: "terminal",
    }.get(int(getattr(state, "_phase", -1)), "unknown")
    if spec:
      allowed = {part.strip() for part in spec.split(",") if part.strip()}
      if phase not in allowed:
        return False
    led_spec = str(
        getattr(self.args, "root_rollout_led_colors", "") or ""
    ).strip()
    if led_spec:
      allowed_led = _split_phase_names(led_spec)
      led_color = getattr(state, "_led_color", None)
      led_name = "none" if led_color is None else str(led_color).lower()
      if led_name not in allowed_led:
        return False
    trick_number = int(getattr(state, "_trick_number", -1))
    min_trick = int(getattr(self.args, "root_rollout_min_trick_number", -1))
    max_trick = int(getattr(self.args, "root_rollout_max_trick_number", -1))
    if min_trick >= 0 and trick_number < min_trick:
      return False
    if max_trick >= 0 and trick_number > max_trick:
      return False
    match_round = int(getattr(state, "_match_round", -1))
    min_round = int(getattr(self.args, "root_rollout_min_match_round", -1))
    max_round = int(getattr(self.args, "root_rollout_max_match_round", -1))
    if min_round >= 0 and match_round < min_round:
      return False
    if max_round >= 0 and match_round > max_round:
      return False
    return True

  def _preserves_tactical_override(self):
    return bool(
        getattr(self.args, "root_rollout_preserve_tactical_overrides", False)
        and self._base_mode == "competitive_tactics"
        and getattr(self._base_neural_bot, "last_tactical_override", False)
    )

  def _leaf_value(self, state, player):
    while state.is_chance_node() and not state.is_terminal():
      actions, probs = zip(*state.chance_outcomes())
      state.apply_action(int(self._search_rng.choice(actions, p=probs)))
    if state.is_terminal():
      return float(terminal_search_value(state, self.args)[player])
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

  def _sample_infostate_worlds(self, state, player, samples):
    """Samples deployment-safe worlds and fails closed on any mismatch."""
    expected_legal = [int(action) for action in state.legal_actions(player)]
    expected_observation = np.asarray(
        state.observation_tensor(player), dtype=np.float32
    )
    sample_count = max(1, int(samples))
    source = str(
        getattr(self.args, "belief_source", "infostate") or "infostate"
    ).strip().lower()
    if source == "infostate":
      sampled_worlds = []
      for _ in range(sample_count):
        try:
          sampled_worlds.append(
              state.resample_from_infostate(player, self._search_rng)
          )
        except Exception:
          self._decision_stats["root_rollout_belief_sample_failures"] += 1
          return []
    else:
      global_rng_state = np.random.get_state()
      sampler_seed = int(self._search_rng.randint(0, 2**32 - 1))
      try:
        np.random.seed(sampler_seed)
        sampled_worlds = sampled_belief_states_for_policy(
            state,
            player,
            sample_count,
            self.args,
            self.model,
            self.device,
            self.args.value_scale,
            context="eval",
        )
      except Exception:
        self._decision_stats["root_rollout_belief_sample_failures"] += 1
        return []
      finally:
        np.random.set_state(global_rng_state)

    worlds = []
    for sampled in sampled_worlds:
      try:
        sampled_legal = [
            int(action) for action in sampled.legal_actions(player)
        ]
        sampled_observation = np.asarray(
            sampled.observation_tensor(player), dtype=np.float32
        )
      except Exception:
        self._decision_stats["root_rollout_belief_sample_failures"] += 1
        return []
      if (
          sampled_legal != expected_legal
          or not np.array_equal(sampled_observation, expected_observation)
      ):
        self._decision_stats["root_rollout_belief_sample_failures"] += 1
        return []
      worlds.append(sampled)
    return worlds

  def _score_actions(self, states, player, actions, rollouts):
    action_results = {
        int(action): {
            "paradox": [],
            "score": [],
            "paradox_world_rates": [],
        }
        for action in actions
    }
    rollouts = max(1, int(rollouts))
    for belief_state in states:
      legal_in_belief = set(belief_state.legal_actions(player))
      legal_actions = [
          int(action) for action in actions if int(action) in legal_in_belief
      ]
      if not legal_actions:
        continue
      world_paradox = {action: [] for action in legal_actions}
      for _ in range(rollouts):
        np_state = self._search_rng.get_state()
        for action in legal_actions:
          self._search_rng.set_state(np_state)
          result = self._rollout_result(belief_state, player, action)
          paradox = float(result["paradox"])
          action_results[action]["paradox"].append(paradox)
          action_results[action]["score"].append(float(result["score"]))
          world_paradox[action].append(paradox)
      for action in legal_actions:
        if world_paradox[action]:
          action_results[action]["paradox_world_rates"].append(
              float(np.mean(world_paradox[action]))
          )
    return {
        action: {
            "paradox": float(np.mean(results["paradox"])),
            "score": float(np.mean(results["score"])),
            "paradox_outcomes": list(results["paradox"]),
            "paradox_world_rates": list(results["paradox_world_rates"]),
            "rollout_worlds": int(len(results["paradox_world_rates"])),
            "rollouts_per_world": int(rollouts),
        }
        for action, results in action_results.items()
        if results["score"]
    }

  @staticmethod
  def _one_sided_binomial_tail(successes, failures):
    successes = int(successes)
    failures = int(failures)
    trials = successes + failures
    if trials <= 0:
      return 1.0
    return float(
        sum(math.comb(trials, k) for k in range(successes, trials + 1))
        / (2 ** trials)
    )

  @classmethod
  def _audit_override_allowed(
      cls,
      scored,
      baseline_action,
      selected_action,
      min_improvement,
      max_paired_binomial_p=1.0,
      min_score_improvement=-1e9,
  ):
    baseline_result = scored.get(int(baseline_action))
    selected_result = scored.get(int(selected_action))
    if baseline_result is None or selected_result is None:
      return False
    improvement = (
        float(baseline_result["paradox"])
        - float(selected_result["paradox"])
    )
    if improvement < float(min_improvement):
      return False
    if (
        float(selected_result["score"]) - float(baseline_result["score"])
        < float(min_score_improvement)
    ):
      return False
    max_p = float(max_paired_binomial_p)
    if max_p >= 1.0:
      return True
    baseline_outcomes = baseline_result.get("paradox_world_rates")
    selected_outcomes = selected_result.get("paradox_world_rates")
    # Legacy/unit-test records with one outcome per independent world remain
    # readable. Newly scored records always carry explicit world-level rates.
    if baseline_outcomes is None and selected_outcomes is None:
      baseline_outcomes = baseline_result.get("paradox_outcomes")
      selected_outcomes = selected_result.get("paradox_outcomes")
    if (
        not isinstance(baseline_outcomes, list)
        or not isinstance(selected_outcomes, list)
        or len(baseline_outcomes) != len(selected_outcomes)
        or not baseline_outcomes
    ):
      return False
    saved = 0
    harmed = 0
    for baseline_value, selected_value in zip(
        baseline_outcomes, selected_outcomes
    ):
      difference = float(baseline_value) - float(selected_value)
      saved += int(difference > 1e-12)
      harmed += int(difference < -1e-12)
    return cls._one_sided_binomial_tail(saved, harmed) <= max_p + 1e-12

  def _rollout_round_result(self, belief_state, player, first_action):
    rollout = belief_state.clone()
    rollout.apply_action(first_action)
    max_plies = max(0, int(getattr(self.args, "root_rollout_max_plies", 0)))
    decision_plies = 0
    while not rollout.is_terminal():
      if rollout.is_chance_node():
        actions, probs = zip(*rollout.chance_outcomes())
        rollout.apply_action(int(self._search_rng.choice(actions, p=probs)))
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
        "score": float(terminal_search_value(rollout, self.args)[player]),
    }

  def _play_model_round(self, state):
    while not state.is_terminal():
      if state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state.apply_action(int(self._search_rng.choice(actions, p=probs)))
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
        rollout.apply_action(int(self._search_rng.choice(actions, p=probs)))
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
    final_terminal_state = rollout
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
      final_terminal_state = next_state
      next_returns = np.array(next_state.returns(), dtype=np.float32)
      if getattr(self.args, "match_context", False):
        totals = next_returns
      else:
        totals += next_returns
    if float(getattr(self.args, "official_outcome_value_weight", 0.0)) > 0:
      paradox_target = np.asarray(
          getattr(final_terminal_state, "_has_paradoxed", [False] * players),
          dtype=np.float32,
      )
      final_round_scores = (
          final_terminal_state.raw_round_scores()
          if hasattr(final_terminal_state, "raw_round_scores")
          else final_terminal_state.returns()
      )
      totals = (
          value_targets_from_scores(
              totals,
              paradox_target,
              self.args,
              final_round_scores=final_round_scores,
          ) * float(getattr(self.args, "value_scale", 1.0))
      )
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
        use_shared_prediction=not self.uses_neural_prediction,
    )
    baseline_action = self._base_action(state, player, legal, policy)
    if not self._enabled_for_phase(state):
      self._decision_stats["root_rollout_disabled"] += 1
      _record_decision(self._decision_stats, state, baseline_action, legal_count)
      return baseline_action
    if self._preserves_tactical_override():
      self._decision_stats["root_rollout_protected_tactical_overrides"] += 1
      _record_decision(self._decision_stats, state, baseline_action, legal_count)
      return baseline_action
    phase = _phase_name(state)
    trick_number = int(getattr(state, "_trick_number", -1))
    self._decision_stats["root_rollout_considered"] += 1
    _inc_counter(self._decision_stats["root_rollout_by_phase"], phase)
    _inc_counter(
        self._decision_stats["root_rollout_by_trick"], str(trick_number)
    )
    search_seed = self._reset_search_rng(state, player, legal)
    candidates = self._candidate_actions(
        state, player, legal, policy, baseline_action=baseline_action
    )
    candidates = self._safe_candidate_actions(
        state, player, candidates, baseline_action
    )
    if not candidates:
      self._decision_stats["root_rollout_no_candidates"] += 1
      _record_decision(self._decision_stats, state, baseline_action, legal_count)
      return baseline_action
    samples = max(1, int(getattr(self.args, "root_rollout_samples", 2)))
    sampled_states = self._sample_infostate_worlds(state, player, samples)
    if not sampled_states:
      self._decision_stats["root_rollout_no_rollouts"] += 1
      _record_decision(self._decision_stats, state, baseline_action, legal_count)
      return baseline_action
    rollouts = max(1, int(getattr(self.args, "root_rollouts", 1)))
    scored = self._score_actions(
        sampled_states, player, candidates, rollouts
    )
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
    if (
        selected_action != baseline_action
        and str(getattr(self.args, "root_rollout_objective", "score"))
        == "paradox_then_score"
    ):
      baseline_result = scored.get(int(baseline_action))
      selected_result = scored.get(int(selected_action))
      min_improvement = float(
          getattr(self.args, "root_rollout_min_paradox_improvement", 0.0)
      )
      if (
          baseline_result is None
          or selected_result is None
          or float(baseline_result["paradox"])
          - float(selected_result["paradox"]) < min_improvement
      ):
        self._decision_stats["root_rollout_blocked_paradox_margin"] += 1
        selected_action = int(baseline_action)
    proposed_action = int(selected_action)
    audit_summaries = []
    audit_samples = max(
        0, int(getattr(self.args, "root_rollout_audit_samples", 0))
    )
    if (
        selected_action != baseline_action
        and str(getattr(self.args, "root_rollout_objective", "score"))
        == "paradox_then_score"
        and audit_samples > 0
    ):
      self._decision_stats["root_rollout_audit_considered"] += 1
      audit_rollouts = max(
          1, int(getattr(self.args, "root_rollout_audit_rollouts", 1))
      )
      audit_passes = max(
          1, int(getattr(self.args, "root_rollout_audit_passes", 1))
      )
      audit_passed = True
      for _ in range(audit_passes):
        audit_states = self._sample_infostate_worlds(
            state, player, audit_samples
        )
        audit_scored = self._score_actions(
            audit_states,
            player,
            [baseline_action, selected_action],
            audit_rollouts,
        ) if audit_states else {}
        if not audit_scored:
          self._decision_stats["root_rollout_audit_no_rollouts"] += 1
          audit_passed = False
          break
        audit_summaries.append(self._paired_audit_summary(
            audit_scored, baseline_action, proposed_action
        ))
        if not self._audit_override_allowed(
            audit_scored,
            baseline_action,
            selected_action,
            getattr(
                self.args,
                "root_rollout_min_audit_paradox_improvement",
                0.0,
            ),
            getattr(
                self.args,
                "root_rollout_max_audit_paired_binomial_p",
                1.0,
            ),
            getattr(
                self.args,
                "root_rollout_min_audit_score_improvement",
                -1e9,
            ),
        ):
          self._decision_stats["root_rollout_audit_blocked"] += 1
          audit_passed = False
          break
        self._decision_stats["root_rollout_audit_passes_passed"] += 1
      if audit_passed:
        self._decision_stats["root_rollout_audit_passed"] += 1
      else:
        selected_action = int(baseline_action)
      self._record_replay_sample(
          state,
          player,
          legal,
          policy,
          baseline_action,
          proposed_action,
          selected_action,
          search_seed,
          scored,
          audit_summaries,
      )
    if selected_action != baseline_action:
      self._decision_stats["root_rollout_overrides"] += 1
      _inc_counter(self._decision_stats["root_rollout_overrides_by_phase"], phase)
      _inc_counter(
          self._decision_stats["root_rollout_overrides_by_trick"],
          str(trick_number),
      )
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
    if self._base_neural_bot is not None:
      if hasattr(self._base_neural_bot, "raw_decision_stats"):
        stats.setdefault("nested", {})["base_bot"] = (
            self._base_neural_bot.raw_decision_stats()
        )
      elif hasattr(self._base_neural_bot, "decision_stats"):
        stats.setdefault("nested", {})["base_bot"] = (
            self._base_neural_bot.decision_stats()
        )
    if self._continuation_neural_bot is not None:
      if hasattr(self._continuation_neural_bot, "raw_decision_stats"):
        stats.setdefault("nested", {})["continuation_bot"] = (
            self._continuation_neural_bot.raw_decision_stats()
        )
      elif hasattr(self._continuation_neural_bot, "decision_stats"):
        stats.setdefault("nested", {})["continuation_bot"] = (
            self._continuation_neural_bot.decision_stats()
        )
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


def _must_win_current_trick_to_hit_prediction(state, player):
  predictions = getattr(state, "_predictions", [])
  tricks_won = getattr(state, "_tricks_won", [])
  player = int(player)
  if player >= len(predictions) or player >= len(tricks_won):
    return False
  prediction = int(predictions[player])
  if prediction < 0:
    return False
  tricks = int(tricks_won[player])
  if tricks >= prediction:
    return False
  num_tricks = int(getattr(state, "_num_tricks", 0) or 0)
  trick_number = int(getattr(state, "_trick_number", 0) or 0)
  remaining_after_current = max(0, num_tricks - trick_number - 1)
  return (prediction - tricks) > remaining_after_current


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
          would_win_now is True and prediction >= 0 and tricks == prediction
      ),
  }
  return row


def action_liveness_key(row, action, base_action=None):
  """Lexicographic deterministic safety key for a candidate action.

  Higher is better. This key is deliberately local and public/own-info only:
  exact own remaining-hand feasibility comes first, then public exit-space
  preservation, then prediction/token-loss tie breakers. A red switch that wins
  a still-needed trick is treated as responsible lane spending instead of a
  generic lost-led-token mistake. It is meant for within-state comparisons, not
  as a calibrated global risk probability.
  """
  base_bonus = (
      1 if base_action is not None and int(action) == int(base_action) else 0
  )
  lost_led_token = bool(row.get("lost_led_token", False))
  responsible_red_win_switch = (
      lost_led_token
      and bool(row.get("would_win_now", False))
      and not bool(row.get("over_target_would_win", False))
  )
  preserves_lane_or_spends_to_win = (
      not lost_led_token or responsible_red_win_switch
  )
  return (
      0 if bool(row.get("is_paradox", False)) else 1,
      1 if bool(row.get("own_feasible", False)) else 0,
      1 if preserves_lane_or_spends_to_win else 0,
      -int(row.get("own_total_deficit", 0)),
      -int(row.get("own_dead_rank_count", 0)),
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
      "led_color": getattr(state, "_led_color", None),
  })
  row["responsible_red_win_switch"] = bool(
      row.get("lost_led_token", False)
      and bool(row.get("would_win_now", False))
      and not bool(row.get("over_target_would_win", False))
  )
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


def exact_table_liveness_after_action(state, action):
  """Full-state post-action table liveness margins for teacher diagnostics.

  This reads every player's hand after applying the candidate action. It is
  therefore only appropriate for exact simulator teachers, audits, and
  training-label generation, not deployed information-legal inference.
  """
  clone = state.clone()
  clone.apply_action(int(action))
  num_players = int(getattr(clone, "_num_players", 0) or 0)
  has_paradoxed = list(getattr(clone, "_has_paradoxed", [False] * num_players))
  any_paradox = any(bool(value) for value in has_paradoxed)
  if clone.is_terminal():
    good_terminal = not any_paradox
    margin = 99 if good_terminal else -99
    return {
        "table_terminal_after": True,
        "table_any_paradox_after": bool(any_paradox),
        "table_dead_margin_players_after": int(
            sum(1 for value in has_paradoxed if value)
        ),
        "table_min_margin_after": int(margin),
        "table_current_margin_after": int(margin),
        "table_total_margin_after": int(margin * max(1, num_players)),
        "table_active_players_after": 0,
    }

  margins = []
  dead_count = 0
  for seat in range(num_players):
    if bool(has_paradoxed[seat]):
      dead_count += 1
      continue
    remaining_cards = int(np.sum(getattr(clone, "_hands", [])[seat]))
    if remaining_cards <= 0:
      continue
    view = liveness_oracle.view_from_state(clone, seat)
    margin = int(view.margin())
    margins.append(margin)
    if margin < 0:
      dead_count += 1

  current = int(clone.current_player())
  current_margin = 99
  if 0 <= current < num_players and not bool(has_paradoxed[current]):
    current_remaining = int(np.sum(getattr(clone, "_hands", [])[current]))
    if current_remaining > 0:
      current_margin = int(liveness_oracle.view_from_state(clone, current).margin())

  return {
      "table_terminal_after": False,
      "table_any_paradox_after": bool(any_paradox),
      "table_dead_margin_players_after": int(dead_count),
      "table_min_margin_after": int(min(margins)) if margins else 99,
      "table_current_margin_after": int(current_margin),
      "table_total_margin_after": int(sum(margins)) if margins else 99 * max(1, num_players),
      "table_active_players_after": int(len(margins)),
  }


class AZBeliefTableLivenessPolicyBot:
  """Information-legal approximation of the exact table-liveness teacher.

  The exact teacher reads every real hand and is diagnostic-only.  This bot
  instead samples hidden worlds exclusively through the acting player's
  information state, evaluates every legal action on the same sampled worlds,
  and fails closed to its base policy unless a paired lower-confidence-bound
  gate clears.  The real state is never passed to an exact liveness scorer.
  """

  def __init__(self, model, name, device, model_args):
    self.model = model
    self.name = name
    self.device = device
    self.args = model_args
    self.uses_neural_prediction = _uses_neural_prediction(model_args)
    base_mode = str(
        getattr(model_args, "belief_table_liveness_base_mode", "policy")
        or "policy"
    )
    if base_mode == "policy":
      self.base_bot = AZPolicyBot(
          model,
          f"{name}_base_policy",
          device,
          model_args.value_scale,
          use_shared_prediction=not self.uses_neural_prediction,
      )
    elif base_mode == "liveness_shield":
      self.base_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_base_liveness_shield", device, model_args
      )
    else:
      raise ValueError(
          "belief_table_liveness_base_mode must be policy or "
          f"liveness_shield, got {base_mode!r}"
      )
    aggregation = str(
        getattr(
            model_args,
            "belief_table_liveness_aggregation",
            "teacher_vote",
        )
        or "teacher_vote"
    ).strip().lower()
    if aggregation not in ("scalar_mean", "scalar_quantile", "teacher_vote"):
      raise ValueError(
          "belief_table_liveness_aggregation must be scalar_mean, "
          f"scalar_quantile, or teacher_vote, got {aggregation!r}"
      )
    quantile = float(
        getattr(model_args, "belief_table_liveness_quantile", 0.25)
    )
    if not 0.0 <= quantile <= 1.0:
      raise ValueError(
          "belief_table_liveness_quantile must be in [0, 1], "
          f"got {quantile}"
      )
    self.aggregation = aggregation
    self.quantile = quantile
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "belief_table_liveness_disabled": 0,
        "belief_table_liveness_considered": 0,
        "belief_table_liveness_base_kept": 0,
        "belief_table_liveness_overrides": 0,
        "belief_table_liveness_shadow_overrides": 0,
        "belief_table_liveness_sample_failures": 0,
        "belief_table_liveness_infostate_mismatches": 0,
        "belief_table_liveness_insufficient_worlds": 0,
        "belief_table_liveness_blocked_paired_lcb": 0,
        "belief_table_liveness_blocked_best_share": 0,
        "belief_table_liveness_blocked_policy_gap": 0,
        "belief_table_liveness_worlds_scored": 0,
        "belief_table_liveness_paired_delta_sum": 0.0,
        "belief_table_liveness_paired_lcb_sum": 0.0,
        "belief_table_liveness_best_share_sum": 0.0,
        "belief_table_liveness_by_trick": {},
        "belief_table_liveness_overrides_by_trick": {},
        "belief_table_liveness_base_mode": base_mode,
        "belief_table_liveness_aggregation": aggregation,
        "belief_table_liveness_quantile": quantile,
        "belief_table_liveness_inference_unit": "acting_information_state_world",
        "belief_table_liveness_real_state_exact_scoring": False,
        "belief_table_liveness_override_samples": [],
    })

  def _enabled_for_state(self, state):
    phases = _split_phase_names(
        getattr(self.args, "belief_table_liveness_phases", "play")
    )
    if phases and _phase_name(state).lower() not in phases:
      return False
    trick = int(getattr(state, "_trick_number", -1))
    min_trick = int(
        getattr(self.args, "belief_table_liveness_min_trick_number", 5)
    )
    max_trick = int(
        getattr(self.args, "belief_table_liveness_max_trick_number", 7)
    )
    if min_trick >= 0 and trick < min_trick:
      return False
    if max_trick >= 0 and trick > max_trick:
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
        use_shared_prediction=not self.uses_neural_prediction,
    )
    return policy

  def _information_state_seed(self, state, player, legal):
    digest = hashlib.sha256()
    observation = np.asarray(
        state.observation_tensor(player), dtype=np.float32
    )
    digest.update(observation.tobytes())
    digest.update(np.asarray(
        sorted(int(action) for action in legal), dtype=np.int32
    ).tobytes())
    digest.update(str(int(player)).encode("ascii"))
    digest.update(str(int(
        getattr(self.args, "belief_table_liveness_seed_salt", 0)
    )).encode("ascii"))
    return int.from_bytes(digest.digest()[:4], byteorder="little", signed=False)

  def _sample_information_state_worlds(self, state, player, legal):
    requested = max(
        1, int(getattr(self.args, "belief_table_liveness_samples", 16))
    )
    rng = np.random.RandomState(
        self._information_state_seed(state, player, legal)
    )
    observation = np.asarray(
        state.observation_tensor(player), dtype=np.float32
    )
    legal_set = set(int(action) for action in legal)
    worlds = []
    for _ in range(requested):
      try:
        sampled = state.resample_from_infostate(player, rng)
      except Exception:
        self._decision_stats["belief_table_liveness_sample_failures"] += 1
        continue
      sampled_observation = np.asarray(
          sampled.observation_tensor(player), dtype=np.float32
      )
      sampled_legal = set(int(action) for action in sampled.legal_actions(player))
      if (
          not np.array_equal(sampled_observation, observation)
          or sampled_legal != legal_set
      ):
        self._decision_stats[
            "belief_table_liveness_infostate_mismatches"
        ] += 1
        continue
      worlds.append(sampled)
    return worlds, requested

  @staticmethod
  def _table_scalar(row):
    if bool(row.get("table_terminal_after", False)):
      return -6.0 if bool(row.get("table_any_paradox_after", False)) else 6.0
    dead = float(row.get("table_dead_margin_players_after", 0.0))
    min_margin = float(row.get("table_min_margin_after", -99.0))
    current_margin = float(row.get("table_current_margin_after", -99.0))
    total_margin = float(row.get("table_total_margin_after", -99.0))
    return float(
        -7.0 * dead
        + 1.10 * max(-6.0, min(min_margin, 8.0))
        + 0.25 * max(-6.0, min(current_margin, 8.0))
        + 0.04 * max(-24.0, min(total_margin, 40.0))
    )

  @staticmethod
  def _teacher_key(row, action, base_action):
    liveness_key = action_liveness_key(
        row, int(action), base_action=int(base_action)
    )
    return (
        0 if bool(row.get("table_any_paradox_after", False)) else 1,
        -int(row.get("table_dead_margin_players_after", 0)),
        int(row.get("table_min_margin_after", -99)),
        int(row.get("table_current_margin_after", -99)),
        int(row.get("table_total_margin_after", -99)),
        1 if bool(row.get("table_terminal_after", False)) else 0,
        -int(row.get("exact_forced_paradox_players_after", 0)),
        int(row.get("exact_min_nonparadox_legal_after", 0)),
        int(row.get("exact_current_nonparadox_legal_after", 0)),
        int(row.get("exact_total_nonparadox_legal_after", 0)),
        0 if bool(row.get("lost_led_token", False)) else 1,
    ) + tuple(int(value) for value in liveness_key)

  def _score_world(self, sampled, player, legal, base_action, policy):
    rows = {}
    scalars = {}
    for action in legal:
      row = action_liveness_certificate(
          sampled, player, int(action), base_action=int(base_action)
      )
      row.update(exact_legal_pressure_after_action(sampled, int(action)))
      row.update(exact_table_liveness_after_action(sampled, int(action)))
      rows[int(action)] = row
      scalars[int(action)] = self._table_scalar(row)
    teacher_best = max(
        legal,
        key=lambda action: (
            self._teacher_key(rows[int(action)], action, base_action),
            float(policy[int(action)]),
            -int(action),
        ),
    )
    return scalars, int(teacher_best)

  @staticmethod
  def _paired_summary(values):
    array = np.asarray(values, dtype=np.float64)
    mean_delta = float(np.mean(array)) if array.size else 0.0
    stderr = (
        float(np.std(array, ddof=1) / math.sqrt(array.size))
        if array.size > 1 else 0.0
    )
    return mean_delta, stderr

  def _score_actions(self, state, player, legal, base_action, policy):
    worlds, requested = self._sample_information_state_worlds(
        state, player, legal
    )
    configured_minimum = int(
        getattr(self.args, "belief_table_liveness_min_valid_samples", 0)
    )
    minimum = requested if configured_minimum <= 0 else configured_minimum
    if len(worlds) < minimum:
      self._decision_stats["belief_table_liveness_insufficient_worlds"] += 1
      return None
    scalar_values = {int(action): [] for action in legal}
    teacher_votes = {int(action): [] for action in legal}
    for sampled in worlds:
      scalars, teacher_best = self._score_world(
          sampled, player, legal, base_action, policy
      )
      for action in legal:
        scalar_values[int(action)].append(float(scalars[int(action)]))
        teacher_votes[int(action)].append(
            1.0 if int(action) == int(teacher_best) else 0.0
        )
    self._decision_stats["belief_table_liveness_worlds_scored"] += len(worlds)
    scalar_means = {
        int(action): float(np.mean(values))
        for action, values in scalar_values.items()
    }
    vote_shares = {
        int(action): float(np.mean(values))
        for action, values in teacher_votes.items()
    }
    if self.aggregation == "scalar_mean":
      selected = max(
          legal,
          key=lambda action: (
              scalar_means[int(action)],
              float(policy[int(action)]),
              -int(action),
          ),
      )
      paired_values = [
          selected_value - base_value
          for selected_value, base_value in zip(
              scalar_values[int(selected)], scalar_values[int(base_action)]
          )
      ]
    elif self.aggregation == "scalar_quantile":
      scalar_quantiles = {
          int(action): float(np.quantile(values, self.quantile))
          for action, values in scalar_values.items()
      }
      selected = max(
          legal,
          key=lambda action: (
              scalar_quantiles[int(action)],
              scalar_means[int(action)],
              float(policy[int(action)]),
              -int(action),
          ),
      )
      paired_values = [
          selected_value - base_value
          for selected_value, base_value in zip(
              scalar_values[int(selected)], scalar_values[int(base_action)]
          )
      ]
    else:
      selected = max(
          legal,
          key=lambda action: (
              vote_shares[int(action)],
              scalar_means[int(action)],
              float(policy[int(action)]),
              -int(action),
          ),
      )
      paired_values = [
          selected_value - base_value
          for selected_value, base_value in zip(
              teacher_votes[int(selected)], teacher_votes[int(base_action)]
          )
      ]
    mean_delta, stderr = self._paired_summary(paired_values)
    z_value = max(
        0.0,
        float(getattr(self.args, "belief_table_liveness_paired_lcb_z", 1.0)),
    )
    paired_lcb = float(mean_delta - z_value * stderr)
    return {
        "selected_action": int(selected),
        "worlds": int(len(worlds)),
        "mean_delta": float(mean_delta),
        "stderr": float(stderr),
        "paired_lcb": float(paired_lcb),
        "best_share": float(vote_shares[int(selected)]),
        "base_share": float(vote_shares[int(base_action)]),
        "selected_scalar_mean": float(scalar_means[int(selected)]),
        "base_scalar_mean": float(scalar_means[int(base_action)]),
    }

  def _within_policy_gap(self, policy, base_action, selected_action):
    max_gap = float(
        getattr(
            self.args, "belief_table_liveness_max_policy_log_gap", -1.0
        )
    )
    if max_gap < 0:
      return True
    gap = math.log(max(float(policy[int(base_action)]), 1e-12)) - math.log(
        max(float(policy[int(selected_action)]), 1e-12)
    )
    return float(gap) <= max_gap

  def _append_override_sample(
      self, state, player, legal, base_action, score, shadow
  ):
    samples = self._decision_stats[
        "belief_table_liveness_override_samples"
    ]
    limit = int(
        getattr(self.args, "belief_table_liveness_sample_limit", 20) or 0
    )
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": _phase_name(state).lower(),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "legal_actions": [int(action) for action in legal],
        "base_action": int(base_action),
        "selected_action": int(score["selected_action"]),
        "shadow": bool(shadow),
        "worlds": int(score["worlds"]),
        "mean_delta": float(score["mean_delta"]),
        "stderr": float(score["stderr"]),
        "paired_lcb": float(score["paired_lcb"]),
        "best_share": float(score["best_share"]),
        "base_share": float(score["base_share"]),
    })

  def step(self, state, player):
    legal = [int(action) for action in state.legal_actions(player)]
    legal_count = len(legal)
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return int(setup_action)
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    base_action = int(self.base_bot.step(state.clone(), player))
    policy = self._policy(state, player)
    if base_action not in legal:
      base_action = int(max(
          legal, key=lambda action: (float(policy[action]), -int(action))
      ))
    if not self._enabled_for_state(state):
      self._decision_stats["belief_table_liveness_disabled"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    self._decision_stats["belief_table_liveness_considered"] += 1
    trick = str(int(getattr(state, "_trick_number", -1)))
    _inc_counter(self._decision_stats["belief_table_liveness_by_trick"], trick)
    score = self._score_actions(
        state, player, legal, base_action, policy
    )
    if score is None:
      self._decision_stats["belief_table_liveness_base_kept"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    selected_action = int(score["selected_action"])
    self._decision_stats["belief_table_liveness_paired_delta_sum"] += float(
        score["mean_delta"]
    )
    self._decision_stats["belief_table_liveness_paired_lcb_sum"] += float(
        score["paired_lcb"]
    )
    self._decision_stats["belief_table_liveness_best_share_sum"] += float(
        score["best_share"]
    )
    should_override = selected_action != base_action
    min_lcb = float(
        getattr(self.args, "belief_table_liveness_min_paired_lcb", 0.0)
    )
    if should_override and float(score["paired_lcb"]) < min_lcb:
      self._decision_stats["belief_table_liveness_blocked_paired_lcb"] += 1
      should_override = False
    min_best_share = float(
        getattr(self.args, "belief_table_liveness_min_best_share", 0.0)
    )
    if should_override and float(score["best_share"]) < min_best_share:
      self._decision_stats["belief_table_liveness_blocked_best_share"] += 1
      should_override = False
    if should_override and not self._within_policy_gap(
        policy, base_action, selected_action
    ):
      self._decision_stats["belief_table_liveness_blocked_policy_gap"] += 1
      should_override = False
    shadow = bool(
        getattr(self.args, "belief_table_liveness_shadow_only", False)
    )
    final_action = int(base_action)
    if should_override:
      self._append_override_sample(
          state, player, legal, base_action, score, shadow=shadow
      )
      if shadow:
        self._decision_stats["belief_table_liveness_shadow_overrides"] += 1
      else:
        final_action = selected_action
        self._decision_stats["belief_table_liveness_overrides"] += 1
        _inc_counter(
            self._decision_stats[
                "belief_table_liveness_overrides_by_trick"
            ],
            trick,
        )
    else:
      self._decision_stats["belief_table_liveness_base_kept"] += 1
    _record_decision(self._decision_stats, state, final_action, legal_count)
    return int(final_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    considered = float(
        stats.get("belief_table_liveness_considered", 0) or 0
    )
    if considered:
      stats["belief_table_liveness_override_rate_when_considered"] = round(
          float(stats.get("belief_table_liveness_overrides", 0)) / considered,
          6,
      )
      stats["belief_table_liveness_paired_delta_avg"] = round(
          float(stats.get("belief_table_liveness_paired_delta_sum", 0.0))
          / considered,
          6,
      )
      stats["belief_table_liveness_paired_lcb_avg"] = round(
          float(stats.get("belief_table_liveness_paired_lcb_sum", 0.0))
          / considered,
          6,
      )
      stats["belief_table_liveness_best_share_avg"] = round(
          float(stats.get("belief_table_liveness_best_share_sum", 0.0))
          / considered,
          6,
      )
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
        "liveness_teacher_base_table_min_margin_sum": 0.0,
        "liveness_teacher_selected_table_min_margin_sum": 0.0,
        "liveness_teacher_base_table_dead_players_sum": 0.0,
        "liveness_teacher_selected_table_dead_players_sum": 0.0,
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
    key_order = str(
        getattr(self.args, "liveness_teacher_exact_key_order", "lane_first")
        or "lane_first"
    )
    use_table = key_order in (
        "table_liveness_first",
        "late_table_liveness_first",
    )
    for action in legal:
      row = action_liveness_certificate(
          state, player, action, base_action=base_action
      )
      if use_exact:
        row.update(exact_legal_pressure_after_action(state, action))
      if use_table:
        row.update(exact_table_liveness_after_action(state, action))
      scored[int(action)] = row
    return scored

  def _teacher_key(self, row):
    liveness_key = tuple(int(value) for value in row.get("liveness_key", []))
    if not bool(getattr(self.args, "liveness_teacher_exact_legal_pressure", False)):
      return liveness_key
    forced = -int(row.get("exact_forced_paradox_players_after", 0))
    min_legal = int(row.get("exact_min_nonparadox_legal_after", 0))
    current_legal = int(row.get("exact_current_nonparadox_legal_after", 0))
    total_legal = int(row.get("exact_total_nonparadox_legal_after", 0))
    preserves_led = 0 if bool(row.get("lost_led_token", False)) else 1
    terminal = 1 if bool(row.get("exact_terminal_after", False)) else 0
    key_order = str(
        getattr(self.args, "liveness_teacher_exact_key_order", "lane_first")
        or "lane_first"
    )
    table_first = key_order == "table_liveness_first"
    if key_order == "late_table_liveness_first":
      led_color = row.get("led_color", None)
      trick_number = int(row.get("trick_number", -1))
      table_first = trick_number >= 5 and led_color in (None, "R")
    if table_first:
      table_terminal = 1 if bool(row.get("table_terminal_after", False)) else 0
      table_no_paradox = 0 if bool(row.get("table_any_paradox_after", False)) else 1
      table_dead = -int(row.get("table_dead_margin_players_after", 0))
      table_min_margin = int(row.get("table_min_margin_after", -99))
      table_current_margin = int(row.get("table_current_margin_after", -99))
      table_total_margin = int(row.get("table_total_margin_after", -99))
      return (
          table_no_paradox,
          table_dead,
          table_min_margin,
          table_current_margin,
          table_total_margin,
          table_terminal,
          forced,
          min_legal,
          current_legal,
          total_legal,
          preserves_led,
      ) + liveness_key
    pressure_first = key_order == "pressure_first"
    if key_order == "late_red_pressure_first":
      led_color = row.get("led_color", None)
      trick_number = int(row.get("trick_number", -1))
      pressure_first = trick_number >= 5 and led_color in (None, "R")
    if pressure_first:
      return (
          terminal,
          forced,
          min_legal,
          current_legal,
          total_legal,
          preserves_led,
      ) + liveness_key
    return (
        terminal,
        forced,
        preserves_led,
        min_legal,
        current_legal,
        total_legal,
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
    self._decision_stats["liveness_teacher_base_table_min_margin_sum"] += float(
        base_row.get("table_min_margin_after", 0.0)
    )
    self._decision_stats["liveness_teacher_selected_table_min_margin_sum"] += float(
        selected_row.get("table_min_margin_after", 0.0)
    )
    self._decision_stats["liveness_teacher_base_table_dead_players_sum"] += float(
        base_row.get("table_dead_margin_players_after", 0.0)
    )
    self._decision_stats["liveness_teacher_selected_table_dead_players_sum"] += float(
        selected_row.get("table_dead_margin_players_after", 0.0)
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
      "game_context_mode": str(
          payload.get("game_context_mode")
          or saved_args.get("game_context_mode")
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
    self.base_mode = str(getattr(model_args, "improve_base_mode", "policy"))
    if self.base_mode == "policy":
      self.base_bot = None
    elif self.base_mode == "q_policy":
      self.base_bot = AZQPolicyBot(model, f"{name}_improve_base", device, model_args)
    elif self.base_mode == "liveness_shield":
      self.base_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_improve_base", device, model_args
      )
    else:
      raise ValueError(f"Unsupported improve_base_mode: {self.base_mode}")
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
        "improve_guard_led_follow_filtered": 0,
        "improve_guard_led_follow_base_follow_states": 0,
        "improve_guard_led_follow_responsible_red_kept": 0,
        "improve_guard_led_follow_nonurgent_red_filtered": 0,
        "improve_policy_gap_filtered": 0,
        "improve_base_off_led_filtered": 0,
        "improve_candidate_color_filtered": 0,
        "improve_responsible_red_win_filtered": 0,
        "improve_by_phase": {},
        "improve_overrides_by_phase": {},
        "improve_shadow_overrides_by_phase": {},
        "improve_override_samples": [],
        "improve_shadow_override_samples": [],
        "improve_max_prob_sum": 0.0,
        "improve_max_safe_prob_sum": 0.0,
        "improve_base_mode": self.base_mode,
    })

  @staticmethod
  def _phase_name(state):
    return _phase_name(state)

  @staticmethod
  def _action_color(state, action):
    action = int(action)
    if action == 999:
      return None
    num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
    if num_card_types <= 0:
      return None
    color_idx = action // num_card_types
    color_names = ["R", "B", "Y", "G"]
    if 0 <= color_idx < len(color_names):
      return color_names[color_idx]
    return None

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
    trick_number = int(getattr(state, "_trick_number", -1))
    if min_trick >= 0 and trick_number < min_trick:
      return False, "trick_number"
    trick_spec = str(getattr(self.args, "improve_trick_numbers", "") or "").strip()
    if trick_spec:
      allowed_tricks = {
          int(part.strip()) for part in trick_spec.split(",") if part.strip()
      }
      if trick_number not in allowed_tricks:
        return False, "trick_number_filter"
    led_spec = str(getattr(self.args, "improve_led_colors", "") or "").strip()
    if led_spec:
      allowed_led = _split_phase_names(led_spec)
      led_color = str(getattr(state, "_led_color", "") or "").lower()
      if led_color not in allowed_led:
        return False, "led_color"
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

  def _guard_led_follow_candidates(self, state, player, policy_action, candidates):
    if not bool(getattr(self.args, "improve_guard_led_follow", False)):
      return candidates
    led_color = getattr(state, "_led_color", None)
    if led_color not in ("B", "Y", "G"):
      return candidates
    if self._action_color(state, policy_action) != led_color:
      return candidates
    self._decision_stats["improve_guard_led_follow_base_follow_states"] += 1
    kept = []
    for action in candidates:
      if self._action_color(state, action) == led_color:
        kept.append(action)
        continue
      row = action_liveness_certificate(
          state, player, action, base_action=policy_action
      )
      if bool(row.get("responsible_red_win_switch", False)):
        if (
            bool(
                getattr(
                    self.args,
                    "improve_guard_led_follow_red_win_must_be_urgent",
                    False,
                )
            )
            and not _must_win_current_trick_to_hit_prediction(state, player)
        ):
          self._decision_stats[
              "improve_guard_led_follow_nonurgent_red_filtered"
          ] += 1
          self._decision_stats["improve_guard_led_follow_filtered"] += 1
          continue
        kept.append(action)
        self._decision_stats["improve_guard_led_follow_responsible_red_kept"] += 1
      else:
        self._decision_stats["improve_guard_led_follow_filtered"] += 1
    return kept

  def _guard_candidate_shape(self, state, player, policy_action, candidates):
    if not candidates:
      return candidates
    led_color = getattr(state, "_led_color", None)
    if bool(getattr(self.args, "improve_require_base_off_led", False)):
      if led_color not in ("R", "B", "Y", "G"):
        self._decision_stats["improve_base_off_led_filtered"] += len(candidates)
        return []
      if self._action_color(state, policy_action) == led_color:
        self._decision_stats["improve_base_off_led_filtered"] += len(candidates)
        return []
    color_spec = str(getattr(self.args, "improve_candidate_colors", "") or "").strip()
    if color_spec:
      allowed_colors = _split_phase_names(color_spec)
      kept = []
      for action in candidates:
        color = str(self._action_color(state, action) or "").lower()
        if color in allowed_colors:
          kept.append(action)
        else:
          self._decision_stats["improve_candidate_color_filtered"] += 1
      candidates = kept
    if bool(getattr(self.args, "improve_require_responsible_red_win", False)):
      kept = []
      for action in candidates:
        row = action_liveness_certificate(
            state, player, action, base_action=policy_action
        )
        if bool(row.get("responsible_red_win_switch", False)):
          kept.append(action)
        else:
          self._decision_stats["improve_responsible_red_win_filtered"] += 1
      candidates = kept
    return candidates

  def _guard_policy_gap_candidates(self, policy, policy_action, candidates):
    max_gap = float(getattr(self.args, "improve_max_policy_log_gap", -1.0))
    if max_gap < 0:
      return candidates
    floor = 1e-12
    base_logp = float(np.log(max(float(policy[policy_action]), floor)))
    kept = []
    for action in candidates:
      candidate_logp = float(np.log(max(float(policy[action]), floor)))
      if base_logp - candidate_logp <= max_gap:
        kept.append(action)
      else:
        self._decision_stats["improve_policy_gap_filtered"] += 1
    return kept

  def _base_action(self, state, player, legal, policy_action):
    if self.base_bot is None:
      return int(policy_action)
    action = int(self.base_bot.step(state.clone(), player))
    if action not in legal:
      return int(policy_action)
    return action

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
    game_context_mode = str(
        self.improve_scorer.get("game_context_mode", "none") or "none"
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
            game_context_mode=game_context_mode,
            trick_number=int(getattr(state, "_trick_number", -1)),
            match_round=int(getattr(state, "_match_round", -1)),
            player=int(player),
            players=int(getattr(self.args, "players", 3)),
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
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
    raw_policy_action = int(max(legal, key=lambda action: policy[action]))
    policy_action = self._base_action(state, player, legal, raw_policy_action)
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
    candidates = self._guard_led_follow_candidates(
        state, player, policy_action, candidates
    )
    candidates = self._guard_candidate_shape(
        state, player, policy_action, candidates
    )
    candidates = self._guard_policy_gap_candidates(policy, policy_action, candidates)
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


def load_intralane_filter(path, device):
  payload = torch.load(path, map_location=device, weights_only=False)
  saved_args = payload.get("args", {})
  feature_size = int(payload.get("feature_size", len(payload["feature_mean"])))
  hidden = int(saved_args.get("hidden", 128))
  dropout = float(saved_args.get("dropout", 0.0))
  model = IntraLaneRankFilter(feature_size, hidden, dropout).to(device)
  model.load_state_dict(payload["model_state"])
  model.eval()
  mean_arr = np.asarray(payload["feature_mean"], dtype=np.float32)
  std_arr = np.asarray(payload["feature_std"], dtype=np.float32)
  std_arr = np.where(std_arr < 1e-6, 1.0, std_arr)
  return {
      "model": model,
      "feature_mean": mean_arr,
      "feature_std": std_arr,
      "feature_size": feature_size,
      "args": saved_args,
      "round_context_schema": list(payload.get("round_context_schema", []) or []),
      "intralane_context_schema": list(
          payload.get("intralane_context_schema", []) or []
      ),
      "replay_sources": payload.get("replay_sources", []),
  }


def load_pairwise_boundary(path, device):
  payload = torch.load(path, map_location=device, weights_only=False)
  saved_args = payload.get("args", {})
  feature_size = int(payload.get("feature_size", len(payload["feature_mean"])))
  hidden = int(saved_args.get("hidden", 128))
  model = PairwiseBoundary(feature_size, hidden).to(device)
  model.load_state_dict(payload["model_state"])
  model.eval()
  mean_arr = np.asarray(payload["feature_mean"], dtype=np.float32)
  std_arr = np.asarray(payload["feature_std"], dtype=np.float32)
  std_arr = np.where(std_arr < 1e-6, 1.0, std_arr)
  return {
      "model": model,
      "feature_mean": mean_arr,
      "feature_std": std_arr,
      "feature_size": feature_size,
      "args": saved_args,
      "replay_sources": payload.get("replay_sources", []),
  }


def load_action_set_transformer(path, device):
  payload = torch.load(path, map_location=device, weights_only=False)
  saved_args = payload.get("args", {})
  feature_size = int(payload.get("feature_size", len(payload["feature_mean"])))
  round_context_schema = list(payload.get("round_context_schema", []) or [])
  feature_schema_version = str(payload.get("feature_schema_version", "") or "")
  if round_context_schema and (
      feature_schema_version != "legal_set_transformer_v1_public_history_v1"
      or tuple(round_context_schema) != tuple(PUBLIC_HISTORY_ACTION_SCHEMA)
  ):
    raise ValueError(
        "live ActionSetTransformer deployment only supports no context or the "
        "exact deployable public-history schema; got "
        f"version={feature_schema_version!r}, schema={round_context_schema!r}"
    )
  model = ActionSetTransformer(
      feature_size,
      int(saved_args.get("hidden", 128)),
      int(saved_args.get("layers", 2)),
      int(saved_args.get("heads", 4)),
      float(saved_args.get("dropout", 0.0)),
  ).to(device)
  model.load_state_dict(payload["model_state"])
  model.eval()
  mean_arr = np.asarray(payload["feature_mean"], dtype=np.float32)
  std_arr = np.asarray(payload["feature_std"], dtype=np.float32)
  std_arr = np.where(std_arr < 1e-6, 1.0, std_arr)
  return {
      "model": model,
      "feature_mean": mean_arr,
      "feature_std": std_arr,
      "feature_size": feature_size,
      "args": saved_args,
      "feature_schema_version": feature_schema_version,
      "round_context_schema": round_context_schema,
      "replay_sources": payload.get("replay_sources", []),
  }


def load_proposal_veto_selector(path, device):
  payload = torch.load(path, map_location=device, weights_only=False)
  saved_args = payload.get("args", {})
  feature_size = int(payload.get("feature_size", len(payload["mean"])))
  model_kind = str(
      payload.get("proposal_model_kind")
      or ("deepset" if str(payload.get("kind", "")).endswith("deepset") else "mlp")
  )
  hidden = int(payload.get("hidden", saved_args.get("hidden", 96)))
  dropout = float(payload.get("dropout", saved_args.get("dropout", 0.0)))
  if model_kind == "deepset":
    model = ProposalVetoDeepSet(feature_size, hidden, dropout).to(device)
  elif model_kind == "mlp":
    model = ProposalVetoMLP(feature_size, hidden, dropout).to(device)
  else:
    raise ValueError(f"unsupported proposal veto model kind: {model_kind}")
  state_dict = payload.get("state_dict", payload.get("model_state"))
  if state_dict is None:
    raise ValueError(f"proposal veto checkpoint has no state_dict: {path}")
  model.load_state_dict(state_dict)
  model.eval()
  mean_arr = np.asarray(payload["mean"], dtype=np.float32)
  std_arr = np.asarray(payload["std"], dtype=np.float32)
  std_arr = np.where(std_arr < 1e-6, 1.0, std_arr)
  return {
      "model": model,
      "model_kind": model_kind,
      "feature_size": feature_size,
      "feature_mean": mean_arr,
      "feature_std": std_arr,
      "args": saved_args,
      "approve_label": payload.get("approve_label", ""),
  }


class AZPairwiseBoundaryBot:
  """Frozen base policy with an abstaining pairwise override boundary."""

  def __init__(self, model, boundary, name, device, model_args):
    self.name = name
    self.model = model
    self.boundary = boundary
    self.device = device
    self.args = model_args
    self.base_mode = str(
        getattr(model_args, "pairwise_boundary_base_mode", "policy")
    )
    if self.base_mode == "policy":
      self.base_bot = None
    elif self.base_mode == "q_policy":
      self.base_bot = AZQPolicyBot(model, f"{name}_pairwise_base", device, model_args)
    elif self.base_mode == "liveness_shield":
      self.base_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_pairwise_base", device, model_args
      )
    else:
      raise ValueError(
          f"Unsupported pairwise_boundary_base_mode: {self.base_mode}"
      )
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "pairwise_boundary_base_mode": self.base_mode,
        "pairwise_boundary_disabled": 0,
        "pairwise_boundary_disabled_by_reason": {},
        "pairwise_boundary_considered": 0,
        "pairwise_boundary_abstained": 0,
        "pairwise_boundary_overrides": 0,
        "pairwise_boundary_shadow_overrides": 0,
        "pairwise_boundary_scored_candidates": 0,
        "pairwise_boundary_forced_lane_filtered": 0,
        "pairwise_boundary_policy_gap_filtered": 0,
        "pairwise_boundary_by_phase": {},
        "pairwise_boundary_overrides_by_phase": {},
        "pairwise_boundary_shadow_overrides_by_phase": {},
        "pairwise_boundary_override_samples": [],
        "pairwise_boundary_shadow_override_samples": [],
        "pairwise_boundary_best_margin_sum": 0.0,
        "pairwise_boundary_base_score_sum": 0.0,
        "pairwise_boundary_selected_score_sum": 0.0,
    })

  def _enabled_for_state(self, state, legal_count):
    spec = str(getattr(self.args, "pairwise_boundary_phases", "play") or "")
    if spec:
      allowed = _split_phase_names(spec)
      if _phase_name(state).lower() not in allowed:
        return False, "phase"
    min_legal = int(getattr(self.args, "pairwise_boundary_min_legal_count", 0) or 0)
    max_legal = int(getattr(self.args, "pairwise_boundary_max_legal_count", 0) or 0)
    if min_legal > 0 and legal_count < min_legal:
      return False, "legal_count_low"
    if max_legal > 0 and legal_count > max_legal:
      return False, "legal_count_high"
    min_trick = int(getattr(self.args, "pairwise_boundary_min_trick_number", -1))
    max_trick = int(getattr(self.args, "pairwise_boundary_max_trick_number", -1))
    trick_number = int(getattr(state, "_trick_number", -1))
    if min_trick >= 0 and trick_number < min_trick:
      return False, "trick_number_low"
    if max_trick >= 0 and trick_number > max_trick:
      return False, "trick_number_high"
    if legal_count <= 1:
      return False, "forced"
    return True, "enabled"

  def _base_action(self, state, player, legal, raw_policy_action):
    if self.base_bot is None:
      return int(raw_policy_action)
    action = int(self.base_bot.step(state.clone(), player))
    if action not in legal:
      return int(raw_policy_action)
    return int(action)

  def _candidate_actions(self, state, player, legal, policy, base_action, action_features):
    legal = [int(action) for action in legal]
    max_actions = int(getattr(self.args, "pairwise_boundary_max_actions", 0) or 0)
    if max_actions > 0:
      sampler_args = SimpleNamespace(
          counterfactual_action_max_legal=max_actions,
          counterfactual_action_top_policy=max(
              0, int(getattr(self.args, "pairwise_boundary_top_policy", 0) or 0)
          ),
          counterfactual_action_include_bots=str(
              getattr(
                  self.args,
                  "pairwise_boundary_include_bots",
                  "heuristic,heuristic_target2,heuristic_adj2",
              )
          ),
          counterfactual_action_feature_candidates=bool(
              getattr(self.args, "pairwise_boundary_feature_candidates", True)
          ),
      )
      candidates = sorted(
          int(action) for action in sampled_counterfactual_legal_actions(
              state, player, legal, sampler_args, policy=policy
          )
          if int(action) != int(base_action)
      )
    else:
      candidates = sorted(int(action) for action in legal if int(action) != int(base_action))
    if bool(
        getattr(self.args, "pairwise_boundary_exclude_forced_led_lane_negatives", True)
    ):
      filter_args = SimpleNamespace(
          force_led_lane_burn_negative=True,
          led_lane_red_exception=str(
              getattr(self.args, "pairwise_boundary_led_lane_red_exception", "win")
          ),
      )
      kept = []
      for action in candidates:
        forced, _ = _led_lane_burn_label_shape(
            action_features, int(base_action), int(action), filter_args
        )
        if forced:
          self._decision_stats["pairwise_boundary_forced_lane_filtered"] += 1
        else:
          kept.append(int(action))
      candidates = kept
    max_gap = float(
        getattr(self.args, "pairwise_boundary_max_policy_log_gap", -1.0)
    )
    if max_gap >= 0.0 and candidates:
      floor = 1e-12
      base_logp = float(np.log(max(float(policy[int(base_action)]), floor)))
      kept = []
      for action in candidates:
        candidate_logp = float(np.log(max(float(policy[int(action)]), floor)))
        if base_logp - candidate_logp <= max_gap:
          kept.append(int(action))
        else:
          self._decision_stats["pairwise_boundary_policy_gap_filtered"] += 1
      candidates = kept
    return candidates

  def _feature_for_action(
      self,
      action_features,
      legal_mask,
      observation,
      policy,
      action,
      base_action,
      legal_count,
  ):
    saved_args = self.boundary.get("args", {})
    return pair_feature_vector_from_matrix(
        action_features,
        legal_mask,
        int(action),
        int(base_action),
        target_action=int(action),
        label_margin=0.0,
        legal_count=int(legal_count),
        candidate_count=int(legal_count),
        include_label_margin=False,
        observation=observation,
        observation_feature_mode=str(
            saved_args.get("observation_feature_mode", "summary") or "summary"
        ),
        policy_context=policy,
        policy_context_mode=str(
            saved_args.get("policy_context_mode", "summary") or "summary"
        ),
        include_target_match_features=False,
    )

  def _score_candidates(
      self,
      state,
      player,
      legal,
      policy,
      base_action,
      candidates,
      action_features,
  ):
    if not candidates:
      return {}
    num_actions = state.num_distinct_actions()
    pool = sorted({int(base_action), *[int(action) for action in candidates]})
    legal_mask = np.zeros(num_actions, dtype=np.float32)
    legal_mask[pool] = 1.0
    observation = np.array(state.observation_tensor(player), dtype=np.float32)
    base_feature = self._feature_for_action(
        action_features,
        legal_mask,
        observation,
        policy,
        int(base_action),
        int(base_action),
        len(pool),
    )
    rows = [
        self._feature_for_action(
            action_features,
            legal_mask,
            observation,
            policy,
            int(action),
            int(base_action),
            len(pool),
        )
        for action in candidates
    ]
    x = np.asarray(rows, dtype=np.float32)
    base = np.asarray(base_feature, dtype=np.float32)
    feature_size = int(self.boundary["feature_size"])
    if x.shape[1] != feature_size or base.shape[0] != feature_size:
      raise ValueError(
          "pairwise boundary feature size mismatch: "
          f"{x.shape[1]} / {base.shape[0]} vs {feature_size}"
      )
    mean_arr = self.boundary["feature_mean"]
    std_arr = self.boundary["feature_std"]
    x = (x - mean_arr) / std_arr
    base = (base - mean_arr) / std_arr
    with torch.no_grad():
      x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
      base_t = torch.tensor(base, dtype=torch.float32, device=self.device)
      base_batch = base_t.unsqueeze(0).expand(x_t.shape[0], -1)
      scores = self.boundary["model"](x_t, base_batch).cpu().numpy()
      base_score = float(
          self.boundary["model"](
              base_t.unsqueeze(0), base_t.unsqueeze(0)
          ).detach().cpu().numpy()[0]
      )
    return {
        int(action): {
            "score": float(score),
            "base_score": float(base_score),
            "margin": float(score - base_score),
        }
        for action, score in zip(candidates, scores)
    }

  def _append_override_sample(
      self,
      key,
      state,
      player,
      legal,
      policy,
      base_action,
      candidate_action,
      score,
  ):
    samples = self._decision_stats.setdefault(key, [])
    limit = int(getattr(self.args, "pairwise_boundary_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": _phase_name(state).lower(),
        "legal_count": int(len(legal)),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "match_round": int(getattr(state, "_match_round", -1)),
        "base_action": int(base_action),
        "candidate_action": int(candidate_action),
        "base_policy_prob": float(policy[int(base_action)]),
        "candidate_policy_prob": float(policy[int(candidate_action)]),
        "score": float(score["score"]),
        "base_score": float(score["base_score"]),
        "margin": float(score["margin"]),
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = [int(action) for action in state.legal_actions(player)]
    legal_count = len(legal)
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    raw_policy_action = int(max(legal, key=lambda action: float(policy[int(action)])))
    base_action = self._base_action(state, player, legal, raw_policy_action)
    enabled, disabled_reason = self._enabled_for_state(state, legal_count)
    if not enabled:
      self._decision_stats["pairwise_boundary_disabled"] += 1
      _inc_counter(
          self._decision_stats["pairwise_boundary_disabled_by_reason"],
          disabled_reason,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = _phase_name(state).lower()
    self._decision_stats["pairwise_boundary_considered"] += 1
    _inc_counter(self._decision_stats["pairwise_boundary_by_phase"], phase)
    action_features = action_feature_matrix(
        state, player, state.num_distinct_actions()
    )
    candidates = self._candidate_actions(
        state, player, legal, policy, int(base_action), action_features
    )
    scores = self._score_candidates(
        state,
        player,
        legal,
        policy,
        int(base_action),
        candidates,
        action_features,
    )
    self._decision_stats["pairwise_boundary_scored_candidates"] += len(scores)
    if not scores:
      self._decision_stats["pairwise_boundary_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    best_action, best_score = max(
        scores.items(),
        key=lambda item: (
            float(item[1]["margin"]),
            float(policy[int(item[0])]),
        ),
    )
    self._decision_stats["pairwise_boundary_best_margin_sum"] += float(
        best_score["margin"]
    )
    self._decision_stats["pairwise_boundary_base_score_sum"] += float(
        best_score["base_score"]
    )
    self._decision_stats["pairwise_boundary_selected_score_sum"] += float(
        best_score["score"]
    )
    threshold = float(getattr(self.args, "pairwise_boundary_threshold", 0.0))
    if float(best_score["margin"]) < threshold:
      self._decision_stats["pairwise_boundary_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    if bool(getattr(self.args, "pairwise_boundary_shadow_only", False)):
      self._decision_stats["pairwise_boundary_shadow_overrides"] += 1
      self._decision_stats["pairwise_boundary_abstained"] += 1
      _inc_counter(
          self._decision_stats["pairwise_boundary_shadow_overrides_by_phase"],
          phase,
      )
      self._append_override_sample(
          "pairwise_boundary_shadow_override_samples",
          state,
          player,
          legal,
          policy,
          base_action,
          best_action,
          best_score,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    self._decision_stats["pairwise_boundary_overrides"] += 1
    _inc_counter(self._decision_stats["pairwise_boundary_overrides_by_phase"], phase)
    self._append_override_sample(
        "pairwise_boundary_override_samples",
        state,
        player,
        legal,
        policy,
        base_action,
        best_action,
        best_score,
    )
    _record_decision(self._decision_stats, state, best_action, legal_count)
    return int(best_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    considered = float(stats.get("pairwise_boundary_considered", 0) or 0)
    if considered:
      stats["pairwise_boundary_override_rate_when_considered"] = round(
          float(stats.get("pairwise_boundary_overrides", 0)) / considered,
          4,
      )
      stats["pairwise_boundary_shadow_override_rate_when_considered"] = round(
          float(stats.get("pairwise_boundary_shadow_overrides", 0)) / considered,
          4,
      )
      stats["pairwise_boundary_abstain_rate_when_considered"] = round(
          float(stats.get("pairwise_boundary_abstained", 0)) / considered,
          4,
      )
      stats["pairwise_boundary_scored_candidates_avg"] = round(
          float(stats.get("pairwise_boundary_scored_candidates", 0)) / considered,
          4,
      )
      stats["pairwise_boundary_best_margin_avg"] = round(
          float(stats.get("pairwise_boundary_best_margin_sum", 0.0)) / considered,
          6,
      )
      stats["pairwise_boundary_base_score_avg"] = round(
          float(stats.get("pairwise_boundary_base_score_sum", 0.0)) / considered,
          6,
      )
      stats["pairwise_boundary_selected_score_avg"] = round(
          float(stats.get("pairwise_boundary_selected_score_sum", 0.0)) / considered,
          6,
      )
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class AZActionSetTransformerBot:
  """Frozen base policy with an abstaining set-aware action override model."""

  def __init__(self, model, selector, name, device, model_args):
    self.name = name
    self.model = model
    self.selector = selector
    self.device = device
    self.args = model_args
    self.saved_args = selector.get("args", {})
    self.base_mode = str(
        getattr(model_args, "action_set_transformer_base_mode", "policy")
    )
    if self.base_mode == "policy":
      self.base_bot = None
    elif self.base_mode == "q_policy":
      self.base_bot = AZQPolicyBot(
          model, f"{name}_action_set_base", device, model_args
      )
    elif self.base_mode == "liveness_shield":
      self.base_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_action_set_base", device, model_args
      )
    else:
      raise ValueError(
          "Unsupported action_set_transformer_base_mode: "
          f"{self.base_mode}"
      )
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "action_set_transformer_base_mode": self.base_mode,
        "action_set_transformer_disabled": 0,
        "action_set_transformer_disabled_by_reason": {},
        "action_set_transformer_considered": 0,
        "action_set_transformer_abstained": 0,
        "action_set_transformer_overrides": 0,
        "action_set_transformer_shadow_overrides": 0,
        "action_set_transformer_scored_candidates": 0,
        "action_set_transformer_forced_lane_filtered": 0,
        "action_set_transformer_policy_gap_filtered": 0,
        "action_set_transformer_by_phase": {},
        "action_set_transformer_overrides_by_phase": {},
        "action_set_transformer_shadow_overrides_by_phase": {},
        "action_set_transformer_override_samples": [],
        "action_set_transformer_shadow_override_samples": [],
        "action_set_transformer_best_margin_sum": 0.0,
        "action_set_transformer_base_score_sum": 0.0,
        "action_set_transformer_selected_score_sum": 0.0,
        "action_set_transformer_selected_value_delta_sum": 0.0,
        "action_set_transformer_selected_positive_prob_sum": 0.0,
    })

  def _enabled_for_state(self, state, legal_count):
    spec = str(
        getattr(self.args, "action_set_transformer_phases", "play") or ""
    )
    if spec:
      allowed = _split_phase_names(spec)
      if _phase_name(state).lower() not in allowed:
        return False, "phase"
    min_legal = int(
        getattr(self.args, "action_set_transformer_min_legal_count", 0) or 0
    )
    max_legal = int(
        getattr(self.args, "action_set_transformer_max_legal_count", 0) or 0
    )
    if min_legal > 0 and legal_count < min_legal:
      return False, "legal_count_low"
    if max_legal > 0 and legal_count > max_legal:
      return False, "legal_count_high"
    min_trick = int(
        getattr(self.args, "action_set_transformer_min_trick_number", -1)
    )
    if min_trick < 0:
      min_trick = int(self.saved_args.get("selection_min_trick_number", -1))
    max_trick = int(
        getattr(self.args, "action_set_transformer_max_trick_number", -1)
    )
    if max_trick < 0:
      max_trick = int(self.saved_args.get("selection_max_trick_number", -1))
    trick_number = int(getattr(state, "_trick_number", -1))
    if min_trick >= 0 and trick_number < min_trick:
      return False, "trick_number_low"
    if max_trick >= 0 and trick_number > max_trick:
      return False, "trick_number_high"
    if legal_count <= 1:
      return False, "forced"
    return True, "enabled"

  def _base_action(self, state, player, legal, raw_policy_action):
    if self.base_bot is None:
      return int(raw_policy_action)
    action = int(self.base_bot.step(state.clone(), player))
    if action not in legal:
      return int(raw_policy_action)
    return int(action)

  def _candidate_actions(self, state, player, legal, policy, base_action, action_features):
    legal = [int(action) for action in legal]
    max_actions = int(
        getattr(self.args, "action_set_transformer_max_actions", 4) or 0
    )
    if max_actions > 0:
      sampler_args = SimpleNamespace(
          counterfactual_action_max_legal=max_actions,
          counterfactual_action_top_policy=max(
              0,
              int(getattr(self.args, "action_set_transformer_top_policy", 2) or 0),
          ),
          counterfactual_action_include_bots=str(
              getattr(
                  self.args,
                  "action_set_transformer_include_bots",
                  "heuristic,heuristic_target2,heuristic_adj2",
              )
          ),
          counterfactual_action_feature_candidates=bool(
              getattr(self.args, "action_set_transformer_feature_candidates", True)
          ),
      )
      candidates = sorted(
          int(action) for action in sampled_counterfactual_legal_actions(
              state, player, legal, sampler_args, policy=policy
          )
          if int(action) != int(base_action)
      )
    else:
      candidates = sorted(
          int(action) for action in legal if int(action) != int(base_action)
      )
    if bool(
        getattr(
            self.args,
            "action_set_transformer_exclude_forced_led_lane_negatives",
            True,
        )
    ):
      filter_args = SimpleNamespace(
          force_led_lane_burn_negative=bool(
              self.saved_args.get("force_led_lane_burn_negative", False)
          ),
          led_lane_red_exception=str(
              self.saved_args.get("led_lane_red_exception", "urgent_win")
          ),
      )
      kept = []
      for action in candidates:
        forced, _ = _led_lane_burn_label_shape(
            action_features, int(base_action), int(action), filter_args
        )
        if forced:
          self._decision_stats["action_set_transformer_forced_lane_filtered"] += 1
        else:
          kept.append(int(action))
      candidates = kept
    max_gap = float(
        getattr(self.args, "action_set_transformer_max_policy_log_gap", -1.0)
    )
    if max_gap >= 0.0 and candidates:
      floor = 1e-12
      base_logp = float(np.log(max(float(policy[int(base_action)]), floor)))
      kept = []
      for action in candidates:
        candidate_logp = float(np.log(max(float(policy[int(action)]), floor)))
        if base_logp - candidate_logp <= max_gap:
          kept.append(int(action))
        else:
          self._decision_stats["action_set_transformer_policy_gap_filtered"] += 1
      candidates = kept
    return candidates

  def _feature_for_action(
      self,
      action_features,
      legal_mask,
      observation,
      policy,
      action,
      base_action,
      history_features=None,
  ):
    feature_args = SimpleNamespace(**self.saved_args)
    feature = selector_pair_feature(
        feature_args,
        action_features,
        np.asarray(legal_mask, dtype=bool),
        int(action),
        int(base_action),
        observation,
        policy,
    )
    if history_features is None:
      return feature
    action = int(action)
    follows = bool(
        float(action_features[action, ACTION_FEATURE_FOLLOWS_LED_INDEX]) > 0.5
    )
    red = bool(
        float(action_features[action, ACTION_FEATURE_IS_RED_INDEX]) > 0.5
    )
    token_loss = bool(
        float(
            action_features[action, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX]
        ) > 0.5
        or float(
            action_features[
                action,
                APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"],
            ]
        ) > 0.5
    )
    return np.concatenate([
        np.asarray(feature, dtype=np.float32),
        public_history_action_features(
            history_features,
            candidate_follows_led=follows,
            candidate_is_red=red,
            candidate_token_loss=token_loss,
        ),
    ]).astype(np.float32)

  def _score_actions(
      self,
      state,
      player,
      legal,
      policy,
      base_action,
      candidates,
      action_features,
  ):
    pool = sorted({int(base_action), *[int(action) for action in candidates]})
    if len(pool) < 2:
      return {}
    num_actions = state.num_distinct_actions()
    legal_mask = np.zeros(num_actions, dtype=np.float32)
    legal_mask[[int(action) for action in legal]] = 1.0
    observation = np.array(state.observation_tensor(player), dtype=np.float32)
    history_features = None
    if self.selector.get("round_context_schema"):
      history_features = public_history_features(state, player)
    rows = [
        self._feature_for_action(
            action_features,
            legal_mask,
            observation,
          policy,
          int(action),
          int(base_action),
          history_features,
        )
        for action in pool
    ]
    x = np.asarray(rows, dtype=np.float32)
    feature_size = int(self.selector["feature_size"])
    if x.shape[1] != feature_size:
      raise ValueError(
          "action-set transformer feature size mismatch: "
          f"{x.shape[1]} vs {feature_size}"
      )
    x = (x - self.selector["feature_mean"]) / self.selector["feature_std"]
    with torch.no_grad():
      x_t = torch.tensor(
          x[None, :, :], dtype=torch.float32, device=self.device
      )
      mask_t = torch.ones((1, len(pool)), dtype=torch.bool, device=self.device)
      scores, value_p, _risk_p, positive_logit, _safe_logit = (
          self.selector["model"](x_t, mask_t)
      )
      scores_np = scores[0].detach().cpu().numpy()
      value_np = value_p[0].detach().cpu().numpy()
      positive_np = torch.sigmoid(positive_logit[0]).detach().cpu().numpy()
    base_idx = pool.index(int(base_action))
    base_score = float(scores_np[base_idx])
    return {
        int(action): {
            "score": float(score),
            "base_score": base_score,
            "margin": float(score - base_score),
            "pred_value_delta": float(value_delta),
            "pred_positive_prob": float(positive_prob),
        }
        for action, score, value_delta, positive_prob in zip(
            pool, scores_np, value_np, positive_np
        )
    }

  def _append_override_sample(
      self,
      key,
      state,
      player,
      legal,
      policy,
      base_action,
      candidate_action,
      score,
  ):
    samples = self._decision_stats.setdefault(key, [])
    limit = int(
        getattr(self.args, "action_set_transformer_sample_limit", 20) or 0
    )
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": _phase_name(state).lower(),
        "legal_count": int(len(legal)),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "match_round": int(getattr(state, "_match_round", -1)),
        "base_action": int(base_action),
        "candidate_action": int(candidate_action),
        "base_policy_prob": float(policy[int(base_action)]),
        "candidate_policy_prob": float(policy[int(candidate_action)]),
        "score": float(score["score"]),
        "base_score": float(score["base_score"]),
        "margin": float(score["margin"]),
        "pred_value_delta": float(score["pred_value_delta"]),
        "pred_positive_prob": float(score["pred_positive_prob"]),
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = [int(action) for action in state.legal_actions(player)]
    legal_count = len(legal)
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    raw_policy_action = int(max(legal, key=lambda action: float(policy[int(action)])))
    base_action = self._base_action(state, player, legal, raw_policy_action)
    enabled, disabled_reason = self._enabled_for_state(state, legal_count)
    if not enabled:
      self._decision_stats["action_set_transformer_disabled"] += 1
      _inc_counter(
          self._decision_stats["action_set_transformer_disabled_by_reason"],
          disabled_reason,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = _phase_name(state).lower()
    self._decision_stats["action_set_transformer_considered"] += 1
    _inc_counter(self._decision_stats["action_set_transformer_by_phase"], phase)
    action_features = action_feature_matrix(
        state, player, state.num_distinct_actions()
    )
    candidates = self._candidate_actions(
        state, player, legal, policy, int(base_action), action_features
    )
    scores = self._score_actions(
        state,
        player,
        legal,
        policy,
        int(base_action),
        candidates,
        action_features,
    )
    self._decision_stats["action_set_transformer_scored_candidates"] += max(
        0, len(scores) - 1
    )
    if not scores:
      self._decision_stats["action_set_transformer_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    base_score = scores[int(base_action)]
    score_margin_threshold = float(
        getattr(self.args, "action_set_transformer_score_margin_threshold", 0.0)
    )
    value_threshold = float(
        getattr(self.args, "action_set_transformer_value_threshold", 0.0)
    )
    positive_threshold = float(
        getattr(self.args, "action_set_transformer_positive_threshold", 0.5)
    )
    value_bonus = float(
        getattr(self.args, "action_set_transformer_selection_value_bonus", 0.05)
    )
    policy_tiebreak = float(
        getattr(self.args, "action_set_transformer_policy_tiebreak", 0.01)
    )
    best_action = None
    best_score = None
    best_rank = None
    for action, score in scores.items():
      if int(action) == int(base_action):
        continue
      if float(score["margin"]) < score_margin_threshold:
        continue
      if float(score["pred_value_delta"]) < value_threshold:
        continue
      if float(score["pred_positive_prob"]) < positive_threshold:
        continue
      rank = (
          float(score["margin"])
          + value_bonus * float(score["pred_value_delta"])
          + policy_tiebreak * float(policy[int(action)])
      )
      if best_action is None or rank > float(best_rank):
        best_action = int(action)
        best_score = score
        best_rank = float(rank)
    if best_action is None:
      self._decision_stats["action_set_transformer_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    self._decision_stats["action_set_transformer_best_margin_sum"] += float(
        best_score["margin"]
    )
    self._decision_stats["action_set_transformer_base_score_sum"] += float(
        base_score["score"]
    )
    self._decision_stats["action_set_transformer_selected_score_sum"] += float(
        best_score["score"]
    )
    self._decision_stats["action_set_transformer_selected_value_delta_sum"] += float(
        best_score["pred_value_delta"]
    )
    self._decision_stats[
        "action_set_transformer_selected_positive_prob_sum"
    ] += float(best_score["pred_positive_prob"])
    if bool(getattr(self.args, "action_set_transformer_shadow_only", False)):
      self._decision_stats["action_set_transformer_shadow_overrides"] += 1
      self._decision_stats["action_set_transformer_abstained"] += 1
      _inc_counter(
          self._decision_stats["action_set_transformer_shadow_overrides_by_phase"],
          phase,
      )
      self._append_override_sample(
          "action_set_transformer_shadow_override_samples",
          state,
          player,
          legal,
          policy,
          base_action,
          best_action,
          best_score,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    self._decision_stats["action_set_transformer_overrides"] += 1
    _inc_counter(
        self._decision_stats["action_set_transformer_overrides_by_phase"],
        phase,
    )
    self._append_override_sample(
        "action_set_transformer_override_samples",
        state,
        player,
        legal,
        policy,
        base_action,
        best_action,
        best_score,
    )
    _record_decision(self._decision_stats, state, best_action, legal_count)
    return int(best_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    considered = float(
        stats.get("action_set_transformer_considered", 0) or 0
    )
    if considered:
      stats["action_set_transformer_override_rate_when_considered"] = round(
          float(stats.get("action_set_transformer_overrides", 0)) / considered,
          4,
      )
      stats["action_set_transformer_shadow_override_rate_when_considered"] = round(
          float(stats.get("action_set_transformer_shadow_overrides", 0))
          / considered,
          4,
      )
      stats["action_set_transformer_abstain_rate_when_considered"] = round(
          float(stats.get("action_set_transformer_abstained", 0)) / considered,
          4,
      )
      stats["action_set_transformer_scored_candidates_avg"] = round(
          float(stats.get("action_set_transformer_scored_candidates", 0))
          / considered,
          4,
      )
      stats["action_set_transformer_best_margin_avg"] = round(
          float(stats.get("action_set_transformer_best_margin_sum", 0.0))
          / considered,
          6,
      )
      stats["action_set_transformer_base_score_avg"] = round(
          float(stats.get("action_set_transformer_base_score_sum", 0.0))
          / considered,
          6,
      )
      stats["action_set_transformer_selected_score_avg"] = round(
          float(stats.get("action_set_transformer_selected_score_sum", 0.0))
          / considered,
          6,
      )
      stats["action_set_transformer_selected_value_delta_avg"] = round(
          float(
              stats.get("action_set_transformer_selected_value_delta_sum", 0.0)
          )
          / considered,
          6,
      )
      stats["action_set_transformer_selected_positive_prob_avg"] = round(
          float(
              stats.get(
                  "action_set_transformer_selected_positive_prob_sum", 0.0
              )
          )
          / considered,
          6,
      )
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class AZProposalVetoTop1Bot:
  """Top-1 proposal-veto ranker stacked on a frozen action-set proposal slate."""

  _HISTORY_KEYS = [
      "history_available",
      "history_prior_events_norm",
      "history_prior_play_events_norm",
      "history_own_prior_play_events_norm",
      "history_other_prior_play_events_norm",
      "history_prior_follow_led_rate",
      "history_prior_switch_rate",
      "history_prior_avoidable_switch_rate",
      "history_prior_red_when_switch_rate",
      "history_prior_red_when_avoidable_switch_rate",
      "history_prior_nonred_when_avoidable_switch_rate",
      "history_prior_token_loss_rate",
      "history_prior_avoidable_token_loss_rate",
      "history_own_prior_token_loss_rate",
      "history_other_prior_token_loss_rate",
      "history_last_event_is_play",
      "history_last_event_follows_led",
      "history_last_event_is_red",
      "history_last_event_switches_off_led",
      "history_last_event_avoidable_switch",
      "history_last_event_avoidable_nonred_switch",
      "history_last_event_off_led_loses_token",
      "history_last_event_led_legal_ratio",
      "history_last_event_red_legal_ratio",
      "history_last_event_switch_legal_ratio",
  ]

  def __init__(self, model, source_selector, proposal_veto, name, device, model_args):
    self.name = name
    self.model = model
    self.source_selector = source_selector
    self.proposal_veto = proposal_veto
    self.device = device
    self.args = model_args
    self.source_args = source_selector.get("args", {})
    self.proposal_args = SimpleNamespace(**proposal_veto.get("args", {}))
    self.proposal_saved_args = proposal_veto.get("args", {})
    self.base_mode = str(getattr(model_args, "proposal_veto_base_mode", "policy"))
    if self.base_mode == "policy":
      self.base_bot = None
    elif self.base_mode == "q_policy":
      self.base_bot = AZQPolicyBot(
          model, f"{name}_proposal_veto_base", device, model_args
      )
    elif self.base_mode == "liveness_shield":
      self.base_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_proposal_veto_base", device, model_args
      )
    else:
      raise ValueError(
          f"Unsupported proposal_veto_base_mode: {self.base_mode}"
      )
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "proposal_veto_base_mode": self.base_mode,
        "proposal_veto_disabled": 0,
        "proposal_veto_disabled_by_reason": {},
        "proposal_veto_considered": 0,
        "proposal_veto_abstained": 0,
        "proposal_veto_overrides": 0,
        "proposal_veto_shadow_overrides": 0,
        "proposal_veto_scored_candidates": 0,
        "proposal_veto_augmented_candidates": 0,
        "proposal_veto_forced_lane_filtered": 0,
        "proposal_veto_policy_gap_filtered": 0,
        "proposal_veto_no_source_scores": 0,
        "proposal_veto_no_rows": 0,
        "proposal_veto_blocked_score_threshold": 0,
        "proposal_veto_blocked_margin_threshold": 0,
        "proposal_veto_blocked_source_score_threshold": 0,
        "proposal_veto_blocked_source_margin_threshold": 0,
        "proposal_veto_blocked_source_value_threshold": 0,
        "proposal_veto_blocked_source_positive_threshold": 0,
        "proposal_veto_history_available": 0,
        "proposal_veto_history_unavailable": 0,
        "proposal_veto_by_phase": {},
        "proposal_veto_overrides_by_phase": {},
        "proposal_veto_shadow_overrides_by_phase": {},
        "proposal_veto_override_samples": [],
        "proposal_veto_shadow_override_samples": [],
        "proposal_veto_selected_logit_sum": 0.0,
        "proposal_veto_selected_margin_sum": 0.0,
        "proposal_veto_selected_source_margin_sum": 0.0,
        "proposal_veto_selected_value_delta_sum": 0.0,
        "proposal_veto_selected_positive_prob_sum": 0.0,
    })

  def _enabled_for_state(self, state, legal_count):
    spec = str(getattr(self.args, "proposal_veto_phases", "play") or "")
    if spec:
      allowed = _split_phase_names(spec)
      if _phase_name(state).lower() not in allowed:
        return False, "phase"
    min_legal = int(getattr(self.args, "proposal_veto_min_legal_count", 0) or 0)
    max_legal = int(getattr(self.args, "proposal_veto_max_legal_count", 0) or 0)
    if min_legal > 0 and legal_count < min_legal:
      return False, "legal_count_low"
    if max_legal > 0 and legal_count > max_legal:
      return False, "legal_count_high"
    min_trick = int(getattr(self.args, "proposal_veto_min_trick_number", -1))
    max_trick = int(getattr(self.args, "proposal_veto_max_trick_number", -1))
    trick_number = int(getattr(state, "_trick_number", -1))
    if min_trick >= 0 and trick_number < min_trick:
      return False, "trick_number_low"
    if max_trick >= 0 and trick_number > max_trick:
      return False, "trick_number_high"
    if legal_count <= 1:
      return False, "forced"
    return True, "enabled"

  def _base_action(self, state, player, legal, raw_policy_action):
    if self.base_bot is None:
      return int(raw_policy_action)
    action = int(self.base_bot.step(state.clone(), player))
    if action not in legal:
      return int(raw_policy_action)
    return int(action)

  def _candidate_actions(self, state, player, legal, policy, base_action, action_features):
    legal = [int(action) for action in legal]
    max_actions = int(getattr(self.args, "proposal_veto_max_actions", 6) or 0)
    mode = str(
        getattr(
            self.args,
            "proposal_veto_candidate_mode",
            "source_plus_red_follow_policy",
        ) or "source_plus_red_follow_policy"
    )
    red_switch_only = mode in ("red_switch", "red_switch_only")
    if red_switch_only:
      candidates = []
    elif max_actions > 0:
      sampler_args = SimpleNamespace(
          counterfactual_action_max_legal=max_actions,
          counterfactual_action_top_policy=max(
              0,
              int(getattr(self.args, "proposal_veto_top_policy", 4) or 0),
          ),
          counterfactual_action_include_bots=str(
              getattr(
                  self.args,
                  "proposal_veto_include_bots",
                  "heuristic,heuristic_target2,heuristic_adj2",
              )
          ),
          counterfactual_action_feature_candidates=bool(
              getattr(self.args, "proposal_veto_feature_candidates", True)
          ),
      )
      candidates = sorted(
          int(action) for action in sampled_counterfactual_legal_actions(
              state, player, legal, sampler_args, policy=policy
          )
          if int(action) != int(base_action)
      )
    else:
      candidates = sorted(
          int(action) for action in legal if int(action) != int(base_action)
      )
    if mode in (
        "source_plus_red_follow_policy",
        "source_plus_red_follow",
        "augmented",
    ) or red_switch_only:
      cap = max_actions if max_actions > 0 else len(legal)
      seen = {int(action) for action in candidates}
      led_color = getattr(state, "_led_color", None)

      def add(action):
        action = int(action)
        if (
            action == int(base_action)
            or action == 999
            or action not in legal
            or action in seen
            or len(candidates) >= cap
        ):
          return
        candidates.append(action)
        seen.add(action)
        self._decision_stats["proposal_veto_augmented_candidates"] += 1

      legal_nonbase = [
          int(action) for action in legal
          if int(action) != int(base_action) and int(action) != 999
      ]
      red_switches = [
          action for action in legal_nonbase
          if _action_color(state, action) == "R"
          and (
              led_color != "R"
              if not red_switch_only
              else led_color not in (None, "R")
          )
      ]
      if red_switch_only:
        buckets = (red_switches,)
      else:
        follows_led = [
            action for action in legal_nonbase
            if led_color is not None and _action_color(state, action) == led_color
        ]
        top_policy = sorted(
            legal_nonbase,
            key=lambda action: float(policy[int(action)]),
            reverse=True,
        )
        buckets = (red_switches, follows_led, top_policy)
      for bucket in buckets:
        for action in sorted(
            bucket,
            key=lambda item: float(policy[int(item)]),
            reverse=True,
        ):
          add(action)
          if len(candidates) >= cap:
            break
        if len(candidates) >= cap:
          break
    elif mode not in ("source", "source_only"):
      raise ValueError(f"unknown proposal_veto_candidate_mode={mode}")
    candidates = sorted({int(action) for action in candidates})
    if bool(
        getattr(
            self.args,
            "proposal_veto_exclude_forced_led_lane_negatives",
            True,
        )
    ):
      filter_args = SimpleNamespace(
          force_led_lane_burn_negative=bool(
              self.source_args.get("force_led_lane_burn_negative", False)
          ),
          led_lane_red_exception=str(
              self.source_args.get("led_lane_red_exception", "urgent_win")
          ),
      )
      kept = []
      for action in candidates:
        forced, _ = _led_lane_burn_label_shape(
            action_features, int(base_action), int(action), filter_args
        )
        if forced:
          self._decision_stats["proposal_veto_forced_lane_filtered"] += 1
        else:
          kept.append(int(action))
      candidates = kept
    max_gap = float(getattr(self.args, "proposal_veto_max_policy_log_gap", -1.0))
    if max_gap >= 0.0 and candidates:
      floor = 1e-12
      base_logp = float(np.log(max(float(policy[int(base_action)]), floor)))
      kept = []
      for action in candidates:
        candidate_logp = float(np.log(max(float(policy[int(action)]), floor)))
        if base_logp - candidate_logp <= max_gap:
          kept.append(int(action))
        else:
          self._decision_stats["proposal_veto_policy_gap_filtered"] += 1
      candidates = kept
    return candidates

  def _pair_feature(
      self,
      saved_args,
      action_features,
      legal_mask,
      observation,
      policy,
      action,
      base_action,
      candidate_count,
  ):
    return pair_feature_vector_from_matrix(
        action_features,
        legal_mask,
        int(action),
        int(base_action),
        target_action=int(action),
        label_margin=0.0,
        legal_count=int(candidate_count),
        candidate_count=int(candidate_count),
        include_label_margin=False,
        observation=observation,
        observation_feature_mode=str(
            saved_args.get("observation_feature_mode", "summary") or "summary"
        ),
        policy_context=policy,
        policy_context_mode=str(
            saved_args.get("policy_context_mode", "summary") or "summary"
        ),
        include_target_match_features=False,
    )

  def _score_source_actions(
      self,
      state,
      player,
      legal,
      policy,
      base_action,
      candidates,
      action_features,
  ):
    pool = sorted({int(base_action), *[int(action) for action in candidates]})
    if len(pool) < 2:
      return {}
    num_actions = state.num_distinct_actions()
    legal_mask = np.zeros(num_actions, dtype=np.float32)
    legal_mask[[int(action) for action in legal]] = 1.0
    observation = np.array(state.observation_tensor(player), dtype=np.float32)
    rows = [
        self._pair_feature(
            self.source_args,
            action_features,
            legal_mask,
            observation,
            policy,
            int(action),
            int(base_action),
            len(pool),
        )
        for action in pool
    ]
    x = np.asarray(rows, dtype=np.float32)
    feature_size = int(self.source_selector["feature_size"])
    if x.shape[1] != feature_size:
      raise ValueError(
          "proposal-veto source feature size mismatch: "
          f"{x.shape[1]} vs {feature_size}"
      )
    x = (
        x - self.source_selector["feature_mean"]
    ) / self.source_selector["feature_std"]
    with torch.no_grad():
      x_t = torch.tensor(x[None, :, :], dtype=torch.float32, device=self.device)
      mask_t = torch.ones((1, len(pool)), dtype=torch.bool, device=self.device)
      scores, value_p, _risk_p, positive_logit, _safe_logit = (
          self.source_selector["model"](x_t, mask_t)
      )
      scores_np = scores[0].detach().cpu().numpy()
      value_np = value_p[0].detach().cpu().numpy()
      positive_np = torch.sigmoid(positive_logit[0]).detach().cpu().numpy()
    base_idx = pool.index(int(base_action))
    base_score = float(scores_np[base_idx])
    return {
        int(action): {
            "action": int(action),
            "score": float(score),
            "base_score": base_score,
            "margin": float(score - base_score),
            "pred_value_delta": float(value_delta),
            "pred_positive_prob": float(positive_prob),
            "policy_prob": float(policy[int(action)]),
        }
        for action, score, value_delta, positive_prob in zip(
            pool, scores_np, value_np, positive_np
        )
    }

  def _state_context_fields(self, state, legal):
    legal = [int(action) for action in legal]
    legal_count = float(len(legal))
    led_color = getattr(state, "_led_color", None)
    led_legal = sum(1 for action in legal if _action_color(state, action) == led_color)
    red_legal = sum(1 for action in legal if _action_color(state, action) == "R")
    switch_legal = sum(
        1 for action in legal
        if action != 999 and _action_color(state, action) not in (None, led_color)
    )
    denom = max(1.0, legal_count)
    return {
        "state_trick_number_norm": float(getattr(state, "_trick_number", 0)) / 8.0,
        "state_legal_count_norm": min(1.0, legal_count / 40.0),
        "state_decision_index_norm": 0.0,
        "state_round_index_norm": float(getattr(state, "_match_round", 0)) / 5.0,
        "state_start_player_norm": float(
            getattr(state, "_round_start_player", getattr(state, "_start_player", 0))
        ) / 4.0,
        "state_led_legal_count_norm": min(1.0, float(led_legal) / 20.0),
        "state_red_legal_count_norm": min(1.0, float(red_legal) / 20.0),
        "state_switch_legal_count_norm": min(1.0, float(switch_legal) / 40.0),
        "state_led_legal_ratio": float(led_legal) / denom,
        "state_red_legal_ratio": float(red_legal) / denom,
        "state_switch_legal_ratio": float(switch_legal) / denom,
        "state_avoidable_switch": 1.0 if led_legal > 0 and switch_legal > 0 else 0.0,
    }

  def _round_history_context_fields(self):
    return {key: 0.0 for key in self._HISTORY_KEYS}

  def _proposal_rows(
      self,
      state,
      player,
      legal,
      policy,
      base_action,
      scored_items,
      action_features,
  ):
    num_actions = state.num_distinct_actions()
    legal_mask = np.zeros(num_actions, dtype=np.float32)
    legal_mask[[int(action) for action in legal]] = 1.0
    observation = np.array(state.observation_tensor(player), dtype=np.float32)
    full_legal_count = int(len(legal))
    base_feature = self._pair_feature(
        self.proposal_saved_args,
        action_features,
        legal_mask,
        observation,
        policy,
        int(base_action),
        int(base_action),
        full_legal_count,
    )
    total = int(len(scored_items))
    context = self._state_context_fields(state, legal)
    history = self._round_history_context_fields()
    aux_values = np.zeros(num_actions, dtype=np.float32)
    aux_risks = np.zeros(num_actions, dtype=np.float32)
    if "_aux_" in str(
        getattr(self.proposal_args, "proposal_feature_mode", "") or ""
    ):
      _, _, aux_values, aux_risks, _ = model_policy_value_action_aux(
          self.model,
          state,
          player,
          num_actions,
          self.args.value_scale,
          self.device,
          need_action_values=True,
          need_action_risks=True,
      )
    rows = []
    views = []
    for rank, item in enumerate(scored_items):
      action = int(item["action"])
      candidate_feature = self._pair_feature(
          self.proposal_saved_args,
          action_features,
          legal_mask,
          observation,
          policy,
          action,
          int(base_action),
          full_legal_count,
      )
      view = {
          "candidate_action": action,
          "score_margin": float(item["margin"]),
          "margin": float(item["margin"]),
          "candidate_score": float(item["score"]),
          "score": float(item["score"]),
          "base_score": float(item["base_score"]),
          "candidate_pred_value_delta": float(item["pred_value_delta"]),
          "pred_value_delta": float(item["pred_value_delta"]),
          "candidate_pred_positive_prob": float(item["pred_positive_prob"]),
          "pred_positive_prob": float(item["pred_positive_prob"]),
          "candidate_policy_prob": float(item["policy_prob"]),
          "policy_prob": float(item["policy_prob"]),
          "base_policy_prob": float(policy[int(base_action)]),
          "candidate_count": int(total),
          "scored_candidate_count": int(total),
          "candidate_aux_action_value": float(aux_values[int(action)]),
          "base_aux_action_value": float(aux_values[int(base_action)]),
          "aux_action_value_delta": float(
              aux_values[int(action)] - aux_values[int(base_action)]
          ),
          "candidate_aux_action_paradox_risk": float(aux_risks[int(action)]),
          "base_aux_action_paradox_risk": float(aux_risks[int(base_action)]),
          "aux_action_paradox_risk_delta": float(
              aux_risks[int(action)] - aux_risks[int(base_action)]
          ),
      }
      view.update(_slate_relative_fields(item, scored_items, int(rank), total))
      view.update(context)
      view.update(history)
      rows.append(
          _proposal_feature(
              self.proposal_args,
              candidate_feature,
              base_feature,
              view,
              np.asarray(action_features[int(action)], dtype=np.float32),
              np.asarray(action_features[int(base_action)], dtype=np.float32),
          )
      )
      views.append(view)
    if not rows:
      return np.zeros((0, 0), dtype=np.float32), []
    return np.asarray(rows, dtype=np.float32), views

  def _score_proposals(self, x):
    feature_size = int(self.proposal_veto["feature_size"])
    if int(x.shape[1]) != feature_size:
      raise ValueError(
          "proposal-veto feature size mismatch: "
          f"{x.shape[1]} vs {feature_size}"
      )
    x = (
        x - self.proposal_veto["feature_mean"]
    ) / self.proposal_veto["feature_std"]
    with torch.no_grad():
      x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
      if str(self.proposal_veto.get("model_kind", "mlp")) == "deepset":
        logits = self.proposal_veto["model"](x_t, [list(range(int(x_t.shape[0])))])
      else:
        logits = self.proposal_veto["model"](x_t)
    return logits.detach().cpu().numpy().astype(np.float32)

  @staticmethod
  def _source_score_view(item):
    return {
        "candidate_action": int(item["action"]),
        "score_margin": float(item.get("margin", 0.0)),
        "margin": float(item.get("margin", 0.0)),
        "candidate_score": float(item.get("score", 0.0)),
        "score": float(item.get("score", 0.0)),
        "base_score": float(item.get("base_score", 0.0)),
        "candidate_pred_value_delta": float(item.get("pred_value_delta", 0.0)),
        "pred_value_delta": float(item.get("pred_value_delta", 0.0)),
        "candidate_pred_positive_prob": float(
            item.get("pred_positive_prob", 0.0)
        ),
        "pred_positive_prob": float(item.get("pred_positive_prob", 0.0)),
        "candidate_policy_prob": float(item.get("policy_prob", 0.0)),
        "policy_prob": float(item.get("policy_prob", 0.0)),
    }

  def _append_override_sample(
      self,
      key,
      state,
      player,
      legal,
      policy,
      base_action,
      selected_view,
      selected_logit,
      selected_margin,
  ):
    samples = self._decision_stats.setdefault(key, [])
    limit = int(getattr(self.args, "proposal_veto_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    action = int(selected_view["candidate_action"])
    samples.append({
        "player": int(player),
        "phase": _phase_name(state).lower(),
        "legal_count": int(len(legal)),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "match_round": int(getattr(state, "_match_round", -1)),
        "base_action": int(base_action),
        "candidate_action": int(action),
        "base_policy_prob": float(policy[int(base_action)]),
        "candidate_policy_prob": float(policy[int(action)]),
        "proposal_logit": float(selected_logit),
        "proposal_margin": float(selected_margin),
        "source_score": float(selected_view.get("candidate_score", 0.0)),
        "source_base_score": float(selected_view.get("base_score", 0.0)),
        "source_margin": float(selected_view.get("score_margin", 0.0)),
        "pred_value_delta": float(
            selected_view.get("candidate_pred_value_delta", 0.0)
        ),
        "pred_positive_prob": float(
            selected_view.get("candidate_pred_positive_prob", 0.0)
        ),
        "source_candidate_rank": int(selected_view.get("source_candidate_rank", 0)),
        "source_candidate_count": int(
            selected_view.get("source_candidate_count", 0)
        ),
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = [int(action) for action in state.legal_actions(player)]
    legal_count = len(legal)
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
    if legal_count == 1:
      _record_decision(self._decision_stats, state, legal[0], legal_count)
      return int(legal[0])
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    raw_policy_action = int(max(legal, key=lambda action: float(policy[int(action)])))
    base_action = self._base_action(state, player, legal, raw_policy_action)
    enabled, disabled_reason = self._enabled_for_state(state, legal_count)
    if not enabled:
      self._decision_stats["proposal_veto_disabled"] += 1
      _inc_counter(
          self._decision_stats["proposal_veto_disabled_by_reason"],
          disabled_reason,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = _phase_name(state).lower()
    self._decision_stats["proposal_veto_considered"] += 1
    self._decision_stats["proposal_veto_history_unavailable"] += 1
    _inc_counter(self._decision_stats["proposal_veto_by_phase"], phase)
    action_features = action_feature_matrix(
        state, player, state.num_distinct_actions()
    )
    candidates = self._candidate_actions(
        state, player, legal, policy, int(base_action), action_features
    )
    source_scores = self._score_source_actions(
        state,
        player,
        legal,
        policy,
        int(base_action),
        candidates,
        action_features,
    )
    if not source_scores:
      self._decision_stats["proposal_veto_no_source_scores"] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    scored_items = [
        dict(source_scores[int(action)])
        for action in candidates
        if int(action) in source_scores and int(action) != int(base_action)
    ]
    scored_items.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("pred_value_delta", 0.0)),
            float(item.get("policy_prob", 0.0)),
        ),
        reverse=True,
    )
    self._decision_stats["proposal_veto_scored_candidates"] += len(scored_items)
    if not scored_items:
      self._decision_stats["proposal_veto_no_rows"] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    selection_mode = str(
        getattr(self.args, "proposal_veto_selection_mode", "proposal_model")
        or "proposal_model"
    ).lower()
    if selection_mode in ("source", "source_score", "source_score_top1"):
      ranked_source_items = list(scored_items)
      selected_item = ranked_source_items[0]
      selected_view = self._source_score_view(selected_item)
      selected_logit = float(selected_item.get("score", 0.0))
      if len(ranked_source_items) > 1:
        second_logit = float(ranked_source_items[1].get("score", 0.0))
        selected_margin = selected_logit - second_logit
      else:
        selected_margin = selected_logit
    elif selection_mode in ("source_value", "source_value_delta"):
      ranked_source_items = sorted(
          scored_items,
          key=lambda item: (
              float(item.get("pred_value_delta", 0.0)),
              float(item.get("score", 0.0)),
              float(item.get("policy_prob", 0.0)),
          ),
          reverse=True,
      )
      selected_item = ranked_source_items[0]
      selected_view = self._source_score_view(selected_item)
      selected_logit = float(selected_item.get("pred_value_delta", 0.0))
      if len(ranked_source_items) > 1:
        second_logit = float(ranked_source_items[1].get("pred_value_delta", 0.0))
        selected_margin = selected_logit - second_logit
      else:
        selected_margin = selected_logit
    elif selection_mode in ("proposal_model", "proposal", "model"):
      x, views = self._proposal_rows(
          state,
          player,
          legal,
          policy,
          int(base_action),
          scored_items,
          action_features,
      )
      if x.shape[0] == 0:
        self._decision_stats["proposal_veto_no_rows"] += 1
        self._decision_stats["proposal_veto_abstained"] += 1
        _record_decision(self._decision_stats, state, base_action, legal_count)
        return int(base_action)
      logits = self._score_proposals(x)
      best_idx = int(np.argmax(logits))
      selected_logit = float(logits[best_idx])
      if len(logits) > 1:
        second_logit = float(np.partition(logits, -2)[-2])
        selected_margin = selected_logit - second_logit
      else:
        selected_margin = selected_logit
      selected_view = views[best_idx]
    else:
      raise ValueError(f"unknown proposal_veto_selection_mode={selection_mode}")
    selected_action = int(selected_view["candidate_action"])
    self._decision_stats["proposal_veto_selected_logit_sum"] += selected_logit
    self._decision_stats["proposal_veto_selected_margin_sum"] += selected_margin
    self._decision_stats["proposal_veto_selected_source_margin_sum"] += float(
        selected_view.get("score_margin", 0.0)
    )
    self._decision_stats["proposal_veto_selected_value_delta_sum"] += float(
        selected_view.get("candidate_pred_value_delta", 0.0)
    )
    self._decision_stats["proposal_veto_selected_positive_prob_sum"] += float(
        selected_view.get("candidate_pred_positive_prob", 0.0)
    )
    score_threshold = float(
        getattr(self.args, "proposal_veto_score_threshold", -1e9)
    )
    margin_threshold = float(
        getattr(self.args, "proposal_veto_margin_threshold", -1e9)
    )
    source_score_threshold = float(
        getattr(self.args, "proposal_veto_min_source_score", -1e9)
    )
    source_margin_threshold = float(
        getattr(self.args, "proposal_veto_min_source_margin", -1e9)
    )
    source_value_threshold = float(
        getattr(self.args, "proposal_veto_min_source_value_delta", -1e9)
    )
    source_positive_threshold = float(
        getattr(self.args, "proposal_veto_min_source_positive_prob", -1e9)
    )
    if selected_logit < score_threshold:
      self._decision_stats["proposal_veto_blocked_score_threshold"] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    if selected_margin < margin_threshold:
      self._decision_stats["proposal_veto_blocked_margin_threshold"] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    if float(selected_view.get("candidate_score", 0.0)) < source_score_threshold:
      self._decision_stats[
          "proposal_veto_blocked_source_score_threshold"
      ] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    if float(selected_view.get("score_margin", 0.0)) < source_margin_threshold:
      self._decision_stats[
          "proposal_veto_blocked_source_margin_threshold"
      ] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    if (
        float(selected_view.get("candidate_pred_value_delta", 0.0))
        < source_value_threshold
    ):
      self._decision_stats[
          "proposal_veto_blocked_source_value_threshold"
      ] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    if (
        float(selected_view.get("candidate_pred_positive_prob", 0.0))
        < source_positive_threshold
    ):
      self._decision_stats[
          "proposal_veto_blocked_source_positive_threshold"
      ] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    if bool(getattr(self.args, "proposal_veto_shadow_only", False)):
      self._decision_stats["proposal_veto_shadow_overrides"] += 1
      self._decision_stats["proposal_veto_abstained"] += 1
      _inc_counter(
          self._decision_stats["proposal_veto_shadow_overrides_by_phase"],
          phase,
      )
      self._append_override_sample(
          "proposal_veto_shadow_override_samples",
          state,
          player,
          legal,
          policy,
          base_action,
          selected_view,
          selected_logit,
          selected_margin,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    self._decision_stats["proposal_veto_overrides"] += 1
    _inc_counter(self._decision_stats["proposal_veto_overrides_by_phase"], phase)
    self._append_override_sample(
        "proposal_veto_override_samples",
        state,
        player,
        legal,
        policy,
        base_action,
        selected_view,
        selected_logit,
        selected_margin,
    )
    _record_decision(self._decision_stats, state, selected_action, legal_count)
    return int(selected_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


class AZIntraLaneRankFilterBot:
  """Conservative runtime wrapper for follow-led/no-token-loss rank allocation."""

  def __init__(self, model, intralane_filter, name, device, model_args):
    self.name = name
    self.model = model
    self.filter = intralane_filter
    self.device = device
    self.args = model_args
    self.base_mode = str(getattr(model_args, "intralane_filter_base_mode", "policy"))
    if self.base_mode == "policy":
      self.base_bot = None
    elif self.base_mode == "q_policy":
      self.base_bot = AZQPolicyBot(model, f"{name}_intralane_base", device, model_args)
    elif self.base_mode == "liveness_shield":
      self.base_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_intralane_base", device, model_args
      )
    else:
      raise ValueError(f"Unsupported intralane filter base mode: {self.base_mode}")
    self._filter_args = SimpleNamespace(
        candidate_filter=str(
            getattr(model_args, "intralane_filter_candidate_filter",
                    "follow_no_token_loss")
        )
    )
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "intralane_filter_disabled": 0,
        "intralane_filter_disabled_by_reason": {},
        "intralane_filter_considered": 0,
        "intralane_filter_abstained": 0,
        "intralane_filter_overrides": 0,
        "intralane_filter_shadow_overrides": 0,
        "intralane_filter_scored_candidates": 0,
        "intralane_filter_policy_gap_filtered": 0,
        "intralane_filter_by_phase": {},
        "intralane_filter_overrides_by_phase": {},
        "intralane_filter_shadow_overrides_by_phase": {},
        "intralane_filter_override_samples": [],
        "intralane_filter_shadow_override_samples": [],
        "intralane_filter_max_positive_prob_sum": 0.0,
        "intralane_filter_max_safe_prob_sum": 0.0,
        "intralane_filter_max_value_delta_sum": 0.0,
        "intralane_filter_min_risk_delta_sum": 0.0,
        "intralane_filter_base_mode": self.base_mode,
    })

  def _enabled_for_state(self, state, legal_count):
    spec = str(getattr(self.args, "intralane_filter_phases", "play") or "").strip()
    if spec:
      allowed = _split_phase_names(spec)
      if _phase_name(state).lower() not in allowed:
        return False, "phase"
    min_trick = int(getattr(self.args, "intralane_filter_min_trick_number", -1))
    max_trick = int(getattr(self.args, "intralane_filter_max_trick_number", -1))
    trick_number = int(getattr(state, "_trick_number", -1))
    if min_trick >= 0 and trick_number < min_trick:
      return False, "trick_number_low"
    if max_trick >= 0 and trick_number > max_trick:
      return False, "trick_number_high"
    if legal_count <= 1:
      return False, "forced"
    return True, "enabled"

  def _base_action(self, state, player, legal, raw_policy_action):
    if self.base_bot is None:
      return int(raw_policy_action)
    action = int(self.base_bot.step(state.clone(), player))
    if action not in legal:
      return int(raw_policy_action)
    return action

  def _candidate_actions(self, state, player, legal, action_features, base_action):
    legal = [int(action) for action in legal]
    if bool(getattr(
        self.args,
        "intralane_filter_require_forced_switch_red_contrast",
        False,
    )):
      if not self._has_forced_switch_red_contrast(action_features, legal):
        return [], "forced_switch_red_contrast"
    if (
        bool(getattr(self.args, "intralane_filter_require_base_allowed", True))
        and not intralane_candidate_allowed(
            self._filter_args, action_features, int(base_action)
        )
    ):
      return [], "base_filter"
    candidates = [
        int(action) for action in legal
        if int(action) != int(base_action)
        and intralane_candidate_allowed(self._filter_args, action_features, int(action))
    ]
    max_gap = float(getattr(self.args, "intralane_filter_max_policy_log_gap", -1.0))
    if max_gap < 0 or not candidates:
      return sorted(candidates), "ok"
    policy, _ = model_policy_value(
        self.model,
        state,
        player,
        state.num_distinct_actions(),
        self.args.value_scale,
        self.device,
    )
    floor = 1e-12
    base_logp = float(np.log(max(float(policy[int(base_action)]), floor)))
    kept = []
    for action in candidates:
      candidate_logp = float(np.log(max(float(policy[int(action)]), floor)))
      if base_logp - candidate_logp <= max_gap:
        kept.append(int(action))
      else:
        self._decision_stats["intralane_filter_policy_gap_filtered"] += 1
    return sorted(kept), "ok"

  @staticmethod
  def _token_loss_mask(action_features):
    appended_idx = APPENDED_ACTION_FEATURE_INDEX.get("token_loss_newly_loses_led")
    appended = (
        action_features[:, appended_idx] > 0.5
        if appended_idx is not None and appended_idx < action_features.shape[1]
        else np.zeros(action_features.shape[0], dtype=bool)
    )
    return (
        (action_features[:, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX] > 0.5)
        | appended
    )

  @classmethod
  def _has_forced_switch_red_contrast(cls, action_features, legal):
    if not legal:
      return False
    legal_arr = np.asarray([int(action) for action in legal], dtype=np.int64)
    follows = action_features[legal_arr, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
    token_loss = cls._token_loss_mask(action_features)[legal_arr]
    red = action_features[legal_arr, ACTION_FEATURE_IS_RED_INDEX] > 0.5
    red_switch = token_loss & red
    nonred_switch = token_loss & (~red)
    return bool(
        not np.any(follows)
        and np.any(red_switch)
        and np.any(nonred_switch)
    )

  def _score_candidates(
      self,
      state,
      player,
      legal,
      policy,
      base_action,
      candidates,
      action_features,
  ):
    if not candidates:
      return {}
    num_actions = state.num_distinct_actions()
    pool = sorted({int(base_action), *[int(action) for action in candidates]})
    legal_mask = np.zeros(num_actions, dtype=np.float32)
    legal_mask[pool] = 1.0
    observation = np.array(state.observation_tensor(player), dtype=np.float32)
    saved_args = self.filter.get("args", {})
    observation_feature_mode = str(
        saved_args.get("observation_feature_mode", "summary") or "summary"
    )
    policy_context_mode = str(
        saved_args.get("policy_context_mode", "summary") or "summary"
    )
    use_intralane_context = bool(self.filter.get("intralane_context_schema")) or (
        str(saved_args.get("intralane_context_mode", "none")) == "rank_scarcity"
    )
    rows = []
    for candidate in candidates:
      feature = pair_feature_vector_from_matrix(
          action_features,
          legal_mask,
          int(candidate),
          int(base_action),
          target_action=int(candidate),
          label_margin=0.0,
          legal_count=len(pool),
          candidate_count=len(pool),
          include_label_margin=False,
          observation=observation,
          observation_feature_mode=observation_feature_mode,
          policy_context=policy,
          policy_context_mode=policy_context_mode,
          include_target_match_features=False,
      )
      if use_intralane_context:
        feature = np.concatenate([
            feature,
            _intralane_context_vector(
                action_features,
                int(candidate),
                int(base_action),
                pool,
            ),
        ]).astype(np.float32)
      rows.append(feature)
    x = np.asarray(rows, dtype=np.float32)
    feature_size = int(self.filter["feature_size"])
    if x.shape[1] != feature_size:
      raise ValueError(
          "intralane filter feature size mismatch: "
          f"{x.shape[1]} vs {feature_size}"
      )
    x = (x - self.filter["feature_mean"]) / self.filter["feature_std"]
    with torch.no_grad():
      x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
      positive_logit, safe_logit, value_delta, risk_delta = (
          self.filter["model"](x_t)
      )
      positive_prob = torch.sigmoid(positive_logit).cpu().numpy()
      safe_prob = torch.sigmoid(safe_logit).cpu().numpy()
      value_np = value_delta.cpu().numpy()
      risk_np = risk_delta.cpu().numpy()
    return {
        int(action): {
            "positive_prob": float(pos_prob),
            "safe_prob": float(safe_p),
            "value_delta": float(value_d),
            "risk_delta": float(risk_d),
        }
        for action, pos_prob, safe_p, value_d, risk_d in zip(
            candidates, positive_prob, safe_prob, value_np, risk_np
        )
    }

  def _append_override_sample(
      self,
      key,
      state,
      player,
      legal,
      policy,
      base_action,
      candidate_action,
      score,
  ):
    samples = self._decision_stats.setdefault(key, [])
    limit = int(getattr(self.args, "intralane_filter_sample_limit", 20) or 0)
    if limit <= 0 or len(samples) >= limit:
      return
    samples.append({
        "player": int(player),
        "phase": _phase_name(state).lower(),
        "legal_count": int(len(legal)),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "match_round": int(getattr(state, "_match_round", -1)),
        "base_action": int(base_action),
        "candidate_action": int(candidate_action),
        "base_policy_prob": float(policy[int(base_action)]),
        "candidate_policy_prob": float(policy[int(candidate_action)]),
        "positive_prob": float(score["positive_prob"]),
        "safe_prob": float(score["safe_prob"]),
        "value_delta": float(score["value_delta"]),
        "risk_delta": float(score["risk_delta"]),
        "legal_actions": [int(action) for action in legal],
    })

  def step(self, state, player):
    legal = [int(action) for action in state.legal_actions(player)]
    legal_count = len(legal)
    setup_action = _shared_discard_decision(
        self._decision_stats, state, player, legal
    )
    if setup_action is not None:
      return setup_action
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
    raw_policy_action = int(max(legal, key=lambda action: float(policy[int(action)])))
    base_action = self._base_action(state, player, legal, raw_policy_action)
    enabled, disabled_reason = self._enabled_for_state(state, legal_count)
    if not enabled:
      self._decision_stats["intralane_filter_disabled"] += 1
      _inc_counter(
          self._decision_stats["intralane_filter_disabled_by_reason"],
          disabled_reason,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    phase = _phase_name(state).lower()
    self._decision_stats["intralane_filter_considered"] += 1
    _inc_counter(self._decision_stats["intralane_filter_by_phase"], phase)
    action_features = action_feature_matrix(
        state, player, state.num_distinct_actions()
    )
    candidates, candidate_status = self._candidate_actions(
        state, player, legal, action_features, int(base_action)
    )
    if candidate_status != "ok":
      self._decision_stats["intralane_filter_abstained"] += 1
      _inc_counter(
          self._decision_stats["intralane_filter_disabled_by_reason"],
          candidate_status,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    scores = self._score_candidates(
        state,
        player,
        legal,
        policy,
        int(base_action),
        candidates,
        action_features,
    )
    self._decision_stats["intralane_filter_scored_candidates"] += len(scores)
    if not scores:
      self._decision_stats["intralane_filter_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    pos_threshold = float(
        getattr(self.args, "intralane_filter_positive_threshold", 0.50)
    )
    safe_threshold = float(
        getattr(self.args, "intralane_filter_safe_threshold", 0.85)
    )
    value_threshold = float(
        getattr(self.args, "intralane_filter_value_threshold", 0.08)
    )
    risk_threshold = float(
        getattr(self.args, "intralane_filter_risk_threshold", -0.01)
    )
    passing = {
        action: score for action, score in scores.items()
        if float(score["positive_prob"]) >= pos_threshold
        and float(score["safe_prob"]) >= safe_threshold
        and float(score["value_delta"]) >= value_threshold
        and float(score["risk_delta"]) <= risk_threshold
    }
    best_all = max(
        scores.items(),
        key=lambda item: (
            float(item[1]["positive_prob"]),
            float(item[1]["safe_prob"]),
            float(item[1]["value_delta"]),
            -float(item[1]["risk_delta"]),
            float(policy[int(item[0])]),
        ),
    )
    self._decision_stats["intralane_filter_max_positive_prob_sum"] += float(
        best_all[1]["positive_prob"]
    )
    self._decision_stats["intralane_filter_max_safe_prob_sum"] += float(
        best_all[1]["safe_prob"]
    )
    self._decision_stats["intralane_filter_max_value_delta_sum"] += float(
        best_all[1]["value_delta"]
    )
    self._decision_stats["intralane_filter_min_risk_delta_sum"] += float(
        min(float(score["risk_delta"]) for score in scores.values())
    )
    if not passing:
      self._decision_stats["intralane_filter_abstained"] += 1
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    value_bonus = float(
        getattr(self.args, "intralane_filter_selection_value_bonus", 0.05)
    )
    risk_penalty = float(
        getattr(self.args, "intralane_filter_selection_risk_penalty", 0.5)
    )
    policy_tiebreak = float(
        getattr(self.args, "intralane_filter_policy_tiebreak", 0.01)
    )
    best_action, best_score = max(
        passing.items(),
        key=lambda item: (
            float(item[1]["positive_prob"])
            + value_bonus * float(item[1]["value_delta"])
            - risk_penalty * max(0.0, float(item[1]["risk_delta"]))
            + policy_tiebreak * float(policy[int(item[0])])
        ),
    )
    if bool(getattr(self.args, "intralane_filter_shadow_only", False)):
      self._decision_stats["intralane_filter_shadow_overrides"] += 1
      self._decision_stats["intralane_filter_abstained"] += 1
      _inc_counter(self._decision_stats["intralane_filter_shadow_overrides_by_phase"], phase)
      self._append_override_sample(
          "intralane_filter_shadow_override_samples",
          state,
          player,
          legal,
          policy,
          base_action,
          best_action,
          best_score,
      )
      _record_decision(self._decision_stats, state, base_action, legal_count)
      return int(base_action)
    self._decision_stats["intralane_filter_overrides"] += 1
    _inc_counter(self._decision_stats["intralane_filter_overrides_by_phase"], phase)
    self._append_override_sample(
        "intralane_filter_override_samples",
        state,
        player,
        legal,
        policy,
        base_action,
        best_action,
        best_score,
    )
    _record_decision(self._decision_stats, state, best_action, legal_count)
    return int(best_action)

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    considered = float(stats.get("intralane_filter_considered", 0) or 0)
    if considered:
      stats["intralane_filter_override_rate_when_considered"] = round(
          float(stats.get("intralane_filter_overrides", 0)) / considered, 4
      )
      stats["intralane_filter_shadow_override_rate_when_considered"] = round(
          float(stats.get("intralane_filter_shadow_overrides", 0)) / considered,
          4,
      )
      stats["intralane_filter_abstain_rate_when_considered"] = round(
          float(stats.get("intralane_filter_abstained", 0)) / considered, 4
      )
      stats["intralane_filter_scored_candidates_avg"] = round(
          float(stats.get("intralane_filter_scored_candidates", 0)) / considered,
          4,
      )
      stats["intralane_filter_max_positive_prob_avg"] = round(
          float(stats.get("intralane_filter_max_positive_prob_sum", 0.0))
          / considered,
          6,
      )
      stats["intralane_filter_max_safe_prob_avg"] = round(
          float(stats.get("intralane_filter_max_safe_prob_sum", 0.0))
          / considered,
          6,
      )
      stats["intralane_filter_max_value_delta_avg"] = round(
          float(stats.get("intralane_filter_max_value_delta_sum", 0.0))
          / considered,
          6,
      )
      stats["intralane_filter_min_risk_delta_avg"] = round(
          float(stats.get("intralane_filter_min_risk_delta_sum", 0.0))
          / considered,
          6,
      )
    return stats

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))


def _split_csv(value):
  return [item.strip() for item in str(value).split(",") if item.strip()]


def _competitive_provisional_winner(state, player, action):
  """Return the current trick leader using only public cards and our action."""
  action = int(action)
  player = int(player)
  num_ranks = int(getattr(state, "_num_card_types", 0) or 0)
  if action == 999 or num_ranks <= 0:
    return None
  colors = ("R", "B", "Y", "G")
  color_index, rank_index = divmod(action, num_ranks)
  if color_index >= len(colors):
    return None
  played = list(getattr(state, "_cards_played_this_trick", []))
  if player >= len(played):
    return None
  played[player] = (rank_index + 1, colors[color_index])
  visible = [(seat, card) for seat, card in enumerate(played) if card is not None]
  red = [(seat, int(card[0])) for seat, card in visible if card[1] == "R"]
  if red:
    return max(red, key=lambda item: item[1])[0]
  led_color = getattr(state, "_led_color", None) or colors[color_index]
  led = [
      (seat, int(card[0])) for seat, card in visible if card[1] == led_color
  ]
  return max(led, key=lambda item: item[1])[0] if led else None


def _competitive_publicly_unbeatable(state, player, action):
  """Check whether any remaining opponent has a public higher winning cell."""
  player = int(player)
  action = int(action)
  if _competitive_provisional_winner(state, player, action) != player:
    return False
  num_ranks = int(state._num_card_types)
  color_index, rank_index = divmod(action, num_ranks)
  board = np.asarray(state._board_ownership)
  tokens = np.asarray(state._color_tokens, dtype=bool)
  remaining = [
      seat for seat, card in enumerate(state._cards_played_this_trick)
      if seat != player and card is None
  ]
  for seat in remaining:
    if bool(tokens[seat, 0]):
      red_start = rank_index + 1 if color_index == 0 else 0
      if np.any(board[0, red_start:] == -1):
        return False
    if color_index != 0:
      led_color = getattr(state, "_led_color", None) or _action_color(state, action)
      led_index = ("R", "B", "Y", "G").index(led_color)
      if (
          bool(tokens[seat, led_index])
          and np.any(board[led_index, rank_index + 1:] == -1)
      ):
        return False
  return True


def _competitive_terminal_action_rows(state, player, legal):
  """Exhaustively score final-player actions from actor/public information."""
  if int(getattr(state, "_phase", -1)) != 3:
    return []
  if int(state._trick_number) != int(state._num_tricks) - 1:
    return []
  if sum(card is not None for card in state._cards_played_this_trick) != (
      int(state._num_players) - 1
  ):
    return []
  rows = []
  for action in legal:
    action = int(action)
    if action == 999:
      continue
    terminal = state.clone()
    terminal.apply_action(action)
    if not terminal.is_terminal():
      return []
    totals = [float(value) for value in terminal.returns()]
    final_scores = [float(value) for value in terminal.raw_round_scores()]
    own_official = (totals[int(player)], final_scores[int(player)])
    placement_points = sum(
        1.0 if own_official > (totals[seat], final_scores[seat])
        else 0.5 if own_official == (totals[seat], final_scores[seat])
        else 0.0
        for seat in range(int(state._num_players)) if seat != int(player)
    )
    rows.append({
        "action": action,
        "placement_points": float(placement_points),
        "own_total": totals[int(player)],
        "own_round_score": final_scores[int(player)],
        "own_connected_bonus": float(
            terminal._player_adjacency_bonus[int(player)]
        ),
    })
  return rows


class AZCompetitiveTacticsBot:
  """Opt-in actor-relative tactics; never inspect opponents' private hands."""

  def __init__(
      self, model, name, device, model_args, *, phase_risk_model=None
  ):
    self.name = name
    self.model = model
    self.device = device
    self.args = model_args
    self.base_mode = str(
        getattr(model_args, "competitive_tactics_base_mode", "policy") or "policy"
    )
    self.prediction_style = str(
        getattr(model_args, "competitive_tactics_prediction_style", "shared")
        or "shared"
    ).strip().lower().replace("-", "_")
    if self.prediction_style not in (
        "shared", "conservative", "aggressive", "rank_mass", "one", "two"
    ):
      raise ValueError(
          "Unsupported competitive_tactics_prediction_style: "
          f"{self.prediction_style}"
      )
    if self.base_mode == "policy":
      self.base_bot = AZPolicyBot(
          model, f"{name}_base", device, model_args.value_scale,
          use_shared_prediction=not _uses_neural_prediction(model_args),
      )
    elif self.base_mode == "q_policy":
      self.base_bot = AZQPolicyBot(
          model, f"{name}_base", device, model_args,
          phase_risk_model=phase_risk_model,
          phase_risk_phases=getattr(
              model_args, "action_paradox_phase_risk_phases", ""
          ),
      )
    elif self.base_mode == "liveness_shield":
      self.base_bot = AZLivenessShieldPolicyBot(
          model, f"{name}_base", device, model_args
      )
    else:
      raise ValueError(
          f"Unsupported competitive_tactics_base_mode: {self.base_mode}"
      )
    self.uses_neural_prediction = bool(
        self.prediction_style != "shared"
        or getattr(self.base_bot, "uses_neural_prediction", False)
    )
    self.last_tactical_override = False
    self._decision_stats = _empty_decision_stats()
    self._decision_stats.update({
        "competitive_tactics_base_mode": self.base_mode,
        "competitive_tactics_prediction_style": self.prediction_style,
        "competitive_tactics_exact_terminal_considered": 0,
        "competitive_tactics_exact_terminal_overrides": 0,
        "competitive_tactics_round_terminal_overrides": 0,
        "competitive_tactics_bid_play_considered": 0,
        "competitive_tactics_needed_trick_overrides": 0,
        "competitive_tactics_declined_excess_trick": 0,
        "competitive_tactics_bid_regime_considered": 0,
        "competitive_tactics_bid_regime_overrides": 0,
        "competitive_tactics_bid_regime_overrides_by_regime": {},
        "competitive_tactics_prediction_overrides": 0,
        "competitive_tactics_public_weak_field_decisions": 0,
    })

  def _prediction_action(self, state, player, legal):
    shared = shared_prediction_action(state, player, legal)
    if self.prediction_style == "shared":
      return None
    if shared is None:
      return None
    shared_guess = int(shared) - 100
    if self.prediction_style == "one":
      guess = 1
    elif self.prediction_style == "two":
      guess = 2
    elif self.prediction_style == "conservative":
      guess = shared_guess - 1
    elif self.prediction_style == "aggressive":
      guess = shared_guess + 1
    else:
      own_hand = np.asarray(state._hands[int(player)], dtype=np.float64)
      top = own_hand[::-1]
      weights = np.asarray((1.0, 0.55, 0.25, 0.1), dtype=np.float64)
      score = float(np.dot(top[:len(weights)], weights[:len(top)]))
      if int(player) == int(getattr(state, "_round_start_player", -1)):
        score += 0.2
      guess = int(math.floor(score + 0.5))
    guess = max(1, min(4, int(guess)))
    predictions = [int(action) for action in legal if 101 <= int(action) <= 104]
    if not predictions:
      return None
    selected = min(predictions, key=lambda action: abs(action - 100 - guess))
    if int(selected) != int(shared):
      self._decision_stats["competitive_tactics_prediction_overrides"] += 1
    return int(selected)

  def _terminal_action(self, state, player, legal, base_action):
    is_final_match_round = (
        int(getattr(state, "_match_round", -1))
        == int(getattr(state, "_num_players", 0)) - 1
    )
    use_exact = bool(
        getattr(self.args, "competitive_tactics_exact_terminal", True)
    )
    use_round = bool(
        getattr(self.args, "competitive_tactics_round_terminal", False)
    )
    if not ((is_final_match_round and use_exact) or use_round):
      return None
    rows = _competitive_terminal_action_rows(state, player, legal)
    if not rows:
      return None
    if is_final_match_round and use_exact:
      self._decision_stats["competitive_tactics_exact_terminal_considered"] += 1
      best = max(float(row["placement_points"]) for row in rows)
      tied = [
          row for row in rows if np.isclose(row["placement_points"], best)
      ]
      if any(int(row["action"]) == int(base_action) for row in tied):
        return int(base_action)
      selected = max(
          tied,
          key=lambda row: (
              row["own_total"], row["own_round_score"],
              row["own_connected_bonus"], -int(row["action"]),
          ),
      )
      self._decision_stats["competitive_tactics_exact_terminal_overrides"] += 1
      return int(selected["action"])
    best_own_round = max(float(row["own_round_score"]) for row in rows)
    tied = [
        row for row in rows if np.isclose(row["own_round_score"], best_own_round)
    ]
    if any(int(row["action"]) == int(base_action) for row in tied):
      return int(base_action)
    selected = max(
        tied,
        key=lambda row: (
            row["placement_points"], row["own_connected_bonus"],
            -int(row["action"]),
        ),
    )
    self._decision_stats["competitive_tactics_round_terminal_overrides"] += 1
    return int(selected["action"])

  def _safe_tactical_action(self, state, player, candidates, base_action):
    candidates = sorted({int(action) for action in candidates if int(action) != 999})
    if not candidates:
      return int(base_action)
    view = liveness_oracle.view_from_state(state, player)
    base_margin = liveness_oracle.post_play_margin(view, int(base_action))
    minimum_margin = max(
        int(base_margin),
        int(getattr(self.args, "competitive_tactics_min_own_margin", 0)),
    )
    scored = []
    for action in candidates:
      margin = liveness_oracle.post_play_margin(view, action)
      if int(margin) < int(minimum_margin):
        continue
      after = state.clone()
      after.apply_action(action)
      cluster = int(after._largest_cluster_for_player(int(player)))
      scored.append((cluster, int(margin), int(action == base_action), -action))
    if not scored:
      return int(base_action)
    return int(-max(scored)[-1])

  def _bid_regime_action(self, state, player, legal, base_action):
    if not bool(getattr(self.args, "competitive_tactics_bid_regime", False)):
      return None
    if int(getattr(state, "_phase", -1)) != 3:
      return None
    if sum(card is not None for card in state._cards_played_this_trick) != (
        int(state._num_players) - 1
    ):
      return None
    player = int(player)
    prediction = int(state._predictions[player])
    if prediction < 0:
      return None
    won = int(state._tricks_won[player])
    remaining = max(0, int(state._num_tricks) - int(state._trick_number))
    if won == prediction:
      regime = "exact"
      wants_win = False
    elif won > prediction:
      regime = "above"
      wants_win = True
    elif won + remaining < prediction:
      regime = "unreachable"
      wants_win = True
    else:
      regime = "below"
      wants_win = True
    view = liveness_oracle.view_from_state(state, player)
    rows = []
    for raw_action in legal:
      action = int(raw_action)
      if action == 999:
        continue
      after = state.clone()
      after.apply_action(action)
      feasibility = own_hand_feasibility(after, player)
      rows.append({
          "action": action,
          "wins": bool(_action_would_win_after_play(state, player, action)),
          "feasible": bool(feasibility["feasible"])
          and not bool(after._has_paradoxed[player]),
          "margin": int(liveness_oracle.post_play_margin(view, action)),
          "buffer": int(feasibility["buffer_deficit"]),
          "tokens": int(np.sum(after._color_tokens[player])),
          "cluster": int(after._largest_cluster_for_player(player)),
      })
    preferred = [row for row in rows if bool(row["wins"]) == wants_win]
    opposite = [row for row in rows if bool(row["wins"]) != wants_win]
    if not preferred or not opposite:
      return None
    self._decision_stats["competitive_tactics_bid_regime_considered"] += 1
    reference = max(opposite, key=lambda row: (
        int(row["feasible"]), row["margin"], -row["buffer"], row["tokens"]
    ))
    safe = [
        row for row in preferred
        if bool(row["feasible"])
        and int(row["margin"]) >= int(reference["margin"])
        and int(row["buffer"]) <= int(reference["buffer"])
        and int(row["tokens"]) >= int(reference["tokens"])
    ]
    if not safe:
      return None
    if any(int(row["action"]) == int(base_action) for row in safe):
      return int(base_action)
    chosen = max(safe, key=lambda row: (
        row["margin"], -row["buffer"], row["tokens"],
        row["cluster"] if regime != "above" else 0,
        -row["action"],
    ))
    self._decision_stats["competitive_tactics_bid_regime_overrides"] += 1
    _inc_counter(
        self._decision_stats["competitive_tactics_bid_regime_overrides_by_regime"],
        regime,
    )
    return int(chosen["action"])

  def _bid_aware_action(self, state, player, legal, base_action):
    if not bool(getattr(self.args, "competitive_tactics_bid_play", False)):
      return int(base_action)
    public_weak_field = bool(
        getattr(self.args, "competitive_tactics_adapt_negative_opponent", False)
        and int(getattr(state, "_match_round", 0)) > 0
        and any(
            int(seat) != int(player) and float(total) < 0.0
            for seat, total in enumerate(getattr(state, "_match_totals", []))
        )
    )
    if public_weak_field:
      self._decision_stats[
          "competitive_tactics_public_weak_field_decisions"
      ] += 1
    trick_number = int(getattr(state, "_trick_number", -1))
    min_trick = int(
        getattr(self.args, "competitive_tactics_min_trick_number", -1)
    )
    max_trick = int(
        getattr(self.args, "competitive_tactics_max_trick_number", -1)
    )
    if not public_weak_field and min_trick >= 0 and trick_number < min_trick:
      return int(base_action)
    if not public_weak_field and max_trick >= 0 and trick_number > max_trick:
      return int(base_action)
    led = getattr(state, "_led_color", None)
    if (
        not public_weak_field
        and bool(getattr(self.args, "competitive_tactics_lead_only", False))
        and led is not None
    ):
      return int(base_action)
    prediction = int(state._predictions[int(player)])
    if prediction < 0:
      return int(base_action)
    minimum_prediction = int(
        getattr(self.args, "competitive_tactics_min_prediction", -1)
    )
    if not public_weak_field and prediction < minimum_prediction:
      return int(base_action)
    needed_tricks = prediction - int(state._tricks_won[int(player)])
    # After overshooting, the prediction bonus is already impossible: do not
    # reinterpret a useful additional trick as one that breaks an exact bid.
    if needed_tricks < 0:
      return int(base_action)
    minimum_needed = int(
        getattr(self.args, "competitive_tactics_min_needed_tricks", -1)
    )
    if not public_weak_field and needed_tricks < minimum_needed:
      return int(base_action)
    self._decision_stats["competitive_tactics_bid_play_considered"] += 1
    needs_trick = needed_tricks > 0
    playable = [int(action) for action in legal if int(action) != 999]
    if not playable:
      return int(base_action)
    if needs_trick and led is None:
      nontrump = [action for action in playable if _action_color(state, action) != "R"]
      if not nontrump:
        return int(base_action)
      num_ranks = int(state._num_card_types)
      highest = max(action % num_ranks for action in nontrump)
      minimum_rank_gain = int(
          getattr(self.args, "competitive_tactics_min_lead_rank_gain", 0)
      )
      if (
          not public_weak_field
          and highest - int(base_action) % num_ranks < minimum_rank_gain
      ):
        return int(base_action)
      candidates = [action for action in nontrump if action % num_ranks == highest]
      action = self._safe_tactical_action(state, player, candidates, base_action)
      if action != int(base_action):
        self._decision_stats["competitive_tactics_needed_trick_overrides"] += 1
      return int(action)
    if needs_trick:
      follow_winners = [
          action for action in playable
          if _action_color(state, action) == led
          and _competitive_provisional_winner(state, player, action) == int(player)
      ]
      unbeatable_red = [
          action for action in playable
          if _action_color(state, action) == "R"
          and _competitive_publicly_unbeatable(state, player, action)
      ]
      candidates = follow_winners or unbeatable_red
      if not candidates:
        return int(base_action)
      action = self._safe_tactical_action(state, player, candidates, base_action)
      if action != int(base_action):
        self._decision_stats["competitive_tactics_needed_trick_overrides"] += 1
      return int(action)
    if led is None:
      return int(base_action)
    if _competitive_provisional_winner(state, player, base_action) != int(player):
      return int(base_action)
    losers = [
        action for action in playable
        if _competitive_provisional_winner(state, player, action) != int(player)
    ]
    same_suit = [action for action in losers if _action_color(state, action) == led]
    action = self._safe_tactical_action(
        state, player, same_suit or losers, base_action
    )
    if action != int(base_action):
      self._decision_stats["competitive_tactics_declined_excess_trick"] += 1
    return int(action)

  def step(self, state, player):
    self.last_tactical_override = False
    legal = list(state.legal_actions(player))
    if not legal:
      return None
    if len(legal) == 1:
      _record_decision(self._decision_stats, state, legal[0], len(legal))
      return int(legal[0])
    if int(getattr(state, "_phase", -1)) == 2:
      selected = self._prediction_action(state, player, legal)
      if selected is None:
        selected = self.base_bot.step(state.clone(), player)
      _record_decision(self._decision_stats, state, selected, len(legal))
      return int(selected)
    base_action = int(self.base_bot.step(state.clone(), player))
    if int(getattr(state, "_phase", -1)) != 3 or base_action == 999:
      _record_decision(self._decision_stats, state, base_action, len(legal))
      return int(base_action)
    selected = self._terminal_action(state, player, legal, base_action)
    if selected is None:
      selected = self._bid_regime_action(state, player, legal, base_action)
    if selected is None:
      selected = self._bid_aware_action(state, player, legal, base_action)
    self.last_tactical_override = int(selected) != int(base_action)
    _record_decision(self._decision_stats, state, selected, len(legal))
    return int(selected)

  def raw_decision_stats(self):
    return json.loads(json.dumps(self._decision_stats))

  def decision_stats(self):
    stats = self.raw_decision_stats()
    _decision_stats_rates(stats)
    return stats


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
    entry_belief_samples = int(
        getattr(entry_args, "belief_samples", 0) or belief_samples
    )
    entry_belief_sims = int(
        getattr(entry_args, "belief_sims", 0) or belief_sims
    )
    if entry_belief_samples < 1:
      raise ValueError("belief_samples must be at least 1")
    if entry_belief_sims < 1:
      raise ValueError("belief_sims must be at least 1")
    if mode == "belief_policy":
      return AZBeliefPolicyBot(
          entry_model, bot_name, device, entry_args.value_scale,
          entry_belief_samples, entry_args
      )
    if mode == "belief":
      return AZBeliefSearchBot(
          entry_model,
          bot_name,
          device,
          entry_args,
          entry_belief_samples,
          entry_belief_sims,
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
    if mode == "competitive_tactics":
      return AZCompetitiveTacticsBot(
          entry_model, bot_name, device, entry_args,
          phase_risk_model=entry.get("phase_risk_model"),
      )
    if mode == "belief_table_liveness":
      return AZBeliefTableLivenessPolicyBot(
          entry_model, bot_name, device, entry_args
      )
    if mode == "oracle_shield":
      return AZOracleShieldPolicyBot(entry_model, bot_name, device, entry_args)
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
    if mode == "intralane_filter":
      return AZIntraLaneRankFilterBot(
          entry_model,
          entry["intralane_filter"],
          bot_name,
          device,
          entry_args,
      )
    if mode == "pairwise_boundary":
      return AZPairwiseBoundaryBot(
          entry_model,
          entry["pairwise_boundary"],
          bot_name,
          device,
          entry_args,
      )
    if mode == "action_set_transformer":
      return AZActionSetTransformerBot(
          entry_model,
          entry["action_set_transformer"],
          bot_name,
          device,
          entry_args,
      )
    if mode == "proposal_veto_top1":
      return AZProposalVetoTop1Bot(
          entry_model,
          entry["action_set_transformer"],
          entry["proposal_veto"],
          bot_name,
          device,
          entry_args,
      )
    return AZPolicyBot(
        entry_model,
        bot_name,
        device,
        entry_args.value_scale,
        use_shared_prediction=not _uses_neural_prediction(entry_args),
    )

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
    return AZPolicyBot(
        model,
        name,
        device,
        model_args.value_scale,
        use_shared_prediction=not _uses_neural_prediction(model_args),
    )
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


def play_round(
    game,
    bots,
    seed,
    match_totals=None,
    round_index=0,
    first_paradox_trace_window=5,
):
  if seed is not None:
    np.random.seed(seed)
  state = game.new_initial_state()
  if match_totals is not None:
    state.set_match_context(match_totals, round_index)
  recent_decisions = deque(maxlen=max(1, int(first_paradox_trace_window)))
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
      action = shared_discard_action(state, player, legal)
      if (
          action is None
          and not bool(getattr(bots[player], "uses_neural_prediction", False))
      ):
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


def play_match(
    players,
    bot_names,
    bots,
    seed,
    initial_start=0,
    match_context=False,
    first_paradox_trace_window=5,
):
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
        first_paradox_trace_window=first_paradox_trace_window,
    )
    returns = np.array(returns, dtype=np.float32)
    if match_context:
      totals = returns
      round_returns = returns - prior_totals
    else:
      totals += returns
      round_returns = returns
    paradoxes += np.array(state._has_paradoxed, dtype=np.int32)
    predictions = [int(value) for value in getattr(state, "_predictions", [])]
    tricks_won = [int(value) for value in getattr(state, "_tricks_won", [])]
    adjacency_bonus = [
        float(value) for value in getattr(state, "_player_adjacency_bonus", [])
    ]
    potential_prediction_bonus = list(adjacency_bonus)
    for seat, paradoxed in enumerate(state._has_paradoxed):
      if (
          paradoxed
          and seat < len(predictions)
          and seat < len(tricks_won)
          and predictions[seat] == tricks_won[seat]
          and seat < len(potential_prediction_bonus)
      ):
        potential_prediction_bonus[seat] = float(
            state._largest_cluster_for_player(seat)
        )
    rounds.append({
        "start_player": start_player,
        "returns": [float(x) for x in round_returns],
        "totals_after_round": [float(x) for x in totals],
        "paradoxed": list(state._has_paradoxed),
        "predictions": predictions,
        "tricks_won": tricks_won,
        "adjacency_bonus": adjacency_bonus,
        "potential_prediction_bonus": potential_prediction_bonus,
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
      first_paradox_trace_window=getattr(args, "first_paradox_trace_window", 5),
  )
  return match_idx, seated, match


def match_jobs(args, names, lineups, use_match_context):
  jobs = []
  paired_deals = int(getattr(args, "paired_deals", 0))
  deal_offset = int(getattr(args, "deal_offset", 0))
  if deal_offset < 0:
    raise ValueError("--deal-offset cannot be negative")
  if deal_offset and paired_deals <= 0:
    raise ValueError("--deal-offset requires --paired-deals")
  if paired_deals > 0:
    if args.candidate:
      raise ValueError("--paired-deals is only supported in ladder mode")
    if args.schedule not in ("permutations", "cyclic"):
      raise ValueError(
          "--paired-deals requires --schedule=permutations or --schedule=cyclic"
      )
    match_idx = 0
    for deal_idx in range(deal_offset, deal_offset + paired_deals):
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
          first_paradox_trace_window=getattr(args, "first_paradox_trace_window", 5),
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
  if args.schedule == "cyclic":
    lineups = []
    for combination in itertools.combinations(names, args.players):
      for shift in range(args.players):
        lineups.append(combination[shift:] + combination[:shift])
    return lineups
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


def _wilson_score_interval(successes, trials, z=1.959963984540054):
  """Returns a two-sided Wilson score interval for a binomial proportion."""
  successes = int(successes)
  trials = int(trials)
  if trials <= 0:
    return [0.0, 1.0]
  proportion = successes / trials
  z_squared = float(z) ** 2
  denominator = 1.0 + z_squared / trials
  center = (proportion + z_squared / (2.0 * trials)) / denominator
  margin = (
      float(z)
      * math.sqrt(
          proportion * (1.0 - proportion) / trials
          + z_squared / (4.0 * trials * trials)
      )
      / denominator
  )
  return [max(0.0, center - margin), min(1.0, center + margin)]


def _independent_block_mean_summary(
    values, *, neutral=None, lower_bound=None, upper_bound=None
):
  """Summarizes block means without treating correlated seats as independent."""
  values = [float(value) for value in values]
  if not values:
    return {
        "mean": None,
        "ci95": [None, None],
        "independent_deal_blocks": 0,
        "two_sided_p": None,
    }
  average = float(np.mean(values))
  count = len(values)
  if count < 2:
    interval = [lower_bound, upper_bound]
    p_value = None
  else:
    standard_error = float(np.std(values, ddof=1) / math.sqrt(count))
    margin = float(scipy_stats.t.ppf(0.975, count - 1)) * standard_error
    interval = [average - margin, average + margin]
    if lower_bound is not None:
      interval[0] = max(float(lower_bound), interval[0])
    if upper_bound is not None:
      interval[1] = min(float(upper_bound), interval[1])
    if neutral is None:
      p_value = None
    elif standard_error > 0:
      p_value = 2.0 * float(scipy_stats.t.sf(
          abs(average - float(neutral)) / standard_error, count - 1
      ))
    else:
      p_value = 1.0 if math.isclose(average, float(neutral)) else 0.0
  return {
      "mean": round(average, 6),
      "ci95": [
          None if value is None else round(float(value), 6)
          for value in interval
      ],
      "independent_deal_blocks": count,
      "two_sided_p": (
          None if p_value is None else round(float(p_value), 6)
      ),
      "method": "independent_deal_block_student_t",
  }


def summarize_competitive_promotion_gate(
    names,
    players,
    match_results,
    candidate,
    *,
    reference_names=None,
    deal_blocks_by_match=None,
    min_independent_deals=30,
    max_own_paradox_regression=0.0,
):
  """Gates official placement using common deals as the inference unit."""
  candidate = str(candidate or "")
  reference_names = (
      list(reference_names)
      if reference_names
      else [name for name in names if name != candidate]
  )
  result = {
      "kind": "competitive_promotion_gate_v1",
      "eligible": False,
      "passed": False,
      "candidate": candidate,
      "reference_opponents": reference_names,
      "players": int(players),
      "min_independent_deal_blocks": int(min_independent_deals),
      "max_own_paradox_rate_regression": float(max_own_paradox_regression),
      "table_paradox_is_promotion_veto": False,
  }
  if candidate not in names:
    result["reason"] = "candidate_not_in_evaluation"
    return result
  if (
      not reference_names
      or candidate in reference_names
      or any(name not in names for name in reference_names)
  ):
    result["reason"] = "invalid_reference_opponents"
    return result

  references = set(reference_names)
  blocks = {}
  seat_counts = {str(seat): 0 for seat in range(int(players))}
  placement_counts = {}
  first_place_share_sum = 0.0
  pairwise_wins = 0
  pairwise_losses = 0
  pairwise_ties = 0
  candidate_paradoxes = 0
  candidate_round_opportunities = 0
  opponent_paradoxes = 0
  opponent_round_opportunities = 0
  opponent_paradox_rounds = 0
  table_paradox_rounds = 0
  observed_rounds = 0
  candidate_paradox_points_lost = 0.0
  candidate_bid_hits = 0
  candidate_scored_bid_hits = 0
  candidate_bid_opportunities = 0
  candidate_adjacency_bonus = 0.0

  for match_idx, seated, match in sorted(
      match_results, key=lambda item: int(item[0])
  ):
    if candidate not in seated:
      continue
    candidate_seat = seated.index(candidate)
    opponent_seats = [
        seat for seat, name in enumerate(seated) if name in references
    ]
    if not opponent_seats:
      continue
    totals = [float(value) for value in match.get("totals", [])]
    final_returns = [
        float(value) for value in match.get("final_round_returns", [])
    ]
    if len(totals) != int(players) or len(final_returns) != int(players):
      result["reason"] = "missing_official_match_outcomes"
      return result
    outcome_scores = match_outcome_scores(totals, final_returns)
    candidate_outcome = outcome_scores[candidate_seat]
    better_seats = sum(
        float(value) > candidate_outcome for value in outcome_scores
    )
    tied_other_seats = sum(
        seat != candidate_seat and float(value) == candidate_outcome
        for seat, value in enumerate(outcome_scores)
    )
    placement = 1.0 + better_seats + 0.5 * tied_other_seats
    placement_key = str(int(placement)) if placement.is_integer() else str(placement)
    placement_counts[placement_key] = (
        int(placement_counts.get(placement_key, 0)) + 1
    )
    if better_seats == 0:
      first_place_share_sum += 1.0 / (tied_other_seats + 1)

    wins = sum(candidate_outcome > outcome_scores[seat] for seat in opponent_seats)
    losses = sum(candidate_outcome < outcome_scores[seat] for seat in opponent_seats)
    ties = len(opponent_seats) - wins - losses
    pairwise_wins += int(wins)
    pairwise_losses += int(losses)
    pairwise_ties += int(ties)
    score_rate = (wins + 0.5 * ties) / len(opponent_seats)
    score_advantage = totals[candidate_seat] - float(np.mean([
        totals[seat] for seat in opponent_seats
    ]))

    round_rows = match.get("rounds", [])
    round_count = len(round_rows) if round_rows else int(players)
    paradox_totals = [
        int(value) for value in match.get("paradoxes", [0] * int(players))
    ]
    if len(paradox_totals) != int(players):
      result["reason"] = "missing_player_paradox_outcomes"
      return result
    own_paradoxes = paradox_totals[candidate_seat]
    other_paradoxes = sum(paradox_totals[seat] for seat in opponent_seats)
    candidate_paradoxes += own_paradoxes
    candidate_round_opportunities += round_count
    opponent_paradoxes += other_paradoxes
    opponent_round_opportunities += round_count * len(opponent_seats)
    own_rate = own_paradoxes / round_count
    opponent_rate = other_paradoxes / (round_count * len(opponent_seats))

    for round_row in round_rows:
      paradoxed = [
          bool(value) for value in round_row.get("paradoxed", [])
      ]
      if len(paradoxed) == int(players):
        observed_rounds += 1
        table_paradox_rounds += int(any(paradoxed))
        opponent_paradox_rounds += int(
            any(paradoxed[seat] for seat in opponent_seats)
        )
      predictions = round_row.get("predictions", [])
      tricks_won = round_row.get("tricks_won", [])
      bonuses = round_row.get("adjacency_bonus", [])
      potential_bonuses = round_row.get("potential_prediction_bonus", [])
      if candidate_seat < len(bonuses):
        candidate_adjacency_bonus += float(bonuses[candidate_seat])
      if (
          candidate_seat < len(predictions)
          and candidate_seat < len(tricks_won)
          and int(predictions[candidate_seat]) >= 0
      ):
        candidate_bid_opportunities += 1
        bid_hit = (
            int(predictions[candidate_seat])
            == int(tricks_won[candidate_seat])
        )
        candidate_bid_hits += int(bid_hit)
        personally_paradoxed = (
            candidate_seat < len(paradoxed) and paradoxed[candidate_seat]
        )
        candidate_scored_bid_hits += int(bid_hit and not personally_paradoxed)
        if personally_paradoxed:
          candidate_paradox_points_lost += 2.0 * float(
              tricks_won[candidate_seat]
          )
          if bid_hit and candidate_seat < len(potential_bonuses):
            candidate_paradox_points_lost += float(
                potential_bonuses[candidate_seat]
            )

    block_id = (
        int(deal_blocks_by_match.get(int(match_idx), match_idx))
        if deal_blocks_by_match is not None
        else int(match_idx)
    )
    block = blocks.setdefault(block_id, {
        "score_rates": [],
        "score_advantages": [],
        "own_paradox_rates": [],
        "opponent_paradox_rates": [],
        "placements": [],
        "seat_counts": {str(seat): 0 for seat in range(int(players))},
    })
    block["score_rates"].append(float(score_rate))
    block["score_advantages"].append(float(score_advantage))
    block["own_paradox_rates"].append(float(own_rate))
    block["opponent_paradox_rates"].append(float(opponent_rate))
    block["placements"].append(float(placement))
    block["seat_counts"][str(candidate_seat)] += 1
    seat_counts[str(candidate_seat)] += 1

  if not blocks:
    result["reason"] = "no_candidate_reference_matches"
    return result

  block_summaries = []
  for block_id, block in sorted(blocks.items()):
    own_rate = float(np.mean(block["own_paradox_rates"]))
    opponent_rate = float(np.mean(block["opponent_paradox_rates"]))
    block_summaries.append({
        "deal_block": int(block_id),
        "matches": len(block["score_rates"]),
        "candidate_seat_counts": block["seat_counts"],
        "pairwise_score_rate": round(float(np.mean(block["score_rates"])), 6),
        "official_score_advantage": round(
            float(np.mean(block["score_advantages"])), 6
        ),
        "own_paradox_round_rate": round(own_rate, 6),
        "opponent_personal_paradox_round_rate": round(opponent_rate, 6),
        "own_paradox_rate_difference": round(own_rate - opponent_rate, 6),
        "mean_official_placement": round(
            float(np.mean(block["placements"])), 6
        ),
    })

  ordered_blocks = [block for _block_id, block in sorted(blocks.items())]
  block_score_rates = [
      float(np.mean(block["score_rates"])) for block in ordered_blocks
  ]
  block_score_advantages = [
      float(np.mean(block["score_advantages"])) for block in ordered_blocks
  ]
  block_own_rates = [
      float(np.mean(block["own_paradox_rates"])) for block in ordered_blocks
  ]
  block_opponent_rates = [
      float(np.mean(block["opponent_paradox_rates"])) for block in ordered_blocks
  ]
  block_paradox_differences = [
      own_rate - opponent_rate
      for own_rate, opponent_rate in zip(block_own_rates, block_opponent_rates)
  ]
  placement_summary = _independent_block_mean_summary(
      block_score_rates, neutral=0.5, lower_bound=0.0, upper_bound=1.0
  )
  score_summary = _independent_block_mean_summary(
      block_score_advantages, neutral=0.0
  )
  own_rate_summary = _independent_block_mean_summary(
      block_own_rates, lower_bound=0.0, upper_bound=1.0
  )
  opponent_rate_summary = _independent_block_mean_summary(
      block_opponent_rates, lower_bound=0.0, upper_bound=1.0
  )
  paradox_delta_summary = _independent_block_mean_summary(
      block_paradox_differences, neutral=0.0,
      lower_bound=-1.0, upper_bound=1.0,
  )
  block_seat_balance = all(
      min(block["candidate_seat_counts"].values()) > 0
      and (
          max(block["candidate_seat_counts"].values())
          == min(block["candidate_seat_counts"].values())
      )
      for block in block_summaries
  )
  deal_count = len(block_summaries)
  match_count = sum(seat_counts.values())
  placement_lower = placement_summary["ci95"][0]
  score_upper = score_summary["ci95"][1]
  paradox_lower = paradox_delta_summary["ci95"][0]
  placement_passed = (
      placement_lower is not None and float(placement_lower) > 0.5
  )
  score_guard_passed = score_upper is None or float(score_upper) >= 0.0
  paradox_guard_passed = (
      paradox_lower is None
      or float(paradox_lower) <= float(max_own_paradox_regression)
  )
  errors = []
  if not block_seat_balance:
    errors.append("candidate seats are not balanced within each common deal")
  if deal_count < int(min_independent_deals):
    errors.append(
        f"independent deal blocks {deal_count} < {int(min_independent_deals)}"
    )
  if not placement_passed:
    errors.append(
        "independent-deal placement score CI95 lower is not above 0.5"
    )
  if not score_guard_passed:
    errors.append("official match score shows significant regression")
  if not paradox_guard_passed:
    errors.append("candidate personal paradox rate significantly regresses")
  score_rate = float(np.mean(block_score_rates))
  average_official_placement = float(np.mean([
      block["mean_official_placement"] for block in block_summaries
  ]))
  clipped_score_rate = min(0.999, max(0.001, score_rate))
  elo_difference = -400.0 * math.log10(
      1.0 / clipped_score_rate - 1.0
  )
  result.update({
      "eligible": bool(block_seat_balance),
      "passed": not errors,
      "reason": None if not errors else errors[0],
      "errors": errors,
      "inference_unit": "independent_common_deal_block",
      "official_tiebreak": "total match score, then final-round score",
      "independent_deal_blocks": deal_count,
      "matches": match_count,
      "candidate_seat_counts": seat_counts,
      "seat_balanced_within_each_deal": block_seat_balance,
      "placement_counts": placement_counts,
      "mean_official_placement": round(average_official_placement, 6),
      "first_place_share": round(
          first_place_share_sum / match_count, 6
      ),
      "bga_style_rank_points": {
          "per_match": round(
              2.0 * (int(players) - average_official_placement), 6
          ),
          "neutral": int(players) - 1,
          "maximum": 2 * (int(players) - 1),
      },
      "pairwise": {
          "wins": pairwise_wins,
          "losses": pairwise_losses,
          "ties": pairwise_ties,
          "score_rate": round(score_rate, 6),
          "score_rate_ci95": placement_summary["ci95"],
          "head_to_head_elo_difference": round(elo_difference, 2),
          "two_sided_p": placement_summary["two_sided_p"],
          "comparisons_are_correlated": True,
      },
      "placement_advantage": placement_summary,
      "official_score_advantage": score_summary,
      "candidate_own_paradox": {
          "paradoxes": candidate_paradoxes,
          "round_opportunities": candidate_round_opportunities,
          "round_rate": round(
              candidate_paradoxes / candidate_round_opportunities, 6
          ) if candidate_round_opportunities else None,
          "independent_block_summary": own_rate_summary,
          "points_lost": round(candidate_paradox_points_lost, 6),
      },
      "reference_personal_paradox": {
          "paradoxes": opponent_paradoxes,
          "round_opportunities": opponent_round_opportunities,
          "round_rate": round(
              opponent_paradoxes / opponent_round_opportunities, 6
          ) if opponent_round_opportunities else None,
          "independent_block_summary": opponent_rate_summary,
      },
      "own_paradox_rate_difference": paradox_delta_summary,
      "opponent_paradox_rounds": opponent_paradox_rounds,
      "table_paradox": {
          "rounds_with_any_paradox": table_paradox_rounds,
          "hand_rounds": observed_rounds,
          "any_paradox_round_rate": round(
              table_paradox_rounds / observed_rounds, 6
          ) if observed_rounds else None,
          "diagnostic_only": True,
      },
      "candidate_prediction": {
          "correct_predictions": candidate_scored_bid_hits,
          "raw_prediction_matches": candidate_bid_hits,
          "opportunities": candidate_bid_opportunities,
          "hit_rate": round(
              candidate_scored_bid_hits / candidate_bid_opportunities, 6
          ) if candidate_bid_opportunities else None,
          "raw_match_rate": round(
              candidate_bid_hits / candidate_bid_opportunities, 6
          ) if candidate_bid_opportunities else None,
          "adjacency_bonus_points": round(candidate_adjacency_bonus, 6),
      },
      "placement_advantage_passed": placement_passed,
      "official_score_guard_passed": score_guard_passed,
      "own_paradox_guard_passed": paradox_guard_passed,
      "deal_blocks": block_summaries,
  })
  return result


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
  hand_round_keys = []
  any_paradox_outcomes = []
  paradox_seat_masks = []
  if match_results is not None:
    observed_hand_rounds = 0
    observed_any_paradox_rounds = 0
    for match_idx, _seated, match in sorted(
        match_results, key=lambda row: int(row[0])
    ):
      for round_idx, round_row in enumerate(match.get("rounds", [])):
        if "paradoxed" not in round_row:
          continue
        paradoxed = [bool(value) for value in round_row.get("paradoxed", [])]
        any_paradox = any(paradoxed)
        observed_hand_rounds += 1
        if any_paradox:
          observed_any_paradox_rounds += 1
        hand_round_keys.append([int(match_idx), int(round_idx)])
        any_paradox_outcomes.append(int(any_paradox))
        paradox_seat_masks.append(int(sum(
            (1 << seat) for seat, value in enumerate(paradoxed) if value
        )))
    if observed_hand_rounds > 0:
      hand_rounds = observed_hand_rounds
      rounds_with_any_paradox = observed_any_paradox_rounds
  if bot_round_opportunities <= 0:
    result["reason"] = "no_completed_bot_rounds"
    return result

  bot_round_rate = total_paradoxes / bot_round_opportunities
  bot_round_ci95 = _wilson_score_interval(
      total_paradoxes, bot_round_opportunities
  )
  hand_round_rate = (
      rounds_with_any_paradox / hand_rounds if hand_rounds > 0 else 0.0
  )
  hand_round_ci95 = _wilson_score_interval(
      rounds_with_any_paradox, hand_rounds
  )
  ci95_upper_below_threshold = bool(hand_round_ci95[1] < threshold)
  return {
      "threshold": threshold,
      "eligible": True,
      "passed": ci95_upper_below_threshold,
      "reason": None,
      "gate_basis": "any_paradox_round_rate_ci95_upper",
      "ci95_method": "wilson_score",
      "point_estimate_below_threshold": bool(hand_round_rate < threshold),
      "bot_spec": first_spec,
      "checkpoint": first_spec.get("checkpoint") if first_spec else None,
      "mode": first_spec.get("mode") if first_spec else None,
      "aliases": list(names),
      "completed_matches": completed_matches,
      "seat_match_appearances": seat_match_appearances,
      "hand_rounds": hand_rounds,
      "rounds_with_any_paradox": rounds_with_any_paradox,
      "hand_round_outcome_order": "sorted_match_index_then_round_index",
      "hand_round_keys": hand_round_keys,
      "any_paradox_outcomes": any_paradox_outcomes,
      "paradox_seat_masks": paradox_seat_masks,
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
      "ci95_upper_below_threshold": ci95_upper_below_threshold,
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
  deal_blocks_by_match = {
      int(job[0]): int(_unpack_match_job(job)[3]) for job in jobs
  }
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
    if len(sample_matches) < int(getattr(args, "sample_match_limit", 3)):
      sample_matches.append(match)

  for stats in decision_stats_by_bot.values():
    _decision_stats_rates(stats)

  result = {
      "players": args.players,
      "seed": int(args.seed),
      "matches": len(match_results),
      "requested_matches": args.matches,
      "paired_deals": args.paired_deals,
      "deal_offset": int(getattr(args, "deal_offset", 0)),
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
      "first_paradox_summary": summarize_first_paradox_traces(
          match_results,
          sample_limit=int(getattr(args, "first_paradox_summary_sample_limit", 12)),
      ),
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
  competitive_candidate = str(
      getattr(args, "competitive_gate_candidate", "") or ""
  ).strip()
  if competitive_candidate:
    reference_names = [
        name.strip()
        for name in str(
            getattr(args, "competitive_gate_opponents", "") or ""
        ).split(",")
        if name.strip()
    ]
    result["competitive_promotion_gate"] = (
        summarize_competitive_promotion_gate(
            names,
            args.players,
            match_results,
            competitive_candidate,
            reference_names=reference_names or None,
            deal_blocks_by_match=deal_blocks_by_match,
            min_independent_deals=int(
                getattr(args, "competitive_min_independent_deals", 30)
            ),
            max_own_paradox_regression=float(
                getattr(args, "competitive_max_own_paradox_regression", 0.0)
            ),
        )
    )
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
