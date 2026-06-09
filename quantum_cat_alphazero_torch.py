#!/usr/bin/env python3
"""Small multiplayer AlphaZero-style trainer for python_quantum_cat.

This is intentionally self-contained because OpenSpiel's bundled Python
AlphaZero runner is 2-player-only, while BGA's current Cat arena target is 3p.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import (
    make_bot,
    multiplayer_elo_update,
    play_game,
    prediction_top_rank_score,
    shared_prediction_action,
)


BASE_ACTION_FEATURE_SIZE = 67
APPENDED_ACTION_FEATURE_NAMES = [
    "rank_7",
    "rank_8",
    "rank_9",
    "trick_result_known",
    "hits_prediction",
    "overshoots_prediction",
    "short_of_prediction",
    "prediction_gap_after",
    "ends_round",
    "end_round_score_estimate",
    "remaining_tricks_after",
    "wins_needed_after",
    "can_still_hit_after",
    "must_win_all_remaining_after",
    "hit_with_future_tricks",
    "post_hit_low_card_frac",
    "post_hit_high_card_frac",
    "post_hit_legal_lead_count_after",
    "post_hit_dead_card_frac_after",
    "post_hit_low_legal_lead_ratio",
    "post_hit_low_card_survival_margin",
    "post_hit_forced_card_pressure",
    "token_loss_newly_loses_led",
    "token_loss_legal_lead_damage",
    "token_loss_dead_card_damage",
    "token_loss_singleton_card_damage",
    "token_loss_rank_damage",
    "token_loss_creates_no_exit",
    "cluster_frontier_empty_after",
    "cluster_frontier_gain",
    "cluster_component_delta",
    "cluster_connects_components",
    "cluster_dead_end_after",
    "legal_z_adjacency_gain",
    "legal_pct_adjacency_gain",
    "legal_z_largest_after",
    "legal_pct_largest_after",
    "legal_z_prediction_gap_after",
    "legal_pct_prediction_gap_after",
    "legal_z_end_round_score_estimate",
    "legal_pct_end_round_score_estimate",
    "legal_z_remaining_tricks_after",
    "legal_pct_remaining_tricks_after",
    "legal_z_wins_needed_after",
    "legal_pct_wins_needed_after",
    "legal_z_post_hit_low_legal_ratio",
    "legal_pct_post_hit_low_legal_ratio",
    "legal_z_post_hit_survival_margin",
    "legal_pct_post_hit_survival_margin",
    "legal_z_post_hit_forced_pressure",
    "legal_pct_post_hit_forced_pressure",
    "legal_z_token_loss_legal_damage",
    "legal_pct_token_loss_legal_damage",
    "legal_z_cluster_frontier_after",
    "legal_pct_cluster_frontier_after",
    "legal_z_future_token_loss_rank_deficit",
    "legal_pct_future_token_loss_rank_deficit",
    "legal_z_future_token_loss_buffer_deficit",
    "legal_pct_future_token_loss_buffer_deficit",
    "legal_z_future_token_loss_safe_flex_drop",
    "legal_pct_future_token_loss_safe_flex_drop",
    "legal_z_discard_rank_deficit_relief",
    "legal_pct_discard_rank_deficit_relief",
    "legal_z_discard_rank_buffer_relief",
    "legal_pct_discard_rank_buffer_relief",
    "legal_z_discard_safe_flex_delta",
    "legal_pct_discard_safe_flex_delta",
    "legal_z_discard_rank_slot_surplus_after",
    "legal_pct_discard_rank_slot_surplus_after",
    "future_rank_slots_after",
    "future_rank_deficit_after",
    "future_tight_rank_count_after",
    "future_rank_deficit_delta",
    "future_safe_flex_score_after",
    "future_min_rank_surplus_after",
    "future_max_rank_deficit_after",
    "future_buffer_deficit_after",
    "future_no_exit_after",
    "future_dead_rank_count_after",
    "future_token_loss_max_rank_deficit_after",
    "future_token_loss_max_buffer_deficit_after",
    "future_token_loss_no_exit_frac_after",
    "future_token_loss_max_dead_rank_count_after",
    "future_token_loss_worst_safe_flex_score_after",
    "future_token_loss_safe_flex_drop_after",
    "prediction_hand_high_card_frac",
    "prediction_hand_duplicate_frac",
    "prediction_expected_tricks_norm",
    "prediction_action_gap_to_expected",
    "prediction_action_abs_gap_to_expected",
    "prediction_action_under_expected",
    "prediction_action_over_expected",
    "prediction_action_is_min",
    "prediction_action_is_max",
    "discard_hand_count_before_frac",
    "discard_hand_count_after_frac",
    "discard_rank_open_slots_frac",
    "discard_rank_slot_surplus_before",
    "discard_rank_slot_surplus_after",
    "discard_rank_deficit_relief",
    "discard_rank_buffer_relief",
    "discard_safe_flex_delta",
    "discard_dead_rank_relief",
    "discard_tight_rank_relief",
    "discard_removes_singleton",
    "discard_from_duplicate",
    "discard_no_exit_after",
    "exit_public_slot_damage",
    "exit_own_public_slot_damage",
    "exit_board_open_cell_damage",
    "exit_min_player_open_slots_after",
    "exit_total_player_open_slots_after",
    "exit_own_lane_surplus_after",
    "exit_min_player_lane_surplus_after",
    "exit_total_player_lane_surplus_after",
    "exit_lane_surplus_damage",
    "exit_min_lane_surplus_damage",
    "exit_lane_pressure_player_count_after",
    "exit_lost_led_token",
    "exit_over_target_would_win",
    "legal_z_exit_public_slot_damage",
    "legal_pct_exit_public_slot_damage",
    "legal_z_exit_own_public_slot_damage",
    "legal_pct_exit_own_public_slot_damage",
    "legal_z_exit_min_player_open_slots_after",
    "legal_pct_exit_min_player_open_slots_after",
    "legal_z_exit_min_player_lane_surplus_after",
    "legal_pct_exit_min_player_lane_surplus_after",
    "legal_z_exit_lane_surplus_damage",
    "legal_pct_exit_lane_surplus_damage",
    "legal_z_exit_lane_pressure_player_count_after",
    "legal_pct_exit_lane_pressure_player_count_after",
    "own_future_min_colors_after",
    "own_future_mean_colors_after",
    "own_future_zero_exit_frac_after",
    "own_future_one_exit_frac_after",
    "own_future_two_or_less_exit_frac_after",
    "own_future_sum_log_colors_after",
    "own_future_min_lead_colors_after",
    "own_future_zero_lead_exit_frac_after",
    "own_future_legal_lead_count_after",
    "legal_z_own_future_min_colors_after",
    "legal_pct_own_future_min_colors_after",
    "legal_z_own_future_zero_exit_frac_after",
    "legal_pct_own_future_zero_exit_frac_after",
    "legal_z_own_future_one_exit_frac_after",
    "legal_pct_own_future_one_exit_frac_after",
]
APPENDED_ACTION_FEATURE_START = BASE_ACTION_FEATURE_SIZE
ACTION_FEATURE_SIZE = (
    APPENDED_ACTION_FEATURE_START + len(APPENDED_ACTION_FEATURE_NAMES)
)
APPENDED_ACTION_FEATURE_INDEX = {
    name: APPENDED_ACTION_FEATURE_START + idx
    for idx, name in enumerate(APPENDED_ACTION_FEATURE_NAMES)
}
ACTION_FEATURE_PREFIX_SIZE = 26
ACTION_FEATURE_RANK_NORM_INDEX = ACTION_FEATURE_PREFIX_SIZE + 9
ACTION_FEATURE_FOLLOWS_LED_INDEX = ACTION_FEATURE_PREFIX_SIZE + 16
ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX = ACTION_FEATURE_PREFIX_SIZE + 17
ACTION_FEATURE_ADJACENCY_GAIN_INDEX = ACTION_FEATURE_PREFIX_SIZE + 21
ACTION_FEATURE_LARGEST_AFTER_INDEX = ACTION_FEATURE_PREFIX_SIZE + 23
ACTION_FEATURE_IS_RED_INDEX = ACTION_FEATURE_PREFIX_SIZE + 24
TACTICAL_POLICY_BUCKET_NAMES = (
    "token_loss",
    "follows_led",
    "prediction_feasible",
    "hits_prediction",
    "future_hit",
    "low_rank",
    "mid_rank",
    "mid_rank_cluster_growth",
    "mid_rank_hits_prediction",
    "mid_rank_follows_led",
    "high_rank",
    "cluster_growth",
    "red",
    "red_cluster_growth",
    "red_prediction_feasible",
    "red_hits_prediction",
)
LEGAL_SET_CONTEXT_FEATURE_NAMES = [
    name for name in APPENDED_ACTION_FEATURE_NAMES
    if name.startswith("legal_z_") or name.startswith("legal_pct_")
]
EXIT_LIQUIDITY_LEGAL_SET_CONTEXT_FEATURE_NAMES = [
    "legal_z_exit_public_slot_damage",
    "legal_pct_exit_public_slot_damage",
    "legal_z_exit_own_public_slot_damage",
    "legal_pct_exit_own_public_slot_damage",
    "legal_z_exit_min_player_open_slots_after",
    "legal_pct_exit_min_player_open_slots_after",
    "legal_z_exit_min_player_lane_surplus_after",
    "legal_pct_exit_min_player_lane_surplus_after",
    "legal_z_exit_lane_surplus_damage",
    "legal_pct_exit_lane_surplus_damage",
    "legal_z_exit_lane_pressure_player_count_after",
    "legal_pct_exit_lane_pressure_player_count_after",
]
OWN_HAND_FEASIBILITY_LEGAL_SET_CONTEXT_FEATURE_NAMES = [
    "legal_z_own_future_min_colors_after",
    "legal_pct_own_future_min_colors_after",
    "legal_z_own_future_zero_exit_frac_after",
    "legal_pct_own_future_zero_exit_frac_after",
    "legal_z_own_future_one_exit_frac_after",
    "legal_pct_own_future_one_exit_frac_after",
]
LEGAL_SET_CONTEXT_BLOCK_FEATURE_NAMES = [
    name for name in LEGAL_SET_CONTEXT_FEATURE_NAMES
    if name not in set(
        EXIT_LIQUIDITY_LEGAL_SET_CONTEXT_FEATURE_NAMES
        + OWN_HAND_FEASIBILITY_LEGAL_SET_CONTEXT_FEATURE_NAMES
    )
]
LEGAL_SET_CONTEXT_SOURCES = (
    (
        ACTION_FEATURE_ADJACENCY_GAIN_INDEX,
        "legal_z_adjacency_gain",
        "legal_pct_adjacency_gain",
    ),
    (
        ACTION_FEATURE_LARGEST_AFTER_INDEX,
        "legal_z_largest_after",
        "legal_pct_largest_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["prediction_gap_after"],
        "legal_z_prediction_gap_after",
        "legal_pct_prediction_gap_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["end_round_score_estimate"],
        "legal_z_end_round_score_estimate",
        "legal_pct_end_round_score_estimate",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["remaining_tricks_after"],
        "legal_z_remaining_tricks_after",
        "legal_pct_remaining_tricks_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["wins_needed_after"],
        "legal_z_wins_needed_after",
        "legal_pct_wins_needed_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["post_hit_low_legal_lead_ratio"],
        "legal_z_post_hit_low_legal_ratio",
        "legal_pct_post_hit_low_legal_ratio",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["post_hit_low_card_survival_margin"],
        "legal_z_post_hit_survival_margin",
        "legal_pct_post_hit_survival_margin",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["post_hit_forced_card_pressure"],
        "legal_z_post_hit_forced_pressure",
        "legal_pct_post_hit_forced_pressure",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["token_loss_legal_lead_damage"],
        "legal_z_token_loss_legal_damage",
        "legal_pct_token_loss_legal_damage",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["cluster_frontier_empty_after"],
        "legal_z_cluster_frontier_after",
        "legal_pct_cluster_frontier_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX[
            "future_token_loss_max_rank_deficit_after"
        ],
        "legal_z_future_token_loss_rank_deficit",
        "legal_pct_future_token_loss_rank_deficit",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX[
            "future_token_loss_max_buffer_deficit_after"
        ],
        "legal_z_future_token_loss_buffer_deficit",
        "legal_pct_future_token_loss_buffer_deficit",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX[
            "future_token_loss_safe_flex_drop_after"
        ],
        "legal_z_future_token_loss_safe_flex_drop",
        "legal_pct_future_token_loss_safe_flex_drop",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["discard_rank_deficit_relief"],
        "legal_z_discard_rank_deficit_relief",
        "legal_pct_discard_rank_deficit_relief",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["discard_rank_buffer_relief"],
        "legal_z_discard_rank_buffer_relief",
        "legal_pct_discard_rank_buffer_relief",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["discard_safe_flex_delta"],
        "legal_z_discard_safe_flex_delta",
        "legal_pct_discard_safe_flex_delta",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["discard_rank_slot_surplus_after"],
        "legal_z_discard_rank_slot_surplus_after",
        "legal_pct_discard_rank_slot_surplus_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["exit_public_slot_damage"],
        "legal_z_exit_public_slot_damage",
        "legal_pct_exit_public_slot_damage",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["exit_own_public_slot_damage"],
        "legal_z_exit_own_public_slot_damage",
        "legal_pct_exit_own_public_slot_damage",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["exit_min_player_open_slots_after"],
        "legal_z_exit_min_player_open_slots_after",
        "legal_pct_exit_min_player_open_slots_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["exit_min_player_lane_surplus_after"],
        "legal_z_exit_min_player_lane_surplus_after",
        "legal_pct_exit_min_player_lane_surplus_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["exit_lane_surplus_damage"],
        "legal_z_exit_lane_surplus_damage",
        "legal_pct_exit_lane_surplus_damage",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX[
            "exit_lane_pressure_player_count_after"
        ],
        "legal_z_exit_lane_pressure_player_count_after",
        "legal_pct_exit_lane_pressure_player_count_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["own_future_min_colors_after"],
        "legal_z_own_future_min_colors_after",
        "legal_pct_own_future_min_colors_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["own_future_zero_exit_frac_after"],
        "legal_z_own_future_zero_exit_frac_after",
        "legal_pct_own_future_zero_exit_frac_after",
    ),
    (
        APPENDED_ACTION_FEATURE_INDEX["own_future_one_exit_frac_after"],
        "legal_z_own_future_one_exit_frac_after",
        "legal_pct_own_future_one_exit_frac_after",
    ),
)
MAX_RANK_FEATURES = 6
FULL_RANK_FEATURES = 9
MAX_COLOR_FEATURES = 4
ACTION_CONDITIONED_ARCHS = ("action_mlp", "action_attn", "action_setpool")


class AZNet(nn.Module):
  def __init__(
      self, obs_size, num_actions, num_players, width=256, depth=3,
      arch="mlp", separate_action_value_encoder=False,
      separate_action_paradox_encoder=False
  ):
    super().__init__()
    if arch not in ("mlp", "residual", *ACTION_CONDITIONED_ARCHS):
      raise ValueError(f"Unsupported architecture: {arch}")
    self.arch = arch
    self.num_actions = num_actions
    self.action_feature_size = ACTION_FEATURE_SIZE
    self.separate_action_value_encoder = bool(separate_action_value_encoder)
    self.separate_action_paradox_encoder = bool(separate_action_paradox_encoder)
    if arch in ("mlp", *ACTION_CONDITIONED_ARCHS):
      layers = []
      in_size = obs_size
      for _ in range(depth):
        layers.append(nn.Linear(in_size, width))
        layers.append(nn.ReLU())
        in_size = width
      self.body = nn.Sequential(*layers)
    else:
      self.body = ResidualBody(obs_size, width, depth)
    self.value = nn.Linear(width, num_players)
    self.paradox = nn.Linear(width, num_players)
    if arch in ACTION_CONDITIONED_ARCHS:
      action_width = max(32, width // 2)
      self.action_width = action_width
      self.action_encoder = nn.Sequential(
          nn.Linear(ACTION_FEATURE_SIZE, action_width),
          nn.ReLU(),
          nn.Linear(action_width, action_width),
          nn.ReLU(),
      )
      self.state_action_projection = nn.Sequential(
          nn.Linear(width, action_width),
          nn.ReLU(),
      )
      if arch == "action_attn":
        attention_heads = 4 if action_width % 4 == 0 else 1
        self.action_attention = nn.MultiheadAttention(
            action_width,
            attention_heads,
            batch_first=True,
        )
        self.action_attention_gate = nn.Parameter(torch.zeros(1))
      pair_width = (
          width + 8 * action_width
          if arch == "action_setpool"
          else width + 5 * action_width
      )
      self.policy = nn.Sequential(
          nn.Linear(pair_width, width),
          nn.ReLU(),
          nn.Linear(width, 1),
      )
      value_width = (
          width + 4 * action_width
          if arch == "action_setpool"
          else width + 2 * action_width
      )
      self.value = nn.Sequential(
          nn.Linear(value_width, width),
          nn.ReLU(),
          nn.Linear(width, num_players),
      )
      self.action_paradox = nn.Sequential(
          nn.Linear(pair_width, width),
          nn.ReLU(),
          nn.Linear(width, 1),
      )
      self.action_value = nn.Sequential(
          nn.Linear(pair_width, width),
          nn.ReLU(),
          nn.Linear(width, 1),
      )
      if self.separate_action_value_encoder:
        self.action_value_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURE_SIZE, action_width),
            nn.ReLU(),
            nn.Linear(action_width, action_width),
            nn.ReLU(),
        )
        self.action_value_state_action_projection = nn.Sequential(
            nn.Linear(width, action_width),
            nn.ReLU(),
        )
        if arch == "action_attn":
          self.action_value_attention = nn.MultiheadAttention(
              action_width,
              attention_heads,
              batch_first=True,
          )
          self.action_value_attention_gate = nn.Parameter(torch.zeros(1))
      if self.separate_action_paradox_encoder:
        self.action_paradox_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURE_SIZE, action_width),
            nn.ReLU(),
            nn.Linear(action_width, action_width),
            nn.ReLU(),
        )
        self.action_paradox_state_action_projection = nn.Sequential(
            nn.Linear(width, action_width),
            nn.ReLU(),
        )
        if arch == "action_attn":
          self.action_paradox_attention = nn.MultiheadAttention(
              action_width,
              attention_heads,
              batch_first=True,
          )
          self.action_paradox_attention_gate = nn.Parameter(torch.zeros(1))
    else:
      self.policy = nn.Linear(width, num_actions)
      self.action_paradox = nn.Linear(width, num_actions)
      self.action_value = nn.Linear(width, num_actions)

  def _encoded_action_embeddings(
      self, state_embedding, action_features, action_encoder=None
  ):
    if action_features is None:
      action_features = torch.zeros(
          (
              state_embedding.shape[0],
              self.num_actions,
              self.action_feature_size,
          ),
          dtype=state_embedding.dtype,
          device=state_embedding.device,
      )
    if action_encoder is None:
      action_encoder = self.action_encoder
    return action_encoder(action_features)

  def _action_embeddings(
      self,
      state_embedding,
      action_features,
      action_encoder=None,
      state_action_projection=None,
      action_attention=None,
      action_attention_gate=None,
  ):
    action_embedding = self._encoded_action_embeddings(
        state_embedding, action_features, action_encoder
    )
    if self.arch != "action_attn":
      return action_embedding
    if state_action_projection is None:
      state_action_projection = self.state_action_projection
    if action_attention is None:
      action_attention = self.action_attention
    if action_attention_gate is None:
      action_attention_gate = self.action_attention_gate
    legal = self._legal_action_mask(action_embedding, action_features)
    state_for_action = state_action_projection(state_embedding).unsqueeze(1)
    conditioned = action_embedding + state_for_action
    attended, _ = action_attention(
        conditioned,
        conditioned,
        conditioned,
        key_padding_mask=~legal,
        need_weights=False,
    )
    return action_embedding + action_attention_gate * attended

  def _legal_action_mask(self, action_embedding, action_features):
    if action_features is None:
      return torch.ones(
          action_embedding.shape[:2],
          dtype=torch.bool,
          device=action_embedding.device,
      )
    return action_features[..., 0] > 0.5

  def _pooled_action_embeddings(self, action_embedding, action_features):
    legal = self._legal_action_mask(action_embedding, action_features)
    legal_f = legal.unsqueeze(-1).to(action_embedding.dtype)
    legal_count = legal_f.sum(dim=1).clamp_min(1.0)
    mean_embedding = (action_embedding * legal_f).sum(dim=1) / legal_count
    masked_embedding = action_embedding.masked_fill(~legal.unsqueeze(-1), -1e9)
    max_embedding = masked_embedding.max(dim=1).values
    max_embedding = torch.where(
        legal.any(dim=1, keepdim=True),
        max_embedding,
        torch.zeros_like(max_embedding),
    )
    return mean_embedding, max_embedding

  def _setpool_action_embeddings(self, action_embedding, action_features):
    legal = self._legal_action_mask(action_embedding, action_features)
    legal_f = legal.unsqueeze(-1).to(action_embedding.dtype)
    legal_count = legal_f.sum(dim=1).clamp_min(1.0)
    mean_embedding = (action_embedding * legal_f).sum(dim=1) / legal_count
    centered = (action_embedding - mean_embedding.unsqueeze(1)) * legal_f
    variance = (centered * centered).sum(dim=1) / legal_count
    std_embedding = torch.sqrt(variance.clamp_min(1e-8))

    max_masked = action_embedding.masked_fill(~legal.unsqueeze(-1), -1e9)
    min_masked = action_embedding.masked_fill(~legal.unsqueeze(-1), 1e9)
    max_embedding = max_masked.max(dim=1).values
    min_embedding = min_masked.min(dim=1).values
    has_legal = legal.any(dim=1, keepdim=True)
    max_embedding = torch.where(has_legal, max_embedding, torch.zeros_like(max_embedding))
    min_embedding = torch.where(has_legal, min_embedding, torch.zeros_like(min_embedding))
    return mean_embedding, max_embedding, min_embedding, std_embedding

  def _pair_features(
      self,
      state_embedding,
      action_features,
      action_encoder=None,
      state_action_projection=None,
      action_attention=None,
      action_attention_gate=None,
  ):
    action_embedding = self._action_embeddings(
        state_embedding,
        action_features,
        action_encoder,
        state_action_projection,
        action_attention,
        action_attention_gate,
    )
    state_expanded = state_embedding.unsqueeze(1).expand(
        -1, action_embedding.shape[1], -1
    )
    if state_action_projection is None:
      state_action_projection = self.state_action_projection
    state_for_action = state_action_projection(state_embedding)
    state_for_action = state_for_action.unsqueeze(1).expand_as(action_embedding)
    mean_embedding, max_embedding = self._pooled_action_embeddings(
        action_embedding, action_features
    )
    mean_expanded = mean_embedding.unsqueeze(1).expand_as(action_embedding)
    max_expanded = max_embedding.unsqueeze(1).expand_as(action_embedding)
    if self.arch == "action_setpool":
      mean_embedding, max_embedding, min_embedding, std_embedding = (
          self._setpool_action_embeddings(action_embedding, action_features)
      )
      mean_expanded = mean_embedding.unsqueeze(1).expand_as(action_embedding)
      max_expanded = max_embedding.unsqueeze(1).expand_as(action_embedding)
      min_expanded = min_embedding.unsqueeze(1).expand_as(action_embedding)
      std_expanded = std_embedding.unsqueeze(1).expand_as(action_embedding)
      centered_embedding = action_embedding - mean_expanded
      return torch.cat([
          state_expanded,
          action_embedding,
          state_for_action * action_embedding,
          torch.abs(state_for_action - action_embedding),
          centered_embedding,
          mean_expanded,
          max_expanded,
          min_expanded,
          std_expanded,
      ], dim=-1)
    return torch.cat([
        state_expanded,
        action_embedding,
        state_for_action * action_embedding,
        torch.abs(state_for_action - action_embedding),
        mean_expanded,
        max_expanded,
    ], dim=-1)

  def _value_features(self, state_embedding, action_features):
    if self.arch not in ACTION_CONDITIONED_ARCHS:
      return state_embedding
    action_embedding = self._action_embeddings(state_embedding, action_features)
    if self.arch == "action_setpool":
      mean_embedding, max_embedding, min_embedding, std_embedding = (
          self._setpool_action_embeddings(action_embedding, action_features)
      )
      return torch.cat([
          state_embedding,
          mean_embedding,
          max_embedding,
          min_embedding,
          std_embedding,
      ], dim=-1)
    mean_embedding, max_embedding = self._pooled_action_embeddings(
        action_embedding, action_features
    )
    return torch.cat([state_embedding, mean_embedding, max_embedding], dim=-1)

  def _policy_logits(self, state_embedding, action_features=None):
    if self.arch in ACTION_CONDITIONED_ARCHS:
      return self.policy(
          self._pair_features(state_embedding, action_features)
      ).squeeze(-1)
    return self.policy(state_embedding)

  def _action_value_pair_features(self, state_embedding, action_features):
    if (
        self.arch in ACTION_CONDITIONED_ARCHS
        and self.separate_action_value_encoder
    ):
      return self._pair_features(
          state_embedding,
          action_features,
          self.action_value_encoder,
          self.action_value_state_action_projection,
          getattr(self, "action_value_attention", None),
          getattr(self, "action_value_attention_gate", None),
      )
    return self._pair_features(state_embedding, action_features)

  def _action_paradox_pair_features(self, state_embedding, action_features):
    if (
        self.arch in ACTION_CONDITIONED_ARCHS
        and self.separate_action_paradox_encoder
    ):
      return self._pair_features(
          state_embedding,
          action_features,
          self.action_paradox_encoder,
          self.action_paradox_state_action_projection,
          getattr(self, "action_paradox_attention", None),
          getattr(self, "action_paradox_attention_gate", None),
      )
    return self._pair_features(state_embedding, action_features)

  def _action_paradox_logits(self, state_embedding, action_features=None):
    if self.arch in ACTION_CONDITIONED_ARCHS:
      return self.action_paradox(
          self._action_paradox_pair_features(state_embedding, action_features)
      ).squeeze(-1)
    return self.action_paradox(state_embedding)

  def _action_values(self, state_embedding, action_features=None):
    if self.arch in ACTION_CONDITIONED_ARCHS:
      return torch.tanh(
          self.action_value(
              self._action_value_pair_features(state_embedding, action_features)
          ).squeeze(-1)
      )
    return torch.tanh(self.action_value(state_embedding))

  def forward(self, obs, action_features=None):
    x = self.body(obs)
    return (
        self._policy_logits(x, action_features),
        torch.tanh(self.value(self._value_features(x, action_features))),
    )

  def forward_with_aux(self, obs, action_features=None):
    x = self.body(obs)
    return (
        self._policy_logits(x, action_features),
        torch.tanh(self.value(self._value_features(x, action_features))),
        self.paradox(x),
    )

  def forward_with_action_aux(self, obs, action_features=None):
    x = self.body(obs)
    return (
        self._policy_logits(x, action_features),
        torch.tanh(self.value(self._value_features(x, action_features))),
        self.paradox(x),
        self._action_paradox_logits(x, action_features),
    )

  def forward_with_all_aux(self, obs, action_features=None):
    x = self.body(obs)
    return (
        self._policy_logits(x, action_features),
        torch.tanh(self.value(self._value_features(x, action_features))),
        self.paradox(x),
        self._action_paradox_logits(x, action_features),
        self._action_values(x, action_features),
    )


class ResidualBlock(nn.Module):
  def __init__(self, width):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(width, width),
        nn.ReLU(),
        nn.LayerNorm(width),
        nn.Linear(width, width),
        nn.LayerNorm(width),
    )

  def forward(self, x):
    return F.relu(x + self.layers(x))


class ResidualBody(nn.Module):
  def __init__(self, obs_size, width, depth):
    super().__init__()
    self.input = nn.Sequential(
        nn.Linear(obs_size, width),
        nn.ReLU(),
        nn.LayerNorm(width),
    )
    self.blocks = nn.Sequential(
        *[ResidualBlock(width) for _ in range(max(0, depth))]
    )

  def forward(self, obs):
    return self.blocks(self.input(obs))


class Node:
  def __init__(self, prior, action_risk=0.0, action_value=0.0):
    self.prior = float(prior)
    self.action_risk = float(action_risk)
    self.action_value = float(action_value)
    self.visit_count = 0
    self.value_sum = None
    self.children = {}

  def q_value(self, player):
    if self.visit_count == 0 or self.value_sum is None:
      return 0.0
    return float(self.value_sum[player] / self.visit_count)

  def add_value(self, value):
    if self.value_sum is None:
      self.value_sum = np.zeros_like(value, dtype=np.float32)
    self.value_sum += value
    self.visit_count += 1


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--iterations", type=int, default=5)
  parser.add_argument("--games-per-iter", type=int, default=20)
  parser.add_argument("--sims", type=int, default=64)
  parser.add_argument("--train-steps", type=int, default=100)
  parser.add_argument("--train-snapshot-interval", type=int, default=0)
  parser.add_argument(
      "--loaded-replay-save-best",
      action="store_true",
      help=(
          "During snapshot-based --loaded-replay-train-steps, save the best "
          "validation snapshot as checkpoint_0000_loaded_replay_best.pt."
      ),
  )
  parser.add_argument(
      "--loaded-replay-best-metric",
      choices=(
          "validation_top1",
          "validation_corr",
          "validation_mean_regret",
          "q_policy_validation_top1",
          "q_policy_validation_corr",
          "q_policy_validation_mean_regret",
          "policy_validation_top1",
          "policy_validation_cross_entropy",
          "action_paradox_validation_auc",
          "action_paradox_validation_brier",
          "action_paradox_validation_corr",
          "value_validation_acting_auc",
          "value_validation_acting_brier",
          "value_validation_acting_corr",
          "value_validation_all_auc",
          "value_validation_all_brier",
          "value_validation_all_corr",
      ),
      default="validation_top1",
      help="Validation metric used by --loaded-replay-save-best.",
  )
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--buffer-size", type=int, default=20000)
  parser.add_argument(
      "--arch",
      choices=("mlp", "residual", *ACTION_CONDITIONED_ARCHS),
      default="mlp",
  )
  parser.add_argument("--width", type=int, default=256)
  parser.add_argument("--depth", type=int, default=3)
  parser.add_argument(
      "--separate-action-value-encoder",
      action="store_true",
      help=(
          "Use a dedicated action encoder/projection/attention stack for the "
          "auxiliary per-action value head. This lets Q-ranking diagnostics "
          "train without changing policy action logits."
      ),
  )
  parser.add_argument(
      "--separate-action-paradox-encoder",
      action="store_true",
      help=(
          "Use a dedicated action encoder/projection/attention stack for the "
          "auxiliary per-action paradox/risk head. This lets survival-risk "
          "fits train without changing policy action logits."
      ),
  )
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument(
      "--weight-decay",
      type=float,
      default=0.0,
      help="Adam weight decay used for training, including loaded-replay fits.",
  )
  parser.add_argument(
      "--train-only-appended-action-features",
      action="store_true",
      help=(
          "Freeze all model weights except the first action-encoder input "
          "weights for appended action-feature columns. This lets new tactical "
          "features adapt without drifting the legacy policy."
      ),
  )
  parser.add_argument(
      "--train-action-stack-only",
      action="store_true",
      help=(
          "Freeze the state encoder/value body and train only the action "
          "scoring stack. This gives action_mlp/action_attn more room than "
          "appended-feature adapters while limiting full-network drift."
      ),
  )
  parser.add_argument(
      "--train-policy-action-stack-only",
      action="store_true",
      help=(
          "Freeze the state encoder and auxiliary action heads, training only "
          "the policy action encoder/projection/head. This is useful for "
          "policy imitation when q_policy safety depends on a calibrated "
          "separate action-paradox stack."
      ),
  )
  parser.add_argument(
      "--train-value-head-only",
      action="store_true",
      help=(
          "Freeze all model weights except the state-value head. This is useful "
          "for fitting round-survival calibration targets without drifting the "
          "policy or shared encoders."
      ),
  )
  parser.add_argument(
      "--train-action-value-head-only",
      action="store_true",
      help=(
          "Freeze all model weights except the auxiliary per-action value "
          "head. This isolates sparse counterfactual Q-label diagnostics from "
          "policy/value drift."
      ),
  )
  parser.add_argument(
      "--train-action-value-stack-only",
      action="store_true",
      help=(
          "Freeze all model weights except the dedicated auxiliary action "
          "value stack and head. Requires --separate-action-value-encoder."
      ),
  )
  parser.add_argument(
      "--train-action-paradox-stack-only",
      action="store_true",
      help=(
          "Freeze all model weights except the dedicated auxiliary action "
          "paradox/risk stack and head. Requires "
          "--separate-action-paradox-encoder."
      ),
  )
  parser.add_argument(
      "--train-action-aux-heads-only",
      action="store_true",
      help=(
          "Freeze all model weights except the auxiliary per-action value and "
          "paradox/risk heads. This trains vector-style action diagnostics "
          "without drifting the policy, value trunk, or action encoder."
      ),
  )
  parser.add_argument(
      "--train-action-attention-only",
      action="store_true",
      help=(
          "Freeze all model weights except the legal-action attention layer "
          "and its residual gate. Requires --arch action_attn and is intended "
          "as a low-drift test of attention over legal actions."
      ),
  )
  parser.add_argument(
      "--appended-action-feature-start",
      type=int,
      default=APPENDED_ACTION_FEATURE_START,
      help=(
          "First action-feature column considered appended/adapter-only when "
          "--train-only-appended-action-features is enabled."
      ),
  )
  parser.add_argument("--c-puct", type=float, default=1.8)
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--temperature-drop", type=int, default=16)
  parser.add_argument("--policy-loss-weight", type=float, default=1.0)
  parser.add_argument("--value-loss-weight", type=float, default=1.0)
  parser.add_argument(
      "--policy-target-action-type-weights",
      default="",
      help=(
          "Optional comma-separated policy-loss multipliers by target action "
          "type, for example 'prediction=2,play=1.25'. Action types are "
          "discard,prediction,play,paradox,other. The weighted policy loss is "
          "normalized by total batch weight."
      ),
  )
  parser.add_argument(
      "--policy-target-bucket-weights",
      default="",
      help=(
          "Optional comma-separated policy-loss multipliers by target action "
          "feature bucket. Supported buckets are "
          f"{','.join(TACTICAL_POLICY_BUCKET_NAMES)}. Matching bucket weights "
          "are multiplied, then normalized by total batch weight."
      ),
  )
  parser.add_argument(
      "--policy-target-ranking-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional pairwise policy-logit ranking loss for one-hot policy "
          "targets. The target action must outrank legal alternatives, giving "
          "survival-teacher rows a direct contrastive signal instead of only "
          "cross-entropy mass."
      ),
  )
  parser.add_argument(
      "--policy-target-ranking-margin",
      type=float,
      default=0.0,
      help=(
          "Required logit margin for --policy-target-ranking-loss-weight. "
          "A value of 1 asks the target action to sit at least one logit above "
          "each selected legal alternative."
      ),
  )
  parser.add_argument(
      "--policy-target-ranking-max-negatives",
      type=int,
      default=0,
      help=(
          "Maximum legal alternatives per row for policy-target ranking. "
          "Zero uses all legal alternatives; positive values use the current "
          "hardest alternatives by logit."
      ),
  )
  parser.add_argument(
      "--policy-target-ranking-min-target-prob",
      type=float,
      default=0.999,
      help=(
          "Only apply policy-target ranking to rows whose largest policy "
          "target is at least this high. The default limits it to effectively "
          "one-hot teacher labels."
      ),
  )
  parser.add_argument(
      "--value-loss-mode",
      choices=("all", "acting"),
      default="all",
      help=(
          "Use all players' returns for value loss, or only the acting "
          "player's return from that information state."
      ),
  )
  parser.add_argument("--paradox-loss-weight", type=float, default=0.0)
  parser.add_argument("--paradox-value-penalty", type=float, default=0.0)
  parser.add_argument("--terminal-paradox-penalty", type=float, default=0.0)
  parser.add_argument(
      "--terminal-any-paradox-penalty",
      type=float,
      default=0.0,
      help=(
          "Subtract this score penalty from every player's terminal value "
          "target when any player paradoxed. This trains the same-policy "
          "anti-paradox objective without changing game rules or legal moves."
      ),
  )
  parser.add_argument("--ordinal-value-weight", type=float, default=0.0)
  parser.add_argument("--official-outcome-value-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-loss-weight", type=float, default=0.0)
  parser.add_argument(
      "--action-paradox-labeled-batch-fraction",
      type=float,
      default=0.0,
      help=(
          "When training with sparse counterfactual action-paradox labels, "
          "draw at least this fraction of each batch from examples whose "
          "action-paradox target mask is non-empty. 0 keeps uniform replay "
          "sampling."
      ),
  )
  parser.add_argument(
      "--action-paradox-positive-weight",
      type=float,
      default=1.0,
      help=(
          "Optional BCE multiplier for labeled risky action-paradox targets. "
          "Use with --action-paradox-loss-weight to handle imbalanced "
          "counterfactual action-risk labels."
      ),
  )
  parser.add_argument(
      "--action-paradox-negative-weight",
      type=float,
      default=1.0,
      help=(
          "Optional BCE multiplier for labeled safe action-paradox targets. "
          "Values above 1 make safe labeled actions pull predicted risk down "
          "harder when the replay is dominated by risky labels."
      ),
  )
  parser.add_argument(
      "--action-paradox-ranking-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional pairwise ranking loss over labeled action-paradox risk "
          "targets. This trains the risk head to order candidate actions by "
          "relative paradox danger, matching threshold q_policy reranking more "
          "directly than pointwise BCE alone."
      ),
  )
  parser.add_argument(
      "--action-paradox-ranking-target-scale",
      type=float,
      default=1.0,
      help=(
          "Target-difference scale used to weight pairwise action-paradox "
          "ranking examples; differences at or above this value get full "
          "weight."
      ),
  )
  parser.add_argument(
      "--action-paradox-ranking-min-diff",
      type=float,
      default=1e-6,
      help=(
          "Ignore action-paradox ranking pairs with target differences below "
          "this."
      ),
  )
  parser.add_argument(
      "--action-paradox-terminal-fallback",
      action="store_true",
      help=(
          "Fallback to labeling selected actions with the acting player's "
          "eventual terminal paradox flag when explicit per-action paradox "
          "targets are unavailable. This is noisy and off by default."
      ),
  )
  parser.add_argument(
      "--action-paradox-terminal-fallback-scope",
      choices=("acting", "any"),
      default="acting",
      help=(
          "Target used by --action-paradox-terminal-fallback. 'acting' labels "
          "the acting player's terminal paradox flag; 'any' labels selected "
          "actions with whether any player paradoxed in the round."
      ),
  )
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument("--action-paradox-root-only", action="store_true")
  parser.add_argument(
      "--action-paradox-rerank-mode",
      choices=("additive", "threshold", "relative"),
      default="additive",
      help=(
          "How q_policy uses the action paradox-risk head. additive preserves "
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
          "Optional q_policy bonus for actions that leave the acting player's "
          "own hand with more future color exits. This only affects root "
          "reranking and defaults to off."
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
      "--prediction-hit-policy-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional action-feature auxiliary loss. In states with legal actions "
          "that immediately hit the acting player's prediction, nudge model "
          "probability mass on those actions toward "
          "--prediction-hit-policy-target-mass."
      ),
  )
  parser.add_argument(
      "--prediction-hit-policy-target-mass",
      type=float,
      default=0.45,
      help=(
          "Probability-mass floor used by "
          "--prediction-hit-policy-loss-weight."
      ),
  )
  parser.add_argument(
      "--future-hit-policy-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional action-feature auxiliary loss that discourages excessive "
          "probability mass on actions that hit the prediction while future "
          "tricks remain."
      ),
  )
  parser.add_argument(
      "--future-hit-policy-max-mass",
      type=float,
      default=0.25,
      help="Probability-mass ceiling used by --future-hit-policy-loss-weight.",
  )
  parser.add_argument(
      "--led-token-loss-policy-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional action-feature auxiliary loss that discourages policy mass "
          "on legal play actions that newly lose the led color token when a "
          "follow-led alternative is legal. This targets the local lane "
          "collapse that often causes later forced paradoxes."
      ),
  )
  parser.add_argument(
      "--led-token-loss-policy-max-mass",
      type=float,
      default=0.05,
      help=(
          "Probability-mass ceiling used by "
          "--led-token-loss-policy-loss-weight."
      ),
  )
  parser.add_argument(
      "--lane-capacity-ranking-policy-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional pairwise policy-logit ranking loss over legal actions. "
          "It prefers led-color lane preservation and higher future lane "
          "capacity over actions that newly burn the led-color token or "
          "damage public lane surplus."
      ),
  )
  parser.add_argument(
      "--lane-capacity-ranking-min-diff",
      type=float,
      default=0.25,
      help=(
          "Ignore lane-capacity policy-ranking pairs whose feature-derived "
          "preference score differs by less than this amount."
      ),
  )
  parser.add_argument(
      "--lane-capacity-ranking-target-scale",
      type=float,
      default=2.0,
      help=(
          "Preference-score difference that gives full weight to a pair in "
          "the lane-capacity policy-ranking loss."
      ),
  )
  parser.add_argument(
      "--lane-capacity-ranking-require-led-choice",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "When true, apply lane-capacity policy ranking only in states where "
          "a legal follow-led action exists and another legal action would "
          "newly lose the led-color token."
      ),
  )
  parser.add_argument(
      "--lane-capacity-ranking-token-loss-penalty",
      type=float,
      default=3.0,
      help="Preference-score penalty for newly losing the led-color token.",
  )
  parser.add_argument(
      "--lane-capacity-ranking-follow-led-bonus",
      type=float,
      default=1.0,
      help="Preference-score bonus for following the led color.",
  )
  parser.add_argument(
      "--lane-capacity-ranking-min-surplus-weight",
      type=float,
      default=1.0,
      help=(
          "Weight on legal-z-scored resulting minimum player lane surplus in "
          "the lane-capacity preference score."
      ),
  )
  parser.add_argument(
      "--lane-capacity-ranking-damage-penalty",
      type=float,
      default=0.75,
      help=(
          "Penalty weight on legal-z-scored resulting lane-surplus damage in "
          "the lane-capacity preference score."
      ),
  )
  parser.add_argument(
      "--lane-capacity-ranking-pressure-penalty",
      type=float,
      default=0.25,
      help=(
          "Penalty weight on legal-z-scored resulting lane-pressure player "
          "count in the lane-capacity preference score."
      ),
  )
  parser.add_argument(
      "--dangerous-future-hit-policy-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional action-feature auxiliary loss that only caps early hit "
          "actions whose post-hit hand looks hard to survive without "
          "overshooting."
      ),
  )
  parser.add_argument(
      "--dangerous-future-hit-policy-max-mass",
      type=float,
      default=0.25,
      help=(
          "Probability-mass ceiling used by "
          "--dangerous-future-hit-policy-loss-weight."
      ),
  )
  parser.add_argument(
      "--dangerous-future-hit-low-legal-ratio-threshold",
      type=float,
      default=0.25,
      help=(
          "Marks an early hit as dangerous when the post-hit low-card legal "
          "lead ratio is at or below this threshold."
      ),
  )
  parser.add_argument(
      "--dangerous-future-hit-survival-margin-threshold",
      type=float,
      default=0.0,
      help=(
          "Marks an early hit as dangerous when low-card count minus remaining "
          "tricks, normalized by round length, is below this threshold."
      ),
  )
  parser.add_argument(
      "--dangerous-future-hit-forced-pressure-threshold",
      type=float,
      default=0.75,
      help=(
          "Marks an early hit as dangerous when remaining tricks divided by "
          "post-hit hand size is above this threshold."
      ),
  )
  parser.add_argument("--counterfactual-action-rollouts", type=int, default=0)
  parser.add_argument(
      "--counterfactual-action-paradox-scope",
      choices=("acting", "any"),
      default="acting",
      help=(
          "Target used by counterfactual per-action paradox labels. 'acting' "
          "keeps the historical acting-player risk label; 'any' labels whether "
          "any player paradoxed in the rolled-out round, matching the "
          "homogeneous same-policy paradox gate."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-paradox-target-mode",
      choices=("binary", "survival"),
      default="binary",
      help=(
          "How counterfactual per-action paradox labels are scored. 'binary' "
          "keeps the historical 0/1 terminal-or-leaf paradox flag. 'survival' "
          "keeps the same event but discounts paradoxes that happen later in "
          "the rollout, giving the learner an ordering signal when most "
          "candidate actions eventually paradox."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-paradox-survival-weight",
      type=float,
      default=0.5,
      help=(
          "For survival paradox labels, maximum discount applied to a "
          "paradoxing line based on how much of the rollout horizon it "
          "survived. 0 makes survival labels binary; 1 makes late paradoxes "
          "much less risky than immediate paradoxes."
      ),
  )
  parser.add_argument(
      "--counterfactual-paradox-policy-target-weight",
      type=float,
      default=0.0,
      help=(
          "Blend policy targets toward a softmax over low counterfactual "
          "per-action paradox risk labels. Use with "
          "--counterfactual-action-rollouts; when "
          "--counterfactual-action-paradox-scope=any this directly trains the "
          "same event measured by the homogeneous paradox gate."
      ),
  )
  parser.add_argument(
      "--counterfactual-paradox-policy-target-temperature",
      type=float,
      default=0.08,
      help="Softmax temperature for low-risk counterfactual policy shaping.",
  )
  parser.add_argument(
      "--counterfactual-paradox-policy-target-min-actions",
      type=int,
      default=2,
      help=(
          "Minimum number of labeled legal actions required before blending "
          "counterfactual paradox-risk labels into the policy target."
      ),
  )
  parser.add_argument(
      "--counterfactual-paradox-policy-target-min-spread",
      type=float,
      default=0.0,
      help=(
          "Minimum max-min risk spread among labeled counterfactual paradox "
          "labels before policy-target blending is applied."
      ),
  )
  parser.add_argument("--action-value-loss-weight", type=float, default=0.0)
  parser.add_argument(
      "--action-value-ranking-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional pairwise ranking loss over labeled action-value targets. "
          "This trains the Q head to order candidate actions correctly, which "
          "is closer to q_policy reranking than pointwise MSE alone."
      ),
  )
  parser.add_argument(
      "--action-value-ranking-target-scale",
      type=float,
      default=0.10,
      help=(
          "Target-difference scale used to weight pairwise action-value "
          "ranking examples; differences at or above this value get full "
          "weight."
      ),
  )
  parser.add_argument(
      "--action-value-ranking-min-diff",
      type=float,
      default=1e-6,
      help="Ignore action-value ranking pairs with target differences below this.",
  )
  parser.add_argument(
      "--action-value-labeled-batch-fraction",
      type=float,
      default=0.0,
      help=(
          "When training with sparse counterfactual action-value labels, draw "
          "at least this fraction of each batch from examples whose "
          "action-value target mask is non-empty. 0 keeps uniform replay "
          "sampling."
      ),
  )
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument("--action-value-rerank-phases", default="")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  parser.add_argument(
      "--reset-action-value-head",
      action="store_true",
      help=(
          "Reinitialize only the action-value head after loading a checkpoint. "
          "Useful for sparse counterfactual Q-label training when the inherited "
          "head is saturated or was trained on a different target."
      ),
  )
  parser.add_argument(
      "--reset-action-paradox-head",
      action="store_true",
      help=(
          "Reinitialize only the per-action paradox/risk head after loading a "
          "checkpoint. Useful when training from cached per-action risk labels."
      ),
  )
  parser.add_argument("--action-value-root-only", action="store_true")
  parser.add_argument(
      "--action-value-terminal-fallback",
      action="store_true",
      help=(
          "When no per-action counterfactual value labels are present in a "
          "batch, train the selected action-value head toward the terminal "
          "acting-player value. Off by default because it is a noisy fallback "
          "for sparse cached counterfactual-label training."
      ),
  )
  parser.add_argument("--counterfactual-action-value-rollouts", type=int, default=0)
  parser.add_argument(
      "--counterfactual-action-value-objective",
      choices=("score", "survival"),
      default="score",
      help=(
          "Target for sparse counterfactual action-value labels. 'score' keeps "
          "the historical raw-score/value-scale target. 'survival' labels "
          "candidate actions as +1 when the current round reaches terminal "
          "with no scoped paradox and -1 when a scoped paradox occurs, using "
          "--counterfactual-action-paradox-scope for acting-vs-any scope."
      ),
  )
  parser.add_argument("--counterfactual-action-value-advantage", action="store_true")
  parser.add_argument(
      "--counterfactual-action-value-min-spread",
      type=float,
      default=0.0,
      help=(
          "Only train action-value labels for a state when sampled legal "
          "actions differ by at least this raw score spread. This filters "
          "low-signal counterfactual labels."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-value-max-stderr",
      type=float,
      default=0.0,
      help=(
          "Only train action-value labels whose rollout standard error is at "
          "most this raw score value. 0 disables the filter."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-value-audit-rollouts",
      type=int,
      default=0,
      help=(
          "After first-pass counterfactual action-value labels pick a best "
          "action, independently re-evaluate that label-best action against "
          "the policy-best labeled action with this many fresh rollout seeds "
          "per belief state. 0 disables the audit."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-value-audit-min-margin",
      type=float,
      default=0.0,
      help=(
          "Raw-score margin required for the audited label-best action to beat "
          "the policy-best labeled action. Used only when audit rollouts are "
          "enabled and the two actions differ."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-value-confidence-weight",
      action="store_true",
      help=(
          "Weight action-value label loss by rollout confidence instead of "
          "using a uniform mask. Requires at least two samples per action to "
          "estimate uncertainty."
      ),
  )
  parser.add_argument(
      "--counterfactual-full-match-rollout",
      action="store_true",
      help=(
          "When labeling counterfactual action values during full-match "
          "training, roll out the rest of the match from the current decision "
          "and use cumulative match total instead of only the current round "
          "score."
      ),
  )
  parser.add_argument(
      "--counterfactual-rollout-max-plies",
      type=int,
      default=0,
      help=(
          "If positive, counterfactual action-value and paradox-risk rollouts "
          "stop after this many non-chance plies after the candidate action. "
          "Action-value labels use the neural value head as a leaf estimate; "
          "paradox-risk labels use the paradox flags observed by the truncated "
          "leaf. This keeps policy-weighted label generation practical."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-survival-truncated-value",
      type=float,
      default=0.0,
      help=(
          "Neutral target returned by --counterfactual-action-value-objective="
          "survival when a positive --counterfactual-rollout-max-plies stops "
          "before round terminal without observing a scoped paradox."
      ),
  )
  parser.add_argument(
      "--counterfactual-policy-target-weight",
      type=float,
      default=0.0,
      help=(
          "Blend MCTS visit-count policy targets with a softmax policy over "
          "sampled counterfactual action-value labels on states where those "
          "labels are available."
      ),
  )
  parser.add_argument(
      "--counterfactual-policy-target-temperature",
      type=float,
      default=0.08,
      help="Softmax temperature for counterfactual policy-target shaping.",
  )
  parser.add_argument(
      "--counterfactual-policy-target-min-actions",
      type=int,
      default=2,
      help=(
          "Minimum number of labeled legal actions required before blending "
          "counterfactual action-value labels into the policy target."
      ),
  )
  parser.add_argument(
      "--counterfactual-policy-target-min-spread",
      type=float,
      default=0.0,
      help=(
          "Minimum max-min spread among labeled counterfactual policy-target "
          "values after value scaling. States below this threshold keep the "
          "MCTS policy target unchanged."
      ),
  )
  parser.add_argument("--counterfactual-action-max-legal", type=int, default=0)
  parser.add_argument(
      "--counterfactual-action-top-policy",
      type=int,
      default=0,
      help=(
          "When limiting counterfactual legal actions, always include up to this "
          "many highest-policy legal actions before random fill. 0 keeps the "
          "old uniform legal-action sampling."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-include-bots",
      default="",
      help=(
          "Optional comma-separated heuristic bot names whose chosen legal "
          "actions are always included before random fill when "
          "--counterfactual-action-max-legal caps expensive labels. Useful for "
          "paired action-value labels that compare champion actions against "
          "tactical baselines."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-feature-candidates",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "When limiting counterfactual legal actions, reserve candidate slots "
          "for tactical action-feature extremes such as adjacency gain, "
          "prediction-hit, token safety, and future flexibility."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-min-policy-entropy",
      type=float,
      default=0.0,
      help=(
          "Only generate counterfactual labels when the normalized legal policy "
          "entropy is at least this value. 0 disables the hard-state filter."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-max-policy-top-prob",
      type=float,
      default=1.0,
      help=(
          "Only generate counterfactual labels when the model's top legal action "
          "probability is at most this value. 1 disables the hard-state filter."
      ),
  )
  parser.add_argument("--counterfactual-action-label-prob", type=float, default=1.0)
  parser.add_argument(
      "--counterfactual-action-label-max-per-game",
      type=int,
      default=0,
      help=(
          "Maximum counterfactual label attempts per generated game/round. "
          "0 disables the cap. This bounds expensive label probes without "
          "changing the played trajectory."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-label-max-per-phase-per-game",
      type=int,
      default=0,
      help=(
          "Maximum counterfactual label attempts for each phase within a "
          "generated game/round. 0 disables the per-phase cap. Use with "
          "--counterfactual-action-label-phases to keep early phases from "
          "spending the whole bounded-label budget before play states."
      ),
  )
  parser.add_argument(
      "--counterfactual-label-progress-interval",
      type=int,
      default=0,
      help=(
          "If positive, print a compact progress row every N counterfactual "
          "label attempts inside single-process generation. This is diagnostic "
          "only and does not change labels."
      ),
  )
  parser.add_argument(
      "--counterfactual-action-label-phases",
      default="",
      help=(
          "Optional comma-separated phase filter for counterfactual action "
          "labels. Supported names: discard,prediction,play. Empty labels all "
          "decision phases."
      ),
  )
  parser.add_argument(
      "--counterfactual-belief-samples",
      type=int,
      default=1,
      help=(
          "Average counterfactual action labels over this many determinizations "
          "from the acting player's information state. Values above 1 avoid "
          "training action heads on hidden-card facts the observation cannot see."
      ),
  )
  parser.add_argument(
      "--counterfactual-belief-source",
      choices=("infostate", "actual", "mixed", "policy_weighted"),
      default="infostate",
      help=(
          "Hidden-state source for counterfactual action labels. infostate "
          "resamples hidden information from the acting player's view; actual "
          "uses the simulator's true hidden state; mixed uses the true state "
          "plus information-state resamples; policy_weighted resamples many "
          "information-state worlds and reweights them by the reference "
          "policy likelihood of observed public trick plays."
      ),
  )
  parser.add_argument(
      "--counterfactual-belief-candidates",
      type=int,
      default=8,
      help=(
          "Candidate information-state determinizations to score before "
          "resampling --counterfactual-belief-samples policy_weighted worlds."
      ),
  )
  parser.add_argument(
      "--counterfactual-belief-policy-temperature",
      type=float,
      default=1.0,
      help=(
          "Softmax temperature for policy-likelihood belief weights. Lower "
          "values make the posterior sharper."
      ),
  )
  parser.add_argument(
      "--counterfactual-belief-uniform-mix",
      type=float,
      default=0.15,
      help=(
          "Uniform mixture mass for policy_weighted belief sampling, keeping "
          "some exploration even when the reference policy is overconfident."
      ),
  )
  parser.add_argument(
      "--counterfactual-belief-ref-policy-mix",
      default="model:1.0",
      help=(
          "Reference mixture for policy_weighted belief likelihoods. "
          "Comma-separated weights over model, model_avg, model0/model1/etc, "
          "uniform, heuristic, heuristic_target2, and heuristic_adj2; e.g. "
          "model0:0.50,model1:0.25,heuristic:0.15,uniform:0.10."
      ),
  )
  parser.add_argument(
      "--counterfactual-belief-ref-policy-mix-by-phase",
      default="",
      help=(
          "Optional semicolon-separated phase overrides for policy_weighted "
          "belief likelihoods, e.g. "
          "prediction=model:0.65,heuristic:0.25,uniform:0.10;play=model:1.0. "
          "Unspecified phases use --counterfactual-belief-ref-policy-mix."
      ),
  )
  parser.add_argument(
      "--counterfactual-belief-logprob-floor",
      type=float,
      default=1e-6,
      help="Minimum action probability used in belief history log-likelihoods.",
  )
  parser.add_argument(
      "--counterfactual-rollout-opponent-mode",
      choices=("", "policy", "mcts", "belief", "belief_policy"),
      default="",
      help=(
          "Override old-policy opponent mode inside counterfactual label "
          "rollouts. Leave empty to reuse league-opponent-mode."
      ),
  )
  parser.add_argument(
      "--counterfactual-rollout-learner-bot",
      default="",
      help=(
          "Optional fixed bot name to use for learner-seat continuation moves "
          "inside counterfactual label rollouts. Empty keeps the neural "
          "learner policy. This can make anti-paradox label generation much "
          "faster and can evaluate candidate actions under a safe continuation "
          "teacher such as heuristic_safe3."
      ),
  )
  parser.add_argument("--anchor-checkpoint", default=None)
  parser.add_argument("--anchor-kl-weight", type=float, default=0.0)
  parser.add_argument(
      "--anchor-top-action-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Optional cross-entropy loss toward the anchor checkpoint's top "
          "legal action. This is a sharper anti-drift regularizer than full "
          "policy KL for action-stack-only fine-tuning."
      ),
  )
  parser.add_argument(
      "--anchor-top-action-min-prob",
      type=float,
      default=0.0,
      help=(
          "Only apply --anchor-top-action-loss-weight on states where the "
          "anchor assigns at least this probability to its selected action."
      ),
  )
  parser.add_argument(
      "--self-play-policy-mode",
      choices=("mcts", "belief", "policy", "belief_policy", "q_policy"),
      default="belief",
      help=(
          "Actor used when generating learner self-play or league examples. "
          "mcts/belief are AlphaZero-style search targets; policy/belief_policy "
          "use the raw network policy, belief_policy averages over "
          "information-state resamples, and q_policy uses the deployed root "
          "action-value/risk reranker. Defaults to belief search so policy "
          "targets do not depend on the simulator's true hidden cards."
      ),
  )
  parser.add_argument("--self-play-belief-samples", type=int, default=4)
  parser.add_argument("--self-play-belief-sims", type=int, default=16)
  parser.add_argument(
      "--self-play-belief-source",
      choices=("infostate", "policy_weighted", "ranker_resample"),
      default="infostate",
      help=(
          "Hidden-state sampler for self-play belief_policy/belief actors. "
          "policy_weighted uses the shared policy-likelihood posterior and "
          "the --counterfactual-belief-* posterior knobs; ranker_resample "
          "uses a learned contrastive hidden-world scorer."
      ),
  )
  parser.add_argument("--self-play-belief-ranker", default="")
  parser.add_argument("--self-play-belief-ranker-candidates", type=int, default=64)
  parser.add_argument("--self-play-belief-ranker-temperature", type=float, default=0.7)
  parser.add_argument("--self-play-belief-ranker-uniform-mix", type=float, default=0.25)
  parser.add_argument("--value-scale", type=float, default=20.0)
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
  parser.add_argument(
      "--self-play-workers",
      type=int,
      default=0,
      help=(
          "Worker processes for independent self-play/teacher/league game "
          "generation. 0 auto-sizes from CPU count, capped at 16. Workers "
          "always run on CPU; the parent can still train batched tensors on "
          "--device=mps."
      ),
  )
  parser.add_argument(
      "--worker-torch-threads",
      type=int,
      default=1,
      help="torch.set_num_threads value inside CPU generation workers.",
  )
  parser.add_argument(
      "--auto-worker-min-games",
      type=int,
      default=32,
      help=(
          "Minimum requested games before --self-play-workers=0 auto-spawns "
          "workers. Explicit positive worker counts ignore this floor."
      ),
  )
  parser.add_argument("--seed", type=int, default=20260602)
  parser.add_argument("--out-dir", default="az_runs/latest")
  parser.add_argument("--eval-games", type=int, default=120)
  parser.add_argument(
      "--eval-full-match",
      action="store_true",
      help=(
          "Evaluate each sample as a full base-game match using total score "
          "and final-round score tiebreak instead of one-round returns."
      ),
  )
  parser.add_argument("--eval-rotate-start-player", action="store_true")
  parser.add_argument("--eval-mcts-sims", type=int, default=0)
  parser.add_argument("--eval-belief-samples", type=int, default=0)
  parser.add_argument("--eval-belief-sims", type=int, default=16)
  parser.add_argument(
      "--eval-belief-source",
      choices=("infostate", "policy_weighted", "ranker_resample"),
      default="infostate",
      help=(
          "Hidden-state sampler for eval belief_policy/belief actors. "
          "policy_weighted uses the shared policy-likelihood posterior and "
          "the --counterfactual-belief-* posterior knobs; ranker_resample "
          "uses a learned contrastive hidden-world scorer."
      ),
  )
  parser.add_argument("--eval-belief-ranker", default="")
  parser.add_argument("--eval-belief-ranker-candidates", type=int, default=64)
  parser.add_argument("--eval-belief-ranker-temperature", type=float, default=0.7)
  parser.add_argument("--eval-belief-ranker-uniform-mix", type=float, default=0.25)
  parser.add_argument("--bootstrap-games", type=int, default=0)
  parser.add_argument("--bootstrap-bots", default="heuristic_target2,heuristic,heuristic_adj2")
  parser.add_argument("--bootstrap-train-steps", type=int, default=0)
  parser.add_argument("--teacher-checkpoint", default=None)
  parser.add_argument("--teacher-games", type=int, default=0)
  parser.add_argument("--teacher-train-steps", type=int, default=0)
  parser.add_argument("--teacher-temperature", type=float, default=0.35)
  parser.add_argument(
      "--teacher-min-target-prob",
      type=float,
      default=0.0,
      help=(
          "When >0, keep only teacher replay rows whose legal target policy "
          "assigns at least this probability to its top action. The teacher "
          "still plays through all states; this filters distillation targets."
      ),
  )
  parser.add_argument(
      "--teacher-min-target-margin",
      type=float,
      default=0.0,
      help=(
          "When >0, keep only teacher replay rows whose best legal action "
          "exceeds the second-best legal action by at least this margin."
      ),
  )
  parser.add_argument(
      "--teacher-max-target-entropy",
      type=float,
      default=1.0,
      help=(
          "Keep only teacher rows with normalized legal-policy entropy at or "
          "below this value. 1 keeps all rows; values near 0 keep only sharp "
          "targets."
      ),
  )
  parser.add_argument(
      "--teacher-output-json",
      default=None,
      help="Optional path for durable teacher generation/training JSON output.",
  )
  parser.add_argument(
      "--teacher-progress-interval",
      type=int,
      default=0,
      help=(
          "When >0, write partial teacher JSON to --teacher-output-json every "
          "N generated teacher games. Defaults to "
          "--generate-replay-progress-interval when unset/0."
      ),
  )
  parser.add_argument(
      "--teacher-mode",
      choices=(
          "policy",
          "q_policy",
          "rollout_select",
          "builtin_policy",
          "mcts",
          "belief_policy",
          "belief_mcts",
      ),
      default="policy",
      help=(
          "Generate teacher targets from the raw policy, q-policy risk/value "
          "reranker, rollout-selected legal actions, built-in bot actions, "
          "teacher MCTS, "
          "belief-averaged raw policy, or belief-averaged teacher MCTS."
      ),
  )
  parser.add_argument(
      "--teacher-builtin-bot",
      default="heuristic_safe14",
      help=(
          "Built-in bot name used by --teacher-mode=builtin_policy. Combine "
          "with --counterfactual-action-label-phases to distill only selected "
          "phases."
      ),
  )
  parser.add_argument(
      "--teacher-belief-samples",
      type=int,
      default=4,
      help="Belief samples for --teacher-mode=belief_policy or belief_mcts.",
  )
  parser.add_argument(
      "--teacher-belief-source",
      choices=("infostate", "policy_weighted", "ranker_resample"),
      default="infostate",
      help=(
          "Hidden-state sampler for teacher belief_policy/belief_mcts targets. "
          "policy_weighted uses public-history likelihood instead of uniform "
          "information-state samples."
      ),
  )
  parser.add_argument("--teacher-belief-candidates", type=int, default=8)
  parser.add_argument(
      "--teacher-belief-policy-temperature", type=float, default=1.0
  )
  parser.add_argument("--teacher-belief-uniform-mix", type=float, default=0.15)
  parser.add_argument("--teacher-belief-ref-policy-mix", default="model:1.0")
  parser.add_argument("--teacher-belief-ref-policy-mix-by-phase", default="")
  parser.add_argument("--teacher-belief-logprob-floor", type=float, default=1e-6)
  parser.add_argument("--teacher-belief-ranker", default="")
  parser.add_argument("--teacher-belief-ranker-candidates", type=int, default=64)
  parser.add_argument("--teacher-belief-ranker-temperature", type=float, default=0.7)
  parser.add_argument("--teacher-belief-ranker-uniform-mix", type=float, default=0.25)
  parser.add_argument(
      "--teacher-sims",
      type=int,
      default=0,
      help="MCTS simulations for --teacher-mode=mcts; defaults to --sims.",
  )
  parser.add_argument(
      "--q-policy-teacher-confirm-rollouts",
      type=int,
      default=0,
      help=(
          "When >0 with --teacher-mode=q_policy, keep only q-policy reranker "
          "targets whose proposed action beats the raw-policy action under "
          "paired counterfactual rollouts."
      ),
  )
  parser.add_argument(
      "--q-policy-teacher-confirm-min-paradox-improvement",
      type=float,
      default=1e-6,
      help=(
          "Minimum baseline_paradox_rate - q_policy_paradox_rate required by "
          "--q-policy-teacher-confirm-rollouts."
      ),
  )
  parser.add_argument(
      "--q-policy-teacher-confirm-min-score-margin",
      type=float,
      default=0.0,
      help=(
          "Minimum q_policy_score - baseline_score required by "
          "--q-policy-teacher-confirm-rollouts."
      ),
  )
  parser.add_argument(
      "--rollout-select-teacher-rollouts",
      type=int,
      default=1,
      help=(
          "Paired rollouts per sampled belief state and candidate action for "
          "--teacher-mode=rollout_select."
      ),
  )
  parser.add_argument(
      "--rollout-select-teacher-min-actions",
      type=int,
      default=2,
      help="Minimum labeled legal actions required for rollout-select targets.",
  )
  parser.add_argument(
      "--rollout-select-teacher-min-paradox-improvement",
      type=float,
      default=1e-6,
      help=(
          "Minimum baseline-policy-top paradox-rate reduction required before "
          "keeping a rollout-select teacher target."
      ),
  )
  parser.add_argument(
      "--rollout-select-teacher-min-score-margin",
      type=float,
      default=0.0,
      help=(
          "Minimum score margin versus the baseline policy-top action required "
          "before keeping a rollout-select teacher target."
      ),
  )
  parser.add_argument(
      "--rollout-select-teacher-continuation-role",
      choices=("learner", "q_policy_teacher"),
      default="learner",
      help=(
          "Policy role used after the first candidate action in rollout-select "
          "teacher probes. Use learner plus --counterfactual-rollout-learner-bot "
          "to evaluate under a fixed tactical continuation bot."
      ),
  )
  parser.add_argument(
      "--rollout-select-teacher-keep-policy-best",
      action="store_true",
      help=(
          "Also keep rollout-select targets when the rollout-selected action is "
          "already the raw policy top action. Off by default to focus on "
          "behavior-changing anti-paradox labels."
      ),
  )
  parser.add_argument("--league-games", type=int, default=0)
  parser.add_argument("--league-checkpoint", default=None)
  parser.add_argument("--league-bots", default="heuristic")
  parser.add_argument(
      "--league-opponent-mode",
      choices=("policy", "mcts", "belief", "belief_policy", "mixed"),
      default="policy",
      help=(
          "How frozen neural league opponents choose actions. The default "
          "keeps the historical greedy policy behavior; mcts/belief train "
          "against search-backed checkpoint opponents. mixed cycles frozen "
          "checkpoint roles through policy, mcts, and belief modes."
      ),
  )
  parser.add_argument(
      "--league-belief-source",
      choices=("infostate", "policy_weighted", "ranker_resample"),
      default="infostate",
      help=(
          "Hidden-state sampler for frozen neural league opponents when "
          "--league-opponent-mode uses belief_policy. policy_weighted uses "
          "the shared policy-likelihood posterior and the "
          "--counterfactual-belief-* posterior knobs; ranker_resample uses "
          "a learned contrastive hidden-world scorer."
      ),
  )
  parser.add_argument("--league-belief-ranker", default="")
  parser.add_argument("--league-belief-ranker-candidates", type=int, default=64)
  parser.add_argument("--league-belief-ranker-temperature", type=float, default=0.7)
  parser.add_argument("--league-belief-ranker-uniform-mix", type=float, default=0.25)
  parser.add_argument("--league-train-steps", type=int, default=0)
  parser.add_argument("--league-progress-interval", type=int, default=0)
  parser.add_argument("--replay-warmup-games", type=int, default=0)
  parser.add_argument("--replay-warmup-train-steps", type=int, default=0)
  parser.add_argument("--load-replay", default=None)
  parser.add_argument(
      "--loaded-replay-train-steps",
      type=int,
      default=0,
      help=(
          "After --load-replay, train on the loaded replay without generating "
          "new games. Useful for cached counterfactual-label corpora."
      ),
  )
  parser.add_argument(
      "--action-value-validation-fraction",
      type=float,
      default=0.0,
      help=(
          "For loaded replay training, withhold this fraction of rows that "
          "have action-value labels from optimization and report them as "
          "action_value_validation_report. Defaults to 0 for backward "
          "compatibility."
      ),
  )
  parser.add_argument(
      "--loaded-replay-validation-label-kind",
      choices=("action_value", "action_paradox", "policy", "any"),
      default="action_value",
      help=(
          "Which sparse replay labels define rows withheld by "
          "--action-value-validation-fraction. Defaults to the historical "
          "action_value behavior."
      ),
  )
  parser.add_argument(
      "--action-value-validation-replay",
      default="",
      help=(
          "Optional comma-separated replay_latest.npz files used only for "
          "action-value validation reports during --loaded-replay-train-steps. "
          "When set, no labeled rows are withheld from --load-replay; this "
          "enables independent validation corpora."
      ),
  )
  parser.add_argument(
      "--value-validation-replay",
      dest="action_value_validation_replay",
      help=(
          "Alias for --action-value-validation-replay when the held-out replay "
          "is used for state-value calibration reports."
      ),
  )
  parser.add_argument(
      "--action-value-validation-seed",
      type=int,
      default=20260603,
      help="Deterministic seed for the loaded-replay action-value validation split.",
  )
  parser.add_argument(
      "--action-value-filter-min-spread",
      type=float,
      default=0.0,
      help=(
          "For loaded replay training, zero action-value labels on rows whose "
          "labeled target max-min spread is below this threshold."
      ),
  )
  parser.add_argument(
      "--action-value-filter-min-top-margin",
      type=float,
      default=0.0,
      help=(
          "For loaded replay training, zero action-value labels on rows whose "
          "best-vs-second-best labeled target margin is below this threshold."
      ),
  )
  parser.add_argument(
      "--action-value-filter-phases",
      default="",
      help=(
          "For loaded replay training, zero action-value labels on rows whose "
          "phase is not in this comma-separated set. Supported names include "
          "discard,prediction,play. Empty keeps all phases."
      ),
  )
  parser.add_argument(
      "--generate-replay-games",
      type=int,
      default=0,
      help=(
          "Generate self-play examples, including any configured "
          "counterfactual labels, save replay_latest.npz, and optionally stop "
          "before training when --generate-replay-only is set."
      ),
  )
  parser.add_argument("--generate-replay-only", action="store_true")
  parser.add_argument("--generate-replay-progress-interval", type=int, default=0)
  parser.add_argument("--random-start-player", action="store_true")
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--random-match-context", action="store_true")
  parser.add_argument("--full-match-training", action="store_true")
  parser.add_argument("--eval-checkpoint", default=None)
  parser.add_argument("--eval-candidate", default=None)
  parser.add_argument("--eval-opponents", default="heuristic,heuristic_target2,random")
  parser.add_argument(
      "--eval-opponent-checkpoint",
      default=None,
      help=(
          "Optional second neural checkpoint exposed to --eval-opponents as "
          "az_opponent. Useful for direct checkpoint-vs-checkpoint promotion "
          "matches."
      ),
  )
  parser.add_argument(
      "--eval-output-json",
      default=None,
      help="Optional path for durable final eval JSON output.",
  )
  parser.add_argument(
      "--eval-progress-interval",
      type=int,
      default=0,
      help=(
          "When >0, write partial eval JSON to --eval-output-json every N "
          "completed games/matches. Useful for long search confirmations."
      ),
  )
  parser.add_argument("--resume-checkpoint", default=None)
  parser.add_argument("--discard-resume-metrics", action="store_true")
  parser.add_argument(
      "--resume-architecture",
      choices=("checkpoint", "cli"),
      default="checkpoint",
      help=(
          "When resuming, use checkpoint architecture metadata or keep the "
          "CLI-provided --arch/--width/--depth and partially load compatible "
          "weights. Use cli for capacity-growth experiments."
      ),
  )
  return parser.parse_args()


def make_game(args, start_player=0):
  return pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": args.players,
          "start_player": int(start_player),
          "match_context": int(getattr(args, "match_context", False)),
      },
  )


def split_csv(value):
  return [item.strip() for item in str(value).split(",") if item.strip()]


def is_old_policy_role(role):
  return role == "old_policy" or role.startswith("old_policy_")


def old_policy_index(role):
  if role == "old_policy":
    return 0
  if role.startswith("old_policy_"):
    return int(role.rsplit("_", 1)[1])
  raise ValueError(f"Not an old-policy role: {role}")


def old_policy_model(opponent_models, role):
  if isinstance(opponent_models, (list, tuple)):
    return opponent_models[old_policy_index(role) % len(opponent_models)]
  return opponent_models


def maybe_set_match_context(state, args):
  if not getattr(args, "match_context", False):
    return
  if getattr(args, "random_match_context", False):
    totals = np.random.randint(-8, 17, size=args.players).astype(np.float32)
    round_index = int(np.random.randint(args.players))
  else:
    totals = np.zeros(args.players, dtype=np.float32)
    round_index = int(getattr(args, "current_match_round", 0))
  state.set_match_context(totals, round_index)


def largest_cluster_from_board(board, player):
  visited = np.zeros(board.shape, dtype=bool)
  max_cluster = 0
  num_colors, num_ranks = board.shape
  for color_idx in range(num_colors):
    for rank_idx in range(num_ranks):
      if board[color_idx, rank_idx] != player or visited[color_idx, rank_idx]:
        continue
      size = 0
      stack = [(color_idx, rank_idx)]
      visited[color_idx, rank_idx] = True
      while stack:
        color, rank = stack.pop()
        size += 1
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
          next_color = color + dc
          next_rank = rank + dr
          if (
              0 <= next_color < num_colors
              and 0 <= next_rank < num_ranks
              and not visited[next_color, next_rank]
              and board[next_color, next_rank] == player
          ):
            visited[next_color, next_rank] = True
            stack.append((next_color, next_rank))
      max_cluster = max(max_cluster, size)
  return max_cluster


def cluster_frontier_profile(board, player):
  """Return component/frontier counts for a player's public board cells."""
  visited = np.zeros(board.shape, dtype=bool)
  num_colors, num_ranks = board.shape
  components = 0
  frontier = set()
  for color_idx in range(num_colors):
    for rank_idx in range(num_ranks):
      if board[color_idx, rank_idx] != player or visited[color_idx, rank_idx]:
        continue
      components += 1
      stack = [(color_idx, rank_idx)]
      visited[color_idx, rank_idx] = True
      while stack:
        color, rank = stack.pop()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
          next_color = color + dc
          next_rank = rank + dr
          if not (0 <= next_color < num_colors and 0 <= next_rank < num_ranks):
            continue
          if board[next_color, next_rank] == -1:
            frontier.add((next_color, next_rank))
          elif (
              board[next_color, next_rank] == player
              and not visited[next_color, next_rank]
          ):
            visited[next_color, next_rank] = True
            stack.append((next_color, next_rank))
  return {
      "components": float(components),
      "frontier": float(len(frontier)),
  }


def would_win_after_play(state, player, rank, color):
  plays = list(getattr(state, "_cards_played_this_trick", []))
  if not plays:
    return None
  plays[player] = (rank, color)
  if any(card is None for card in plays):
    return None
  red_plays = [(seat, card[0]) for seat, card in enumerate(plays)
               if card[1] == "R"]
  if red_plays:
    return max(red_plays, key=lambda item: item[1])[0] == player
  led = getattr(state, "_led_color", None) or color
  led_plays = [(seat, card[0]) for seat, card in enumerate(plays)
               if card[1] == led]
  if not led_plays:
    led_plays = [(seat, card[0]) for seat, card in enumerate(plays)]
  return max(led_plays, key=lambda item: item[1])[0] == player


def action_type(action, state):
  if action == 999:
    return "paradox"
  phase = int(getattr(state, "_phase", 0))
  if phase == 1:
    return "discard"
  if phase == 2:
    return "prediction"
  if phase == 3:
    return "play"
  return "other"


def hand_play_flexibility(
    hand, board, color_tokens, player, num_ranks, leading=True,
    red_broken=False,
):
  legal_count = 0
  legal_ranks = set()
  legal_colors = set()
  dead_cards = 0
  singleton_cards = 0
  for rank_idx in range(num_ranks):
    card_count = int(hand[rank_idx]) if rank_idx < len(hand) else 0
    if card_count <= 0:
      continue
    rank_legal_colors = []
    for color_idx in range(min(board.shape[0], color_tokens.shape[0])):
      if not color_tokens[color_idx]:
        continue
      if board[color_idx, rank_idx] != -1:
        continue
      rank_legal_colors.append(color_idx)
    if (
        leading
        and not red_broken
        and rank_legal_colors
        and any(color_idx != 0 for color_idx in rank_legal_colors)
    ):
      rank_legal_colors = [
          color_idx for color_idx in rank_legal_colors if color_idx != 0
      ]
    if rank_legal_colors:
      legal_count += len(rank_legal_colors) * card_count
      legal_ranks.add(rank_idx)
      legal_colors.update(rank_legal_colors)
      if len(rank_legal_colors) == 1:
        singleton_cards += card_count
    else:
      dead_cards += card_count
  total_cards = max(1, int(np.sum(hand)))
  return {
      "legal_count": float(legal_count),
      "legal_rank_count": float(len(legal_ranks)),
      "legal_color_count": float(len(legal_colors)),
      "dead_card_count": float(dead_cards),
      "singleton_card_count": float(singleton_cards),
      "total_cards": float(total_cards),
  }


def rank_slot_pressure(hand, board, color_tokens, num_ranks, buffer=1):
  """Safe-flex pressure used to expose avoidable future paradox risk."""
  total_slots = 0.0
  rank_deficit = 0.0
  tight_rank_count = 0.0
  min_rank_surplus = None
  max_rank_deficit = 0.0
  buffer_deficit = 0.0
  dead_rank_count = 0.0
  remaining_cards = 0
  for rank_idx in range(num_ranks):
    card_count = int(hand[rank_idx]) if rank_idx < len(hand) else 0
    if card_count <= 0:
      continue
    remaining_cards += card_count
    available = 0
    for color_idx in range(min(board.shape[0], color_tokens.shape[0])):
      if not color_tokens[color_idx]:
        continue
      if board[color_idx, rank_idx] != -1:
        continue
      available += 1
    total_slots += float(available)
    deficit = float(max(0, card_count - available))
    rank_deficit += deficit
    max_rank_deficit = max(max_rank_deficit, deficit)
    surplus = float(available - card_count)
    min_rank_surplus = (
        surplus if min_rank_surplus is None else min(min_rank_surplus, surplus)
    )
    buffer_deficit += float(max(0, card_count + int(buffer) - available))
    if available <= 0:
      dead_rank_count += 1.0
    if available <= card_count:
      tight_rank_count += 1.0
  return {
      "total_slots": total_slots,
      "rank_deficit": rank_deficit,
      "tight_rank_count": tight_rank_count,
      "min_rank_surplus": 0.0 if min_rank_surplus is None else min_rank_surplus,
      "max_rank_deficit": max_rank_deficit,
      "buffer_deficit": buffer_deficit,
      "no_exit": remaining_cards > 0 and total_slots <= 0,
      "dead_rank_count": dead_rank_count,
  }


def safe_flex_score_from_pressure(pressure, num_ranks):
  return (
      0.20 * pressure["total_slots"]
      - pressure["tight_rank_count"]
      - 30.0 * pressure["rank_deficit"]
  ) / max(1.0, 30.0 * float(num_ranks))


def public_remaining_cards_by_player(state, num_players, num_tricks):
  """Public remaining-card counts from trick progress, not hidden hand makeup."""
  current_trick = list(getattr(state, "_cards_played_this_trick", []))
  trick_number = int(getattr(state, "_trick_number", 0))
  remaining = []
  for seat in range(num_players):
    try:
      played = int(state._count_cards_played_by(seat))  # pylint: disable=protected-access
    except Exception:
      played = trick_number
      if seat < len(current_trick) and current_trick[seat] is not None:
        played += 1
    remaining.append(max(0, int(num_tricks) - played))
  return remaining


def public_exit_liquidity_from_board(board, color_tokens, remaining_cards=None):
  """Public future color-rank exit slots implied by board and tokens only."""
  board = np.asarray(board, dtype=np.int32)
  tokens = np.asarray(color_tokens, dtype=bool)
  if tokens.ndim == 1:
    tokens = tokens.reshape(1, -1)
  num_players = int(tokens.shape[0]) if tokens.ndim == 2 else 1
  num_colors = int(board.shape[0]) if board.ndim >= 2 else 0
  open_by_color = [
      int(np.sum(board[color_idx] == -1)) for color_idx in range(num_colors)
  ]
  player_open_slots = []
  for seat in range(num_players):
    slots = 0
    for color_idx, open_slots in enumerate(open_by_color):
      if color_idx < tokens.shape[1] and bool(tokens[seat, color_idx]):
        slots += int(open_slots)
    player_open_slots.append(int(slots))
  if remaining_cards is None:
    player_remaining_cards = [0 for _ in range(num_players)]
  else:
    player_remaining_cards = [
        int(remaining_cards[seat]) if seat < len(remaining_cards) else 0
        for seat in range(num_players)
    ]
  player_lane_surplus = [
      int(player_open_slots[seat] - player_remaining_cards[seat])
      for seat in range(num_players)
  ]
  return {
      "open_cells": int(np.sum(board == -1)),
      "player_open_slots": player_open_slots,
      "player_remaining_cards": player_remaining_cards,
      "player_lane_surplus": player_lane_surplus,
      "total_player_open_slots": int(sum(player_open_slots)),
      "min_player_open_slots": int(
          min(player_open_slots) if player_open_slots else 0
      ),
      "total_player_remaining_cards": int(sum(player_remaining_cards)),
      "total_player_lane_surplus": int(sum(player_lane_surplus)),
      "min_player_lane_surplus": int(
          min(player_lane_surplus) if player_lane_surplus else 0
      ),
      "lane_pressure_player_count": int(
          sum(1 for surplus in player_lane_surplus if surplus < 0)
      ),
  }


def normalized_color_tokens_array(color_tokens, num_players, num_colors):
  if color_tokens is None:
    return np.ones((num_players, num_colors), dtype=bool)
  tokens = np.asarray(color_tokens, dtype=bool)
  if tokens.ndim == 1:
    tokens = tokens.reshape(1, -1)
  normalized = np.zeros((num_players, num_colors), dtype=bool)
  rows = min(num_players, tokens.shape[0])
  cols = min(num_colors, tokens.shape[1])
  if rows > 0 and cols > 0:
    normalized[:rows, :cols] = tokens[:rows, :cols]
  return normalized


def own_hand_feasibility_profile(
    hand, board, color_tokens, num_ranks, num_colors, *,
    leading=False, red_broken=True
):
  """Per-card feasible color counts for the acting player's remaining hand."""
  hand = np.asarray(hand, dtype=np.float32)
  board = np.asarray(board, dtype=np.int32)
  tokens = np.asarray(color_tokens, dtype=bool)
  if tokens.ndim != 1:
    tokens = tokens.reshape(-1)
  total_cards = float(np.sum(hand))
  color_denominator = max(1.0, float(num_colors))
  if total_cards <= 0:
    return {
        "min_colors": color_denominator,
        "mean_colors": color_denominator,
        "zero_exit_frac": 0.0,
        "one_exit_frac": 0.0,
        "two_or_less_exit_frac": 0.0,
        "sum_log_colors": 1.0,
        "legal_count_norm": 1.0,
    }

  min_colors = color_denominator
  total_feasible = 0.0
  zero_exit_cards = 0.0
  one_exit_cards = 0.0
  two_or_less_exit_cards = 0.0
  sum_log_colors = 0.0
  log_denominator = math.log1p(color_denominator)
  for rank_idx in range(num_ranks):
    card_count = float(hand[rank_idx]) if rank_idx < len(hand) else 0.0
    if card_count <= 0:
      continue
    feasible_colors = []
    for color_idx in range(min(board.shape[0], tokens.shape[0], num_colors)):
      if not tokens[color_idx]:
        continue
      if rank_idx >= board.shape[1] or board[color_idx, rank_idx] != -1:
        continue
      feasible_colors.append(color_idx)
    if (
        leading
        and not red_broken
        and feasible_colors
        and any(color_idx != 0 for color_idx in feasible_colors)
    ):
      feasible_colors = [
          color_idx for color_idx in feasible_colors if color_idx != 0
      ]
    feasible_count = float(len(feasible_colors))
    min_colors = min(min_colors, feasible_count)
    total_feasible += card_count * feasible_count
    if feasible_count <= 0:
      zero_exit_cards += card_count
    if feasible_count <= 1:
      one_exit_cards += card_count
    if feasible_count <= 2:
      two_or_less_exit_cards += card_count
    sum_log_colors += card_count * math.log1p(feasible_count)
  return {
      "min_colors": min_colors,
      "mean_colors": total_feasible / max(1.0, total_cards),
      "zero_exit_frac": zero_exit_cards / max(1.0, total_cards),
      "one_exit_frac": one_exit_cards / max(1.0, total_cards),
      "two_or_less_exit_frac": two_or_less_exit_cards / max(1.0, total_cards),
      "sum_log_colors": (
          sum_log_colors / max(1e-8, total_cards * log_denominator)
      ),
      "legal_count_norm": (
          total_feasible / max(1.0, total_cards * color_denominator)
      ),
  }


def future_token_loss_fragility_features(
    hand, board, color_tokens, num_ranks, base_pressure=None
):
  """Worst remaining-hand pressure if one additional color token is lost.

  Current rank-slot pressure assumes the player's color tokens stay fixed after
  the candidate action. In Cat in the Box, future off-led plays can remove one
  more token and make previously playable ranks dead. These features expose
  that fragility without using hidden cards or rollout outcomes.
  """
  tokens = np.array(color_tokens, dtype=bool)
  token_count = int(np.sum(tokens))
  post_cards = max(1.0, float(np.sum(hand)))
  if base_pressure is None:
    base_pressure = rank_slot_pressure(hand, board, tokens, num_ranks)
  base_score = safe_flex_score_from_pressure(base_pressure, num_ranks)
  if token_count <= 0:
    return [0.0, 0.0, 0.0, 0.0, base_score, 0.0]

  max_rank_deficit = 0.0
  max_buffer_deficit = 0.0
  max_dead_rank_count = 0.0
  no_exit_count = 0.0
  worst_score = base_score
  for color_idx, available in enumerate(tokens):
    if not available:
      continue
    reduced_tokens = np.copy(tokens)
    reduced_tokens[color_idx] = False
    pressure = rank_slot_pressure(hand, board, reduced_tokens, num_ranks)
    max_rank_deficit = max(max_rank_deficit, pressure["rank_deficit"])
    max_buffer_deficit = max(max_buffer_deficit, pressure["buffer_deficit"])
    max_dead_rank_count = max(max_dead_rank_count, pressure["dead_rank_count"])
    if pressure["no_exit"]:
      no_exit_count += 1.0
    worst_score = min(
        worst_score, safe_flex_score_from_pressure(pressure, num_ranks)
    )
  return [
      max_rank_deficit / post_cards,
      max_buffer_deficit / post_cards,
      no_exit_count / max(1.0, float(token_count)),
      max_dead_rank_count / max(1.0, float(num_ranks)),
      worst_score,
      max(0.0, base_score - worst_score),
  ]


def action_flexibility_features(
    state, player, action, type_name, rank_idx, color_idx
):
  num_ranks = int(getattr(state, "_num_card_types", MAX_RANK_FEATURES))
  hand_before = np.array(getattr(state, "_hands", [np.zeros(num_ranks)])[player])
  board_before = np.array(getattr(state, "_board_ownership", np.zeros((4, 6))))
  color_tokens_all = getattr(state, "_color_tokens", None)
  if color_tokens_all is None:
    tokens_before = np.ones(MAX_COLOR_FEATURES, dtype=bool)
  else:
    tokens_before = np.array(color_tokens_all[player], dtype=bool)
  red_broken_before = bool(
      getattr(state, "_trump_broken", False)
      or np.any(board_before[0] >= 0)
  )
  lead_before = hand_play_flexibility(
      hand_before, board_before, tokens_before, player, num_ranks,
      leading=True, red_broken=red_broken_before,
  )
  nonlead_before = hand_play_flexibility(
      hand_before, board_before, tokens_before, player, num_ranks,
      leading=False, red_broken=True,
  )

  hand_after = np.copy(hand_before)
  board_after = np.copy(board_before)
  tokens_after = np.copy(tokens_before)
  red_broken_after = red_broken_before
  led_color = getattr(state, "_led_color", None)
  color_names = ["R", "B", "Y", "G"]
  if type_name in ("discard", "play") and 0 <= rank_idx < len(hand_after):
    hand_after[rank_idx] = max(0, hand_after[rank_idx] - 1)
  if (
      type_name == "play"
      and 0 <= color_idx < board_after.shape[0]
      and 0 <= rank_idx < board_after.shape[1]
  ):
    board_after[color_idx, rank_idx] = player
    if color_idx == 0:
      red_broken_after = True
    if led_color is not None and color_idx < len(color_names):
      color = color_names[color_idx]
      if color != led_color and led_color in color_names:
        tokens_after[color_names.index(led_color)] = False

  lead_after = hand_play_flexibility(
      hand_after, board_after, tokens_after, player, num_ranks,
      leading=True, red_broken=red_broken_after,
  )
  nonlead_after = hand_play_flexibility(
      hand_after, board_after, tokens_after, player, num_ranks,
      leading=False, red_broken=True,
  )
  denom_actions = max(1.0, float(num_ranks * MAX_COLOR_FEATURES))
  denom_cards = max(1.0, lead_before["total_cards"])
  return [
      lead_before["legal_count"] / denom_actions,
      lead_after["legal_count"] / denom_actions,
      (lead_after["legal_count"] - lead_before["legal_count"]) / denom_actions,
      nonlead_after["legal_count"] / denom_actions,
      lead_after["dead_card_count"] / denom_cards,
      (lead_after["dead_card_count"] - lead_before["dead_card_count"]) /
      denom_cards,
      lead_after["legal_rank_count"] / max(1.0, float(num_ranks)),
      lead_after["legal_color_count"] / max(1.0, float(MAX_COLOR_FEATURES)),
      (
          1.0
          if hand_after.sum() > 0 and lead_after["legal_count"] <= 0
          else 0.0
      ),
  ]


def action_feature_vector(state, player, action, legal_actions=None):
  legal_set = set(legal_actions if legal_actions is not None else [])
  phase = int(getattr(state, "_phase", 0))
  num_players = int(getattr(state, "_num_players", 1))
  num_ranks = int(getattr(state, "_num_card_types", MAX_RANK_FEATURES))
  num_colors = int(getattr(state, "_num_colors", MAX_COLOR_FEATURES))
  num_tricks = max(1, int(getattr(state, "_num_tricks", 1)))
  hand = np.array(getattr(state, "_hands", [np.zeros(num_ranks)])[player])
  board = np.array(getattr(state, "_board_ownership", np.zeros((4, 6))))
  predictions = getattr(state, "_predictions", [-1])
  tricks_won = getattr(state, "_tricks_won", np.zeros(player + 1))
  prediction = int(predictions[player]) if player < len(predictions) else -1
  tricks = int(tricks_won[player]) if player < len(tricks_won) else 0
  wants_more = prediction < 0 or tricks < prediction
  total_hand = float(np.sum(hand))
  legal_actions = list(legal_actions if legal_actions is not None
                       else state.legal_actions(player))
  type_name = action_type(action, state)
  rank_idx = -1
  color_idx = -1
  prediction_value = -1
  if type_name == "discard":
    rank_idx = int(action)
  elif type_name == "prediction":
    prediction_value = int(action) - 100
  elif type_name == "play":
    color_idx = int(action) // num_ranks
    rank_idx = int(action) % num_ranks

  features = [1.0 if action in legal_set else 0.0]
  for value in range(5):
    features.append(1.0 if phase == value else 0.0)
  for name in ("discard", "prediction", "play", "paradox", "other"):
    features.append(1.0 if type_name == name else 0.0)
  for idx in range(MAX_RANK_FEATURES):
    features.append(1.0 if rank_idx == idx else 0.0)
  for idx in range(MAX_COLOR_FEATURES):
    features.append(1.0 if color_idx == idx else 0.0)

  led_color = getattr(state, "_led_color", None)
  color_names = ["R", "B", "Y", "G"]
  features.append(1.0 if led_color is None else 0.0)
  for name in color_names:
    features.append(1.0 if led_color == name else 0.0)

  rank = rank_idx + 1 if rank_idx >= 0 else 0
  color = color_names[color_idx] if 0 <= color_idx < len(color_names) else None
  hand_count = float(hand[rank_idx]) if 0 <= rank_idx < len(hand) else 0.0
  hand_count_after = max(0.0, hand_count - 1.0) if type_name in (
      "discard", "play") else hand_count
  largest_before = largest_cluster_from_board(board, player)
  largest_after = largest_before
  adjacency_gain = 0.0
  board_free = 0.0
  cluster_before = cluster_frontier_profile(board, player)
  cluster_after = cluster_before
  if type_name == "play" and 0 <= color_idx < board.shape[0] and 0 <= rank_idx < board.shape[1]:
    board_free = 1.0 if board[color_idx, rank_idx] == -1 else 0.0
    after_board = np.copy(board)
    after_board[color_idx, rank_idx] = player
    largest_after = largest_cluster_from_board(after_board, player)
    adjacency_gain = float(largest_after - largest_before)
    cluster_after = cluster_frontier_profile(after_board, player)

  follows_led = 0.0
  off_led_loses_token = 0.0
  if type_name == "play" and led_color is not None and color is not None:
    follows_led = 1.0 if color == led_color else 0.0
    off_led_loses_token = 0.0 if color == led_color else 1.0
  would_complete = (
      type_name == "play"
      and sum(card is not None for card in getattr(
          state, "_cards_played_this_trick", []
      )) + 1 >= int(getattr(state, "_num_players", 1))
  )
  would_win = (
      would_win_after_play(state, player, rank, color)
      if type_name == "play" and color is not None else None
  )
  win_aligns_target = (
      1.0 if would_win is not None and bool(would_win) == bool(wants_more)
      else 0.0
  )
  outcome_tricks = tricks
  if would_win is not None and bool(would_win):
    outcome_tricks += 1
  outcome_prediction_gap = (
      (prediction - outcome_tricks) / max(1.0, float(num_tricks))
      if prediction >= 0 and would_win is not None else 0.0
  )
  outcome_hits_prediction = (
      prediction >= 0 and would_win is not None and outcome_tricks == prediction
  )
  outcome_overshoots_prediction = (
      prediction >= 0 and would_win is not None and outcome_tricks > prediction
  )
  outcome_still_short_prediction = (
      prediction >= 0 and would_win is not None and outcome_tricks < prediction
  )
  would_end_round = (
      would_win is not None
      and int(getattr(state, "_trick_number", 0)) + 1 >= num_tricks
  )
  current_trick_number = int(getattr(state, "_trick_number", 0))
  remaining_tricks_after_count = max(
      0,
      num_tricks - current_trick_number - (1 if would_complete else 0),
  )
  wins_needed_after_count = (
      max(0, prediction - outcome_tricks)
      if prediction >= 0 and would_win is not None else 0
  )
  can_still_hit_after = (
      prediction >= 0
      and would_win is not None
      and 0 <= wins_needed_after_count <= remaining_tricks_after_count
  )
  must_win_all_remaining_after = (
      can_still_hit_after
      and remaining_tricks_after_count > 0
      and wins_needed_after_count == remaining_tricks_after_count
  )
  hit_with_future_tricks = (
      outcome_hits_prediction and remaining_tricks_after_count > 0
  )
  end_round_score_estimate = 0.0
  if would_end_round:
    end_round_score_estimate = float(outcome_tricks)
    if outcome_hits_prediction:
      end_round_score_estimate += float(largest_after)
    end_round_score_estimate /= max(1.0, float(num_tricks + num_ranks))
  same_rank_legal = sum(
      1 for legal in legal_actions
      if type_name == "play" and legal != 999 and legal < 100
      and legal % num_ranks == rank_idx
  )
  same_color_legal = sum(
      1 for legal in legal_actions
      if type_name == "play" and legal != 999 and legal < 100
      and legal // num_ranks == color_idx
  )
  color_token_available = 0.0
  color_tokens = getattr(state, "_color_tokens", None)
  all_tokens_before = normalized_color_tokens_array(
      color_tokens, num_players, num_colors
  )
  if color_tokens is None or player >= all_tokens_before.shape[0]:
    player_tokens_before = np.ones(MAX_COLOR_FEATURES, dtype=bool)
  else:
    player_tokens_before = np.array(all_tokens_before[player], dtype=bool)
  if (
      color_tokens is not None
      and 0 <= color_idx < all_tokens_before.shape[1]
      and player < all_tokens_before.shape[0]
  ):
    color_token_available = 1.0 if all_tokens_before[player][color_idx] else 0.0

  post_hand = np.copy(hand)
  post_board = np.copy(board)
  if color_tokens is None:
    post_tokens = np.ones(MAX_COLOR_FEATURES, dtype=bool)
  else:
    post_tokens = np.array(color_tokens[player], dtype=bool)
  post_red_broken = bool(
      getattr(state, "_trump_broken", False)
      or (post_board.shape[0] > 0 and np.any(post_board[0] >= 0))
  )
  if type_name in ("discard", "play") and 0 <= rank_idx < len(post_hand):
    post_hand[rank_idx] = max(0, post_hand[rank_idx] - 1)
  if (
      type_name == "play"
      and 0 <= color_idx < post_board.shape[0]
      and 0 <= rank_idx < post_board.shape[1]
  ):
    post_board[color_idx, rank_idx] = player
    if color_idx == 0:
      post_red_broken = True
    if led_color is not None and 0 <= color_idx < len(color_names):
      played_color = color_names[color_idx]
      if played_color != led_color and led_color in color_names:
        post_tokens[color_names.index(led_color)] = False
  post_lead_flex = hand_play_flexibility(
      post_hand, post_board, post_tokens, player, num_ranks,
      leading=True, red_broken=post_red_broken,
  )
  lost_token_idx = -1
  token_loss_newly_loses_led = 0.0
  if (
      type_name == "play"
      and led_color is not None
      and color is not None
      and color != led_color
      and led_color in color_names
  ):
    lost_token_idx = color_names.index(led_color)
    token_loss_newly_loses_led = (
        1.0
        if (
            0 <= lost_token_idx < len(player_tokens_before)
            and bool(player_tokens_before[lost_token_idx])
            and not bool(post_tokens[lost_token_idx])
        )
        else 0.0
    )
  no_loss_lead_flex = post_lead_flex
  if token_loss_newly_loses_led > 0.5 and 0 <= lost_token_idx < len(post_tokens):
    no_loss_tokens = np.copy(post_tokens)
    no_loss_tokens[lost_token_idx] = True
    no_loss_lead_flex = hand_play_flexibility(
        post_hand, post_board, no_loss_tokens, player, num_ranks,
        leading=True, red_broken=post_red_broken,
    )
  token_loss_legal_lead_damage = max(
      0.0, no_loss_lead_flex["legal_count"] - post_lead_flex["legal_count"]
  )
  token_loss_dead_card_damage = max(
      0.0, post_lead_flex["dead_card_count"] - no_loss_lead_flex["dead_card_count"]
  )
  token_loss_singleton_card_damage = max(
      0.0,
      post_lead_flex["singleton_card_count"] -
      no_loss_lead_flex["singleton_card_count"],
  )
  token_loss_rank_damage = max(
      0.0,
      no_loss_lead_flex["legal_rank_count"] -
      post_lead_flex["legal_rank_count"],
  )
  token_loss_creates_no_exit = (
      1.0
      if (
          token_loss_newly_loses_led > 0.5
          and no_loss_lead_flex["legal_count"] > 0
          and post_lead_flex["legal_count"] <= 0
      )
      else 0.0
  )
  all_tokens_after = np.copy(all_tokens_before)
  if player < all_tokens_after.shape[0]:
    cols = min(all_tokens_after.shape[1], post_tokens.shape[0])
    all_tokens_after[player, :cols] = post_tokens[:cols]
  remaining_cards_before = public_remaining_cards_by_player(
      state, num_players, num_tricks
  )
  remaining_cards_after = list(remaining_cards_before)
  if type_name == "play" and player < len(remaining_cards_after):
    remaining_cards_after[player] = max(0, remaining_cards_after[player] - 1)
  before_exit = public_exit_liquidity_from_board(
      board, all_tokens_before, remaining_cards_before
  )
  after_exit = public_exit_liquidity_from_board(
      post_board, all_tokens_after, remaining_cards_after
  )
  board_cell_denominator = max(1.0, float(num_colors * num_ranks))
  public_slot_denominator = max(
      1.0, float(max(1, num_players)) * board_cell_denominator
  )
  pressure_count_denominator = max(1.0, float(num_players))
  before_own_slots = (
      before_exit["player_open_slots"][player]
      if player < len(before_exit["player_open_slots"]) else 0
  )
  after_own_slots = (
      after_exit["player_open_slots"][player]
      if player < len(after_exit["player_open_slots"]) else 0
  )
  after_own_lane_surplus = (
      after_exit["player_lane_surplus"][player]
      if player < len(after_exit["player_lane_surplus"]) else 0
  )
  if type_name == "paradox":
    exit_public_slot_damage = (
        before_exit["total_player_open_slots"] / public_slot_denominator
    )
    exit_own_public_slot_damage = before_own_slots / board_cell_denominator
    exit_board_open_cell_damage = before_exit["open_cells"] / board_cell_denominator
    exit_min_player_open_slots_after = 0.0
    exit_total_player_open_slots_after = 0.0
    exit_own_lane_surplus_after = 0.0
    exit_min_player_lane_surplus_after = 0.0
    exit_total_player_lane_surplus_after = 0.0
    exit_lane_surplus_damage = (
        before_exit["total_player_lane_surplus"] / public_slot_denominator
    )
    exit_min_lane_surplus_damage = (
        before_exit["min_player_lane_surplus"] / board_cell_denominator
    )
    exit_lane_pressure_player_count_after = 1.0
  else:
    exit_public_slot_damage = max(
        0.0,
        before_exit["total_player_open_slots"] -
        after_exit["total_player_open_slots"],
    ) / public_slot_denominator
    exit_own_public_slot_damage = max(
        0.0, before_own_slots - after_own_slots
    ) / board_cell_denominator
    exit_board_open_cell_damage = max(
        0.0, before_exit["open_cells"] - after_exit["open_cells"]
    ) / board_cell_denominator
    exit_min_player_open_slots_after = (
        after_exit["min_player_open_slots"] / board_cell_denominator
    )
    exit_total_player_open_slots_after = (
        after_exit["total_player_open_slots"] / public_slot_denominator
    )
    exit_own_lane_surplus_after = (
        after_own_lane_surplus / board_cell_denominator
    )
    exit_min_player_lane_surplus_after = (
        after_exit["min_player_lane_surplus"] / board_cell_denominator
    )
    exit_total_player_lane_surplus_after = (
        after_exit["total_player_lane_surplus"] / public_slot_denominator
    )
    exit_lane_surplus_damage = max(
        0.0,
        before_exit["total_player_lane_surplus"] -
        after_exit["total_player_lane_surplus"],
    ) / public_slot_denominator
    exit_min_lane_surplus_damage = max(
        0.0,
        before_exit["min_player_lane_surplus"] -
        after_exit["min_player_lane_surplus"],
    ) / board_cell_denominator
    exit_lane_pressure_player_count_after = (
        after_exit["lane_pressure_player_count"] / pressure_count_denominator
    )
  cluster_denominator = max(1.0, float(board.shape[0] * board.shape[1]))
  cluster_frontier_before = cluster_before["frontier"]
  cluster_frontier_after = cluster_after["frontier"]
  cluster_components_before = cluster_before["components"]
  cluster_components_after = cluster_after["components"]
  cluster_connects_components = (
      1.0
      if (
          type_name == "play"
          and cluster_components_before > 0
          and cluster_components_after < cluster_components_before
      )
      else 0.0
  )
  cluster_dead_end_after = (
      1.0
      if (
          type_name == "play"
          and largest_after > largest_before
          and cluster_frontier_after <= 0
      )
      else 0.0
  )
  low_cutoff = max(1, min(num_ranks, 2))
  high_start = max(0, num_ranks - 2)
  post_cards = max(1.0, float(np.sum(post_hand)))
  low_card_count = float(np.sum(post_hand[:low_cutoff]))
  high_card_count = float(np.sum(post_hand[high_start:num_ranks]))
  low_legal_lead_count = 0.0
  for low_rank_idx in range(low_cutoff):
    card_count = float(post_hand[low_rank_idx]) if low_rank_idx < len(post_hand) else 0.0
    if card_count <= 0:
      continue
    legal_color_indices = []
    for candidate_color_idx in range(min(post_board.shape[0], post_tokens.shape[0])):
      if not post_tokens[candidate_color_idx]:
        continue
      if post_board[candidate_color_idx, low_rank_idx] != -1:
        continue
      legal_color_indices.append(candidate_color_idx)
    if (
        not post_red_broken
        and legal_color_indices
        and any(candidate_color_idx != 0 for candidate_color_idx in legal_color_indices)
    ):
      legal_color_indices = [
          candidate_color_idx
          for candidate_color_idx in legal_color_indices
          if candidate_color_idx != 0
      ]
    low_legal_lead_count += card_count * len(legal_color_indices)
  post_hit_gate = 1.0 if hit_with_future_tricks else 0.0
  post_hit_low_card_frac = post_hit_gate * low_card_count / post_cards
  post_hit_high_card_frac = post_hit_gate * high_card_count / post_cards
  post_hit_legal_lead_count_after = (
      post_hit_gate * post_lead_flex["legal_count"] /
      max(1.0, float(num_ranks * MAX_COLOR_FEATURES))
  )
  post_hit_dead_card_frac_after = (
      post_hit_gate * post_lead_flex["dead_card_count"] / post_cards
  )
  post_hit_low_legal_lead_ratio = (
      post_hit_gate * low_legal_lead_count /
      max(1.0, post_lead_flex["legal_count"])
  )
  post_hit_low_card_survival_margin = (
      post_hit_gate * (low_card_count - remaining_tricks_after_count) /
      max(1.0, float(num_tricks))
  )
  post_hit_forced_card_pressure = (
      post_hit_gate * remaining_tricks_after_count / post_cards
  )
  rank_pressure_before = rank_slot_pressure(
      hand, board, player_tokens_before, num_ranks
  )
  rank_pressure_after = rank_slot_pressure(
      post_hand, post_board, post_tokens, num_ranks
  )
  own_future_profile = own_hand_feasibility_profile(
      post_hand,
      post_board,
      post_tokens,
      num_ranks,
      num_colors,
      leading=False,
      red_broken=True,
  )
  own_future_lead_profile = own_hand_feasibility_profile(
      post_hand,
      post_board,
      post_tokens,
      num_ranks,
      num_colors,
      leading=True,
      red_broken=post_red_broken,
  )
  future_safe_flex_score_before = safe_flex_score_from_pressure(
      rank_pressure_before, num_ranks
  )
  future_safe_flex_score_after = safe_flex_score_from_pressure(
      rank_pressure_after, num_ranks
  )
  future_token_loss_fragility = future_token_loss_fragility_features(
      post_hand,
      post_board,
      post_tokens,
      num_ranks,
      base_pressure=rank_pressure_after,
  )
  discard_hand_count_before_frac = 0.0
  discard_hand_count_after_frac = 0.0
  discard_rank_open_slots_frac = 0.0
  discard_rank_slot_surplus_before = 0.0
  discard_rank_slot_surplus_after = 0.0
  discard_rank_deficit_relief = 0.0
  discard_rank_buffer_relief = 0.0
  discard_safe_flex_delta = 0.0
  discard_dead_rank_relief = 0.0
  discard_tight_rank_relief = 0.0
  discard_removes_singleton = 0.0
  discard_from_duplicate = 0.0
  discard_no_exit_after = 0.0
  if type_name == "discard" and 0 <= rank_idx < num_ranks:
    rank_open_slots = 0.0
    for candidate_color_idx in range(
        min(post_board.shape[0], player_tokens_before.shape[0])
    ):
      if not player_tokens_before[candidate_color_idx]:
        continue
      if post_board[candidate_color_idx, rank_idx] != -1:
        continue
      rank_open_slots += 1.0
    rank_deficit_before = max(0.0, hand_count - rank_open_slots)
    rank_deficit_after = max(0.0, hand_count_after - rank_open_slots)
    rank_buffer_before = max(0.0, hand_count + 1.0 - rank_open_slots)
    rank_buffer_after = max(0.0, hand_count_after + 1.0 - rank_open_slots)
    discard_hand_count_before_frac = hand_count / max(1.0, total_hand)
    discard_hand_count_after_frac = hand_count_after / max(
        1.0, total_hand - 1.0
    )
    discard_rank_open_slots_frac = rank_open_slots / max(
        1.0, float(MAX_COLOR_FEATURES)
    )
    discard_rank_slot_surplus_before = (
        rank_open_slots - hand_count
    ) / max(1.0, float(MAX_COLOR_FEATURES))
    discard_rank_slot_surplus_after = (
        rank_open_slots - hand_count_after
    ) / max(1.0, float(MAX_COLOR_FEATURES))
    discard_rank_deficit_relief = max(
        0.0, rank_deficit_before - rank_deficit_after
    ) / max(1.0, total_hand)
    discard_rank_buffer_relief = max(
        0.0, rank_buffer_before - rank_buffer_after
    ) / max(1.0, total_hand)
    discard_safe_flex_delta = (
        future_safe_flex_score_after - future_safe_flex_score_before
    )
    discard_dead_rank_relief = max(
        0.0,
        rank_pressure_before["dead_rank_count"] -
        rank_pressure_after["dead_rank_count"],
    ) / max(1.0, float(num_ranks))
    discard_tight_rank_relief = max(
        0.0,
        rank_pressure_before["tight_rank_count"] -
        rank_pressure_after["tight_rank_count"],
    ) / max(1.0, float(num_ranks))
    discard_removes_singleton = 1.0 if hand_count <= 1.0 else 0.0
    discard_from_duplicate = 1.0 if hand_count > 1.0 else 0.0
    discard_no_exit_after = 1.0 if rank_pressure_after["no_exit"] else 0.0
  high_card_start = max(0, num_ranks - 2)
  prediction_high_cards = float(np.sum(hand[high_card_start:num_ranks]))
  prediction_duplicate_pressure = float(
      np.sum(np.maximum(0.0, hand.astype(np.float32) - 1.0))
  )
  prediction_expected_tricks = prediction_top_rank_score(state, player)
  prediction_action_gap = 0.0
  prediction_action_abs_gap = 0.0
  prediction_action_under = 0.0
  prediction_action_over = 0.0
  prediction_action_is_min = 0.0
  prediction_action_is_max = 0.0
  if type_name == "prediction" and prediction_value > 0:
    prediction_action_gap = (
        float(prediction_value) - prediction_expected_tricks
    ) / 4.0
    prediction_action_abs_gap = abs(prediction_action_gap)
    prediction_action_under = (
        1.0 if float(prediction_value) < prediction_expected_tricks else 0.0
    )
    prediction_action_over = (
        1.0 if float(prediction_value) > prediction_expected_tricks else 0.0
    )
    prediction_action_is_min = 1.0 if prediction_value == 1 else 0.0
    prediction_action_is_max = 1.0 if prediction_value == 4 else 0.0

  numeric = [
      phase / 4.0,
      float(getattr(state, "_trick_number", 0)) / num_tricks,
      sum(card is not None for card in getattr(
          state, "_cards_played_this_trick", []
      )) / max(1, int(getattr(state, "_num_players", 1))),
      tricks / num_tricks,
      (prediction / 4.0) if prediction >= 0 else 0.0,
      1.0 if prediction >= 0 else 0.0,
      1.0 if wants_more else 0.0,
      1.0 if prediction >= 0 and tricks >= prediction else 0.0,
      ((prediction - tricks) / 4.0) if prediction >= 0 else 0.0,
      rank / max(1, num_ranks),
      hand_count / max(1.0, total_hand),
      hand_count_after / max(1.0, total_hand),
      total_hand / max(1, getattr(state, "_cards_per_player_initial", num_ranks)),
      max(0.0, hand_count - 1.0) / max(1.0, total_hand),
      board_free,
      color_token_available,
      follows_led,
      off_led_loses_token,
      1.0 if would_complete else 0.0,
      0.0 if would_win is None else (1.0 if would_win else -1.0),
      win_aligns_target,
      adjacency_gain / max(1, num_ranks),
      largest_before / max(1, num_ranks),
      largest_after / max(1, num_ranks),
      1.0 if color == "R" else 0.0,
      1.0 if (
          type_name == "play"
          and color == "R"
          and led_color is None
          and not getattr(state, "_trump_broken", False)
      ) else 0.0,
      1.0 if getattr(state, "_trump_broken", False) else 0.0,
      len(legal_actions) / max(1, state.num_distinct_actions()),
      same_rank_legal / max(1, len(legal_actions)),
      same_color_legal / max(1, len(legal_actions)),
      (prediction_value / 4.0) if prediction_value > 0 else 0.0,
      1.0 if action == 999 else 0.0,
  ]
  numeric.extend(
      action_flexibility_features(
          state, player, action, type_name, rank_idx, color_idx
      )
  )
  numeric.extend([
      1.0 if rank_idx == 6 else 0.0,
      1.0 if rank_idx == 7 else 0.0,
      1.0 if rank_idx == 8 else 0.0,
      1.0 if would_win is not None else 0.0,
      1.0 if outcome_hits_prediction else 0.0,
      1.0 if outcome_overshoots_prediction else 0.0,
      1.0 if outcome_still_short_prediction else 0.0,
      outcome_prediction_gap,
      1.0 if would_end_round else 0.0,
      end_round_score_estimate,
      remaining_tricks_after_count / max(1.0, float(num_tricks)),
      wins_needed_after_count / max(1.0, float(num_tricks)),
        1.0 if can_still_hit_after else 0.0,
        1.0 if must_win_all_remaining_after else 0.0,
        1.0 if hit_with_future_tricks else 0.0,
        post_hit_low_card_frac,
        post_hit_high_card_frac,
        post_hit_legal_lead_count_after,
        post_hit_dead_card_frac_after,
        post_hit_low_legal_lead_ratio,
        post_hit_low_card_survival_margin,
        post_hit_forced_card_pressure,
        token_loss_newly_loses_led,
        token_loss_legal_lead_damage / max(1.0, float(num_ranks * MAX_COLOR_FEATURES)),
        token_loss_dead_card_damage / post_cards,
        token_loss_singleton_card_damage / post_cards,
        token_loss_rank_damage / max(1.0, float(num_ranks)),
        token_loss_creates_no_exit,
        cluster_frontier_after / cluster_denominator,
        (cluster_frontier_after - cluster_frontier_before) / cluster_denominator,
        (cluster_components_after - cluster_components_before) /
        max(1.0, float(num_ranks)),
        cluster_connects_components,
        cluster_dead_end_after,
    ])
  numeric.extend([0.0] * len(LEGAL_SET_CONTEXT_BLOCK_FEATURE_NAMES))
  numeric.extend([
      rank_pressure_after["total_slots"] /
      max(1.0, float(num_ranks * MAX_COLOR_FEATURES)),
      rank_pressure_after["rank_deficit"] / post_cards,
      rank_pressure_after["tight_rank_count"] / max(1.0, float(num_ranks)),
      (
          rank_pressure_after["rank_deficit"] -
          rank_pressure_before["rank_deficit"]
      ) / max(1.0, total_hand),
      future_safe_flex_score_after,
      rank_pressure_after["min_rank_surplus"] /
      max(1.0, float(MAX_COLOR_FEATURES)),
      rank_pressure_after["max_rank_deficit"] / post_cards,
      rank_pressure_after["buffer_deficit"] / post_cards,
      1.0 if rank_pressure_after["no_exit"] else 0.0,
      rank_pressure_after["dead_rank_count"] / max(1.0, float(num_ranks)),
  ])
  numeric.extend(future_token_loss_fragility)
  numeric.extend([
      prediction_high_cards / max(1.0, total_hand),
      prediction_duplicate_pressure / max(1.0, total_hand),
      prediction_expected_tricks / 4.0,
      prediction_action_gap,
      prediction_action_abs_gap,
      prediction_action_under,
      prediction_action_over,
      prediction_action_is_min,
      prediction_action_is_max,
      discard_hand_count_before_frac,
      discard_hand_count_after_frac,
      discard_rank_open_slots_frac,
      discard_rank_slot_surplus_before,
      discard_rank_slot_surplus_after,
      discard_rank_deficit_relief,
      discard_rank_buffer_relief,
      discard_safe_flex_delta,
      discard_dead_rank_relief,
      discard_tight_rank_relief,
      discard_removes_singleton,
      discard_from_duplicate,
      discard_no_exit_after,
      exit_public_slot_damage,
      exit_own_public_slot_damage,
      exit_board_open_cell_damage,
      exit_min_player_open_slots_after,
      exit_total_player_open_slots_after,
      exit_own_lane_surplus_after,
      exit_min_player_lane_surplus_after,
      exit_total_player_lane_surplus_after,
      exit_lane_surplus_damage,
      exit_min_lane_surplus_damage,
      exit_lane_pressure_player_count_after,
      token_loss_newly_loses_led,
      1.0 if outcome_overshoots_prediction else 0.0,
  ])
  numeric.extend([0.0] * len(EXIT_LIQUIDITY_LEGAL_SET_CONTEXT_FEATURE_NAMES))
  numeric.extend([
      own_future_profile["min_colors"] / max(1.0, float(num_colors)),
      own_future_profile["mean_colors"] / max(1.0, float(num_colors)),
      own_future_profile["zero_exit_frac"],
      own_future_profile["one_exit_frac"],
      own_future_profile["two_or_less_exit_frac"],
      own_future_profile["sum_log_colors"],
      own_future_lead_profile["min_colors"] / max(1.0, float(num_colors)),
      own_future_lead_profile["zero_exit_frac"],
      own_future_lead_profile["legal_count_norm"],
  ])
  numeric.extend([0.0] * len(
      OWN_HAND_FEASIBILITY_LEGAL_SET_CONTEXT_FEATURE_NAMES
  ))
  features.extend(numeric)
  if len(features) != ACTION_FEATURE_SIZE:
    raise ValueError(f"action feature size drifted to {len(features)}")
  return np.array(features, dtype=np.float32)


def _legal_set_percentiles(values):
  values = np.asarray(values, dtype=np.float32)
  if len(values) <= 1:
    return np.full(len(values), 0.5, dtype=np.float32)
  percentiles = np.zeros(len(values), dtype=np.float32)
  for idx, value in enumerate(values):
    less = float(np.sum(values < value))
    equal = float(np.sum(values == value))
    percentiles[idx] = (less + 0.5 * equal) / float(len(values))
  return percentiles


def add_legal_set_context_features(features, legal):
  legal = list(legal)
  if not legal:
    return features
  legal_rows = features[legal]
  for source_idx, z_name, pct_name in LEGAL_SET_CONTEXT_SOURCES:
    values = legal_rows[:, source_idx].astype(np.float32)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-6:
      z_scores = np.zeros_like(values, dtype=np.float32)
    else:
      z_scores = ((values - mean) / std).astype(np.float32)
    features[legal, APPENDED_ACTION_FEATURE_INDEX[z_name]] = z_scores
    features[legal, APPENDED_ACTION_FEATURE_INDEX[pct_name]] = (
        _legal_set_percentiles(values)
    )
  return features


def action_feature_matrix(state, player, num_actions):
  legal = state.legal_actions(player)
  features = np.zeros((num_actions, ACTION_FEATURE_SIZE), dtype=np.float32)
  for action in legal:
    features[action] = action_feature_vector(state, player, action, legal)
  return add_legal_set_context_features(features, legal)


def adapt_action_features(features, num_actions):
  """Pad/truncate persisted action features to the current feature contract."""
  adapted = np.zeros((num_actions, ACTION_FEATURE_SIZE), dtype=np.float32)
  if features is None:
    return adapted
  arr = np.asarray(features, dtype=np.float32)
  if arr.ndim == 1:
    if num_actions == 1:
      arr = arr.reshape(1, -1)
    elif arr.size % max(1, num_actions) == 0:
      arr = arr.reshape(num_actions, -1)
    else:
      arr = arr.reshape(1, -1)
  if arr.ndim != 2:
    arr = arr.reshape(-1, arr.shape[-1])
  rows = min(num_actions, arr.shape[0])
  cols = min(ACTION_FEATURE_SIZE, arr.shape[1])
  if rows > 0 and cols > 0:
    adapted[:rows, :cols] = arr[:rows, :cols]
  return adapted


def model_input_size(model):
  if getattr(model, "arch", "mlp") == "residual":
    return model.body.input[0].in_features
  return model.body[0].in_features


def adapt_observation(obs, expected_size):
  if obs.shape[0] == expected_size:
    return obs
  if obs.shape[0] > expected_size:
    return obs[:expected_size]
  padded = np.zeros(expected_size, dtype=np.float32)
  padded[:obs.shape[0]] = obs
  return padded


def adapt_observation_batch(obs, expected_size):
  current_size = obs.shape[1]
  if current_size == expected_size:
    return obs
  if current_size > expected_size:
    return obs[:, :expected_size]
  padding = torch.zeros(
      (obs.shape[0], expected_size - current_size),
      dtype=obs.dtype,
      device=obs.device,
  )
  return torch.cat([obs, padding], dim=1)


def load_compatible_state_dict(model, state_dict):
  current = model.state_dict()
  compatible = {}
  for key, value in state_dict.items():
    if key not in current:
      continue
    if current[key].shape == value.shape:
      compatible[key] = value
      continue
    if current[key].dim() == value.dim():
      adapted = torch.zeros_like(current[key])
      slices = tuple(
          slice(0, min(current_dim, saved_dim))
          for current_dim, saved_dim in zip(current[key].shape, value.shape)
      )
      adapted[slices] = value[slices]
      compatible[key] = adapted
  model.load_state_dict(compatible, strict=False)


def initialize_missing_action_value_stack_from_policy(model, state_dict):
  """Seed a new separated Q stack from policy action modules when absent."""
  if not getattr(model, "separate_action_value_encoder", False):
    return
  if "action_value_encoder.0.weight" in state_dict:
    return
  if not hasattr(model, "action_value_encoder"):
    return
  model.action_value_encoder.load_state_dict(model.action_encoder.state_dict())
  model.action_value_state_action_projection.load_state_dict(
      model.state_action_projection.state_dict()
  )
  if hasattr(model, "action_value_attention") and hasattr(model, "action_attention"):
    model.action_value_attention.load_state_dict(model.action_attention.state_dict())
  if (
      hasattr(model, "action_value_attention_gate")
      and hasattr(model, "action_attention_gate")
  ):
    with torch.no_grad():
      model.action_value_attention_gate.copy_(model.action_attention_gate)


def initialize_missing_action_paradox_stack_from_policy(model, state_dict):
  """Seed a new separated risk stack from policy action modules when absent."""
  if not getattr(model, "separate_action_paradox_encoder", False):
    return
  if "action_paradox_encoder.0.weight" in state_dict:
    return
  if not hasattr(model, "action_paradox_encoder"):
    return
  model.action_paradox_encoder.load_state_dict(model.action_encoder.state_dict())
  model.action_paradox_state_action_projection.load_state_dict(
      model.state_action_projection.state_dict()
  )
  if hasattr(model, "action_paradox_attention") and hasattr(
      model, "action_attention"
  ):
    model.action_paradox_attention.load_state_dict(
        model.action_attention.state_dict()
    )
  if (
      hasattr(model, "action_paradox_attention_gate")
      and hasattr(model, "action_attention_gate")
  ):
    with torch.no_grad():
      model.action_paradox_attention_gate.copy_(model.action_attention_gate)


def configure_trainable_parameters(model, args):
  """Optionally restrict training to action-scoring parameters."""
  restricted_modes = [
      getattr(args, "train_only_appended_action_features", False),
      getattr(args, "train_policy_action_stack_only", False),
      getattr(args, "train_value_head_only", False),
      getattr(args, "train_action_stack_only", False),
      getattr(args, "train_action_value_head_only", False),
      getattr(args, "train_action_value_stack_only", False),
      getattr(args, "train_action_paradox_stack_only", False),
      getattr(args, "train_action_aux_heads_only", False),
      getattr(args, "train_action_attention_only", False),
  ]
  if sum(1 for enabled in restricted_modes if enabled) > 1:
    raise ValueError(
      "--train-only-appended-action-features, --train-action-stack-only, "
      "--train-policy-action-stack-only, --train-value-head-only, "
      "--train-action-value-head-only, --train-action-value-stack-only, "
      "--train-action-paradox-stack-only, --train-action-aux-heads-only, "
      "and --train-action-attention-only are mutually exclusive"
    )
  if getattr(args, "train_value_head_only", False):
    if not hasattr(model, "value"):
      raise ValueError("--train-value-head-only requires value")
    trainable = []
    for name, param in model.named_parameters():
      allowed = name.startswith("value.")
      param.requires_grad = allowed
      if allowed:
        trainable.append(param)
    return trainable
  if getattr(args, "train_action_value_head_only", False):
    if not hasattr(model, "action_value"):
      raise ValueError("--train-action-value-head-only requires action_value")
    trainable = []
    for name, param in model.named_parameters():
      allowed = name.startswith("action_value.")
      param.requires_grad = allowed
      if allowed:
        trainable.append(param)
    return trainable
  if getattr(args, "train_action_value_stack_only", False):
    if not getattr(model, "separate_action_value_encoder", False):
      raise ValueError(
          "--train-action-value-stack-only requires "
          "--separate-action-value-encoder"
      )
    trainable = []
    prefixes = (
        "action_value.",
        "action_value_encoder.",
        "action_value_state_action_projection.",
        "action_value_attention.",
    )
    allowed_names = {"action_value_attention_gate"}
    for name, param in model.named_parameters():
      allowed = name in allowed_names or name.startswith(prefixes)
      param.requires_grad = allowed
      if allowed:
        trainable.append(param)
    return trainable
  if getattr(args, "train_action_paradox_stack_only", False):
    if not getattr(model, "separate_action_paradox_encoder", False):
      raise ValueError(
          "--train-action-paradox-stack-only requires "
          "--separate-action-paradox-encoder"
      )
    trainable = []
    prefixes = (
        "action_paradox.",
        "action_paradox_encoder.",
        "action_paradox_state_action_projection.",
        "action_paradox_attention.",
    )
    allowed_names = {"action_paradox_attention_gate"}
    for name, param in model.named_parameters():
      allowed = name in allowed_names or name.startswith(prefixes)
      param.requires_grad = allowed
      if allowed:
        trainable.append(param)
    return trainable
  if getattr(args, "train_action_aux_heads_only", False):
    missing = [
        name for name in ("action_value", "action_paradox")
        if not hasattr(model, name)
    ]
    if missing:
      raise ValueError(
          "--train-action-aux-heads-only requires " + ", ".join(missing)
      )
    trainable = []
    for name, param in model.named_parameters():
      allowed = name.startswith("action_value.") or name.startswith(
          "action_paradox."
      )
      param.requires_grad = allowed
      if allowed:
        trainable.append(param)
    return trainable
  if getattr(args, "train_action_attention_only", False):
    if not hasattr(model, "action_attention") or model.arch != "action_attn":
      raise ValueError("--train-action-attention-only requires action_attn")
    trainable = []
    for name, param in model.named_parameters():
      allowed = name == "action_attention_gate" or name.startswith(
          "action_attention."
      )
      param.requires_grad = allowed
      if allowed:
        trainable.append(param)
    return trainable
  if getattr(args, "train_policy_action_stack_only", False):
    if not hasattr(model, "action_encoder"):
      raise ValueError(
          "--train-policy-action-stack-only requires an action-conditioned arch"
      )
    prefixes = (
        "action_encoder.",
        "state_action_projection.",
        "policy.",
        "action_attention.",
    )
    allowed_names = {"action_attention_gate"}
    trainable = []
    for name, param in model.named_parameters():
      allowed = name in allowed_names or name.startswith(prefixes)
      param.requires_grad = allowed
      if allowed:
        trainable.append(param)
    return trainable
  if getattr(args, "train_action_stack_only", False):
    if not hasattr(model, "action_encoder"):
      raise ValueError(
          "--train-action-stack-only requires an action-conditioned arch"
      )
    prefixes = (
        "action_encoder.",
        "state_action_projection.",
        "policy.",
        "action_paradox.",
        "action_value.",
        "action_attention.",
    )
    allowed_names = {"action_attention_gate"}
    trainable = []
    for name, param in model.named_parameters():
      allowed = name in allowed_names or name.startswith(prefixes)
      param.requires_grad = allowed
      if allowed:
        trainable.append(param)
    return trainable
  if not getattr(args, "train_only_appended_action_features", False):
    return [param for param in model.parameters() if param.requires_grad]
  if not hasattr(model, "action_encoder"):
    raise ValueError(
        "--train-only-appended-action-features requires an action-conditioned arch"
    )
  first_layer = model.action_encoder[0]
  if not isinstance(first_layer, nn.Linear):
    raise ValueError("Expected first action encoder layer to be nn.Linear")
  start = int(
      getattr(args, "appended_action_feature_start",
              APPENDED_ACTION_FEATURE_START)
  )
  if start < 0 or start >= first_layer.weight.shape[1]:
    raise ValueError(
        "--appended-action-feature-start must point inside action features"
    )
  for param in model.parameters():
    param.requires_grad = False
  first_layer.weight.requires_grad = True
  mask = torch.zeros_like(first_layer.weight)
  mask[:, start:] = 1.0

  def mask_legacy_feature_grads(grad):
    return grad * mask.to(device=grad.device, dtype=grad.dtype)

  first_layer.weight.register_hook(mask_legacy_feature_grads)
  return [first_layer.weight]


def parameter_group_name(name):
  if name in (
      "action_attention_gate",
      "action_value_attention_gate",
      "action_paradox_attention_gate",
  ):
    return name
  if "." not in name:
    return name
  return name.split(".", 1)[0]


def parameter_training_report(model):
  """Summarize trainable and frozen model parameters for experiment audits."""
  trainable_names = []
  frozen_names = []
  trainable_numel = 0
  frozen_numel = 0
  group_numel = {}
  group_norm_sq = {}
  for name, param in model.named_parameters():
    numel = int(param.numel())
    group = parameter_group_name(name)
    group_numel[group] = group_numel.get(group, 0) + numel
    with torch.no_grad():
      group_norm_sq[group] = group_norm_sq.get(group, 0.0) + float(
          torch.sum(param.detach().float() ** 2).item()
      )
    if param.requires_grad:
      trainable_names.append(name)
      trainable_numel += numel
    else:
      frozen_names.append(name)
      frozen_numel += numel
  return {
      "trainable_parameter_count": len(trainable_names),
      "frozen_parameter_count": len(frozen_names),
      "trainable_numel": trainable_numel,
      "frozen_numel": frozen_numel,
      "trainable_parameter_names": trainable_names,
      "frozen_parameter_names": frozen_names,
      "parameter_group_numel": {
          key: group_numel[key] for key in sorted(group_numel)
      },
      "parameter_group_norm": {
          key: round(group_norm_sq[key] ** 0.5, 6)
          for key in sorted(group_norm_sq)
      },
  }


def frozen_parameter_snapshot(model):
  """Clone all currently frozen parameters for exact post-train checks."""
  return {
      name: param.detach().cpu().clone()
      for name, param in model.named_parameters()
      if not param.requires_grad
  }


def frozen_parameter_integrity_report(model, before):
  """Report whether any parameter frozen at snapshot time changed later."""
  current_by_name = dict(model.named_parameters())
  changed_names = []
  missing_names = []
  max_abs_delta = 0.0
  checked = 0
  for name, old_value in before.items():
    current = current_by_name.get(name)
    if current is None:
      missing_names.append(name)
      continue
    checked += 1
    new_value = current.detach().cpu()
    if old_value.shape != new_value.shape:
      changed_names.append(name)
      max_abs_delta = float("inf")
      continue
    if not torch.equal(old_value, new_value):
      changed_names.append(name)
      with torch.no_grad():
        delta = float(torch.max(torch.abs(old_value - new_value)).item())
      max_abs_delta = max(max_abs_delta, delta)
  return {
      "checked_parameter_count": checked,
      "missing_parameter_count": len(missing_names),
      "changed_parameter_count": len(changed_names),
      "max_abs_delta": (
          None if max_abs_delta == float("inf") else round(max_abs_delta, 12)
      ),
      "changed_parameter_names": changed_names[:50],
      "missing_parameter_names": missing_names[:50],
      "exact_match": not changed_names and not missing_names,
  }


def add_frozen_parameter_integrity_or_raise(row, model, before, args):
  if before is None:
    return
  report = frozen_parameter_integrity_report(model, before)
  row["frozen_parameter_integrity"] = report
  if (
      getattr(args, "train_policy_action_stack_only", False)
      and not report["exact_match"]
  ):
    raise RuntimeError(
        "--train-policy-action-stack-only changed frozen parameters: "
        + ", ".join(report["changed_parameter_names"])
    )


def reset_action_value_head(model):
  """Reinitialize the auxiliary per-action value head without touching policy."""
  if not hasattr(model, "action_value"):
    raise ValueError("--reset-action-value-head requires an action_value head")
  for module in model.action_value.modules():
    if hasattr(module, "reset_parameters"):
      module.reset_parameters()
  linear_layers = [
      module for module in model.action_value.modules()
      if isinstance(module, nn.Linear)
  ]
  if linear_layers:
    final_layer = linear_layers[-1]
    nn.init.zeros_(final_layer.weight)
    if final_layer.bias is not None:
      nn.init.zeros_(final_layer.bias)


def reset_action_paradox_head(model):
  """Reinitialize the auxiliary per-action risk head without touching policy."""
  if not hasattr(model, "action_paradox"):
    raise ValueError("--reset-action-paradox-head requires an action_paradox head")
  for module in model.action_paradox.modules():
    if hasattr(module, "reset_parameters"):
      module.reset_parameters()
  linear_layers = [
      module for module in model.action_paradox.modules()
      if isinstance(module, nn.Linear)
  ]
  if linear_layers:
    final_layer = linear_layers[-1]
    nn.init.zeros_(final_layer.weight)
    if final_layer.bias is not None:
      nn.init.zeros_(final_layer.bias)


def model_policy_value(
    model, state, player, num_actions, value_scale, device,
    paradox_value_penalty=0.0,
):
  prediction_action = shared_prediction_action(state, player)
  obs_np = np.array(state.observation_tensor(player), dtype=np.float32)
  obs_np = adapt_observation(obs_np, model_input_size(model))
  obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
  action_features_np = action_feature_matrix(state, player, num_actions)
  action_features_t = torch.tensor(
      action_features_np, dtype=torch.float32, device=device
  ).unsqueeze(0)
  with torch.no_grad():
    if paradox_value_penalty > 0 and hasattr(model, "forward_with_aux"):
      logits, value, paradox_logits = model.forward_with_aux(
          obs, action_features_t
      )
      paradox_risk = torch.sigmoid(paradox_logits).squeeze(0).cpu().numpy()
    else:
      logits, value = model(obs, action_features_t)
      paradox_risk = None
    logits = logits.squeeze(0).cpu().numpy()
    value = value.squeeze(0).cpu().numpy() * value_scale
    if paradox_risk is not None:
      value = value - paradox_value_penalty * paradox_risk
  legal = state.legal_actions(player)
  mask = np.full(num_actions, -np.inf, dtype=np.float32)
  mask[legal] = 0.0
  masked = logits + mask
  max_logit = np.max(masked[legal])
  exp = np.zeros(num_actions, dtype=np.float32)
  exp[legal] = np.exp(masked[legal] - max_logit)
  total = float(exp.sum())
  if not math.isfinite(total) or total <= 0:
    exp[legal] = 1.0 / len(legal)
  else:
    exp /= total
  if prediction_action is not None:
    exp[:] = 0.0
    exp[int(prediction_action)] = 1.0
  return exp, value.astype(np.float32)


def model_action_risks(model, state, player, num_actions, device):
  if not hasattr(model, "forward_with_action_aux"):
    return np.zeros(num_actions, dtype=np.float32)
  obs_np = np.array(state.observation_tensor(player), dtype=np.float32)
  obs_np = adapt_observation(obs_np, model_input_size(model))
  obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
  action_features_t = torch.tensor(
      action_feature_matrix(state, player, num_actions),
      dtype=torch.float32,
      device=device,
  ).unsqueeze(0)
  with torch.no_grad():
    _, _, _, action_logits = model.forward_with_action_aux(
        obs, action_features_t
    )
    risks = torch.sigmoid(action_logits).squeeze(0).cpu().numpy()
  return risks.astype(np.float32)


def model_action_values(model, state, player, num_actions, device):
  if not hasattr(model, "action_value"):
    return np.zeros(num_actions, dtype=np.float32)
  obs_np = np.array(state.observation_tensor(player), dtype=np.float32)
  obs_np = adapt_observation(obs_np, model_input_size(model))
  obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
  action_features_t = torch.tensor(
      action_feature_matrix(state, player, num_actions),
      dtype=torch.float32,
      device=device,
  ).unsqueeze(0)
  with torch.no_grad():
    x = model.body(obs)
    values = model._action_values(x, action_features_t).squeeze(0).cpu().numpy()
  return values.astype(np.float32)


def model_action_value_batch(model, obs_t, action_features_t=None):
  x = model.body(obs_t)
  return model._action_values(x, action_features_t)


def terminal_search_value(state, args):
  returns = np.array(state.returns(), dtype=np.float32)
  if args is None:
    return returns
  has_terminal_shaping = (
      float(getattr(args, "terminal_paradox_penalty", 0.0)) > 0
      or float(getattr(args, "terminal_any_paradox_penalty", 0.0)) > 0
      or float(getattr(args, "ordinal_value_weight", 0.0)) > 0
  )
  if not has_terminal_shaping:
    return returns
  paradox_target = np.array(
      getattr(state, "_has_paradoxed", [False] * len(returns)),
      dtype=np.float32,
  )
  return (
      value_targets_from_scores(returns, paradox_target, args)
      * float(getattr(args, "value_scale", 1.0))
  ).astype(np.float32)


def expand(
    node, state, model, num_actions, value_scale, device,
    paradox_value_penalty=0.0,
    action_paradox_selection_penalty=0.0,
    action_value_selection_weight=0.0,
    terminal_value_args=None,
):
  if state.is_terminal():
    return terminal_search_value(state, terminal_value_args)
  player = state.current_player()
  priors, value = model_policy_value(
      model, state, player, num_actions, value_scale, device,
      paradox_value_penalty
  )
  action_risks = (
      model_action_risks(model, state, player, num_actions, device)
      if action_paradox_selection_penalty > 0
      else np.zeros(num_actions, dtype=np.float32)
  )
  action_values = (
      model_action_values(model, state, player, num_actions, device)
      if action_value_selection_weight > 0
      else np.zeros(num_actions, dtype=np.float32)
  )
  for action in state.legal_actions(player):
    node.children[action] = Node(
        priors[action], action_risks[action], action_values[action]
    )
  return value


def run_simulation(
    node, state, model, num_actions, c_puct, value_scale, device,
    paradox_value_penalty=0.0,
    action_paradox_selection_penalty=0.0,
    action_paradox_root_only=False,
    action_value_selection_weight=0.0,
    action_value_root_only=False,
    terminal_value_args=None,
    depth=0,
):
  if state.is_terminal():
    return terminal_search_value(state, terminal_value_args)
  if not node.children:
    value = expand(
        node, state, model, num_actions, value_scale, device,
        paradox_value_penalty,
        (
            action_paradox_selection_penalty
            if not action_paradox_root_only or depth == 0
            else 0.0
        ),
        (
            action_value_selection_weight
            if not action_value_root_only or depth == 0
            else 0.0
        ),
        terminal_value_args,
    )
    return value

  player = state.current_player()
  sqrt_visits = math.sqrt(max(1, node.visit_count))
  best_action = None
  best_score = -float("inf")
  for action, child in node.children.items():
    u = c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
    risk_penalty = (
        action_paradox_selection_penalty
        if not action_paradox_root_only or depth == 0
        else 0.0
    )
    value_bonus = (
        action_value_selection_weight
        if not action_value_root_only or depth == 0
        else 0.0
    )
    score = (
        child.q_value(player)
        - risk_penalty * child.action_risk
        + value_bonus * child.action_value * value_scale
        + u
    )
    if score > best_score:
      best_score = score
      best_action = action

  child = node.children[best_action]
  state.apply_action(best_action)
  value = run_simulation(
      child, state, model, num_actions, c_puct, value_scale, device,
      paradox_value_penalty,
      action_paradox_selection_penalty,
      action_paradox_root_only,
      action_value_selection_weight,
      action_value_root_only,
      terminal_value_args,
      depth + 1,
  )
  child.add_value(value)
  node.add_value(value)
  return value


def mcts_policy(state, model, args, device, add_noise=True, sims=None):
  if not state.is_chance_node() and not state.is_terminal():
    prediction_action = shared_prediction_action(state, state.current_player())
    if prediction_action is not None:
      policy = np.zeros(state.num_distinct_actions(), dtype=np.float32)
      policy[int(prediction_action)] = 1.0
      return policy
  root = Node(1.0)
  paradox_value_penalty = getattr(args, "paradox_value_penalty", 0.0)
  action_paradox_selection_penalty = getattr(
      args, "action_paradox_selection_penalty", 0.0
  )
  action_paradox_root_only = getattr(args, "action_paradox_root_only", False)
  action_value_selection_weight = getattr(args, "action_value_selection_weight", 0.0)
  action_value_root_only = getattr(args, "action_value_root_only", False)
  expand(
      root, state.clone(), model, state.num_distinct_actions(), args.value_scale,
      device, paradox_value_penalty, action_paradox_selection_penalty,
      action_value_selection_weight,
      args,
  )
  if add_noise:
    add_dirichlet_noise(root)
  for _ in range(args.sims if sims is None else sims):
    run_simulation(
        root,
        state.clone(),
        model,
        state.num_distinct_actions(),
        args.c_puct,
        args.value_scale,
        device,
        paradox_value_penalty,
        action_paradox_selection_penalty,
        action_paradox_root_only,
        action_value_selection_weight,
        action_value_root_only,
        args,
        0,
    )
  visits = np.zeros(state.num_distinct_actions(), dtype=np.float32)
  for action, child in root.children.items():
    visits[action] = child.visit_count
  if visits.sum() <= 0:
    legal = state.legal_actions(state.current_player())
    visits[legal] = 1.0 / len(legal)
  else:
    visits /= visits.sum()
  return visits


def belief_mcts_policy(
    state, player, model, args, device, add_noise=True, context="self_play"
):
  """Average root policies over states sampled from player's information state."""
  legal = state.legal_actions(player)
  combined = np.zeros(state.num_distinct_actions(), dtype=np.float32)
  samples = max(1, args.self_play_belief_samples)
  sampled_states = sampled_belief_states_for_policy(
      state,
      player,
      samples,
      args,
      model,
      device,
      args.value_scale,
      context=context,
  )
  for sampled in sampled_states:
    policy = mcts_policy(
        sampled,
        model,
        args,
        device,
        add_noise=add_noise,
        sims=max(1, args.self_play_belief_sims),
    )
    combined += policy
  combined /= float(max(1, len(sampled_states)))
  masked = np.zeros_like(combined)
  masked[legal] = combined[legal]
  total = float(masked.sum())
  if total <= 0 or not math.isfinite(total):
    masked[legal] = 1.0 / len(legal)
  else:
    masked /= total
  return masked


def teacher_belief_mcts_policy(state, player, model, args, device):
  """Average teacher MCTS policies over player's information-state samples."""
  legal = state.legal_actions(player)
  combined = np.zeros(state.num_distinct_actions(), dtype=np.float32)
  sample_count = max(1, int(getattr(args, "teacher_belief_samples", 4)))
  sims = (
      int(getattr(args, "teacher_sims", 0))
      if int(getattr(args, "teacher_sims", 0)) > 0
      else max(1, int(getattr(args, "sims", 1)))
  )
  sampled_states = sampled_belief_states_for_policy(
      state,
      player,
      sample_count,
      args,
      model,
      device,
      args.value_scale,
      context="teacher",
  )
  for sampled in sampled_states:
    combined += mcts_policy(
        sampled,
        model,
        args,
        device,
        add_noise=False,
        sims=sims,
    )
  combined /= float(max(1, len(sampled_states)))
  masked = np.zeros_like(combined)
  masked[legal] = combined[legal]
  total = float(masked.sum())
  if total <= 0 or not math.isfinite(total):
    masked[legal] = 1.0 / len(legal)
  else:
    masked /= total
  return masked


def legal_policy_confidence(policy, legal_mask):
  policy = np.array(policy, dtype=np.float32)
  legal_mask = np.array(legal_mask, dtype=np.float32) > 0.5
  legal_probs = policy[legal_mask]
  if legal_probs.size == 0:
    return {
        "top_prob": 0.0,
        "top_margin": 0.0,
        "normalized_entropy": 1.0,
    }
  total = float(np.sum(legal_probs))
  if total <= 0 or not math.isfinite(total):
    legal_probs = np.full(
        legal_probs.shape, 1.0 / float(len(legal_probs)), dtype=np.float32
    )
  else:
    legal_probs = legal_probs / total
  sorted_probs = np.sort(legal_probs)[::-1]
  top_prob = float(sorted_probs[0])
  second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
  entropy = -float(
      np.sum(legal_probs * np.log(np.maximum(legal_probs, 1e-12)))
  )
  normalized_entropy = (
      entropy / math.log(len(legal_probs)) if len(legal_probs) > 1 else 0.0
  )
  return {
      "top_prob": top_prob,
      "top_margin": top_prob - second_prob,
      "normalized_entropy": normalized_entropy,
  }


def keep_teacher_policy_target(policy, legal_mask, args):
  confidence = legal_policy_confidence(policy, legal_mask)
  min_prob = float(getattr(args, "teacher_min_target_prob", 0.0) or 0.0)
  min_margin = float(getattr(args, "teacher_min_target_margin", 0.0) or 0.0)
  max_entropy = float(getattr(args, "teacher_max_target_entropy", 1.0))
  if confidence["top_prob"] < min_prob:
    return False
  if confidence["top_margin"] < min_margin:
    return False
  if confidence["normalized_entropy"] > max_entropy:
    return False
  return True


def _selection_add_stat(stats, key, amount=1):
  if stats is None:
    return
  if isinstance(amount, (float, np.floating)):
    stats[key] = float(stats.get(key, 0.0)) + float(amount)
  else:
      stats[key] = int(stats.get(key, 0)) + int(amount)


def action_feasibility_scores(action_features):
  """Feature-derived bonus for actions that preserve future own-hand exits."""
  if action_features is None:
    return None
  features = np.asarray(action_features, dtype=np.float32)
  if features.ndim != 2 or features.shape[1] < ACTION_FEATURE_SIZE:
    return np.zeros(features.shape[0], dtype=np.float32)

  def col(name):
    return features[:, APPENDED_ACTION_FEATURE_INDEX[name]]

  min_colors = col("own_future_min_colors_after")
  mean_colors = col("own_future_mean_colors_after")
  legal_min_pct = col("legal_pct_own_future_min_colors_after")
  zero_exit = col("own_future_zero_exit_frac_after")
  one_exit = col("own_future_one_exit_frac_after")
  two_exit = col("own_future_two_or_less_exit_frac_after")
  legal_zero_pct = col("legal_pct_own_future_zero_exit_frac_after")
  legal_one_pct = col("legal_pct_own_future_one_exit_frac_after")
  legal_lead = col("own_future_legal_lead_count_after")
  score = (
      1.20 * legal_min_pct
      + 0.80 * min_colors
      + 0.35 * mean_colors
      + 0.25 * legal_lead
      - 0.90 * legal_zero_pct
      - 0.65 * legal_one_pct
      - 0.65 * zero_exit
      - 0.45 * one_exit
      - 0.20 * two_exit
  )
  return np.asarray(score, dtype=np.float32)


def q_policy_select_action(
    legal,
    policy,
    action_values,
    action_risks,
    args,
    value_scale,
    stats=None,
    feasibility_scores=None,
):
  """Select the q-policy action from already-computed policy/value/risk heads."""
  legal = list(legal)
  baseline_action = max(legal, key=lambda action: float(policy[action]))
  value_weight = float(getattr(args, "action_value_selection_weight", 0.0))
  risk_penalty = float(getattr(args, "action_paradox_selection_penalty", 0.0))
  value_bonus = value_weight * action_values * float(value_scale)
  feasibility_weight = float(
      getattr(args, "action_feasibility_selection_weight", 0.0)
  )
  feasibility_bonus = (
      feasibility_weight * feasibility_scores
      if feasibility_scores is not None and feasibility_weight != 0.0 else
      np.zeros_like(policy, dtype=np.float32)
  )
  log_policy = {
      action: np.log(max(float(policy[action]), 1e-12)) for action in legal
  }
  mode = str(
      getattr(args, "action_paradox_rerank_mode", "additive") or "additive"
  ).lower()
  risk_available = action_risks is not None and len(action_risks) > 0
  if risk_available:
    legal_risks = [float(action_risks[action]) for action in legal]
    baseline_risk = float(action_risks[baseline_action])
    min_risk = float(min(legal_risks))
    _selection_add_stat(stats, "rerank_risk_diagnostics")
    _selection_add_stat(stats, "rerank_baseline_risk_sum", baseline_risk)
    _selection_add_stat(stats, "rerank_min_risk_sum", min_risk)
    _selection_add_stat(
        stats, "rerank_risk_spread_sum", float(max(legal_risks) - min_risk)
    )
  if mode == "threshold":
    _selection_add_stat(stats, "rerank_threshold_considered")
    if not risk_available:
      _selection_add_stat(stats, "rerank_threshold_missing_risk")
      return baseline_action
    risk_threshold = float(getattr(args, "action_paradox_risk_threshold", 0.0))
    min_risk_margin = float(
        getattr(args, "action_paradox_min_risk_margin", 0.0)
    )
    max_policy_gap = float(
        getattr(args, "action_paradox_max_policy_log_gap", 2.0)
    )
    baseline_risk = float(action_risks[baseline_action])
    if risk_threshold > 0.0 and baseline_risk < risk_threshold:
      _selection_add_stat(stats, "rerank_threshold_blocked_baseline_risk")
      return baseline_action
    margin_candidates = []
    gap_candidates = []
    for action in legal:
      if action == baseline_action:
        continue
      risk_margin = baseline_risk - float(action_risks[action])
      if risk_margin < min_risk_margin:
        continue
      margin_candidates.append(action)
      policy_gap = float(log_policy[baseline_action] - log_policy[action])
      if max_policy_gap >= 0.0 and policy_gap > max_policy_gap:
        continue
      gap_candidates.append(action)
    _selection_add_stat(stats, "rerank_threshold_candidates", len(gap_candidates))
    if not gap_candidates:
      if margin_candidates:
        _selection_add_stat(stats, "rerank_threshold_blocked_policy_gap")
      else:
        _selection_add_stat(stats, "rerank_threshold_blocked_risk_margin")
      return baseline_action
    selected = max(
        gap_candidates,
        key=lambda action: (
            baseline_risk - float(action_risks[action]),
            -float(action_risks[action]),
            log_policy[action]
            + float(value_bonus[action])
            + float(feasibility_bonus[action]),
        ),
    )
    _selection_add_stat(stats, "rerank_threshold_applied")
  elif mode == "relative":
    _selection_add_stat(stats, "rerank_relative_considered")
    if not risk_available:
      _selection_add_stat(stats, "rerank_relative_missing_risk")
      return baseline_action
    safe_slack = max(
        0.0, float(getattr(args, "action_paradox_min_risk_margin", 0.0))
    )
    max_policy_gap = float(
        getattr(args, "action_paradox_max_policy_log_gap", -1.0)
    )
    min_risk = min(float(action_risks[action]) for action in legal)
    baseline_log_policy = float(log_policy[baseline_action])
    safe_candidates = []
    gap_filtered = []
    for action in legal:
      if float(action_risks[action]) > min_risk + safe_slack:
        continue
      safe_candidates.append(action)
      policy_gap = baseline_log_policy - float(log_policy[action])
      if max_policy_gap >= 0.0 and policy_gap > max_policy_gap:
        continue
      gap_filtered.append(action)
    _selection_add_stat(stats, "rerank_relative_candidates", len(safe_candidates))
    _selection_add_stat(
        stats, "rerank_relative_gap_filtered_candidates", len(gap_filtered)
    )
    if not gap_filtered:
      _selection_add_stat(stats, "rerank_relative_blocked_policy_gap")
      return baseline_action
    selected = max(
        gap_filtered,
        key=lambda action: (
            log_policy[action]
            + float(value_bonus[action])
            + float(feasibility_bonus[action]),
            -float(action_risks[action]),
        ),
    )
    _selection_add_stat(stats, "rerank_relative_applied")
  else:
    selected = max(
        legal,
        key=lambda action: log_policy[action]
        + float(value_bonus[action])
        + float(feasibility_bonus[action])
        - risk_penalty * (
            float(action_risks[action]) if risk_available else 0.0
        ),
    )
  if risk_available:
    selected_risk = float(action_risks[selected])
    _selection_add_stat(stats, "rerank_selected_risk_count")
    _selection_add_stat(stats, "rerank_selected_risk_sum", selected_risk)
    _selection_add_stat(
        stats,
        "rerank_selected_risk_margin_sum",
        float(action_risks[baseline_action]) - selected_risk,
    )
  return selected


def q_policy_rerank_policy(state, player, model, args, device):
  """Teacher policy from the same root risk/value reranker used by q_policy."""
  legal = state.legal_actions(player)
  prediction_action = shared_prediction_action(state, player, legal)
  if prediction_action is not None:
    target = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    target[int(prediction_action)] = 1.0
    return target
  policy, _ = model_policy_value(
      model,
      state,
      player,
      state.num_distinct_actions(),
      args.value_scale,
      device,
  )
  if len(legal) <= 1:
    return policy
  phase_spec = str(getattr(args, "action_value_rerank_phases", "") or "")
  if phase_spec.strip():
    allowed = {part.strip().lower() for part in phase_spec.split(",") if part.strip()}
    if phase_name_for_state(state) not in allowed:
      return policy
  value_weight = float(getattr(args, "action_value_selection_weight", 0.0))
  risk_penalty = float(getattr(args, "action_paradox_selection_penalty", 0.0))
  risk_mode = str(
      getattr(args, "action_paradox_rerank_mode", "additive") or "additive"
  ).lower()
  if (
      value_weight <= 0
      and risk_penalty <= 0
      and float(getattr(args, "action_feasibility_selection_weight", 0.0)) == 0.0
      and risk_mode not in ("threshold", "relative")
  ):
    return policy
  action_values = (
      model_action_values(
          model, state, player, state.num_distinct_actions(), device
      )
      if value_weight > 0 else
      np.zeros(state.num_distinct_actions(), dtype=np.float32)
  )
  clip = max(0.0, float(getattr(args, "action_value_rerank_clip", 0.5)))
  if clip > 0:
    action_values = np.clip(action_values, -clip, clip)
  min_margin = float(getattr(args, "action_value_rerank_min_margin", 0.0))
  if min_margin > 0 and len(legal) > 1:
    legal_values = np.array([float(action_values[action]) for action in legal])
    top_two = np.partition(legal_values, -2)[-2:]
    value_margin = float(np.max(top_two) - np.min(top_two))
    if value_margin < min_margin:
      return policy
  action_risks = (
      model_action_risks(
          model, state, player, state.num_distinct_actions(), device
      )
      if risk_penalty > 0 or risk_mode in ("threshold", "relative") else
      None
  )
  feasibility_scores = (
      action_feasibility_scores(
          action_feature_matrix(state, player, state.num_distinct_actions())
      )
      if float(getattr(args, "action_feasibility_selection_weight", 0.0)) != 0.0
      else None
  )
  selected = q_policy_select_action(
      legal,
      policy,
      action_values,
      action_risks,
      args,
      getattr(args, "value_scale", 1.0),
      feasibility_scores=feasibility_scores,
  )
  target = np.zeros_like(policy, dtype=np.float32)
  target[selected] = 1.0
  return target


def _add_stat(stats, key, amount=1):
  if stats is None:
    return
  if isinstance(amount, (float, np.floating)):
    stats[key] = float(stats.get(key, 0.0)) + float(amount)
  else:
    stats[key] = int(stats.get(key, 0)) + int(amount)


def q_policy_teacher_target_rollout_confirmed(
    state,
    round_game,
    player,
    raw_policy,
    q_policy,
    model,
    args,
    device,
    stats=None,
):
  """Checks whether a q-policy override improves paired rollout outcomes."""
  rollouts = int(getattr(args, "q_policy_teacher_confirm_rollouts", 0))
  if rollouts <= 0:
    return True
  _add_stat(stats, "q_policy_teacher_confirm_considered")
  legal = state.legal_actions(player)
  if len(legal) <= 1:
    _add_stat(stats, "q_policy_teacher_confirm_forced")
    return False
  baseline_action = max(legal, key=lambda action: float(raw_policy[action]))
  q_action = max(legal, key=lambda action: float(q_policy[action]))
  if q_action == baseline_action:
    _add_stat(stats, "q_policy_teacher_confirm_non_overrides")
    return False

  belief_states = counterfactual_belief_states(
      state, player, args, model=model, device=device
  )
  seat_roles = ["q_policy_teacher"] * int(getattr(args, "players", 0))
  fixed_bots = {}
  rollout_score_fn = (
      rollout_match_score_after_action
      if (
          getattr(args, "counterfactual_full_match_rollout", False)
          and getattr(args, "full_match_training", False)
      )
      else rollout_score_after_action
  )
  baseline_paradox = []
  q_paradox = []
  baseline_scores = []
  q_scores = []
  for belief_state in belief_states:
    legal_actions = belief_state.legal_actions(player)
    if baseline_action not in legal_actions or q_action not in legal_actions:
      continue
    for _ in range(rollouts):
      py_random_state = random.getstate()
      np_random_state = np.random.get_state()
      random.setstate(py_random_state)
      np.random.set_state(np_random_state)
      baseline_paradox.append(
          rollout_paradox_after_action(
              belief_state,
              round_game,
              player,
              baseline_action,
              seat_roles,
              fixed_bots,
              model,
              model,
              args,
              device,
          )
      )
      random.setstate(py_random_state)
      np.random.set_state(np_random_state)
      q_paradox.append(
          rollout_paradox_after_action(
              belief_state,
              round_game,
              player,
              q_action,
              seat_roles,
              fixed_bots,
              model,
              model,
              args,
              device,
          )
      )
      random.setstate(py_random_state)
      np.random.set_state(np_random_state)
      baseline_scores.append(
          rollout_score_fn(
              belief_state,
              round_game,
              player,
              baseline_action,
              seat_roles,
              fixed_bots,
              model,
              model,
              args,
              device,
          )
      )
      random.setstate(py_random_state)
      np.random.set_state(np_random_state)
      q_scores.append(
          rollout_score_fn(
              belief_state,
              round_game,
              player,
              q_action,
              seat_roles,
              fixed_bots,
              model,
              model,
              args,
              device,
          )
      )
  if not baseline_paradox or not q_paradox:
    _add_stat(stats, "q_policy_teacher_confirm_no_rollouts")
    return False
  paradox_improvement = float(np.mean(baseline_paradox) - np.mean(q_paradox))
  score_margin = float(np.mean(q_scores) - np.mean(baseline_scores))
  _add_stat(stats, "q_policy_teacher_confirm_evaluated")
  _add_stat(stats, "q_policy_teacher_confirm_paradox_improvement_sum",
            paradox_improvement)
  _add_stat(stats, "q_policy_teacher_confirm_score_margin_sum", score_margin)
  min_paradox_improvement = float(
      getattr(args, "q_policy_teacher_confirm_min_paradox_improvement", 1e-6)
  )
  min_score_margin = float(
      getattr(args, "q_policy_teacher_confirm_min_score_margin", 0.0)
  )
  accepted = (
      paradox_improvement >= min_paradox_improvement
      and score_margin >= min_score_margin
  )
  if accepted:
    _add_stat(stats, "q_policy_teacher_confirm_accepted")
  else:
    if paradox_improvement < min_paradox_improvement:
      _add_stat(stats, "q_policy_teacher_confirm_rejected_paradox")
    if score_margin < min_score_margin:
      _add_stat(stats, "q_policy_teacher_confirm_rejected_score")
  return accepted


def rollout_select_teacher_policy(
    state,
    round_game,
    player,
    model,
    args,
    device,
    stats=None,
):
  """Builds a one-hot target from paired rollout-ranked legal actions."""
  legal = state.legal_actions(player)
  raw_policy, _ = model_policy_value(
      model,
      state,
      player,
      state.num_distinct_actions(),
      args.value_scale,
      device,
  )
  rollouts = int(getattr(args, "rollout_select_teacher_rollouts", 1))
  if rollouts <= 0:
    _add_stat(stats, "rollout_select_teacher_disabled")
    return raw_policy, False
  _add_stat(stats, "rollout_select_teacher_considered")
  if len(legal) <= 1:
    _add_stat(stats, "rollout_select_teacher_forced")
    return raw_policy, False
  if not counterfactual_phase_allowed(state, args):
    _add_stat(stats, "rollout_select_teacher_skipped_phase")
    return raw_policy, False
  if not counterfactual_policy_hard_enough(raw_policy, legal, args):
    _add_stat(stats, "rollout_select_teacher_skipped_easy_policy")
    return raw_policy, False

  baseline_action = max(legal, key=lambda action: float(raw_policy[action]))
  candidate_actions = list(
      sampled_counterfactual_legal_actions(
          state, player, legal, args, policy=raw_policy
      )
  )
  if baseline_action not in candidate_actions:
    candidate_actions.append(baseline_action)
  candidate_actions = sorted(set(int(action) for action in candidate_actions))
  min_actions = max(
      2, int(getattr(args, "rollout_select_teacher_min_actions", 2))
  )
  if len(candidate_actions) < min_actions:
    _add_stat(stats, "rollout_select_teacher_too_few_candidates")
    return raw_policy, False
  _add_stat(stats, "rollout_select_teacher_candidate_actions_sum",
            len(candidate_actions))

  belief_states = counterfactual_belief_states(
      state, player, args, model=model, device=device
  )
  continuation_role = str(
      getattr(args, "rollout_select_teacher_continuation_role", "learner")
  )
  seat_roles = [continuation_role] * int(getattr(args, "players", 0))
  fixed_bots = {}
  rollout_score_fn = (
      rollout_match_score_after_action
      if (
          getattr(args, "counterfactual_full_match_rollout", False)
          and getattr(args, "full_match_training", False)
      )
      else rollout_score_after_action
  )
  action_paradox = {action: [] for action in candidate_actions}
  action_scores = {action: [] for action in candidate_actions}
  for belief_state in belief_states:
    legal_candidates = [
        action
        for action in candidate_actions
        if action in belief_state.legal_actions(player)
    ]
    if len(legal_candidates) < min_actions:
      continue
    for _ in range(rollouts):
      py_random_state = random.getstate()
      np_random_state = np.random.get_state()
      for action in legal_candidates:
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)
        action_paradox[action].append(
            float(
                rollout_paradox_after_action(
                    belief_state,
                    round_game,
                    player,
                    action,
                    seat_roles,
                    fixed_bots,
                    model,
                    model,
                    args,
                    device,
                )
            )
        )
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)
        action_scores[action].append(
            float(
                rollout_score_fn(
                    belief_state,
                    round_game,
                    player,
                    action,
                    seat_roles,
                    fixed_bots,
                    model,
                    model,
                    args,
                    device,
                )
            )
        )

  action_means = {}
  for action in candidate_actions:
    if action_paradox[action] and action_scores[action]:
      action_means[action] = (
          float(np.mean(action_paradox[action])),
          float(np.mean(action_scores[action])),
      )
  if len(action_means) < min_actions or baseline_action not in action_means:
    _add_stat(stats, "rollout_select_teacher_no_rollouts")
    return raw_policy, False

  ordered = sorted(
      action_means,
      key=lambda action: (
          action_means[action][0],
          -action_means[action][1],
          -float(raw_policy[action]),
      ),
  )
  best_action = ordered[0]
  baseline_paradox, baseline_score = action_means[baseline_action]
  best_paradox, best_score = action_means[best_action]
  paradox_improvement = baseline_paradox - best_paradox
  score_margin = best_score - baseline_score
  _add_stat(stats, "rollout_select_teacher_evaluated")
  _add_stat(stats, "rollout_select_teacher_paradox_improvement_sum",
            paradox_improvement)
  _add_stat(stats, "rollout_select_teacher_score_margin_sum", score_margin)

  if best_action == baseline_action:
    _add_stat(stats, "rollout_select_teacher_non_overrides")
    if not bool(getattr(args, "rollout_select_teacher_keep_policy_best", False)):
      return raw_policy, False

  min_paradox_improvement = float(
      getattr(args, "rollout_select_teacher_min_paradox_improvement", 1e-6)
  )
  min_score_margin = float(
      getattr(args, "rollout_select_teacher_min_score_margin", 0.0)
  )
  accepted = (
      paradox_improvement >= min_paradox_improvement
      and score_margin >= min_score_margin
  )
  if not accepted:
    if paradox_improvement < min_paradox_improvement:
      _add_stat(stats, "rollout_select_teacher_rejected_paradox")
    if score_margin < min_score_margin:
      _add_stat(stats, "rollout_select_teacher_rejected_score")
    return raw_policy, False

  _add_stat(stats, "rollout_select_teacher_accepted")
  if best_action != baseline_action:
    _add_stat(stats, "rollout_select_teacher_accepted_overrides")
  target = np.zeros_like(raw_policy, dtype=np.float32)
  target[best_action] = 1.0
  return target, True


def belief_raw_policy(
    state, player, model, args, device, samples=None, value_scale=None,
    context="self_play",
):
  """Average raw network policy over information-state determinizations."""
  legal = state.legal_actions(player)
  sample_count = samples
  if sample_count is None:
    sample_count = max(
        1,
        int(getattr(args, "self_play_belief_samples", 0))
        or int(getattr(args, "eval_belief_samples", 0))
        or int(getattr(args, "counterfactual_belief_samples", 0))
        or 1,
    )
  value_scale = args.value_scale if value_scale is None else value_scale
  combined = np.zeros(state.num_distinct_actions(), dtype=np.float32)
  sampled_states = sampled_belief_states_for_policy(
      state,
      player,
      sample_count,
      args,
      model,
      device,
      value_scale,
      context=context,
  )
  for sampled in sampled_states:
    policy, _ = model_policy_value(
        model,
        sampled,
        player,
        sampled.num_distinct_actions(),
        value_scale,
        device,
    )
    combined += policy
  combined /= float(max(1, len(sampled_states)))
  masked = np.zeros_like(combined)
  masked[legal] = combined[legal]
  total = float(masked.sum())
  if total <= 0 or not math.isfinite(total):
    masked[legal] = 1.0 / len(legal)
  else:
    masked /= total
  return masked


def learner_policy(state, player, model, args, device, add_noise=True):
  prediction_action = shared_prediction_action(state, player)
  if prediction_action is not None:
    policy = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    policy[int(prediction_action)] = 1.0
    return policy
  mode = getattr(args, "self_play_policy_mode", "belief")
  if mode == "policy":
    policy, _ = model_policy_value(
        model,
        state,
        player,
        state.num_distinct_actions(),
        args.value_scale,
        device,
    )
    return policy
  if mode == "belief_policy":
    return belief_raw_policy(state, player, model, args, device)
  if mode == "q_policy":
    return q_policy_rerank_policy(state, player, model, args, device)
  if mode == "belief" or args.self_play_belief_samples > 0:
    return belief_mcts_policy(state, player, model, args, device, add_noise=add_noise)
  return mcts_policy(state, model, args, device, add_noise=add_noise)


def add_dirichlet_noise(root, alpha=0.3, frac=0.25):
  actions = list(root.children)
  if not actions:
    return
  noise = np.random.dirichlet([alpha] * len(actions))
  for action, sample in zip(actions, noise):
    child = root.children[action]
    child.prior = (1 - frac) * child.prior + frac * float(sample)


def sample_action(policy, legal, temperature):
  if temperature <= 1e-6:
    return max(legal, key=lambda action: policy[action])
  probs = np.array([policy[action] for action in legal], dtype=np.float64)
  probs = probs ** (1.0 / temperature)
  total = probs.sum()
  if total <= 0 or not np.isfinite(total):
    probs = np.ones(len(legal)) / len(legal)
  else:
    probs /= total
  return int(np.random.choice(legal, p=probs))


def adapt_player_target_vector(target, players):
  out = np.zeros(players, dtype=np.float32)
  if target is None:
    return out
  arr = np.array(target, dtype=np.float32).reshape(-1)
  count = min(players, len(arr))
  if count > 0:
    out[:count] = arr[:count]
  return out


def terminal_paradox_target(state, players):
  return adapt_player_target_vector(
      getattr(state, "_has_paradoxed", [False] * players), players
  )


def self_play_round(state, game, model, args, device):
  examples = []
  ply = 0
  label_attempts = 0
  phase_label_attempts = {}
  seat_roles = ["learner"] * args.players
  fixed_bots = {}
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    player = state.current_player()
    policy = learner_policy(state, player, model, args, device)
    obs = np.array(state.observation_tensor(player), dtype=np.float32)
    legal_mask = np.zeros(game.num_distinct_actions(), dtype=np.float32)
    legal = state.legal_actions(player)
    legal_mask[legal] = 1.0
    action_features = action_feature_matrix(
        state, player, game.num_distinct_actions()
    )
    temp = args.temperature if ply < args.temperature_drop else 0.0
    action = sample_action(policy, legal, temp)
    should_label_actions = (
        counterfactual_phase_allowed(state, args)
        and counterfactual_phase_label_budget_allows(
            state, args, label_attempts, phase_label_attempts
        )
        and random.random()
        <= float(getattr(args, "counterfactual_action_label_prob", 1.0))
    )
    if should_label_actions:
      label_attempts += 1
      phase_name = phase_name_for_state(state)
      phase_label_attempts[phase_name] = (
          int(phase_label_attempts.get(phase_name, 0)) + 1
      )
      maybe_print_counterfactual_label_progress(
          args, "self_play_round", label_attempts, phase_label_attempts,
          phase_name, len(legal)
      )
      cf_targets, cf_mask = counterfactual_action_targets(
          state,
          game,
          player,
          legal,
          seat_roles,
          fixed_bots,
          model,
          model,
          args,
          device,
          policy=policy,
      )
      cf_value_targets, cf_value_mask = counterfactual_action_value_targets(
          state,
          game,
          player,
          legal,
          seat_roles,
          fixed_bots,
          model,
          model,
          args,
          device,
          policy=policy,
      )
      policy = blend_policy_with_counterfactual_paradox(
          policy, cf_targets, cf_mask, args
      )
      policy = blend_policy_with_counterfactual_values(
          policy, cf_value_targets, cf_value_mask, args
      )
    else:
      cf_targets = cf_mask = None
      cf_value_targets = cf_value_mask = None
    examples.append((
        obs,
        legal_mask,
        policy,
        None,
        action,
        player,
        cf_targets,
        cf_mask,
        cf_value_targets,
        cf_value_mask,
        action_features,
    ))
    state.apply_action(action)
    ply += 1
  return examples, state


def self_play_game(game, model, args, device):
  state = game.new_initial_state()
  maybe_set_match_context(state, args)
  examples, state = self_play_round(state, game, model, args, device)
  paradox_target = terminal_paradox_target(state, args.players)
  returns = value_targets_from_scores(state.returns(), paradox_target, args)
  return with_terminal_targets(
      examples,
      returns,
      paradox_target,
      terminal_action_paradox_targets=[paradox_target] * len(examples),
  ), state


def self_play_match(model, args, device, initial_start):
  match_totals = np.zeros(args.players, dtype=np.float32)
  final_round_scores = np.zeros(args.players, dtype=np.float32)
  all_examples = []
  terminal_action_paradox_targets = []
  start_counts = [0] * args.players
  terminal_states = []
  for round_index in range(args.players):
    start_player = (initial_start + round_index) % args.players
    start_counts[start_player] += 1
    round_game = make_game(args, start_player)
    state = round_game.new_initial_state()
    if getattr(args, "match_context", False):
      state.set_match_context(match_totals, round_index)
    examples, terminal_state = self_play_round(
        state, round_game, model, args, device
    )
    raw_scores = (
        terminal_state.raw_round_scores()
        if hasattr(terminal_state, "raw_round_scores")
        else terminal_state.returns()
    )
    round_paradox_target = terminal_paradox_target(terminal_state, args.players)
    match_totals += np.array(raw_scores, dtype=np.float32)
    final_round_scores = np.array(raw_scores, dtype=np.float32)
    all_examples.extend(examples)
    terminal_action_paradox_targets.extend(
        [round_paradox_target] * len(examples)
    )
    terminal_states.append(terminal_state)
  paradox_target = np.zeros(args.players, dtype=np.float32)
  for terminal_state in terminal_states:
    paradox_target = np.maximum(
        paradox_target,
        terminal_paradox_target(terminal_state, args.players),
    )
  final_returns = value_targets_from_scores(
      match_totals, paradox_target, args, final_round_scores
  )
  return (
      with_terminal_targets(
          all_examples,
          final_returns,
          paradox_target,
          terminal_action_paradox_targets=terminal_action_paradox_targets,
      ),
      match_totals,
      start_counts,
      terminal_states,
  )


def bootstrap_game(game, bot_names, args, game_index):
  bots = [
      make_bot(bot_names[(game_index + seat) % len(bot_names)],
               seed=args.seed + 700000 + game_index * 17 + seat)
      for seat in range(args.players)
  ]
  state = game.new_initial_state()
  maybe_set_match_context(state, args)
  examples = []
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    player = state.current_player()
    legal = state.legal_actions(player)
    action = bots[player].step(state, player)
    if action not in legal:
      raise ValueError(f"Bootstrap bot chose illegal action {action}: {legal}")
    obs = np.array(state.observation_tensor(player), dtype=np.float32)
    legal_mask = np.zeros(game.num_distinct_actions(), dtype=np.float32)
    legal_mask[legal] = 1.0
    action_features = action_feature_matrix(
        state, player, game.num_distinct_actions()
    )
    policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
    policy[action] = 1.0
    examples.append((
        obs, legal_mask, policy, None, action, player, None, None, None, None,
        action_features,
    ))
    state.apply_action(action)
  paradox_target = terminal_paradox_target(state, args.players)
  returns = value_targets_from_scores(state.returns(), paradox_target, args)
  return with_terminal_targets(
      examples,
      returns,
      paradox_target,
      terminal_action_paradox_targets=[paradox_target] * len(examples),
  ), state


def teacher_policy_round(state, game, teacher_model, args, device,
                         teacher_stats=None):
  examples = []
  ply = 0
  label_attempts = 0
  phase_label_attempts = {}
  builtin_teacher = None
  if getattr(args, "teacher_mode", "policy") == "builtin_policy":
    builtin_teacher = make_bot(
        str(getattr(args, "teacher_builtin_bot", "heuristic_safe14")),
        seed=int(getattr(args, "seed", 0)),
    )
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    player = state.current_player()
    legal = state.legal_actions(player)
    teacher_mode = getattr(args, "teacher_mode", "policy")
    keep_target = True
    if teacher_mode == "mcts":
      policy = mcts_policy(
          state,
          teacher_model,
          args,
          device,
          add_noise=False,
          sims=(
              int(getattr(args, "teacher_sims", 0))
              if int(getattr(args, "teacher_sims", 0)) > 0
              else None
          ),
      )
    elif teacher_mode == "q_policy":
      raw_policy, _ = model_policy_value(
          teacher_model,
          state,
          player,
          game.num_distinct_actions(),
          args.value_scale,
          device,
      )
      policy = q_policy_rerank_policy(
          state, player, teacher_model, args, device
      )
      keep_target = q_policy_teacher_target_rollout_confirmed(
          state,
          game,
          player,
          raw_policy,
          policy,
          teacher_model,
          args,
          device,
          teacher_stats,
      )
    elif teacher_mode == "rollout_select":
      should_rollout_select = (
          counterfactual_phase_allowed(state, args)
          and counterfactual_phase_label_budget_allows(
              state, args, label_attempts, phase_label_attempts
          )
          and random.random()
          <= float(getattr(args, "counterfactual_action_label_prob", 1.0))
      )
      if should_rollout_select:
        label_attempts += 1
        phase_name = phase_name_for_state(state)
        phase_label_attempts[phase_name] = (
            int(phase_label_attempts.get(phase_name, 0)) + 1
        )
        maybe_print_counterfactual_label_progress(
            args, "teacher_rollout_select", label_attempts,
            phase_label_attempts, phase_name, len(legal)
        )
        policy, keep_target = rollout_select_teacher_policy(
            state,
            game,
            player,
            teacher_model,
            args,
            device,
            teacher_stats,
        )
      else:
        policy, _ = model_policy_value(
            teacher_model,
            state,
            player,
            game.num_distinct_actions(),
            args.value_scale,
            device,
        )
        keep_target = False
    elif teacher_mode == "builtin_policy":
      action = int(builtin_teacher.step(state.clone(), player))
      if action not in legal:
        raise ValueError(
            f"Builtin teacher selected illegal action {action} from {legal}"
        )
      policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
      policy[action] = 1.0
      keep_target = counterfactual_phase_allowed(state, args)
    elif teacher_mode == "belief_policy":
      policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
      sample_count = max(1, int(getattr(args, "teacher_belief_samples", 4)))
      sampled_states = sampled_belief_states_for_policy(
          state,
          player,
          sample_count,
          args,
          teacher_model,
          device,
          args.value_scale,
          context="teacher",
      )
      for sampled in sampled_states:
        sampled_policy, _ = model_policy_value(
            teacher_model,
            sampled,
            player,
            game.num_distinct_actions(),
            args.value_scale,
          device,
        )
        policy += sampled_policy
      policy /= float(max(1, len(sampled_states)))
    elif teacher_mode == "belief_mcts":
      policy = teacher_belief_mcts_policy(
          state, player, teacher_model, args, device
      )
    else:
      policy, _ = model_policy_value(
          teacher_model,
          state,
          player,
          game.num_distinct_actions(),
          args.value_scale,
          device,
      )
    legal_mask = np.zeros(game.num_distinct_actions(), dtype=np.float32)
    legal_mask[legal] = 1.0
    action_features = action_feature_matrix(
        state, player, game.num_distinct_actions()
    )
    temp = args.teacher_temperature if ply < args.temperature_drop else 0.0
    action = sample_action(policy, legal, temp)
    if keep_target and keep_teacher_policy_target(policy, legal_mask, args):
      examples.append((
          np.array(state.observation_tensor(player), dtype=np.float32),
          legal_mask,
          policy,
          None,
          action,
          player,
          None,
          None,
          None,
          None,
          action_features,
      ))
    state.apply_action(action)
    ply += 1
  return examples, state


def teacher_policy_game(game, teacher_model, args, device, teacher_stats=None):
  state = game.new_initial_state()
  maybe_set_match_context(state, args)
  examples, state = teacher_policy_round(
      state, game, teacher_model, args, device, teacher_stats
  )
  paradox_target = terminal_paradox_target(state, args.players)
  returns = value_targets_from_scores(state.returns(), paradox_target, args)
  return with_terminal_targets(
      examples,
      returns,
      paradox_target,
      terminal_action_paradox_targets=[paradox_target] * len(examples),
  ), state


def teacher_policy_match(teacher_model, args, device, initial_start,
                         teacher_stats=None):
  match_totals = np.zeros(args.players, dtype=np.float32)
  final_round_scores = np.zeros(args.players, dtype=np.float32)
  all_examples = []
  terminal_action_paradox_targets = []
  start_counts = [0] * args.players
  terminal_states = []
  for round_index in range(args.players):
    start_player = (initial_start + round_index) % args.players
    start_counts[start_player] += 1
    round_game = make_game(args, start_player)
    state = round_game.new_initial_state()
    if getattr(args, "match_context", False):
      state.set_match_context(match_totals, round_index)
    examples, terminal_state = teacher_policy_round(
        state, round_game, teacher_model, args, device, teacher_stats
    )
    raw_scores = (
        terminal_state.raw_round_scores()
        if hasattr(terminal_state, "raw_round_scores")
        else terminal_state.returns()
    )
    round_paradox_target = terminal_paradox_target(terminal_state, args.players)
    match_totals += np.array(raw_scores, dtype=np.float32)
    final_round_scores = np.array(raw_scores, dtype=np.float32)
    all_examples.extend(examples)
    terminal_action_paradox_targets.extend(
        [round_paradox_target] * len(examples)
    )
    terminal_states.append(terminal_state)
  paradox_target = np.zeros(args.players, dtype=np.float32)
  for terminal_state in terminal_states:
    paradox_target = np.maximum(
        paradox_target,
        terminal_paradox_target(terminal_state, args.players),
    )
  final_returns = value_targets_from_scores(
      match_totals, paradox_target, args, final_round_scores
  )
  return (
      with_terminal_targets(
          all_examples,
          final_returns,
          paradox_target,
          terminal_action_paradox_targets=terminal_action_paradox_targets,
      ),
      match_totals,
      start_counts,
      terminal_states,
  )


def with_terminal_targets(
    examples, returns, paradox_target=None, terminal_action_paradox_targets=None
):
  if paradox_target is None:
    paradox_target = np.zeros_like(returns, dtype=np.float32)
  paradox_target = np.array(paradox_target, dtype=np.float32)
  rows = []
  for idx, example in enumerate(examples):
    obs, mask, policy = example[:3]
    action = int(example[4]) if len(example) > 4 else -1
    player = int(example[5]) if len(example) > 5 else -1
    action_targets = example[6] if len(example) > 6 else None
    action_mask = example[7] if len(example) > 7 else None
    action_value_targets = example[8] if len(example) > 8 else None
    action_value_mask = example[9] if len(example) > 9 else None
    action_features = example[10] if len(example) > 10 else None
    terminal_action_paradox_target = None
    if terminal_action_paradox_targets is not None:
      if isinstance(terminal_action_paradox_targets, np.ndarray):
        if terminal_action_paradox_targets.ndim == 1:
          terminal_action_paradox_target = terminal_action_paradox_targets
        else:
          terminal_action_paradox_target = terminal_action_paradox_targets[idx]
      else:
        terminal_action_paradox_target = terminal_action_paradox_targets[idx]
      terminal_action_paradox_target = adapt_player_target_vector(
          terminal_action_paradox_target, len(paradox_target)
      )
    rows.append((
        obs, mask, policy, returns, paradox_target, action, player,
        action_targets, action_mask, action_value_targets, action_value_mask,
        action_features, terminal_action_paradox_target,
    ))
  return rows


_GENERATION_WORKER = {}


def model_state_to_cpu(model):
  return {
      key: value.detach().cpu()
      for key, value in model.state_dict().items()
  }


def positive_worker_count(args, requested_games):
  workers = int(getattr(args, "self_play_workers", 1))
  if workers <= 0:
    if requested_games < int(getattr(args, "auto_worker_min_games", 32)):
      return 1
    workers = min(16, max(1, (os.cpu_count() or 2) - 2))
  if requested_games <= 1:
    return 1
  return max(1, min(workers, requested_games))


def seed_generation_job(args, salt, game_index, start_player):
  seed = (
      int(getattr(args, "seed", 0))
      + int(salt)
      + int(game_index) * 1009
      + int(start_player) * 37
  )
  random.seed(seed)
  np.random.seed(seed % (2**32 - 1))
  torch.manual_seed(seed)


def _make_worker_model(args, state_dict, device):
  game = make_game(args, 0)
  model = AZNet(
      game.observation_tensor_shape()[0],
      game.num_distinct_actions(),
      args.players,
      args.width,
      args.depth,
      args.arch,
      getattr(args, "separate_action_value_encoder", False),
      getattr(args, "separate_action_paradox_encoder", False),
  ).to(device)
  load_compatible_state_dict(model, state_dict)
  initialize_missing_action_value_stack_from_policy(model, state_dict)
  initialize_missing_action_paradox_stack_from_policy(model, state_dict)
  model.eval()
  return model


def _generation_worker_init(args_dict, model_state, checkpoint_paths):
  args = argparse.Namespace(**args_dict)
  torch.set_num_threads(max(1, int(getattr(args, "worker_torch_threads", 1))))
  device = torch.device("cpu")
  game = make_game(args, 0)
  model = _make_worker_model(args, model_state, device) if model_state else None
  checkpoint_models = []
  for checkpoint_path in checkpoint_paths or []:
    checkpoint_model, _, _ = load_model_payload(checkpoint_path, game, args, device)
    checkpoint_model.eval()
    checkpoint_models.append(checkpoint_model)
  _GENERATION_WORKER.clear()
  _GENERATION_WORKER.update({
      "args": args,
      "device": device,
      "model": model,
      "checkpoint_models": checkpoint_models,
  })


def _self_play_worker(job):
  game_index, start_player = job
  args = _GENERATION_WORKER["args"]
  device = _GENERATION_WORKER["device"]
  model = _GENERATION_WORKER["model"]
  seed_generation_job(args, 1100000, game_index, start_player)
  if getattr(args, "full_match_training", False):
    examples, match_totals, start_counts, _ = self_play_match(
        model, args, device, start_player
    )
    return {
        "examples": examples,
        "returns": np.array(match_totals, dtype=np.float32),
        "start_counts": start_counts,
    }
  round_game = make_game(args, start_player)
  examples, terminal_state = self_play_game(round_game, model, args, device)
  start_counts = [0] * args.players
  start_counts[start_player] += 1
  return {
      "examples": examples,
      "returns": np.array(terminal_state.returns(), dtype=np.float32),
      "start_counts": start_counts,
  }


def _teacher_worker(job):
  game_index, start_player = job
  args = _GENERATION_WORKER["args"]
  device = _GENERATION_WORKER["device"]
  checkpoint_models = _GENERATION_WORKER["checkpoint_models"]
  teacher_model = checkpoint_models[0] if checkpoint_models else None
  seed_generation_job(args, 1200000, game_index, start_player)
  teacher_stats = {}
  if getattr(args, "full_match_training", False):
    examples, match_totals, start_counts, _ = teacher_policy_match(
        teacher_model, args, device, start_player, teacher_stats
    )
    return {
        "examples": examples,
        "returns": np.array(match_totals, dtype=np.float32),
        "start_counts": start_counts,
        "teacher_stats": teacher_stats,
    }
  round_game = make_game(args, start_player)
  examples, terminal_state = teacher_policy_game(
      round_game, teacher_model, args, device, teacher_stats
  )
  start_counts = [0] * args.players
  start_counts[start_player] += 1
  return {
      "examples": examples,
      "returns": np.array(terminal_state.returns(), dtype=np.float32),
      "start_counts": start_counts,
      "teacher_stats": teacher_stats,
  }


def _league_worker(job):
  game_index, start_player, league_bots = job
  args = _GENERATION_WORKER["args"]
  device = _GENERATION_WORKER["device"]
  model = _GENERATION_WORKER["model"]
  checkpoint_models = _GENERATION_WORKER["checkpoint_models"]
  league_opponents = (
      checkpoint_models[0] if len(checkpoint_models) == 1 else checkpoint_models
  )
  seed_generation_job(args, 1300000, game_index, start_player)
  (
      examples,
      match_totals,
      start_counts,
      _seat_roles,
      learner_seat,
      paradoxes,
  ) = league_policy_match(
      model,
      league_opponents,
      league_bots,
      args,
      device,
      game_index,
      start_player,
  )
  return {
      "examples": examples,
      "returns": np.array(match_totals, dtype=np.float32),
      "start_counts": start_counts,
      "learner_seat": int(learner_seat),
      "learner_return": float(match_totals[learner_seat]),
      "paradoxes": np.array(paradoxes, dtype=np.float32),
  }


def random_start_players(args, count):
  return [
      int(np.random.randint(args.players)) if args.random_start_player else 0
      for _ in range(count)
  ]


def run_generation_pool(args, jobs, worker_fn, model_state=None,
                        checkpoint_paths=None, progress_interval=0,
                        progress_label=None):
  workers = positive_worker_count(args, len(jobs))
  if workers <= 1:
    return None
  args_dict = vars(args).copy()
  ctx = mp.get_context("spawn")
  results = []
  started = time.perf_counter()
  with ctx.Pool(
      processes=workers,
      initializer=_generation_worker_init,
      initargs=(args_dict, model_state, checkpoint_paths or []),
  ) as pool:
    for result in pool.imap_unordered(worker_fn, jobs):
      results.append(result)
      if progress_interval > 0 and len(results) % progress_interval == 0:
        progress_row = {
            "iteration": progress_label or "generation_progress",
            "completed_games": len(results),
            "total_games": len(jobs),
            "workers": workers,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        print(json.dumps(progress_row), flush=True)
  return results


def aggregate_generation_results(results, players):
  all_examples = []
  returns = []
  start_counts = [0] * players
  for result in results:
    all_examples.extend(result["examples"])
    returns.append(result["returns"])
    for seat in range(players):
      start_counts[seat] += result["start_counts"][seat]
  return all_examples, returns, start_counts


def aggregate_numeric_result_stats(results, key):
  stats = {}
  for result in results:
    for stat_key, value in result.get(key, {}).items():
      if isinstance(value, (float, np.floating)):
        stats[stat_key] = float(stats.get(stat_key, 0.0)) + float(value)
      elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        existing = stats.get(stat_key, 0)
        if isinstance(existing, float):
          stats[stat_key] = float(existing) + float(value)
        else:
          stats[stat_key] = int(existing) + int(value)
  return stats


def collect_self_play_games(model, args, device, count):
  start_players = random_start_players(args, count)
  jobs = list(enumerate(start_players))
  workers = positive_worker_count(args, count)
  started = time.perf_counter()
  results = run_generation_pool(
      args,
      jobs,
      _self_play_worker,
      model_state=model_state_to_cpu(model),
      progress_interval=getattr(args, "generate_replay_progress_interval", 0),
      progress_label="generate_replay_progress",
  )
  if results is None:
    results = []
    progress_interval = int(getattr(args, "generate_replay_progress_interval", 0))
    for game_index, start_player in jobs:
      seed_generation_job(args, 1100000, game_index, start_player)
      if args.full_match_training:
        examples, match_totals, match_start_counts, _ = self_play_match(
            model, args, device, start_player
        )
        results.append({
            "examples": examples,
            "returns": np.array(match_totals, dtype=np.float32),
            "start_counts": match_start_counts,
        })
      else:
        round_game = make_game(args, start_player)
        examples, terminal_state = self_play_game(round_game, model, args, device)
        start_counts = [0] * args.players
        start_counts[start_player] += 1
        results.append({
            "examples": examples,
          "returns": np.array(terminal_state.returns(), dtype=np.float32),
          "start_counts": start_counts,
        })
      if progress_interval > 0 and (game_index + 1) % progress_interval == 0:
        generated_examples = sum(len(result["examples"]) for result in results)
        generated_policy_label_rows = 0
        generated_policy_label_actions = 0
        generated_action_value_labels = sum(
            1
            for result in results
            for example in result["examples"]
            if has_action_value_labels(example)
        )
        for result in results:
          for example in result["examples"]:
            policy_mask = example[8] if len(example) > 8 else None
            if policy_mask is None:
              continue
            policy_count = int(
                np.sum(np.array(policy_mask, dtype=np.float32) > 0.0)
            )
            if policy_count > 0:
              generated_policy_label_rows += 1
              generated_policy_label_actions += policy_count
        progress_row = {
            "iteration": "generate_replay_progress",
            "completed_games": game_index + 1,
            "total_games": len(jobs),
            "workers": workers,
            "examples": int(generated_examples),
            "policy_labeled_rows": int(generated_policy_label_rows),
            "policy_labeled_actions": int(generated_policy_label_actions),
            "action_value_labeled_rows": int(generated_action_value_labels),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        print(json.dumps(progress_row), flush=True)
  examples, returns, start_counts = aggregate_generation_results(
      results, args.players
  )
  return {
      "examples": examples,
      "returns": returns,
      "start_counts": start_counts,
      "workers": workers,
      "elapsed_sec": time.perf_counter() - started,
  }


def collect_teacher_games(teacher_model, args, device, count):
  start_players = random_start_players(args, count)
  jobs = list(enumerate(start_players))
  workers = positive_worker_count(args, count)
  started = time.perf_counter()
  results = run_generation_pool(
      args,
      jobs,
      _teacher_worker,
      checkpoint_paths=(
          []
          if getattr(args, "teacher_mode", "policy") == "builtin_policy"
          else [args.teacher_checkpoint]
      ),
      progress_interval=getattr(args, "generate_replay_progress_interval", 0),
      progress_label="teacher_replay_progress",
  )
  if results is None:
    results = []
    progress_interval = int(getattr(args, "generate_replay_progress_interval", 0))
    for game_index, start_player in jobs:
      seed_generation_job(args, 1200000, game_index, start_player)
      teacher_stats = {}
      if args.full_match_training:
        examples, match_totals, match_start_counts, _ = teacher_policy_match(
            teacher_model, args, device, start_player, teacher_stats
        )
        results.append({
            "examples": examples,
            "returns": np.array(match_totals, dtype=np.float32),
            "start_counts": match_start_counts,
            "teacher_stats": teacher_stats,
        })
      else:
        round_game = make_game(args, start_player)
        examples, terminal_state = teacher_policy_game(
            round_game, teacher_model, args, device, teacher_stats
        )
        start_counts = [0] * args.players
        start_counts[start_player] += 1
        results.append({
            "examples": examples,
            "returns": np.array(terminal_state.returns(), dtype=np.float32),
            "start_counts": start_counts,
            "teacher_stats": teacher_stats,
        })
      if progress_interval > 0 and (game_index + 1) % progress_interval == 0:
        generated_examples = sum(len(result["examples"]) for result in results)
        progress_row = {
            "iteration": "teacher_replay_progress",
            "completed_games": game_index + 1,
            "total_games": len(jobs),
            "workers": workers,
            "examples": int(generated_examples),
            "teacher_stats": aggregate_numeric_result_stats(
                results, "teacher_stats"
            ),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        print(json.dumps(progress_row), flush=True)
        maybe_write_teacher_progress(args, progress_row)
  examples, returns, start_counts = aggregate_generation_results(
      results, args.players
  )
  teacher_stats = aggregate_numeric_result_stats(results, "teacher_stats")
  return {
      "examples": examples,
      "returns": returns,
      "start_counts": start_counts,
      "teacher_stats": teacher_stats,
      "workers": workers,
      "elapsed_sec": time.perf_counter() - started,
  }


def collect_league_games(model, league_opponents, league_bots, args, device):
  start_players = random_start_players(args, args.league_games)
  jobs = [
      (game_index, start_player, league_bots)
      for game_index, start_player in enumerate(start_players)
  ]
  workers = positive_worker_count(args, args.league_games)
  started = time.perf_counter()
  progress_interval = int(getattr(args, "league_progress_interval", 0))
  results = run_generation_pool(
      args,
      jobs,
      _league_worker,
      model_state=model_state_to_cpu(model),
      checkpoint_paths=split_csv(args.league_checkpoint),
      progress_interval=progress_interval,
      progress_label="league_progress",
  )
  if results is None:
    results = []
    learner_returns = []
    for game_index, start_player, _ in jobs:
      seed_generation_job(args, 1300000, game_index, start_player)
      (
          examples,
          match_totals,
          match_start_counts,
          _seat_roles,
          learner_seat,
          paradoxes,
      ) = league_policy_match(
          model,
          league_opponents,
          league_bots,
          args,
          device,
          game_index,
          start_player,
      )
      learner_returns.append(float(match_totals[learner_seat]))
      results.append({
          "examples": examples,
          "returns": np.array(match_totals, dtype=np.float32),
          "start_counts": match_start_counts,
          "learner_seat": int(learner_seat),
          "learner_return": float(match_totals[learner_seat]),
          "paradoxes": np.array(paradoxes, dtype=np.float32),
      })
      if progress_interval > 0 and (game_index + 1) % progress_interval == 0:
        progress_row = {
            "iteration": "league_progress",
            "completed_games": game_index + 1,
            "league_games": args.league_games,
            "mean_learner_return_so_far": float(np.mean(learner_returns)),
        }
        print(json.dumps(progress_row), flush=True)
  examples, returns, start_counts = aggregate_generation_results(
      results, args.players
  )
  learner_returns = [result["learner_return"] for result in results]
  learner_seat_counts = [0] * args.players
  paradox_sums = np.zeros(args.players, dtype=np.float32)
  for result in results:
    learner_seat_counts[result["learner_seat"]] += 1
    paradox_sums += result["paradoxes"]
  return {
      "examples": examples,
      "returns": returns,
      "start_counts": start_counts,
      "learner_returns": learner_returns,
      "learner_seat_counts": learner_seat_counts,
      "paradox_sums": paradox_sums,
      "workers": workers,
      "elapsed_sec": time.perf_counter() - started,
  }

def value_targets_from_scores(scores, paradox_target, args, final_round_scores=None):
  scores = np.array(scores, dtype=np.float32)
  original_scores = np.array(scores, dtype=np.float32)
  penalty = float(getattr(args, "terminal_paradox_penalty", 0.0))
  if penalty > 0:
    paradox_target = np.array(paradox_target, dtype=np.float32)
    scores = scores - penalty * paradox_target
  any_penalty = float(getattr(args, "terminal_any_paradox_penalty", 0.0))
  if any_penalty > 0 and np.any(np.array(paradox_target, dtype=np.float32) > 0):
    scores = scores - any_penalty
  score_targets = scores / args.value_scale
  ordinal_weight = float(getattr(args, "ordinal_value_weight", 0.0))
  ordinal_weight = min(1.0, ordinal_weight)
  official_weight = float(getattr(args, "official_outcome_value_weight", 0.0))
  official_weight = min(1.0, max(0.0, official_weight))
  blended = score_targets
  if ordinal_weight > 0:
    ordinal_targets = ordinal_value_targets(scores)
    blended = (1.0 - ordinal_weight) * blended + ordinal_weight * ordinal_targets
  if official_weight > 0 and final_round_scores is not None:
    final_round_scores = np.array(final_round_scores, dtype=np.float32)
    official_scores = match_outcome_scores(original_scores, final_round_scores)
    official_targets = ordinal_value_targets(official_scores)
    blended = (1.0 - official_weight) * blended + official_weight * official_targets
  return blended


def ordinal_value_targets(scores):
  scores = np.array(scores, dtype=np.float32)
  players = len(scores)
  if players <= 1:
    return np.zeros_like(scores, dtype=np.float32)
  targets = np.zeros(players, dtype=np.float32)
  for player in range(players):
    wins = float(np.sum(scores[player] > scores))
    losses = float(np.sum(scores[player] < scores))
    targets[player] = (wins - losses) / float(players - 1)
  return targets


def rollout_policy_action(
    state, round_game, player, role, fixed_bots, model, opponent_model, args,
    device,
):
  legal = state.legal_actions(player)
  if role == "q_policy_teacher":
    policy = q_policy_rerank_policy(state, player, model, args, device)
    return max(legal, key=lambda legal_action: policy[legal_action])
  if role == "learner":
    rollout_bot_name = str(
        getattr(args, "counterfactual_rollout_learner_bot", "") or ""
    ).strip()
    if rollout_bot_name:
      try:
        action = make_bot(rollout_bot_name, seed=0).step(state.clone(), player)
      except Exception:
        action = None
      if action in legal:
        return int(action)
    policy = learner_policy(state, player, model, args, device, add_noise=False)
    return max(legal, key=lambda legal_action: policy[legal_action])
  if is_old_policy_role(role):
    rollout_mode = getattr(args, "counterfactual_rollout_opponent_mode", "")
    return old_policy_action(
        state, player, legal, opponent_model, role, args, device,
        mode_override=rollout_mode or None,
        belief_context="counterfactual" if rollout_mode else "league",
    )
  action = fixed_bots[player].step(state, player)
  if action not in legal:
    return random.choice(legal)
  return action


def old_policy_action(
    state, player, legal, opponent_model, role, args, device, mode_override=None,
    belief_context="league",
):
  model = old_policy_model(opponent_model, role)
  mode = mode_override or old_policy_mode(role, args)
  if mode == "mcts":
    policy = mcts_policy(state, model, args, device, add_noise=False)
  elif mode == "belief":
    policy = belief_mcts_policy(
        state, player, model, args, device, add_noise=False,
        context=belief_context
    )
  elif mode == "belief_policy":
    policy = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    sample_count = max(1, int(getattr(args, "eval_belief_samples", 4)))
    sampled_states = sampled_belief_states_for_policy(
        state,
        player,
        sample_count,
        args,
        model,
        device,
        args.value_scale,
        context=belief_context,
    )
    for sampled in sampled_states:
      sampled_policy, _ = model_policy_value(
          model,
          sampled,
          player,
          sampled.num_distinct_actions(),
          args.value_scale,
          device,
      )
      policy += sampled_policy
    policy /= float(max(1, len(sampled_states)))
  else:
    policy, _ = model_policy_value(
        model,
        state,
        player,
        state.num_distinct_actions(),
        args.value_scale,
        device,
    )
  return max(legal, key=lambda legal_action: policy[legal_action])


def old_policy_mode(role, args):
  mode = getattr(args, "league_opponent_mode", "policy")
  if mode != "mixed":
    return mode
  modes = ("policy", "mcts", "belief", "belief_policy")
  if role == "old_policy":
    return random.choice(modes)
  return modes[old_policy_index(role) % len(modes)]


def scoped_paradox_value(state, acting_player, args):
  players = int(getattr(args, "players", 0))
  paradoxes = np.array(
      getattr(state, "_has_paradoxed", [False] * players),
      dtype=np.float32,
  )
  if paradoxes.size == 0:
    return 0.0
  if getattr(args, "counterfactual_action_paradox_scope", "acting") == "any":
    return float(np.max(paradoxes))
  if 0 <= int(acting_player) < paradoxes.size:
    return float(paradoxes[int(acting_player)])
  return 0.0


def estimated_paradox_rollout_horizon(state, args, max_decision_plies):
  if max_decision_plies > 0:
    return max(1, int(max_decision_plies))
  hand_counts = 0
  hands = getattr(state, "_hands", None)
  if hands is not None:
    for hand in hands:
      hand_counts += int(np.sum(np.array(hand, dtype=np.int32)))
  phase = int(getattr(state, "_phase", 0))
  players = int(getattr(args, "players", getattr(state, "_num_players", 0)))
  prediction_count = 0
  if phase <= 2:
    predictions = getattr(state, "_predictions", None)
    if predictions is None:
      prediction_count = players
    else:
      prediction_count = sum(
          1
          for prediction in predictions
          if prediction is None or int(prediction) < 0
      )
  return max(1, hand_counts + prediction_count)


def survival_paradox_risk(paradox, survived_decision_plies, horizon, args):
  weight = float(
      getattr(args, "counterfactual_action_paradox_survival_weight", 0.5)
  )
  weight = min(1.0, max(0.0, weight))
  survival_fraction = min(
      1.0, max(0.0, float(survived_decision_plies) / float(max(1, horizon)))
  )
  if paradox:
    return float(max(0.0, min(1.0, 1.0 - weight * survival_fraction)))
  return float(max(0.0, min(1.0, 0.5 * (1.0 - survival_fraction))))


def rollout_survival_paradox_after_action(
    state, round_game, acting_player, first_action, seat_roles, fixed_bots,
    model, opponent_model, args, device,
):
  rollout_state = state.clone()
  rollout_state.apply_action(first_action)
  max_plies = max(0, int(getattr(args, "counterfactual_rollout_max_plies", 0)))
  horizon = estimated_paradox_rollout_horizon(rollout_state, args, max_plies)
  if scoped_paradox_value(rollout_state, acting_player, args) > 0.0:
    return survival_paradox_risk(True, 0, horizon, args)

  decision_plies = 0
  while not rollout_state.is_terminal():
    if rollout_state.is_chance_node():
      outcomes = rollout_state.chance_outcomes()
      actions, probs = zip(*outcomes)
      rollout_state.apply_action(int(np.random.choice(actions, p=probs)))
      if scoped_paradox_value(rollout_state, acting_player, args) > 0.0:
        return survival_paradox_risk(True, decision_plies, horizon, args)
      continue
    if max_plies > 0 and decision_plies >= max_plies:
      break
    player = rollout_state.current_player()
    action = rollout_policy_action(
        rollout_state,
        round_game,
        player,
        seat_roles[player],
        fixed_bots,
        model,
        opponent_model,
        args,
        device,
    )
    rollout_state.apply_action(action)
    decision_plies += 1
    if scoped_paradox_value(rollout_state, acting_player, args) > 0.0:
      return survival_paradox_risk(True, decision_plies, horizon, args)

  if rollout_state.is_terminal():
    return 0.0
  return survival_paradox_risk(False, decision_plies, horizon, args)


def rollout_paradox_after_action(
    state, round_game, acting_player, first_action, seat_roles, fixed_bots,
    model, opponent_model, args, device,
):
  if getattr(args, "counterfactual_action_paradox_target_mode", "binary") == "survival":
    return rollout_survival_paradox_after_action(
        state, round_game, acting_player, first_action, seat_roles, fixed_bots,
        model, opponent_model, args, device
    )
  rollout_state = state.clone()
  rollout_state.apply_action(first_action)
  max_plies = max(0, int(getattr(args, "counterfactual_rollout_max_plies", 0)))
  play_rollout_to_terminal(
      rollout_state, round_game, seat_roles, fixed_bots, model, opponent_model,
      args, device, max_decision_plies=max_plies
  )
  return scoped_paradox_value(rollout_state, acting_player, args)


def rollout_survival_value_after_action(
    state, round_game, acting_player, first_action, seat_roles, fixed_bots,
    model, opponent_model, args, device,
):
  """Return +1 for round survival and -1 for a scoped paradox."""
  rollout_state = state.clone()
  rollout_state.apply_action(first_action)
  if scoped_paradox_value(rollout_state, acting_player, args) > 0.0:
    return -1.0
  max_plies = max(0, int(getattr(args, "counterfactual_rollout_max_plies", 0)))
  play_rollout_to_terminal(
      rollout_state, round_game, seat_roles, fixed_bots, model, opponent_model,
      args, device, max_decision_plies=max_plies
  )
  if scoped_paradox_value(rollout_state, acting_player, args) > 0.0:
    return -1.0
  if rollout_state.is_terminal():
    return 1.0
  return float(
      getattr(args, "counterfactual_action_survival_truncated_value", 0.0)
  )


def rollout_score_after_action(
    state, round_game, acting_player, first_action, seat_roles, fixed_bots,
    model, opponent_model, args, device,
):
  rollout_state = state.clone()
  rollout_state.apply_action(first_action)
  max_plies = max(0, int(getattr(args, "counterfactual_rollout_max_plies", 0)))
  play_rollout_to_terminal(
      rollout_state, round_game, seat_roles, fixed_bots, model,
      opponent_model, args, device, max_decision_plies=max_plies
  )
  if not rollout_state.is_terminal():
    return leaf_value_estimate(rollout_state, acting_player, model, args, device)
  raw_scores = (
      rollout_state.raw_round_scores()
      if hasattr(rollout_state, "raw_round_scores")
      else rollout_state.returns()
  )
  return float(np.array(raw_scores, dtype=np.float32)[acting_player])


def leaf_value_estimate(state, player, model, args, device):
  """Raw-score value estimate for a nonterminal rollout leaf."""
  state = state.clone()
  while state.is_chance_node() and not state.is_terminal():
    outcomes = state.chance_outcomes()
    actions, probs = zip(*outcomes)
    state.apply_action(int(np.random.choice(actions, p=probs)))
  if state.is_terminal():
    raw_scores = (
        state.raw_round_scores()
        if hasattr(state, "raw_round_scores")
        else state.returns()
    )
    return float(np.array(raw_scores, dtype=np.float32)[player])
  _, value = model_policy_value(
      model,
      state,
      player,
      state.num_distinct_actions(),
      args.value_scale,
      device,
  )
  return float(np.array(value, dtype=np.float32)[player])


def play_rollout_to_terminal(
    rollout_state, round_game, seat_roles, fixed_bots, model, opponent_model,
    args, device, max_decision_plies=0,
):
  decision_plies = 0
  while not rollout_state.is_terminal():
    if rollout_state.is_chance_node():
      outcomes = rollout_state.chance_outcomes()
      actions, probs = zip(*outcomes)
      rollout_state.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    if max_decision_plies > 0 and decision_plies >= max_decision_plies:
      break
    player = rollout_state.current_player()
    action = rollout_policy_action(
        rollout_state,
        round_game,
        player,
        seat_roles[player],
        fixed_bots,
        model,
        opponent_model,
        args,
        device,
    )
    rollout_state.apply_action(action)
    decision_plies += 1
  return rollout_state


def rollout_match_score_after_action(
    state, round_game, acting_player, first_action, seat_roles, fixed_bots,
    model, opponent_model, args, device,
):
  """Roll out the current decision through the remaining full match."""
  rollout_state = state.clone()
  current_round = int(getattr(rollout_state, "_match_round", 0))
  current_start = int(getattr(rollout_state, "_round_start_player", 0))
  initial_start = (current_start - current_round) % args.players
  match_totals = np.array(
      getattr(rollout_state, "_match_totals", np.zeros(args.players)),
      dtype=np.float32,
  )

  rollout_state.apply_action(first_action)
  max_plies = max(0, int(getattr(args, "counterfactual_rollout_max_plies", 0)))
  play_rollout_to_terminal(
      rollout_state, round_game, seat_roles, fixed_bots, model,
      opponent_model, args, device, max_decision_plies=max_plies
  )
  if not rollout_state.is_terminal():
    return leaf_value_estimate(rollout_state, acting_player, model, args, device)
  raw_scores = (
      rollout_state.raw_round_scores()
      if hasattr(rollout_state, "raw_round_scores")
      else rollout_state.returns()
  )
  match_totals += np.array(raw_scores, dtype=np.float32)

  if max_plies > 0 and current_round + 1 < args.players:
    next_round = current_round + 1
    start_player = (initial_start + next_round) % args.players
    next_game = make_game(args, start_player)
    next_state = next_game.new_initial_state()
    if getattr(args, "match_context", False):
      next_state.set_match_context(match_totals, next_round)
    return leaf_value_estimate(next_state, acting_player, model, args, device)

  for next_round in range(current_round + 1, args.players):
    start_player = (initial_start + next_round) % args.players
    next_game = make_game(args, start_player)
    next_state = next_game.new_initial_state()
    if getattr(args, "match_context", False):
      next_state.set_match_context(match_totals, next_round)
    play_rollout_to_terminal(
        next_state, next_game, seat_roles, fixed_bots, model, opponent_model,
        args, device
    )
    raw_scores = (
        next_state.raw_round_scores()
        if hasattr(next_state, "raw_round_scores")
        else next_state.returns()
    )
    match_totals += np.array(raw_scores, dtype=np.float32)

  return float(match_totals[acting_player])


def sampled_counterfactual_legal_actions(state, player, legal, args, policy=None):
  legal = list(legal)
  max_legal = int(getattr(args, "counterfactual_action_max_legal", 0))
  if max_legal > 0 and len(legal) > max_legal:
    top_policy_count = max(
        0, int(getattr(args, "counterfactual_action_top_policy", 0))
    )
    selected = []
    if top_policy_count > 0 and policy is not None:
      ordered = sorted(legal, key=lambda action: float(policy[action]), reverse=True)
      selected.extend(ordered[:min(top_policy_count, max_legal)])
    selected_set = set(selected)
    feature_action_features = None
    if bool(getattr(args, "counterfactual_action_feature_candidates", True)):
      feature_action_features = action_feature_matrix(
          state, player, state.num_distinct_actions()
      )

      def add_feature_action(action):
        if len(selected) >= max_legal or action in selected_set:
          return False
        selected.append(int(action))
        selected_set.add(int(action))
        return True

      def is_play_action(features):
        return bool(features[6 + 2] > 0.5)

      token_loss_idx = APPENDED_ACTION_FEATURE_INDEX[
          "token_loss_newly_loses_led"
      ]
      min_surplus_idx = APPENDED_ACTION_FEATURE_INDEX[
          "exit_min_player_lane_surplus_after"
      ]
      lane_damage_idx = APPENDED_ACTION_FEATURE_INDEX[
          "exit_lane_surplus_damage"
      ]
      follow_led_actions = [
          action for action in legal
          if action not in selected_set
          and is_play_action(feature_action_features[action])
          and feature_action_features[action, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
      ]
      newly_loses_led_actions = [
          action for action in legal
          if action not in selected_set
          and is_play_action(feature_action_features[action])
          and feature_action_features[action, token_loss_idx] > 0.5
      ]
      if follow_led_actions and newly_loses_led_actions:
        best_follow = max(
            follow_led_actions,
            key=lambda action: (
                float(feature_action_features[action, min_surplus_idx]),
                float(policy[action]) if policy is not None else 0.0,
            ),
        )
        add_feature_action(best_follow)
        worst_lane_loss = max(
            newly_loses_led_actions,
            key=lambda action: (
                float(feature_action_features[action, lane_damage_idx]),
                float(feature_action_features[action, token_loss_idx]),
                float(policy[action]) if policy is not None else 0.0,
            ),
        )
        add_feature_action(worst_lane_loss)
    for bot_name in split_csv(
        getattr(args, "counterfactual_action_include_bots", "")
    ):
      if len(selected) >= max_legal:
        break
      try:
        action = make_bot(bot_name, seed=0).step(state.clone(), player)
      except Exception:
        continue
      if action in legal and action not in selected_set:
        selected.append(action)
        selected_set.add(action)
    if bool(getattr(args, "counterfactual_action_feature_candidates", True)):
      action_features = (
          feature_action_features
          if feature_action_features is not None
          else action_feature_matrix(state, player, state.num_distinct_actions())
      )

      def add_extreme(feature_idx, reverse=True, predicate=None):
        if len(selected) >= max_legal:
          return
        candidates = [
            action for action in legal
            if action not in selected_set
            and (
                predicate is None
                or bool(predicate(action_features[action], action))
            )
        ]
        if not candidates:
          return
        best = max(
            candidates,
            key=lambda action: float(action_features[action, feature_idx]),
        ) if reverse else min(
            candidates,
            key=lambda action: float(action_features[action, feature_idx]),
        )
        selected.append(int(best))
        selected_set.add(int(best))

      numeric_start = 1 + 5 + 5 + MAX_RANK_FEATURES + MAX_COLOR_FEATURES + 5
      board_free_idx = numeric_start + 14
      off_led_token_loss_idx = numeric_start + 17
      would_win_idx = numeric_start + 19
      win_align_idx = numeric_start + 20
      adjacency_gain_idx = numeric_start + 21
      largest_after_idx = numeric_start + 23
      lead_after_legal_count_idx = numeric_start + 33
      lead_dead_after_idx = numeric_start + 36
      lead_dead_delta_idx = numeric_start + 37
      hit_prediction_idx = APPENDED_ACTION_FEATURE_INDEX["hits_prediction"]
      overshoot_idx = APPENDED_ACTION_FEATURE_INDEX["overshoots_prediction"]
      can_still_hit_idx = APPENDED_ACTION_FEATURE_INDEX["can_still_hit_after"]
      hit_future_idx = APPENDED_ACTION_FEATURE_INDEX["hit_with_future_tricks"]

      play_action = lambda features, action: bool(features[6 + 2] > 0.5)
      add_extreme(hit_prediction_idx, True, play_action)
      add_extreme(can_still_hit_idx, True, play_action)
      add_extreme(win_align_idx, True, play_action)
      add_extreme(adjacency_gain_idx, True, play_action)
      add_extreme(largest_after_idx, True, play_action)
      add_extreme(lead_after_legal_count_idx, True, play_action)
      add_extreme(board_free_idx, True, play_action)
      add_extreme(off_led_token_loss_idx, False, play_action)
      add_extreme(overshoot_idx, False, play_action)
      add_extreme(lead_dead_after_idx, False, play_action)
      add_extreme(lead_dead_delta_idx, False, play_action)
      add_extreme(would_win_idx, False, play_action)
      add_extreme(would_win_idx, True, play_action)
      add_extreme(hit_future_idx, False, play_action)
    remaining = [action for action in legal if action not in selected_set]
    fill_count = max(0, max_legal - len(selected))
    if fill_count > 0 and remaining:
      selected.extend(random.sample(remaining, min(fill_count, len(remaining))))
    return sorted(selected)
  return legal


def counterfactual_phase_allowed(state, args):
  phases = str(getattr(args, "counterfactual_action_label_phases", "")).strip()
  if not phases:
    return True
  phase_name = phase_name_for_state(state)
  allowed = {part.strip() for part in phases.split(",") if part.strip()}
  return phase_name in allowed


def phase_name_for_state(state):
  return {
      0: "chance",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "terminal",
  }.get(int(getattr(state, "_phase", -1)), "unknown")


def counterfactual_phase_label_budget_allows(
    state, args, total_attempts, phase_attempts
):
  total_cap = int(getattr(args, "counterfactual_action_label_max_per_game", 0))
  if total_cap > 0 and total_attempts >= total_cap:
    return False
  phase_cap = int(
      getattr(args, "counterfactual_action_label_max_per_phase_per_game", 0)
  )
  if phase_cap <= 0:
    return True
  phase = phase_name_for_state(state)
  return int(phase_attempts.get(phase, 0)) < phase_cap


def maybe_print_counterfactual_label_progress(
    args, context, attempts, phase_attempts, phase_name, legal_count
):
  interval = int(getattr(args, "counterfactual_label_progress_interval", 0))
  if interval <= 0 or attempts <= 0 or attempts % interval != 0:
    return
  row = {
      "iteration": "counterfactual_label_progress",
      "context": context,
      "attempts": int(attempts),
      "phase": str(phase_name),
      "phase_attempts": {
          str(key): int(value) for key, value in sorted(phase_attempts.items())
      },
      "legal_actions": int(legal_count),
  }
  print(json.dumps(row), flush=True)


def counterfactual_policy_hard_enough(policy, legal, args):
  if policy is None or not legal:
    return True
  legal_probs = np.array([float(policy[action]) for action in legal], dtype=np.float64)
  total = float(np.sum(legal_probs))
  if total <= 0 or not math.isfinite(total):
    legal_probs = np.ones(len(legal), dtype=np.float64) / float(len(legal))
  else:
    legal_probs /= total
  min_entropy = float(
      getattr(args, "counterfactual_action_min_policy_entropy", 0.0)
  )
  if min_entropy > 0 and len(legal_probs) > 1:
    entropy = -float(np.sum(legal_probs * np.log(np.maximum(legal_probs, 1e-12))))
    normalized_entropy = entropy / math.log(len(legal_probs))
    if normalized_entropy < min_entropy:
      return False
  max_top_prob = float(
      getattr(args, "counterfactual_action_max_policy_top_prob", 1.0)
  )
  if max_top_prob < 1.0 and float(np.max(legal_probs)) > max_top_prob:
    return False
  return True


_ACTION_COLORS = ("R", "B", "Y", "G")


def _belief_reference_model(model):
  return model


def _observed_trick_cards(state):
  observed = []
  for trick in getattr(state, "_completed_tricks", []):
    for p, cardinfo in trick:
      if cardinfo is not None:
        observed.append((int(p), cardinfo))
  for p, cardinfo in enumerate(getattr(state, "_cards_played_this_trick", [])):
    if cardinfo is not None:
      observed.append((int(p), cardinfo))
  return observed


def _action_from_cardinfo(cardinfo, num_card_types):
  rank_val, color_str = cardinfo
  if color_str not in _ACTION_COLORS:
    raise ValueError(f"Unknown action color {color_str!r}")
  return _ACTION_COLORS.index(color_str) * num_card_types + int(rank_val) - 1


def _replay_start_hands(sampled_state):
  num_card_types = int(getattr(sampled_state, "_num_card_types"))
  replay_hands = [
      np.array(hand, dtype=int, copy=True) for hand in sampled_state._hands
  ]
  for p, cardinfo in _observed_trick_cards(sampled_state):
    rank_val, _ = cardinfo
    rank_idx = int(rank_val) - 1
    if not (0 <= rank_idx < num_card_types):
      raise ValueError("Observed trick rank outside this game's rank range")
    replay_hands[p][rank_idx] += 1
  return replay_hands


def _initial_public_board(sampled_state):
  initial_board = -1 * np.ones_like(sampled_state._board_ownership)
  initial_board[np.asarray(sampled_state._board_ownership) == -2] = -2
  return initial_board


def _reset_to_prediction_replay_start(sampled_state):
  """Builds a sampled-world shadow state at the start of prediction."""
  shadow = sampled_state.clone()
  num_players = int(getattr(shadow, "_num_players"))
  shadow._phase = 2
  shadow._game_over = False
  shadow._returns = [0.0] * num_players
  shadow._rewards = [0.0] * num_players
  shadow._last_raw_scores = [0.0] * num_players
  shadow._player_adjacency_bonus = [0.0] * num_players
  shadow._hands = _replay_start_hands(sampled_state)
  shadow._has_discarded = [True] * num_players
  shadow._predictions = [-1] * num_players
  shadow._trick_number = 0
  shadow._start_player = int(getattr(shadow, "_round_start_player", 0))
  shadow._current_player = shadow._start_player
  shadow._led_color = None
  shadow._cards_played_this_trick = [None] * num_players
  shadow._tricks_won = np.zeros(num_players, dtype=int)
  shadow._board_ownership = _initial_public_board(shadow)
  shadow._has_paradoxed = [False] * num_players
  shadow._color_tokens = np.ones((num_players, shadow._num_colors), dtype=bool)
  shadow._trump_broken = False
  shadow._completed_tricks = []
  return shadow


def _reset_to_trick_replay_start(sampled_state):
  """Builds a sampled-world shadow state at the start of trick-taking."""
  shadow = sampled_state.clone()
  num_players = int(getattr(shadow, "_num_players"))

  shadow._phase = 3
  shadow._game_over = False
  shadow._returns = [0.0] * num_players
  shadow._rewards = [0.0] * num_players
  shadow._last_raw_scores = [0.0] * num_players
  shadow._player_adjacency_bonus = [0.0] * num_players
  shadow._hands = _replay_start_hands(sampled_state)
  shadow._trick_number = 0
  shadow._start_player = int(getattr(shadow, "_round_start_player", 0))
  shadow._current_player = shadow._start_player
  shadow._led_color = None
  shadow._cards_played_this_trick = [None] * num_players
  shadow._tricks_won = np.zeros(num_players, dtype=int)
  shadow._board_ownership = _initial_public_board(shadow)
  shadow._has_paradoxed = [False] * num_players
  shadow._color_tokens = np.ones((num_players, shadow._num_colors), dtype=bool)
  shadow._trump_broken = False
  shadow._completed_tricks = []
  return shadow


def _parse_ref_policy_mix(spec):
  if isinstance(spec, dict):
    items = spec.items()
  else:
    items = []
    for raw_part in str(spec or "model:1.0").replace("|", ",").split(","):
      part = raw_part.strip()
      if not part:
        continue
      if ":" in part:
        name, raw_weight = part.split(":", 1)
        weight = float(raw_weight)
      else:
        name, weight = part, 1.0
      items.append((name, weight))
  weights = {}
  for name, weight in items:
    key = str(name).strip().lower()
    if not key:
      continue
    is_indexed_model = key.startswith("model") and key[5:].isdigit()
    if key not in {
        "model", "model_avg", "uniform", "heuristic",
        "heuristic_target2", "heuristic_adj2"
    } and not is_indexed_model:
      raise ValueError(f"Unknown belief reference policy component: {name}")
    weight = max(0.0, float(weight))
    if weight > 0:
      weights[key] = weights.get(key, 0.0) + weight
  total = float(sum(weights.values()))
  if total <= 0:
    return {"model": 1.0}
  return {name: weight / total for name, weight in weights.items()}


def _reference_model_components(model, component_name):
  if model is None:
    return []
  if isinstance(model, (list, tuple)):
    models = list(model)
  else:
    models = [model]
  if not models:
    return []
  if component_name == "model":
    return [(models[0], 1.0)]
  if component_name == "model_avg":
    weight = 1.0 / float(len(models))
    return [(candidate, weight) for candidate in models]
  if component_name.startswith("model") and component_name[5:].isdigit():
    idx = int(component_name[5:])
    if 0 <= idx < len(models):
      return [(models[idx], 1.0)]
    return []
  return []


def _ref_policy_mix_for_phase(args, state):
  base_mix = str(getattr(args, "counterfactual_belief_ref_policy_mix", "model:1.0"))
  by_phase = str(
      getattr(args, "counterfactual_belief_ref_policy_mix_by_phase", "") or ""
  ).strip()
  if not by_phase:
    return base_mix
  current_phase = phase_name_for_state(state)
  default_mix = None
  for raw_part in by_phase.split(";"):
    part = raw_part.strip()
    if not part or "=" not in part:
      continue
    phase_spec, mix_spec = part.split("=", 1)
    phase_names = {
        name.strip().lower()
        for name in phase_spec.split("|")
        if name.strip()
    }
    if current_phase in phase_names:
      return mix_spec.strip()
    if "default" in phase_names or "*" in phase_names:
      default_mix = mix_spec.strip()
  return default_mix or base_mix


def _args_with_phase_ref_mix(args, state):
  selected_mix = _ref_policy_mix_for_phase(args, state)
  if selected_mix == getattr(args, "counterfactual_belief_ref_policy_mix", "model:1.0"):
    return args
  copied = SimpleNamespace(**vars(args))
  copied.counterfactual_belief_ref_policy_mix = selected_mix
  return copied


def _reference_policy_action_prob(shadow, player, action, model, args, device):
  legal = shadow.legal_actions(player)
  if action not in legal:
    return 0.0
  mix = _parse_ref_policy_mix(
      getattr(args, "counterfactual_belief_ref_policy_mix", "model:1.0")
  )
  prob = 0.0
  for component_name, component_weight in mix.items():
    if not component_name.startswith("model") or component_weight <= 0:
      continue
    if device is None:
      return None
    model_components = _reference_model_components(model, component_name)
    if not model_components:
      return None
    for ref_model, ref_weight in model_components:
      policy, _ = model_policy_value(
          ref_model,
          shadow,
          player,
          int(shadow.num_distinct_actions()),
          args.value_scale,
          device,
      )
      prob += component_weight * ref_weight * float(policy[action])
  if mix.get("uniform", 0.0) > 0:
    prob += mix["uniform"] / float(len(legal))
  for bot_name in ("heuristic", "heuristic_target2", "heuristic_adj2"):
    weight = mix.get(bot_name, 0.0)
    if weight <= 0:
      continue
    try:
      choice = make_bot(bot_name, seed=0).step(shadow.clone(), player)
    except Exception:
      continue
    if choice == action:
      prob += weight
  return prob


def _model_policy_action_probs_batch(ref_model, shadows, players, actions, args, device):
  if not shadows:
    return []
  num_actions = int(shadows[0].num_distinct_actions())
  obs_rows = []
  action_feature_rows = []
  legal_rows = []
  for shadow, player in zip(shadows, players):
    obs_rows.append(np.array(shadow.observation_tensor(player), dtype=np.float32))
    action_feature_rows.append(action_feature_matrix(shadow, player, num_actions))
    legal_rows.append(shadow.legal_actions(player))
  obs_t = torch.tensor(np.stack(obs_rows), dtype=torch.float32, device=device)
  obs_t = adapt_observation_batch(obs_t, model_input_size(ref_model))
  action_features_t = torch.tensor(
      np.stack(action_feature_rows), dtype=torch.float32, device=device
  )
  with torch.no_grad():
    logits_t, _ = ref_model(obs_t, action_features_t)
    logits = logits_t.detach().cpu().numpy()
  probs = []
  for row_idx, (legal, action) in enumerate(zip(legal_rows, actions)):
    if action not in legal:
      probs.append(0.0)
      continue
    legal_logits = logits[row_idx, legal]
    max_logit = float(np.max(legal_logits))
    exp = np.exp(legal_logits - max_logit)
    total = float(np.sum(exp))
    if total <= 0 or not math.isfinite(total):
      probs.append(1.0 / float(len(legal)))
      continue
    action_pos = legal.index(action)
    probs.append(float(exp[action_pos] / total))
  return probs


def _reference_policy_action_probs_batch(shadows, players, actions, model, args, device):
  if not shadows:
    return []
  mix = _parse_ref_policy_mix(
      getattr(args, "counterfactual_belief_ref_policy_mix", "model:1.0")
  )
  probs = [0.0 for _ in shadows]
  for component_name, component_weight in mix.items():
    if not component_name.startswith("model") or component_weight <= 0:
      continue
    if device is None:
      return [None for _ in shadows]
    model_components = _reference_model_components(model, component_name)
    if not model_components:
      return [None for _ in shadows]
    for ref_model, ref_weight in model_components:
      model_probs = _model_policy_action_probs_batch(
          ref_model, shadows, players, actions, args, device
      )
      for idx, model_prob in enumerate(model_probs):
        probs[idx] += component_weight * ref_weight * float(model_prob)
  uniform_weight = float(mix.get("uniform", 0.0))
  if uniform_weight > 0:
    for idx, (shadow, player, action) in enumerate(zip(shadows, players, actions)):
      legal = shadow.legal_actions(player)
      if action in legal:
        probs[idx] += uniform_weight / float(len(legal))
  for bot_name in ("heuristic", "heuristic_target2", "heuristic_adj2"):
    weight = float(mix.get(bot_name, 0.0))
    if weight <= 0:
      continue
    for idx, (shadow, player, action) in enumerate(zip(shadows, players, actions)):
      try:
        choice = make_bot(bot_name, seed=0).step(shadow.clone(), player)
      except Exception:
        continue
      if choice == action:
        probs[idx] += weight
  return probs


def _score_observed_trick_likelihood_batch(sampled_states, model, args, device):
  """Batched variant of _score_observed_trick_likelihood for belief rankers."""
  states = list(sampled_states)
  if not states:
    return []
  num_states = len(states)
  scores = [0.0 for _ in states]
  scored = [0 for _ in states]
  status = ["ok" for _ in states]
  prob_floor = max(
      1e-12, float(getattr(args, "counterfactual_belief_logprob_floor", 1e-6))
  )

  def mark_none(idx):
    if status[idx] == "ok":
      status[idx] = "none"

  def mark_illegal(idx):
    status[idx] = "illegal"
    scores[idx] = -1e9

  def apply_probability(records, probs):
    for (idx, shadow, action), prob in zip(records, probs):
      if status[idx] != "ok":
        continue
      if prob is None:
        mark_none(idx)
        continue
      try:
        scores[idx] += math.log(max(prob_floor, float(prob)))
        scored[idx] += 1
        shadow.apply_action(action)
      except Exception:
        mark_none(idx)

  try:
    prediction_shadows = [None for _ in states]
    max_players = max(int(getattr(state, "_num_players", 0)) for state in states)
    for idx, state in enumerate(states):
      if int(getattr(state, "_phase", -1)) < 2:
        status[idx] = "done"
        continue
      if int(getattr(state, "_num_players")) > 2:
        prediction_shadows[idx] = _reset_to_prediction_replay_start(state)
    for _ in range(max_players):
      records = []
      players = []
      actions = []
      for idx, state in enumerate(states):
        shadow = prediction_shadows[idx]
        if status[idx] != "ok" or shadow is None:
          continue
        replay_player = int(shadow.current_player())
        prediction = int(state._predictions[replay_player])
        if prediction < 0:
          prediction_shadows[idx] = None
          continue
        action = 100 + prediction
        if action not in shadow.legal_actions(replay_player):
          mark_illegal(idx)
          continue
        records.append((idx, shadow, action))
        players.append(replay_player)
        actions.append(action)
      if records:
        probs = _reference_policy_action_probs_batch(
            [record[1] for record in records], players, actions, model, args, device
        )
        apply_probability(records, probs)

    trick_shadows = [None for _ in states]
    num_card_types = [0 for _ in states]
    max_completed = 0
    for idx, state in enumerate(states):
      if status[idx] != "ok":
        continue
      if int(getattr(state, "_phase", -1)) >= 3 and len(_observed_trick_cards(state)) > 0:
        trick_shadows[idx] = _reset_to_trick_replay_start(state)
        num_card_types[idx] = int(getattr(state, "_num_card_types"))
        max_completed = max(max_completed, len(getattr(state, "_completed_tricks", [])))

    for trick_idx in range(max_completed):
      for _ in range(max_players):
        records = []
        players = []
        actions = []
        for idx, state in enumerate(states):
          shadow = trick_shadows[idx]
          if status[idx] != "ok" or shadow is None:
            continue
          completed = getattr(state, "_completed_tricks", [])
          if trick_idx >= len(completed):
            continue
          replay_player = int(shadow.current_player())
          card_by_player = {int(p): cardinfo for p, cardinfo in completed[trick_idx]}
          cardinfo = card_by_player.get(replay_player)
          if cardinfo is None:
            mark_none(idx)
            continue
          action = _action_from_cardinfo(cardinfo, num_card_types[idx])
          if action not in shadow.legal_actions(replay_player):
            mark_illegal(idx)
            continue
          records.append((idx, shadow, action))
          players.append(replay_player)
          actions.append(action)
        if records:
          probs = _reference_policy_action_probs_batch(
              [record[1] for record in records], players, actions, model, args, device
          )
          apply_probability(records, probs)

    current_done = [False for _ in states]
    for _ in range(max_players):
      records = []
      players = []
      actions = []
      for idx, state in enumerate(states):
        shadow = trick_shadows[idx]
        if status[idx] != "ok" or shadow is None or current_done[idx]:
          continue
        replay_player = int(shadow.current_player())
        cardinfo = state._cards_played_this_trick[replay_player]
        if cardinfo is None:
          current_done[idx] = True
          continue
        action = _action_from_cardinfo(cardinfo, num_card_types[idx])
        if action not in shadow.legal_actions(replay_player):
          mark_illegal(idx)
          continue
        records.append((idx, shadow, action))
        players.append(replay_player)
        actions.append(action)
      if records:
        probs = _reference_policy_action_probs_batch(
            [record[1] for record in records], players, actions, model, args, device
        )
        apply_probability(records, probs)
  except Exception:
    return [None for _ in states]

  results = []
  for idx in range(num_states):
    if status[idx] == "none":
      results.append(None)
    elif status[idx] == "illegal":
      results.append(-1e9)
    elif scored[idx] <= 0:
      results.append(0.0)
    else:
      results.append(scores[idx])
  return results


def _score_observed_trick_likelihood(sampled_state, model, args, device):
  phase = int(getattr(sampled_state, "_phase", -1))
  if phase < 2:
    return 0.0
  num_card_types = int(getattr(sampled_state, "_num_card_types"))
  prob_floor = max(
      1e-12, float(getattr(args, "counterfactual_belief_logprob_floor", 1e-6))
  )

  score = 0.0
  scored = 0
  try:
    if int(getattr(sampled_state, "_num_players")) > 2:
      prediction_shadow = _reset_to_prediction_replay_start(sampled_state)
      for _ in range(int(getattr(sampled_state, "_num_players"))):
        replay_player = int(prediction_shadow.current_player())
        prediction = int(sampled_state._predictions[replay_player])
        if prediction < 0:
          break
        action = 100 + prediction
        legal = prediction_shadow.legal_actions(replay_player)
        if action not in legal:
          return -1e9
        prob = _reference_policy_action_prob(
            prediction_shadow, replay_player, action, model, args, device
        )
        if prob is None:
          return None
        score += math.log(max(prob_floor, float(prob)))
        scored += 1
        prediction_shadow.apply_action(action)

    if phase >= 3 and len(_observed_trick_cards(sampled_state)) > 0:
      shadow = _reset_to_trick_replay_start(sampled_state)
      for trick in getattr(sampled_state, "_completed_tricks", []):
        for _ in range(int(getattr(sampled_state, "_num_players"))):
          replay_player = int(shadow.current_player())
          card_by_player = {int(p): cardinfo for p, cardinfo in trick}
          cardinfo = card_by_player.get(replay_player)
          if cardinfo is None:
            return None
          action = _action_from_cardinfo(cardinfo, num_card_types)
          legal = shadow.legal_actions(replay_player)
          if action not in legal:
            return -1e9
          prob = _reference_policy_action_prob(
              shadow, replay_player, action, model, args, device
          )
          if prob is None:
            return None
          score += math.log(max(prob_floor, float(prob)))
          scored += 1
          shadow.apply_action(action)

      for _ in range(int(getattr(sampled_state, "_num_players"))):
        replay_player = int(shadow.current_player())
        cardinfo = sampled_state._cards_played_this_trick[replay_player]
        if cardinfo is None:
          break
        action = _action_from_cardinfo(cardinfo, num_card_types)
        legal = shadow.legal_actions(replay_player)
        if action not in legal:
          return -1e9
        prob = _reference_policy_action_prob(
            shadow, replay_player, action, model, args, device
        )
        if prob is None:
          return None
        score += math.log(max(prob_floor, float(prob)))
        scored += 1
        shadow.apply_action(action)
  except Exception:
    return None

  if scored <= 0:
    return 0.0
  return score


def _uniform_counterfactual_resamples(state, player, count):
  states = []
  for _ in range(max(0, int(count))):
    try:
      states.append(state.resample_from_infostate(player, np.random))
    except Exception:
      states.append(state.clone())
  return states


def policy_weighted_belief_states(state, player, args, model, device):
  samples = max(1, int(getattr(args, "counterfactual_belief_samples", 1)))
  candidate_count = max(
      samples, int(getattr(args, "counterfactual_belief_candidates", 8))
  )
  score_args = _args_with_phase_ref_mix(args, state)
  candidates = _uniform_counterfactual_resamples(state, player, candidate_count)
  scores = _score_observed_trick_likelihood_batch(candidates, model, score_args, device)
  if not candidates or any(score is None for score in scores):
    return _uniform_counterfactual_resamples(state, player, samples)

  scores_np = np.array(scores, dtype=np.float64)
  finite = np.isfinite(scores_np)
  if not np.any(finite):
    return _uniform_counterfactual_resamples(state, player, samples)
  min_finite = float(np.min(scores_np[finite]))
  scores_np[~finite] = min_finite - 1e6
  temperature = max(
      1e-6, float(getattr(args, "counterfactual_belief_policy_temperature", 1.0))
  )
  logits = (scores_np - float(np.max(scores_np))) / temperature
  weights = np.exp(logits)
  total = float(weights.sum())
  if total <= 0 or not math.isfinite(total):
    probs = np.ones(len(candidates), dtype=np.float64) / float(len(candidates))
  else:
    probs = weights / total
  mix = min(
      1.0,
      max(0.0, float(getattr(args, "counterfactual_belief_uniform_mix", 0.15))),
  )
  probs = (1.0 - mix) * probs + mix / float(len(candidates))
  probs = probs / float(probs.sum())
  chosen = np.random.choice(len(candidates), size=samples, replace=True, p=probs)
  return [candidates[int(idx)].clone() for idx in chosen]


def counterfactual_belief_states(state, player, args, model=None, device=None):
  source = getattr(args, "counterfactual_belief_source", "infostate")
  samples = max(1, int(getattr(args, "counterfactual_belief_samples", 1)))
  if source == "actual":
    return [state]
  if source == "policy_weighted":
    ref_model = _belief_reference_model(model)
    return policy_weighted_belief_states(state, player, args, ref_model, device)
  states = [state] if source == "mixed" else []
  resample_count = samples if source == "infostate" else max(0, samples - 1)
  states.extend(_uniform_counterfactual_resamples(state, player, resample_count))
  return states


def _belief_context_value(args, context, name, default):
  """Reads belief sampler knobs across train/eval/league argument namespaces."""
  attrs = []
  if context:
    attrs.append(f"{context}_belief_{name}")
  attrs.append(f"belief_{name}")
  if name != "source":
    attrs.append(f"counterfactual_belief_{name}")
  elif context == "counterfactual":
    attrs.append("counterfactual_belief_source")
  for attr in attrs:
    if hasattr(args, attr):
      return getattr(args, attr)
  return default


def sampled_belief_states_for_policy(
    state, player, samples, args, model, device, value_scale, context="self_play"
):
  """Samples deployment-safe hidden worlds for belief policy/search actors.

  Defaults to the historical uniform information-state sampler. When a caller
  opts into ``policy_weighted`` for its context, this reuses the same public
  history likelihood posterior used by counterfactual-label generation.
  """
  sample_count = max(1, int(samples))
  source = str(
      _belief_context_value(args, context, "source", "infostate")
  ).strip()
  if source == "policy_weighted":
    belief_args = SimpleNamespace(
        counterfactual_belief_samples=sample_count,
        counterfactual_belief_candidates=max(
            sample_count,
            int(_belief_context_value(args, context, "candidates", 8)),
        ),
        counterfactual_belief_policy_temperature=float(
            _belief_context_value(args, context, "policy_temperature", 1.0)
        ),
        counterfactual_belief_uniform_mix=float(
            _belief_context_value(args, context, "uniform_mix", 0.15)
        ),
        counterfactual_belief_ref_policy_mix=str(
            _belief_context_value(args, context, "ref_policy_mix", "model:1.0")
        ),
        counterfactual_belief_ref_policy_mix_by_phase=str(
            _belief_context_value(
                args, context, "ref_policy_mix_by_phase", ""
            )
        ),
        counterfactual_belief_logprob_floor=float(
            _belief_context_value(args, context, "logprob_floor", 1e-6)
        ),
        value_scale=value_scale,
    )
    return policy_weighted_belief_states(
        state, player, belief_args, _belief_reference_model(model), device
    )
  if source == "ranker_resample":
    ranker_path = str(_belief_context_value(args, context, "ranker", "") or "")
    if not ranker_path:
      return _uniform_counterfactual_resamples(state, player, sample_count)
    from quantum_cat_belief_ranker import ranker_resampled_belief_states
    score_args = SimpleNamespace(
        counterfactual_belief_ref_policy_mix=str(
            _belief_context_value(args, context, "ref_policy_mix", "model:1.0")
        ),
        counterfactual_belief_ref_policy_mix_by_phase=str(
            _belief_context_value(args, context, "ref_policy_mix_by_phase", "")
        ),
        counterfactual_belief_logprob_floor=float(
            _belief_context_value(args, context, "logprob_floor", 1e-6)
        ),
        value_scale=value_scale,
    )
    return ranker_resampled_belief_states(
        state,
        player,
        sample_count,
        max(
            sample_count,
            int(_belief_context_value(args, context, "ranker_candidates", 64)),
        ),
        ranker_path,
        device or "cpu",
        temperature=float(
            _belief_context_value(args, context, "ranker_temperature", 0.7)
        ),
        uniform_mix=float(
            _belief_context_value(args, context, "ranker_uniform_mix", 0.25)
        ),
        likelihood_score_fn=lambda candidate: _score_observed_trick_likelihood(
            candidate, _belief_reference_model(model), score_args, device
        ),
        likelihood_scores_fn=lambda candidates: _score_observed_trick_likelihood_batch(
            candidates, _belief_reference_model(model), score_args, device
        ),
    )
  return _uniform_counterfactual_resamples(state, player, sample_count)


def counterfactual_action_targets(
    state, round_game, player, legal, seat_roles, fixed_bots, model,
    opponent_model, args, device, policy=None,
):
  rollouts = int(getattr(args, "counterfactual_action_rollouts", 0))
  if rollouts <= 0 or not counterfactual_phase_allowed(state, args):
    return None, None
  if not counterfactual_policy_hard_enough(policy, legal, args):
    return None, None
  num_actions = round_game.num_distinct_actions()
  targets = np.zeros(num_actions, dtype=np.float32)
  mask = np.zeros(num_actions, dtype=np.float32)
  belief_states = counterfactual_belief_states(
      state, player, args, model=opponent_model or model, device=device
  )
  candidate_actions = sampled_counterfactual_legal_actions(
      state, player, legal, args, policy=policy
  )
  action_outcomes = {action: [] for action in candidate_actions}
  for belief_state in belief_states:
    legal_candidate_actions = [
        action
        for action in candidate_actions
        if action in belief_state.legal_actions(player)
    ]
    if not legal_candidate_actions:
      continue
    for _ in range(rollouts):
      py_random_state = random.getstate()
      np_random_state = np.random.get_state()
      for action in legal_candidate_actions:
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)
        action_outcomes[action].append(
          float(rollout_paradox_after_action(
              belief_state,
              round_game,
              player,
              action,
              seat_roles,
              fixed_bots,
              model,
              opponent_model,
              args,
              device,
          ))
        )
  for action, outcomes in action_outcomes.items():
    if outcomes:
      targets[action] = float(np.mean(outcomes))
      mask[action] = 1.0
  return targets, mask


def counterfactual_action_value_audit_passes(
    state,
    round_game,
    player,
    action_scores,
    policy,
    seat_roles,
    fixed_bots,
    model,
    opponent_model,
    args,
    device,
    rollout_fn,
):
  audit_rollouts = int(
      getattr(args, "counterfactual_action_value_audit_rollouts", 0)
  )
  if audit_rollouts <= 0 or len(action_scores) < 2 or policy is None:
    return True
  label_best = max(action_scores, key=lambda action: action_scores[action])
  policy_best = max(action_scores, key=lambda action: float(policy[action]))
  if label_best == policy_best:
    return True

  audit_actions = [label_best, policy_best]
  audit_outcomes = {action: [] for action in audit_actions}
  audit_belief_states = counterfactual_belief_states(
      state, player, args, model=opponent_model or model, device=device
  )
  for belief_state in audit_belief_states:
    legal_actions = belief_state.legal_actions(player)
    if any(action not in legal_actions for action in audit_actions):
      continue
    for _ in range(audit_rollouts):
      py_random_state = random.getstate()
      np_random_state = np.random.get_state()
      for action in audit_actions:
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)
        audit_outcomes[action].append(
            float(
                rollout_fn(
                    belief_state,
                    round_game,
                    player,
                    action,
                    seat_roles,
                    fixed_bots,
                    model,
                    opponent_model,
                    args,
                    device,
                )
            )
        )
  if not audit_outcomes[label_best] or not audit_outcomes[policy_best]:
    return False
  audited_margin = (
      float(np.mean(audit_outcomes[label_best]))
      - float(np.mean(audit_outcomes[policy_best]))
  )
  min_margin = float(
      getattr(args, "counterfactual_action_value_audit_min_margin", 0.0)
  )
  return audited_margin >= min_margin


def counterfactual_action_value_targets(
    state, round_game, player, legal, seat_roles, fixed_bots, model,
    opponent_model, args, device, policy=None,
):
  rollouts = int(getattr(args, "counterfactual_action_value_rollouts", 0))
  if rollouts <= 0 or not counterfactual_phase_allowed(state, args):
    return None, None
  if not counterfactual_policy_hard_enough(policy, legal, args):
    return None, None
  num_actions = round_game.num_distinct_actions()
  targets = np.zeros(num_actions, dtype=np.float32)
  mask = np.zeros(num_actions, dtype=np.float32)
  action_outcomes = {}
  action_stderrs = {}
  belief_states = counterfactual_belief_states(
      state, player, args, model=opponent_model or model, device=device
  )
  objective = str(
      getattr(args, "counterfactual_action_value_objective", "score") or "score"
  )
  if objective == "survival":
    rollout_fn = rollout_survival_value_after_action
    target_scale = 1.0
  else:
    rollout_fn = (
        rollout_match_score_after_action
        if (
            getattr(args, "counterfactual_full_match_rollout", False)
            and getattr(args, "full_match_training", False)
        )
        else rollout_score_after_action
    )
    target_scale = float(getattr(args, "value_scale", 1.0) or 1.0)
  candidate_actions = sampled_counterfactual_legal_actions(
      state, player, legal, args, policy=policy
  )
  use_advantage = getattr(args, "counterfactual_action_value_advantage", False)
  for belief_state in belief_states:
    legal_candidate_actions = [
        action
        for action in candidate_actions
        if action in belief_state.legal_actions(player)
    ]
    if not legal_candidate_actions:
      continue
    for _ in range(rollouts):
      py_random_state = random.getstate()
      np_random_state = np.random.get_state()
      row_scores = {}
      for action in legal_candidate_actions:
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)
        row_scores[action] = rollout_fn(
            belief_state,
            round_game,
            player,
            action,
            seat_roles,
            fixed_bots,
            model,
            opponent_model,
            args,
            device,
        )
      baseline = float(np.mean(list(row_scores.values()))) if use_advantage else 0.0
      for action, score in row_scores.items():
        action_outcomes.setdefault(action, []).append(float(score) - baseline)
  if not action_outcomes:
    return targets, mask
  action_scores = {}
  for action, outcomes in action_outcomes.items():
    outcomes = np.array(outcomes, dtype=np.float32)
    action_scores[action] = float(np.mean(outcomes))
    if len(outcomes) > 1:
      action_stderrs[action] = float(
          np.std(outcomes, ddof=1) / math.sqrt(len(outcomes))
      )
    else:
      action_stderrs[action] = None
  if not action_scores:
    return targets, mask
  min_spread = float(
      getattr(args, "counterfactual_action_value_min_spread", 0.0)
  )
  if min_spread > 0:
    spread = float(max(action_scores.values()) - min(action_scores.values()))
    if spread < min_spread:
      return targets, mask
  if not counterfactual_action_value_audit_passes(
      state,
      round_game,
      player,
      action_scores,
      policy,
      seat_roles,
      fixed_bots,
      model,
      opponent_model,
      args,
      device,
      rollout_fn,
  ):
    return targets, mask
  max_stderr = float(
      getattr(args, "counterfactual_action_value_max_stderr", 0.0)
  )
  for action, score in action_scores.items():
    stderr = action_stderrs.get(action)
    if max_stderr > 0 and (stderr is None or stderr > max_stderr):
      continue
    targets[action] = score / target_scale
    if getattr(args, "counterfactual_action_value_confidence_weight", False):
      if stderr is None:
        weight = 0.5
      else:
        weight = 1.0 / (1.0 + max(0.0, stderr))
      mask[action] = max(0.05, min(1.0, weight))
    else:
      mask[action] = 1.0
  return targets, mask


def blend_policy_with_counterfactual_values(policy, value_targets, value_mask, args):
  weight = float(getattr(args, "counterfactual_policy_target_weight", 0.0))
  if weight <= 0 or value_targets is None or value_mask is None:
    return policy
  labeled = np.array(value_mask, dtype=np.float32) > 0.0
  min_actions = max(
      2, int(getattr(args, "counterfactual_policy_target_min_actions", 2))
  )
  if int(np.sum(labeled)) < min_actions:
    return policy
  shaped = np.array(policy, dtype=np.float32, copy=True)
  temperature = max(
      1e-6, float(getattr(args, "counterfactual_policy_target_temperature", 0.08))
  )
  label_values = np.array(value_targets, dtype=np.float32)[labeled]
  min_spread = float(
      getattr(args, "counterfactual_policy_target_min_spread", 0.0)
  )
  if min_spread > 0:
    spread = float(np.max(label_values) - np.min(label_values))
    if spread < min_spread:
      return policy
  label_values = label_values / temperature
  label_values -= np.max(label_values)
  label_exp = np.exp(label_values)
  label_total = float(label_exp.sum())
  if label_total <= 0 or not math.isfinite(label_total):
    return policy
  cf_policy = np.zeros_like(shaped, dtype=np.float32)
  cf_policy[labeled] = label_exp / label_total
  shaped = (1.0 - weight) * shaped + weight * cf_policy
  total = float(shaped.sum())
  if total <= 0 or not math.isfinite(total):
    return policy
  return (shaped / total).astype(np.float32)


def blend_policy_with_counterfactual_paradox(policy, risk_targets, risk_mask, args):
  weight = float(
      getattr(args, "counterfactual_paradox_policy_target_weight", 0.0)
  )
  if weight <= 0 or risk_targets is None or risk_mask is None:
    return policy
  labeled = np.array(risk_mask, dtype=np.float32) > 0.0
  min_actions = max(
      2,
      int(getattr(args, "counterfactual_paradox_policy_target_min_actions", 2)),
  )
  if int(np.sum(labeled)) < min_actions:
    return policy
  shaped = np.array(policy, dtype=np.float32, copy=True)
  risks = np.array(risk_targets, dtype=np.float32)[labeled]
  min_spread = float(
      getattr(args, "counterfactual_paradox_policy_target_min_spread", 0.0)
  )
  if min_spread > 0:
    spread = float(np.max(risks) - np.min(risks))
    if spread < min_spread:
      return policy
  temperature = max(
      1e-6,
      float(getattr(
          args, "counterfactual_paradox_policy_target_temperature", 0.08
      )),
  )
  label_values = -risks / temperature
  label_values -= np.max(label_values)
  label_exp = np.exp(label_values)
  label_total = float(label_exp.sum())
  if label_total <= 0 or not math.isfinite(label_total):
    return policy
  cf_policy = np.zeros_like(shaped, dtype=np.float32)
  cf_policy[labeled] = label_exp / label_total
  shaped = (1.0 - weight) * shaped + weight * cf_policy
  total = float(shaped.sum())
  if total <= 0 or not math.isfinite(total):
    return policy
  return (shaped / total).astype(np.float32)


def league_policy_match(
    model, opponent_model, bot_names, args, device, game_index, initial_start
):
  if isinstance(opponent_model, (list, tuple)):
    old_policy_roles = [
        f"old_policy_{idx}" for idx in range(len(opponent_model))
    ]
  else:
    old_policy_roles = ["old_policy"]
  opponents = old_policy_roles + list(bot_names)
  if len(opponents) < args.players - 1:
    raise ValueError("league needs at least players-1 opponents")

  seat_roles = ["learner"] + [
      opponents[(game_index + idx) % len(opponents)]
      for idx in range(args.players - 1)
  ]
  shift = game_index % args.players
  if shift:
    seat_roles = seat_roles[-shift:] + seat_roles[:-shift]

  fixed_bots = {}
  for seat, role in enumerate(seat_roles):
    if role != "learner" and not is_old_policy_role(role):
      fixed_bots[seat] = make_bot(
          role, seed=args.seed + 900000 + game_index * 31 + seat
      )

  match_totals = np.zeros(args.players, dtype=np.float32)
  final_round_scores = np.zeros(args.players, dtype=np.float32)
  all_examples = []
  terminal_action_paradox_targets = []
  start_counts = [0] * args.players
  learner_seat = seat_roles.index("learner")
  learner_plys = 0
  label_attempts = 0
  phase_label_attempts = {}
  paradoxes = np.zeros(args.players, dtype=np.float32)

  for round_index in range(args.players):
    start_player = (initial_start + round_index) % args.players
    start_counts[start_player] += 1
    round_game = make_game(args, start_player)
    state = round_game.new_initial_state()
    round_example_start = len(all_examples)
    if getattr(args, "match_context", False):
      state.set_match_context(match_totals, round_index)

    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        state.apply_action(int(np.random.choice(actions, p=probs)))
        continue

      player = state.current_player()
      role = seat_roles[player]
      legal = state.legal_actions(player)
      if role == "learner":
        policy = learner_policy(state, player, model, args, device)
        legal_mask = np.zeros(round_game.num_distinct_actions(), dtype=np.float32)
        legal_mask[legal] = 1.0
        action_features = action_feature_matrix(
            state, player, round_game.num_distinct_actions()
        )
        temp = args.temperature if learner_plys < args.temperature_drop else 0.0
        action = sample_action(policy, legal, temp)
        should_label_actions = (
            counterfactual_phase_allowed(state, args)
            and counterfactual_phase_label_budget_allows(
                state, args, label_attempts, phase_label_attempts
            )
            and random.random()
            <= float(getattr(args, "counterfactual_action_label_prob", 1.0))
        )
        if should_label_actions:
          label_attempts += 1
          phase_name = phase_name_for_state(state)
          phase_label_attempts[phase_name] = (
              int(phase_label_attempts.get(phase_name, 0)) + 1
          )
          maybe_print_counterfactual_label_progress(
              args, "self_play_match", label_attempts, phase_label_attempts,
              phase_name, len(legal)
          )
          cf_targets, cf_mask = counterfactual_action_targets(
              state,
              round_game,
              player,
              legal,
              seat_roles,
              fixed_bots,
              model,
              opponent_model,
              args,
              device,
              policy=policy,
          )
          cf_value_targets, cf_value_mask = counterfactual_action_value_targets(
              state,
              round_game,
              player,
              legal,
              seat_roles,
              fixed_bots,
              model,
              opponent_model,
              args,
              device,
              policy=policy,
          )
          policy = blend_policy_with_counterfactual_paradox(
              policy, cf_targets, cf_mask, args
          )
          policy = blend_policy_with_counterfactual_values(
              policy, cf_value_targets, cf_value_mask, args
          )
        else:
          cf_targets = cf_mask = None
          cf_value_targets = cf_value_mask = None
        all_examples.append((
            np.array(state.observation_tensor(player), dtype=np.float32),
            legal_mask,
            policy,
            None,
            action,
            player,
            cf_targets,
            cf_mask,
            cf_value_targets,
            cf_value_mask,
            action_features,
        ))
        learner_plys += 1
      elif is_old_policy_role(role):
        action = old_policy_action(
            state, player, legal, opponent_model, role, args, device
        )
      else:
        action = fixed_bots[player].step(state, player)
        if action not in legal:
          raise ValueError(f"League bot chose illegal action {action}: {legal}")
      state.apply_action(action)

    raw_scores = (
        state.raw_round_scores()
        if hasattr(state, "raw_round_scores")
        else state.returns()
    )
    match_totals += np.array(raw_scores, dtype=np.float32)
    final_round_scores = np.array(raw_scores, dtype=np.float32)
    round_paradox_target = terminal_paradox_target(state, args.players)
    paradoxes += round_paradox_target
    terminal_action_paradox_targets.extend(
        [round_paradox_target] * (len(all_examples) - round_example_start)
    )

  final_returns = value_targets_from_scores(
      match_totals,
      (paradoxes > 0).astype(np.float32),
      args,
      final_round_scores,
  )
  return (
      with_terminal_targets(
          all_examples,
          final_returns,
          (paradoxes > 0),
          terminal_action_paradox_targets=terminal_action_paradox_targets,
      ),
      match_totals,
      start_counts,
      seat_roles,
      learner_seat,
      paradoxes,
  )


def parse_named_weights(spec, allowed_names):
  weights = {name: 1.0 for name in allowed_names}
  if not spec:
    return weights
  for part in str(spec).split(","):
    part = part.strip()
    if not part:
      continue
    if "=" not in part:
      raise ValueError(f"Expected name=value in weight spec: {part!r}")
    name, value = part.split("=", 1)
    name = name.strip()
    if name not in weights:
      raise ValueError(
          f"Unknown weight name {name!r}; expected one of {allowed_names}"
      )
    weights[name] = max(0.0, float(value))
  return weights


def policy_target_action_type_weights(action_features_t, policy_t, args, device):
  spec = getattr(args, "policy_target_action_type_weights", "")
  if not spec:
    return torch.ones(policy_t.shape[0], dtype=policy_t.dtype, device=device)
  type_names = ["discard", "prediction", "play", "paradox", "other"]
  weights = parse_named_weights(spec, type_names)
  weight_values = torch.tensor(
      [weights[name] for name in type_names],
      dtype=policy_t.dtype,
      device=device,
  )
  target_actions = torch.argmax(policy_t, dim=1)
  batch_idx = torch.arange(policy_t.shape[0], device=device)
  type_one_hot = action_features_t[batch_idx, target_actions, 6:11]
  type_idx = torch.argmax(type_one_hot, dim=1)
  return weight_values[type_idx]


def target_action_feature_rows(action_features_t, policy_t):
  target_actions = torch.argmax(policy_t, dim=1)
  batch_idx = torch.arange(policy_t.shape[0], device=policy_t.device)
  return action_features_t[batch_idx, target_actions]


def policy_target_bucket_masks(action_rows):
  rank_norm = action_rows[:, ACTION_FEATURE_RANK_NORM_INDEX]
  cluster_growth = (
      (action_rows[:, ACTION_FEATURE_ADJACENCY_GAIN_INDEX] > 0.0)
      | (
          action_rows[:, APPENDED_ACTION_FEATURE_INDEX["cluster_frontier_gain"]]
          > 0.0
      )
      | (
          action_rows[
              :, APPENDED_ACTION_FEATURE_INDEX["cluster_connects_components"]
          ] > 0.5
      )
  )
  follows_led = action_rows[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
  hits_prediction = (
      action_rows[:, APPENDED_ACTION_FEATURE_INDEX["hits_prediction"]]
      > 0.5
  )
  prediction_feasible = (
      action_rows[:, APPENDED_ACTION_FEATURE_INDEX["can_still_hit_after"]]
      > 0.5
  )
  mid_rank = (rank_norm > 0.5) & (rank_norm <= (5.0 / 6.0))
  red = action_rows[:, ACTION_FEATURE_IS_RED_INDEX] > 0.5
  masks = {
      "token_loss": (
          (action_rows[:, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX] > 0.5)
          | (
              action_rows[
                  :, APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"]
              ] > 0.5
          )
      ),
      "follows_led": follows_led,
      "prediction_feasible": prediction_feasible,
      "hits_prediction": hits_prediction,
      "future_hit": (
          action_rows[:, APPENDED_ACTION_FEATURE_INDEX["hit_with_future_tricks"]]
          > 0.5
      ),
      "low_rank": rank_norm <= 0.5,
      "mid_rank": mid_rank,
      "mid_rank_cluster_growth": mid_rank & cluster_growth,
      "mid_rank_hits_prediction": mid_rank & hits_prediction,
      "mid_rank_follows_led": mid_rank & follows_led,
      "high_rank": rank_norm > (5.0 / 6.0),
      "cluster_growth": cluster_growth,
      "red": red,
      "red_cluster_growth": red & cluster_growth,
      "red_prediction_feasible": red & prediction_feasible,
      "red_hits_prediction": red & hits_prediction,
  }
  return masks


def policy_target_bucket_weights(action_features_t, policy_t, args, device):
  spec = getattr(args, "policy_target_bucket_weights", "")
  if not spec:
    return torch.ones(policy_t.shape[0], dtype=policy_t.dtype, device=device)
  weights = parse_named_weights(spec, TACTICAL_POLICY_BUCKET_NAMES)
  action_rows = target_action_feature_rows(action_features_t, policy_t)
  bucket_masks = policy_target_bucket_masks(action_rows)
  result = torch.ones(policy_t.shape[0], dtype=policy_t.dtype, device=device)
  for name, mask in bucket_masks.items():
    weight = float(weights.get(name, 1.0))
    if weight == 1.0:
      continue
    result = torch.where(
        mask,
        result * torch.tensor(weight, dtype=policy_t.dtype, device=device),
        result,
    )
  return result


def has_action_value_labels(example):
  if len(example) <= 10 or example[10] is None:
    return False
  return float(np.sum(example[10])) > 0


def has_action_paradox_labels(example):
  if len(example) <= 8 or example[8] is None:
    return False
  return float(np.sum(example[8])) > 0


def has_policy_target_labels(example):
  if len(example) <= 2 or example[2] is None:
    return False
  return float(np.sum(example[2])) > 0


def has_loaded_replay_validation_labels(example, args):
  kind = str(
      getattr(args, "loaded_replay_validation_label_kind", "action_value")
  )
  if kind == "action_value":
    return has_action_value_labels(example)
  if kind == "action_paradox":
    return has_action_paradox_labels(example)
  if kind == "policy":
    return has_policy_target_labels(example)
  if kind == "any":
    return (
        has_action_value_labels(example)
        or has_action_paradox_labels(example)
        or has_policy_target_labels(example)
    )
  raise ValueError(f"Unsupported loaded replay validation label kind: {kind}")


def action_value_label_quality(example):
  if not has_action_value_labels(example) or example[9] is None:
    return None
  targets = np.array(example[9], dtype=np.float32)
  mask = np.array(example[10], dtype=np.float32) > 0
  if int(np.sum(mask)) < 2:
    return None
  values = targets[mask]
  spread = float(np.max(values) - np.min(values))
  sorted_values = np.sort(values)
  top_margin = float(sorted_values[-1] - sorted_values[-2])
  return spread, top_margin


def action_value_label_phase(example):
  if not has_action_value_labels(example):
    return None
  action_features = example[11] if len(example) > 11 else None
  if action_features is None or example[10] is None:
    return None
  mask = np.array(example[10], dtype=np.float32) > 0
  if not np.any(mask):
    return None
  first_labeled = int(np.argmax(mask))
  adapted = adapt_action_features(action_features, len(example[10]))
  phase_idx = int(np.argmax(adapted[first_labeled, 1:6]))
  return {
      0: "chance",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "terminal",
  }.get(phase_idx, "unknown")


def clear_action_value_label(example):
  if len(example) <= 10 or example[10] is None:
    return example
  items = list(example)
  items[10] = np.zeros_like(example[10], dtype=np.float32)
  return tuple(items)


def filter_action_value_labels_for_training(replay, args):
  min_spread = float(getattr(args, "action_value_filter_min_spread", 0.0))
  min_top_margin = float(
      getattr(args, "action_value_filter_min_top_margin", 0.0)
  )
  phase_spec = str(getattr(args, "action_value_filter_phases", "") or "")
  allowed_phases = {
      part.strip() for part in phase_spec.split(",") if part.strip()
  }
  if min_spread <= 0.0 and min_top_margin <= 0.0 and not allowed_phases:
    return list(replay), None
  kept = 0
  removed = 0
  unlabeled = 0
  phase_removed = {}
  filtered = []
  for example in replay:
    quality = action_value_label_quality(example)
    if quality is None:
      unlabeled += 1
      filtered.append(example)
      continue
    phase = action_value_label_phase(example)
    if allowed_phases and phase not in allowed_phases:
      removed += 1
      key = phase or "unknown"
      phase_removed[key] = phase_removed.get(key, 0) + 1
      filtered.append(clear_action_value_label(example))
      continue
    spread, top_margin = quality
    if spread >= min_spread and top_margin >= min_top_margin:
      kept += 1
      filtered.append(example)
    else:
      removed += 1
      filtered.append(clear_action_value_label(example))
  return filtered, {
      "action_value_filter_min_spread": min_spread,
      "action_value_filter_min_top_margin": min_top_margin,
      "action_value_filter_phases": sorted(allowed_phases),
      "action_value_filter_kept_rows": kept,
      "action_value_filter_removed_rows": removed,
      "action_value_filter_unlabeled_rows": unlabeled,
      "action_value_filter_phase_removed_rows": phase_removed,
  }


def sample_training_batch(replay, batch_size, args):
  value_labeled_fraction = float(
      getattr(args, "action_value_labeled_batch_fraction", 0.0)
  )
  paradox_labeled_fraction = float(
      getattr(args, "action_paradox_labeled_batch_fraction", 0.0)
  )
  use_value_labeled = (
      value_labeled_fraction > 0
      and (
          getattr(args, "action_value_loss_weight", 0.0) > 0
          or getattr(args, "action_value_ranking_loss_weight", 0.0) > 0
      )
  )
  use_paradox_labeled = (
      paradox_labeled_fraction > 0
      and (
          getattr(args, "action_paradox_loss_weight", 0.0) > 0
          or getattr(args, "action_paradox_ranking_loss_weight", 0.0) > 0
      )
  )
  if not use_value_labeled and not use_paradox_labeled:
    return random.sample(replay, batch_size)
  batch = []
  remaining = batch_size

  def add_labeled_rows(labeled, fraction):
    nonlocal remaining
    if remaining <= 0 or not labeled:
      return
    labeled_count = min(
        remaining,
        max(1, int(round(batch_size * min(1.0, fraction)))),
    )
    if labeled_count <= len(labeled):
      batch.extend(random.sample(labeled, labeled_count))
    else:
      batch.extend(random.choices(labeled, k=labeled_count))
    remaining -= labeled_count

  if use_paradox_labeled:
    add_labeled_rows(
        [example for example in replay if has_action_paradox_labels(example)],
        paradox_labeled_fraction,
    )
  if use_value_labeled:
    add_labeled_rows(
        [example for example in replay if has_action_value_labels(example)],
        value_labeled_fraction,
    )
  if remaining == batch_size:
    return random.sample(replay, batch_size)
  if remaining > 0:
    batch.extend(random.sample(replay, remaining))
  random.shuffle(batch)
  return batch


def split_action_value_validation_replay(replay, args):
  replay = list(replay)
  fraction = float(getattr(args, "action_value_validation_fraction", 0.0))
  label_kind = str(
      getattr(args, "loaded_replay_validation_label_kind", "action_value")
  )
  if fraction <= 0.0:
    return replay, [], None
  labeled_indices = [
      idx for idx, example in enumerate(replay)
      if has_loaded_replay_validation_labels(example, args)
  ]
  if len(labeled_indices) < 2:
    return replay, [], {
      "action_value_validation_fraction": fraction,
      "loaded_replay_validation_label_kind": label_kind,
      "action_value_labeled_rows": len(labeled_indices),
      "loaded_replay_validation_labeled_rows": len(labeled_indices),
      "action_value_validation_rows": 0,
      "loaded_replay_validation_rows": 0,
      "action_value_train_rows": len(replay),
      "loaded_replay_train_rows": len(replay),
      "action_value_validation_seed": int(
          getattr(args, "action_value_validation_seed", 20260603)
      ),
      }
  rng = random.Random(int(getattr(args, "action_value_validation_seed", 20260603)))
  validation_count = int(round(len(labeled_indices) * min(1.0, fraction)))
  validation_count = max(1, min(len(labeled_indices) - 1, validation_count))
  validation_indices = set(rng.sample(labeled_indices, validation_count))
  train_replay = [
      example for idx, example in enumerate(replay)
      if idx not in validation_indices
  ]
  validation_replay = [
      example for idx, example in enumerate(replay)
      if idx in validation_indices
  ]
  return train_replay, validation_replay, {
      "action_value_validation_fraction": fraction,
      "loaded_replay_validation_label_kind": label_kind,
      "action_value_labeled_rows": len(labeled_indices),
      "loaded_replay_validation_labeled_rows": len(labeled_indices),
      "action_value_validation_rows": len(validation_replay),
      "loaded_replay_validation_rows": len(validation_replay),
      "action_value_train_rows": len(train_replay),
      "loaded_replay_train_rows": len(train_replay),
      "action_value_validation_seed": int(
          getattr(args, "action_value_validation_seed", 20260603)
      ),
  }


def load_action_value_validation_replay(args):
  replay_spec = str(getattr(args, "action_value_validation_replay", "") or "")
  if not replay_spec.strip():
    return None, None
  label_kind = str(
      getattr(args, "loaded_replay_validation_label_kind", "action_value")
  )
  validation_replay = []
  replay_paths = split_csv(replay_spec)
  replay_counts = {}
  replay_metadata = []
  for replay_path in replay_paths:
    examples, metadata = load_replay(replay_path, return_metadata=True)
    replay_counts[replay_path] = len(examples)
    replay_metadata.append(metadata)
    validation_replay.extend(examples)
  validation_replay, action_value_filter = filter_action_value_labels_for_training(
      validation_replay, args
  )
  labeled_rows = sum(
      1 for example in validation_replay
      if has_loaded_replay_validation_labels(example, args)
  )
  validation_split = {
      "action_value_validation_replay": replay_spec,
      "loaded_replay_validation_label_kind": label_kind,
      "action_value_validation_replay_paths": replay_paths,
      "action_value_validation_replay_counts": replay_counts,
      "action_value_validation_replay_metadata": replay_metadata,
      "action_value_validation_rows": len(validation_replay),
      "action_value_validation_labeled_rows": int(labeled_rows),
      "loaded_replay_validation_rows": len(validation_replay),
      "loaded_replay_validation_labeled_rows": int(labeled_rows),
      "action_value_train_rows_withheld": 0,
  }
  if action_value_filter is not None:
    validation_split["action_value_validation_filter"] = action_value_filter
  return validation_replay, validation_split


def action_value_pairwise_ranking_loss(
    action_value_pred, action_value_target_t, action_value_target_mask_t, args
):
  losses = []
  weights = []
  min_diff = float(getattr(args, "action_value_ranking_min_diff", 1e-6))
  target_scale = max(
      1e-6, float(getattr(args, "action_value_ranking_target_scale", 0.10))
  )
  for row_idx in range(action_value_pred.shape[0]):
    valid_indices = torch.nonzero(
        action_value_target_mask_t[row_idx] > 0, as_tuple=False
    ).flatten()
    if valid_indices.numel() < 2:
      continue
    pred_row = action_value_pred[row_idx, valid_indices]
    target_row = action_value_target_t[row_idx, valid_indices]
    mask_row = action_value_target_mask_t[row_idx, valid_indices]
    pair_i, pair_j = torch.triu_indices(
        valid_indices.numel(),
        valid_indices.numel(),
        offset=1,
        device=action_value_pred.device,
    )
    target_diff = target_row[pair_i] - target_row[pair_j]
    useful = torch.abs(target_diff) >= min_diff
    if not useful.any():
      continue
    pred_diff = pred_row[pair_i] - pred_row[pair_j]
    pair_targets = (target_diff > 0).to(action_value_pred.dtype)
    pair_loss = F.binary_cross_entropy_with_logits(
        pred_diff[useful],
        pair_targets[useful],
        reduction="none",
    )
    confidence_weight = torch.sqrt(mask_row[pair_i] * mask_row[pair_j])
    diff_weight = torch.clamp(torch.abs(target_diff) / target_scale, max=1.0)
    pair_weight = confidence_weight * diff_weight
    losses.append(pair_loss * pair_weight[useful])
    weights.append(pair_weight[useful])
  if not losses:
    return torch.tensor(0.0, device=action_value_pred.device)
  loss_vec = torch.cat(losses)
  weight_vec = torch.cat(weights)
  return loss_vec.sum() / weight_vec.sum().clamp_min(1e-6)


def weighted_masked_action_paradox_bce_loss(
    action_paradox_logits,
    action_target_t,
    action_target_mask_t,
    positive_weight=1.0,
    negative_weight=1.0,
):
  raw_loss = F.binary_cross_entropy_with_logits(
      action_paradox_logits,
      action_target_t,
      reduction="none",
  )
  pos_weight = max(0.0, float(positive_weight))
  neg_weight = max(0.0, float(negative_weight))
  label_weights = torch.where(
      action_target_t > 0.5,
      torch.tensor(
          pos_weight,
          dtype=raw_loss.dtype,
          device=raw_loss.device,
      ),
      torch.tensor(
          neg_weight,
          dtype=raw_loss.dtype,
          device=raw_loss.device,
      ),
  )
  weighted_mask = action_target_mask_t * label_weights
  return (raw_loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def action_paradox_pairwise_ranking_loss(
    action_paradox_logits, action_target_t, action_target_mask_t, args
):
  losses = []
  weights = []
  min_diff = float(getattr(args, "action_paradox_ranking_min_diff", 1e-6))
  target_scale = max(
      1e-6, float(getattr(args, "action_paradox_ranking_target_scale", 1.0))
  )
  for row_idx in range(action_paradox_logits.shape[0]):
    valid_indices = torch.nonzero(
        action_target_mask_t[row_idx] > 0, as_tuple=False
    ).flatten()
    if valid_indices.numel() < 2:
      continue
    pred_row = action_paradox_logits[row_idx, valid_indices]
    target_row = action_target_t[row_idx, valid_indices]
    mask_row = action_target_mask_t[row_idx, valid_indices]
    pair_i, pair_j = torch.triu_indices(
        valid_indices.numel(),
        valid_indices.numel(),
        offset=1,
        device=action_paradox_logits.device,
    )
    target_diff = target_row[pair_i] - target_row[pair_j]
    useful = torch.abs(target_diff) >= min_diff
    if not useful.any():
      continue
    pred_diff = pred_row[pair_i] - pred_row[pair_j]
    pair_targets = (target_diff > 0).to(action_paradox_logits.dtype)
    pair_loss = F.binary_cross_entropy_with_logits(
        pred_diff[useful],
        pair_targets[useful],
        reduction="none",
    )
    confidence_weight = torch.sqrt(mask_row[pair_i] * mask_row[pair_j])
    diff_weight = torch.clamp(torch.abs(target_diff) / target_scale, max=1.0)
    pair_weight = confidence_weight * diff_weight
    losses.append(pair_loss * pair_weight[useful])
    weights.append(pair_weight[useful])
  if not losses:
    return torch.tensor(0.0, device=action_paradox_logits.device)
  loss_vec = torch.cat(losses)
  weight_vec = torch.cat(weights)
  return loss_vec.sum() / weight_vec.sum().clamp_min(1e-6)


def policy_target_pairwise_ranking_loss(logits, policy_t, mask_t, args):
  """Rank the one-hot policy target above legal alternatives."""
  losses = []
  min_target_prob = min(
      1.0, max(0.0, float(getattr(
          args, "policy_target_ranking_min_target_prob", 0.999
      )))
  )
  margin = float(getattr(args, "policy_target_ranking_margin", 0.0))
  max_negatives = max(
      0, int(getattr(args, "policy_target_ranking_max_negatives", 0))
  )
  target_prob, target_action = policy_t.max(dim=1)
  for row_idx in range(logits.shape[0]):
    if target_prob[row_idx] < min_target_prob:
      continue
    legal_indices = torch.nonzero(mask_t[row_idx], as_tuple=False).flatten()
    if legal_indices.numel() < 2:
      continue
    target_idx = target_action[row_idx]
    if not mask_t[row_idx, target_idx]:
      continue
    negative_indices = legal_indices[legal_indices != target_idx]
    if negative_indices.numel() == 0:
      continue
    if max_negatives > 0 and negative_indices.numel() > max_negatives:
      _, hardest_order = torch.topk(
          logits[row_idx, negative_indices],
          k=max_negatives,
      )
      negative_indices = negative_indices[hardest_order]
    target_logit = logits[row_idx, target_idx]
    target_margin = target_logit - logits[row_idx, negative_indices]
    losses.append(F.softplus(margin - target_margin).mean())
  if not losses:
    return torch.tensor(0.0, device=logits.device)
  return torch.stack(losses).mean()


def lane_capacity_pairwise_policy_loss(
    logits,
    action_features_t,
    mask_t,
    args,
):
  """Rank legal policy logits by feature-derived lane-preservation quality."""
  losses = []
  weights = []
  min_diff = float(getattr(args, "lane_capacity_ranking_min_diff", 0.25))
  target_scale = max(
      1e-6, float(getattr(args, "lane_capacity_ranking_target_scale", 2.0))
  )
  token_loss_idx = APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"]
  min_surplus_idx = APPENDED_ACTION_FEATURE_INDEX[
      "legal_z_exit_min_player_lane_surplus_after"
  ]
  damage_idx = APPENDED_ACTION_FEATURE_INDEX["legal_z_exit_lane_surplus_damage"]
  pressure_idx = APPENDED_ACTION_FEATURE_INDEX[
      "legal_z_exit_lane_pressure_player_count_after"
  ]
  token_loss_penalty = float(
      getattr(args, "lane_capacity_ranking_token_loss_penalty", 3.0)
  )
  follow_led_bonus = float(
      getattr(args, "lane_capacity_ranking_follow_led_bonus", 1.0)
  )
  min_surplus_weight = float(
      getattr(args, "lane_capacity_ranking_min_surplus_weight", 1.0)
  )
  damage_penalty = float(
      getattr(args, "lane_capacity_ranking_damage_penalty", 0.75)
  )
  pressure_penalty = float(
      getattr(args, "lane_capacity_ranking_pressure_penalty", 0.25)
  )
  require_led_choice = bool(
      getattr(args, "lane_capacity_ranking_require_led_choice", True)
  )
  features = action_features_t
  lane_score = (
      follow_led_bonus * features[:, :, ACTION_FEATURE_FOLLOWS_LED_INDEX]
      - token_loss_penalty * features[:, :, token_loss_idx]
      + min_surplus_weight * features[:, :, min_surplus_idx]
      - damage_penalty * features[:, :, damage_idx]
      - pressure_penalty * features[:, :, pressure_idx]
  )
  for row_idx in range(logits.shape[0]):
    valid_indices = torch.nonzero(mask_t[row_idx], as_tuple=False).flatten()
    if valid_indices.numel() < 2:
      continue
    feature_row = features[row_idx, valid_indices]
    if require_led_choice:
      has_token_loss = (feature_row[:, token_loss_idx] > 0.5).any()
      has_follow_led = (
          feature_row[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
      ).any()
      if not (has_token_loss and has_follow_led):
        continue
    score_row = lane_score[row_idx, valid_indices]
    pair_i, pair_j = torch.triu_indices(
        valid_indices.numel(),
        valid_indices.numel(),
        offset=1,
        device=logits.device,
    )
    target_diff = score_row[pair_i] - score_row[pair_j]
    useful = torch.abs(target_diff) >= min_diff
    if not useful.any():
      continue
    pred_row = logits[row_idx, valid_indices]
    pred_diff = pred_row[pair_i] - pred_row[pair_j]
    pair_targets = (target_diff > 0).to(logits.dtype)
    pair_loss = F.binary_cross_entropy_with_logits(
        pred_diff[useful],
        pair_targets[useful],
        reduction="none",
    )
    pair_weight = torch.clamp(torch.abs(target_diff) / target_scale, max=1.0)
    losses.append(pair_loss * pair_weight[useful])
    weights.append(pair_weight[useful])
  if not losses:
    return torch.tensor(0.0, device=logits.device)
  loss_vec = torch.cat(losses)
  weight_vec = torch.cat(weights)
  return loss_vec.sum() / weight_vec.sum().clamp_min(1e-6)


def selected_terminal_paradox_targets(
    paradox_t,
    player_t,
    batch_indices,
    args,
    terminal_action_paradox_t=None,
    terminal_action_paradox_mask_t=None,
):
  source_t = paradox_t[batch_indices]
  if (
      terminal_action_paradox_t is not None
      and terminal_action_paradox_mask_t is not None
  ):
    sidecar_t = terminal_action_paradox_t[batch_indices]
    sidecar_valid_t = terminal_action_paradox_mask_t[batch_indices].view(-1, 1)
    source_t = torch.where(sidecar_valid_t > 0.5, sidecar_t, source_t)
  scope = str(
      getattr(args, "action_paradox_terminal_fallback_scope", "acting")
      or "acting"
  )
  if scope == "any":
    return (source_t > 0.5).any(dim=1).to(source_t.dtype)
  safe_player_t = torch.clamp(player_t, 0, source_t.shape[1] - 1)
  return source_t[
      torch.arange(source_t.shape[0], device=source_t.device),
      safe_player_t,
  ]


def train_steps(model, optimizer, replay, args, device, anchor_model=None):
  if args.train_steps <= 0:
    return None
  if len(replay) < args.batch_size:
    return None
  losses = []
  for _ in range(args.train_steps):
    batch = sample_training_batch(replay, args.batch_size, args)
    obs = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    policies = [item[2] for item in batch]
    values = [item[3] for item in batch]
    paradoxes = [
        item[4] if len(item) > 4 else np.zeros(args.players, dtype=np.float32)
        for item in batch
    ]
    actions = [item[5] if len(item) > 5 else -1 for item in batch]
    players = [item[6] if len(item) > 6 else -1 for item in batch]
    action_target_vectors = [
        (
            item[7]
            if len(item) > 8 and item[7] is not None
            else np.zeros_like(masks[0], dtype=np.float32)
        )
        for item in batch
    ]
    action_target_masks = [
        (
            item[8]
            if len(item) > 8 and item[8] is not None
            else np.zeros_like(masks[0], dtype=np.float32)
        )
        for item in batch
    ]
    action_value_target_vectors = [
        (
            item[9]
            if len(item) > 10 and item[9] is not None
            else np.zeros_like(masks[0], dtype=np.float32)
        )
        for item in batch
    ]
    action_value_target_masks = [
        (
            item[10]
            if len(item) > 10 and item[10] is not None
            else np.zeros_like(masks[0], dtype=np.float32)
        )
        for item in batch
    ]
    action_features = [
        adapt_action_features(
            item[11] if len(item) > 11 else None,
            len(item[1]),
        )
        for item in batch
    ]
    terminal_action_paradox_targets = []
    terminal_action_paradox_target_masks = []
    for item in batch:
      if len(item) > 12 and item[12] is not None:
        terminal_action_paradox_targets.append(
            adapt_player_target_vector(item[12], args.players)
        )
        terminal_action_paradox_target_masks.append(1.0)
      else:
        terminal_action_paradox_targets.append(
            np.zeros(args.players, dtype=np.float32)
        )
        terminal_action_paradox_target_masks.append(0.0)
    obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
    obs_t = adapt_observation_batch(obs_t, model_input_size(model))
    mask_t = torch.tensor(np.array(masks), dtype=torch.bool, device=device)
    policy_t = torch.tensor(np.array(policies), dtype=torch.float32, device=device)
    value_t = torch.tensor(np.array(values), dtype=torch.float32, device=device)
    paradox_t = torch.tensor(np.array(paradoxes), dtype=torch.float32, device=device)
    action_t = torch.tensor(actions, dtype=torch.long, device=device)
    player_t = torch.tensor(players, dtype=torch.long, device=device)
    action_target_t = torch.tensor(
        np.array(action_target_vectors), dtype=torch.float32, device=device
    )
    action_target_mask_t = torch.tensor(
        np.array(action_target_masks), dtype=torch.float32, device=device
    )
    action_value_target_t = torch.tensor(
        np.array(action_value_target_vectors), dtype=torch.float32, device=device
    )
    action_value_target_mask_t = torch.tensor(
        np.array(action_value_target_masks), dtype=torch.float32, device=device
    )
    action_features_t = torch.tensor(
        np.array(action_features), dtype=torch.float32, device=device
    )
    terminal_action_paradox_t = torch.tensor(
        np.array(terminal_action_paradox_targets),
        dtype=torch.float32,
        device=device,
    )
    terminal_action_paradox_mask_t = torch.tensor(
        np.array(terminal_action_paradox_target_masks),
        dtype=torch.float32,
        device=device,
    )

    use_aux = (
        args.paradox_loss_weight > 0
        or args.action_paradox_loss_weight > 0
        or getattr(args, "action_paradox_ranking_loss_weight", 0.0) > 0
        or args.action_value_loss_weight > 0
        or getattr(args, "action_value_ranking_loss_weight", 0.0) > 0
    )
    if use_aux and (
        args.action_paradox_loss_weight > 0
        or getattr(args, "action_paradox_ranking_loss_weight", 0.0) > 0
        or args.action_value_loss_weight > 0
        or getattr(args, "action_value_ranking_loss_weight", 0.0) > 0
    ):
      (
          logits,
          value_pred,
          paradox_logits,
          action_paradox_logits,
          action_value_pred,
      ) = (
          model.forward_with_all_aux(obs_t, action_features_t)
      )
    elif use_aux:
      logits, value_pred, paradox_logits = model.forward_with_aux(
          obs_t, action_features_t
      )
      action_paradox_logits = None
      action_value_pred = None
    else:
      logits, value_pred = model(obs_t, action_features_t)
      paradox_logits = None
      action_paradox_logits = None
      action_value_pred = None
    logits = logits.masked_fill(~mask_t, -1e9)
    log_probs = F.log_softmax(logits, dim=1)
    per_example_policy_loss = -(policy_t * log_probs).sum(dim=1)
    policy_weights = policy_target_action_type_weights(
        action_features_t, policy_t, args, device
    )
    policy_weights = policy_weights * policy_target_bucket_weights(
        action_features_t, policy_t, args, device
    )
    policy_loss = (
        per_example_policy_loss * policy_weights
    ).sum() / policy_weights.sum().clamp_min(1e-6)
    prediction_hit_loss = torch.tensor(0.0, device=device)
    probs = None
    if getattr(args, "prediction_hit_policy_loss_weight", 0.0) > 0:
      hit_idx = APPENDED_ACTION_FEATURE_INDEX["hits_prediction"]
      hit_mask = (action_features_t[:, :, hit_idx] > 0.5) & mask_t
      valid_hit_state = hit_mask.any(dim=1)
      if valid_hit_state.any():
        probs = log_probs.exp()
        hit_mass = (probs * hit_mask.to(probs.dtype)).sum(dim=1)
        target_mass = float(
            getattr(args, "prediction_hit_policy_target_mass", 0.45)
        )
        target_mass = min(0.999, max(0.0, target_mass))
        shortfall = F.relu(target_mass - hit_mass[valid_hit_state])
        prediction_hit_loss = torch.mean(shortfall * shortfall)
    future_hit_loss = torch.tensor(0.0, device=device)
    if getattr(args, "future_hit_policy_loss_weight", 0.0) > 0:
      future_hit_idx = APPENDED_ACTION_FEATURE_INDEX["hit_with_future_tricks"]
      future_hit_mask = (
          action_features_t[:, :, future_hit_idx] > 0.5
      ) & mask_t
      valid_future_hit_state = future_hit_mask.any(dim=1)
      if valid_future_hit_state.any():
        if probs is None:
          probs = log_probs.exp()
        future_hit_mass = (
            probs * future_hit_mask.to(probs.dtype)
        ).sum(dim=1)
        max_mass = float(getattr(args, "future_hit_policy_max_mass", 0.25))
        max_mass = min(0.999, max(0.0, max_mass))
        excess = F.relu(future_hit_mass[valid_future_hit_state] - max_mass)
        future_hit_loss = torch.mean(excess * excess)
    led_token_loss = torch.tensor(0.0, device=device)
    if getattr(args, "led_token_loss_policy_loss_weight", 0.0) > 0:
      led_loss_idx = APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"]
      led_loss_mask = (
          action_features_t[:, :, led_loss_idx] > 0.5
      ) & mask_t
      follow_led_mask = (
          action_features_t[:, :, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
      ) & mask_t
      valid_led_loss_state = led_loss_mask.any(dim=1) & follow_led_mask.any(dim=1)
      if valid_led_loss_state.any():
        if probs is None:
          probs = log_probs.exp()
        led_loss_mass = (
            probs * led_loss_mask.to(probs.dtype)
        ).sum(dim=1)
        max_mass = float(
            getattr(args, "led_token_loss_policy_max_mass", 0.05)
        )
        max_mass = min(0.999, max(0.0, max_mass))
        excess = F.relu(led_loss_mass[valid_led_loss_state] - max_mass)
        led_token_loss = torch.mean(excess * excess)
    policy_target_ranking_loss = torch.tensor(0.0, device=device)
    if getattr(args, "policy_target_ranking_loss_weight", 0.0) > 0:
      policy_target_ranking_loss = policy_target_pairwise_ranking_loss(
          logits,
          policy_t,
          mask_t,
          args,
      )
    lane_capacity_ranking_loss = torch.tensor(0.0, device=device)
    if getattr(args, "lane_capacity_ranking_policy_loss_weight", 0.0) > 0:
      lane_capacity_ranking_loss = lane_capacity_pairwise_policy_loss(
          logits,
          action_features_t,
          mask_t,
          args,
      )
    dangerous_future_hit_loss = torch.tensor(0.0, device=device)
    if getattr(args, "dangerous_future_hit_policy_loss_weight", 0.0) > 0:
      future_hit_idx = APPENDED_ACTION_FEATURE_INDEX["hit_with_future_tricks"]
      low_legal_ratio_idx = APPENDED_ACTION_FEATURE_INDEX[
          "post_hit_low_legal_lead_ratio"
      ]
      survival_margin_idx = APPENDED_ACTION_FEATURE_INDEX[
          "post_hit_low_card_survival_margin"
      ]
      forced_pressure_idx = APPENDED_ACTION_FEATURE_INDEX[
          "post_hit_forced_card_pressure"
      ]
      future_hit_mask = (
          action_features_t[:, :, future_hit_idx] > 0.5
      ) & mask_t
      low_ratio_risky = (
          action_features_t[:, :, low_legal_ratio_idx]
          <= float(getattr(
              args, "dangerous_future_hit_low_legal_ratio_threshold", 0.25
          ))
      )
      survival_risky = (
          action_features_t[:, :, survival_margin_idx]
          < float(getattr(
              args, "dangerous_future_hit_survival_margin_threshold", 0.0
          ))
      )
      pressure_risky = (
          action_features_t[:, :, forced_pressure_idx]
          > float(getattr(
              args, "dangerous_future_hit_forced_pressure_threshold", 0.75
          ))
      )
      dangerous_future_hit_mask = future_hit_mask & (
          low_ratio_risky | survival_risky | pressure_risky
      )
      valid_dangerous_state = dangerous_future_hit_mask.any(dim=1)
      if valid_dangerous_state.any():
        if probs is None:
          probs = log_probs.exp()
        dangerous_mass = (
            probs * dangerous_future_hit_mask.to(probs.dtype)
        ).sum(dim=1)
        max_mass = float(
            getattr(args, "dangerous_future_hit_policy_max_mass", 0.25)
        )
        max_mass = min(0.999, max(0.0, max_mass))
        excess = F.relu(dangerous_mass[valid_dangerous_state] - max_mass)
        dangerous_future_hit_loss = torch.mean(excess * excess)
    if args.value_loss_mode == "acting":
      value_valid = (player_t >= 0) & (player_t < args.players)
      if value_valid.any():
        value_batch_indices = torch.arange(len(batch), device=device)[value_valid]
        value_loss = F.mse_loss(
            value_pred[value_batch_indices, player_t[value_valid]],
            value_t[value_batch_indices, player_t[value_valid]],
        )
      else:
        value_loss = F.mse_loss(value_pred, value_t)
    else:
      value_loss = F.mse_loss(value_pred, value_t)
    paradox_loss = torch.tensor(0.0, device=device)
    if paradox_logits is not None:
      paradox_loss = F.binary_cross_entropy_with_logits(
          paradox_logits, paradox_t
      )
    action_paradox_loss = torch.tensor(0.0, device=device)
    action_paradox_ranking_loss = torch.tensor(0.0, device=device)
    valid_action = (
        (action_t >= 0)
        & (action_t < logits.shape[1])
        & (player_t >= 0)
        & (player_t < args.players)
    )
    if action_paradox_logits is not None and action_target_mask_t.sum() > 0:
      action_paradox_loss = weighted_masked_action_paradox_bce_loss(
          action_paradox_logits,
          action_target_t,
          action_target_mask_t,
          positive_weight=getattr(args, "action_paradox_positive_weight", 1.0),
          negative_weight=getattr(args, "action_paradox_negative_weight", 1.0),
      )
      if getattr(args, "action_paradox_ranking_loss_weight", 0.0) > 0:
        action_paradox_ranking_loss = action_paradox_pairwise_ranking_loss(
            action_paradox_logits,
            action_target_t,
            action_target_mask_t,
            args,
        )
    elif (
        action_paradox_logits is not None
        and valid_action.any()
        and getattr(args, "action_paradox_terminal_fallback", False)
    ):
      batch_indices = torch.arange(len(batch), device=device)[valid_action]
      selected_logits = action_paradox_logits[batch_indices, action_t[valid_action]]
      selected_targets = selected_terminal_paradox_targets(
          paradox_t,
          player_t[valid_action],
          batch_indices,
          args,
          terminal_action_paradox_t=terminal_action_paradox_t,
          terminal_action_paradox_mask_t=terminal_action_paradox_mask_t,
      )
      action_paradox_loss = F.binary_cross_entropy_with_logits(
          selected_logits, selected_targets
      )
    action_value_loss = torch.tensor(0.0, device=device)
    action_value_ranking_loss = torch.tensor(0.0, device=device)
    if action_value_pred is not None and action_value_target_mask_t.sum() > 0:
      raw_loss = F.mse_loss(
          action_value_pred,
          action_value_target_t,
          reduction="none",
      )
      action_value_loss = (
          raw_loss * action_value_target_mask_t
      ).sum() / action_value_target_mask_t.sum().clamp_min(1.0)
      if getattr(args, "action_value_ranking_loss_weight", 0.0) > 0:
        action_value_ranking_loss = action_value_pairwise_ranking_loss(
            action_value_pred,
            action_value_target_t,
            action_value_target_mask_t,
            args,
        )
    elif (
        action_value_pred is not None
        and valid_action.any()
        and getattr(args, "action_value_terminal_fallback", False)
    ):
      batch_indices = torch.arange(len(batch), device=device)[valid_action]
      selected_values = action_value_pred[batch_indices, action_t[valid_action]]
      selected_targets = value_t[batch_indices, player_t[valid_action]]
      action_value_loss = F.mse_loss(selected_values, selected_targets)
    anchor_loss = torch.tensor(0.0, device=device)
    anchor_top_action_loss = torch.tensor(0.0, device=device)
    if (
        anchor_model is not None
        and (
            args.anchor_kl_weight > 0
            or getattr(args, "anchor_top_action_loss_weight", 0.0) > 0
        )
    ):
      with torch.no_grad():
        anchor_obs_t = adapt_observation_batch(
            obs_t, model_input_size(anchor_model)
        )
        anchor_logits, _ = anchor_model(anchor_obs_t, action_features_t)
        anchor_logits = anchor_logits.masked_fill(~mask_t, -1e9)
        anchor_log_probs = F.log_softmax(anchor_logits, dim=1)
        anchor_probs = anchor_log_probs.exp()
      if args.anchor_kl_weight > 0:
        anchor_loss = (
            anchor_probs * (anchor_log_probs - log_probs)
        ).sum(dim=1).mean()
      if getattr(args, "anchor_top_action_loss_weight", 0.0) > 0:
        anchor_top_prob, anchor_top_action = anchor_probs.max(dim=1)
        min_prob = float(getattr(args, "anchor_top_action_min_prob", 0.0))
        min_prob = min(0.999, max(0.0, min_prob))
        confident_anchor = anchor_top_prob >= min_prob
        if confident_anchor.any():
          top_action_nll = -log_probs[
              torch.arange(log_probs.shape[0], device=device),
              anchor_top_action,
          ]
          anchor_top_action_loss = top_action_nll[confident_anchor].mean()
    loss = (
        args.policy_loss_weight * policy_loss
        + args.value_loss_weight * value_loss
        + args.paradox_loss_weight * paradox_loss
        + args.action_paradox_loss_weight * action_paradox_loss
        + getattr(args, "action_paradox_ranking_loss_weight", 0.0) *
        action_paradox_ranking_loss
        + args.action_value_loss_weight * action_value_loss
        + getattr(args, "action_value_ranking_loss_weight", 0.0) *
        action_value_ranking_loss
        + args.prediction_hit_policy_loss_weight * prediction_hit_loss
        + args.future_hit_policy_loss_weight * future_hit_loss
        + getattr(args, "led_token_loss_policy_loss_weight", 0.0) *
        led_token_loss
        + getattr(args, "policy_target_ranking_loss_weight", 0.0) *
        policy_target_ranking_loss
        + getattr(args, "lane_capacity_ranking_policy_loss_weight", 0.0) *
        lane_capacity_ranking_loss
        + args.dangerous_future_hit_policy_loss_weight *
        dangerous_future_hit_loss
        + args.anchor_kl_weight * anchor_loss
        + getattr(args, "anchor_top_action_loss_weight", 0.0) *
        anchor_top_action_loss
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append((
        float(loss.detach()),
        float(policy_loss.detach()),
        float(value_loss.detach()),
        float(anchor_loss.detach()),
        float(paradox_loss.detach()),
        float(action_paradox_loss.detach()),
        float(action_paradox_ranking_loss.detach()),
        float(action_value_loss.detach()),
        float(action_value_ranking_loss.detach()),
        float(prediction_hit_loss.detach()),
        float(future_hit_loss.detach()),
        float(led_token_loss.detach()),
        float(dangerous_future_hit_loss.detach()),
        float(anchor_top_action_loss.detach()),
        float(policy_target_ranking_loss.detach()),
        float(lane_capacity_ranking_loss.detach()),
    ))
  return np.mean(losses, axis=0).tolist()


def mean_loss(losses):
  valid = [loss for loss in losses if loss is not None]
  if not valid:
    return None
  return np.mean(np.array(valid, dtype=np.float32), axis=0).tolist()


def loaded_replay_report_row(
    model, anchor_model, replay, args, loaded_args, device, loss, trained_steps,
    eval_result=None, checkpoint_path=None, iteration="loaded_replay_train",
    validation_replay=None, validation_split=None,
):
  row = {
      "iteration": iteration,
      "loaded_replay_train_steps": trained_steps,
      "load_replay": args.load_replay,
      "replay_size": len(replay),
      "loss": loss,
  }
  if eval_result is not None:
    row["eval"] = eval_result
  if checkpoint_path is not None:
    row["checkpoint"] = str(checkpoint_path)
  if validation_split is not None:
    row["validation_split"] = validation_split
  maybe_add_value_prediction_report(row, model, replay, loaded_args, device)
  maybe_add_action_report(row, model, replay, loaded_args, device)
  maybe_add_action_value_report(row, model, replay, loaded_args, device)
  q_policy_report = q_label_policy_report(
      model, list(replay), loaded_args, device
  )
  if q_policy_report is not None:
    row["q_label_policy_report"] = q_policy_report
  if validation_replay:
    maybe_add_value_prediction_report(
        row,
        model,
        list(validation_replay),
        loaded_args,
        device,
        name="value_prediction_validation_report",
    )
    validation_report = action_value_report(
        model, list(validation_replay), loaded_args, device
    )
    if validation_report is not None:
      row["action_value_validation_report"] = validation_report
    action_paradox_validation_report = action_paradox_report(
        model, list(validation_replay), loaded_args, device
    )
    if action_paradox_validation_report is not None:
      row["action_paradox_validation_report"] = action_paradox_validation_report
    q_policy_validation_report = q_label_policy_report(
        model, list(validation_replay), loaded_args, device
    )
    if q_policy_validation_report is not None:
      row["q_label_policy_validation_report"] = q_policy_validation_report
    maybe_add_policy_target_report(
        row,
        model,
        list(validation_replay),
        loaded_args,
        device,
        name="loaded_replay_policy_target_validation_report",
    )
    maybe_add_anchor_policy_target_report(
        row,
        anchor_model,
        list(validation_replay),
        loaded_args,
        device,
        name="anchor_policy_target_validation_report",
    )
  maybe_add_prediction_hit_report(row, model, replay, loaded_args, device)
  maybe_add_policy_target_report(
      row,
      model,
      replay,
      loaded_args,
      device,
      name="loaded_replay_policy_target_report",
  )
  maybe_add_anchor_policy_target_report(
      row,
      anchor_model,
      replay,
      loaded_args,
      device,
      name="anchor_policy_target_report",
  )
  maybe_add_anchor_policy_report(
      row, model, anchor_model, replay, loaded_args, device
  )
  return row


def loaded_replay_validation_score(row, args):
  metric = getattr(args, "loaded_replay_best_metric", "validation_top1")
  if metric.startswith("value_validation_"):
    report = row.get("value_prediction_validation_report")
    if not report:
      return None
    scope_metric = metric.removeprefix("value_validation_")
    if scope_metric.startswith("acting_"):
      scope = "acting_player"
      value_name = scope_metric.removeprefix("acting_")
    elif scope_metric.startswith("all_"):
      scope = "all_players"
      value_name = scope_metric.removeprefix("all_")
    else:
      return None
    value_report = report.get(scope)
    if not value_report:
      return None
    value = value_report.get(value_name)
    if value is None:
      return None
    value = float(value)
    if value_name in ("brier", "rmse", "mae"):
      value = -value
    return value
  if metric.startswith("policy_validation_"):
    report = row.get("loaded_replay_policy_target_validation_report")
    if not report:
      return None
    if metric == "policy_validation_top1":
      return report.get("top1_rate")
    if metric == "policy_validation_cross_entropy":
      value = report.get("cross_entropy")
      return None if value is None else -float(value)
    return None
  if metric.startswith("action_paradox_validation_"):
    report = row.get("action_paradox_validation_report")
    if not report:
      return None
    value_name = metric.removeprefix("action_paradox_validation_")
    value = report.get(value_name)
    if value is None:
      return None
    value = float(value)
    if value_name == "brier":
      value = -value
    return value
  if metric.startswith("q_policy_"):
    report = row.get("q_label_policy_validation_report")
  else:
    report = row.get("action_value_validation_report")
  if not report:
    return None
  if metric in ("validation_top1", "q_policy_validation_top1"):
    value = report.get("top1_rate")
  elif metric in ("validation_corr", "q_policy_validation_corr"):
    value = report.get("corr")
  elif metric in ("validation_mean_regret", "q_policy_validation_mean_regret"):
    value = report.get("mean_regret")
    if value is not None:
      value = -float(value)
  else:
    value = None
  if value is None:
    return None
  return float(value)


def replay_metadata_from_args(args):
  return {
      "counterfactual_full_match_rollout": bool(
          getattr(args, "counterfactual_full_match_rollout", False)
      ),
      "counterfactual_rollout_max_plies": int(
          getattr(args, "counterfactual_rollout_max_plies", 0)
      ),
      "counterfactual_belief_source": str(
          getattr(args, "counterfactual_belief_source", "infostate")
      ),
      "counterfactual_belief_samples": int(
          getattr(args, "counterfactual_belief_samples", 1)
      ),
      "counterfactual_action_value_rollouts": int(
          getattr(args, "counterfactual_action_value_rollouts", 0)
      ),
      "counterfactual_action_value_objective": str(
          getattr(args, "counterfactual_action_value_objective", "score")
      ),
      "counterfactual_action_survival_truncated_value": float(
          getattr(args, "counterfactual_action_survival_truncated_value", 0.0)
      ),
      "counterfactual_action_rollouts": int(
          getattr(args, "counterfactual_action_rollouts", 0)
      ),
      "counterfactual_action_paradox_scope": str(
          getattr(args, "counterfactual_action_paradox_scope", "acting")
      ),
      "counterfactual_action_paradox_target_mode": str(
          getattr(args, "counterfactual_action_paradox_target_mode", "binary")
      ),
      "counterfactual_action_paradox_survival_weight": float(
          getattr(args, "counterfactual_action_paradox_survival_weight", 0.5)
      ),
      "action_paradox_terminal_fallback_scope": str(
          getattr(args, "action_paradox_terminal_fallback_scope", "acting")
      ),
      "counterfactual_rollout_learner_bot": str(
          getattr(args, "counterfactual_rollout_learner_bot", "")
      ),
      "counterfactual_paradox_policy_target_weight": float(
          getattr(args, "counterfactual_paradox_policy_target_weight", 0.0)
      ),
      "counterfactual_paradox_policy_target_temperature": float(
          getattr(args, "counterfactual_paradox_policy_target_temperature", 0.08)
      ),
      "counterfactual_paradox_policy_target_min_spread": float(
          getattr(args, "counterfactual_paradox_policy_target_min_spread", 0.0)
      ),
      "counterfactual_action_label_phases": str(
          getattr(args, "counterfactual_action_label_phases", "")
      ),
      "rollout_select_teacher_rollouts": int(
          getattr(args, "rollout_select_teacher_rollouts", 1)
      ),
      "rollout_select_teacher_min_paradox_improvement": float(
          getattr(args, "rollout_select_teacher_min_paradox_improvement", 1e-6)
      ),
      "rollout_select_teacher_min_score_margin": float(
          getattr(args, "rollout_select_teacher_min_score_margin", 0.0)
      ),
      "rollout_select_teacher_continuation_role": str(
          getattr(args, "rollout_select_teacher_continuation_role", "learner")
      ),
  }


def _iter_replay_metadata_dicts(metadata):
  if not metadata:
    return
  if isinstance(metadata, dict):
    yield metadata
    for value in metadata.values():
      if isinstance(value, (dict, list, tuple)):
        yield from _iter_replay_metadata_dicts(value)
    return
  if isinstance(metadata, (list, tuple)):
    for item in metadata:
      yield from _iter_replay_metadata_dicts(item)


def apply_replay_metadata_to_args(args, metadata):
  metadata_items = list(_iter_replay_metadata_dicts(metadata))
  if not metadata_items:
    return args
  copied = argparse.Namespace(**vars(args))
  if any(
      bool(item.get("counterfactual_full_match_rollout", False))
      for item in metadata_items
  ):
    copied.counterfactual_full_match_rollout = True
  value_objectives = {
      str(item.get("counterfactual_action_value_objective", "score") or "score")
      for item in metadata_items
      if item.get("counterfactual_action_value_objective") is not None
  }
  if value_objectives == {"survival"}:
    copied.counterfactual_action_value_objective = "survival"
  return copied


def save_training_artifacts(model, args, metrics, replay, checkpoint_path, out_dir):
  torch.save({
      "model": model.state_dict(),
      "args": vars(args),
      "metrics": metrics,
  }, checkpoint_path)
  (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
  save_replay(
      list(replay), out_dir / "replay_latest.npz",
      metadata=replay_metadata_from_args(args),
  )


def binary_auc(preds, targets):
  preds = np.array(preds, dtype=np.float64)
  targets = np.array(targets, dtype=np.int32)
  positives = int(targets.sum())
  negatives = int(len(targets) - positives)
  if positives == 0 or negatives == 0:
    return None
  order = np.argsort(preds)
  ranks = np.empty(len(preds), dtype=np.float64)
  start = 0
  while start < len(preds):
    end = start + 1
    while end < len(preds) and preds[order[end]] == preds[order[start]]:
      end += 1
    avg_rank = (start + 1 + end) / 2.0
    ranks[order[start:end]] = avg_rank
    start = end
  pos_rank_sum = float(ranks[targets == 1].sum())
  auc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (
      positives * negatives
  )
  return float(auc)


def value_prediction_summary(preds, targets, bucket_count=10):
  preds = np.array(preds, dtype=np.float64).reshape(-1)
  targets = np.array(targets, dtype=np.float64).reshape(-1)
  if len(preds) == 0:
    return None
  rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
  mae = float(np.mean(np.abs(preds - targets)))
  corr = None
  if (
      len(preds) > 1
      and float(np.std(preds)) > 0.0
      and float(np.std(targets)) > 0.0
  ):
    corr_candidate = float(np.corrcoef(preds, targets)[0, 1])
    if math.isfinite(corr_candidate):
      corr = corr_candidate
  binary_targets = np.isin(targets, [-1.0, 0.0, 1.0])
  labels = (targets > 0.0).astype(np.float64)
  probs = np.clip((preds + 1.0) / 2.0, 0.0, 1.0)
  auc = binary_auc(probs, labels.astype(np.int32)) if np.all(binary_targets) else None
  brier = float(np.mean((probs - labels) ** 2)) if np.all(binary_targets) else None
  buckets = []
  if np.all(binary_targets):
    for bucket_idx in range(max(1, int(bucket_count))):
      low = bucket_idx / float(max(1, int(bucket_count)))
      high = (bucket_idx + 1) / float(max(1, int(bucket_count)))
      if bucket_idx == int(bucket_count) - 1:
        mask = (probs >= low) & (probs <= high)
      else:
        mask = (probs >= low) & (probs < high)
      if not np.any(mask):
        continue
      buckets.append({
          "low": round(low, 4),
          "high": round(high, 4),
          "count": int(np.sum(mask)),
          "mean_pred_prob": round(float(np.mean(probs[mask])), 4),
          "empirical_positive_rate": round(float(np.mean(labels[mask])), 4),
      })
  return {
      "count": int(len(preds)),
      "target_mean": round(float(np.mean(targets)), 4),
      "pred_mean": round(float(np.mean(preds)), 4),
      "rmse": round(rmse, 4),
      "mae": round(mae, 4),
      "corr": None if corr is None else round(corr, 4),
      "auc": None if auc is None else round(float(auc), 4),
      "brier": None if brier is None else round(float(brier), 4),
      "positive_rate": (
          None if not np.all(binary_targets) else round(float(np.mean(labels)), 4)
      ),
      "reliability_buckets": buckets,
  }


def value_prediction_report(model, replay, args, device, max_examples=5000):
  if not replay:
    return None
  valid = []
  for item in replay:
    if len(item) < 4 or item[3] is None:
      continue
    valid.append(item)
  if not valid:
    return None
  if len(valid) > max_examples:
    valid = random.sample(valid, max_examples)
  obs = [item[0] for item in valid]
  targets = np.array([item[3] for item in valid], dtype=np.float32)
  players = np.array([
      item[6] if len(item) > 6 else -1 for item in valid
  ], dtype=np.int64)
  action_features = np.array([
      adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
      for item in valid
  ], dtype=np.float32)
  pred_chunks = []
  report_batch_size = (
      128
      if getattr(model, "arch", "mlp") in ACTION_CONDITIONED_ARCHS
      else 512
  )
  obs_array = np.array(obs, dtype=np.float32)
  with torch.no_grad():
    for start in range(0, len(valid), report_batch_size):
      end = min(len(valid), start + report_batch_size)
      obs_t = torch.tensor(
          obs_array[start:end], dtype=torch.float32, device=device
      )
      obs_t = adapt_observation_batch(obs_t, model_input_size(model))
      action_features_t = torch.tensor(
          action_features[start:end], dtype=torch.float32, device=device
      )
      _, values = model(obs_t, action_features_t)
      pred_chunks.append(values.detach().cpu().numpy())
  preds = np.concatenate(pred_chunks, axis=0)
  report = {
      "kind": "state_value",
      "rows": int(len(valid)),
      "all_players": value_prediction_summary(
          preds.reshape(-1), targets.reshape(-1)
      ),
  }
  valid_players = (
      (players >= 0)
      & (players < preds.shape[1])
      & (players < targets.shape[1])
  )
  if np.any(valid_players):
    row_indices = np.arange(len(valid))[valid_players]
    acting_preds = preds[row_indices, players[valid_players]]
    acting_targets = targets[row_indices, players[valid_players]]
    report["acting_player"] = value_prediction_summary(
        acting_preds, acting_targets
    )
  return report


def maybe_add_value_prediction_report(
    row, model, replay, args, device, name="value_prediction_report"
):
  report = value_prediction_report(model, list(replay), args, device)
  if report is not None:
    row[name] = report


def action_paradox_report(model, replay, args, device, max_examples=5000):
  if not replay or not hasattr(model, "forward_with_action_aux"):
    return None
  full_valid = [
      item for item in replay
      if (
          len(item) > 8
          and item[7] is not None
          and item[8] is not None
          and float(np.sum(item[8])) > 0
      )
  ]
  if full_valid:
    if len(full_valid) > max_examples:
      full_valid = random.sample(full_valid, max_examples)
    obs = [item[0] for item in full_valid]
    target_vectors = np.array([item[7] for item in full_valid], dtype=np.float32)
    target_masks = np.array([item[8] for item in full_valid], dtype=np.float32)
    action_features = np.array([
        adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
        for item in full_valid
    ], dtype=np.float32)
    obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
    obs_t = adapt_observation_batch(obs_t, model_input_size(model))
    action_features_t = torch.tensor(
        action_features, dtype=torch.float32, device=device
    )
    with torch.no_grad():
      _, _, _, action_logits = model.forward_with_action_aux(
          obs_t, action_features_t
      )
      pred_matrix = torch.sigmoid(action_logits).detach().cpu().numpy()
    selected = target_masks > 0.5
    preds = pred_matrix[selected]
    targets = target_vectors[selected]
    if len(preds) == 0:
      return None
    report_kind = "counterfactual_legal_actions"
  elif getattr(args, "action_paradox_terminal_fallback", False):
    report_kind = "selected_actions"
    valid = [
        item for item in replay
        if (
            len(item) > 6
            and 0 <= int(item[5]) < len(item[1])
            and 0 <= int(item[6]) < args.players
        )
    ]
    if not valid:
      return None
    if len(valid) > max_examples:
      valid = random.sample(valid, max_examples)
    obs = [item[0] for item in valid]
    actions = [int(item[5]) for item in valid]
    players = [int(item[6]) for item in valid]
    paradoxes = [item[4] for item in valid]
    terminal_action_paradox_targets = []
    terminal_action_paradox_target_masks = []
    for item in valid:
      if len(item) > 12 and item[12] is not None:
        terminal_action_paradox_targets.append(
            adapt_player_target_vector(item[12], args.players)
        )
        terminal_action_paradox_target_masks.append(1.0)
      else:
        terminal_action_paradox_targets.append(
            np.zeros(args.players, dtype=np.float32)
        )
        terminal_action_paradox_target_masks.append(0.0)
    obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
    obs_t = adapt_observation_batch(obs_t, model_input_size(model))
    action_features = np.array([
        adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
        for item in valid
    ], dtype=np.float32)
    action_features_t = torch.tensor(
        action_features, dtype=torch.float32, device=device
    )
    action_t = torch.tensor(actions, dtype=torch.long, device=device)
    player_t = torch.tensor(players, dtype=torch.long, device=device)
    paradox_t = torch.tensor(
        np.array(paradoxes), dtype=torch.float32, device=device
    )
    terminal_action_paradox_t = torch.tensor(
        np.array(terminal_action_paradox_targets),
        dtype=torch.float32,
        device=device,
    )
    terminal_action_paradox_mask_t = torch.tensor(
        np.array(terminal_action_paradox_target_masks),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
      _, _, _, action_logits = model.forward_with_action_aux(
          obs_t, action_features_t
      )
      batch_indices = torch.arange(len(valid), device=device)
      selected_logits = action_logits[batch_indices, action_t]
      preds = torch.sigmoid(selected_logits).detach().cpu().numpy()
      targets = selected_terminal_paradox_targets(
          paradox_t,
          player_t,
          batch_indices,
          args,
          terminal_action_paradox_t=terminal_action_paradox_t,
          terminal_action_paradox_mask_t=terminal_action_paradox_mask_t,
      ).detach().cpu().numpy()
  else:
    return None
  brier = float(np.mean((preds - targets) ** 2))
  positive_rate = float(np.mean(targets))
  positive_mask = targets > 0.5
  negative_mask = ~positive_mask
  binary_targets = bool(
      np.all(np.logical_or(np.isclose(targets, 0.0), np.isclose(targets, 1.0)))
  )
  report = {
      "kind": report_kind,
      "target_kind": "binary" if binary_targets else "continuous",
      "count": int(len(preds)),
      "positive_rate": round(positive_rate, 4),
      "target_std": round(float(np.std(targets)), 4),
      "target_min": round(float(np.min(targets)), 4),
      "target_max": round(float(np.max(targets)), 4),
      "mean_pred": round(float(np.mean(preds)), 4),
      "brier": round(brier, 4),
      "auc": None,
      "corr": None,
      "positive_mean_pred": None,
      "negative_mean_pred": None,
  }
  if report_kind == "selected_actions":
    report["terminal_action_paradox_sidecar_rows"] = int(
        np.sum(terminal_action_paradox_target_masks)
    )
  if binary_targets:
    auc = binary_auc(preds, targets)
    if auc is not None:
      report["auc"] = round(auc, 4)
  if (
      len(preds) > 1
      and float(np.std(preds)) > 0.0
      and float(np.std(targets)) > 0.0
  ):
    corr = float(np.corrcoef(preds, targets)[0, 1])
    if math.isfinite(corr):
      report["corr"] = round(corr, 4)
  if positive_mask.any():
    report["positive_mean_pred"] = round(float(np.mean(preds[positive_mask])), 4)
  if negative_mask.any():
    report["negative_mean_pred"] = round(float(np.mean(preds[negative_mask])), 4)
  return report


def maybe_add_action_report(row, model, replay, args, device):
  if (
      getattr(args, "action_paradox_loss_weight", 0.0) <= 0
      and getattr(args, "action_paradox_ranking_loss_weight", 0.0) <= 0
  ):
    return
  report = action_paradox_report(model, list(replay), args, device)
  if report is not None:
    row["action_paradox_report"] = report


def counterfactual_action_value_report_kind(args):
  if getattr(args, "counterfactual_action_value_objective", "score") == "survival":
    return "counterfactual_round_survival"
  if getattr(args, "counterfactual_full_match_rollout", False):
    return "counterfactual_full_match_score"
  return "counterfactual_round_score"


def action_value_report(model, replay, args, device, max_examples=5000):
  if not replay or not hasattr(model, "action_value"):
    return None
  full_valid = [
      item for item in replay
      if (
          len(item) > 10
          and item[9] is not None
          and item[10] is not None
          and float(np.sum(item[10])) > 0
      )
  ]
  if not full_valid:
    return None
  if len(full_valid) > max_examples:
    full_valid = random.sample(full_valid, max_examples)
  obs = [item[0] for item in full_valid]
  target_vectors = np.array([item[9] for item in full_valid], dtype=np.float32)
  target_masks = np.array([item[10] for item in full_valid], dtype=np.float32)
  action_features = np.array([
      adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
      for item in full_valid
  ], dtype=np.float32)
  obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
  obs_t = adapt_observation_batch(obs_t, model_input_size(model))
  action_features_t = torch.tensor(
      action_features, dtype=torch.float32, device=device
  )
  with torch.no_grad():
    action_values = model_action_value_batch(model, obs_t, action_features_t)
    pred_matrix = action_values.detach().cpu().numpy()

  def summarize_value_predictions(row_mask, action_mask=None):
    if action_mask is None:
      selected_mask = np.array(target_masks, dtype=np.float32)
    else:
      selected_mask = np.array(action_mask, dtype=np.float32)
    selected_mask = selected_mask * row_mask[:, None]
    selected = selected_mask > 0.0
    preds = pred_matrix[selected]
    targets = target_vectors[selected]
    if len(preds) == 0:
      return None
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    corr = None
    if (
        len(preds) > 1
        and float(np.std(preds)) > 0
        and float(np.std(targets)) > 0
    ):
      corr_candidate = float(np.corrcoef(preds, targets)[0, 1])
      if math.isfinite(corr_candidate):
        corr = corr_candidate
    state_corrs = []
    top_hits = []
    chosen_targets = []
    best_targets = []
    mean_targets = []
    regrets = []
    for pred_row, target_row, mask_row in zip(
        pred_matrix, target_vectors, selected_mask
    ):
      legal = mask_row > 0.0
      if int(np.sum(legal)) < 2:
        continue
      legal_preds = pred_row[legal]
      legal_targets = target_row[legal]
      if float(np.std(legal_preds)) > 0 and float(np.std(legal_targets)) > 0:
        state_corr = float(np.corrcoef(legal_preds, legal_targets)[0, 1])
        if math.isfinite(state_corr):
          state_corrs.append(state_corr)
      chosen_index = int(np.argmax(legal_preds))
      best_index = int(np.argmax(legal_targets))
      chosen_target = float(legal_targets[chosen_index])
      best_target = float(legal_targets[best_index])
      chosen_targets.append(chosen_target)
      best_targets.append(best_target)
      mean_targets.append(float(np.mean(legal_targets)))
      regrets.append(best_target - chosen_target)
      top_hits.append(1.0 if chosen_index == best_index else 0.0)
    return {
        "count": int(len(preds)),
        "target_mean": round(float(np.mean(targets)), 4),
        "pred_mean": round(float(np.mean(preds)), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "corr": None if corr is None else round(corr, 4),
        "state_count": int(len(chosen_targets)),
        "mean_state_corr": (
            None if not state_corrs else round(float(np.mean(state_corrs)), 4)
        ),
        "top1_rate": (
            None if not top_hits else round(float(np.mean(top_hits)), 4)
        ),
        "top_pred_target_mean": (
            None if not chosen_targets
            else round(float(np.mean(chosen_targets)), 4)
        ),
        "best_target_mean": (
            None if not best_targets else round(float(np.mean(best_targets)), 4)
        ),
        "mean_target_mean": (
            None if not mean_targets else round(float(np.mean(mean_targets)), 4)
        ),
        "mean_regret": (
            None if not regrets else round(float(np.mean(regrets)), 4)
        ),
    }

  all_rows = np.ones(len(full_valid), dtype=bool)
  summary = summarize_value_predictions(all_rows, target_masks)
  if summary is None:
    return None

  phase_names = ["chance", "discard", "prediction", "play", "terminal"]
  type_names = ["discard", "prediction", "play", "paradox", "other"]
  labeled_row_indices = np.argmax(target_masks > 0.0, axis=1)
  row_phase_idx = np.argmax(
      action_features[
          np.arange(len(full_valid)), labeled_row_indices, 1:6
      ],
      axis=1,
  )
  by_phase = {}
  for idx, name in enumerate(phase_names):
    bucket = summarize_value_predictions(row_phase_idx == idx, target_masks)
    if bucket is not None:
      by_phase[name] = bucket

  action_type_idx = np.argmax(action_features[:, :, 6:11], axis=2)
  by_target_action_type = {}
  for idx, name in enumerate(type_names):
    bucket_mask = target_masks * (action_type_idx == idx)
    bucket = summarize_value_predictions(all_rows, bucket_mask)
    if bucket is not None:
      by_target_action_type[name] = bucket

  report = {"kind": counterfactual_action_value_report_kind(args)}
  report.update(summary)
  report["by_phase"] = by_phase
  report["by_target_action_type"] = by_target_action_type
  return report


def q_label_policy_report(model, replay, args, device, max_examples=5000):
  if not replay:
    return None
  full_valid = [
      item for item in replay
      if (
          len(item) > 10
          and item[9] is not None
          and item[10] is not None
          and float(np.sum(item[10])) > 0
      )
  ]
  if not full_valid:
    return None
  if len(full_valid) > max_examples:
    full_valid = random.sample(full_valid, max_examples)
  obs = [item[0] for item in full_valid]
  legal_masks = np.array([item[1] for item in full_valid], dtype=bool)
  target_vectors = np.array([item[9] for item in full_valid], dtype=np.float32)
  target_masks = np.array([item[10] for item in full_valid], dtype=np.float32)
  action_features = np.array([
      adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
      for item in full_valid
  ], dtype=np.float32)
  obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
  obs_t = adapt_observation_batch(obs_t, model_input_size(model))
  action_features_t = torch.tensor(
      action_features, dtype=torch.float32, device=device
  )
  legal_mask_t = torch.tensor(legal_masks, dtype=torch.bool, device=device)
  with torch.no_grad():
    logits, _ = model(obs_t, action_features_t)
    logits = logits.masked_fill(~legal_mask_t, -1e9)
    pred_matrix = logits.detach().cpu().numpy()

  def summarize_policy_rank(row_mask, action_mask=None):
    if action_mask is None:
      selected_mask = np.array(target_masks, dtype=np.float32)
    else:
      selected_mask = np.array(action_mask, dtype=np.float32)
    selected_mask = selected_mask * row_mask[:, None]
    selected = selected_mask > 0.0
    preds = pred_matrix[selected]
    targets = target_vectors[selected]
    if len(preds) == 0:
      return None
    corr = None
    if (
        len(preds) > 1
        and float(np.std(preds)) > 0
        and float(np.std(targets)) > 0
    ):
      corr_candidate = float(np.corrcoef(preds, targets)[0, 1])
      if math.isfinite(corr_candidate):
        corr = corr_candidate
    state_corrs = []
    top_hits = []
    chosen_targets = []
    best_targets = []
    mean_targets = []
    regrets = []
    for pred_row, target_row, mask_row in zip(
        pred_matrix, target_vectors, selected_mask
    ):
      labeled = mask_row > 0.0
      if int(np.sum(labeled)) < 2:
        continue
      labeled_preds = pred_row[labeled]
      labeled_targets = target_row[labeled]
      if float(np.std(labeled_preds)) > 0 and float(np.std(labeled_targets)) > 0:
        state_corr = float(np.corrcoef(labeled_preds, labeled_targets)[0, 1])
        if math.isfinite(state_corr):
          state_corrs.append(state_corr)
      chosen_index = int(np.argmax(labeled_preds))
      best_index = int(np.argmax(labeled_targets))
      chosen_target = float(labeled_targets[chosen_index])
      best_target = float(labeled_targets[best_index])
      chosen_targets.append(chosen_target)
      best_targets.append(best_target)
      mean_targets.append(float(np.mean(labeled_targets)))
      regrets.append(best_target - chosen_target)
      top_hits.append(1.0 if chosen_index == best_index else 0.0)
    return {
        "count": int(len(preds)),
        "target_mean": round(float(np.mean(targets)), 4),
        "logit_mean": round(float(np.mean(preds)), 4),
        "corr": None if corr is None else round(corr, 4),
        "state_count": int(len(chosen_targets)),
        "mean_state_corr": (
            None if not state_corrs else round(float(np.mean(state_corrs)), 4)
        ),
        "top1_rate": (
            None if not top_hits else round(float(np.mean(top_hits)), 4)
        ),
        "top_policy_target_mean": (
            None if not chosen_targets
            else round(float(np.mean(chosen_targets)), 4)
        ),
        "best_target_mean": (
            None if not best_targets else round(float(np.mean(best_targets)), 4)
        ),
        "mean_target_mean": (
            None if not mean_targets else round(float(np.mean(mean_targets)), 4)
        ),
        "mean_regret": (
            None if not regrets else round(float(np.mean(regrets)), 4)
        ),
    }

  all_rows = np.ones(len(full_valid), dtype=bool)
  summary = summarize_policy_rank(all_rows, target_masks)
  if summary is None:
    return None

  phase_names = ["chance", "discard", "prediction", "play", "terminal"]
  type_names = ["discard", "prediction", "play", "paradox", "other"]
  labeled_row_indices = np.argmax(target_masks > 0.0, axis=1)
  row_phase_idx = np.argmax(
      action_features[
          np.arange(len(full_valid)), labeled_row_indices, 1:6
      ],
      axis=1,
  )
  by_phase = {}
  for idx, name in enumerate(phase_names):
    bucket = summarize_policy_rank(row_phase_idx == idx, target_masks)
    if bucket is not None:
      by_phase[name] = bucket

  action_type_idx = np.argmax(action_features[:, :, 6:11], axis=2)
  by_target_action_type = {}
  for idx, name in enumerate(type_names):
    bucket_mask = target_masks * (action_type_idx == idx)
    bucket = summarize_policy_rank(all_rows, bucket_mask)
    if bucket is not None:
      by_target_action_type[name] = bucket

  masked_targets = np.where(target_masks > 0.0, target_vectors, -np.inf)
  q_best_actions = np.argmax(masked_targets, axis=1)
  q_best_rows = action_features[np.arange(len(full_valid)), q_best_actions]
  by_q_best_tactical_bucket = {}
  for name, bucket_mask in policy_target_bucket_masks(q_best_rows).items():
    bucket = summarize_policy_rank(bucket_mask, target_masks)
    if bucket is not None:
      by_q_best_tactical_bucket[name] = bucket

  report = {"kind": counterfactual_action_value_report_kind(args)}
  report.update(summary)
  report["by_phase"] = by_phase
  report["by_target_action_type"] = by_target_action_type
  report["by_q_best_tactical_bucket"] = by_q_best_tactical_bucket
  return report


def counterfactual_label_coverage_report(replay):
  if not replay:
    return None
  phase_names = ["chance", "discard", "prediction", "play", "terminal"]
  report = {
      "rows": int(len(replay)),
      "policy_labeled_rows": 0,
      "policy_labeled_actions": 0,
      "policy_labels_per_labeled_row_mean": 0.0,
      "policy_target_spread_mean": None,
      "policy_top_margin_mean": None,
      "policy_target_positive_rate": None,
      "policy_by_phase": {},
      "action_value_labeled_rows": 0,
      "action_value_labeled_actions": 0,
      "action_value_labels_per_labeled_row_mean": 0.0,
      "action_value_target_spread_mean": None,
      "action_value_top_margin_mean": None,
      "action_value_by_phase": {},
  }
  value_action_counts = []
  value_spreads = []
  value_margins = []
  by_phase = {
      name: {"rows": 0, "actions": 0, "spreads": [], "margins": []}
      for name in phase_names
  }
  policy_action_counts = []
  policy_spreads = []
  policy_margins = []
  policy_targets_seen = []
  policy_by_phase = {
      name: {
          "rows": 0,
          "actions": 0,
          "spreads": [],
          "margins": [],
          "targets": [],
      }
      for name in phase_names
  }
  for item in replay:
    policy_targets = (
        item[7]
        if len(item) > 8 and item[7] is not None
        else None
    )
    policy_mask = (
        item[8]
        if len(item) > 8 and item[8] is not None
        else None
    )
    if policy_mask is not None:
      policy_mask = np.array(policy_mask, dtype=np.float32)
      policy_labeled = policy_mask > 0.0
      policy_count = int(np.sum(policy_labeled))
      if policy_count > 0:
        report["policy_labeled_rows"] += 1
        report["policy_labeled_actions"] += policy_count
        policy_action_counts.append(policy_count)
        if policy_targets is not None:
          labeled_policy_targets = np.array(
              policy_targets, dtype=np.float32
          )[policy_labeled]
          policy_targets_seen.extend(float(value) for value in labeled_policy_targets)
          if policy_count >= 2:
            sorted_targets = np.sort(labeled_policy_targets)
            policy_spread = float(sorted_targets[-1] - sorted_targets[0])
            policy_margin = float(sorted_targets[-1] - sorted_targets[-2])
            policy_spreads.append(policy_spread)
            policy_margins.append(policy_margin)
          action_features = item[11] if len(item) > 11 else None
          if action_features is not None:
            first_labeled = int(np.argmax(policy_labeled))
            adapted = adapt_action_features(action_features, len(policy_mask))
            phase_idx = int(np.argmax(adapted[first_labeled, 1:6]))
            if 0 <= phase_idx < len(phase_names):
              bucket = policy_by_phase[phase_names[phase_idx]]
              bucket["rows"] += 1
              bucket["actions"] += policy_count
              bucket["targets"].extend(
                  float(value) for value in labeled_policy_targets
              )
              if policy_count >= 2:
                bucket["spreads"].append(policy_spread)
                bucket["margins"].append(policy_margin)
    value_targets = (
        item[9]
        if len(item) > 10 and item[9] is not None
        else None
    )
    value_mask = (
        item[10]
        if len(item) > 10 and item[10] is not None
        else None
    )
    if value_targets is None or value_mask is None:
      continue
    value_mask = np.array(value_mask, dtype=np.float32)
    labeled = value_mask > 0.0
    value_count = int(np.sum(labeled))
    if value_count <= 0:
      continue
    report["action_value_labeled_rows"] += 1
    report["action_value_labeled_actions"] += value_count
    value_action_counts.append(value_count)
    labeled_targets = np.array(value_targets, dtype=np.float32)[labeled]
    if value_count >= 2:
      sorted_targets = np.sort(labeled_targets)
      spread = float(sorted_targets[-1] - sorted_targets[0])
      margin = float(sorted_targets[-1] - sorted_targets[-2])
      value_spreads.append(spread)
      value_margins.append(margin)
    action_features = item[11] if len(item) > 11 else None
    if action_features is not None:
      first_labeled = int(np.argmax(labeled))
      adapted = adapt_action_features(action_features, len(value_mask))
      phase_idx = int(np.argmax(adapted[first_labeled, 1:6]))
      if 0 <= phase_idx < len(phase_names):
        bucket = by_phase[phase_names[phase_idx]]
        bucket["rows"] += 1
        bucket["actions"] += value_count
        if value_count >= 2:
          bucket["spreads"].append(spread)
          bucket["margins"].append(margin)

  if policy_action_counts:
    report["policy_labels_per_labeled_row_mean"] = round(
        float(np.mean(policy_action_counts)), 4
    )
  if policy_spreads:
    report["policy_target_spread_mean"] = round(float(np.mean(policy_spreads)), 4)
  if policy_margins:
    report["policy_top_margin_mean"] = round(float(np.mean(policy_margins)), 4)
  if policy_targets_seen:
    report["policy_target_positive_rate"] = round(
        float(np.mean(np.array(policy_targets_seen, dtype=np.float32) > 0.5)),
        4,
    )
  for phase_name, bucket in policy_by_phase.items():
    if bucket["rows"] <= 0:
      continue
    targets = np.array(bucket["targets"], dtype=np.float32)
    report["policy_by_phase"][phase_name] = {
        "rows": int(bucket["rows"]),
        "actions": int(bucket["actions"]),
        "target_positive_rate": (
            None if len(targets) == 0
            else round(float(np.mean(targets > 0.5)), 4)
        ),
        "target_spread_mean": (
            None if not bucket["spreads"]
            else round(float(np.mean(bucket["spreads"])), 4)
        ),
        "top_margin_mean": (
            None if not bucket["margins"]
            else round(float(np.mean(bucket["margins"])), 4)
        ),
    }
  if value_action_counts:
    report["action_value_labels_per_labeled_row_mean"] = round(
        float(np.mean(value_action_counts)), 4
    )
  if value_spreads:
    report["action_value_target_spread_mean"] = round(float(np.mean(value_spreads)), 4)
  if value_margins:
    report["action_value_top_margin_mean"] = round(float(np.mean(value_margins)), 4)
  for phase_name, bucket in by_phase.items():
    if bucket["rows"] <= 0:
      continue
    report["action_value_by_phase"][phase_name] = {
        "rows": int(bucket["rows"]),
        "actions": int(bucket["actions"]),
        "target_spread_mean": (
            None if not bucket["spreads"]
            else round(float(np.mean(bucket["spreads"])), 4)
        ),
        "top_margin_mean": (
            None if not bucket["margins"]
            else round(float(np.mean(bucket["margins"])), 4)
        ),
    }
  return report


def maybe_add_action_value_report(row, model, replay, args, device):
  if getattr(args, "action_value_loss_weight", 0.0) <= 0:
    return
  report = action_value_report(model, list(replay), args, device)
  if report is not None:
    row["action_value_report"] = report


def prediction_hit_report(model, replay, args, device, max_examples=5000):
  if not replay:
    return None
  hit_idx = APPENDED_ACTION_FEATURE_INDEX["hits_prediction"]
  future_hit_idx = APPENDED_ACTION_FEATURE_INDEX["hit_with_future_tricks"]
  low_legal_ratio_idx = APPENDED_ACTION_FEATURE_INDEX[
      "post_hit_low_legal_lead_ratio"
  ]
  survival_margin_idx = APPENDED_ACTION_FEATURE_INDEX[
      "post_hit_low_card_survival_margin"
  ]
  forced_pressure_idx = APPENDED_ACTION_FEATURE_INDEX[
      "post_hit_forced_card_pressure"
  ]
  valid = []
  for item in replay:
    if len(item) <= 11:
      continue
    mask = np.array(item[1], dtype=np.float32) > 0.5
    features = adapt_action_features(item[11], len(item[1]))
    hit_mask = (features[:, hit_idx] > 0.5) & mask
    if hit_mask.any():
      valid.append((item, hit_mask))
  if not valid:
    return None
  if len(valid) > max_examples:
    valid = random.sample(valid, max_examples)
  items = [entry[0] for entry in valid]
  hit_masks = np.array([entry[1] for entry in valid], dtype=bool)
  obs = [item[0] for item in items]
  masks = np.array([item[1] for item in items], dtype=np.float32)
  policies = np.array([item[2] for item in items], dtype=np.float32)
  action_features = np.array([
      adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
      for item in items
  ], dtype=np.float32)
  probs_chunks = []
  batch_size = (
      128 if getattr(model, "arch", "mlp") in ACTION_CONDITIONED_ARCHS else 512
  )
  obs_array = np.array(obs, dtype=np.float32)
  with torch.no_grad():
    for start in range(0, len(items), batch_size):
      end = min(len(items), start + batch_size)
      obs_t = torch.tensor(
          obs_array[start:end], dtype=torch.float32, device=device
      )
      obs_t = adapt_observation_batch(obs_t, model_input_size(model))
      mask_t = torch.tensor(
          masks[start:end], dtype=torch.bool, device=device
      )
      action_features_t = torch.tensor(
          action_features[start:end], dtype=torch.float32, device=device
      )
      logits, _ = model(obs_t, action_features_t)
      logits = logits.masked_fill(~mask_t, -1e9)
      probs_chunks.append(F.softmax(logits, dim=1).detach().cpu().numpy())
  probs = np.concatenate(probs_chunks, axis=0)
  model_hit_mass = np.sum(probs * hit_masks, axis=1)
  target_hit_mass = np.sum(policies * hit_masks, axis=1)
  future_hit_masks = (
      action_features[:, :, future_hit_idx] > 0.5
  ) & (masks > 0.5)
  dangerous_future_hit_masks = future_hit_masks & (
      (
          action_features[:, :, low_legal_ratio_idx]
          <= float(getattr(
              args, "dangerous_future_hit_low_legal_ratio_threshold", 0.25
          ))
      )
      | (
          action_features[:, :, survival_margin_idx]
          < float(getattr(
              args, "dangerous_future_hit_survival_margin_threshold", 0.0
          ))
      )
      | (
          action_features[:, :, forced_pressure_idx]
          > float(getattr(
              args, "dangerous_future_hit_forced_pressure_threshold", 0.75
          ))
      )
  )
  model_future_hit_mass = np.sum(probs * future_hit_masks, axis=1)
  target_future_hit_mass = np.sum(policies * future_hit_masks, axis=1)
  model_dangerous_future_hit_mass = np.sum(
      probs * dangerous_future_hit_masks, axis=1
  )
  target_dangerous_future_hit_mass = np.sum(
      policies * dangerous_future_hit_masks, axis=1
  )
  pred_actions = np.argmax(probs, axis=1)
  target_actions = np.argmax(policies, axis=1)
  return {
      "count": int(len(items)),
      "mean_hit_action_count": round(float(np.mean(hit_masks.sum(axis=1))), 4),
      "mean_future_hit_action_count": round(
          float(np.mean(future_hit_masks.sum(axis=1))), 4
      ),
      "mean_dangerous_future_hit_action_count": round(
          float(np.mean(dangerous_future_hit_masks.sum(axis=1))), 4
      ),
      "model_hit_mass": round(float(np.mean(model_hit_mass)), 4),
      "target_hit_mass": round(float(np.mean(target_hit_mass)), 4),
      "model_future_hit_mass": round(float(np.mean(model_future_hit_mass)), 4),
      "target_future_hit_mass": round(float(np.mean(target_future_hit_mass)), 4),
      "model_dangerous_future_hit_mass": round(
          float(np.mean(model_dangerous_future_hit_mass)), 4
      ),
      "target_dangerous_future_hit_mass": round(
          float(np.mean(target_dangerous_future_hit_mass)), 4
      ),
      "model_top1_hit_rate": round(
          float(np.mean(hit_masks[np.arange(len(items)), pred_actions])), 4
      ),
      "target_top1_hit_rate": round(
          float(np.mean(hit_masks[np.arange(len(items)), target_actions])), 4
      ),
      "model_top1_future_hit_rate": round(
          float(np.mean(future_hit_masks[np.arange(len(items)), pred_actions])), 4
      ),
      "target_top1_future_hit_rate": round(
          float(np.mean(future_hit_masks[np.arange(len(items)), target_actions])), 4
      ),
      "model_top1_dangerous_future_hit_rate": round(
          float(np.mean(
              dangerous_future_hit_masks[np.arange(len(items)), pred_actions]
          )),
          4,
      ),
      "target_top1_dangerous_future_hit_rate": round(
          float(np.mean(
              dangerous_future_hit_masks[np.arange(len(items)), target_actions]
          )),
          4,
      ),
  }


def maybe_add_prediction_hit_report(row, model, replay, args, device):
  if (
      getattr(args, "prediction_hit_policy_loss_weight", 0.0) <= 0
      and getattr(args, "future_hit_policy_loss_weight", 0.0) <= 0
      and getattr(args, "dangerous_future_hit_policy_loss_weight", 0.0) <= 0
  ):
    return
  report = prediction_hit_report(model, list(replay), args, device)
  if report is not None:
    row["prediction_hit_report"] = report


def policy_alignment_summary(probs, target_policies, action_features):
  target_actions = np.argmax(target_policies, axis=1)
  pred_actions = np.argmax(probs, axis=1)
  target_probs = probs[np.arange(len(target_policies)), target_actions]
  cross_entropy = -np.sum(
      target_policies * np.log(np.maximum(probs, 1e-12)), axis=1
  )
  top1 = pred_actions == target_actions
  phase_names = ["chance", "discard", "prediction", "play", "terminal"]
  type_names = ["discard", "prediction", "play", "paradox", "other"]
  target_phase_idx = np.argmax(
      action_features[np.arange(len(target_actions)), target_actions, 1:6],
      axis=1,
  )
  target_type_idx = np.argmax(
      action_features[np.arange(len(target_actions)), target_actions, 6:11],
      axis=1,
  )
  pred_type_idx = np.argmax(
      action_features[np.arange(len(pred_actions)), pred_actions, 6:11],
      axis=1,
  )
  target_action_rows = action_features[np.arange(len(target_actions)), target_actions]

  def bucket_report(bucket_values, names):
    buckets = {}
    for idx, name in enumerate(names):
      bucket_mask = bucket_values == idx
      if not np.any(bucket_mask):
        continue
      buckets[name] = {
          "count": int(np.sum(bucket_mask)),
          "top1_rate": round(float(np.mean(top1[bucket_mask])), 4),
          "mean_target_prob": round(
              float(np.mean(target_probs[bucket_mask])), 4
          ),
          "cross_entropy": round(float(np.mean(cross_entropy[bucket_mask])), 4),
      }
    return buckets

  confusion = {}
  for target_idx, target_name in enumerate(type_names):
    target_mask = target_type_idx == target_idx
    if not np.any(target_mask):
      continue
    pred_counts = {}
    for pred_idx, pred_name in enumerate(type_names):
      count = int(np.sum(target_mask & (pred_type_idx == pred_idx)))
      if count:
        pred_counts[pred_name] = count
    confusion[target_name] = pred_counts

  return {
      "top1_rate": round(float(np.mean(top1)), 4),
      "mean_target_prob": round(float(np.mean(target_probs)), 4),
      "cross_entropy": round(float(np.mean(cross_entropy)), 4),
      "by_phase": bucket_report(target_phase_idx, phase_names),
      "by_target_action_type": bucket_report(target_type_idx, type_names),
      "by_target_tactical_bucket": {
          name: {
              "count": int(np.sum(bucket_mask)),
              "top1_rate": round(float(np.mean(top1[bucket_mask])), 4),
              "mean_target_prob": round(
                  float(np.mean(target_probs[bucket_mask])), 4
              ),
              "cross_entropy": round(
                  float(np.mean(cross_entropy[bucket_mask])), 4
              ),
          }
          for name, bucket_mask in policy_target_bucket_masks(
              target_action_rows
          ).items()
          if np.any(bucket_mask)
      },
      "target_action_type_confusion": confusion,
  }


def policy_target_report(
    model, replay, args, device, max_examples=5000, one_hot_only=False
):
  valid = []
  for item in replay:
    if len(item) < 4:
      continue
    policy = np.array(item[2], dtype=np.float32)
    mask = np.array(item[1], dtype=np.float32) > 0.5
    if not mask.any():
      continue
    if one_hot_only and (
        float(np.max(policy)) < 0.999 or not np.isclose(float(policy.sum()), 1.0)
    ):
      continue
    valid.append(item)
  if not valid:
    return None
  if len(valid) > max_examples:
    valid = random.sample(valid, max_examples)
  obs = [item[0] for item in valid]
  masks = np.array([item[1] for item in valid], dtype=np.float32)
  policies = np.array([item[2] for item in valid], dtype=np.float32)
  action_features = np.array([
      adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
      for item in valid
  ], dtype=np.float32)
  prob_chunks = []
  report_batch_size = (
      128
      if getattr(model, "arch", "mlp") in ACTION_CONDITIONED_ARCHS
      else 512
  )
  obs_array = np.array(obs, dtype=np.float32)
  with torch.no_grad():
    for start in range(0, len(valid), report_batch_size):
      end = min(len(valid), start + report_batch_size)
      obs_t = torch.tensor(
          obs_array[start:end], dtype=torch.float32, device=device
      )
      obs_t = adapt_observation_batch(obs_t, model_input_size(model))
      mask_t = torch.tensor(
          masks[start:end], dtype=torch.bool, device=device
      )
      action_features_t = torch.tensor(
          action_features[start:end], dtype=torch.float32, device=device
      )
      logits, _ = model(obs_t, action_features_t)
      logits = logits.masked_fill(~mask_t, -1e9)
      prob_chunks.append(F.softmax(logits, dim=1).detach().cpu().numpy())
  probs = np.concatenate(prob_chunks, axis=0)
  report = {
      "count": int(len(valid)),
      "one_hot_only": bool(one_hot_only),
  }
  report.update(policy_alignment_summary(probs, policies, action_features))
  return report


def anchor_policy_report(
    model, anchor_model, replay, args, device, max_examples=5000
):
  if anchor_model is None or not replay:
    return None
  valid = []
  for item in replay:
    if len(item) < 4:
      continue
    mask = np.array(item[1], dtype=np.float32) > 0.5
    if mask.any():
      valid.append(item)
  if not valid:
    return None
  if len(valid) > max_examples:
    valid = random.sample(valid, max_examples)
  obs = [item[0] for item in valid]
  masks = np.array([item[1] for item in valid], dtype=np.float32)
  action_features = np.array([
      adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
      for item in valid
  ], dtype=np.float32)
  model_prob_chunks = []
  anchor_prob_chunks = []
  report_batch_size = (
      128
      if getattr(model, "arch", "mlp") in ACTION_CONDITIONED_ARCHS
      else 512
  )
  obs_array = np.array(obs, dtype=np.float32)
  with torch.no_grad():
    for start in range(0, len(valid), report_batch_size):
      end = min(len(valid), start + report_batch_size)
      obs_t = torch.tensor(
          obs_array[start:end], dtype=torch.float32, device=device
      )
      model_obs_t = adapt_observation_batch(obs_t, model_input_size(model))
      anchor_obs_t = adapt_observation_batch(
          obs_t, model_input_size(anchor_model)
      )
      mask_t = torch.tensor(
          masks[start:end], dtype=torch.bool, device=device
      )
      action_features_t = torch.tensor(
          action_features[start:end], dtype=torch.float32, device=device
      )
      logits, _ = model(model_obs_t, action_features_t)
      anchor_logits, _ = anchor_model(anchor_obs_t, action_features_t)
      logits = logits.masked_fill(~mask_t, -1e9)
      anchor_logits = anchor_logits.masked_fill(~mask_t, -1e9)
      model_prob_chunks.append(F.softmax(logits, dim=1).detach().cpu().numpy())
      anchor_prob_chunks.append(
          F.softmax(anchor_logits, dim=1).detach().cpu().numpy()
      )
  probs = np.concatenate(model_prob_chunks, axis=0)
  anchor_policies = np.concatenate(anchor_prob_chunks, axis=0)
  anchor_top_probs = np.max(anchor_policies, axis=1)
  report = {
      "count": int(len(valid)),
      "anchor_mean_top_prob": round(float(np.mean(anchor_top_probs)), 4),
  }
  report.update(policy_alignment_summary(probs, anchor_policies, action_features))
  return report


def maybe_add_policy_target_report(
    row, model, replay, args, device, name="policy_target_report",
    one_hot_only=False,
):
  report = policy_target_report(
      model, list(replay), args, device, one_hot_only=one_hot_only
  )
  if report is not None:
    row[name] = report


def maybe_add_anchor_policy_report(
    row, model, anchor_model, replay, args, device,
    name="anchor_policy_report",
):
  report = anchor_policy_report(model, anchor_model, list(replay), args, device)
  if report is not None:
    row[name] = report


def maybe_add_anchor_policy_target_report(
    row, anchor_model, replay, args, device,
    name="anchor_policy_target_report",
):
  if anchor_model is None:
    return
  report = policy_target_report(anchor_model, list(replay), args, device)
  if report is not None:
    row[name] = report


def save_replay(replay, path, metadata=None):
  if not replay:
    return
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_name(
      f".{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
  )
  obs = [item[0] for item in replay]
  masks = [item[1] for item in replay]
  policies = [item[2] for item in replay]
  values = [item[3] for item in replay]
  paradoxes = [
      item[4] if len(item) > 4 else np.zeros_like(values[0], dtype=np.float32)
      for item in replay
  ]
  actions = [item[5] if len(item) > 5 else -1 for item in replay]
  players = [item[6] if len(item) > 6 else -1 for item in replay]
  action_target_vectors = [
      (
          item[7]
          if len(item) > 8 and item[7] is not None
          else np.zeros_like(masks[0], dtype=np.float32)
      )
      for item in replay
  ]
  action_target_masks = [
      (
          item[8]
          if len(item) > 8 and item[8] is not None
          else np.zeros_like(masks[0], dtype=np.float32)
      )
      for item in replay
  ]
  action_value_target_vectors = [
      (
          item[9]
          if len(item) > 10 and item[9] is not None
          else np.zeros_like(masks[0], dtype=np.float32)
      )
      for item in replay
  ]
  action_value_target_masks = [
      (
          item[10]
          if len(item) > 10 and item[10] is not None
          else np.zeros_like(masks[0], dtype=np.float32)
      )
      for item in replay
  ]
  action_features = [
      adapt_action_features(item[11] if len(item) > 11 else None, len(item[1]))
      for item in replay
  ]
  terminal_action_paradox_targets = []
  terminal_action_paradox_target_masks = []
  for item in replay:
    if len(item) > 12 and item[12] is not None:
      terminal_action_paradox_targets.append(
          adapt_player_target_vector(item[12], len(values[0]))
      )
      terminal_action_paradox_target_masks.append(1.0)
    else:
      terminal_action_paradox_targets.append(
          np.zeros(len(values[0]), dtype=np.float32)
      )
      terminal_action_paradox_target_masks.append(0.0)
  try:
    with tmp_path.open("wb") as tmp_file:
      np.savez_compressed(
          tmp_file,
          obs=np.array(obs, dtype=np.float32),
          masks=np.array(masks, dtype=np.float32),
          policies=np.array(policies, dtype=np.float32),
          values=np.array(values, dtype=np.float32),
          paradoxes=np.array(paradoxes, dtype=np.float32),
          actions=np.array(actions, dtype=np.int64),
          players=np.array(players, dtype=np.int64),
          action_target_vectors=np.array(
              action_target_vectors, dtype=np.float32
          ),
          action_target_masks=np.array(
              action_target_masks, dtype=np.float32
          ),
          action_value_target_vectors=np.array(
              action_value_target_vectors, dtype=np.float32
          ),
          action_value_target_masks=np.array(
              action_value_target_masks, dtype=np.float32
          ),
          action_features=np.array(action_features, dtype=np.float32),
          terminal_action_paradox_targets=np.array(
              terminal_action_paradox_targets, dtype=np.float32
          ),
          terminal_action_paradox_target_masks=np.array(
              terminal_action_paradox_target_masks, dtype=np.float32
          ),
          replay_meta_json=np.array(
              json.dumps(metadata or {}, sort_keys=True), dtype=np.str_
          ),
      )
    os.replace(tmp_path, path)
  finally:
    if tmp_path.exists():
      tmp_path.unlink()


def load_replay(path, return_metadata=False):
  data = np.load(path)
  metadata = {}
  if "replay_meta_json" in data:
    try:
      metadata = json.loads(str(data["replay_meta_json"].item()))
    except Exception:
      metadata = {}
  actions = data["actions"] if "actions" in data else None
  players = data["players"] if "players" in data else None
  action_target_vectors = (
      data["action_target_vectors"] if "action_target_vectors" in data else None
  )
  action_target_masks = (
      data["action_target_masks"] if "action_target_masks" in data else None
  )
  action_value_target_vectors = (
      data["action_value_target_vectors"]
      if "action_value_target_vectors" in data else None
  )
  action_value_target_masks = (
      data["action_value_target_masks"]
      if "action_value_target_masks" in data else None
  )
  action_features = data["action_features"] if "action_features" in data else None
  terminal_action_paradox_targets = (
      data["terminal_action_paradox_targets"]
      if "terminal_action_paradox_targets" in data else None
  )
  terminal_action_paradox_target_masks = (
      data["terminal_action_paradox_target_masks"]
      if "terminal_action_paradox_target_masks" in data else None
  )
  if "paradoxes" in data:
    rows = []
    for idx, (obs, mask, policy, value, paradox) in enumerate(zip(
        data["obs"], data["masks"], data["policies"],
        data["values"], data["paradoxes"]
    )):
      action = int(actions[idx]) if actions is not None else -1
      player = int(players[idx]) if players is not None else -1
      action_targets = (
          action_target_vectors[idx] if action_target_vectors is not None else None
      )
      action_mask = (
          action_target_masks[idx] if action_target_masks is not None else None
      )
      action_value_targets = (
          action_value_target_vectors[idx]
          if action_value_target_vectors is not None else None
      )
      action_value_mask = (
          action_value_target_masks[idx]
          if action_value_target_masks is not None else None
      )
      action_feature_row = (
          adapt_action_features(
              action_features[idx] if action_features is not None else None,
              len(mask),
          )
      )
      terminal_action_paradox_target = None
      if terminal_action_paradox_targets is not None:
        target_mask = (
            terminal_action_paradox_target_masks[idx]
            if terminal_action_paradox_target_masks is not None
            else np.ones_like(terminal_action_paradox_targets[idx])
        )
        if float(np.sum(target_mask)) > 0.0:
          terminal_action_paradox_target = adapt_player_target_vector(
              terminal_action_paradox_targets[idx], len(paradox)
          )
      rows.append((
          obs, mask, policy, value, paradox, action, player,
          action_targets, action_mask, action_value_targets, action_value_mask,
          action_feature_row, terminal_action_paradox_target,
      ))
    return (rows, metadata) if return_metadata else rows
  rows = [
      (obs, mask, policy, value)
      for obs, mask, policy, value in zip(
          data["obs"], data["masks"], data["policies"], data["values"]
      )
  ]
  return (rows, metadata) if return_metadata else rows


def load_model_payload(checkpoint_path, game, args, device):
  payload = torch.load(checkpoint_path, map_location=device)
  saved_args = payload.get("args", {})
  arch = saved_args.get("arch", args.arch)
  shape_game = pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": args.players,
          "start_player": 0,
          "match_context": int(saved_args.get("match_context", False)),
      },
  )
  model = AZNet(
      shape_game.observation_tensor_shape()[0],
      shape_game.num_distinct_actions(),
      args.players,
      saved_args.get("width", args.width),
      saved_args.get("depth", args.depth),
      arch,
      saved_args.get(
          "separate_action_value_encoder",
          getattr(args, "separate_action_value_encoder", False),
      ),
      saved_args.get(
          "separate_action_paradox_encoder",
          getattr(args, "separate_action_paradox_encoder", False),
      ),
  ).to(device)
  load_compatible_state_dict(model, payload["model"])
  initialize_missing_action_value_stack_from_policy(model, payload["model"])
  initialize_missing_action_paradox_stack_from_policy(model, payload["model"])
  return model, payload, saved_args


class AZPolicyBot:
  def __init__(self, model, name, device, value_scale):
    self.name = name
    self.model = model
    self.device = device
    self.value_scale = value_scale

  def step(self, state, player):
    prediction_action = shared_prediction_action(state, player)
    if prediction_action is not None:
      return int(prediction_action)
    policy, _ = model_policy_value(
        self.model, state, player, state.num_distinct_actions(),
        self.value_scale, self.device
    )
    legal = state.legal_actions(player)
    return max(legal, key=lambda action: policy[action])


class AZBeliefPolicyBot:
  """Raw network policy averaged over sampled hidden-information states."""

  def __init__(self, model, name, device, value_scale, samples, args=None):
    self.name = name
    self.model = model
    self.device = device
    self.value_scale = value_scale
    self.samples = samples
    self.args = args

  def step(self, state, player):
    legal = state.legal_actions(player)
    prediction_action = shared_prediction_action(state, player, legal)
    if prediction_action is not None:
      return int(prediction_action)
    if len(legal) == 1:
      return legal[0]
    combined = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    sampled_states = sampled_belief_states_for_policy(
        state,
        player,
        self.samples,
        self.args or SimpleNamespace(),
        self.model,
        self.device,
        self.value_scale,
        context="eval",
    )
    for sampled in sampled_states:
      policy, _ = model_policy_value(
          self.model,
          sampled,
          player,
          sampled.num_distinct_actions(),
          self.value_scale,
          self.device,
      )
      combined += policy
    combined /= float(max(1, len(sampled_states)))
    masked = np.zeros_like(combined)
    masked[legal] = combined[legal]
    total = float(masked.sum())
    if total <= 0 or not math.isfinite(total):
      masked[legal] = 1.0 / len(legal)
    else:
      masked /= total
    return max(legal, key=lambda action: masked[action])


class AZSearchBot:
  """Diagnostic network+MCTS actor over the simulator state.

  This is useful for measuring whether the value/policy network helps search,
  but it should not be treated as online human-play proof until the search is
  changed to sample hidden information from the player's observation.
  """

  def __init__(self, model, name, device, args, sims):
    self.name = name
    self.model = model
    self.device = device
    self.args = args
    self.sims = sims

  def step(self, state, player):
    prediction_action = shared_prediction_action(state, player)
    if prediction_action is not None:
      return int(prediction_action)
    policy = mcts_policy(
        state, self.model, self.args, self.device, add_noise=False, sims=self.sims
    )
    legal = state.legal_actions(player)
    return max(legal, key=lambda action: policy[action])


class AZBeliefSearchBot:
  """Network-guided search averaged over sampled hidden-information states."""

  def __init__(self, model, name, device, args, samples, sims):
    self.name = name
    self.model = model
    self.device = device
    self.args = args
    self.samples = samples
    self.sims = sims

  def step(self, state, player):
    legal = state.legal_actions(player)
    prediction_action = shared_prediction_action(state, player, legal)
    if prediction_action is not None:
      return int(prediction_action)
    if len(legal) == 1:
      return legal[0]
    combined = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    sampled_states = sampled_belief_states_for_policy(
        state,
        player,
        self.samples,
        self.args,
        self.model,
        self.device,
        self.args.value_scale,
        context="eval",
    )
    for sampled in sampled_states:
      policy = mcts_policy(
          sampled, self.model, self.args, self.device,
          add_noise=False, sims=self.sims
      )
      combined += policy
    combined /= float(max(1, len(sampled_states)))
    return max(legal, key=lambda action: combined[action])


def evaluate_checkpoint(model, game, args, device):
  if getattr(args, "eval_full_match", False):
    return evaluate_checkpoint_full_match(model, game, args, device)
  names = ["az_policy", "heuristic", "heuristic_target2", "random"]
  if args.eval_mcts_sims > 0:
    names.append("az_search")
  if args.eval_belief_samples > 0:
    names.append("az_belief_policy")
    names.append("az_belief_search")
  ratings = {name: 1000.0 for name in names}
  stats = {name: [] for name in ratings}
  for idx in range(args.eval_games):
    round_game = (
        make_game(args, idx % args.players)
        if args.eval_rotate_start_player else game
    )
    seated = [names[(idx + offset) % len(names)] for offset in range(args.players)]
    bots = []
    for seat, name in enumerate(seated):
      if name == "az_policy":
        bots.append(AZPolicyBot(model, name, device, args.value_scale))
      elif name == "az_belief_policy":
        bots.append(AZBeliefPolicyBot(
            model, name, device, args.value_scale, args.eval_belief_samples
        ))
      elif name == "az_search":
        bots.append(AZSearchBot(
            model, name, device, args, args.eval_mcts_sims
        ))
      elif name == "az_belief_search":
        bots.append(AZBeliefSearchBot(
            model, name, device, args,
            args.eval_belief_samples, args.eval_belief_sims
        ))
      else:
        bots.append(make_bot(name, seed=args.seed + idx * 13 + seat))
    returns, _ = play_game(round_game, bots, seed=args.seed + 100000 + idx)
    multiplayer_elo_update(ratings, seated, returns, 24.0)
    for seat, name in enumerate(seated):
      stats[name].append(float(returns[seat]))
  return {
      "ratings": {k: round(v, 2) for k, v in sorted(
          ratings.items(), key=lambda item: item[1], reverse=True)},
      "avg_returns": {
          name: round(float(np.mean(vals)), 4) if vals else None
          for name, vals in stats.items()
      },
  }


def make_eval_bot(name, model, args, device, seat, game_index):
  if name == "az_policy":
    return AZPolicyBot(model, name, device, args.value_scale)
  if name == "az_opponent":
    opponent_model = getattr(args, "_eval_opponent_model", None)
    if opponent_model is None:
      raise ValueError(
          "Opponent name az_opponent requires --eval-opponent-checkpoint"
      )
    return AZPolicyBot(
        opponent_model,
        name,
        device,
        getattr(args, "_eval_opponent_value_scale", args.value_scale),
    )
  if name == "az_belief_policy":
    return AZBeliefPolicyBot(
        model, name, device, args.value_scale, max(1, args.eval_belief_samples)
    )
  if name == "az_search":
    return AZSearchBot(model, name, device, args, max(1, args.eval_mcts_sims))
  if name == "az_belief_search":
    return AZBeliefSearchBot(
        model, name, device, args,
        max(1, args.eval_belief_samples),
        max(1, args.eval_belief_sims),
    )
  return make_bot(name, seed=args.seed + game_index * 13 + seat)


def match_outcome_scores(totals, final_round_returns):
  return [
      float(total) * 1000.0 + float(final_round)
      for total, final_round in zip(totals, final_round_returns)
  ]


def play_eval_round(round_game, bots, args, seed, match_totals=None, round_index=0):
  if seed is not None:
    np.random.seed(seed)
  state = round_game.new_initial_state()
  if getattr(args, "match_context", False) and match_totals is not None:
    state.set_match_context(match_totals, round_index)
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      action = int(np.random.choice(actions, p=probs))
    else:
      player = state.current_player()
      action = shared_prediction_action(state, player)
      if action is None:
        action = bots[player].step(state, player)
    state.apply_action(action)
  return state.returns(), state


def play_eval_match(seated, bots, args, initial_start, seed):
  totals = np.zeros(args.players, dtype=np.float32)
  final_round_returns = np.zeros(args.players, dtype=np.float32)
  paradoxes = np.zeros(args.players, dtype=np.int32)
  for round_index in range(args.players):
    start_player = (initial_start + round_index) % args.players
    round_game = make_game(args, start_player)
    prior_totals = np.copy(totals)
    returns, state = play_eval_round(
        round_game,
        bots,
        args,
        seed=seed + round_index * 9973,
        match_totals=prior_totals if getattr(args, "match_context", False) else None,
        round_index=round_index,
    )
    returns = np.array(returns, dtype=np.float32)
    if getattr(args, "match_context", False):
      totals = returns
      round_returns = returns - prior_totals
    else:
      raw_scores = (
          state.raw_round_scores()
          if hasattr(state, "raw_round_scores")
          else returns
      )
      round_returns = np.array(raw_scores, dtype=np.float32)
      totals += round_returns
    final_round_returns = round_returns
    paradoxes += np.array(
        getattr(state, "_has_paradoxed", [False] * args.players), dtype=np.int32
    )
  return totals, final_round_returns, paradoxes


def summarize_full_match_eval(
    names, players, ratings, totals_by_name, paradoxes_by_name
):
  games_by_name = {name: len(totals_by_name[name]) for name in names}
  avg_match_total = {
      name: round(float(np.mean(vals)), 4) if vals else None
      for name, vals in totals_by_name.items()
  }
  avg_match_total_se = {}
  avg_match_total_ci95 = {}
  for name, vals in totals_by_name.items():
    if not vals:
      avg_match_total_se[name] = None
      avg_match_total_ci95[name] = None
      continue
    values = np.array(vals, dtype=np.float32)
    if len(values) <= 1:
      se = 0.0
    else:
      se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    mean = float(np.mean(values))
    margin = 1.96 * se
    avg_match_total_se[name] = round(se, 4)
    avg_match_total_ci95[name] = [
        round(mean - margin, 4),
        round(mean + margin, 4),
    ]
  return {
      "full_match": True,
      "rounds_per_match": players,
      "ratings": {
          k: round(v, 2)
          for k, v in sorted(ratings.items(), key=lambda item: item[1],
                             reverse=True)
      },
      "avg_returns": avg_match_total,
      "avg_match_total": avg_match_total,
      "avg_match_total_se": avg_match_total_se,
      "avg_match_total_ci95": avg_match_total_ci95,
      "paradoxes_per_match": {
          name: (
              round(paradoxes_by_name[name] / games_by_name[name], 4)
              if games_by_name[name] else None
          )
          for name in names
      },
      "paradoxes_per_round": {
          name: (
              round(paradoxes_by_name[name] / (games_by_name[name] * players), 4)
              if games_by_name[name] else None
          )
          for name in names
      },
      "paradox_round_rate": {
          name: (
              round(paradoxes_by_name[name] / (games_by_name[name] * players), 4)
              if games_by_name[name] else None
          )
          for name in names
      },
      "games": games_by_name,
  }


def write_json_artifact(path, payload):
  artifact_path = Path(path)
  artifact_path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = artifact_path.with_name(artifact_path.name + ".tmp")
  tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  tmp_path.replace(artifact_path)


def wrap_eval_result(args, eval_result, complete=True, completed_games=None):
  wrapped = {
      "checkpoint": args.eval_checkpoint,
      "opponent_checkpoint": getattr(args, "eval_opponent_checkpoint", None),
      "eval_games": args.eval_games,
      "eval_mcts_sims": args.eval_mcts_sims,
      "eval_belief_samples": args.eval_belief_samples,
      "eval_belief_sims": args.eval_belief_sims,
      "eval_candidate": args.eval_candidate,
      "action_value_selection_weight": getattr(
          args, "action_value_selection_weight", 0.0
      ),
      "action_paradox_selection_penalty": getattr(
          args, "action_paradox_selection_penalty", 0.0
      ),
      "action_paradox_rerank_mode": getattr(
          args, "action_paradox_rerank_mode", "additive"
      ),
      "action_paradox_risk_threshold": getattr(
          args, "action_paradox_risk_threshold", 0.0
      ),
      "action_paradox_min_risk_margin": getattr(
          args, "action_paradox_min_risk_margin", 0.0
      ),
      "action_paradox_max_policy_log_gap": getattr(
          args, "action_paradox_max_policy_log_gap", 2.0
      ),
      "complete": bool(complete),
      "eval": eval_result,
  }
  if completed_games is not None:
    wrapped["completed_games"] = int(completed_games)
  return wrapped


def maybe_write_eval_progress(args, eval_result, completed_games):
  output_path = getattr(args, "eval_output_json", None)
  interval = int(getattr(args, "eval_progress_interval", 0) or 0)
  if not output_path or interval <= 0:
    return
  if completed_games < int(getattr(args, "eval_games", 0)):
    if completed_games % interval != 0:
      return
  write_json_artifact(
      output_path,
      wrap_eval_result(
          args, eval_result, complete=False, completed_games=completed_games
      ),
  )


def wrap_teacher_result(args, result, complete=True, completed_games=None):
  wrapped = {
      "teacher_checkpoint": getattr(args, "teacher_checkpoint", None),
      "teacher_games": int(getattr(args, "teacher_games", 0)),
      "teacher_mode": getattr(args, "teacher_mode", "policy"),
      "teacher_builtin_bot": str(
          getattr(args, "teacher_builtin_bot", "heuristic_safe14")
      ),
      "teacher_sims": int(getattr(args, "teacher_sims", 0)),
      "teacher_temperature": float(getattr(args, "teacher_temperature", 0.0)),
      "action_value_selection_weight": getattr(
          args, "action_value_selection_weight", 0.0
      ),
      "action_paradox_selection_penalty": getattr(
          args, "action_paradox_selection_penalty", 0.0
      ),
      "action_paradox_rerank_mode": getattr(
          args, "action_paradox_rerank_mode", "additive"
      ),
      "action_paradox_risk_threshold": getattr(
          args, "action_paradox_risk_threshold", 0.0
      ),
      "action_paradox_min_risk_margin": getattr(
          args, "action_paradox_min_risk_margin", 0.0
      ),
      "action_paradox_max_policy_log_gap": getattr(
          args, "action_paradox_max_policy_log_gap", 2.0
      ),
      "action_value_rerank_phases": getattr(
          args, "action_value_rerank_phases", ""
      ),
      "q_policy_teacher_confirm_rollouts": int(
          getattr(args, "q_policy_teacher_confirm_rollouts", 0)
      ),
      "q_policy_teacher_confirm_min_paradox_improvement": float(
          getattr(args, "q_policy_teacher_confirm_min_paradox_improvement", 1e-6)
      ),
      "q_policy_teacher_confirm_min_score_margin": float(
          getattr(args, "q_policy_teacher_confirm_min_score_margin", 0.0)
      ),
      "rollout_select_teacher_rollouts": int(
          getattr(args, "rollout_select_teacher_rollouts", 1)
      ),
      "rollout_select_teacher_min_paradox_improvement": float(
          getattr(args, "rollout_select_teacher_min_paradox_improvement", 1e-6)
      ),
      "rollout_select_teacher_min_score_margin": float(
          getattr(args, "rollout_select_teacher_min_score_margin", 0.0)
      ),
      "rollout_select_teacher_continuation_role": str(
          getattr(args, "rollout_select_teacher_continuation_role", "learner")
      ),
      "full_match_training": bool(getattr(args, "full_match_training", False)),
      "complete": bool(complete),
      "teacher": result,
  }
  if completed_games is not None:
    wrapped["completed_games"] = int(completed_games)
  return wrapped


def teacher_progress_interval(args):
  interval = int(getattr(args, "teacher_progress_interval", 0) or 0)
  if interval > 0:
    return interval
  return int(getattr(args, "generate_replay_progress_interval", 0) or 0)


def maybe_write_teacher_progress(args, progress_row):
  output_path = getattr(args, "teacher_output_json", None)
  interval = teacher_progress_interval(args)
  if not output_path or interval <= 0:
    return
  completed_games = int(progress_row.get("completed_games", 0))
  total_games = int(progress_row.get("total_games", 0))
  if completed_games < total_games and completed_games % interval != 0:
    return
  write_json_artifact(
      output_path,
      wrap_teacher_result(
          args,
          progress_row,
          complete=False,
          completed_games=completed_games,
      ),
  )


def evaluate_checkpoint_full_match(model, game, args, device):
  del game
  names = ["az_policy", "heuristic", "heuristic_target2", "random"]
  if args.eval_mcts_sims > 0:
    names.append("az_search")
  if args.eval_belief_samples > 0:
    names.append("az_belief_policy")
    names.append("az_belief_search")
  ratings = {name: 1000.0 for name in names}
  totals_by_name = {name: [] for name in names}
  paradoxes_by_name = {name: 0 for name in names}
  for idx in range(args.eval_games):
    seated = [names[(idx + offset) % len(names)] for offset in range(args.players)]
    shift = idx % args.players
    seated = seated[-shift:] + seated[:-shift] if shift else seated
    bots = [
        make_eval_bot(name, model, args, device, seat, idx)
        for seat, name in enumerate(seated)
    ]
    totals, final_round_returns, paradoxes = play_eval_match(
        seated,
        bots,
        args,
        initial_start=idx % args.players,
        seed=args.seed + 300000 + idx * 37,
    )
    outcome_scores = match_outcome_scores(totals, final_round_returns)
    multiplayer_elo_update(ratings, seated, outcome_scores, 24.0)
    for seat, name in enumerate(seated):
      totals_by_name[name].append(float(totals[seat]))
      paradoxes_by_name[name] += int(paradoxes[seat])
    partial = summarize_full_match_eval(
        names, args.players, ratings, totals_by_name, paradoxes_by_name
    )
    maybe_write_eval_progress(args, partial, idx + 1)
  return summarize_full_match_eval(
      names, args.players, ratings, totals_by_name, paradoxes_by_name
  )


def evaluate_candidate(model, game, args, device):
  if getattr(args, "eval_full_match", False):
    return evaluate_candidate_full_match(model, game, args, device)
  candidate = args.eval_candidate
  opponents = [name.strip() for name in args.eval_opponents.split(",") if name.strip()]
  if not candidate:
    raise ValueError("--eval-candidate is required for candidate evaluation")
  if len(opponents) < game.num_players() - 1:
    raise ValueError("Need at least players-1 opponents for candidate evaluation")
  names = [candidate] + opponents
  ratings = {name: 1000.0 for name in names}
  stats = {name: [] for name in names}
  games_by_name = {name: 0 for name in names}
  for idx in range(args.eval_games):
    round_game = (
        make_game(args, idx % args.players)
        if args.eval_rotate_start_player else game
    )
    opponent_seats = [
        opponents[(idx + offset) % len(opponents)]
        for offset in range(game.num_players() - 1)
    ]
    seated = [candidate] + opponent_seats
    shift = idx % game.num_players()
    seated = seated[-shift:] + seated[:-shift] if shift else seated
    bots = [
        make_eval_bot(name, model, args, device, seat, idx)
        for seat, name in enumerate(seated)
    ]
    returns, _ = play_game(round_game, bots, seed=args.seed + 200000 + idx)
    multiplayer_elo_update(ratings, seated, returns, 24.0)
    for seat, name in enumerate(seated):
      stats[name].append(float(returns[seat]))
      games_by_name[name] += 1
    partial = {
        "candidate": candidate,
        "opponents": opponents,
        "ratings": {k: round(v, 2) for k, v in sorted(
            ratings.items(), key=lambda item: item[1], reverse=True)},
        "avg_returns": {
            name: round(float(np.mean(vals)), 4) if vals else None
            for name, vals in stats.items()
        },
        "games": games_by_name,
    }
    maybe_write_eval_progress(args, partial, idx + 1)
  return {
      "candidate": candidate,
      "opponents": opponents,
      "ratings": {k: round(v, 2) for k, v in sorted(
          ratings.items(), key=lambda item: item[1], reverse=True)},
      "avg_returns": {
          name: round(float(np.mean(vals)), 4) if vals else None
          for name, vals in stats.items()
      },
      "games": games_by_name,
  }


def evaluate_candidate_full_match(model, game, args, device):
  candidate = args.eval_candidate
  opponents = [name.strip() for name in args.eval_opponents.split(",") if name.strip()]
  if not candidate:
    raise ValueError("--eval-candidate is required for candidate evaluation")
  if len(opponents) < game.num_players() - 1:
    raise ValueError("Need at least players-1 opponents for candidate evaluation")
  names = [candidate] + opponents
  ratings = {name: 1000.0 for name in names}
  totals_by_name = {name: [] for name in names}
  paradoxes_by_name = {name: 0 for name in names}
  for idx in range(args.eval_games):
    opponent_seats = [
        opponents[(idx + offset) % len(opponents)]
        for offset in range(game.num_players() - 1)
    ]
    seated = [candidate] + opponent_seats
    shift = idx % game.num_players()
    seated = seated[-shift:] + seated[:-shift] if shift else seated
    bots = [
        make_eval_bot(name, model, args, device, seat, idx)
        for seat, name in enumerate(seated)
    ]
    totals, final_round_returns, paradoxes = play_eval_match(
        seated,
        bots,
        args,
        initial_start=idx % args.players,
        seed=args.seed + 400000 + idx * 37,
    )
    outcome_scores = match_outcome_scores(totals, final_round_returns)
    multiplayer_elo_update(ratings, seated, outcome_scores, 24.0)
    for seat, name in enumerate(seated):
      totals_by_name[name].append(float(totals[seat]))
      paradoxes_by_name[name] += int(paradoxes[seat])
    partial = summarize_full_match_eval(
        names, args.players, ratings, totals_by_name, paradoxes_by_name
    )
    partial["candidate"] = candidate
    partial["opponents"] = opponents
    maybe_write_eval_progress(args, partial, idx + 1)
  result = summarize_full_match_eval(
      names, args.players, ratings, totals_by_name, paradoxes_by_name
  )
  result["candidate"] = candidate
  result["opponents"] = opponents
  return result


def main():
  args = parse_args()
  if (
      getattr(args, "anchor_top_action_loss_weight", 0.0) > 0
      and not args.anchor_checkpoint
  ):
    raise ValueError(
        "--anchor-top-action-loss-weight requires --anchor-checkpoint"
    )
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device=mps requested, but torch MPS is not available")
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  game = make_game(args, 0)
  if args.eval_checkpoint:
    model, _, saved_args = load_model_payload(args.eval_checkpoint, game, args, device)
    args.value_scale = saved_args.get("value_scale", args.value_scale)
    args.arch = saved_args.get("arch", args.arch)
    args.match_context = saved_args.get("match_context", args.match_context)
    if args.eval_opponent_checkpoint:
      opponent_model, _, opponent_saved_args = load_model_payload(
          args.eval_opponent_checkpoint, game, args, device
      )
      opponent_model.eval()
      args._eval_opponent_model = opponent_model
      args._eval_opponent_value_scale = opponent_saved_args.get(
          "value_scale", args.value_scale
      )
    eval_result = (
        evaluate_candidate(model, game, args, device)
        if args.eval_candidate else evaluate_checkpoint(model, game, args, device)
    )
    result = wrap_eval_result(
        args, eval_result, complete=True, completed_games=args.eval_games
    )
    if args.eval_output_json:
      write_json_artifact(args.eval_output_json, result)
    print(json.dumps(result, indent=2), flush=True)
    return
  resume_payload = None
  if args.resume_checkpoint:
    resume_payload = torch.load(args.resume_checkpoint, map_location=device)
    saved_args = resume_payload.get("args", {})
    if args.resume_architecture == "checkpoint":
      args.width = saved_args.get("width", args.width)
      args.depth = saved_args.get("depth", args.depth)
      args.arch = saved_args.get("arch", args.arch)
      args.separate_action_value_encoder = saved_args.get(
          "separate_action_value_encoder",
          args.separate_action_value_encoder,
      )
      args.separate_action_paradox_encoder = saved_args.get(
          "separate_action_paradox_encoder",
          args.separate_action_paradox_encoder,
      )
    args.value_scale = saved_args.get("value_scale", args.value_scale)
  model = AZNet(
      game.observation_tensor_shape()[0],
      game.num_distinct_actions(),
      args.players,
      args.width,
      args.depth,
      args.arch,
      args.separate_action_value_encoder,
      args.separate_action_paradox_encoder,
  ).to(device)
  if resume_payload is not None:
    load_compatible_state_dict(model, resume_payload["model"])
    initialize_missing_action_value_stack_from_policy(
        model, resume_payload["model"]
    )
    initialize_missing_action_paradox_stack_from_policy(
        model, resume_payload["model"]
    )
  if getattr(args, "reset_action_value_head", False):
    reset_action_value_head(model)
  if getattr(args, "reset_action_paradox_head", False):
    reset_action_paradox_head(model)
  trainable_params = configure_trainable_parameters(model, args)
  if not trainable_params:
    raise ValueError("No trainable parameters selected")
  training_parameter_report = parameter_training_report(model)
  frozen_params_before_training = frozen_parameter_snapshot(model)
  optimizer = torch.optim.Adam(
      trainable_params,
      lr=args.lr,
      weight_decay=max(0.0, float(getattr(args, "weight_decay", 0.0))),
  )
  anchor_model = None
  if args.anchor_checkpoint:
    anchor_model, _, _ = load_model_payload(args.anchor_checkpoint, game, args, device)
    anchor_model.eval()
    for param in anchor_model.parameters():
      param.requires_grad = False
  replay = deque(maxlen=args.buffer_size)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  metrics = (
      []
      if getattr(args, "discard_resume_metrics", False)
      else list(resume_payload.get("metrics", [])) if resume_payload else []
  )
  parameter_row = {
      "iteration": "parameter_training_audit",
      **training_parameter_report,
  }
  metrics.append(parameter_row)
  print(json.dumps(parameter_row, indent=2), flush=True)

  if args.load_replay:
    replay_paths = split_csv(args.load_replay)
    replay_counts = {}
    replay_metadata = []
    for replay_path in replay_paths:
      examples, metadata = load_replay(replay_path, return_metadata=True)
      replay_counts[replay_path] = len(examples)
      replay_metadata.append(metadata)
      for example in examples:
        replay.append(example)
    replay, action_value_filter = filter_action_value_labels_for_training(
        replay, args
    )
    load_row = {
        "iteration": "load_replay",
        "load_replay": args.load_replay,
        "load_replay_paths": replay_paths,
        "load_replay_counts": replay_counts,
        "load_replay_metadata": replay_metadata,
        "replay_size": len(replay),
    }
    if action_value_filter is not None:
      load_row["action_value_filter"] = action_value_filter
    metrics.append(load_row)
    print(json.dumps(load_row, indent=2), flush=True)

  if args.loaded_replay_train_steps > 0:
    if not replay:
      raise ValueError("--loaded-replay-train-steps requires non-empty --load-replay")
    loaded_args = argparse.Namespace(**vars(args))
    loaded_args = apply_replay_metadata_to_args(
        loaded_args, locals().get("replay_metadata", [])
    )
    validation_replay, validation_split = load_action_value_validation_replay(args)
    if validation_split is not None:
      loaded_args = apply_replay_metadata_to_args(
          loaded_args,
          validation_split.get("action_value_validation_replay_metadata", []),
      )
    if validation_replay is None:
      train_replay, validation_replay, validation_split = (
          split_action_value_validation_replay(replay, args)
      )
    else:
      train_replay = list(replay)
    if len(train_replay) < args.batch_size:
      raise ValueError(
          "Loaded replay training has fewer train rows than --batch-size "
          f"after validation split ({len(train_replay)} < {args.batch_size}). "
          "Lower --batch-size or reduce --action-value-validation-fraction."
      )
    if (
        args.train_snapshot_interval > 0
        and args.loaded_replay_train_steps > args.train_snapshot_interval
    ):
      losses = []
      trained_steps = 0
      best_snapshot_score = None
      best_snapshot_row = None
      best_snapshot_path = out_dir / "checkpoint_0000_loaded_replay_best.pt"
      while trained_steps < args.loaded_replay_train_steps:
        chunk_args = argparse.Namespace(**vars(loaded_args))
        chunk_args.train_steps = min(
            args.train_snapshot_interval,
            args.loaded_replay_train_steps - trained_steps,
        )
        chunk_loss = train_steps(
            model, optimizer, list(train_replay), chunk_args, device, anchor_model
        )
        trained_steps += chunk_args.train_steps
        losses.append(chunk_loss)
        snapshot_eval = evaluate_checkpoint(model, game, args, device)
        snapshot_path = (
            out_dir / f"checkpoint_0000_loaded_replay_step_{trained_steps:04d}.pt"
        )
        snapshot_row = loaded_replay_report_row(
            model,
            anchor_model,
            replay,
            args,
            chunk_args,
            device,
            chunk_loss,
            trained_steps,
            eval_result=snapshot_eval,
            checkpoint_path=snapshot_path,
            iteration="loaded_replay_train_snapshot",
            validation_replay=validation_replay,
            validation_split=validation_split,
        )
        add_frozen_parameter_integrity_or_raise(
            snapshot_row, model, frozen_params_before_training, args
        )
        metrics.append(snapshot_row)
        print(json.dumps(snapshot_row, indent=2), flush=True)
        save_training_artifacts(model, args, metrics, replay, snapshot_path, out_dir)
        if getattr(args, "loaded_replay_save_best", False):
          snapshot_score = loaded_replay_validation_score(snapshot_row, args)
          if (
              snapshot_score is not None
              and (
                  best_snapshot_score is None
                  or snapshot_score > best_snapshot_score
              )
          ):
            best_snapshot_score = snapshot_score
            best_snapshot_row = {
                "iteration": "loaded_replay_best_snapshot",
                "loaded_replay_best_metric": getattr(
                    args, "loaded_replay_best_metric", "validation_top1"
                ),
                "loaded_replay_best_score": round(float(snapshot_score), 6),
                "loaded_replay_best_steps": int(trained_steps),
                "loaded_replay_best_source_checkpoint": str(snapshot_path),
                "checkpoint": str(best_snapshot_path),
            }
            metrics.append(best_snapshot_row)
            print(json.dumps(best_snapshot_row, indent=2), flush=True)
            save_training_artifacts(
                model, args, metrics, replay, best_snapshot_path, out_dir
            )
      loss = mean_loss(losses)
      eval_result = snapshot_eval
    else:
      loaded_args.train_steps = args.loaded_replay_train_steps
      loss = train_steps(
          model, optimizer, list(train_replay), loaded_args, device, anchor_model
      )
      eval_result = evaluate_checkpoint(model, game, args, device)
    checkpoint_path = out_dir / "checkpoint_0000_loaded_replay.pt"
    loaded_row = loaded_replay_report_row(
        model,
        anchor_model,
        replay,
        args,
        loaded_args,
        device,
        loss,
        args.loaded_replay_train_steps,
        eval_result=eval_result,
        checkpoint_path=checkpoint_path,
        validation_replay=validation_replay,
        validation_split=validation_split,
    )
    add_frozen_parameter_integrity_or_raise(
        loaded_row, model, frozen_params_before_training, args
    )
    metrics.append(loaded_row)
    print(json.dumps(loaded_row, indent=2), flush=True)
    save_training_artifacts(model, args, metrics, replay, checkpoint_path, out_dir)

  if args.generate_replay_games > 0:
    generation = collect_self_play_games(
        model, args, device, args.generate_replay_games
    )
    replay.extend(generation["examples"])
    generate_row = {
        "iteration": "generate_replay",
        "generate_replay_games": args.generate_replay_games,
        "full_match_training": args.full_match_training,
        "generation_workers": generation["workers"],
        "generation_elapsed_sec": round(generation["elapsed_sec"], 3),
        "start_counts": generation["start_counts"],
        "replay_size": len(replay),
        "self_play_avg_returns": np.mean(
            np.array(generation["returns"]), axis=0
        ).tolist(),
        "replay_path": str(out_dir / "replay_latest.npz"),
    }
    generate_row["counterfactual_label_coverage"] = (
        counterfactual_label_coverage_report(list(replay))
    )
    save_replay(
        list(replay), out_dir / "replay_latest.npz",
        metadata=replay_metadata_from_args(args),
    )
    maybe_add_action_report(generate_row, model, replay, args, device)
    maybe_add_action_value_report(generate_row, model, replay, args, device)
    maybe_add_prediction_hit_report(generate_row, model, replay, args, device)
    metrics.append(generate_row)
    print(json.dumps(generate_row, indent=2), flush=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }, out_dir / "checkpoint_0000_generated_replay.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if args.generate_replay_only:
      return

  if args.teacher_games > 0:
    if (
        not args.teacher_checkpoint
        and getattr(args, "teacher_mode", "policy") != "builtin_policy"
    ):
      raise ValueError("--teacher-checkpoint is required with --teacher-games")
    if getattr(args, "teacher_mode", "policy") == "builtin_policy":
      teacher_model = None
    else:
      teacher_model, _, _ = load_model_payload(
          args.teacher_checkpoint, game, args, device
      )
      teacher_model.eval()
    teacher_generation = collect_teacher_games(
        teacher_model, args, device, args.teacher_games
    )
    teacher_returns = teacher_generation["returns"]
    teacher_start_counts = teacher_generation["start_counts"]
    replay.extend(teacher_generation["examples"])
    teacher_args = argparse.Namespace(**vars(args))
    teacher_args.train_steps = (
        args.teacher_train_steps if args.teacher_train_steps > 0
        else args.train_steps
    )
    teacher_snapshot_rows = []
    if args.train_snapshot_interval > 0 and teacher_args.train_steps > 0:
      losses = []
      trained_steps = 0
      while trained_steps < teacher_args.train_steps:
        chunk_args = argparse.Namespace(**vars(teacher_args))
        chunk_args.train_steps = min(
            args.train_snapshot_interval,
            teacher_args.train_steps - trained_steps,
        )
        chunk_loss = train_steps(
            model, optimizer, list(replay), chunk_args, device, anchor_model
        )
        trained_steps += chunk_args.train_steps
        losses.append(chunk_loss)
        snapshot_eval = evaluate_checkpoint(model, game, args, device)
        snapshot_path = (
            out_dir / f"checkpoint_0000_teacher_step_{trained_steps:04d}.pt"
        )
        snapshot_row = {
            "iteration": "teacher_train_snapshot",
            "train_steps_completed": trained_steps,
            "checkpoint": str(snapshot_path),
            "loss": chunk_loss,
            "eval": snapshot_eval,
        }
        maybe_add_action_report(
            snapshot_row, model, replay, chunk_args, device
        )
        maybe_add_action_value_report(
            snapshot_row, model, replay, chunk_args, device
        )
        maybe_add_prediction_hit_report(
            snapshot_row, model, replay, chunk_args, device
        )
        maybe_add_policy_target_report(
            snapshot_row,
            model,
            replay,
            chunk_args,
            device,
            name="teacher_policy_target_report",
        )
        maybe_add_anchor_policy_report(
            snapshot_row, model, anchor_model, replay, chunk_args, device
        )
        teacher_snapshot_rows.append(snapshot_row)
        metrics.append(snapshot_row)
        print(json.dumps(snapshot_row, indent=2), flush=True)
        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        }, snapshot_path)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
      loss = mean_loss(losses)
      eval_result = (
          teacher_snapshot_rows[-1]["eval"] if teacher_snapshot_rows else None
      )
    else:
      loss = train_steps(
          model, optimizer, list(replay), teacher_args, device, anchor_model
      )
      eval_result = evaluate_checkpoint(model, game, args, device)
    teacher_row = {
        "iteration": "teacher_distill",
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_games": args.teacher_games,
        "full_match_training": args.full_match_training,
        "teacher_temperature": args.teacher_temperature,
        "teacher_mode": args.teacher_mode,
        "teacher_builtin_bot": str(
            getattr(args, "teacher_builtin_bot", "heuristic_safe14")
        ),
        "teacher_sims": args.teacher_sims,
        "teacher_min_target_prob": float(args.teacher_min_target_prob),
        "teacher_min_target_margin": float(args.teacher_min_target_margin),
        "teacher_max_target_entropy": float(args.teacher_max_target_entropy),
        "teacher_belief_samples": int(args.teacher_belief_samples),
        "teacher_belief_source": str(getattr(args, "teacher_belief_source", "infostate")),
        "teacher_belief_candidates": int(getattr(args, "teacher_belief_candidates", 8)),
        "teacher_belief_policy_temperature": float(
            getattr(args, "teacher_belief_policy_temperature", 1.0)
        ),
        "teacher_belief_uniform_mix": float(
            getattr(args, "teacher_belief_uniform_mix", 0.15)
        ),
        "teacher_belief_ref_policy_mix": str(
            getattr(args, "teacher_belief_ref_policy_mix", "model:1.0")
        ),
        "action_paradox_selection_penalty": float(
            getattr(args, "action_paradox_selection_penalty", 0.0)
        ),
        "action_paradox_rerank_mode": str(
            getattr(args, "action_paradox_rerank_mode", "additive")
        ),
        "action_paradox_risk_threshold": float(
            getattr(args, "action_paradox_risk_threshold", 0.0)
        ),
        "action_paradox_min_risk_margin": float(
            getattr(args, "action_paradox_min_risk_margin", 0.0)
        ),
        "action_paradox_max_policy_log_gap": float(
            getattr(args, "action_paradox_max_policy_log_gap", 2.0)
        ),
        "action_value_rerank_phases": str(
            getattr(args, "action_value_rerank_phases", "")
        ),
        "q_policy_teacher_confirm_rollouts": int(
            getattr(args, "q_policy_teacher_confirm_rollouts", 0)
        ),
        "q_policy_teacher_confirm_min_paradox_improvement": float(
            getattr(args, "q_policy_teacher_confirm_min_paradox_improvement", 1e-6)
        ),
        "q_policy_teacher_confirm_min_score_margin": float(
            getattr(args, "q_policy_teacher_confirm_min_score_margin", 0.0)
        ),
        "rollout_select_teacher_rollouts": int(
            getattr(args, "rollout_select_teacher_rollouts", 1)
        ),
        "rollout_select_teacher_min_paradox_improvement": float(
            getattr(args, "rollout_select_teacher_min_paradox_improvement", 1e-6)
        ),
        "rollout_select_teacher_min_score_margin": float(
            getattr(args, "rollout_select_teacher_min_score_margin", 0.0)
        ),
        "rollout_select_teacher_continuation_role": str(
            getattr(args, "rollout_select_teacher_continuation_role", "learner")
        ),
        "generation_workers": teacher_generation["workers"],
        "generation_elapsed_sec": round(teacher_generation["elapsed_sec"], 3),
        "teacher_stats": teacher_generation.get("teacher_stats", {}),
        "start_counts": teacher_start_counts,
        "replay_size": len(replay),
        "loss": loss,
        "teacher_avg_returns": np.mean(np.array(teacher_returns), axis=0).tolist(),
        "eval": eval_result,
        "train_snapshots": teacher_snapshot_rows,
    }
    maybe_add_action_report(teacher_row, model, replay, teacher_args, device)
    maybe_add_action_value_report(
        teacher_row, model, replay, teacher_args, device
    )
    maybe_add_prediction_hit_report(
        teacher_row, model, replay, teacher_args, device
    )
    maybe_add_policy_target_report(
        teacher_row,
        model,
        replay,
        teacher_args,
        device,
        name="teacher_policy_target_report",
    )
    maybe_add_anchor_policy_report(
        teacher_row, model, anchor_model, replay, teacher_args, device
    )
    metrics.append(teacher_row)
    if args.teacher_output_json:
      write_json_artifact(
          args.teacher_output_json,
          wrap_teacher_result(
              args,
              teacher_row,
              complete=True,
              completed_games=args.teacher_games,
          ),
      )
    print(json.dumps(teacher_row, indent=2), flush=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }, out_dir / "checkpoint_0000_teacher.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_replay(
        list(replay), out_dir / "replay_latest.npz",
        metadata=replay_metadata_from_args(args),
    )
    if int(getattr(args, "iterations", 0)) <= 0 and args.league_games <= 0:
      return

  if args.league_games > 0:
    if not args.league_checkpoint:
      raise ValueError("--league-checkpoint is required with --league-games")
    league_checkpoints = split_csv(args.league_checkpoint)
    league_models = []
    for checkpoint in league_checkpoints:
      league_model, _, _ = load_model_payload(checkpoint, game, args, device)
      league_model.eval()
      league_models.append(league_model)
    league_opponents = (
        league_models[0] if len(league_models) == 1 else league_models
    )
    league_bots = split_csv(args.league_bots)
    league_generation = collect_league_games(
        model, league_opponents, league_bots, args, device
    )
    league_returns = league_generation["returns"]
    learner_returns = league_generation["learner_returns"]
    league_start_counts = league_generation["start_counts"]
    learner_seat_counts = league_generation["learner_seat_counts"]
    paradox_sums = league_generation["paradox_sums"]
    replay.extend(league_generation["examples"])
    league_args = argparse.Namespace(**vars(args))
    league_args.train_steps = (
        args.league_train_steps if args.league_train_steps > 0
        else args.train_steps
    )
    league_snapshot_rows = []
    if args.train_snapshot_interval > 0 and league_args.train_steps > 0:
      losses = []
      trained_steps = 0
      while trained_steps < league_args.train_steps:
        chunk_args = argparse.Namespace(**vars(league_args))
        chunk_args.train_steps = min(
            args.train_snapshot_interval,
            league_args.train_steps - trained_steps,
        )
        chunk_loss = train_steps(
            model, optimizer, list(replay), chunk_args, device, anchor_model
        )
        trained_steps += chunk_args.train_steps
        losses.append(chunk_loss)
        snapshot_eval = evaluate_checkpoint(model, game, args, device)
        snapshot_path = (
            out_dir / f"checkpoint_0000_league_step_{trained_steps:04d}.pt"
        )
        snapshot_row = {
            "iteration": "league_train_snapshot",
            "train_steps_completed": trained_steps,
            "checkpoint": str(snapshot_path),
            "loss": chunk_loss,
            "eval": snapshot_eval,
        }
        maybe_add_action_report(
            snapshot_row, model, replay, chunk_args, device
        )
        maybe_add_action_value_report(
            snapshot_row, model, replay, chunk_args, device
        )
        maybe_add_prediction_hit_report(
            snapshot_row, model, replay, chunk_args, device
        )
        maybe_add_policy_target_report(
            snapshot_row,
            model,
            replay,
            chunk_args,
            device,
            name="league_policy_target_report",
        )
        maybe_add_anchor_policy_report(
            snapshot_row, model, anchor_model, replay, chunk_args, device
        )
        league_snapshot_rows.append(snapshot_row)
        metrics.append(snapshot_row)
        print(json.dumps(snapshot_row, indent=2), flush=True)
        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        }, snapshot_path)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
      loss = mean_loss(losses)
      eval_result = (
          league_snapshot_rows[-1]["eval"] if league_snapshot_rows else None
      )
    else:
      loss = train_steps(
          model, optimizer, list(replay), league_args, device, anchor_model
      )
      eval_result = evaluate_checkpoint(model, game, args, device)
    league_row = {
        "iteration": "league",
        "league_games": args.league_games,
        "league_checkpoint": args.league_checkpoint,
        "league_checkpoints": league_checkpoints,
        "league_bots": league_bots,
        "league_opponent_mode": args.league_opponent_mode,
        "checkpoint": str(out_dir / "checkpoint_0000_league.pt"),
        "full_match_training": True,
        "generation_workers": league_generation["workers"],
        "generation_elapsed_sec": round(league_generation["elapsed_sec"], 3),
        "start_counts": league_start_counts,
        "learner_seat_counts": learner_seat_counts,
        "replay_size": len(replay),
        "loss": loss,
        "league_avg_returns": np.mean(np.array(league_returns), axis=0).tolist(),
        "learner_avg_return": float(np.mean(learner_returns)),
        "paradoxes_per_match_by_seat": (
            paradox_sums / max(1, args.league_games)
        ).tolist(),
        "eval": eval_result,
        "train_snapshots": league_snapshot_rows,
    }
    league_row["counterfactual_label_coverage"] = (
        counterfactual_label_coverage_report(list(replay))
    )
    maybe_add_action_report(league_row, model, replay, league_args, device)
    maybe_add_action_value_report(league_row, model, replay, league_args, device)
    maybe_add_prediction_hit_report(
        league_row, model, replay, league_args, device
    )
    maybe_add_policy_target_report(
        league_row,
        model,
        replay,
        league_args,
        device,
        name="league_policy_target_report",
    )
    maybe_add_anchor_policy_report(
        league_row, model, anchor_model, replay, league_args, device
    )
    metrics.append(league_row)
    print(json.dumps(league_row, indent=2), flush=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }, out_dir / "checkpoint_0000_league.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_replay(
        list(replay), out_dir / "replay_latest.npz",
        metadata=replay_metadata_from_args(league_args),
    )

  if args.bootstrap_games > 0:
    bot_names = [name.strip() for name in args.bootstrap_bots.split(",")
                 if name.strip()]
    if not bot_names:
      raise ValueError("--bootstrap-bots must include at least one bot name")
    bootstrap_returns = []
    bootstrap_start_counts = [0] * args.players
    for game_index in range(args.bootstrap_games):
      start_player = (
          int(np.random.randint(args.players)) if args.random_start_player else 0
      )
      bootstrap_start_counts[start_player] += 1
      round_game = make_game(args, start_player)
      examples, terminal_state = bootstrap_game(
          round_game, bot_names, args, game_index
      )
      replay.extend(examples)
      bootstrap_returns.append(terminal_state.returns())
    bootstrap_args = argparse.Namespace(**vars(args))
    bootstrap_args.train_steps = (
        args.bootstrap_train_steps if args.bootstrap_train_steps > 0
        else args.train_steps
    )
    loss = train_steps(
        model, optimizer, list(replay), bootstrap_args, device, anchor_model
    )
    bootstrap_row = {
        "iteration": 0,
        "bootstrap_games": args.bootstrap_games,
        "bootstrap_bots": bot_names,
        "start_counts": bootstrap_start_counts,
        "replay_size": len(replay),
        "loss": loss,
        "bootstrap_avg_returns": np.mean(np.array(bootstrap_returns), axis=0).tolist(),
        "eval": evaluate_checkpoint(model, game, args, device),
    }
    maybe_add_action_report(bootstrap_row, model, replay, bootstrap_args, device)
    maybe_add_action_value_report(
        bootstrap_row, model, replay, bootstrap_args, device
    )
    maybe_add_prediction_hit_report(
        bootstrap_row, model, replay, bootstrap_args, device
    )
    maybe_add_policy_target_report(
        bootstrap_row,
        model,
        replay,
        bootstrap_args,
        device,
        name="heuristic_imitation_report",
        one_hot_only=True,
    )
    metrics.append(bootstrap_row)
    print(json.dumps(bootstrap_row, indent=2), flush=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }, out_dir / "checkpoint_0000_bootstrap.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_replay(
        list(replay), out_dir / "replay_latest.npz",
        metadata=replay_metadata_from_args(bootstrap_args),
    )

  if args.replay_warmup_games > 0:
    warmup_generation = collect_self_play_games(
        model, args, device, args.replay_warmup_games
    )
    warmup_returns = warmup_generation["returns"]
    warmup_start_counts = warmup_generation["start_counts"]
    replay.extend(warmup_generation["examples"])
    warmup_args = argparse.Namespace(**vars(args))
    warmup_args.train_steps = args.replay_warmup_train_steps
    loss = train_steps(
        model, optimizer, list(replay), warmup_args, device, anchor_model
    )
    warmup_row = {
        "iteration": "warmup",
        "replay_warmup_games": args.replay_warmup_games,
        "generation_workers": warmup_generation["workers"],
        "generation_elapsed_sec": round(warmup_generation["elapsed_sec"], 3),
        "start_counts": warmup_start_counts,
        "replay_size": len(replay),
        "loss": loss,
        "warmup_avg_returns": np.mean(np.array(warmup_returns), axis=0).tolist(),
        "eval": evaluate_checkpoint(model, game, args, device),
    }
    maybe_add_action_report(warmup_row, model, replay, warmup_args, device)
    maybe_add_action_value_report(warmup_row, model, replay, warmup_args, device)
    maybe_add_prediction_hit_report(
        warmup_row, model, replay, warmup_args, device
    )
    metrics.append(warmup_row)
    print(json.dumps(warmup_row, indent=2), flush=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }, out_dir / "checkpoint_0000_warmup.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_replay(
        list(replay), out_dir / "replay_latest.npz",
        metadata=replay_metadata_from_args(warmup_args),
    )

  for iteration in range(1, args.iterations + 1):
    generation = collect_self_play_games(model, args, device, args.games_per_iter)
    game_returns = generation["returns"]
    start_counts = generation["start_counts"]
    replay.extend(generation["examples"])
    snapshot_rows = []
    if args.train_snapshot_interval > 0 and args.train_steps > 0:
      losses = []
      trained_steps = 0
      while trained_steps < args.train_steps:
        chunk_args = argparse.Namespace(**vars(args))
        chunk_args.train_steps = min(
            args.train_snapshot_interval, args.train_steps - trained_steps
        )
        chunk_loss = train_steps(
            model, optimizer, list(replay), chunk_args, device, anchor_model
        )
        trained_steps += chunk_args.train_steps
        losses.append(chunk_loss)
        snapshot_eval = evaluate_checkpoint(model, game, args, device)
        snapshot_path = (
            out_dir / f"checkpoint_{iteration:04d}_step_{trained_steps:04d}.pt"
        )
        snapshot_row = {
            "iteration": "train_snapshot",
            "parent_iteration": iteration,
            "train_steps_completed": trained_steps,
            "checkpoint": str(snapshot_path),
            "loss": chunk_loss,
            "eval": snapshot_eval,
        }
        maybe_add_action_report(snapshot_row, model, replay, chunk_args, device)
        maybe_add_action_value_report(
            snapshot_row, model, replay, chunk_args, device
        )
        maybe_add_prediction_hit_report(
            snapshot_row, model, replay, chunk_args, device
        )
        snapshot_rows.append(snapshot_row)
        metrics.append(snapshot_row)
        print(json.dumps(snapshot_row, indent=2), flush=True)
        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        }, snapshot_path)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
      loss = mean_loss(losses)
      eval_result = snapshot_rows[-1]["eval"] if snapshot_rows else None
    else:
      loss = train_steps(model, optimizer, list(replay), args, device, anchor_model)
      eval_result = evaluate_checkpoint(model, game, args, device)
    row = {
        "iteration": iteration,
        "self_play_games": args.games_per_iter,
        "full_match_training": args.full_match_training,
        "generation_workers": generation["workers"],
        "generation_elapsed_sec": round(generation["elapsed_sec"], 3),
        "start_counts": start_counts,
        "replay_size": len(replay),
        "loss": loss,
        "self_play_avg_returns": np.mean(np.array(game_returns), axis=0).tolist(),
        "eval": eval_result,
        "train_snapshots": snapshot_rows,
    }
    maybe_add_action_report(row, model, replay, args, device)
    maybe_add_action_value_report(row, model, replay, args, device)
    maybe_add_prediction_hit_report(row, model, replay, args, device)
    metrics.append(row)
    print(json.dumps(row, indent=2), flush=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }, out_dir / f"checkpoint_{iteration:04d}.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
  save_replay(
      list(replay), out_dir / "replay_latest.npz",
      metadata=replay_metadata_from_args(args),
  )


if __name__ == "__main__":
  main()
