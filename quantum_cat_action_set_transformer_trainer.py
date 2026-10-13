#!/usr/bin/env python3
"""Train a set-aware legal-action transformer for table-liveness overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    APPENDED_ACTION_FEATURE_INDEX,
)
from quantum_cat_legal_set_selector_trainer import (
    _build_states,
    _load_replays_and_metadata,
    _parse_int_set,
    _standardize_state_features,
    _state_summary,
    _state_summary_by_source,
)
from quantum_cat_public_history import (
    PUBLIC_HISTORY_ACTION_SCHEMA,
    PUBLIC_HISTORY_SCHEMA,
    PUBLIC_HISTORY_VERSION,
    public_history_action_features,
)
from quantum_cat_table_liveness_improve_trainer import (
    _load_metadata_rows,
    _metadata_path_for,
    _parse_float_list,
)


class ActionSetTransformer(nn.Module):
  """Scores each legal action while attending over the full legal set."""

  def __init__(
      self,
      feature_size: int,
      hidden: int,
      layers: int,
      heads: int,
      dropout: float,
  ):
    super().__init__()
    self.input = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden,
        nhead=heads,
        dim_feedforward=hidden * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
    self.head_body = nn.Sequential(
        nn.Linear(hidden * 3, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.GELU(),
    )
    self.score = nn.Linear(hidden, 1)
    self.value = nn.Linear(hidden, 1)
    self.risk = nn.Linear(hidden, 1)
    self.positive = nn.Linear(hidden, 1)
    self.safe = nn.Linear(hidden, 1)

  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    h = self.input(x)
    h = self.encoder(h, src_key_padding_mask=~mask)
    mask_f = mask.to(dtype=h.dtype).unsqueeze(-1)
    count = torch.clamp(mask_f.sum(dim=1, keepdim=True), min=1.0)
    mean = (h * mask_f).sum(dim=1, keepdim=True) / count
    masked_h = h.masked_fill(~mask.unsqueeze(-1), -1e9)
    maxv = masked_h.max(dim=1, keepdim=True).values
    maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
    z = torch.cat([h, mean.expand_as(h), maxv.expand_as(h)], dim=-1)
    body = self.head_body(z)
    return (
        self.score(body).squeeze(-1).masked_fill(~mask, -1e9),
        self.value(body).squeeze(-1),
        self.risk(body).squeeze(-1),
        self.positive(body).squeeze(-1),
        self.safe(body).squeeze(-1),
    )


class LinearActionSet(nn.Module):
  """Low-capacity baseline for sparse source-disjoint correction banks."""

  def __init__(self, feature_size: int):
    super().__init__()
    self.score = nn.Linear(feature_size, 1)
    self.value = nn.Linear(feature_size, 1)
    self.risk = nn.Linear(feature_size, 1)
    self.positive = nn.Linear(feature_size, 1)
    self.safe = nn.Linear(feature_size, 1)

  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    return (
        self.score(x).squeeze(-1).masked_fill(~mask, -1e9),
        self.value(x).squeeze(-1),
        self.risk(x).squeeze(-1),
        self.positive(x).squeeze(-1),
        self.safe(x).squeeze(-1),
    )


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20261261)
  parser.add_argument(
      "--model-kind",
      choices=("action_set_transformer", "linear_pair"),
      default="action_set_transformer",
  )
  parser.add_argument("--train-source-indices", default="")
  parser.add_argument("--validation-source-indices", default="2")
  parser.add_argument(
      "--selection-min-trick-number",
      type=int,
      default=-1,
      help=(
          "Information-legal live selection scope used by held-out threshold "
          "gates and saved into the deployable selector contract."
      ),
  )
  parser.add_argument(
      "--selection-max-trick-number",
      type=int,
      default=-1,
      help="Optional inclusive last trick for the saved live selection scope.",
  )
  parser.add_argument("--epochs", type=int, default=120)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--hidden", type=int, default=128)
  parser.add_argument("--layers", type=int, default=2)
  parser.add_argument("--heads", type=int, default=4)
  parser.add_argument("--dropout", type=float, default=0.08)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--positive-value-margin", type=float, default=0.01)
  parser.add_argument("--safe-risk-delta-max", type=float, default=0.0)
  parser.add_argument(
      "--allow-value-only-action-labels",
      action="store_true",
      help=(
          "Allow replay rows with action-value masks but no action-risk masks. "
          "Risk/safe labels are treated as unavailable/zero and this should "
          "be paired with zero risk/safe losses; intended for belief-rollout "
          "policy/value replay."
      ),
  )
  parser.add_argument(
      "--observation-feature-mode",
      choices=("none", "summary", "raw", "raw_summary"),
      default="summary",
  )
  parser.add_argument(
      "--policy-context-mode",
      choices=("none", "summary"),
      default="summary",
  )
  parser.add_argument(
      "--pair-feature-mode",
      choices=("full", "compact_action_delta"),
      default="full",
      help=(
          "Use the legacy expanded pair feature or the lower-capacity compact "
          "candidate/action-delta representation."
      ),
  )
  parser.add_argument(
      "--round-context-mode",
      choices=("none", "summary", "public_state_summary"),
      default="none",
      help=(
          "Append information-legal within-round history features. "
          "public_state_summary requires the exact deployable public-history "
          "vector recorded from each state. The legacy summary mode uses "
          "earlier labeled rows and is offline-only."
      ),
  )
  parser.add_argument(
      "--base-action-source",
      choices=("metadata_policy_best", "policy_context_argmax"),
      default="policy_context_argmax",
  )
  parser.add_argument(
      "--target-mode",
      choices=(
          "best_positive_else_base",
          "best_value_positive_else_base",
          "group_relative_soft",
          "replay_action",
      ),
      default="best_value_positive_else_base",
  )
  parser.add_argument(
      "--chained-audit-pair-only",
      action="store_true",
      help=(
          "Restrict chained rows to the incumbent and independently audited "
          "proposal instead of treating unaudited legal actions as negatives."
      ),
  )
  parser.add_argument("--soft-target-temperature", type=float, default=0.35)
  parser.add_argument("--soft-target-risk-penalty", type=float, default=1.0)
  parser.add_argument("--soft-target-unsafe-penalty", type=float, default=1.0)
  parser.add_argument("--force-led-lane-burn-negative", action="store_true")
  parser.add_argument(
      "--led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="urgent_win",
  )
  parser.add_argument("--select-loss-weight", type=float, default=1.0)
  parser.add_argument("--value-loss-weight", type=float, default=0.8)
  parser.add_argument("--risk-loss-weight", type=float, default=0.8)
  parser.add_argument("--positive-loss-weight", type=float, default=0.8)
  parser.add_argument("--safe-loss-weight", type=float, default=0.4)
  parser.add_argument("--positive-target-weight", type=float, default=4.0)
  parser.add_argument("--base-target-weight", type=float, default=1.0)
  parser.add_argument("--score-margin-loss-weight", type=float, default=0.4)
  parser.add_argument("--score-margin", type=float, default=1.0)
  parser.add_argument(
      "--proposal-contrast-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Extra loss for live proposal-mined rows. False-positive proposal "
          "candidates are pushed below the verified target/base; true-positive "
          "proposal candidates are pushed above the base."
      ),
  )
  parser.add_argument("--proposal-contrast-margin", type=float, default=1.0)
  parser.add_argument(
      "--proposal-positive-contrast-margin",
      type=float,
      default=0.5,
  )
  parser.add_argument(
      "--proposal-contrast-include-positive",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--score-margin-thresholds",
      default="-1.0,0.0,0.5,1.0,2.0,3.0",
  )
  parser.add_argument(
      "--predicted-value-delta-thresholds",
      default="0.00,0.02,0.04,0.08,0.12,0.14",
  )
  parser.add_argument(
      "--predicted-risk-delta-max-thresholds",
      default="-0.05,-0.03,-0.02,-0.01,0.00,0.02",
  )
  parser.add_argument(
      "--positive-prob-thresholds",
      default="0.50,0.70,0.90,0.97,0.99",
  )
  parser.add_argument(
      "--safe-prob-thresholds",
      default="0.50,0.70,0.90,0.97",
  )
  parser.add_argument("--min-validation-value-only-overrides", type=int, default=8)
  parser.add_argument(
      "--min-validation-value-only-positive-precision", type=float, default=0.85
  )
  parser.add_argument(
      "--min-validation-value-only-positive-recall", type=float, default=0.25
  )
  parser.add_argument(
      "--min-validation-value-only-decision-accuracy", type=float, default=0.75
  )
  parser.add_argument(
      "--min-validation-value-only-mean-value-delta", type=float, default=0.0
  )
  parser.add_argument("--selection-risk-penalty", type=float, default=0.5)
  parser.add_argument("--selection-value-bonus", type=float, default=0.05)
  parser.add_argument("--selection-policy-tiebreak", type=float, default=0.01)
  parser.add_argument(
      "--exclude-forced-led-lane-negatives-at-selection",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--min-validation-overrides-for-live", type=int, default=8)
  parser.add_argument(
      "--min-validation-positive-precision-for-live", type=float, default=0.85
  )
  parser.add_argument("--min-validation-safe-rate-for-live", type=float, default=0.90)
  parser.add_argument(
      "--min-validation-mean-value-delta-for-live", type=float, default=0.0
  )
  parser.add_argument(
      "--max-validation-mean-risk-delta-for-live", type=float, default=0.0
  )
  return parser.parse_args()


def _source_states(states: list[dict], sources: set[int]) -> list[dict]:
  return [
      state for state in states
      if int(state.get("source_replay_index", -1)) in sources
  ]


def _selection_scope_states(args, states: list[dict]) -> list[dict]:
  min_trick = int(getattr(args, "selection_min_trick_number", -1))
  max_trick = int(getattr(args, "selection_max_trick_number", -1))
  scoped = []
  for state in states:
    trick_number = int(state.get("trick_number", -1))
    if min_trick >= 0 and trick_number < min_trick:
      continue
    if max_trick >= 0 and trick_number > max_trick:
      continue
    scoped.append(state)
  return scoped


def _source_group_split_report(
    train_states: list[dict], validation_states: list[dict]
) -> dict:
  """Prove canonical collector groups do not cross the held-out split."""
  train_groups = {
      str(state.get("source_group_id", ""))
      for state in train_states
      if str(state.get("source_group_id", ""))
  }
  validation_groups = {
      str(state.get("source_group_id", ""))
      for state in validation_states
      if str(state.get("source_group_id", ""))
  }
  overlap = sorted(train_groups & validation_groups)
  if overlap:
    raise ValueError(
        "canonical source groups cross train/validation split: "
        + ", ".join(overlap)
    )
  return {
      "train_source_groups": sorted(train_groups),
      "validation_source_groups": sorted(validation_groups),
      "overlap": overlap,
      "passed": True,
  }


ROUND_CONTEXT_SCHEMA = (
    "round_position_norm",
    "prior_count_norm",
    "prior_follow_rate",
    "prior_token_loss_rate",
    "prior_red_rate",
    "prior_red_token_loss_rate",
    "prior_nonred_token_loss_rate",
    "own_prior_count_norm",
    "own_prior_follow_rate",
    "own_prior_token_loss_rate",
    "own_prior_nonred_token_loss_rate",
    "others_prior_token_loss_rate",
    "prev_base_follow",
    "prev_base_token_loss",
    "prev_base_red",
    "prev_base_nonred_token_loss",
    "acting_player_norm",
    "current_legal_count_norm",
    "candidate_token_loss_x_prior_token_loss_rate",
    "candidate_nonred_token_loss_x_prior_nonred_token_loss_rate",
    "candidate_token_loss_x_own_prior_token_loss_rate",
    "candidate_nonred_token_loss_x_own_prior_nonred_token_loss_rate",
    "candidate_token_loss_x_prev_base_token_loss",
    "candidate_nonred_token_loss_x_prev_base_nonred_token_loss",
    "candidate_not_follow_x_prior_follow_rate",
    "candidate_red_token_loss_x_prior_token_loss_rate",
    "candidate_follows_led_x_prior_token_loss_rate",
    "candidate_token_loss_x_round_position_norm",
    "candidate_nonred_token_loss_x_round_position_norm",
)


def _rate(num: float, denom: float) -> float:
  return 0.0 if float(denom) <= 0.0 else float(num) / float(denom)


def _feature_bool(action_features: np.ndarray, action: int, feature_idx: int) -> bool:
  if int(action) < 0 or int(action) >= int(action_features.shape[0]):
    return False
  if int(feature_idx) < 0 or int(feature_idx) >= int(action_features.shape[1]):
    return False
  return bool(float(action_features[int(action), int(feature_idx)]) > 0.5)


def _token_loss(action_features: np.ndarray, action: int) -> bool:
  return bool(
      _feature_bool(action_features, action, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX)
      or _feature_bool(
          action_features,
          action,
          APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"],
      )
  )


def _round_context_stats(players: int) -> dict:
  return {
      "count": 0,
      "follow": 0,
      "token_loss": 0,
      "red": 0,
      "red_token_loss": 0,
      "nonred_token_loss": 0,
      "prev_follow": 0.0,
      "prev_token_loss": 0.0,
      "prev_red": 0.0,
      "prev_nonred_token_loss": 0.0,
      "by_player": [
          {
              "count": 0,
              "follow": 0,
              "token_loss": 0,
              "nonred_token_loss": 0,
          }
          for _ in range(max(1, int(players)))
      ],
  }


def _round_context_vector(
    stats: dict,
    *,
    player: int,
    players: int,
    legal_count: int,
) -> np.ndarray:
  count = int(stats["count"])
  player_idx = min(max(0, int(player)), max(0, int(players) - 1))
  own = stats["by_player"][player_idx]
  other_count = max(0, count - int(own["count"]))
  other_loss = max(0, int(stats["token_loss"]) - int(own["token_loss"]))
  context = [
      min(1.0, count / 25.0),
      min(1.0, count / 25.0),
      _rate(stats["follow"], count),
      _rate(stats["token_loss"], count),
      _rate(stats["red"], count),
      _rate(stats["red_token_loss"], count),
      _rate(stats["nonred_token_loss"], count),
      min(1.0, int(own["count"]) / 7.0),
      _rate(own["follow"], own["count"]),
      _rate(own["token_loss"], own["count"]),
      _rate(own["nonred_token_loss"], own["count"]),
      _rate(other_loss, other_count),
      float(stats["prev_follow"]),
      float(stats["prev_token_loss"]),
      float(stats["prev_red"]),
      float(stats["prev_nonred_token_loss"]),
      _rate(player_idx, max(1, int(players) - 1)),
      min(1.0, int(legal_count) / 20.0),
  ]
  return np.asarray(context, dtype=np.float32)


def _update_round_context_stats(
    stats: dict,
    *,
    player: int,
    players: int,
    action_features: np.ndarray,
    base_action: int,
) -> None:
  follows = _feature_bool(action_features, base_action, ACTION_FEATURE_FOLLOWS_LED_INDEX)
  red = _feature_bool(action_features, base_action, ACTION_FEATURE_IS_RED_INDEX)
  loses = _token_loss(action_features, base_action)
  nonred_loss = bool(loses and not red)
  red_loss = bool(loses and red)
  player_idx = min(max(0, int(player)), max(0, int(players) - 1))
  own = stats["by_player"][player_idx]
  stats["count"] += 1
  stats["follow"] += int(follows)
  stats["token_loss"] += int(loses)
  stats["red"] += int(red)
  stats["red_token_loss"] += int(red_loss)
  stats["nonred_token_loss"] += int(nonred_loss)
  stats["prev_follow"] = float(follows)
  stats["prev_token_loss"] = float(loses)
  stats["prev_red"] = float(red)
  stats["prev_nonred_token_loss"] = float(nonred_loss)
  own["count"] += 1
  own["follow"] += int(follows)
  own["token_loss"] += int(loses)
  own["nonred_token_loss"] += int(nonred_loss)


def _append_round_context_features(args, rows, states: list[dict]) -> list[str]:
  """Append information-legal within-round history features to action rows."""
  mode = str(getattr(args, "round_context_mode", "none"))
  if mode == "none":
    return []
  if mode == "public_state_summary":
    for state in states:
      metadata = state.get("public_history")
      if not isinstance(metadata, dict):
        raise ValueError(
            "public_state_summary requires public_history metadata on every row"
        )
      if str(metadata.get("version", "")) != PUBLIC_HISTORY_VERSION:
        raise ValueError(
            "public history version mismatch: "
            f"{metadata.get('version')!r} vs {PUBLIC_HISTORY_VERSION!r}"
        )
      if tuple(metadata.get("schema", ())) != tuple(PUBLIC_HISTORY_SCHEMA):
        raise ValueError("public history schema mismatch")
      history = np.asarray(metadata.get("features", ()), dtype=np.float32)
      if history.shape != (len(PUBLIC_HISTORY_SCHEMA),):
        raise ValueError(
            "public history feature shape mismatch: "
            f"{history.shape} vs {(len(PUBLIC_HISTORY_SCHEMA),)}"
        )
      row = rows[int(state["state_index"])]
      action_features = np.asarray(row[11], dtype=np.float32)
      for action in state["actions"]:
        action_idx = int(action["action"])
        context = public_history_action_features(
            history,
            candidate_follows_led=_feature_bool(
                action_features,
                action_idx,
                ACTION_FEATURE_FOLLOWS_LED_INDEX,
            ),
            candidate_is_red=_feature_bool(
                action_features,
                action_idx,
                ACTION_FEATURE_IS_RED_INDEX,
            ),
            candidate_token_loss=_token_loss(action_features, action_idx),
        )
        action["feature"] = np.concatenate([
            np.asarray(action["feature"], dtype=np.float32),
            context,
        ]).astype(np.float32)
      state["round_context"] = history.tolist()
    return list(PUBLIC_HISTORY_ACTION_SCHEMA)
  if mode != "summary":
    raise ValueError(f"unknown round_context_mode={args.round_context_mode}")
  players = int(getattr(args, "players", 5))
  groups: dict[str, dict] = {}
  state_by_index = {int(state["state_index"]): state for state in states}
  for state_idx in sorted(state_by_index):
    state = state_by_index[state_idx]
    row = rows[state_idx]
    action_features = np.asarray(row[11], dtype=np.float32)
    legal_count = int(np.sum(np.asarray(row[1], dtype=np.float32) > 0.5))
    player = int(row[6])
    group_key = str(state.get("group_key", ""))
    stats = groups.setdefault(group_key, _round_context_stats(players))
    context = _round_context_vector(
        stats,
        player=player,
        players=players,
        legal_count=legal_count,
    )
    for action in state["actions"]:
      action_idx = int(action["action"])
      candidate_follows = _feature_bool(
          action_features, action_idx, ACTION_FEATURE_FOLLOWS_LED_INDEX
      )
      candidate_red = _feature_bool(action_features, action_idx, ACTION_FEATURE_IS_RED_INDEX)
      candidate_token_loss = _token_loss(action_features, action_idx)
      candidate_nonred_token_loss = bool(candidate_token_loss and not candidate_red)
      candidate_red_token_loss = bool(candidate_token_loss and candidate_red)
      interactions = np.asarray([
          float(candidate_token_loss) * float(context[3]),
          float(candidate_nonred_token_loss) * float(context[6]),
          float(candidate_token_loss) * float(context[9]),
          float(candidate_nonred_token_loss) * float(context[10]),
          float(candidate_token_loss) * float(context[13]),
          float(candidate_nonred_token_loss) * float(context[15]),
          float(not candidate_follows) * float(context[2]),
          float(candidate_red_token_loss) * float(context[3]),
          float(candidate_follows) * float(context[3]),
          float(candidate_token_loss) * float(context[0]),
          float(candidate_nonred_token_loss) * float(context[0]),
      ], dtype=np.float32)
      action["feature"] = np.concatenate([
          np.asarray(action["feature"], dtype=np.float32),
          context,
          interactions,
      ]).astype(np.float32)
    state["round_context"] = context.tolist()
    state["current_player"] = int(player)
    _update_round_context_stats(
        stats,
        player=player,
        players=players,
        action_features=action_features,
        base_action=int(state["base_action"]),
    )
  return list(ROUND_CONTEXT_SCHEMA)


def _padded_batch(states: list[dict]):
  max_actions = max(len(state["actions"]) for state in states)
  feature_size = len(states[0]["actions"][0]["feature_std"])
  x = torch.zeros((len(states), max_actions, feature_size), dtype=torch.float32)
  mask = torch.zeros((len(states), max_actions), dtype=torch.bool)
  target = torch.zeros((len(states),), dtype=torch.long)
  target_weight = torch.ones((len(states),), dtype=torch.float32)
  value = torch.zeros((len(states), max_actions), dtype=torch.float32)
  risk = torch.zeros((len(states), max_actions), dtype=torch.float32)
  positive = torch.zeros((len(states), max_actions), dtype=torch.float32)
  safe = torch.zeros((len(states), max_actions), dtype=torch.float32)
  base = torch.zeros((len(states), max_actions), dtype=torch.bool)
  soft_target = torch.zeros((len(states), max_actions), dtype=torch.float32)
  for row_idx, state in enumerate(states):
    action_count = len(state["actions"])
    x[row_idx, :action_count] = torch.tensor(
        np.asarray(
            [action["feature_std"] for action in state["actions"]],
            dtype=np.float32,
        ),
        dtype=torch.float32,
    )
    mask[row_idx, :action_count] = True
    target[row_idx] = int(state["target_index"])
    target_weight[row_idx] = float(state.get("target_weight", 1.0))
    value[row_idx, :action_count] = torch.tensor(
        [float(action["value_delta"]) for action in state["actions"]],
        dtype=torch.float32,
    )
    risk[row_idx, :action_count] = torch.tensor(
        [float(action["risk_delta"]) for action in state["actions"]],
        dtype=torch.float32,
    )
    positive[row_idx, :action_count] = torch.tensor(
        [float(action["positive"]) for action in state["actions"]],
        dtype=torch.float32,
    )
    safe[row_idx, :action_count] = torch.tensor(
        [float(action["safe"]) for action in state["actions"]],
        dtype=torch.float32,
    )
    base[row_idx, :action_count] = torch.tensor(
        [bool(action["is_base"]) for action in state["actions"]],
        dtype=torch.bool,
    )
    soft_values = state.get("soft_target")
    if soft_values is None:
      soft_values = [0.0] * action_count
      soft_values[int(state["target_index"])] = 1.0
    soft_target[row_idx, :action_count] = torch.tensor(
        soft_values,
        dtype=torch.float32,
    )
  return x, mask, target, target_weight, value, risk, positive, safe, base, soft_target


def _masked_action_loss(loss_per_action: torch.Tensor, mask: torch.Tensor):
  valid_counts = torch.clamp(mask.sum(dim=1).to(dtype=torch.float32), min=1.0)
  return torch.mean((loss_per_action * mask.to(dtype=torch.float32)).sum(dim=1) / valid_counts)


def _bce_pos_weight(states: list[dict], key: str):
  positives = 0
  total = 0
  for state in states:
    for action in state["actions"]:
      positives += int(bool(action[key]))
      total += 1
  negatives = total - positives
  if positives <= 0:
    return torch.tensor(1.0, dtype=torch.float32)
  return torch.tensor(max(1.0, negatives / positives), dtype=torch.float32)


def _score_margin_loss(
    scores: torch.Tensor,
    positive: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
):
  has_positive = torch.any((positive > 0.5) & mask, dim=1)
  if not bool(torch.any(has_positive)):
    return scores.sum() * 0.0
  pos_scores = scores.masked_fill(~((positive > 0.5) & mask), -1e9).max(dim=1).values
  base_scores = scores.masked_fill(~(base & mask), -1e9).max(dim=1).values
  diff = pos_scores[has_positive] - base_scores[has_positive]
  return F.softplus(float(margin) - diff).mean()


def _selection_loss(args, scores: torch.Tensor, target: torch.Tensor, soft_target: torch.Tensor):
  if str(getattr(args, "target_mode", "")) == "group_relative_soft":
    log_probs = F.log_softmax(scores, dim=1)
    return -(soft_target * log_probs).sum(dim=1)
  return F.cross_entropy(scores, target, reduction="none")


def _load_metadata_for_proposal_contrast(args, source_reports):
  metadata_paths = list(getattr(args, "metadata", []) or [])
  replay_paths = list(getattr(args, "replay", []) or [])
  if metadata_paths and len(metadata_paths) != len(replay_paths):
    raise ValueError(
        "when --metadata is provided with multi-replay training, pass exactly "
        f"one --metadata per --replay ({len(metadata_paths)} vs "
        f"{len(replay_paths)})"
    )
  rows = []
  for source_idx, report in enumerate(source_reports):
    replay_path = str(report.get("replay", replay_paths[source_idx]))
    expected_rows = int(report.get("rows", 0))
    metadata_arg = metadata_paths[source_idx] if metadata_paths else ""
    metadata_path = Path(metadata_arg) if metadata_arg else _metadata_path_for(
        replay_path
    )
    if not metadata_path.exists():
      rows.extend({} for _ in range(expected_rows))
      continue
    source_rows = _load_metadata_rows(metadata_path)
    if len(source_rows) != expected_rows:
      raise ValueError(
          "replay/metadata row mismatch for proposal contrast: "
          f"{replay_path}: {expected_rows} vs {len(source_rows)}"
      )
    rows.extend(source_rows)
  return rows


def _annotate_proposal_contrast(states: list[dict], metadata_rows: list[dict]):
  summary = {
      "metadata_rows": int(len(metadata_rows)),
      "states_seen": int(len(states)),
      "states_with_above_threshold_proposal": 0,
      "states_with_candidate_action": 0,
      "states_with_base_action": 0,
      "proposal_positive_candidates": 0,
      "proposal_false_positive_candidates": 0,
      "proposal_candidate_missing": 0,
      "proposal_base_missing": 0,
  }
  for state in states:
    state_idx = int(state.get("state_index", -1))
    if state_idx < 0 or state_idx >= len(metadata_rows):
      continue
    proposal = metadata_rows[state_idx].get("proposal")
    if not isinstance(proposal, dict):
      continue
    if not bool(proposal.get("above_threshold", False)):
      continue
    summary["states_with_above_threshold_proposal"] += 1
    candidate_action = int(proposal.get("candidate_action", -1))
    base_action = int(proposal.get("base_action", state.get("base_action", -1)))
    candidate_index = None
    base_index = None
    for idx, action in enumerate(state["actions"]):
      if int(action["action"]) == candidate_action:
        candidate_index = int(idx)
        action["proposal_candidate"] = True
      if int(action["action"]) == base_action:
        base_index = int(idx)
        action["proposal_base"] = True
    if candidate_index is None:
      summary["proposal_candidate_missing"] += 1
      continue
    summary["states_with_candidate_action"] += 1
    if base_index is None:
      summary["proposal_base_missing"] += 1
    else:
      summary["states_with_base_action"] += 1
    proposal_positive = bool(proposal.get("positive_by_dense_delta", False))
    proposal_safe = bool(proposal.get("safe_by_paradox_delta", False))
    proposal_negative = bool(not proposal_positive or not proposal_safe)
    state["proposal_candidate_index"] = int(candidate_index)
    state["proposal_base_index"] = (
        int(base_index) if base_index is not None else -1
    )
    state["proposal_candidate_positive"] = bool(
        proposal_positive and proposal_safe
    )
    state["proposal_candidate_negative"] = bool(proposal_negative)
    state["proposal_dense_value_delta_vs_base"] = float(
        proposal.get("dense_action_value_delta_vs_base", 0.0)
    )
    state["proposal_paradox_delta_vs_base"] = float(
        proposal.get("paradox_delta_vs_base", 0.0)
    )
    state["actions"][candidate_index]["proposal_candidate_positive"] = bool(
        state["proposal_candidate_positive"]
    )
    state["actions"][candidate_index]["proposal_candidate_negative"] = bool(
        state["proposal_candidate_negative"]
    )
    if state["proposal_candidate_positive"]:
      summary["proposal_positive_candidates"] += 1
    if state["proposal_candidate_negative"]:
      summary["proposal_false_positive_candidates"] += 1
  return summary


def _proposal_contrast_loss(args, scores: torch.Tensor, batch: list[dict]):
  if float(getattr(args, "proposal_contrast_loss_weight", 0.0)) <= 0.0:
    return scores.sum() * 0.0
  losses = []
  negative_margin = float(getattr(args, "proposal_contrast_margin", 1.0))
  positive_margin = float(
      getattr(args, "proposal_positive_contrast_margin", 0.5)
  )
  include_positive = bool(
      getattr(args, "proposal_contrast_include_positive", True)
  )
  for row_idx, state in enumerate(batch):
    candidate_idx = int(state.get("proposal_candidate_index", -1))
    if candidate_idx < 0 or candidate_idx >= len(state["actions"]):
      continue
    if bool(state.get("proposal_candidate_negative", False)):
      anchor_idx = int(state.get("target_index", -1))
      if anchor_idx < 0 or anchor_idx >= len(state["actions"]):
        anchor_idx = int(state.get("proposal_base_index", -1))
      if anchor_idx < 0 or anchor_idx >= len(state["actions"]):
        continue
      if anchor_idx == candidate_idx:
        continue
      losses.append(
          F.softplus(
              negative_margin
              + scores[row_idx, candidate_idx]
              - scores[row_idx, anchor_idx]
          )
      )
      continue
    if include_positive and bool(state.get("proposal_candidate_positive", False)):
      base_idx = int(state.get("proposal_base_index", -1))
      if base_idx < 0 or base_idx >= len(state["actions"]):
        continue
      losses.append(
          F.softplus(
              positive_margin
              - (scores[row_idx, candidate_idx] - scores[row_idx, base_idx])
          )
      )
  if not losses:
    return scores.sum() * 0.0
  return torch.stack(losses).mean()


def _train(args, train_states: list[dict], feature_size: int):
  torch.manual_seed(int(args.seed))
  if str(args.model_kind) == "linear_pair":
    model = LinearActionSet(feature_size)
  else:
    model = ActionSetTransformer(
        feature_size,
        int(args.hidden),
        int(args.layers),
        int(args.heads),
        float(args.dropout),
    )
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  for state in train_states:
    state["target_weight"] = (
        float(args.base_target_weight)
        if bool(state["target_is_base"]) else
        float(args.positive_target_weight)
    )
  positive_weight = _bce_pos_weight(train_states, "positive")
  safe_weight = _bce_pos_weight(train_states, "safe")
  rng = random.Random(int(args.seed))
  indices = list(range(len(train_states)))
  losses = []
  select_losses = []
  value_losses = []
  risk_losses = []
  positive_losses = []
  safe_losses = []
  margin_losses = []
  proposal_contrast_losses = []
  batch_size = max(1, int(args.batch_size))
  for _ in range(max(1, int(args.epochs))):
    rng.shuffle(indices)
    totals = [0.0] * 8
    seen = 0
    for start in range(0, len(indices), batch_size):
      batch = [train_states[idx] for idx in indices[start:start + batch_size]]
      (
          x,
          mask,
          target,
          target_weight,
          value_t,
          risk_t,
          positive_t,
          safe_t,
          base_t,
          soft_t,
      ) = (
          _padded_batch(batch)
      )
      scores, value_p, risk_p, positive_logit, safe_logit = model(x, mask)
      select_per = _selection_loss(args, scores, target, soft_t)
      select_loss = torch.mean(select_per * target_weight)
      value_loss = _masked_action_loss(
          F.smooth_l1_loss(value_p, value_t, reduction="none"), mask
      )
      risk_loss = _masked_action_loss(
          F.smooth_l1_loss(risk_p, risk_t, reduction="none"), mask
      )
      positive_loss = _masked_action_loss(
          F.binary_cross_entropy_with_logits(
              positive_logit,
              positive_t,
              pos_weight=positive_weight,
              reduction="none",
          ),
          mask,
      )
      safe_loss = _masked_action_loss(
          F.binary_cross_entropy_with_logits(
              safe_logit,
              safe_t,
              pos_weight=safe_weight,
              reduction="none",
          ),
          mask,
      )
      margin_loss = _score_margin_loss(
          scores, positive_t, base_t, mask, float(args.score_margin)
      )
      proposal_contrast_loss = _proposal_contrast_loss(args, scores, batch)
      loss = (
          float(args.select_loss_weight) * select_loss
          + float(args.value_loss_weight) * value_loss
          + float(args.risk_loss_weight) * risk_loss
          + float(args.positive_loss_weight) * positive_loss
          + float(args.safe_loss_weight) * safe_loss
          + float(args.score_margin_loss_weight) * margin_loss
          + float(args.proposal_contrast_loss_weight) * proposal_contrast_loss
      )
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      count = len(batch)
      for idx_total, item in enumerate((
          loss,
          select_loss,
          value_loss,
          risk_loss,
          positive_loss,
          safe_loss,
          margin_loss,
          proposal_contrast_loss,
      )):
        totals[idx_total] += float(item.detach()) * count
      seen += count
    denom = max(1, seen)
    losses.append(totals[0] / denom)
    select_losses.append(totals[1] / denom)
    value_losses.append(totals[2] / denom)
    risk_losses.append(totals[3] / denom)
    positive_losses.append(totals[4] / denom)
    safe_losses.append(totals[5] / denom)
    margin_losses.append(totals[6] / denom)
    proposal_contrast_losses.append(totals[7] / denom)
  return model, {
      "epochs": int(args.epochs),
      "model_kind": str(args.model_kind),
      "batch_size": int(args.batch_size),
      "hidden": int(args.hidden),
      "layers": int(args.layers),
      "heads": int(args.heads),
      "dropout": float(args.dropout),
      "lr": float(args.lr),
      "weight_decay": float(args.weight_decay),
      "positive_pos_weight": round(float(positive_weight.item()), 6),
      "safe_pos_weight": round(float(safe_weight.item()), 6),
      "initial_loss": round(losses[0], 6),
      "final_loss": round(losses[-1], 6),
      "initial_select_loss": round(select_losses[0], 6),
      "final_select_loss": round(select_losses[-1], 6),
      "initial_value_loss": round(value_losses[0], 6),
      "final_value_loss": round(value_losses[-1], 6),
      "initial_risk_loss": round(risk_losses[0], 6),
      "final_risk_loss": round(risk_losses[-1], 6),
      "initial_positive_loss": round(positive_losses[0], 6),
      "final_positive_loss": round(positive_losses[-1], 6),
      "initial_safe_loss": round(safe_losses[0], 6),
      "final_safe_loss": round(safe_losses[-1], 6),
      "initial_margin_loss": round(margin_losses[0], 6),
      "final_margin_loss": round(margin_losses[-1], 6),
      "initial_proposal_contrast_loss": round(proposal_contrast_losses[0], 6),
      "final_proposal_contrast_loss": round(proposal_contrast_losses[-1], 6),
      "proposal_contrast_loss_weight": float(
          getattr(args, "proposal_contrast_loss_weight", 0.0)
      ),
      "proposal_contrast_margin": float(
          getattr(args, "proposal_contrast_margin", 1.0)
      ),
      "proposal_positive_contrast_margin": float(
          getattr(args, "proposal_positive_contrast_margin", 0.5)
      ),
      "proposal_contrast_include_positive": bool(
          getattr(args, "proposal_contrast_include_positive", True)
      ),
      "target_mode": str(args.target_mode),
      "soft_target_temperature": float(getattr(args, "soft_target_temperature", 0.35)),
      "soft_target_risk_penalty": float(getattr(args, "soft_target_risk_penalty", 1.0)),
      "soft_target_unsafe_penalty": float(getattr(args, "soft_target_unsafe_penalty", 1.0)),
  }


def _predict_states(model, states: list[dict], batch_size: int):
  model.eval()
  with torch.no_grad():
    for start in range(0, len(states), max(1, int(batch_size))):
      batch = states[start:start + max(1, int(batch_size))]
      x, mask, *_ = _padded_batch(batch)
      scores, value_p, risk_p, positive_logit, safe_logit = model(x, mask)
      positive_p = torch.sigmoid(positive_logit).cpu().numpy()
      safe_p = torch.sigmoid(safe_logit).cpu().numpy()
      for row_idx, state in enumerate(batch):
        for action_idx, action in enumerate(state["actions"]):
          action["set_score"] = float(scores[row_idx, action_idx].item())
          action["pred_value_delta"] = float(value_p[row_idx, action_idx].item())
          action["pred_risk_delta"] = float(risk_p[row_idx, action_idx].item())
          action["pred_positive_prob"] = float(positive_p[row_idx, action_idx])
          action["pred_safe_prob"] = float(safe_p[row_idx, action_idx])


def _base_action(state: dict) -> dict:
  for action in state["actions"]:
    if bool(action["is_base"]):
      return action
  raise ValueError("base action missing")


def _selected_action(
    state: dict,
    *,
    score_margin_threshold: float,
    value_threshold: float,
    risk_threshold: float,
    positive_threshold: float,
    safe_threshold: float,
    risk_penalty: float,
    value_bonus: float,
    policy_tiebreak: float,
    exclude_forced: bool,
):
  base = _base_action(state)
  base_score = float(base.get("set_score", 0.0))
  best = None
  best_rank = None
  for idx, action in enumerate(state["actions"]):
    if bool(action["is_base"]):
      continue
    if exclude_forced and bool(action.get("forced_led_lane_negative", False)):
      continue
    score_margin = float(action.get("set_score", 0.0)) - base_score
    if score_margin < float(score_margin_threshold):
      continue
    if float(action.get("pred_value_delta", 0.0)) < float(value_threshold):
      continue
    if float(action.get("pred_risk_delta", 0.0)) > float(risk_threshold):
      continue
    if float(action.get("pred_positive_prob", 0.0)) < float(positive_threshold):
      continue
    if float(action.get("pred_safe_prob", 0.0)) < float(safe_threshold):
      continue
    rank = (
        score_margin
        + float(value_bonus) * float(action.get("pred_value_delta", 0.0))
        - float(risk_penalty) * max(0.0, float(action.get("pred_risk_delta", 0.0)))
        + float(policy_tiebreak) * float(action["policy_prob"])
    )
    if best is None or rank > float(best_rank):
      best = dict(action)
      best["selected_index"] = int(idx)
      best["score_margin"] = float(score_margin)
      best_rank = float(rank)
  return best


def _threshold_report(args, states: list[dict]):
  reports = []
  for score_margin_threshold in _parse_float_list(args.score_margin_thresholds):
    for value_threshold in _parse_float_list(args.predicted_value_delta_thresholds):
      for risk_threshold in _parse_float_list(
          args.predicted_risk_delta_max_thresholds
      ):
        for positive_threshold in _parse_float_list(args.positive_prob_thresholds):
          for safe_threshold in _parse_float_list(args.safe_prob_thresholds):
            overrides = 0
            positives = 0
            safe = 0
            forced = 0
            value_deltas = []
            risk_deltas = []
            score_margins = []
            pred_values = []
            pred_risks = []
            positive_probs = []
            safe_probs = []
            for state in states:
              selected = _selected_action(
                  state,
                  score_margin_threshold=float(score_margin_threshold),
                  value_threshold=float(value_threshold),
                  risk_threshold=float(risk_threshold),
                  positive_threshold=float(positive_threshold),
                  safe_threshold=float(safe_threshold),
                  risk_penalty=float(args.selection_risk_penalty),
                  value_bonus=float(args.selection_value_bonus),
                  policy_tiebreak=float(args.selection_policy_tiebreak),
                  exclude_forced=bool(args.exclude_forced_led_lane_negatives_at_selection),
              )
              if selected is None:
                continue
              overrides += 1
              positives += int(bool(selected["positive"]))
              safe += int(bool(selected["safe"]))
              forced += int(bool(selected.get("forced_led_lane_negative", False)))
              value_deltas.append(float(selected["value_delta"]))
              risk_deltas.append(float(selected["risk_delta"]))
              score_margins.append(float(selected["score_margin"]))
              pred_values.append(float(selected["pred_value_delta"]))
              pred_risks.append(float(selected["pred_risk_delta"]))
              positive_probs.append(float(selected["pred_positive_prob"]))
              safe_probs.append(float(selected["pred_safe_prob"]))
            reports.append({
                "score_margin_threshold": float(score_margin_threshold),
                "value_threshold": float(value_threshold),
                "risk_threshold": float(risk_threshold),
                "positive_threshold": float(positive_threshold),
                "safe_threshold": float(safe_threshold),
                "states": int(len(states)),
                "overrides": int(overrides),
                "coverage": round(float(overrides / max(1, len(states))), 6),
                "positive_precision": (
                    None if overrides == 0 else round(float(positives / overrides), 6)
                ),
                "safe_rate": None if overrides == 0 else round(float(safe / overrides), 6),
                "forced_led_lane_negative_selected": int(forced),
                "mean_value_delta": (
                    None if not value_deltas else round(float(np.mean(value_deltas)), 6)
                ),
                "mean_risk_delta": (
                    None if not risk_deltas else round(float(np.mean(risk_deltas)), 6)
                ),
                "mean_score_margin": (
                    None if not score_margins else round(float(np.mean(score_margins)), 6)
                ),
                "mean_pred_value_delta": (
                    None if not pred_values else round(float(np.mean(pred_values)), 6)
                ),
                "mean_pred_risk_delta": (
                    None if not pred_risks else round(float(np.mean(pred_risks)), 6)
                ),
                "mean_positive_prob": (
                    None if not positive_probs else round(float(np.mean(positive_probs)), 6)
                ),
                "mean_safe_prob": (
                    None if not safe_probs else round(float(np.mean(safe_probs)), 6)
                ),
            })
  return reports


def _value_only_selected_action(
    state: dict,
    *,
    score_margin_threshold: float,
    value_threshold: float,
    positive_threshold: float,
    value_bonus: float,
    policy_tiebreak: float,
    exclude_forced: bool,
):
  base = _base_action(state)
  base_score = float(base.get("set_score", 0.0))
  best = None
  best_rank = None
  for idx, action in enumerate(state["actions"]):
    if bool(action["is_base"]):
      continue
    if exclude_forced and bool(action.get("forced_led_lane_negative", False)):
      continue
    score_margin = float(action.get("set_score", 0.0)) - base_score
    if score_margin < float(score_margin_threshold):
      continue
    if float(action.get("pred_value_delta", 0.0)) < float(value_threshold):
      continue
    if float(action.get("pred_positive_prob", 0.0)) < float(positive_threshold):
      continue
    rank = (
        score_margin
        + float(value_bonus) * float(action.get("pred_value_delta", 0.0))
        + float(policy_tiebreak) * float(action["policy_prob"])
    )
    if best is None or rank > float(best_rank):
      best = dict(action)
      best["selected_index"] = int(idx)
      best["score_margin"] = float(score_margin)
      best_rank = float(rank)
  return best


def _value_only_threshold_report(args, states: list[dict]):
  reports = []
  positive_states_total = int(sum(int(bool(state["has_positive"])) for state in states))
  base_target_states_total = int(
      sum(int(bool(state.get("target_is_base", False))) for state in states)
  )
  for score_margin_threshold in _parse_float_list(args.score_margin_thresholds):
    for value_threshold in _parse_float_list(args.predicted_value_delta_thresholds):
      for positive_threshold in _parse_float_list(args.positive_prob_thresholds):
        overrides = 0
        positives = 0
        positive_state_hits = 0
        correct_decisions = 0
        base_target_abstains = 0
        false_base_overrides = 0
        forced = 0
        value_deltas = []
        score_margins = []
        pred_values = []
        positive_probs = []
        for state in states:
          selected = _value_only_selected_action(
              state,
              score_margin_threshold=float(score_margin_threshold),
              value_threshold=float(value_threshold),
              positive_threshold=float(positive_threshold),
              value_bonus=float(args.selection_value_bonus),
              policy_tiebreak=float(args.selection_policy_tiebreak),
              exclude_forced=bool(args.exclude_forced_led_lane_negatives_at_selection),
          )
          has_positive = bool(state["has_positive"])
          target_is_base = bool(state.get("target_is_base", False))
          if selected is None:
            correct_decisions += int(not has_positive)
            base_target_abstains += int(target_is_base)
            continue
          overrides += 1
          selected_positive = bool(selected["positive"])
          positives += int(selected_positive)
          positive_state_hits += int(has_positive and selected_positive)
          correct_decisions += int(selected_positive)
          false_base_overrides += int(target_is_base)
          forced += int(bool(selected.get("forced_led_lane_negative", False)))
          value_deltas.append(float(selected["value_delta"]))
          score_margins.append(float(selected["score_margin"]))
          pred_values.append(float(selected["pred_value_delta"]))
          positive_probs.append(float(selected["pred_positive_prob"]))
        reports.append({
            "score_margin_threshold": float(score_margin_threshold),
            "value_threshold": float(value_threshold),
            "positive_threshold": float(positive_threshold),
            "states": int(len(states)),
            "positive_states": int(positive_states_total),
            "base_target_states": int(base_target_states_total),
            "overrides": int(overrides),
            "coverage": round(float(overrides / max(1, len(states))), 6),
            "positive_precision": (
                None if overrides == 0 else round(float(positives / overrides), 6)
            ),
            "positive_recall": (
                None if positive_states_total == 0 else
                round(float(positive_state_hits / positive_states_total), 6)
            ),
            "decision_accuracy": round(
                float(correct_decisions / max(1, len(states))), 6
            ),
            "base_target_abstain_rate": (
                None if base_target_states_total == 0 else
                round(float(base_target_abstains / base_target_states_total), 6)
            ),
            "false_base_overrides": int(false_base_overrides),
            "forced_led_lane_negative_selected": int(forced),
            "mean_value_delta": (
                None if not value_deltas else round(float(np.mean(value_deltas)), 6)
            ),
            "mean_score_margin": (
                None if not score_margins else round(float(np.mean(score_margins)), 6)
            ),
            "mean_pred_value_delta": (
                None if not pred_values else round(float(np.mean(pred_values)), 6)
            ),
            "mean_positive_prob": (
                None if not positive_probs else round(float(np.mean(positive_probs)), 6)
            ),
        })
  return reports


def _best_report(reports):
  if not reports:
    return None
  return max(
      reports,
      key=lambda row: (
          row["positive_precision"] is not None,
          row["positive_precision"] or -1.0,
          row["safe_rate"] or -1.0,
          row["overrides"] or -1,
          row["mean_value_delta"] or -999.0,
          -(row["mean_risk_delta"] or 999.0),
      ),
  )


def _eligible_reports(args, reports):
  eligible = []
  for row in reports:
    if int(row.get("overrides") or 0) < int(args.min_validation_overrides_for_live):
      continue
    if row.get("positive_precision") is None or float(
        row["positive_precision"]
    ) < float(args.min_validation_positive_precision_for_live):
      continue
    if row.get("safe_rate") is None or float(row["safe_rate"]) < float(
        args.min_validation_safe_rate_for_live
    ):
      continue
    if row.get("mean_value_delta") is None or float(
        row["mean_value_delta"]
    ) < float(args.min_validation_mean_value_delta_for_live):
      continue
    if row.get("mean_risk_delta") is None or float(
        row["mean_risk_delta"]
    ) > float(args.max_validation_mean_risk_delta_for_live):
      continue
    eligible.append(row)
  return eligible


def _best_value_only_report(reports):
  if not reports:
    return None
  return max(
      reports,
      key=lambda row: (
          row["positive_precision"] is not None,
          row["positive_precision"] or -1.0,
          row["decision_accuracy"] or -1.0,
          row["positive_recall"] or -1.0,
          row["overrides"] or -1,
          row["mean_value_delta"] or -999.0,
      ),
  )


def _value_only_eligible_reports(args, reports):
  eligible = []
  for row in reports:
    if int(row.get("overrides") or 0) < int(
        args.min_validation_value_only_overrides
    ):
      continue
    if row.get("positive_precision") is None or float(
        row["positive_precision"]
    ) < float(args.min_validation_value_only_positive_precision):
      continue
    if row.get("positive_recall") is None or float(
        row["positive_recall"]
    ) < float(args.min_validation_value_only_positive_recall):
      continue
    if float(row.get("decision_accuracy") or 0.0) < float(
        args.min_validation_value_only_decision_accuracy
    ):
      continue
    if row.get("mean_value_delta") is None or float(
        row["mean_value_delta"]
    ) < float(args.min_validation_value_only_mean_value_delta):
      continue
    eligible.append(row)
  return eligible


def _pearson(xs, ys):
  xs = np.asarray(xs, dtype=np.float64)
  ys = np.asarray(ys, dtype=np.float64)
  if xs.size < 2 or float(np.std(xs)) <= 1e-12 or float(np.std(ys)) <= 1e-12:
    return None
  return round(float(np.corrcoef(xs, ys)[0, 1]), 6)


def _prediction_summary(states: list[dict]):
  value_true = []
  value_pred = []
  risk_true = []
  risk_pred = []
  positive_probs = []
  positive_targets = []
  safe_probs = []
  safe_targets = []
  for state in states:
    for action in state["actions"]:
      value_true.append(float(action["value_delta"]))
      value_pred.append(float(action["pred_value_delta"]))
      risk_true.append(float(action["risk_delta"]))
      risk_pred.append(float(action["pred_risk_delta"]))
      positive_probs.append(float(action["pred_positive_prob"]))
      positive_targets.append(float(action["positive"]))
      safe_probs.append(float(action["pred_safe_prob"]))
      safe_targets.append(float(action["safe"]))
  value_err = np.asarray(value_pred) - np.asarray(value_true)
  risk_err = np.asarray(risk_pred) - np.asarray(risk_true)
  return {
      "actions": int(len(value_true)),
      "value_corr": _pearson(value_true, value_pred),
      "risk_corr": _pearson(risk_true, risk_pred),
      "value_mae": round(float(np.mean(np.abs(value_err))), 6),
      "risk_mae": round(float(np.mean(np.abs(risk_err))), 6),
      "positive_mean_prob_on_positive": round(float(np.mean([
          p for p, y in zip(positive_probs, positive_targets) if y > 0.5
      ] or [0.0])), 6),
      "positive_mean_prob_on_negative": round(float(np.mean([
          p for p, y in zip(positive_probs, positive_targets) if y <= 0.5
      ] or [0.0])), 6),
      "safe_mean_prob_on_safe": round(float(np.mean([
          p for p, y in zip(safe_probs, safe_targets) if y > 0.5
      ] or [0.0])), 6),
      "safe_mean_prob_on_unsafe": round(float(np.mean([
          p for p, y in zip(safe_probs, safe_targets) if y <= 0.5
      ] or [0.0])), 6),
  }


def main():
  args = parse_args()
  rows, metadata_rows, source_reports = _load_replays_and_metadata(args)
  states, skipped_report = _build_states(args, rows, metadata_rows)
  proposal_contrast_summary = {
      "enabled": bool(float(getattr(args, "proposal_contrast_loss_weight", 0.0)) > 0.0),
      "metadata_loaded": False,
  }
  if bool(proposal_contrast_summary["enabled"]):
    proposal_metadata_rows = _load_metadata_for_proposal_contrast(
        args, source_reports
    )
    proposal_contrast_summary = {
        **proposal_contrast_summary,
        "metadata_loaded": True,
        **_annotate_proposal_contrast(states, proposal_metadata_rows),
    }
  round_context_schema = _append_round_context_features(args, rows, states)
  validation_sources = _parse_int_set(str(args.validation_source_indices))
  all_sources = {
      int(state.get("source_replay_index", -1)) for state in states
  }
  train_sources = (
      _parse_int_set(str(args.train_source_indices))
      if str(args.train_source_indices or "").strip() else
      set(all_sources) - set(validation_sources)
  )
  train_states = _source_states(states, train_sources)
  validation_states = _source_states(states, validation_sources)
  if not train_states or not validation_states:
    raise ValueError("empty train/validation source split")
  source_group_split = _source_group_split_report(
      train_states, validation_states
  )
  mean, std = _standardize_state_features(train_states, validation_states, states)
  feature_size = len(states[0]["actions"][0]["feature_std"])
  model, train_report = _train(args, train_states, feature_size)
  _predict_states(model, states, int(args.batch_size))

  train_screen_states = _selection_scope_states(args, train_states)
  validation_screen_states = _selection_scope_states(args, validation_states)
  all_screen_states = _selection_scope_states(args, states)
  if not train_screen_states or not validation_screen_states:
    raise ValueError("empty train/validation live selection scope")
  train_thresholds = _threshold_report(args, train_screen_states)
  validation_thresholds = _threshold_report(args, validation_screen_states)
  all_thresholds = _threshold_report(args, all_screen_states)
  gate_eligible = _eligible_reports(args, validation_thresholds)
  best_gate = _best_report(gate_eligible)
  train_value_only_thresholds = _value_only_threshold_report(
      args, train_screen_states
  )
  validation_value_only_thresholds = _value_only_threshold_report(
      args, validation_screen_states
  )
  all_value_only_thresholds = _value_only_threshold_report(
      args, all_screen_states
  )
  value_only_gate_eligible = _value_only_eligible_reports(
      args, validation_value_only_thresholds
  )
  best_value_only_gate = _best_value_only_report(value_only_gate_eligible)

  model_out = str(args.model_out or "")
  if model_out:
    path = Path(model_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "feature_size": int(feature_size),
        "model_class": (
            "LinearActionSet" if str(args.model_kind) == "linear_pair" else
            "ActionSetTransformer"
        ),
        "model_kind": str(args.model_kind),
        "feature_schema_version": (
            (
                "legal_set_transformer_v1" if
                str(args.pair_feature_mode) == "full" else
                "legal_set_transformer_compact_action_delta_v1"
            )
            if not round_context_schema else (
                (
                    "legal_set_transformer_v1_public_history_v1" if
                    str(args.pair_feature_mode) == "full" else
                    "legal_set_transformer_compact_action_delta_v1_public_history_v1"
                )
                if str(args.round_context_mode) == "public_state_summary" else
                (
                    "legal_set_transformer_v1_round_context_summary" if
                    str(args.pair_feature_mode) == "full" else
                    "legal_set_transformer_compact_action_delta_v1_round_context_summary"
                )
            )
        ),
        "pair_feature_mode": str(args.pair_feature_mode),
        "round_context_schema": round_context_schema,
        "args": vars(args),
        "train_report": train_report,
        "replay_sources": source_reports,
    }, path)

  result = {
      "replay": list(args.replay),
      "metadata": list(args.metadata or []),
      "replay_sources": source_reports,
      "model_out": model_out,
      "model_kind": str(args.model_kind),
      "source_split": {
          "train_source_indices": sorted(int(x) for x in train_sources),
          "validation_source_indices": sorted(int(x) for x in validation_sources),
          **source_group_split,
      },
      "round_context_schema": round_context_schema,
      "skipped": skipped_report,
      "proposal_contrast_summary": proposal_contrast_summary,
      "feature_size": int(feature_size),
      "pair_feature_mode": str(args.pair_feature_mode),
      "selection_scope": {
          "min_trick_number": int(args.selection_min_trick_number),
          "max_trick_number": int(args.selection_max_trick_number),
          "train_states": int(len(train_screen_states)),
          "validation_states": int(len(validation_screen_states)),
          "all_states": int(len(all_screen_states)),
      },
      "train_summary": _state_summary(train_states),
      "validation_summary": _state_summary(validation_states),
      "all_summary": _state_summary(states),
      "train_summary_by_source": _state_summary_by_source(train_states),
      "validation_summary_by_source": _state_summary_by_source(validation_states),
      "all_summary_by_source": _state_summary_by_source(states),
      "train_report": train_report,
      "train_prediction_summary": _prediction_summary(train_states),
      "validation_prediction_summary": _prediction_summary(validation_states),
      "all_prediction_summary": _prediction_summary(states),
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "all_thresholds": all_thresholds,
      "best_validation_threshold": _best_report(validation_thresholds),
      "train_value_only_thresholds": train_value_only_thresholds,
      "validation_value_only_thresholds": validation_value_only_thresholds,
      "all_value_only_thresholds": all_value_only_thresholds,
      "best_validation_value_only_threshold": _best_value_only_report(
          validation_value_only_thresholds
      ),
      "live_screen_gate": {
          "passed": bool(best_gate is not None),
          "min_validation_overrides": int(args.min_validation_overrides_for_live),
          "min_positive_precision": float(
              args.min_validation_positive_precision_for_live
          ),
          "min_safe_rate": float(args.min_validation_safe_rate_for_live),
          "min_mean_value_delta": float(
              args.min_validation_mean_value_delta_for_live
          ),
          "max_mean_risk_delta": float(args.max_validation_mean_risk_delta_for_live),
          "eligible_thresholds": int(len(gate_eligible)),
          "best_threshold": best_gate,
      },
      "value_only_preference_gate": {
          "passed": bool(best_value_only_gate is not None),
          "min_validation_overrides": int(
              args.min_validation_value_only_overrides
          ),
          "min_positive_precision": float(
              args.min_validation_value_only_positive_precision
          ),
          "min_positive_recall": float(
              args.min_validation_value_only_positive_recall
          ),
          "min_decision_accuracy": float(
              args.min_validation_value_only_decision_accuracy
          ),
          "min_mean_value_delta": float(
              args.min_validation_value_only_mean_value_delta
          ),
          "eligible_thresholds": int(len(value_only_gate_eligible)),
          "best_threshold": best_value_only_gate,
      },
      "args": vars(args),
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
