#!/usr/bin/env python3
"""Train a direct approve/veto boundary for live proposal candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_calibration_transfer_audit import _standardize
from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    ACTION_FEATURE_PREFIX_SIZE,
    ACTION_FEATURE_RANK_NORM_INDEX,
    APPENDED_ACTION_FEATURE_INDEX,
)
from quantum_cat_legal_set_selector_trainer import (
    _build_states,
    _load_replays_and_metadata,
    _parse_int_set,
    _state_summary_by_source,
)
from quantum_cat_table_liveness_improve_trainer import _parse_float_list


ACTION_FEATURE_WOULD_WIN_INDEX = ACTION_FEATURE_PREFIX_SIZE + 19


class _GradReverse(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x: torch.Tensor, scale: float):
    ctx.scale = float(scale)
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor):
    return -float(ctx.scale) * grad_output, None


def _grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
  return _GradReverse.apply(x, float(scale))


class SourceAdversary(nn.Module):
  def __init__(self, feature_size: int, hidden: int, source_count: int):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.ReLU(),
        nn.Dropout(0.08),
        nn.Linear(hidden, source_count),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class ProposalVetoMLP(nn.Module):
  """Scores whether to approve one proposed candidate over the base action."""

  def __init__(self, feature_size: int, hidden: int, dropout: float):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x).squeeze(-1)

  def source_features(
      self,
      x: torch.Tensor,
      groups: list[list[int]] | None = None,
  ) -> torch.Tensor:
    del groups
    return self.net[:-1](x)


class ProposalVetoDeepSet(nn.Module):
  """Scores candidates using pooled context from their same-state slate."""

  def __init__(self, feature_size: int, hidden: int, dropout: float):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )
    self.score = nn.Sequential(
        nn.Linear(hidden * 4, hidden),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden, 1),
    )

  def forward(
      self,
      x: torch.Tensor,
      groups: list[list[int]] | None = None,
  ) -> torch.Tensor:
    embeddings = self.encoder(x)
    if not groups:
      groups = [list(range(int(x.shape[0])))]
    logits = torch.empty(int(x.shape[0]), dtype=x.dtype, device=x.device)
    for indices in groups:
      if not indices:
        continue
      idx_t = torch.tensor(indices, dtype=torch.long, device=x.device)
      group_embeddings = embeddings.index_select(0, idx_t)
      mean_pool = group_embeddings.mean(dim=0, keepdim=True)
      max_pool = group_embeddings.max(dim=0, keepdim=True).values
      mean_expand = mean_pool.expand(group_embeddings.shape[0], -1)
      max_expand = max_pool.expand(group_embeddings.shape[0], -1)
      score_input = torch.cat(
          [
              group_embeddings,
              mean_expand,
              max_expand,
              group_embeddings - mean_expand,
          ],
          dim=1,
      )
      logits.index_copy_(0, idx_t, self.score(score_input).squeeze(-1))
    return logits

  def source_features(
      self,
      x: torch.Tensor,
      groups: list[list[int]] | None = None,
  ) -> torch.Tensor:
    del groups
    return self.encoder(x)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20261560)
  parser.add_argument("--train-source-indices", default="")
  parser.add_argument("--validation-source-indices", default="6")
  parser.add_argument("--epochs", type=int, default=240)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--hidden", type=int, default=96)
  parser.add_argument("--dropout", type=float, default=0.08)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument(
      "--pos-weight-mode",
      choices=("balanced", "fixed"),
      default="balanced",
  )
  parser.add_argument("--positive-weight", type=float, default=1.0)
  parser.add_argument("--focal-gamma", type=float, default=0.0)
  parser.add_argument(
      "--loss-mode",
      choices=("bce", "pairwise", "bce_pairwise", "listwise", "bce_listwise"),
      default="bce",
      help=(
          "bce preserves the historical independent approval loss. pairwise "
          "ranks approved scored candidates above rejected candidates inside "
          "the same proposal state. listwise trains a top-1 choice inside each "
          "proposal state. Combined modes add the grouped loss to BCE."
      ),
  )
  parser.add_argument("--pairwise-loss-weight", type=float, default=1.0)
  parser.add_argument("--pairwise-margin", type=float, default=1.0)
  parser.add_argument("--listwise-loss-weight", type=float, default=1.0)
  parser.add_argument(
      "--unsafe-margin-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Extra grouped margin loss that pushes clean alternatives above "
          "candidate-any / forced-lane-burn proposals in the same state. This "
          "is default-off to preserve old checkpoints and screens."
      ),
  )
  parser.add_argument("--unsafe-margin", type=float, default=1.0)
  parser.add_argument(
      "--unsafe-positive-mode",
      choices=("approved", "clean_safe", "clean_any"),
      default="clean_safe",
      help=(
          "Which same-state alternatives should outrank unsafe candidates for "
          "--unsafe-margin-loss-weight. clean_safe uses paradox-safe, no-any, "
          "non-forced-lane-burn candidates, even when they are not dense-value "
          "positive."
      ),
  )
  parser.add_argument(
      "--density-contrast-loss-weight",
      type=float,
      default=0.0,
      help=(
          "Default-off grouped contrast loss for source-10-density slates. "
          "When a same-state scored candidate slate has high candidate-any "
          "density and sparse approved/no-any actions, push the sparse clean "
          "actions above candidate-any actions."
      ),
  )
  parser.add_argument("--density-contrast-margin", type=float, default=2.0)
  parser.add_argument(
      "--density-contrast-min-candidate-any-rate",
      type=float,
      default=0.65,
  )
  parser.add_argument(
      "--density-contrast-max-approved-rate",
      type=float,
      default=0.30,
  )
  parser.add_argument("--density-contrast-min-group-size", type=int, default=6)
  parser.add_argument(
      "--density-contrast-source-indices",
      type=str,
      default="",
      help=(
          "Comma-separated source_replay_index allowlist for density contrast "
          "groups. Empty means all sources. This lets new high-density slate "
          "lessons avoid perturbing older replay families."
      ),
  )
  parser.add_argument(
      "--source-adversary-weight",
      type=float,
      default=0.0,
      help=(
          "Optional training-only gradient-reversal source classifier weight. "
          "Use this to discourage proposal-veto representations from "
          "memorizing replay/source-specific score quirks."
      ),
  )
  parser.add_argument(
      "--source-balance-weight",
      type=float,
      default=1.0,
      help=(
          "0 disables inverse-source weighting for the adversary; 1 fully "
          "balances source classes when source-adversary-weight is enabled."
      ),
  )
  parser.add_argument(
      "--main-source-balance-weight",
      type=float,
      default=0.0,
      help=(
          "Default-off inverse-source weighting for the main BCE/listwise/"
          "pairwise losses. Use this when small hard-negative sources should "
          "count like large positive sources."
      ),
  )
  parser.add_argument("--positive-value-margin", type=float, default=0.01)
  parser.add_argument("--safe-risk-delta-max", type=float, default=0.0)
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
      "--base-action-source",
      choices=("metadata_policy_best", "policy_context_argmax"),
      default="metadata_policy_best",
  )
  parser.add_argument(
      "--proposal-feature-mode",
      choices=(
          "full",
          "scalar_only",
          "scalar_action_flags",
          "scalar_action_flags_slate",
          "scalar_action_flags_slate_state_context",
          "scalar_action_flags_slate_state_round_history_context",
          "scalar_action_flags_win_slate_state_context",
          "scalar_action_flags_win_slate_state_round_history_context",
          "scalar_action_flags_win_aux_slate_state_context",
          "scalar_action_flags_win_aux_slate_state_round_history_context",
          "full_no_source",
          "full_no_source_public_slate",
      ),
      default="full",
      help=(
          "Feature family for the candidate-vs-base approval MLP. full keeps "
          "the original high-dimensional candidate/base action features; "
          "scalar_only uses only proposal scores/probabilities; "
          "scalar_action_flags adds compact public follow-led/red/token-loss "
          "flags for candidate and base; scalar_action_flags_slate also adds "
          "candidate-rank and score/probability context within the scored "
          "candidate slate; scalar_action_flags_slate_state_context appends "
          "compact legal state/history counters plus the existing "
          "observation/policy-context tail from the pair features; "
          "scalar_action_flags_slate_state_round_history_context also appends "
          "prior round-history summary counters from --record-round-history "
          "collector metadata. The *_win_* modes add whether a lane-burning "
          "switch would win the current trick. The *_win_aux_* modes also add "
          "candidate/base action-value and paradox-risk aux-head predictions, "
          "preserving old feature sizes for existing checkpoints. "
          "full_no_source keeps the full candidate/base feature vectors and "
          "their difference, but drops proposal source-score/probability "
          "scalars so source-rank quirks cannot dominate same-lane contrasts. "
          "full_no_source_public_slate keeps that source-score-free base and "
          "adds outcome-free rank/color/follow-led/liquidity context within "
          "the current scored candidate slate."
      ),
  )
  parser.add_argument(
      "--proposal-model-kind",
      choices=("mlp", "deepset"),
      default="mlp",
      help=(
          "mlp scores each candidate independently from its feature vector. "
          "deepset adds same-state slate context by pooling candidate "
          "embeddings before scoring."
      ),
  )
  parser.add_argument(
      "--proposal-example-source",
      choices=("top_proposal", "scored_candidates"),
      default="top_proposal",
      help=(
          "top_proposal keeps the historical one-example-per-proposal path. "
          "scored_candidates expands every outcome-labeled scored candidate "
          "against the same base action, enabling multi-candidate best-of-N "
          "verifier training from Run914-style metadata."
      ),
  )
  parser.add_argument(
      "--target-mode",
      choices=(
          "best_positive_else_base",
          "best_value_positive_else_base",
          "group_relative_soft",
      ),
      default="best_value_positive_else_base",
  )
  parser.add_argument("--soft-target-temperature", type=float, default=0.35)
  parser.add_argument("--soft-target-risk-penalty", type=float, default=1.0)
  parser.add_argument("--soft-target-unsafe-penalty", type=float, default=1.0)
  parser.add_argument(
      "--soft-target-source-value-penalty",
      type=float,
      default=0.0,
      help=(
          "Optional group-relative reward penalty for candidates whose source "
          "action scorer predicts value delta below --soft-target-min-source-value-delta."
      ),
  )
  parser.add_argument(
      "--soft-target-source-positive-penalty",
      type=float,
      default=0.0,
      help=(
          "Optional group-relative reward penalty for candidates whose source "
          "action scorer predicts positive probability below "
          "--soft-target-min-source-positive-prob."
      ),
  )
  parser.add_argument(
      "--soft-target-source-margin-penalty",
      type=float,
      default=0.0,
      help=(
          "Optional group-relative reward penalty for candidates whose source "
          "score margin is below --soft-target-min-source-margin."
      ),
  )
  parser.add_argument("--soft-target-min-source-value-delta", type=float, default=0.0)
  parser.add_argument("--soft-target-min-source-positive-prob", type=float, default=0.5)
  parser.add_argument("--soft-target-min-source-margin", type=float, default=0.0)
  parser.add_argument(
      "--validation-min-source-value-delta",
      type=float,
      default=-1e9,
      help=(
          "Validation/report-only source agreement guard. Threshold reports "
          "will not approve candidates whose source model predicted value "
          "delta is below this value. Default keeps historical behavior."
      ),
  )
  parser.add_argument(
      "--validation-min-source-positive-prob",
      type=float,
      default=-1e9,
      help=(
          "Validation/report-only source agreement guard. Threshold reports "
          "will not approve candidates whose source model positive "
          "probability is below this value."
      ),
  )
  parser.add_argument(
      "--validation-min-source-margin",
      type=float,
      default=-1e9,
      help=(
          "Validation/report-only source agreement guard. Threshold reports "
          "will not approve candidates whose source score margin over the "
          "base action is below this value."
      ),
  )
  parser.add_argument("--force-led-lane-burn-negative", action="store_true")
  parser.add_argument(
      "--bad-red-no-win-negative",
      action="store_true",
      help=(
          "Default-off stricter negative label for red off-led lane burns that "
          "do not win the current trick. This targets the bad-red/no-win hard "
          "negative family directly instead of relying only on generic unsafe "
          "or forced-lane-burn labels."
      ),
  )
  parser.add_argument(
      "--bad-red-no-win-penalty",
      type=float,
      default=0.0,
      help=(
          "Extra group-relative reward penalty for candidates marked by "
          "--bad-red-no-win-negative."
      ),
  )
  parser.add_argument(
      "--led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="urgent_win",
  )
  parser.add_argument(
      "--approve-label",
      choices=(
          "positive_safe",
          "positive",
          "safe",
          "positive_safe_no_candidate_any_paradox",
          "positive_no_candidate_any_paradox",
          "safe_no_candidate_any_paradox",
      ),
      default="positive_safe",
      help=(
          "Which verified proposal outcome is treated as an approval target. "
          "positive_safe requires both dense-positive and paradox-safe. "
          "*_no_candidate_any_paradox variants also require the candidate's "
          "own continuation to avoid any paradox."
      ),
  )
  parser.add_argument(
      "--approval-thresholds",
      default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.97,0.99",
  )
  parser.add_argument("--min-validation-approvals-for-live", type=int, default=4)
  parser.add_argument(
      "--min-validation-approval-precision-for-live", type=float, default=0.75
  )
  parser.add_argument(
      "--min-validation-approval-recall-for-live", type=float, default=0.10
  )
  parser.add_argument(
      "--max-validation-false-approval-rate-for-live", type=float, default=0.25
  )
  parser.add_argument("--min-validation-safe-rate-for-live", type=float, default=0.90)
  parser.add_argument(
      "--min-validation-mean-value-delta-for-live", type=float, default=0.0
  )
  parser.add_argument(
      "--max-validation-mean-risk-delta-for-live", type=float, default=0.0
  )
  parser.add_argument(
      "--max-validation-candidate-any-paradox-rate-for-live",
      type=float,
      default=0.40,
      help=(
          "Maximum allowed fraction of approved validation candidates whose "
          "own continuation has any paradox. This catches value-positive "
          "choices that are only safe by delta because the base was bad too."
      ),
  )
  parser.add_argument(
      "--max-negative-validation-approvals-for-live",
      type=int,
      default=0,
      help=(
          "For validation sources with zero approval targets, allow live "
          "screening only when the guarded threshold approves at most this "
          "many candidates. Default requires strict abstention."
      ),
  )
  parser.add_argument(
      "--max-negative-validation-candidate-any-approvals-for-live",
      type=int,
      default=0,
      help=(
          "For validation sources with zero approval targets, reject thresholds "
          "that approve more than this many candidate-any-paradox examples."
      ),
  )
  parser.add_argument(
      "--max-negative-validation-approval-rate-for-live",
      type=float,
      default=0.0,
      help=(
          "For validation sources with zero approval targets, allow at most "
          "this guarded approval fraction. Default requires zero approvals."
      ),
  )
  return parser.parse_args()


def _source_examples(examples: list[dict], sources: set[int]) -> list[dict]:
  return [
      example for example in examples
      if int(example.get("source_replay_index", -1)) in sources
  ]


def _find_action(state: dict, action_id: int) -> dict | None:
  for action in state["actions"]:
    if int(action["action"]) == int(action_id):
      return action
  return None


def _approval_label(proposal: dict, mode: str) -> bool:
  positive = bool(proposal.get("positive_by_dense_delta", False))
  safe = bool(proposal.get("safe_by_paradox_delta", False))
  no_candidate_any = not bool(proposal.get("candidate_any_paradox", False))
  if mode == "positive":
    return positive
  if mode == "safe":
    return safe
  if mode == "positive_safe_no_candidate_any_paradox":
    return bool(positive and safe and no_candidate_any)
  if mode == "positive_no_candidate_any_paradox":
    return bool(positive and no_candidate_any)
  if mode == "safe_no_candidate_any_paradox":
    return bool(safe and no_candidate_any)
  return bool(positive and safe)


def _group_relative_reward(args, proposal: dict, approved: bool) -> float:
  reward = float(proposal.get("dense_action_value_delta_vs_base", 0.0))
  risk_delta = float(proposal.get("paradox_delta_vs_base", 0.0))
  reward -= float(args.soft_target_risk_penalty) * max(0.0, risk_delta)
  source_value = float(proposal.get("candidate_pred_value_delta", 0.0))
  source_positive = float(proposal.get("candidate_pred_positive_prob", 0.0))
  source_margin = float(proposal.get("score_margin", 0.0))
  reward -= float(args.soft_target_source_value_penalty) * max(
      0.0,
      float(args.soft_target_min_source_value_delta) - source_value,
  )
  reward -= float(args.soft_target_source_positive_penalty) * max(
      0.0,
      float(args.soft_target_min_source_positive_prob) - source_positive,
  )
  reward -= float(args.soft_target_source_margin_penalty) * max(
      0.0,
      float(args.soft_target_min_source_margin) - source_margin,
  )
  if bool(proposal.get("candidate_any_paradox", False)):
    reward -= float(args.soft_target_unsafe_penalty)
  if bool(proposal.get("candidate_actor_paradox", False)):
    reward -= float(args.soft_target_unsafe_penalty)
  if bool(proposal.get("candidate_forced_paradox", False)):
    reward -= float(args.soft_target_unsafe_penalty)
  if bool(proposal.get("forced_led_lane_burn_negative", False)):
    reward -= float(args.soft_target_unsafe_penalty)
  if bool(proposal.get("bad_red_no_win_negative", False)):
    reward -= float(getattr(args, "bad_red_no_win_penalty", 0.0))
  if approved:
    reward += 0.25
  return float(reward)


def _float_from_any(mapping: dict, keys: tuple[str, ...]) -> float:
  for key in keys:
    if key in mapping:
      try:
        return float(mapping.get(key, 0.0))
      except (TypeError, ValueError):
        return 0.0
  return 0.0


def _slate_metric_values(scored: list, aliases: tuple[str, ...]) -> list[float]:
  values = []
  for item in scored:
    if isinstance(item, dict):
      values.append(_float_from_any(item, aliases))
  return values


def _slate_metric_fields(
    name: str,
    current: float,
    values: list[float],
    rank: int,
) -> dict:
  if not values:
    values = [0.0]
  arr = np.asarray(values, dtype=np.float32)
  max_value = float(np.max(arr))
  min_value = float(np.min(arr))
  mean_value = float(np.mean(arr))
  std_value = float(np.std(arr))
  other_values = [
      float(value) for idx, value in enumerate(values) if int(idx) != int(rank)
  ]
  best_other = max(other_values) if other_values else float(current)
  scale = max(1e-6, std_value)
  return {
      f"{name}_slate_max": max_value,
      f"{name}_slate_min": min_value,
      f"{name}_slate_mean": mean_value,
      f"{name}_slate_std": std_value,
      f"{name}_delta_to_slate_best": float(current) - max_value,
      f"{name}_delta_to_slate_mean": float(current) - mean_value,
      f"{name}_slate_z": (float(current) - mean_value) / scale,
      f"{name}_delta_to_best_other": float(current) - float(best_other),
      f"{name}_is_slate_best": 1.0 if float(current) >= max_value - 1e-9 else 0.0,
  }


def _slate_relative_fields(
    item: dict,
    scored: list,
    rank: int,
    total: int,
) -> dict:
  rank_norm = 0.0 if int(total) <= 1 else float(rank) / float(max(1, total - 1))
  fields = {
      "source_candidate_rank": float(rank),
      "source_candidate_count": float(total),
      "source_candidate_rank_norm": rank_norm,
      "source_candidate_inverse_rank_norm": 1.0 - rank_norm,
      "source_candidate_is_top_ranked": 1.0 if int(rank) == 0 else 0.0,
  }
  metrics = [
      ("score", ("score",)),
      ("margin", ("margin", "score_margin")),
      ("pred_value_delta", ("pred_value_delta", "candidate_pred_value_delta")),
      (
          "pred_positive_prob",
          ("pred_positive_prob", "candidate_pred_positive_prob"),
      ),
      ("policy_prob", ("policy_prob", "candidate_policy_prob")),
  ]
  for name, aliases in metrics:
    current = _float_from_any(item, aliases)
    values = _slate_metric_values(scored, aliases)
    fields.update(_slate_metric_fields(name, current, values, int(rank)))
  return fields


PUBLIC_SLATE_FEATURE_NAMES = (
    "public_candidate_rank_norm",
    "public_base_rank_norm",
    "public_rank_delta_to_base",
    "public_candidate_follows_led",
    "public_candidate_is_red",
    "public_candidate_would_win",
    "public_candidate_token_loss",
    "public_slate_count_norm",
    "public_slate_follow_led_count_norm",
    "public_slate_red_count_norm",
    "public_slate_switch_count_norm",
    "public_slate_token_loss_count_norm",
    "public_slate_win_count_norm",
    "public_slate_responsible_red_win_count_norm",
    "public_color_group_count_norm",
    "public_color_rank_pct",
    "public_color_rank_inv_pct",
    "public_color_lower_count_norm",
    "public_color_higher_count_norm",
    "public_color_is_lowest_rank",
    "public_color_is_highest_rank",
    "public_color_rank_delta_to_min_norm",
    "public_color_rank_delta_to_max_norm",
    "public_color_rank_delta_to_mean_norm",
    "public_follow_group_count_norm",
    "public_follow_rank_pct",
    "public_follow_rank_inv_pct",
    "public_follow_lower_count_norm",
    "public_follow_higher_count_norm",
    "public_follow_is_lowest_rank",
    "public_follow_is_highest_rank",
    "public_follow_rank_delta_to_min_norm",
    "public_follow_rank_delta_to_max_norm",
    "public_follow_rank_delta_to_mean_norm",
    "public_red_group_count_norm",
    "public_red_rank_pct",
    "public_red_rank_inv_pct",
    "public_red_lower_count_norm",
    "public_red_higher_count_norm",
    "public_red_is_lowest_rank",
    "public_red_is_highest_rank",
    "public_red_rank_delta_to_min_norm",
    "public_red_rank_delta_to_max_norm",
    "public_red_rank_delta_to_mean_norm",
    "public_candidate_future_lead_delta_to_all_max",
    "public_candidate_future_lead_delta_to_all_mean",
    "public_candidate_lane_damage_delta_to_all_min",
    "public_candidate_lane_damage_delta_to_all_mean",
    "public_follow_future_lead_delta_to_max",
    "public_follow_future_lead_delta_to_mean",
    "public_follow_lane_damage_delta_to_min",
    "public_follow_lane_damage_delta_to_mean",
    "public_color_future_lead_delta_to_max",
    "public_color_future_lead_delta_to_mean",
    "public_color_lane_damage_delta_to_min",
    "public_color_lane_damage_delta_to_mean",
)


def _parse_rank_color(action_string: str | None) -> tuple[int | None, str | None]:
  if not isinstance(action_string, str):
    return None, None
  marker = "rank="
  start = action_string.find(marker)
  if start < 0:
    return None, None
  start += len(marker)
  end = start
  while end < len(action_string) and action_string[end].isdigit():
    end += 1
  try:
    rank = int(action_string[start:end])
  except ValueError:
    return None, None
  color_marker = "color="
  color_start = action_string.find(color_marker, end)
  if color_start < 0:
    return rank, None
  color_start += len(color_marker)
  color_end = color_start
  while color_end < len(action_string) and action_string[color_end].isalpha():
    color_end += 1
  color = action_string[color_start:color_end] or None
  return rank, color


def _norm_rank(rank: int | None) -> float:
  if rank is None:
    return 0.0
  return min(1.0, max(0.0, float(rank) / 9.0))


def _feature_value(feature: np.ndarray, idx: int) -> float:
  if int(idx) < 0 or int(idx) >= int(feature.shape[0]):
    return 0.0
  try:
    return float(feature[int(idx)])
  except (TypeError, ValueError):
    return 0.0


def _public_group_rank_fields(
    prefix: str,
    candidate_rank: int | None,
    ranks: list[int],
    include_rank_position: bool,
    total: int,
) -> dict:
  denom = max(1.0, float(total))
  fields = {
      f"{prefix}_count_norm": min(1.0, float(len(ranks)) / denom),
      f"{prefix}_rank_pct": 0.0,
      f"{prefix}_rank_inv_pct": 0.0,
      f"{prefix}_lower_count_norm": 0.0,
      f"{prefix}_higher_count_norm": 0.0,
      f"{prefix}_is_lowest_rank": 0.0,
      f"{prefix}_is_highest_rank": 0.0,
      f"{prefix}_rank_delta_to_min_norm": 0.0,
      f"{prefix}_rank_delta_to_max_norm": 0.0,
      f"{prefix}_rank_delta_to_mean_norm": 0.0,
  }
  if not include_rank_position or candidate_rank is None or not ranks:
    return fields
  lower = sum(1 for rank in ranks if int(rank) < int(candidate_rank))
  higher = sum(1 for rank in ranks if int(rank) > int(candidate_rank))
  group_denom = max(1.0, float(len(ranks) - 1))
  rank_pct = float(lower) / group_denom
  min_rank = float(min(ranks))
  max_rank = float(max(ranks))
  mean_rank = float(np.mean(np.asarray(ranks, dtype=np.float32)))
  fields.update({
      f"{prefix}_rank_pct": rank_pct,
      f"{prefix}_rank_inv_pct": 1.0 - rank_pct,
      f"{prefix}_lower_count_norm": float(lower) / denom,
      f"{prefix}_higher_count_norm": float(higher) / denom,
      f"{prefix}_is_lowest_rank": 1.0 if lower == 0 else 0.0,
      f"{prefix}_is_highest_rank": 1.0 if higher == 0 else 0.0,
      f"{prefix}_rank_delta_to_min_norm": (
          float(candidate_rank) - min_rank
      ) / 8.0,
      f"{prefix}_rank_delta_to_max_norm": (
          float(candidate_rank) - max_rank
      ) / 8.0,
      f"{prefix}_rank_delta_to_mean_norm": (
          float(candidate_rank) - mean_rank
      ) / 8.0,
  })
  return fields


def _public_delta_fields(
    prefix: str,
    current: float,
    values: list[float],
    high_is_good: bool,
) -> dict:
  if not values:
    return {
        f"{prefix}_max" if high_is_good else f"{prefix}_min": 0.0,
        f"{prefix}_mean": 0.0,
    }
  arr = np.asarray(values, dtype=np.float32)
  mean_value = float(np.mean(arr))
  if high_is_good:
    best = float(np.max(arr))
  else:
    best = float(np.min(arr))
  return {
      f"{prefix}_max" if high_is_good else f"{prefix}_min": float(current) - best,
      f"{prefix}_mean": float(current) - mean_value,
  }


def _public_action_slate_fields(
    candidate_view: dict,
    candidate_feature: np.ndarray,
    base_feature: np.ndarray,
    state: dict,
    candidate_views: list[dict],
) -> dict:
  total = max(1, len(candidate_views))
  candidate_rank, candidate_color = _parse_rank_color(
      candidate_view.get("candidate_action_string")
  )
  base_rank, _ = _parse_rank_color(candidate_view.get("base_action_string"))
  candidate_token_loss = max(
      _action_feature_bool(candidate_feature, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX),
      _action_loses_led_lane(candidate_feature),
  )
  fields = {
      "public_candidate_rank_norm": _norm_rank(candidate_rank),
      "public_base_rank_norm": _norm_rank(base_rank),
      "public_rank_delta_to_base": (
          _norm_rank(candidate_rank) - _norm_rank(base_rank)
      ),
      "public_candidate_follows_led": _action_feature_bool(
          candidate_feature,
          ACTION_FEATURE_FOLLOWS_LED_INDEX,
      ),
      "public_candidate_is_red": _action_feature_bool(
          candidate_feature,
          ACTION_FEATURE_IS_RED_INDEX,
      ),
      "public_candidate_would_win": _action_feature_bool(
          candidate_feature,
          ACTION_FEATURE_WOULD_WIN_INDEX,
      ),
      "public_candidate_token_loss": candidate_token_loss,
      "public_slate_count_norm": min(1.0, float(total) / 20.0),
  }

  slate_items = []
  for view in candidate_views:
    action_id = int(view.get("candidate_action", -1))
    action = _find_action(state, action_id)
    if action is None:
      continue
    feature = np.asarray(
        action.get("raw_action_feature", action.get("feature")),
        dtype=np.float32,
    )
    rank, color = _parse_rank_color(view.get("candidate_action_string"))
    token_loss = max(
        _action_feature_bool(feature, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX),
        _action_loses_led_lane(feature),
    )
    slate_items.append({
        "rank": rank,
        "color": color,
        "follows_led": _action_feature_bool(
            feature,
            ACTION_FEATURE_FOLLOWS_LED_INDEX,
        ),
        "is_red": _action_feature_bool(feature, ACTION_FEATURE_IS_RED_INDEX),
        "switches_off_led": 1.0
        if bool(view.get("candidate_switches_off_led", False)) else 0.0,
        "token_loss": token_loss,
        "would_win": _action_feature_bool(feature, ACTION_FEATURE_WOULD_WIN_INDEX),
        "responsible_red_win": 1.0
        if bool(view.get("candidate_responsible_red_win_switch", False)) else 0.0,
        "future_lead": _feature_value(
            feature,
            APPENDED_ACTION_FEATURE_INDEX.get(
                "own_future_legal_lead_count_after",
                -1,
            ),
        ),
        "lane_damage": _feature_value(
            feature,
            APPENDED_ACTION_FEATURE_INDEX.get("exit_lane_surplus_damage", -1),
        ),
    })

  def sum_flag(name: str) -> float:
    return min(
        1.0,
        float(sum(1 for item in slate_items if float(item.get(name, 0.0)) > 0.5))
        / float(total),
    )

  fields.update({
      "public_slate_follow_led_count_norm": sum_flag("follows_led"),
      "public_slate_red_count_norm": sum_flag("is_red"),
      "public_slate_switch_count_norm": sum_flag("switches_off_led"),
      "public_slate_token_loss_count_norm": sum_flag("token_loss"),
      "public_slate_win_count_norm": sum_flag("would_win"),
      "public_slate_responsible_red_win_count_norm": sum_flag(
          "responsible_red_win"
      ),
  })

  candidate_follows = fields["public_candidate_follows_led"] > 0.5
  candidate_is_red = fields["public_candidate_is_red"] > 0.5
  color_ranks = [
      int(item["rank"]) for item in slate_items
      if item.get("rank") is not None and item.get("color") == candidate_color
  ]
  follow_ranks = [
      int(item["rank"]) for item in slate_items
      if item.get("rank") is not None and float(item.get("follows_led", 0.0)) > 0.5
  ]
  red_ranks = [
      int(item["rank"]) for item in slate_items
      if item.get("rank") is not None and float(item.get("is_red", 0.0)) > 0.5
  ]
  fields.update(_public_group_rank_fields(
      "public_color_group",
      candidate_rank,
      color_ranks,
      candidate_color is not None,
      total,
  ))
  fields.update(_public_group_rank_fields(
      "public_follow_group",
      candidate_rank,
      follow_ranks,
      candidate_follows,
      total,
  ))
  fields.update(_public_group_rank_fields(
      "public_red_group",
      candidate_rank,
      red_ranks,
      candidate_is_red,
      total,
  ))

  current_future_lead = _feature_value(
      candidate_feature,
      APPENDED_ACTION_FEATURE_INDEX.get("own_future_legal_lead_count_after", -1),
  )
  current_lane_damage = _feature_value(
      candidate_feature,
      APPENDED_ACTION_FEATURE_INDEX.get("exit_lane_surplus_damage", -1),
  )
  all_future_leads = [float(item["future_lead"]) for item in slate_items]
  all_lane_damages = [float(item["lane_damage"]) for item in slate_items]
  fields.update(_public_delta_fields(
      "public_candidate_future_lead_delta_to_all",
      current_future_lead,
      all_future_leads,
      True,
  ))
  fields.update(_public_delta_fields(
      "public_candidate_lane_damage_delta_to_all",
      current_lane_damage,
      all_lane_damages,
      False,
  ))

  follow_items = [
      item for item in slate_items if float(item.get("follows_led", 0.0)) > 0.5
  ]
  color_items = [
      item for item in slate_items if item.get("color") == candidate_color
  ]
  fields.update(_public_delta_fields(
      "public_follow_future_lead_delta_to",
      current_future_lead,
      [float(item["future_lead"]) for item in follow_items] if candidate_follows else [],
      True,
  ))
  fields.update(_public_delta_fields(
      "public_follow_lane_damage_delta_to",
      current_lane_damage,
      [float(item["lane_damage"]) for item in follow_items] if candidate_follows else [],
      False,
  ))
  fields.update(_public_delta_fields(
      "public_color_future_lead_delta_to",
      current_future_lead,
      [float(item["future_lead"]) for item in color_items],
      True,
  ))
  fields.update(_public_delta_fields(
      "public_color_lane_damage_delta_to",
      current_lane_damage,
      [float(item["lane_damage"]) for item in color_items],
      False,
  ))
  return {
      name: float(fields.get(name, 0.0))
      for name in PUBLIC_SLATE_FEATURE_NAMES
  }


def _proposal_public_slate_features(proposal: dict) -> np.ndarray:
  return np.asarray(
      [
          _float_from_any(proposal, (name,))
          for name in PUBLIC_SLATE_FEATURE_NAMES
      ],
      dtype=np.float32,
  )


def _proposal_scalar_features(proposal: dict) -> np.ndarray:
  keys = [
      ("score_margin", "margin"),
      ("candidate_score", "score"),
      ("base_score",),
      ("candidate_pred_value_delta", "pred_value_delta"),
      ("candidate_pred_positive_prob", "pred_positive_prob"),
      ("candidate_policy_prob", "policy_prob"),
      ("base_policy_prob",),
      ("candidate_count",),
      ("scored_candidate_count",),
  ]
  values = []
  for aliases in keys:
    values.append(_float_from_any(proposal, aliases))
  return np.asarray(values, dtype=np.float32)


def _proposal_aux_features(proposal: dict) -> np.ndarray:
  candidate_value = _float_from_any(proposal, ("candidate_aux_action_value",))
  base_value = _float_from_any(proposal, ("base_aux_action_value",))
  value_delta = _float_from_any(
      proposal,
      ("aux_action_value_delta", "candidate_aux_action_value_delta"),
  )
  candidate_risk = _float_from_any(
      proposal,
      ("candidate_aux_action_paradox_risk",),
  )
  base_risk = _float_from_any(proposal, ("base_aux_action_paradox_risk",))
  risk_delta = _float_from_any(
      proposal,
      (
          "aux_action_paradox_risk_delta",
          "candidate_aux_action_paradox_risk_delta",
      ),
  )
  candidate_survival = candidate_value - candidate_risk
  base_survival = base_value - base_risk
  return np.asarray(
      [
          candidate_value,
          base_value,
          value_delta,
          candidate_risk,
          base_risk,
          risk_delta,
          candidate_survival,
          base_survival,
          candidate_survival - base_survival,
      ],
      dtype=np.float32,
  )


def _proposal_slate_features(proposal: dict) -> np.ndarray:
  keys = [
      ("source_candidate_rank",),
      ("source_candidate_count",),
      ("source_candidate_rank_norm",),
      ("source_candidate_inverse_rank_norm",),
      ("source_candidate_is_top_ranked",),
      ("score_slate_max",),
      ("score_slate_min",),
      ("score_slate_mean",),
      ("score_slate_std",),
      ("score_delta_to_slate_best",),
      ("score_delta_to_slate_mean",),
      ("score_slate_z",),
      ("score_delta_to_best_other",),
      ("score_is_slate_best",),
      ("margin_slate_max",),
      ("margin_slate_min",),
      ("margin_slate_mean",),
      ("margin_slate_std",),
      ("margin_delta_to_slate_best",),
      ("margin_delta_to_slate_mean",),
      ("margin_slate_z",),
      ("margin_delta_to_best_other",),
      ("margin_is_slate_best",),
      ("pred_value_delta_slate_max",),
      ("pred_value_delta_slate_min",),
      ("pred_value_delta_slate_mean",),
      ("pred_value_delta_slate_std",),
      ("pred_value_delta_delta_to_slate_best",),
      ("pred_value_delta_delta_to_slate_mean",),
      ("pred_value_delta_slate_z",),
      ("pred_value_delta_delta_to_best_other",),
      ("pred_value_delta_is_slate_best",),
      ("pred_positive_prob_slate_max",),
      ("pred_positive_prob_slate_min",),
      ("pred_positive_prob_slate_mean",),
      ("pred_positive_prob_slate_std",),
      ("pred_positive_prob_delta_to_slate_best",),
      ("pred_positive_prob_delta_to_slate_mean",),
      ("pred_positive_prob_slate_z",),
      ("pred_positive_prob_delta_to_best_other",),
      ("pred_positive_prob_is_slate_best",),
      ("policy_prob_slate_max",),
      ("policy_prob_slate_min",),
      ("policy_prob_slate_mean",),
      ("policy_prob_slate_std",),
      ("policy_prob_delta_to_slate_best",),
      ("policy_prob_delta_to_slate_mean",),
      ("policy_prob_slate_z",),
      ("policy_prob_delta_to_best_other",),
      ("policy_prob_is_slate_best",),
  ]
  return np.asarray([
      _float_from_any(proposal, aliases) for aliases in keys
  ], dtype=np.float32)


def _state_context_fields(metadata: dict) -> dict:
  group = metadata.get("group") if isinstance(metadata, dict) else {}
  if not isinstance(group, dict):
    group = {}
  led = metadata.get("led_contrast") if isinstance(metadata, dict) else {}
  if not isinstance(led, dict):
    led = {}
  legal_count = float(metadata.get("legal_count", 0.0))
  led_legal = float(led.get("led_legal_count", 0.0))
  red_legal = float(led.get("red_legal_count", 0.0))
  switch_legal = float(led.get("switch_legal_count", 0.0))
  denom = max(1.0, legal_count)
  return {
      "state_trick_number_norm": float(metadata.get("trick_number", 0.0)) / 8.0,
      "state_legal_count_norm": min(1.0, legal_count / 40.0),
      "state_decision_index_norm": (
          float(group.get("decision_index", 0.0)) / 35.0
      ),
      "state_round_index_norm": float(group.get("round_index", 0.0)) / 5.0,
      "state_start_player_norm": float(group.get("start_player", 0.0)) / 4.0,
      "state_led_legal_count_norm": min(1.0, led_legal / 20.0),
      "state_red_legal_count_norm": min(1.0, red_legal / 20.0),
      "state_switch_legal_count_norm": min(1.0, switch_legal / 40.0),
      "state_led_legal_ratio": led_legal / denom,
      "state_red_legal_ratio": red_legal / denom,
      "state_switch_legal_ratio": switch_legal / denom,
      "state_avoidable_switch": 1.0 if bool(led.get("avoidable_switch", False)) else 0.0,
  }


def _round_history_context_fields(metadata: dict) -> dict:
  if not isinstance(metadata, dict):
    metadata = {}
  summary = metadata.get("round_history_summary")
  if not isinstance(summary, dict):
    summary = {}
  last_event = summary.get("last_event")
  if not isinstance(last_event, dict):
    last_event = {}

  def get_float(key: str) -> float:
    try:
      return float(summary.get(key, 0.0))
    except (TypeError, ValueError):
      return 0.0

  def get_last_float(key: str) -> float:
    try:
      return float(last_event.get(key, 0.0))
    except (TypeError, ValueError):
      return 0.0

  def bool_float(mapping: dict, key: str) -> float:
    return 1.0 if bool(mapping.get(key, False)) else 0.0

  prior_events = get_float("prior_events")
  prior_play = get_float("prior_play_events")
  own_play = get_float("own_prior_play_events")
  other_play = get_float("other_prior_play_events")
  return {
      "history_available": 1.0 if bool(summary) else 0.0,
      "history_prior_events_norm": min(1.0, prior_events / 40.0),
      "history_prior_play_events_norm": min(1.0, prior_play / 30.0),
      "history_own_prior_play_events_norm": min(1.0, own_play / 10.0),
      "history_other_prior_play_events_norm": min(1.0, other_play / 25.0),
      "history_prior_follow_led_rate": get_float("prior_follow_led_rate"),
      "history_prior_switch_rate": get_float("prior_switch_rate"),
      "history_prior_avoidable_switch_rate": get_float(
          "prior_avoidable_switch_rate"
      ),
      "history_prior_red_when_switch_rate": get_float(
          "prior_red_when_switch_rate"
      ),
      "history_prior_red_when_avoidable_switch_rate": get_float(
          "prior_red_when_avoidable_switch_rate"
      ),
      "history_prior_nonred_when_avoidable_switch_rate": get_float(
          "prior_nonred_when_avoidable_switch_rate"
      ),
      "history_prior_token_loss_rate": get_float("prior_token_loss_rate"),
      "history_prior_avoidable_token_loss_rate": get_float(
          "prior_avoidable_token_loss_rate"
      ),
      "history_own_prior_token_loss_rate": get_float(
          "own_prior_token_loss_rate"
      ),
      "history_other_prior_token_loss_rate": get_float(
          "other_prior_token_loss_rate"
      ),
      "history_last_event_is_play": 1.0
      if str(last_event.get("phase", "")) == "play" else 0.0,
      "history_last_event_follows_led": bool_float(last_event, "follows_led"),
      "history_last_event_is_red": bool_float(last_event, "is_red"),
      "history_last_event_switches_off_led": bool_float(
          last_event, "switches_off_led"
      ),
      "history_last_event_avoidable_switch": bool_float(
          last_event, "avoidable_switch"
      ),
      "history_last_event_avoidable_nonred_switch": bool_float(
          last_event, "avoidable_nonred_switch"
      ),
      "history_last_event_off_led_loses_token": bool_float(
          last_event, "off_led_loses_token"
      ),
      "history_last_event_led_legal_ratio": get_last_float("led_legal_count")
      / max(1.0, get_last_float("legal_count")),
      "history_last_event_red_legal_ratio": get_last_float("red_legal_count")
      / max(1.0, get_last_float("legal_count")),
      "history_last_event_switch_legal_ratio": get_last_float(
          "switch_legal_count"
      ) / max(1.0, get_last_float("legal_count")),
  }


def _proposal_state_context_features(proposal: dict) -> np.ndarray:
  keys = [
      ("state_trick_number_norm",),
      ("state_legal_count_norm",),
      ("state_decision_index_norm",),
      ("state_round_index_norm",),
      ("state_start_player_norm",),
      ("state_led_legal_count_norm",),
      ("state_red_legal_count_norm",),
      ("state_switch_legal_count_norm",),
      ("state_led_legal_ratio",),
      ("state_red_legal_ratio",),
      ("state_switch_legal_ratio",),
      ("state_avoidable_switch",),
  ]
  return np.asarray([
      _float_from_any(proposal, aliases) for aliases in keys
  ], dtype=np.float32)


def _proposal_round_history_features(proposal: dict) -> np.ndarray:
  keys = [
      ("history_available",),
      ("history_prior_events_norm",),
      ("history_prior_play_events_norm",),
      ("history_own_prior_play_events_norm",),
      ("history_other_prior_play_events_norm",),
      ("history_prior_follow_led_rate",),
      ("history_prior_switch_rate",),
      ("history_prior_avoidable_switch_rate",),
      ("history_prior_red_when_switch_rate",),
      ("history_prior_red_when_avoidable_switch_rate",),
      ("history_prior_nonred_when_avoidable_switch_rate",),
      ("history_prior_token_loss_rate",),
      ("history_prior_avoidable_token_loss_rate",),
      ("history_own_prior_token_loss_rate",),
      ("history_other_prior_token_loss_rate",),
      ("history_last_event_is_play",),
      ("history_last_event_follows_led",),
      ("history_last_event_is_red",),
      ("history_last_event_switches_off_led",),
      ("history_last_event_avoidable_switch",),
      ("history_last_event_avoidable_nonred_switch",),
      ("history_last_event_off_led_loses_token",),
      ("history_last_event_led_legal_ratio",),
      ("history_last_event_red_legal_ratio",),
      ("history_last_event_switch_legal_ratio",),
  ]
  return np.asarray([
      _float_from_any(proposal, aliases) for aliases in keys
  ], dtype=np.float32)


def _pair_context_tail(args, feature: np.ndarray) -> np.ndarray:
  obs_mode = str(getattr(args, "observation_feature_mode", "none") or "none")
  policy_mode = str(getattr(args, "policy_context_mode", "none") or "none")
  obs_len = 8 if obs_mode == "summary" else 0
  policy_len = 13 if policy_mode == "summary" else 0
  tail_len = int(obs_len + policy_len)
  if tail_len <= 0:
    return np.zeros(0, dtype=np.float32)
  feature = np.asarray(feature, dtype=np.float32)
  if int(feature.shape[0]) < tail_len:
    return np.zeros(tail_len, dtype=np.float32)
  return feature[-tail_len:].astype(np.float32)


def _action_feature_bool(feature: np.ndarray, idx: int) -> float:
  if int(idx) < 0 or int(idx) >= int(feature.shape[0]):
    return 0.0
  return 1.0 if float(feature[int(idx)]) > 0.5 else 0.0


def _named_action_feature_bool(feature: np.ndarray, name: str) -> float:
  idx = APPENDED_ACTION_FEATURE_INDEX.get(str(name))
  if idx is None:
    return 0.0
  return _action_feature_bool(feature, int(idx))


def _action_loses_led_lane(feature: np.ndarray) -> float:
  return max(
      _action_feature_bool(feature, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX),
      _named_action_feature_bool(feature, "token_loss_newly_loses_led"),
  )


def _led_lane_burn_allowed_by_exception(
    args,
    candidate_feature: np.ndarray,
) -> bool:
  exception = str(getattr(args, "led_lane_red_exception", "urgent_win"))
  if exception == "none":
    return False
  candidate_red = bool(_action_feature_bool(
      candidate_feature,
      ACTION_FEATURE_IS_RED_INDEX,
  ))
  would_win = bool(_action_feature_bool(
      candidate_feature,
      ACTION_FEATURE_WOULD_WIN_INDEX,
  ))
  if exception == "red":
    return candidate_red
  if exception == "win":
    return would_win
  return bool(candidate_red and would_win)


def _forced_led_lane_burn_negative(
    args,
    candidate_feature: np.ndarray,
) -> bool:
  if not bool(getattr(args, "force_led_lane_burn_negative", False)):
    return False
  loses_lane = bool(_action_loses_led_lane(candidate_feature))
  if not loses_lane:
    return False
  follows_led = bool(_action_feature_bool(
      candidate_feature,
      ACTION_FEATURE_FOLLOWS_LED_INDEX,
  ))
  if follows_led:
    return False
  return not _led_lane_burn_allowed_by_exception(args, candidate_feature)


def _bad_red_no_win_negative(
    args,
    candidate_feature: np.ndarray,
    candidate_view: dict,
) -> bool:
  if not bool(getattr(args, "bad_red_no_win_negative", False)):
    return False
  if {
      "candidate_is_red_switch",
      "candidate_lost_led_token",
      "candidate_would_win_now",
  }.issubset(candidate_view):
    return bool(
        candidate_view.get("candidate_is_red_switch", False)
        and candidate_view.get("candidate_lost_led_token", False)
        and not candidate_view.get("candidate_would_win_now", False)
        and not candidate_view.get("candidate_responsible_red_win_switch", False)
    )
  candidate_red = bool(_action_feature_bool(
      candidate_feature,
      ACTION_FEATURE_IS_RED_INDEX,
  ))
  loses_lane = bool(_action_loses_led_lane(candidate_feature))
  would_win = bool(_action_feature_bool(
      candidate_feature,
      ACTION_FEATURE_WOULD_WIN_INDEX,
  ))
  return bool(candidate_red and loses_lane and not would_win)


def _action_flag_features(
    candidate_feature: np.ndarray,
    base_feature: np.ndarray,
) -> np.ndarray:
  candidate_follows = _action_feature_bool(
      candidate_feature,
      ACTION_FEATURE_FOLLOWS_LED_INDEX,
  )
  base_follows = _action_feature_bool(base_feature, ACTION_FEATURE_FOLLOWS_LED_INDEX)
  candidate_red = _action_feature_bool(candidate_feature, ACTION_FEATURE_IS_RED_INDEX)
  base_red = _action_feature_bool(base_feature, ACTION_FEATURE_IS_RED_INDEX)
  candidate_token_loss = _action_feature_bool(
      candidate_feature,
      ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
  )
  base_token_loss = _action_feature_bool(
      base_feature,
      ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
  )
  candidate_token_loss = max(candidate_token_loss, _action_loses_led_lane(
      candidate_feature
  ))
  base_token_loss = max(base_token_loss, _action_loses_led_lane(base_feature))
  candidate_red_loss = candidate_red * candidate_token_loss
  base_red_loss = base_red * base_token_loss
  candidate_nonred_loss = (1.0 - candidate_red) * candidate_token_loss
  base_nonred_loss = (1.0 - base_red) * base_token_loss
  values = [
      candidate_follows,
      base_follows,
      candidate_follows - base_follows,
      candidate_red,
      base_red,
      candidate_red - base_red,
      candidate_token_loss,
      base_token_loss,
      candidate_token_loss - base_token_loss,
      candidate_red_loss,
      base_red_loss,
      candidate_red_loss - base_red_loss,
      candidate_nonred_loss,
      base_nonred_loss,
      candidate_nonred_loss - base_nonred_loss,
  ]
  return np.asarray(values, dtype=np.float32)


def _action_flag_win_features(
    candidate_feature: np.ndarray,
    base_feature: np.ndarray,
) -> np.ndarray:
  base_flags = _action_flag_features(candidate_feature, base_feature)
  candidate_red = _action_feature_bool(candidate_feature, ACTION_FEATURE_IS_RED_INDEX)
  base_red = _action_feature_bool(base_feature, ACTION_FEATURE_IS_RED_INDEX)
  candidate_token_loss = max(
      _action_feature_bool(candidate_feature, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX),
      _action_loses_led_lane(candidate_feature),
  )
  base_token_loss = max(
      _action_feature_bool(base_feature, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX),
      _action_loses_led_lane(base_feature),
  )
  candidate_would_win = _action_feature_bool(
      candidate_feature,
      ACTION_FEATURE_WOULD_WIN_INDEX,
  )
  base_would_win = _action_feature_bool(
      base_feature,
      ACTION_FEATURE_WOULD_WIN_INDEX,
  )
  candidate_lane_loss_no_win = candidate_token_loss * (1.0 - candidate_would_win)
  base_lane_loss_no_win = base_token_loss * (1.0 - base_would_win)
  candidate_red_loss_win = candidate_red * candidate_token_loss * candidate_would_win
  base_red_loss_win = base_red * base_token_loss * base_would_win
  candidate_red_loss_no_win = (
      candidate_red * candidate_token_loss * (1.0 - candidate_would_win)
  )
  base_red_loss_no_win = base_red * base_token_loss * (1.0 - base_would_win)
  win_values = [
      candidate_would_win,
      base_would_win,
      candidate_would_win - base_would_win,
      candidate_lane_loss_no_win,
      base_lane_loss_no_win,
      candidate_lane_loss_no_win - base_lane_loss_no_win,
      candidate_red_loss_win,
      base_red_loss_win,
      candidate_red_loss_win - base_red_loss_win,
      candidate_red_loss_no_win,
      base_red_loss_no_win,
      candidate_red_loss_no_win - base_red_loss_no_win,
  ]
  return np.concatenate(
      [base_flags, np.asarray(win_values, dtype=np.float32)]
  ).astype(np.float32)


def _proposal_feature(
    args,
    candidate_feature: np.ndarray,
    base_feature: np.ndarray,
    proposal: dict,
    candidate_action_feature: np.ndarray | None = None,
    base_action_feature: np.ndarray | None = None,
) -> np.ndarray:
  scalars = _proposal_scalar_features(proposal)
  mode = str(getattr(args, "proposal_feature_mode", "full"))
  candidate_action_feature = (
      candidate_feature if candidate_action_feature is None else candidate_action_feature
  )
  base_action_feature = base_feature if base_action_feature is None else base_action_feature
  if mode == "scalar_only":
    return scalars.astype(np.float32)
  if mode == "full_no_source":
    return np.concatenate(
        [
            candidate_feature,
            base_feature,
            candidate_feature - base_feature,
        ]
    ).astype(np.float32)
  if mode == "full_no_source_public_slate":
    return np.concatenate(
        [
            candidate_feature,
            base_feature,
            candidate_feature - base_feature,
            _proposal_public_slate_features(proposal),
        ]
    ).astype(np.float32)
  if mode in (
      "scalar_action_flags",
      "scalar_action_flags_slate",
      "scalar_action_flags_slate_state_context",
      "scalar_action_flags_slate_state_round_history_context",
      "scalar_action_flags_win_slate_state_context",
      "scalar_action_flags_win_slate_state_round_history_context",
      "scalar_action_flags_win_aux_slate_state_context",
      "scalar_action_flags_win_aux_slate_state_round_history_context",
  ):
    action_flags = (
        _action_flag_win_features(candidate_action_feature, base_action_feature)
        if mode.startswith("scalar_action_flags_win_") else
        _action_flag_features(candidate_action_feature, base_action_feature)
    )
    pieces = [
        scalars,
        action_flags,
    ]
    if mode in (
        "scalar_action_flags_slate",
        "scalar_action_flags_slate_state_context",
        "scalar_action_flags_slate_state_round_history_context",
        "scalar_action_flags_win_slate_state_context",
        "scalar_action_flags_win_slate_state_round_history_context",
        "scalar_action_flags_win_aux_slate_state_context",
        "scalar_action_flags_win_aux_slate_state_round_history_context",
    ):
      pieces.append(_proposal_slate_features(proposal))
    if "_aux_" in mode:
      pieces.append(_proposal_aux_features(proposal))
    if mode in (
        "scalar_action_flags_slate_state_context",
        "scalar_action_flags_slate_state_round_history_context",
        "scalar_action_flags_win_slate_state_context",
        "scalar_action_flags_win_slate_state_round_history_context",
        "scalar_action_flags_win_aux_slate_state_context",
        "scalar_action_flags_win_aux_slate_state_round_history_context",
    ):
      candidate_tail = _pair_context_tail(args, candidate_feature)
      base_tail = _pair_context_tail(args, base_feature)
      pieces.extend([
          _proposal_state_context_features(proposal),
          candidate_tail,
          base_tail,
          candidate_tail - base_tail,
      ])
    if mode in (
        "scalar_action_flags_slate_state_round_history_context",
        "scalar_action_flags_win_slate_state_round_history_context",
        "scalar_action_flags_win_aux_slate_state_round_history_context",
    ):
      pieces.append(_proposal_round_history_features(proposal))
    return np.concatenate([
        piece.astype(np.float32) for piece in pieces
    ]).astype(np.float32)
  return np.concatenate(
      [
          candidate_feature,
          base_feature,
          candidate_feature - base_feature,
          scalars,
      ]
  ).astype(np.float32)


def _candidate_view_from_top_proposal(proposal: dict) -> dict:
  has_repeat = "repeat_validation_mean_value_delta" in proposal
  if has_repeat:
    dense_delta = float(proposal.get("repeat_validation_mean_value_delta", 0.0))
  else:
    dense_delta = float(proposal.get("dense_action_value_delta_vs_base", 0.0))
  if "repeat_validation_candidate_any_paradox_rate" in proposal:
    candidate_any_rate = float(
        proposal.get("repeat_validation_candidate_any_paradox_rate", 1.0)
    )
    base_any_rate = float(
        proposal.get("repeat_validation_base_any_paradox_rate", 1.0)
    )
    paradox_delta = float(candidate_any_rate - base_any_rate)
    candidate_any = bool(candidate_any_rate > 0.0)
  else:
    paradox_delta = float(proposal.get("paradox_delta_vs_base", 0.0))
    candidate_any = bool(proposal.get("candidate_any_paradox", False))
  positive = bool(dense_delta >= 0.01 and paradox_delta <= 0.0)
  safe = bool(paradox_delta <= 0.0)
  return {
      "candidate_action": int(proposal.get("candidate_action", -1)),
      "candidate_action_string": proposal.get("candidate_action_string"),
      "score_margin": float(proposal.get("score_margin", 0.0)),
      "candidate_score": float(proposal.get("candidate_score", 0.0)),
      "base_score": float(proposal.get("base_score", 0.0)),
      "candidate_pred_value_delta": float(
          proposal.get("candidate_pred_value_delta", 0.0)
      ),
      "candidate_pred_positive_prob": float(
          proposal.get("candidate_pred_positive_prob", 0.0)
      ),
      "candidate_policy_prob": float(proposal.get("candidate_policy_prob", 0.0)),
      "base_policy_prob": float(proposal.get("base_policy_prob", 0.0)),
      "candidate_count": int(proposal.get("candidate_count", 0)),
      "scored_candidate_count": int(proposal.get("scored_candidate_count", 0)),
      "candidate_is_red_switch": bool(proposal.get("is_red_switch", False)),
      "candidate_switches_off_led": bool(proposal.get("switches_off_led", False)),
      "candidate_lost_led_token": bool(proposal.get("lost_led_token", False)),
      "candidate_would_win_now": bool(proposal.get("would_win_now", False)),
      "candidate_responsible_red_win_switch": bool(
          proposal.get("responsible_red_win_switch", False)
      ),
      "positive_by_dense_delta": bool(
          positive if has_repeat else proposal.get("positive_by_dense_delta", False)
      ),
      "safe_by_paradox_delta": bool(
          safe if has_repeat else proposal.get("safe_by_paradox_delta", False)
      ),
      "candidate_any_paradox": bool(candidate_any),
      "candidate_actor_paradox": bool(
          proposal.get("candidate_actor_paradox", False)
      ),
      "candidate_forced_paradox": bool(
          proposal.get("candidate_forced_paradox", False)
      ),
      "dense_action_value_delta_vs_base": float(dense_delta),
      "paradox_delta_vs_base": float(paradox_delta),
      "source_candidate_rank": 0,
      "source_candidate_count": int(proposal.get("scored_candidate_count", 0)),
  }


def _candidate_view_from_scored_item(
    proposal: dict,
    item: dict,
    scored: list,
    rank: int,
    total: int,
) -> dict:
  has_repeat = "repeat_validation_mean_value_delta" in item
  if has_repeat:
    dense_delta = float(item.get("repeat_validation_mean_value_delta", 0.0))
  else:
    dense_delta = float(item.get("dense_action_value_delta_vs_base", 0.0))
  if "repeat_validation_candidate_any_paradox_rate" in item:
    candidate_any_rate = float(
        item.get("repeat_validation_candidate_any_paradox_rate", 1.0)
    )
    base_any_rate = float(
        item.get(
            "repeat_validation_base_any_paradox_rate",
            proposal.get("repeat_validation_base_any_paradox_rate", 1.0),
        )
    )
    paradox_delta = float(candidate_any_rate - base_any_rate)
    candidate_any = bool(candidate_any_rate > 0.0)
  else:
    paradox_delta = float(item.get("paradox_delta_vs_base", 0.0))
    candidate_any = bool(item.get("any_paradox", False))
  positive = bool(dense_delta >= 0.01 and paradox_delta <= 0.0)
  safe = bool(paradox_delta <= 0.0)
  view = {
      "candidate_action": int(item.get("action", -1)),
      "candidate_action_string": item.get("action_string"),
      "score_margin": float(item.get("margin", 0.0)),
      "candidate_score": float(item.get("score", 0.0)),
      "base_score": float(item.get("base_score", proposal.get("base_score", 0.0))),
      "candidate_pred_value_delta": float(item.get("pred_value_delta", 0.0)),
      "candidate_pred_positive_prob": float(item.get("pred_positive_prob", 0.0)),
      "candidate_policy_prob": float(item.get("policy_prob", 0.0)),
      "base_policy_prob": float(proposal.get("base_policy_prob", 0.0)),
      "candidate_count": int(proposal.get("candidate_count", 0)),
      "scored_candidate_count": int(proposal.get("scored_candidate_count", total)),
      "candidate_is_red_switch": bool(item.get("is_red_switch", False)),
      "candidate_switches_off_led": bool(item.get("switches_off_led", False)),
      "candidate_lost_led_token": bool(item.get("lost_led_token", False)),
      "candidate_would_win_now": bool(item.get("would_win_now", False)),
      "candidate_responsible_red_win_switch": bool(
          item.get("responsible_red_win_switch", False)
      ),
      "positive_by_dense_delta": bool(
          positive if has_repeat else item.get("positive_by_dense_delta", False)
      ),
      "safe_by_paradox_delta": bool(
          safe if has_repeat else item.get("safe_by_paradox_delta", False)
      ),
      "candidate_any_paradox": bool(candidate_any),
      "candidate_actor_paradox": bool(item.get("actor_paradox", False)),
      "candidate_forced_paradox": bool(item.get("forced_paradox", False)),
      "dense_action_value_delta_vs_base": float(dense_delta),
      "paradox_delta_vs_base": float(paradox_delta),
      "source_candidate_rank": int(rank),
      "source_candidate_count": int(total),
  }
  view.update(_slate_relative_fields(item, scored, int(rank), int(total)))
  return view


def _attach_action_aux_fields(view: dict, candidate: dict, base: dict) -> None:
  candidate_value = float(candidate.get("aux_action_value", 0.0))
  base_value = float(
      base.get("aux_action_value", base.get("base_aux_action_value", 0.0))
  )
  candidate_risk = float(candidate.get("aux_action_paradox_risk", 0.0))
  base_risk = float(
      base.get(
          "aux_action_paradox_risk",
          base.get("base_aux_action_paradox_risk", 0.0),
      )
  )
  view.update({
      "candidate_aux_action_value": candidate_value,
      "base_aux_action_value": base_value,
      "aux_action_value_delta": candidate_value - base_value,
      "candidate_aux_action_paradox_risk": candidate_risk,
      "base_aux_action_paradox_risk": base_risk,
      "aux_action_paradox_risk_delta": candidate_risk - base_risk,
  })


def _build_proposal_examples(args, states: list[dict], metadata_rows: list[dict]):
  examples = []
  skipped = {
      "missing_metadata": 0,
      "missing_proposal": 0,
      "below_threshold": 0,
      "missing_candidate": 0,
      "missing_base": 0,
      "missing_scored_candidates": 0,
      "candidate_outcome_unavailable": 0,
  }
  for state in states:
    state_idx = int(state["state_index"])
    if state_idx < 0 or state_idx >= len(metadata_rows):
      skipped["missing_metadata"] += 1
      continue
    proposal = metadata_rows[state_idx].get("proposal")
    if not isinstance(proposal, dict):
      skipped["missing_proposal"] += 1
      continue
    if not bool(proposal.get("above_threshold", False)):
      skipped["below_threshold"] += 1
      continue
    base_id = int(proposal.get("base_action", state.get("base_action", -1)))
    base = _find_action(state, base_id)
    if base is None:
      skipped["missing_base"] += 1
      continue
    base_feature = np.asarray(base["feature"], dtype=np.float32)
    candidate_views = []
    if str(args.proposal_example_source) == "scored_candidates":
      scored = proposal.get("scored_candidates")
      if not isinstance(scored, list) or not scored:
        skipped["missing_scored_candidates"] += 1
        continue
      for rank, item in enumerate(scored):
        if not isinstance(item, dict):
          continue
        if not bool(item.get("outcome_available", False)):
          skipped["candidate_outcome_unavailable"] += 1
          continue
        candidate_views.append(
            _candidate_view_from_scored_item(
                proposal,
                item,
                scored,
                rank,
                len(scored),
            )
        )
    else:
      candidate_views.append(_candidate_view_from_top_proposal(proposal))
    for candidate_view in candidate_views:
      candidate_view.update(_state_context_fields(metadata_rows[state_idx]))
      candidate_view.update(_round_history_context_fields(metadata_rows[state_idx]))
      candidate_id = int(candidate_view.get("candidate_action", -1))
      candidate = _find_action(state, candidate_id)
      if candidate is None:
        skipped["missing_candidate"] += 1
        continue
      _attach_action_aux_fields(candidate_view, candidate, base)
      candidate_feature = np.asarray(candidate["feature"], dtype=np.float32)
      candidate_action_feature = np.asarray(
          candidate.get("raw_action_feature", candidate_feature),
          dtype=np.float32,
      )
      base_action_feature = np.asarray(
          base.get("raw_action_feature", base_feature),
          dtype=np.float32,
      )
      candidate_view.update(_public_action_slate_fields(
          candidate_view,
          candidate_action_feature,
          base_action_feature,
          state,
          candidate_views,
      ))
      feature = _proposal_feature(
          args,
          candidate_feature,
          base_feature,
          candidate_view,
          candidate_action_feature,
          base_action_feature,
      )
      dense_delta = float(candidate_view.get(
          "dense_action_value_delta_vs_base",
          candidate.get("value_delta", 0.0),
      ))
      paradox_delta = float(candidate_view.get(
          "paradox_delta_vs_base",
          candidate.get("risk_delta", 0.0),
      ))
      positive = bool(candidate_view.get("positive_by_dense_delta", False))
      safe = bool(candidate_view.get("safe_by_paradox_delta", False))
      forced_lane_burn_negative = _forced_led_lane_burn_negative(
          args,
          candidate_action_feature,
      )
      bad_red_no_win_negative = _bad_red_no_win_negative(
          args,
          candidate_action_feature,
          candidate_view,
      )
      candidate_view["forced_led_lane_burn_negative"] = bool(
          forced_lane_burn_negative
      )
      candidate_view["bad_red_no_win_negative"] = bool(bad_red_no_win_negative)
      approved = bool(
          _approval_label(candidate_view, str(args.approve_label))
          and not forced_lane_burn_negative
          and not bad_red_no_win_negative
      )
      group_relative_reward = _group_relative_reward(
          args,
          candidate_view,
          bool(approved),
      )
      examples.append({
          "state_index": int(state_idx),
          "group_key": state.get("group_key"),
          "source_replay_index": int(state.get("source_replay_index", -1)),
          "base_action": int(base_id),
          "candidate_action": int(candidate_id),
          "base_action_string": proposal.get("base_action_string"),
          "candidate_action_string": candidate_view.get("candidate_action_string"),
          "approved": bool(approved),
          "positive_by_dense_delta": bool(positive),
          "safe_by_paradox_delta": bool(safe),
          "candidate_any_paradox": bool(
              candidate_view.get("candidate_any_paradox", False)
          ),
          "base_any_paradox": bool(proposal.get("base_any_paradox", False)),
          "candidate_actor_paradox": bool(
              candidate_view.get("candidate_actor_paradox", False)
          ),
          "candidate_forced_paradox": bool(
              candidate_view.get("candidate_forced_paradox", False)
          ),
          "forced_led_lane_burn_negative": bool(forced_lane_burn_negative),
          "bad_red_no_win_negative": bool(bad_red_no_win_negative),
          "dense_value_delta": float(dense_delta),
          "paradox_delta": float(paradox_delta),
          "group_relative_reward": float(group_relative_reward),
          "feature": feature,
          "proposal": {
              "score_margin": float(candidate_view.get("score_margin", 0.0)),
              "candidate_pred_value_delta": float(
                  candidate_view.get("candidate_pred_value_delta", 0.0)
              ),
              "candidate_pred_positive_prob": float(
                  candidate_view.get("candidate_pred_positive_prob", 0.0)
              ),
              "candidate_policy_prob": float(
                  candidate_view.get("candidate_policy_prob", 0.0)
              ),
              "base_policy_prob": float(candidate_view.get("base_policy_prob", 0.0)),
              "source_candidate_rank": int(
                  candidate_view.get("source_candidate_rank", 0)
              ),
              "source_candidate_count": int(
                  candidate_view.get("source_candidate_count", 0)
              ),
          },
      })
  if not examples:
    raise ValueError("no proposal examples built")
  return examples, skipped


def _summary(examples: list[dict]) -> dict:
  if not examples:
    return {
        "examples": 0,
        "approved": 0,
        "positive_by_dense_delta": 0,
        "safe_by_paradox_delta": 0,
        "candidate_any_paradox": 0,
        "base_any_paradox": 0,
        "forced_led_lane_burn_negative": 0,
        "bad_red_no_win_negative": 0,
        "mean_dense_value_delta": None,
        "mean_paradox_delta": None,
        "mean_group_relative_reward": None,
    }
  return {
      "examples": int(len(examples)),
      "approved": int(sum(bool(ex["approved"]) for ex in examples)),
      "positive_by_dense_delta": int(
          sum(bool(ex["positive_by_dense_delta"]) for ex in examples)
      ),
      "safe_by_paradox_delta": int(
          sum(bool(ex["safe_by_paradox_delta"]) for ex in examples)
      ),
      "candidate_any_paradox": int(
          sum(bool(ex["candidate_any_paradox"]) for ex in examples)
      ),
      "base_any_paradox": int(sum(bool(ex["base_any_paradox"]) for ex in examples)),
      "forced_led_lane_burn_negative": int(
          sum(bool(ex.get("forced_led_lane_burn_negative", False)) for ex in examples)
      ),
      "bad_red_no_win_negative": int(
          sum(bool(ex.get("bad_red_no_win_negative", False)) for ex in examples)
      ),
      "mean_dense_value_delta": round(
          float(np.mean([ex["dense_value_delta"] for ex in examples])), 6
      ),
      "mean_paradox_delta": round(
          float(np.mean([ex["paradox_delta"] for ex in examples])), 6
      ),
      "mean_group_relative_reward": round(
          float(np.mean([ex["group_relative_reward"] for ex in examples])),
          6,
      ),
  }


def _summary_by_source(examples: list[dict]) -> dict:
  sources = sorted({int(ex.get("source_replay_index", -1)) for ex in examples})
  return {
      str(source): _summary([
          ex for ex in examples
          if int(ex.get("source_replay_index", -1)) == int(source)
      ])
      for source in sources
  }


def _state_groups(examples: list[dict], min_size: int = 1) -> list[list[int]]:
  by_state: dict[tuple[int, int], list[int]] = {}
  for idx, example in enumerate(examples):
    key = (
        int(example.get("source_replay_index", -1)),
        int(example.get("state_index", -1)),
    )
    by_state.setdefault(key, []).append(int(idx))
  return [
      indices for indices in by_state.values()
      if len(indices) >= int(min_size)
  ]


def _pairwise_groups(examples: list[dict]) -> list[list[int]]:
  by_state = _state_groups(examples)
  groups = []
  for indices in by_state:
    positives = [idx for idx in indices if bool(examples[idx].get("approved", False))]
    negatives = [
        idx for idx in indices
        if not bool(examples[idx].get("approved", False))
    ]
    if positives and negatives:
      groups.append(indices)
  return groups


def _unsafe_negative(example: dict) -> bool:
  return bool(
      example.get("candidate_any_paradox", False)
      or example.get("candidate_actor_paradox", False)
      or example.get("candidate_forced_paradox", False)
      or example.get("forced_led_lane_burn_negative", False)
      or example.get("bad_red_no_win_negative", False)
  )


def _unsafe_positive(example: dict, args) -> bool:
  mode = str(getattr(args, "unsafe_positive_mode", "clean_safe") or "clean_safe")
  if mode == "approved":
    return bool(example.get("approved", False))
  if _unsafe_negative(example):
    return False
  if mode == "clean_any":
    return True
  return bool(example.get("safe_by_paradox_delta", False))


def _unsafe_margin_groups(examples: list[dict], args) -> list[list[int]]:
  groups = []
  for indices in _state_groups(examples):
    positives = [idx for idx in indices if _unsafe_positive(examples[idx], args)]
    negatives = [idx for idx in indices if _unsafe_negative(examples[idx])]
    if positives and negatives:
      groups.append(indices)
  return groups


def _density_contrast_groups(examples: list[dict], args) -> list[list[int]]:
  groups = []
  source_filter = _parse_int_set(
      str(getattr(args, "density_contrast_source_indices", "") or "")
  )
  min_size = int(getattr(args, "density_contrast_min_group_size", 6))
  min_any_rate = float(
      getattr(args, "density_contrast_min_candidate_any_rate", 0.65)
  )
  max_approved_rate = float(
      getattr(args, "density_contrast_max_approved_rate", 0.30)
  )
  for indices in _state_groups(examples):
    if len(indices) < min_size:
      continue
    source_index = int(examples[indices[0]].get("source_replay_index", -1))
    if source_filter and source_index not in source_filter:
      continue
    approved = [
        idx for idx in indices
        if bool(examples[idx].get("approved", False))
    ]
    candidate_any = [
        idx for idx in indices
        if bool(examples[idx].get("candidate_any_paradox", False))
    ]
    if not approved or not candidate_any:
      continue
    any_rate = float(len(candidate_any)) / float(max(1, len(indices)))
    approved_rate = float(len(approved)) / float(max(1, len(indices)))
    if any_rate < min_any_rate or approved_rate > max_approved_rate:
      continue
    groups.append(indices)
  return groups


def _listwise_groups(examples: list[dict], args=None) -> list[list[int]]:
  groups = []
  for indices in _state_groups(examples):
    if (
        args is not None
        and str(getattr(args, "target_mode", "")) == "group_relative_soft"
    ):
      if len(indices) >= 2:
        groups.append(indices)
      continue
    positives = [idx for idx in indices if bool(examples[idx].get("approved", False))]
    if positives and len(indices) >= 2:
      groups.append(indices)
  return groups


def _pairwise_group_summary(examples: list[dict]) -> dict:
  state_groups = _state_groups(examples)
  group_sizes = [len(indices) for indices in state_groups]
  pairwise_groups = _pairwise_groups(examples)
  listwise_groups = _listwise_groups(examples)
  pairs = 0
  for indices in pairwise_groups:
    positives = sum(bool(examples[idx].get("approved", False)) for idx in indices)
    negatives = len(indices) - positives
    pairs += int(positives * negatives)
  return {
      "states": int(len(state_groups)),
      "examples": int(len(examples)),
      "mean_group_size": (
          None if not group_sizes else round(float(np.mean(group_sizes)), 6)
      ),
      "max_group_size": 0 if not group_sizes else int(max(group_sizes)),
      "pairwise_groups": int(len(pairwise_groups)),
      "pairwise_pairs": int(pairs),
      "listwise_groups": int(len(listwise_groups)),
  }


def _unsafe_margin_group_summary(examples: list[dict], args) -> dict:
  groups = _unsafe_margin_groups(examples, args)
  positive_count = 0
  unsafe_count = 0
  pairs = 0
  for indices in groups:
    positives = sum(_unsafe_positive(examples[idx], args) for idx in indices)
    negatives = sum(_unsafe_negative(examples[idx]) for idx in indices)
    positive_count += int(positives)
    unsafe_count += int(negatives)
    pairs += int(positives * negatives)
  return {
      "unsafe_margin_groups": int(len(groups)),
      "unsafe_margin_positive_items": int(positive_count),
      "unsafe_margin_negative_items": int(unsafe_count),
      "unsafe_margin_pairs": int(pairs),
      "unsafe_positive_mode": str(
          getattr(args, "unsafe_positive_mode", "clean_safe")
      ),
  }


def _density_contrast_group_summary(examples: list[dict], args) -> dict:
  groups = _density_contrast_groups(examples, args)
  approved_count = 0
  candidate_any_count = 0
  pairs = 0
  any_rates = []
  approved_rates = []
  for indices in groups:
    approved = sum(bool(examples[idx].get("approved", False)) for idx in indices)
    candidate_any = sum(
        bool(examples[idx].get("candidate_any_paradox", False))
        for idx in indices
    )
    approved_count += int(approved)
    candidate_any_count += int(candidate_any)
    pairs += int(approved * candidate_any)
    any_rates.append(float(candidate_any) / float(max(1, len(indices))))
    approved_rates.append(float(approved) / float(max(1, len(indices))))
  return {
      "density_contrast_groups": int(len(groups)),
      "density_contrast_approved_items": int(approved_count),
      "density_contrast_candidate_any_items": int(candidate_any_count),
      "density_contrast_pairs": int(pairs),
      "density_contrast_min_candidate_any_rate": float(
          getattr(args, "density_contrast_min_candidate_any_rate", 0.65)
      ),
      "density_contrast_max_approved_rate": float(
          getattr(args, "density_contrast_max_approved_rate", 0.30)
      ),
      "density_contrast_min_group_size": int(
          getattr(args, "density_contrast_min_group_size", 6)
      ),
      "density_contrast_source_indices": sorted(
          _parse_int_set(
              str(getattr(args, "density_contrast_source_indices", "") or "")
          )
      ),
      "mean_density_candidate_any_rate": (
          None if not any_rates else round(float(np.mean(any_rates)), 6)
      ),
      "mean_density_approved_rate": (
          None if not approved_rates else round(float(np.mean(approved_rates)), 6)
      ),
  }


def _pairwise_loss(
    logits: torch.Tensor,
    y_t: torch.Tensor,
    groups: list[list[int]],
    margin: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
  losses = []
  loss_weights = []
  for indices in groups:
    idx_t = torch.tensor(indices, dtype=torch.long, device=logits.device)
    group_logits = logits.index_select(0, idx_t)
    group_y = y_t.index_select(0, idx_t)
    positives = group_logits[group_y > 0.5]
    negatives = group_logits[group_y <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
      continue
    deltas = positives.unsqueeze(1) - negatives.unsqueeze(0)
    losses.append(F.softplus(float(margin) - deltas).mean())
    if weights is not None:
      loss_weights.append(torch.mean(weights.index_select(0, idx_t)))
  if not losses:
    return logits.sum() * 0.0
  stacked = torch.stack(losses)
  if weights is None:
    return stacked.mean()
  weight_t = torch.stack(loss_weights)
  return torch.sum(stacked * weight_t) / torch.clamp(torch.sum(weight_t), min=1e-6)


def _unsafe_margin_loss(
    logits: torch.Tensor,
    groups: list[list[int]],
    examples: list[dict],
    args,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
  losses = []
  loss_weights = []
  margin = float(getattr(args, "unsafe_margin", 1.0))
  for indices in groups:
    pos_idx = [idx for idx in indices if _unsafe_positive(examples[idx], args)]
    neg_idx = [idx for idx in indices if _unsafe_negative(examples[idx])]
    if not pos_idx or not neg_idx:
      continue
    pos_t = torch.tensor(pos_idx, dtype=torch.long, device=logits.device)
    neg_t = torch.tensor(neg_idx, dtype=torch.long, device=logits.device)
    positives = logits.index_select(0, pos_t)
    negatives = logits.index_select(0, neg_t)
    deltas = positives.unsqueeze(1) - negatives.unsqueeze(0)
    losses.append(F.softplus(margin - deltas).mean())
    if weights is not None:
      idx_t = torch.tensor(indices, dtype=torch.long, device=logits.device)
      loss_weights.append(torch.mean(weights.index_select(0, idx_t)))
  if not losses:
    return logits.sum() * 0.0
  stacked = torch.stack(losses)
  if weights is None:
    return stacked.mean()
  weight_t = torch.stack(loss_weights)
  return torch.sum(stacked * weight_t) / torch.clamp(torch.sum(weight_t), min=1e-6)


def _density_contrast_loss(
    logits: torch.Tensor,
    groups: list[list[int]],
    examples: list[dict],
    args,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
  losses = []
  loss_weights = []
  margin = float(getattr(args, "density_contrast_margin", 2.0))
  for indices in groups:
    pos_idx = [
        idx for idx in indices
        if bool(examples[idx].get("approved", False))
    ]
    neg_idx = [
        idx for idx in indices
        if bool(examples[idx].get("candidate_any_paradox", False))
    ]
    if not pos_idx or not neg_idx:
      continue
    pos_t = torch.tensor(pos_idx, dtype=torch.long, device=logits.device)
    neg_t = torch.tensor(neg_idx, dtype=torch.long, device=logits.device)
    positives = logits.index_select(0, pos_t)
    negatives = logits.index_select(0, neg_t)
    deltas = positives.unsqueeze(1) - negatives.unsqueeze(0)
    losses.append(F.softplus(margin - deltas).mean())
    if weights is not None:
      idx_t = torch.tensor(indices, dtype=torch.long, device=logits.device)
      loss_weights.append(torch.mean(weights.index_select(0, idx_t)))
  if not losses:
    return logits.sum() * 0.0
  stacked = torch.stack(losses)
  if weights is None:
    return stacked.mean()
  weight_t = torch.stack(loss_weights)
  return torch.sum(stacked * weight_t) / torch.clamp(torch.sum(weight_t), min=1e-6)


def _listwise_target_index(indices: list[int], examples: list[dict]) -> int | None:
  approved = [idx for idx in indices if bool(examples[idx].get("approved", False))]
  if not approved:
    return None

  def key(idx: int):
    example = examples[idx]
    return (
        float(example.get("dense_value_delta", 0.0)),
        -float(example.get("paradox_delta", 0.0)),
        float(example.get("safe_by_paradox_delta", False)),
    )

  best_global_idx = max(approved, key=key)
  return indices.index(best_global_idx)


def _listwise_soft_target(
    indices: list[int],
    examples: list[dict],
    args,
    device,
) -> torch.Tensor:
  rewards = torch.tensor(
      [
          float(examples[idx].get("group_relative_reward", 0.0))
          for idx in indices
      ],
      dtype=torch.float32,
      device=device,
  )
  temperature = float(getattr(args, "soft_target_temperature", 0.35))
  if temperature <= 0.0:
    target = torch.zeros_like(rewards)
    target[int(torch.argmax(rewards).item())] = 1.0
    return target
  shifted = rewards - torch.max(rewards)
  return torch.softmax(shifted / max(1e-6, temperature), dim=0)


def _listwise_loss(
    args,
    logits: torch.Tensor,
    groups: list[list[int]],
    examples: list[dict],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
  losses = []
  loss_weights = []
  for indices in groups:
    idx_t = torch.tensor(indices, dtype=torch.long, device=logits.device)
    group_logits = logits.index_select(0, idx_t)
    if str(getattr(args, "target_mode", "")) == "group_relative_soft":
      target_probs = _listwise_soft_target(
          indices,
          examples,
          args,
          logits.device,
      )
      losses.append(-(target_probs * F.log_softmax(group_logits, dim=0)).sum())
    else:
      target = _listwise_target_index(indices, examples)
      if target is None:
        continue
      target_t = torch.tensor([int(target)], dtype=torch.long, device=logits.device)
      losses.append(F.cross_entropy(group_logits.unsqueeze(0), target_t))
    if weights is not None:
      loss_weights.append(torch.mean(weights.index_select(0, idx_t)))
  if not losses:
    return logits.sum() * 0.0
  stacked = torch.stack(losses)
  if weights is None:
    return stacked.mean()
  weight_t = torch.stack(loss_weights)
  return torch.sum(stacked * weight_t) / torch.clamp(torch.sum(weight_t), min=1e-6)


def _source_training_tensors(
    examples: list[dict],
    balance: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
  sources = [int(ex.get("source_replay_index", -1)) for ex in examples]
  unique_sources = sorted(set(sources))
  source_to_label = {source: idx for idx, source in enumerate(unique_sources)}
  counts = {source: sources.count(source) for source in unique_sources}
  mean_count = float(np.mean(list(counts.values()))) if counts else 1.0
  labels = [source_to_label[source] for source in sources]
  weights = []
  for source in sources:
    inv = mean_count / max(1.0, float(counts.get(source, 1)))
    weights.append((1.0 - float(balance)) + float(balance) * inv)
  report = {
      "source_count": int(len(unique_sources)),
      "source_to_label": {
          str(source): int(label)
          for source, label in sorted(source_to_label.items())
      },
      "source_counts": {
          str(source): int(count)
          for source, count in sorted(counts.items())
      },
  }
  return (
      torch.tensor(labels, dtype=torch.long),
      torch.tensor(weights, dtype=torch.float32),
      report,
  )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-6)


def _train(
    args,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_examples: list[dict],
):
  torch.manual_seed(int(args.seed))
  model_kind = str(getattr(args, "proposal_model_kind", "mlp"))
  model_cls = ProposalVetoDeepSet if model_kind == "deepset" else ProposalVetoMLP
  model = model_cls(int(x_train.shape[1]), max(1, int(args.hidden)), float(args.dropout))
  source_labels, source_weights, source_report = _source_training_tensors(
      train_examples,
      float(getattr(args, "source_balance_weight", 1.0)),
  )
  main_source_weight = float(getattr(args, "main_source_balance_weight", 0.0))
  _, main_loss_weights, main_source_report = _source_training_tensors(
      train_examples,
      main_source_weight,
  )
  source_adversary = None
  if (
      float(getattr(args, "source_adversary_weight", 0.0)) > 0.0
      and int(source_report["source_count"]) > 1
  ):
    with torch.no_grad():
      feature_probe = model.source_features(
          torch.zeros((1, int(x_train.shape[1])), dtype=torch.float32),
          [[0]],
      )
    source_adversary = SourceAdversary(
        int(feature_probe.shape[1]),
        max(1, int(args.hidden)),
        int(source_report["source_count"]),
    )
  params = list(model.parameters())
  if source_adversary is not None:
    params.extend(source_adversary.parameters())
  optimizer = torch.optim.AdamW(
      params,
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  x_t = torch.tensor(x_train, dtype=torch.float32)
  y_t = torch.tensor(y_train, dtype=torch.float32)
  positives = float(torch.sum(y_t).item())
  negatives = float(len(y_train) - positives)
  if (
      str(args.pos_weight_mode) == "balanced"
      and positives > 0.0
      and negatives > 0.0
  ):
    pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32)
  else:
      pos_weight = torch.tensor([float(args.positive_weight)], dtype=torch.float32)
  losses = []
  bce_losses = []
  pairwise_losses = []
  listwise_losses = []
  unsafe_margin_losses = []
  density_contrast_losses = []
  source_losses = []
  source_accs = []
  pair_groups = _pairwise_groups(train_examples)
  list_groups = _listwise_groups(train_examples, args)
  unsafe_groups = _unsafe_margin_groups(train_examples, args)
  density_groups = _density_contrast_groups(train_examples, args)
  state_groups = _state_groups(train_examples)
  loss_mode = str(getattr(args, "loss_mode", "bce"))
  for _ in range(max(1, int(args.epochs))):
    logits = (
        model(x_t, state_groups)
        if model_kind == "deepset" else
        model(x_t)
    )
    loss_items = F.binary_cross_entropy_with_logits(
        logits, y_t, pos_weight=pos_weight, reduction="none"
    )
    gamma = max(0.0, float(args.focal_gamma))
    if gamma > 0.0:
      probs = torch.sigmoid(logits)
      p_t = torch.where(y_t > 0.5, probs, 1.0 - probs)
      loss_items = loss_items * torch.pow(torch.clamp(1.0 - p_t, min=0.0), gamma)
    bce_loss = torch.mean(loss_items)
    if main_source_weight > 0.0:
      bce_loss = _weighted_mean(loss_items, main_loss_weights)
    pair_loss = _pairwise_loss(
        logits,
        y_t,
        pair_groups,
        float(getattr(args, "pairwise_margin", 1.0)),
        main_loss_weights if main_source_weight > 0.0 else None,
    )
    list_loss = _listwise_loss(
        args,
        logits,
        list_groups,
        train_examples,
        main_loss_weights if main_source_weight > 0.0 else None,
    )
    unsafe_loss = _unsafe_margin_loss(
        logits,
        unsafe_groups,
        train_examples,
        args,
        main_loss_weights if main_source_weight > 0.0 else None,
    )
    density_loss = _density_contrast_loss(
        logits,
        density_groups,
        train_examples,
        args,
        main_loss_weights if main_source_weight > 0.0 else None,
    )
    if loss_mode == "pairwise":
      loss = pair_loss
    elif loss_mode == "bce_pairwise":
      loss = bce_loss + float(args.pairwise_loss_weight) * pair_loss
    elif loss_mode == "listwise":
      loss = list_loss
    elif loss_mode == "bce_listwise":
      loss = bce_loss + float(args.listwise_loss_weight) * list_loss
    else:
      loss = bce_loss
    unsafe_weight = float(getattr(args, "unsafe_margin_loss_weight", 0.0))
    if unsafe_weight > 0.0:
      loss = loss + unsafe_weight * unsafe_loss
    density_weight = float(getattr(args, "density_contrast_loss_weight", 0.0))
    if density_weight > 0.0:
      loss = loss + density_weight * density_loss
    if source_adversary is not None:
      source_features = model.source_features(x_t, state_groups)
      source_logits = source_adversary(_grad_reverse(source_features, 1.0))
      source_loss_items = F.cross_entropy(
          source_logits,
          source_labels,
          reduction="none",
      )
      source_loss = _weighted_mean(source_loss_items, source_weights)
      loss = loss + float(args.source_adversary_weight) * source_loss
      source_pred = torch.argmax(source_logits.detach(), dim=1)
      source_acc = float(
          torch.mean((source_pred == source_labels).float()).item()
      )
    else:
      source_loss = loss * 0.0
      source_acc = 0.0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.item()))
    bce_losses.append(float(bce_loss.item()))
    pairwise_losses.append(float(pair_loss.item()))
    listwise_losses.append(float(list_loss.item()))
    unsafe_margin_losses.append(float(unsafe_loss.item()))
    density_contrast_losses.append(float(density_loss.item()))
    source_losses.append(float(source_loss.item()))
    source_accs.append(float(source_acc))
  report = {
      "initial_loss": round(float(losses[0]), 6),
      "final_loss": round(float(losses[-1]), 6),
      "initial_bce_loss": round(float(bce_losses[0]), 6),
      "final_bce_loss": round(float(bce_losses[-1]), 6),
      "initial_pairwise_loss": round(float(pairwise_losses[0]), 6),
      "final_pairwise_loss": round(float(pairwise_losses[-1]), 6),
      "initial_listwise_loss": round(float(listwise_losses[0]), 6),
      "final_listwise_loss": round(float(listwise_losses[-1]), 6),
      "initial_unsafe_margin_loss": round(float(unsafe_margin_losses[0]), 6),
      "final_unsafe_margin_loss": round(float(unsafe_margin_losses[-1]), 6),
      "initial_density_contrast_loss": round(
          float(density_contrast_losses[0]),
          6,
      ),
      "final_density_contrast_loss": round(
          float(density_contrast_losses[-1]),
          6,
      ),
      "pos_weight": round(float(pos_weight.item()), 6),
      "loss_mode": loss_mode,
      "proposal_model_kind": model_kind,
      "pairwise_loss_weight": float(args.pairwise_loss_weight),
      "pairwise_margin": float(args.pairwise_margin),
      "listwise_loss_weight": float(args.listwise_loss_weight),
      "unsafe_margin_loss_weight": float(
          getattr(args, "unsafe_margin_loss_weight", 0.0)
      ),
      "unsafe_margin": float(getattr(args, "unsafe_margin", 1.0)),
      "unsafe_positive_mode": str(
          getattr(args, "unsafe_positive_mode", "clean_safe")
      ),
      "density_contrast_loss_weight": float(
          getattr(args, "density_contrast_loss_weight", 0.0)
      ),
      "density_contrast_margin": float(
          getattr(args, "density_contrast_margin", 2.0)
      ),
      "source_adversary_weight": float(args.source_adversary_weight),
      "source_balance_weight": float(args.source_balance_weight),
      "main_source_balance_weight": float(main_source_weight),
      "main_source_weight_report": main_source_report,
      "source_adversary_enabled": bool(source_adversary is not None),
      "initial_source_loss": round(float(source_losses[0]), 6),
      "final_source_loss": round(float(source_losses[-1]), 6),
      "initial_source_acc": round(float(source_accs[0]), 6),
      "final_source_acc": round(float(source_accs[-1]), 6),
      "state_groups": int(len(state_groups)),
      "pairwise_groups": int(len(pair_groups)),
      "listwise_groups": int(len(list_groups)),
      "unsafe_margin_groups": int(len(unsafe_groups)),
      "density_contrast_groups": int(len(density_groups)),
  }
  report.update(source_report)
  return model, report, source_adversary


def _predict(model, x: np.ndarray, examples: list[dict]) -> np.ndarray:
  with torch.no_grad():
    x_t = torch.tensor(x, dtype=torch.float32)
    if isinstance(model, ProposalVetoDeepSet):
      logits = model(x_t, _state_groups(examples))
    else:
      logits = model(x_t)
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def _prediction_summary(examples: list[dict], probs: np.ndarray) -> dict:
  if not examples:
    return {"examples": 0}
  y = np.asarray([1.0 if ex["approved"] else 0.0 for ex in examples], dtype=np.float32)
  positives = probs[y > 0.5]
  negatives = probs[y <= 0.5]
  return {
      "examples": int(len(examples)),
      "approved_targets": int(np.sum(y > 0.5)),
      "mean_prob_on_approved": (
          None if len(positives) == 0 else round(float(np.mean(positives)), 6)
      ),
      "mean_prob_on_rejected": (
          None if len(negatives) == 0 else round(float(np.mean(negatives)), 6)
      ),
  }


def _group_top1_summary(examples: list[dict], probs: np.ndarray) -> dict:
  if not examples:
    return {"groups": 0}
  by_state: dict[tuple[int, int], list[int]] = {}
  for idx, example in enumerate(examples):
    key = (
        int(example.get("source_replay_index", -1)),
        int(example.get("state_index", -1)),
    )
    by_state.setdefault(key, []).append(int(idx))
  groups = [indices for indices in by_state.values() if len(indices) >= 2]
  if not groups:
    return {"groups": 0, "examples": int(len(examples))}
  selected = []
  groups_with_approved = 0
  oracle_best_values = []
  for indices in groups:
    approved_indices = [
        idx for idx in indices if bool(examples[idx].get("approved", False))
    ]
    if approved_indices:
      groups_with_approved += 1
      oracle_best_values.append(
          max(float(examples[idx].get("dense_value_delta", 0.0)) for idx in approved_indices)
      )
    top_idx = max(indices, key=lambda idx: float(probs[int(idx)]))
    selected.append(examples[top_idx])
  selected_values = [float(ex["dense_value_delta"]) for ex in selected]
  selected_risks = [float(ex["paradox_delta"]) for ex in selected]
  selected_rewards = [
      float(ex.get("group_relative_reward", 0.0))
      for ex in selected
  ]
  oracle_best_rewards = [
      max(float(examples[idx].get("group_relative_reward", 0.0)) for idx in indices)
      for indices in groups
  ]
  approved = int(sum(bool(ex.get("approved", False)) for ex in selected))
  safe = int(sum(bool(ex.get("safe_by_paradox_delta", False)) for ex in selected))
  candidate_any = int(sum(bool(ex.get("candidate_any_paradox", False)) for ex in selected))
  bad_red_no_win = int(
      sum(bool(ex.get("bad_red_no_win_negative", False)) for ex in selected)
  )
  return {
      "groups": int(len(groups)),
      "groups_with_approved": int(groups_with_approved),
      "top1_approved": int(approved),
      "top1_approved_rate": round(float(approved / max(1, len(groups))), 6),
      "top1_safe": int(safe),
      "top1_safe_rate": round(float(safe / max(1, len(groups))), 6),
      "top1_candidate_any_paradox": int(candidate_any),
      "top1_candidate_any_paradox_rate": round(
          float(candidate_any / max(1, len(groups))),
          6,
      ),
      "top1_bad_red_no_win_negative": int(bad_red_no_win),
      "top1_bad_red_no_win_negative_rate": round(
          float(bad_red_no_win / max(1, len(groups))),
          6,
      ),
      "top1_mean_dense_value_delta": round(float(np.mean(selected_values)), 6),
      "top1_mean_paradox_delta": round(float(np.mean(selected_risks)), 6),
      "top1_mean_group_relative_reward": round(
          float(np.mean(selected_rewards)),
          6,
      ),
      "oracle_best_group_relative_reward": round(
          float(np.mean(oracle_best_rewards)),
          6,
      ),
      "oracle_best_approved_mean_value_delta": (
          None if not oracle_best_values else
          round(float(np.mean(oracle_best_values)), 6)
      ),
  }


def _passes_validation_source_guards(args, example: dict) -> bool:
  proposal = example.get("proposal", {})
  source_value = float(proposal.get("candidate_pred_value_delta", 0.0))
  source_positive = float(proposal.get("candidate_pred_positive_prob", 0.0))
  source_margin = float(proposal.get("score_margin", 0.0))
  if source_value < float(getattr(args, "validation_min_source_value_delta", -1e9)):
    return False
  if source_positive < float(
      getattr(args, "validation_min_source_positive_prob", -1e9)
  ):
    return False
  if source_margin < float(getattr(args, "validation_min_source_margin", -1e9)):
    return False
  return True


def _threshold_report(args, examples: list[dict], probs: np.ndarray) -> list[dict]:
  report = []
  total_approved_targets = int(sum(bool(ex["approved"]) for ex in examples))
  total_rejected_targets = int(len(examples) - total_approved_targets)
  for threshold in _parse_float_list(args.approval_thresholds):
    above_threshold = [
        (ex, float(prob))
        for ex, prob in zip(examples, probs)
        if float(prob) >= float(threshold)
    ]
    selected = [
        (ex, prob)
        for ex, prob in above_threshold
        if _passes_validation_source_guards(args, ex)
    ]
    approvals = len(selected)
    true_approvals = int(sum(bool(ex["approved"]) for ex, _ in selected))
    false_approvals = int(approvals - true_approvals)
    safe = int(sum(bool(ex["safe_by_paradox_delta"]) for ex, _ in selected))
    candidate_any = int(sum(bool(ex["candidate_any_paradox"]) for ex, _ in selected))
    bad_red_no_win = int(
        sum(bool(ex.get("bad_red_no_win_negative", False)) for ex, _ in selected)
    )
    dense_values = [float(ex["dense_value_delta"]) for ex, _ in selected]
    paradox_deltas = [float(ex["paradox_delta"]) for ex, _ in selected]
    rewards = [
        float(ex.get("group_relative_reward", 0.0))
        for ex, _ in selected
    ]
    precision = None if approvals == 0 else float(true_approvals / approvals)
    recall = (
        None if total_approved_targets == 0 else
        float(true_approvals / total_approved_targets)
    )
    false_rate = (
        None if total_rejected_targets == 0 else
        float(false_approvals / total_rejected_targets)
    )
    safe_rate = None if approvals == 0 else float(safe / approvals)
    row = {
        "threshold": float(threshold),
        "examples": int(len(examples)),
        "approved_targets": int(total_approved_targets),
        "rejected_targets": int(total_rejected_targets),
        "approvals": int(approvals),
        "probability_approvals_before_source_guards": int(len(above_threshold)),
        "blocked_by_validation_source_guards": int(len(above_threshold) - approvals),
        "approval_rate": round(float(approvals / max(1, len(examples))), 6),
        "approval_precision": (
            None if precision is None else round(float(precision), 6)
        ),
        "approval_recall": None if recall is None else round(float(recall), 6),
        "false_approvals": int(false_approvals),
        "false_approval_rate": (
            None if false_rate is None else round(float(false_rate), 6)
        ),
        "safe_approvals": int(safe),
        "safe_rate": None if safe_rate is None else round(float(safe_rate), 6),
        "candidate_any_paradox_approvals": int(candidate_any),
        "candidate_any_paradox_rate": (
            None if approvals == 0 else round(float(candidate_any / approvals), 6)
        ),
        "bad_red_no_win_approvals": int(bad_red_no_win),
        "bad_red_no_win_rate": (
            None if approvals == 0 else round(float(bad_red_no_win / approvals), 6)
        ),
        "mean_dense_value_delta": (
            None if not dense_values else round(float(np.mean(dense_values)), 6)
        ),
        "mean_paradox_delta": (
            None if not paradox_deltas else round(float(np.mean(paradox_deltas)), 6)
        ),
        "mean_group_relative_reward": (
            None if not rewards else round(float(np.mean(rewards)), 6)
        ),
        "mean_approval_prob": (
            None if approvals == 0 else
            round(float(np.mean([prob for _, prob in selected])), 6)
        ),
    }
    report.append(row)
  return report


def _threshold_passes(args, row: dict) -> bool:
  if int(row["approved_targets"]) <= 0:
    return False
  if int(row["approvals"]) < int(args.min_validation_approvals_for_live):
    return False
  precision = row.get("approval_precision")
  if precision is None or float(precision) < float(args.min_validation_approval_precision_for_live):
    return False
  recall = row.get("approval_recall")
  if recall is None or float(recall) < float(args.min_validation_approval_recall_for_live):
    return False
  false_rate = row.get("false_approval_rate")
  if (
      false_rate is not None
      and float(false_rate) > float(args.max_validation_false_approval_rate_for_live)
  ):
    return False
  safe_rate = row.get("safe_rate")
  if safe_rate is None or float(safe_rate) < float(args.min_validation_safe_rate_for_live):
    return False
  mean_value = row.get("mean_dense_value_delta")
  if mean_value is None or float(mean_value) < float(args.min_validation_mean_value_delta_for_live):
    return False
  mean_risk = row.get("mean_paradox_delta")
  if mean_risk is None or float(mean_risk) > float(args.max_validation_mean_risk_delta_for_live):
    return False
  candidate_any_rate = row.get("candidate_any_paradox_rate")
  if (
      candidate_any_rate is None
      or float(candidate_any_rate)
      > float(args.max_validation_candidate_any_paradox_rate_for_live)
  ):
    return False
  return True


def _negative_threshold_passes(args, row: dict) -> bool:
  if int(row["approved_targets"]) > 0:
    return False
  approvals = int(row.get("approvals", 0))
  if approvals > int(args.max_negative_validation_approvals_for_live):
    return False
  approval_rate = float(row.get("approval_rate", 0.0))
  if approval_rate > float(args.max_negative_validation_approval_rate_for_live):
    return False
  candidate_any = int(row.get("candidate_any_paradox_approvals", 0))
  if (
      candidate_any
      > int(args.max_negative_validation_candidate_any_approvals_for_live)
  ):
    return False
  return True


def _best_threshold(rows: list[dict]) -> dict | None:
  if not rows:
    return None
  def key(row):
    precision = -1.0 if row.get("approval_precision") is None else float(row["approval_precision"])
    recall = -1.0 if row.get("approval_recall") is None else float(row["approval_recall"])
    false_rate = 999.0 if row.get("false_approval_rate") is None else float(row["false_approval_rate"])
    safe = -1.0 if row.get("safe_rate") is None else float(row["safe_rate"])
    value = -999.0 if row.get("mean_dense_value_delta") is None else float(row["mean_dense_value_delta"])
    approvals = int(row.get("approvals", 0))
    return (
        precision,
        safe,
        value,
        recall,
        -false_rate,
        approvals,
    )
  return max(rows, key=key)


def _best_negative_threshold(rows: list[dict]) -> dict | None:
  if not rows:
    return None

  def key(row):
    return (
        -int(row.get("approvals", 0)),
        -int(row.get("candidate_any_paradox_approvals", 0)),
        -float(row.get("approval_rate", 0.0)),
        float(row.get("threshold", 0.0)),
    )

  return max(rows, key=key)


def _gate(args, rows: list[dict]) -> dict:
  eligible = [row for row in rows if _threshold_passes(args, row)]
  return {
      "passed": bool(eligible),
      "eligible_thresholds": int(len(eligible)),
      "best_threshold": _best_threshold(eligible) if eligible else None,
      "min_approvals": int(args.min_validation_approvals_for_live),
      "min_precision": float(args.min_validation_approval_precision_for_live),
      "min_recall": float(args.min_validation_approval_recall_for_live),
      "max_false_approval_rate": float(
          args.max_validation_false_approval_rate_for_live
      ),
      "min_safe_rate": float(args.min_validation_safe_rate_for_live),
      "min_mean_value_delta": float(args.min_validation_mean_value_delta_for_live),
      "max_mean_risk_delta": float(args.max_validation_mean_risk_delta_for_live),
      "max_candidate_any_paradox_rate": float(
          args.max_validation_candidate_any_paradox_rate_for_live
      ),
      "validation_min_source_margin": float(args.validation_min_source_margin),
      "validation_min_source_value_delta": float(
          args.validation_min_source_value_delta
      ),
      "validation_min_source_positive_prob": float(
          args.validation_min_source_positive_prob
      ),
  }


def _negative_gate(args, rows: list[dict]) -> dict:
  eligible = [row for row in rows if _negative_threshold_passes(args, row)]
  return {
      "passed": bool(eligible),
      "eligible_thresholds": int(len(eligible)),
      "best_threshold": (
          _best_negative_threshold(eligible) if eligible else None
      ),
      "applies_when_approved_targets": 0,
      "max_approvals": int(args.max_negative_validation_approvals_for_live),
      "max_approval_rate": float(
          args.max_negative_validation_approval_rate_for_live
      ),
      "max_candidate_any_paradox_approvals": int(
          args.max_negative_validation_candidate_any_approvals_for_live
      ),
      "validation_min_source_margin": float(args.validation_min_source_margin),
      "validation_min_source_value_delta": float(
          args.validation_min_source_value_delta
      ),
      "validation_min_source_positive_prob": float(
          args.validation_min_source_positive_prob
      ),
  }


def _selected_validation_gate(
    positive_gate: dict,
    negative_gate: dict,
    rows: list[dict],
) -> dict:
  approved_targets = int(rows[0].get("approved_targets", 0)) if rows else 0
  if approved_targets == 0:
    selected = dict(negative_gate)
    selected["mode"] = "negative_only_abstention"
    return selected
  selected = dict(positive_gate)
  selected["mode"] = "positive_precision"
  return selected


def main():
  args = parse_args()
  np.random.seed(int(args.seed))
  torch.manual_seed(int(args.seed))

  rows, metadata_rows, source_reports = _load_replays_and_metadata(args)
  states, build_summary = _build_states(args, rows, metadata_rows)
  examples, skipped = _build_proposal_examples(args, states, metadata_rows)
  all_sources = sorted({int(ex.get("source_replay_index", -1)) for ex in examples})
  validation_sources = _parse_int_set(str(args.validation_source_indices))
  train_sources = (
      _parse_int_set(str(args.train_source_indices))
      if str(args.train_source_indices or "").strip() else
      set(all_sources) - set(validation_sources)
  )
  train_examples = _source_examples(examples, train_sources)
  validation_examples = _source_examples(examples, validation_sources)
  if not train_examples or not validation_examples:
    raise ValueError("empty train/validation proposal example split")
  train_x_raw = np.asarray([ex["feature"] for ex in train_examples], dtype=np.float32)
  validation_x_raw = np.asarray(
      [ex["feature"] for ex in validation_examples], dtype=np.float32
  )
  all_x_raw = np.asarray([ex["feature"] for ex in examples], dtype=np.float32)
  (train_x, validation_x, all_x), mean, std = _standardize(
      train_x_raw, train_x_raw, validation_x_raw, all_x_raw
  )
  train_y = np.asarray(
      [1.0 if ex["approved"] else 0.0 for ex in train_examples],
      dtype=np.float32,
  )
  model, train_report, source_adversary = _train(
      args,
      train_x,
      train_y,
      train_examples,
  )
  train_probs = _predict(model, train_x, train_examples)
  validation_probs = _predict(model, validation_x, validation_examples)
  all_probs = _predict(model, all_x, examples)
  train_thresholds = _threshold_report(args, train_examples, train_probs)
  validation_thresholds = _threshold_report(
      args, validation_examples, validation_probs
  )
  all_thresholds = _threshold_report(args, examples, all_probs)
  validation_gate = _gate(args, validation_thresholds)
  validation_negative_gate = _negative_gate(args, validation_thresholds)
  validation_gate_for_live = _selected_validation_gate(
      validation_gate,
      validation_negative_gate,
      validation_thresholds,
  )
  report = {
      "kind": "proposal_veto_trainer",
      "args": vars(args),
      "replay_sources": source_reports,
      "state_build_summary": build_summary,
      "state_summary_by_source": _state_summary_by_source(states),
      "proposal_example_skipped": skipped,
      "source_split": {
          "train_source_indices": sorted(int(x) for x in train_sources),
          "validation_source_indices": sorted(int(x) for x in validation_sources),
      },
      "feature_size": int(train_x.shape[1]),
      "approve_label": str(args.approve_label),
      "train_summary": _summary(train_examples),
      "validation_summary": _summary(validation_examples),
      "all_summary": _summary(examples),
      "train_summary_by_source": _summary_by_source(train_examples),
      "validation_summary_by_source": _summary_by_source(validation_examples),
      "all_summary_by_source": _summary_by_source(examples),
      "train_pairwise_group_summary": _pairwise_group_summary(train_examples),
      "validation_pairwise_group_summary": _pairwise_group_summary(
          validation_examples
      ),
      "all_pairwise_group_summary": _pairwise_group_summary(examples),
      "train_unsafe_margin_group_summary": _unsafe_margin_group_summary(
          train_examples,
          args,
      ),
      "validation_unsafe_margin_group_summary": _unsafe_margin_group_summary(
          validation_examples,
          args,
      ),
      "all_unsafe_margin_group_summary": _unsafe_margin_group_summary(
          examples,
          args,
      ),
      "train_density_contrast_group_summary": _density_contrast_group_summary(
          train_examples,
          args,
      ),
      "validation_density_contrast_group_summary": _density_contrast_group_summary(
          validation_examples,
          args,
      ),
      "all_density_contrast_group_summary": _density_contrast_group_summary(
          examples,
          args,
      ),
      "train_report": train_report,
      "train_prediction_summary": _prediction_summary(train_examples, train_probs),
      "validation_prediction_summary": _prediction_summary(
          validation_examples, validation_probs
      ),
      "all_prediction_summary": _prediction_summary(examples, all_probs),
      "train_group_top1_summary": _group_top1_summary(
          train_examples,
          train_probs,
      ),
      "validation_group_top1_summary": _group_top1_summary(
          validation_examples,
          validation_probs,
      ),
      "all_group_top1_summary": _group_top1_summary(examples, all_probs),
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "all_thresholds": all_thresholds,
      "best_validation_threshold": _best_threshold(validation_thresholds),
      "validation_gate": validation_gate,
      "validation_negative_gate": validation_negative_gate,
      "validation_gate_for_live": validation_gate_for_live,
      "model_out": str(args.model_out or ""),
  }
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  if args.model_out:
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": (
                "proposal_veto_deepset"
                if str(args.proposal_model_kind) == "deepset" else
                "proposal_veto_mlp"
            ),
            "proposal_model_kind": str(args.proposal_model_kind),
            "state_dict": model.state_dict(),
            "source_adversary_state_dict": (
                None if source_adversary is None else source_adversary.state_dict()
            ),
            "feature_size": int(train_x.shape[1]),
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "approve_label": str(args.approve_label),
            "args": vars(args),
        },
        args.model_out,
    )
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
