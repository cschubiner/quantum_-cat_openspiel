#!/usr/bin/env python3
"""Train a legal-set selector that can choose base/abstain or an override."""

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
    adapt_action_features,
    adapt_observation,
    load_replay,
    model_input_size,
)
from quantum_cat_calibration_transfer_audit import (
    _load_policy_context_model,
    _policy_probs_from_row,
    _standardize,
    observation_feature_vector,
    pair_feature_vector_from_matrix,
    policy_context_feature_vector,
)
from quantum_cat_table_liveness_improve_trainer import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    _feature_bool,
    _led_lane_burn_label_shape,
    _load_metadata_rows,
    _metadata_group_key,
    _metadata_path_for,
    _parse_float_list,
    _split_state_ids,
    _split_state_ids_by_group,
)


class LegalSetSelector(nn.Module):
  """DeepSets-style selector over the full candidate set including base."""

  def __init__(self, feature_size: int, hidden: int):
    super().__init__()
    self.item = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )
    self.score = nn.Sequential(
        nn.Linear(hidden * 3, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    self.safe = nn.Sequential(
        nn.Linear(hidden * 3, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    self.positive = nn.Sequential(
        nn.Linear(hidden * 3, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )

  def forward(self, x: torch.Tensor):
    h = self.item(x)
    mean = torch.mean(h, dim=0, keepdim=True).expand_as(h)
    maxv = torch.max(h, dim=0, keepdim=True).values.expand_as(h)
    z = torch.cat([h, mean, maxv], dim=-1)
    return (
        self.score(z).squeeze(-1),
        self.safe(z).squeeze(-1),
        self.positive(z).squeeze(-1),
    )

  def forward_padded(self, x: torch.Tensor, mask: torch.Tensor):
    h = self.item(x)
    mask_f = mask.to(dtype=h.dtype).unsqueeze(-1)
    count = torch.clamp(mask_f.sum(dim=1, keepdim=True), min=1.0)
    mean = (h * mask_f).sum(dim=1, keepdim=True) / count
    masked_h = h.masked_fill(~mask.unsqueeze(-1), -1e9)
    maxv = masked_h.max(dim=1, keepdim=True).values
    maxv = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))
    z = torch.cat([h, mean.expand_as(h), maxv.expand_as(h)], dim=-1)
    score = self.score(z).squeeze(-1).masked_fill(~mask, -1e9)
    safe = self.safe(z).squeeze(-1)
    positive = self.positive(z).squeeze(-1)
    return score, safe, positive


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument(
      "--metadata",
      action="append",
      default=[],
      help=(
          "Optional metadata path for each --replay. When omitted, each replay's "
          ".metadata.json sidecar is used."
      ),
  )
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--seed", type=int, default=20299321)
  parser.add_argument("--validation-fraction", type=float, default=0.25)
  parser.add_argument(
      "--validation-split",
      choices=("state", "metadata_group", "source"),
      default="metadata_group",
  )
  parser.add_argument(
      "--validation-source-indices",
      default="",
      help=(
          "Comma-separated replay source indices to hold out when "
          "--validation-split=source. Source indices match the order of "
          "repeated --replay arguments."
      ),
  )
  parser.add_argument("--epochs", type=int, default=160)
  parser.add_argument(
      "--batch-size",
      type=int,
      default=0,
      help=(
          "Optional padded legal-set batch size. 0 preserves the historical "
          "per-state trainer."
      ),
  )
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--hidden", type=int, default=96)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--positive-value-margin", type=float, default=0.04)
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
          "full keeps the historical 11-block action-set expansion; "
          "compact_action_delta uses candidate features, candidate-minus-base, "
          "small observation/policy summaries, and two scalars."
      ),
  )
  parser.add_argument(
      "--base-action-source",
      choices=("metadata_policy_best", "policy_context_argmax"),
      default="metadata_policy_best",
  )
  parser.add_argument(
      "--target-mode",
      choices=(
          "best_positive_else_base",
          "best_value_positive_else_base",
          "group_relative_soft",
          "replay_action",
      ),
      default="best_positive_else_base",
  )
  parser.add_argument(
      "--chained-audit-pair-only",
      action="store_true",
      help=(
          "For chained best-of-rollouts rows, train only on the incumbent and "
          "the proposed action that received independent audit passes. Other "
          "legal actions were not independently audited and must not be "
          "silently treated as hard negatives."
      ),
  )
  parser.add_argument("--soft-target-temperature", type=float, default=0.35)
  parser.add_argument("--soft-target-risk-penalty", type=float, default=1.0)
  parser.add_argument("--soft-target-unsafe-penalty", type=float, default=1.0)
  parser.add_argument("--target-positive-weight", type=float, default=1.0)
  parser.add_argument("--target-base-weight", type=float, default=1.0)
  parser.add_argument("--safe-loss-weight", type=float, default=0.25)
  parser.add_argument("--positive-loss-weight", type=float, default=0.50)
  parser.add_argument(
      "--thresholds",
      default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95",
  )
  parser.add_argument(
      "--risk-thresholds",
      default="0.50,0.70,0.85,0.90,0.95,0.97,0.99",
  )
  parser.add_argument(
      "--positive-thresholds",
      default="0.50,0.70,0.85,0.90,0.95,0.97,0.99",
  )
  parser.add_argument("--min-validation-overrides-for-live", type=int, default=8)
  parser.add_argument(
      "--min-validation-positive-precision-for-live", type=float, default=0.90
  )
  parser.add_argument("--min-validation-safe-rate-for-live", type=float, default=0.90)
  parser.add_argument(
      "--min-validation-mean-value-delta-for-live", type=float, default=0.0
  )
  parser.add_argument(
      "--max-validation-mean-risk-delta-for-live", type=float, default=0.0
  )
  parser.add_argument(
      "--force-led-lane-burn-negative",
      action="store_true",
      help=(
          "Force off-led candidates that newly burn the led-color token to be "
          "negative/unsafe when the base action follows led."
      ),
  )
  parser.add_argument(
      "--led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="urgent_win",
      help=(
          "Exception for forced led-lane negatives. win keeps any red switch "
          "that would win now; urgent_win keeps only non-over-target red wins "
          "needed to hit prediction."
      ),
  )
  return parser.parse_args()


def _target_index(actions: list[dict], target_mode: str):
  positives = [idx for idx, row in enumerate(actions) if bool(row["positive"])]
  if positives:
    if target_mode in ("best_value_positive_else_base", "group_relative_soft"):
      return max(
          positives,
          key=lambda idx: (
              float(actions[idx]["value_delta"]),
              -float(actions[idx]["risk_delta"]),
              float(actions[idx]["policy_prob"]),
          ),
      )
    return max(
        positives,
        key=lambda idx: (
            -float(actions[idx]["risk_delta"]),
            float(actions[idx]["value_delta"]),
            float(actions[idx]["policy_prob"]),
        ),
    )
  for idx, row in enumerate(actions):
    if bool(row["is_base"]):
      return idx
  raise ValueError("base action missing from legal-set state")


def _hard_target_probs(actions: list[dict], target: int) -> list[float]:
  probs = [0.0] * len(actions)
  if 0 <= int(target) < len(probs):
    probs[int(target)] = 1.0
  return probs


def _group_relative_target_probs(actions: list[dict], args) -> list[float]:
  hard_target = _target_index(actions, "best_value_positive_else_base")
  if not any(bool(action.get("positive", False)) for action in actions):
    return _hard_target_probs(actions, hard_target)
  temperature = max(1e-6, float(getattr(args, "soft_target_temperature", 0.35)))
  risk_penalty = float(getattr(args, "soft_target_risk_penalty", 1.0))
  unsafe_penalty = float(getattr(args, "soft_target_unsafe_penalty", 1.0))
  scores = []
  for action in actions:
    if bool(action.get("forced_led_lane_negative", False)):
      scores.append(-np.inf)
      continue
    score = float(action["value_delta"]) - risk_penalty * float(action["risk_delta"])
    if not bool(action.get("safe", False)):
      score -= unsafe_penalty
    scores.append(float(score))
  finite = np.isfinite(np.asarray(scores, dtype=np.float64))
  if not bool(np.any(finite)):
    return _hard_target_probs(actions, hard_target)
  logits = np.asarray(scores, dtype=np.float64)
  logits[~finite] = -np.inf
  logits = logits / temperature
  logits = logits - float(np.max(logits[finite]))
  exp = np.zeros_like(logits, dtype=np.float64)
  exp[finite] = np.exp(logits[finite])
  total = float(np.sum(exp))
  if total <= 0.0 or not np.isfinite(total):
    return _hard_target_probs(actions, hard_target)
  return [float(x) for x in (exp / total)]


def _selection_target_probs(actions: list[dict], target: int, args) -> list[float]:
  if str(getattr(args, "target_mode", "")) == "group_relative_soft":
    return _group_relative_target_probs(actions, args)
  return _hard_target_probs(actions, target)


def _metadata_path_for_replay(replay_path: str, metadata: str):
  return Path(metadata) if metadata else _metadata_path_for(replay_path)


def _load_replays_and_metadata(args):
  replay_paths = list(args.replay)
  metadata_paths = list(args.metadata or [])
  if metadata_paths and len(metadata_paths) != len(replay_paths):
    raise ValueError(
        "when --metadata is provided with multi-replay training, pass exactly "
        f"one --metadata per --replay ({len(metadata_paths)} vs "
        f"{len(replay_paths)})"
    )

  rows = []
  metadata_rows = []
  source_reports = []
  # Explicit sidecars may carry audit-rejection labels, public-history
  # features, and provenance even when the live base action comes from the
  # policy checkpoint.  Silently discarding them changes supervision.
  needs_metadata = bool(metadata_paths) or (
      str(args.base_action_source) == "metadata_policy_best"
  )
  for source_idx, replay_path in enumerate(replay_paths):
    source_rows, embedded_metadata = load_replay(replay_path, return_metadata=True)
    if not source_rows:
      raise ValueError(f"no replay rows loaded from {replay_path}")
    rows.extend(source_rows)
    row_group_ids = []
    if isinstance(embedded_metadata, dict):
      raw_group_ids = embedded_metadata.get("row_group_ids")
      if isinstance(raw_group_ids, list) and len(raw_group_ids) == len(source_rows):
        row_group_ids = [str(value) for value in raw_group_ids]
    source_report = {
        "source_index": int(source_idx),
        "replay": str(replay_path),
        "rows": int(len(source_rows)),
        "embedded_row_group_ids": int(len(row_group_ids)),
    }
    source_row_infos = []
    for local_idx in range(len(source_rows)):
      if row_group_ids:
        row_group_id = (
            f"source={source_idx}|replay={Path(replay_path).name}|"
            f"{row_group_ids[local_idx]}"
        )
      else:
        row_group_id = (
            f"source={source_idx}|replay={Path(replay_path).name}|"
            f"row={local_idx}"
        )
      source_row_infos.append({
          "_source_replay_index": int(source_idx),
          "_source_replay": str(replay_path),
          "_source_row_index": int(local_idx),
          "row_group_id": row_group_id,
      })
    if needs_metadata:
      metadata_arg = metadata_paths[source_idx] if metadata_paths else ""
      metadata_path = _metadata_path_for_replay(replay_path, metadata_arg)
      if not metadata_path.exists():
        raise FileNotFoundError(f"required metadata sidecar not found: {metadata_path}")
      source_metadata = _load_metadata_rows(metadata_path)
      if len(source_metadata) != len(source_rows):
        raise ValueError(
            "replay/metadata row mismatch for "
            f"{replay_path}: {len(source_rows)} vs {len(source_metadata)}"
        )
      for local_idx, meta in enumerate(source_metadata):
        enriched = dict(meta)
        for key, value in source_row_infos[local_idx].items():
          enriched.setdefault(key, value)
        if "row_group_id" not in enriched:
          local_group_key = _metadata_group_key(meta, int(local_idx))
          enriched["row_group_id"] = (
              f"source={source_idx}|replay={Path(replay_path).name}|"
              f"{local_group_key}"
          )
        metadata_rows.append(enriched)
      source_report["metadata"] = str(metadata_path)
    else:
      metadata_rows.extend(source_row_infos)
    source_reports.append(source_report)
  if len(metadata_rows) != len(rows):
    raise ValueError(
        f"replay/metadata row mismatch after merge: {len(rows)} vs "
        f"{len(metadata_rows)}"
    )
  return rows, metadata_rows, source_reports


def _action_aux_predictions_from_row(row, model, device):
  num_actions = int(len(row[1]))
  zero = np.zeros(num_actions, dtype=np.float32)
  if model is None:
    return zero, zero
  obs_np = adapt_observation(
      np.asarray(row[0], dtype=np.float32).reshape(-1),
      model_input_size(model),
  )
  action_features = adapt_action_features(
      row[11] if len(row) > 11 else None,
      num_actions,
  )
  with torch.no_grad():
    obs_t = torch.tensor(
        obs_np, dtype=torch.float32, device=device
    ).unsqueeze(0)
    action_t = torch.tensor(
        action_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    action_values = None
    action_risks = None
    if hasattr(model, "forward_with_all_aux"):
      _, _, _, action_paradox_logits, action_values_t = (
          model.forward_with_all_aux(obs_t, action_t)
      )
      action_values = action_values_t.squeeze(0).detach().cpu().numpy()
      action_risks = torch.sigmoid(
          action_paradox_logits
      ).squeeze(0).detach().cpu().numpy()
    else:
      state_embedding = model.body(obs_t)
      if hasattr(model, "_action_values"):
        action_values = (
            model._action_values(state_embedding, action_t)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
      if hasattr(model, "forward_with_action_aux"):
        _, _, _, action_logits = model.forward_with_action_aux(obs_t, action_t)
        action_risks = torch.sigmoid(
            action_logits
        ).squeeze(0).detach().cpu().numpy()
  if action_values is None:
    action_values = zero
  if action_risks is None:
    action_risks = zero
  return (
      np.asarray(action_values, dtype=np.float32),
      np.asarray(action_risks, dtype=np.float32),
  )


def _metadata_base_action(meta: dict) -> int:
  """Resolve the incumbent action without confusing it with a teacher target."""
  decision = meta.get("decision") if isinstance(meta.get("decision"), dict) else {}
  return int(
      meta.get(
          "policy_base_action",
          meta.get("policy_best_action", decision.get("policy_action", -1)),
      )
  )


def _chained_audit_pair_actions(
    meta: dict,
    base_action: int,
    labeled_mask: np.ndarray,
) -> list[int]:
  """Return the only two actions justified by a chained audit row."""
  target_kind = str(meta.get("target_kind", ""))
  if target_kind not in ("audited_correction", "audit_rejected_base_keep"):
    raise ValueError(f"unsupported chained audit target kind: {target_kind}")
  proposed_action = int(meta.get("proposed_best_action", -1))
  base_action = int(base_action)
  if proposed_action < 0 or proposed_action == base_action:
    raise ValueError("chained audit row lacks a distinct proposed action")
  mask = np.asarray(labeled_mask, dtype=bool)
  if (
      base_action < 0
      or base_action >= len(mask)
      or proposed_action >= len(mask)
      or not bool(mask[base_action])
      or not bool(mask[proposed_action])
  ):
    raise ValueError("chained audit pair is absent from labeled action masks")
  return sorted((base_action, proposed_action))


def _selector_pair_feature(
    args,
    action_features: np.ndarray,
    legal_mask: np.ndarray,
    candidate_action: int,
    base_action: int,
    observation: np.ndarray,
    policy: np.ndarray,
) -> np.ndarray:
  """Build either the legacy expanded pair or a compact public pair."""
  mode = str(getattr(args, "pair_feature_mode", "full"))
  if mode == "full":
    return pair_feature_vector_from_matrix(
        action_features,
        np.asarray(legal_mask, dtype=np.float32),
        int(candidate_action),
        int(base_action),
        target_action=int(candidate_action),
        label_margin=0.0,
        legal_count=int(np.sum(legal_mask)),
        candidate_count=int(np.sum(legal_mask)),
        include_label_margin=False,
        observation=observation,
        observation_feature_mode=str(args.observation_feature_mode),
        policy_context=policy,
        policy_context_mode=str(args.policy_context_mode),
        include_target_match_features=False,
    )
  if mode != "compact_action_delta":
    raise ValueError(f"unknown pair_feature_mode={mode}")
  features = np.asarray(action_features, dtype=np.float32)
  candidate = features[int(candidate_action)]
  base = features[int(base_action)]
  scalars = np.asarray([
      float(int(candidate_action) == int(base_action)),
      float(np.sum(legal_mask)) / max(1.0, float(len(legal_mask))),
  ], dtype=np.float32)
  return np.concatenate([
      scalars,
      candidate,
      candidate - base,
      observation_feature_vector(
          observation, mode=str(args.observation_feature_mode)
      ),
      policy_context_feature_vector(
          policy,
          legal_mask,
          int(candidate_action),
          int(base_action),
          mode=str(args.policy_context_mode),
      ),
  ]).astype(np.float32)


def _build_states(args, rows, metadata_rows):
  policy_model, policy_device = _load_policy_context_model(args, rows[0])
  states = []
  skipped = 0
  skipped_base_action = 0
  skipped_chained_audit_pair = 0
  chained_audit_pair_states = 0
  allow_value_only = bool(getattr(args, "allow_value_only_action_labels", False))
  value_only_action_label_states = 0
  class_counts = {"POSITIVE": 0, "NEAR_POSITIVE": 0, "NEGATIVE": 0}
  label_shape_counts = {
      "led_lane_base_follow_candidates": 0,
      "led_lane_forced_negative": 0,
      "led_lane_red_exception_kept": 0,
      "led_lane_raw_positive_forced_negative": 0,
      "led_lane_raw_safe_forced_unsafe": 0,
  }
  for state_idx, row in enumerate(rows):
    legal_mask = np.asarray(row[1], dtype=np.float32) > 0.5
    value_mask = np.asarray(row[10], dtype=np.float32) > 0.5
    risk_mask = np.asarray(row[8], dtype=np.float32) > 0.5
    if allow_value_only:
      labeled_mask = legal_mask & value_mask
      value_only_action_label_states += int(
          bool(np.any(legal_mask & value_mask)) and not bool(np.any(risk_mask))
      )
    else:
      labeled_mask = legal_mask & value_mask & risk_mask
    legal = [int(action) for action in np.flatnonzero(labeled_mask)]
    if len(legal) < 2:
      skipped += 1
      continue
    policy = _policy_probs_from_row(row, policy_model, policy_device)
    if policy is None or float(np.sum(policy[legal])) <= 0.0:
      skipped += 1
      continue
    if str(args.base_action_source) == "metadata_policy_best":
      meta = metadata_rows[state_idx]
      base_action = _metadata_base_action(meta)
      group_key = _metadata_group_key(meta, state_idx)
      source_replay_index = int(
          meta.get("source_replay_index", meta.get("_source_replay_index", -1))
      )
    else:
      meta = metadata_rows[state_idx] if metadata_rows else {}
      base_action = int(max(legal, key=lambda action: float(policy[action])))
      group_key = _metadata_group_key(meta, state_idx)
      source_replay_index = int(
          meta.get("source_replay_index", meta.get("_source_replay_index", -1))
      )
    if base_action not in set(legal):
      skipped += 1
      skipped_base_action += 1
      continue
    feature_legal_mask = legal_mask.astype(bool, copy=True)
    if bool(getattr(args, "chained_audit_pair_only", False)):
      try:
        legal = _chained_audit_pair_actions(meta, base_action, labeled_mask)
      except ValueError:
        skipped += 1
        skipped_chained_audit_pair += 1
        continue
      feature_legal_mask = np.zeros_like(legal_mask, dtype=bool)
      feature_legal_mask[np.asarray(legal, dtype=np.int64)] = True
      chained_audit_pair_states += 1
    audit_base_keep = bool(meta.get("audit_base_keep_validation", False))
    action_features = np.asarray(row[11], dtype=np.float32)
    aux_values, aux_risks = _action_aux_predictions_from_row(
        row,
        policy_model,
        policy_device,
    )
    values = np.asarray(row[9], dtype=np.float32)
    risks = np.asarray(row[7], dtype=np.float32)
    base_value = float(values[base_action])
    base_risk = float(risks[base_action])
    base_aux_value = float(aux_values[base_action])
    base_aux_risk = float(aux_risks[base_action])
    observation = row[0]
    actions = []
    for action in legal:
      value_delta = float(values[action]) - base_value
      risk_delta = float(risks[action]) - base_risk
      if audit_base_keep and int(action) != int(base_action):
        # These candidates looked better on the selection worlds but failed
        # the independent belief audit.  Their selection-world auxiliary
        # targets must not teach the selector to recreate the rejected
        # override; conservatively shape every unaudited alternative as a
        # non-improvement while the replay action hard-targets the base.
        value_delta = min(0.0, float(value_delta))
        risk_delta = max(0.0, float(risk_delta))
      aux_value = float(aux_values[action])
      aux_risk = float(aux_risks[action])
      raw_positive = (
          action != base_action
          and value_delta >= float(args.positive_value_margin)
          and risk_delta <= float(args.safe_risk_delta_max)
      )
      raw_safe = bool(risk_delta <= float(args.safe_risk_delta_max))
      positive = bool(raw_positive)
      safe = bool(raw_safe)
      forced_led_lane_negative, kept_by_red_exception = _led_lane_burn_label_shape(
          action_features, int(base_action), int(action), args
      )
      if bool(getattr(args, "force_led_lane_burn_negative", False)) and _feature_bool(
          action_features, int(base_action), ACTION_FEATURE_FOLLOWS_LED_INDEX
      ):
        label_shape_counts["led_lane_base_follow_candidates"] += 1
      if kept_by_red_exception:
        label_shape_counts["led_lane_red_exception_kept"] += 1
      if forced_led_lane_negative:
        label_shape_counts["led_lane_forced_negative"] += 1
        label_shape_counts["led_lane_raw_positive_forced_negative"] += int(
            bool(raw_positive)
        )
        label_shape_counts["led_lane_raw_safe_forced_unsafe"] += int(bool(raw_safe))
        positive = False
        safe = False
      if positive:
        row_class = "POSITIVE"
      elif value_delta > 0.0:
        row_class = "NEAR_POSITIVE"
      else:
        row_class = "NEGATIVE"
      if forced_led_lane_negative:
        row_class = "FORCED_LED_LANE_BURN_NEGATIVE"
      class_counts[row_class] = int(class_counts.get(row_class, 0)) + 1
      feature = _selector_pair_feature(
          args,
          action_features,
          feature_legal_mask,
          int(action),
          int(base_action),
          observation,
          policy,
      )
      actions.append({
          "action": int(action),
          "is_base": bool(action == base_action),
          "audit_base_keep": bool(audit_base_keep),
          "policy_prob": float(policy[action]),
          "value_delta": float(value_delta),
          "risk_delta": float(risk_delta),
          "aux_action_value": float(aux_value),
          "base_aux_action_value": float(base_aux_value),
          "aux_action_value_delta": float(aux_value - base_aux_value),
          "aux_action_paradox_risk": float(aux_risk),
          "base_aux_action_paradox_risk": float(base_aux_risk),
          "aux_action_paradox_risk_delta": float(aux_risk - base_aux_risk),
          "risk_label_available": bool(risk_mask[action]),
          "positive": bool(positive),
          "safe": bool(safe),
          "raw_positive": bool(raw_positive),
          "raw_safe": bool(raw_safe),
          "forced_led_lane_negative": bool(forced_led_lane_negative),
          "kept_by_led_lane_red_exception": bool(kept_by_red_exception),
          "class": row_class,
          "feature": feature,
          "raw_action_feature": np.asarray(
              action_features[int(action)],
              dtype=np.float32,
          ),
      })
    if str(args.target_mode) == "replay_action":
      replay_action = int(row[5])
      target = next(
          (
              idx for idx, action_row in enumerate(actions)
              if int(action_row["action"]) == replay_action
          ),
          -1,
      )
      if target < 0:
        skipped += 1
        continue
    else:
      target = _target_index(actions, str(args.target_mode))
    soft_target = _selection_target_probs(actions, int(target), args)
    for action, prob in zip(actions, soft_target):
      action["target_prob"] = float(prob)
    states.append({
        "state_index": int(state_idx),
        "group_key": group_key,
        "source_group_id": str(
            meta.get("source_group_id", meta.get("row_group_id", group_key))
        ),
        "source_replay_index": int(source_replay_index),
        "public_history": meta.get("public_history"),
        "trick_number": int(meta.get("trick_number", -1)),
        "base_action": int(base_action),
        "target_index": int(target),
        "target_action": int(actions[target]["action"]),
        "target_is_base": bool(actions[target]["is_base"]),
        "chained_audit_pair_only": bool(
            getattr(args, "chained_audit_pair_only", False)
        ),
        "soft_target": [float(value) for value in soft_target],
        "has_positive": bool(any(bool(row["positive"]) for row in actions)),
        "actions": actions,
    })
  if not states:
    raise ValueError("no legal-set states built")
  return states, {
      "skipped_states": int(skipped),
      "skipped_base_action_states": int(skipped_base_action),
      "skipped_chained_audit_pair_states": int(skipped_chained_audit_pair),
      "chained_audit_pair_states": int(chained_audit_pair_states),
      "allow_value_only_action_labels": bool(allow_value_only),
      "value_only_action_label_states": int(value_only_action_label_states),
      "class_counts": class_counts,
      "label_shape_counts": label_shape_counts,
  }


def _parse_int_set(spec: str) -> set[int]:
  values: set[int] = set()
  for part in str(spec or "").replace("|", ",").replace("+", ",").split(","):
    text = part.strip()
    if not text:
      continue
    values.add(int(text))
  return values


def _split_states(args, states):
  if str(args.validation_split) == "source":
    validation_sources = _parse_int_set(str(args.validation_source_indices or ""))
    if not validation_sources:
      raise ValueError(
          "--validation-split=source requires --validation-source-indices"
      )
    train_ids = [
        int(state["state_index"])
        for state in states
        if int(state.get("source_replay_index", -1)) not in validation_sources
    ]
    validation_ids = [
        int(state["state_index"])
        for state in states
        if int(state.get("source_replay_index", -1)) in validation_sources
    ]
    split_report = {
        "split": "source",
        "validation_source_indices": sorted(int(x) for x in validation_sources),
        "train_source_indices": sorted({
            int(state.get("source_replay_index", -1))
            for state in states
            if int(state.get("state_index")) in set(train_ids)
        }),
        "validation_groups": None,
        "train_groups": None,
    }
  elif str(args.validation_split) == "metadata_group":
    train_ids, validation_ids, split_report = _split_state_ids_by_group(
        states, float(args.validation_fraction), int(args.seed)
    )
  else:
    state_ids = [int(state["state_index"]) for state in states]
    train_ids, validation_ids = _split_state_ids(
        state_ids, float(args.validation_fraction), int(args.seed)
    )
    split_report = {
        "split": "state",
        "groups": None,
        "train_groups": None,
        "validation_groups": None,
    }
  train_set = set(train_ids)
  validation_set = set(validation_ids)
  train_states = [
      state for state in states if int(state["state_index"]) in train_set
  ]
  validation_states = [
      state for state in states if int(state["state_index"]) in validation_set
  ]
  if not train_states or not validation_states:
    raise ValueError("empty train/validation split")
  return train_states, validation_states, split_report


def _standardize_state_features(train_states, validation_states, all_states):
  train_features = np.asarray([
      action["feature"]
      for state in train_states
      for action in state["actions"]
  ], dtype=np.float32)
  all_features = np.asarray([
      action["feature"]
      for state in all_states
      for action in state["actions"]
  ], dtype=np.float32)
  (_, all_std), mean, std = _standardize(train_features, train_features, all_features)
  offset = 0
  for state in all_states:
    count = len(state["actions"])
    for action, feature in zip(state["actions"], all_std[offset:offset + count]):
      action["feature_std"] = feature.astype(np.float32)
    offset += count
  return mean.astype(np.float32), std.astype(np.float32)


def _tensor_for_state(state):
  return torch.tensor(
      np.asarray([action["feature_std"] for action in state["actions"]], dtype=np.float32),
      dtype=torch.float32,
  )


def _padded_batch_for_states(states):
  max_actions = max(len(state["actions"]) for state in states)
  feature_size = len(states[0]["actions"][0]["feature_std"])
  x = torch.zeros((len(states), max_actions, feature_size), dtype=torch.float32)
  mask = torch.zeros((len(states), max_actions), dtype=torch.bool)
  target = torch.zeros((len(states),), dtype=torch.long)
  target_weight = torch.zeros((len(states),), dtype=torch.float32)
  safe = torch.zeros((len(states), max_actions), dtype=torch.float32)
  positive = torch.zeros((len(states), max_actions), dtype=torch.float32)
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
    target_weight[row_idx] = float(state["target_weight"])
    safe[row_idx, :action_count] = torch.tensor(
        [float(action["safe"]) for action in state["actions"]],
        dtype=torch.float32,
    )
    positive[row_idx, :action_count] = torch.tensor(
        [float(action["positive"]) for action in state["actions"]],
        dtype=torch.float32,
    )
    soft_target[row_idx, :action_count] = torch.tensor(
        state.get("soft_target", _hard_target_probs(state["actions"], int(state["target_index"]))),
        dtype=torch.float32,
    )
  return x, mask, target, target_weight, safe, positive, soft_target


def _select_loss(args, logits, target_t, soft_t=None, reduction="mean"):
  if str(getattr(args, "target_mode", "")) == "group_relative_soft":
    if soft_t is None:
      raise ValueError("group_relative_soft requires soft targets")
    per_state = -(soft_t * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    if reduction == "none":
      return per_state
    return per_state.mean()
  return F.cross_entropy(logits, target_t, reduction=reduction)


def _train(args, train_states, feature_size):
  torch.manual_seed(int(args.seed))
  model = LegalSetSelector(feature_size, int(args.hidden))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  for state in train_states:
    state["target_weight"] = (
        float(args.target_base_weight)
        if bool(state["target_is_base"]) else
        float(args.target_positive_weight)
    )
  batch_size = int(getattr(args, "batch_size", 0) or 0)
  if batch_size > 1:
    return _train_batched(args, train_states, feature_size, model, optimizer, batch_size)
  state_tensors = [_tensor_for_state(state) for state in train_states]
  target_tensors = [
      torch.tensor([int(state["target_index"])], dtype=torch.long)
      for state in train_states
  ]
  target_weights = torch.tensor([
      float(state["target_weight"])
      for state in train_states
  ], dtype=torch.float32)
  safe_targets = [
      torch.tensor(
          [float(action["safe"]) for action in state["actions"]],
          dtype=torch.float32,
      )
      for state in train_states
  ]
  positive_targets = [
      torch.tensor(
          [float(action["positive"]) for action in state["actions"]],
          dtype=torch.float32,
      )
      for state in train_states
  ]
  soft_targets = [
      torch.tensor(
          state.get("soft_target", _hard_target_probs(state["actions"], int(state["target_index"]))),
          dtype=torch.float32,
      )
      for state in train_states
  ]
  losses = []
  select_losses = []
  safe_losses = []
  positive_losses = []
  for _ in range(max(1, int(args.epochs))):
    total = torch.tensor(0.0)
    select_total = torch.tensor(0.0)
    safe_total = torch.tensor(0.0)
    positive_total = torch.tensor(0.0)
    for x_t, target_t, safe_t, positive_t, soft_t, weight_t in zip(
        state_tensors,
        target_tensors,
        safe_targets,
        positive_targets,
        soft_targets,
        target_weights,
    ):
      logits, safe_logits, positive_logits = model(x_t)
      select_loss = _select_loss(
          args,
          logits.unsqueeze(0),
          target_t,
          soft_t.unsqueeze(0),
      ) * weight_t
      safe_loss = F.binary_cross_entropy_with_logits(safe_logits, safe_t)
      positive_loss = F.binary_cross_entropy_with_logits(
          positive_logits, positive_t
      )
      total = total + select_loss + float(args.safe_loss_weight) * safe_loss
      total = total + float(args.positive_loss_weight) * positive_loss
      select_total = select_total + select_loss.detach()
      safe_total = safe_total + safe_loss.detach()
      positive_total = positive_total + positive_loss.detach()
    denom = max(1, len(train_states))
    loss = total / denom
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
    select_losses.append(float(select_total / denom))
    safe_losses.append(float(safe_total / denom))
    positive_losses.append(float(positive_total / denom))
  return model, {
      "initial_loss": round(losses[0], 6),
      "final_loss": round(losses[-1], 6),
      "initial_select_loss": round(select_losses[0], 6),
      "final_select_loss": round(select_losses[-1], 6),
      "initial_safe_loss": round(safe_losses[0], 6),
      "final_safe_loss": round(safe_losses[-1], 6),
      "initial_positive_loss": round(positive_losses[0], 6),
      "final_positive_loss": round(positive_losses[-1], 6),
      "epochs": int(args.epochs),
      "lr": float(args.lr),
      "hidden": int(args.hidden),
      "safe_loss_weight": float(args.safe_loss_weight),
      "positive_loss_weight": float(args.positive_loss_weight),
      "target_positive_weight": float(args.target_positive_weight),
      "target_base_weight": float(args.target_base_weight),
      "target_mode": str(args.target_mode),
      "soft_target_temperature": float(getattr(args, "soft_target_temperature", 0.35)),
      "soft_target_risk_penalty": float(getattr(args, "soft_target_risk_penalty", 1.0)),
      "soft_target_unsafe_penalty": float(getattr(args, "soft_target_unsafe_penalty", 1.0)),
      "batch_size": int(batch_size),
  }


def _train_batched(args, train_states, feature_size, model, optimizer, batch_size):
  rng = random.Random(int(args.seed))
  losses = []
  select_losses = []
  safe_losses = []
  positive_losses = []
  indices = list(range(len(train_states)))
  for _ in range(max(1, int(args.epochs))):
    rng.shuffle(indices)
    total_loss_sum = 0.0
    select_loss_sum = 0.0
    safe_loss_sum = 0.0
    positive_loss_sum = 0.0
    seen = 0
    for start in range(0, len(indices), batch_size):
      batch_indices = indices[start:start + batch_size]
      batch_states = [train_states[idx] for idx in batch_indices]
      x_t, mask_t, target_t, weight_t, safe_t, positive_t, soft_t = (
          _padded_batch_for_states(batch_states)
      )
      logits, safe_logits, positive_logits = model.forward_padded(x_t, mask_t)
      select_per_state = _select_loss(
          args,
          logits,
          target_t,
          soft_t,
          reduction="none",
      )
      select_loss = torch.mean(select_per_state * weight_t)
      valid_counts = torch.clamp(mask_t.sum(dim=1).to(dtype=torch.float32), min=1.0)
      safe_loss_per_action = F.binary_cross_entropy_with_logits(
          safe_logits, safe_t, reduction="none"
      )
      positive_loss_per_action = F.binary_cross_entropy_with_logits(
          positive_logits, positive_t, reduction="none"
      )
      safe_loss = torch.mean(
          (safe_loss_per_action * mask_t.to(dtype=torch.float32)).sum(dim=1)
          / valid_counts
      )
      positive_loss = torch.mean(
          (positive_loss_per_action * mask_t.to(dtype=torch.float32)).sum(dim=1)
          / valid_counts
      )
      loss = (
          select_loss
          + float(args.safe_loss_weight) * safe_loss
          + float(args.positive_loss_weight) * positive_loss
      )
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      count = len(batch_states)
      total_loss_sum += float(loss.detach()) * count
      select_loss_sum += float(select_loss.detach()) * count
      safe_loss_sum += float(safe_loss.detach()) * count
      positive_loss_sum += float(positive_loss.detach()) * count
      seen += count
    denom = max(1, seen)
    losses.append(total_loss_sum / denom)
    select_losses.append(select_loss_sum / denom)
    safe_losses.append(safe_loss_sum / denom)
    positive_losses.append(positive_loss_sum / denom)
  return model, {
      "initial_loss": round(losses[0], 6),
      "final_loss": round(losses[-1], 6),
      "initial_select_loss": round(select_losses[0], 6),
      "final_select_loss": round(select_losses[-1], 6),
      "initial_safe_loss": round(safe_losses[0], 6),
      "final_safe_loss": round(safe_losses[-1], 6),
      "initial_positive_loss": round(positive_losses[0], 6),
      "final_positive_loss": round(positive_losses[-1], 6),
      "epochs": int(args.epochs),
      "lr": float(args.lr),
      "hidden": int(args.hidden),
      "safe_loss_weight": float(args.safe_loss_weight),
      "positive_loss_weight": float(args.positive_loss_weight),
      "target_positive_weight": float(args.target_positive_weight),
      "target_base_weight": float(args.target_base_weight),
      "target_mode": str(args.target_mode),
      "soft_target_temperature": float(getattr(args, "soft_target_temperature", 0.35)),
      "soft_target_risk_penalty": float(getattr(args, "soft_target_risk_penalty", 1.0)),
      "soft_target_unsafe_penalty": float(getattr(args, "soft_target_unsafe_penalty", 1.0)),
      "batch_size": int(batch_size),
  }


def _predict_states(model, states):
  model.eval()
  with torch.no_grad():
    for state in states:
      x_t = _tensor_for_state(state)
      logits, safe_logits, positive_logits = model(x_t)
      probs = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
      safe_probs = torch.sigmoid(safe_logits).cpu().numpy().astype(np.float32)
      positive_probs = torch.sigmoid(positive_logits).cpu().numpy().astype(np.float32)
      for action, prob, safe_prob, positive_prob, logit in zip(
          state["actions"],
          probs,
          safe_probs,
          positive_probs,
          logits.cpu().numpy().astype(np.float32),
      ):
        action["prob"] = float(prob)
        action["safe_prob"] = float(safe_prob)
        action["positive_prob"] = float(positive_prob)
        action["logit"] = float(logit)


def _selected_action(state, threshold, risk_threshold, positive_threshold):
  best_idx, best = max(
      enumerate(state["actions"]),
      key=lambda item: (
          float(item[1]["prob"]),
          float(item[1]["policy_prob"]),
      ),
  )
  if bool(best["is_base"]):
    return None
  if float(best["prob"]) < threshold:
    return None
  if risk_threshold is not None and float(best["safe_prob"]) < risk_threshold:
    return None
  if positive_threshold is not None and float(
      best["positive_prob"]
  ) < positive_threshold:
    return None
  row = dict(best)
  row["selected_index"] = int(best_idx)
  return row


def _threshold_report(states, thresholds, risk_thresholds, positive_thresholds):
  reports = []
  for threshold in thresholds:
    for risk_threshold in [None] + list(risk_thresholds):
      for positive_threshold in [None] + list(positive_thresholds):
        overrides = 0
        positives = 0
        safe = 0
        value_deltas = []
        risk_deltas = []
        for state in states:
          selected = _selected_action(
              state, threshold, risk_threshold, positive_threshold
          )
          if selected is None:
            continue
          overrides += 1
          positives += int(bool(selected["positive"]))
          safe += int(bool(selected["safe"]))
          value_deltas.append(float(selected["value_delta"]))
          risk_deltas.append(float(selected["risk_delta"]))
        precision = positives / overrides if overrides else None
        safe_rate = safe / overrides if overrides else None
        coverage = overrides / len(states) if states else None
        reports.append({
            "threshold": float(threshold),
            "risk_threshold": (
                None if risk_threshold is None else float(risk_threshold)
            ),
            "positive_threshold": (
                None
                if positive_threshold is None else
                float(positive_threshold)
            ),
            "states": int(len(states)),
            "overrides": int(overrides),
            "coverage": None if coverage is None else round(float(coverage), 6),
            "positive_precision": (
                None if precision is None else round(float(precision), 6)
            ),
            "safe_rate": None if safe_rate is None else round(float(safe_rate), 6),
            "mean_value_delta": (
                None if not value_deltas else round(float(np.mean(value_deltas)), 6)
            ),
            "mean_risk_delta": (
                None if not risk_deltas else round(float(np.mean(risk_deltas)), 6)
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


def _gate_eligible_reports(reports, args):
  eligible = []
  for report in reports:
    if int(report.get("overrides") or 0) < int(
        args.min_validation_overrides_for_live
    ):
      continue
    if report.get("positive_precision") is None or float(
        report["positive_precision"]
    ) < float(args.min_validation_positive_precision_for_live):
      continue
    if report.get("safe_rate") is None or float(report["safe_rate"]) < float(
        args.min_validation_safe_rate_for_live
    ):
      continue
    if report.get("mean_value_delta") is None or float(
        report["mean_value_delta"]
    ) < float(args.min_validation_mean_value_delta_for_live):
      continue
    if report.get("mean_risk_delta") is None or float(
        report["mean_risk_delta"]
    ) > float(args.max_validation_mean_risk_delta_for_live):
      continue
    eligible.append(report)
  return eligible


def _state_summary(states):
  action_count = sum(len(state["actions"]) for state in states)
  positive_states = sum(int(bool(state["has_positive"])) for state in states)
  positive_targets = sum(int(not bool(state["target_is_base"])) for state in states)
  return {
      "states": int(len(states)),
      "actions": int(action_count),
      "positive_states": int(positive_states),
      "positive_targets": int(positive_targets),
      "base_targets": int(len(states) - positive_targets),
      "mean_actions_per_state": round(action_count / max(1, len(states)), 6),
  }


def _state_summary_by_source(states):
  by_source: dict[int, list[dict]] = {}
  for state in states:
    source_idx = int(state.get("source_replay_index", -1))
    by_source.setdefault(source_idx, []).append(state)
  return {
      str(source_idx): _state_summary(source_states)
      for source_idx, source_states in sorted(by_source.items())
  }


def main():
  args = parse_args()
  rows, metadata_rows, source_reports = _load_replays_and_metadata(args)
  states, skipped_report = _build_states(args, rows, metadata_rows)
  train_states, validation_states, split_report = _split_states(args, states)
  mean, std = _standardize_state_features(train_states, validation_states, states)
  feature_size = len(states[0]["actions"][0]["feature_std"])
  model, train_report = _train(args, train_states, feature_size)
  _predict_states(model, states)

  thresholds = _parse_float_list(args.thresholds)
  risk_thresholds = _parse_float_list(args.risk_thresholds)
  positive_thresholds = _parse_float_list(args.positive_thresholds)
  train_thresholds = _threshold_report(
      train_states, thresholds, risk_thresholds, positive_thresholds
  )
  validation_thresholds = _threshold_report(
      validation_states, thresholds, risk_thresholds, positive_thresholds
  )
  all_thresholds = _threshold_report(
      states, thresholds, risk_thresholds, positive_thresholds
  )
  best_validation = _best_report(validation_thresholds)
  gate_eligible = _gate_eligible_reports(validation_thresholds, args)
  best_gate_eligible = _best_report(gate_eligible)
  live_screen_gate = {
      "passed": bool(best_gate_eligible is not None),
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
      "best_threshold": best_gate_eligible,
  }

  model_path = Path(args.model_out)
  model_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
      "model_state": model.state_dict(),
      "feature_mean": mean,
      "feature_std": std,
      "feature_size": int(feature_size),
      "model_class": "LegalSetSelector",
      "feature_schema_version": (
          "legal_set_pair_v1" if str(args.pair_feature_mode) == "full" else
          "legal_set_compact_action_delta_v1"
      ),
      "pair_feature_mode": str(args.pair_feature_mode),
      "observation_feature_mode": str(args.observation_feature_mode),
      "policy_context_mode": str(args.policy_context_mode),
      "policy_context_checkpoint": str(args.policy_context_checkpoint),
      "base_action_source": str(args.base_action_source),
      "target_mode": str(args.target_mode),
      "force_led_lane_burn_negative": bool(args.force_led_lane_burn_negative),
      "led_lane_red_exception": str(args.led_lane_red_exception),
      "replay_sources": source_reports,
      "args": vars(args),
      "train_report": train_report,
  }, model_path)

  result = {
      "replay": list(args.replay),
      "metadata": list(args.metadata or []),
      "replay_sources": source_reports,
      "model_out": str(model_path),
      "validation_split": str(args.validation_split),
      "validation_source_indices": str(args.validation_source_indices or ""),
      "split_report": split_report,
      "skipped": skipped_report,
      "feature_size": int(feature_size),
      "pair_feature_mode": str(args.pair_feature_mode),
      "positive_value_margin": float(args.positive_value_margin),
      "safe_risk_delta_max": float(args.safe_risk_delta_max),
      "target_mode": str(args.target_mode),
      "force_led_lane_burn_negative": bool(args.force_led_lane_burn_negative),
      "led_lane_red_exception": str(args.led_lane_red_exception),
      "train_summary": _state_summary(train_states),
      "validation_summary": _state_summary(validation_states),
      "all_summary": _state_summary(states),
      "train_summary_by_source": _state_summary_by_source(train_states),
      "validation_summary_by_source": _state_summary_by_source(validation_states),
      "all_summary_by_source": _state_summary_by_source(states),
      "train_report": train_report,
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "all_thresholds": all_thresholds,
      "best_validation_threshold": best_validation,
      "live_screen_gate": live_screen_gate,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
