#!/usr/bin/env python3
"""Train a deployable improve_graft scorer from dense table-liveness labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_PREFIX_SIZE,
    APPENDED_ACTION_FEATURE_INDEX,
    load_replay,
)
from quantum_cat_calibration_transfer_audit import (
    ImproveMLP,
    _load_policy_context_model,
    _policy_probs_from_row,
    _standardize,
    _train_model,
    pair_feature_vector_from_matrix,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", required=True)
  parser.add_argument(
      "--metadata",
      default="",
      help=(
          "Optional replay metadata JSON. Required when --base-action-source "
          "uses metadata; defaults to REPLAY + '.metadata.json'."
      ),
  )
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--seed", type=int, default=20297811)
  parser.add_argument("--validation-fraction", type=float, default=0.25)
  parser.add_argument(
      "--validation-split",
      choices=("state", "metadata_group"),
      default="state",
      help=(
          "state preserves the historical random state split. metadata_group "
          "keeps live-substitution rows from the same match/round together."
      ),
  )
  parser.add_argument("--epochs", type=int, default=240)
  parser.add_argument("--lr", type=float, default=0.002)
  parser.add_argument("--hidden", type=int, default=128)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument(
      "--positive-value-margin",
      type=float,
      default=0.08,
      help="Candidate value delta over raw policy action required for POSITIVE.",
  )
  parser.add_argument(
      "--safe-risk-delta-max",
      type=float,
      default=0.0,
      help="Candidate risk delta over raw policy action allowed for safety.",
  )
  parser.add_argument(
      "--negative-value-margin",
      type=float,
      default=0.0,
      help=(
          "Candidates at or below this value delta are hard negatives. "
          "Candidates between negative and positive margins remain negatives "
          "but are reported as NEAR_POSITIVE."
      ),
  )
  parser.add_argument(
      "--max-candidates-per-state",
      type=int,
      default=0,
      help=(
          "Optional deterministic cap after keeping policy top, value best, "
          "risk best, and high absolute value-delta candidates."
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
      "--base-action-source",
      choices=(
          "policy_context_argmax",
          "metadata_policy_best",
          "metadata_selected_action",
      ),
      default="policy_context_argmax",
      help=(
          "Action to compare candidates against. metadata_policy_best is for "
          "live-substitution replays collected under a shielded behavior "
          "policy, where metadata policy_best_action is the deployed base. "
          "metadata_selected_action is for miner rows that store the deployed "
          "base action as selected_action."
      ),
  )
  parser.add_argument(
      "--thresholds",
      default="0.50,0.60,0.70,0.80,0.85,0.90,0.93,0.95,0.97,0.99",
  )
  parser.add_argument(
      "--risk-thresholds",
      default="0.50,0.70,0.85,0.90,0.95,0.97,0.99",
  )
  parser.add_argument(
      "--live-screen-trick-numbers",
      default="",
      help=(
          "Optional comma-separated trick numbers for a secondary live-screen "
          "gate/report. This lets a selector be validated only on the trick "
          "strata where it would be deployed."
      ),
  )
  parser.add_argument(
      "--min-validation-overrides-for-live",
      type=int,
      default=0,
      help=(
          "Offline live-screen gate: require at least this many held-out "
          "overrides for a threshold to be considered gate-eligible."
      ),
  )
  parser.add_argument(
      "--min-validation-positive-precision-for-live",
      type=float,
      default=0.95,
      help="Offline live-screen gate precision requirement.",
  )
  parser.add_argument(
      "--min-validation-safe-rate-for-live",
      type=float,
      default=0.95,
      help="Offline live-screen gate safe-rate requirement.",
  )
  parser.add_argument(
      "--min-validation-mean-value-delta-for-live",
      type=float,
      default=0.0,
      help="Offline live-screen gate mean value-delta requirement.",
  )
  parser.add_argument(
      "--max-validation-mean-risk-delta-for-live",
      type=float,
      default=0.0,
      help="Offline live-screen gate maximum mean risk-delta.",
  )
  parser.add_argument(
      "--pos-weight-mode",
      choices=("balanced", "none"),
      default="none",
  )
  parser.add_argument(
      "--training-objective",
      choices=(
          "bce",
          "state_ce",
          "bce_state_ce",
          "pairwise_margin",
          "bce_pairwise_margin",
      ),
      default="bce",
      help=(
          "bce preserves the historical independent-row classifier. "
          "state_ce trains a same-state argmax objective. bce_state_ce combines "
          "both for calibrated hard-negative ranking. pairwise_margin ranks "
          "same-state positives above hard-negative lookalikes."
      ),
  )
  parser.add_argument(
      "--state-ce-loss-weight",
      type=float,
      default=1.0,
      help="Multiplier for the same-state cross-entropy loss.",
  )
  parser.add_argument(
      "--state-pairwise-loss-weight",
      type=float,
      default=1.0,
      help="Multiplier for same-state positive-vs-hard-negative margin loss.",
  )
  parser.add_argument(
      "--state-pairwise-margin",
      type=float,
      default=1.0,
      help="Logit margin required between positive and same-state negative.",
  )
  parser.add_argument(
      "--state-pairwise-negative-top-k",
      type=int,
      default=4,
      help=(
          "Per positive, compare against this many hardest same-state negatives "
          "by label value delta and policy probability. <=0 uses all negatives."
      ),
  )
  parser.add_argument("--positive-loss-weight", type=float, default=1.0)
  parser.add_argument("--negative-loss-weight", type=float, default=1.0)
  parser.add_argument("--false-positive-loss-weight", type=float, default=1.0)
  parser.add_argument("--near-positive-loss-weight", type=float, default=1.0)
  parser.add_argument(
      "--force-led-lane-burn-negative",
      action="store_true",
      help=(
          "Force off-led candidates that newly burn the led-color token to be "
          "negative when the base action follows led. This bakes the "
          "led-lane preservation rule into the labels instead of relying only "
          "on a deployment guard."
      ),
  )
  parser.add_argument(
      "--led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="urgent_win",
      help=(
          "Exception for forced led-lane negatives. win keeps any red switch "
          "that would win the current trick; urgent_win keeps only red wins "
          "that are not over-target and are needed to hit prediction."
      ),
  )
  parser.add_argument(
      "--forced-led-lane-negative-weight",
      type=float,
      default=1.0,
      help="Extra sample-weight multiplier for forced led-lane negatives.",
  )
  parser.add_argument("--focal-gamma", type=float, default=0.0)
  return parser.parse_args()


def _parse_float_list(spec: str) -> list[float]:
  return [float(part.strip()) for part in str(spec).split(",") if part.strip()]


def _parse_int_set(spec: str) -> set[int]:
  return {int(part.strip()) for part in str(spec).split(",") if part.strip()}


def _choose_candidates(
    legal: np.ndarray,
    policy_action: int,
    values: np.ndarray,
    risks: np.ndarray,
    max_candidates: int,
) -> list[int]:
  candidates = [int(action) for action in legal if int(action) != policy_action]
  if max_candidates <= 0 or len(candidates) <= max_candidates:
    return sorted(candidates)
  kept: list[int] = []

  def add(action):
    action = int(action)
    if action != policy_action and action in candidates and action not in kept:
      kept.append(action)

  add(max(candidates, key=lambda action: float(values[action])))
  add(min(candidates, key=lambda action: float(risks[action])))
  policy_value = float(values[policy_action])
  for action in sorted(
      candidates,
      key=lambda item: abs(float(values[item]) - policy_value),
      reverse=True,
  ):
    add(action)
    if len(kept) >= max_candidates:
      break
  return sorted(kept[:max_candidates])


def _split_state_ids(state_ids: list[int], validation_fraction: float, seed: int):
  rng = random.Random(seed)
  indices = list(state_ids)
  rng.shuffle(indices)
  if len(indices) <= 1:
    return indices, indices
  val_count = max(1, int(round(len(indices) * validation_fraction)))
  val_count = min(val_count, len(indices) - 1)
  validation = set(indices[:val_count])
  train = [idx for idx in indices if idx not in validation]
  return train, sorted(validation)


def _metadata_group_key(meta: dict, fallback_index: int) -> str:
  row_group_id = meta.get("row_group_id")
  if row_group_id is not None:
    return f"row_group_id={row_group_id}"
  group = meta.get("group")
  if not isinstance(group, dict):
    pieces = ["group"]
    for key in ("match_index", "round_index", "start_player"):
      if key in meta:
        pieces.append(f"{key}={meta[key]}")
    if len(pieces) > 1:
      return "|".join(pieces)
    return f"row={fallback_index}"
  pieces = [
      "group",
      str(group.get("proposal_source", "")),
  ]
  for key in ("match_index", "round_index", "start_player"):
    if key in group:
      pieces.append(f"{key}={group[key]}")
  if len(pieces) <= 2:
    return f"row={fallback_index}"
  return "|".join(pieces)


def _split_state_ids_by_group(
    states: list[dict], validation_fraction: float, seed: int
):
  rng = random.Random(seed)
  group_state_ids: dict[str, list[int]] = {}
  for state in states:
    group_state_ids.setdefault(str(state["group_key"]), []).append(
        int(state["state_index"])
    )
  groups = list(group_state_ids)
  rng.shuffle(groups)
  if len(groups) <= 1:
    state_ids = [int(state["state_index"]) for state in states]
    return state_ids, state_ids, {
        "split": "metadata_group",
        "groups": int(len(groups)),
        "train_groups": int(len(groups)),
        "validation_groups": int(len(groups)),
    }
  val_count = max(1, int(round(len(groups) * validation_fraction)))
  val_count = min(val_count, len(groups) - 1)
  validation_groups = set(groups[:val_count])
  train_groups = set(groups[val_count:])
  validation_ids = []
  train_ids = []
  for group, ids in group_state_ids.items():
    if group in validation_groups:
      validation_ids.extend(ids)
    else:
      train_ids.extend(ids)
  return sorted(train_ids), sorted(validation_ids), {
      "split": "metadata_group",
      "groups": int(len(groups)),
      "train_groups": int(len(train_groups)),
      "validation_groups": int(len(validation_groups)),
  }


def _metadata_path_for(replay_path: str) -> Path:
  replay = Path(replay_path)
  return replay.with_suffix(replay.suffix + ".metadata.json")


def _load_metadata_rows(path: Path) -> list[dict]:
  data = json.loads(path.read_text())
  if isinstance(data, list):
    rows = data
  elif isinstance(data.get("row_metadata"), list):
    rows = data["row_metadata"]
  else:
    rows = data.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"metadata rows not found in {path}")
  return rows


def _feature_value(action_features: np.ndarray, action: int, feature_idx: int) -> float:
  if action < 0 or action >= action_features.shape[0]:
    return 0.0
  if feature_idx < 0 or feature_idx >= action_features.shape[1]:
    return 0.0
  return float(action_features[int(action), int(feature_idx)])


def _feature_bool(
    action_features: np.ndarray,
    action: int,
    feature_idx: int,
    *,
    threshold: float = 0.5,
) -> bool:
  return bool(
      _feature_value(action_features, int(action), int(feature_idx)) > threshold
  )


def _urgent_red_win_exception(
    action_features: np.ndarray,
    candidate: int,
    red_exception_mode: str,
) -> bool:
  if str(red_exception_mode) == "none":
    return False
  if not _feature_bool(action_features, candidate, ACTION_FEATURE_IS_RED_INDEX):
    return False
  if str(red_exception_mode) == "red":
    return True
  would_win_idx = ACTION_FEATURE_PREFIX_SIZE + 19
  if not _feature_bool(action_features, candidate, would_win_idx):
    return False
  if str(red_exception_mode) == "win":
    return True
  over_target = (
      _feature_bool(
          action_features,
          candidate,
          APPENDED_ACTION_FEATURE_INDEX["exit_over_target_would_win"],
      )
      or _feature_bool(
          action_features,
          candidate,
          APPENDED_ACTION_FEATURE_INDEX["overshoots_prediction"],
      )
  )
  if over_target:
    return False
  return bool(
      _feature_bool(
          action_features,
          candidate,
          APPENDED_ACTION_FEATURE_INDEX["hits_prediction"],
      )
      or _feature_bool(
          action_features,
          candidate,
          APPENDED_ACTION_FEATURE_INDEX["must_win_all_remaining_after"],
      )
  )


def _led_lane_burn_label_shape(
    action_features: np.ndarray,
    policy_action: int,
    candidate: int,
    args,
) -> tuple[bool, bool]:
  """Returns (force_negative, kept_by_red_exception)."""
  if not bool(getattr(args, "force_led_lane_burn_negative", False)):
    return False, False
  base_follows_led = _feature_bool(
      action_features, policy_action, ACTION_FEATURE_FOLLOWS_LED_INDEX
  )
  if not base_follows_led:
    return False, False
  candidate_follows_led = _feature_bool(
      action_features, candidate, ACTION_FEATURE_FOLLOWS_LED_INDEX
  )
  if candidate_follows_led:
    return False, False
  burns_led_token = _feature_bool(
      action_features,
      candidate,
      APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"],
  )
  if not burns_led_token:
    return False, False
  if _urgent_red_win_exception(
      action_features, candidate, str(getattr(args, "led_lane_red_exception", ""))
  ):
    return False, True
  return True, False


def _selector_sample_weights(args, y_train, examples):
  weights = []
  for label, example in zip(y_train, examples):
    if float(label) > 0.5:
      weight = float(args.positive_loss_weight)
    else:
      weight = float(args.negative_loss_weight) * float(
          args.false_positive_loss_weight
      )
      if str(example.get("class")) == "NEAR_POSITIVE":
        weight *= float(args.near_positive_loss_weight)
      if bool(example.get("forced_led_lane_negative", False)):
        weight *= float(getattr(args, "forced_led_lane_negative_weight", 1.0))
    weights.append(weight)
  return np.asarray(weights, dtype=np.float32)


def _state_ce_targets(examples):
  grouped: dict[int, list[int]] = {}
  for local_idx, example in enumerate(examples):
    grouped.setdefault(int(example["state_index"]), []).append(int(local_idx))
  targets = []
  for rows in grouped.values():
    positives = [idx for idx in rows if bool(examples[idx]["positive"])]
    if not positives or len(rows) < 2:
      continue
    target = max(
        positives,
        key=lambda idx: (
            float(examples[idx]["value_delta"]),
            -float(examples[idx]["risk_delta"]),
            float(examples[idx]["policy_prob"]),
        ),
    )
    targets.append((rows, rows.index(target)))
  return targets


def _state_pairwise_targets(examples, negative_top_k: int):
  grouped: dict[int, list[int]] = {}
  for local_idx, example in enumerate(examples):
    grouped.setdefault(int(example["state_index"]), []).append(int(local_idx))
  pairs = []
  for rows in grouped.values():
    positives = [idx for idx in rows if bool(examples[idx]["positive"])]
    negatives = [idx for idx in rows if not bool(examples[idx]["positive"])]
    if not positives or not negatives:
      continue
    negatives = sorted(
        negatives,
        key=lambda idx: (
            float(examples[idx]["value_delta"]),
            float(examples[idx]["policy_prob"]),
            -float(examples[idx]["risk_delta"]),
        ),
        reverse=True,
    )
    if negative_top_k > 0:
      negatives = negatives[:negative_top_k]
    for pos_idx in positives:
      for neg_idx in negatives:
        pairs.append((int(pos_idx), int(neg_idx)))
  return pairs


def _train_state_ce_model(args, x_train, y_train, examples):
  torch.manual_seed(int(args.seed))
  model = ImproveMLP(x_train.shape[1], max(1, int(args.hidden)))
  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=float(args.lr),
      weight_decay=float(args.weight_decay),
  )
  x_t = torch.tensor(x_train, dtype=torch.float32)
  y_t = torch.tensor(y_train, dtype=torch.float32)
  positives = float(torch.sum(y_t).item())
  negatives = float(len(y_train) - positives)
  if (
      str(getattr(args, "pos_weight_mode", "balanced")) == "balanced"
      and positives > 0
      and negatives > 0
  ):
    pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32)
  else:
    pos_weight = torch.tensor([1.0], dtype=torch.float32)
  weights_t = torch.tensor(
      _selector_sample_weights(args, y_train, examples), dtype=torch.float32
  )
  weight_mean = torch.mean(weights_t)
  if float(weight_mean.item()) > 0:
    weights_t = weights_t / weight_mean

  objective = str(args.training_objective)
  use_bce = objective in ("bce", "bce_state_ce")
  use_state_ce = objective in ("state_ce", "bce_state_ce")
  use_pairwise = objective in ("pairwise_margin", "bce_pairwise_margin")
  if objective == "bce_pairwise_margin":
    use_bce = True
  state_targets = _state_ce_targets(examples) if use_state_ce else []
  if use_state_ce and not state_targets:
    raise ValueError("state_ce objective needs at least one state with positives")
  state_index_tensors = [
      (torch.tensor(rows, dtype=torch.long), torch.tensor([target], dtype=torch.long))
      for rows, target in state_targets
  ]
  pairwise_targets = (
      _state_pairwise_targets(
          examples, int(getattr(args, "state_pairwise_negative_top_k", 4))
      )
      if use_pairwise else
      []
  )
  if use_pairwise and not pairwise_targets:
    raise ValueError("pairwise_margin objective needs positive/negative pairs")
  pairwise_pos_t = torch.tensor(
      [pos for pos, _ in pairwise_targets], dtype=torch.long
  )
  pairwise_neg_t = torch.tensor(
      [neg for _, neg in pairwise_targets], dtype=torch.long
  )

  losses = []
  bce_losses = []
  state_losses = []
  pairwise_losses = []
  for _ in range(max(1, int(args.epochs))):
    logits = model(x_t)
    loss_parts = []
    bce_loss = None
    if use_bce:
      loss_items = F.binary_cross_entropy_with_logits(
          logits, y_t, pos_weight=pos_weight, reduction="none"
      )
      gamma = max(0.0, float(getattr(args, "focal_gamma", 0.0)))
      if gamma > 0:
        probs = torch.sigmoid(logits)
        p_t = torch.where(y_t > 0.5, probs, 1.0 - probs)
        loss_items = loss_items * torch.pow(torch.clamp(1.0 - p_t, min=0.0), gamma)
      bce_loss = torch.mean(loss_items * weights_t)
      loss_parts.append(bce_loss)
    state_ce_loss = None
    if use_state_ce:
      per_state_losses = []
      for rows_t, target_t in state_index_tensors:
        per_state_losses.append(
            F.cross_entropy(logits[rows_t].unsqueeze(0), target_t)
        )
      state_ce_loss = torch.mean(torch.stack(per_state_losses))
      loss_parts.append(float(args.state_ce_loss_weight) * state_ce_loss)
    pairwise_loss = None
    if use_pairwise:
      margin = float(getattr(args, "state_pairwise_margin", 1.0))
      pairwise_loss = torch.mean(
          F.softplus(margin - (logits[pairwise_pos_t] - logits[pairwise_neg_t]))
      )
      loss_parts.append(float(args.state_pairwise_loss_weight) * pairwise_loss)
    loss = torch.sum(torch.stack(loss_parts))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
    bce_losses.append(None if bce_loss is None else float(bce_loss.detach()))
    state_losses.append(
        None if state_ce_loss is None else float(state_ce_loss.detach())
    )
    pairwise_losses.append(
        None if pairwise_loss is None else float(pairwise_loss.detach())
    )
  bce_non_null = [loss for loss in bce_losses if loss is not None]
  state_non_null = [loss for loss in state_losses if loss is not None]
  pairwise_non_null = [loss for loss in pairwise_losses if loss is not None]
  return model, {
      "initial_loss": round(losses[0], 6),
      "final_loss": round(losses[-1], 6),
      "initial_bce_loss": (
          None if not bce_non_null else round(float(bce_non_null[0]), 6)
      ),
      "final_bce_loss": (
          None if not bce_non_null else round(float(bce_non_null[-1]), 6)
      ),
      "initial_state_ce_loss": (
          None if not state_non_null else round(float(state_non_null[0]), 6)
      ),
      "final_state_ce_loss": (
          None if not state_non_null else round(float(state_non_null[-1]), 6)
      ),
      "initial_pairwise_margin_loss": (
          None if not pairwise_non_null else round(float(pairwise_non_null[0]), 6)
      ),
      "final_pairwise_margin_loss": (
          None if not pairwise_non_null else round(float(pairwise_non_null[-1]), 6)
      ),
      "state_ce_groups": int(len(state_targets)),
      "state_ce_loss_weight": float(args.state_ce_loss_weight),
      "state_pairwise_pairs": int(len(pairwise_targets)),
      "state_pairwise_loss_weight": float(args.state_pairwise_loss_weight),
      "state_pairwise_margin": float(args.state_pairwise_margin),
      "state_pairwise_negative_top_k": int(args.state_pairwise_negative_top_k),
      "training_objective": objective,
      "epochs": int(args.epochs),
      "lr": float(args.lr),
      "hidden": int(args.hidden),
      "pos_weight": round(float(pos_weight.item()), 6),
      "pos_weight_mode": str(getattr(args, "pos_weight_mode", "balanced")),
      "positive_loss_weight": float(getattr(args, "positive_loss_weight", 1.0)),
      "negative_loss_weight": float(getattr(args, "negative_loss_weight", 1.0)),
      "false_positive_loss_weight": float(
          getattr(args, "false_positive_loss_weight", 1.0)
      ),
      "near_positive_loss_weight": float(
          getattr(args, "near_positive_loss_weight", 1.0)
      ),
      "focal_gamma": float(getattr(args, "focal_gamma", 0.0)),
      "sample_weight_min": round(float(torch.min(weights_t).item()), 6),
      "sample_weight_mean": round(float(torch.mean(weights_t).item()), 6),
      "sample_weight_max": round(float(torch.max(weights_t).item()), 6),
  }


def _train(args, x_train, y_train, examples=None, force_bce=False):
  objective = "bce" if force_bce else str(args.training_objective)
  if objective != "bce":
    if examples is None:
      raise ValueError("examples are required for state-aware training")
    return _train_state_ce_model(args, x_train, y_train, examples)
  if examples is None or force_bce:
    sample_weights = np.where(
        y_train > 0.5,
        float(args.positive_loss_weight),
        float(args.negative_loss_weight),
    ).astype(np.float32)
  else:
    sample_weights = _selector_sample_weights(args, y_train, examples)
  return _train_model(
      x_train,
      y_train,
      args,
      sample_weights=sample_weights,
  )


def _predict(model, x):
  with torch.no_grad():
    logits = model(torch.tensor(x, dtype=torch.float32))
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def _threshold_report(states, thresholds, risk_thresholds):
  def summarize_bucket(bucket, considered):
    overrides = int(bucket.get("overrides", 0))
    positives = int(bucket.get("positives", 0))
    safe = int(bucket.get("safe", 0))
    value_deltas = bucket.get("value_deltas", [])
    risk_deltas = bucket.get("risk_deltas", [])
    return {
        "states": int(considered),
        "overrides": overrides,
        "coverage": (
            None if not considered else round(float(overrides / considered), 6)
        ),
        "positive_precision": (
            None if not overrides else round(float(positives / overrides), 6)
        ),
        "safe_rate": (
            None if not overrides else round(float(safe / overrides), 6)
        ),
        "mean_value_delta": (
            None if not value_deltas else round(float(np.mean(value_deltas)), 6)
        ),
        "mean_risk_delta": (
            None if not risk_deltas else round(float(np.mean(risk_deltas)), 6)
        ),
    }

  reports = []
  for threshold in thresholds:
    for risk_threshold in [None] + list(risk_thresholds):
      considered = 0
      overrides = 0
      positives = 0
      safe = 0
      value_deltas = []
      risk_deltas = []
      considered_by_trick = {}
      selected_by_trick = {}
      for state in states:
        candidates = state["candidates"]
        if not candidates:
          continue
        considered += 1
        trick_key = str(int(state.get("trick_number", -1)))
        considered_by_trick[trick_key] = int(
            considered_by_trick.get(trick_key, 0)
        ) + 1
        best = max(
            candidates,
            key=lambda row: (float(row["prob"]), float(row["policy_prob"])),
        )
        if float(best["prob"]) < threshold:
          continue
        if risk_threshold is not None and float(best["safe_prob"]) < risk_threshold:
          continue
        overrides += 1
        positives += int(bool(best["positive"]))
        safe += int(bool(best["safe"]))
        value_deltas.append(float(best["value_delta"]))
        risk_deltas.append(float(best["risk_delta"]))
        bucket = selected_by_trick.setdefault(
            trick_key,
            {
                "overrides": 0,
                "positives": 0,
                "safe": 0,
                "value_deltas": [],
                "risk_deltas": [],
            },
        )
        bucket["overrides"] += 1
        bucket["positives"] += int(bool(best["positive"]))
        bucket["safe"] += int(bool(best["safe"]))
        bucket["value_deltas"].append(float(best["value_delta"]))
        bucket["risk_deltas"].append(float(best["risk_delta"]))
      precision = positives / overrides if overrides else None
      safe_rate = safe / overrides if overrides else None
      coverage = overrides / considered if considered else None
      by_trick = []
      for trick_key in sorted(considered_by_trick, key=lambda item: int(item)):
        row = summarize_bucket(
            selected_by_trick.get(trick_key, {}), considered_by_trick[trick_key]
        )
        row["trick_number"] = int(trick_key)
        by_trick.append(row)
      reports.append({
          "threshold": float(threshold),
          "risk_threshold": None if risk_threshold is None else float(risk_threshold),
          "states": int(considered),
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
          "by_trick": by_trick,
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
          row["coverage"] or -1.0,
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


def _live_screen_gate(reports, args):
  gate_eligible = _gate_eligible_reports(reports, args)
  best_gate_eligible = _best_report(gate_eligible)
  return {
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


def main():
  args = parse_args()
  rows = load_replay(args.replay)
  if not rows:
    raise ValueError(f"no replay rows loaded from {args.replay}")
  metadata_rows = None
  if str(args.base_action_source) in (
      "metadata_policy_best",
      "metadata_selected_action",
  ):
    metadata_path = Path(args.metadata) if args.metadata else _metadata_path_for(
        args.replay
    )
    metadata_rows = _load_metadata_rows(metadata_path)
    if len(metadata_rows) != len(rows):
      raise ValueError(
          f"replay/metadata row mismatch: {len(rows)} vs {len(metadata_rows)}"
      )
  policy_model, policy_device = _load_policy_context_model(args, rows[0])
  examples = []
  states = []
  class_counts = {"POSITIVE": 0, "NEAR_POSITIVE": 0, "NEGATIVE": 0}
  label_shape_counts = {
      "led_lane_base_follow_candidates": 0,
      "led_lane_forced_negative": 0,
      "led_lane_red_exception_kept": 0,
      "led_lane_raw_positive_forced_negative": 0,
      "led_lane_raw_safe_forced_unsafe": 0,
  }
  skipped = 0
  skipped_base_action = 0
  for state_idx, row in enumerate(rows):
    legal_mask = np.asarray(row[1], dtype=np.float32) > 0.5
    value_mask = np.asarray(row[10], dtype=np.float32) > 0.5
    risk_mask = np.asarray(row[8], dtype=np.float32) > 0.5
    labeled_mask = legal_mask & value_mask & risk_mask
    legal = np.flatnonzero(labeled_mask)
    if len(legal) < 2:
      skipped += 1
      continue
    policy = _policy_probs_from_row(row, policy_model, policy_device)
    if policy is None or float(np.sum(policy[legal])) <= 0.0:
      skipped += 1
      continue
    if str(args.base_action_source) in (
        "metadata_policy_best",
        "metadata_selected_action",
    ):
      assert metadata_rows is not None
      meta = metadata_rows[state_idx]
      if str(args.base_action_source) == "metadata_policy_best":
        policy_action = int(meta.get("policy_best_action", -1))
      else:
        policy_action = int(meta.get("selected_action", -1))
      if policy_action not in set(int(action) for action in legal):
        skipped += 1
        skipped_base_action += 1
        continue
    else:
      meta = None
      policy_action = int(max(legal, key=lambda action: float(policy[action])))
    values = np.asarray(row[9], dtype=np.float32)
    risks = np.asarray(row[7], dtype=np.float32)
    policy_value = float(values[policy_action])
    policy_risk = float(risks[policy_action])
    candidates = _choose_candidates(
        legal,
        policy_action,
        values,
        risks,
        int(args.max_candidates_per_state),
    )
    action_features = np.asarray(row[11], dtype=np.float32)
    state_record = {
        "state_index": int(state_idx),
        "policy_action": int(policy_action),
        "policy_value": float(policy_value),
        "policy_risk": float(policy_risk),
        "trick_number": int(meta.get("trick_number", -1)) if meta is not None else -1,
        "group_key": (
            _metadata_group_key(meta, state_idx)
            if meta is not None else
            f"row={state_idx}"
        ),
        "candidates": [],
    }
    for candidate in candidates:
      value_delta = float(values[candidate]) - policy_value
      risk_delta = float(risks[candidate]) - policy_risk
      raw_positive = (
          value_delta >= float(args.positive_value_margin)
          and risk_delta <= float(args.safe_risk_delta_max)
      )
      raw_safe = bool(risk_delta <= float(args.safe_risk_delta_max))
      positive = bool(raw_positive)
      safe = bool(raw_safe)
      forced_led_lane_negative, kept_by_red_exception = _led_lane_burn_label_shape(
          action_features, int(policy_action), int(candidate), args
      )
      if bool(getattr(args, "force_led_lane_burn_negative", False)) and _feature_bool(
          action_features, int(policy_action), ACTION_FEATURE_FOLLOWS_LED_INDEX
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
      elif value_delta > float(args.negative_value_margin):
        row_class = "NEAR_POSITIVE"
      else:
        row_class = "NEGATIVE"
      if forced_led_lane_negative:
        row_class = "FORCED_LED_LANE_BURN_NEGATIVE"
      class_counts[row_class] = int(class_counts.get(row_class, 0)) + 1
      feature = pair_feature_vector_from_matrix(
          action_features,
          legal_mask.astype(np.float32),
          int(candidate),
          int(policy_action),
          target_action=int(candidate),
          label_margin=0.0,
          legal_count=int(len(legal)),
          candidate_count=int(len(candidates)),
          include_label_margin=False,
          observation=row[0],
          observation_feature_mode=str(args.observation_feature_mode),
          policy_context=policy,
          policy_context_mode=str(args.policy_context_mode),
          include_target_match_features=False,
      )
      example = {
          "state_index": int(state_idx),
          "candidate_action": int(candidate),
          "policy_action": int(policy_action),
          "policy_prob": float(policy[candidate]),
          "value_delta": float(value_delta),
          "risk_delta": float(risk_delta),
          "positive": bool(positive),
          "safe": bool(safe),
          "raw_positive": bool(raw_positive),
          "raw_safe": bool(raw_safe),
          "forced_led_lane_negative": bool(forced_led_lane_negative),
          "kept_by_led_lane_red_exception": bool(kept_by_red_exception),
          "class": row_class,
          "feature": feature,
      }
      examples.append(example)
      state_record["candidates"].append(example)
    states.append(state_record)
  if not examples:
    raise ValueError("no candidate-pair examples built")
  if (
      str(args.validation_split) == "metadata_group"
      and metadata_rows is not None
  ):
    train_state_indices, val_state_indices, split_report = _split_state_ids_by_group(
        states, float(args.validation_fraction), int(args.seed)
    )
  else:
    state_ids = [int(state["state_index"]) for state in states]
    train_state_indices, val_state_indices = _split_state_ids(
        state_ids, float(args.validation_fraction), int(args.seed)
    )
    split_report = {
        "split": "state",
        "groups": None,
        "train_groups": None,
        "validation_groups": None,
    }
  train_state_set = set(train_state_indices)
  val_state_set = set(val_state_indices)
  train_indices = [
      idx for idx, example in enumerate(examples)
      if int(example["state_index"]) in train_state_set
  ]
  val_indices = [
      idx for idx, example in enumerate(examples)
      if int(example["state_index"]) in val_state_set
  ]
  if not train_indices or not val_indices:
    raise ValueError("empty train/validation split")
  x_all = np.asarray([example["feature"] for example in examples], dtype=np.float32)
  y_all = np.asarray([float(example["positive"]) for example in examples], dtype=np.float32)
  safe_all = np.asarray([float(example["safe"]) for example in examples], dtype=np.float32)
  standardized, mean, std = _standardize(
      x_all[train_indices], x_all[train_indices], x_all[val_indices], x_all
  )
  x_train, x_val, x_all_std = standardized
  y_train = y_all[train_indices]
  train_examples = [examples[idx] for idx in train_indices]
  model, train_report = _train(args, x_train, y_train, train_examples)
  risk_model = None
  risk_train_report = None
  if np.any(safe_all[train_indices] > 0.5) and np.any(safe_all[train_indices] <= 0.5):
    risk_model, risk_train_report = _train(
        args, x_train, safe_all[train_indices], force_bce=True
    )
  probs_all = _predict(model, x_all_std)
  safe_probs_all = (
      _predict(risk_model, x_all_std)
      if risk_model is not None else
      np.ones_like(probs_all, dtype=np.float32)
  )
  for idx, example in enumerate(examples):
    example["prob"] = float(probs_all[idx])
    example["safe_prob"] = float(safe_probs_all[idx])
  train_states = [
      state for state in states if int(state["state_index"]) in train_state_set
  ]
  val_states = [
      state for state in states if int(state["state_index"]) in val_state_set
  ]
  thresholds = _parse_float_list(args.thresholds)
  risk_thresholds = _parse_float_list(args.risk_thresholds)
  train_thresholds = _threshold_report(train_states, thresholds, risk_thresholds)
  validation_thresholds = _threshold_report(val_states, thresholds, risk_thresholds)
  all_thresholds = _threshold_report(states, thresholds, risk_thresholds)
  best_validation = _best_report(validation_thresholds)
  live_screen_gate = _live_screen_gate(validation_thresholds, args)
  live_screen_tricks = _parse_int_set(args.live_screen_trick_numbers)
  train_thresholds_trick_filtered = None
  validation_thresholds_trick_filtered = None
  all_thresholds_trick_filtered = None
  best_validation_trick_filtered = None
  live_screen_gate_trick_filtered = None
  if live_screen_tricks:
    train_states_filtered = [
        state for state in train_states
        if int(state.get("trick_number", -1)) in live_screen_tricks
    ]
    val_states_filtered = [
        state for state in val_states
        if int(state.get("trick_number", -1)) in live_screen_tricks
    ]
    all_states_filtered = [
        state for state in states
        if int(state.get("trick_number", -1)) in live_screen_tricks
    ]
    train_thresholds_trick_filtered = _threshold_report(
        train_states_filtered, thresholds, risk_thresholds
    )
    validation_thresholds_trick_filtered = _threshold_report(
        val_states_filtered, thresholds, risk_thresholds
    )
    all_thresholds_trick_filtered = _threshold_report(
        all_states_filtered, thresholds, risk_thresholds
    )
    best_validation_trick_filtered = _best_report(
        validation_thresholds_trick_filtered
    )
    live_screen_gate_trick_filtered = _live_screen_gate(
        validation_thresholds_trick_filtered, args
    )
    live_screen_gate_trick_filtered["trick_numbers"] = sorted(live_screen_tricks)
    live_screen_gate_trick_filtered["validation_states"] = int(
        len(val_states_filtered)
    )
    live_screen_gate_trick_filtered["train_states"] = int(len(train_states_filtered))

  model_path = Path(args.model_out)
  model_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
      "model_state": model.state_dict(),
      "risk_model_state": None if risk_model is None else risk_model.state_dict(),
      "feature_mean": mean.astype(np.float32),
      "feature_std": std.astype(np.float32),
      "feature_size": int(x_all.shape[1]),
      "feature_schema_version": "deployable_pair_v2",
      "deployable_feature_schema": True,
      "include_target_match_features": False,
      "observation_feature_mode": str(args.observation_feature_mode),
      "policy_context_mode": str(args.policy_context_mode),
      "policy_context_checkpoint": str(args.policy_context_checkpoint),
      "base_action_source": str(args.base_action_source),
      "args": vars(args),
      "risk_train_report": risk_train_report,
      "risk_label_mode": "candidate_safe_over_policy",
      "risk_target_paradox_delta_max": float(args.safe_risk_delta_max),
  }, model_path)
  result = {
      "replay": str(args.replay),
      "model_out": str(model_path),
      "states": int(len(states)),
      "skipped_states": int(skipped),
      "skipped_base_action_states": int(skipped_base_action),
      "candidate_examples": int(len(examples)),
      "train_states": int(len(train_states)),
      "validation_states": int(len(val_states)),
      "train_examples": int(len(train_indices)),
      "validation_examples": int(len(val_indices)),
      "validation_split": str(args.validation_split),
      "split_report": split_report,
      "classes": {key: int(value) for key, value in sorted(class_counts.items())},
      "label_shape_counts": {
          key: int(value) for key, value in sorted(label_shape_counts.items())
      },
      "positive_value_margin": float(args.positive_value_margin),
      "safe_risk_delta_max": float(args.safe_risk_delta_max),
      "force_led_lane_burn_negative": bool(args.force_led_lane_burn_negative),
      "led_lane_red_exception": str(args.led_lane_red_exception),
      "forced_led_lane_negative_weight": float(
          args.forced_led_lane_negative_weight
      ),
      "feature_size": int(x_all.shape[1]),
      "observation_feature_mode": str(args.observation_feature_mode),
      "policy_context_mode": str(args.policy_context_mode),
      "policy_context_checkpoint": str(args.policy_context_checkpoint),
      "base_action_source": str(args.base_action_source),
      "train_report": train_report,
      "risk_train_report": risk_train_report,
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "all_thresholds": all_thresholds,
      "best_validation_threshold": best_validation,
      "live_screen_gate": live_screen_gate,
      "live_screen_trick_numbers": sorted(live_screen_tricks),
      "train_thresholds_trick_filtered": train_thresholds_trick_filtered,
      "validation_thresholds_trick_filtered": validation_thresholds_trick_filtered,
      "all_thresholds_trick_filtered": all_thresholds_trick_filtered,
      "best_validation_threshold_trick_filtered": best_validation_trick_filtered,
      "live_screen_gate_trick_filtered": live_screen_gate_trick_filtered,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
