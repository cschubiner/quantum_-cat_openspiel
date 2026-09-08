#!/usr/bin/env python3
"""Train a tiny residual selector on exact shield-veto behavior rows.

This keeps the champion checkpoint frozen.  The selector only learns an
action-set score from deployable features plus the frozen model's policy and
action-value outputs.  It is an offline gate: if it cannot learn when to abstain
versus follow the exact late-liveness shield, it is not worth adding runtime
support.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import (
    adapt_action_features,
    adapt_observation_batch,
    load_model_payload,
    load_replay,
    model_action_value_batch,
    model_input_size,
)
from quantum_cat_table_liveness_improve_trainer import _parse_float_list


class ResidualSelector(nn.Module):
  def __init__(self, feature_size: int, hidden: int, dropout: float):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(feature_size, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x).squeeze(-1)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-replay", required=True)
  parser.add_argument("--train-summary", required=True)
  parser.add_argument("--validation-replay", required=True)
  parser.add_argument("--validation-summary", required=True)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--seed", type=int, default=20261381)
  parser.add_argument("--epochs", type=int, default=220)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--hidden", type=int, default=128)
  parser.add_argument("--dropout", type=float, default=0.10)
  parser.add_argument(
      "--feature-profile",
      choices=("full", "scalar"),
      default="full",
      help=(
          "full uses scalar policy/Q context plus action feature comparisons; "
          "scalar uses only frozen policy/Q ranks, margins, and set summaries."
      ),
  )
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--weight-decay", type=float, default=0.001)
  parser.add_argument("--veto-state-weight", type=float, default=4.0)
  parser.add_argument("--margin-loss-weight", type=float, default=0.35)
  parser.add_argument("--score-margin", type=float, default=0.75)
  parser.add_argument(
      "--thresholds",
      default="-3,-2,-1,-0.5,0,0.25,0.5,0.75,1,1.5,2,3,4,5",
  )
  parser.add_argument("--min-validation-overrides-for-live", type=int, default=8)
  parser.add_argument("--min-validation-precision-for-live", type=float, default=0.85)
  parser.add_argument("--min-validation-no-veto-abstain-rate", type=float, default=0.90)
  parser.add_argument("--device", default="cpu")
  return parser.parse_args()


def _device(name: str) -> torch.device:
  if str(name) == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(str(name))


def _load_frozen_model(checkpoint: str, players: int, device: torch.device):
  args = SimpleNamespace(
      players=int(players),
      arch="action_mlp",
      width=256,
      depth=3,
      match_context=True,
      separate_action_value_encoder=True,
      separate_action_paradox_encoder=True,
  )
  game = pyspiel.load_game(
      "python_quantum_cat",
      {"players": int(players), "start_player": 0, "match_context": 1},
  )
  model, payload, saved_args = load_model_payload(checkpoint, game, args, device)
  model.eval()
  return model, payload, saved_args


def _masked_policy_and_q(model, rows, device: torch.device):
  obs = np.asarray([np.asarray(row[0], dtype=np.float32).reshape(-1) for row in rows])
  masks = np.asarray([np.asarray(row[1], dtype=np.float32) > 0.5 for row in rows])
  num_actions = masks.shape[1]
  action_features = np.asarray([
      adapt_action_features(row[11] if len(row) > 11 else None, num_actions)
      for row in rows
  ], dtype=np.float32)
  policies = []
  q_values = []
  batch_size = 128 if getattr(model, "arch", "mlp") in ("action_mlp", "action_attn") else 512
  with torch.no_grad():
    for start in range(0, len(rows), batch_size):
      end = min(len(rows), start + batch_size)
      obs_t = torch.tensor(obs[start:end], dtype=torch.float32, device=device)
      obs_t = adapt_observation_batch(obs_t, model_input_size(model))
      mask_t = torch.tensor(masks[start:end], dtype=torch.bool, device=device)
      feat_t = torch.tensor(action_features[start:end], dtype=torch.float32, device=device)
      logits, _ = model(obs_t, feat_t)
      logits = logits.masked_fill(~mask_t, -1e9)
      probs = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
      policies.append(probs)
      if hasattr(model, "action_value"):
        values = model_action_value_batch(model, obs_t, feat_t)
        values = values.detach().cpu().numpy().astype(np.float32)
      else:
        values = np.zeros((end - start, num_actions), dtype=np.float32)
      values = np.where(masks[start:end], values, np.nan)
      q_values.append(values)
  return np.concatenate(policies, axis=0), np.concatenate(q_values, axis=0), action_features


def _rank_norm(values: np.ndarray, legal: np.ndarray, action: int, high_good: bool = True) -> float:
  legal_values = values[legal]
  value = float(values[int(action)])
  if legal_values.size <= 1:
    return 0.0
  if high_good:
    rank = int(np.sum(legal_values > value))
  else:
    rank = int(np.sum(legal_values < value))
  return float(rank / max(1, legal_values.size - 1))


def _entropy_norm(probs: np.ndarray, legal: np.ndarray) -> float:
  vals = np.asarray(probs[legal], dtype=np.float64)
  vals = vals / max(1e-12, float(np.sum(vals)))
  if vals.size <= 1:
    return 0.0
  entropy = -float(np.sum(vals * np.log(np.maximum(vals, 1e-12))))
  return float(entropy / math.log(float(vals.size)))


def _state_features(
    row,
    policy: np.ndarray,
    q_values: np.ndarray,
    action_features: np.ndarray,
    stat: dict,
    source_index: int,
    state_index: int,
    feature_profile: str,
) -> dict | None:
  legal_mask = np.asarray(row[1], dtype=np.float32) > 0.5
  legal = np.flatnonzero(legal_mask).astype(np.int64)
  if legal.size < 2:
    return None
  base_action = int(stat.get("baseline_action", -1))
  target_action = int(stat.get("best_action", -1))
  if base_action not in set(int(x) for x in legal):
    return None
  if target_action not in set(int(x) for x in legal):
    return None
  legal_features = action_features[legal]
  feat_mean = legal_features.mean(axis=0)
  feat_std = legal_features.std(axis=0)
  feat_std = np.where(feat_std < 1e-6, 1.0, feat_std).astype(np.float32)
  feat_min = legal_features.min(axis=0)
  feat_max = legal_features.max(axis=0)
  base_features = action_features[base_action]
  base_policy = float(policy[base_action])
  base_q = float(q_values[base_action]) if np.isfinite(q_values[base_action]) else 0.0
  legal_policy = policy[legal]
  legal_q = q_values[legal]
  finite_q = np.where(np.isfinite(legal_q), legal_q, 0.0)
  q_mean = float(np.mean(finite_q))
  q_std = float(np.std(finite_q)) if float(np.std(finite_q)) > 1e-6 else 1.0
  q_top = float(np.max(finite_q))
  sorted_probs = np.sort(legal_policy)[::-1]
  top_prob = float(sorted_probs[0]) if sorted_probs.size else 0.0
  second_prob = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
  entropy = _entropy_norm(policy, legal)
  rows = []
  target_index = None
  base_index = None
  for idx, action in enumerate(legal):
    action = int(action)
    act_features = action_features[action]
    q = float(q_values[action]) if np.isfinite(q_values[action]) else 0.0
    prob = float(policy[action])
    scalar = np.asarray([
        float(action == base_action),
        float(action == int(np.argmax(np.where(legal_mask, policy, -1.0)))),
        prob,
        float(np.log(prob + 1e-8)),
        prob - base_policy,
        _rank_norm(policy, legal, action, high_good=True),
        q,
        q - base_q,
        q - q_top,
        (q - q_mean) / q_std,
        _rank_norm(np.where(np.isfinite(q_values), q_values, -1e9), legal, action, high_good=True),
        min(1.0, float(legal.size) / 20.0),
        entropy,
        top_prob,
        second_prob,
        top_prob - second_prob,
        base_policy,
        base_q,
    ], dtype=np.float32)
    if str(feature_profile) == "scalar":
      feature = scalar.astype(np.float32)
    else:
      feature = np.concatenate([
          scalar,
          act_features,
          base_features,
          act_features - base_features,
          feat_mean,
          feat_min,
          feat_max,
          (act_features - feat_mean) / feat_std,
      ]).astype(np.float32)
    if action == target_action:
      target_index = idx
    if action == base_action:
      base_index = idx
    rows.append({
        "action": action,
        "feature": feature,
        "policy_prob": prob,
        "q_value": q,
    })
  if target_index is None or base_index is None:
    return None
  return {
      "source_index": int(source_index),
      "state_index": int(state_index),
      "group_key": str(stat.get("group_key", f"s{source_index}:r{state_index}")),
      "base_action": int(base_action),
      "target_action": int(target_action),
      "target_is_base": bool(target_action == base_action),
      "target_index": int(target_index),
      "base_index": int(base_index),
      "actions": rows,
  }


def _load_states(
    replay_path: str,
    summary_path: str,
    source_index: int,
    model,
    device,
    feature_profile: str,
):
  replay, _ = load_replay(replay_path, return_metadata=True)
  summary = json.loads(Path(summary_path).read_text())
  row_stats = summary.get("row_stats")
  if not isinstance(row_stats, list) or len(row_stats) != len(replay):
    raise ValueError(f"{summary_path} must contain row_stats matching replay rows")
  policies, q_values, action_features = _masked_policy_and_q(model, replay, device)
  states = []
  skipped = 0
  for idx, row in enumerate(replay):
    state = _state_features(
        row,
        policies[idx],
        q_values[idx],
        action_features[idx],
        row_stats[idx],
        source_index,
        idx,
        feature_profile,
    )
    if state is None:
      skipped += 1
    else:
      states.append(state)
  return states, {
      "replay": str(replay_path),
      "summary": str(summary_path),
      "rows": int(len(replay)),
      "states": int(len(states)),
      "skipped": int(skipped),
      "source_any_paradox_rate": summary.get("same_policy_paradox_round_rate"),
      "source_rounds": summary.get("rounds"),
      "source_rounds_with_any_paradox": summary.get("rounds_with_any_paradox"),
  }


def _standardize(train_states, all_states):
  train_features = np.asarray([
      action["feature"] for state in train_states for action in state["actions"]
  ], dtype=np.float32)
  mean = train_features.mean(axis=0)
  std = train_features.std(axis=0)
  std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
  for state in all_states:
    for action in state["actions"]:
      action["feature_std"] = ((action["feature"] - mean) / std).astype(np.float32)
  return mean.astype(np.float32), std.astype(np.float32)


def _state_summary(states):
  return {
      "states": int(len(states)),
      "veto_states": int(sum(not state["target_is_base"] for state in states)),
      "no_veto_states": int(sum(state["target_is_base"] for state in states)),
      "actions": int(sum(len(state["actions"]) for state in states)),
      "mean_actions": (
          None if not states else
          round(float(np.mean([len(state["actions"]) for state in states])), 6)
      ),
  }


def _state_batch(states, device):
  max_actions = max(len(state["actions"]) for state in states)
  feature_size = len(states[0]["actions"][0]["feature_std"])
  x = torch.zeros((len(states), max_actions, feature_size), dtype=torch.float32, device=device)
  mask = torch.zeros((len(states), max_actions), dtype=torch.bool, device=device)
  target = torch.zeros((len(states),), dtype=torch.long, device=device)
  base = torch.zeros((len(states),), dtype=torch.long, device=device)
  weights = torch.ones((len(states),), dtype=torch.float32, device=device)
  veto = torch.zeros((len(states),), dtype=torch.bool, device=device)
  for row_idx, state in enumerate(states):
    action_count = len(state["actions"])
    x[row_idx, :action_count] = torch.tensor(
        np.asarray([a["feature_std"] for a in state["actions"]], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    mask[row_idx, :action_count] = True
    target[row_idx] = int(state["target_index"])
    base[row_idx] = int(state["base_index"])
    veto[row_idx] = not bool(state["target_is_base"])
  return x, mask, target, base, weights, veto


def _train(args, train_states, feature_size: int, device: torch.device):
  torch.manual_seed(int(args.seed))
  model = ResidualSelector(feature_size, int(args.hidden), float(args.dropout)).to(device)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
  )
  rng = np.random.default_rng(int(args.seed))
  batch_size = max(1, int(args.batch_size))
  losses = []
  ce_losses = []
  margin_losses = []
  for _ in range(max(1, int(args.epochs))):
    order = rng.permutation(len(train_states))
    total = ce_total = margin_total = 0.0
    seen = 0
    for start in range(0, len(order), batch_size):
      batch = [train_states[int(idx)] for idx in order[start:start + batch_size]]
      x, mask, target, base, weights, veto = _state_batch(batch, device)
      weights = torch.where(
          veto,
          torch.full_like(weights, float(args.veto_state_weight)),
          weights,
      )
      scores = model(x).masked_fill(~mask, -1e9)
      ce = F.cross_entropy(scores, target, reduction="none")
      ce_loss = torch.mean(ce * weights)
      row_idx = torch.arange(scores.shape[0], device=device)
      target_scores = scores[row_idx, target]
      base_scores = scores[row_idx, base]
      nonbase_scores = scores.masked_fill(
          F.one_hot(base, scores.shape[1]).to(dtype=torch.bool), -1e9
      ).max(dim=1).values
      veto_margin = F.relu(float(args.score_margin) - (target_scores - base_scores))
      abstain_margin = F.relu(float(args.score_margin) - (base_scores - nonbase_scores))
      margin_loss = torch.mean(torch.where(veto, veto_margin, abstain_margin) * weights)
      loss = ce_loss + float(args.margin_loss_weight) * margin_loss
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      count = len(batch)
      total += float(loss.detach()) * count
      ce_total += float(ce_loss.detach()) * count
      margin_total += float(margin_loss.detach()) * count
      seen += count
    denom = max(1, seen)
    losses.append(total / denom)
    ce_losses.append(ce_total / denom)
    margin_losses.append(margin_total / denom)
  return model, {
      "initial_loss": round(float(losses[0]), 6),
      "final_loss": round(float(losses[-1]), 6),
      "initial_ce_loss": round(float(ce_losses[0]), 6),
      "final_ce_loss": round(float(ce_losses[-1]), 6),
      "initial_margin_loss": round(float(margin_losses[0]), 6),
      "final_margin_loss": round(float(margin_losses[-1]), 6),
  }


def _predict_model_scores(model, states, device):
  model.eval()
  with torch.no_grad():
    for state in states:
      x = torch.tensor(
          np.asarray([a["feature_std"] for a in state["actions"]], dtype=np.float32),
          dtype=torch.float32,
          device=device,
      )
      scores = model(x).detach().cpu().numpy().astype(np.float32)
      state["model_scores"] = scores


def _state_score_arrays(states, score_key: str):
  for state in states:
    if score_key == "model":
      scores = np.asarray(state["model_scores"], dtype=np.float32)
    elif score_key == "q":
      scores = np.asarray([a["q_value"] for a in state["actions"]], dtype=np.float32)
    elif score_key == "policy":
      scores = np.asarray([
          math.log(max(1e-8, float(a["policy_prob"]))) for a in state["actions"]
      ], dtype=np.float32)
    else:
      raise ValueError(f"unknown score key {score_key}")
    yield state, scores


def _top1_report(states, score_key: str):
  correct = 0
  veto_correct = 0
  veto_total = 0
  no_veto_correct = 0
  no_veto_total = 0
  for state, scores in _state_score_arrays(states, score_key):
    selected = int(np.argmax(scores))
    hit = selected == int(state["target_index"])
    correct += int(hit)
    if bool(state["target_is_base"]):
      no_veto_total += 1
      no_veto_correct += int(hit)
    else:
      veto_total += 1
      veto_correct += int(hit)
  return {
      "score_key": score_key,
      "states": int(len(states)),
      "top1": round(float(correct / max(1, len(states))), 6),
      "veto_top1": round(float(veto_correct / max(1, veto_total)), 6),
      "no_veto_top1": round(float(no_veto_correct / max(1, no_veto_total)), 6),
      "veto_states": int(veto_total),
      "no_veto_states": int(no_veto_total),
  }


def _threshold_report(args, states, score_key: str):
  reports = []
  thresholds = _parse_float_list(str(args.thresholds))
  for threshold in thresholds:
    overrides = correct_overrides = false_overrides = 0
    no_veto_states = no_veto_abstain = 0
    veto_states = veto_correct_or_abstained = 0
    top1 = 0
    margins = []
    for state, scores in _state_score_arrays(states, score_key):
      base = int(state["base_index"])
      target = int(state["target_index"])
      nonbase_scores = np.array(scores, copy=True)
      nonbase_scores[base] = -np.inf
      best_nonbase = int(np.argmax(nonbase_scores))
      margin = float(nonbase_scores[best_nonbase] - scores[base])
      selected = best_nonbase if margin >= float(threshold) else base
      margins.append(margin)
      top1 += int(selected == target)
      if bool(state["target_is_base"]):
        no_veto_states += 1
        if selected == base:
          no_veto_abstain += 1
        else:
          overrides += 1
          false_overrides += 1
      else:
        veto_states += 1
        if selected != base:
          overrides += 1
          correct_overrides += int(selected == target)
        if selected == target:
          veto_correct_or_abstained += 1
    precision = correct_overrides / overrides if overrides else None
    no_veto_abstain_rate = no_veto_abstain / max(1, no_veto_states)
    reports.append({
        "score_key": score_key,
        "threshold": float(threshold),
        "states": int(len(states)),
        "top1": round(float(top1 / max(1, len(states))), 6),
        "overrides": int(overrides),
        "correct_overrides": int(correct_overrides),
        "false_overrides": int(false_overrides),
        "override_precision": None if precision is None else round(float(precision), 6),
        "veto_recall": round(float(correct_overrides / max(1, veto_states)), 6),
        "no_veto_abstain_rate": round(float(no_veto_abstain_rate), 6),
        "mean_margin": round(float(np.mean(margins)), 6) if margins else None,
        "median_margin": round(float(np.median(margins)), 6) if margins else None,
    })
  return reports


def _eligible(args, rows):
  out = []
  for row in rows:
    if int(row["overrides"]) < int(args.min_validation_overrides_for_live):
      continue
    if row["override_precision"] is None or float(row["override_precision"]) < float(
        args.min_validation_precision_for_live
    ):
      continue
    if float(row["no_veto_abstain_rate"]) < float(args.min_validation_no_veto_abstain_rate):
      continue
    out.append(row)
  return out


def _best(rows):
  if not rows:
    return None
  return max(
      rows,
      key=lambda row: (
          row["override_precision"] is not None,
          row["override_precision"] or -1.0,
          row["no_veto_abstain_rate"],
          row["veto_recall"],
          row["overrides"],
          row["top1"],
      ),
  )


def main():
  args = parse_args()
  run_device = _device(args.device)
  model, payload, saved_args = _load_frozen_model(args.checkpoint, args.players, run_device)
  del payload
  train_states, train_source = _load_states(
      args.train_replay, args.train_summary, 0, model, run_device,
      str(args.feature_profile),
  )
  validation_states, validation_source = _load_states(
      args.validation_replay, args.validation_summary, 1, model, run_device,
      str(args.feature_profile),
  )
  all_states = train_states + validation_states
  mean, std = _standardize(train_states, all_states)
  feature_size = len(train_states[0]["actions"][0]["feature_std"])
  selector, train_report = _train(args, train_states, feature_size, run_device)
  _predict_model_scores(selector, all_states, run_device)

  train_thresholds = _threshold_report(args, train_states, "model")
  validation_thresholds = _threshold_report(args, validation_states, "model")
  q_validation_thresholds = _threshold_report(args, validation_states, "q")
  policy_validation_thresholds = _threshold_report(args, validation_states, "policy")
  eligible = _eligible(args, validation_thresholds)

  model_out = str(args.model_out or "")
  if model_out:
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": selector.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "feature_size": int(feature_size),
        "model_class": "ResidualSelector",
        "args": vars(args),
        "checkpoint_saved_args": dict(saved_args),
        "train_report": train_report,
        "train_source": train_source,
        "validation_source": validation_source,
    }, model_out)

  result = {
      "kind": "veto_residual_selector",
      "checkpoint": str(args.checkpoint),
      "checkpoint_saved_args": dict(saved_args),
      "model_out": model_out,
      "feature_size": int(feature_size),
      "train_source": train_source,
      "validation_source": validation_source,
      "train_summary": _state_summary(train_states),
      "validation_summary": _state_summary(validation_states),
      "train_report": train_report,
      "top1": {
          "train_model": _top1_report(train_states, "model"),
          "validation_model": _top1_report(validation_states, "model"),
          "validation_q": _top1_report(validation_states, "q"),
          "validation_policy": _top1_report(validation_states, "policy"),
      },
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "validation_q_thresholds": q_validation_thresholds,
      "validation_policy_thresholds": policy_validation_thresholds,
      "live_screen_gate": {
          "passed": bool(eligible),
          "eligible_thresholds": int(len(eligible)),
          "best_eligible_threshold": _best(eligible),
          "best_validation_threshold": _best(validation_thresholds),
          "best_q_validation_threshold": _best(q_validation_thresholds),
          "best_policy_validation_threshold": _best(policy_validation_thresholds),
          "min_overrides": int(args.min_validation_overrides_for_live),
          "min_precision": float(args.min_validation_precision_for_live),
          "min_no_veto_abstain_rate": float(args.min_validation_no_veto_abstain_rate),
      },
      "args": vars(args),
  }
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
