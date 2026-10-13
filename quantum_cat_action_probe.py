#!/usr/bin/env python3
"""Cheap action-policy diagnostics for Quantum Cat checkpoints.

This probe is intentionally not a strength gate. It answers a narrower but
important question before expensive full-match Elo runs: where does a candidate
policy disagree with a teacher/champion policy over legal actions?
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
import torch.nn.functional as F

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import (
    action_feature_matrix,
    adapt_observation_batch,
    load_model_payload,
    model_action_value_batch,
    mcts_policy,
    model_input_size,
    model_policy_value,
    phase_name_for_state,
    policy_alignment_summary,
)


def _device(name: str) -> torch.device:
  if name == "auto":
    if torch.backends.mps.is_available():
      return torch.device("mps")
    return torch.device("cpu")
  return torch.device(name)


def _model_args(players: int, value_scale: float):
  return SimpleNamespace(
      players=players,
      arch="mlp",
      width=256,
      depth=3,
      value_scale=value_scale,
      match_context=True,
  )


def _load_checkpoint(path: str, game, args, device):
  model, _, saved_args = load_model_payload(path, game, args, device)
  model.eval()
  return model, float(saved_args.get("value_scale", args.value_scale))


def _masked_model_probs(model, obs, masks, action_features, device):
  chunks = []
  batch_size = 128 if getattr(model, "arch", "mlp") in ("action_mlp", "action_attn") else 512
  with torch.no_grad():
    for start in range(0, len(obs), batch_size):
      end = min(len(obs), start + batch_size)
      obs_t = torch.tensor(obs[start:end], dtype=torch.float32, device=device)
      obs_t = adapt_observation_batch(obs_t, model_input_size(model))
      mask_t = torch.tensor(masks[start:end], dtype=torch.bool, device=device)
      action_features_t = torch.tensor(
          action_features[start:end], dtype=torch.float32, device=device
      )
      logits, _ = model(obs_t, action_features_t)
      logits = logits.masked_fill(~mask_t, -1e9)
      chunks.append(F.softmax(logits, dim=1).detach().cpu().numpy())
  return np.concatenate(chunks, axis=0)


def _masked_model_action_values(model, obs, masks, action_features, device):
  if not hasattr(model, "action_value"):
    return None
  chunks = []
  batch_size = 128 if getattr(model, "arch", "mlp") in ("action_mlp", "action_attn") else 512
  with torch.no_grad():
    for start in range(0, len(obs), batch_size):
      end = min(len(obs), start + batch_size)
      obs_t = torch.tensor(obs[start:end], dtype=torch.float32, device=device)
      obs_t = adapt_observation_batch(obs_t, model_input_size(model))
      action_features_t = torch.tensor(
          action_features[start:end], dtype=torch.float32, device=device
      )
      values = model_action_value_batch(model, obs_t, action_features_t)
      values = values.detach().cpu().numpy()
      values = np.where(masks[start:end].astype(bool), values, np.nan)
      chunks.append(values)
  return np.concatenate(chunks, axis=0)


def _sample_action(policy, legal, temperature):
  legal = list(legal)
  if not legal:
    raise ValueError("No legal actions to sample")
  if temperature <= 0:
    return int(max(legal, key=lambda action: float(policy[action])))
  probs = np.array([max(0.0, float(policy[action])) for action in legal], dtype=np.float64)
  total = float(np.sum(probs))
  if total <= 0 or not np.isfinite(total):
    probs = np.ones(len(legal), dtype=np.float64) / float(len(legal))
  else:
    probs /= total
  if temperature != 1.0:
    probs = np.power(probs, 1.0 / max(1e-6, float(temperature)))
    probs /= float(np.sum(probs))
  return int(np.random.choice(legal, p=probs))


def _teacher_policy(state, teacher_model, args, device):
  player = int(state.current_player())
  if args.teacher_mode == "mcts":
    return mcts_policy(
        state,
        teacher_model,
        args,
        device,
        add_noise=False,
        sims=args.teacher_sims,
    )
  if args.teacher_mode == "belief_policy":
    policy = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    for _ in range(max(1, args.teacher_belief_samples)):
      try:
        sampled = state.resample_from_infostate(player, np.random)
      except Exception:
        sampled = state.clone()
      sampled_policy, _ = model_policy_value(
          teacher_model,
          sampled,
          player,
          state.num_distinct_actions(),
          args.value_scale,
          device,
      )
      policy += sampled_policy
    policy /= float(max(1, args.teacher_belief_samples))
    return policy
  policy, _ = model_policy_value(
      teacher_model,
      state,
      player,
      state.num_distinct_actions(),
      args.value_scale,
      device,
  )
  return policy


def collect_probe_rows(game, teacher_model, args, device):
  rows = []
  phase_counts: dict[str, int] = {}
  for game_idx in range(args.games):
    state = game.new_initial_state()
    if args.match_context:
      state.set_match_context(
          np.zeros(args.players, dtype=np.float32),
          game_idx % args.players,
      )
    ply = 0
    while not state.is_terminal():
      if state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state.apply_action(int(np.random.choice(actions, p=probs)))
        continue
      player = int(state.current_player())
      legal = state.legal_actions(player)
      policy = _teacher_policy(state, teacher_model, args, device)
      legal_mask = np.zeros(game.num_distinct_actions(), dtype=np.float32)
      legal_mask[legal] = 1.0
      features = action_feature_matrix(state, player, game.num_distinct_actions())
      rows.append({
          "obs": np.array(state.observation_tensor(player), dtype=np.float32),
          "mask": legal_mask,
          "policy": np.array(policy, dtype=np.float32),
          "action_features": features,
          "phase": phase_name_for_state(state),
          "player": player,
      })
      phase_counts[rows[-1]["phase"]] = phase_counts.get(rows[-1]["phase"], 0) + 1
      if len(rows) >= args.max_states:
        return rows, phase_counts
      action = _sample_action(policy, legal, args.temperature if ply < args.temperature_drop else 0.0)
      state.apply_action(action)
      ply += 1
  return rows, phase_counts


def _phase_bucket_report(probs, targets, phases):
  report = {}
  phases = np.array(phases)
  target_actions = np.argmax(targets, axis=1)
  pred_actions = np.argmax(probs, axis=1)
  target_probs = probs[np.arange(len(probs)), target_actions]
  cross_entropy = -np.sum(targets * np.log(np.maximum(probs, 1e-12)), axis=1)
  for phase in sorted(set(phases.tolist())):
    mask = phases == phase
    report[phase] = {
        "count": int(np.sum(mask)),
        "top1_rate": round(float(np.mean(pred_actions[mask] == target_actions[mask])), 4),
        "mean_target_prob": round(float(np.mean(target_probs[mask])), 4),
        "cross_entropy": round(float(np.mean(cross_entropy[mask])), 4),
    }
  return report


def _allowed_phase_set(spec: str):
  if not str(spec or "").strip():
    return None
  return {part.strip().lower() for part in str(spec).split(",") if part.strip()}


def _q_rerank_report(probs, action_values, masks, targets, phases, args):
  if action_values is None or float(args.action_value_selection_weight) <= 0:
    return None
  allowed_phases = _allowed_phase_set(args.action_value_rerank_phases)
  clip = max(0.0, float(args.action_value_rerank_clip))
  min_margin = float(args.action_value_rerank_min_margin)
  weight = float(args.action_value_selection_weight)
  value_scale = float(args.value_scale)
  target_actions = np.argmax(targets, axis=1)
  base_actions = np.argmax(probs, axis=1)
  q_actions = np.array(base_actions, copy=True)
  margins = []
  passed_margins = []
  override_margins = []
  phase_rows = {}
  enabled = 0
  margin_pass = 0
  overrides = 0
  for idx, phase in enumerate(phases):
    phase = str(phase)
    bucket = phase_rows.setdefault(
        phase,
        {
            "count": 0,
            "enabled": 0,
            "margin_pass": 0,
            "overrides": 0,
            "base_teacher_top1": 0,
            "q_teacher_top1": 0,
        },
    )
    bucket["count"] += 1
    legal = np.flatnonzero(masks[idx].astype(bool))
    if len(legal) == 0:
      continue
    if int(base_actions[idx]) == int(target_actions[idx]):
      bucket["base_teacher_top1"] += 1
    phase_enabled = allowed_phases is None or phase.lower() in allowed_phases
    if not phase_enabled or len(legal) == 1:
      if int(q_actions[idx]) == int(target_actions[idx]):
        bucket["q_teacher_top1"] += 1
      continue
    enabled += 1
    bucket["enabled"] += 1
    legal_values = np.array([float(action_values[idx, action]) for action in legal])
    if clip > 0:
      legal_values = np.clip(legal_values, -clip, clip)
    top_two = np.partition(legal_values, -2)[-2:] if len(legal_values) > 1 else legal_values
    margin = float(np.max(top_two) - np.min(top_two)) if len(top_two) > 1 else 0.0
    margins.append(margin)
    if min_margin > 0 and margin < min_margin:
      if int(q_actions[idx]) == int(target_actions[idx]):
        bucket["q_teacher_top1"] += 1
      continue
    margin_pass += 1
    bucket["margin_pass"] += 1
    passed_margins.append(margin)
    full_values = np.array(action_values[idx], copy=True)
    if clip > 0:
      full_values = np.clip(full_values, -clip, clip)
    scores = np.full_like(probs[idx], -np.inf, dtype=np.float64)
    for action in legal:
      scores[action] = (
          np.log(max(float(probs[idx, action]), 1e-12))
          + weight * float(full_values[action]) * value_scale
      )
    q_action = int(np.argmax(scores))
    q_actions[idx] = q_action
    if q_action != int(base_actions[idx]):
      overrides += 1
      bucket["overrides"] += 1
      override_margins.append(margin)
    if q_action == int(target_actions[idx]):
      bucket["q_teacher_top1"] += 1

  def _summary(values):
    if not values:
      return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "max": round(float(np.max(arr)), 4),
    }

  total = len(phases)
  base_teacher_top1 = float(np.mean(base_actions == target_actions))
  q_teacher_top1 = float(np.mean(q_actions == target_actions))
  by_phase = {}
  for phase, bucket in phase_rows.items():
    count = max(1, int(bucket["count"]))
    by_phase[phase] = {
        **bucket,
        "enabled_rate": round(bucket["enabled"] / count, 4),
        "margin_pass_rate": round(bucket["margin_pass"] / count, 4),
        "override_rate": round(bucket["overrides"] / count, 4),
        "base_teacher_top1_rate": round(bucket["base_teacher_top1"] / count, 4),
        "q_teacher_top1_rate": round(bucket["q_teacher_top1"] / count, 4),
    }
  return {
      "weight": weight,
      "value_scale": value_scale,
      "clip": clip,
      "min_margin": min_margin,
      "rerank_phases": args.action_value_rerank_phases,
      "states": total,
      "enabled_states": enabled,
      "margin_pass_states": margin_pass,
      "overrides": overrides,
      "enabled_rate": round(enabled / max(1, total), 4),
      "margin_pass_rate": round(margin_pass / max(1, total), 4),
      "override_rate": round(overrides / max(1, total), 4),
      "base_teacher_top1_rate": round(base_teacher_top1, 4),
      "q_teacher_top1_rate": round(q_teacher_top1, 4),
      "margin_summary": _summary(margins),
      "passed_margin_summary": _summary(passed_margins),
      "override_margin_summary": _summary(override_margins),
      "by_phase": by_phase,
  }


def run_probe(args):
  started = time.perf_counter()
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  device = _device(args.device)
  game = pyspiel.load_game(
      "python_quantum_cat",
      {"players": args.players, "match_context": int(args.match_context)},
  )
  base_args = _model_args(args.players, args.value_scale)
  teacher_model, teacher_value_scale = _load_checkpoint(
      args.teacher_checkpoint, game, base_args, device
  )
  args.value_scale = teacher_value_scale
  candidate_model, _ = _load_checkpoint(
      args.candidate_checkpoint or args.teacher_checkpoint, game, base_args, device
  )

  rows, phase_counts = collect_probe_rows(game, teacher_model, args, device)
  if not rows:
    raise RuntimeError("No probe rows collected")
  obs = np.array([row["obs"] for row in rows], dtype=np.float32)
  masks = np.array([row["mask"] for row in rows], dtype=np.float32)
  targets = np.array([row["policy"] for row in rows], dtype=np.float32)
  action_features = np.array([row["action_features"] for row in rows], dtype=np.float32)
  phases = [str(row["phase"]) for row in rows]
  probs = _masked_model_probs(candidate_model, obs, masks, action_features, device)
  summary = policy_alignment_summary(probs, targets, action_features)
  summary["by_phase"] = _phase_bucket_report(probs, targets, phases)
  action_values = _masked_model_action_values(
      candidate_model, obs, masks, action_features, device
  )
  result = {
      "teacher_checkpoint": args.teacher_checkpoint,
      "candidate_checkpoint": args.candidate_checkpoint or args.teacher_checkpoint,
      "teacher_mode": args.teacher_mode,
      "teacher_sims": args.teacher_sims,
      "teacher_belief_samples": args.teacher_belief_samples,
      "players": args.players,
      "match_context": bool(args.match_context),
      "games": args.games,
      "max_states": args.max_states,
      "seed": args.seed,
      "elapsed_sec": round(time.perf_counter() - started, 3),
      "collected_states": len(rows),
      "collection_phase_counts": phase_counts,
      "alignment": summary,
      "q_rerank": _q_rerank_report(probs, action_values, masks, targets, phases, args),
  }
  if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--teacher-checkpoint", required=True)
  parser.add_argument("--candidate-checkpoint", default="")
  parser.add_argument("--teacher-mode", choices=("policy", "mcts", "belief_policy"), default="policy")
  parser.add_argument("--teacher-sims", type=int, default=16)
  parser.add_argument("--teacher-belief-samples", type=int, default=4)
  parser.add_argument("--sims", type=int, default=16)
  parser.add_argument("--c-puct", type=float, default=1.5)
  parser.add_argument("--paradox-value-penalty", type=float, default=0.0)
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument("--action-paradox-root-only", action="store_true")
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-root-only", action="store_true")
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument("--action-value-rerank-phases", default="")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--games", type=int, default=24)
  parser.add_argument("--max-states", type=int, default=512)
  parser.add_argument("--temperature", type=float, default=0.35)
  parser.add_argument("--temperature-drop", type=int, default=8)
  parser.add_argument("--value-scale", type=float, default=20.0)
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
  parser.add_argument("--seed", type=int, default=20260604)
  parser.add_argument("--out", default="")
  run_probe(parser.parse_args())


if __name__ == "__main__":
  main()
