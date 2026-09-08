#!/usr/bin/env python3
"""Audit belief-consistent late-exit labels for Quantum Cat.

Run818 showed that exact hidden-state shield vetoes are not directly
recoverable from actor-side features.  This script asks the next question:
if we sample hidden worlds from the acting player's information state and score
late exits across those worlds, do we get stable action labels worth training?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import (
    _args_with_phase_ref_mix,
    _score_observed_trick_likelihood_batch,
    _uniform_counterfactual_resamples,
    action_feature_matrix,
    counterfactual_belief_states,
    load_model_payload,
    make_game,
    model_policy_value,
    phase_name_for_state,
)
from quantum_cat_counterfactual_liquidity_replay import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    exact_table_liveness_scalar,
    red_switch_mask,
    token_loss_mask,
)
from quantum_cat_full_match_elo import (
    exact_legal_pressure_after_action,
    exact_table_liveness_after_action,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--matches", type=int, default=8)
  parser.add_argument("--seed", type=int, default=20261391)
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument("--match-context", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--source-temperature", type=float, default=0.20)
  parser.add_argument("--min-trick-number", type=int, default=5)
  parser.add_argument("--max-states", type=int, default=0)
  parser.add_argument("--belief-samples", type=int, default=8)
  parser.add_argument(
      "--belief-source",
      choices=("infostate", "policy_weighted", "policy_weighted_cap", "mixed", "actual"),
      default="infostate",
  )
  parser.add_argument("--belief-candidates", type=int, default=32)
  parser.add_argument(
      "--policy-weighted-score-candidates",
      type=int,
      default=0,
      help=(
          "For belief_source=policy_weighted_cap, score at most this many "
          "candidate worlds. 0 means use belief-candidates."
      ),
  )
  parser.add_argument("--belief-policy-temperature", type=float, default=1.0)
  parser.add_argument("--belief-uniform-mix", type=float, default=0.15)
  parser.add_argument("--belief-ref-policy-mix", default="model:1.0")
  parser.add_argument("--belief-ref-policy-mix-by-phase", default="")
  parser.add_argument(
      "--policy-weighted-action-aggregation",
      choices=("sample", "expected"),
      default="sample",
      help=(
          "For policy_weighted_cap, either sample posterior worlds as before "
          "or compute action scores as posterior-weighted expectations over "
          "all scored candidates."
      ),
  )
  parser.add_argument(
      "--policy-weighted-score-predictions",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Whether policy-weighted posterior scoring includes prediction actions.",
  )
  parser.add_argument(
      "--policy-weighted-score-completed-trick-window",
      type=int,
      default=0,
      help=(
          "Score only the last N completed tricks when weighting candidates. "
          "0 scores all completed tricks."
      ),
  )
  parser.add_argument("--min-belief-action-coverage", type=float, default=0.75)
  parser.add_argument("--stable-margin", type=float, default=0.25)
  parser.add_argument(
      "--audit-batch-size",
      type=int,
      default=1,
      help=(
          "Batch this many audited states when scoring policy-weighted belief "
          "candidates. 1 preserves the historical per-state scorer."
      ),
  )
  parser.add_argument(
      "--torch-threads",
      type=int,
      default=0,
      help="If >0, set torch intra-op and inter-op threads for this audit process.",
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
  return parser.parse_args()


def _device(name: str) -> torch.device:
  if name == "auto":
    if torch.backends.mps.is_available():
      return torch.device("mps")
    return torch.device("cpu")
  return torch.device(name)


def _model_args(args):
  return SimpleNamespace(
      players=int(args.players),
      arch="action_mlp",
      width=256,
      depth=3,
      match_context=bool(args.match_context),
      separate_action_value_encoder=True,
      separate_action_paradox_encoder=True,
  )


def _belief_args(args):
  return SimpleNamespace(
      players=int(args.players),
      value_scale=20.0,
      counterfactual_belief_source=str(args.belief_source),
      counterfactual_belief_samples=max(1, int(args.belief_samples)),
      counterfactual_belief_candidates=max(
          int(args.belief_samples), int(args.belief_candidates)
      ),
      counterfactual_belief_policy_temperature=float(args.belief_policy_temperature),
      counterfactual_belief_uniform_mix=float(args.belief_uniform_mix),
      counterfactual_belief_ref_policy_mix=str(args.belief_ref_policy_mix),
      counterfactual_belief_ref_policy_mix_by_phase=str(
          args.belief_ref_policy_mix_by_phase
      ),
      counterfactual_belief_score_predictions=bool(
          args.policy_weighted_score_predictions
      ),
      counterfactual_belief_score_completed_trick_window=max(
          0, int(args.policy_weighted_score_completed_trick_window)
      ),
      counterfactual_belief_logprob_floor=1e-6,
  )


def _chance_action(state, rng: np.random.Generator) -> int:
  outcomes = state.chance_outcomes()
  actions, probs = zip(*outcomes)
  return int(rng.choice(actions, p=np.asarray(probs, dtype=np.float64)))


def _sample_action_rng(policy: np.ndarray, legal: list[int], temperature: float, rng) -> int:
  if temperature <= 1e-6:
    return int(max(legal, key=lambda action: float(policy[action])))
  probs = np.asarray([float(policy[action]) for action in legal], dtype=np.float64)
  probs = probs ** (1.0 / float(temperature))
  total = float(probs.sum())
  if total <= 0 or not np.isfinite(total):
    probs = np.ones(len(legal), dtype=np.float64) / float(len(legal))
  else:
    probs /= total
  return int(rng.choice(legal, p=probs))


def _policy_action(state, player, model, args, device, rng) -> tuple[int, np.ndarray]:
  policy, _ = model_policy_value(
      model,
      state,
      player,
      state.num_distinct_actions(),
      float(getattr(args, "value_scale", 20.0)),
      device,
  )
  legal = state.legal_actions(player)
  if phase_name_for_state(state) == "play":
    return _sample_action_rng(policy, legal, float(args.source_temperature), rng), policy
  return _sample_action_rng(policy, legal, 0.0, rng), policy


def _policy_weighted_cap_belief_states(state, player, args, model, device):
  sample_count = max(1, int(args.belief_samples))
  configured_candidates = max(sample_count, int(args.belief_candidates))
  score_cap = int(args.policy_weighted_score_candidates)
  if score_cap > 0:
    candidate_count = max(sample_count, min(configured_candidates, score_cap))
  else:
    candidate_count = configured_candidates

  belief_args = _args_with_phase_ref_mix(_belief_args(args), state)
  candidates = _uniform_counterfactual_resamples(state, player, candidate_count)
  scores = _score_observed_trick_likelihood_batch(candidates, model, belief_args, device)
  meta = {
      "belief_source_effective": "policy_weighted_cap",
      "policy_weighted_score_candidates": int(candidate_count),
      "policy_weighted_score_predictions": bool(args.policy_weighted_score_predictions),
      "policy_weighted_score_completed_trick_window": int(
          args.policy_weighted_score_completed_trick_window
      ),
      "policy_weighted_fallback": False,
      "policy_weighted_fallback_reason": None,
      "policy_weighted_finite_scores": 0,
      "policy_weighted_min_score": None,
      "policy_weighted_max_score": None,
      "policy_weighted_mean_score": None,
  }
  if not candidates:
    meta["policy_weighted_fallback"] = True
    meta["policy_weighted_fallback_reason"] = "no_candidates"
    return _uniform_counterfactual_resamples(state, player, sample_count), meta
  if len(scores) != len(candidates) or any(score is None for score in scores):
    meta["policy_weighted_fallback"] = True
    meta["policy_weighted_fallback_reason"] = "unscorable_candidate"
    return _uniform_counterfactual_resamples(state, player, sample_count), meta

  scores_np = np.asarray(scores, dtype=np.float64)
  finite = np.isfinite(scores_np)
  meta["policy_weighted_finite_scores"] = int(np.sum(finite))
  if not np.any(finite):
    meta["policy_weighted_fallback"] = True
    meta["policy_weighted_fallback_reason"] = "no_finite_scores"
    return _uniform_counterfactual_resamples(state, player, sample_count), meta

  finite_scores = scores_np[finite]
  meta["policy_weighted_min_score"] = float(np.min(finite_scores))
  meta["policy_weighted_max_score"] = float(np.max(finite_scores))
  meta["policy_weighted_mean_score"] = float(np.mean(finite_scores))
  scores_np[~finite] = float(np.min(finite_scores)) - 1e6
  temperature = max(1e-6, float(args.belief_policy_temperature))
  logits = (scores_np - float(np.max(scores_np))) / temperature
  weights = np.exp(logits)
  total = float(weights.sum())
  if total <= 0 or not math.isfinite(total):
    probs = np.ones(len(candidates), dtype=np.float64) / float(len(candidates))
  else:
    probs = weights / total
  mix = min(1.0, max(0.0, float(args.belief_uniform_mix)))
  probs = (1.0 - mix) * probs + mix / float(len(candidates))
  probs = probs / float(probs.sum())
  chosen = np.random.choice(len(candidates), size=sample_count, replace=True, p=probs)
  meta["policy_weighted_selected_unique"] = int(len(set(int(idx) for idx in chosen)))
  return [candidates[int(idx)].clone() for idx in chosen], meta


def _policy_weighted_cap_belief_states_batch(items, args, model, device):
  sample_count = max(1, int(args.belief_samples))
  configured_candidates = max(sample_count, int(args.belief_candidates))
  score_cap = int(args.policy_weighted_score_candidates)
  if score_cap > 0:
    candidate_count = max(sample_count, min(configured_candidates, score_cap))
  else:
    candidate_count = configured_candidates

  candidate_groups = []
  all_candidates = []
  for item in items:
    candidates = _uniform_counterfactual_resamples(
        item["state"], item["player"], candidate_count
    )
    candidate_groups.append(candidates)
    all_candidates.extend(candidates)

  score_args = _args_with_phase_ref_mix(_belief_args(args), items[0]["state"])
  all_scores = _score_observed_trick_likelihood_batch(
      all_candidates, model, score_args, device
  )
  out = []
  offset = 0
  for candidates in candidate_groups:
    scores = all_scores[offset:offset + len(candidates)]
    offset += len(candidates)
    meta = {
        "belief_source_effective": "policy_weighted_cap",
        "policy_weighted_score_candidates": int(candidate_count),
        "policy_weighted_score_predictions": bool(args.policy_weighted_score_predictions),
        "policy_weighted_score_completed_trick_window": int(
            args.policy_weighted_score_completed_trick_window
        ),
        "policy_weighted_fallback": False,
        "policy_weighted_fallback_reason": None,
        "policy_weighted_finite_scores": 0,
        "policy_weighted_min_score": None,
        "policy_weighted_max_score": None,
        "policy_weighted_mean_score": None,
        "policy_weighted_batch_size": int(len(items)),
    }
    if not candidates:
      meta["policy_weighted_fallback"] = True
      meta["policy_weighted_fallback_reason"] = "no_candidates"
      out.append((_uniform_counterfactual_resamples(
          items[len(out)]["state"], items[len(out)]["player"], sample_count
      ), meta))
      continue
    if len(scores) != len(candidates) or any(score is None for score in scores):
      meta["policy_weighted_fallback"] = True
      meta["policy_weighted_fallback_reason"] = "unscorable_candidate"
      out.append((_uniform_counterfactual_resamples(
          items[len(out)]["state"], items[len(out)]["player"], sample_count
      ), meta))
      continue

    scores_np = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores_np)
    meta["policy_weighted_finite_scores"] = int(np.sum(finite))
    if not np.any(finite):
      meta["policy_weighted_fallback"] = True
      meta["policy_weighted_fallback_reason"] = "no_finite_scores"
      out.append((_uniform_counterfactual_resamples(
          items[len(out)]["state"], items[len(out)]["player"], sample_count
      ), meta))
      continue

    finite_scores = scores_np[finite]
    meta["policy_weighted_min_score"] = float(np.min(finite_scores))
    meta["policy_weighted_max_score"] = float(np.max(finite_scores))
    meta["policy_weighted_mean_score"] = float(np.mean(finite_scores))
    scores_np[~finite] = float(np.min(finite_scores)) - 1e6
    temperature = max(1e-6, float(args.belief_policy_temperature))
    logits = (scores_np - float(np.max(scores_np))) / temperature
    weights = np.exp(logits)
    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
      probs = np.ones(len(candidates), dtype=np.float64) / float(len(candidates))
    else:
      probs = weights / total
    mix = min(1.0, max(0.0, float(args.belief_uniform_mix)))
    probs = (1.0 - mix) * probs + mix / float(len(candidates))
    probs = probs / float(probs.sum())
    chosen = np.random.choice(len(candidates), size=sample_count, replace=True, p=probs)
    meta["policy_weighted_selected_unique"] = int(len(set(int(idx) for idx in chosen)))
    out.append(([candidates[int(idx)].clone() for idx in chosen], meta))
  return out


def _posterior_probs_from_scores(candidates, scores, args, meta):
  if not candidates:
    meta["policy_weighted_fallback"] = True
    meta["policy_weighted_fallback_reason"] = "no_candidates"
    return None
  if len(scores) != len(candidates) or any(score is None for score in scores):
    meta["policy_weighted_fallback"] = True
    meta["policy_weighted_fallback_reason"] = "unscorable_candidate"
    return None

  scores_np = np.asarray(scores, dtype=np.float64)
  finite = np.isfinite(scores_np)
  meta["policy_weighted_finite_scores"] = int(np.sum(finite))
  if not np.any(finite):
    meta["policy_weighted_fallback"] = True
    meta["policy_weighted_fallback_reason"] = "no_finite_scores"
    return None

  finite_scores = scores_np[finite]
  meta["policy_weighted_min_score"] = float(np.min(finite_scores))
  meta["policy_weighted_max_score"] = float(np.max(finite_scores))
  meta["policy_weighted_mean_score"] = float(np.mean(finite_scores))
  scores_np[~finite] = float(np.min(finite_scores)) - 1e6
  temperature = max(1e-6, float(args.belief_policy_temperature))
  logits = (scores_np - float(np.max(scores_np))) / temperature
  weights = np.exp(logits)
  total = float(weights.sum())
  if total <= 0 or not math.isfinite(total):
    probs = np.ones(len(candidates), dtype=np.float64) / float(len(candidates))
  else:
    probs = weights / total
  mix = min(1.0, max(0.0, float(args.belief_uniform_mix)))
  probs = (1.0 - mix) * probs + mix / float(len(candidates))
  return probs / float(probs.sum())


def _policy_weighted_cap_candidate_weights_batch(items, args, model, device):
  sample_count = max(1, int(args.belief_samples))
  configured_candidates = max(sample_count, int(args.belief_candidates))
  score_cap = int(args.policy_weighted_score_candidates)
  if score_cap > 0:
    candidate_count = max(sample_count, min(configured_candidates, score_cap))
  else:
    candidate_count = configured_candidates

  candidate_groups = []
  all_candidates = []
  for item in items:
    candidates = _uniform_counterfactual_resamples(
        item["state"], item["player"], candidate_count
    )
    candidate_groups.append(candidates)
    all_candidates.extend(candidates)

  score_args = _args_with_phase_ref_mix(_belief_args(args), items[0]["state"])
  all_scores = _score_observed_trick_likelihood_batch(
      all_candidates, model, score_args, device
  )
  out = []
  offset = 0
  for idx, candidates in enumerate(candidate_groups):
    scores = all_scores[offset:offset + len(candidates)]
    offset += len(candidates)
    meta = {
        "belief_source_effective": "policy_weighted_cap",
        "belief_weighted_expected": True,
        "policy_weighted_score_candidates": int(candidate_count),
        "policy_weighted_score_predictions": bool(args.policy_weighted_score_predictions),
        "policy_weighted_score_completed_trick_window": int(
            args.policy_weighted_score_completed_trick_window
        ),
        "policy_weighted_fallback": False,
        "policy_weighted_fallback_reason": None,
        "policy_weighted_finite_scores": 0,
        "policy_weighted_min_score": None,
        "policy_weighted_max_score": None,
        "policy_weighted_mean_score": None,
        "policy_weighted_batch_size": int(len(items)),
    }
    probs = _posterior_probs_from_scores(candidates, scores, args, meta)
    if probs is None:
      fallback = _uniform_counterfactual_resamples(
          items[idx]["state"], items[idx]["player"], sample_count
      )
      uniform = np.ones(len(fallback), dtype=np.float64) / float(max(1, len(fallback)))
      out.append((fallback, uniform, meta))
      continue
    meta["policy_weighted_effective_candidates"] = float(
        1.0 / np.sum(np.square(probs))
    )
    out.append((candidates, probs, meta))
  return out


def _action_table_score(state, action: int) -> float:
  row = exact_legal_pressure_after_action(state, int(action))
  row.update(exact_table_liveness_after_action(state, int(action)))
  return float(exact_table_liveness_scalar(row))


def _score_actions_from_belief_samples(state, player, legal, samples, source_meta, args):
  action_scores = {int(action): [] for action in legal}
  skipped_worlds = 0
  for sampled in samples:
    sampled_legal = set(int(action) for action in sampled.legal_actions(player))
    usable = [int(action) for action in legal if int(action) in sampled_legal]
    if not usable:
      skipped_worlds += 1
      continue
    for action in usable:
      action_scores[int(action)].append(_action_table_score(sampled, int(action)))
  means = {}
  stderrs = {}
  coverages = {}
  for action, values in action_scores.items():
    coverages[int(action)] = float(len(values) / max(1, len(samples)))
    if values:
      arr = np.asarray(values, dtype=np.float32)
      means[int(action)] = float(np.mean(arr))
      if len(arr) > 1:
        stderrs[int(action)] = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
      else:
        stderrs[int(action)] = None
  return means, stderrs, coverages, {
      "belief_samples_requested": int(args.belief_samples),
      "belief_samples_used": int(len(samples)),
      "belief_worlds_skipped": int(skipped_worlds),
      **source_meta,
  }


def _score_actions_from_weighted_belief_candidates(
    state, player, legal, candidates, weights, source_meta, args
):
  del state
  action_weighted_values = {int(action): [] for action in legal}
  skipped_worlds = 0
  for sampled, raw_weight in zip(candidates, weights):
    weight = float(raw_weight)
    if weight <= 0:
      continue
    sampled_legal = set(int(action) for action in sampled.legal_actions(player))
    usable = [int(action) for action in legal if int(action) in sampled_legal]
    if not usable:
      skipped_worlds += 1
      continue
    for action in usable:
      action_weighted_values[int(action)].append(
          (weight, _action_table_score(sampled, int(action)))
      )
  means = {}
  stderrs = {}
  coverages = {}
  for action, values in action_weighted_values.items():
    weight_sum = float(sum(weight for weight, _ in values))
    coverages[int(action)] = weight_sum
    if weight_sum <= 0:
      continue
    mean = float(sum(weight * value for weight, value in values) / weight_sum)
    means[int(action)] = mean
    if len(values) > 1:
      variance = float(
          sum(weight * ((value - mean) ** 2) for weight, value in values)
          / weight_sum
      )
      effective_n = (weight_sum ** 2) / max(
          1e-12, sum(weight ** 2 for weight, _ in values)
      )
      stderrs[int(action)] = float(math.sqrt(max(0.0, variance) / effective_n))
    else:
      stderrs[int(action)] = None
  return means, stderrs, coverages, {
      "belief_samples_requested": int(args.belief_samples),
      "belief_samples_used": int(len(candidates)),
      "belief_worlds_skipped": int(skipped_worlds),
      **source_meta,
  }


def _score_belief_actions(state, player, legal, model, args, device):
  if (
      str(args.belief_source) == "policy_weighted_cap"
      and str(args.policy_weighted_action_aggregation) == "expected"
  ):
    candidates, weights, source_meta = _policy_weighted_cap_candidate_weights_batch(
        [{
            "state": state,
            "player": player,
            "legal": legal,
        }],
        args,
        model,
        device,
    )[0]
    return _score_actions_from_weighted_belief_candidates(
        state, player, legal, candidates, weights, source_meta, args
    )
  if str(args.belief_source) == "policy_weighted_cap":
    samples, source_meta = _policy_weighted_cap_belief_states(
        state, player, args, model, device
    )
  else:
    belief_args = _belief_args(args)
    samples = counterfactual_belief_states(
        state, player, belief_args, model=model, device=device
    )
    source_meta = {"belief_source_effective": str(args.belief_source)}
  return _score_actions_from_belief_samples(
      state, player, legal, samples, source_meta, args
  )


def _score_belief_actions_batch(items, model, args, device):
  if (
      str(args.belief_source) == "policy_weighted_cap"
      and str(args.policy_weighted_action_aggregation) == "expected"
  ):
    weighted_groups = _policy_weighted_cap_candidate_weights_batch(
        items, args, model, device
    )
    return [
        _score_actions_from_weighted_belief_candidates(
            item["state"],
            item["player"],
            item["legal"],
            candidates,
            weights,
            source_meta,
            args,
        )
        for item, (candidates, weights, source_meta) in zip(items, weighted_groups)
    ]
  if str(args.belief_source) != "policy_weighted_cap" or len(items) <= 1:
    return [
        _score_belief_actions(
            item["state"], item["player"], item["legal"], model, args, device
        )
        for item in items
    ]
  sample_groups = _policy_weighted_cap_belief_states_batch(items, args, model, device)
  return [
      _score_actions_from_belief_samples(
          item["state"],
          item["player"],
          item["legal"],
          samples,
          source_meta,
          args,
      )
      for item, (samples, source_meta) in zip(items, sample_groups)
  ]


def _rank_report(scores: dict[int, float], policy: np.ndarray, legal: list[int]):
  if not scores:
    return None
  ordered = sorted(
      scores,
      key=lambda action: (
          float(scores[int(action)]),
          float(policy[int(action)]),
          -int(action),
      ),
      reverse=True,
  )
  best = int(ordered[0])
  second = int(ordered[1]) if len(ordered) > 1 else best
  return {
      "best_action": best,
      "second_action": second,
      "best_score": float(scores[best]),
      "second_score": float(scores[second]),
      "margin": float(scores[best] - scores[second]) if best != second else 0.0,
      "ordered_actions": [int(action) for action in ordered],
  }


def _feature_flags(state, player, action: int):
  features = action_feature_matrix(state, player, state.num_distinct_actions())
  token_losses = token_loss_mask(features)
  red_switches = red_switch_mask(features)
  return {
      "follows_led": bool(features[int(action), ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5),
      "red": bool(features[int(action), ACTION_FEATURE_IS_RED_INDEX] > 0.5),
      "token_loss": bool(token_losses[int(action)]),
      "red_switch": bool(red_switches[int(action)]),
  }


def _audit_state_from_belief_scores(
    state,
    player,
    policy,
    args,
    match_idx,
    state_idx,
    belief_bundle,
):
  legal = [int(action) for action in state.legal_actions(player)]
  if len(legal) < 2:
    return None
  base_action = int(max(legal, key=lambda action: (float(policy[action]), -action)))
  belief_scores, stderrs, coverages, belief_meta = belief_bundle
  min_coverage = float(args.min_belief_action_coverage)
  belief_scores = {
      int(action): score for action, score in belief_scores.items()
      if float(coverages.get(int(action), 0.0)) >= min_coverage
  }
  if len(belief_scores) < 2 or base_action not in belief_scores:
    return None
  true_scores = {int(action): _action_table_score(state, int(action)) for action in legal}
  belief_rank = _rank_report(belief_scores, policy, legal)
  true_rank = _rank_report(true_scores, policy, legal)
  if belief_rank is None or true_rank is None:
    return None
  belief_best = int(belief_rank["best_action"])
  true_best = int(true_rank["best_action"])
  return {
      "match_index": int(match_idx),
      "state_index": int(state_idx),
      "player": int(player),
      "phase": phase_name_for_state(state),
      "trick_number": int(getattr(state, "_trick_number", -1)),
      "legal_actions": legal,
      "base_action": int(base_action),
      "belief_best_action": int(belief_best),
      "true_best_action": int(true_best),
      "belief_margin": float(belief_rank["margin"]),
      "true_margin": float(true_rank["margin"]),
      "belief_overrides_base": bool(belief_best != base_action),
      "true_overrides_base": bool(true_best != base_action),
      "belief_matches_true": bool(belief_best == true_best),
      "belief_stable": bool(float(belief_rank["margin"]) >= float(args.stable_margin)),
      "base_policy_prob": float(policy[base_action]),
      "belief_best_policy_prob": float(policy[belief_best]),
      "true_best_policy_prob": float(policy[true_best]),
      "belief_scores": {str(k): float(v) for k, v in sorted(belief_scores.items())},
      "true_scores": {str(k): float(v) for k, v in sorted(true_scores.items())},
      "belief_stderrs": {
          str(k): (None if stderrs.get(k) is None else float(stderrs[k]))
          for k in sorted(belief_scores)
      },
      "belief_coverages": {
          str(k): float(coverages.get(k, 0.0)) for k in sorted(belief_scores)
      },
      "base_flags": _feature_flags(state, player, base_action),
      "belief_best_flags": _feature_flags(state, player, belief_best),
      "true_best_flags": _feature_flags(state, player, true_best),
      **belief_meta,
  }


def _audit_state(state, player, policy, model, args, device, match_idx, state_idx):
  legal = [int(action) for action in state.legal_actions(player)]
  if len(legal) < 2:
    return None
  belief_bundle = _score_belief_actions(state, player, legal, model, args, device)
  return _audit_state_from_belief_scores(
      state, player, policy, args, match_idx, state_idx, belief_bundle
  )


def _summarize(rows: list[dict]) -> dict:
  if not rows:
    return {
        "states": 0,
        "belief_overrides": 0,
        "belief_stable_rows": 0,
        "belief_matches_true_rate": None,
    }
  belief_overrides = [row for row in rows if bool(row["belief_overrides_base"])]
  true_overrides = [row for row in rows if bool(row["true_overrides_base"])]
  stable = [row for row in rows if bool(row["belief_stable"])]
  stable_overrides = [
      row for row in belief_overrides if bool(row["belief_stable"])
  ]
  return {
      "states": int(len(rows)),
      "belief_overrides": int(len(belief_overrides)),
      "belief_override_rate": round(float(len(belief_overrides) / len(rows)), 6),
      "true_overrides": int(len(true_overrides)),
      "true_override_rate": round(float(len(true_overrides) / len(rows)), 6),
      "belief_stable_rows": int(len(stable)),
      "belief_stable_rate": round(float(len(stable) / len(rows)), 6),
      "stable_belief_overrides": int(len(stable_overrides)),
      "stable_belief_override_rate": round(
          float(len(stable_overrides) / len(rows)), 6
      ),
      "belief_matches_true": int(sum(bool(row["belief_matches_true"]) for row in rows)),
      "belief_matches_true_rate": round(
          float(sum(bool(row["belief_matches_true"]) for row in rows) / len(rows)),
          6,
      ),
      "belief_override_matches_true": int(
          sum(bool(row["belief_matches_true"]) for row in belief_overrides)
      ),
      "belief_override_matches_true_rate": (
          None if not belief_overrides else
          round(
              float(
                  sum(bool(row["belief_matches_true"]) for row in belief_overrides)
                  / len(belief_overrides)
              ),
              6,
          )
      ),
      "mean_belief_margin": round(
          float(np.mean([row["belief_margin"] for row in rows])), 6
      ),
      "mean_true_margin": round(
          float(np.mean([row["true_margin"] for row in rows])), 6
      ),
  }


def main():
  args = parse_args()
  if int(args.torch_threads) > 0:
    torch.set_num_threads(int(args.torch_threads))
    try:
      torch.set_num_interop_threads(int(args.torch_threads))
    except RuntimeError:
      pass
  rng = np.random.default_rng(int(args.seed))
  random.seed(int(args.seed))
  np.random.seed(int(args.seed) % (2**32 - 1))
  torch.manual_seed(int(args.seed))
  device = _device(args.device)
  game_args = _model_args(args)
  game = make_game(game_args, int(args.initial_start))
  model, _, saved_args = load_model_payload(args.checkpoint, game, game_args, device)
  model.eval()
  value_scale = float(saved_args.get("value_scale", 20.0))
  setattr(args, "value_scale", value_scale)
  rows = []
  pending = []
  visited_late_states = 0
  started = time.perf_counter()

  def flush_pending():
    if not pending:
      return
    batch_items = list(pending)
    pending.clear()
    bundles = _score_belief_actions_batch(batch_items, model, args, device)
    for item, belief_bundle in zip(batch_items, bundles):
      audited = _audit_state_from_belief_scores(
          item["state"],
          item["player"],
          item["policy"],
          args,
          item["match_idx"],
          item["state_idx"],
          belief_bundle,
      )
      if audited is not None:
        rows.append(audited)

  for match_idx in range(int(args.matches)):
    start_player = (int(args.initial_start) + match_idx) % int(args.players)
    game = make_game(game_args, start_player)
    state = game.new_initial_state()
    state_idx = 0
    while not state.is_terminal():
      if state.is_chance_node():
        state.apply_action(_chance_action(state, rng))
        continue
      player = state.current_player()
      action, policy = _policy_action(state, player, model, args, device, rng)
      if (
          phase_name_for_state(state) == "play"
          and int(getattr(state, "_trick_number", -1)) >= int(args.min_trick_number)
          and len(state.legal_actions(player)) >= 2
      ):
        visited_late_states += 1
        state_clone = state.clone()
        pending.append({
            "state": state_clone,
            "player": int(player),
            "policy": np.array(policy, dtype=np.float32, copy=True),
            "match_idx": int(match_idx),
            "state_idx": int(state_idx),
            "legal": [int(action) for action in state_clone.legal_actions(player)],
        })
        if len(pending) >= max(1, int(args.audit_batch_size)):
          flush_pending()
        if int(args.max_states) > 0 and len(rows) >= int(args.max_states):
          break
      state.apply_action(action)
      state_idx += 1
    if int(args.max_states) > 0 and len(rows) >= int(args.max_states):
      break
  flush_pending()
  if int(args.max_states) > 0 and len(rows) > int(args.max_states):
    rows = rows[:int(args.max_states)]
  result = {
      "kind": "belief_late_exit_audit",
      "checkpoint": str(args.checkpoint),
      "checkpoint_saved_args": dict(saved_args),
      "players": int(args.players),
      "matches_requested": int(args.matches),
      "seed": int(args.seed),
      "source_temperature": float(args.source_temperature),
      "min_trick_number": int(args.min_trick_number),
      "belief_source": str(args.belief_source),
      "belief_samples": int(args.belief_samples),
      "belief_candidates": int(args.belief_candidates),
      "policy_weighted_score_candidates": int(args.policy_weighted_score_candidates),
      "policy_weighted_score_predictions": bool(args.policy_weighted_score_predictions),
      "policy_weighted_score_completed_trick_window": int(
          args.policy_weighted_score_completed_trick_window
      ),
      "policy_weighted_action_aggregation": str(args.policy_weighted_action_aggregation),
      "source_action_rng_isolated": True,
      "audit_batch_size": int(args.audit_batch_size),
      "torch_threads": int(args.torch_threads),
      "min_belief_action_coverage": float(args.min_belief_action_coverage),
      "stable_margin": float(args.stable_margin),
      "visited_late_states": int(visited_late_states),
      "summary": _summarize(rows),
      "rows": rows,
      "elapsed_sec": round(time.perf_counter() - started, 3),
      "args": vars(args),
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
