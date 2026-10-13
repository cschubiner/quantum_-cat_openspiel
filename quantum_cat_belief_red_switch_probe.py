#!/usr/bin/env python3
"""Probe belief-averaged red-switch labels for early Cat in the Box states.

This is a small diagnostic for the Strawberry/RFT-style training loop.  The
same-deal repeat-validation bank can say "this red card was best" because it
knows the sampled hidden deal.  This probe asks whether that label survives
when each candidate is scored across hidden worlds consistent with the acting
player's observation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import (
    action_feature_matrix,
    load_model_payload,
    make_game,
    phase_name_for_state,
    save_replay,
)
from quantum_cat_belief_late_exit_audit import (
    _action_table_score,
    _device,
    _model_args,
    _score_belief_actions,
)
from quantum_cat_counterfactual_liquidity_replay import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    red_switch_mask,
    token_loss_mask,
)
from quantum_cat_failure_miner import (
    action_string,
    play_round_with_history,
    rollout_from_action,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--matches", type=int, default=40)
  parser.add_argument("--seed", type=int, default=20262620)
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument(
      "--match-context", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--min-trick-number", type=int, default=0)
  parser.add_argument("--max-trick-number", type=int, default=1)
  parser.add_argument("--max-states", type=int, default=80)
  parser.add_argument(
      "--partial-save-interval-states",
      type=int,
      default=0,
      help="If >0, write a partial JSON snapshot after this many audited rows.",
  )
  parser.add_argument("--belief-samples", type=int, default=12)
  parser.add_argument(
      "--score-mode",
      choices=("table_liveness", "rollout_survival"),
      default="table_liveness",
  )
  parser.add_argument("--rollout-actual-rollouts", type=int, default=2)
  parser.add_argument("--rollout-paradox-penalty", type=float, default=5.0)
  parser.add_argument(
      "--max-red-candidates",
      type=int,
      default=0,
      help="If >0, keep only the top-N policy-probability red-switch candidates.",
  )
  parser.add_argument(
      "--max-follow-candidates",
      type=int,
      default=0,
      help="If >0, keep only the top-N policy-probability follow-led candidates.",
  )
  parser.add_argument(
      "--belief-source",
      choices=("infostate", "policy_weighted", "policy_weighted_cap", "mixed", "actual"),
      default="infostate",
  )
  parser.add_argument("--belief-candidates", type=int, default=48)
  parser.add_argument("--belief-policy-temperature", type=float, default=1.0)
  parser.add_argument("--belief-uniform-mix", type=float, default=0.15)
  parser.add_argument("--belief-ref-policy-mix", default="model:1.0")
  parser.add_argument("--belief-ref-policy-mix-by-phase", default="")
  parser.add_argument("--policy-weighted-score-candidates", type=int, default=0)
  parser.add_argument(
      "--policy-weighted-action-aggregation",
      choices=("sample", "expected"),
      default="sample",
  )
  parser.add_argument(
      "--policy-weighted-score-predictions",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--policy-weighted-score-completed-trick-window", type=int, default=0
  )
  parser.add_argument("--min-belief-action-coverage", type=float, default=0.75)
  parser.add_argument("--stable-margin", type=float, default=0.20)
  parser.add_argument(
      "--require-actual-best-red-switch",
      action="store_true",
      help=(
          "Before expensive belief scoring, only audit candidate sets whose "
          "actual table/rollout ranker prefers a red-switch action."
      ),
  )
  parser.add_argument(
      "--require-actual-best-overrides-base",
      action="store_true",
      help=(
          "Before expensive belief scoring, only audit candidate sets whose "
          "actual table/rollout ranker prefers something other than the base "
          "policy action."
      ),
  )
  parser.add_argument(
      "--replay-out",
      default="",
      help=(
          "Optional .npz replay path. When set, export belief-stable rows as "
          "normal Quantum Cat replay examples for downstream distillation."
      ),
  )
  parser.add_argument(
      "--replay-metadata-out",
      default="",
      help=(
          "Optional JSON metadata path for exported replay rows. Defaults to "
          "<replay-out>.metadata.json when --replay-out is set."
      ),
  )
  parser.add_argument(
      "--replay-row-filter",
      choices=(
          "stable_red_overrides",
          "stable_overrides",
          "stable_base",
          "stable_red_or_base",
          "all_stable",
          "all",
      ),
      default="stable_red_overrides",
      help=(
          "Which audited rows become replay rows. The default only keeps stable "
          "belief labels that override the base action with a red switch."
      ),
  )
  parser.add_argument("--replay-target-temperature", type=float, default=0.35)
  parser.add_argument("--source-bot-mode", default="policy")
  parser.add_argument("--behavior-exploration-prob", type=float, default=0.0)
  parser.add_argument("--behavior-led-contrast-switch-prob", type=float, default=0.35)
  parser.add_argument("--behavior-red-switch-prob", type=float, default=0.90)
  parser.add_argument("--behavior-exploration-temperature", type=float, default=0.8)
  parser.add_argument("--torch-threads", type=int, default=1)
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
  return parser.parse_args()


def _source_args(args, value_scale: float):
  return SimpleNamespace(
      players=int(args.players),
      match_context=bool(args.match_context),
      bot_mode=str(args.source_bot_mode),
      value_scale=float(value_scale),
      enable_behavior_exploration=(
          float(args.behavior_exploration_prob) > 0.0
          or float(args.behavior_led_contrast_switch_prob) > 0.0
      ),
      behavior_exploration_prob=float(args.behavior_exploration_prob),
      behavior_led_contrast_switch_prob=float(args.behavior_led_contrast_switch_prob),
      behavior_red_switch_prob=float(args.behavior_red_switch_prob),
      behavior_exploration_temperature=float(args.behavior_exploration_temperature),
  )


def _belief_args(args, value_scale: float):
  return SimpleNamespace(
      players=int(args.players),
      value_scale=float(value_scale),
      belief_samples=max(1, int(args.belief_samples)),
      belief_source=str(args.belief_source),
      belief_candidates=max(int(args.belief_samples), int(args.belief_candidates)),
      belief_policy_temperature=float(args.belief_policy_temperature),
      belief_uniform_mix=float(args.belief_uniform_mix),
      belief_ref_policy_mix=str(args.belief_ref_policy_mix),
      belief_ref_policy_mix_by_phase=str(args.belief_ref_policy_mix_by_phase),
      score_mode=str(args.score_mode),
      rollout_actual_rollouts=max(1, int(args.rollout_actual_rollouts)),
      rollout_paradox_penalty=float(args.rollout_paradox_penalty),
      max_red_candidates=max(0, int(args.max_red_candidates)),
      max_follow_candidates=max(0, int(args.max_follow_candidates)),
      policy_weighted_score_candidates=int(args.policy_weighted_score_candidates),
      policy_weighted_action_aggregation=str(args.policy_weighted_action_aggregation),
      policy_weighted_score_predictions=bool(args.policy_weighted_score_predictions),
      policy_weighted_score_completed_trick_window=int(
          args.policy_weighted_score_completed_trick_window
      ),
      min_belief_action_coverage=float(args.min_belief_action_coverage),
      stable_margin=float(args.stable_margin),
      require_actual_best_red_switch=bool(args.require_actual_best_red_switch),
      require_actual_best_overrides_base=bool(args.require_actual_best_overrides_base),
      replay_target_temperature=float(args.replay_target_temperature),
  )


def _softmax_from_scores(scores: dict[int, float], temperature: float, num_actions: int):
  if not scores:
    return None
  actions = sorted(int(action) for action in scores)
  logits = np.asarray([float(scores[action]) for action in actions], dtype=np.float32)
  logits = logits / max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  exp = np.exp(logits)
  total = float(exp.sum())
  if total <= 0.0 or not math.isfinite(total):
    return None
  target = np.zeros(int(num_actions), dtype=np.float32)
  for action, prob in zip(actions, exp / total):
    target[int(action)] = float(prob)
  return target


def _average_belief_scores(scores_a, scores_b):
  common = sorted(set(int(action) for action in scores_a) & set(int(action) for action in scores_b))
  return {
      int(action): 0.5 * (
          float(scores_a[int(action)]) + float(scores_b[int(action)])
      )
      for action in common
  }


def _build_replay_row(
    state,
    player: int,
    legal,
    base_action: int,
    belief_best_action: int,
    features: np.ndarray,
    belief_scores_a,
    belief_scores_b,
    target_temperature: float,
):
  num_actions = int(state.num_distinct_actions())
  try:
    num_players = int(state.get_game().num_players())
  except AttributeError:
    num_players = int(getattr(state, "_num_players", 1))
  average_scores = _average_belief_scores(belief_scores_a, belief_scores_b)
  if int(base_action) not in average_scores or int(belief_best_action) not in average_scores:
    return None, None
  policy_target = _softmax_from_scores(
      average_scores, float(target_temperature), int(num_actions)
  )
  if policy_target is None:
    return None, None
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[[int(action) for action in legal]] = 1.0
  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  base_score = float(average_scores[int(base_action)])
  for action, score in average_scores.items():
    action_value_targets[int(action)] = float(score) - base_score
    action_value_mask[int(action)] = 1.0
  replay_row = (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      policy_target,
      np.zeros(num_players, dtype=np.float32),
      np.zeros(num_players, dtype=np.float32),
      int(np.argmax(policy_target)),
      int(player),
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      np.array(features, dtype=np.float32),
  )
  metadata = {
      "policy_best_action": int(base_action),
      "label_best_action": int(np.argmax(policy_target)),
      "belief_best_action": int(belief_best_action),
      "belief_average_scores": {
          str(action): float(score)
          for action, score in sorted(average_scores.items())
      },
      "belief_score_delta_vs_base": {
          str(action): float(score) - base_score
          for action, score in sorted(average_scores.items())
      },
      "replay_target_temperature": float(target_temperature),
      "labeled_actions": [int(action) for action in sorted(average_scores)],
  }
  return replay_row, metadata


def _include_replay_row(row, filter_name: str) -> bool:
  if str(filter_name) == "all":
    return True
  if str(filter_name) == "all_stable":
    return bool(row.get("belief_stable", False))
  if str(filter_name) == "stable_base":
    return bool(row.get("belief_stable", False)) and not bool(
        row.get("belief_best_overrides_base", False)
    )
  if str(filter_name) == "stable_red_or_base":
    if not bool(row.get("belief_stable", False)):
      return False
    if not bool(row.get("belief_best_overrides_base", False)):
      return True
    return bool(row.get("belief_best_flags", {}).get("red_switch", False))
  if str(filter_name) == "stable_overrides":
    return bool(row.get("belief_stable", False)) and bool(
        row.get("belief_best_overrides_base", False)
    )
  return (
      bool(row.get("belief_stable", False))
      and bool(row.get("belief_best_overrides_base", False))
      and bool(row.get("belief_best_flags", {}).get("red_switch", False))
  )


def _chance_action(state, rng: np.random.Generator) -> int:
  outcomes = state.chance_outcomes()
  actions, probs = zip(*outcomes)
  return int(rng.choice(actions, p=np.asarray(probs, dtype=np.float64)))


def _rank(scores: dict[int, float], policy: np.ndarray):
  if not scores:
    return None
  ordered = sorted(
      [int(action) for action in scores],
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
      "ordered_actions": ordered,
  }


def _filtered_scores(scores, coverages, min_coverage: float):
  return {
      int(action): float(score)
      for action, score in scores.items()
      if float(coverages.get(int(action), 0.0)) >= float(min_coverage)
  }


def _score_rollout_actions(
    state,
    player,
    candidates,
    model,
    rollout_args,
    device,
    value_scale: float,
    seed: int,
    samples: int,
    paradox_penalty: float,
    belief_sample: bool,
):
  np.random.seed(int(seed) % (2**32 - 1))
  action_scores = {int(action): [] for action in candidates}
  skipped_worlds = 0
  for sample_idx in range(max(1, int(samples))):
    if belief_sample:
      sampled = state.resample_from_infostate(player, np.random)
    else:
      sampled = state.clone()
    sampled_legal = set(int(action) for action in sampled.legal_actions(player))
    usable = [int(action) for action in candidates if int(action) in sampled_legal]
    if not usable:
      skipped_worlds += 1
      continue
    for action in usable:
      result = rollout_from_action(
          model,
          sampled,
          int(action),
          float(value_scale),
          device,
          int(seed) + sample_idx * 1000003 + int(action) * 9176,
          rollout_args,
      )
      raw_scores = result.get("raw_scores", [])
      actor_score = (
          float(raw_scores[int(player)])
          if int(player) < len(raw_scores) else 0.0
      )
      score = actor_score - float(paradox_penalty) * float(
          bool(result.get("any_paradox", False))
      )
      action_scores[int(action)].append(float(score))
  means = {}
  stderrs = {}
  coverages = {}
  for action, values in action_scores.items():
    coverages[int(action)] = float(len(values) / max(1, int(samples)))
    if not values:
      continue
    arr = np.asarray(values, dtype=np.float32)
    means[int(action)] = float(np.mean(arr))
    stderrs[int(action)] = (
        None if len(arr) <= 1 else float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    )
  return means, stderrs, coverages, {
      "belief_samples_requested": int(samples),
      "belief_samples_used": int(samples),
      "belief_worlds_skipped": int(skipped_worlds),
      "score_mode": "rollout_survival",
      "rollout_paradox_penalty": float(paradox_penalty),
      "belief_sample": bool(belief_sample),
  }


def _score_belief_split(
    state,
    player,
    candidates,
    model,
    args,
    rollout_args,
    device,
    value_scale,
    seed,
):
  if str(getattr(args, "score_mode", "table_liveness")) == "rollout_survival":
    scores, stderrs, coverages, meta = _score_rollout_actions(
        state,
        player,
        candidates,
        model,
        rollout_args,
        device,
        float(value_scale),
        int(seed),
        int(args.belief_samples),
        float(getattr(args, "rollout_paradox_penalty", 5.0)),
        belief_sample=True,
    )
    filtered = _filtered_scores(
        scores, coverages, float(args.min_belief_action_coverage)
    )
    return filtered, stderrs, coverages, meta
  np.random.seed(int(seed) % (2**32 - 1))
  scores, stderrs, coverages, meta = _score_belief_actions(
      state, player, candidates, model, args, device
  )
  filtered = _filtered_scores(
      scores, coverages, float(args.min_belief_action_coverage)
  )
  return filtered, stderrs, coverages, meta


def _action_flags(state, player, action: int, features: np.ndarray):
  red_switches = red_switch_mask(features)
  token_losses = token_loss_mask(features)
  return {
      "action": int(action),
      "action_string": action_string(state, player, int(action)),
      "follows_led": bool(
          features[int(action), ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
      ),
      "red": bool(features[int(action), ACTION_FEATURE_IS_RED_INDEX] > 0.5),
      "red_switch": bool(red_switches[int(action)]),
      "token_loss": bool(token_losses[int(action)]),
  }


def _cap_by_policy(actions, policy: np.ndarray, cap: int) -> list[int]:
  actions = [int(action) for action in actions]
  if int(cap) <= 0 or len(actions) <= int(cap):
    return sorted(actions)
  return sorted(
      actions,
      key=lambda action: (float(policy[int(action)]), -int(action)),
      reverse=True,
  )[:int(cap)]


def _audit_decision(
    decision,
    model,
    belief_args,
    rollout_args,
    device,
    value_scale,
    seed,
):
  state = decision["state"]
  player = int(decision["player"])
  if phase_name_for_state(state) != "play":
    return None
  led_color = getattr(state, "_led_color", None)
  if led_color not in ("B", "Y", "G"):
    return None
  trick_number = int(getattr(state, "_trick_number", -1))
  legal = [int(action) for action in state.legal_actions(player)]
  if len(legal) < 2:
    return None
  policy = np.asarray(decision["policy"], dtype=np.float32)
  base_action = int(max(legal, key=lambda action: (float(policy[action]), -action)))
  features = action_feature_matrix(state, player, state.num_distinct_actions())
  follows_led = features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
  red_switches = red_switch_mask(features)
  follow_actions = [action for action in legal if bool(follows_led[int(action)])]
  red_actions = [action for action in legal if bool(red_switches[int(action)])]
  if not follow_actions or not red_actions:
    return None
  if not bool(follows_led[int(base_action)]):
    return None
  follow_actions = _cap_by_policy(
      follow_actions, policy, int(getattr(belief_args, "max_follow_candidates", 0))
  )
  red_actions = _cap_by_policy(
      red_actions, policy, int(getattr(belief_args, "max_red_candidates", 0))
  )
  candidates = sorted(set([base_action, *follow_actions, *red_actions]))
  if len(candidates) < 2:
    return None

  if str(getattr(belief_args, "score_mode", "table_liveness")) == "rollout_survival":
    actual_scores, actual_stderrs, actual_coverages, actual_meta = _score_rollout_actions(
        state,
        player,
        candidates,
        model,
        rollout_args,
        device,
        float(value_scale),
        int(seed) + 99991,
        int(getattr(belief_args, "rollout_actual_rollouts", 2)),
        float(getattr(belief_args, "rollout_paradox_penalty", 5.0)),
        belief_sample=False,
    )
    actual_scores = _filtered_scores(
        actual_scores,
        actual_coverages,
        float(belief_args.min_belief_action_coverage),
    )
  else:
    actual_stderrs = {}
    actual_coverages = {int(action): 1.0 for action in candidates}
    actual_meta = {"score_mode": "table_liveness", "belief_sample": False}
    actual_scores = {
        int(action): _action_table_score(state, int(action))
        for action in candidates
    }
  actual_rank = _rank(actual_scores, policy)
  if actual_rank is None:
    return None
  actual_best = int(actual_rank["best_action"])
  if (
      bool(getattr(belief_args, "require_actual_best_overrides_base", False))
      and int(actual_best) == int(base_action)
  ):
    return None
  if (
      bool(getattr(belief_args, "require_actual_best_red_switch", False))
      and not bool(red_switches[int(actual_best)])
  ):
    return None

  belief_scores_a, stderrs_a, coverages_a, meta_a = _score_belief_split(
      state,
      player,
      candidates,
      model,
      belief_args,
      rollout_args,
      device,
      float(value_scale),
      int(seed),
  )
  belief_scores_b, stderrs_b, coverages_b, meta_b = _score_belief_split(
      state,
      player,
      candidates,
      model,
      belief_args,
      rollout_args,
      device,
      float(value_scale),
      int(seed) + 104729,
  )
  belief_rank_a = _rank(belief_scores_a, policy)
  belief_rank_b = _rank(belief_scores_b, policy)
  if belief_rank_a is None or belief_rank_b is None:
    return None

  best_a = int(belief_rank_a["best_action"])
  best_b = int(belief_rank_b["best_action"])
  stable = bool(best_a == best_b and float(belief_rank_a["margin"]) >= float(belief_args.stable_margin))
  candidate_flags = {
      str(action): _action_flags(state, player, action, features)
      for action in candidates
  }
  for action in candidates:
    candidate_flags[str(action)]["policy_prob"] = float(policy[int(action)])
    if int(action) in actual_scores:
      candidate_flags[str(action)]["actual_score"] = float(actual_scores[int(action)])
      candidate_flags[str(action)]["actual_coverage"] = float(
          actual_coverages.get(int(action), 0.0)
      )
      candidate_flags[str(action)]["actual_stderr"] = (
          None if actual_stderrs.get(int(action)) is None
          else float(actual_stderrs[int(action)])
      )
    if int(action) in belief_scores_a:
      candidate_flags[str(action)]["belief_score_a"] = float(belief_scores_a[int(action)])
      candidate_flags[str(action)]["belief_coverage_a"] = float(coverages_a.get(int(action), 0.0))
      candidate_flags[str(action)]["belief_stderr_a"] = (
          None if stderrs_a.get(int(action)) is None else float(stderrs_a[int(action)])
      )
    if int(action) in belief_scores_b:
      candidate_flags[str(action)]["belief_score_b"] = float(belief_scores_b[int(action)])
      candidate_flags[str(action)]["belief_coverage_b"] = float(coverages_b.get(int(action), 0.0))
      candidate_flags[str(action)]["belief_stderr_b"] = (
          None if stderrs_b.get(int(action)) is None else float(stderrs_b[int(action)])
      )

  replay_row, replay_metadata = _build_replay_row(
      state,
      int(player),
      legal,
      int(base_action),
      int(best_a),
      features,
      belief_scores_a,
      belief_scores_b,
      float(getattr(belief_args, "replay_target_temperature", 0.35)),
  )

  return {
      "player": int(player),
      "phase": "play",
      "trick_number": int(trick_number),
      "led_color": str(led_color),
      "legal_count": int(len(legal)),
      "candidate_count": int(len(candidates)),
      "red_candidate_count": int(len(red_actions)),
      "follow_candidate_count": int(len(follow_actions)),
      "base_action": int(base_action),
      "base_action_string": action_string(state, player, base_action),
      "actual_best_action": int(actual_best),
      "actual_best_action_string": action_string(state, player, actual_best),
      "belief_best_action_a": int(best_a),
      "belief_best_action_string_a": action_string(state, player, best_a),
      "belief_best_action_b": int(best_b),
      "belief_best_action_string_b": action_string(state, player, best_b),
      "belief_split_agree": bool(best_a == best_b),
      "belief_stable": bool(stable),
      "belief_matches_actual": bool(best_a == actual_best),
      "belief_best_overrides_base": bool(best_a != base_action),
      "actual_best_overrides_base": bool(actual_best != base_action),
      "belief_best_flags": candidate_flags[str(best_a)],
      "actual_best_flags": candidate_flags[str(actual_best)],
      "base_flags": candidate_flags[str(base_action)],
      "belief_margin_a": float(belief_rank_a["margin"]),
      "belief_margin_b": float(belief_rank_b["margin"]),
      "actual_margin": float(actual_rank["margin"]),
      "belief_scores_a": {str(k): float(v) for k, v in sorted(belief_scores_a.items())},
      "belief_scores_b": {str(k): float(v) for k, v in sorted(belief_scores_b.items())},
      "actual_scores": {str(k): float(v) for k, v in sorted(actual_scores.items())},
      "candidate_flags": candidate_flags,
      "actual_meta": actual_meta,
      "belief_meta_a": meta_a,
      "belief_meta_b": meta_b,
      "_replay_row": replay_row,
      "_replay_metadata": replay_metadata,
  }


def _summarize(rows):
  if not rows:
    return {"states": 0}

  def rate(pred):
    return float(sum(1 for row in rows if pred(row)) / max(1, len(rows)))

  stable_rows = [row for row in rows if bool(row["belief_stable"])]
  override_rows = [row for row in rows if bool(row["belief_best_overrides_base"])]
  stable_override_rows = [
      row for row in stable_rows if bool(row["belief_best_overrides_base"])
  ]
  stable_red_override_rows = [
      row for row in stable_override_rows
      if bool(row["belief_best_flags"]["red_switch"])
  ]
  return {
      "states": int(len(rows)),
      "belief_split_agree": int(sum(bool(row["belief_split_agree"]) for row in rows)),
      "belief_split_agree_rate": round(rate(lambda row: bool(row["belief_split_agree"])), 6),
      "belief_stable_rows": int(len(stable_rows)),
      "belief_stable_rate": round(float(len(stable_rows) / max(1, len(rows))), 6),
      "belief_overrides_base": int(len(override_rows)),
      "belief_override_rate": round(float(len(override_rows) / max(1, len(rows))), 6),
      "stable_belief_overrides_base": int(len(stable_override_rows)),
      "stable_belief_override_rate": round(
          float(len(stable_override_rows) / max(1, len(rows))), 6
      ),
      "stable_belief_red_overrides_base": int(len(stable_red_override_rows)),
      "stable_belief_red_override_rate": round(
          float(len(stable_red_override_rows) / max(1, len(rows))), 6
      ),
      "belief_matches_actual": int(sum(bool(row["belief_matches_actual"]) for row in rows)),
      "belief_matches_actual_rate": round(rate(lambda row: bool(row["belief_matches_actual"])), 6),
      "belief_best_red_switch": int(
          sum(bool(row["belief_best_flags"]["red_switch"]) for row in rows)
      ),
      "belief_best_red_switch_rate": round(
          rate(lambda row: bool(row["belief_best_flags"]["red_switch"])), 6
      ),
      "actual_best_red_switch": int(
          sum(bool(row["actual_best_flags"]["red_switch"]) for row in rows)
      ),
      "actual_best_red_switch_rate": round(
          rate(lambda row: bool(row["actual_best_flags"]["red_switch"])), 6
      ),
      "stable_belief_best_red_switch": int(
          sum(bool(row["belief_best_flags"]["red_switch"]) for row in stable_rows)
      ),
      "stable_belief_best_red_switch_rate": (
          None if not stable_rows else round(
              float(
                  sum(bool(row["belief_best_flags"]["red_switch"]) for row in stable_rows)
                  / len(stable_rows)
              ),
              6,
          )
      ),
      "mean_belief_margin_a": round(
          float(np.mean([row["belief_margin_a"] for row in rows])), 6
      ),
      "mean_actual_margin": round(
          float(np.mean([row["actual_margin"] for row in rows])), 6
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
  random.seed(int(args.seed))
  np.random.seed(int(args.seed) % (2**32 - 1))
  torch.manual_seed(int(args.seed))
  rng = np.random.default_rng(int(args.seed))
  device = _device(str(args.device))
  game_args = _model_args(args)
  game = make_game(game_args, int(args.initial_start))
  model, _, saved_args = load_model_payload(args.checkpoint, game, game_args, device)
  model.eval()
  value_scale = float(saved_args.get("value_scale", 20.0))
  source_args = _source_args(args, value_scale)
  belief_args = _belief_args(args, value_scale)
  rows = []
  replay_rows = []
  replay_metadata_rows = []
  replay_skipped = {
      "disabled": 0,
      "no_private_row": 0,
      "row_filter": 0,
  }
  visited_decisions = 0
  eligible_decisions = 0
  rounds = 0
  last_partial_rows = 0

  def replay_export_summary():
    metadata_out = str(args.replay_metadata_out)
    if str(args.replay_out) and not metadata_out:
      metadata_out = str(Path(args.replay_out).with_suffix(
          Path(args.replay_out).suffix + ".metadata.json"
      ))
    return {
        "enabled": bool(args.replay_out),
        "replay_out": str(args.replay_out),
        "metadata_out": metadata_out,
        "row_filter": str(args.replay_row_filter),
        "target_temperature": float(args.replay_target_temperature),
        "rows": int(len(replay_rows)),
        "skipped": dict(replay_skipped),
    }

  def write_snapshot(partial: bool):
    snapshot = {
        "kind": "belief_red_switch_probe",
        "partial": bool(partial),
        "checkpoint": str(args.checkpoint),
        "checkpoint_saved_args": dict(saved_args),
        "players": int(args.players),
        "matches_requested": int(args.matches),
        "rounds_visited": int(rounds),
        "seed": int(args.seed),
        "match_context": bool(args.match_context),
        "min_trick_number": int(args.min_trick_number),
        "max_trick_number": int(args.max_trick_number),
        "belief_source": str(args.belief_source),
        "score_mode": str(args.score_mode),
        "belief_samples": int(args.belief_samples),
        "belief_candidates": int(args.belief_candidates),
        "rollout_actual_rollouts": int(args.rollout_actual_rollouts),
        "rollout_paradox_penalty": float(args.rollout_paradox_penalty),
        "max_red_candidates": int(args.max_red_candidates),
        "max_follow_candidates": int(args.max_follow_candidates),
        "min_belief_action_coverage": float(args.min_belief_action_coverage),
        "stable_margin": float(args.stable_margin),
        "source_args": vars(source_args),
        "visited_decisions": int(visited_decisions),
        "eligible_decisions": int(eligible_decisions),
        "replay_export": replay_export_summary(),
        "summary": _summarize(rows),
        "rows": rows,
        "args": vars(args),
    }
    out = Path(args.out)
    path = (
        out.with_suffix(out.suffix + ".partial.json")
        if partial else
        out
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return snapshot, path
  for match_idx in range(int(args.matches)):
    if int(args.max_states) > 0 and len(rows) >= int(args.max_states):
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if int(args.max_states) > 0 and len(rows) >= int(args.max_states):
        break
      start_player = (int(args.initial_start) + round_idx) % int(args.players)
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": int(args.players),
              "start_player": int(start_player),
              "match_context": int(bool(args.match_context)),
          },
      )
      prior_totals = np.copy(totals)
      round_seed = int(args.seed) + match_idx * 1000003 + round_idx * 9973
      terminal, decisions, _first_new_paradox = play_round_with_history(
          model,
          game,
          seed=round_seed,
          value_scale=value_scale,
          device=device,
          match_totals=prior_totals if bool(args.match_context) else None,
          round_index=round_idx,
          args=source_args,
      )
      returns = np.array(terminal.returns(), dtype=np.float32)
      totals = returns if bool(args.match_context) else totals + returns
      rounds += 1
      for decision in decisions:
        visited_decisions += 1
        trick_number = int(decision.get("trick_number", -1))
        if trick_number < int(args.min_trick_number):
          continue
        if int(args.max_trick_number) >= 0 and trick_number > int(args.max_trick_number):
          continue
        audited = _audit_decision(
            decision,
            model,
            belief_args,
            source_args,
            device,
            float(value_scale),
            int(round_seed) + int(decision.get("index", 0)) * 15485863,
        )
        if audited is None:
          continue
        eligible_decisions += 1
        replay_row = audited.pop("_replay_row", None)
        replay_metadata = audited.pop("_replay_metadata", None)
        audited["match_index"] = int(match_idx)
        audited["round_index"] = int(round_idx)
        audited["decision_index"] = int(decision.get("index", -1))
        audited["round_seed"] = int(round_seed)
        if str(args.replay_out):
          if replay_row is None or replay_metadata is None:
            replay_skipped["no_private_row"] += 1
          elif not _include_replay_row(audited, str(args.replay_row_filter)):
            replay_skipped["row_filter"] += 1
          else:
            replay_metadata.update({
                "phase": str(audited["phase"]),
                "player": int(audited["player"]),
                "trick_number": int(audited["trick_number"]),
                "led_color": str(audited["led_color"]),
                "legal_count": int(audited["legal_count"]),
                "candidate_count": int(audited["candidate_count"]),
                "policy_best_action_string": str(audited["base_action_string"]),
                "label_best_action_string": str(
                    audited["belief_best_action_string_a"]
                ),
                "belief_stable": bool(audited["belief_stable"]),
                "belief_margin_a": float(audited["belief_margin_a"]),
                "belief_margin_b": float(audited["belief_margin_b"]),
                "belief_best_flags": audited["belief_best_flags"],
                "base_flags": audited["base_flags"],
                "candidate_flags": audited["candidate_flags"],
                "match_index": int(match_idx),
                "round_index": int(round_idx),
                "decision_index": int(decision.get("index", -1)),
                "round_seed": int(round_seed),
            })
            replay_rows.append(replay_row)
            replay_metadata_rows.append(replay_metadata)
        else:
          replay_skipped["disabled"] += 1
        rows.append(audited)
        interval = int(getattr(args, "partial_save_interval_states", 0) or 0)
        if interval > 0 and len(rows) - int(last_partial_rows) >= interval:
          write_snapshot(partial=True)
          last_partial_rows = len(rows)
        if int(args.max_states) > 0 and len(rows) >= int(args.max_states):
          break

  result, out = write_snapshot(partial=False)
  replay_metadata_path = None
  if str(args.replay_out):
    replay_metadata_path = (
        Path(args.replay_metadata_out)
        if str(args.replay_metadata_out)
        else Path(args.replay_out).with_suffix(
            Path(args.replay_out).suffix + ".metadata.json"
        )
    )
    replay_meta = {
        "kind": "belief_red_switch_replay_metadata",
        "source_probe_out": str(out),
        "checkpoint": str(args.checkpoint),
        "players": int(args.players),
        "seed": int(args.seed),
        "row_filter": str(args.replay_row_filter),
        "target_temperature": float(args.replay_target_temperature),
        "rows": replay_metadata_rows,
        "summary": replay_export_summary(),
    }
    replay_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    replay_metadata_path.write_text(
        json.dumps(replay_meta, indent=2, sort_keys=True) + "\n"
    )
    if replay_rows:
      save_replay(
          replay_rows,
          args.replay_out,
          metadata={
              "kind": "belief_red_switch_replay",
              "source_probe_out": str(out),
              "metadata_out": str(replay_metadata_path),
              "summary": replay_export_summary(),
              "rows": replay_metadata_rows,
          },
      )
  print(json.dumps({
      "out": str(out),
      "summary": result["summary"],
      "visited_decisions": int(visited_decisions),
      "eligible_decisions": int(eligible_decisions),
      "replay_export": replay_export_summary(),
      "replay_metadata_out": (
          None if replay_metadata_path is None else str(replay_metadata_path)
      ),
  }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
