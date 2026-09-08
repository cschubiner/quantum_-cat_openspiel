#!/usr/bin/env python3
"""Mine DAgger-style calibration rows from a live improve-graft policy."""

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

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import (
    load_model_payload,
    make_game,
    model_policy_value,
    save_replay,
)
from quantum_cat_full_match_elo import AZImproveGraftBot, load_improve_scorer
from quantum_cat_label_oracle_audit import (
    accepted_replay_example,
    accepted_replay_summary,
    action_string,
    audit_namespace,
    calibration_row_class,
    calibration_summary,
    calibration_target_action,
    evaluate_actions_with_stability,
    model_args_for,
    phase_name_for_state,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  proposal_source = parser.add_mutually_exclusive_group(required=True)
  proposal_source.add_argument("--improve-model")
  proposal_source.add_argument(
      "--proposal-checkpoint",
      help=(
          "Use another policy checkpoint as the proposal source. The base "
          "checkpoint remains the deployment policy and rollout continuation."
      ),
  )
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--states", type=int, default=48)
  parser.add_argument("--max-matches", type=int, default=96)
  parser.add_argument("--max-states-per-match", type=int, default=2)
  parser.add_argument("--phases", default="play")
  parser.add_argument("--improve-threshold", type=float, default=0.99999)
  parser.add_argument("--improve-risk-threshold", type=float, default=0.0)
  parser.add_argument(
      "--improve-base-mode",
      choices=("policy", "q_policy", "liveness_shield"),
      default="policy",
      help="Base action the improve model must beat before it becomes a proposal.",
  )
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-root-only", action="store_true")
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument("--action-paradox-root-only", action="store_true")
  parser.add_argument(
      "--action-paradox-rerank-mode",
      choices=("additive", "threshold", "relative", "relative_dominance"),
      default="additive",
  )
  parser.add_argument("--action-paradox-risk-threshold", type=float, default=0.0)
  parser.add_argument("--action-paradox-min-risk-margin", type=float, default=0.0)
  parser.add_argument(
      "--action-paradox-rerank-led-contrast-only",
      action="store_true",
  )
  parser.add_argument("--action-feasibility-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-max-policy-log-gap", type=float, default=2.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument("--action-value-rerank-phases", default="")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  parser.add_argument("--action-value-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument("--improve-max-actions", type=int, default=12)
  parser.add_argument("--improve-top-policy", type=int, default=4)
  parser.add_argument(
      "--proposal-audit-top-k",
      type=int,
      default=1,
      help=(
          "Audit this many top improve-model candidates per collected state, "
          "plus the configured base action. Values >1 create same-state "
          "hard negatives for pairwise/legal-set training."
      ),
  )
  parser.add_argument("--improve-min-legal-count", type=int, default=0)
  parser.add_argument("--improve-max-legal-count", type=int, default=0)
  parser.add_argument("--improve-min-trick-number", type=int, default=-1)
  parser.add_argument("--improve-trick-numbers", default="")
  parser.add_argument("--improve-led-colors", default="")
  parser.add_argument("--improve-candidate-colors", default="")
  parser.add_argument("--improve-require-base-off-led", action="store_true")
  parser.add_argument("--improve-require-responsible-red-win", action="store_true")
  parser.add_argument("--improve-guard-led-follow", action="store_true")
  parser.add_argument(
      "--improve-guard-led-follow-red-win-must-be-urgent",
      action="store_true",
  )
  parser.add_argument("--improve-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument(
      "--proposal-trick-quotas",
      default="",
      help=(
          "Comma-separated collection quotas by trick number, e.g. "
          "'4:32,5:32,6:32,7:32'. When set, only proposals from listed tricks "
          "are collected and collection stops when every quota is filled."
      ),
  )
  parser.add_argument("--improve-min-match-round", type=int, default=-1)
  parser.add_argument("--improve-sample-limit", type=int, default=20)
  parser.add_argument(
      "--improve-include-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument(
      "--improve-feature-candidates",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--audit-rollouts", type=int, default=1)
  parser.add_argument("--stability-audit-passes", type=int, default=1)
  parser.add_argument("--stability-seed-stride", type=int, default=1000003)
  parser.add_argument(
      "--stability-min-positive-sign-rate", type=float, default=1.0
  )
  parser.add_argument(
      "--stability-min-safe-risk-rate", type=float, default=1.0
  )
  parser.add_argument(
      "--unstable-positive-as-negative",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Fail closed by teaching the base action when an aggregate POSITIVE "
          "candidate does not repeat across independent audit passes."
      ),
  )
  parser.add_argument("--belief-samples", type=int, default=1)
  parser.add_argument("--belief-candidates", type=int, default=4)
  parser.add_argument("--belief-policy-temperature", type=float, default=1.0)
  parser.add_argument("--belief-uniform-mix", type=float, default=0.15)
  parser.add_argument(
      "--belief-ref-policy-mix",
      default="model:0.65,heuristic:0.25,uniform:0.10",
  )
  parser.add_argument("--belief-ref-policy-mix-by-phase", default="")
  parser.add_argument("--full-match-rollout", action="store_true")
  parser.add_argument(
      "--counterfactual-action-paradox-scope",
      choices=("acting", "any"),
      default="any",
      help="Use any-player risk by default to match the promotion gate.",
  )
  parser.add_argument("--rollout-max-plies", type=int, default=0)
  parser.add_argument("--match-context", action="store_true", default=True)
  parser.add_argument(
      "--shadow-run119-actions",
      "--shadow-base-actions",
      dest="shadow_run119_actions",
      action="store_true",
      help="Collect would-overrides while applying the configured base actions.",
  )
  parser.add_argument("--liveness-shield-base-mode", default="q_policy")
  parser.add_argument("--liveness-shield-phases", default="")
  parser.add_argument("--liveness-shield-min-trick-number", type=int, default=-1)
  parser.add_argument("--liveness-shield-led-colors", default="")
  parser.add_argument("--liveness-shield-min-open-slot-delta", type=int, default=1)
  parser.add_argument("--liveness-shield-min-public-damage-delta", type=int, default=2)
  parser.add_argument("--liveness-shield-exact-legal-pressure", action="store_true")
  parser.add_argument("--liveness-shield-exact-forced-only", action="store_true")
  parser.add_argument("--liveness-shield-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument("--liveness-shield-oracle-min-margin", type=int, default=-1)
  parser.add_argument("--calibration-positive-margin", type=float, default=0.5)
  parser.add_argument("--calibration-max-positive-paradox-delta", type=float, default=0.0)
  parser.add_argument("--seed", type=int, default=20262106)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument("--calibration-replay-out", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--progress-interval", type=int, default=0)
  return parser.parse_args()


def _oracle_cli_namespace(args):
  return SimpleNamespace(
      players=args.players,
      states=args.states,
      max_matches=args.max_matches,
      max_states_per_match=args.max_states_per_match,
      state_selection="proposal",
      state_pool_per_match=0,
      hard_entropy_weight=1.0,
      hard_loss_weight=0.25,
      hard_paradox_weight=0.75,
      seed=args.seed,
      match_context=bool(args.match_context),
      belief_samples=args.belief_samples,
      belief_candidates=args.belief_candidates,
      belief_policy_temperature=args.belief_policy_temperature,
      belief_uniform_mix=args.belief_uniform_mix,
      belief_ref_policy_mix=args.belief_ref_policy_mix,
      belief_ref_policy_mix_by_phase=args.belief_ref_policy_mix_by_phase,
      label_rollouts=1,
      audit_rollouts=args.audit_rollouts,
      stability_min_positive_sign_rate=float(
          args.stability_min_positive_sign_rate
      ),
      stability_min_safe_risk_rate=float(args.stability_min_safe_risk_rate),
      phases=args.phases,
      label_max_actions=args.improve_max_actions,
      label_top_policy=args.improve_top_policy,
      label_include_bots=args.improve_include_bots,
      label_feature_candidates=args.improve_feature_candidates,
      min_label_margin=0.0,
      label_paradox_penalty=0.0,
      label_max_paradox_delta=None,
      min_audit_margin=0.0,
      pass_min_score_rate=0.54,
      pass_min_avg_diff=0.0,
      pass_max_paradox_delta=0.02,
      full_match_rollout=bool(args.full_match_rollout),
      counterfactual_action_paradox_scope=str(
          args.counterfactual_action_paradox_scope
      ),
      rollout_max_plies=args.rollout_max_plies,
  )


def _improve_args(args, value_scale):
  return SimpleNamespace(
      players=args.players,
      value_scale=float(value_scale),
      match_context=bool(args.match_context),
      improve_model=args.improve_model,
      improve_base_mode=str(args.improve_base_mode),
      action_value_selection_weight=float(args.action_value_selection_weight),
      action_value_root_only=bool(args.action_value_root_only),
      action_paradox_selection_penalty=float(args.action_paradox_selection_penalty),
      action_paradox_root_only=bool(args.action_paradox_root_only),
      action_paradox_rerank_mode=str(args.action_paradox_rerank_mode),
      action_paradox_risk_threshold=float(args.action_paradox_risk_threshold),
      action_paradox_min_risk_margin=float(args.action_paradox_min_risk_margin),
      action_paradox_rerank_led_contrast_only=bool(
          args.action_paradox_rerank_led_contrast_only
      ),
      action_feasibility_selection_weight=float(
          args.action_feasibility_selection_weight
      ),
      action_paradox_max_policy_log_gap=float(args.action_paradox_max_policy_log_gap),
      action_value_rerank_clip=float(args.action_value_rerank_clip),
      action_value_rerank_phases=str(args.action_value_rerank_phases),
      action_value_rerank_min_margin=float(args.action_value_rerank_min_margin),
      action_value_max_policy_log_gap=float(args.action_value_max_policy_log_gap),
      improve_threshold=float(args.improve_threshold),
      improve_risk_threshold=float(args.improve_risk_threshold),
      improve_phases=args.phases,
      improve_max_actions=int(args.improve_max_actions),
      improve_top_policy=int(args.improve_top_policy),
      improve_include_bots=args.improve_include_bots,
      improve_feature_candidates=bool(args.improve_feature_candidates),
      improve_min_legal_count=int(args.improve_min_legal_count),
      improve_max_legal_count=int(args.improve_max_legal_count),
      improve_min_trick_number=int(args.improve_min_trick_number),
      improve_trick_numbers=str(args.improve_trick_numbers),
      improve_led_colors=str(args.improve_led_colors),
      improve_candidate_colors=str(args.improve_candidate_colors),
      improve_require_base_off_led=bool(args.improve_require_base_off_led),
      improve_require_responsible_red_win=bool(
          args.improve_require_responsible_red_win
      ),
      improve_min_match_round=int(args.improve_min_match_round),
      improve_guard_led_follow=bool(args.improve_guard_led_follow),
      improve_guard_led_follow_red_win_must_be_urgent=bool(
          args.improve_guard_led_follow_red_win_must_be_urgent
      ),
      improve_max_policy_log_gap=float(args.improve_max_policy_log_gap),
      improve_shadow_only=False,
      improve_sample_limit=int(args.improve_sample_limit),
      liveness_shield_base_mode=str(args.liveness_shield_base_mode),
      liveness_shield_phases=str(args.liveness_shield_phases),
      liveness_shield_min_trick_number=int(args.liveness_shield_min_trick_number),
      liveness_shield_led_colors=str(args.liveness_shield_led_colors),
      liveness_shield_min_open_slot_delta=int(args.liveness_shield_min_open_slot_delta),
      liveness_shield_min_public_damage_delta=int(
          args.liveness_shield_min_public_damage_delta
      ),
      liveness_shield_exact_legal_pressure=bool(
          args.liveness_shield_exact_legal_pressure
      ),
      liveness_shield_exact_forced_only=bool(args.liveness_shield_exact_forced_only),
      liveness_shield_max_policy_log_gap=float(args.liveness_shield_max_policy_log_gap),
      liveness_shield_oracle_min_margin=int(args.liveness_shield_oracle_min_margin),
  )


def _set_seed(seed):
  random.seed(int(seed))
  np.random.seed(int(seed) % (2**32 - 1))
  torch.manual_seed(int(seed))


def _parse_trick_quotas(spec):
  quotas = {}
  if not str(spec or "").strip():
    return quotas
  for raw_part in str(spec).split(","):
    part = raw_part.strip()
    if not part:
      continue
    if ":" in part:
      key, value = part.split(":", 1)
    elif "=" in part:
      key, value = part.split("=", 1)
    else:
      raise ValueError(
          f"Invalid --proposal-trick-quotas entry {part!r}; expected TRICK:COUNT"
      )
    trick = int(key.strip())
    count = int(value.strip())
    if count < 0:
      raise ValueError("--proposal-trick-quotas counts must be nonnegative")
    if count == 0:
      continue
    quotas[str(trick)] = int(quotas.get(str(trick), 0)) + count
  return quotas


def _collection_target(args, quotas):
  if quotas:
    return int(sum(int(value) for value in quotas.values()))
  return int(args.states)


def _quota_all_filled(quotas, collected_by_trick):
  if not quotas:
    return False
  for trick, wanted in quotas.items():
    if int(collected_by_trick.get(str(trick), 0)) < int(wanted):
      return False
  return True


def _proposal_quota_open(quotas, collected_by_trick, trick_key):
  if not quotas:
    return True
  if str(trick_key) not in quotas:
    return False
  return int(collected_by_trick.get(str(trick_key), 0)) < int(quotas[str(trick_key)])


def _policy_action(model, oracle_args, device, state, player):
  policy, _ = model_policy_value(
      model,
      state,
      player,
      state.num_distinct_actions(),
      oracle_args.value_scale,
      device,
  )
  legal = state.legal_actions(player)
  return policy, int(max(legal, key=lambda action: policy[action]))


def _proposal_from_policy_disagreement(
    base_policy, proposal_policy, legal, base_action, top_k
):
  """Return a proposal checkpoint's top disagreement and ranked slate."""
  legal = [int(action) for action in legal]
  base_action = int(base_action)
  if len(legal) <= 1:
    return base_action, "single_legal", []
  proposal_best = int(
      max(legal, key=lambda action: float(proposal_policy[int(action)]))
  )
  ranked = sorted(
      (action for action in legal if action != base_action),
      key=lambda action: (
          float(proposal_policy[int(action)]),
          float(base_policy[int(action)]),
          -int(action),
      ),
      reverse=True,
  )
  candidate_records = [
      {
          "action": int(action),
          # Keep the established proposal-context names so existing pairwise
          # and listwise audit tooling can consume policy-checkpoint sources.
          "improve": float(proposal_policy[int(action)]),
          "safe": 1.0,
          "policy_prob": float(base_policy[int(action)]),
          "proposal_checkpoint_prob": float(proposal_policy[int(action)]),
      }
      for action in ranked[:max(1, int(top_k))]
  ]
  if proposal_best == base_action:
    return base_action, "policy_agrees", candidate_records
  return proposal_best, "proposal", candidate_records


def _improve_base_action(improve_bot, state, player, raw_policy_best):
  legal = list(state.legal_actions(player))
  try:
    action = int(
        improve_bot._base_action(  # pylint: disable=protected-access
            state.clone(), int(player), legal, int(raw_policy_best)
        )
    )
  except Exception:
    action = int(raw_policy_best)
  if action not in legal:
    return int(raw_policy_best)
  return action


def _proposal_from_cached_base(
    improve_bot, args, state, player, policy, base_action
):
  legal = list(state.legal_actions(player))
  if len(legal) <= 1:
    return int(base_action), "single_legal", []
  enabled, disabled_reason = improve_bot._enabled_for_state(  # pylint: disable=protected-access
      state, len(legal)
  )
  if not enabled:
    return int(base_action), f"disabled:{disabled_reason}", []
  candidates = improve_bot._candidate_actions(  # pylint: disable=protected-access
      state, player, legal, policy, int(base_action)
  )
  candidates = improve_bot._guard_led_follow_candidates(  # pylint: disable=protected-access
      state, player, int(base_action), candidates
  )
  candidates = improve_bot._guard_candidate_shape(  # pylint: disable=protected-access
      state, player, int(base_action), candidates
  )
  candidates = improve_bot._guard_policy_gap_candidates(  # pylint: disable=protected-access
      policy, int(base_action), candidates
  )
  scores = improve_bot._score_candidates(  # pylint: disable=protected-access
      state, player, legal, policy, int(base_action), candidates
  )
  if not scores:
    return int(base_action), "no_scores", []
  ranked = sorted(
      scores.items(),
      key=lambda item: (
          float(item[1]["improve"]),
          float(item[1]["safe"]),
          float(policy[item[0]]),
      ),
      reverse=True,
  )
  top_k = max(1, int(getattr(args, "proposal_audit_top_k", 1) or 1))
  candidate_records = [
      {
          "action": int(action),
          "improve": float(score_row["improve"]),
          "safe": float(score_row["safe"]),
          "policy_prob": float(policy[int(action)]),
      }
      for action, score_row in ranked[:top_k]
  ]
  best_action, best_scores = ranked[0]
  threshold = float(args.improve_threshold)
  if float(best_scores["improve"]) < threshold:
    return int(base_action), "below_threshold", candidate_records
  risk_threshold = float(args.improve_risk_threshold)
  if risk_threshold > 0 and improve_bot.improve_scorer.get("risk_model") is None:
    return int(base_action), "risk_model_missing", candidate_records
  if risk_threshold > 0 and float(best_scores["safe"]) < risk_threshold:
    return int(base_action), "risk_veto", candidate_records
  return int(best_action), "proposal", candidate_records


def _candidate_audit_class(margin, paradox_delta, positive_margin, max_delta):
  if margin is None:
    return "UNAUDITED"
  if float(margin) <= 0.0:
    return "NEGATIVE"
  if paradox_delta is None:
    return "NEAR_POSITIVE"
  if (
      float(margin) >= float(positive_margin)
      and float(paradox_delta) <= float(max_delta)
  ):
    return "POSITIVE"
  return "NEAR_POSITIVE"


def _candidate_target_from_audit(audit, base_action, candidate_actions, args):
  base_stats = audit.get(int(base_action), {})
  base_mean = base_stats.get("mean")
  base_paradox = base_stats.get("mean_paradox")
  evaluations = []
  for action in candidate_actions:
    action = int(action)
    stats = audit.get(action, {})
    action_mean = stats.get("mean")
    action_paradox = stats.get("mean_paradox")
    margin = (
        None if action_mean is None or base_mean is None
        else float(action_mean - base_mean)
    )
    paradox_delta = (
        None if action_paradox is None or base_paradox is None
        else float(action_paradox - base_paradox)
    )
    row_class = _candidate_audit_class(
        margin,
        paradox_delta,
        float(args.calibration_positive_margin),
        float(args.calibration_max_positive_paradox_delta),
    )
    stability = dict(stats.get("candidate_stability") or {})
    if (
        row_class == "POSITIVE"
        and int(getattr(args, "stability_audit_passes", 1)) > 1
        and not bool(stability.get("stable_positive", False))
    ):
      row_class = (
          "NEGATIVE"
          if bool(getattr(args, "unstable_positive_as_negative", True))
          else "NEAR_POSITIVE"
      )
    evaluations.append({
        "action": int(action),
        "audit_margin_vs_policy": margin,
        "audit_paradox_delta_vs_policy": paradox_delta,
        "calibration_class": row_class,
        "stability": stability,
    })
  viable = [
      item for item in evaluations
      if item["calibration_class"] == "POSITIVE"
  ]
  if not viable:
    return int(base_action), "NEGATIVE", evaluations
  best = max(
      viable,
      key=lambda item: (
          float(item["audit_margin_vs_policy"] or -1e9),
          -float(item["audit_paradox_delta_vs_policy"] or 0.0),
          -int(item["action"]),
      ),
  )
  return int(best["action"]), str(best["calibration_class"]), evaluations


def annotate_candidate_stability(audit, base_action, candidate_actions, args):
  """Attach per-candidate repeatability evidence to an aggregated audit."""
  passes = max(1, int(getattr(args, "stability_audit_passes", 1)))
  base_stats = audit.get(int(base_action), {})
  base_means = list(base_stats.get("pass_means") or [])
  base_paradoxes = list(base_stats.get("pass_paradoxes") or [])
  for action in candidate_actions:
    stats = audit.get(int(action), {})
    if passes <= 1:
      stats["candidate_stability"] = {
          "passes": 1,
          "valid_passes": int(stats.get("mean") is not None),
          "stable_positive": True,
      }
      continue
    action_means = list(stats.get("pass_means") or [])
    action_paradoxes = list(stats.get("pass_paradoxes") or [])
    valid = min(
        len(base_means), len(action_means),
        len(base_paradoxes), len(action_paradoxes),
    )
    margins = [
        float(action_means[idx] - base_means[idx]) for idx in range(valid)
    ]
    paradox_deltas = [
        float(action_paradoxes[idx] - base_paradoxes[idx])
        for idx in range(valid)
    ]
    positive_sign_rate = (
        None if not margins
        else float(np.mean([margin > 0.0 for margin in margins]))
    )
    safe_risk_rate = (
        None if not paradox_deltas
        else float(np.mean([
            delta <= float(args.calibration_max_positive_paradox_delta)
            for delta in paradox_deltas
        ]))
    )
    stable = bool(
        valid == passes
        and positive_sign_rate is not None
        and safe_risk_rate is not None
        and positive_sign_rate
        >= float(getattr(args, "stability_min_positive_sign_rate", 1.0))
        and safe_risk_rate
        >= float(getattr(args, "stability_min_safe_risk_rate", 1.0))
    )
    stats["candidate_stability"] = {
        "passes": int(passes),
        "valid_passes": int(valid),
        "margins": margins,
        "paradox_deltas": paradox_deltas,
        "positive_sign_rate": positive_sign_rate,
        "safe_risk_rate": safe_risk_rate,
        "stable_positive": stable,
    }
  return audit


def proposal_calibration_fields(
    proposed_action,
    proposed_action_string,
    proposal_class,
    audit_margin,
    paradox_delta,
):
  """Returns pair labels for scoring the actual live proposal."""
  return {
      "label_best_action": int(proposed_action),
      "label_best_action_string": str(proposed_action_string),
      "calibration_class": str(proposal_class),
      "false_positive_for_positive_gate": bool(
          str(proposal_class) != "POSITIVE"
      ),
      "audit_margin_vs_policy": audit_margin,
      "audit_paradox_delta_vs_policy": paradox_delta,
  }


def collect_proposals(
    model,
    improve_bot,
    proposal_model,
    oracle_args,
    args,
    device,
    wanted,
    quotas,
):
  proposals = []
  matches = 0
  collection_started = time.perf_counter()
  target_states = _collection_target(args, quotas)
  stats = {
      "states_seen": 0,
      "wanted_phase_states": 0,
      "proposal_candidates": 0,
      "proposals_collected": 0,
      "proposal_trick_quotas": quotas,
      "disabled_or_abstained": {},
      "proposal_skip_reasons": {},
      "proposal_candidates_by_trick": {},
      "proposals_collected_by_trick": {},
  }
  while len(proposals) < target_states and matches < args.max_matches:
    if _quota_all_filled(quotas, stats["proposals_collected_by_trick"]):
      break
    initial_start = matches % args.players
    totals = np.zeros(args.players, dtype=np.float32)
    collected_this_match = 0
    for round_idx in range(args.players):
      game = make_game(oracle_args, (initial_start + round_idx) % args.players)
      state = game.new_initial_state()
      if args.match_context:
        state.set_match_context(totals, round_idx)
      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(int(np.random.choice(actions, p=probs)))
          continue
        stats["states_seen"] += 1
        player = int(state.current_player())
        phase = phase_name_for_state(state)
        policy, raw_policy_best = _policy_action(
            model, oracle_args, device, state, player
        )
        if proposal_model is not None:
          proposal_policy, _ = _policy_action(
              proposal_model, oracle_args, device, state, player
          )
          base_action = int(raw_policy_best)
          proposed, proposal_reason, candidate_records = (
              _proposal_from_policy_disagreement(
                  policy,
                  proposal_policy,
                  state.legal_actions(player),
                  base_action,
                  args.proposal_audit_top_k,
              )
          )
          del proposal_policy
        else:
          base_action = _improve_base_action(
              improve_bot, state, player, raw_policy_best
          )
          proposed, proposal_reason, candidate_records = _proposal_from_cached_base(
              improve_bot, args, state.clone(), player, policy, base_action
          )
        del policy
        if phase in wanted:
          stats["wanted_phase_states"] += 1
        trick_key = str(int(getattr(state, "_trick_number", -1)))
        if proposal_reason == "proposal":
          stats["proposal_candidates"] += 1
          by_trick = stats["proposal_candidates_by_trick"]
          by_trick[trick_key] = int(by_trick.get(trick_key, 0)) + 1
        else:
          disabled = stats["disabled_or_abstained"]
          disabled[proposal_reason] = int(disabled.get(proposal_reason, 0)) + 1
        cap_open = (
            args.max_states_per_match <= 0
            or collected_this_match < args.max_states_per_match
        )
        quota_open = _proposal_quota_open(
            quotas, stats["proposals_collected_by_trick"], trick_key
        )
        if phase in wanted and proposed != base_action and cap_open and quota_open:
          proposals.append({
              "state": state.clone(),
              "player": int(player),
              "phase": phase,
              "match_index": int(matches),
              "initial_start_player": int(initial_start),
              "round_index": int(round_idx),
              "match_totals": [float(x) for x in totals],
              "policy_best_action": int(base_action),
              "raw_policy_best_action": int(raw_policy_best),
              "proposed_action": int(proposed),
              "proposal_candidates": candidate_records,
              "legal_count": int(len(state.legal_actions(player))),
              "shadow_run119_actions": bool(args.shadow_run119_actions),
          })
          collected_this_match += 1
          stats["proposals_collected"] = len(proposals)
          collected_by_trick = stats["proposals_collected_by_trick"]
          collected_by_trick[trick_key] = int(collected_by_trick.get(trick_key, 0)) + 1
          if args.progress_interval > 0 and (
              len(proposals) % args.progress_interval == 0
              or len(proposals) == target_states
          ):
            print(json.dumps({
                "collected_proposals": int(len(proposals)),
                "target_proposals": int(target_states),
                "matches_started": int(matches + 1),
                "collected_by_trick": stats["proposals_collected_by_trick"],
                "elapsed_sec": round(time.perf_counter() - collection_started, 3),
            }), flush=True)
          if len(proposals) >= target_states or _quota_all_filled(
              quotas, stats["proposals_collected_by_trick"]
          ):
            break
        elif phase in wanted and proposed != base_action:
          skip_reasons = stats["proposal_skip_reasons"]
          if not cap_open:
            skip_key = "match_cap"
          elif not quota_open:
            skip_key = f"trick_quota:{trick_key}"
          else:
            skip_key = "unknown"
          skip_reasons[skip_key] = int(skip_reasons.get(skip_key, 0)) + 1
        state.apply_action(base_action if args.shadow_run119_actions else proposed)
      raw_scores = (
          state.raw_round_scores()
          if hasattr(state, "raw_round_scores")
          else state.returns()
      )
      totals += np.array(raw_scores, dtype=np.float32)
      if len(proposals) >= target_states or _quota_all_filled(
          quotas, stats["proposals_collected_by_trick"]
      ):
        break
    matches += 1
  return proposals, matches, stats


def main():
  args = parse_args()
  _set_seed(args.seed)
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  load_game = make_game(SimpleNamespace(players=args.players, match_context=True), 0)
  model, _, saved_args = load_model_payload(
      args.checkpoint, load_game, model_args_for(args.players), device
  )
  model.eval()
  oracle_args = audit_namespace(_oracle_cli_namespace(args), saved_args)
  improve_bot = None
  proposal_model = None
  if args.proposal_checkpoint:
    proposal_model, _, _ = load_model_payload(
        args.proposal_checkpoint,
        load_game,
        model_args_for(args.players),
        device,
    )
    proposal_model.eval()
  else:
    scorer = load_improve_scorer(args.improve_model, device)
    improve_bot = AZImproveGraftBot(
        model,
        scorer,
        "improve_graft_miner",
        device,
        _improve_args(args, oracle_args.value_scale),
    )
  wanted = {part.strip() for part in args.phases.split(",") if part.strip()}
  proposal_trick_quotas = _parse_trick_quotas(args.proposal_trick_quotas)
  proposal_source_mode = (
      "policy_checkpoint" if proposal_model is not None else "improve_graft"
  )
  proposal_source_id = (
      f"{proposal_source_mode}_"
      f"{'shadow_' if args.shadow_run119_actions else ''}seed{args.seed}"
  )
  collection_target = _collection_target(args, proposal_trick_quotas)
  started = time.perf_counter()
  proposals, matches_used, collection_stats = collect_proposals(
      model,
      improve_bot,
      proposal_model,
      oracle_args,
      args,
      device,
      wanted,
      proposal_trick_quotas,
  )

  rows = []
  replay = []
  metadata = []
  for idx, item in enumerate(proposals):
    state = item["state"]
    player = int(item["player"])
    proposed = int(item["proposed_action"])
    policy_best = int(item["policy_best_action"])
    raw_policy_best = int(item.get("raw_policy_best_action", policy_best))
    proposal_candidates = [
        int(row["action"])
        for row in item.get("proposal_candidates", [])
        if int(row.get("action", policy_best)) != int(policy_best)
    ]
    if proposed not in proposal_candidates:
      proposal_candidates.insert(0, int(proposed))
    proposal_candidates = list(dict.fromkeys(proposal_candidates))
    audited_actions = list(dict.fromkeys(proposal_candidates + [int(policy_best)]))
    audit_seed = int(args.seed) + 3000001 + idx * 104729
    round_game = make_game(oracle_args, int(getattr(state, "_round_start_player", 0)))
    audit, primary_stability = evaluate_actions_with_stability(
        state,
        player,
        audited_actions,
        round_game,
        model,
        oracle_args,
        device,
        args.audit_rollouts,
        base_seed=audit_seed,
        passes=args.stability_audit_passes,
        seed_stride=args.stability_seed_stride,
        label_action=proposed,
        policy_action=policy_best,
        safe_paradox_delta=args.calibration_max_positive_paradox_delta,
    )
    audit = annotate_candidate_stability(
        audit, policy_best, proposal_candidates, args
    )
    proposed_mean = audit[int(proposed)]["mean"]
    policy_mean = audit[int(policy_best)]["mean"]
    proposed_paradox = audit[int(proposed)].get("mean_paradox")
    policy_paradox = audit[int(policy_best)].get("mean_paradox")
    audit_margin = (
        None if proposed_mean is None or policy_mean is None
        else float(proposed_mean - policy_mean)
    )
    paradox_delta = (
        None if proposed_paradox is None or policy_paradox is None
        else float(proposed_paradox - policy_paradox)
    )
    target_action, target_class, candidate_evaluations = _candidate_target_from_audit(
        audit, policy_best, proposal_candidates, args
    )
    for evaluation in candidate_evaluations:
      action = int(evaluation["action"])
      evaluation["action_string"] = action_string(state, player, action)
      proposal_score = next(
          (
              row for row in item.get("proposal_candidates", [])
              if int(row.get("action", -1)) == action
          ),
          None,
      )
      if proposal_score is not None:
        evaluation["proposal_improve_prob"] = float(proposal_score["improve"])
        evaluation["proposal_safe_prob"] = float(proposal_score["safe"])
        evaluation["proposal_policy_prob"] = float(proposal_score["policy_prob"])
        if "proposal_checkpoint_prob" in proposal_score:
          evaluation["proposal_checkpoint_prob"] = float(
              proposal_score["proposal_checkpoint_prob"]
          )
    row = {
        "index": int(idx),
        "phase": item["phase"],
        "player": int(player),
        "group": {
            "match_index": int(item["match_index"]),
            "initial_start_player": int(item["initial_start_player"]),
            "round_index": int(item["round_index"]),
            "match_totals": item["match_totals"],
            "proposal_source": proposal_source_id,
            "improve_base_mode": str(args.improve_base_mode),
            "improve_threshold": float(args.improve_threshold),
            "improve_risk_threshold": float(args.improve_risk_threshold),
          },
        "match_round": int(getattr(state, "_match_round", -1)),
        "round_start_player": int(getattr(state, "_round_start_player", -1)),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "legal_count": int(item["legal_count"]),
        "policy_best_action": int(policy_best),
        "policy_best_action_string": action_string(state, player, policy_best),
        "raw_policy_best_action": int(raw_policy_best),
        "raw_policy_best_action_string": action_string(state, player, raw_policy_best),
        "label_best_action": int(target_action),
        "label_best_action_string": action_string(state, player, target_action),
        "proposed_action": int(proposed),
        "proposed_action_string": action_string(state, player, proposed),
        "primary_proposal_action": int(proposed),
        "primary_proposal_action_string": action_string(state, player, proposed),
        "target_action": int(target_action),
        "target_action_string": action_string(state, player, target_action),
        "candidate_evaluations": candidate_evaluations,
        "audited_actions": [int(action) for action in audited_actions],
        "audited": audit_margin is not None,
        "audit": {str(k): v for k, v in sorted(audit.items())},
        "audit_margin_vs_policy": audit_margin,
        "audit_paradox_delta_vs_policy": paradox_delta,
    }
    primary_evaluation = next(
        (
            evaluation for evaluation in candidate_evaluations
            if int(evaluation["action"]) == int(proposed)
        ),
        None,
    )
    primary_row_class = (
        str(primary_evaluation["calibration_class"])
        if primary_evaluation is not None
        else calibration_row_class(
            row,
            args.calibration_positive_margin,
            args.calibration_max_positive_paradox_delta,
        )
    )
    row["primary_stability"] = primary_stability
    row["primary_proposal_class"] = primary_row_class
    row["target_calibration_class"] = target_class
    row["calibration_class"] = primary_row_class
    row["false_positive_for_positive_gate"] = bool(
        primary_row_class != "POSITIVE"
    )
    replay.append(
        accepted_replay_example(
            state,
            player,
            target_action,
            policy_best,
            audit,
            oracle_args,
            label_stats=audit,
            scored_actions=set(int(action) for action in audited_actions),
        )
    )
    metadata.append({
        "source_index": int(idx),
        "phase": item["phase"],
        "player": int(player),
        "group": row["group"],
        "match_round": int(row["match_round"]),
        "round_start_player": int(row["round_start_player"]),
        "trick_number": int(row["trick_number"]),
        "policy_best_action": int(policy_best),
        "policy_best_action_string": row["policy_best_action_string"],
        "raw_policy_best_action": int(raw_policy_best),
        "raw_policy_best_action_string": row["raw_policy_best_action_string"],
        # The abstention model scores the live proposal against the incumbent.
        # On a rejected proposal the replay target remains the incumbent, but
        # the pair candidate must remain the rejected proposed action.
        **proposal_calibration_fields(
            proposed,
            row["proposed_action_string"],
            primary_row_class,
            audit_margin,
            paradox_delta,
        ),
        "primary_proposal_action": int(proposed),
        "primary_proposal_action_string": row["primary_proposal_action_string"],
        "deployment_target_action": int(target_action),
        "deployment_target_action_string": action_string(state, player, target_action),
        "target_calibration_class": target_class,
        "primary_proposal_class": primary_row_class,
        "label_margin_vs_policy": None,
        "primary_audit_margin_vs_policy": audit_margin,
        "primary_audit_paradox_delta_vs_policy": paradox_delta,
        "legal_count": int(item["legal_count"]),
        "candidate_count": int(len(audited_actions)),
        "candidate_set_contains_label_best": True,
        "candidate_actions": [int(action) for action in audited_actions],
        "proposal_candidates": item.get("proposal_candidates", []),
        "candidate_evaluations": candidate_evaluations,
        "audit_seed": int(audit_seed),
        "primary_stability": primary_stability,
        "proposal_policy": proposal_source_mode,
        "improve_base_mode": str(args.improve_base_mode),
    })
    rows.append(row)
    if args.progress_interval > 0 and (
        (idx + 1) % args.progress_interval == 0
        or (idx + 1) == len(proposals)
    ):
      print(json.dumps({
          "audited_proposals": int(idx + 1),
          "total_proposals": int(len(proposals)),
          "elapsed_sec": round(time.perf_counter() - started, 3),
      }), flush=True)

  calibration_path = Path(args.calibration_replay_out)
  calibration_path.parent.mkdir(parents=True, exist_ok=True)
  replay_metadata = {
      "proposal_source_id": proposal_source_id,
      "counterfactual_full_match_rollout": bool(args.full_match_rollout),
      "counterfactual_action_paradox_scope": str(
          args.counterfactual_action_paradox_scope
      ),
      "counterfactual_rollout_max_plies": int(args.rollout_max_plies),
      "calibration_positive_margin": float(args.calibration_positive_margin),
      "calibration_max_positive_paradox_delta": float(
          args.calibration_max_positive_paradox_delta
      ),
      "audit_rollouts": int(args.audit_rollouts),
      "stability_audit_passes": int(args.stability_audit_passes),
      "stability_seed_stride": int(args.stability_seed_stride),
      "stability_min_positive_sign_rate": float(
          args.stability_min_positive_sign_rate
      ),
      "stability_min_safe_risk_rate": float(
          args.stability_min_safe_risk_rate
      ),
      "unstable_positive_as_negative": bool(
          args.unstable_positive_as_negative
      ),
  }
  save_replay(replay, calibration_path, metadata=replay_metadata)
  metadata_path = calibration_path.with_suffix(
      calibration_path.suffix + ".metadata.json"
  )
  metadata_path.write_text(json.dumps({
      "checkpoint": args.checkpoint,
      "improve_model": args.improve_model,
      "proposal_checkpoint": args.proposal_checkpoint,
      "proposal_source_mode": proposal_source_mode,
      "seed": args.seed,
      "proposal_source_id": proposal_source_id,
      "positive_margin": args.calibration_positive_margin,
      "max_positive_paradox_delta": args.calibration_max_positive_paradox_delta,
      "counterfactual_action_paradox_scope": str(
          args.counterfactual_action_paradox_scope
      ),
      "proposal_audit_top_k": int(args.proposal_audit_top_k),
      "audit_rollouts": int(args.audit_rollouts),
      "stability_audit_passes": int(args.stability_audit_passes),
      "stability_seed_stride": int(args.stability_seed_stride),
      "stability_min_positive_sign_rate": float(
          args.stability_min_positive_sign_rate
      ),
      "stability_min_safe_risk_rate": float(
          args.stability_min_safe_risk_rate
      ),
      "unstable_positive_as_negative": bool(
          args.unstable_positive_as_negative
      ),
      "proposal_trick_quotas": proposal_trick_quotas,
      "improve_threshold": float(args.improve_threshold),
      "improve_risk_threshold": float(args.improve_risk_threshold),
      "improve_base_mode": str(args.improve_base_mode),
      "action_value_selection_weight": float(args.action_value_selection_weight),
      "action_feasibility_selection_weight": float(
          args.action_feasibility_selection_weight
      ),
      "action_paradox_rerank_mode": str(args.action_paradox_rerank_mode),
      "action_paradox_min_risk_margin": float(args.action_paradox_min_risk_margin),
      "action_paradox_max_policy_log_gap": float(args.action_paradox_max_policy_log_gap),
      "improve_min_trick_number": int(args.improve_min_trick_number),
      "improve_trick_numbers": str(args.improve_trick_numbers),
      "improve_led_colors": str(args.improve_led_colors),
      "improve_candidate_colors": str(args.improve_candidate_colors),
      "improve_require_base_off_led": bool(args.improve_require_base_off_led),
      "improve_require_responsible_red_win": bool(
          args.improve_require_responsible_red_win
      ),
      "improve_guard_led_follow": bool(args.improve_guard_led_follow),
      "improve_guard_led_follow_red_win_must_be_urgent": bool(
          args.improve_guard_led_follow_red_win_must_be_urgent
      ),
      "improve_max_policy_log_gap": float(args.improve_max_policy_log_gap),
      "collection_target": int(collection_target),
      "rows": metadata,
  }, indent=2) + "\n")
  result = {
      "checkpoint": args.checkpoint,
      "improve_model": args.improve_model,
      "proposal_checkpoint": args.proposal_checkpoint,
      "proposal_source_mode": proposal_source_mode,
      "players": args.players,
      "states_requested": args.states,
      "collection_target": int(collection_target),
      "proposals_collected": len(proposals),
      "matches_used": int(matches_used),
      "proposal_collection_stats": collection_stats,
      "elapsed_sec": round(time.perf_counter() - started, 3),
      "phases": sorted(wanted),
      "improve_threshold": float(args.improve_threshold),
      "improve_risk_threshold": float(args.improve_risk_threshold),
      "improve_base_mode": str(args.improve_base_mode),
      "action_value_selection_weight": float(args.action_value_selection_weight),
      "action_feasibility_selection_weight": float(
          args.action_feasibility_selection_weight
      ),
      "action_paradox_rerank_mode": str(args.action_paradox_rerank_mode),
      "action_paradox_min_risk_margin": float(args.action_paradox_min_risk_margin),
      "action_paradox_max_policy_log_gap": float(args.action_paradox_max_policy_log_gap),
      "proposal_audit_top_k": int(args.proposal_audit_top_k),
      "proposal_trick_quotas": proposal_trick_quotas,
      "shadow_run119_actions": bool(args.shadow_run119_actions),
      "improve_max_actions": int(args.improve_max_actions),
      "improve_top_policy": int(args.improve_top_policy),
      "improve_min_legal_count": int(args.improve_min_legal_count),
      "improve_max_legal_count": int(args.improve_max_legal_count),
      "improve_min_trick_number": int(args.improve_min_trick_number),
      "improve_trick_numbers": str(args.improve_trick_numbers),
      "improve_led_colors": str(args.improve_led_colors),
      "improve_candidate_colors": str(args.improve_candidate_colors),
      "improve_require_base_off_led": bool(args.improve_require_base_off_led),
      "improve_require_responsible_red_win": bool(
          args.improve_require_responsible_red_win
      ),
      "improve_min_match_round": int(args.improve_min_match_round),
      "improve_guard_led_follow": bool(args.improve_guard_led_follow),
      "improve_guard_led_follow_red_win_must_be_urgent": bool(
          args.improve_guard_led_follow_red_win_must_be_urgent
      ),
      "improve_max_policy_log_gap": float(args.improve_max_policy_log_gap),
      "audit_rollouts": int(args.audit_rollouts),
      "stability_audit_passes": int(args.stability_audit_passes),
      "stability_seed_stride": int(args.stability_seed_stride),
      "stability_min_positive_sign_rate": float(
          args.stability_min_positive_sign_rate
      ),
      "stability_min_safe_risk_rate": float(
          args.stability_min_safe_risk_rate
      ),
      "unstable_positive_as_negative": bool(
          args.unstable_positive_as_negative
      ),
      "full_match_rollout": bool(args.full_match_rollout),
      "counterfactual_action_paradox_scope": str(
          args.counterfactual_action_paradox_scope
      ),
      "calibration_positive_margin": float(args.calibration_positive_margin),
      "calibration_max_positive_paradox_delta": float(
          args.calibration_max_positive_paradox_delta
      ),
      "calibration_replay_out": str(calibration_path),
      "calibration_replay_rows": len(replay),
      "calibration_summary": calibration_summary(metadata),
      "accepted_replay_summary": accepted_replay_summary(replay),
      "improve_decision_stats": (
          improve_bot.decision_stats() if improve_bot is not None else {}
      ),
      "rows": rows,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
