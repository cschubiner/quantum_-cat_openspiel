"""Generate action-conditioned survival/liquidity replay from full matches.

This is the next survival-curriculum miner after dense state-value labels.  It
labels legal actions within the same public/own-information state, so the model
gets direct comparisons like "follow led color preserves exits, off-led burns a
lane and later forces paradox."
"""

from __future__ import annotations

import argparse
import json
import math
import time
from argparse import BooleanOptionalAction
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from quantum_cat_ai import make_bot
from quantum_cat_alphazero_torch import (
  ACTION_FEATURE_FOLLOWS_LED_INDEX,
  ACTION_FEATURE_IS_RED_INDEX,
  ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
  ACTION_FEATURE_PREFIX_SIZE,
  ACTION_FEATURE_RANK_NORM_INDEX,
  APPENDED_ACTION_FEATURE_INDEX,
  action_feasibility_scores,
  action_feature_matrix,
  make_game,
  model_policy_value,
  model_policy_value_action_aux,
  q_policy_select_action,
  save_replay,
  terminal_paradox_target,
)
from quantum_cat_full_match_elo import (
    AZLivenessShieldPolicyBot,
    LivenessKeyTeacherBot,
    _model_args,
    action_liveness_certificate,
    exact_legal_pressure_after_action,
    exact_table_liveness_after_action,
    load_neural,
)
from quantum_cat_public_history import public_history_metadata
from quantum_cat_survival_grammar_replay import (
    action_kind_name,
    phase_name,
    safe_feature,
    structural_scores,
)


PHASE_NAMES = ("chance", "discard", "prediction", "play", "terminal")


def state_phase_name(state) -> str:
  phase_idx = int(getattr(state, "_phase", -1))
  if 0 <= phase_idx < len(PHASE_NAMES):
    return PHASE_NAMES[phase_idx]
  return "unknown"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=20)
  parser.add_argument("--seed", type=int, default=20268800)
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument("--no-match-context", dest="match_context", action="store_false")
  parser.set_defaults(match_context=True)
  parser.add_argument("--bot", default="")
  parser.add_argument(
      "--liveness-teacher-base-bot",
      default="heuristic_safe15",
      help=(
          "Base built-in bot when --bot is liveness_key_teacher. This mirrors "
          "the evaluator teacher used for survival-grammar mining."
      ),
  )
  parser.add_argument(
      "--liveness-teacher-phases",
      default="play",
      help="Enabled phases when --bot is liveness_key_teacher.",
  )
  parser.add_argument(
      "--liveness-teacher-min-trick-number",
      type=int,
      default=0,
      help="Minimum trick number for liveness_key_teacher source overrides.",
  )
  parser.add_argument(
      "--liveness-teacher-exact-legal-pressure",
      action=BooleanOptionalAction,
      default=False,
      help=(
          "Use exact post-action legal-exit pressure inside the liveness teacher "
          "source bot."
      ),
  )
  parser.add_argument(
      "--liveness-teacher-exact-key-order",
      choices=(
          "lane_first",
          "pressure_first",
          "late_red_pressure_first",
          "table_liveness_first",
          "late_table_liveness_first",
      ),
      default="lane_first",
      help=(
          "Lexicographic key used when the liveness teacher has exact labels. "
          "table_liveness_first is a hidden-info training teacher that ranks "
          "candidate actions by all players' exact post-action hand liveness."
      ),
  )
  parser.add_argument("--checkpoint", default="")
  parser.add_argument(
      "--checkpoint-mode",
      choices=("policy", "q_policy", "liveness_shield"),
      default="policy",
      help=(
          "How --checkpoint chooses trajectory/continuation actions. "
          "policy samples the raw policy head; q_policy uses the deployed "
          "action-value/risk reranker with the knobs below; liveness_shield "
          "uses the same q/policy base plus deterministic liveness guard used "
          "by the homogeneous paradox gate."
      ),
  )
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
  parser.add_argument("--source-temperature", type=float, default=0.20)
  parser.add_argument("--continuation-temperature", type=float, default=0.20)
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument(
      "--action-paradox-rerank-mode",
      choices=("additive", "threshold", "relative"),
      default="additive",
  )
  parser.add_argument("--action-paradox-risk-threshold", type=float, default=0.0)
  parser.add_argument("--action-paradox-min-risk-margin", type=float, default=0.0)
  parser.add_argument("--action-paradox-max-policy-log-gap", type=float, default=2.0)
  parser.add_argument("--action-feasibility-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument(
      "--action-value-rerank-phases",
      default="",
      help=(
          "Optional comma-separated phase names where checkpoint q_policy "
          "reranking is enabled. Empty applies in every phase."
      ),
  )
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  parser.add_argument(
      "--liveness-shield-base-mode",
      choices=("policy", "q_policy"),
      default="q_policy",
      help="Base neural mode used by --checkpoint-mode=liveness_shield.",
  )
  parser.add_argument(
      "--liveness-shield-phases",
      default="discard,prediction,play",
      help="Comma-separated phases where liveness_shield may override.",
  )
  parser.add_argument("--liveness-shield-min-trick-number", type=int, default=-1)
  parser.add_argument("--liveness-shield-led-colors", default="")
  parser.add_argument("--liveness-shield-min-open-slot-delta", type=int, default=1)
  parser.add_argument(
      "--liveness-shield-min-public-damage-delta",
      type=int,
      default=2,
  )
  parser.add_argument("--liveness-shield-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument(
      "--liveness-shield-exact-legal-pressure",
      action=BooleanOptionalAction,
      default=False,
      help=(
          "Use exact post-action legal-exit pressure inside "
          "--checkpoint-mode=liveness_shield. This mirrors the evaluator "
          "teacher used for Run653 exact-pressure screens."
      ),
  )
  parser.add_argument(
      "--liveness-shield-exact-forced-only",
      action=BooleanOptionalAction,
      default=False,
      help=(
          "With --liveness-shield-exact-legal-pressure, only allow exact "
          "shield overrides that reduce immediately forced paradox players."
      ),
  )
  parser.add_argument(
      "--liveness-shield-shadow-only",
      action=BooleanOptionalAction,
      default=False,
  )
  parser.add_argument(
      "--liveness-shield-label-policy",
      choices=("deployed", "raw"),
      default="deployed",
      help=(
          "Policy vector returned for labeled liveness_shield states. "
          "deployed preserves historical behavior-cloning semantics by making "
          "the shielded action the policy top action. raw keeps the neural "
          "base policy visible, so contrast labels can penalize the raw action "
          "that the shield would veto while still following shielded "
          "trajectories."
      ),
  )
  parser.add_argument("--liveness-shield-sample-limit", type=int, default=0)
  parser.add_argument("--phases", default="play")
  parser.add_argument(
      "--min-trick-number",
      type=int,
      default=-1,
      help=(
          "When >=0, collect play-position labels only at or after this "
          "zero-based trick number. Non-play phases are unaffected."
      ),
  )
  parser.add_argument(
      "--max-trick-number",
      type=int,
      default=-1,
      help=(
          "When >=0, collect play-position labels only at or before this "
          "zero-based trick number. Non-play phases are unaffected."
      ),
  )
  parser.add_argument(
      "--row-filter",
      choices=(
          "all",
          "lane_contrast",
          "bottleneck",
          "lane_or_bottleneck",
          "red_switch_contrast",
          "lane_or_red_switch",
          "base_lost_led_follow_available",
          "base_nonred_lost_led_follow_available",
          "forced_switch_red_contrast",
          "forced_switch_red_win_contrast",
      ),
      default="lane_or_bottleneck",
  )
  parser.add_argument(
      "--source-round-filter",
      choices=("all", "survived", "paradox"),
      default="all",
      help=(
          "Keep replay rows only from source rounds with the requested terminal "
          "outcome. survived means no player paradoxed in that round."
      ),
  )
  parser.add_argument(
      "--behavior-policy-target",
      action=BooleanOptionalAction,
      default=False,
      help=(
          "Use the actual source trajectory action as a one-hot policy target. "
          "This is intended for behavior cloning from survived full rounds."
      ),
  )
  parser.add_argument(
      "--behavior-action-filter",
      choices=(
          "all",
          "follow_led",
          "follow_or_red_switch",
          "follow_or_red_win_switch",
          "not_nonred_token_loss",
      ),
      default="all",
      help=(
          "Optional action-level filter applied only with "
          "--behavior-policy-target. follow_or_red_win_switch keeps source "
          "actions that follow led, plus red switches that win the current "
          "trick."
      ),
  )
  parser.add_argument("--max-rows", type=int, default=0)
  parser.add_argument("--max-actions-per-state", type=int, default=0)
  parser.add_argument(
      "--candidate-action-filter",
      choices=("all", "follow_led", "no_token_loss", "follow_no_token_loss"),
      default="all",
      help=(
          "Filter labeled non-base candidate actions before rollout/deterministic "
          "targets are computed. The policy base action is always retained so "
          "downstream override trainers can compare candidates against it."
      ),
  )
  parser.add_argument(
      "--top-policy-actions",
      type=int,
      default=2,
      help=(
          "When --max-actions-per-state caps labels, keep this many policy-top "
          "actions before adding feature-extreme lane/liquidity candidates."
      ),
  )
  parser.add_argument(
      "--feature-candidates",
      action=BooleanOptionalAction,
      default=True,
      help=(
          "When candidate labels are capped, force in feature-extreme actions "
          "such as follow-led, newly-loses-led, lane-surplus, and bottleneck "
          "contrasts instead of relying only on policy-top actions."
      ),
  )
  parser.add_argument("--rollouts-per-action", type=int, default=1)
  parser.add_argument(
      "--paired-rollout-seeds",
      action=BooleanOptionalAction,
      default=True,
      help=(
          "Use the same random seeds for every candidate action in a state. "
          "This common-random-number pairing reduces label noise in the "
          "within-state action comparison."
      ),
  )
  parser.add_argument(
      "--rollout-horizon-plies",
      type=int,
      default=0,
      help="0 rolls to round terminal; positive values stop early.",
  )
  parser.add_argument(
      "--horizon-survival-prob",
      type=float,
      default=0.72,
      help="Survival probability assigned when a capped rollout reaches horizon.",
  )
  parser.add_argument(
      "--rollout-weight",
      type=float,
      default=0.70,
      help="Blend weight for continuation survival versus immediate liquidity.",
  )
  parser.add_argument(
      "--immediate-target-scale",
      type=float,
      default=2.0,
      help="Scale for tanh of relative immediate liquidity scores.",
  )
  parser.add_argument("--policy-target-temperature", type=float, default=0.12)
  parser.add_argument(
      "--deterministic-teacher-policy-target",
      action=BooleanOptionalAction,
      default=False,
      help=(
          "When deterministic liveness rows are available, set the policy "
          "target to the lexicographic liveness-teacher best action instead "
          "of the smoothed value softmax. This directly distills the survival "
          "teacher action rule while keeping dense value/risk labels."
      ),
  )
  parser.add_argument(
      "--deterministic-teacher-key-order",
      choices=("auto", "liveness", "pressure", "table_liveness"),
      default="auto",
      help=(
          "Key used when --deterministic-teacher-policy-target is enabled. "
          "auto preserves historical behavior: exact-pressure key when exact "
          "pressure labels are enabled, otherwise public liveness. "
          "table_liveness uses hidden-information exact table liveness as a "
          "training-time teacher target."
      ),
  )
  parser.add_argument(
      "--deterministic-teacher-auxiliary-mode",
      choices=("auto", "preserve", "teacher_rank"),
      default="auto",
      help=(
          "Keep policy/value/risk supervision consistent for deterministic "
          "teacher replay. auto replaces non-rollout structural proxies with "
          "the teacher's within-state ordinal preferences while preserving "
          "real terminal-rollout targets. preserve retains historical proxies; "
          "teacher_rank explicitly uses teacher-order value/risk labels."
      ),
  )
  parser.add_argument(
      "--switch-curriculum-policy-target",
      action=BooleanOptionalAction,
      default=False,
      help=(
          "For rows where following led and red-switch alternatives coexist, "
          "use a soft target that keeps follow-led mass while explicitly "
          "favoring the best red switch over non-red token-burning switches."
      ),
  )
  parser.add_argument(
      "--switch-curriculum-follow-mass",
      type=float,
      default=0.55,
      help="Policy-target mass assigned to the best follow-led action.",
  )
  parser.add_argument(
      "--switch-curriculum-red-mass",
      type=float,
      default=0.40,
      help="Policy-target mass assigned to the best red-switch action.",
  )
  parser.add_argument(
      "--switch-curriculum-other-mass",
      type=float,
      default=0.05,
      help="Residual policy-target mass assigned to the best remaining action.",
  )
  parser.add_argument(
      "--switch-curriculum-target-mode",
      choices=("balanced", "red_exception_top"),
      default="balanced",
      help=(
          "balanced preserves the historical soft target. red_exception_top "
          "lets high-quality red switches become the top target when they win "
          "or improve deterministic liveness/exact-pressure enough."
      ),
  )
  parser.add_argument(
      "--switch-curriculum-red-top-mass",
      type=float,
      default=0.68,
      help=(
          "Policy-target mass assigned to the best red switch when "
          "--switch-curriculum-target-mode=red_exception_top triggers."
      ),
  )
  parser.add_argument(
      "--switch-curriculum-red-top-min-value-delta",
      type=float,
      default=0.08,
      help=(
          "Minimum policy-score advantage over the best follow-led action for "
          "a red switch to become the top target without already winning."
      ),
  )
  parser.add_argument(
      "--switch-curriculum-red-top-require-win",
      action=BooleanOptionalAction,
      default=False,
      help=(
          "When true, only red switches that win the current trick may become "
          "the top switch-curriculum target."
      ),
  )
  parser.add_argument(
      "--switch-curriculum-red-win-bonus",
      type=float,
      default=0.40,
      help="Action-value bonus for red switches that win the current trick.",
  )
  parser.add_argument(
      "--switch-curriculum-red-bonus",
      type=float,
      default=0.18,
      help="Action-value bonus for red switches that do not currently win.",
  )
  parser.add_argument(
      "--switch-curriculum-nonred-penalty",
      type=float,
      default=0.35,
      help="Action-value penalty for non-red actions that burn the led lane.",
  )
  parser.add_argument("--min-target-spread", type=float, default=0.08)
  parser.add_argument(
      "--min-risk-spread",
      type=float,
      default=0.0,
      help=(
          "Optional minimum max-min continuation paradox-risk spread. 0 keeps "
          "the historical value-spread-only filter."
      ),
  )
  parser.add_argument(
      "--emit-action-risk-labels",
      action=BooleanOptionalAction,
      default=True,
      help=(
          "Store per-action paradox-risk labels in replay item[7]/item[8]. "
          "Risk is 1 - full-round continuation survival."
      ),
  )
  parser.add_argument("--min-structural-spread", type=float, default=0.10)
  parser.add_argument("--progress-interval", type=int, default=0)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument(
      "--metadata-out",
      default="",
      help=(
          "Optional per-row JSON sidecar. Rows include the deployable public "
          "history contract captured from the exact source state."
      ),
  )
  parser.add_argument(
      "--include-row-stats",
      action=BooleanOptionalAction,
      default=False,
      help=(
          "Include per-replay-row stats in the JSON summary. This is useful "
          "for downstream contrast transforms that need baseline_action vs "
          "best_action pairs, but can make summaries large."
      ),
  )
  parser.add_argument(
      "--deterministic-liveness-weight",
      type=float,
      default=0.0,
      help=(
          "Blend in exact own-hand feasibility plus public exit-liquidity "
          "labels from action_liveness_certificate. This is a no-rollout, "
          "within-state survival grammar target."
      ),
  )
  parser.add_argument(
      "--deterministic-liveness-scale",
      type=float,
      default=3.0,
      help="Scale for tanh-normalizing centered deterministic liveness scores.",
  )
  parser.add_argument(
      "--deterministic-liveness-risk-weight",
      type=float,
      default=0.0,
      help=(
          "Blend action-risk labels toward deterministic liveness risk. "
          "0 keeps continuation-survival risk; 1 uses deterministic risk."
      ),
  )
  parser.add_argument(
      "--deterministic-exact-legal-pressure-weight",
      type=float,
      default=0.0,
      help=(
          "Blend deterministic liveness action values toward exact post-action "
          "legal-exit pressure labels. This is an oracle teacher/audit signal "
          "from the full simulator state."
      ),
  )
  parser.add_argument(
      "--deterministic-exact-legal-pressure-scale",
      type=float,
      default=2.0,
      help="Scale for tanh-normalizing centered exact legal-pressure scores.",
  )
  parser.add_argument(
      "--deterministic-exact-legal-pressure-risk-weight",
      type=float,
      default=0.0,
      help=(
          "Blend action-risk labels toward exact post-action forced-legal "
          "pressure. 0 keeps the existing deterministic/rollout risk."
      ),
  )
  parser.add_argument(
      "--deterministic-exact-table-liveness-weight",
      type=float,
      default=0.0,
      help=(
          "Blend deterministic action values toward exact post-action "
          "all-player hand-liveness labels. This reads hidden hands and is a "
          "training-time teacher signal only."
      ),
  )
  parser.add_argument(
      "--deterministic-exact-table-liveness-scale",
      type=float,
      default=2.0,
      help="Scale for tanh-normalizing centered exact table-liveness scores.",
  )
  parser.add_argument(
      "--deterministic-exact-table-liveness-risk-weight",
      type=float,
      default=0.0,
      help=(
          "Blend action-risk labels toward exact table-liveness collapse "
          "risk. 0 keeps the existing deterministic/rollout risk."
      ),
  )

  # Keep these names aligned with quantum_cat_survival_grammar_replay so we can
  # reuse its feature-derived structural score.
  parser.add_argument("--prior-weight", type=float, default=0.04)
  parser.add_argument("--action-value-weight", type=float, default=0.0)
  parser.add_argument("--follow-led-bonus", type=float, default=1.6)
  parser.add_argument("--token-loss-penalty", type=float, default=4.2)
  parser.add_argument("--min-lane-surplus-weight", type=float, default=1.25)
  parser.add_argument("--lane-damage-penalty", type=float, default=1.10)
  parser.add_argument("--lane-pressure-penalty", type=float, default=0.45)
  parser.add_argument("--own-min-colors-weight", type=float, default=0.85)
  parser.add_argument("--own-zero-exit-penalty", type=float, default=2.2)
  parser.add_argument("--own-one-exit-penalty", type=float, default=1.25)
  parser.add_argument("--own-two-less-exit-penalty", type=float, default=0.45)
  parser.add_argument("--future-no-exit-penalty", type=float, default=1.7)
  parser.add_argument("--forced-pressure-penalty", type=float, default=0.70)
  parser.add_argument("--paradox-penalty", type=float, default=8.0)
  parser.add_argument("--discard-relief-weight", type=float, default=1.0)
  parser.add_argument("--discard-no-exit-penalty", type=float, default=1.5)
  parser.add_argument("--prediction-abs-gap-penalty", type=float, default=0.35)
  parser.add_argument("--prediction-over-penalty", type=float, default=0.15)
  parser.add_argument("--bottleneck-zero-threshold", type=float, default=0.01)
  parser.add_argument("--bottleneck-one-threshold", type=float, default=0.12)
  return parser.parse_args()


def split_names(spec: str) -> set[str]:
  return {
      part.strip().lower()
      for part in str(spec or "").replace("|", ",").replace("+", ",").split(",")
      if part.strip()
  }


def legal_mask(num_actions: int, legal: list[int]) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=np.float32)
  for action in legal:
    mask[int(action)] = 1.0
  return mask


def one_hot_policy(num_actions: int, action: int) -> np.ndarray:
  policy = np.zeros(num_actions, dtype=np.float32)
  if 0 <= int(action) < num_actions:
    policy[int(action)] = 1.0
  return policy


def softmax_over_actions(
    scores: dict[int, float],
    num_actions: int,
    temperature: float,
) -> np.ndarray:
  target = np.zeros(num_actions, dtype=np.float32)
  if not scores:
    return target
  actions = sorted(int(action) for action in scores)
  logits = np.array([float(scores[action]) for action in actions], dtype=np.float32)
  logits /= max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  weights = np.exp(logits)
  total = float(np.sum(weights))
  if not math.isfinite(total) or total <= 0.0:
    weights = np.ones_like(weights, dtype=np.float32)
    total = float(np.sum(weights))
  weights /= total
  for action, weight in zip(actions, weights):
    target[int(action)] = float(weight)
  return target


def sample_policy(
    policy: np.ndarray,
    legal: list[int],
    temperature: float,
    rng: np.random.Generator,
) -> int:
  if len(legal) == 1:
    return int(legal[0])
  if temperature <= 1e-9:
    return int(max(legal, key=lambda action: float(policy[int(action)])))
  probs = np.array([max(float(policy[int(action)]), 1e-12) for action in legal])
  logits = np.log(probs) / max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  weights = np.exp(logits)
  total = float(np.sum(weights))
  if not math.isfinite(total) or total <= 0.0:
    weights = np.ones_like(weights, dtype=np.float32)
    total = float(np.sum(weights))
  weights /= total
  return int(rng.choice(np.array(legal, dtype=np.int64), p=weights))


def phase_enabled_for_q_policy(state, phase_spec: str) -> bool:
  spec = str(phase_spec or "")
  if not spec.strip():
    return True
  enabled = {
      item.strip().lower()
      for item in spec.replace("|", ",").replace("+", ",").split(",")
      if item.strip()
  }
  return state_phase_name(state) in enabled


def checkpoint_liveness_shield_action(
    state,
    player: int,
    num_actions: int,
    model,
    model_args,
    device,
    label_policy: str = "deployed",
) -> tuple[np.ndarray, int]:
  """Selects an action with the deployed liveness shield for replay rollouts."""
  legal = [int(action) for action in state.legal_actions(player)]
  bot = AZLivenessShieldPolicyBot(
      model,
      "counterfactual_replay_liveness_shield",
      device,
      model_args,
  )
  action = int(bot.step(state.clone(), player))
  if action not in legal:
    action = int(legal[0])
  if str(label_policy) == "raw":
    policy, _ = model_policy_value(
        model,
        state,
        player,
        num_actions,
        model_args.value_scale,
        device,
    )
    return np.asarray(policy, dtype=np.float32), int(action)
  # The replay miner uses the returned policy to choose the "baseline" action
  # and to retain policy-top candidates under caps. For liveness-shield
  # behavior-cloning trajectories the baseline should be the actual deployed
  # action, not the raw neural top action hidden underneath the shield.
  return one_hot_policy(num_actions, action), int(action)


def checkpoint_policy_action(
    args: argparse.Namespace,
    state,
    player: int,
    num_actions: int,
    model,
    model_args,
    device,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
  legal = [int(action) for action in state.legal_actions(player)]
  if str(getattr(args, "checkpoint_mode", "policy")) == "liveness_shield":
    return checkpoint_liveness_shield_action(
        state,
        player,
        num_actions,
        model,
        model_args,
        device,
        label_policy=str(getattr(args, "liveness_shield_label_policy", "deployed")),
    )
  checkpoint_mode = str(getattr(args, "checkpoint_mode", "policy"))
  q_policy_enabled = (
      checkpoint_mode == "q_policy"
      and phase_enabled_for_q_policy(
          state, str(getattr(args, "action_value_rerank_phases", ""))
      )
  )
  value_weight = float(
      getattr(model_args, "action_value_selection_weight", 0.0)
  )
  risk_mode = str(
      getattr(model_args, "action_paradox_rerank_mode", "additive")
      or "additive"
  ).lower()
  risk_weight = float(
      getattr(model_args, "action_paradox_selection_penalty", 0.0)
  )
  feasibility_weight = float(
      getattr(model_args, "action_feasibility_selection_weight", 0.0)
  )
  if q_policy_enabled:
    policy, _, action_values, action_risks, action_features = (
        model_policy_value_action_aux(
            model,
            state,
            player,
            num_actions,
            model_args.value_scale,
            device,
            need_action_values=value_weight > 0.0,
            need_action_risks=(
                risk_weight > 0.0 or risk_mode in ("threshold", "relative")
            ),
            return_action_features=feasibility_weight != 0.0,
        )
    )
  else:
    policy, _ = model_policy_value(
        model,
        state,
        player,
        num_actions,
        model_args.value_scale,
        device,
    )
    action_values = None
    action_risks = None
    action_features = None
  if (
      checkpoint_mode != "q_policy"
      or not q_policy_enabled
  ):
    action = sample_policy(policy, legal, float(temperature), rng)
    return np.asarray(policy, dtype=np.float32), int(action)

  if action_values is None:
    action_values = np.zeros(num_actions, dtype=np.float32)
  if action_risks is None and (
      risk_weight > 0.0 or risk_mode in ("threshold", "relative")
  ):
    action_risks = np.zeros(num_actions, dtype=np.float32)
  clip = max(0.0, float(getattr(model_args, "action_value_rerank_clip", 0.5)))
  if clip > 0:
    action_values = np.clip(action_values, -clip, clip)
  feasibility_scores = (
      action_feasibility_scores(action_features)
      if feasibility_weight != 0.0 else
      None
  )
  action = q_policy_select_action(
      legal,
      policy,
      action_values,
      action_risks,
      model_args,
      getattr(model_args, "value_scale", 1.0),
      stats=None,
      feasibility_scores=feasibility_scores,
  )
  return np.asarray(policy, dtype=np.float32), int(action)


def source_policy_action(
    args: argparse.Namespace,
    state,
    player: int,
    num_actions: int,
    model,
    model_args,
    device,
    bots,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
  legal = [int(action) for action in state.legal_actions(player)]
  if model is not None:
    return checkpoint_policy_action(
        args,
        state,
        player,
        num_actions,
        model,
        model_args,
        device,
        float(args.source_temperature),
        rng,
    )
  action = int(bots[player].step(state.clone(), player))
  if action not in legal:
    action = int(legal[0])
  return one_hot_policy(num_actions, action), action


def continuation_action(
    args: argparse.Namespace,
    state,
    player: int,
    num_actions: int,
    model,
    model_args,
    device,
    bots,
    rng: np.random.Generator,
) -> int:
  legal = [int(action) for action in state.legal_actions(player)]
  if model is not None:
    _, action = checkpoint_policy_action(
        args,
        state,
        player,
        num_actions,
        model,
        model_args,
        device,
        float(args.continuation_temperature),
        rng,
    )
    return int(action)
  action = int(bots[player].step(state.clone(), player))
  return action if action in legal else int(legal[0])


def any_paradox(state) -> bool:
  return any(bool(value) for value in getattr(state, "_has_paradoxed", []))


def rollout_survival_probability(
    args: argparse.Namespace,
    state,
    first_action: int,
    model,
    model_args,
    device,
    bots,
    rng: np.random.Generator,
) -> float:
  trial = state.clone()
  try:
    trial.apply_action(int(first_action))
  except Exception:
    return 0.0
  if any_paradox(trial):
    return 0.0
  num_actions = trial.num_distinct_actions()
  horizon = int(args.rollout_horizon_plies)
  plies = 0
  while not trial.is_terminal():
    if horizon > 0 and plies >= horizon:
      return float(np.clip(args.horizon_survival_prob, 0.0, 1.0))
    if trial.is_chance_node():
      actions, probs = zip(*trial.chance_outcomes())
      trial.apply_action(int(rng.choice(np.array(actions, dtype=np.int64), p=probs)))
      continue
    player = int(trial.current_player())
    action = continuation_action(
        args, trial, player, num_actions, model, model_args, device, bots, rng
    )
    trial.apply_action(int(action))
    plies += 1
    if any_paradox(trial):
      return 0.0
  return 1.0 if not any_paradox(trial) else 0.0


def deterministic_liveness_scalar(row: dict) -> float:
  """Converts a liveness certificate into a smooth within-state score."""
  score = 0.0
  if bool(row.get("is_paradox", False)):
    score -= 10.0
  score += 4.0 if bool(row.get("own_feasible", False)) else -4.0
  score -= 2.2 * float(row.get("own_total_deficit", 0.0))
  score -= 0.7 * float(row.get("own_buffer_deficit", 0.0))
  score -= 2.0 * float(row.get("own_dead_rank_count", 0.0))
  score -= 0.35 * float(row.get("own_tight_rank_count", 0.0))
  score -= 0.20 * float(row.get("own_singleton_card_count", 0.0))
  score += 0.55 * float(row.get("min_player_lane_surplus_after", 0.0))
  score += 0.10 * float(row.get("total_player_lane_surplus_after", 0.0))
  score += 0.25 * float(row.get("own_lane_surplus_after", 0.0))
  score -= 1.1 * float(row.get("lane_pressure_player_count_after", 0.0))
  score += 0.15 * float(row.get("min_player_open_slots_after", 0.0))
  score += 0.03 * float(row.get("total_player_open_slots_after", 0.0))
  score += 0.45 * float(row.get("min_player_rank_slots_after", 0.0))
  score += 0.35 * float(row.get("own_min_rank_slots_after", 0.0))
  score += 0.20 * float(row.get("own_min_choices", 0.0))
  score += 0.15 * float(row.get("own_slot_surplus", 0.0))
  if bool(row.get("lost_led_token", False)):
    score -= 0.8 if bool(row.get("responsible_red_win_switch", False)) else 3.0
  if bool(row.get("over_target_would_win", False)):
    score -= 1.0
  score -= 0.25 * float(row.get("public_slot_damage", 0.0))
  score -= 0.20 * float(row.get("own_public_slot_damage", 0.0))
  score -= 0.10 * float(row.get("board_open_cell_damage", 0.0))
  return float(score)


def deterministic_liveness_risk(row: dict) -> float:
  """Approximate collapse risk from exact feasibility/liquidity diagnostics."""
  if bool(row.get("is_paradox", False)):
    return 1.0
  risk = 0.0
  if not bool(row.get("own_feasible", False)):
    risk += 0.35
  risk += min(0.30, 0.08 * float(row.get("own_total_deficit", 0.0)))
  risk += min(0.20, 0.04 * float(row.get("own_buffer_deficit", 0.0)))
  risk += min(0.25, 0.12 * float(row.get("own_dead_rank_count", 0.0)))
  risk += min(0.12, 0.04 * float(row.get("own_tight_rank_count", 0.0)))
  if bool(row.get("lost_led_token", False)):
    risk += 0.05 if bool(row.get("responsible_red_win_switch", False)) else 0.18
  min_lane = float(row.get("min_player_lane_surplus_after", 0.0))
  if min_lane < 0.0:
    risk += 0.16
  elif min_lane <= 0.0:
    risk += 0.08
  if float(row.get("min_player_rank_slots_after", 0.0)) <= 0.0:
    risk += 0.10
  if float(row.get("own_min_rank_slots_after", 0.0)) <= 0.0:
    risk += 0.08
  risk += min(0.12, 0.03 * float(row.get("lane_pressure_player_count_after", 0.0)))
  return float(np.clip(risk, 0.0, 1.0))


def exact_legal_pressure_scalar(row: dict) -> float:
  """Dense score for full-state post-action legal exits."""
  if bool(row.get("exact_terminal_after", False)):
    return 6.0
  forced = float(row.get("exact_forced_paradox_players_after", 0.0))
  min_legal = float(row.get("exact_min_nonparadox_legal_after", 0.0))
  current_legal = float(row.get("exact_current_nonparadox_legal_after", 0.0))
  total_legal = float(row.get("exact_total_nonparadox_legal_after", 0.0))
  score = float(
      -7.0 * forced
      + 0.45 * min(min_legal, 8.0)
      + 0.10 * min(current_legal, 8.0)
      + 0.02 * min(total_legal, 40.0)
  )
  if bool(row.get("lost_led_token", False)):
    score -= 2.5
  return score


def exact_legal_pressure_risk(row: dict) -> float:
  if bool(row.get("exact_terminal_after", False)):
    return 0.0
  forced = float(row.get("exact_forced_paradox_players_after", 0.0))
  min_legal = float(row.get("exact_min_nonparadox_legal_after", 0.0))
  risk = min(1.0, 0.55 * forced)
  if min_legal <= 0.0:
    risk += 0.40
  elif min_legal <= 1.0:
    risk += 0.22
  elif min_legal <= 2.0:
    risk += 0.10
  if bool(row.get("lost_led_token", False)):
    risk += 0.18
  return float(np.clip(risk, 0.0, 1.0))


def exact_table_liveness_scalar(row: dict) -> float:
  """Dense score for full-state post-action table liveness."""
  if bool(row.get("table_terminal_after", False)):
    return -6.0 if bool(row.get("table_any_paradox_after", False)) else 6.0
  dead = float(row.get("table_dead_margin_players_after", 0.0))
  min_margin = float(row.get("table_min_margin_after", -99.0))
  current_margin = float(row.get("table_current_margin_after", -99.0))
  total_margin = float(row.get("table_total_margin_after", -99.0))
  return float(
      -7.0 * dead
      + 1.10 * max(-6.0, min(min_margin, 8.0))
      + 0.25 * max(-6.0, min(current_margin, 8.0))
      + 0.04 * max(-24.0, min(total_margin, 40.0))
  )


def exact_table_liveness_risk(row: dict) -> float:
  if bool(row.get("table_terminal_after", False)):
    return 1.0 if bool(row.get("table_any_paradox_after", False)) else 0.0
  risk = 0.0
  dead = float(row.get("table_dead_margin_players_after", 0.0))
  min_margin = float(row.get("table_min_margin_after", 0.0))
  current_margin = float(row.get("table_current_margin_after", 0.0))
  risk += min(0.80, 0.45 * dead)
  if min_margin < 0.0:
    risk += 0.35
  elif min_margin <= 0.0:
    risk += 0.22
  elif min_margin <= 1.0:
    risk += 0.12
  if current_margin < 0.0:
    risk += 0.20
  elif current_margin <= 0.0:
    risk += 0.12
  return float(np.clip(risk, 0.0, 1.0))


def deterministic_teacher_key(
    row: dict, *, use_exact: bool, key_order: str = "auto"
) -> tuple[int, ...]:
  liveness_key = tuple(int(value) for value in row.get("liveness_key", []))
  key_order = str(key_order or "auto")
  if key_order == "liveness" or (key_order == "auto" and not use_exact):
    return liveness_key
  if key_order == "table_liveness":
    return (
        0 if bool(row.get("table_any_paradox_after", False)) else 1,
        -int(row.get("table_dead_margin_players_after", 0)),
        int(row.get("table_min_margin_after", -99)),
        int(row.get("table_current_margin_after", -99)),
        int(row.get("table_total_margin_after", -99)),
        1 if bool(row.get("table_terminal_after", False)) else 0,
        1 if bool(row.get("exact_terminal_after", False)) else 0,
        -int(row.get("exact_forced_paradox_players_after", 0)),
        int(row.get("exact_min_nonparadox_legal_after", 0)),
        int(row.get("exact_current_nonparadox_legal_after", 0)),
        int(row.get("exact_total_nonparadox_legal_after", 0)),
        0 if bool(row.get("lost_led_token", False)) else 1,
    ) + liveness_key
  return (
      1 if bool(row.get("exact_terminal_after", False)) else 0,
      -int(row.get("exact_forced_paradox_players_after", 0)),
      0 if bool(row.get("lost_led_token", False)) else 1,
      int(row.get("exact_min_nonparadox_legal_after", 0)),
      int(row.get("exact_current_nonparadox_legal_after", 0)),
      int(row.get("exact_total_nonparadox_legal_after", 0)),
  ) + liveness_key


def align_deterministic_teacher_auxiliary_targets(
    candidates,
    rows,
    value_targets,
    risk_targets,
    *,
    use_exact,
    key_order,
):
  """Write ordinal auxiliary targets consistent with the teacher action.

  These labels encode within-state preference, not calibrated terminal-failure
  probabilities. Genuine terminal-rollout labels must be preserved instead.
  """
  ordered = sorted(
      (int(action) for action in candidates),
      key=lambda action: deterministic_teacher_key(
          rows[action], use_exact=use_exact, key_order=key_order
      ),
      reverse=True,
  )
  denominator = max(1, len(ordered) - 1)
  for rank, action in enumerate(ordered):
    ordinal_risk = float(rank) / denominator
    value_targets[action] = 1.0 - 2.0 * ordinal_risk
    risk_targets[action] = ordinal_risk
  return ordered


def should_align_deterministic_teacher_auxiliary_targets(
    teacher_action, mode, rollout_count, switch_curriculum_active=False
):
  """Default alignment protects genuine terminal-rollout supervision."""
  if teacher_action is None or switch_curriculum_active or mode == "preserve":
    return False
  return mode == "teacher_rank" or int(rollout_count) <= 0


def deterministic_liveness_labels(
    state,
    player: int,
    candidates: list[int],
    base_action: int,
    args: argparse.Namespace,
) -> tuple[dict[int, float], dict[int, float], dict[int, dict]]:
  exact_weight = float(
      getattr(args, "deterministic_exact_legal_pressure_weight", 0.0)
  )
  exact_risk_weight = float(
      getattr(args, "deterministic_exact_legal_pressure_risk_weight", 0.0)
  )
  table_weight = float(
      getattr(args, "deterministic_exact_table_liveness_weight", 0.0)
  )
  table_risk_weight = float(
      getattr(args, "deterministic_exact_table_liveness_risk_weight", 0.0)
  )
  teacher_key_order = str(
      getattr(args, "deterministic_teacher_key_order", "auto") or "auto"
  )
  if (
      float(args.deterministic_liveness_weight) <= 0.0
      and float(args.deterministic_liveness_risk_weight) <= 0.0
      and exact_weight <= 0.0
      and exact_risk_weight <= 0.0
      and table_weight <= 0.0
      and table_risk_weight <= 0.0
      and teacher_key_order != "table_liveness"
  ):
    return {}, {}, {}
  rows: dict[int, dict] = {}
  raw_scores: dict[int, float] = {}
  risks: dict[int, float] = {}
  exact_scores: dict[int, float] = {}
  exact_risks: dict[int, float] = {}
  table_scores: dict[int, float] = {}
  table_risks: dict[int, float] = {}
  use_exact = exact_weight > 0.0 or exact_risk_weight > 0.0
  use_table = (
      table_weight > 0.0
      or table_risk_weight > 0.0
      or teacher_key_order == "table_liveness"
  )
  for action in candidates:
    row = action_liveness_certificate(
        state, int(player), int(action), base_action=int(base_action)
    )
    if use_exact or use_table:
      row.update(exact_legal_pressure_after_action(state, int(action)))
    if use_table:
      row.update(exact_table_liveness_after_action(state, int(action)))
    rows[int(action)] = row
    raw_scores[int(action)] = deterministic_liveness_scalar(row)
    risks[int(action)] = deterministic_liveness_risk(row)
    if use_exact or use_table:
      exact_scores[int(action)] = exact_legal_pressure_scalar(row)
      exact_risks[int(action)] = exact_legal_pressure_risk(row)
    if use_table:
      table_scores[int(action)] = exact_table_liveness_scalar(row)
      table_risks[int(action)] = exact_table_liveness_risk(row)
  if not raw_scores:
    return {}, risks, rows
  baseline = float(np.mean(list(raw_scores.values())))
  scale = max(1e-6, float(args.deterministic_liveness_scale))
  values = {
      int(action): float(math.tanh((float(score) - baseline) / scale))
      for action, score in raw_scores.items()
  }
  exact_weight = float(np.clip(exact_weight, 0.0, 1.0))
  if exact_weight > 0.0 and exact_scores:
    exact_baseline = float(np.mean(list(exact_scores.values())))
    exact_scale = max(
        1e-6,
        float(getattr(args, "deterministic_exact_legal_pressure_scale", 2.0)),
    )
    for action, score in exact_scores.items():
      exact_value = float(math.tanh((float(score) - exact_baseline) / exact_scale))
      values[int(action)] = (
          (1.0 - exact_weight) * float(values.get(int(action), 0.0))
          + exact_weight * exact_value
      )
  exact_risk_weight = float(np.clip(exact_risk_weight, 0.0, 1.0))
  if exact_risk_weight > 0.0 and exact_risks:
    for action, risk in exact_risks.items():
      risks[int(action)] = (
          (1.0 - exact_risk_weight) * float(risks.get(int(action), 0.0))
          + exact_risk_weight * float(risk)
      )
  table_weight = float(np.clip(table_weight, 0.0, 1.0))
  if table_weight > 0.0 and table_scores:
    table_baseline = float(np.mean(list(table_scores.values())))
    table_scale = max(
        1e-6,
        float(getattr(args, "deterministic_exact_table_liveness_scale", 2.0)),
    )
    for action, score in table_scores.items():
      table_value = float(math.tanh((float(score) - table_baseline) / table_scale))
      values[int(action)] = (
          (1.0 - table_weight) * float(values.get(int(action), 0.0))
          + table_weight * table_value
      )
  table_risk_weight = float(np.clip(table_risk_weight, 0.0, 1.0))
  if table_risk_weight > 0.0 and table_risks:
    for action, risk in table_risks.items():
      risks[int(action)] = (
          (1.0 - table_risk_weight) * float(risks.get(int(action), 0.0))
          + table_risk_weight * float(risk)
      )
  return values, risks, rows


def phase_from_features(features: np.ndarray, legal: np.ndarray) -> str:
  return phase_name(features, legal)


def row_kind_flags(features: np.ndarray, legal_mask_row: np.ndarray, args):
  legal = np.flatnonzero(legal_mask_row > 0.5)
  if legal.size == 0:
    return False, False, False, False, False
  legal_features = features[legal]
  legal_follow = legal_features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
  legal_token_loss = token_loss_mask(features)[legal]
  legal_red_switch = red_switch_mask(features)[legal]
  legal_red_win_switch = red_win_switch_mask(features)[legal]
  legal_nonred_switch = legal_token_loss & (
      legal_features[:, ACTION_FEATURE_IS_RED_INDEX] <= 0.5
  )
  has_lane = bool(np.any(legal_follow) and np.any(legal_token_loss))
  has_red_switch_contrast = bool(
      np.any(legal_follow) and np.any(legal_red_switch)
      and (np.any(legal_nonred_switch) or np.sum(legal_red_switch) >= 1)
  )
  has_forced_switch_red_contrast = bool(
      not np.any(legal_follow)
      and np.any(legal_red_switch)
      and np.any(legal_nonred_switch)
  )
  has_forced_switch_red_win_contrast = bool(
      has_forced_switch_red_contrast and np.any(legal_red_win_switch)
  )
  zero_exit = safe_feature(legal_features, "own_future_zero_exit_frac_after")
  one_exit = safe_feature(legal_features, "own_future_one_exit_frac_after")
  future_no_exit = safe_feature(legal_features, "future_no_exit_after")
  bottleneck = bool(
      float(np.max(zero_exit, initial=0.0)) >= float(args.bottleneck_zero_threshold)
      or float(np.max(one_exit, initial=0.0)) >=
      float(args.bottleneck_one_threshold)
      or float(np.max(future_no_exit, initial=0.0)) > 0.5
  )
  return (
      has_lane,
      bottleneck,
      has_red_switch_contrast,
      has_forced_switch_red_contrast,
      has_forced_switch_red_win_contrast,
  )


def base_lost_led_follow_available(
    features: np.ndarray,
    legal_mask_row: np.ndarray,
    base_action: int | None,
    *,
    require_nonred_base: bool = False,
) -> bool:
  if base_action is None:
    return False
  base_action = int(base_action)
  if base_action < 0 or base_action >= int(features.shape[0]):
    return False
  legal = np.flatnonzero(legal_mask_row > 0.5)
  if legal.size == 0 or base_action not in set(int(action) for action in legal):
    return False
  follows = features[legal, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
  if not bool(np.any(follows)):
    return False
  if not bool(token_loss_mask(features)[base_action]):
    return False
  if require_nonred_base and bool(features[base_action, ACTION_FEATURE_IS_RED_INDEX] > 0.5):
    return False
  return True


def keep_row(
    features: np.ndarray,
    mask: np.ndarray,
    structural: np.ndarray,
    phases: set[str],
    args: argparse.Namespace,
    base_action: int | None = None,
) -> tuple[bool, str, bool, bool, bool, bool, bool, float]:
  legal = mask > 0.5
  phase = phase_from_features(features, legal)
  if phases and phase.lower() not in phases:
    return False, phase, False, False, False, False, False, 0.0
  legal_scores = structural[legal]
  spread = (
      float(np.max(legal_scores) - np.min(legal_scores))
      if legal_scores.size else 0.0
  )
  mode = str(args.row_filter)
  is_base_rescue_mode = mode in {
      "base_lost_led_follow_available",
      "base_nonred_lost_led_follow_available",
  }
  if spread < float(args.min_structural_spread) and not is_base_rescue_mode:
    return False, phase, False, False, False, False, False, spread
  (
      lane,
      bottleneck,
      red_switch_contrast,
      forced_switch_red_contrast,
      forced_switch_red_win_contrast,
  ) = row_kind_flags(features, mask, args)
  if mode == "lane_contrast":
    keep = lane
  elif mode == "bottleneck":
    keep = bottleneck
  elif mode == "lane_or_bottleneck":
    keep = lane or bottleneck
  elif mode == "red_switch_contrast":
    keep = red_switch_contrast
  elif mode == "lane_or_red_switch":
    keep = lane or red_switch_contrast
  elif mode == "base_lost_led_follow_available":
    keep = base_lost_led_follow_available(
        features, mask, base_action, require_nonred_base=False
    )
  elif mode == "base_nonred_lost_led_follow_available":
    keep = base_lost_led_follow_available(
        features, mask, base_action, require_nonred_base=True
    )
  elif mode == "forced_switch_red_contrast":
    keep = forced_switch_red_contrast
  elif mode == "forced_switch_red_win_contrast":
    keep = forced_switch_red_win_contrast
  else:
    keep = True
  return (
      bool(keep),
      phase,
      lane,
      bottleneck,
      red_switch_contrast,
      forced_switch_red_contrast,
      forced_switch_red_win_contrast,
      spread,
  )


def indexed_feature(features: np.ndarray, name: str) -> np.ndarray:
  if name not in APPENDED_ACTION_FEATURE_INDEX:
    return np.zeros(features.shape[0], dtype=np.float32)
  return features[:, APPENDED_ACTION_FEATURE_INDEX[name]]


def token_loss_mask(features: np.ndarray) -> np.ndarray:
  return (
      (features[:, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX] > 0.5)
      | (indexed_feature(features, "token_loss_newly_loses_led") > 0.5)
  )


def red_switch_mask(features: np.ndarray) -> np.ndarray:
  return (
      (features[:, ACTION_FEATURE_IS_RED_INDEX] > 0.5)
      & token_loss_mask(features)
  )


def red_win_switch_mask(features: np.ndarray) -> np.ndarray:
  # Numeric slot 19 is would_win_after_play encoded as 1.0 win, -1.0 loss,
  # and 0.0 when the action is not a normal play action.
  return red_switch_mask(features) & (
      features[:, ACTION_FEATURE_PREFIX_SIZE + 19] > 0.5
  )


def switch_curriculum_policy_target(
    num_actions: int,
    candidates: list[int],
    features: np.ndarray,
    policy_scores: dict[int, float],
    deterministic_rows: dict[int, dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict]:
  red_switches = red_switch_mask(features)
  red_win_switches = red_win_switch_mask(features)
  token_losses = token_loss_mask(features)
  follows_led = features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
  red_candidates = [int(a) for a in candidates if red_switches[int(a)]]
  if not red_candidates:
    return None, {}

  def red_key(action: int) -> tuple:
    row = deterministic_rows.get(int(action), {})
    return (
        1 if bool(red_win_switches[int(action)]) else 0,
        1 if bool(row.get("responsible_red_win_switch", False)) else 0,
        1 if bool(row.get("own_feasible", False)) else 0,
        0 if bool(row.get("over_target_would_win", False)) else 1,
        -float(row.get("own_total_deficit", 0.0)),
        float(policy_scores.get(int(action), 0.0)),
        -int(action),
    )

  def follow_key(action: int) -> tuple:
    row = deterministic_rows.get(int(action), {})
    return (
        1 if bool(row.get("own_feasible", False)) else 0,
        float(policy_scores.get(int(action), 0.0)),
        -float(row.get("own_public_slot_damage", 0.0)),
        -int(action),
    )

  best_red = int(max(red_candidates, key=red_key))
  follow_candidates = [int(a) for a in candidates if follows_led[int(a)]]
  best_follow = (
      int(max(follow_candidates, key=follow_key)) if follow_candidates else None
  )
  excluded = {best_red}
  if best_follow is not None:
    excluded.add(int(best_follow))
  other_candidates = [
      int(a) for a in candidates
      if int(a) not in excluded
      and not (bool(token_losses[int(a)]) and not bool(red_switches[int(a)]))
  ]
  if not other_candidates:
    other_candidates = [int(a) for a in candidates if int(a) not in excluded]

  target = np.zeros(num_actions, dtype=np.float32)
  red_mass = float(np.clip(args.switch_curriculum_red_mass, 0.0, 1.0))
  follow_mass = float(np.clip(args.switch_curriculum_follow_mass, 0.0, 1.0))
  other_mass = float(np.clip(args.switch_curriculum_other_mass, 0.0, 1.0))
  target_mode = str(
      getattr(args, "switch_curriculum_target_mode", "balanced") or "balanced"
  )
  red_top_used = False
  red_value_delta = None
  if best_follow is None:
    red_mass = max(red_mass, 1.0 - other_mass)
    follow_mass = 0.0
  else:
    red_value = float(policy_scores.get(int(best_red), 0.0))
    follow_value = float(policy_scores.get(int(best_follow), 0.0))
    red_value_delta = red_value - follow_value
    red_row = deterministic_rows.get(int(best_red), {})
    red_top_min_delta = float(
        getattr(args, "switch_curriculum_red_top_min_value_delta", 0.08)
    )
    red_top_require_win = bool(
        getattr(args, "switch_curriculum_red_top_require_win", False)
    )
    red_top_qualified = (
        bool(red_win_switches[int(best_red)])
        or bool(red_row.get("responsible_red_win_switch", False))
        or red_value_delta >= red_top_min_delta
    )
    if red_top_require_win:
      red_top_qualified = (
          bool(red_win_switches[int(best_red)])
          or bool(red_row.get("responsible_red_win_switch", False))
      )
    if target_mode == "red_exception_top" and red_top_qualified:
      red_top_used = True
      red_mass = float(np.clip(
          getattr(args, "switch_curriculum_red_top_mass", 0.68), 0.0, 1.0
      ))
      follow_mass = min(follow_mass, max(0.0, 1.0 - red_mass - other_mass))
    target[int(best_follow)] += follow_mass
  target[best_red] += red_mass

  used_mass = float(np.sum(target))
  residual = max(0.0, min(other_mass, 1.0 - used_mass))
  if residual > 0.0 and other_candidates:
    best_other = int(max(
        other_candidates,
        key=lambda action: (float(policy_scores.get(int(action), 0.0)), -int(action)),
    ))
    target[best_other] += residual
  total = float(np.sum(target))
  if total <= 0.0:
    return None, {}
  target /= total
  return target, {
      "switch_curriculum_policy_target_used": True,
      "switch_curriculum_best_follow_action": best_follow,
      "switch_curriculum_best_red_action": int(best_red),
      "switch_curriculum_best_red_win": bool(red_win_switches[int(best_red)]),
      "switch_curriculum_red_top_used": bool(red_top_used),
      "switch_curriculum_red_value_delta": (
          None if red_value_delta is None else float(red_value_delta)
      ),
      "switch_curriculum_red_mass": float(target[best_red]),
      "switch_curriculum_follow_mass": (
          None if best_follow is None else float(target[int(best_follow)])
      ),
      "switch_curriculum_nonred_switch_mass": float(np.sum(
          target[token_losses & ~red_switches]
      )),
  }


def add_candidate(
    selected: list[int],
    legal_set: set[int],
    action: int,
) -> None:
  action = int(action)
  if action in legal_set and action not in selected:
    selected.append(action)


def add_extreme_candidate(
    selected: list[int],
    legal: list[int],
    values: np.ndarray,
    *,
    high: bool,
) -> None:
  if not legal:
    return
  legal_values = np.asarray(values, dtype=np.float32)
  if high:
    action = max(legal, key=lambda a: (float(legal_values[int(a)]), -int(a)))
  else:
    action = min(legal, key=lambda a: (float(legal_values[int(a)]), int(a)))
  add_candidate(selected, set(legal), int(action))


def candidate_actions(
    legal: list[int],
    policy: np.ndarray,
    features: np.ndarray,
    structural: np.ndarray,
    args: argparse.Namespace,
) -> list[int]:
  legal = sorted(int(action) for action in legal)
  max_actions = int(args.max_actions_per_state)
  if max_actions <= 0 or len(legal) <= max_actions:
    return legal
  ranked = sorted(
      legal,
      key=lambda action: (float(policy[int(action)]), -int(action)),
      reverse=True,
  )
  selected: list[int] = []
  legal_set = set(legal)
  for action in ranked[:max(1, int(args.top_policy_actions))]:
    add_candidate(selected, legal_set, int(action))

  if bool(args.feature_candidates):
    # These are the core lane-liquidity contrasts this replay is meant to
    # teach. Add them before generic extremes so capped candidate sets do not
    # silently drop red/trump switch exceptions.
    add_extreme_candidate(
        selected, legal, features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX], high=True
    )
    add_extreme_candidate(
        selected,
        legal,
        indexed_feature(features, "token_loss_newly_loses_led"),
        high=True,
    )
    add_extreme_candidate(
        selected,
        legal,
        indexed_feature(features, "token_loss_newly_loses_led"),
        high=False,
    )
    add_extreme_candidate(
        selected,
        legal,
        red_switch_mask(features).astype(np.float32),
        high=True,
    )
    add_extreme_candidate(
        selected,
        legal,
        red_win_switch_mask(features).astype(np.float32),
        high=True,
    )
    add_extreme_candidate(selected, legal, structural, high=True)
    add_extreme_candidate(selected, legal, structural, high=False)
    add_extreme_candidate(
        selected,
        legal,
        indexed_feature(features, "legal_z_exit_min_player_lane_surplus_after"),
        high=True,
    )
    add_extreme_candidate(
        selected,
        legal,
        indexed_feature(features, "legal_z_exit_lane_surplus_damage"),
        high=False,
    )
    add_extreme_candidate(
        selected,
        legal,
        indexed_feature(features, "own_future_min_colors_after"),
        high=True,
    )
    add_extreme_candidate(
        selected,
        legal,
        indexed_feature(features, "own_future_zero_exit_frac_after"),
        high=False,
    )
    add_extreme_candidate(
        selected,
        legal,
        indexed_feature(features, "own_future_one_exit_frac_after"),
        high=False,
    )
    add_extreme_candidate(
        selected,
        legal,
        indexed_feature(features, "future_no_exit_after"),
        high=False,
    )
    add_extreme_candidate(
        selected, legal, features[:, ACTION_FEATURE_RANK_NORM_INDEX], high=False
    )
    add_extreme_candidate(
        selected, legal, features[:, ACTION_FEATURE_RANK_NORM_INDEX], high=True
    )
    add_extreme_candidate(
        selected, legal, features[:, ACTION_FEATURE_IS_RED_INDEX], high=False
    )
    add_extreme_candidate(
        selected, legal, features[:, ACTION_FEATURE_IS_RED_INDEX], high=True
    )
    add_extreme_candidate(
        selected,
        legal,
        red_switch_mask(features).astype(np.float32),
        high=True,
    )
    add_extreme_candidate(
        selected,
        legal,
        red_win_switch_mask(features).astype(np.float32),
        high=True,
    )

  for action in ranked:
    add_candidate(selected, legal_set, int(action))
    if len(selected) >= max_actions:
      break
  return sorted(selected[:max_actions])


def candidate_action_filter_allows(
    args: argparse.Namespace,
    features: np.ndarray,
    action: int,
) -> bool:
  mode = str(getattr(args, "candidate_action_filter", "all"))
  if mode == "all":
    return True
  follows = bool(features[int(action), ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5)
  loses = bool(token_loss_mask(features)[int(action)])
  if mode == "follow_led":
    return follows
  if mode == "no_token_loss":
    return not loses
  if mode == "follow_no_token_loss":
    return bool(follows and not loses)
  raise ValueError(f"unknown candidate_action_filter={mode!r}")


def action_targets_for_state(
    args: argparse.Namespace,
    state,
    player: int,
    policy: np.ndarray,
    legal: list[int],
    features: np.ndarray,
    model,
    model_args,
    device,
    bots,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict | None,
]:
  num_actions = state.num_distinct_actions()
  mask = legal_mask(num_actions, legal)
  structural = structural_scores(features, policy, None, None, args)
  base = int(max(legal, key=lambda action: float(policy[int(action)])))
  (
      keep,
      phase,
      lane,
      bottleneck,
      red_switch_contrast,
      forced_switch_red_contrast,
      forced_switch_red_win_contrast,
      structural_spread,
  ) = keep_row(
      features, mask, structural, split_names(args.phases), args, base_action=base
  )
  if not keep:
    return None, None, None, None, None, None
  trick_number = int(getattr(state, "_trick_number", 0))
  if not phase_trick_allowed(args, phase, trick_number):
    return None, None, None, None, None, None
  candidates = candidate_actions(legal, policy, features, structural, args)
  if str(getattr(args, "candidate_action_filter", "all")) != "all":
    candidates = sorted({
        int(base),
        *[
            int(action)
            for action in candidates
            if candidate_action_filter_allows(args, features, int(action))
        ],
    })
  if len(candidates) < 2:
    return None, None, None, None, None, None
  red_switches_all = red_switch_mask(features)
  red_win_switches_all = red_win_switch_mask(features)
  token_losses_all = token_loss_mask(features)
  switch_curriculum_row = bool(red_switch_contrast or forced_switch_red_contrast)

  legal_structural = structural[np.asarray(candidates, dtype=np.int64)]
  baseline = float(np.mean(legal_structural)) if legal_structural.size else 0.0
  scale = max(1e-6, float(args.immediate_target_scale))
  rollout_weight = float(np.clip(args.rollout_weight, 0.0, 1.0))
  rollouts = max(0, int(args.rollouts_per_action))
  targets = np.zeros(num_actions, dtype=np.float32)
  target_mask = np.zeros(num_actions, dtype=np.float32)
  risk_targets = np.zeros(num_actions, dtype=np.float32)
  risk_mask = np.zeros(num_actions, dtype=np.float32)
  policy_scores: dict[int, float] = {}
  rollout_probs = {}
  (
      deterministic_values,
      deterministic_risks,
      deterministic_rows,
  ) = deterministic_liveness_labels(state, player, candidates, base, args)
  deterministic_weight = float(
      np.clip(float(args.deterministic_liveness_weight), 0.0, 1.0)
  )
  deterministic_risk_weight = float(
      np.clip(float(args.deterministic_liveness_risk_weight), 0.0, 1.0)
  )
  paired_seeds = []
  if rollouts > 0 and bool(args.paired_rollout_seeds):
    paired_seeds = [
        int(seed)
        for seed in rng.integers(
            0, np.iinfo(np.uint32).max, size=rollouts, dtype=np.uint32
        )
    ]
  for action in candidates:
    immediate = math.tanh((float(structural[int(action)]) - baseline) / scale)
    if rollouts > 0:
      outcomes = []
      for rollout_idx in range(rollouts):
        rollout_rng = (
            np.random.default_rng(paired_seeds[rollout_idx])
            if paired_seeds else rng
        )
        outcomes.append(rollout_survival_probability(
            args,
            state,
            int(action),
            model,
            model_args,
            device,
            bots,
            rollout_rng,
        ))
      survival = float(np.mean(outcomes)) if outcomes else 0.0
      rollout_probs[int(action)] = survival
      rollout_value = 2.0 * survival - 1.0
      value = rollout_weight * rollout_value + (1.0 - rollout_weight) * immediate
    else:
      value = immediate
      survival = (value + 1.0) / 2.0
    if deterministic_weight > 0.0 and int(action) in deterministic_values:
      value = (
          (1.0 - deterministic_weight) * float(value)
          + deterministic_weight * float(deterministic_values[int(action)])
      )
    if (
        bool(getattr(args, "switch_curriculum_policy_target", False))
        and bool(switch_curriculum_row)
    ):
      if bool(red_win_switches_all[int(action)]):
        value += float(getattr(args, "switch_curriculum_red_win_bonus", 0.40))
      elif bool(red_switches_all[int(action)]):
        value += float(getattr(args, "switch_curriculum_red_bonus", 0.18))
      elif bool(token_losses_all[int(action)]):
        value -= float(getattr(args, "switch_curriculum_nonred_penalty", 0.35))
    value = float(np.clip(value, -1.0, 1.0))
    survival_risk = float(np.clip(1.0 - survival, 0.0, 1.0))
    if deterministic_risk_weight > 0.0 and int(action) in deterministic_risks:
      survival_risk = (
          (1.0 - deterministic_risk_weight) * survival_risk
          + deterministic_risk_weight * float(deterministic_risks[int(action)])
      )
    targets[int(action)] = value
    target_mask[int(action)] = 1.0
    risk_targets[int(action)] = float(np.clip(survival_risk, 0.0, 1.0))
    risk_mask[int(action)] = 1.0
    policy_scores[int(action)] = value

  deterministic_teacher_best = None
  use_exact_teacher_key = False
  teacher_key_order = str(
      getattr(args, "deterministic_teacher_key_order", "auto") or "auto"
  )
  if bool(getattr(args, "deterministic_teacher_policy_target", False)):
    use_exact_teacher_key = bool(
        float(getattr(args, "deterministic_exact_legal_pressure_weight", 0.0)) > 0.0
        or float(
            getattr(args, "deterministic_exact_legal_pressure_risk_weight", 0.0)
        ) > 0.0
        or float(
            getattr(args, "deterministic_exact_table_liveness_weight", 0.0)
        ) > 0.0
        or float(
            getattr(args, "deterministic_exact_table_liveness_risk_weight", 0.0)
        ) > 0.0
        or teacher_key_order == "table_liveness"
    )
    if deterministic_rows:
      deterministic_teacher_best = int(max(
          deterministic_rows,
          key=lambda action: deterministic_teacher_key(
              deterministic_rows[int(action)],
              use_exact=use_exact_teacher_key,
              key_order=teacher_key_order,
          ),
      ))
  auxiliary_mode = str(
      getattr(args, "deterministic_teacher_auxiliary_mode", "auto") or "auto"
  )
  align_teacher_auxiliary = should_align_deterministic_teacher_auxiliary_targets(
      deterministic_teacher_best,
      auxiliary_mode,
      rollouts,
      switch_curriculum_active=(
          bool(getattr(args, "switch_curriculum_policy_target", False))
          and bool(switch_curriculum_row)
      ),
  )
  if align_teacher_auxiliary:
    teacher_order = align_deterministic_teacher_auxiliary_targets(
        candidates,
        deterministic_rows,
        targets,
        risk_targets,
        use_exact=use_exact_teacher_key,
        key_order=teacher_key_order,
    )
    if teacher_order[0] != deterministic_teacher_best:
      raise AssertionError("Teacher policy and auxiliary target order disagree")
    for action in candidates:
      policy_scores[int(action)] = float(targets[int(action)])

  labeled_values = targets[target_mask > 0.5]
  target_spread = (
      float(np.max(labeled_values) - np.min(labeled_values))
      if labeled_values.size else 0.0
  )
  if target_spread < float(args.min_target_spread):
    return None, None, None, None, None, None
  labeled_risks = risk_targets[risk_mask > 0.5]
  risk_spread = (
      float(np.max(labeled_risks) - np.min(labeled_risks))
      if labeled_risks.size else 0.0
  )
  if risk_spread < float(args.min_risk_spread):
    return None, None, None, None, None, None
  if deterministic_teacher_best is not None:
    policy_target = np.zeros(num_actions, dtype=np.float32)
    policy_target[int(deterministic_teacher_best)] = 1.0
  else:
    policy_target = softmax_over_actions(
        policy_scores, num_actions, float(args.policy_target_temperature)
    )
  switch_target_details = {}
  if (
      bool(getattr(args, "switch_curriculum_policy_target", False))
      and bool(switch_curriculum_row)
  ):
    switch_target, switch_target_details = switch_curriculum_policy_target(
        num_actions,
        candidates,
        features,
        policy_scores,
        deterministic_rows,
        args,
    )
    if switch_target is not None:
      policy_target = switch_target
  best = int(max(policy_scores, key=lambda action: policy_scores[action]))
  policy_target_top = int(np.argmax(policy_target))
  risk_best = int(min(
      [int(action) for action in candidates],
      key=lambda action: (float(risk_targets[action]), -float(policy[action])),
  ))
  deterministic_best = None
  if deterministic_values:
    deterministic_best = int(max(
        deterministic_values,
        key=lambda action: (
            float(deterministic_values[action]),
            -float(deterministic_risks.get(action, 0.0)),
            float(policy[action]),
        ),
    ))
  deterministic_base_row = deterministic_rows.get(base, {})
  deterministic_best_row = deterministic_rows.get(deterministic_best, {})
  stats = {
      "phase": phase,
      "trick_number": int(trick_number),
      "lane_contrast": bool(lane),
      "bottleneck": bool(bottleneck),
      "red_switch_contrast": bool(red_switch_contrast),
      "forced_switch_red_contrast": bool(forced_switch_red_contrast),
      "forced_switch_red_win_contrast": bool(forced_switch_red_win_contrast),
      "structural_spread": float(structural_spread),
      "target_spread": float(target_spread),
      "risk_spread": float(risk_spread),
      "candidate_count": int(len(candidates)),
      "baseline_action": int(base),
      "best_action": int(best),
      "policy_target_top_action": int(policy_target_top),
      "risk_best_action": int(risk_best),
      "baseline_follows_led": bool(
          features[base, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
      ),
      "baseline_token_loss": bool(token_losses_all[base]),
      "baseline_red": bool(features[base, ACTION_FEATURE_IS_RED_INDEX] > 0.5),
      "baseline_red_switch": bool(red_switches_all[base]),
      "baseline_lost_led_follow_available": bool(
          base_lost_led_follow_available(
              features, mask, base, require_nonred_base=False
          )
      ),
      "baseline_nonred_lost_led_follow_available": bool(
          base_lost_led_follow_available(
              features, mask, base, require_nonred_base=True
          )
      ),
      "baseline_target": float(targets[base]) if target_mask[base] > 0.5 else None,
      "baseline_risk": float(risk_targets[base]) if risk_mask[base] > 0.5 else None,
      "best_target": float(targets[best]),
      "best_risk": float(risk_targets[best]),
      "risk_best_target": float(targets[risk_best]),
      "risk_best_risk": float(risk_targets[risk_best]),
      "deterministic_liveness_used": bool(deterministic_values),
      "deterministic_best_action": deterministic_best,
      "deterministic_teacher_policy_target_used": (
          deterministic_teacher_best is not None
      ),
      "deterministic_teacher_auxiliary_aligned": align_teacher_auxiliary,
      "deterministic_teacher_best_action": deterministic_teacher_best,
      "deterministic_best_is_target_best": (
          None if deterministic_best is None else bool(deterministic_best == best)
      ),
      "deterministic_teacher_best_is_target_best": (
          None if deterministic_teacher_best is None
          else bool(deterministic_teacher_best == best)
      ),
      "deterministic_best_risk": (
          None if deterministic_best is None
          else float(deterministic_risks.get(deterministic_best, 0.0))
      ),
      "deterministic_base_risk": (
          float(deterministic_risks[base]) if base in deterministic_risks else None
      ),
      "deterministic_best_own_feasible": (
          None if deterministic_best is None
          else bool(deterministic_best_row.get("own_feasible", False))
      ),
      "deterministic_base_own_feasible": (
          bool(deterministic_base_row.get("own_feasible", False))
          if deterministic_base_row else None
      ),
      "deterministic_best_lost_led_token": (
          None if deterministic_best is None
          else bool(deterministic_best_row.get("lost_led_token", False))
      ),
      "deterministic_teacher_best_lost_led_token": (
          None if deterministic_teacher_best is None
          else bool(
              deterministic_rows.get(deterministic_teacher_best, {}).get(
                  "lost_led_token", False
              )
          )
      ),
      "deterministic_base_lost_led_token": (
          bool(deterministic_base_row.get("lost_led_token", False))
          if deterministic_base_row else None
      ),
      "deterministic_best_min_lane_surplus": (
          None if deterministic_best is None
          else int(deterministic_best_row.get("min_player_lane_surplus_after", 0))
      ),
      "deterministic_base_min_lane_surplus": (
          int(deterministic_base_row.get("min_player_lane_surplus_after", 0))
          if deterministic_base_row else None
      ),
      "deterministic_teacher_best_table_min_margin": (
          None if deterministic_teacher_best is None
          else int(
              deterministic_rows.get(deterministic_teacher_best, {}).get(
                  "table_min_margin_after", 0
              )
          )
      ),
      "deterministic_base_table_min_margin": (
          int(deterministic_base_row.get("table_min_margin_after", 0))
          if deterministic_base_row else None
      ),
      "deterministic_teacher_best_table_dead_players": (
          None if deterministic_teacher_best is None
          else int(
              deterministic_rows.get(deterministic_teacher_best, {}).get(
                  "table_dead_margin_players_after", 0
              )
          )
      ),
      "deterministic_base_table_dead_players": (
          int(deterministic_base_row.get("table_dead_margin_players_after", 0))
          if deterministic_base_row else None
      ),
      "best_action_kind": action_kind_name(features, best),
      "best_follows_led": bool(features[best, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5),
      "best_token_loss": bool(
          safe_feature(features, "token_loss_newly_loses_led")[best] > 0.5
      ),
      "best_red": bool(features[best, ACTION_FEATURE_IS_RED_INDEX] > 0.5),
      "best_red_switch": bool(red_switches_all[best]),
      "best_red_win_switch": bool(red_win_switches_all[best]),
      "policy_target_top_follows_led": bool(
          features[policy_target_top, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
      ),
      "policy_target_top_token_loss": bool(
          safe_feature(features, "token_loss_newly_loses_led")[policy_target_top]
          > 0.5
      ),
      "policy_target_top_red": bool(
          features[policy_target_top, ACTION_FEATURE_IS_RED_INDEX] > 0.5
      ),
      "policy_target_top_red_switch": bool(red_switches_all[policy_target_top]),
      "policy_target_top_red_win_switch": bool(
          red_win_switches_all[policy_target_top]
      ),
      "policy_target_follow_mass": float(np.sum(
          policy_target[features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5]
      )),
      "policy_target_red_switch_mass": float(np.sum(
          policy_target[red_switches_all]
      )),
      "policy_target_red_win_switch_mass": float(np.sum(
          policy_target[red_win_switches_all]
      )),
      "policy_target_nonred_switch_mass": float(np.sum(
          policy_target[token_losses_all & ~red_switches_all]
      )),
      "best_rollout_survival": rollout_probs.get(best),
      "risk_best_rollout_survival": rollout_probs.get(risk_best),
  }
  stats.update(switch_target_details)
  if not bool(args.emit_action_risk_labels):
    risk_targets = None
    risk_mask = None
  return targets, target_mask, risk_targets, risk_mask, policy_target, stats


def phase_trick_allowed(
    args: argparse.Namespace, phase: str, trick_number: int
) -> bool:
  if phase != "play":
    return True
  min_trick = int(getattr(args, "min_trick_number", -1))
  max_trick = int(getattr(args, "max_trick_number", -1))
  if min_trick >= 0 and trick_number < min_trick:
    return False
  if max_trick >= 0 and trick_number > max_trick:
    return False
  return True


def behavior_policy_target_for_state(
    args: argparse.Namespace,
    state,
    player: int,
    action: int,
    policy: np.ndarray,
    legal: list[int],
    features: np.ndarray,
) -> tuple[np.ndarray | None, dict | None]:
  """Builds a one-hot behavior-cloning target for the source trajectory action."""
  num_actions = state.num_distinct_actions()
  mask = legal_mask(num_actions, legal)
  structural = structural_scores(features, policy, None, None, args)
  base = int(max(legal, key=lambda candidate: float(policy[int(candidate)])))
  (
      keep,
      phase,
      lane,
      bottleneck,
      red_switch_contrast,
      forced_switch_red_contrast,
      forced_switch_red_win_contrast,
      structural_spread,
  ) = keep_row(
      features, mask, structural, split_names(args.phases), args, base_action=base
  )
  trick_number = int(getattr(state, "_trick_number", 0))
  if not keep or not phase_trick_allowed(args, phase, trick_number):
    return None, None
  action = int(action)
  if action < 0 or action >= num_actions or mask[action] <= 0.5:
    return None, None
  red_switches_all = red_switch_mask(features)
  red_win_switches_all = red_win_switch_mask(features)
  token_losses_all = token_loss_mask(features)
  follows_led = bool(features[action, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5)
  red_switch = bool(red_switches_all[action])
  red_win_switch = bool(red_win_switches_all[action])
  token_loss = bool(token_losses_all[action])
  action_filter = str(getattr(args, "behavior_action_filter", "all"))
  if action_filter == "follow_led" and not follows_led:
    return None, None
  if action_filter == "follow_or_red_switch" and not (
      follows_led or red_switch
  ):
    return None, None
  if action_filter == "follow_or_red_win_switch" and not (
      follows_led or red_win_switch
  ):
    return None, None
  if action_filter == "not_nonred_token_loss" and token_loss and not red_switch:
    return None, None

  policy_target = one_hot_policy(num_actions, action)
  return policy_target, {
      "phase": phase,
      "trick_number": int(trick_number),
      "lane_contrast": bool(lane),
      "bottleneck": bool(bottleneck),
      "red_switch_contrast": bool(red_switch_contrast),
      "forced_switch_red_contrast": bool(forced_switch_red_contrast),
      "forced_switch_red_win_contrast": bool(forced_switch_red_win_contrast),
      "structural_spread": float(structural_spread),
      "target_spread": 1.0,
      "risk_spread": 0.0,
      "candidate_count": int(len(legal)),
      "baseline_action": int(base),
      "best_action": int(action),
      "policy_target_top_action": int(action),
      "baseline_follows_led": bool(
          features[base, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
      ),
      "baseline_token_loss": bool(token_losses_all[base]),
      "baseline_red": bool(features[base, ACTION_FEATURE_IS_RED_INDEX] > 0.5),
      "baseline_red_switch": bool(red_switches_all[base]),
      "baseline_lost_led_follow_available": bool(
          base_lost_led_follow_available(
              features, mask, base, require_nonred_base=False
          )
      ),
      "baseline_nonred_lost_led_follow_available": bool(
          base_lost_led_follow_available(
              features, mask, base, require_nonred_base=True
          )
      ),
      "baseline_target": None,
      "baseline_risk": None,
      "best_target": 1.0,
      "best_risk": 0.0,
      "risk_best_target": 1.0,
      "risk_best_risk": 0.0,
      "deterministic_liveness_used": False,
      "deterministic_best_action": None,
      "deterministic_teacher_policy_target_used": False,
      "deterministic_teacher_best_action": None,
      "deterministic_best_is_target_best": None,
      "deterministic_teacher_best_is_target_best": None,
      "deterministic_best_risk": None,
      "deterministic_base_risk": None,
      "deterministic_best_own_feasible": None,
      "deterministic_base_own_feasible": None,
      "deterministic_best_lost_led_token": None,
      "deterministic_teacher_best_lost_led_token": None,
      "deterministic_base_lost_led_token": None,
      "deterministic_best_min_lane_surplus": None,
      "deterministic_base_min_lane_surplus": None,
      "best_action_kind": action_kind_name(features, action),
      "best_follows_led": follows_led,
      "best_token_loss": token_loss,
      "best_red": bool(features[action, ACTION_FEATURE_IS_RED_INDEX] > 0.5),
      "best_red_switch": red_switch,
      "best_red_win_switch": red_win_switch,
      "policy_target_top_follows_led": follows_led,
      "policy_target_top_token_loss": token_loss,
      "policy_target_top_red": bool(
          features[action, ACTION_FEATURE_IS_RED_INDEX] > 0.5
      ),
      "policy_target_top_red_switch": red_switch,
      "policy_target_top_red_win_switch": red_win_switch,
      "policy_target_follow_mass": float(
          np.sum(policy_target[features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5])
      ),
      "policy_target_red_switch_mass": float(np.sum(policy_target[red_switches_all])),
      "policy_target_red_win_switch_mass": float(
          np.sum(policy_target[red_win_switches_all])
      ),
      "policy_target_nonred_switch_mass": float(
          np.sum(policy_target[token_losses_all & ~red_switches_all])
      ),
      "best_rollout_survival": None,
      "risk_best_rollout_survival": None,
      "behavior_policy_target": True,
      "behavior_action_filter": action_filter,
  }


def round_raw_scores(state) -> np.ndarray:
  if hasattr(state, "raw_round_scores"):
    return np.array(state.raw_round_scores(), dtype=np.float32)
  return np.array(state.returns(), dtype=np.float32)


def make_replay_bot(name: str, seed: int, args: argparse.Namespace):
  if str(name).startswith("liveness_key_teacher"):
    teacher_args = _model_args(int(args.players))
    teacher_args.liveness_teacher_base_bot = str(args.liveness_teacher_base_bot)
    teacher_args.liveness_teacher_phases = str(args.liveness_teacher_phases)
    teacher_args.liveness_teacher_min_trick_number = int(
        args.liveness_teacher_min_trick_number
    )
    teacher_args.liveness_teacher_exact_legal_pressure = bool(
        args.liveness_teacher_exact_legal_pressure
    )
    teacher_args.liveness_teacher_exact_key_order = str(
        args.liveness_teacher_exact_key_order
    )
    teacher_args.liveness_teacher_sample_limit = 0
    return LivenessKeyTeacherBot(str(name), teacher_args, seed=seed)
  return make_bot(str(name), seed=seed)


def collect_round(
    args: argparse.Namespace,
    game,
    state,
    model,
    model_args,
    device,
    bots,
    rng: np.random.Generator,
    *,
    match_idx: int,
    round_idx: int,
) -> tuple[list[tuple], list[dict], object]:
  rows = []
  stats_rows = []
  num_actions = game.num_distinct_actions()
  while not state.is_terminal():
    if state.is_chance_node():
      actions, probs = zip(*state.chance_outcomes())
      state.apply_action(int(rng.choice(np.array(actions, dtype=np.int64), p=probs)))
      continue
    player = int(state.current_player())
    legal = [int(action) for action in state.legal_actions(player)]
    policy, action = source_policy_action(
        args, state, player, num_actions, model, model_args, device, bots, rng
    )
    if len(legal) >= 2:
      features = action_feature_matrix(state, player, num_actions)
      if bool(getattr(args, "behavior_policy_target", False)):
        policy_target, stats = behavior_policy_target_for_state(
            args,
            state,
            player,
            int(action),
            policy,
            legal,
            features,
        )
        action_values = None
        action_mask = None
        action_risks = None
        action_risk_mask = None
      else:
        (
            action_values,
            action_mask,
            action_risks,
            action_risk_mask,
            policy_target,
            stats,
        ) = action_targets_for_state(
            args,
            state,
            player,
            policy,
            legal,
            features,
            model,
            model_args,
            device,
            bots,
            rng,
        )
      if stats is not None:
        stats["match_idx"] = int(match_idx)
        stats["round_idx"] = int(round_idx)
        stats["player"] = int(player)
        stats["source_action"] = int(action)
        # The source trajectory may be produced by a deployed wrapper such as
        # the public liveness shield while the saved policy context remains the
        # raw neural policy.  Downstream selectors deliberately reserve
        # ``policy_base_action`` for that deployed incumbent so a teacher's
        # ``best_action`` can never be mistaken for the live base action.
        stats["policy_base_action"] = int(action)
        stats["legal_actions"] = [int(value) for value in legal]
        stats["public_history"] = public_history_metadata(state, player)
        rows.append((
            np.array(state.observation_tensor(player), dtype=np.float32),
            legal_mask(num_actions, legal),
            policy_target,
            np.zeros(int(args.players), dtype=np.float32),
            np.zeros(int(args.players), dtype=np.float32),
            int(action),
            int(player),
            action_risks,
            action_risk_mask,
            action_values,
            action_mask,
            features,
            None,
        ))
        stats_rows.append(stats)
    state.apply_action(int(action))
  paradox_target = terminal_paradox_target(state, int(args.players))
  any_round_paradox = bool(np.any(paradox_target > 0.5))
  source_filter = str(getattr(args, "source_round_filter", "all"))
  if source_filter == "survived" and any_round_paradox:
    return [], [], state
  if source_filter == "paradox" and not any_round_paradox:
    return [], [], state
  finalized = []
  for row in rows:
    finalized.append((
        row[0],
        row[1],
        row[2],
        np.zeros(int(args.players), dtype=np.float32),
        paradox_target,
        row[5],
        row[6],
        row[7],
        row[8],
        row[9],
        row[10],
        row[11],
        paradox_target,
    ))
  return finalized, stats_rows, state


def row_group_ids_from_stats(
    stats_rows: list[dict], *, source_seed: int | None = None
) -> list[str]:
  group_ids = []
  source_prefix = "" if source_seed is None else f"seed={int(source_seed)}|"
  for idx, row in enumerate(stats_rows):
    if "match_idx" in row and "round_idx" in row:
      group_ids.append(
          f"{source_prefix}m{int(row['match_idx'])}:r{int(row['round_idx'])}"
      )
    else:
      group_ids.append(f"{source_prefix}row:{idx}")
  return group_ids


def summarize_stats(stats_rows: list[dict]) -> dict:
  phase_counts = Counter(row["phase"] for row in stats_rows)
  trick_counts = Counter(
      str(row["trick_number"])
      for row in stats_rows
      if row.get("phase") == "play" and row.get("trick_number") is not None
  )
  kind_counts = Counter(row["best_action_kind"] for row in stats_rows)
  baseline_values = [
      float(row["baseline_target"])
      for row in stats_rows
      if row.get("baseline_target") is not None
  ]
  best_values = [float(row["best_target"]) for row in stats_rows]
  baseline_risks = [
      float(row["baseline_risk"])
      for row in stats_rows
      if row.get("baseline_risk") is not None
  ]
  best_risks = [float(row["best_risk"]) for row in stats_rows]
  risk_best_risks = [float(row["risk_best_risk"]) for row in stats_rows]
  deterministic_base_risks = [
      float(row["deterministic_base_risk"])
      for row in stats_rows
      if row.get("deterministic_base_risk") is not None
  ]
  deterministic_best_risks = [
      float(row["deterministic_best_risk"])
      for row in stats_rows
      if row.get("deterministic_best_risk") is not None
  ]
  deterministic_lane_deltas = [
      float(row["deterministic_best_min_lane_surplus"])
      - float(row["deterministic_base_min_lane_surplus"])
      for row in stats_rows
      if row.get("deterministic_best_min_lane_surplus") is not None
      and row.get("deterministic_base_min_lane_surplus") is not None
  ]
  deterministic_table_margin_deltas = [
      float(row["deterministic_teacher_best_table_min_margin"])
      - float(row["deterministic_base_table_min_margin"])
      for row in stats_rows
      if row.get("deterministic_teacher_best_table_min_margin") is not None
      and row.get("deterministic_base_table_min_margin") is not None
  ]
  deterministic_table_dead_deltas = [
      float(row["deterministic_base_table_dead_players"])
      - float(row["deterministic_teacher_best_table_dead_players"])
      for row in stats_rows
      if row.get("deterministic_teacher_best_table_dead_players") is not None
      and row.get("deterministic_base_table_dead_players") is not None
  ]
  policy_target_follow_masses = [
      float(row.get("policy_target_follow_mass", 0.0)) for row in stats_rows
  ]
  policy_target_red_switch_masses = [
      float(row.get("policy_target_red_switch_mass", 0.0)) for row in stats_rows
  ]
  policy_target_red_win_switch_masses = [
      float(row.get("policy_target_red_win_switch_mass", 0.0))
      for row in stats_rows
  ]
  policy_target_nonred_switch_masses = [
      float(row.get("policy_target_nonred_switch_mass", 0.0))
      for row in stats_rows
  ]
  improvements = [
      float(row["best_target"]) - float(row["baseline_target"])
      for row in stats_rows
      if row.get("baseline_target") is not None
  ]
  return {
      "phase_counts": dict(sorted(phase_counts.items())),
      "play_trick_counts": dict(
          sorted(trick_counts.items(), key=lambda item: int(item[0]))
      ),
      "best_action_kind_counts": dict(sorted(kind_counts.items())),
      "lane_contrast_rows": int(sum(1 for row in stats_rows if row["lane_contrast"])),
      "bottleneck_rows": int(sum(1 for row in stats_rows if row["bottleneck"])),
      "red_switch_contrast_rows": int(
          sum(1 for row in stats_rows if row.get("red_switch_contrast"))
      ),
      "forced_switch_red_contrast_rows": int(
          sum(1 for row in stats_rows if row.get("forced_switch_red_contrast"))
      ),
      "forced_switch_red_win_contrast_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("forced_switch_red_win_contrast")
          )
      ),
      "baseline_follows_led_rows": int(
          sum(1 for row in stats_rows if row.get("baseline_follows_led"))
      ),
      "baseline_token_loss_rows": int(
          sum(1 for row in stats_rows if row.get("baseline_token_loss"))
      ),
      "baseline_red_rows": int(
          sum(1 for row in stats_rows if row.get("baseline_red"))
      ),
      "baseline_red_switch_rows": int(
          sum(1 for row in stats_rows if row.get("baseline_red_switch"))
      ),
      "baseline_lost_led_follow_available_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("baseline_lost_led_follow_available")
          )
      ),
      "baseline_nonred_lost_led_follow_available_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("baseline_nonred_lost_led_follow_available")
          )
      ),
      "best_follows_led_rows": int(
          sum(1 for row in stats_rows if row["best_follows_led"])
      ),
      "best_token_loss_rows": int(
          sum(1 for row in stats_rows if row["best_token_loss"])
      ),
      "best_red_rows": int(
          sum(1 for row in stats_rows if row.get("best_red"))
      ),
      "best_red_switch_rows": int(
          sum(1 for row in stats_rows if row.get("best_red_switch"))
      ),
      "best_red_win_switch_rows": int(
          sum(1 for row in stats_rows if row.get("best_red_win_switch"))
      ),
      "policy_target_top_follows_led_rows": int(
          sum(1 for row in stats_rows if row.get("policy_target_top_follows_led"))
      ),
      "policy_target_top_token_loss_rows": int(
          sum(1 for row in stats_rows if row.get("policy_target_top_token_loss"))
      ),
      "policy_target_top_red_rows": int(
          sum(1 for row in stats_rows if row.get("policy_target_top_red"))
      ),
      "policy_target_top_red_switch_rows": int(
          sum(1 for row in stats_rows if row.get("policy_target_top_red_switch"))
      ),
      "policy_target_top_red_win_switch_rows": int(
          sum(1 for row in stats_rows if row.get("policy_target_top_red_win_switch"))
      ),
      "switch_curriculum_policy_target_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("switch_curriculum_policy_target_used") is True
          )
      ),
      "switch_curriculum_red_top_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("switch_curriculum_red_top_used") is True
          )
      ),
      "switch_curriculum_red_value_delta_avg": (
          float(np.mean([
              float(row["switch_curriculum_red_value_delta"])
              for row in stats_rows
              if row.get("switch_curriculum_red_value_delta") is not None
          ]))
          if any(
              row.get("switch_curriculum_red_value_delta") is not None
              for row in stats_rows
          )
          else 0.0
      ),
      "policy_target_follow_mass_avg": (
          float(np.mean(policy_target_follow_masses))
          if policy_target_follow_masses else 0.0
      ),
      "policy_target_red_switch_mass_avg": (
          float(np.mean(policy_target_red_switch_masses))
          if policy_target_red_switch_masses else 0.0
      ),
      "policy_target_red_win_switch_mass_avg": (
          float(np.mean(policy_target_red_win_switch_masses))
          if policy_target_red_win_switch_masses else 0.0
      ),
      "policy_target_nonred_switch_mass_avg": (
          float(np.mean(policy_target_nonred_switch_masses))
          if policy_target_nonred_switch_masses else 0.0
      ),
      "mean_candidate_count": (
          float(np.mean([row["candidate_count"] for row in stats_rows]))
          if stats_rows else 0.0
      ),
      "mean_structural_spread": (
          float(np.mean([row["structural_spread"] for row in stats_rows]))
          if stats_rows else 0.0
      ),
      "mean_target_spread": (
          float(np.mean([row["target_spread"] for row in stats_rows]))
          if stats_rows else 0.0
      ),
      "mean_risk_spread": (
          float(np.mean([row["risk_spread"] for row in stats_rows]))
          if stats_rows else 0.0
      ),
      "mean_baseline_target": float(np.mean(baseline_values)) if baseline_values else 0.0,
      "mean_best_target": float(np.mean(best_values)) if best_values else 0.0,
      "mean_best_over_baseline": (
          float(np.mean(improvements)) if improvements else 0.0
      ),
      "mean_baseline_risk": float(np.mean(baseline_risks)) if baseline_risks else 0.0,
      "mean_best_risk": float(np.mean(best_risks)) if best_risks else 0.0,
      "mean_risk_best_risk": (
          float(np.mean(risk_best_risks)) if risk_best_risks else 0.0
      ),
      "deterministic_liveness_rows": int(
          sum(1 for row in stats_rows if row.get("deterministic_liveness_used"))
      ),
      "deterministic_best_matches_target_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_best_is_target_best") is True
          )
      ),
      "deterministic_teacher_policy_target_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_teacher_policy_target_used") is True
          )
      ),
      "deterministic_teacher_auxiliary_aligned_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_teacher_auxiliary_aligned") is True
          )
      ),
      "deterministic_teacher_best_matches_value_best_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_teacher_best_is_target_best") is True
          )
      ),
      "deterministic_best_own_feasible_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_best_own_feasible") is True
          )
      ),
      "deterministic_base_own_feasible_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_base_own_feasible") is True
          )
      ),
      "deterministic_best_lost_led_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_best_lost_led_token") is True
          )
      ),
      "deterministic_teacher_best_lost_led_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_teacher_best_lost_led_token") is True
          )
      ),
      "deterministic_base_lost_led_rows": int(
          sum(
              1 for row in stats_rows
              if row.get("deterministic_base_lost_led_token") is True
          )
      ),
      "mean_deterministic_base_risk": (
          float(np.mean(deterministic_base_risks))
          if deterministic_base_risks else 0.0
      ),
      "mean_deterministic_best_risk": (
          float(np.mean(deterministic_best_risks))
          if deterministic_best_risks else 0.0
      ),
      "mean_deterministic_best_over_base_lane_surplus": (
          float(np.mean(deterministic_lane_deltas))
          if deterministic_lane_deltas else 0.0
      ),
      "mean_deterministic_teacher_over_base_table_margin": (
          float(np.mean(deterministic_table_margin_deltas))
          if deterministic_table_margin_deltas else 0.0
      ),
      "mean_deterministic_teacher_dead_players_reduction": (
          float(np.mean(deterministic_table_dead_deltas))
          if deterministic_table_dead_deltas else 0.0
      ),
  }


def main() -> None:
  args = parse_args()
  if not args.bot and not args.checkpoint:
    raise ValueError("Provide --bot or --checkpoint")
  started = time.perf_counter()
  rng = np.random.default_rng(int(args.seed))
  torch.manual_seed(int(args.seed))
  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device=mps requested but unavailable")
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model = None
  model_args = None
  if args.checkpoint:
    model, model_args = load_neural(
        args.checkpoint, int(args.players), device, args=args
    )
    if model is None or model_args is None:
      raise ValueError(f"Could not load checkpoint {args.checkpoint}")
  bots = [
      make_replay_bot(args.bot, seed=int(args.seed) + seat * 17, args=args)
      for seat in range(int(args.players))
  ] if args.bot else []

  all_rows = []
  stats_rows = []
  round_summaries = []
  match_totals = np.zeros(int(args.players), dtype=np.float32)
  for match_idx in range(int(args.matches)):
    for round_idx in range(int(args.players)):
      start_player = (
          int(args.initial_start) + int(match_idx) + int(round_idx)
      ) % int(args.players)
      game = make_game(args, start_player)
      state = game.new_initial_state()
      if bool(args.match_context):
        state.set_match_context(match_totals, round_idx)
      rows, row_stats, terminal_state = collect_round(
          args,
          game,
          state,
          model,
          model_args,
          device,
          bots,
          rng,
          match_idx=int(match_idx),
          round_idx=int(round_idx),
      )
      all_rows.extend(rows)
      stats_rows.extend(row_stats)
      paradox_target = terminal_paradox_target(terminal_state, int(args.players))
      round_summaries.append({
          "match_idx": int(match_idx),
          "round_idx": int(round_idx),
          "start_player": int(start_player),
          "rows": int(len(rows)),
          "any_paradox": bool(np.any(paradox_target > 0.5)),
          "paradoxed": [bool(value) for value in paradox_target],
      })
      match_totals += round_raw_scores(terminal_state)
    if (
        int(args.max_rows) > 0
        and len(all_rows) >= int(args.max_rows)
    ):
      break
    if (
        int(args.progress_interval) > 0
        and (match_idx + 1) % int(args.progress_interval) == 0
    ):
      print(json.dumps({
          "completed_matches": int(match_idx + 1),
          "rows": int(len(all_rows)),
          "elapsed_sec": round(time.perf_counter() - started, 3),
      }), flush=True)

  if int(args.max_rows) > 0 and len(all_rows) > int(args.max_rows):
    all_rows = all_rows[:int(args.max_rows)]
    stats_rows = stats_rows[:int(args.max_rows)]
  if not all_rows:
    raise SystemExit("No rows selected; relax --row-filter or phase/spread knobs")

  total_rounds = len(round_summaries)
  paradox_rounds = sum(1 for row in round_summaries if row["any_paradox"])
  selected_rounds = sum(1 for row in round_summaries if int(row["rows"]) > 0)
  selected_paradox_rounds = sum(
      1 for row in round_summaries
      if int(row["rows"]) > 0 and bool(row["any_paradox"])
  )
  row_group_ids = row_group_ids_from_stats(
      stats_rows, source_seed=int(args.seed)
  )
  row_group_error = ""
  if len(row_group_ids) != len(all_rows):
    row_group_error = (
        f"group id count {len(row_group_ids)} did not match row count "
        f"{len(all_rows)}"
    )
    row_group_ids = []
  if row_group_ids:
    for stats, row_group_id in zip(stats_rows, row_group_ids):
      stats["row_group_id"] = str(row_group_id)
      stats["source_group_id"] = str(row_group_id)
  metadata = {
      "kind": "counterfactual_action_liquidity_survival",
      "players": int(args.players),
      "matches": int(args.matches),
      "bot": str(args.bot),
      "liveness_teacher_base_bot": str(args.liveness_teacher_base_bot),
      "liveness_teacher_phases": str(args.liveness_teacher_phases),
      "liveness_teacher_min_trick_number": int(
          args.liveness_teacher_min_trick_number
      ),
      "liveness_teacher_exact_legal_pressure": bool(
          args.liveness_teacher_exact_legal_pressure
      ),
      "liveness_teacher_exact_key_order": str(
          args.liveness_teacher_exact_key_order
      ),
      "checkpoint": str(args.checkpoint),
      "checkpoint_mode": str(args.checkpoint_mode),
      "action_value_selection_weight": float(args.action_value_selection_weight),
      "action_paradox_selection_penalty": float(
          args.action_paradox_selection_penalty
      ),
      "action_paradox_rerank_mode": str(args.action_paradox_rerank_mode),
      "action_paradox_risk_threshold": float(args.action_paradox_risk_threshold),
      "action_paradox_min_risk_margin": float(
          args.action_paradox_min_risk_margin
      ),
      "action_paradox_max_policy_log_gap": float(
          args.action_paradox_max_policy_log_gap
      ),
      "action_feasibility_selection_weight": float(
          args.action_feasibility_selection_weight
      ),
      "action_value_rerank_clip": float(args.action_value_rerank_clip),
      "action_value_rerank_phases": str(args.action_value_rerank_phases),
      "action_value_rerank_min_margin": float(
          args.action_value_rerank_min_margin
      ),
      "liveness_shield_base_mode": str(args.liveness_shield_base_mode),
      "liveness_shield_phases": str(args.liveness_shield_phases),
      "liveness_shield_min_trick_number": int(
          args.liveness_shield_min_trick_number
      ),
      "liveness_shield_led_colors": str(args.liveness_shield_led_colors),
      "liveness_shield_min_open_slot_delta": int(
          args.liveness_shield_min_open_slot_delta
      ),
      "liveness_shield_min_public_damage_delta": int(
          args.liveness_shield_min_public_damage_delta
      ),
      "liveness_shield_max_policy_log_gap": float(
          args.liveness_shield_max_policy_log_gap
      ),
      "liveness_shield_exact_legal_pressure": bool(
          args.liveness_shield_exact_legal_pressure
      ),
      "liveness_shield_exact_forced_only": bool(
          args.liveness_shield_exact_forced_only
      ),
      "liveness_shield_shadow_only": bool(args.liveness_shield_shadow_only),
      "liveness_shield_label_policy": str(args.liveness_shield_label_policy),
      "liveness_shield_sample_limit": int(args.liveness_shield_sample_limit),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "source_temperature": float(args.source_temperature),
      "continuation_temperature": float(args.continuation_temperature),
      "rollouts_per_action": int(args.rollouts_per_action),
      "paired_rollout_seeds": bool(args.paired_rollout_seeds),
      "emit_action_risk_labels": bool(args.emit_action_risk_labels),
      "max_actions_per_state": int(args.max_actions_per_state),
      "top_policy_actions": int(args.top_policy_actions),
      "feature_candidates": bool(args.feature_candidates),
      "candidate_action_filter": str(args.candidate_action_filter),
      "rollout_horizon_plies": int(args.rollout_horizon_plies),
      "rollout_weight": float(args.rollout_weight),
      "deterministic_liveness_weight": float(args.deterministic_liveness_weight),
      "deterministic_liveness_scale": float(args.deterministic_liveness_scale),
      "deterministic_liveness_risk_weight": float(
          args.deterministic_liveness_risk_weight
      ),
      "deterministic_exact_legal_pressure_weight": float(
          args.deterministic_exact_legal_pressure_weight
      ),
      "deterministic_exact_legal_pressure_scale": float(
          args.deterministic_exact_legal_pressure_scale
      ),
      "deterministic_exact_legal_pressure_risk_weight": float(
          args.deterministic_exact_legal_pressure_risk_weight
      ),
      "deterministic_exact_table_liveness_weight": float(
          args.deterministic_exact_table_liveness_weight
      ),
      "deterministic_exact_table_liveness_scale": float(
          args.deterministic_exact_table_liveness_scale
      ),
      "deterministic_exact_table_liveness_risk_weight": float(
          args.deterministic_exact_table_liveness_risk_weight
      ),
      "deterministic_teacher_policy_target": bool(
          args.deterministic_teacher_policy_target
      ),
      "deterministic_teacher_key_order": str(args.deterministic_teacher_key_order),
      "deterministic_teacher_auxiliary_mode": str(
          args.deterministic_teacher_auxiliary_mode
      ),
      "switch_curriculum_policy_target": bool(
          args.switch_curriculum_policy_target
      ),
      "switch_curriculum_follow_mass": float(
          args.switch_curriculum_follow_mass
      ),
      "switch_curriculum_red_mass": float(args.switch_curriculum_red_mass),
      "switch_curriculum_other_mass": float(args.switch_curriculum_other_mass),
      "switch_curriculum_red_win_bonus": float(
          args.switch_curriculum_red_win_bonus
      ),
      "switch_curriculum_red_bonus": float(args.switch_curriculum_red_bonus),
      "switch_curriculum_nonred_penalty": float(
          args.switch_curriculum_nonred_penalty
      ),
      "phases": str(args.phases),
      "min_trick_number": int(args.min_trick_number),
      "max_trick_number": int(args.max_trick_number),
      "row_filter": str(args.row_filter),
      "source_round_filter": str(args.source_round_filter),
      "behavior_policy_target": bool(args.behavior_policy_target),
      "behavior_action_filter": str(args.behavior_action_filter),
      "target": (
          "actual source trajectory action"
          if bool(args.behavior_policy_target) else
          "per-action blend of immediate liquidity, continuation no-paradox "
          "survival, and optional deterministic feasibility/liveness"
      ),
      "row_group_ids": row_group_ids,
      "row_group_id_kind": "source_match_round",
      "row_group_count": int(len(set(row_group_ids))) if row_group_ids else 0,
  }
  if row_group_error:
    metadata["row_group_error"] = row_group_error
  save_replay(all_rows, args.out, metadata=metadata)
  metadata_path = Path(str(args.metadata_out)) if args.metadata_out else None
  if metadata_path is not None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        **metadata,
        "replay": str(args.out),
        "rows": stats_rows,
    }, indent=2, sort_keys=True) + "\n")
  summary = {
      **metadata,
      "out": str(args.out),
      "rows": int(len(all_rows)),
      "rounds": int(total_rounds),
      "rounds_with_any_paradox": int(paradox_rounds),
      "same_policy_paradox_round_rate": (
          round(float(paradox_rounds) / float(total_rounds), 4)
          if total_rounds else None
      ),
      "selected_source_rounds": int(selected_rounds),
      "selected_source_rounds_with_any_paradox": int(selected_paradox_rounds),
      "selected_source_paradox_round_rate": (
          round(float(selected_paradox_rounds) / float(selected_rounds), 4)
          if selected_rounds else None
      ),
      **summarize_stats(stats_rows),
      "round_samples": round_summaries[:5],
      "elapsed_sec": round(time.perf_counter() - started, 3),
  }
  if metadata_path is not None:
    summary["metadata_out"] = str(metadata_path)
  if bool(args.include_row_stats):
    summary["row_stats"] = stats_rows
  summary_path = (
      Path(args.summary_out)
      if args.summary_out else
      Path(args.out).with_suffix(Path(args.out).suffix + ".summary.json")
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
