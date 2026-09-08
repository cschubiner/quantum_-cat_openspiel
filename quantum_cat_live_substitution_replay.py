#!/usr/bin/env python3
"""Generate dense labels from live same-policy decision states.

Unlike the failure miners, this samples ordinary on-policy decisions from all
rounds, then evaluates the selected action and legal substitutions under the
same continuation policy. The resulting replay asks the training code to learn
which replacement actions actually improve survival from the live decision
distribution, not only from late failure traces.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from quantum_cat_calibration_transfer_audit import pair_feature_vector_from_matrix
from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    action_feature_matrix,
    model_policy_value,
    save_replay,
)
from quantum_cat_failure_miner import (
    action_string,
    choose_policy_action,
    dense_action_value_target,
    dense_label_row_for_decision,
    evaluate_decision,
    parse_phase_filter,
    play_round_with_history,
    rollout_from_action,
    should_evaluate_decision,
    terminal_round_result,
)
from quantum_cat_full_match_elo import (
  AZActionSetTransformerBot,
  AZProposalVetoTop1Bot,
  action_liveness_certificate,
  load_action_set_transformer,
  load_neural,
  load_pairwise_boundary,
  load_proposal_veto_selector,
)
from quantum_cat_table_liveness_improve_trainer import _led_lane_burn_label_shape

COLOR_NAMES = ["R", "B", "Y", "G"]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=20)
  parser.add_argument("--seed", type=int, default=20288801)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--max-decisions", type=int, default=80)
  parser.add_argument(
      "--max-accepted-rows",
      type=int,
      default=0,
      help=(
          "Optional cap on rows that survive all filters and are written to the "
          "replay. This is separate from --max-decisions, which counts evaluated "
          "candidate decisions before late proposal-outcome filters."
      ),
  )
  parser.add_argument("--max-decisions-per-round", type=int, default=2)
  parser.add_argument("--min-legal-count", type=int, default=2)
  parser.add_argument("--max-legal-count", type=int, default=0)
  parser.add_argument("--min-trick-number", type=int, default=0)
  parser.add_argument("--max-trick-number", type=int, default=99)
  parser.add_argument(
      "--led-contrast-only",
      action="store_true",
      help=(
          "Only evaluate play decisions with a non-red led color where following "
          "that led color is legal and at least one legal switch is available."
      ),
  )
  parser.add_argument(
      "--require-red-switch-legal",
      action="store_true",
      help=(
          "With --led-contrast-only, require at least one legal red switch. This "
          "focuses labels on the follow-led vs trump-red decision."
      ),
  )
  parser.add_argument(
      "--max-led-bucket-rows",
      type=int,
      default=0,
      help=(
          "Optional cap per selected-action led-contrast bucket. Buckets are "
          "selected_follow_led, selected_switch_red, selected_switch_nonred, "
          "and selected_other."
      ),
  )
  parser.add_argument(
      "--require-action-paradox-contrast",
      action="store_true",
      help=(
          "After evaluating substitutions, keep only rows where labeled legal "
          "actions include both any-paradox and no-paradox outcomes."
      ),
  )
  parser.add_argument(
      "--require-policy-target-changed",
      action="store_true",
      help="After labeling, keep only rows where the dense target changes top action.",
  )
  parser.add_argument(
      "--repeat-validate-labels",
      action="store_true",
      help=(
          "Keep only rows whose dense label-best action repeatedly improves "
          "the label-base action under common-seed continuation rollouts."
      ),
  )
  parser.add_argument(
      "--repeat-validation-candidate-mode",
      choices=(
          "label_best",
          "follow_red_value",
          "proposal_scored_candidates",
          "proposal_scored_plus_red_follow_value",
      ),
      default="label_best",
      help=(
          "Candidate source for repeat validation. label_best preserves the "
          "historical behavior. follow_red_value also tests led-follow, "
          "red-switch, and top dense-value actions. proposal_scored_candidates "
          "tests the live proposal slate itself. "
          "proposal_scored_plus_red_follow_value starts with the proposal slate "
          "and appends red-switch/follow/value candidates so the proposal "
          "verifier can learn red-rescue alternatives. Non-label_best modes "
          "relabel to the best repeat-validated candidate."
      ),
  )
  parser.add_argument(
      "--repeat-validation-max-candidates",
      type=int,
      default=6,
      help=(
          "Maximum candidate actions tested when repeat-validation-candidate-mode "
          "is not label_best. The dense label-best action is always considered "
          "first when legal."
      ),
  )
  parser.add_argument("--repeat-validation-rollouts", type=int, default=3)
  parser.add_argument("--repeat-validation-min-win-rate", type=float, default=1.0)
  parser.add_argument(
      "--repeat-validation-min-nonworse-rate",
      type=float,
      default=1.0,
  )
  parser.add_argument("--repeat-validation-min-value-delta", type=float, default=0.01)
  parser.add_argument("--repeat-validation-max-risk-delta", type=float, default=0.0)
  parser.add_argument(
      "--repeat-validation-max-candidate-any-paradox-rate",
      type=float,
      default=1.0,
  )
  parser.add_argument(
      "--repeat-validation-min-base-any-paradox-rate",
      type=float,
      default=0.0,
  )
  parser.add_argument(
      "--repeat-validation-require-passed-red-switch",
      action="store_true",
      help=(
          "After repeat-validating a candidate queue, keep only rows where at "
          "least one passed candidate is an off-led red switch. This builds "
          "red-rescue contrast banks instead of relying on incidental red "
          "coverage."
      ),
  )
  parser.add_argument(
      "--repeat-validation-require-passed-follow-led",
      action="store_true",
      help=(
          "After repeat-validating a candidate queue, keep only rows where at "
          "least one passed candidate follows the led color."
      ),
  )
  parser.add_argument(
      "--repeat-validation-prefilter-candidates",
      action="store_true",
      help=(
          "Before running expensive all-legal dense substitution labels, "
          "repeat-validate a small direct candidate queue and skip states where "
          "no candidate passes. Accepted rows still use normal dense rows plus "
          "repeat-validation action targets."
      ),
  )
  parser.add_argument(
      "--record-repeat-validation-attempts",
      action="store_true",
      help=(
          "Write diagnostic metadata for repeat-validation candidate queues, "
          "including rejected queues skipped before replay row creation. This "
          "does not change replay rows or training targets."
      ),
  )
  parser.add_argument(
      "--record-round-history",
      action="store_true",
      help=(
          "Attach JSON-safe prior-decision led-lane history to accepted rows. "
          "This is intended for best-of-N/trajectory distillation diagnostics."
      ),
  )
  parser.add_argument(
      "--round-history-max-events",
      type=int,
      default=40,
      help=(
          "Maximum number of prior decisions saved per accepted row when "
          "--record-round-history is enabled. The most recent events are kept."
      ),
  )
  parser.add_argument(
      "--min-round-history-avoidable-switch-events",
      type=int,
      default=0,
      help=(
          "After dense labeling, keep only rows whose prior round history has at "
          "least this many avoidable off-led switches."
      ),
  )
  parser.add_argument(
      "--min-round-history-avoidable-nonred-switch-events",
      type=int,
      default=0,
      help=(
          "After dense labeling, keep only rows whose prior round history has at "
          "least this many avoidable non-red off-led switches."
      ),
  )
  parser.add_argument(
      "--min-round-history-avoidable-token-loss-events",
      type=int,
      default=0,
      help=(
          "After dense labeling, keep only rows whose prior round history has at "
          "least this many avoidable off-led token-loss events."
      ),
  )
  parser.add_argument(
      "--partial-save-interval-decisions",
      type=int,
      default=0,
      help=(
          "If positive, periodically write .partial metadata and replay files "
          "after this many evaluated decisions. Intended for long collection "
          "runs so interrupts preserve accepted rows and attempt diagnostics."
      ),
  )
  parser.add_argument(
      "--repeat-validation-exclusive-action-targets",
      action="store_true",
      help=(
          "When writing repeat-validation action targets, clear pre-existing "
          "per-action risk/value masks first so selector trainers only see the "
          "repeat-validated base and candidate group. This avoids mixing "
          "common-seed labels with older one-rollout dense labels."
      ),
  )
  parser.add_argument(
      "--max-repeat-validation-label-color-rows",
      type=int,
      default=0,
      help=(
          "Optional cap per accepted repeat-validation target color. Useful "
          "for collecting balanced banks; once a color is full, later rows "
          "whose selected target has that color are skipped."
      ),
  )
  parser.add_argument("--min-score-delta", type=float, default=-99.0)
  parser.add_argument("--dense-target-temperature", type=float, default=0.05)
  parser.add_argument(
      "--dense-action-value-objective",
      choices=("safety", "safety_raw", "score"),
      default="safety",
  )
  parser.add_argument("--dense-safety-value-safe-target", type=float, default=0.9)
  parser.add_argument(
      "--dense-safety-value-paradox-target", type=float, default=-0.55
  )
  parser.add_argument("--dense-safety-value-forced-penalty", type=float, default=0.25)
  parser.add_argument("--dense-safety-value-actor-penalty", type=float, default=0.10)
  parser.add_argument("--dense-safety-value-score-weight", type=float, default=0.05)
  parser.add_argument("--dense-safety-value-min", type=float, default=-0.95)
  parser.add_argument("--dense-safety-value-max", type=float, default=0.95)
  parser.add_argument(
      "--bot-mode",
      choices=("policy", "q_policy", "liveness_policy", "q_liveness_policy"),
      default="q_policy",
  )
  parser.add_argument("--liveness-shield-phases", default="play")
  parser.add_argument("--liveness-shield-min-open-slot-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-min-public-damage-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument(
      "--action-paradox-rerank-mode",
      choices=("additive", "threshold", "relative"),
      default="additive",
  )
  parser.add_argument("--action-paradox-risk-threshold", type=float, default=0.0)
  parser.add_argument("--action-paradox-min-risk-margin", type=float, default=0.0)
  parser.add_argument("--action-feasibility-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-max-policy-log-gap", type=float, default=2.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument("--action-value-rerank-phases", default="")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  parser.add_argument(
      "--proposal-mode",
      choices=(
          "none",
          "pairwise_boundary",
          "action_set_transformer",
          "proposal_veto_top1",
      ),
      default="none",
      help=(
          "Optionally shadow-score a live proposal before dense labeling. "
          "Rows still use the frozen base action as policy_best_action."
      ),
  )
  parser.add_argument(
      "--label-base-action-source",
      choices=("behavior", "frozen_policy"),
      default="behavior",
      help=(
          "Which action is treated as policy_best_action when no proposal base "
          "overrides it. 'behavior' preserves the actually played collection "
          "action. 'frozen_policy' recomputes the configured bot policy with "
          "behavior exploration disabled, useful for exploratory state "
          "collection whose labels must still be deployment-relative."
      ),
  )
  parser.add_argument(
      "--require-proposal",
      action="store_true",
      help="Keep only rows where the configured proposal mode would override.",
  )
  parser.add_argument(
      "--require-proposal-outcome",
      choices=(
          "none",
          "positive",
          "positive_safe",
          "safe",
          "unsafe",
          "candidate_any_paradox",
          "candidate_worse",
          "scored_has_positive_safe",
          "scored_mixed_positive_safe",
          "scored_has_positive_safe_no_candidate_any_paradox",
          "scored_mixed_positive_safe_no_candidate_any_paradox",
          "scored_has_candidate_any_paradox",
          "scored_candidate_any_with_positive_safe_no_candidate_any_alternative",
          "scored_same_led_candidate_any_with_same_led_positive_safe_no_candidate_any_alternative",
          "scored_source10_dense_candidate_any_sparse_approved",
          "scored_bad_red_lane_burn_no_win_with_safe_alternative",
          "scored_bad_red_lane_burn_no_win_with_safe_any_alternative",
      ),
      default="none",
      help=(
          "After dense proposal labeling, keep only proposal rows in this "
          "verified outcome bucket. Useful for balanced proposal-verifier banks."
      ),
  )
  parser.add_argument(
      "--source10-density-min-candidate-any-rate",
      type=float,
      default=0.70,
      help=(
          "Minimum scored-candidate any-paradox rate for the source-10-density "
          "proposal outcome bucket."
      ),
  )
  parser.add_argument(
      "--source10-density-max-approved-rate",
      type=float,
      default=0.15,
      help=(
          "Maximum positive-safe/no-any scored-candidate rate for the "
          "source-10-density proposal outcome bucket."
      ),
  )
  parser.add_argument(
      "--source10-density-min-scored-candidates",
      type=int,
      default=8,
      help=(
          "Minimum scored-candidate count for the source-10-density proposal "
          "outcome bucket."
      ),
  )
  parser.add_argument(
      "--prefilter-proposal-tactical-shape",
      choices=("none", "bad_red_no_win_and_lane_preserver"),
      default="none",
      help=(
          "Default-off cheap prefilter before dense continuation labeling. "
          "The final --require-proposal-outcome check still decides admission; "
          "this only skips expensive states whose proposal slate lacks the "
          "requested tactical shape."
      ),
  )
  parser.add_argument("--pairwise-boundary-model", default="")
  parser.add_argument(
      "--pairwise-boundary-base-mode",
      choices=("policy", "q_policy", "liveness_policy", "q_liveness_policy"),
      default="policy",
  )
  parser.add_argument("--pairwise-boundary-threshold", type=float, default=0.0)
  parser.add_argument("--pairwise-boundary-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument(
      "--pairwise-boundary-exclude-forced-led-lane-negatives",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--pairwise-boundary-led-lane-red-exception",
      choices=("none", "red", "win", "urgent_win"),
      default="win",
  )
  parser.add_argument("--action-set-transformer-model", default="")
  parser.add_argument(
      "--action-set-transformer-base-mode",
      choices=("policy", "q_policy", "liveness_shield"),
      default="policy",
  )
  parser.add_argument("--action-set-transformer-phases", default="play")
  parser.add_argument("--action-set-transformer-min-legal-count", type=int, default=0)
  parser.add_argument("--action-set-transformer-max-legal-count", type=int, default=0)
  parser.add_argument("--action-set-transformer-min-trick-number", type=int, default=-1)
  parser.add_argument("--action-set-transformer-max-trick-number", type=int, default=-1)
  parser.add_argument(
      "--action-set-transformer-score-margin-threshold",
      type=float,
      default=0.0,
  )
  parser.add_argument("--action-set-transformer-value-threshold", type=float, default=0.0)
  parser.add_argument(
      "--action-set-transformer-positive-threshold",
      type=float,
      default=0.5,
  )
  parser.add_argument(
      "--action-set-transformer-max-policy-log-gap",
      type=float,
      default=-1.0,
  )
  parser.add_argument("--action-set-transformer-max-actions", type=int, default=4)
  parser.add_argument("--action-set-transformer-top-policy", type=int, default=2)
  parser.add_argument(
      "--action-set-transformer-include-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument(
      "--action-set-transformer-feature-candidates",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--action-set-transformer-exclude-forced-led-lane-negatives",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--action-set-transformer-selection-value-bonus",
      type=float,
      default=0.05,
  )
  parser.add_argument(
      "--action-set-transformer-policy-tiebreak",
      type=float,
      default=0.01,
  )
  parser.add_argument("--proposal-veto-model", default="")
  parser.add_argument(
      "--proposal-veto-base-mode",
      choices=("policy", "q_policy", "liveness_shield"),
      default="policy",
  )
  parser.add_argument("--proposal-veto-phases", default="play")
  parser.add_argument("--proposal-veto-min-legal-count", type=int, default=0)
  parser.add_argument("--proposal-veto-max-legal-count", type=int, default=0)
  parser.add_argument("--proposal-veto-min-trick-number", type=int, default=-1)
  parser.add_argument("--proposal-veto-max-trick-number", type=int, default=-1)
  parser.add_argument("--proposal-veto-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument("--proposal-veto-max-actions", type=int, default=6)
  parser.add_argument("--proposal-veto-top-policy", type=int, default=4)
  parser.add_argument(
      "--proposal-veto-include-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument(
      "--proposal-veto-candidate-mode",
      default="source_plus_red_follow_policy",
  )
  parser.add_argument("--proposal-veto-selection-mode", default="proposal_model")
  parser.add_argument(
      "--proposal-veto-feature-candidates",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--proposal-veto-exclude-forced-led-lane-negatives",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--proposal-veto-score-threshold", type=float, default=-1e9)
  parser.add_argument("--proposal-veto-margin-threshold", type=float, default=-1e9)
  parser.add_argument("--proposal-veto-min-source-score", type=float, default=-1e9)
  parser.add_argument("--proposal-veto-min-source-margin", type=float, default=-1e9)
  parser.add_argument(
      "--proposal-veto-min-source-value-delta",
      type=float,
      default=-1e9,
  )
  parser.add_argument(
      "--proposal-veto-min-source-positive-prob",
      type=float,
      default=-1e9,
  )
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument(
      "--enable-behavior-exploration",
      action="store_true",
      help=(
          "Use controlled exploration while collecting live decision states. "
          "Counterfactual labels still disable this exploration."
      ),
  )
  parser.add_argument(
      "--behavior-exploration-prob",
      type=float,
      default=0.0,
      help="Epsilon probability for sampling a legal behavior action.",
  )
  parser.add_argument(
      "--behavior-exploration-temperature",
      type=float,
      default=1.0,
      help="Temperature used when sampling exploratory behavior actions.",
  )
  parser.add_argument(
      "--behavior-led-contrast-switch-prob",
      type=float,
      default=0.0,
      help=(
          "In non-red-led play states with legal follow and switch options, "
          "sample a switch action with this probability."
      ),
  )
  parser.add_argument(
      "--behavior-red-switch-prob",
      type=float,
      default=0.75,
      help=(
          "When behavior-led-contrast-switch-prob fires and red is legal, "
          "prefer sampling from red switches with this probability."
      ),
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  return parser.parse_args()


def _action_color(state, action):
  action = int(action)
  if action == 999:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  if num_card_types <= 0:
    return None
  color_idx = action // num_card_types
  if 0 <= color_idx < len(COLOR_NAMES):
    return COLOR_NAMES[color_idx]
  return None


def _legal_color_count(state, legal, color):
  return sum(1 for action in legal if _action_color(state, action) == color)


def _action_feature_flag(features, action, feature_index):
  action = int(action)
  feature_index = int(feature_index)
  if action < 0 or action >= int(features.shape[0]):
    return False
  if feature_index < 0 or feature_index >= int(features.shape[1]):
    return False
  return bool(float(features[action, feature_index]) > 0.5)


def _round_history_event(decision):
  state = decision["state"]
  player = int(decision["player"])
  action = int(decision["action"])
  legal = [int(item) for item in decision["legal"]]
  phase = str(decision["phase"])
  led_color = getattr(state, "_led_color", None)
  selected_color = _action_color(state, action)
  follows_led = False
  is_red = bool(selected_color == "R")
  off_led_loses_token = False
  feature_shape = [0, 0]
  if phase == "play" and action != 999:
    features = action_feature_matrix(state, player, int(state.num_distinct_actions()))
    feature_shape = [int(features.shape[0]), int(features.shape[1])]
    follows_led = _action_feature_flag(
        features, action, ACTION_FEATURE_FOLLOWS_LED_INDEX
    )
    is_red = _action_feature_flag(features, action, ACTION_FEATURE_IS_RED_INDEX)
    off_led_loses_token = _action_feature_flag(
        features, action, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX
    )

  switches_off_led = bool(
      phase == "play"
      and selected_color not in (None, led_color)
      and led_color is not None
  )
  led_legal_count = int(_legal_color_count(state, legal, led_color))
  red_legal_count = int(_legal_color_count(state, legal, "R"))
  switch_legal_count = int(
      sum(
          1
          for legal_action in legal
          if legal_action != 999
          and _action_color(state, legal_action) not in (None, led_color)
      )
  )
  avoidable_switch = bool(switches_off_led and led_legal_count > 0)
  return {
      "decision_index": int(decision["index"]),
      "player": int(player),
      "phase": phase,
      "trick_number": int(decision.get("trick_number", -1)),
      "legal_count": int(len(legal)),
      "action": int(action),
      "action_string": str(decision["action_string"]),
      "selected_color": selected_color,
      "led_color": led_color,
      "follows_led": bool(follows_led),
      "is_red": bool(is_red),
      "switches_off_led": bool(switches_off_led),
      "avoidable_switch": bool(avoidable_switch),
      "avoidable_red_switch": bool(avoidable_switch and is_red),
      "avoidable_nonred_switch": bool(
          avoidable_switch and selected_color not in (None, "R")
      ),
      "off_led_loses_token": bool(off_led_loses_token),
      "avoidable_token_loss": bool(
          avoidable_switch and off_led_loses_token
      ),
      "led_legal_count": int(led_legal_count),
      "red_legal_count": int(red_legal_count),
      "switch_legal_count": int(switch_legal_count),
      "policy_prob": float(decision.get("policy_prob", 0.0)),
      "new_paradox_players": [
          int(player_idx)
          for player_idx in decision.get("new_paradox_players", [])
      ],
      "action_feature_shape": feature_shape,
  }


def _round_history_before(decisions, current_decision, max_events):
  current_index = int(current_decision["index"])
  events = [
      _round_history_event(decision)
      for decision in decisions
      if int(decision["index"]) < current_index
  ]
  max_events = int(max_events)
  if max_events > 0 and len(events) > max_events:
    return events[-max_events:]
  return events


def _round_history_summary(events, current_player):
  play_events = [event for event in events if event.get("phase") == "play"]
  current_player = int(current_player)
  own_play = [
      event for event in play_events
      if int(event.get("player", -1)) == current_player
  ]
  other_play = [
      event for event in play_events
      if int(event.get("player", -1)) != current_player
  ]

  def frac(count, total):
    return float(count / total) if total else 0.0

  follow_count = sum(int(bool(event.get("follows_led", False))) for event in play_events)
  switch_count = sum(
      int(bool(event.get("switches_off_led", False))) for event in play_events
  )
  avoidable_switch_count = sum(
      int(bool(event.get("avoidable_switch", False))) for event in play_events
  )
  avoidable_red_switch_count = sum(
      int(bool(event.get("avoidable_red_switch", False))) for event in play_events
  )
  avoidable_nonred_switch_count = sum(
      int(bool(event.get("avoidable_nonred_switch", False))) for event in play_events
  )
  red_switch_count = sum(
      int(
          bool(event.get("switches_off_led", False))
          and bool(event.get("is_red", False))
      )
      for event in play_events
  )
  token_loss_count = sum(
      int(bool(event.get("off_led_loses_token", False))) for event in play_events
  )
  avoidable_token_loss_count = sum(
      int(bool(event.get("avoidable_token_loss", False))) for event in play_events
  )
  own_token_loss_count = sum(
      int(bool(event.get("off_led_loses_token", False))) for event in own_play
  )
  other_token_loss_count = sum(
      int(bool(event.get("off_led_loses_token", False))) for event in other_play
  )
  return {
      "prior_events": int(len(events)),
      "prior_play_events": int(len(play_events)),
      "prior_follow_led_events": int(follow_count),
      "prior_switch_events": int(switch_count),
      "prior_avoidable_switch_events": int(avoidable_switch_count),
      "prior_avoidable_red_switch_events": int(avoidable_red_switch_count),
      "prior_avoidable_nonred_switch_events": int(avoidable_nonred_switch_count),
      "prior_red_switch_events": int(red_switch_count),
      "prior_token_loss_events": int(token_loss_count),
      "prior_avoidable_token_loss_events": int(avoidable_token_loss_count),
      "own_prior_play_events": int(len(own_play)),
      "own_prior_token_loss_events": int(own_token_loss_count),
      "other_prior_play_events": int(len(other_play)),
      "other_prior_token_loss_events": int(other_token_loss_count),
      "prior_follow_led_rate": frac(follow_count, len(play_events)),
      "prior_switch_rate": frac(switch_count, len(play_events)),
      "prior_avoidable_switch_rate": frac(
          avoidable_switch_count, len(play_events)
      ),
      "prior_red_when_avoidable_switch_rate": frac(
          avoidable_red_switch_count, avoidable_switch_count
      ),
      "prior_nonred_when_avoidable_switch_rate": frac(
          avoidable_nonred_switch_count, avoidable_switch_count
      ),
      "prior_red_when_switch_rate": frac(red_switch_count, switch_count),
      "prior_token_loss_rate": frac(token_loss_count, len(play_events)),
      "prior_avoidable_token_loss_rate": frac(
          avoidable_token_loss_count, len(play_events)
      ),
      "own_prior_token_loss_rate": frac(own_token_loss_count, len(own_play)),
      "other_prior_token_loss_rate": frac(other_token_loss_count, len(other_play)),
      "last_event": events[-1] if events else None,
  }


def _round_history_filter_thresholds(args):
  return {
      "avoidable_switch_events": int(
          getattr(args, "min_round_history_avoidable_switch_events", 0) or 0
      ),
      "avoidable_nonred_switch_events": int(
          getattr(args, "min_round_history_avoidable_nonred_switch_events", 0) or 0
      ),
      "avoidable_token_loss_events": int(
          getattr(args, "min_round_history_avoidable_token_loss_events", 0) or 0
      ),
  }


def _requires_round_history_filter(args):
  return any(value > 0 for value in _round_history_filter_thresholds(args).values())


def _wants_round_history(args):
  return bool(getattr(args, "record_round_history", False)) or _requires_round_history_filter(args)


def _round_history_filter_failures(history_summary, args):
  thresholds = _round_history_filter_thresholds(args)
  checks = (
      (
          "avoidable_switch_events",
          "prior_avoidable_switch_events",
      ),
      (
          "avoidable_nonred_switch_events",
          "prior_avoidable_nonred_switch_events",
      ),
      (
          "avoidable_token_loss_events",
          "prior_avoidable_token_loss_events",
      ),
  )
  failures = []
  for threshold_key, summary_key in checks:
    required = int(thresholds[threshold_key])
    actual = int(history_summary.get(summary_key, 0))
    if required > 0 and actual < required:
      failures.append(
          {
              "threshold": threshold_key,
              "summary_key": summary_key,
              "required": int(required),
              "actual": int(actual),
          }
      )
  return failures


def _record_round_history_summary_counters(summary, history, history_summary):
  summary["round_history_rows"] += 1
  summary["round_history_events"] += int(len(history))
  summary["round_history_prior_play_events"] += int(
      history_summary["prior_play_events"]
  )
  summary["round_history_prior_token_loss_events"] += int(
      history_summary["prior_token_loss_events"]
  )
  summary["round_history_prior_avoidable_switch_events"] += int(
      history_summary["prior_avoidable_switch_events"]
  )
  summary["round_history_prior_avoidable_nonred_switch_events"] += int(
      history_summary["prior_avoidable_nonred_switch_events"]
  )
  summary["round_history_prior_avoidable_token_loss_events"] += int(
      history_summary["prior_avoidable_token_loss_events"]
  )


def _led_contrast_profile(decision):
  state = decision["state"]
  legal = [int(action) for action in decision["legal"]]
  selected = int(decision["action"])
  led_color = getattr(state, "_led_color", None)
  selected_color = _action_color(state, selected)
  led_legal = _legal_color_count(state, legal, led_color)
  red_legal = _legal_color_count(state, legal, "R")
  switch_legal = sum(
      1
      for action in legal
      if action != 999 and _action_color(state, action) not in (None, led_color)
  )
  if selected_color == led_color:
    selected_bucket = "selected_follow_led"
  elif selected_color == "R":
    selected_bucket = "selected_switch_red"
  elif selected_color is None:
    selected_bucket = "selected_other"
  else:
    selected_bucket = "selected_switch_nonred"
  return {
      "led_color": led_color,
      "selected_color": selected_color,
      "selected_action": int(selected),
      "led_legal_count": int(led_legal),
      "red_legal_count": int(red_legal),
      "switch_legal_count": int(switch_legal),
      "selected_bucket": selected_bucket,
      "avoidable_switch": bool(
          selected_color not in (None, led_color) and int(led_legal) > 0
      ),
  }


def _annotate_led_contrast_meta(dense_meta, decision, profile):
  led_color = profile.get("led_color")
  label_action = int(dense_meta.get("label_best_action", -1))
  label_color = _action_color(decision["state"], label_action)
  dense_meta["led_contrast"] = {
      **profile,
      "label_best_color": label_color,
      "label_best_follows_led": bool(label_color == led_color),
      "label_best_is_red_switch": bool(label_color == "R" and led_color != "R"),
      "label_best_switches_off_led": bool(
          label_color not in (None, led_color)
      ),
  }
  for outcome in dense_meta.get("action_outcomes", []):
    action = int(outcome.get("action", -1))
    color = _action_color(decision["state"], action)
    outcome["color"] = color
    outcome["follows_led"] = bool(color == led_color)
    outcome["is_red_switch"] = bool(color == "R" and led_color != "R")
    outcome["switches_off_led"] = bool(color not in (None, led_color))


def _proposal_eval_args(args, bot_mode):
  eval_args = copy.copy(args)
  eval_args.bot_mode = str(bot_mode)
  eval_args.enable_behavior_exploration = False
  eval_args.behavior_exploration_prob = 0.0
  eval_args.behavior_led_contrast_switch_prob = 0.0
  return eval_args


def _replace_decision_action(decision, action):
  action = int(action)
  replacement = dict(decision)
  replacement["action"] = int(action)
  replacement["action_string"] = action_string(
      decision["state"], int(decision["player"]), action
  )
  policy = np.asarray(decision["policy"], dtype=np.float32)
  replacement["policy_prob"] = (
      float(policy[action]) if 0 <= action < len(policy) else 0.0
  )
  return replacement


def _frozen_policy_base_action(model, decision, value_scale, device, args):
  base_args = _proposal_eval_args(args, str(getattr(args, "bot_mode", "policy")))
  action, _policy = choose_policy_action(
      model,
      decision["state"],
      int(decision["player"]),
      value_scale,
      device,
      base_args,
  )
  if int(action) not in set(int(item) for item in decision["legal"]):
    return int(decision["action"])
  return int(action)


def _repeat_record(result, player, acting_score_delta=0.0, total_score_delta=0.0):
  return {
      "any_paradox": bool(result["any_paradox"]),
      "actor_paradox": bool(result["paradoxed"][int(player)]),
      "forced_paradox": bool(result.get("forced_paradox", False)),
      "acting_score_delta": float(acting_score_delta),
      "total_score_delta": float(total_score_delta),
      "raw_scores": [float(value) for value in result["raw_scores"]],
  }


def _repeat_validate_label_best(
    model,
    decision,
    value_scale,
    device,
    rollout_args,
    label_action,
    seed_base,
):
  base_action = int(decision["action"])
  label_action = int(label_action)
  player = int(decision["player"])
  rollouts = max(1, int(getattr(rollout_args, "repeat_validation_rollouts", 3)))
  if label_action == base_action:
    return {
        "passed": False,
        "reason": "label_matches_base",
        "base_action": int(base_action),
        "candidate_action": int(label_action),
        "rollouts": int(rollouts),
    }

  wins = 0
  nonworse = 0
  candidate_any = 0
  base_any = 0
  value_deltas = []
  risk_deltas = []
  acting_score_deltas = []
  total_score_deltas = []
  seeds = []
  min_value_delta = float(
      getattr(rollout_args, "repeat_validation_min_value_delta", 0.01)
  )
  max_risk_delta = float(
      getattr(rollout_args, "repeat_validation_max_risk_delta", 0.0)
  )

  for idx in range(rollouts):
    seed = int(seed_base) + 5000003 + idx * 7919
    seeds.append(int(seed))
    base_result = rollout_from_action(
        model,
        decision["state"],
        base_action,
        value_scale,
        device,
        seed,
        rollout_args,
    )
    candidate_result = rollout_from_action(
        model,
        decision["state"],
        label_action,
        value_scale,
        device,
        seed,
        rollout_args,
    )
    acting_delta = (
        float(candidate_result["raw_scores"][player])
        - float(base_result["raw_scores"][player])
    )
    total_delta = (
        float(sum(candidate_result["raw_scores"]))
        - float(sum(base_result["raw_scores"]))
    )
    base_record = _repeat_record(base_result, player)
    candidate_record = _repeat_record(
        candidate_result,
        player,
        acting_score_delta=acting_delta,
        total_score_delta=total_delta,
    )
    base_value = dense_action_value_target(base_record, rollout_args, value_scale)
    candidate_value = dense_action_value_target(
        candidate_record, rollout_args, value_scale
    )
    value_delta = float(candidate_value - base_value)
    risk_delta = float(bool(candidate_result["any_paradox"])) - float(
        bool(base_result["any_paradox"])
    )
    value_deltas.append(value_delta)
    risk_deltas.append(risk_delta)
    acting_score_deltas.append(float(acting_delta))
    total_score_deltas.append(float(total_delta))
    candidate_any += int(bool(candidate_result["any_paradox"]))
    base_any += int(bool(base_result["any_paradox"]))
    if value_delta >= min_value_delta and risk_delta <= max_risk_delta:
      wins += 1
    if value_delta >= 0.0 and risk_delta <= max_risk_delta:
      nonworse += 1

  win_rate = float(wins / rollouts)
  nonworse_rate = float(nonworse / rollouts)
  base_any_rate = float(base_any / rollouts)
  candidate_any_rate = float(candidate_any / rollouts)
  passed = (
      win_rate >= float(getattr(rollout_args, "repeat_validation_min_win_rate", 1.0))
      and nonworse_rate >= float(
          getattr(rollout_args, "repeat_validation_min_nonworse_rate", 1.0)
      )
      and candidate_any_rate <= float(
          getattr(
              rollout_args,
              "repeat_validation_max_candidate_any_paradox_rate",
              1.0,
          )
      )
      and base_any_rate >= float(
          getattr(rollout_args, "repeat_validation_min_base_any_paradox_rate", 0.0)
      )
  )
  return {
      "passed": bool(passed),
      "reason": "passed" if passed else "failed_threshold",
      "base_action": int(base_action),
      "base_action_string": action_string(decision["state"], player, base_action),
      "candidate_action": int(label_action),
      "candidate_action_string": action_string(
          decision["state"], player, label_action
      ),
      "rollouts": int(rollouts),
      "wins": int(wins),
      "nonworse": int(nonworse),
      "win_rate": float(win_rate),
      "nonworse_rate": float(nonworse_rate),
      "base_any_paradox_rate": float(base_any_rate),
      "candidate_any_paradox_rate": float(candidate_any_rate),
      "mean_value_delta": float(np.mean(value_deltas)),
      "min_value_delta": float(np.min(value_deltas)),
      "max_value_delta": float(np.max(value_deltas)),
      "mean_risk_delta": float(np.mean(risk_deltas)),
      "max_risk_delta": float(np.max(risk_deltas)),
      "mean_acting_score_delta": float(np.mean(acting_score_deltas)),
      "mean_total_score_delta": float(np.mean(total_score_deltas)),
      "seeds": seeds,
      "required_min_win_rate": float(
          getattr(rollout_args, "repeat_validation_min_win_rate", 1.0)
      ),
      "required_min_nonworse_rate": float(
          getattr(rollout_args, "repeat_validation_min_nonworse_rate", 1.0)
      ),
      "required_min_value_delta": float(min_value_delta),
      "required_max_risk_delta": float(max_risk_delta),
      "required_max_candidate_any_paradox_rate": float(
          getattr(
              rollout_args,
              "repeat_validation_max_candidate_any_paradox_rate",
              1.0,
          )
      ),
      "required_min_base_any_paradox_rate": float(
          getattr(rollout_args, "repeat_validation_min_base_any_paradox_rate", 0.0)
      ),
  }


def _outcome_value(outcome):
  return float(outcome.get("dense_action_value_target", -1e9))


def _outcomes_by_action(dense_meta):
  return {
      int(outcome.get("action", -1)): outcome
      for outcome in dense_meta.get("action_outcomes", [])
  }


def _repeat_validation_candidates(dense_meta, decision, profile, args):
  base_action = int(decision["action"])
  label_best = int(dense_meta.get("label_best_action", -1))
  outcomes = [
      outcome for outcome in dense_meta.get("action_outcomes", [])
      if int(outcome.get("action", -1)) != base_action
      and int(outcome.get("action", -1)) != 999
  ]
  by_action = {int(outcome.get("action", -1)): outcome for outcome in outcomes}
  candidates = []

  def add(action):
    action = int(action)
    if action == base_action or action == 999 or action not in by_action:
      return
    if action not in candidates:
      candidates.append(action)

  mode = str(getattr(args, "repeat_validation_candidate_mode", "label_best"))
  if mode in ("proposal_scored_candidates", "proposal_scored_plus_red_follow_value"):
    proposal = dense_meta.get("proposal")
    scored = proposal.get("scored_candidates") if isinstance(proposal, dict) else None
    if isinstance(scored, list):
      for item in scored:
        if not isinstance(item, dict):
          continue
        add(int(item.get("action", -1)))
        if len(candidates) >= int(getattr(args, "repeat_validation_max_candidates", 6)):
          return candidates
    if mode == "proposal_scored_candidates":
      return candidates
    led_color = profile.get("led_color")
    red_switches = [
        outcome for outcome in outcomes
        if outcome.get("color") == "R" and led_color != "R"
    ]
    follows_led = [outcome for outcome in outcomes if outcome.get("color") == led_color]
    top_value = sorted(outcomes, key=_outcome_value, reverse=True)
    for bucket in (red_switches, follows_led, top_value):
      for outcome in sorted(bucket, key=_outcome_value, reverse=True):
        add(int(outcome.get("action", -1)))
        if len(candidates) >= int(getattr(args, "repeat_validation_max_candidates", 6)):
          return candidates
    return candidates

  add(label_best)
  if mode == "label_best":
    return candidates

  led_color = profile.get("led_color")
  red_switches = [
      outcome for outcome in outcomes
      if outcome.get("color") == "R" and led_color != "R"
  ]
  follows_led = [outcome for outcome in outcomes if outcome.get("color") == led_color]
  top_value = sorted(outcomes, key=_outcome_value, reverse=True)
  for bucket in (red_switches, follows_led, top_value):
    for outcome in sorted(bucket, key=_outcome_value, reverse=True):
      add(int(outcome.get("action", -1)))
      if len(candidates) >= int(getattr(args, "repeat_validation_max_candidates", 6)):
        return candidates
  return candidates


def _policy_probability(decision, action):
  policy = np.asarray(decision.get("policy", []), dtype=np.float32)
  action = int(action)
  if 0 <= action < len(policy):
    return float(policy[action])
  return 0.0


def _direct_candidate_record(decision, profile, action):
  action = int(action)
  color = _action_color(decision["state"], action)
  led_color = profile.get("led_color")
  return {
      "action": int(action),
      "candidate_action": int(action),
      "candidate_color": color,
      "candidate_follows_led": bool(color == led_color),
      "candidate_is_red_switch": bool(color == "R" and led_color != "R"),
      "candidate_switches_off_led": bool(color not in (None, led_color)),
      "candidate_dense_action_value_target": _policy_probability(decision, action),
      "policy_prob": _policy_probability(decision, action),
  }


def _direct_repeat_validation_candidates(decision, profile, args):
  base_action = int(decision["action"])
  legal = [
      int(action) for action in decision["legal"]
      if int(action) not in (base_action, 999)
  ]
  led_color = profile.get("led_color")
  candidates = []

  def add(action):
    action = int(action)
    if action == base_action or action == 999 or action not in legal:
      return
    if action not in candidates:
      candidates.append(action)

  def policy_sorted(actions):
    return sorted(
        (int(action) for action in actions),
        key=lambda item: _policy_probability(decision, item),
        reverse=True,
    )

  red_switches = [
      action for action in legal
      if _action_color(decision["state"], action) == "R" and led_color != "R"
  ]
  follows_led = [
      action for action in legal
      if _action_color(decision["state"], action) == led_color
  ]
  for bucket in (policy_sorted(red_switches), policy_sorted(follows_led),
                 policy_sorted(legal)):
    for action in bucket:
      add(action)
      if len(candidates) >= int(getattr(args, "repeat_validation_max_candidates", 6)):
        return [
            _direct_candidate_record(decision, profile, item)
            for item in candidates
        ]
  return [
      _direct_candidate_record(decision, profile, item)
      for item in candidates
  ]


def _relabel_dense_row(dense_row, dense_meta, decision, action):
  action = int(action)
  row_parts = list(dense_row)
  target = np.zeros_like(np.asarray(row_parts[2], dtype=np.float32))
  if 0 <= action < len(target):
    target[action] = 1.0
  row_parts[2] = target
  row_parts[5] = int(action)
  dense_meta["label_best_action"] = int(action)
  dense_meta["label_best_action_string"] = action_string(
      decision["state"], int(decision["player"]), action
  )
  for outcome in dense_meta.get("action_outcomes", []):
    outcome_action = int(outcome.get("action", -1))
    outcome["dense_policy_target"] = 1.0 if outcome_action == action else 0.0
    if outcome_action == action:
      dense_meta["label_best_safety_tuple"] = [
          int(bool(outcome.get("any_paradox", False))),
          int(bool(outcome.get("forced_paradox", False))),
          int(bool(outcome.get("actor_paradox", False))),
      ]
  dense_meta["policy_target_changed"] = bool(action != int(decision["action"]))
  return tuple(row_parts), dense_meta


def _write_repeat_validation_action_targets(
    dense_row,
    dense_meta,
    repeat_validation,
    *,
    exclusive=False,
):
  row_parts = list(dense_row)
  num_actions = len(row_parts[1])
  action_paradox_targets = np.asarray(row_parts[7], dtype=np.float32).copy()
  action_paradox_mask = np.asarray(row_parts[8], dtype=np.float32).copy()
  action_value_targets = np.asarray(row_parts[9], dtype=np.float32).copy()
  action_value_mask = np.asarray(row_parts[10], dtype=np.float32).copy()
  if bool(exclusive):
    action_paradox_targets = np.zeros_like(action_paradox_targets, dtype=np.float32)
    action_paradox_mask = np.zeros_like(action_paradox_mask, dtype=np.float32)
    action_value_targets = np.zeros_like(action_value_targets, dtype=np.float32)
    action_value_mask = np.zeros_like(action_value_mask, dtype=np.float32)

  base_action = int(repeat_validation.get("base_action", -1))
  if 0 <= base_action < num_actions:
    action_paradox_targets[base_action] = float(
        repeat_validation.get("base_any_paradox_rate", 1.0)
    )
    action_paradox_mask[base_action] = 1.0
    action_value_targets[base_action] = 0.0
    action_value_mask[base_action] = 1.0

  candidates = list(repeat_validation.get("candidates", []) or [])
  if not candidates and int(repeat_validation.get("candidate_action", -1)) >= 0:
    candidates = [repeat_validation]
  by_action = {}
  for item in candidates:
    action = int(item.get("candidate_action", -1))
    if not (0 <= action < num_actions):
      continue
    by_action[action] = dict(item)
    action_paradox_targets[action] = float(
        item.get("candidate_any_paradox_rate", 1.0)
    )
    action_paradox_mask[action] = 1.0
    action_value_targets[action] = float(item.get("mean_value_delta", 0.0))
    action_value_mask[action] = 1.0

  for outcome in dense_meta.get("action_outcomes", []):
    action = int(outcome.get("action", -1))
    if action == base_action:
      outcome["repeat_validation_base_any_paradox_rate"] = float(
          repeat_validation.get("base_any_paradox_rate", 1.0)
      )
      outcome["repeat_validation_value_delta_vs_base"] = 0.0
      outcome["dense_action_value_target"] = 0.0
      continue
    item = by_action.get(action)
    if item is None:
      continue
    outcome["repeat_validation_candidate_any_paradox_rate"] = float(
        item.get("candidate_any_paradox_rate", 1.0)
    )
    outcome["repeat_validation_base_any_paradox_rate"] = float(
        item.get("base_any_paradox_rate", repeat_validation.get("base_any_paradox_rate", 1.0))
    )
    outcome["repeat_validation_mean_value_delta"] = float(
        item.get("mean_value_delta", 0.0)
    )
    outcome["repeat_validation_min_value_delta"] = float(
        item.get("min_value_delta", 0.0)
    )
    outcome["repeat_validation_passed"] = bool(item.get("passed", False))
    outcome["dense_action_value_target"] = float(
        item.get("mean_value_delta", 0.0)
    )

  row_parts[7] = action_paradox_targets
  row_parts[8] = action_paradox_mask
  row_parts[9] = action_value_targets
  row_parts[10] = action_value_mask
  dense_meta["repeat_validation_action_targets_written"] = True
  dense_meta["repeat_validation_exclusive_action_targets"] = bool(exclusive)
  dense_meta["repeat_validation_labeled_action_count"] = int(len(by_action))
  return tuple(row_parts), dense_meta


def _append_repeat_validation_candidates_to_proposal(dense_meta, repeat_validation):
  proposal = dense_meta.get("proposal")
  if not isinstance(proposal, dict):
    return
  scored = proposal.get("scored_candidates")
  if not isinstance(scored, list):
    scored = []
    proposal["scored_candidates"] = scored
  existing = {
      int(item.get("action", -1))
      for item in scored
      if isinstance(item, dict)
  }
  outcomes = _outcomes_by_action(dense_meta)
  base_score = float(proposal.get("base_score", 0.0))
  added = 0
  for item in repeat_validation.get("candidates", []) or []:
    action = int(item.get("candidate_action", -1))
    if action < 0 or action in existing:
      continue
    outcome = outcomes.get(action, {})
    dense_value = float(outcome.get("dense_action_value_target", 0.0))
    candidate = {
        "action": int(action),
        "action_string": str(
            item.get("candidate_action_string")
            or outcome.get("action_string")
            or ""
        ),
        "score": float(dense_value),
        "base_score": float(base_score),
        "margin": float(dense_value - base_score),
        "pred_value_delta": float(dense_value),
        "pred_positive_prob": float(1.0 if bool(item.get("passed", False)) else 0.0),
        "policy_prob": float(item.get("candidate_policy_prob", 0.0)),
        "repeat_validation_augmented_candidate": True,
    }
    scored.append(candidate)
    existing.add(action)
    added += 1
  proposal["repeat_validation_augmented_scored_candidates"] = int(
      proposal.get("repeat_validation_augmented_scored_candidates", 0)
  ) + int(added)
  proposal["scored_candidate_count"] = int(len(scored))
  proposal["candidate_count"] = max(
      int(proposal.get("candidate_count", 0)),
      int(len(scored)),
  )


def _repeat_validate_candidate_queue(
    model,
    decision,
    value_scale,
    device,
    rollout_args,
    dense_meta,
    profile,
    seed_base,
):
  candidates = _repeat_validation_candidates(dense_meta, decision, profile, rollout_args)
  by_action = _outcomes_by_action(dense_meta)
  validations = []
  for candidate in candidates:
    result = _repeat_validate_label_best(
        model,
        decision,
        value_scale,
        device,
        rollout_args,
        int(candidate),
        seed_base,
    )
    outcome = by_action.get(int(candidate), {})
    result["candidate_color"] = outcome.get("color")
    result["candidate_follows_led"] = bool(outcome.get("follows_led", False))
    result["candidate_is_red_switch"] = bool(outcome.get("is_red_switch", False))
    result["candidate_switches_off_led"] = bool(
        outcome.get("switches_off_led", False)
    )
    result["candidate_dense_action_value_target"] = float(
        outcome.get("dense_action_value_target", -1e9)
    )
    validations.append(dict(result))
  passed = [item for item in validations if bool(item.get("passed", False))]
  if not passed:
    return {
        "passed": False,
        "reason": "no_candidate_passed" if validations else "no_candidates",
        "candidate_count": int(len(candidates)),
        "candidates": validations,
    }
  best = max(
      passed,
      key=lambda item: (
          -float(item.get("candidate_any_paradox_rate", 1.0)),
          float(item.get("win_rate", 0.0)),
          float(item.get("nonworse_rate", 0.0)),
          float(item.get("mean_value_delta", -1e9)),
          -float(item.get("mean_risk_delta", 1e9)),
          float(item.get("mean_acting_score_delta", -1e9)),
      ),
  )
  return {
      **best,
      "candidate_count": int(len(candidates)),
      "candidates": validations,
      "selected_by_candidate_queue": True,
  }


def _repeat_validate_direct_candidate_queue(
    model,
    decision,
    value_scale,
    device,
    rollout_args,
    profile,
    seed_base,
):
  candidates = _direct_repeat_validation_candidates(decision, profile, rollout_args)
  validations = []
  for candidate in candidates:
    result = _repeat_validate_label_best(
        model,
        decision,
        value_scale,
        device,
        rollout_args,
        int(candidate["candidate_action"]),
        seed_base,
    )
    result["candidate_color"] = candidate.get("candidate_color")
    result["candidate_follows_led"] = bool(
        candidate.get("candidate_follows_led", False)
    )
    result["candidate_is_red_switch"] = bool(
        candidate.get("candidate_is_red_switch", False)
    )
    result["candidate_switches_off_led"] = bool(
        candidate.get("candidate_switches_off_led", False)
    )
    result["candidate_dense_action_value_target"] = float(
        candidate.get("candidate_dense_action_value_target", -1e9)
    )
    result["candidate_policy_prob"] = float(candidate.get("policy_prob", 0.0))
    validations.append(dict(result))
  passed = [item for item in validations if bool(item.get("passed", False))]
  if not passed:
    return {
        "passed": False,
        "reason": "no_candidate_passed" if validations else "no_candidates",
        "candidate_count": int(len(candidates)),
        "candidates": validations,
        "direct_candidate_prefilter": True,
    }
  best = max(
      passed,
      key=lambda item: (
          -float(item.get("candidate_any_paradox_rate", 1.0)),
          float(item.get("win_rate", 0.0)),
          float(item.get("nonworse_rate", 0.0)),
          float(item.get("mean_value_delta", -1e9)),
          -float(item.get("mean_risk_delta", 1e9)),
          float(item.get("mean_acting_score_delta", -1e9)),
          float(item.get("candidate_policy_prob", 0.0)),
      ),
  )
  return {
      **best,
      "candidate_count": int(len(candidates)),
      "candidates": validations,
      "selected_by_candidate_queue": True,
      "direct_candidate_prefilter": True,
  }


def _repeat_validation_attempt_meta(
    label_decision,
    led_profile,
    repeat_validation,
    *,
    match_idx,
    round_idx,
    round_seed,
    admission,
):
  return {
      "admission": str(admission),
      "phase": str(label_decision["phase"]),
      "player": int(label_decision["player"]),
      "trick_number": int(label_decision.get("trick_number", -1)),
      "legal_count": int(len(label_decision["legal"])),
      "match_index": int(match_idx),
      "round_index": int(round_idx),
      "decision_index": int(label_decision["index"]),
      "round_seed": int(round_seed),
      "base_action": int(label_decision["action"]),
      "base_action_string": str(label_decision["action_string"]),
      "led_contrast": dict(led_profile),
      "repeat_validation": dict(repeat_validation),
  }


def _pairwise_base_action(model, decision, value_scale, device, args):
  base_mode = str(getattr(args, "pairwise_boundary_base_mode", "policy"))
  if base_mode == str(getattr(args, "bot_mode", "policy")):
    return int(decision["action"]), np.asarray(decision["policy"], dtype=np.float32)
  state = decision["state"]
  player = int(decision["player"])
  base_args = _proposal_eval_args(args, base_mode)
  action, policy = choose_policy_action(
      model, state, player, value_scale, device, base_args
  )
  return int(action), np.asarray(policy, dtype=np.float32)


def _pairwise_feature(
    boundary,
    action_features,
    legal_mask,
    observation,
    policy,
    action,
    base_action,
    legal_count,
):
  saved_args = boundary.get("args", {})
  feature = pair_feature_vector_from_matrix(
      action_features,
      legal_mask,
      int(action),
      int(base_action),
      target_action=int(action),
      label_margin=0.0,
      legal_count=int(legal_count),
      candidate_count=int(legal_count),
      include_label_margin=False,
      observation=observation,
      observation_feature_mode=str(
          saved_args.get("observation_feature_mode", "summary") or "summary"
      ),
      policy_context=policy,
      policy_context_mode=str(
          saved_args.get("policy_context_mode", "summary") or "summary"
      ),
      include_target_match_features=False,
  )
  feature_size = int(boundary["feature_size"])
  if int(feature.shape[0]) != feature_size:
    raise ValueError(
        "pairwise proposal feature size mismatch: "
        f"{feature.shape[0]} vs {feature_size}"
    )
  return np.asarray(feature, dtype=np.float32)


def pairwise_boundary_proposal(model, boundary, decision, value_scale, device, args):
  if boundary is None:
    return None
  state = decision["state"]
  player = int(decision["player"])
  legal = [int(action) for action in decision["legal"]]
  if len(legal) <= 1:
    return None
  base_action, policy = _pairwise_base_action(
      model, decision, value_scale, device, args
  )
  if int(base_action) not in set(legal):
    base_action = int(decision["action"])
    policy = np.asarray(decision["policy"], dtype=np.float32)
  candidates = [int(action) for action in legal if int(action) != int(base_action)]
  num_actions = int(state.num_distinct_actions())
  action_features = action_feature_matrix(state, player, num_actions)
  filtered_forced = []
  if bool(getattr(args, "pairwise_boundary_exclude_forced_led_lane_negatives", True)):
    filter_args = SimpleNamespace(
        force_led_lane_burn_negative=True,
        led_lane_red_exception=str(
            getattr(args, "pairwise_boundary_led_lane_red_exception", "win")
        ),
    )
    kept = []
    for action in candidates:
      forced, _kept_by_exception = _led_lane_burn_label_shape(
          action_features, int(base_action), int(action), filter_args
      )
      if forced:
        filtered_forced.append(int(action))
      else:
        kept.append(int(action))
    candidates = kept
  filtered_policy_gap = []
  max_gap = float(getattr(args, "pairwise_boundary_max_policy_log_gap", -1.0))
  if max_gap >= 0.0:
    floor = 1e-12
    base_logp = math.log(max(float(policy[int(base_action)]), floor))
    kept = []
    for action in candidates:
      candidate_logp = math.log(max(float(policy[int(action)]), floor))
      if base_logp - candidate_logp <= max_gap:
        kept.append(int(action))
      else:
        filtered_policy_gap.append(int(action))
    candidates = kept
  if not candidates:
    return None
  pool = sorted({int(base_action), *[int(action) for action in candidates]})
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[pool] = 1.0
  observation = np.asarray(state.observation_tensor(player), dtype=np.float32)
  base_feature = _pairwise_feature(
      boundary,
      action_features,
      legal_mask,
      observation,
      policy,
      int(base_action),
      int(base_action),
      len(pool),
  )
  rows = [
      _pairwise_feature(
          boundary,
          action_features,
          legal_mask,
          observation,
          policy,
          int(action),
          int(base_action),
          len(pool),
      )
      for action in candidates
  ]
  x = np.asarray(rows, dtype=np.float32)
  mean_arr = np.asarray(boundary["feature_mean"], dtype=np.float32)
  std_arr = np.asarray(boundary["feature_std"], dtype=np.float32)
  base_std = (base_feature - mean_arr) / std_arr
  x_std = (x - mean_arr) / std_arr
  with torch.no_grad():
    base_t = torch.tensor(base_std, dtype=torch.float32, device=device)
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    base_batch = base_t.unsqueeze(0).expand(x_t.shape[0], -1)
    scores = boundary["model"](x_t, base_batch).detach().cpu().numpy()
    base_score = float(
        boundary["model"](
            base_t.unsqueeze(0), base_t.unsqueeze(0)
        ).detach().cpu().numpy()[0]
    )
  scored = []
  for action, score in zip(candidates, scores):
    scored.append({
        "action": int(action),
        "score": float(score),
        "base_score": float(base_score),
        "margin": float(score - base_score),
        "policy_prob": float(policy[int(action)]),
    })
  scored.sort(
      key=lambda item: (float(item["margin"]), float(item["policy_prob"])),
      reverse=True,
  )
  best = dict(scored[0])
  threshold = float(getattr(args, "pairwise_boundary_threshold", 0.0))
  if float(best["margin"]) < threshold:
    return {
        "mode": "pairwise_boundary",
        "above_threshold": False,
        "threshold": float(threshold),
        "base_mode": str(getattr(args, "pairwise_boundary_base_mode", "policy")),
        "base_action": int(base_action),
        "base_action_string": action_string(state, player, base_action),
        "best_action": int(best["action"]),
        "best_action_string": action_string(state, player, best["action"]),
        "best_score": float(best["score"]),
        "base_score": float(best["base_score"]),
        "best_margin": float(best["margin"]),
        "candidate_count": int(len(candidates)),
        "filtered_forced_led_lane": [int(action) for action in filtered_forced],
        "filtered_policy_gap": [int(action) for action in filtered_policy_gap],
    }
  return {
      "mode": "pairwise_boundary",
      "above_threshold": True,
      "threshold": float(threshold),
      "base_mode": str(getattr(args, "pairwise_boundary_base_mode", "policy")),
      "base_action": int(base_action),
      "base_action_string": action_string(state, player, base_action),
      "candidate_action": int(best["action"]),
      "candidate_action_string": action_string(state, player, best["action"]),
      "candidate_score": float(best["score"]),
      "base_score": float(best["base_score"]),
      "score_margin": float(best["margin"]),
      "base_policy_prob": float(policy[int(base_action)]),
      "candidate_policy_prob": float(policy[int(best["action"])]),
      "candidate_count": int(len(candidates)),
      "filtered_forced_led_lane": [int(action) for action in filtered_forced],
      "filtered_policy_gap": [int(action) for action in filtered_policy_gap],
  }


def _action_set_proposal_args(args, value_scale):
  proposal_args = copy.copy(args)
  proposal_args.value_scale = float(value_scale)
  proposal_args.action_set_transformer_shadow_only = False
  return proposal_args


def _proposal_veto_args(args, value_scale):
  proposal_args = copy.copy(args)
  proposal_args.value_scale = float(value_scale)
  proposal_args.proposal_veto_shadow_only = False
  return proposal_args


def action_set_transformer_proposal(bot, decision, summary=None):
  if bot is None:
    return None
  state = decision["state"]
  player = int(decision["player"])
  legal = [int(action) for action in decision["legal"]]
  legal_set = set(legal)
  if len(legal) <= 1:
    return None
  policy, _ = model_policy_value(
      bot.model,
      state,
      player,
      state.num_distinct_actions(),
      bot.args.value_scale,
      bot.device,
  )
  raw_policy_action = int(max(legal, key=lambda action: float(policy[int(action)])))
  base_action = bot._base_action(state, player, legal, raw_policy_action)
  if int(base_action) not in legal_set:
    base_action = int(decision["action"])
  enabled, disabled_reason = bot._enabled_for_state(state, len(legal))
  if not enabled:
    return {
        "mode": "action_set_transformer",
        "above_threshold": False,
        "disabled_reason": str(disabled_reason),
        "base_mode": str(bot.base_mode),
        "base_action": int(base_action),
        "base_action_string": action_string(state, player, base_action),
    }
  if summary is not None and not _decision_prefilter_tactical_shape(
      decision,
      bot.args,
      summary,
      base_action=int(base_action),
      counter_prefix="preproposal_base_tactical",
  ):
    return {
        "mode": "action_set_transformer",
        "above_threshold": False,
        "disabled_reason": "preproposal_base_tactical_shape",
        "base_mode": str(bot.base_mode),
        "base_action": int(base_action),
        "base_action_string": action_string(state, player, base_action),
    }
  action_features = action_feature_matrix(
      state, player, state.num_distinct_actions()
  )
  candidates = bot._candidate_actions(
      state,
      player,
      legal,
      policy,
      int(base_action),
      action_features,
  )
  scores = bot._score_actions(
      state,
      player,
      legal,
      policy,
      int(base_action),
      candidates,
      action_features,
  )
  base_score = scores.get(int(base_action)) if scores else None
  if not scores or base_score is None:
    return {
        "mode": "action_set_transformer",
        "above_threshold": False,
        "disabled_reason": "no_scores",
        "base_mode": str(bot.base_mode),
        "base_action": int(base_action),
        "base_action_string": action_string(state, player, base_action),
        "candidate_count": int(len(candidates)),
    }
  score_margin_threshold = float(
      getattr(bot.args, "action_set_transformer_score_margin_threshold", 0.0)
  )
  value_threshold = float(
      getattr(bot.args, "action_set_transformer_value_threshold", 0.0)
  )
  positive_threshold = float(
      getattr(bot.args, "action_set_transformer_positive_threshold", 0.5)
  )
  value_bonus = float(
      getattr(bot.args, "action_set_transformer_selection_value_bonus", 0.05)
  )
  policy_tiebreak = float(
      getattr(bot.args, "action_set_transformer_policy_tiebreak", 0.01)
  )
  scored_candidates = []
  best_action = None
  best_score = None
  best_rank = None
  for action, score in scores.items():
    if int(action) == int(base_action):
      continue
    candidate = {
        "action": int(action),
        "action_string": action_string(state, player, int(action)),
        "score": float(score["score"]),
        "base_score": float(score["base_score"]),
        "margin": float(score["margin"]),
        "pred_value_delta": float(score["pred_value_delta"]),
        "pred_positive_prob": float(score["pred_positive_prob"]),
        "policy_prob": float(policy[int(action)]),
    }
    scored_candidates.append(candidate)
    if float(score["margin"]) < score_margin_threshold:
      continue
    if float(score["pred_value_delta"]) < value_threshold:
      continue
    if float(score["pred_positive_prob"]) < positive_threshold:
      continue
    rank = (
        float(score["margin"])
        + value_bonus * float(score["pred_value_delta"])
        + policy_tiebreak * float(policy[int(action)])
    )
    if best_action is None or rank > float(best_rank):
      best_action = int(action)
      best_score = score
      best_rank = float(rank)
  scored_candidates.sort(
      key=lambda item: (
          float(item["margin"])
          + value_bonus * float(item["pred_value_delta"])
          + policy_tiebreak * float(item["policy_prob"])
      ),
      reverse=True,
  )
  if best_action is None:
    best = scored_candidates[0] if scored_candidates else None
    return {
        "mode": "action_set_transformer",
        "above_threshold": False,
        "disabled_reason": "below_threshold",
        "threshold": float(score_margin_threshold),
        "value_threshold": float(value_threshold),
        "positive_threshold": float(positive_threshold),
        "base_mode": str(bot.base_mode),
        "base_action": int(base_action),
        "base_action_string": action_string(state, player, base_action),
        "best_action": int(best["action"]) if best else -1,
        "best_action_string": (
            action_string(state, player, int(best["action"])) if best else ""
        ),
        "best_score": float(best["score"]) if best else 0.0,
        "base_score": float(base_score["score"]),
        "best_margin": float(best["margin"]) if best else 0.0,
        "best_pred_value_delta": (
            float(best["pred_value_delta"]) if best else 0.0
        ),
        "best_pred_positive_prob": (
            float(best["pred_positive_prob"]) if best else 0.0
        ),
        "candidate_count": int(len(candidates)),
        "scored_candidate_count": int(max(0, len(scores) - 1)),
        "scored_candidates": scored_candidates,
    }
  return {
      "mode": "action_set_transformer",
      "above_threshold": True,
      "threshold": float(score_margin_threshold),
      "value_threshold": float(value_threshold),
      "positive_threshold": float(positive_threshold),
      "base_mode": str(bot.base_mode),
      "base_action": int(base_action),
      "base_action_string": action_string(state, player, base_action),
      "candidate_action": int(best_action),
      "candidate_action_string": action_string(state, player, best_action),
      "candidate_score": float(best_score["score"]),
      "base_score": float(base_score["score"]),
      "score_margin": float(best_score["margin"]),
      "candidate_pred_value_delta": float(best_score["pred_value_delta"]),
      "candidate_pred_positive_prob": float(best_score["pred_positive_prob"]),
      "base_policy_prob": float(policy[int(base_action)]),
      "candidate_policy_prob": float(policy[int(best_action)]),
      "candidate_count": int(len(candidates)),
      "scored_candidate_count": int(max(0, len(scores) - 1)),
      "scored_candidates": scored_candidates,
  }


def proposal_veto_top1_proposal(bot, decision):
  if bot is None:
    return None
  state = decision["state"]
  player = int(decision["player"])
  legal = [int(action) for action in decision["legal"]]
  legal_set = set(legal)
  if len(legal) <= 1:
    return None
  policy, _ = model_policy_value(
      bot.model,
      state,
      player,
      state.num_distinct_actions(),
      bot.args.value_scale,
      bot.device,
  )
  raw_policy_action = int(max(legal, key=lambda action: float(policy[int(action)])))
  base_action = bot._base_action(state, player, legal, raw_policy_action)
  if int(base_action) not in legal_set:
    base_action = int(decision["action"])
  enabled, disabled_reason = bot._enabled_for_state(state, len(legal))
  if not enabled:
    return {
        "mode": "proposal_veto_top1",
        "above_threshold": False,
        "disabled_reason": str(disabled_reason),
        "base_mode": str(bot.base_mode),
        "base_action": int(base_action),
        "base_action_string": action_string(state, player, base_action),
    }
  action_features = action_feature_matrix(
      state, player, state.num_distinct_actions()
  )
  candidates = bot._candidate_actions(
      state,
      player,
      legal,
      policy,
      int(base_action),
      action_features,
  )
  source_scores = bot._score_source_actions(
      state,
      player,
      legal,
      policy,
      int(base_action),
      candidates,
      action_features,
  )
  if not source_scores:
    return {
        "mode": "proposal_veto_top1",
        "above_threshold": False,
        "disabled_reason": "no_source_scores",
        "base_mode": str(bot.base_mode),
        "base_action": int(base_action),
        "base_action_string": action_string(state, player, base_action),
        "candidate_count": int(len(candidates)),
    }
  scored_items = [
      dict(source_scores[int(action)])
      for action in candidates
      if int(action) in source_scores and int(action) != int(base_action)
  ]
  scored_items.sort(
      key=lambda item: (
          float(item.get("score", 0.0)),
          float(item.get("pred_value_delta", 0.0)),
          float(item.get("policy_prob", 0.0)),
      ),
      reverse=True,
  )
  if not scored_items:
    return {
        "mode": "proposal_veto_top1",
        "above_threshold": False,
        "disabled_reason": "no_rows",
        "base_mode": str(bot.base_mode),
        "base_action": int(base_action),
        "base_action_string": action_string(state, player, base_action),
        "candidate_count": int(len(candidates)),
    }
  selection_mode = str(
      getattr(bot.args, "proposal_veto_selection_mode", "proposal_model")
      or "proposal_model"
  ).lower()
  if selection_mode in ("source", "source_score", "source_score_top1"):
    views = [bot._source_score_view(item) for item in scored_items]
    logits = np.asarray(
        [float(item.get("score", 0.0)) for item in scored_items],
        dtype=np.float32,
    )
    best_idx = 0
    selected_logit = float(logits[best_idx])
    if len(logits) > 1:
      selected_margin = float(logits[0] - logits[1])
    else:
      selected_margin = float(selected_logit)
  elif selection_mode in ("source_value", "source_value_delta"):
    scored_items = sorted(
        scored_items,
        key=lambda item: (
            float(item.get("pred_value_delta", 0.0)),
            float(item.get("score", 0.0)),
            float(item.get("policy_prob", 0.0)),
        ),
        reverse=True,
    )
    views = [bot._source_score_view(item) for item in scored_items]
    logits = np.asarray(
        [float(item.get("pred_value_delta", 0.0)) for item in scored_items],
        dtype=np.float32,
    )
    best_idx = 0
    selected_logit = float(logits[best_idx])
    if len(logits) > 1:
      selected_margin = float(logits[0] - logits[1])
    else:
      selected_margin = float(selected_logit)
  elif selection_mode in ("proposal_model", "proposal", "model"):
    rows, views = bot._proposal_rows(
        state,
        player,
        legal,
        policy,
        int(base_action),
        scored_items,
        action_features,
    )
    if rows.shape[0] == 0:
      return {
          "mode": "proposal_veto_top1",
          "above_threshold": False,
          "disabled_reason": "empty_feature_rows",
          "base_mode": str(bot.base_mode),
          "base_action": int(base_action),
          "base_action_string": action_string(state, player, base_action),
          "candidate_count": int(len(candidates)),
          "scored_candidate_count": int(len(scored_items)),
      }
    logits = bot._score_proposals(rows)
    best_idx = int(np.argmax(logits))
    selected_logit = float(logits[best_idx])
    if len(logits) > 1:
      second_logit = float(np.partition(logits, -2)[-2])
      selected_margin = float(selected_logit - second_logit)
    else:
      selected_margin = float(selected_logit)
  else:
    raise ValueError(f"unknown proposal_veto_selection_mode={selection_mode}")
  selected_view = views[best_idx]
  selected_action = int(selected_view["candidate_action"])
  selected_item = dict(scored_items[best_idx])
  score_threshold = float(
      getattr(bot.args, "proposal_veto_score_threshold", -1e9)
  )
  margin_threshold = float(
      getattr(bot.args, "proposal_veto_margin_threshold", -1e9)
  )
  source_score_threshold = float(
      getattr(bot.args, "proposal_veto_min_source_score", -1e9)
  )
  source_margin_threshold = float(
      getattr(bot.args, "proposal_veto_min_source_margin", -1e9)
  )
  source_value_threshold = float(
      getattr(bot.args, "proposal_veto_min_source_value_delta", -1e9)
  )
  source_positive_threshold = float(
      getattr(bot.args, "proposal_veto_min_source_positive_prob", -1e9)
  )
  blocked_reason = ""
  if selected_logit < score_threshold:
    blocked_reason = "score_threshold"
  elif selected_margin < margin_threshold:
    blocked_reason = "margin_threshold"
  elif float(selected_view.get("candidate_score", 0.0)) < source_score_threshold:
    blocked_reason = "source_score_threshold"
  elif float(selected_view.get("score_margin", 0.0)) < source_margin_threshold:
    blocked_reason = "source_margin_threshold"
  elif (
      float(selected_view.get("candidate_pred_value_delta", 0.0))
      < source_value_threshold
  ):
    blocked_reason = "source_value_threshold"
  elif (
      float(selected_view.get("candidate_pred_positive_prob", 0.0))
      < source_positive_threshold
  ):
    blocked_reason = "source_positive_threshold"
  scored_candidates = []
  for rank, (item, logit) in enumerate(zip(scored_items, logits)):
    action = int(item["action"])
    scored_candidates.append({
        "action": int(action),
        "action_string": action_string(state, player, action),
        "score": float(item.get("score", 0.0)),
        "base_score": float(item.get("base_score", 0.0)),
        "margin": float(item.get("margin", 0.0)),
        "pred_value_delta": float(item.get("pred_value_delta", 0.0)),
        "pred_positive_prob": float(item.get("pred_positive_prob", 0.0)),
        "policy_prob": float(policy[int(action)]),
        "proposal_logit": float(logit),
        "proposal_rank": int(rank),
        "proposal_selected": bool(action == selected_action),
    })
  return {
      "mode": "proposal_veto_top1",
      "above_threshold": bool(not blocked_reason),
      "disabled_reason": str(blocked_reason),
      "threshold": float(score_threshold),
      "proposal_score_threshold": float(score_threshold),
      "proposal_margin_threshold": float(margin_threshold),
      "proposal_selection_mode": str(selection_mode),
      "proposal_min_source_score": float(source_score_threshold),
      "proposal_min_source_margin": float(source_margin_threshold),
      "proposal_min_source_value_delta": float(source_value_threshold),
      "proposal_min_source_positive_prob": float(source_positive_threshold),
      "base_mode": str(bot.base_mode),
      "base_action": int(base_action),
      "base_action_string": action_string(state, player, base_action),
      "candidate_action": int(selected_action),
      "candidate_action_string": action_string(state, player, selected_action),
      "candidate_score": float(selected_item.get("score", 0.0)),
      "base_score": float(selected_item.get("base_score", 0.0)),
      "score_margin": float(selected_item.get("margin", 0.0)),
      "candidate_pred_value_delta": float(
          selected_item.get("pred_value_delta", 0.0)
      ),
      "candidate_pred_positive_prob": float(
          selected_item.get("pred_positive_prob", 0.0)
      ),
      "base_policy_prob": float(policy[int(base_action)]),
      "candidate_policy_prob": float(policy[int(selected_action)]),
      "proposal_logit": float(selected_logit),
      "proposal_margin": float(selected_margin),
      "candidate_count": int(len(candidates)),
      "scored_candidate_count": int(len(scored_candidates)),
      "scored_candidates": scored_candidates,
  }


def _annotate_candidate_outcome(candidate, outcome, base):
  if "repeat_validation_mean_value_delta" in outcome:
    value_delta = float(outcome.get("repeat_validation_mean_value_delta", 0.0))
  else:
    value_delta = (
        float(outcome.get("dense_action_value_target", 0.0))
        - float(base.get("dense_action_value_target", 0.0))
    )
  if "repeat_validation_candidate_any_paradox_rate" in outcome:
    candidate_any_rate = float(
        outcome.get("repeat_validation_candidate_any_paradox_rate", 1.0)
    )
    base_any_rate = float(
        outcome.get(
            "repeat_validation_base_any_paradox_rate",
            base.get("repeat_validation_base_any_paradox_rate",
                     float(bool(base.get("any_paradox", False)))),
        )
    )
  else:
    candidate_any_rate = float(bool(outcome.get("any_paradox", False)))
    base_any_rate = float(bool(base.get("any_paradox", False)))
  risk_delta = float(candidate_any_rate - base_any_rate)
  candidate_any = bool(candidate_any_rate > 0.0)
  candidate["outcome_available"] = True
  candidate["any_paradox"] = bool(candidate_any)
  candidate["actor_paradox"] = bool(outcome.get("actor_paradox", False))
  candidate["forced_paradox"] = bool(outcome.get("forced_paradox", False))
  if "repeat_validation_candidate_any_paradox_rate" in outcome:
    candidate["repeat_validation_candidate_any_paradox_rate"] = float(
        candidate_any_rate
    )
    candidate["repeat_validation_base_any_paradox_rate"] = float(base_any_rate)
    candidate["repeat_validation_mean_value_delta"] = float(value_delta)
    candidate["repeat_validation_min_value_delta"] = float(
        outcome.get("repeat_validation_min_value_delta", value_delta)
    )
    candidate["repeat_validation_passed"] = bool(
        outcome.get("repeat_validation_passed", False)
    )
  liveness = outcome.get("liveness")
  led_color = liveness.get("led_color") if isinstance(liveness, dict) else None
  color = outcome.get("color")
  if color is None:
    action_text = str(candidate.get("action_string", outcome.get("action_string", "")))
    marker = "color="
    if marker in action_text:
      color = action_text.split(marker, 1)[1].split(")", 1)[0].strip()
  follows_led = bool(color is not None and led_color is not None and color == led_color)
  is_red_switch = bool(color == "R" and led_color not in (None, "R"))
  switches_off_led = bool(
      color is not None and led_color is not None and color != led_color
  )
  candidate["led_color"] = led_color
  candidate["color"] = color
  candidate["follows_led"] = follows_led
  candidate["is_red_switch"] = is_red_switch
  candidate["switches_off_led"] = switches_off_led
  if isinstance(liveness, dict):
    candidate["lost_led_token"] = bool(liveness.get("lost_led_token", False))
    candidate["would_win_now"] = bool(liveness.get("would_win_now", False))
    candidate["responsible_red_win_switch"] = bool(
        liveness.get("responsible_red_win_switch", False)
    )
  candidate["acting_score_delta"] = float(outcome.get("acting_score_delta", 0.0))
  candidate["total_score_delta"] = float(outcome.get("total_score_delta", 0.0))
  candidate["dense_action_value_delta_vs_base"] = float(value_delta)
  candidate["paradox_delta_vs_base"] = float(risk_delta)
  candidate["positive_by_dense_delta"] = bool(value_delta >= 0.01 and risk_delta <= 0.0)
  candidate["safe_by_paradox_delta"] = bool(risk_delta <= 0.0)
  candidate["positive_safe_no_candidate_any_paradox"] = bool(
      candidate["positive_by_dense_delta"]
      and candidate["safe_by_paradox_delta"]
      and not candidate["any_paradox"]
  )


def _candidate_source_agrees(candidate):
  return bool(
      float(candidate.get("margin", -1e9)) >= 0.0
      and float(candidate.get("pred_value_delta", -1e9)) >= 0.0
      and float(candidate.get("pred_positive_prob", 0.0)) >= 0.5
  )


def _candidate_is_bad_red_lane_burn_no_win(candidate):
  return bool(
      bool(candidate.get("is_red_switch", False))
      and bool(candidate.get("switches_off_led", False))
      and bool(candidate.get("lost_led_token", False))
      and candidate.get("would_win_now") is False
      and not bool(candidate.get("responsible_red_win_switch", False))
      and bool(candidate.get("any_paradox", False))
  )


def _proposal_candidate_color(candidate):
  color = candidate.get("color")
  if color is not None:
    return color
  action_text = str(candidate.get("action_string", ""))
  marker = "color="
  if marker in action_text:
    return action_text.split(marker, 1)[1].split(")", 1)[0].strip()
  return None


def _decision_prefilter_tactical_shape(
    decision,
    args,
    summary,
    base_action=None,
    counter_prefix="preproposal_tactical",
):
  shape = str(getattr(args, "prefilter_proposal_tactical_shape", "none") or "none")
  if shape == "none":
    return True
  if shape != "bad_red_no_win_and_lane_preserver":
    raise ValueError(f"unknown proposal tactical prefilter shape: {shape}")
  state = decision["state"]
  player = int(decision["player"])
  led_color = getattr(state, "_led_color", None)
  summary[f"{counter_prefix}_checked"] += 1
  if led_color in (None, "R"):
    summary[f"{counter_prefix}_no_nonred_led"] += 1
    return False
  legal = [int(action) for action in decision.get("legal", [])]
  if base_action is None:
    base_action = int(decision.get("action", legal[0] if legal else 999))
  else:
    base_action = int(base_action)
  has_bad_red_no_win = False
  has_lane_preserver = False
  for action in legal:
    if int(action) == 999:
      continue
    cert = action_liveness_certificate(
        state,
        player,
        int(action),
        base_action=base_action,
    )
    color = _action_color(state, int(action))
    responsible_red_win = bool(cert.get("responsible_red_win_switch", False))
    if color == led_color or responsible_red_win:
      has_lane_preserver = True
    if (
        color == "R"
        and bool(cert.get("lost_led_token", False))
        and cert.get("would_win_now") is False
        and not responsible_red_win
    ):
      has_bad_red_no_win = True
    if has_bad_red_no_win and has_lane_preserver:
      summary[f"{counter_prefix}_passed"] += 1
      return True
  if has_bad_red_no_win:
    summary[f"{counter_prefix}_has_bad_red_no_win"] += 1
  if has_lane_preserver:
    summary[f"{counter_prefix}_has_lane_preserver"] += 1
  summary[f"{counter_prefix}_rejected_{shape}"] += 1
  return False


def _proposal_prefilter_tactical_shape(proposal, decision, args, summary):
  shape = str(getattr(args, "prefilter_proposal_tactical_shape", "none") or "none")
  if shape == "none":
    return True
  if proposal is None or not bool(proposal.get("above_threshold", False)):
    return False
  if shape != "bad_red_no_win_and_lane_preserver":
    raise ValueError(f"unknown proposal tactical prefilter shape: {shape}")
  state = decision["state"]
  player = int(decision["player"])
  led_color = getattr(state, "_led_color", None)
  if led_color in (None, "R"):
    summary["prefilter_tactical_no_nonred_led"] += 1
    return False
  scored = proposal.get("scored_candidates")
  if not isinstance(scored, list):
    summary["prefilter_tactical_no_scored_candidates"] += 1
    return False
  base_action = int(proposal.get("base_action", decision["action"]))
  has_bad_red_no_win = False
  has_lane_preserver = False
  for candidate in scored:
    if not isinstance(candidate, dict) or "action" not in candidate:
      continue
    action = int(candidate["action"])
    if action == 999:
      continue
    cert = action_liveness_certificate(
        state,
        player,
        action,
        base_action=base_action,
    )
    color = _proposal_candidate_color(candidate)
    follows_led = bool(color == led_color)
    responsible_red_win = bool(cert.get("responsible_red_win_switch", False))
    if follows_led or responsible_red_win:
      has_lane_preserver = True
    if (
        color == "R"
        and bool(cert.get("lost_led_token", False))
        and cert.get("would_win_now") is False
        and not responsible_red_win
    ):
      has_bad_red_no_win = True
  summary["prefilter_tactical_checked"] += 1
  if has_bad_red_no_win:
    summary["prefilter_tactical_has_bad_red_no_win"] += 1
  if has_lane_preserver:
    summary["prefilter_tactical_has_lane_preserver"] += 1
  if has_bad_red_no_win and has_lane_preserver:
    summary["prefilter_tactical_passed"] += 1
    return True
  summary[f"prefilter_tactical_rejected_{shape}"] += 1
  return False


def _candidate_is_safe_source_agreeing_alternative(candidate):
  preserves_lane_or_responsible_red = bool(
      bool(candidate.get("follows_led", False))
      or bool(candidate.get("responsible_red_win_switch", False))
  )
  return bool(
      preserves_lane_or_responsible_red
      and _candidate_source_agrees(candidate)
      and bool(candidate.get("safe_by_paradox_delta", False))
      and not bool(candidate.get("any_paradox", False))
  )


def _candidate_is_safe_lane_preserving_alternative(candidate):
  preserves_lane_or_responsible_red = bool(
      bool(candidate.get("follows_led", False))
      or bool(candidate.get("responsible_red_win_switch", False))
  )
  return bool(
      preserves_lane_or_responsible_red
      and bool(candidate.get("safe_by_paradox_delta", False))
      and not bool(candidate.get("any_paradox", False))
  )


def _annotate_proposal_outcome(dense_meta, proposal, args=None):
  if not proposal or not bool(proposal.get("above_threshold", False)):
    return
  candidate_action = int(proposal["candidate_action"])
  base_action = int(proposal["base_action"])
  outcomes = {
      int(outcome.get("action", -1)): outcome
      for outcome in dense_meta.get("action_outcomes", [])
  }
  candidate = outcomes.get(candidate_action)
  base = outcomes.get(base_action)
  if candidate is None or base is None:
    proposal["outcome_available"] = False
    return
  if "repeat_validation_mean_value_delta" in candidate:
    value_delta = float(candidate.get("repeat_validation_mean_value_delta", 0.0))
  else:
    value_delta = (
        float(candidate.get("dense_action_value_target", 0.0))
        - float(base.get("dense_action_value_target", 0.0))
    )
  if "repeat_validation_candidate_any_paradox_rate" in candidate:
    candidate_any_rate = float(
        candidate.get("repeat_validation_candidate_any_paradox_rate", 1.0)
    )
    base_any_rate = float(
        candidate.get(
            "repeat_validation_base_any_paradox_rate",
            base.get("repeat_validation_base_any_paradox_rate",
                     float(bool(base.get("any_paradox", False)))),
        )
    )
  else:
    candidate_any_rate = float(bool(candidate.get("any_paradox", False)))
    base_any_rate = float(bool(base.get("any_paradox", False)))
  risk_delta = float(candidate_any_rate - base_any_rate)
  proposal["outcome_available"] = True
  proposal["candidate_any_paradox"] = bool(candidate_any_rate > 0.0)
  proposal["base_any_paradox"] = bool(base_any_rate > 0.0)
  if "repeat_validation_candidate_any_paradox_rate" in candidate:
    proposal["repeat_validation_candidate_any_paradox_rate"] = float(
        candidate_any_rate
    )
    proposal["repeat_validation_base_any_paradox_rate"] = float(base_any_rate)
    proposal["repeat_validation_mean_value_delta"] = float(value_delta)
    proposal["repeat_validation_passed"] = bool(
        candidate.get("repeat_validation_passed", False)
    )
  proposal["candidate_actor_paradox"] = bool(candidate.get("actor_paradox", False))
  proposal["candidate_forced_paradox"] = bool(candidate.get("forced_paradox", False))
  proposal["acting_score_delta"] = float(candidate.get("acting_score_delta", 0.0))
  proposal["total_score_delta"] = float(candidate.get("total_score_delta", 0.0))
  proposal["dense_action_value_delta_vs_base"] = float(value_delta)
  proposal["paradox_delta_vs_base"] = float(risk_delta)
  proposal["positive_by_dense_delta"] = bool(value_delta >= 0.01 and risk_delta <= 0.0)
  proposal["safe_by_paradox_delta"] = bool(risk_delta <= 0.0)
  scored = proposal.get("scored_candidates")
  if isinstance(scored, list):
    available = 0
    positive = 0
    positive_safe = 0
    positive_safe_no_candidate_any = 0
    same_led_candidate_any = 0
    same_led_positive_safe_no_candidate_any = 0
    safe = 0
    candidate_any = 0
    bad_red_lane_burn_no_win = 0
    safe_lane_preserving_alternative = 0
    safe_source_agreeing_alternative = 0
    non_red_led_scored = 0
    bad_red_lane_burn_no_win_actions = []
    safe_lane_preserving_alternative_actions = []
    safe_source_agreeing_alternative_actions = []
    for item in scored:
      if not isinstance(item, dict):
        continue
      action_outcome = outcomes.get(int(item.get("action", -1)))
      if action_outcome is None:
        item["outcome_available"] = False
        continue
      _annotate_candidate_outcome(item, action_outcome, base)
      available += 1
      if item.get("led_color") in ("B", "Y", "G"):
        non_red_led_scored += 1
      positive += int(bool(item.get("positive_by_dense_delta", False)))
      safe += int(bool(item.get("safe_by_paradox_delta", False)))
      positive_safe += int(
          bool(item.get("positive_by_dense_delta", False))
          and bool(item.get("safe_by_paradox_delta", False))
      )
      positive_safe_no_candidate_any += int(
          bool(item.get("positive_safe_no_candidate_any_paradox", False))
      )
      candidate_any += int(bool(item.get("any_paradox", False)))
      if bool(item.get("follows_led", False)) and bool(item.get("any_paradox", False)):
        same_led_candidate_any += 1
      if (
          bool(item.get("follows_led", False))
          and bool(item.get("positive_safe_no_candidate_any_paradox", False))
      ):
        same_led_positive_safe_no_candidate_any += 1
      if _candidate_is_bad_red_lane_burn_no_win(item):
        bad_red_lane_burn_no_win += 1
        bad_red_lane_burn_no_win_actions.append(int(item.get("action", -1)))
      if _candidate_is_safe_lane_preserving_alternative(item):
        safe_lane_preserving_alternative += 1
        safe_lane_preserving_alternative_actions.append(int(item.get("action", -1)))
      if _candidate_is_safe_source_agreeing_alternative(item):
        safe_source_agreeing_alternative += 1
        safe_source_agreeing_alternative_actions.append(int(item.get("action", -1)))
    proposal["scored_outcome_available_count"] = int(available)
    proposal["scored_positive_by_dense_delta_count"] = int(positive)
    proposal["scored_safe_by_paradox_delta_count"] = int(safe)
    proposal["scored_positive_safe_count"] = int(positive_safe)
    proposal["scored_positive_safe_no_candidate_any_paradox_count"] = int(
        positive_safe_no_candidate_any
    )
    proposal["scored_candidate_any_paradox_count"] = int(candidate_any)
    proposal["scored_candidate_any_paradox_rate"] = (
        float(candidate_any) / float(available) if available > 0 else 0.0
    )
    proposal["scored_same_led_candidate_any_paradox_count"] = int(
        same_led_candidate_any
    )
    proposal[
        "scored_same_led_positive_safe_no_candidate_any_paradox_count"
    ] = int(same_led_positive_safe_no_candidate_any)
    proposal["scored_positive_safe_no_candidate_any_paradox_rate"] = (
        float(positive_safe_no_candidate_any) / float(available)
        if available > 0 else 0.0
    )
    proposal["scored_non_red_led_count"] = int(non_red_led_scored)
    proposal["scored_bad_red_lane_burn_no_win_count"] = int(
        bad_red_lane_burn_no_win
    )
    proposal["scored_safe_source_agreeing_alternative_count"] = int(
        safe_source_agreeing_alternative
    )
    proposal["scored_safe_lane_preserving_alternative_count"] = int(
        safe_lane_preserving_alternative
    )
    proposal["scored_bad_red_lane_burn_no_win_actions"] = [
        int(action) for action in bad_red_lane_burn_no_win_actions
    ]
    proposal["scored_safe_lane_preserving_alternative_actions"] = [
        int(action) for action in safe_lane_preserving_alternative_actions
    ]
    proposal["scored_safe_source_agreeing_alternative_actions"] = [
        int(action) for action in safe_source_agreeing_alternative_actions
    ]
    proposal["scored_bad_red_lane_burn_no_win_with_safe_alternative"] = bool(
        bad_red_lane_burn_no_win > 0 and safe_source_agreeing_alternative > 0
    )
    proposal["scored_bad_red_lane_burn_no_win_with_safe_any_alternative"] = bool(
        bad_red_lane_burn_no_win > 0 and safe_lane_preserving_alternative > 0
    )
    min_any_rate = float(
        getattr(args, "source10_density_min_candidate_any_rate", 0.70)
    )
    max_approved_rate = float(
        getattr(args, "source10_density_max_approved_rate", 0.15)
    )
    min_scored_candidates = int(
        getattr(args, "source10_density_min_scored_candidates", 8)
    )
    proposal["scored_source10_dense_candidate_any_sparse_approved"] = bool(
        available >= min_scored_candidates
        and non_red_led_scored > 0
        and same_led_candidate_any > 0
        and positive_safe_no_candidate_any > 0
        and (float(candidate_any) / float(available)) >= min_any_rate
        and (
            float(positive_safe_no_candidate_any) / float(available)
        ) <= max_approved_rate
    )


def _proposal_outcome_matches(proposal, required_outcome):
  required_outcome = str(required_outcome or "none")
  if required_outcome == "none":
    return True
  if not proposal or not bool(proposal.get("above_threshold", False)):
    return False
  if not bool(proposal.get("outcome_available", False)):
    return False
  positive = bool(proposal.get("positive_by_dense_delta", False))
  safe = bool(proposal.get("safe_by_paradox_delta", False))
  candidate_any = bool(proposal.get("candidate_any_paradox", False))
  value_delta = float(proposal.get("dense_action_value_delta_vs_base", 0.0))
  risk_delta = float(proposal.get("paradox_delta_vs_base", 0.0))
  scored_available = int(proposal.get("scored_outcome_available_count", 0))
  scored_positive_safe = int(proposal.get("scored_positive_safe_count", 0))
  scored_positive_safe_no_candidate_any = int(
      proposal.get("scored_positive_safe_no_candidate_any_paradox_count", 0)
  )
  scored_candidate_any = int(proposal.get("scored_candidate_any_paradox_count", 0))
  scored_same_led_candidate_any = int(
      proposal.get("scored_same_led_candidate_any_paradox_count", 0)
  )
  scored_same_led_positive_safe_no_candidate_any = int(
      proposal.get(
          "scored_same_led_positive_safe_no_candidate_any_paradox_count",
          0,
      )
  )
  scored_bad_red_lane_burn_no_win = int(
      proposal.get("scored_bad_red_lane_burn_no_win_count", 0)
  )
  scored_safe_lane_preserving_alternative = int(
      proposal.get("scored_safe_lane_preserving_alternative_count", 0)
  )
  scored_safe_source_agreeing_alternative = int(
      proposal.get("scored_safe_source_agreeing_alternative_count", 0)
  )
  if required_outcome == "positive":
    return positive
  if required_outcome == "positive_safe":
    return bool(positive and safe)
  if required_outcome == "safe":
    return safe
  if required_outcome == "unsafe":
    return not safe
  if required_outcome == "candidate_any_paradox":
    return candidate_any
  if required_outcome == "candidate_worse":
    return bool(value_delta < -0.01 or risk_delta > 0.0)
  if required_outcome == "scored_has_positive_safe":
    return bool(scored_positive_safe > 0)
  if required_outcome == "scored_mixed_positive_safe":
    return bool(scored_positive_safe > 0 and scored_available > scored_positive_safe)
  if required_outcome == "scored_has_positive_safe_no_candidate_any_paradox":
    return bool(scored_positive_safe_no_candidate_any > 0)
  if required_outcome == "scored_mixed_positive_safe_no_candidate_any_paradox":
    return bool(
        scored_positive_safe_no_candidate_any > 0
        and scored_available > scored_positive_safe_no_candidate_any
    )
  if required_outcome == "scored_has_candidate_any_paradox":
    return bool(scored_candidate_any > 0)
  if (
      required_outcome
      == "scored_candidate_any_with_positive_safe_no_candidate_any_alternative"
  ):
    return bool(
        scored_candidate_any > 0
        and scored_positive_safe_no_candidate_any > 0
    )
  if (
      required_outcome
      == "scored_same_led_candidate_any_with_same_led_positive_safe_no_candidate_any_alternative"
  ):
    return bool(
        scored_same_led_candidate_any > 0
        and scored_same_led_positive_safe_no_candidate_any > 0
    )
  if required_outcome == "scored_source10_dense_candidate_any_sparse_approved":
    return bool(
        proposal.get("scored_source10_dense_candidate_any_sparse_approved", False)
    )
  if required_outcome == "scored_bad_red_lane_burn_no_win_with_safe_alternative":
    return bool(
        scored_bad_red_lane_burn_no_win > 0
        and scored_safe_source_agreeing_alternative > 0
    )
  if (
      required_outcome
      == "scored_bad_red_lane_burn_no_win_with_safe_any_alternative"
  ):
    return bool(
        scored_bad_red_lane_burn_no_win > 0
        and scored_safe_lane_preserving_alternative > 0
    )
  raise ValueError(f"unknown proposal outcome bucket: {required_outcome}")


def _decision_allowed(decision, allowed_phases, args):
  if not should_evaluate_decision(decision, allowed_phases):
    return False
  legal_count = len(decision["legal"])
  if legal_count < int(args.min_legal_count):
    return False
  max_legal = int(args.max_legal_count)
  if max_legal > 0 and legal_count > max_legal:
    return False
  trick = int(decision.get("trick_number", -1))
  if trick < int(args.min_trick_number):
    return False
  if trick > int(args.max_trick_number):
    return False
  if bool(getattr(args, "led_contrast_only", False)):
    if str(decision.get("phase")) != "play":
      return False
    profile = _led_contrast_profile(decision)
    if profile["led_color"] not in ("B", "Y", "G"):
      return False
    if int(profile["led_legal_count"]) <= 0:
      return False
    if int(profile["switch_legal_count"]) <= 0:
      return False
    if (
        bool(getattr(args, "require_red_switch_legal", False))
        and int(profile["red_legal_count"]) <= 0
    ):
      return False
  return True


def main():
  args = parse_args()
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model, model_args = load_neural(args.checkpoint, args.players, device, args)
  value_scale = float(model_args.value_scale)
  proposal_boundary = None
  proposal_action_set_bot = None
  proposal_veto_bot = None
  if str(args.proposal_mode) == "pairwise_boundary":
    if not str(args.pairwise_boundary_model or "").strip():
      raise ValueError("--pairwise-boundary-model is required for pairwise proposals")
    proposal_boundary = load_pairwise_boundary(args.pairwise_boundary_model, device)
  elif str(args.proposal_mode) == "action_set_transformer":
    if not str(args.action_set_transformer_model or "").strip():
      raise ValueError(
          "--action-set-transformer-model is required for action-set proposals"
      )
    proposal_action_set_bot = AZActionSetTransformerBot(
        model,
        load_action_set_transformer(args.action_set_transformer_model, device),
        "proposal_action_set_transformer",
        device,
        _action_set_proposal_args(args, value_scale),
    )
  elif str(args.proposal_mode) == "proposal_veto_top1":
    if not str(args.action_set_transformer_model or "").strip():
      raise ValueError(
          "--action-set-transformer-model is required for proposal-veto proposals"
      )
    if not str(args.proposal_veto_model or "").strip():
      raise ValueError("--proposal-veto-model is required for proposal-veto proposals")
    proposal_veto_bot = AZProposalVetoTop1Bot(
        model,
        load_action_set_transformer(args.action_set_transformer_model, device),
        load_proposal_veto_selector(args.proposal_veto_model, device),
        "proposal_veto_top1",
        device,
        _proposal_veto_args(args, value_scale),
    )
  eval_args = copy.copy(args)
  eval_args.enable_behavior_exploration = False
  eval_args.behavior_exploration_prob = 0.0
  eval_args.behavior_led_contrast_switch_prob = 0.0
  allowed_phases = parse_phase_filter(args.decision_phases)
  rows = []
  metadata_rows = []
  repeat_validation_attempt_rows = []
  summary = Counter()
  led_bucket_rows = Counter()
  repeat_validation_label_color_rows = Counter()
  last_partial_decisions = -1

  def reached_collection_cap():
    if (
        int(args.max_decisions) > 0
        and summary["decisions_evaluated"] >= int(args.max_decisions)
    ):
      return True
    return bool(
        int(getattr(args, "max_accepted_rows", 0) or 0) > 0
        and summary["dense_label_rows"] >= int(args.max_accepted_rows)
    )

  def write_partial_snapshot(force=False):
    nonlocal last_partial_decisions
    interval = int(getattr(args, "partial_save_interval_decisions", 0) or 0)
    if interval <= 0 and not force:
      return
    decisions = int(summary["decisions_evaluated"])
    if not force and last_partial_decisions >= 0:
      if decisions - int(last_partial_decisions) < interval:
        return
    if not force and not rows and not repeat_validation_attempt_rows:
      return
    out_path = Path(args.out)
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    partial_replay_path = partial_path.with_suffix(partial_path.suffix + ".npz")
    partial_metadata = {
        "source": "quantum_cat_live_substitution_replay",
        "partial": True,
        "checkpoint": str(args.checkpoint),
        "players": int(args.players),
        "seed": int(args.seed),
        "rows": int(len(rows)),
        "repeat_validation_attempt_rows": int(len(repeat_validation_attempt_rows)),
        "record_round_history": bool(args.record_round_history),
        "round_history_max_events": int(args.round_history_max_events),
        "min_round_history_avoidable_switch_events": int(
            args.min_round_history_avoidable_switch_events
        ),
        "min_round_history_avoidable_nonred_switch_events": int(
            args.min_round_history_avoidable_nonred_switch_events
        ),
        "min_round_history_avoidable_token_loss_events": int(
            args.min_round_history_avoidable_token_loss_events
        ),
        "summary": dict(summary),
        "led_bucket_rows": dict(led_bucket_rows),
        "repeat_validation_label_color_rows": dict(
            repeat_validation_label_color_rows
        ),
    }
    if rows:
      save_replay(rows, partial_replay_path, metadata=partial_metadata)
    metadata_path = out_path.with_suffix(out_path.suffix + ".partial.metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                **partial_metadata,
                "replay": str(partial_replay_path) if rows else "",
                "repeat_validation_attempt_rows": repeat_validation_attempt_rows,
                "rows": metadata_rows,
            },
            indent=2,
        )
        + "\n"
    )
    summary_path = out_path.with_suffix(out_path.suffix + ".partial.summary.json")
    summary_path.write_text(json.dumps(partial_metadata, indent=2) + "\n")
    last_partial_decisions = decisions

  for match_idx in range(int(args.matches)):
    if reached_collection_cap():
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if reached_collection_cap():
        break
      start_player = round_idx % int(args.players)
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
          match_totals=prior_totals if args.match_context else None,
          round_index=round_idx,
          args=args,
      )
      result = terminal_round_result(terminal)
      returns = np.array(terminal.returns(), dtype=np.float32)
      totals = returns if args.match_context else totals + returns
      summary["rounds"] += 1
      if result["any_paradox"]:
        summary["paradox_rounds"] += 1
      evaluated_this_round = 0
      for decision in decisions:
        if reached_collection_cap():
          break
        if (
            int(args.max_decisions_per_round) > 0
            and evaluated_this_round >= int(args.max_decisions_per_round)
        ):
          break
        if not _decision_allowed(decision, allowed_phases, args):
          continue
        if not _decision_prefilter_tactical_shape(
            decision,
            args,
            summary,
            counter_prefix="preproposal_raw_tactical",
        ):
          summary[
              "skipped_preproposal_raw_tactical_shape_"
              + str(getattr(args, "prefilter_proposal_tactical_shape", "none"))
          ] += 1
          continue
        proposal = None
        if str(args.proposal_mode) == "pairwise_boundary":
          summary["proposal_states_considered"] += 1
          proposal = pairwise_boundary_proposal(
              model,
              proposal_boundary,
              decision,
              value_scale,
              device,
              args,
          )
          if proposal is None:
            summary["proposal_none"] += 1
          elif bool(proposal.get("above_threshold", False)):
            summary["proposal_above_threshold"] += 1
          else:
            summary["proposal_below_threshold"] += 1
        elif str(args.proposal_mode) == "action_set_transformer":
          summary["proposal_states_considered"] += 1
          proposal = action_set_transformer_proposal(
              proposal_action_set_bot,
              decision,
              summary,
          )
          if proposal is None:
            summary["proposal_none"] += 1
          elif bool(proposal.get("above_threshold", False)):
            summary["proposal_above_threshold"] += 1
          else:
            summary["proposal_below_threshold"] += 1
        elif str(args.proposal_mode) == "proposal_veto_top1":
          summary["proposal_states_considered"] += 1
          proposal = proposal_veto_top1_proposal(
              proposal_veto_bot,
              decision,
          )
          if proposal is None:
            summary["proposal_none"] += 1
          elif bool(proposal.get("above_threshold", False)):
            summary["proposal_above_threshold"] += 1
          else:
            summary["proposal_below_threshold"] += 1
        if bool(args.require_proposal) and not (
            proposal is not None and bool(proposal.get("above_threshold", False))
        ):
          summary["skipped_no_required_proposal"] += 1
          continue
        if not _proposal_prefilter_tactical_shape(
            proposal,
            decision,
            args,
            summary,
        ):
          shape = str(
              getattr(args, "prefilter_proposal_tactical_shape", "none")
              or "none"
          )
          summary[f"skipped_prefilter_proposal_tactical_shape_{shape}"] += 1
          continue
        label_decision = decision
        if proposal is not None and "base_action" in proposal:
          base_action = int(proposal["base_action"])
          if base_action != int(decision["action"]):
            label_decision = _replace_decision_action(decision, base_action)
            summary["proposal_base_replaced_behavior_action"] += 1
        elif str(args.label_base_action_source) == "frozen_policy":
          base_action = _frozen_policy_base_action(
              model,
              decision,
              value_scale,
              device,
              args,
          )
          if base_action != int(decision["action"]):
            label_decision = _replace_decision_action(decision, base_action)
            summary["frozen_base_replaced_behavior_action"] += 1
        led_profile = _led_contrast_profile(label_decision)
        selected_bucket = str(led_profile["selected_bucket"])
        max_bucket_rows = int(getattr(args, "max_led_bucket_rows", 0) or 0)
        if (
            bool(getattr(args, "led_contrast_only", False))
            and max_bucket_rows > 0
            and led_bucket_rows[selected_bucket] >= max_bucket_rows
        ):
          summary[f"skipped_full_led_bucket_{selected_bucket}"] += 1
          continue
        evaluated_this_round += 1
        summary["decisions_evaluated"] += 1
        summary[f"decisions_evaluated_phase_{decision['phase']}"] += 1
        if bool(getattr(args, "led_contrast_only", False)):
          summary[f"led_contrast_bucket_{selected_bucket}"] += 1
          summary[f"led_color_{led_profile['led_color']}"] += 1
          if int(led_profile["red_legal_count"]) > 0:
            summary["led_contrast_red_legal"] += 1
          if bool(led_profile["avoidable_switch"]):
            summary["led_contrast_avoidable_switch_selected"] += 1
        if (
            int(args.progress_interval) > 0
            and summary["decisions_evaluated"] % int(args.progress_interval) == 0
        ):
          print(json.dumps({
              "iteration": "live_substitution_progress",
              "matches": int(match_idx + 1),
              "rounds": int(summary["rounds"]),
              "rows": int(len(rows)),
              "decisions_evaluated": int(summary["decisions_evaluated"]),
              "paradox_rounds": int(summary["paradox_rounds"]),
              "phase": str(label_decision["phase"]),
              "legal_count": int(len(label_decision["legal"])),
              "trick_number": int(label_decision.get("trick_number", -1)),
              "led_bucket_rows": dict(led_bucket_rows),
              "proposal_above_threshold": int(summary["proposal_above_threshold"]),
          }), flush=True)
          write_partial_snapshot()
        prefilter_repeat_validation = None
        prefilter_attempt_index = None
        if (
            bool(getattr(args, "repeat_validate_labels", False))
            and bool(getattr(args, "repeat_validation_prefilter_candidates", False))
        ):
          prefilter_repeat_validation = _repeat_validate_direct_candidate_queue(
              model,
              label_decision,
              value_scale,
              device,
              eval_args,
              led_profile,
              seed_base=round_seed + int(label_decision["index"]) * 1009,
          )
          summary["repeat_validation_attempted"] += 1
          summary["repeat_validation_prefilter_attempted"] += 1
          if bool(getattr(args, "record_repeat_validation_attempts", False)):
            prefilter_attempt_index = len(repeat_validation_attempt_rows)
            repeat_validation_attempt_rows.append(
                _repeat_validation_attempt_meta(
                    label_decision,
                    led_profile,
                    prefilter_repeat_validation,
                    match_idx=match_idx,
                    round_idx=round_idx,
                    round_seed=round_seed,
                    admission="prefilter_evaluated",
                )
            )
          if bool(prefilter_repeat_validation.get("passed", False)):
            passed_candidates = [
                item for item in prefilter_repeat_validation.get("candidates", [])
                if bool(item.get("passed", False))
            ]
            if bool(args.repeat_validation_require_passed_red_switch) and not any(
                bool(item.get("candidate_is_red_switch", False))
                for item in passed_candidates
            ):
              summary["skipped_repeat_validation_no_passed_red_switch"] += 1
              if prefilter_attempt_index is not None:
                repeat_validation_attempt_rows[prefilter_attempt_index][
                    "admission"
                ] = "rejected_no_passed_red_switch"
              write_partial_snapshot(force=True)
              continue
            if bool(args.repeat_validation_require_passed_follow_led) and not any(
                bool(item.get("candidate_follows_led", False))
                for item in passed_candidates
            ):
              summary["skipped_repeat_validation_no_passed_follow_led"] += 1
              if prefilter_attempt_index is not None:
                repeat_validation_attempt_rows[prefilter_attempt_index][
                    "admission"
                ] = "rejected_no_passed_follow_led"
              write_partial_snapshot(force=True)
              continue
            selected_candidate = int(
                prefilter_repeat_validation.get(
                    "candidate_action", label_decision["action"]
                )
            )
            if (
                bool(getattr(args, "require_policy_target_changed", False))
                and selected_candidate == int(label_decision["action"])
            ):
              summary["skipped_policy_target_unchanged"] += 1
              if prefilter_attempt_index is not None:
                repeat_validation_attempt_rows[prefilter_attempt_index][
                    "admission"
                ] = "rejected_policy_target_unchanged"
              write_partial_snapshot(force=True)
              continue
            summary["repeat_validation_prefilter_passed"] += 1
            if prefilter_attempt_index is not None:
              repeat_validation_attempt_rows[prefilter_attempt_index][
                  "admission"
              ] = "prefilter_passed"
          else:
            reason = str(
                prefilter_repeat_validation.get("reason", "unknown")
            )
            summary[f"repeat_validation_failed_{reason}"] += 1
            summary["repeat_validation_prefilter_failed"] += 1
            summary["skipped_repeat_validation_failed"] += 1
            if prefilter_attempt_index is not None:
              repeat_validation_attempt_rows[prefilter_attempt_index][
                  "admission"
              ] = f"rejected_{reason}"
            write_partial_snapshot(force=True)
            continue
        evaluated, _avoiding, alternatives = evaluate_decision(
            model,
            label_decision,
            value_scale,
            device,
            eval_args,
            rollout_seed=round_seed,
        )
        dense_row, dense_meta = dense_label_row_for_decision(
            label_decision,
            evaluated["baseline"],
            alternatives,
            args,
            value_scale,
        )
        if dense_row is None:
          summary["null_dense_rows"] += 1
          if prefilter_attempt_index is not None:
            repeat_validation_attempt_rows[prefilter_attempt_index][
                "admission"
            ] = "rejected_null_dense_row"
          write_partial_snapshot(force=True)
          continue
        dense_meta["source_index"] = int(len(metadata_rows))
        dense_meta["group"] = {
            "proposal_source": "live_substitution",
            "match_index": int(match_idx),
            "round_index": int(round_idx),
            "decision_index": int(label_decision["index"]),
            "start_player": int(start_player),
            "terminal_any_paradox": bool(result["any_paradox"]),
        }
        if proposal is not None:
          dense_meta["proposal"] = dict(proposal)
          _annotate_proposal_outcome(dense_meta, dense_meta["proposal"], args)
          dense_meta["group"]["proposal_source"] = (
              "live_" + str(proposal.get("mode", args.proposal_mode))
          )
          if bool(dense_meta["proposal"].get("above_threshold", False)):
            if bool(dense_meta["proposal"].get("outcome_available", False)):
              summary["proposal_outcome_available"] += 1
              if bool(dense_meta["proposal"].get("positive_by_dense_delta", False)):
                summary["proposal_positive_by_dense_delta"] += 1
              if bool(dense_meta["proposal"].get("safe_by_paradox_delta", False)):
                summary["proposal_safe_by_paradox_delta"] += 1
              if bool(dense_meta["proposal"].get("candidate_any_paradox", False)):
                summary["proposal_candidate_any_paradox"] += 1
              if bool(dense_meta["proposal"].get("base_any_paradox", False)):
                summary["proposal_base_any_paradox"] += 1
              summary[
                  "proposal_scored_bad_red_lane_burn_no_win"
              ] += int(
                  dense_meta["proposal"].get(
                      "scored_bad_red_lane_burn_no_win_count",
                      0,
                  )
              )
              summary[
                  "proposal_scored_safe_source_agreeing_alternative"
              ] += int(
                  dense_meta["proposal"].get(
                      "scored_safe_source_agreeing_alternative_count",
                      0,
                  )
              )
              summary[
                  "proposal_scored_safe_lane_preserving_alternative"
              ] += int(
                  dense_meta["proposal"].get(
                      "scored_safe_lane_preserving_alternative_count",
                      0,
                  )
              )
              if bool(
                  dense_meta["proposal"].get(
                      "scored_bad_red_lane_burn_no_win_with_safe_alternative",
                      False,
                  )
              ):
                summary[
                    "proposal_scored_bad_red_lane_burn_no_win_with_safe_alternative"
                ] += 1
              if bool(
                  dense_meta["proposal"].get(
                      "scored_bad_red_lane_burn_no_win_with_safe_any_alternative",
                      False,
                  )
              ):
                summary[
                    "proposal_scored_bad_red_lane_burn_no_win_with_safe_any_alternative"
                ] += 1
              summary[
                  "proposal_scored_positive_safe_no_candidate_any_paradox"
              ] += int(
                  dense_meta["proposal"].get(
                      "scored_positive_safe_no_candidate_any_paradox_count",
                      0,
                  )
              )
              summary["proposal_scored_same_led_candidate_any_paradox"] += int(
                  dense_meta["proposal"].get(
                      "scored_same_led_candidate_any_paradox_count",
                      0,
                  )
              )
              summary[
                  "proposal_scored_same_led_positive_safe_no_candidate_any_paradox"
              ] += int(
                  dense_meta["proposal"].get(
                      "scored_same_led_positive_safe_no_candidate_any_paradox_count",
                      0,
                  )
              )
              if bool(
                  dense_meta["proposal"].get(
                      "scored_source10_dense_candidate_any_sparse_approved",
                      False,
                  )
              ):
                summary[
                    "proposal_scored_source10_dense_candidate_any_sparse_approved"
                ] += 1
        required_proposal_outcome = str(
            getattr(args, "require_proposal_outcome", "none") or "none"
        )
        if not _proposal_outcome_matches(
            dense_meta.get("proposal"), required_proposal_outcome
        ):
          summary[f"skipped_proposal_outcome_{required_proposal_outcome}"] += 1
          if prefilter_attempt_index is not None:
            repeat_validation_attempt_rows[prefilter_attempt_index][
                "admission"
            ] = f"rejected_proposal_outcome_{required_proposal_outcome}"
          write_partial_snapshot(force=True)
          continue
        if required_proposal_outcome != "none":
          summary[f"proposal_outcome_required_{required_proposal_outcome}"] += 1
        dense_meta["behavior_action"] = int(decision["action"])
        dense_meta["behavior_action_string"] = str(decision["action_string"])
        dense_meta["label_base_action_source"] = str(args.label_base_action_source)
        dense_meta["label_base_replaced_behavior_action"] = bool(
            int(label_decision["action"]) != int(decision["action"])
        )
        _annotate_led_contrast_meta(dense_meta, label_decision, led_profile)
        round_history = None
        round_history_summary = None
        if _wants_round_history(args):
          round_history = _round_history_before(
              decisions,
              label_decision,
              int(getattr(args, "round_history_max_events", 40)),
          )
          round_history_summary = _round_history_summary(
              round_history,
              int(label_decision["player"]),
          )
          round_history_failures = _round_history_filter_failures(
              round_history_summary,
              args,
          )
          if round_history_failures:
            summary["skipped_round_history_filter"] += 1
            for failure in round_history_failures:
              summary[f"skipped_round_history_{failure['threshold']}"] += 1
            if prefilter_attempt_index is not None:
              repeat_validation_attempt_rows[prefilter_attempt_index][
                  "admission"
              ] = "rejected_round_history_filter"
            write_partial_snapshot(force=True)
            continue
          dense_meta["round_history_summary"] = round_history_summary
          dense_meta["round_history_filter_thresholds"] = (
              _round_history_filter_thresholds(args)
          )
          if bool(getattr(args, "record_round_history", False)):
            dense_meta["round_history"] = round_history
        outcome_paradox_values = [
            bool(outcome.get("any_paradox", False))
            for outcome in dense_meta.get("action_outcomes", [])
        ]
        has_action_paradox_contrast = bool(
            any(outcome_paradox_values) and not all(outcome_paradox_values)
        )
        dense_meta["has_action_paradox_contrast"] = has_action_paradox_contrast
        if bool(getattr(args, "require_action_paradox_contrast", False)):
          if not has_action_paradox_contrast:
            summary["skipped_no_action_paradox_contrast"] += 1
            continue
        if bool(getattr(args, "require_policy_target_changed", False)):
          if (
              prefilter_repeat_validation is None
              and not bool(dense_meta.get("policy_target_changed", False))
          ):
            summary["skipped_policy_target_unchanged"] += 1
            continue
        if bool(getattr(args, "repeat_validate_labels", False)):
          if prefilter_repeat_validation is not None:
            repeat_validation = prefilter_repeat_validation
          elif str(getattr(args, "repeat_validation_candidate_mode", "label_best")) == "label_best":
            repeat_validation = _repeat_validate_label_best(
                model,
                label_decision,
                value_scale,
                device,
                eval_args,
                int(dense_meta.get("label_best_action", -1)),
                seed_base=round_seed + int(label_decision["index"]) * 1009,
            )
          else:
            repeat_validation = _repeat_validate_candidate_queue(
                model,
                label_decision,
                value_scale,
                device,
                eval_args,
                dense_meta,
                led_profile,
                seed_base=round_seed + int(label_decision["index"]) * 1009,
            )
          dense_meta["repeat_validation"] = dict(repeat_validation)
          if (
              prefilter_attempt_index is None
              and bool(getattr(args, "record_repeat_validation_attempts", False))
          ):
            prefilter_attempt_index = len(repeat_validation_attempt_rows)
            repeat_validation_attempt_rows.append(
                _repeat_validation_attempt_meta(
                    label_decision,
                    led_profile,
                    repeat_validation,
                    match_idx=match_idx,
                    round_idx=round_idx,
                    round_seed=round_seed,
                    admission="repeat_validation_evaluated",
                )
            )
          if prefilter_repeat_validation is None:
            summary["repeat_validation_attempted"] += 1
          if bool(repeat_validation.get("passed", False)):
            passed_candidates = [
                item for item in repeat_validation.get("candidates", [])
                if bool(item.get("passed", False))
            ]
            if bool(args.repeat_validation_require_passed_red_switch) and not any(
                bool(item.get("candidate_is_red_switch", False))
                for item in passed_candidates
            ):
              summary["skipped_repeat_validation_no_passed_red_switch"] += 1
              if prefilter_attempt_index is not None:
                repeat_validation_attempt_rows[prefilter_attempt_index][
                    "admission"
                ] = "rejected_no_passed_red_switch"
              write_partial_snapshot(force=True)
              continue
            if bool(args.repeat_validation_require_passed_follow_led) and not any(
                bool(item.get("candidate_follows_led", False))
                for item in passed_candidates
            ):
              summary["skipped_repeat_validation_no_passed_follow_led"] += 1
              if prefilter_attempt_index is not None:
                repeat_validation_attempt_rows[prefilter_attempt_index][
                    "admission"
                ] = "rejected_no_passed_follow_led"
              write_partial_snapshot(force=True)
              continue
            summary["repeat_validation_passed"] += 1
            selected_candidate = int(
                repeat_validation.get(
                    "candidate_action", dense_meta.get("label_best_action", -1)
                )
            )
            selected_candidate_color = _action_color(
                label_decision["state"], selected_candidate
            )
            max_label_color_rows = int(
                getattr(args, "max_repeat_validation_label_color_rows", 0) or 0
            )
            if (
                max_label_color_rows > 0
                and repeat_validation_label_color_rows[selected_candidate_color]
                >= max_label_color_rows
            ):
              summary[
                  f"skipped_full_repeat_validation_label_color_"
                  f"{selected_candidate_color}"
              ] += 1
              if prefilter_attempt_index is not None:
                repeat_validation_attempt_rows[prefilter_attempt_index][
                    "admission"
                ] = f"rejected_full_label_color_{selected_candidate_color}"
              write_partial_snapshot(force=True)
              continue
            if selected_candidate != int(dense_meta.get("label_best_action", -1)):
              dense_row, dense_meta = _relabel_dense_row(
                  dense_row,
                  dense_meta,
                  label_decision,
                  selected_candidate,
              )
              _annotate_led_contrast_meta(dense_meta, label_decision, led_profile)
              summary["repeat_validation_relabelled_target"] += 1
            dense_row, dense_meta = _write_repeat_validation_action_targets(
                dense_row,
                dense_meta,
                repeat_validation,
                exclusive=bool(args.repeat_validation_exclusive_action_targets),
            )
            if isinstance(dense_meta.get("proposal"), dict):
              _append_repeat_validation_candidates_to_proposal(
                  dense_meta,
                  repeat_validation,
              )
              _annotate_proposal_outcome(dense_meta, dense_meta["proposal"])
            summary["repeat_validation_action_targets_written"] += 1
            summary["repeat_validation_exclusive_action_targets_written"] += int(
                bool(args.repeat_validation_exclusive_action_targets)
            )
            dense_meta["repeat_validation_selected_candidate_color"] = (
                selected_candidate_color
            )
            if prefilter_attempt_index is not None:
              repeat_validation_attempt_rows[prefilter_attempt_index][
                  "admission"
              ] = "accepted"
          else:
            summary[f"repeat_validation_failed_{repeat_validation.get('reason', 'unknown')}"] += 1
            summary["skipped_repeat_validation_failed"] += 1
            if prefilter_attempt_index is not None:
              repeat_validation_attempt_rows[prefilter_attempt_index][
                  "admission"
              ] = f"rejected_{repeat_validation.get('reason', 'unknown')}"
            write_partial_snapshot(force=True)
            continue
        if round_history is not None and round_history_summary is not None:
          _record_round_history_summary_counters(
              summary,
              round_history,
              round_history_summary,
          )
        rows.append(dense_row)
        metadata_rows.append(dense_meta)
        if bool(getattr(args, "led_contrast_only", False)):
          led_bucket_rows[selected_bucket] += 1
        summary["dense_label_rows"] += 1
        if bool(dense_meta.get("policy_target_changed", False)):
          summary["dense_policy_target_changed"] += 1
        if bool(getattr(args, "repeat_validate_labels", False)):
          selected_color = dense_meta.get("repeat_validation_selected_candidate_color")
          if selected_color is not None:
            repeat_validation_label_color_rows[str(selected_color)] += 1
        write_partial_snapshot()
        baseline = evaluated["baseline"]
        if bool(baseline.get("any_paradox", False)):
          summary["baseline_any_paradox"] += 1
        if bool(baseline.get("actor_paradox", False)):
          summary["baseline_actor_paradox"] += 1

  metadata = {
      "source": "quantum_cat_live_substitution_replay",
      "checkpoint": args.checkpoint,
      "players": int(args.players),
      "matches_requested": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "bot_mode": str(args.bot_mode),
      "decision_phases": str(args.decision_phases),
      "max_decisions": int(args.max_decisions),
      "max_accepted_rows": int(args.max_accepted_rows),
      "max_decisions_per_round": int(args.max_decisions_per_round),
      "led_contrast_only": bool(args.led_contrast_only),
      "require_red_switch_legal": bool(args.require_red_switch_legal),
      "max_led_bucket_rows": int(args.max_led_bucket_rows),
      "require_action_paradox_contrast": bool(args.require_action_paradox_contrast),
      "require_policy_target_changed": bool(args.require_policy_target_changed),
      "repeat_validate_labels": bool(args.repeat_validate_labels),
      "repeat_validation_candidate_mode": str(args.repeat_validation_candidate_mode),
      "repeat_validation_max_candidates": int(args.repeat_validation_max_candidates),
      "repeat_validation_rollouts": int(args.repeat_validation_rollouts),
      "repeat_validation_min_win_rate": float(args.repeat_validation_min_win_rate),
      "repeat_validation_min_nonworse_rate": float(
          args.repeat_validation_min_nonworse_rate
      ),
      "repeat_validation_min_value_delta": float(
          args.repeat_validation_min_value_delta
      ),
      "repeat_validation_max_risk_delta": float(
          args.repeat_validation_max_risk_delta
      ),
      "repeat_validation_max_candidate_any_paradox_rate": float(
          args.repeat_validation_max_candidate_any_paradox_rate
      ),
      "repeat_validation_min_base_any_paradox_rate": float(
          args.repeat_validation_min_base_any_paradox_rate
      ),
      "repeat_validation_require_passed_red_switch": bool(
          args.repeat_validation_require_passed_red_switch
      ),
      "repeat_validation_require_passed_follow_led": bool(
          args.repeat_validation_require_passed_follow_led
      ),
      "repeat_validation_prefilter_candidates": bool(
          args.repeat_validation_prefilter_candidates
      ),
      "record_repeat_validation_attempts": bool(
          args.record_repeat_validation_attempts
      ),
      "record_round_history": bool(args.record_round_history),
      "round_history_max_events": int(args.round_history_max_events),
      "min_round_history_avoidable_switch_events": int(
          args.min_round_history_avoidable_switch_events
      ),
      "min_round_history_avoidable_nonred_switch_events": int(
          args.min_round_history_avoidable_nonred_switch_events
      ),
      "min_round_history_avoidable_token_loss_events": int(
          args.min_round_history_avoidable_token_loss_events
      ),
      "repeat_validation_exclusive_action_targets": bool(
          args.repeat_validation_exclusive_action_targets
      ),
      "max_repeat_validation_label_color_rows": int(
          args.max_repeat_validation_label_color_rows
      ),
      "enable_behavior_exploration": bool(args.enable_behavior_exploration),
      "behavior_exploration_prob": float(args.behavior_exploration_prob),
      "behavior_exploration_temperature": float(
          args.behavior_exploration_temperature
      ),
      "behavior_led_contrast_switch_prob": float(
          args.behavior_led_contrast_switch_prob
      ),
      "behavior_red_switch_prob": float(args.behavior_red_switch_prob),
      "dense_target_temperature": float(args.dense_target_temperature),
      "dense_action_value_objective": str(args.dense_action_value_objective),
      "proposal_mode": str(args.proposal_mode),
      "label_base_action_source": str(args.label_base_action_source),
      "require_proposal": bool(args.require_proposal),
      "require_proposal_outcome": str(args.require_proposal_outcome),
      "source10_density_min_candidate_any_rate": float(
          args.source10_density_min_candidate_any_rate
      ),
      "source10_density_max_approved_rate": float(
          args.source10_density_max_approved_rate
      ),
      "source10_density_min_scored_candidates": int(
          args.source10_density_min_scored_candidates
      ),
      "pairwise_boundary_model": str(args.pairwise_boundary_model or ""),
      "pairwise_boundary_base_mode": str(args.pairwise_boundary_base_mode),
      "pairwise_boundary_threshold": float(args.pairwise_boundary_threshold),
      "pairwise_boundary_max_policy_log_gap": float(
          args.pairwise_boundary_max_policy_log_gap
      ),
      "pairwise_boundary_exclude_forced_led_lane_negatives": bool(
          args.pairwise_boundary_exclude_forced_led_lane_negatives
      ),
      "pairwise_boundary_led_lane_red_exception": str(
          args.pairwise_boundary_led_lane_red_exception
      ),
      "action_set_transformer_model": str(args.action_set_transformer_model or ""),
      "action_set_transformer_base_mode": str(args.action_set_transformer_base_mode),
      "action_set_transformer_phases": str(args.action_set_transformer_phases),
      "action_set_transformer_min_legal_count": int(
          args.action_set_transformer_min_legal_count
      ),
      "action_set_transformer_max_legal_count": int(
          args.action_set_transformer_max_legal_count
      ),
      "action_set_transformer_min_trick_number": int(
          args.action_set_transformer_min_trick_number
      ),
      "action_set_transformer_max_trick_number": int(
          args.action_set_transformer_max_trick_number
      ),
      "action_set_transformer_score_margin_threshold": float(
          args.action_set_transformer_score_margin_threshold
      ),
      "action_set_transformer_value_threshold": float(
          args.action_set_transformer_value_threshold
      ),
      "action_set_transformer_positive_threshold": float(
          args.action_set_transformer_positive_threshold
      ),
      "action_set_transformer_max_policy_log_gap": float(
          args.action_set_transformer_max_policy_log_gap
      ),
      "action_set_transformer_max_actions": int(args.action_set_transformer_max_actions),
      "action_set_transformer_top_policy": int(args.action_set_transformer_top_policy),
      "action_set_transformer_include_bots": str(
          args.action_set_transformer_include_bots
      ),
      "action_set_transformer_feature_candidates": bool(
          args.action_set_transformer_feature_candidates
      ),
      "action_set_transformer_exclude_forced_led_lane_negatives": bool(
          args.action_set_transformer_exclude_forced_led_lane_negatives
      ),
      "action_set_transformer_selection_value_bonus": float(
          args.action_set_transformer_selection_value_bonus
      ),
      "action_set_transformer_policy_tiebreak": float(
          args.action_set_transformer_policy_tiebreak
      ),
      "proposal_veto_model": str(args.proposal_veto_model or ""),
      "proposal_veto_base_mode": str(args.proposal_veto_base_mode),
      "proposal_veto_phases": str(args.proposal_veto_phases),
      "proposal_veto_min_legal_count": int(args.proposal_veto_min_legal_count),
      "proposal_veto_max_legal_count": int(args.proposal_veto_max_legal_count),
      "proposal_veto_min_trick_number": int(args.proposal_veto_min_trick_number),
      "proposal_veto_max_trick_number": int(args.proposal_veto_max_trick_number),
      "proposal_veto_max_policy_log_gap": float(args.proposal_veto_max_policy_log_gap),
      "proposal_veto_max_actions": int(args.proposal_veto_max_actions),
      "proposal_veto_top_policy": int(args.proposal_veto_top_policy),
      "proposal_veto_include_bots": str(args.proposal_veto_include_bots),
      "proposal_veto_candidate_mode": str(args.proposal_veto_candidate_mode),
      "proposal_veto_selection_mode": str(args.proposal_veto_selection_mode),
      "proposal_veto_feature_candidates": bool(args.proposal_veto_feature_candidates),
      "proposal_veto_exclude_forced_led_lane_negatives": bool(
          args.proposal_veto_exclude_forced_led_lane_negatives
      ),
      "proposal_veto_score_threshold": float(args.proposal_veto_score_threshold),
      "proposal_veto_margin_threshold": float(args.proposal_veto_margin_threshold),
      "proposal_veto_min_source_score": float(args.proposal_veto_min_source_score),
      "proposal_veto_min_source_margin": float(args.proposal_veto_min_source_margin),
      "proposal_veto_min_source_value_delta": float(
          args.proposal_veto_min_source_value_delta
      ),
      "proposal_veto_min_source_positive_prob": float(
          args.proposal_veto_min_source_positive_prob
      ),
      "summary": dict(summary),
      "led_bucket_rows": dict(led_bucket_rows),
      "repeat_validation_label_color_rows": dict(
          repeat_validation_label_color_rows
      ),
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_replay(rows, out_path, metadata=metadata)
  out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(
      json.dumps(
          {
              **metadata,
              "repeat_validation_attempt_rows": repeat_validation_attempt_rows,
              "rows": metadata_rows,
          },
          indent=2,
      )
      + "\n"
  )
  summary_out = Path(args.summary_out) if args.summary_out else out_path.with_suffix(
      out_path.suffix + ".summary.json"
  )
  summary_out.parent.mkdir(parents=True, exist_ok=True)
  summary_out.write_text(json.dumps({
      **metadata,
      "out": str(out_path),
      "rows": len(rows),
      "policy_target_changed_rows": int(summary["dense_policy_target_changed"]),
      "baseline_any_paradox_rows": int(summary["baseline_any_paradox"]),
      "baseline_actor_paradox_rows": int(summary["baseline_actor_paradox"]),
  }, indent=2) + "\n")
  print(json.dumps({
      "iteration": "live_substitution_done",
      "out": str(out_path),
      "rows": int(len(rows)),
      "summary": dict(summary),
  }, indent=2), flush=True)


if __name__ == "__main__":
  main()
