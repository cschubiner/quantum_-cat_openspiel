#!/usr/bin/env python3
"""Mine exact-pressure liveness-shield veto contrast rows directly.

This avoids the bulky two-step path:

  counterfactual behavior replay with row_stats -> shield_veto transform

The trajectory policy is a liveness shield, usually the Run653-style exact
trick-5+ shield.  For each eligible play state, the miner compares the shielded
action to a baseline action (q-policy by default).  When they differ, it emits a
single action-conditioned contrast row:

  shield action -> positive target
  baseline action -> negative target
  optional raw policy top action -> negative target

Exact simulator pressure is used only to choose the teacher at data-generation
time.  The stored observation/features remain information-legal tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from quantum_cat_alphazero_torch import (
    action_feature_matrix,
    make_game,
    model_policy_value,
    save_replay,
    terminal_paradox_target,
)
from quantum_cat_counterfactual_liquidity_replay import round_raw_scores
from quantum_cat_failure_miner import action_string
from quantum_cat_full_match_elo import (
    AZLivenessShieldPolicyBot,
    AZQPolicyBot,
    action_liveness_certificate,
    exact_legal_pressure_after_action,
    exact_table_liveness_after_action,
    load_neural,
    public_exit_liquidity_after_action,
)


PHASE_NAMES = ("chance", "discard", "prediction", "play", "terminal")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=20)
  parser.add_argument("--seed", type=int, default=20291001)
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument("--no-match-context", dest="match_context", action="store_false")
  parser.set_defaults(match_context=True)
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
  parser.add_argument(
      "--baseline-mode",
      choices=("q_policy", "raw_policy"),
      default="q_policy",
      help="Action treated as the vetoed baseline action.",
  )
  parser.add_argument(
      "--teacher-mode",
      choices=("liveness_shield", "liveness_key", "table_liveness"),
      default="liveness_shield",
      help=(
          "Teacher used for shield-veto rows. liveness_key directly selects "
          "the highest table-protective liveness certificate action. "
          "table_liveness uses hidden-hand exact table margins and is only "
          "valid for training-label generation."
      ),
  )
  parser.add_argument(
      "--include-raw-top-negative",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Also label the raw policy top legal action negative when not shielded.",
  )
  parser.add_argument(
      "--collect-raw-veto-when-base-agrees",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Collect rows when the shield equals the configured baseline but "
          "differs from the raw policy top action."
      ),
  )
  parser.add_argument(
      "--collect-all-eligible-dense-rows",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "When dense exact-pressure labels are enabled, emit rows for every "
          "eligible state matching --dense-row-filter, even when the shield "
          "does not veto the baseline. The policy target is the exact-pressure "
          "best legal action instead of the shield action."
      ),
  )
  parser.add_argument(
      "--collect-shield-veto-rows",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Keep the original shield-veto row collection path. Disable this "
          "when generating a pure broad dense replay controlled only by "
          "--dense-row-filter."
      ),
  )
  parser.add_argument(
      "--dense-row-filter",
      choices=("all", "led_follow_available", "nonred_led_follow_available"),
      default="all",
      help=(
          "Optional extra filter for --collect-all-eligible-dense-rows. "
          "The led-follow variants focus on states where following the led "
          "color is legal and switching away is also legal."
      ),
  )
  parser.add_argument(
      "--dense-policy-teacher",
      choices=("exact_pressure_best", "led_preservation", "table_liveness_best"),
      default="exact_pressure_best",
      help=(
          "Policy target for broad dense rows. led_preservation explicitly "
          "prefers preserving the led-color token, with responsible red trick "
          "wins as the first fallback."
      ),
  )
  parser.add_argument(
      "--dense-label-kind",
      choices=("exact_pressure", "table_liveness", "led_preservation"),
      default="exact_pressure",
      help=(
          "All-legal sparse action-value/risk label family used with "
          "--label-all-legal-exact-pressure. exact_pressure preserves the "
          "older own/current legal-pressure labels; table_liveness ranks all "
          "legal actions by exact table-wide liveness margins; "
          "led_preservation directly rewards following the led color, with "
          "red trick-winning switches as the main exception."
      ),
  )
  parser.add_argument("--positive-target", type=float, default=1.0)
  parser.add_argument("--negative-target", type=float, default=-1.5)
  parser.add_argument("--safe-risk-target", type=float, default=0.05)
  parser.add_argument("--risky-risk-target", type=float, default=0.95)
  parser.add_argument(
      "--label-all-legal-exact-pressure",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Label every non-paradox legal action by exact post-action pressure "
          "instead of only the shield/base contrast pair. The shield action "
          "remains the one-hot policy target."
      ),
  )
  parser.add_argument("--phases", default="play")
  parser.add_argument("--min-trick-number", type=int, default=5)
  parser.add_argument("--max-trick-number", type=int, default=-1)
  parser.add_argument(
      "--source-round-filter",
      choices=("all", "survived", "paradox"),
      default="all",
  )
  parser.add_argument("--max-rows", type=int, default=1000)
  parser.add_argument("--progress-interval", type=int, default=5)
  parser.add_argument(
      "--partial-save-interval",
      type=int,
      default=0,
      help=(
          "If positive, write usable partial replay artifacts every N completed "
          "matches after at least one row has been selected."
      ),
  )
  parser.add_argument("--sample-limit", type=int, default=12)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")

  # q-policy baseline/base knobs.
  parser.add_argument("--action-value-selection-weight", type=float, default=1.5)
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
  parser.add_argument("--action-value-rerank-clip", type=float, default=1.2)
  parser.add_argument("--action-value-rerank-phases", default="play")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.1)
  parser.add_argument("--action-value-max-policy-log-gap", type=float, default=2.0)

  # Liveness shield knobs.
  parser.add_argument(
      "--liveness-shield-base-mode",
      choices=("policy", "q_policy"),
      default="q_policy",
  )
  parser.add_argument("--liveness-shield-phases", default="play")
  parser.add_argument("--liveness-shield-min-trick-number", type=int, default=5)
  parser.add_argument("--liveness-shield-led-colors", default="")
  parser.add_argument("--liveness-shield-min-open-slot-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-min-public-damage-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument(
      "--liveness-shield-exact-legal-pressure",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--liveness-shield-exact-forced-only",
      action=argparse.BooleanOptionalAction,
      default=False,
  )
  parser.add_argument("--liveness-shield-sample-limit", type=int, default=0)
  return parser.parse_args()


def phase_name(state) -> str:
  phase = int(getattr(state, "_phase", -1))
  if 0 <= phase < len(PHASE_NAMES):
    return PHASE_NAMES[phase]
  return "unknown"


def legal_mask(num_actions: int, legal: list[int]) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=np.float32)
  mask[[int(action) for action in legal]] = 1.0
  return mask


def action_color(state, action: int) -> str | None:
  if int(action) == 999:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0))
  if num_card_types <= 0:
    return None
  color_idx = int(action) // num_card_types
  colors = ("R", "B", "Y", "G")
  if 0 <= color_idx < len(colors):
    return colors[color_idx]
  return None


def dense_row_filter_passes(
    state, legal: list[int], filter_name: str
) -> tuple[bool, dict]:
  led = getattr(state, "_led_color", None)
  colors = [action_color(state, action) for action in legal]
  has_led_follow = led is not None and any(color == led for color in colors)
  has_switch = led is not None and any(
      color is not None and color != led for color in colors
  )
  info = {
      "led_color": "None" if led is None else str(led),
      "has_led_follow": bool(has_led_follow),
      "has_switch": bool(has_switch),
  }
  if filter_name == "all":
    return True, info
  if filter_name == "led_follow_available":
    return bool(has_led_follow and has_switch), info
  if filter_name == "nonred_led_follow_available":
    return bool(led in ("B", "Y", "G") and has_led_follow and has_switch), info
  return False, info


def raw_policy_and_top(model, state, player: int, num_actions: int, model_args, device):
  policy, _ = model_policy_value(
      model,
      state,
      player,
      num_actions,
      model_args.value_scale,
      device,
  )
  legal = [int(action) for action in state.legal_actions(player)]
  top = int(max(legal, key=lambda action: (float(policy[int(action)]), -int(action))))
  return np.asarray(policy, dtype=np.float32), top


def pressure_key(row: dict, action: int) -> tuple[int, int, int, int, int]:
  return (
      -int(row.get("exact_forced_paradox_players_after", 0)),
      int(row.get("exact_min_nonparadox_legal_after", 0)),
      int(row.get("exact_current_nonparadox_legal_after", 0)),
      int(row.get("exact_total_nonparadox_legal_after", 0)),
      0 if bool(row.get("exact_terminal_after", False)) else 1,
      -int(action),
  )


def dense_exact_pressure_targets(state, legal: list[int], args: argparse.Namespace):
  actions = [int(action) for action in legal if int(action) != 999]
  if not actions:
    return {}, {}
  pressure = {
      action: exact_legal_pressure_after_action(state, action)
      for action in actions
  }
  ranked = sorted(actions, key=lambda action: pressure_key(pressure[action], action))
  if len(ranked) == 1:
    return {
        ranked[0]: float(args.positive_target)
    }, {
        ranked[0]: float(args.safe_risk_target)
    }
  value_targets = {}
  risk_targets = {}
  denom = float(len(ranked) - 1)
  low = float(args.negative_target)
  high = float(args.positive_target)
  risk_low = float(args.safe_risk_target)
  risk_high = float(args.risky_risk_target)
  for idx, action in enumerate(ranked):
    frac = float(idx) / denom
    value_targets[int(action)] = low + frac * (high - low)
    risk_targets[int(action)] = risk_high - frac * (risk_high - risk_low)
  return value_targets, risk_targets


def dense_table_liveness_targets(
    state,
    player: int,
    legal: list[int],
    baseline_action: int,
    args: argparse.Namespace,
):
  actions = [int(action) for action in legal if int(action) != 999]
  if not actions:
    return {}, {}
  ranked = sorted(
      actions,
      key=lambda action: table_liveness_teacher_key(
          state, player, action, baseline_action, args
      ),
  )
  if len(ranked) == 1:
    return {
        ranked[0]: float(args.positive_target)
    }, {
        ranked[0]: float(args.safe_risk_target)
    }
  value_targets = {}
  risk_targets = {}
  denom = float(len(ranked) - 1)
  low = float(args.negative_target)
  high = float(args.positive_target)
  risk_low = float(args.safe_risk_target)
  risk_high = float(args.risky_risk_target)
  for idx, action in enumerate(ranked):
    frac = float(idx) / denom
    value_targets[int(action)] = low + frac * (high - low)
    risk_targets[int(action)] = risk_high - frac * (risk_high - risk_low)
  return value_targets, risk_targets


def led_preservation_label_key(state, player: int, action: int) -> tuple:
  row = public_exit_liquidity_after_action(state, player, int(action))
  led = getattr(state, "_led_color", None)
  color = action_color(state, int(action))
  follows_led = bool(led is not None and color == led)
  lost_led = bool(row.get("lost_led_token", False))
  red_win_switch = bool(
      lost_led
      and color == "R"
      and row.get("would_win_now") is True
      and not bool(row.get("over_target_would_win", False))
  )
  red_switch = bool(lost_led and color == "R")
  if follows_led:
    lane_class = 4
  elif red_win_switch:
    lane_class = 3
  elif red_switch:
    lane_class = 2
  elif not lost_led:
    lane_class = 1
  else:
    lane_class = 0
  exact = exact_legal_pressure_after_action(state, int(action))
  return (
      lane_class,
      -int(exact.get("exact_forced_paradox_players_after", 0)),
      int(exact.get("exact_min_nonparadox_legal_after", 0)),
      int(exact.get("exact_current_nonparadox_legal_after", 0)),
      int(exact.get("exact_total_nonparadox_legal_after", 0)),
      0 if bool(exact.get("exact_terminal_after", False)) else 1,
      -int(action),
  )


def dense_led_preservation_targets(
    state,
    player: int,
    legal: list[int],
    args: argparse.Namespace,
):
  actions = [int(action) for action in legal if int(action) != 999]
  if not actions:
    return {}, {}
  ranked = sorted(
      actions,
      key=lambda action: led_preservation_label_key(state, player, action),
  )
  if len(ranked) == 1:
    return {
        ranked[0]: float(args.positive_target)
    }, {
        ranked[0]: float(args.safe_risk_target)
    }
  value_targets = {}
  risk_targets = {}
  denom = float(len(ranked) - 1)
  low = float(args.negative_target)
  high = float(args.positive_target)
  risk_low = float(args.safe_risk_target)
  risk_high = float(args.risky_risk_target)
  for idx, action in enumerate(ranked):
    frac = float(idx) / denom
    value_targets[int(action)] = low + frac * (high - low)
    risk_targets[int(action)] = risk_high - frac * (risk_high - risk_low)
  return value_targets, risk_targets


def dense_action_targets(
    state,
    player: int,
    legal: list[int],
    baseline_action: int,
    args: argparse.Namespace,
):
  if str(args.dense_label_kind) == "table_liveness":
    return dense_table_liveness_targets(state, player, legal, baseline_action, args)
  if str(args.dense_label_kind) == "led_preservation":
    return dense_led_preservation_targets(state, player, legal, args)
  return dense_exact_pressure_targets(state, legal, args)


def best_dense_pressure_action(
    state, legal: list[int], raw_policy: np.ndarray, args: argparse.Namespace
) -> int | None:
  _, dense_risks = dense_exact_pressure_targets(state, legal, args)
  if not dense_risks:
    return None
  best_risk = min(float(value) for value in dense_risks.values())
  best_actions = [
      int(action)
      for action, risk in dense_risks.items()
      if abs(float(risk) - best_risk) <= 1e-6
  ]
  return int(max(best_actions, key=lambda action: float(raw_policy[int(action)])))


def best_dense_table_liveness_action(
    state,
    player: int,
    legal: list[int],
    baseline_action: int,
    raw_policy: np.ndarray,
    args: argparse.Namespace,
) -> int | None:
  dense_values, _ = dense_table_liveness_targets(
      state, player, legal, baseline_action, args
  )
  if not dense_values:
    return None
  best_value = max(float(value) for value in dense_values.values())
  best_actions = [
      int(action)
      for action, value in dense_values.items()
      if abs(float(value) - best_value) <= 1e-6
  ]
  return int(max(best_actions, key=lambda action: float(raw_policy[int(action)])))


def led_preservation_key(state, player: int, action: int, raw_policy: np.ndarray):
  row = public_exit_liquidity_after_action(state, player, int(action))
  led = getattr(state, "_led_color", None)
  color = action_color(state, int(action))
  follows_led = bool(led is not None and color == led)
  lost_led = bool(row.get("lost_led_token", False))
  red_win_switch = bool(
      lost_led
      and color == "R"
      and row.get("would_win_now") is True
      and not bool(row.get("over_target_would_win", False))
  )
  if follows_led:
    lane_class = 3
  elif red_win_switch:
    lane_class = 2
  elif not lost_led:
    lane_class = 1
  else:
    lane_class = 0
  exact = exact_legal_pressure_after_action(state, int(action))
  return (
      lane_class,
      -int(exact.get("exact_forced_paradox_players_after", 0)),
      int(exact.get("exact_min_nonparadox_legal_after", 0)),
      int(exact.get("exact_current_nonparadox_legal_after", 0)),
      int(exact.get("exact_total_nonparadox_legal_after", 0)),
      float(raw_policy[int(action)]),
      -int(action),
  )


def best_led_preservation_action(
    state, player: int, legal: list[int], raw_policy: np.ndarray
) -> int | None:
  actions = [int(action) for action in legal if int(action) != 999]
  if not actions:
    return None
  return int(max(
      actions,
      key=lambda action: led_preservation_key(state, player, action, raw_policy),
  ))


def liveness_key_teacher_key(
    state,
    player: int,
    action: int,
    baseline_action: int,
    args: argparse.Namespace,
):
  row = action_liveness_certificate(
      state, player, int(action), base_action=int(baseline_action)
  )
  if bool(args.liveness_shield_exact_legal_pressure):
    exact = exact_legal_pressure_after_action(state, int(action))
    row.update(exact)
    return (
        1 if bool(exact.get("exact_terminal_after", False)) else 0,
        -int(exact.get("exact_forced_paradox_players_after", 0)),
        0 if bool(row.get("lost_led_token", False)) else 1,
        int(exact.get("exact_min_nonparadox_legal_after", 0)),
        int(exact.get("exact_current_nonparadox_legal_after", 0)),
        int(exact.get("exact_total_nonparadox_legal_after", 0)),
    ) + tuple(int(value) for value in row.get("liveness_key", []))
  return tuple(int(value) for value in row.get("liveness_key", []))


def table_liveness_teacher_key(
    state,
    player: int,
    action: int,
    baseline_action: int,
    args: argparse.Namespace,
):
  row = action_liveness_certificate(
      state, player, int(action), base_action=int(baseline_action)
  )
  exact = exact_legal_pressure_after_action(state, int(action))
  table = exact_table_liveness_after_action(state, int(action))
  row.update(exact)
  row.update(table)
  return (
      0 if bool(table.get("table_any_paradox_after", False)) else 1,
      -int(table.get("table_dead_margin_players_after", 0)),
      int(table.get("table_min_margin_after", -99)),
      int(table.get("table_current_margin_after", -99)),
      int(table.get("table_total_margin_after", -99)),
      1 if bool(table.get("table_terminal_after", False)) else 0,
      -int(exact.get("exact_forced_paradox_players_after", 0)),
      int(exact.get("exact_min_nonparadox_legal_after", 0)),
      int(exact.get("exact_current_nonparadox_legal_after", 0)),
      int(exact.get("exact_total_nonparadox_legal_after", 0)),
      0 if bool(row.get("lost_led_token", False)) else 1,
  ) + tuple(int(value) for value in row.get("liveness_key", []))


def best_liveness_key_action(
    state,
    player: int,
    legal: list[int],
    baseline_action: int,
    args: argparse.Namespace,
) -> int | None:
  actions = [int(action) for action in legal if int(action) != 999]
  if not actions:
    return None
  return int(max(
      actions,
      key=lambda action: liveness_key_teacher_key(
          state, player, action, baseline_action, args
      ),
  ))


def best_table_liveness_action(
    state,
    player: int,
    legal: list[int],
    baseline_action: int,
    args: argparse.Namespace,
) -> int | None:
  actions = [int(action) for action in legal if int(action) != 999]
  if not actions:
    return None
  return int(max(
      actions,
      key=lambda action: table_liveness_teacher_key(
          state, player, action, baseline_action, args
      ),
  ))


def teacher_action_tags(state, player: int, action: int) -> dict:
  row = public_exit_liquidity_after_action(state, player, int(action))
  led = getattr(state, "_led_color", None)
  color = action_color(state, int(action))
  follows_led = bool(led is not None and color == led)
  lost_led = bool(row.get("lost_led_token", False))
  return {
      "teacher_color": color,
      "teacher_follows_led": follows_led,
      "teacher_lost_led_token": lost_led,
      "teacher_red_win_switch": bool(
          lost_led
          and color == "R"
          and row.get("would_win_now") is True
          and not bool(row.get("over_target_would_win", False))
      ),
      "teacher_would_win_now": row.get("would_win_now"),
      "teacher_over_target_would_win": bool(row.get("over_target_would_win", False)),
  }


def eligible_state(state, allowed_phases: set[str], args: argparse.Namespace) -> bool:
  if phase_name(state) not in allowed_phases:
    return False
  if phase_name(state) == "play":
    trick = int(getattr(state, "_trick_number", -1))
    if trick < int(args.min_trick_number):
      return False
    if int(args.max_trick_number) >= 0 and trick > int(args.max_trick_number):
      return False
  return True


def build_contrast_row(
    state,
    player: int,
  raw_policy: np.ndarray,
  shield_action: int,
  baseline_action: int,
  negative_actions: list[int],
  args: argparse.Namespace,
):
  num_actions = int(state.num_distinct_actions())
  legal = [int(action) for action in state.legal_actions(player)]
  policy_target = np.zeros(num_actions, dtype=np.float32)
  policy_target[int(shield_action)] = 1.0

  action_values = np.zeros(num_actions, dtype=np.float32)
  action_mask = np.zeros(num_actions, dtype=np.float32)
  action_risks = np.zeros(num_actions, dtype=np.float32)
  action_risk_mask = np.zeros(num_actions, dtype=np.float32)

  action_values[int(shield_action)] = float(args.positive_target)
  action_mask[int(shield_action)] = 1.0
  action_risks[int(shield_action)] = float(args.safe_risk_target)
  action_risk_mask[int(shield_action)] = 1.0

  if bool(args.label_all_legal_exact_pressure):
    dense_values, dense_risks = dense_action_targets(
        state, player, legal, int(baseline_action), args
    )
    for action, target in dense_values.items():
      action_values[int(action)] = float(target)
      action_mask[int(action)] = 1.0
    for action, target in dense_risks.items():
      action_risks[int(action)] = float(target)
      action_risk_mask[int(action)] = 1.0
  else:
    for action in negative_actions:
      action = int(action)
      if action == int(shield_action):
        continue
      action_values[action] = float(args.negative_target)
      action_mask[action] = 1.0
      action_risks[action] = float(args.risky_risk_target)
      action_risk_mask[action] = 1.0

  return (
      np.asarray(state.observation_tensor(player), dtype=np.float32),
      legal_mask(num_actions, legal),
      policy_target,
      np.zeros(int(args.players), dtype=np.float32),
      np.zeros(int(args.players), dtype=np.float32),
      int(shield_action),
      int(player),
      action_risks,
      action_risk_mask,
      action_values,
      action_mask,
      action_feature_matrix(state, player, num_actions),
      None,
  )


def finalize_rows(rows: list[tuple], paradox_target: np.ndarray) -> list[tuple]:
  finalized = []
  for row in rows:
    finalized.append((
        row[0],
        row[1],
        row[2],
        row[3],
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
  return finalized


def main() -> None:
  args = parse_args()
  started = time.perf_counter()
  rng = np.random.default_rng(int(args.seed))
  torch.manual_seed(int(args.seed))
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model, model_args = load_neural(
      args.checkpoint, int(args.players), device, args=args
  )
  if model is None or model_args is None:
    raise ValueError(f"Could not load checkpoint {args.checkpoint}")
  if bool(args.collect_all_eligible_dense_rows) and not bool(
      args.label_all_legal_exact_pressure
  ):
    raise ValueError(
        "--collect-all-eligible-dense-rows requires "
        "--label-all-legal-exact-pressure"
    )

  shield_bot = AZLivenessShieldPolicyBot(
      model,
      "exact_veto_contrast_shield",
      device,
      model_args,
  )
  q_bot = AZQPolicyBot(model, "exact_veto_contrast_q", device, model_args)
  allowed_phases = {
      part.strip().lower()
      for part in str(args.phases).replace("|", ",").split(",")
      if part.strip()
  }

  all_rows = []
  row_group_ids = []
  samples = []
  counters = Counter()
  trick_counts = Counter()
  led_counts = Counter()
  negative_count_hist = Counter()
  match_totals = np.zeros(int(args.players), dtype=np.float32)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  checkpoint_sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()

  def build_metadata(completed_matches: int, *, partial: bool) -> dict:
    return {
        "kind": "exact_liveness_shield_veto_contrast_replay",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "players": int(args.players),
        "matches": int(args.matches),
        "completed_matches": int(completed_matches),
        "partial": bool(partial),
        "seed": int(args.seed),
        "match_context": bool(args.match_context),
        "rows": int(len(all_rows)),
        "row_group_ids": row_group_ids,
        "row_group_id_kind": "match_round",
        "row_group_count": int(len(set(row_group_ids))),
        "rounds": int(counters["rounds"]),
        "paradox_rounds": int(counters["paradox_rounds"]),
        "filtered_rows": int(counters["filtered_rows"]),
        "base_veto_rows": int(counters["base_veto_rows"]),
        "raw_only_veto_rows": int(counters["raw_only_veto_rows"]),
        "dense_eligible_rows": int(counters["dense_eligible_rows"]),
        "dense_best_differs_from_baseline": int(
            counters["dense_best_differs_from_baseline"]
        ),
        "dense_best_differs_from_raw_top": int(
            counters["dense_best_differs_from_raw_top"]
        ),
        "baseline_mode": str(args.baseline_mode),
        "teacher_mode": str(args.teacher_mode),
        "include_raw_top_negative": bool(args.include_raw_top_negative),
        "collect_raw_veto_when_base_agrees": bool(
            args.collect_raw_veto_when_base_agrees
        ),
        "collect_all_eligible_dense_rows": bool(
            args.collect_all_eligible_dense_rows
        ),
        "collect_shield_veto_rows": bool(args.collect_shield_veto_rows),
        "dense_row_filter": str(args.dense_row_filter),
        "dense_policy_teacher": str(args.dense_policy_teacher),
        "dense_label_kind": str(args.dense_label_kind),
        "teacher_follows_led_rows": int(counters["teacher_follows_led_rows"]),
        "teacher_lost_led_token_rows": int(
            counters["teacher_lost_led_token_rows"]
        ),
        "teacher_red_win_switch_rows": int(
            counters["teacher_red_win_switch_rows"]
        ),
        "positive_target": float(args.positive_target),
        "negative_target": float(args.negative_target),
        "safe_risk_target": float(args.safe_risk_target),
        "risky_risk_target": float(args.risky_risk_target),
        "label_all_legal_exact_pressure": bool(
            args.label_all_legal_exact_pressure
        ),
        "phases": str(args.phases),
        "min_trick_number": int(args.min_trick_number),
        "max_trick_number": int(args.max_trick_number),
        "source_round_filter": str(args.source_round_filter),
        "negative_action_count_hist": dict(sorted(negative_count_hist.items())),
        "trick_counts": dict(sorted(trick_counts.items())),
        "led_color_counts": dict(sorted(led_counts.items())),
        "action_value_selection_weight": float(args.action_value_selection_weight),
        "action_value_rerank_clip": float(args.action_value_rerank_clip),
        "action_value_rerank_min_margin": float(
            args.action_value_rerank_min_margin
        ),
        "action_value_max_policy_log_gap": float(
            args.action_value_max_policy_log_gap
        ),
        "liveness_shield_base_mode": str(args.liveness_shield_base_mode),
        "liveness_shield_phases": str(args.liveness_shield_phases),
        "liveness_shield_min_trick_number": int(
            args.liveness_shield_min_trick_number
        ),
        "liveness_shield_led_colors": str(args.liveness_shield_led_colors),
        "liveness_shield_exact_legal_pressure": bool(
            args.liveness_shield_exact_legal_pressure
        ),
        "liveness_shield_exact_forced_only": bool(
            args.liveness_shield_exact_forced_only
        ),
        "liveness_shield_min_open_slot_delta": int(
            args.liveness_shield_min_open_slot_delta
        ),
        "liveness_shield_min_public_damage_delta": int(
            args.liveness_shield_min_public_damage_delta
        ),
        "liveness_shield_max_policy_log_gap": float(
            args.liveness_shield_max_policy_log_gap
        ),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "samples": samples,
    }

  def write_outputs(path: Path, completed_matches: int, *, partial: bool) -> dict:
    metadata = build_metadata(completed_matches, partial=partial)
    save_replay(all_rows, path, metadata=metadata)
    summary_path = (
        Path(args.summary_out)
        if (not partial and args.summary_out)
        else path.with_suffix(path.suffix + ".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(text)
    return {"metadata": metadata, "summary_path": summary_path}

  for match_idx in range(int(args.matches)):
    for round_idx in range(int(args.players)):
      start_player = (
          int(args.initial_start) + int(match_idx) + int(round_idx)
      ) % int(args.players)
      game = make_game(args, start_player)
      state = game.new_initial_state()
      if bool(args.match_context):
        state.set_match_context(match_totals, round_idx)
      round_rows = []
      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(int(rng.choice(np.asarray(actions), p=probs)))
          continue
        player = int(state.current_player())
        legal = [int(action) for action in state.legal_actions(player)]
        raw_policy, raw_top = raw_policy_and_top(
            model, state, player, int(game.num_distinct_actions()), model_args,
            device
        )
        if str(args.baseline_mode) == "raw_policy":
          baseline_action = int(raw_top)
        else:
          baseline_action = int(q_bot.step(state.clone(), player))
          if baseline_action not in legal:
            baseline_action = int(raw_top)
        shield_action = int(shield_bot.step(state.clone(), player))
        if shield_action not in legal:
          shield_action = int(baseline_action)
        if str(args.teacher_mode) == "liveness_key":
          teacher_action = best_liveness_key_action(
              state, player, legal, baseline_action, args
          )
          if teacher_action is not None:
            shield_action = int(teacher_action)
        elif str(args.teacher_mode) == "table_liveness":
          teacher_action = best_table_liveness_action(
              state, player, legal, baseline_action, args
          )
          if teacher_action is not None:
            shield_action = int(teacher_action)

        if len(legal) >= 2 and eligible_state(state, allowed_phases, args):
          negative_actions = []
          collect_dense_row = False
          dense_filter_info = {}
          if bool(args.collect_all_eligible_dense_rows):
            collect_dense_row, dense_filter_info = dense_row_filter_passes(
                state, legal, str(args.dense_row_filter)
            )
          if bool(args.collect_shield_veto_rows) and shield_action != baseline_action:
            negative_actions.append(int(baseline_action))
            counters["base_veto_rows"] += 1
          elif (
              bool(args.collect_shield_veto_rows)
              and
              bool(args.collect_raw_veto_when_base_agrees)
              and shield_action != raw_top
          ):
            counters["raw_only_veto_rows"] += 1
            negative_actions.append(int(raw_top))
          if (
              bool(args.include_raw_top_negative)
              and int(raw_top) != int(shield_action)
              and negative_actions
              and int(raw_top) not in negative_actions
          ):
            negative_actions.append(int(raw_top))
          row_teacher_action = int(shield_action)
          collection_kind = "shield_veto"
          if collect_dense_row:
            if str(args.dense_policy_teacher) == "led_preservation":
              dense_best = best_led_preservation_action(
                  state, player, legal, raw_policy
              )
            elif str(args.dense_policy_teacher) == "table_liveness_best":
              dense_best = best_dense_table_liveness_action(
                  state, player, legal, baseline_action, raw_policy, args
              )
            else:
              dense_best = best_dense_pressure_action(
                  state, legal, raw_policy, args
              )
            if dense_best is None:
              collect_dense_row = False
            else:
              row_teacher_action = int(dense_best)
              collection_kind = "dense_eligible"
              counters["dense_eligible_rows"] += 1
              if row_teacher_action != int(baseline_action):
                counters["dense_best_differs_from_baseline"] += 1
              if row_teacher_action != int(raw_top):
                counters["dense_best_differs_from_raw_top"] += 1
              tags = teacher_action_tags(state, player, row_teacher_action)
              if bool(tags.get("teacher_follows_led", False)):
                counters["teacher_follows_led_rows"] += 1
              if bool(tags.get("teacher_lost_led_token", False)):
                counters["teacher_lost_led_token_rows"] += 1
              if bool(tags.get("teacher_red_win_switch", False)):
                counters["teacher_red_win_switch_rows"] += 1
              if not negative_actions:
                if int(baseline_action) != row_teacher_action:
                  negative_actions.append(int(baseline_action))
                elif int(raw_top) != row_teacher_action:
                  negative_actions.append(int(raw_top))
          if negative_actions or collect_dense_row:
            tags = teacher_action_tags(state, player, row_teacher_action)
            if bool(tags.get("teacher_follows_led", False)):
              counters["teacher_follows_led_rows"] += 1
            if bool(tags.get("teacher_lost_led_token", False)):
              counters["teacher_lost_led_token_rows"] += 1
            if bool(tags.get("teacher_red_win_switch", False)):
              counters["teacher_red_win_switch_rows"] += 1
            row = build_contrast_row(
                state,
                player,
                raw_policy,
                row_teacher_action,
                baseline_action,
                negative_actions,
                args,
            )
            round_rows.append(row)
            counters["rows"] += 1
            negative_count_hist[str(len(negative_actions))] += 1
            trick_counts[str(int(getattr(state, "_trick_number", -1)))] += 1
            led = getattr(state, "_led_color", None)
            led_counts["None" if led is None else str(led)] += 1
            if len(samples) < int(args.sample_limit):
              samples.append({
                  "collection_kind": collection_kind,
                  "match_idx": int(match_idx),
                  "round_idx": int(round_idx),
                  "player": int(player),
                  "phase": phase_name(state),
                  "trick_number": int(getattr(state, "_trick_number", -1)),
                  "led_color": "None" if led is None else str(led),
                  "dense_filter_info": dense_filter_info,
                  "baseline_action": int(baseline_action),
                  "baseline_action_string": action_string(
                      state, player, baseline_action
                  ),
                  "raw_top": int(raw_top),
                  "raw_top_string": action_string(state, player, raw_top),
                  "shield_action": int(shield_action),
                  "shield_action_string": action_string(
                      state, player, shield_action
                  ),
                  "teacher_action": int(row_teacher_action),
                  "teacher_action_string": action_string(
                      state, player, row_teacher_action
                  ),
                  "teacher_tags": tags,
                  "negative_actions": [int(action) for action in negative_actions],
              })
        state.apply_action(int(shield_action))

      paradox_target = terminal_paradox_target(state, int(args.players))
      any_paradox = bool(np.any(paradox_target > 0.5))
      counters["rounds"] += 1
      counters["paradox_rounds"] += int(any_paradox)
      source_filter = str(args.source_round_filter)
      keep_round = (
          source_filter == "all"
          or (source_filter == "survived" and not any_paradox)
          or (source_filter == "paradox" and any_paradox)
      )
      if keep_round:
        finalized = finalize_rows(round_rows, paradox_target)
        all_rows.extend(finalized)
        group_id = f"m{int(match_idx):06d}_r{int(round_idx):02d}"
        row_group_ids.extend([group_id] * len(finalized))
      else:
        counters["filtered_rows"] += len(round_rows)
      match_totals += round_raw_scores(state)
      if int(args.max_rows) > 0 and len(all_rows) >= int(args.max_rows):
        all_rows = all_rows[:int(args.max_rows)]
        row_group_ids = row_group_ids[:int(args.max_rows)]
        break
    if int(args.max_rows) > 0 and len(all_rows) >= int(args.max_rows):
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
    if (
        int(args.partial_save_interval) > 0
        and all_rows
        and (match_idx + 1) % int(args.partial_save_interval) == 0
    ):
      partial_path = out_path.with_name(
          f"{out_path.stem}.partial_m{int(match_idx + 1):06d}{out_path.suffix}"
      )
      written = write_outputs(partial_path, match_idx + 1, partial=True)
      print(json.dumps({
          "iteration": "partial_save",
          "completed_matches": int(match_idx + 1),
          "rows": int(len(all_rows)),
          "out": str(partial_path),
          "summary_out": str(written["summary_path"]),
          "elapsed_sec": round(time.perf_counter() - started, 3),
      }), flush=True)

  if not all_rows:
    raise SystemExit("No exact-veto rows selected; relax filters or run longer.")

  written = write_outputs(out_path, args.matches, partial=False)
  text = json.dumps(written["metadata"], indent=2, sort_keys=True) + "\n"
  print(text, end="")


if __name__ == "__main__":
  main()
