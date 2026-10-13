"""Mine late-play pre-forced states into dense survival policy replay."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import

from quantum_cat_alphazero_torch import (
    action_feature_matrix,
    model_policy_value,
    save_replay,
)
from quantum_cat_ai import make_bot
from quantum_cat_full_match_elo import AZLivenessShieldPolicyBot, load_neural
from quantum_cat_survival_grammar_replay import (
    safe_feature,
    softmax_target,
    structural_scores,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=50)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--seed", type=int, default=20266720)
  parser.add_argument(
      "--source-group-id",
      default="",
      help=(
          "Namespace every match/round validation group by collector source; "
          "defaults to player count and seed."
      ),
  )
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument("--no-match-context", dest="match_context", action="store_false")
  parser.set_defaults(match_context=True)
  parser.add_argument("--min-trick-number", type=int, default=5)
  parser.add_argument(
      "--max-trick-number",
      type=int,
      default=-1,
      help=(
          "Optional inclusive upper trick bound for kept play rows. Use this "
          "to mine early trajectory causes separately from late forced exits."
      ),
  )
  parser.add_argument("--max-rows", type=int, default=0)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--target-temperature", type=float, default=0.62)
  parser.add_argument("--behavior-temperature", type=float, default=0.0)
  parser.add_argument(
      "--behavior-policy-mode",
      choices=("policy", "liveness_shield"),
      default="policy",
      help=(
          "Policy that generates the source trajectory. liveness_shield uses "
          "the deployed public-information Run730 shield instead of the raw "
          "checkpoint policy."
      ),
  )
  parser.add_argument(
      "--anchor-policy-row-prob",
      type=float,
      default=0.0,
      help=(
          "Probability of adding a self-distillation row using the behavior "
          "policy when the state is not kept as a pre-forced target."
      ),
  )
  parser.add_argument(
      "--anchor-policy-phases",
      default="discard,prediction",
      help="Comma-separated phase names eligible for anchor-policy rows.",
  )
  parser.add_argument(
      "--row-filter",
      choices=(
          "immediate_contrast",
          "late_play",
          "all_play",
          "late_red_led_play",
          "liveness_contrast",
          "leader_liveness",
          "late_or_leader_liveness",
          "red_led_liveness",
          "late_red_led_liveness",
      ),
      default="immediate_contrast",
  )
  parser.add_argument(
      "--leader-min-trick-number",
      type=int,
      default=2,
      help=(
          "Minimum trick number for leader-state liveness rows. This is separate "
          "from --min-trick-number because led-color=None mistakes often happen "
          "before the late forced bottleneck."
      ),
  )
  parser.add_argument("--min-score-spread", type=float, default=0.20)
  parser.add_argument("--min-liveness-survival-spread", type=float, default=0.15)
  parser.add_argument("--min-liveness-legal-spread", type=float, default=0.75)
  parser.add_argument("--min-liveness-terminal-spread", type=float, default=0.15)
  parser.add_argument(
      "--require-liveness-safe-action",
      action="store_true",
      help=(
          "For liveness row filters, keep only rows where screened rollouts "
          "contain both a safe action and a risky/forced action. This avoids "
          "training mostly on all-bad late states."
      ),
  )
  parser.add_argument(
      "--require-liveness-terminal-safe-action",
      action="store_true",
      help=(
          "For liveness row filters, count an action as safe only if the "
          "screened rollouts finish the hand without paradox. This is stricter "
          "than merely surviving until the rollout cap."
      ),
  )
  parser.add_argument(
      "--liveness-mask-unsafe-targets",
      action="store_true",
      help=(
          "When any evaluated legal action is liveness-safe, force unsafe "
          "evaluated actions far below the target distribution. This makes "
          "terminal no-paradox survival lexicographic, then lets liquidity "
          "terms rank the surviving actions."
      ),
  )
  parser.add_argument(
      "--require-target-liveness-evaluated",
      action="store_true",
      help=(
          "For liveness row filters, keep only rows whose final top target "
          "action was actually evaluated by the liveness rollout labeler."
      ),
  )
  parser.add_argument(
      "--require-target-liveness-terminal-safe",
      action="store_true",
      help=(
          "For liveness row filters, keep only rows whose final top target "
          "action was evaluated and reached terminal no-paradox at least at "
          "--liveness-safe-terminal-threshold."
      ),
  )
  parser.add_argument(
      "--liveness-unsafe-target-penalty",
      type=float,
      default=1000.0,
      help="Score gap used by --liveness-mask-unsafe-targets.",
  )
  parser.add_argument("--liveness-safe-forced-threshold", type=float, default=0.1)
  parser.add_argument("--liveness-risky-forced-threshold", type=float, default=0.1)
  parser.add_argument(
      "--liveness-safe-terminal-threshold",
      type=float,
      default=0.999,
      help="Minimum terminal-no-paradox rate for strict terminal-safe actions.",
  )
  parser.add_argument("--next-legal-weight", type=float, default=1.15)
  parser.add_argument("--next-forced-penalty", type=float, default=6.0)
  parser.add_argument("--next-zero-penalty", type=float, default=10.0)
  parser.add_argument("--immediate-paradox-penalty", type=float, default=14.0)
  parser.add_argument("--terminal-after-penalty", type=float, default=1.0)
  parser.add_argument("--prior-weight", type=float, default=0.02)
  parser.add_argument("--action-value-weight", type=float, default=0.0)
  parser.add_argument("--follow-led-bonus", type=float, default=1.6)
  parser.add_argument("--token-loss-penalty", type=float, default=5.5)
  parser.add_argument("--min-lane-surplus-weight", type=float, default=1.25)
  parser.add_argument("--lane-damage-penalty", type=float, default=1.10)
  parser.add_argument("--lane-pressure-penalty", type=float, default=0.45)
  parser.add_argument("--own-min-colors-weight", type=float, default=0.85)
  parser.add_argument("--own-zero-exit-penalty", type=float, default=2.6)
  parser.add_argument("--own-one-exit-penalty", type=float, default=1.5)
  parser.add_argument("--own-two-less-exit-penalty", type=float, default=0.65)
  parser.add_argument("--future-no-exit-penalty", type=float, default=2.0)
  parser.add_argument("--forced-pressure-penalty", type=float, default=0.90)
  parser.add_argument("--paradox-penalty", type=float, default=8.0)
  parser.add_argument("--discard-relief-weight", type=float, default=1.0)
  parser.add_argument("--discard-no-exit-penalty", type=float, default=1.5)
  parser.add_argument("--prediction-abs-gap-penalty", type=float, default=0.35)
  parser.add_argument("--prediction-over-penalty", type=float, default=0.15)
  parser.add_argument(
      "--liveness-rollouts-per-action",
      type=int,
      default=0,
      help=(
          "If positive, clone each legal action and roll forward this many "
          "short continuations to label multi-step no-paradox/liquidity risk."
      ),
  )
  parser.add_argument("--liveness-max-plies", type=int, default=18)
  parser.add_argument(
      "--liveness-max-actions-per-state",
      type=int,
      default=0,
      help=(
          "If positive, estimate multi-step liveness for only this many legal "
          "actions per state. The selected set keeps forced-looking actions, "
          "high structural-score actions, high policy-prior actions, and the "
          "behavior action."
      ),
  )
  parser.add_argument("--liveness-top-policy-actions", type=int, default=2)
  parser.add_argument("--liveness-include-forced-actions", action="store_true")
  parser.add_argument("--liveness-temperature", type=float, default=0.35)
  parser.add_argument(
      "--liveness-continuation",
      choices=(
          "policy",
          "liveness_shield",
          "structural",
          "hybrid",
          "structural_fast",
          "heuristic",
          "heuristic_safe8",
          "heuristic_safe14",
      ),
      default="hybrid",
      help="Continuation policy used inside short liveness rollouts.",
  )
  parser.add_argument("--liveness-structural-weight", type=float, default=0.65)
  parser.add_argument(
      "--liveness-common-random-numbers",
      action="store_true",
      help=(
          "Evaluate every candidate action from the same state with identical "
          "continuation random seeds, so apparent action differences cannot "
          "come only from independently lucky rollout draws."
      ),
  )
  parser.add_argument(
      "--liveness-repeat-passes",
      type=int,
      default=1,
      help=(
          "Independent common-random rollout passes per action. Each pass "
          "uses --liveness-rollouts-per-action continuations."
      ),
  )
  parser.add_argument(
      "--liveness-min-repeat-agreement",
      type=float,
      default=0.0,
      help=(
          "For liveness-filtered rows, require the action with the best "
          "aggregate terminal survival to remain tied for best in at least "
          "this fraction of independently seeded repeat passes."
      ),
  )
  parser.add_argument("--liveness-survival-bonus", type=float, default=18.0)
  parser.add_argument("--liveness-min-legal-weight", type=float, default=1.35)
  parser.add_argument("--liveness-forced-penalty", type=float, default=4.0)
  parser.add_argument("--liveness-zero-penalty", type=float, default=8.0)
  parser.add_argument("--liveness-paradox-penalty", type=float, default=18.0)
  parser.add_argument("--liveness-terminal-bonus", type=float, default=1.5)
  parser.add_argument(
      "--liveness-incomplete-penalty",
      type=float,
      default=0.0,
      help=(
          "Penalty per rollout that survives only until the ply cap without "
          "finishing the hand. Use this for terminal-liveness curriculum."
      ),
  )
  parser.add_argument(
      "--emit-liveness-action-labels",
      action="store_true",
      help=(
          "Store evaluated liveness rollout outcomes as action-risk and "
          "action-value labels. Risk is 1 - terminal-no-paradox rate; value is "
          "2 * terminal-no-paradox rate - 1. Unevaluated actions stay masked."
      ),
  )
  parser.add_argument("--progress-interval", type=int, default=0)
  parser.add_argument(
      "--partial-save-interval",
      type=int,
      default=0,
      help=(
          "If positive, write a valid partial replay and summary every N "
          "completed matches. Partial files are siblings of --out and named "
          "*.partial_mNNNNNN.npz."
      ),
  )
  return parser.parse_args()


def make_game(players: int, start_player: int, match_context: bool):
  return pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": int(players),
          "start_player": int(start_player),
          "match_context": int(bool(match_context)),
      },
  )


def state_phase_name(state) -> str:
  return {
      0: "chance",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "terminal",
  }.get(int(getattr(state, "_phase", -1)), "unknown")


def split_csv(value: str) -> set[str]:
  return {item.strip().lower() for item in str(value).split(",") if item.strip()}


def legal_mask(num_actions: int, legal: list[int]) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=np.float32)
  for action in legal:
    if 0 <= int(action) < num_actions:
      mask[int(action)] = 1.0
  return mask


def normalized_policy(policy: np.ndarray, legal: list[int]) -> np.ndarray:
  result = np.zeros_like(policy, dtype=np.float32)
  legal_indices = np.array([int(action) for action in legal], dtype=np.int64)
  if legal_indices.size == 0:
    return result
  values = np.maximum(np.asarray(policy, dtype=np.float32)[legal_indices], 0.0)
  total = float(np.sum(values))
  if not math.isfinite(total) or total <= 0.0:
    values = np.ones_like(values, dtype=np.float32)
    total = float(np.sum(values))
  result[legal_indices] = values / total
  return result


def uniform_policy(num_actions: int, legal: list[int]) -> np.ndarray:
  result = np.zeros(int(num_actions), dtype=np.float32)
  if not legal:
    return result
  value = 1.0 / float(len(legal))
  for action in legal:
    result[int(action)] = value
  return result


def sample_action(
    policy: np.ndarray, legal: list[int], temperature: float, rng: np.random.Generator
) -> int:
  legal_indices = np.array([int(action) for action in legal], dtype=np.int64)
  if legal_indices.size == 1:
    return int(legal_indices[0])
  probs = np.asarray(policy, dtype=np.float32)[legal_indices]
  if float(temperature) <= 0.0:
    return int(legal_indices[int(np.argmax(probs))])
  logits = np.log(np.maximum(probs, 1e-12)) / max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  weights = np.exp(logits)
  total = float(np.sum(weights))
  if not math.isfinite(total) or total <= 0.0:
    weights = np.ones_like(weights, dtype=np.float32)
    total = float(np.sum(weights))
  return int(rng.choice(legal_indices, p=weights / total))


def liveness_shield_bot(model, model_args, device, args):
  """Return the exact deployed public-information 4p policy configuration."""
  cached = getattr(args, "_collector_liveness_shield_bot", None)
  if cached is not None:
    return cached
  model_args.liveness_shield_base_mode = "policy"
  model_args.liveness_shield_phases = "play"
  model_args.liveness_shield_min_open_slot_delta = 0
  model_args.liveness_shield_min_public_damage_delta = 0
  model_args.liveness_shield_max_policy_log_gap = -1.0
  bot = AZLivenessShieldPolicyBot(
      model, "current_public_liveness", device, model_args
  )
  args._collector_liveness_shield_bot = bot
  return bot


def behavior_action_for_state(
    state,
    player,
    legal,
    policy,
    model,
    model_args,
    device,
    args,
    rng,
):
  if str(getattr(args, "behavior_policy_mode", "policy")) == "liveness_shield":
    action = int(
        liveness_shield_bot(model, model_args, device, args).step(state, player)
    )
    if action not in legal:
      raise ValueError(f"liveness shield selected illegal action {action}")
    return action
  return sample_action(policy, legal, args.behavior_temperature, rng)


def continuation_policy(
    model,
    model_args,
    device,
    state,
    player: int,
    legal: list[int],
    args: argparse.Namespace,
) -> np.ndarray:
  mode = str(args.liveness_continuation)
  if mode == "liveness_shield":
    result = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    action = int(
        liveness_shield_bot(model, model_args, device, args).step(state, player)
    )
    if action not in legal:
      raise ValueError(f"liveness shield selected illegal action {action}")
    result[action] = 1.0
    return result
  if mode.startswith("heuristic"):
    result = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    bot = make_bot(mode, seed=int(args.seed))
    action = int(bot.step(state, player))
    if action not in legal:
      action = int(legal[0])
    result[action] = 1.0
    return result
  if mode == "structural_fast":
    policy = uniform_policy(state.num_distinct_actions(), legal)
    features = action_feature_matrix(state, player, state.num_distinct_actions())
    structural = structural_scores(features, policy, None, None, args)
    legal_array = legal_mask(state.num_distinct_actions(), legal) > 0.5
    return softmax_target(
        structural, legal_array, max(1e-6, float(args.liveness_temperature))
    )
  raw_policy, _ = model_policy_value(
      model,
      state,
      player,
      state.num_distinct_actions(),
      model_args.value_scale,
      device,
  )
  policy = normalized_policy(raw_policy, legal)
  if mode == "policy":
    return policy
  features = action_feature_matrix(state, player, state.num_distinct_actions())
  structural = structural_scores(features, policy, None, None, args)
  legal_array = legal_mask(state.num_distinct_actions(), legal) > 0.5
  structural_target = softmax_target(
      structural, legal_array, max(1e-6, float(args.liveness_temperature))
  )
  if mode == "structural":
    return structural_target
  if mode == "hybrid":
    legal_indices = np.array([int(action) for action in legal], dtype=np.int64)
    result = np.zeros_like(policy, dtype=np.float32)
    logits = np.log(np.maximum(policy[legal_indices], 1e-12))
    logits += float(args.liveness_structural_weight) * np.log(
        np.maximum(structural_target[legal_indices], 1e-12)
    )
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
      weights = np.ones_like(weights, dtype=np.float32)
      total = float(np.sum(weights))
    result[legal_indices] = weights / total
    return result
  raise ValueError(f"unknown liveness continuation {mode!r}")


def roll_chance_nodes(state, rng: np.random.Generator) -> None:
  while state.is_chance_node() and not state.is_terminal():
    outcomes = state.chance_outcomes()
    actions, probs = zip(*outcomes)
    state.apply_action(int(rng.choice(actions, p=probs)))


def estimate_action_liveness(
    state,
    action: int,
    model,
    model_args,
    device,
    args: argparse.Namespace,
    rng: np.random.Generator,
    rollout_seeds: list[int] | None = None,
) -> dict:
  rollouts_per_pass = max(0, int(args.liveness_rollouts_per_action))
  repeat_passes = max(1, int(getattr(args, "liveness_repeat_passes", 1)))
  rollouts = rollouts_per_pass * repeat_passes
  if rollouts <= 0:
    return {}
  if rollout_seeds is not None and len(rollout_seeds) != rollouts:
    raise ValueError(
        f"expected {rollouts} matched rollout seeds, got {len(rollout_seeds)}"
    )
  root_paradox = any(bool(value) for value in getattr(state, "_has_paradoxed", []))
  survived = 0
  terminal_no_paradox = 0
  min_legal_counts = []
  forced_counts = []
  zero_counts = []
  decision_steps = []
  plies_to_paradox = []
  terminal_outcomes = []
  for rollout_idx in range(rollouts):
    rollout_rng = (
        rng if rollout_seeds is None
        else np.random.default_rng(int(rollout_seeds[rollout_idx]))
    )
    shadow = state.clone()
    shadow.apply_action(int(action))
    min_legal = 999
    forced = 0
    zero = 0
    decisions = 0
    paradoxed = any(bool(value) for value in getattr(shadow, "_has_paradoxed", []))
    roll_chance_nodes(shadow, rollout_rng)
    while (
        not shadow.is_terminal()
        and decisions < int(args.liveness_max_plies)
        and not paradoxed
    ):
      if shadow.is_chance_node():
        roll_chance_nodes(shadow, rollout_rng)
        continue
      player = int(shadow.current_player())
      legal = [int(legal_action) for legal_action in shadow.legal_actions(player)]
      legal_count = int(len(legal))
      min_legal = min(min_legal, legal_count)
      if legal_count <= 1:
        forced += 1
      if legal_count <= 0:
        zero += 1
        break
      policy = continuation_policy(
          model, model_args, device, shadow, player, legal, args
      )
      next_action = sample_action(
          policy, legal, float(args.liveness_temperature), rollout_rng
      )
      shadow.apply_action(next_action)
      decisions += 1
      paradoxed = (
          any(bool(value) for value in getattr(shadow, "_has_paradoxed", []))
          and not root_paradox
      )
      roll_chance_nodes(shadow, rollout_rng)
    if min_legal == 999:
      min_legal = 0 if not shadow.is_terminal() else int(args.players)
    no_new_paradox = not (
        any(bool(value) for value in getattr(shadow, "_has_paradoxed", []))
        and not root_paradox
    )
    if no_new_paradox:
      survived += 1
    elif paradoxed:
      plies_to_paradox.append(decisions)
    if shadow.is_terminal() and no_new_paradox:
      terminal_no_paradox += 1
      terminal_outcomes.append(1.0)
    else:
      terminal_outcomes.append(0.0)
    min_legal_counts.append(float(min_legal))
    forced_counts.append(float(forced))
    zero_counts.append(float(zero))
    decision_steps.append(float(decisions))
  survival_rate = float(survived) / float(rollouts)
  terminal_rate = float(terminal_no_paradox) / float(rollouts)
  mean_min_legal = float(np.mean(min_legal_counts)) if min_legal_counts else 0.0
  mean_forced = float(np.mean(forced_counts)) if forced_counts else 0.0
  mean_zero = float(np.mean(zero_counts)) if zero_counts else 0.0
  mean_steps = float(np.mean(decision_steps)) if decision_steps else 0.0
  mean_plies_to_paradox = (
      float(np.mean(plies_to_paradox)) if plies_to_paradox else None
  )
  score = 0.0
  incomplete_rate = max(0.0, survival_rate - terminal_rate)
  score += float(args.liveness_survival_bonus) * survival_rate
  score += float(args.liveness_min_legal_weight) * math.log1p(max(0.0, mean_min_legal))
  score -= float(args.liveness_forced_penalty) * mean_forced
  score -= float(args.liveness_zero_penalty) * mean_zero
  score -= float(args.liveness_paradox_penalty) * (1.0 - survival_rate)
  score += float(args.liveness_terminal_bonus) * terminal_rate
  score -= float(args.liveness_incomplete_penalty) * incomplete_rate
  repeat_terminal_rates = [
      float(np.mean(
          terminal_outcomes[
              pass_idx * rollouts_per_pass:(pass_idx + 1) * rollouts_per_pass
          ]
      ))
      for pass_idx in range(repeat_passes)
  ]
  return {
      "liveness_evaluated": True,
      "liveness_rollouts": int(rollouts),
      "liveness_survival_rate": survival_rate,
      "liveness_paradox_rate": 1.0 - survival_rate,
      "liveness_terminal_no_paradox_rate": terminal_rate,
      "liveness_incomplete_no_paradox_rate": incomplete_rate,
      "liveness_mean_min_legal_count": mean_min_legal,
      "liveness_mean_forced_count": mean_forced,
      "liveness_mean_zero_count": mean_zero,
      "liveness_mean_decision_steps": mean_steps,
      "liveness_mean_plies_to_paradox": mean_plies_to_paradox,
      "liveness_score": float(score),
      "liveness_repeat_passes": repeat_passes,
      "liveness_repeat_terminal_rates": repeat_terminal_rates,
  }


def empty_liveness() -> dict:
  return {
      "liveness_evaluated": False,
      "liveness_rollouts": 0,
      "liveness_survival_rate": 1.0,
      "liveness_paradox_rate": 0.0,
      "liveness_terminal_no_paradox_rate": 0.0,
      "liveness_incomplete_no_paradox_rate": 0.0,
      "liveness_mean_min_legal_count": 0.0,
      "liveness_mean_forced_count": 0.0,
      "liveness_mean_zero_count": 0.0,
      "liveness_mean_decision_steps": 0.0,
      "liveness_mean_plies_to_paradox": None,
      "liveness_score": 0.0,
      "liveness_repeat_passes": 0,
      "liveness_repeat_terminal_rates": [],
  }


def liveness_row_is_safe(row: dict, args: argparse.Namespace) -> bool:
  if not bool(row.get("liveness_evaluated", False)):
    return False
  if float(row.get("liveness_survival_rate", 1.0)) < 0.999:
    return False
  if (
      float(row.get("liveness_mean_forced_count", 0.0)) >
      float(args.liveness_safe_forced_threshold)
  ):
    return False
  if bool(args.require_liveness_terminal_safe_action) and (
      float(row.get("liveness_terminal_no_paradox_rate", 0.0)) <
      float(args.liveness_safe_terminal_threshold)
  ):
    return False
  return True


def liveness_row_is_risky(row: dict, args: argparse.Namespace) -> bool:
  if not bool(row.get("liveness_evaluated", False)):
    return False
  if (
      float(row.get("liveness_survival_rate", 1.0)) <
      max(0.0, 1.0 - float(args.min_liveness_survival_spread))
  ):
    return True
  if (
      float(row.get("liveness_mean_forced_count", 0.0)) >
      float(args.liveness_risky_forced_threshold)
  ):
    return True
  if bool(args.require_liveness_terminal_safe_action) and (
      float(row.get("liveness_terminal_no_paradox_rate", 0.0)) <
      float(args.liveness_safe_terminal_threshold)
  ):
    return True
  return False


def liveness_action_label_arrays(
    num_actions: int,
    next_rows: list[dict],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
  """Build sparse trainable action labels from evaluated liveness rows."""
  action_risk_targets = np.zeros(int(num_actions), dtype=np.float32)
  action_risk_mask = np.zeros(int(num_actions), dtype=np.float32)
  action_value_targets = np.zeros(int(num_actions), dtype=np.float32)
  action_value_mask = np.zeros(int(num_actions), dtype=np.float32)
  for row in next_rows:
    if not bool(row.get("liveness_evaluated", False)):
      continue
    action = int(row["action"])
    if action < 0 or action >= int(num_actions):
      continue
    terminal_safe = float(row.get("liveness_terminal_no_paradox_rate", 0.0))
    terminal_safe = min(1.0, max(0.0, terminal_safe))
    action_risk_targets[action] = float(1.0 - terminal_safe)
    action_risk_mask[action] = 1.0
    action_value_targets[action] = float(2.0 * terminal_safe - 1.0)
    action_value_mask[action] = 1.0
  if float(np.sum(action_value_mask)) <= 0.0:
    return None, None, None, None
  return (
      action_risk_targets,
      action_risk_mask,
      action_value_targets,
      action_value_mask,
  )


def select_liveness_actions(
    legal: list[int],
    scores: np.ndarray,
    policy: np.ndarray,
    next_rows: list[dict],
    behavior_action: int | None,
    args: argparse.Namespace,
) -> set[int]:
  if int(args.liveness_rollouts_per_action) <= 0:
    return set()
  if int(args.liveness_max_actions_per_state) <= 0:
    return {int(action) for action in legal}
  limit = max(1, int(args.liveness_max_actions_per_state))
  selected: list[int] = []

  def add(action: int) -> None:
    action = int(action)
    if action in legal and action not in selected:
      selected.append(action)

  if behavior_action is not None:
    add(int(behavior_action))
  if bool(args.liveness_include_forced_actions):
    forced = [
        int(row["action"]) for row in next_rows
        if int(row["next_legal_count"]) <= 1
    ]
    forced.sort(key=lambda action: float(policy[int(action)]), reverse=True)
    for action in forced:
      add(action)
      if len(selected) >= limit:
        return set(selected)
  policy_top = sorted(
      [int(action) for action in legal],
      key=lambda action: float(policy[int(action)]),
      reverse=True,
  )
  for action in policy_top[:max(0, int(args.liveness_top_policy_actions))]:
    add(action)
    if len(selected) >= limit:
      return set(selected)
  score_top = sorted(
      [int(action) for action in legal],
      key=lambda action: float(scores[int(action)]),
      reverse=True,
  )
  for action in score_top:
    add(action)
    if len(selected) >= limit:
      return set(selected)
  return set(selected)


def next_decision_context(state) -> dict:
  if state.is_terminal():
    return {
        "terminal": True,
        "phase": "terminal",
        "player": -1,
        "legal_count": 0,
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "led_color": getattr(state, "_led_color", None),
    }
  if state.is_chance_node():
    return {
        "terminal": False,
        "phase": "chance",
        "player": -1,
        "legal_count": 0,
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "led_color": getattr(state, "_led_color", None),
    }
  player = int(state.current_player())
  legal = state.legal_actions(player)
  return {
      "terminal": False,
      "phase": state_phase_name(state),
      "player": player,
      "legal_count": int(len(legal)),
      "trick_number": int(getattr(state, "_trick_number", -1)),
      "led_color": getattr(state, "_led_color", None),
  }


def score_candidate_actions(
    state,
    player: int,
    legal: list[int],
    policy: np.ndarray,
    features: np.ndarray,
    model,
    model_args,
    device,
    args: argparse.Namespace,
    rng: np.random.Generator,
    behavior_action: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
  before_paradox = [
      bool(value) for value in getattr(state, "_has_paradoxed", [])
  ]
  scores = structural_scores(features, policy, None, None, args)
  next_rows = []
  for action in legal:
    action = int(action)
    shadow = state.clone()
    shadow.apply_action(action)
    after_paradox = [
        bool(value) for value in getattr(shadow, "_has_paradoxed", [])
    ]
    newly_paradoxed = any(after_paradox) and not any(before_paradox)
    context = next_decision_context(shadow)
    next_legal = int(context["legal_count"])
    immediate = float(args.next_legal_weight) * math.log1p(max(0, next_legal))
    if next_legal <= 1 and not bool(context["terminal"]):
      immediate -= float(args.next_forced_penalty)
    if next_legal <= 0 and not bool(context["terminal"]):
      immediate -= float(args.next_zero_penalty)
    if bool(context["terminal"]):
      immediate -= float(args.terminal_after_penalty)
    if newly_paradoxed:
      immediate -= float(args.immediate_paradox_penalty)
    scores[action] += immediate
    next_rows.append({
        "action": action,
        "next_legal_count": next_legal,
        "next_phase": context["phase"],
        "next_player": int(context["player"]),
        "next_trick_number": int(context["trick_number"]),
        "next_led_color": context["led_color"],
        "newly_paradoxed": bool(newly_paradoxed),
        "immediate_score": float(immediate),
        "token_loss": bool(safe_feature(
            features, "token_loss_newly_loses_led"
        )[action] > 0.5),
        **empty_liveness(),
    })
  liveness_actions = select_liveness_actions(
      legal, scores, policy, next_rows, behavior_action, args
  )
  if liveness_actions:
    matched_rollout_seeds = None
    if bool(getattr(args, "liveness_common_random_numbers", False)):
      rollout_count = (
          max(0, int(args.liveness_rollouts_per_action))
          * max(1, int(getattr(args, "liveness_repeat_passes", 1)))
      )
      matched_rollout_seeds = [
          int(value)
          for value in rng.integers(
              0, np.iinfo(np.int64).max, size=rollout_count, dtype=np.int64
          )
      ]
    for row in next_rows:
      action = int(row["action"])
      if action not in liveness_actions:
        continue
      liveness = estimate_action_liveness(
          state,
          action,
          model,
          model_args,
          device,
          args,
          rng,
          rollout_seeds=matched_rollout_seeds,
      )
      row.update(liveness)
      scores[action] += float(liveness["liveness_score"])
    if bool(args.liveness_mask_unsafe_targets):
      safe_rows = [row for row in next_rows if liveness_row_is_safe(row, args)]
      if safe_rows:
        safe_floor = min(float(scores[int(row["action"])]) for row in safe_rows)
        unsafe_cap = safe_floor - float(args.liveness_unsafe_target_penalty)
        for row in next_rows:
          if not bool(row.get("liveness_evaluated", False)):
            continue
          if liveness_row_is_safe(row, args):
            continue
          scores[int(row["action"])] = min(float(scores[int(row["action"])]), unsafe_cap)
  return scores, next_rows


def liveness_repeat_agreement(next_rows: list[dict]) -> float | None:
  """Return how often the aggregate safest action also wins each repeat."""
  evaluated = [
      row for row in next_rows
      if bool(row.get("liveness_evaluated", False))
      and row.get("liveness_repeat_terminal_rates")
  ]
  if len(evaluated) < 2:
    return None
  best_row = max(
      evaluated,
      key=lambda row: (
          float(row.get("liveness_terminal_no_paradox_rate", 0.0)),
          -int(row["action"]),
      ),
  )
  best_rates = best_row["liveness_repeat_terminal_rates"]
  if not best_rates:
    return None
  stable = 0
  for pass_idx, best_rate in enumerate(best_rates):
    alternatives = [
        float(row["liveness_repeat_terminal_rates"][pass_idx])
        for row in evaluated
        if len(row["liveness_repeat_terminal_rates"]) > pass_idx
    ]
    if alternatives and float(best_rate) >= max(alternatives) - 1e-8:
      stable += 1
  return float(stable) / float(len(best_rates))


def early_reject_reason(state, legal: list[int], args: argparse.Namespace):
  """Reject impossible rows before paying for dense features and rollouts."""
  if int(getattr(state, "_phase", -1)) != 3:
    return "not_play"
  if len(legal) < 2:
    return "too_few_legal"
  trick_number = int(getattr(state, "_trick_number", -1))
  is_leader_state = getattr(state, "_led_color", None) is None
  min_trick = int(args.min_trick_number)
  if (
      str(args.row_filter) in ("leader_liveness", "late_or_leader_liveness")
      and is_leader_state
  ):
    min_trick = int(args.leader_min_trick_number)
  if trick_number < min_trick:
    return "too_early"
  if int(args.max_trick_number) >= 0 and trick_number > int(args.max_trick_number):
    return "too_late"
  return None


def should_keep_row(
    state,
    legal: list[int],
    scores: np.ndarray,
    next_rows: list[dict],
    args: argparse.Namespace,
) -> tuple[bool, str]:
  phase = int(getattr(state, "_phase", -1))
  if phase != 3:
    return False, "not_play"
  if len(legal) < 2:
    return False, "too_few_legal"
  trick_number = int(getattr(state, "_trick_number", -1))
  led_color = getattr(state, "_led_color", None)
  is_red_led_state = str(led_color).upper() == "R"
  is_leader_state = led_color is None
  min_trick = int(args.min_trick_number)
  if str(args.row_filter) in ("leader_liveness", "late_or_leader_liveness"):
    if is_leader_state:
      min_trick = int(args.leader_min_trick_number)
  if trick_number < min_trick:
    return False, "too_early"
  if int(args.max_trick_number) >= 0 and trick_number > int(args.max_trick_number):
    return False, "too_late"
  legal_scores = np.array([float(scores[int(action)]) for action in legal])
  spread = float(np.max(legal_scores) - np.min(legal_scores))
  if spread < float(args.min_score_spread):
    return False, "low_spread"
  next_counts = [int(row["next_legal_count"]) for row in next_rows]
  creates_forced = any(count <= 1 for count in next_counts)
  avoids_forced = any(count > 1 for count in next_counts)
  if args.row_filter == "immediate_contrast" and not (
      creates_forced and avoids_forced
  ):
    return False, "no_immediate_contrast"
  if args.row_filter == "late_play":
    return True, "late_play"
  if args.row_filter == "all_play":
    return True, "all_play"
  if args.row_filter == "late_red_led_play":
    if is_red_led_state:
      return True, "late_red_led_play"
    return False, "not_red_led"
  if args.row_filter in (
      "liveness_contrast",
      "leader_liveness",
      "late_or_leader_liveness",
      "red_led_liveness",
      "late_red_led_liveness",
  ):
    if int(args.liveness_rollouts_per_action) <= 0:
      return False, "liveness_disabled"
    evaluated_rows = [
        row for row in next_rows if bool(row.get("liveness_evaluated", False))
    ]
    if len(evaluated_rows) < 2:
      return False, "too_few_liveness_actions"
    min_repeat_agreement = float(
        getattr(args, "liveness_min_repeat_agreement", 0.0)
    )
    if min_repeat_agreement > 0.0:
      agreement = liveness_repeat_agreement(evaluated_rows)
      if agreement is None or agreement + 1e-8 < min_repeat_agreement:
        return False, "unstable_liveness_repeat"
    survival_rates = [
        float(row.get("liveness_survival_rate", 1.0)) for row in evaluated_rows
    ]
    min_legal_counts = [
        float(row.get("liveness_mean_min_legal_count", 0.0))
        for row in evaluated_rows
    ]
    forced_counts = [
        float(row.get("liveness_mean_forced_count", 0.0)) for row in evaluated_rows
    ]
    terminal_rates = [
        float(row.get("liveness_terminal_no_paradox_rate", 0.0))
        for row in evaluated_rows
    ]
    survival_spread = float(max(survival_rates) - min(survival_rates))
    legal_spread = float(max(min_legal_counts) - min(min_legal_counts))
    terminal_spread = float(max(terminal_rates) - min(terminal_rates))
    has_liveness_contrast = (
        survival_spread >= float(args.min_liveness_survival_spread)
        or legal_spread >= float(args.min_liveness_legal_spread)
        or terminal_spread >= float(args.min_liveness_terminal_spread)
        or (max(forced_counts) > 0.0 and min(forced_counts) <= 0.1)
    )
    if not has_liveness_contrast:
      return False, "no_liveness_contrast"
    safe_count = sum(1 for row in evaluated_rows if liveness_row_is_safe(row, args))
    risky_count = sum(1 for row in evaluated_rows if liveness_row_is_risky(row, args))
    if bool(args.require_liveness_safe_action):
      if safe_count <= 0 and risky_count <= 0:
        return False, "no_safe_or_risky_liveness_action"
      if safe_count <= 0:
        return False, "no_liveness_safe_action"
      if risky_count <= 0:
        return False, "no_liveness_risky_action"
    if args.row_filter == "leader_liveness" and not is_leader_state:
      return False, "not_leader"
    if (
        args.row_filter == "late_or_leader_liveness"
        and not is_leader_state
        and trick_number < int(args.min_trick_number)
    ):
      return False, "not_late_or_leader"
    if args.row_filter == "red_led_liveness" and not is_red_led_state:
      return False, "not_red_led"
    if (
        args.row_filter == "late_red_led_liveness"
        and (not is_red_led_state or trick_number < int(args.min_trick_number))
    ):
      return False, "not_late_red_led"
    if bool(args.require_target_liveness_evaluated):
      top_action = max(legal, key=lambda action: float(scores[int(action)]))
      top_row = next(
          row for row in next_rows if int(row["action"]) == int(top_action)
      )
      if not bool(top_row.get("liveness_evaluated", False)):
        return False, "target_liveness_not_evaluated"
    if bool(args.require_target_liveness_terminal_safe):
      top_action = max(legal, key=lambda action: float(scores[int(action)]))
      top_row = next(
          row for row in next_rows if int(row["action"]) == int(top_action)
      )
      if not bool(top_row.get("liveness_evaluated", False)):
        return False, "target_liveness_not_evaluated"
      if (
          float(top_row.get("liveness_terminal_no_paradox_rate", 0.0)) <
          float(args.liveness_safe_terminal_threshold)
      ):
        return False, "target_liveness_terminal_unsafe"
    return True, str(args.row_filter)
  return True, "immediate_contrast"


def collect_match(
    match_idx: int,
    model,
    model_args,
    device,
    args: argparse.Namespace,
) -> dict:
  rng = np.random.default_rng(int(args.seed) + match_idx * 1009)
  label_rng = np.random.default_rng(
      np.random.SeedSequence([int(args.seed), int(match_idx), 1034])
  )
  rows = []
  stats_rows = []
  keep_reasons = Counter()
  anchor_phases = split_csv(args.anchor_policy_phases)
  totals = np.zeros(int(args.players), dtype=np.float32)
  rounds = []
  for round_idx in range(int(args.players)):
    start_player = (
        int(args.initial_start) + match_idx + round_idx
    ) % int(args.players)
    game = make_game(int(args.players), start_player, bool(args.match_context))
    state = game.new_initial_state()
    if bool(args.match_context):
      state.set_match_context(totals, round_idx)
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        state.apply_action(int(rng.choice(actions, p=probs)))
        continue
      player = int(state.current_player())
      legal = [int(action) for action in state.legal_actions(player)]
      raw_policy, _ = model_policy_value(
          model,
          state,
          player,
          state.num_distinct_actions(),
          model_args.value_scale,
          device,
      )
      policy = normalized_policy(raw_policy, legal)
      action = behavior_action_for_state(
          state, player, legal, policy, model, model_args, device, args, rng
      )
      premature_rejection = early_reject_reason(state, legal, args)
      features = None
      scores = None
      next_rows = []
      if premature_rejection is None:
        features = action_feature_matrix(
            state, player, state.num_distinct_actions()
        )
        scores, next_rows = score_candidate_actions(
            state,
            player,
            legal,
            policy,
            features,
            model,
            model_args,
            device,
            args,
            label_rng,
            behavior_action=action,
        )
        keep, reason = should_keep_row(state, legal, scores, next_rows, args)
      else:
        keep, reason = False, premature_rejection
      keep_reasons[reason] += 1
      if keep:
        target = softmax_target(
            scores, legal_mask(state.num_distinct_actions(), legal) > 0.5,
            args.target_temperature,
        )
        obs = np.array(state.observation_tensor(player), dtype=np.float32)
        mask = legal_mask(state.num_distinct_actions(), legal)
        values = np.zeros(int(args.players), dtype=np.float32)
        paradoxes = np.zeros(int(args.players), dtype=np.float32)
        action_risk_targets = None
        action_risk_mask = None
        action_value_targets = None
        action_value_mask = None
        if bool(args.emit_liveness_action_labels):
          (
              action_risk_targets,
              action_risk_mask,
              action_value_targets,
              action_value_mask,
          ) = liveness_action_label_arrays(
              state.num_distinct_actions(),
              next_rows,
          )
        rows.append((
            obs,
            mask,
            target,
            values,
            paradoxes,
            action,
            player,
            action_risk_targets,
            action_risk_mask,
            action_value_targets,
            action_value_mask,
            features,
            None,
        ))
        top_action = int(np.argmax(target))
        top_next = next(
            row for row in next_rows if int(row["action"]) == top_action
        )
        behavior_next = next(
            row for row in next_rows if int(row["action"]) == int(action)
        )
        forced_action_count = sum(
            1 for row in next_rows if int(row["next_legal_count"]) <= 1
        )
        safe_action_count = sum(
            1 for row in next_rows if int(row["next_legal_count"]) > 1
        )
        liveness_evaluated_action_count = sum(
            1 for row in next_rows if bool(row.get("liveness_evaluated", False))
        )
        liveness_safe_action_count = sum(
            1
            for row in next_rows
            if liveness_row_is_safe(row, args)
        )
        liveness_risky_action_count = sum(
            1
            for row in next_rows
            if liveness_row_is_risky(row, args)
        )
        stats_rows.append({
            "role": "preforced",
            "match_idx": int(match_idx),
            "round_idx": int(round_idx),
            "player": int(player),
            "trick_number": int(getattr(state, "_trick_number", -1)),
            "led_color": getattr(state, "_led_color", None),
            "legal_count": int(len(legal)),
            "top_action": top_action,
            "top_prob": float(target[top_action]),
            "top_next_legal_count": int(top_next["next_legal_count"]),
            "behavior_action": int(action),
            "behavior_next_legal_count": int(
                behavior_next["next_legal_count"]
            ),
            "top_token_loss": bool(top_next["token_loss"]),
            "behavior_token_loss": bool(behavior_next["token_loss"]),
            "forced_action_count": int(forced_action_count),
            "safe_action_count": int(safe_action_count),
            "liveness_evaluated_action_count": int(
                liveness_evaluated_action_count
            ),
            "liveness_safe_action_count": int(liveness_safe_action_count),
            "liveness_risky_action_count": int(liveness_risky_action_count),
            "liveness_repeat_agreement": liveness_repeat_agreement(next_rows),
            "liveness_action_label_count": int(
                0 if action_value_mask is None else np.sum(action_value_mask)
            ),
            "top_liveness_evaluated": bool(
                top_next.get("liveness_evaluated", False)
            ),
            "behavior_liveness_evaluated": bool(
                behavior_next.get("liveness_evaluated", False)
            ),
            "top_liveness_survival_rate": float(
                top_next.get("liveness_survival_rate", 1.0)
            ),
            "behavior_liveness_survival_rate": float(
                behavior_next.get("liveness_survival_rate", 1.0)
            ),
            "top_liveness_mean_min_legal_count": float(
                top_next.get("liveness_mean_min_legal_count", 0.0)
            ),
            "behavior_liveness_mean_min_legal_count": float(
                behavior_next.get("liveness_mean_min_legal_count", 0.0)
            ),
            "top_liveness_mean_forced_count": float(
                top_next.get("liveness_mean_forced_count", 0.0)
            ),
            "behavior_liveness_mean_forced_count": float(
                behavior_next.get("liveness_mean_forced_count", 0.0)
            ),
            "top_liveness_paradox_rate": float(
                top_next.get("liveness_paradox_rate", 0.0)
            ),
            "behavior_liveness_paradox_rate": float(
                behavior_next.get("liveness_paradox_rate", 0.0)
            ),
            "top_liveness_terminal_no_paradox_rate": float(
                top_next.get("liveness_terminal_no_paradox_rate", 0.0)
            ),
            "behavior_liveness_terminal_no_paradox_rate": float(
                behavior_next.get("liveness_terminal_no_paradox_rate", 0.0)
            ),
            "top_liveness_incomplete_no_paradox_rate": float(
                top_next.get("liveness_incomplete_no_paradox_rate", 0.0)
            ),
            "behavior_liveness_incomplete_no_paradox_rate": float(
                behavior_next.get("liveness_incomplete_no_paradox_rate", 0.0)
            ),
            "score_spread": float(
                np.max([scores[int(a)] for a in legal])
                - np.min([scores[int(a)] for a in legal])
            ),
        })
      elif (
          float(args.anchor_policy_row_prob) > 0.0
          and state_phase_name(state).lower() in anchor_phases
          and rng.random() <= float(args.anchor_policy_row_prob)
      ):
        if features is None:
          features = action_feature_matrix(
              state, player, state.num_distinct_actions()
          )
        obs = np.array(state.observation_tensor(player), dtype=np.float32)
        mask = legal_mask(state.num_distinct_actions(), legal)
        values = np.zeros(int(args.players), dtype=np.float32)
        paradoxes = np.zeros(int(args.players), dtype=np.float32)
        rows.append((
            obs,
            mask,
            policy,
            values,
            paradoxes,
            action,
            player,
            None,
            None,
            None,
            None,
            features,
            None,
        ))
        stats_rows.append({
            "role": "anchor",
            "match_idx": int(match_idx),
            "round_idx": int(round_idx),
            "player": int(player),
            "phase": state_phase_name(state),
            "keep_reason": reason,
            "legal_count": int(len(legal)),
            "top_action": int(np.argmax(policy)),
            "top_prob": float(np.max(policy)),
        })
      state.apply_action(action)
    returns = np.array(state.returns(), dtype=np.float32)
    if bool(args.match_context):
      totals = returns
    else:
      totals += returns
    paradoxed = [bool(value) for value in getattr(state, "_has_paradoxed", [])]
    rounds.append({
        "match_idx": int(match_idx),
        "round_idx": int(round_idx),
        "start_player": int(start_player),
        "paradoxed": paradoxed,
    })
  return {
      "rows": rows,
      "stats_rows": stats_rows,
      "keep_reasons": dict(keep_reasons),
      "rounds": rounds,
  }


def summarize(
    rows: list[tuple],
    stats_rows: list[dict],
    rounds: list[dict],
    keep_reasons: Counter,
    args: argparse.Namespace,
) -> dict:
  preforced_rows = [row for row in stats_rows if row.get("role") == "preforced"]
  anchor_rows = [row for row in stats_rows if row.get("role") == "anchor"]
  trick_counts = Counter(str(row["trick_number"]) for row in preforced_rows)
  led_counts = Counter(str(row["led_color"]) for row in preforced_rows)
  top_next_counts = Counter(
      str(row["top_next_legal_count"]) for row in preforced_rows
  )
  behavior_next_counts = Counter(
      str(row["behavior_next_legal_count"]) for row in preforced_rows
  )
  top_liveness_survival_counts = Counter(
      f"{float(row.get('top_liveness_survival_rate', 1.0)):.2f}"
      for row in preforced_rows
  )
  behavior_liveness_survival_counts = Counter(
      f"{float(row.get('behavior_liveness_survival_rate', 1.0)):.2f}"
      for row in preforced_rows
  )
  top_liveness_terminal_counts = Counter(
      f"{float(row.get('top_liveness_terminal_no_paradox_rate', 0.0)):.2f}"
      for row in preforced_rows
  )
  behavior_liveness_terminal_counts = Counter(
      f"{float(row.get('behavior_liveness_terminal_no_paradox_rate', 0.0)):.2f}"
      for row in preforced_rows
  )
  liveness_action_label_counts = [
      int(row.get("liveness_action_label_count", 0)) for row in preforced_rows
  ]
  action_risk_targets = []
  action_value_targets = []
  for row in rows:
    if len(row) <= 10 or row[7] is None or row[8] is None or row[9] is None or row[10] is None:
      continue
    risk_mask = np.asarray(row[8], dtype=np.float32) > 0.5
    value_mask = np.asarray(row[10], dtype=np.float32) > 0.5
    if np.any(risk_mask):
      action_risk_targets.extend(
          [round(float(value), 4) for value in np.asarray(row[7])[risk_mask]]
      )
    if np.any(value_mask):
      action_value_targets.extend(
          [round(float(value), 4) for value in np.asarray(row[9])[value_mask]]
      )
  anchor_phase_counts = Counter(str(row["phase"]) for row in anchor_rows)
  rounds_with_any_paradox = sum(
      1 for row in rounds if any(bool(value) for value in row["paradoxed"])
  )
  return {
      "source": "quantum_cat_preforced_replay",
      "players": int(args.players),
      "matches": int(args.matches),
      "rounds": int(len(rounds)),
      "rounds_with_any_paradox": int(rounds_with_any_paradox),
      "same_policy_paradox_round_rate": (
          round(float(rounds_with_any_paradox) / float(len(rounds)), 4)
          if rounds else None
      ),
      "checkpoint": str(args.checkpoint),
      "seed": int(args.seed),
      "source_group_id": str(
          getattr(args, "source_group_id", "")
          or f"preforced:players={args.players}:seed={args.seed}"
      ),
      "match_context": bool(args.match_context),
      "row_filter": str(args.row_filter),
      "min_trick_number": int(args.min_trick_number),
      "max_trick_number": int(args.max_trick_number),
      "target_temperature": float(args.target_temperature),
      "behavior_temperature": float(args.behavior_temperature),
      "behavior_policy_mode": str(
          getattr(args, "behavior_policy_mode", "policy")
      ),
      "anchor_policy_row_prob": float(args.anchor_policy_row_prob),
      "anchor_policy_phases": sorted(split_csv(args.anchor_policy_phases)),
      "liveness_rollouts_per_action": int(args.liveness_rollouts_per_action),
      "liveness_max_plies": int(args.liveness_max_plies),
      "liveness_temperature": float(args.liveness_temperature),
      "liveness_continuation": str(args.liveness_continuation),
      "liveness_common_random_numbers": bool(
          getattr(args, "liveness_common_random_numbers", False)
      ),
      "liveness_repeat_passes": int(
          getattr(args, "liveness_repeat_passes", 1)
      ),
      "liveness_min_repeat_agreement": float(
          getattr(args, "liveness_min_repeat_agreement", 0.0)
      ),
      "require_liveness_safe_action": bool(args.require_liveness_safe_action),
      "require_liveness_terminal_safe_action": bool(
          args.require_liveness_terminal_safe_action
      ),
      "require_target_liveness_evaluated": bool(
          args.require_target_liveness_evaluated
      ),
      "require_target_liveness_terminal_safe": bool(
          args.require_target_liveness_terminal_safe
      ),
      "liveness_safe_forced_threshold": float(args.liveness_safe_forced_threshold),
      "liveness_risky_forced_threshold": float(args.liveness_risky_forced_threshold),
      "liveness_safe_terminal_threshold": float(args.liveness_safe_terminal_threshold),
      "min_liveness_terminal_spread": float(args.min_liveness_terminal_spread),
      "liveness_incomplete_penalty": float(args.liveness_incomplete_penalty),
      "emit_liveness_action_labels": bool(args.emit_liveness_action_labels),
      "rows": int(len(rows)),
      "preforced_rows": int(len(preforced_rows)),
      "anchor_rows": int(len(anchor_rows)),
      "anchor_phase_counts": dict(sorted(anchor_phase_counts.items())),
      "keep_reasons": dict(sorted(keep_reasons.items())),
      "trick_counts": dict(sorted(trick_counts.items())),
      "led_color_counts": dict(sorted(led_counts.items())),
      "top_next_legal_count_counts": dict(sorted(top_next_counts.items())),
      "behavior_next_legal_count_counts": dict(
          sorted(behavior_next_counts.items())
      ),
      "top_liveness_survival_rate_counts": dict(
          sorted(top_liveness_survival_counts.items())
      ),
      "behavior_liveness_survival_rate_counts": dict(
          sorted(behavior_liveness_survival_counts.items())
      ),
      "top_liveness_terminal_no_paradox_rate_counts": dict(
          sorted(top_liveness_terminal_counts.items())
      ),
      "behavior_liveness_terminal_no_paradox_rate_counts": dict(
          sorted(behavior_liveness_terminal_counts.items())
      ),
      "target_top_forced_rows": int(
          sum(1 for row in preforced_rows if row["top_next_legal_count"] <= 1)
      ),
      "avoidable_forced_rows": int(
          sum(
              1 for row in preforced_rows
              if row["forced_action_count"] > 0 and row["safe_action_count"] > 0
          )
      ),
      "target_top_forced_on_avoidable_rows": int(
          sum(
              1 for row in preforced_rows
              if (
                  row["forced_action_count"] > 0
                  and row["safe_action_count"] > 0
                  and row["top_next_legal_count"] <= 1
              )
          )
      ),
      "behavior_forced_on_avoidable_rows": int(
          sum(
              1 for row in preforced_rows
              if (
                  row["forced_action_count"] > 0
                  and row["safe_action_count"] > 0
                  and row["behavior_next_legal_count"] <= 1
              )
          )
      ),
      "all_forced_rows": int(
          sum(
              1 for row in preforced_rows
              if row["forced_action_count"] > 0 and row["safe_action_count"] == 0
          )
      ),
      "behavior_forced_rows": int(
          sum(
              1 for row in preforced_rows
              if row["behavior_next_legal_count"] <= 1
          )
      ),
      "liveness_avoidable_risky_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("liveness_risky_action_count", 0) > 0
              and row.get("liveness_safe_action_count", 0) > 0
          )
      ),
      "mean_liveness_evaluated_actions": (
          float(np.mean([
              row.get("liveness_evaluated_action_count", 0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "liveness_action_labeled_rows": int(
          sum(1 for count in liveness_action_label_counts if count > 0)
      ),
      "liveness_action_labeled_actions": int(
          sum(liveness_action_label_counts)
      ),
      "mean_liveness_action_labels_per_row": (
          float(np.mean(liveness_action_label_counts))
          if liveness_action_label_counts else 0.0
      ),
      "action_risk_target_counts": dict(
          sorted(Counter(f"{value:.2f}" for value in action_risk_targets).items())
      ),
      "action_value_target_counts": dict(
          sorted(Counter(f"{value:.2f}" for value in action_value_targets).items())
      ),
      "target_top_liveness_evaluated_rows": int(
          sum(1 for row in preforced_rows if row.get("top_liveness_evaluated"))
      ),
      "behavior_liveness_evaluated_rows": int(
          sum(1 for row in preforced_rows if row.get("behavior_liveness_evaluated"))
      ),
      "target_top_liveness_paradox_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("top_liveness_evaluated")
              and float(row.get("top_liveness_paradox_rate", 0.0)) > 0.0
          )
      ),
      "behavior_liveness_paradox_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("behavior_liveness_evaluated")
              and float(row.get("behavior_liveness_paradox_rate", 0.0)) > 0.0
          )
      ),
      "target_top_liveness_forced_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("top_liveness_evaluated")
              and float(row.get("top_liveness_mean_forced_count", 0.0)) >
              float(args.liveness_risky_forced_threshold)
          )
      ),
      "target_top_liveness_forced_on_avoidable_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("top_liveness_evaluated")
              and row.get("liveness_risky_action_count", 0) > 0
              and row.get("liveness_safe_action_count", 0) > 0
              and float(row.get("top_liveness_mean_forced_count", 0.0)) >
              float(args.liveness_risky_forced_threshold)
          )
      ),
      "behavior_liveness_forced_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("behavior_liveness_evaluated")
              and float(row.get("behavior_liveness_mean_forced_count", 0.0)) >
              float(args.liveness_risky_forced_threshold)
          )
      ),
      "target_top_liveness_terminal_safe_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("top_liveness_evaluated")
              and float(row.get("top_liveness_terminal_no_paradox_rate", 0.0)) >=
              float(args.liveness_safe_terminal_threshold)
          )
      ),
      "behavior_liveness_terminal_safe_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("behavior_liveness_evaluated")
              and float(
                  row.get("behavior_liveness_terminal_no_paradox_rate", 0.0)
              ) >= float(args.liveness_safe_terminal_threshold)
          )
      ),
      "target_top_liveness_incomplete_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("top_liveness_evaluated")
              and float(row.get("top_liveness_incomplete_no_paradox_rate", 0.0))
              > 0.0
          )
      ),
      "behavior_liveness_incomplete_rows": int(
          sum(
              1 for row in preforced_rows
              if row.get("behavior_liveness_evaluated")
              and float(
                  row.get("behavior_liveness_incomplete_no_paradox_rate", 0.0)
              ) > 0.0
          )
      ),
      "target_top_token_loss_rows": int(
          sum(1 for row in preforced_rows if row["top_token_loss"])
      ),
      "behavior_token_loss_rows": int(
          sum(1 for row in preforced_rows if row["behavior_token_loss"])
      ),
      "mean_target_top_prob": (
          float(np.mean([row["top_prob"] for row in preforced_rows]))
          if preforced_rows else 0.0
      ),
      "mean_score_spread": (
          float(np.mean([row["score_spread"] for row in preforced_rows]))
          if preforced_rows else 0.0
      ),
      "mean_top_liveness_survival_rate": (
          float(np.mean([
              row.get("top_liveness_survival_rate", 1.0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "mean_behavior_liveness_survival_rate": (
          float(np.mean([
              row.get("behavior_liveness_survival_rate", 1.0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "mean_top_liveness_min_legal_count": (
          float(np.mean([
              row.get("top_liveness_mean_min_legal_count", 0.0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "mean_behavior_liveness_min_legal_count": (
          float(np.mean([
              row.get("behavior_liveness_mean_min_legal_count", 0.0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "mean_top_liveness_terminal_no_paradox_rate": (
          float(np.mean([
              row.get("top_liveness_terminal_no_paradox_rate", 0.0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "mean_behavior_liveness_terminal_no_paradox_rate": (
          float(np.mean([
              row.get("behavior_liveness_terminal_no_paradox_rate", 0.0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "mean_top_liveness_incomplete_no_paradox_rate": (
          float(np.mean([
              row.get("top_liveness_incomplete_no_paradox_rate", 0.0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "mean_behavior_liveness_incomplete_no_paradox_rate": (
          float(np.mean([
              row.get("behavior_liveness_incomplete_no_paradox_rate", 0.0)
              for row in preforced_rows
          ]))
          if preforced_rows else 0.0
      ),
      "samples": preforced_rows[:12],
      "anchor_samples": anchor_rows[:8],
  }


def row_group_ids_from_stats(
    stats_rows: list[dict], source_group_id: str = ""
) -> list[str]:
  """Return one validation group id per replay row.

  The trainer can withhold whole groups when fitting action-value/risk labels.
  Grouping by match hand prevents nearly identical states from one trajectory
  leaking across train and validation after liveness/action-label mining.
  """
  group_ids = []
  source_prefix = f"{source_group_id}:" if source_group_id else ""
  for idx, row in enumerate(stats_rows):
    if "match_idx" in row and "round_idx" in row:
      group_ids.append(
          f"{source_prefix}m{int(row['match_idx'])}:r{int(row['round_idx'])}"
      )
    else:
      group_ids.append(f"{source_prefix}row:{idx}")
  return group_ids


def replay_output_paths(
    args: argparse.Namespace,
    partial_match_count: int | None = None,
) -> tuple[Path, Path]:
  out_path = Path(args.out)
  if partial_match_count is not None:
    out_path = out_path.with_suffix(
        f".partial_m{int(partial_match_count):06d}{out_path.suffix}"
    )
    return out_path, out_path.with_suffix(out_path.suffix + ".summary.json")
  summary_path = (
      Path(args.summary_out)
      if args.summary_out
      else out_path.with_suffix(out_path.suffix + ".summary.json")
  )
  return out_path, summary_path


def write_replay_outputs(
    rows: list[tuple],
    stats_rows: list[dict],
    rounds: list[dict],
    keep_reasons: Counter,
    args: argparse.Namespace,
    partial_match_count: int | None = None,
) -> dict:
  summary = summarize(rows, stats_rows, rounds, keep_reasons, args)
  row_group_ids = row_group_ids_from_stats(
      stats_rows, str(summary["source_group_id"])
  )
  if len(row_group_ids) == len(rows):
    summary["row_group_ids"] = row_group_ids
    summary["row_group_count"] = int(len(set(row_group_ids)))
  else:
    summary["row_group_ids"] = []
    summary["row_group_count"] = 0
    summary["row_group_error"] = (
        f"group id count {len(row_group_ids)} did not match row count "
        f"{len(rows)}"
    )
  summary["completed_matches"] = (
      int(partial_match_count)
      if partial_match_count is not None
      else int(args.matches)
  )
  summary["partial"] = partial_match_count is not None
  out_path, summary_path = replay_output_paths(args, partial_match_count)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_replay(rows, str(out_path), metadata=summary)
  text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(text)
  return {
      "out": str(out_path),
      "summary_out": str(summary_path),
      "rows": int(len(rows)),
      "completed_matches": int(summary["completed_matches"]),
      "partial": bool(summary["partial"]),
  }


def main() -> None:
  args = parse_args()
  if int(args.players) < 2 or int(args.players) > 5:
    raise ValueError("--players must be 2..5")
  device = torch.device("cpu")
  model, model_args = load_neural(str(args.checkpoint), int(args.players), device)
  if model is None:
    raise ValueError("--checkpoint failed to load")
  all_rows = []
  all_stats_rows = []
  all_rounds = []
  keep_reasons = Counter()
  for match_idx in range(int(args.matches)):
    result = collect_match(match_idx, model, model_args, device, args)
    all_rows.extend(result["rows"])
    all_stats_rows.extend(result["stats_rows"])
    all_rounds.extend(result["rounds"])
    keep_reasons.update(result["keep_reasons"])
    if int(args.progress_interval) > 0 and (
        match_idx + 1
    ) % int(args.progress_interval) == 0:
      print(json.dumps({
          "completed_matches": match_idx + 1,
          "rows": len(all_rows),
      }), flush=True)
    if (
        int(args.partial_save_interval) > 0
        and (match_idx + 1) % int(args.partial_save_interval) == 0
        and len(all_rows) > 0
    ):
      print(json.dumps({
          "iteration": "partial_save_start",
          "completed_matches": match_idx + 1,
          "rows": len(all_rows),
      }), flush=True)
      partial_info = write_replay_outputs(
          all_rows,
          all_stats_rows,
          all_rounds,
          keep_reasons,
          args,
          partial_match_count=match_idx + 1,
      )
      print(json.dumps({
          "iteration": "partial_save_done",
          **partial_info,
      }, sort_keys=True), flush=True)

  if int(args.max_rows) > 0 and len(all_rows) > int(args.max_rows):
    rng = np.random.default_rng(int(args.seed))
    indices = np.sort(
        rng.choice(len(all_rows), size=int(args.max_rows), replace=False)
    )
    all_rows = [all_rows[int(idx)] for idx in indices]
    all_stats_rows = [all_stats_rows[int(idx)] for idx in indices]

  final_info = write_replay_outputs(
      all_rows,
      all_stats_rows,
      all_rounds,
      keep_reasons,
      args,
      partial_match_count=None,
  )
  print(Path(final_info["summary_out"]).read_text(), end="")


if __name__ == "__main__":
  main()
