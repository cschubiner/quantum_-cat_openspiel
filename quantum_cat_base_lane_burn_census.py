"""Census raw-policy led-lane burn decisions in full homogeneous matches.

This is a cheap diagnostic for the hypothesis that the neural play policy often
refuses to follow a non-red led color even when a follow action is legal, thereby
burning that color token and shrinking future legal exits.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quantum_cat_ai import PARADOX, shared_discard_action, shared_prediction_action
from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    ACTION_FEATURE_PREFIX_SIZE,
    APPENDED_ACTION_FEATURE_INDEX,
    action_feature_matrix,
    make_game,
    model_policy_value,
    terminal_paradox_target,
)
from quantum_cat_counterfactual_liquidity_replay import round_raw_scores
from quantum_cat_full_match_elo import load_neural


COLORS = ("R", "B", "Y", "G")
WOULD_WIN_AFTER_PLAY_INDEX = ACTION_FEATURE_PREFIX_SIZE + 19


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--matches", type=int, default=20)
  parser.add_argument("--seed", type=int, default=20261332)
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
  parser.add_argument("--min-trick-number", type=int, default=0)
  parser.add_argument("--max-trick-number", type=int, default=-1)
  parser.add_argument("--max-samples", type=int, default=30)
  parser.add_argument("--progress-interval", type=int, default=5)
  parser.add_argument("--out", required=True)
  parser.add_argument("--no-match-context", dest="match_context", action="store_false")
  parser.set_defaults(match_context=True)
  return parser.parse_args()


def resolve_device(name: str) -> torch.device:
  if name == "auto":
    if torch.cuda.is_available():
      return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
      return torch.device("mps")
    return torch.device("cpu")
  return torch.device(name)


def indexed_feature(features: np.ndarray, name: str) -> np.ndarray:
  if name not in APPENDED_ACTION_FEATURE_INDEX:
    return np.zeros(features.shape[0], dtype=np.float32)
  return features[:, APPENDED_ACTION_FEATURE_INDEX[name]]


def token_loss_mask(features: np.ndarray) -> np.ndarray:
  return (
      (features[:, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX] > 0.5)
      | (indexed_feature(features, "token_loss_newly_loses_led") > 0.5)
  )


def action_name(state, player: int, action: int) -> str:
  try:
    return str(state.action_to_string(int(player), int(action)))
  except Exception:
    try:
      return str(state._action_to_string(int(player), int(action)))
    except Exception:
      return str(int(action))


def phase_name(state) -> str:
  phase = int(getattr(state, "_phase", -1))
  return {
      0: "chance",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "terminal",
  }.get(phase, "unknown")


def legal_argmax(policy: np.ndarray, legal: list[int]) -> int:
  return max((int(action) for action in legal), key=lambda action: float(policy[action]))


def policy_action(
    state,
    player: int,
    num_actions: int,
    model,
    model_args,
    device: torch.device,
) -> tuple[np.ndarray, int]:
  legal = [int(action) for action in state.legal_actions(player)]
  setup_action = shared_discard_action(state, player, legal)
  if setup_action is None:
    setup_action = shared_prediction_action(state, player, legal)
  if setup_action is not None:
    policy = np.zeros(num_actions, dtype=np.float32)
    policy[int(setup_action)] = 1.0
    return policy, int(setup_action)
  policy, _ = model_policy_value(
      model,
      state,
      player,
      num_actions,
      float(model_args.value_scale),
      device,
  )
  return policy, legal_argmax(policy, legal)


def add_rate(summary: dict, numerator: str, denominator: str, output: str) -> None:
  denom = int(summary.get(denominator, 0))
  if denom <= 0:
    summary[output] = None
  else:
    summary[output] = float(summary.get(numerator, 0)) / float(denom)


def summarize_counts(counts: Counter) -> dict:
  summary = {key: int(value) for key, value in sorted(counts.items())}
  rate_defs = (
      ("paradox_rounds", "rounds", "paradox_round_rate"),
      ("base_lost_led_follow_available", "led_follow_available", "base_lost_led_follow_available_rate"),
      ("base_lost_led_follow_available", "census_decisions", "base_lost_led_per_census_decision_rate"),
      ("nonred_led_base_off_led_switch", "nonred_led_follow_available", "nonred_led_off_led_switch_rate"),
      ("nonred_led_base_red_switch", "nonred_led_base_off_led_switch", "nonred_led_switch_to_red_rate"),
      ("nonred_led_base_switch_wins", "nonred_led_base_off_led_switch", "nonred_led_switch_win_rate"),
      ("nonred_led_base_red_switch_wins", "nonred_led_base_red_switch", "nonred_led_red_switch_win_rate"),
      ("base_red_switch", "base_off_led_switch", "base_switch_to_red_rate"),
      ("base_switch_wins", "base_off_led_switch", "base_switch_win_rate"),
      ("base_filter_follow_available_base_token_loss", "base_filter_total", "base_filter_follow_available_token_loss_share"),
  )
  for numerator, denominator, output in rate_defs:
    add_rate(summary, numerator, denominator, output)
  return summary


def classify_decision(
    state,
    player: int,
    legal: list[int],
    base_action: int,
    features: np.ndarray,
    counts: Counter,
    samples: list[dict],
    args: argparse.Namespace,
) -> None:
  trick_number = int(getattr(state, "_trick_number", -1))
  if trick_number < int(args.min_trick_number):
    return
  if int(args.max_trick_number) >= 0 and trick_number > int(args.max_trick_number):
    return
  if len(legal) < 2:
    return

  led_color = getattr(state, "_led_color", None)
  base_is_paradox = int(base_action) == PARADOX
  follows_led = features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5
  token_losses = token_loss_mask(features)
  red = features[:, ACTION_FEATURE_IS_RED_INDEX] > 0.5
  would_win = features[:, WOULD_WIN_AFTER_PLAY_INDEX] > 0.5

  legal_arr = np.array([int(action) for action in legal], dtype=np.int64)
  follow_available = bool(np.any(follows_led[legal_arr]))
  follow_no_loss_available = bool(np.any(follows_led[legal_arr] & (~token_losses[legal_arr])))

  base_follows = bool(follows_led[int(base_action)]) if not base_is_paradox else False
  base_token_loss = bool(token_losses[int(base_action)]) if not base_is_paradox else False
  base_red = bool(red[int(base_action)]) if not base_is_paradox else False
  base_wins = bool(would_win[int(base_action)]) if not base_is_paradox else False
  base_off_led_switch = bool(led_color is not None and not base_follows and not base_is_paradox)
  base_filter_allowed = bool(base_follows and not base_token_loss)

  counts["census_decisions"] += 1
  counts[f"trick_{trick_number}_decisions"] += 1
  counts[f"led_{led_color if led_color is not None else 'None'}_decisions"] += 1
  if led_color is not None:
    counts["led_decisions"] += 1
  if led_color in COLORS and led_color != "R":
    counts["nonred_led_decisions"] += 1
  if follow_available:
    counts["led_follow_available"] += 1
  if follow_no_loss_available:
    counts["led_follow_no_loss_available"] += 1
  if base_follows:
    counts["base_follows_led"] += 1
  if base_token_loss:
    counts["base_token_loss"] += 1
  if base_red:
    counts["base_red"] += 1
  if base_is_paradox:
    counts["base_paradox"] += 1
  if base_off_led_switch:
    counts["base_off_led_switch"] += 1
    if base_red:
      counts["base_red_switch"] += 1
    if base_wins:
      counts["base_switch_wins"] += 1
  if follow_available and base_token_loss:
    counts["base_lost_led_follow_available"] += 1
    if base_red:
      counts["base_lost_led_red_switch"] += 1
    if base_wins:
      counts["base_lost_led_switch_wins"] += 1
  if led_color in COLORS and led_color != "R" and follow_available:
    counts["nonred_led_follow_available"] += 1
    if base_off_led_switch:
      counts["nonred_led_base_off_led_switch"] += 1
      if base_red:
        counts["nonred_led_base_red_switch"] += 1
      if base_wins:
        counts["nonred_led_base_switch_wins"] += 1
      if base_red and base_wins:
        counts["nonred_led_base_red_switch_wins"] += 1

  if not base_filter_allowed:
    counts["base_filter_total"] += 1
    if led_color is None:
      counts["base_filter_no_led"] += 1
    elif base_is_paradox:
      counts["base_filter_paradox"] += 1
    elif not follow_available:
      counts["base_filter_no_follow_available"] += 1
    elif base_token_loss:
      counts["base_filter_follow_available_base_token_loss"] += 1
    else:
      counts["base_filter_follow_available_base_no_token_loss"] += 1

  should_sample = (
      follow_available
      and (base_token_loss or (led_color in COLORS and led_color != "R" and base_off_led_switch))
      and len(samples) < int(args.max_samples)
  )
  if should_sample:
    follow_actions = [
        int(action) for action in legal
        if bool(follows_led[int(action)])
    ][:8]
    samples.append({
        "match_idx": int(getattr(args, "_current_match_idx", -1)),
        "round_idx": int(getattr(args, "_current_round_idx", -1)),
        "player": int(player),
        "trick_number": int(trick_number),
        "led_color": None if led_color is None else str(led_color),
        "base_action": int(base_action),
        "base_action_name": action_name(state, player, base_action),
        "base_follows_led": bool(base_follows),
        "base_token_loss": bool(base_token_loss),
        "base_red": bool(base_red),
        "base_would_win": bool(base_wins),
        "legal_count": int(len(legal)),
        "follow_available": bool(follow_available),
        "follow_no_loss_available": bool(follow_no_loss_available),
        "sample_follow_actions": [
            {
                "action": int(action),
                "name": action_name(state, player, action),
                "would_win": bool(would_win[int(action)]),
            }
            for action in follow_actions
        ],
    })


def run(args: argparse.Namespace) -> dict:
  rng = np.random.default_rng(int(args.seed))
  device = resolve_device(str(args.device))
  model, model_args = load_neural(str(args.checkpoint), int(args.players), device)
  if model is None or model_args is None:
    raise SystemExit(f"Could not load checkpoint: {args.checkpoint}")

  run_args = SimpleNamespace(
      players=int(args.players),
      match_context=bool(args.match_context),
  )
  counts: Counter = Counter()
  samples: list[dict] = []
  round_summaries: list[dict] = []
  started = time.perf_counter()

  for match_idx in range(int(args.matches)):
    match_totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      start_player = (
          int(args.initial_start) + int(match_idx) + int(round_idx)
      ) % int(args.players)
      game = make_game(run_args, start_player)
      state = game.new_initial_state()
      if bool(args.match_context):
        state.set_match_context(match_totals, round_idx)
      num_actions = int(game.num_distinct_actions())
      args._current_match_idx = int(match_idx)
      args._current_round_idx = int(round_idx)
      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          action = int(rng.choice(np.array(actions, dtype=np.int64), p=probs))
          state.apply_action(action)
          continue
        player = int(state.current_player())
        legal = [int(action) for action in state.legal_actions(player)]
        policy, action = policy_action(
            state,
            player,
            num_actions,
            model,
            model_args,
            device,
        )
        if phase_name(state) == "play":
          features = action_feature_matrix(state, player, num_actions)
          classify_decision(state, player, legal, int(action), features, counts, samples, args)
        state.apply_action(int(action))
      paradox = terminal_paradox_target(state, int(args.players))
      any_paradox = bool(np.any(paradox > 0.5))
      counts["rounds"] += 1
      if any_paradox:
        counts["paradox_rounds"] += 1
      round_summaries.append({
          "match_idx": int(match_idx),
          "round_idx": int(round_idx),
          "start_player": int(start_player),
          "any_paradox": bool(any_paradox),
          "paradoxed": [bool(value) for value in paradox],
      })
      match_totals += round_raw_scores(state)
    if (
        int(args.progress_interval) > 0
        and (match_idx + 1) % int(args.progress_interval) == 0
    ):
      print(json.dumps({
          "completed_matches": int(match_idx + 1),
          "rounds": int(counts["rounds"]),
          "census_decisions": int(counts["census_decisions"]),
          "base_lost_led_follow_available": int(counts["base_lost_led_follow_available"]),
          "elapsed_sec": round(time.perf_counter() - started, 3),
      }), flush=True)

  summary = summarize_counts(counts)
  return {
      "kind": "base_lane_burn_census",
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches": int(args.matches),
      "rounds": int(counts["rounds"]),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "min_trick_number": int(args.min_trick_number),
      "max_trick_number": int(args.max_trick_number),
      "summary": summary,
      "samples": samples,
      "round_summaries": round_summaries,
      "elapsed_sec": round(time.perf_counter() - started, 3),
  }


def main() -> None:
  args = parse_args()
  result = run(args)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
      "out": str(out),
      "summary": result["summary"],
      "elapsed_sec": result["elapsed_sec"],
  }, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
