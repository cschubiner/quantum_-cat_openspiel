#!/usr/bin/env python3
"""Audit upstream causes of future no-led-color token-loss switches.

Run757/758 showed that Run730 does follow led when it can, and that most
no-led-color switches newly burn a still-available led token. This diagnostic
looks one step upstream: for each such future switch, find the earlier decision
that made the target player's potential legal count in that led color drop to
zero while the token was still available.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_full_match_elo import (
    action_liveness_certificate,
    load_neural_bots,
    make_eval_bot,
)


COLOR_NAMES = ("R", "B", "Y", "G")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=40)
  parser.add_argument("--seed", type=int, default=20260615)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--mode", default="policy")
  parser.add_argument("--bot-name", default="bot")
  parser.add_argument("--belief-samples", type=int, default=4)
  parser.add_argument("--belief-sims", type=int, default=8)
  parser.add_argument("--mcts-sims", type=int, default=16)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--sample-limit", type=int, default=24)
  parser.add_argument("--out", required=True)
  return parser.parse_args()


def action_color_idx(state, action: int) -> int | None:
  action = int(action)
  if action == 999:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  if num_card_types <= 0:
    return None
  idx = action // num_card_types
  return idx if 0 <= idx < len(COLOR_NAMES) else None


def action_color(state, action: int) -> str | None:
  idx = action_color_idx(state, action)
  return None if idx is None else COLOR_NAMES[idx]


def action_rank(state, action: int) -> int | None:
  action = int(action)
  if action == 999:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  if num_card_types <= 0:
    return None
  return int(action % num_card_types) + 1


def phase_name(state) -> str:
  return {
      0: "chance",
      1: "discard",
      2: "prediction",
      3: "play",
      4: "terminal",
  }.get(int(getattr(state, "_phase", -1)), "unknown")


def action_string(state, player: int, action: int) -> str:
  try:
    return state._action_to_string(int(player), int(action))  # pylint: disable=protected-access
  except Exception:
    return str(int(action))


def color_index(color: str) -> int:
  return COLOR_NAMES.index(str(color))


def token_available(state, player: int, color: str) -> bool:
  tokens = np.asarray(getattr(state, "_color_tokens"), dtype=bool)
  c_idx = color_index(color)
  return bool(
      0 <= int(player) < tokens.shape[0]
      and c_idx < tokens.shape[1]
      and tokens[int(player), c_idx]
  )


def potential_color_action_count(state, player: int, color: str) -> int:
  """Count playable ranks in color, independent of whose turn it currently is."""
  player = int(player)
  c_idx = color_index(color)
  if not token_available(state, player, color):
    return 0
  hand = np.asarray(getattr(state, "_hands")[player], dtype=int)
  board = np.asarray(getattr(state, "_board_ownership"), dtype=int)
  count = 0
  for rank_idx, qty in enumerate(hand):
    if int(qty) <= 0:
      continue
    if int(board[c_idx, rank_idx]) == -1:
      count += 1
  return int(count)


def make_same_policy_bots(args, device):
  specs = [
      f"{args.bot_name}{seat}={args.checkpoint}:{args.mode}"
      for seat in range(int(args.players))
  ]
  neural_bots = load_neural_bots(specs, int(args.players), device, args)
  bots = [
      make_eval_bot(
          f"{args.bot_name}{seat}",
          int(args.seed) + seat,
          None,
          None,
          device,
          int(args.belief_samples),
          int(args.belief_sims),
          int(args.mcts_sims),
          neural_bots=neural_bots,
          eval_args=args,
      )
      for seat in range(int(args.players))
  ]
  return bots


def is_new_token_loss_bottleneck(decision):
  state = decision["state_before"]
  player = int(decision["player"])
  action = int(decision["action"])
  led_color = getattr(state, "_led_color", None)
  if led_color not in ("B", "Y", "G"):
    return False
  if action == 999:
    return False
  if action_color(state, action) == led_color:
    return False
  if not token_available(state, player, led_color):
    return False
  if potential_color_action_count(state, player, led_color) != 0:
    return False
  cert = action_liveness_certificate(state, player, action, base_action=action)
  return bool(cert.get("lost_led_token", False))


def direct_zero_cause(decisions, event_index: int, target_player: int,
                      target_color: str):
  direct = None
  drops = []
  for decision in decisions[:event_index]:
    before = decision["state_before"]
    after = decision["state_after"]
    before_count = potential_color_action_count(
        before, target_player, target_color
    )
    after_count = potential_color_action_count(
        after, target_player, target_color
    )
    after_token = token_available(after, target_player, target_color)
    if after_token and after_count < before_count:
      row = {
          "decision": decision,
          "before_count": int(before_count),
          "after_count": int(after_count),
      }
      drops.append(row)
      if before_count > 0 and after_count == 0:
        direct = row
  return direct, drops


def cause_metadata(cause, target_player: int, target_color: str):
  if cause is None:
    return None
  decision = cause["decision"]
  state = decision["state_before"]
  action = int(decision["action"])
  rank = action_rank(state, action)
  color = action_color(state, action)
  rank_idx = None if rank is None else int(rank) - 1
  target_had_rank = False
  if rank_idx is not None:
    hand = np.asarray(getattr(state, "_hands")[int(target_player)], dtype=int)
    if 0 <= rank_idx < hand.shape[0]:
      target_had_rank = bool(int(hand[rank_idx]) > 0)
  return {
      "decision_index": int(decision["index"]),
      "phase": str(decision["phase"]),
      "trick_number": int(decision["trick_number"]),
      "actor": int(decision["player"]),
      "same_as_target_player": bool(int(decision["player"]) == int(target_player)),
      "action": int(action),
      "action_string": str(decision["action_string"]),
      "action_color": color,
      "action_rank": rank,
      "before_count": int(cause["before_count"]),
      "after_count": int(cause["after_count"]),
      "action_in_target_color": bool(color == target_color),
      "target_player_had_action_rank_before": bool(target_had_rank),
  }


def event_metadata(decision, direct_cause, drops, match_idx, round_idx,
                   start_player):
  state = decision["state_before"]
  player = int(decision["player"])
  action = int(decision["action"])
  led_color = str(getattr(state, "_led_color"))
  cert = action_liveness_certificate(state, player, action, base_action=action)
  return {
      "match_index": int(match_idx),
      "round_index": int(round_idx),
      "start_player": int(start_player),
      "event_decision_index": int(decision["index"]),
      "event_trick_number": int(decision["trick_number"]),
      "target_player": int(player),
      "led_color": led_color,
      "event_action": int(action),
      "event_action_string": str(decision["action_string"]),
      "event_action_color": action_color(state, action),
      "event_action_rank": action_rank(state, action),
      "event_legal_count": int(len(decision["legal"])),
      "event_red_legal_count": int(
          sum(1 for item in decision["legal"] if action_color(state, item) == "R")
      ),
      "event_responsible_red_win_switch": bool(
          cert.get("responsible_red_win_switch", False)
      ),
      "event_liveness": cert,
      "direct_zero_cause": cause_metadata(direct_cause, player, led_color),
      "drop_cause_count": int(len(drops)),
      "last_drop_cause": (
          cause_metadata(drops[-1], player, led_color) if drops else None
      ),
  }


def play_round(game, bots, args, seed, match_totals=None, round_index=0):
  if seed is not None:
    np.random.seed(int(seed))
  state = game.new_initial_state()
  if match_totals is not None:
    state.set_match_context(match_totals, round_index)
  decisions = []
  while not state.is_terminal():
    if state.is_chance_node():
      actions, probs = zip(*state.chance_outcomes())
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    player = int(state.current_player())
    legal = list(state.legal_actions(player))
    before = state.clone()
    action = int(bots[player].step(state, player))
    state.apply_action(action)
    after = state.clone()
    decisions.append({
        "index": int(len(decisions)),
        "state_before": before,
        "state_after": after,
        "player": int(player),
        "phase": phase_name(before),
        "trick_number": int(getattr(before, "_trick_number", -1)),
        "legal": legal,
        "action": int(action),
        "action_string": action_string(before, player, action),
    })
  return state, decisions


def inc(mapping, key, amount=1):
  mapping[str(key)] = int(mapping.get(str(key), 0)) + int(amount)


def main():
  args = parse_args()
  if int(args.players) < 2 or int(args.players) > 5:
    raise ValueError("--players must be 2..5")
  device = torch.device(args.device)
  bots = make_same_policy_bots(args, device)
  summary = Counter()
  samples = []
  by_led_color = {}
  by_event_trick = {}
  direct_by_phase = {}
  direct_by_actor_relation = {}
  direct_by_action_color = {}

  for match_idx in range(int(args.matches)):
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      start_player = (match_idx + round_idx) % int(args.players)
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": int(args.players),
              "start_player": int(start_player),
              "match_context": int(bool(args.match_context)),
          },
      )
      prior_totals = np.copy(totals)
      round_seed = int(args.seed) + match_idx * 1009 + round_idx * 37
      terminal, decisions = play_round(
          game,
          bots,
          args,
          seed=round_seed,
          match_totals=prior_totals if bool(args.match_context) else None,
          round_index=round_idx,
      )
      returns = np.array(terminal.returns(), dtype=np.float32)
      totals = returns if bool(args.match_context) else totals + returns
      summary["rounds"] += 1
      if any(bool(value) for value in getattr(terminal, "_has_paradoxed", [])):
        summary["paradox_rounds"] += 1
      for idx, decision in enumerate(decisions):
        if not is_new_token_loss_bottleneck(decision):
          continue
        summary["events"] += 1
        led_color = str(getattr(decision["state_before"], "_led_color"))
        inc(by_led_color, led_color)
        inc(by_event_trick, int(decision["trick_number"]))
        direct, drops = direct_zero_cause(
            decisions,
            idx,
            int(decision["player"]),
            led_color,
        )
        if direct is not None:
          summary["events_with_direct_zero_cause"] += 1
          cause = cause_metadata(direct, int(decision["player"]), led_color)
          inc(direct_by_phase, cause["phase"])
          inc(
              direct_by_actor_relation,
              "same_player" if cause["same_as_target_player"] else "other_player",
          )
          inc(direct_by_action_color, cause["action_color"])
          if bool(cause["action_in_target_color"]):
            summary["direct_cause_action_in_target_color"] += 1
          if bool(cause["target_player_had_action_rank_before"]):
            summary["direct_cause_target_had_rank"] += 1
        else:
          summary["events_without_direct_zero_cause"] += 1
        summary["total_drop_causes"] += int(len(drops))
        if len(samples) < int(args.sample_limit):
          samples.append(
              event_metadata(decision, direct, drops, match_idx, round_idx,
                             start_player)
          )

  rounds = int(summary.get("rounds", 0))
  events = int(summary.get("events", 0))
  direct = int(summary.get("events_with_direct_zero_cause", 0))
  result = {
      "checkpoint": str(args.checkpoint),
      "mode": str(args.mode),
      "players": int(args.players),
      "matches": int(args.matches),
      "hand_rounds": rounds,
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "summary": dict(summary),
      "rates": {
          "same_policy_paradox_round_rate": (
              round(float(summary.get("paradox_rounds", 0)) / rounds, 4)
              if rounds else None
          ),
          "events_per_hand_round": (
              round(float(events) / rounds, 4) if rounds else None
          ),
          "direct_zero_cause_rate": (
              round(float(direct) / events, 4) if events else None
          ),
          "mean_drop_causes_per_event": (
              round(float(summary.get("total_drop_causes", 0)) / events, 4)
              if events else None
          ),
      },
      "by_led_color": by_led_color,
      "by_event_trick": by_event_trick,
      "direct_by_phase": direct_by_phase,
      "direct_by_actor_relation": direct_by_actor_relation,
      "direct_by_action_color": direct_by_action_color,
      "samples": samples,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
