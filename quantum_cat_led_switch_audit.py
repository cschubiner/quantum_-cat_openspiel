#!/usr/bin/env python3
"""Audit off-led play choices under a non-red led color.

The main question is whether a policy burns the led-color lane only when it has
an actual trump/win reason. We therefore measure non-red-led play decisions,
switches away from the led color, red switches, and whether switch plays
eventually win their trick.
"""

from __future__ import annotations

import argparse
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
from quantum_cat_ai import make_bot


COLOR_NAMES = ["R", "B", "Y", "G"]


def action_color(state, action):
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


def action_rank(state, action):
  action = int(action)
  if action == 999:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  if num_card_types <= 0:
    return None
  return int(action % num_card_types) + 1


def legal_color_count(state, legal, color):
  return sum(1 for action in legal if action_color(state, action) == color)


def completed_trick_winner(completed, led_color):
  plays = [
      (int(player), card)
      for player, card in completed
      if card is not None
  ]
  red_plays = [(player, int(card[0])) for player, card in plays if card[1] == "R"]
  if red_plays:
    return max(red_plays, key=lambda item: item[1])[0]
  led_plays = [
      (player, int(card[0])) for player, card in plays if card[1] == led_color
  ]
  if led_plays:
    return max(led_plays, key=lambda item: item[1])[0]
  if plays:
    return max((player, int(card[0])) for player, card in plays)[0]
  return None


def pct(numer, denom):
  return None if denom <= 0 else round(100.0 * float(numer) / float(denom), 2)


def make_same_policy_bots(args, device=None):
  if args.checkpoint:
    specs = [
        f"{args.bot_name}{seat}={args.checkpoint}:{args.mode}"
        for seat in range(args.players)
    ]
    neural_bots = load_neural_bots(specs, args.players, device, args)
    bots = [
        make_eval_bot(
            f"{args.bot_name}{seat}",
            args.seed + seat,
            None,
            None,
            device,
            args.belief_samples,
            args.belief_sims,
            args.mcts_sims,
            neural_bots=neural_bots,
            eval_args=args,
        )
        for seat in range(args.players)
    ]
    return bots, neural_bots
  return (
      [make_bot(args.builtin_bot, seed=args.seed + seat)
       for seat in range(args.players)],
      {},
  )


def empty_stats():
  return {
      "nonred_led_opportunities": 0,
      "nonred_led_nonparadox_actions": 0,
      "led_color_legal_actions": 0,
      "no_led_color_legal_actions": 0,
      "led_token_available_actions": 0,
      "led_token_unavailable_actions": 0,
      "follow_led": 0,
      "follow_with_led_color_legal": 0,
      "follow_with_no_led_color_legal": 0,
      "switch_off_led": 0,
      "switch_to_red": 0,
      "switch_to_nonred": 0,
      "switch_with_red_legal": 0,
      "switch_with_no_red_legal": 0,
      "switch_with_red_legal_chose_red": 0,
      "switch_wins_trick": 0,
      "switch_loses_trick": 0,
      "switch_unresolved_by_paradox": 0,
      "red_switch_wins_trick": 0,
      "red_switch_loses_trick": 0,
      "red_switch_unresolved_by_paradox": 0,
      "switch_with_led_color_legal": 0,
      "switch_with_no_led_color_legal": 0,
      "switch_with_led_token_available": 0,
      "switch_with_led_token_unavailable": 0,
      "switch_newly_loses_led_token": 0,
      "switch_newly_loses_led_token_to_red": 0,
      "switch_newly_loses_led_token_to_nonred": 0,
      "switch_newly_loses_led_token_with_red_legal": 0,
      "switch_newly_loses_led_token_with_red_legal_chose_red": 0,
      "switch_newly_loses_led_token_responsible_red_win": 0,
      "switch_newly_loses_led_token_wins_trick": 0,
      "switch_newly_loses_led_token_loses_trick": 0,
      "switch_newly_loses_led_token_unresolved_by_paradox": 0,
      "avoidable_switch_to_red": 0,
      "avoidable_switch_wins_trick": 0,
      "avoidable_switch_loses_trick": 0,
      "avoidable_switch_unresolved_by_paradox": 0,
      "nonred_led_paradox_actions": 0,
      "by_led_color": {},
      "by_trick_number": {},
      "samples_nonred_switch": [],
  }


def inc(mapping, key, amount=1):
  mapping[str(key)] = int(mapping.get(str(key), 0)) + int(amount)


def record_sample(stats, event, winner=None):
  if len(stats["samples_nonred_switch"]) >= 12:
    return
  row = dict(event)
  if winner is not None:
    row["winner"] = int(winner)
    row["won_trick"] = bool(winner == event["player"])
  stats["samples_nonred_switch"].append(row)


def finalize_pending_switches(state, pending, stats):
  completed_count = len(getattr(state, "_completed_tricks", []))
  still_pending = []
  for event in pending:
    trick_number = int(event["trick_number"])
    if trick_number < completed_count:
      completed = state._completed_tricks[trick_number]  # pylint: disable=protected-access
      winner = completed_trick_winner(completed, event["led_color"])
      if winner == int(event["player"]):
        stats["switch_wins_trick"] += 1
        if event["chosen_color"] == "R":
          stats["red_switch_wins_trick"] += 1
        if event["led_color_legal_count"] > 0:
          stats["avoidable_switch_wins_trick"] += 1
        if event.get("newly_lost_led_token"):
          stats["switch_newly_loses_led_token_wins_trick"] += 1
      else:
        stats["switch_loses_trick"] += 1
        if event["chosen_color"] == "R":
          stats["red_switch_loses_trick"] += 1
        if event["led_color_legal_count"] > 0:
          stats["avoidable_switch_loses_trick"] += 1
        if event.get("newly_lost_led_token"):
          stats["switch_newly_loses_led_token_loses_trick"] += 1
      record_sample(stats, event, winner)
    else:
      still_pending.append(event)
  return still_pending


def play_round_with_audit(game, bots, args, seed, match_totals=None, round_index=0):
  if seed is not None:
    np.random.seed(seed)
  state = game.new_initial_state()
  if match_totals is not None:
    state.set_match_context(match_totals, round_index)
  stats = empty_stats()
  pending_switches = []
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      action = int(np.random.choice(actions, p=probs))
      state.apply_action(action)
      continue

    player = int(state.current_player())
    legal = list(state.legal_actions(player))
    action = int(bots[player].step(state, player))
    phase = int(getattr(state, "_phase", -1))
    led_color = getattr(state, "_led_color", None)
    chosen_color = action_color(state, action)
    if phase == 3 and led_color in ("B", "Y", "G"):
      stats["nonred_led_opportunities"] += 1
      inc(stats["by_led_color"], led_color)
      inc(stats["by_trick_number"], int(getattr(state, "_trick_number", -1)))
      if action == 999:
        stats["nonred_led_paradox_actions"] += 1
      elif chosen_color is not None:
        led_color_legal_count = legal_color_count(state, legal, led_color)
        red_legal_count = legal_color_count(state, legal, "R")
        led_idx = COLOR_NAMES.index(led_color)
        tokens = np.asarray(getattr(state, "_color_tokens"), dtype=bool)
        led_token_available = bool(
            player < tokens.shape[0]
            and led_idx < tokens.shape[1]
            and tokens[player, led_idx]
        )
        cert = action_liveness_certificate(
            state, player, action, base_action=action
        )
        newly_lost_led_token = bool(cert.get("lost_led_token", False))
        stats["nonred_led_nonparadox_actions"] += 1
        if led_color_legal_count > 0:
          stats["led_color_legal_actions"] += 1
        else:
          stats["no_led_color_legal_actions"] += 1
        if led_token_available:
          stats["led_token_available_actions"] += 1
        else:
          stats["led_token_unavailable_actions"] += 1
        if chosen_color == led_color:
          stats["follow_led"] += 1
          if led_color_legal_count > 0:
            stats["follow_with_led_color_legal"] += 1
          else:
            stats["follow_with_no_led_color_legal"] += 1
        else:
          stats["switch_off_led"] += 1
          if chosen_color == "R":
            stats["switch_to_red"] += 1
          else:
            stats["switch_to_nonred"] += 1
          if red_legal_count > 0:
            stats["switch_with_red_legal"] += 1
            if chosen_color == "R":
              stats["switch_with_red_legal_chose_red"] += 1
          else:
            stats["switch_with_no_red_legal"] += 1
          if led_color_legal_count > 0:
            stats["switch_with_led_color_legal"] += 1
            if chosen_color == "R":
              stats["avoidable_switch_to_red"] += 1
          else:
            stats["switch_with_no_led_color_legal"] += 1
          if led_token_available:
            stats["switch_with_led_token_available"] += 1
          else:
            stats["switch_with_led_token_unavailable"] += 1
          if newly_lost_led_token:
            stats["switch_newly_loses_led_token"] += 1
            if red_legal_count > 0:
              stats["switch_newly_loses_led_token_with_red_legal"] += 1
            if chosen_color == "R":
              stats["switch_newly_loses_led_token_to_red"] += 1
              if red_legal_count > 0:
                stats[
                    "switch_newly_loses_led_token_with_red_legal_chose_red"
                ] += 1
            else:
              stats["switch_newly_loses_led_token_to_nonred"] += 1
            if bool(cert.get("responsible_red_win_switch", False)):
              stats["switch_newly_loses_led_token_responsible_red_win"] += 1
          pending_switches.append({
              "match": int(args._current_match),
              "round": int(round_index),
              "trick_number": int(getattr(state, "_trick_number", -1)),
              "player": player,
              "led_color": led_color,
              "chosen_color": chosen_color,
              "chosen_rank": action_rank(state, action),
              "action": action,
              "legal_count": int(len(legal)),
              "led_color_legal_count": int(led_color_legal_count),
              "led_token_available_before": bool(led_token_available),
              "newly_lost_led_token": bool(newly_lost_led_token),
              "responsible_red_win_switch": bool(
                  cert.get("responsible_red_win_switch", False)
              ),
              "red_legal_count": int(red_legal_count),
          })

    state.apply_action(action)
    pending_switches = finalize_pending_switches(state, pending_switches, stats)

  for event in pending_switches:
    stats["switch_unresolved_by_paradox"] += 1
    if event["chosen_color"] == "R":
      stats["red_switch_unresolved_by_paradox"] += 1
    if event["led_color_legal_count"] > 0:
      stats["avoidable_switch_unresolved_by_paradox"] += 1
    if event.get("newly_lost_led_token"):
      stats["switch_newly_loses_led_token_unresolved_by_paradox"] += 1
    record_sample(stats, event)
  return np.array(state.returns(), dtype=np.float32), state, stats


def merge_stats(target, source):
  for key, value in source.items():
    if isinstance(value, dict):
      nested = target.setdefault(key, {})
      for inner_key, inner_value in value.items():
        nested[inner_key] = int(nested.get(inner_key, 0)) + int(inner_value)
    elif isinstance(value, list):
      target_list = target.setdefault(key, [])
      for row in value:
        if len(target_list) < 12:
          target_list.append(row)
    else:
      target[key] = int(target.get(key, 0)) + int(value)


def add_rates(stats, players, matches, paradox_rounds):
  nonparadox = int(stats.get("nonred_led_nonparadox_actions", 0))
  opportunities = int(stats.get("nonred_led_opportunities", 0))
  switches = int(stats.get("switch_off_led", 0))
  avoidable = int(stats.get("switch_with_led_color_legal", 0))
  red_switches = int(stats.get("switch_to_red", 0))
  red_legal_switches = int(stats.get("switch_with_red_legal", 0))
  red_legal_chose_red = int(stats.get("switch_with_red_legal_chose_red", 0))
  avoidable_red = int(stats.get("avoidable_switch_to_red", 0))
  led_available_actions = int(stats.get("led_color_legal_actions", 0))
  led_unavailable_actions = int(stats.get("no_led_color_legal_actions", 0))
  follow_when_led_available = int(
      stats.get("follow_with_led_color_legal", 0)
  )
  switch_when_led_available = int(stats.get("switch_with_led_color_legal", 0))
  switch_when_led_unavailable = int(
      stats.get("switch_with_no_led_color_legal", 0)
  )
  led_token_available_actions = int(stats.get("led_token_available_actions", 0))
  led_token_unavailable_actions = int(
      stats.get("led_token_unavailable_actions", 0)
  )
  switch_with_led_token_available = int(
      stats.get("switch_with_led_token_available", 0)
  )
  switch_with_led_token_unavailable = int(
      stats.get("switch_with_led_token_unavailable", 0)
  )
  token_loss_switches = int(stats.get("switch_newly_loses_led_token", 0))
  token_loss_red_legal = int(
      stats.get("switch_newly_loses_led_token_with_red_legal", 0)
  )
  token_loss_red_legal_chose_red = int(
      stats.get("switch_newly_loses_led_token_with_red_legal_chose_red", 0)
  )
  token_loss_switch_wins = int(
      stats.get("switch_newly_loses_led_token_wins_trick", 0)
  )
  token_loss_switch_losses = int(
      stats.get("switch_newly_loses_led_token_loses_trick", 0)
  )
  token_loss_switch_unresolved = int(
      stats.get("switch_newly_loses_led_token_unresolved_by_paradox", 0)
  )
  token_loss_switch_resolved = token_loss_switch_wins + token_loss_switch_losses
  resolved_switches = int(stats.get("switch_wins_trick", 0)) + int(
      stats.get("switch_loses_trick", 0)
  )
  all_switch_outcomes = resolved_switches + int(
      stats.get("switch_unresolved_by_paradox", 0)
  )
  red_resolved_switches = int(stats.get("red_switch_wins_trick", 0)) + int(
      stats.get("red_switch_loses_trick", 0)
  )
  red_all_switch_outcomes = red_resolved_switches + int(
      stats.get("red_switch_unresolved_by_paradox", 0)
  )
  avoidable_resolved = int(stats.get("avoidable_switch_wins_trick", 0)) + (
      int(stats.get("avoidable_switch_loses_trick", 0))
  )
  stats["matches"] = int(matches)
  stats["players"] = int(players)
  stats["hand_rounds"] = int(matches) * int(players)
  stats["rounds_with_any_paradox"] = int(paradox_rounds)
  stats["same_policy_paradox_round_rate"] = round(
      float(paradox_rounds) / float(stats["hand_rounds"]), 4
  ) if stats["hand_rounds"] > 0 else None
  stats["same_policy_paradox_round_rate_pct"] = pct(
      paradox_rounds, stats["hand_rounds"]
  )
  stats["switch_rate_among_nonred_led_actions"] = pct(switches, nonparadox)
  stats["switch_rate_among_all_nonred_led_opportunities"] = pct(
      switches, opportunities
  )
  stats["red_rate_among_switches"] = pct(red_switches, switches)
  stats["red_available_rate_among_switches"] = pct(red_legal_switches, switches)
  stats["red_choice_rate_when_red_legal_on_switch"] = pct(
      red_legal_chose_red, red_legal_switches
  )
  stats["red_rate_among_avoidable_switches"] = pct(avoidable_red, avoidable)
  stats["led_available_rate_among_nonred_led_actions"] = pct(
      led_available_actions, nonparadox
  )
  stats["switch_rate_when_led_available"] = pct(
      switch_when_led_available, led_available_actions
  )
  stats["switch_rate_when_led_unavailable"] = pct(
      switch_when_led_unavailable, led_unavailable_actions
  )
  stats["follow_rate_when_led_available"] = pct(
      follow_when_led_available, led_available_actions
  )
  stats["led_token_available_rate_among_nonred_led_actions"] = pct(
      led_token_available_actions, nonparadox
  )
  stats["switch_rate_when_led_token_available"] = pct(
      switch_with_led_token_available, led_token_available_actions
  )
  stats["switch_rate_when_led_token_unavailable"] = pct(
      switch_with_led_token_unavailable, led_token_unavailable_actions
  )
  stats["newly_loses_led_token_rate_among_switches"] = pct(
      token_loss_switches, switches
  )
  stats["red_rate_among_new_led_token_loss_switches"] = pct(
      int(stats.get("switch_newly_loses_led_token_to_red", 0)),
      token_loss_switches,
  )
  stats["red_available_rate_among_new_led_token_loss_switches"] = pct(
      token_loss_red_legal, token_loss_switches
  )
  stats["red_choice_rate_when_red_legal_on_new_led_token_loss_switch"] = pct(
      token_loss_red_legal_chose_red, token_loss_red_legal
  )
  stats["responsible_red_win_rate_among_new_led_token_loss_switches"] = pct(
      int(stats.get("switch_newly_loses_led_token_responsible_red_win", 0)),
      token_loss_switches,
  )
  stats["new_led_token_loss_switch_win_rate_resolved"] = pct(
      token_loss_switch_wins, token_loss_switch_resolved
  )
  stats["new_led_token_loss_switch_win_rate_counting_unresolved_as_loss"] = pct(
      token_loss_switch_wins,
      token_loss_switch_resolved + token_loss_switch_unresolved,
  )
  stats["switch_win_rate_resolved"] = pct(
      int(stats.get("switch_wins_trick", 0)), resolved_switches
  )
  stats["switch_win_rate_counting_unresolved_as_loss"] = pct(
      int(stats.get("switch_wins_trick", 0)), all_switch_outcomes
  )
  stats["red_switch_win_rate_resolved"] = pct(
      int(stats.get("red_switch_wins_trick", 0)), red_resolved_switches
  )
  stats["red_switch_win_rate_counting_unresolved_as_loss"] = pct(
      int(stats.get("red_switch_wins_trick", 0)), red_all_switch_outcomes
  )
  stats["avoidable_switch_win_rate_resolved"] = pct(
      int(stats.get("avoidable_switch_wins_trick", 0)), max(0, avoidable_resolved)
  )
  stats["follow_rate_among_nonred_led_actions"] = pct(
      int(stats.get("follow_led", 0)), nonparadox
  )
  return stats


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=100)
  parser.add_argument("--seed", type=int, default=20270609)
  parser.add_argument("--checkpoint", default="")
  parser.add_argument(
      "--mode",
      default="policy",
      help=(
          "Neural selector mode. Supports full quantum_cat_full_match_elo "
          "payload suffixes, e.g. policy, q_policy, or "
          "'liveness_shield?liveness_shield_base_mode=policy,...'."
      ),
  )
  parser.add_argument("--bot-name", default="bot")
  parser.add_argument("--builtin-bot", default="heuristic")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--belief-samples", type=int, default=4)
  parser.add_argument("--belief-sims", type=int, default=8)
  parser.add_argument("--mcts-sims", type=int, default=16)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument("--action-value-rerank-phases", default="")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
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
  parser.add_argument("--out", default="experiments/led_switch_audit_latest.json")
  return parser.parse_args()


def main():
  args = parse_args()
  if args.players < 2 or args.players > 5:
    raise ValueError("--players must be 2..5")
  device = torch.device(args.device)
  use_match_context = bool(args.match_context)

  all_stats = empty_stats()
  paradox_rounds = 0
  bots, neural_bots = make_same_policy_bots(args, device=device)
  if neural_bots:
    use_match_context = use_match_context or any(
        bool(getattr(entry["args"], "match_context", False))
        for entry in neural_bots.values()
    )
  for match_idx in range(args.matches):
    totals = np.zeros(args.players, dtype=np.float32)
    args._current_match = match_idx  # pylint: disable=protected-access
    for round_idx in range(args.players):
      start_player = (match_idx + round_idx) % args.players
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": args.players,
              "start_player": start_player,
              "match_context": int(use_match_context),
          },
      )
      prior_totals = np.copy(totals)
      returns, state, round_stats = play_round_with_audit(
          game,
          bots,
          args,
          seed=args.seed + match_idx * 1009 + round_idx * 37,
          match_totals=prior_totals if use_match_context else None,
          round_index=round_idx,
      )
      if use_match_context:
        totals = returns
      else:
        totals += returns
      if any(bool(value) for value in getattr(state, "_has_paradoxed", [])):
        paradox_rounds += 1
      merge_stats(all_stats, round_stats)

  result = {
      "checkpoint": args.checkpoint or None,
      "builtin_bot": None if args.checkpoint else args.builtin_bot,
      "mode": args.mode if args.checkpoint else "builtin",
      "match_context": use_match_context,
      "seed": int(args.seed),
      "stats": add_rates(all_stats, args.players, args.matches, paradox_rounds),
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
