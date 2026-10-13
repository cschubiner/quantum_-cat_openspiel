#!/usr/bin/env python3
"""Information-safe, deployable public play-history features for Quantum Cat.

The normal observation tensor contains the current public board and token state,
but it intentionally does not retain the order in which public cards were
played.  This module derives a compact sequence summary from the game state.
The same functions are used by replay collection, offline training, and live
inference so a history-aware model cannot depend on training-only row order.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


PUBLIC_HISTORY_VERSION = "quantum_cat_public_history_v1"
_COLORS = ("R", "B", "Y", "G")

PUBLIC_HISTORY_SCHEMA = (
    "round_position_norm",
    "public_play_progress_norm",
    "completed_tricks_norm",
    "current_trick_play_count_norm",
    "prior_follow_rate",
    "prior_off_led_rate",
    "prior_red_rate",
    "prior_off_led_red_rate",
    "prior_off_led_nonred_rate",
    "own_prior_play_count_norm",
    "own_prior_follow_rate",
    "own_prior_off_led_rate",
    "own_prior_red_rate",
    "others_prior_off_led_rate",
    "own_completed_win_rate",
    "last_trick_own_winner",
    "last_trick_own_leader",
    "last_trick_own_follow",
    "last_trick_own_off_led",
    "last_trick_had_red",
    "previous_play_own",
    "previous_play_follow",
    "previous_play_off_led",
    "previous_play_red",
    "last_trick_led_red",
    "last_trick_led_blue",
    "last_trick_led_yellow",
    "last_trick_led_green",
    "previous_play_red_color",
    "previous_play_blue_color",
    "previous_play_yellow_color",
    "previous_play_green_color",
)

PUBLIC_HISTORY_INTERACTION_SCHEMA = (
    "candidate_token_loss_x_prior_off_led_rate",
    "candidate_nonred_token_loss_x_prior_off_led_nonred_rate",
    "candidate_token_loss_x_own_prior_off_led_rate",
    "candidate_token_loss_x_others_prior_off_led_rate",
    "candidate_token_loss_x_previous_play_off_led",
    "candidate_nonred_token_loss_x_previous_play_red",
    "candidate_not_follow_x_prior_follow_rate",
    "candidate_red_token_loss_x_prior_off_led_rate",
    "candidate_follows_x_prior_off_led_rate",
    "candidate_token_loss_x_round_position_norm",
    "candidate_nonred_token_loss_x_round_position_norm",
)

PUBLIC_HISTORY_ACTION_SCHEMA = (
    *PUBLIC_HISTORY_SCHEMA,
    *PUBLIC_HISTORY_INTERACTION_SCHEMA,
)


def _rate(numerator: float, denominator: float) -> float:
  return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def _card_map(trick: Iterable) -> dict[int, tuple[int, str]]:
  result = {}
  for player, card in trick:
    if card is None:
      continue
    rank, color = card
    result[int(player)] = (int(rank), str(color))
  return result


def _ordered_trick_events(
    trick: Iterable,
    start_player: int,
    players: int,
) -> list[dict]:
  cards = _card_map(trick)
  if not cards:
    return []
  order = [
      int((int(start_player) + offset) % int(players))
      for offset in range(int(players))
      if int((int(start_player) + offset) % int(players)) in cards
  ]
  led_color = str(cards[order[0]][1])
  return [
      {
          "player": int(actor),
          "rank": int(cards[actor][0]),
          "color": str(cards[actor][1]),
          "led_color": led_color,
          "follows_led": bool(str(cards[actor][1]) == led_color),
          "off_led": bool(str(cards[actor][1]) != led_color),
          "red": bool(str(cards[actor][1]) == "R"),
      }
      for actor in order
  ]


def _trick_winner(events: list[dict]) -> int:
  if not events:
    return -1
  red = [event for event in events if bool(event["red"])]
  eligible = red or [
      event for event in events
      if str(event["color"]) == str(events[0]["led_color"])
  ]
  return int(max(eligible, key=lambda event: int(event["rank"]))["player"])


def public_history_features(state, player: int) -> np.ndarray:
  """Return a fixed-width public sequence summary for ``player``.

  Only completed tricks and the cards already exposed in the current trick are
  read.  Hands and face-down discards are never inspected.
  """
  player = int(player)
  players = int(getattr(state, "_num_players"))
  num_tricks = max(1, int(getattr(state, "_num_tricks", 1)))
  completed = list(getattr(state, "_completed_tricks", []) or [])
  round_start = int(getattr(state, "_round_start_player", 0))

  completed_events = []
  completed_summaries = []
  trick_start = round_start
  for trick in completed:
    events = _ordered_trick_events(trick, trick_start, players)
    if not events:
      continue
    winner = _trick_winner(events)
    completed_events.extend(events)
    completed_summaries.append({
        "events": events,
        "leader": int(trick_start),
        "winner": int(winner),
        "led_color": str(events[0]["led_color"]),
        "had_red": bool(any(bool(event["red"]) for event in events)),
    })
    trick_start = int(winner)

  current_cards = [
      (idx, card)
      for idx, card in enumerate(
          list(getattr(state, "_cards_played_this_trick", []) or [])
      )
      if card is not None
  ]
  current_start = int(getattr(state, "_start_player", trick_start))
  current_events = _ordered_trick_events(current_cards, current_start, players)
  events = [*completed_events, *current_events]

  own = [event for event in events if int(event["player"]) == player]
  others = [event for event in events if int(event["player"]) != player]
  follow = sum(int(bool(event["follows_led"])) for event in events)
  off_led = sum(int(bool(event["off_led"])) for event in events)
  red = sum(int(bool(event["red"])) for event in events)
  off_led_red = sum(
      int(bool(event["off_led"]) and bool(event["red"])) for event in events
  )
  off_led_nonred = sum(
      int(bool(event["off_led"]) and not bool(event["red"])) for event in events
  )
  own_follow = sum(int(bool(event["follows_led"])) for event in own)
  own_off_led = sum(int(bool(event["off_led"])) for event in own)
  own_red = sum(int(bool(event["red"])) for event in own)
  others_off_led = sum(int(bool(event["off_led"])) for event in others)

  last = completed_summaries[-1] if completed_summaries else None
  last_own_event = None
  if last is not None:
    last_own_event = next(
        (
            event for event in last["events"]
            if int(event["player"]) == player
        ),
        None,
    )
  previous = events[-1] if events else None
  last_led = str(last["led_color"]) if last is not None else None
  previous_color = str(previous["color"]) if previous is not None else None

  values = [
      min(1.0, float(getattr(state, "_trick_number", len(completed))) / num_tricks),
      min(1.0, _rate(len(events), num_tricks * players)),
      min(1.0, _rate(len(completed_summaries), num_tricks)),
      min(1.0, _rate(len(current_events), players)),
      _rate(follow, len(events)),
      _rate(off_led, len(events)),
      _rate(red, len(events)),
      _rate(off_led_red, len(events)),
      _rate(off_led_nonred, len(events)),
      min(1.0, _rate(len(own), num_tricks)),
      _rate(own_follow, len(own)),
      _rate(own_off_led, len(own)),
      _rate(own_red, len(own)),
      _rate(others_off_led, len(others)),
      _rate(
          sum(int(int(item["winner"]) == player) for item in completed_summaries),
          len(completed_summaries),
      ),
      float(last is not None and int(last["winner"]) == player),
      float(last is not None and int(last["leader"]) == player),
      float(last_own_event is not None and bool(last_own_event["follows_led"])),
      float(last_own_event is not None and bool(last_own_event["off_led"])),
      float(last is not None and bool(last["had_red"])),
      float(previous is not None and int(previous["player"]) == player),
      float(previous is not None and bool(previous["follows_led"])),
      float(previous is not None and bool(previous["off_led"])),
      float(previous is not None and bool(previous["red"])),
      *[float(last_led == color) for color in _COLORS],
      *[float(previous_color == color) for color in _COLORS],
  ]
  result = np.asarray(values, dtype=np.float32)
  if result.shape != (len(PUBLIC_HISTORY_SCHEMA),):
    raise ValueError(
        "public history feature shape mismatch: "
        f"{result.shape} vs {(len(PUBLIC_HISTORY_SCHEMA),)}"
    )
  return result


def public_history_metadata(state, player: int) -> dict:
  features = public_history_features(state, player)
  return {
      "version": PUBLIC_HISTORY_VERSION,
      "schema": list(PUBLIC_HISTORY_SCHEMA),
      "features": [float(value) for value in features],
  }


def public_history_action_features(
    history_features,
    *,
    candidate_follows_led: bool,
    candidate_is_red: bool,
    candidate_token_loss: bool,
) -> np.ndarray:
  """Append candidate/history interactions to a public history summary."""
  history = np.asarray(history_features, dtype=np.float32)
  if history.shape != (len(PUBLIC_HISTORY_SCHEMA),):
    raise ValueError(
        "public history feature shape mismatch: "
        f"{history.shape} vs {(len(PUBLIC_HISTORY_SCHEMA),)}"
    )
  idx = {name: i for i, name in enumerate(PUBLIC_HISTORY_SCHEMA)}
  follows = bool(candidate_follows_led)
  red = bool(candidate_is_red)
  loses = bool(candidate_token_loss)
  nonred_loss = bool(loses and not red)
  red_loss = bool(loses and red)
  interactions = np.asarray([
      float(loses) * float(history[idx["prior_off_led_rate"]]),
      float(nonred_loss) * float(history[idx["prior_off_led_nonred_rate"]]),
      float(loses) * float(history[idx["own_prior_off_led_rate"]]),
      float(loses) * float(history[idx["others_prior_off_led_rate"]]),
      float(loses) * float(history[idx["previous_play_off_led"]]),
      float(nonred_loss) * float(history[idx["previous_play_red"]]),
      float(not follows) * float(history[idx["prior_follow_rate"]]),
      float(red_loss) * float(history[idx["prior_off_led_rate"]]),
      float(follows) * float(history[idx["prior_off_led_rate"]]),
      float(loses) * float(history[idx["round_position_norm"]]),
      float(nonred_loss) * float(history[idx["round_position_norm"]]),
  ], dtype=np.float32)
  return np.concatenate([history, interactions]).astype(np.float32)
