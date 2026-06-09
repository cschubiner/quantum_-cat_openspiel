"""Bot utilities for Cat in the Box / python_quantum_cat."""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Iterable, Protocol

import numpy as np
import pyspiel

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import


PARADOX = 999
PREDICT_OFFSET = 100


def prediction_top_rank_score(state, player: int) -> float:
  """Expected-trick bid score from top-rank cards.

  The shared bidding rule counts each highest-rank card as one expected trick.
  The second-highest rank is only a fractional kicker, with extra credit for
  the round's first leader because that player has more control over the first
  trick.
  """
  hand = np.asarray(state._hands[int(player)], dtype=np.float32)
  num_ranks = int(getattr(state, "_num_card_types", len(hand)))
  if num_ranks <= 0 or hand.size == 0:
    return 0.0
  top_idx = min(num_ranks, hand.size) - 1
  second_idx = top_idx - 1
  score = float(hand[top_idx])
  if second_idx >= 0:
    start_player = int(
        getattr(
            state,
            "_round_start_player",
            getattr(state, "_start_player", -1),
        )
    )
    second_weight = 0.50 if int(player) == start_player else 0.25
    score += second_weight * float(hand[second_idx])
  return score


def shared_prediction_action(state, player: int, legal: Iterable[int] | None = None):
  """Returns the shared heuristic prediction action, or None outside bidding.

  All bot families use this for phase-2 bidding so prediction behavior is not a
  moving target while we train the play policy.
  """
  if int(getattr(state, "_phase", -1)) != 2:
    return None
  legal_actions = list(state.legal_actions(player) if legal is None else legal)
  prediction_actions = [
      int(action)
      for action in legal_actions
      if PREDICT_OFFSET < int(action) <= PREDICT_OFFSET + 4
  ]
  if not prediction_actions:
    return None
  guess = int(math.floor(prediction_top_rank_score(state, player) + 0.5))
  guess = max(1, min(4, guess))
  action = PREDICT_OFFSET + guess
  if action in prediction_actions:
    return action
  return min(prediction_actions, key=lambda a: abs((a - PREDICT_OFFSET) - guess))


class QuantumCatBot(Protocol):
  name: str

  def step(self, state: pyspiel.State, player: int) -> int:
    ...


@dataclasses.dataclass
class RandomBot:
  name: str = "random"
  seed: int = 0

  def __post_init__(self):
    self._rng = random.Random(self.seed)

  def step(self, state, player):
    legal = state.legal_actions(player)
    prediction_action = shared_prediction_action(state, player, legal)
    if prediction_action is not None:
      return prediction_action
    return self._rng.choice(legal)


@dataclasses.dataclass
class HeuristicBot:
  """Fast rule-aware baseline used before neural self-play is ready."""

  name: str = "heuristic"
  seed: int = 0
  adjacency_weight: float = 1.0
  target_weight: float = 1.5
  winner_weight: float = 1.0
  future_flex_weight: float = 0.0
  rank_deficit_weight: float = 0.0
  token_loss_weight: float = 0.0
  discard_future_flex_weight: float = 0.0
  discard_rank_deficit_weight: float = 0.0
  future_buffer_weight: float = 0.0
  discard_future_buffer_weight: float = 0.0
  future_slot_buffer: int = 1
  future_no_exit_weight: float = 0.0
  token_loss_pressure_weight: float = 0.0
  token_loss_no_exit_weight: float = 0.0
  rank_slot_urgency_weight: float = 0.0
  rank_slot_urgency_buffer: int = 1
  prediction_high_card_weight: float = 0.6
  prediction_duplicate_weight: float = 0.25
  prediction_bias: float = 0.0

  def __post_init__(self):
    self._rng = random.Random(self.seed)

  def step(self, state, player):
    legal = state.legal_actions(player)
    if len(legal) == 1:
      return legal[0]
    phase = state._phase
    if phase == 1:
      return self._discard(state, player, legal)
    if phase == 2:
      return self._predict(state, player, legal)
    return self._trick_action(state, player, legal)

  def _discard(self, state, player, legal):
    hand = state._hands[player]
    if (
        self.discard_future_flex_weight
        or self.discard_rank_deficit_weight
        or self.discard_future_buffer_weight
    ):
      scored = [
          (self._score_discard_action(state, player, rank_idx), rank_idx)
          for rank_idx in legal
      ]
      best = max(score for score, _ in scored)
      candidates = [rank_idx for score, rank_idx in scored if abs(score - best) < 1e-9]
      return self._rng.choice(candidates)
    # Keep singletons as escape hatches; discard from the most duplicated rank,
    # breaking ties toward low ranks that are less useful for winning tricks.
    return max(legal, key=lambda rank_idx: (hand[rank_idx], -rank_idx))

  def _score_discard_action(self, state, player, rank_idx):
    hand = state._hands[player]
    pressure = self._future_pressure_after(state, player, rank_idx)
    total_slots = pressure["total_slots"]
    rank_deficit = pressure["rank_deficit"]
    tight_ranks = pressure["tight_ranks"]
    buffer_deficit = pressure["buffer_deficit"]
    score = (
        self.discard_future_flex_weight * (0.20 * total_slots - tight_ranks)
        - self.discard_rank_deficit_weight * rank_deficit
        - self.discard_future_buffer_weight * buffer_deficit
    )
    score += 0.25 * float(hand[rank_idx])
    score -= 0.02 * float(rank_idx)
    return score

  def _score_future_pressure(self, state, player, action):
    pressure = self._future_pressure_after(state, player, action)
    return self._score_pressure(pressure)

  def _score_pressure(self, pressure):
    no_exit_penalty = (
        self.future_no_exit_weight if pressure.get("no_exit", False) else 0.0
    )
    return (
        self.future_flex_weight *
        (0.20 * pressure["total_slots"] - pressure["tight_ranks"])
        - self.rank_deficit_weight * pressure["rank_deficit"]
        - self.future_buffer_weight * pressure["buffer_deficit"]
        - no_exit_penalty
    )

  def _future_flexibility_after(self, state, player, action):
    pressure = self._future_pressure_after(state, player, action)
    return (
        pressure["total_slots"],
        pressure["rank_deficit"],
        pressure["tight_ranks"],
    )

  def _future_pressure_after(self, state, player, action):
    clone = state.clone()
    clone.apply_action(action)
    return self._pressure_for_position(
        clone,
        clone._hands[player],
        clone._color_tokens[player],
        clone._board_ownership,
    )

  def _pressure_for_position(self, state, hand, tokens, board):
    total_slots = 0
    rank_deficit = 0
    tight_ranks = 0
    buffer_deficit = 0
    for rank_idx, count in enumerate(hand):
      count = int(count)
      if count <= 0:
        continue
      available = 0
      for color_idx in range(state._num_colors):
        if tokens[color_idx] and board[color_idx, rank_idx] == -1:
          available += 1
      total_slots += available
      deficit = max(0, count - available)
      rank_deficit += deficit
      buffer_deficit += max(0, count + int(self.future_slot_buffer) - available)
      if available <= count:
        tight_ranks += 1
    remaining_cards = int(sum(max(0, int(count)) for count in hand))
    return {
        "total_slots": total_slots,
        "rank_deficit": rank_deficit,
        "tight_ranks": tight_ranks,
        "buffer_deficit": buffer_deficit,
        "no_exit": remaining_cards > 0 and total_slots <= 0,
    }

  def _rank_open_slots(self, state, player, rank_idx):
    tokens = state._color_tokens[player]
    board = state._board_ownership
    available = 0
    for color_idx in range(state._num_colors):
      if tokens[color_idx] and board[color_idx, rank_idx] == -1:
        available += 1
    return available

  def _score_rank_slot_urgency(self, state, player, rank_idx):
    if self.rank_slot_urgency_weight <= 0.0:
      return 0.0
    hand_count = int(state._hands[player][rank_idx])
    if hand_count <= 0:
      return 0.0
    open_slots = self._rank_open_slots(state, player, rank_idx)
    if open_slots <= 0:
      return 0.0
    buffer = max(0, int(self.rank_slot_urgency_buffer))
    pressure = max(0, hand_count + buffer - open_slots)
    scarcity = hand_count / max(1.0, float(open_slots))
    return self.rank_slot_urgency_weight * (pressure + scarcity)

  def _score_token_loss_pressure(self, state, player, action):
    if (
        self.token_loss_pressure_weight <= 0.0
        and self.token_loss_no_exit_weight <= 0.0
    ):
      return 0.0
    if action == PARADOX or state._led_color is None:
      return 0.0
    color_idx = action // state._num_card_types
    if color_idx < 0 or color_idx >= state._num_colors:
      return 0.0
    color_names = ["R", "B", "Y", "G"]
    if state._led_color not in color_names:
      return 0.0
    color = color_names[color_idx]
    if color == state._led_color:
      return 0.0
    lost_idx = color_names.index(state._led_color)
    if not bool(state._color_tokens[player][lost_idx]):
      return 0.0

    clone = state.clone()
    clone.apply_action(action)
    actual_pressure = self._pressure_for_position(
        clone,
        clone._hands[player],
        clone._color_tokens[player],
        clone._board_ownership,
    )
    restored_tokens = np.copy(clone._color_tokens[player])
    restored_tokens[lost_idx] = True
    restored_pressure = self._pressure_for_position(
        clone,
        clone._hands[player],
        restored_tokens,
        clone._board_ownership,
    )
    pressure_damage = max(
        0.0, self._score_pressure(restored_pressure) -
        self._score_pressure(actual_pressure)
    )
    score = -self.token_loss_pressure_weight * pressure_damage
    if actual_pressure["no_exit"] and not restored_pressure["no_exit"]:
      score -= self.token_loss_no_exit_weight
    return score

  def _predict(self, state, player, legal):
    action = shared_prediction_action(state, player, legal)
    if action is None:
      raise ValueError(f"No legal prediction action for player {player}: {legal}")
    return action

  def _trick_action(self, state, player, legal):
    scored = [(self._score_trick_action(state, player, action), action)
              for action in legal]
    best = max(score for score, _ in scored)
    candidates = [action for score, action in scored if abs(score - best) < 1e-9]
    return self._rng.choice(candidates)

  def _score_trick_action(self, state, player, action):
    if action == PARADOX:
      return -1000.0

    color_idx = action // state._num_card_types
    rank_idx = action % state._num_card_types
    rank = rank_idx + 1
    color = ["R", "B", "Y", "G"][color_idx]
    score = 0.0

    tricks = int(state._tricks_won[player])
    prediction = state._predictions[player]
    wants_more = prediction < 0 or tricks < prediction

    would_win_now = self._would_win_after_play(state, player, rank, color)
    if would_win_now is not None:
      score += self.winner_weight * (1.0 if would_win_now == wants_more else -1.0)

    score += self.adjacency_weight * self._adjacency_gain(state, player, color_idx, rank_idx)

    if (
        self.future_flex_weight
        or self.rank_deficit_weight
        or self.future_buffer_weight
        or self.future_no_exit_weight
    ):
      score += self._score_future_pressure(state, player, action)

    score += self._score_rank_slot_urgency(state, player, rank_idx)

    if self.token_loss_weight and state._led_color is not None and color != state._led_color:
      score -= self.token_loss_weight
    score += self._score_token_loss_pressure(state, player, action)

    # Preserve high cards when still below target; shed them when already at target.
    highness = rank / max(1, state._num_card_types)
    score += self.target_weight * (highness if wants_more else -highness)

    # Keep red/trump flexible early unless it wins a needed trick.
    if color == "R" and state._led_color is None and not state._trump_broken:
      score -= 0.25
    return score

  def _would_win_after_play(self, state, player, rank, color):
    plays = list(state._cards_played_this_trick)
    plays[player] = (rank, color)
    if any(card is None for card in plays):
      return None
    red = [(p, card[0]) for p, card in enumerate(plays) if card[1] == "R"]
    if red:
      winner = max(red, key=lambda item: item[1])[0]
      return winner == player
    led = state._led_color or color
    led_plays = [(p, card[0]) for p, card in enumerate(plays) if card[1] == led]
    winner = max(led_plays, key=lambda item: item[1])[0]
    return winner == player

  def _adjacency_gain(self, state, player, color_idx, rank_idx):
    before = _largest_cluster(state._board_ownership, player)
    board = np.copy(state._board_ownership)
    board[color_idx, rank_idx] = player
    after = _largest_cluster(board, player)
    return after - before


def base_bot_name(name: str) -> str:
  """Returns the built-in bot name behind an optional evaluator alias."""
  if "=" in name:
    _alias, base = name.split("=", 1)
    return base
  return name


def make_bot(name: str, seed: int = 0) -> QuantumCatBot:
  name = base_bot_name(name)
  if name == "random":
    return RandomBot(seed=seed)
  if name == "heuristic":
    return HeuristicBot(seed=seed)
  if name == "heuristic_adj2":
    return HeuristicBot(name=name, seed=seed, adjacency_weight=2.0)
  if name == "heuristic_target2":
    return HeuristicBot(name=name, seed=seed, target_weight=2.0)
  if name == "heuristic_safe":
    return HeuristicBot(
        name=name,
        seed=seed,
        target_weight=1.0,
        future_flex_weight=1.0,
        rank_deficit_weight=8.0,
        token_loss_weight=1.5,
    )
  if name == "heuristic_safe2":
    return HeuristicBot(
        name=name,
        seed=seed,
        target_weight=0.75,
        winner_weight=0.5,
        future_flex_weight=2.0,
        rank_deficit_weight=14.0,
        token_loss_weight=3.0,
    )
  if name == "heuristic_safe3":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.5,
        target_weight=0.25,
        winner_weight=0.0,
        future_flex_weight=4.0,
        rank_deficit_weight=30.0,
        token_loss_weight=6.0,
    )
  if name == "heuristic_safe4":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.25,
        target_weight=0.0,
        winner_weight=0.0,
        future_flex_weight=8.0,
        rank_deficit_weight=60.0,
        token_loss_weight=10.0,
    )
  if name == "heuristic_safe5":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.75,
        target_weight=0.25,
        winner_weight=0.25,
        future_flex_weight=4.0,
        rank_deficit_weight=30.0,
        token_loss_weight=12.0,
    )
  if name == "heuristic_safe6":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=1.0,
        target_weight=0.5,
        winner_weight=0.0,
        future_flex_weight=3.0,
        rank_deficit_weight=30.0,
        token_loss_weight=12.0,
    )
  if name == "heuristic_safe7":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.5,
        target_weight=0.25,
        winner_weight=0.0,
        future_flex_weight=4.0,
        rank_deficit_weight=30.0,
        token_loss_weight=6.0,
        discard_future_flex_weight=4.0,
        discard_rank_deficit_weight=30.0,
    )
  if name == "heuristic_safe8":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.5,
        target_weight=0.25,
        winner_weight=0.0,
        future_flex_weight=5.0,
        rank_deficit_weight=40.0,
        token_loss_weight=8.0,
        discard_future_flex_weight=5.0,
        discard_rank_deficit_weight=40.0,
    )
  if name == "heuristic_safe9":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.5,
        target_weight=0.25,
        winner_weight=0.0,
        future_flex_weight=4.0,
        rank_deficit_weight=30.0,
        future_buffer_weight=10.0,
        token_loss_weight=6.0,
        discard_future_flex_weight=4.0,
        discard_rank_deficit_weight=30.0,
        discard_future_buffer_weight=10.0,
        future_slot_buffer=1,
    )
  if name == "heuristic_safe10":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.25,
        target_weight=0.0,
        winner_weight=0.0,
        future_flex_weight=5.0,
        rank_deficit_weight=40.0,
        future_buffer_weight=16.0,
        token_loss_weight=8.0,
        discard_future_flex_weight=5.0,
        discard_rank_deficit_weight=40.0,
        discard_future_buffer_weight=16.0,
        future_slot_buffer=1,
    )
  if name == "heuristic_safe11":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=1.0,
        target_weight=0.75,
        winner_weight=0.5,
        future_flex_weight=3.0,
        rank_deficit_weight=16.0,
        token_loss_weight=12.0,
        future_buffer_weight=4.0,
        discard_future_buffer_weight=4.0,
        future_slot_buffer=1,
        future_no_exit_weight=20.0,
        token_loss_pressure_weight=0.5,
        token_loss_no_exit_weight=40.0,
    )
  if name == "heuristic_safe12":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=1.0,
        target_weight=0.75,
        winner_weight=0.5,
        future_flex_weight=3.0,
        rank_deficit_weight=16.0,
        token_loss_weight=12.0,
        future_buffer_weight=4.0,
        discard_future_buffer_weight=4.0,
        future_slot_buffer=1,
        rank_slot_urgency_weight=8.0,
        rank_slot_urgency_buffer=1,
    )
  if name == "heuristic_safe13":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=1.0,
        target_weight=0.75,
        winner_weight=0.5,
        future_flex_weight=3.0,
        rank_deficit_weight=16.0,
        token_loss_weight=12.0,
        future_buffer_weight=4.0,
        discard_future_buffer_weight=4.0,
        future_slot_buffer=1,
        prediction_bias=1.0,
    )
  if name == "heuristic_safe14":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=1.0,
        target_weight=0.75,
        winner_weight=0.5,
        future_flex_weight=3.0,
        rank_deficit_weight=16.0,
        token_loss_weight=12.0,
        future_buffer_weight=4.0,
        discard_future_buffer_weight=4.0,
        future_slot_buffer=1,
        prediction_high_card_weight=0.2,
        prediction_duplicate_weight=0.0,
        prediction_bias=-0.5,
    )
  if name == "heuristic_safe15":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=1.0,
        target_weight=0.75,
        winner_weight=0.5,
        future_flex_weight=3.0,
        rank_deficit_weight=16.0,
        token_loss_weight=12.0,
        future_buffer_weight=4.0,
        discard_future_buffer_weight=4.0,
        future_slot_buffer=1,
        prediction_high_card_weight=0.4,
        prediction_duplicate_weight=0.2,
        prediction_bias=-1.0,
    )
  if name == "heuristic_safe16":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.9,
        target_weight=0.5,
        winner_weight=0.35,
        future_flex_weight=4.0,
        rank_deficit_weight=24.0,
        token_loss_weight=16.0,
        future_buffer_weight=8.0,
        discard_future_flex_weight=5.0,
        discard_rank_deficit_weight=40.0,
        discard_future_buffer_weight=12.0,
        future_slot_buffer=1,
        future_no_exit_weight=80.0,
        token_loss_pressure_weight=2.0,
        token_loss_no_exit_weight=80.0,
        rank_slot_urgency_weight=10.0,
        rank_slot_urgency_buffer=1,
        prediction_high_card_weight=0.2,
        prediction_duplicate_weight=0.0,
        prediction_bias=-0.75,
    )
  if name == "heuristic_safe17":
    return HeuristicBot(
        name=name,
        seed=seed,
        adjacency_weight=0.6,
        target_weight=0.15,
        winner_weight=0.10,
        future_flex_weight=6.0,
        rank_deficit_weight=36.0,
        token_loss_weight=20.0,
        future_buffer_weight=14.0,
        discard_future_flex_weight=6.0,
        discard_rank_deficit_weight=48.0,
        discard_future_buffer_weight=16.0,
        future_slot_buffer=2,
        future_no_exit_weight=140.0,
        token_loss_pressure_weight=4.0,
        token_loss_no_exit_weight=140.0,
        rank_slot_urgency_weight=16.0,
        rank_slot_urgency_buffer=2,
        prediction_high_card_weight=0.15,
        prediction_duplicate_weight=0.0,
        prediction_bias=-1.0,
    )
  raise ValueError(f"Unknown bot: {name}")


def play_game(game, bots: list[QuantumCatBot], seed: int | None = None):
  if seed is not None:
    np.random.seed(seed)
  state = game.new_initial_state()
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      action = int(np.random.choice(actions, p=probs))
    else:
      player = state.current_player()
      action = shared_prediction_action(state, player)
      if action is None:
        action = bots[player].step(state, player)
    state.apply_action(action)
  return state.returns(), state


def multiplayer_elo_update(
    ratings: dict[str, float],
    bot_names: list[str],
    returns: Iterable[float],
    k_factor: float,
) -> None:
  returns = list(returns)
  deltas = {name: 0.0 for name in bot_names}
  counts = {name: 0 for name in bot_names}
  for i in range(len(bot_names)):
    for j in range(i + 1, len(bot_names)):
      name_i = bot_names[i]
      name_j = bot_names[j]
      expected_i = 1.0 / (1.0 + 10 ** ((ratings[name_j] - ratings[name_i]) / 400.0))
      if returns[i] > returns[j]:
        actual_i = 1.0
      elif returns[i] < returns[j]:
        actual_i = 0.0
      else:
        actual_i = 0.5
      change = k_factor * (actual_i - expected_i)
      deltas[name_i] += change
      deltas[name_j] -= change
      counts[name_i] += 1
      counts[name_j] += 1
  for name in bot_names:
    if counts[name]:
      ratings[name] += deltas[name] / counts[name]


def _largest_cluster(board, player):
  seen = np.zeros(board.shape, dtype=bool)
  best = 0
  for c in range(board.shape[0]):
    for r in range(board.shape[1]):
      if seen[c, r] or board[c, r] != player:
        continue
      stack = [(c, r)]
      seen[c, r] = True
      size = 0
      while stack:
        cc, rr = stack.pop()
        size += 1
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
          nc, nr = cc + dc, rr + dr
          if 0 <= nc < board.shape[0] and 0 <= nr < board.shape[1]:
            if not seen[nc, nr] and board[nc, nr] == player:
              seen[nc, nr] = True
              stack.append((nc, nr))
      best = max(best, size)
  return best
