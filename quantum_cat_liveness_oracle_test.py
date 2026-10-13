"""Tests for quantum_cat_liveness_oracle.

Validates the pure core against brute force, the state adapter against the
game's own legality logic, and the soundness guarantee ("oracle says dead =>
the player cannot finish the round unparadoxed") on random rollouts.
"""

import itertools
import random
import unittest

import numpy as np
import pyspiel

from open_spiel.python.games import quantum_cat  # registers python_quantum_cat
import quantum_cat_liveness_oracle as oracle


def brute_force_matching(cards, cells):
  """Max matching: cards is a list of ranks, cells a list of (color, rank)."""
  best = 0
  n = len(cards)
  cells = list(cells)

  def rec(i, used, placed):
    nonlocal best
    best = max(best, placed)
    if i >= n or placed + (n - i) <= best:
      return
    rec(i + 1, used, placed)  # leave card i unplaced
    for j, (_, r) in enumerate(cells):
      if not used[j] and r == cards[i]:
        used[j] = True
        rec(i + 1, used, placed + 1)
        used[j] = False

  rec(0, [False] * len(cells), 0)
  return best


def random_instance(rng, num_ranks=4, max_copies=3, max_cells=8):
  counts = [rng.randint(0, max_copies) for _ in range(num_ranks)]
  cells = []
  for c in range(4):
    for r in range(num_ranks):
      if rng.random() < 0.4 and len(cells) < max_cells:
        cells.append((c, r))
  tokens = [rng.random() < 0.8 for _ in range(4)]
  return counts, cells, tokens


def caps_from(cells, tokens, num_ranks):
  caps = [0] * num_ranks
  for c, r in cells:
    if tokens[c]:
      caps[r] += 1
  return caps


class MatchingTest(unittest.TestCase):

  def test_matching_matches_brute_force(self):
    rng = random.Random(20280701)
    for _ in range(300):
      counts, cells, tokens = random_instance(rng)
      num_ranks = len(counts)
      caps = caps_from(cells, tokens, num_ranks)
      cards = []
      for r, k in enumerate(counts):
        cards.extend([r] * k)
      compat = [(c, r) for (c, r) in cells if tokens[c]]
      self.assertEqual(
          oracle.max_matching(counts, caps),
          brute_force_matching(cards, compat))

  def test_adversarial_margin_matches_brute_force(self):
    rng = random.Random(20280702)
    checked = 0
    for _ in range(200):
      counts, cells, tokens = random_instance(
          rng, num_ranks=3, max_copies=2, max_cells=6)
      num_ranks = len(counts)
      caps = caps_from(cells, tokens, num_ranks)
      compat = [(c, r) for (c, r) in cells if tokens[c]]
      matching = oracle.max_matching(counts, caps)
      if matching == 0:
        continue
      plays = rng.randint(0, matching)
      got = oracle.adversarial_margin(counts, caps, plays)
      # Brute force: largest K such that EVERY K-subset of compatible cells
      # leaves matching >= plays.
      cards = []
      for r, k in enumerate(counts):
        cards.extend([r] * k)
      true_margin = len(compat)
      for k in range(1, len(compat) + 1):
        all_safe = True
        for deleted in itertools.combinations(range(len(compat)), k):
          remaining = [compat[i] for i in range(len(compat))
                       if i not in deleted]
          if brute_force_matching(cards, remaining) < plays:
            all_safe = False
            break
        if not all_safe:
          true_margin = k - 1
          break
      self.assertEqual(got, true_margin,
                       msg=f"counts={counts} caps={caps} plays={plays}")
      checked += 1
    self.assertGreater(checked, 50)

  def test_dead_is_negative(self):
    self.assertEqual(oracle.adversarial_margin([2, 0], [1, 0], 2), -1)

  def test_zero_plays_is_safe(self):
    self.assertGreaterEqual(oracle.adversarial_margin([1], [1], 0), 1)


def play_to_phase(state, rng, target_phase):
  while not state.is_terminal() and state._phase < target_phase:
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      state.apply_action(rng.choice([o[0] for o in outcomes]))
    else:
      state.apply_action(rng.choice(state.legal_actions()))
  return state


class AdapterTest(unittest.TestCase):

  def test_legality_reconstruction(self):
    """Oracle's view reproduces the game's play-phase legal actions."""
    rng = random.Random(20280703)
    decisions = 0
    for _ in range(30):
      game = pyspiel.load_game("python_quantum_cat", {"players": 3})
      state = play_to_phase(game.new_initial_state(), rng, 3)
      while not state.is_terminal():
        player = state.current_player()
        legal = state.legal_actions()
        view = oracle.view_from_state(state, player)
        mine = oracle.legal_play_cells(view)
        # Replicate the red-lead filter.
        if state._led_color is None and not state._red_row_occupied():
          red = [a for a in mine if (a // view.num_ranks) == 0]
          if red and len(mine) > len(red):
            mine = [a for a in mine if (a // view.num_ranks) != 0]
        if not mine:
          mine = [oracle.PARADOX_ACTION]
        self.assertEqual(sorted(mine), sorted(legal))
        decisions += 1
        state.apply_action(rng.choice(legal))
    self.assertGreater(decisions, 200)

  def test_post_play_margin_matches_next_state(self):
    """Analytic post-action margin equals recomputing on the real next state."""
    rng = random.Random(20280704)
    checked = 0
    for _ in range(20):
      game = pyspiel.load_game("python_quantum_cat", {"players": 3})
      state = play_to_phase(game.new_initial_state(), rng, 3)
      while not state.is_terminal():
        player = state.current_player()
        legal = state.legal_actions()
        action = rng.choice(legal)
        if action != oracle.PARADOX_ACTION:
          view = oracle.view_from_state(state, player)
          predicted = oracle.post_play_margin(view, action)
          nxt = state.clone()
          nxt.apply_action(action)
          if not nxt.is_terminal():
            actual = oracle.view_from_state(nxt, player).margin()
            self.assertEqual(predicted, actual,
                             msg=f"action={action} trick={state._trick_number}")
            checked += 1
        state.apply_action(action)
    self.assertGreater(checked, 100)


class SoundnessTest(unittest.TestCase):

  def test_dead_flag_implies_paradox(self):
    """If the oracle flags a player dead, that player never finishes clean.

    The claim is conditional on the round running to completion: another
    player's paradox ends the round early for everyone, which leaves the
    flagged player's fate untested (counted, not failed).
    """
    rng = random.Random(20280705)
    flagged_rounds = 0
    confirmed = 0
    truncated = 0
    for _ in range(120):
      game = pyspiel.load_game("python_quantum_cat", {"players": 3})
      state = play_to_phase(game.new_initial_state(), rng, 3)
      dead_flagged = set()
      while not state.is_terminal():
        player = state.current_player()
        view = oracle.view_from_state(state, player)
        if not view.alive():
          dead_flagged.add(player)
        state.apply_action(rng.choice(state.legal_actions()))
      if not dead_flagged:
        continue
      flagged_rounds += 1
      paradoxed = [p for p in range(3) if state._has_paradoxed[p]]
      for p in dead_flagged:
        if state._has_paradoxed[p]:
          confirmed += 1
        else:
          # Only acceptable if someone else ended the round first.
          self.assertTrue(
              paradoxed,
              msg="oracle flagged a player dead but the round completed "
                  "with no paradox at all - soundness violated")
          truncated += 1
    # Random play should produce plenty of flagged rounds in this game.
    self.assertGreater(flagged_rounds, 10)
    self.assertGreater(confirmed, 0)


class ShieldSelectionTest(unittest.TestCase):

  def test_shield_never_picks_dead_when_alive_exists(self):
    rng = random.Random(20280706)
    np_rng = np.random.RandomState(20280706)
    checked = 0
    for _ in range(20):
      game = pyspiel.load_game("python_quantum_cat", {"players": 3})
      state = play_to_phase(game.new_initial_state(), rng, 3)
      while not state.is_terminal():
        player = state.current_player()
        legal = state.legal_actions()
        policy = np_rng.rand(1000)
        choice = oracle.select_action_with_oracle(
            state, player, policy, min_margin=1)
        if choice is None:
          choice = rng.choice(legal)
        self.assertIn(choice, legal)
        if state._phase == 3 and legal != [oracle.PARADOX_ACTION]:
          view = oracle.view_from_state(state, player)
          margins = {a: oracle.post_play_margin(view, a)
                     for a in legal if a != oracle.PARADOX_ACTION}
          if any(m >= 0 for m in margins.values()):
            self.assertGreaterEqual(
                margins[choice], 0,
                msg="shield chose a provably dead move while an alive "
                    "move existed")
            checked += 1
        state.apply_action(choice)
    self.assertGreater(checked, 100)


if __name__ == "__main__":
  unittest.main()
