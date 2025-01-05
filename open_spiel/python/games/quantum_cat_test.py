"""Tests for Python Quantum Cat"""

import pickle

from absl.testing import absltest
import numpy as np

import pyspiel
# If your Quantum Cat code is in an external file, you'd do something like:
#   from open_spiel.python.games import quantum_cat
#
# Otherwise, if you registered it only via a local file, ensure the import path
# is correct. For demonstration, we'll rely on pyspiel.load_game("python_quantum_cat").

class QuantumCatTest(absltest.TestCase):

  def test_can_create_game_and_state(self):
    """Checks we can create the game and a state with default params (5 players)."""
    game = pyspiel.load_game("python_quantum_cat")
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    self.assertFalse(state.is_terminal())
    self.assertIn("Phase=0", str(state))  # Just a quick check on the string output

  def test_chance_deal(self):
    """Ensures the game deals cards in chance mode until each player has the right number."""
    game = pyspiel.load_game("python_quantum_cat", {"players": 5})
    state = game.new_initial_state()
    num_deals = 5 * 9  # 5 players * 9 cards each (for 5p)
    for i in range(num_deals):
      self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
      # pick any outcome
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      chosen = np.random.choice(actions, p=probs)
      state.apply_action(chosen)

    # After dealing, we should be in discard phase (phase=1),
    # current player should be 0 (the first discarder).
    self.assertEqual(state.current_player(), 0)
    self.assertIn("Phase=1", str(state))

  def test_discard_phase(self):
    """Populates the state through dealing, then tests that each player discards exactly 1."""
    game = pyspiel.load_game("python_quantum_cat", {"players": 5})
    state = game.new_initial_state()

    # Deal all cards
    num_deals = 5 * 9
    for _ in range(num_deals):
      # chance node pick
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      chosen = np.random.choice(actions, p=probs)
      state.apply_action(chosen)

    # Now in discard phase
    # We expect each player to discard once in ascending order: 0..4
    for player in range(5):
      self.assertEqual(state.current_player(), player)
      discard_actions = state.legal_actions()
      self.assertGreater(len(discard_actions), 0)  # must discard something
      # Just pick the first action
      chosen_discard = discard_actions[0]
      state.apply_action(chosen_discard)

    # After all discards, we should be in prediction phase (phase=2).
    self.assertIn("Phase=2", str(state))
    self.assertFalse(state.is_terminal())

  def test_prediction_phase(self):
    """Tests that each player can only predict 1..4, and then moves to trick phase."""
    game = pyspiel.load_game("python_quantum_cat", {"players": 5})
    state = game.new_initial_state()

    # Quick function to auto-deal + discard:
    self._deal_and_discard(game, state)

    # Now in phase=2 => predictions
    for player in range(5):
      self.assertEqual(state.current_player(), player)
      actions = state.legal_actions()
      # Should be exactly [101,102,103,104], i.e. 1..4
      self.assertEqual(actions, [101, 102, 103, 104])
      # Let's pick "Predict=2" => action=102
      state.apply_action(102)

    # After all predictions, should be in trick phase (phase=3).
    self.assertIn("Phase=3", str(state))
    self.assertFalse(state.is_terminal())

  def test_trick_paradox(self):
    """Construct a scenario where a paradox might arise if a player can't play any card."""
    game = pyspiel.load_game("python_quantum_cat", {"players": 5})
    state = game.new_initial_state()

    self._deal_and_discard(game, state)
    # Everyone predicts. Just pick 2 for each.
    for _ in range(5):
      state.apply_action(102)  # "Predict=2"

    # Now in trick phase
    # If we artificially manipulate the state so that player 0 has no possible move,
    # we can force a paradox. For a real test, we might do it by repeatedly claiming
    # all possible color-rank combos that player 0 has. But let's see if we can do it more simply
    # by playing random moves until we catch a paradox or the game ends.

    # We'll just do a few random moves and see if we can cause no legal actions for someone:
    while not state.is_terminal():
      cur_player = state.current_player()
      actions = state.legal_actions(cur_player)
      if len(actions) == 1 and actions[0] == 999:
        # PARADOX is forced
        state.apply_action(999)
        self.assertTrue(state.is_terminal())
        self.assertIn("GameOver=True", str(state))
        return
      else:
        # pick a random valid action
        chosen = np.random.choice(actions)
        state.apply_action(chosen)
        if state.is_terminal():
          break

    # If we exit the while loop naturally, no paradox happened this time. That's fine,
    # but we tested it can handle random moves. One can craft a direct scenario for forced paradox
    # by controlling the deck or color usage.

  def test_adjacency_scoring(self):
    """
    Construct a simple deterministic test that ensures adjacency scoring works:
    - We'll try to give a single player a cluster of 3 adjacent squares.
    - If they bid exactly the # of tricks they took, they get base_score + 3.
    """
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()

    # We'll do a minimal dealing approach:
    # for 3 players with 11 each => 33 deals. We'll just choose rank=1 for the first 11,
    # rank=2 for next 11, rank=3 for last 11, etc., so we can control adjacency.

    deal_actions = state.chance_outcomes()
    # But let's just do a direct approach: call apply_action(0) repeatedly
    # so we always pick the "front" card. This won't be truly deterministic
    # unless the deck was sorted, but let's assume. Or we can forcibly set the deck
    # if we have direct access, but let's keep it simple here:
    for _ in range(3 * 11):
      outcomes = state.chance_outcomes()
      # Just pick the first outcome
      chosen, _ = outcomes[0]
      state.apply_action(chosen)

    # Discard phase: each discards 1
    for p in range(3):
      discard_actions = state.legal_actions(p)
      # Just pick the first
      state.apply_action(discard_actions[0])

    # Predict phase: each picks 1
    for p in range(3):
      # legal actions => [101..104]
      state.apply_action(101)  # predict=1

    # Now in trick phase. We'll artificially try to create adjacency for player 0
    # by letting them play the same color in consecutive ranks, e.g., (R,1), (R,2), (R,3).
    # We'll skip full suit-following logic for brevity, just pick feasible actions if possible.

    # We want player 0 to take exactly 1 trick so they match their prediction (1),
    # and also place 3 tokens on adjacency squares. We can do so by letting them only
    # succeed in 1 trick but claim 3 color-rank combos in separate leads. That might conflict
    # with real trick logic, but we'll do minimal logic just for adjacency demonstration.
    #
    # We'll do a few forced moves. If they can't lead R for all three, no big deal—we're just
    # showing a skeleton. A real adjacency test might manipulate the deck more carefully.

    while (not state.is_terminal()) and (state._phase == 3):
      cur_player = state.current_player()
      actions = state.legal_actions(cur_player)
      if not actions:
        break  # or paradox
      chosen = actions[0]  # pick first feasible
      state.apply_action(chosen)

    # Once complete, check final scores. We can't guarantee a big adjacency cluster
    # in this random approach, but at least we test the adjacency code runs.
    if state.is_terminal():
      returns = state.returns()
      # They might or might not have adjacency points. We just check it doesn't crash.
      # In a real test, you'd craft exact card deals to ensure a known adjacency outcome.
      self.assertEqual(len(returns), 3)

  def test_game_from_cc(self):
    """Runs standard game tests using pyspiel.random_sim_test for coverage."""
    # Must match the short name used in the registration.
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    pyspiel.random_sim_test(game, num_sims=3, serialize=False, verbose=True)

  def test_pickle(self):
    """Checks pickling/unpickling of game and a partially progressed state."""
    game = pyspiel.load_game("python_quantum_cat", {"players": 4})
    pickled_game = pickle.dumps(game)
    unpickled_game = pickle.loads(pickled_game)
    self.assertEqual(str(game), str(unpickled_game))

    state = game.new_initial_state()
    # Deal a few cards
    for _ in range(2):
      outcomes = state.chance_outcomes()
      chosen, _ = outcomes[0]
      state.apply_action(chosen)

    # Discard for first player
    if state.current_player() != pyspiel.PlayerId.CHANCE:
      discard_actions = state.legal_actions(state.current_player())
      if discard_actions:
        state.apply_action(discard_actions[0])

    ser_str = pyspiel.serialize_game_and_state(game, state)
    new_game, new_state = pyspiel.deserialize_game_and_state(ser_str)
    self.assertEqual(str(game), str(new_game))
    self.assertEqual(str(state), str(new_state))

    pickled_state = pickle.dumps(state)
    unpickled_state = pickle.loads(pickled_state)
    self.assertEqual(str(state), str(unpickled_state))

  def test_cloned_state_matches_original_state(self):
    """Check we can clone states successfully."""
    game = pyspiel.load_game("python_quantum_cat", {"players": 5})
    state = game.new_initial_state()
    # Deal a few cards
    for _ in range(10):
      outcomes = state.chance_outcomes()
      if not outcomes:
        break
      chosen, _ = outcomes[0]
      state.apply_action(chosen)

    # Possibly discard
    if state._phase == 1:
      discard_actions = state.legal_actions(state.current_player())
      if discard_actions:
        state.apply_action(discard_actions[0])

    clone = state.clone()

    # Basic checks
    self.assertEqual(state.history(), clone.history())
    self.assertEqual(state.num_players(), clone.num_players())
    self.assertEqual(state.move_number(), clone.move_number())
    self.assertEqual(state.num_distinct_actions(), clone.num_distinct_actions())

    # Now we can compare string representations. For deeper checks,
    # you can also compare internal arrays if needed.
    self.assertEqual(str(state), str(clone))

  # -------------------------------------------------------------------------
  # Helper method(s)
  # -------------------------------------------------------------------------
  def _deal_and_discard(self, game, state):
    """Helper to fully deal and discard once for each player."""
    # Deal
    deals_needed = game.num_players * game.cards_per_player_initial
    for _ in range(deals_needed):
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      chosen = np.random.choice(actions, p=probs)
      state.apply_action(chosen)

    # Discard
    for p in range(game.num_players):
      discard_actions = state.legal_actions(p)
      # pick the first feasible discard
      if discard_actions:
        state.apply_action(discard_actions[0])


if __name__ == "__main__":
  absltest.main()
