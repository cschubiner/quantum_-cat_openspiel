"""Tests for Python Quantum Cat"""

import pickle

from absl.testing import absltest
import numpy as np

import pyspiel
from quantum_cat_ai import make_bot
from quantum_cat_full_match_elo import match_outcome_scores, play_match
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

  def test_supported_player_count_contracts(self):
    contracts = {
        2: {"ranks": 5, "cards": 10, "tricks": 9, "bids": None},
        3: {"ranks": 6, "cards": 10, "tricks": 8, "bids": [101, 102, 103, 104]},
        4: {"ranks": 8, "cards": 10, "tricks": 8, "bids": [101, 102, 103, 104]},
        5: {"ranks": 9, "cards": 9, "tricks": 7, "bids": [101, 102, 103, 104]},
    }
    for players, expected in contracts.items():
      game = pyspiel.load_game("python_quantum_cat", {"players": players})
      state = game.new_initial_state()
      self.assertEqual(state._num_card_types, expected["ranks"])
      self.assertEqual(game.cards_per_player_initial, expected["cards"])
      self.assertEqual(state._num_tricks, expected["tricks"])

      for _ in range(players * expected["cards"]):
        action, _ = state.chance_outcomes()[0]
        state.apply_action(action)
      if players == 2:
        self.assertEqual(state._phase, 3)
        continue
      for player in range(players):
        self.assertEqual(state.current_player(), player)
        state.apply_action(state.legal_actions(player)[0])
      self.assertEqual(state._phase, 2)
      for player in range(players):
        self.assertEqual(state.current_player(), player)
        self.assertEqual(state.legal_actions(player), expected["bids"])
        state.apply_action(expected["bids"][0])
      self.assertEqual(state._phase, 3)

  def test_chance_history_replays_deterministically(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    first = game.new_initial_state()
    second = game.new_initial_state()
    for _ in range(30):
      self.assertTrue(first.is_chance_node())
      action = first.chance_outcomes()[0][0]
      first.apply_action(action)
      second.apply_action(action)
    for first_hand, second_hand in zip(first._hands, second._hands):
      np.testing.assert_array_equal(first_hand, second_hand)

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
    """Tests prediction actions and then moves to trick phase."""
    game = pyspiel.load_game("python_quantum_cat", {"players": 5})
    state = game.new_initial_state()

    # Quick function to auto-deal + discard:
    self._deal_and_discard(game, state)

    # Now in phase=2 => predictions
    for player in range(5):
      self.assertEqual(state.current_player(), player)
      actions = state.legal_actions()
      self.assertEqual(actions, [101, 102, 103, 104])
      # Let's pick "Predict=2" => action=102
      state.apply_action(102)

    # After all predictions, should be in trick phase (phase=3).
    self.assertIn("Phase=3", str(state))
    self.assertFalse(state.is_terminal())

  def test_three_player_prediction_can_bid_two(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    self._deal_and_discard(game, state)

    for player in range(3):
      self.assertEqual(state.current_player(), player)
      self.assertEqual(state.legal_actions(), [101, 102, 103, 104])
      state.apply_action(102)

    self.assertIn("Phase=3", str(state))

  def test_configured_round_start_player_controls_prediction_and_first_trick(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 3, "start_player": 2})
    state = game.new_initial_state()
    self._deal_and_discard(game, state)

    for player in [2, 0, 1]:
      self.assertEqual(state.current_player(), player)
      state.apply_action(101)

    self.assertEqual(state._phase, 3)
    self.assertEqual(state._start_player, 2)
    self.assertEqual(state.current_player(), 2)

  def test_two_player_configured_round_start_player_starts_tricks(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 2, "start_player": 1})
    state = game.new_initial_state()

    for _ in range(2 * 10):
      action, _ = state.chance_outcomes()[0]
      state.apply_action(action)

    self.assertEqual(state._phase, 3)
    self.assertEqual(state._start_player, 1)
    self.assertEqual(state.current_player(), 1)

  def test_full_match_rotates_start_player_once_per_player(self):
    bots = [
        make_bot("random", seed=11),
        make_bot("random", seed=12),
        make_bot("random", seed=13),
    ]
    match = play_match(
        3,
        ["a", "b", "c"],
        bots,
        seed=20260602,
        initial_start=1,
    )

    self.assertEqual([r["start_player"] for r in match["rounds"]], [1, 2, 0])
    self.assertEqual(len(match["totals"]), 3)
    self.assertEqual(len(match["rounds"]), 3)

  def test_full_match_outcome_uses_final_round_tiebreak(self):
    outcome = match_outcome_scores(
        totals=[12.0, 12.0, 10.0],
        final_round_returns=[2.0, 5.0, 9.0],
    )

    self.assertGreater(outcome[1], outcome[0])
    self.assertGreater(outcome[0], outcome[2])
    self.assertEqual(
        match_outcome_scores([7.0, 7.0], [3.0, 3.0])[0],
        match_outcome_scores([7.0, 7.0], [3.0, 3.0])[1],
    )

  def test_match_context_extends_observation_and_returns_cumulative_score(self):
    plain = pyspiel.load_game("python_quantum_cat", {"players": 3})
    contextual = pyspiel.load_game(
        "python_quantum_cat",
        {"players": 3, "match_context": 1},
    )
    self.assertEqual(
        contextual.observation_tensor_shape()[0],
        plain.observation_tensor_shape()[0] + 4,
    )

    state = contextual.new_initial_state()
    state.set_match_context([10.0, -2.0, 3.0], 1)
    obs = state.observation_tensor(0)
    np.testing.assert_allclose(obs[-4:-1], [0.5, -0.1, 0.15])
    self.assertEqual(obs[-1], 1.0)

    state._game_over = True
    state._phase = 4
    state._tricks_won = np.array([1, 1, 1])
    state._predictions = [1, 1, 1]
    state._board_ownership[:, :] = -1
    state._board_ownership[0, 0] = 0
    state._board_ownership[1, 0] = 1
    state._board_ownership[2, 0] = 2
    state._compute_final_scores()
    self.assertEqual(state.returns(), [12.0, 0.0, 5.0])
    self.assertEqual(state.raw_round_scores(), [2.0, 2.0, 2.0])

  def test_raw_round_scores_clone_with_match_context(self):
    contextual = pyspiel.load_game(
        "python_quantum_cat",
        {"players": 3, "match_context": 1},
    )
    state = contextual.new_initial_state()
    state.set_match_context([4.0, 5.0, 6.0], 2)
    state._game_over = True
    state._phase = 4
    state._tricks_won = np.array([2, 1, 0])
    state._predictions = [4, 1, 3]
    state._board_ownership[:, :] = -1
    state._board_ownership[0, 0] = 1
    state._compute_final_scores()

    self.assertEqual(state.raw_round_scores(), [2.0, 2.0, 0.0])
    self.assertEqual(state.returns(), [6.0, 7.0, 6.0])
    self.assertEqual(state.clone().raw_round_scores(), [2.0, 2.0, 0.0])

  def test_resample_hides_opponent_discards(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    self._deal_and_discard(game, state)

    original_legal = state.legal_actions(0)
    sampled_other_discards = set()
    for _ in range(80):
      sampled = state.resample_from_infostate(0, np.random)
      self.assertEqual(sampled._hands[0].tolist(), state._hands[0].tolist())
      self.assertEqual(sampled._discarded_cards[0], state._discarded_cards[0])
      self.assertEqual(sampled.legal_actions(0), original_legal)
      sampled_other_discards.add(tuple(sampled._discarded_cards[1:]))

    self.assertGreater(len(sampled_other_discards), 1)

  def test_resample_uses_supplied_sampler(self):
    class RecordingSampler:

      def __init__(self):
        self.shuffle_calls = 0

      def shuffle(self, values):
        self.shuffle_calls += 1
        values.reverse()

    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    self._deal_and_discard(game, state)

    sampler = RecordingSampler()
    sampled = state.resample_from_infostate(0, sampler)

    self.assertEqual(sampler.shuffle_calls, 1)
    self.assertEqual(sampled._hands[0].tolist(), state._hands[0].tolist())
    self.assertEqual(sampled._discarded_cards[0], state._discarded_cards[0])

  def test_action_feature_appended_columns_are_at_end(self):
    from quantum_cat_alphazero_torch import (
        ACTION_FEATURE_SIZE,
        APPENDED_ACTION_FEATURE_NAMES,
        APPENDED_ACTION_FEATURE_START,
        BASE_ACTION_FEATURE_SIZE,
        action_feature_matrix,
    )

    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    player = state.current_player()
    features = action_feature_matrix(state, player, game.num_distinct_actions())

    self.assertEqual(APPENDED_ACTION_FEATURE_START, BASE_ACTION_FEATURE_SIZE)
    self.assertEqual(
        ACTION_FEATURE_SIZE,
        BASE_ACTION_FEATURE_SIZE + len(APPENDED_ACTION_FEATURE_NAMES),
    )
    self.assertEqual(features.shape[1], ACTION_FEATURE_SIZE)
    self.assertEqual(
        features.shape[1] - APPENDED_ACTION_FEATURE_START,
        len(APPENDED_ACTION_FEATURE_NAMES),
    )

  def test_action_features_and_logits_are_infostate_safe(self):
    from quantum_cat_alphazero_torch import AZNet, action_feature_matrix
    import torch

    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    self._deal_and_discard(game, state)
    for _ in range(3):
      state.apply_action(101)
    player = state.current_player()
    base_obs = np.array(state.observation_tensor(player), dtype=np.float32)
    base_legal = state.legal_actions(player)
    base_features = action_feature_matrix(
        state, player, game.num_distinct_actions()
    )

    hidden_variation_seen = False
    for _ in range(80):
      sampled = state.resample_from_infostate(player, np.random)
      if any(
          sampled._hands[p].tolist() != state._hands[p].tolist()
          for p in range(game.num_players())
          if p != player
      ):
        hidden_variation_seen = True
      self.assertEqual(sampled.legal_actions(player), base_legal)
      np.testing.assert_allclose(
          sampled.observation_tensor(player), base_obs, atol=0.0
      )
      np.testing.assert_allclose(
          action_feature_matrix(sampled, player, game.num_distinct_actions()),
          base_features,
          atol=0.0,
      )
    self.assertTrue(hidden_variation_seen)

    torch.manual_seed(20260603)
    model = AZNet(
        game.observation_tensor_shape()[0],
        game.num_distinct_actions(),
        game.num_players(),
        width=64,
        depth=2,
        arch="action_mlp",
    )
    model.eval()
    sampled = state.resample_from_infostate(player, np.random)
    obs_t = torch.tensor(base_obs, dtype=torch.float32).unsqueeze(0)
    features_t = torch.tensor(base_features, dtype=torch.float32).unsqueeze(0)
    sampled_obs_t = torch.tensor(
        sampled.observation_tensor(player), dtype=torch.float32
    ).unsqueeze(0)
    sampled_features_t = torch.tensor(
        action_feature_matrix(sampled, player, game.num_distinct_actions()),
        dtype=torch.float32,
    ).unsqueeze(0)
    with torch.no_grad():
      logits, values = model(obs_t, features_t)
      sampled_logits, sampled_values = model(sampled_obs_t, sampled_features_t)
    torch.testing.assert_close(sampled_logits, logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(sampled_values, values, atol=0.0, rtol=0.0)

  def test_two_player_duplicate_leftover_blocks_stacked_rows(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 2})
    state = game.new_initial_state()
    state._deck = np.array([1] * 23 + [2, 3], dtype=int)

    for _ in range(2 * 10):
      state.apply_action(0)

    self.assertEqual(state._board_ownership[3, 0], -2)
    self.assertEqual(state._board_ownership[2, 0], -2)
    self.assertEqual(state._board_ownership[1, 0], -2)
    self.assertEqual(state._phase, 3)

  def test_red_can_be_led_after_any_red_space_is_occupied(self):
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    self._deal_and_discard(game, state)
    for _ in range(3):
      state.apply_action(101)

    state._current_player = 0
    state._start_player = 0
    state._led_color = None
    state._trump_broken = False
    state._cards_played_this_trick = [None] * 3
    state._hands[0] = np.zeros(state._num_card_types, dtype=int)
    state._hands[0][1] = 1
    state._board_ownership[0, 0] = 1

    legal = state.legal_actions(0)
    self.assertIn(1, legal)  # Play rank 2 as red.
    self.assertIn(state._num_card_types + 1, legal)  # Non-red remains legal too.

  def test_trick_paradox(self):
    """Forced paradox is legal and terminal when a player has no playable card."""
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()

    state._phase = 3
    state._current_player = 0
    state._start_player = 0
    state._led_color = None
    state._cards_played_this_trick = [None] * 3
    state._predictions = [1, 1, 1]
    state._tricks_won = np.array([2, 0, 0], dtype=int)
    state._hands = [
        np.zeros(state._num_card_types, dtype=int) for _ in range(3)
    ]
    state._hands[0][0] = 1
    state._board_ownership[:, :] = -1
    state._board_ownership[:, 0] = [1, 1, 2, -2]

    self.assertEqual(state.legal_actions(0), [999])
    state.apply_action(999)

    self.assertTrue(state.is_terminal())
    self.assertEqual(state.current_player(), pyspiel.PlayerId.TERMINAL)
    self.assertEqual(state._phase, 4)
    self.assertEqual(state._has_paradoxed, [True, False, False])
    self.assertEqual(state.raw_round_scores(), [-2.0, 0.0, 0.0])
    self.assertEqual(state.returns(), [-2.0, 0.0, 0.0])
    self.assertIn("GameOver=True", str(state))

  def test_adjacency_scoring(self):
    """
    Construct a simple deterministic test that ensures adjacency scoring works:
    - Player 0 owns a cluster of 3 adjacent squares.
    - If they bid exactly the # of tricks they took, they get base_score + 3.
    """
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()

    state._phase = 4
    state._game_over = True
    state._returns = [0.0, 0.0, 0.0]
    state._rewards = [0.0, 0.0, 0.0]
    state._has_paradoxed = [False, False, False]
    state._tricks_won = np.array([1, 2, 0], dtype=int)
    state._predictions = [1, 1, 1]
    state._board_ownership[:, :] = -1
    state._board_ownership[0, 0] = 0
    state._board_ownership[0, 1] = 0
    state._board_ownership[0, 2] = 0
    state._board_ownership[1, 0] = 1
    state._board_ownership[3, 5] = 1

    state._compute_final_scores()

    self.assertEqual(state._largest_cluster_for_player(0), 3)
    self.assertEqual(state.raw_round_scores(), [4.0, 2.0, 0.0])
    self.assertEqual(state.returns(), [4.0, 2.0, 0.0])
    self.assertEqual(state._player_adjacency_bonus, [3, 0.0, 0.0])

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
    deals_needed = game.num_players() * game.cards_per_player_initial
    for _ in range(deals_needed):
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      chosen = np.random.choice(actions, p=probs)
      state.apply_action(chosen)

    # Discard
    for p in range(game.num_players()):
      discard_actions = state.legal_actions(p)
      # pick the first feasible discard
      if discard_actions:
        state.apply_action(discard_actions[0])


if __name__ == "__main__":
  absltest.main()
