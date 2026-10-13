# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dynamic_routing_to_mean_field_game."""
from absl.testing import absltest

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.games import dynamic_routing
from open_spiel.python.games import dynamic_routing_to_mean_field_game
from open_spiel.python.games import dynamic_routing_utils
from open_spiel.python.mfg import games as mfg_games  # pylint:disable=unused-import
from open_spiel.python.mfg.algorithms import mirror_descent
import pyspiel


class DerivedNPlayerPolicyFromMeanFieldPolicyTest(absltest.TestCase):

  def _small_two_player_game(self):
    network = dynamic_routing_utils.Network({
        "O": ["A", "B"],
        "A": ["C"],
        "B": ["C"],
        "C": ["D"],
        "D": [],
    })
    vehicles = [
        dynamic_routing_utils.Vehicle("O->A", "C->D"),
        dynamic_routing_utils.Vehicle("O->B", "C->D"),
    ]
    return dynamic_routing.DynamicRoutingGame(
        {
            "max_num_time_step": 10,
            "time_step_length": 0.5,
            "players": -1
        },
        network=network,
        vehicles=vehicles)

  def test_state_conversion_method(self):
    """Test N player game state to mean field game state conversion."""
    n_player_game = self._small_two_player_game()
    mfg_empty_state = dynamic_routing_to_mean_field_game._create_empty_mfg_state(  # pylint:disable=protected-access
        n_player_game)
    derived_policy = (
        dynamic_routing_to_mean_field_game
        .DerivedNPlayerPolicyFromMeanFieldPolicy(
            n_player_game, policy.UniformRandomPolicy(
                mfg_empty_state.get_game())))

    state = n_player_game.new_initial_state()
    state.apply_actions([
        n_player_game.network.get_action_id_from_movement("A", "C"),
        n_player_game.network.get_action_id_from_movement("B", "C"),
    ])

    converted = derived_policy._convert_state_to_mean_field_state(state, 1)  # pylint:disable=protected-access

    self.assertEqual(converted.current_player(),
                     pyspiel.PlayerId.DEFAULT_PLAYER_ID)
    self.assertEqual(converted.current_time_step, state.current_time_step)
    self.assertFalse(converted.is_terminal())
    self.assertEqual(str(converted),
                     "Location=B->C, waiting_time=1, t=1, destination='C->D'")
    self.assertEqual(converted.legal_actions(),
                     [dynamic_routing_utils.NO_POSSIBLE_ACTION])
    self.assertEqual(
        converted,
        derived_policy._convert_state_to_mean_field_state(state, 1))  # pylint:disable=protected-access

  def test_uniform_mfg_policy_conversion_to_n_player_uniform_policy(self):
    """Test conversion of uniform to uniform policy."""
    mfg_game = pyspiel.load_game("python_mfg_dynamic_routing", {
        "time_step_length": 0.05,
        "max_num_time_step": 100
    })
    n_player_game = pyspiel.load_game("python_dynamic_routing", {
        "time_step_length": 0.05,
        "max_num_time_step": 100
    })
    mfg_derived_policy = (
        dynamic_routing_to_mean_field_game
        .DerivedNPlayerPolicyFromMeanFieldPolicy(
            n_player_game, policy.UniformRandomPolicy(mfg_game)))
    derived_policy_value = expected_game_score.policy_value(
        n_player_game.new_initial_state(), mfg_derived_policy)
    uniform_policy_value = expected_game_score.policy_value(
        n_player_game.new_initial_state(),
        policy.UniformRandomPolicy(n_player_game))
    self.assertSequenceAlmostEqual(derived_policy_value, uniform_policy_value)

  def test_pigou_network_game_outcome_optimal_mfg_policy_in_n_player_game(self):
    """Test MFG Nash equilibrium policy for the Pigou network."""
    mfg_game = pyspiel.load_game("python_mfg_dynamic_routing", {
        "time_step_length": 0.05,
        "max_num_time_step": 100
    })
    omd = mirror_descent.MirrorDescent(mfg_game, lr=1)
    for _ in range(10):
      omd.iteration()
    n_player_game = pyspiel.load_game("python_dynamic_routing", {
        "time_step_length": 0.05,
        "max_num_time_step": 100
    })

    mfg_derived_policy = (
        dynamic_routing_to_mean_field_game
        .DerivedNPlayerPolicyFromMeanFieldPolicy(n_player_game,
                                                 omd.get_policy()))
    derived_policy_value = expected_game_score.policy_value(
        n_player_game.new_initial_state(), mfg_derived_policy)

    self.assertLen(derived_policy_value, n_player_game.num_players())
    self.assertSequenceAlmostEqual(
        derived_policy_value, [-3.69785841] * n_player_game.num_players())

  def test_learning_and_applying_mfg_policy_in_n_player_game(self):
    """Test converting learnt MFG policy default game."""
    # learning the Braess MFG Nash equilibrium
    mfg_game = pyspiel.load_game("python_mfg_dynamic_routing")
    omd = mirror_descent.MirrorDescent(mfg_game, lr=1)
    for _ in range(10):
      omd.iteration()
    mfg_policy = omd.get_policy()
    n_player_game = pyspiel.load_game("python_dynamic_routing")
    mfg_derived_policy = (
        dynamic_routing_to_mean_field_game
        .DerivedNPlayerPolicyFromMeanFieldPolicy(n_player_game, mfg_policy))
    expected_game_score.policy_value(n_player_game.new_initial_state(),
                                     mfg_derived_policy)


if __name__ == "__main__":
  absltest.main()
