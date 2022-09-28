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
"""Tests for open_spiel.python.jax.lola."""

from typing import Tuple

import distrax
import haiku as hk
import jax
import numpy as np
import pyspiel
from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import rl_environment
from open_spiel.python.jax.lola import LolaPolicyGradientAgent

SEED = 24984617


def make_iterated_matrix_game(game: str, iterations=5, batch_size=8) -> rl_environment.Environment:
    matrix_game = pyspiel.load_matrix_game(game)
    config = {"num_repetitions": iterations, "batch_size": batch_size}
    game = pyspiel.create_repeated_game(matrix_game, config)
    env = rl_environment.Environment(game)
    return env


def make_agent_networks(num_actions: int) -> Tuple[hk.Transformed, hk.Transformed]:
    def policy(obs):
        logits = hk.nets.MLP(output_sizes=[8, 8, num_actions], with_bias=True)(obs)
        return distrax.Categorical(logits=logits)

    def value_fn(obs):
        values = hk.nets.MLP(output_sizes=[8, 8, 1], with_bias=True)(obs)
        return values

    return hk.without_apply_rng(hk.transform(policy)), hk.without_apply_rng(hk.transform(value_fn))


class LolaPolicyGradientTest(parameterized.TestCase, absltest.TestCase):

    @parameterized.parameters(["matrix_pd"])
    def test_run_game(self, game_name):
        batch_size = 8
        iterations = 5
        env = make_iterated_matrix_game(game_name, batch_size=batch_size, iterations=iterations)
        env.seed(SEED)
        key = jax.random.PRNGKey(SEED)
        num_actions = env.action_spec()["num_actions"]
        policy_network, critic_network = make_agent_networks(num_actions=num_actions)

        agents = [
            LolaPolicyGradientAgent(
                player_id=i,
                opponent_ids=[1 - i],
                seed=key,
                info_state_size=env.observation_spec()["info_state"],
                num_actions=env.action_spec()["num_actions"],
                policy=policy_network,
                critic=critic_network,
                batch_size=batch_size,
                pi_learning_rate=0.005,
                critic_learning_rate=1.0,
                policy_update_interval=2,
                discount=0.96,
                lola_weight=1.0,
                use_jit=True
            )
            for i in range(2)
        ]

        for _ in range(2 * batch_size):
            time_step = env.reset()
            while not time_step.last():
                actions = [agent.step(time_step).action for agent in agents]
                time_step = env.step(actions)
                time_step.observations["actions"] = np.array(actions)

            for agent in agents:
                agent.step(time_step)


if __name__ == "__main__":
    np.random.seed(SEED)
    absltest.main()
