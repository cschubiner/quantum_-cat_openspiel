import numpy as np
import pyspiel
from pyspiel import PlayerId

import open_spiel.python.rl_environment
from open_spiel.python import rl_environment

from open_spiel.python.rl_environment import Environment, TimeStep, StepType


class IteratedMatrixGameEnv(Environment):

    def __init__(self, payoff_matrix: np.ndarray, iterations: int, batch_size=1):
        self._payoff_matrix = np.array(payoff_matrix, dtype=np.float32)
        self._iterations = iterations
        self._num_players = payoff_matrix.ndim - 1
        self._batch_size = batch_size
        self._t = 0

    def one_hot(self, x, n):
        return np.eye(n)[x]

    @property
    def num_players(self):
        return self._num_players

    def observation_spec(self):
        return dict(
            info_state=tuple([np.sum(self._payoff_matrix.shape[:-1])] for _ in range(self._num_players)),
            legal_actions=tuple([self._payoff_matrix.shape[p] for p in range(self._num_players)]),
            current_player=()
        )

    def action_spec(self):
        return dict(
            num_actions=tuple([self._payoff_matrix.shape[p] for p in range(self._num_players)]),
            min=tuple([0 for p in range(self._num_players)]),
            max=tuple([self._payoff_matrix.shape[p]-1 for p in range(self._num_players)]),
            dtype=int,
        )

    def step(self, actions: np.ndarray):
        if actions.ndim == 1:
            actions = actions[None, :]
        payoffs = self._payoff_matrix[tuple(actions.T)]
        info_state = np.concatenate([self.one_hot(actions[:, p], self._payoff_matrix.shape[p]) for p in range(self.num_players)], axis=-1)
        info_state = [np.squeeze(info_state).astype(np.float32)] * self._num_players
        rewards = [np.squeeze(p) for p in np.split(payoffs, indices_or_sections=self._num_players, axis=1)]
        discounts = [np.ones_like(r) for r in rewards]
        if self._t == self._iterations - 1:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        self._t += 1
        return TimeStep(
            observations=dict(
                info_state=info_state,
                legal_actions=[np.arange(self.action_spec()['num_actions'][p]) for p in range(self.num_players)],
                batch_size=actions.shape[0],
                current_player=PlayerId.SIMULTANEOUS
            ),
            rewards=rewards,
            discounts=discounts,
            step_type=step_type
        )

    def reset(self):
        self._t = 0
        info_state = np.squeeze(np.zeros((self.num_players, self._batch_size, *self.observation_spec()["info_state"][0])))
        rewards = np.squeeze(np.zeros((self.num_players, self._batch_size)))
        discounts = np.squeeze(np.ones((self.num_players, self._batch_size)))
        return TimeStep(
            observations=dict(
                info_state=[np.squeeze(s).astype(np.float32) for s in info_state],
                legal_actions=[np.arange(self.action_spec()['num_actions'][p]) for p in range(self.num_players)],
                batch_size=self._batch_size,
                current_player=PlayerId.SIMULTANEOUS
            ),
            rewards=[np.squeeze(a).astype(np.float32) for a in rewards],
            discounts=[np.squeeze(a).astype(np.float32) for a in discounts],
            step_type=StepType.FIRST
        )

def IteratedPrisonersDilemmaEnv(iterations: int, batch_size=1):
    return IteratedMatrixGameEnv(np.array([[[-1,-1], [-3,0]], [[0,-3], [-2,-2]]]), iterations, batch_size)

def make_iterated_matrix_game(game: str, config: dict) -> rl_environment.Environment:
    matrix_game = pyspiel.load_matrix_game(game)
    game = pyspiel.create_repeated_game(matrix_game, config)
    env = rl_environment.Environment(game)
    return env

if __name__ == '__main__':
    env = IteratedPrisonersDilemmaEnv(iterations=5)
    obs = env.reset()
    obs = env.step(np.array([0, 0]))
    obs = env.step(np.array([[-1,-1], [0, 1], [1, 0], [1, 1]]))

    pd_env = make_iterated_matrix_game("matrix_pd", {"num_players": 2, "game_iterations": 5})
    pd_obs = pd_env.reset()
    pd_step = pd_env.step(np.array([0, 0]))
    print(obs)
