"""Training script for Quantum Cat using PPO (Proximal Policy Optimization).

This trains multiple PPO agents to play Quantum Cat, supporting 3-5 players.
"""

import os
from datetime import datetime
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

from open_spiel.python.pytorch import ppo
from open_spiel.python import rl_environment
from open_spiel.python.utils import spawn
from open_spiel.python.vector_env import SyncVectorEnv
from open_spiel.python.games import quantum_cat

# run_quantum_cat_ppo.py
import random
import numpy as np
import torch

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.vector_env import SyncVectorEnv
from open_spiel.python.rl_agent import StepOutput

# Import your existing PPO implementation
# (You said you have this in open_spiel/python/pytorch/ppo.py)
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import legal_actions_to_mask

def run_ppo_on_quantum_cat(
    num_players=3,
    total_timesteps=2000,
    steps_per_batch=16,
    player_id=0,
    seed=1234,
):
    """
    Trains a single PPO agent on the quantum_cat game, controlling
    one player and letting the other players act randomly.

    Args:
      num_players: number of players in quantum_cat. (3..5)
      total_timesteps: total environment steps to run training.
      steps_per_batch: number of environment steps before each PPO update.
      player_id: which seat the PPO agent controls. 0 <= player_id < num_players.
      seed: random seed.
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the quantum_cat game
    game = pyspiel.load_game("python_quantum_cat", {"players": num_players})
    # Create an OpenSpiel RL environment for the game
    # By default, rl_environment.Environment will run all players inside a single env step,
    # but we only control the agent for 'player_id'; the others are random.
    def make_env():
        return rl_environment.Environment(
            game=game, seed=seed, players=num_players
        )

    # For better PPO performance, create multiple synchronous environments
    # that run in parallel. Here we do 1 or 2 as an example, but you can
    # increase this.
    num_envs = 2
    # Create actual environment instances by calling the make_env function
    envs = SyncVectorEnv([make_env() for _ in range(num_envs)])

    # The environment returns time steps for all players, but we only
    # train on player_id's perspective. Let's get that info.
    sample_timestep = envs.reset()
    obs_spec = sample_timestep[0].observations["info_state"][player_id]
    info_state_shape = (len(obs_spec),)  # Flattened shape of info_state (19,)
    num_actions = game.num_distinct_actions()

    # Initialize the PPO agent with PPOAgent for vector observations
    agent = PPO(
        input_shape=info_state_shape,
        num_actions=num_actions,
        num_players=num_players,
        player_id=player_id,
        num_envs=num_envs,
        steps_per_batch=steps_per_batch,
        update_epochs=4,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",  # or "cuda" if GPU is available
        agent_fn=ppo.PPOAgent,  # Use PPOAgent for vector inputs instead of PPOAtariAgent
    )

    # Helper function: for players we do NOT control,
    # pick random legal actions:
    def random_policy(time_steps):
        actions = []
        for ts in time_steps:
            # ts is a TimeStep for exactly one player
            legal_acts = ts.observations["legal_actions"][ts.current_player()]
            actions.append(random.choice(legal_acts))
        return actions

    # Training loop
    steps_done = 0
    time_step = envs.reset()
    while steps_done < total_timesteps:
        # Step until we fill up agent's batch
        for _ in range(steps_per_batch):
            # Create an action array for all players
            actions = [None] * num_players

            # For each environment, we have num_players time_steps
            # but we only have to pick an action for the "current_player"
            # if it's our agent's seat (player_id).
            for env_idx in range(num_envs):
                # The environment's observation for each seat:
                all_player_ts = time_step[env_idx]

                # We'll pick random actions for everyone
                # except if it's the agent's turn.
                current_p = all_player_ts.current_player()
                if current_p == player_id and not all_player_ts.last():
                    # PPO agent picks action
                    agent_output = agent.step([all_player_ts], is_evaluation=False)
                    actions[current_p] = agent_output[0].action
                else:
                    # For other seats or if it's terminal for the agent, random action
                    # (Though if last() is True, the environment step will ignore actions.)
                    legal_acts = all_player_ts.observations["legal_actions"][current_p]
                    actions[current_p] = random.choice(legal_acts)

            # Because SyncVectorEnv expects an array of shape [num_envs, num_players]
            # But each environment only needs the action for the "current_player."
            # We fill in random actions for each environment's other seats:
            # So let's do that for each environment's players:
            # If the seat is not already filled, fill it randomly.
            # (In many cases, the environment only uses the current player's
            #  chosen action and ignores the rest, but we fill them anyway.)
            for env_idx in range(num_envs):
                for p_id in range(num_players):
                    if actions[p_id] is None:
                        ts_p = time_step[env_idx]
                        if ts_p.last():
                            # If the environment is done for this env, action is irrelevant
                            actions[p_id] = 0
                        else:
                            legal_acts = ts_p.observations["legal_actions"][p_id]
                            actions[p_id] = random.choice(legal_acts)

            # Convert actions to StepOutput objects - one per environment
            step_outputs = [StepOutput(action=actions[time_step[i].current_player()], probs=None) 
                          for i in range(num_envs)]

            # Step the vector environment
            next_time_step, reward, done, _ = envs.step(step_outputs)
            # Extract just our agent's rewards and done flags
            agent_rewards = [r[player_id] for r in reward]
            agent_dones = [d[player_id] for d in done]
            agent.post_step(agent_rewards, agent_dones)

            # Bookkeeping
            time_step = next_time_step
            steps_done += num_envs
            if steps_done >= total_timesteps:
                break

        # Once we have a full batch, do learning
        agent.learn(time_step[:, player_id])

    # After training, do a quick evaluation:
    # We'll evaluate by letting our agent pick actions deterministically
    # (is_evaluation=True) and keep other players random.
    # We'll run ~20 episodes for a rough measure of agent's performance.
    total_eval_reward = 0
    n_episodes = 20
    episodes_done = 0
    time_step = envs.reset()
    while episodes_done < n_episodes:
        actions = [None] * num_players
        for env_idx in range(num_envs):
            ts = time_step[env_idx]
            if ts.last():
                # the environment ended for this player
                pass
            current_p = ts.current_player()
            if current_p == player_id and not ts.last():
                # Agent picks an action deterministically
                agent_output = agent.step([ts], is_evaluation=True)
                actions[current_p] = agent_output[0].action
            else:
                # random for other seats
                legal_acts = ts.observations["legal_actions"][current_p]
                actions[current_p] = random.choice(legal_acts)

        # Convert actions to StepOutput objects - one per environment
        step_outputs = [StepOutput(action=actions[time_step[i].current_player()], probs=None)
                       for i in range(num_envs)]

        next_time_step, reward, done, _ = envs.step(step_outputs)
        total_eval_reward += sum(reward)
        # Count how many envs finished an episode
        episodes_done += sum(1 for d in done if d)
        time_step = next_time_step

    avg_eval_reward = total_eval_reward / n_episodes
    print(f"Evaluation over {n_episodes} episodes, avg reward = {avg_eval_reward:.2f}")

def main():
    run_ppo_on_quantum_cat(
        num_players=3,
        total_timesteps=2000,
        steps_per_batch=16,
        player_id=0,
        seed=1234,
    )

if __name__ == "__main__":
    main()
