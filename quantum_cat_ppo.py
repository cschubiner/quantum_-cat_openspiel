"""Training script for Quantum Cat using PPO (Proximal Policy Optimization).

This trains multiple PPO agents to play Quantum Cat, supporting 3-5 players.
"""

import os
from datetime import datetime
import time
from tqdm import tqdm
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
    num_episodes=1000,
    steps_per_batch=16,
    player_id=0,
    seed=1234,
):
    """
    Trains a single PPO agent on the quantum_cat game vs. PPO opponents,
    for num_episodes full games.

    Args:
      num_players: number of players in quantum_cat. (3..5)
      num_episodes: number of full games to train for
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

    # Create PPO opponents for other seats
    opponents = {}
    for opp_id in range(num_players):
        if opp_id != player_id:
            opp_agent = PPO(
                input_shape=info_state_shape,
                num_actions=num_actions,
                num_players=num_players,
                player_id=opp_id,
                num_envs=num_envs,
                steps_per_batch=steps_per_batch,
                update_epochs=4,  # They will learn too
                learning_rate=2.5e-4,
                gae=True,
                gamma=0.99,
                gae_lambda=0.95,
                device="cpu",
                agent_fn=ppo.PPOAgent,
            )
            opponents[opp_id] = opp_agent

    # Helper function: for players we do NOT control (fallback),
    # pick random legal actions:
    def random_policy(time_steps):
        actions = []
        for ts in time_steps:
            # ts is a TimeStep for exactly one player
            legal_acts = ts.observations["legal_actions"][ts.current_player()]
            actions.append(random.choice(legal_acts))
        return actions

    # Training loop
    episodes_done = 0
    time_step = envs.reset()
    
    # Training progress bar
    with tqdm(total=num_episodes, desc="Training Episodes") as pbar:
        while episodes_done < num_episodes:
        # Step until we fill up agent's batch
        for _ in range(steps_per_batch):
            # We only need ONE action per environment (the current player's action)
            env_actions = []
            for i in range(num_envs):
                ts = time_step[i]
                current_p = ts.current_player()
                if current_p == player_id and not ts.last():
                    # Main PPO agent picks action
                    agent_output = agent.step([ts], is_evaluation=False)
                    env_actions.append(agent_output[0].action)
                else:
                    # PPO opponents or terminal state handling
                    if current_p == pyspiel.PlayerId.TERMINAL or ts.last():
                        action = 0  # Dummy action for terminal states
                    else:
                        # Get the PPO opponent for this seat
                        opp = opponents.get(current_p)
                        if opp is not None:
                            opp_output = opp.step([ts], is_evaluation=False)
                            action = opp_output[0].action
                        else:
                            # Fallback to random if no opponent found
                            legal_acts = ts.observations["legal_actions"][current_p]
                            action = random.choice(legal_acts) if legal_acts else 0
                    env_actions.append(action)

            # Convert to StepOutput objects - one per environment
            step_outputs = [StepOutput(action=a, probs=None) for a in env_actions]

            # Step the vector environment with the StepOutput objects
            next_time_step, reward, done, _ = envs.step(step_outputs)
            # Handle rewards and done flags for all agents
            for pid in range(num_players):
                if pid == player_id:
                    agent_rewards = [r[pid] if r is not None else 0.0 for r in reward]
                    agent.post_step(agent_rewards, done)
                elif pid in opponents:
                    opp_rewards = [r[pid] if r is not None else 0.0 for r in reward]
                    opponents[pid].post_step(opp_rewards, done)

            # Count completed episodes
            completed = sum(1 for d in done if d)
            episodes_done += completed
            pbar.update(completed)
            if episodes_done >= num_episodes:
                break

            # Continue to next timestep
            time_step = next_time_step

        # Once we have a full batch, do learning for all agents
        agent_timesteps = [ts for ts in time_step]
        agent.learn(agent_timesteps)
        for opp in opponents.values():
            opp.learn(agent_timesteps)

    # Save the trained agent
    save_path = f"quantum_cat_agent_p{player_id}.pth"
    torch.save(agent.state_dict(), save_path)
    print(f"Saved trained agent to {save_path}")
    
    # Evaluate vs random
    print("\nEvaluating vs random opponents:")
    evaluate_agent(agent, envs, num_episodes=20, player_id=player_id)
    
    # Evaluate self-play
    print("\nEvaluating self-play:")
    evaluate_agent(agent, envs, num_episodes=20, player_id=player_id, opponent_agent=agent)
        # We only need ONE action per environment (the current player's action)
        env_actions = []
        for i in range(num_envs):
            ts = time_step[i]
            current_p = ts.current_player()
            if current_p == player_id and not ts.last():
                # Agent picks action deterministically
                agent_output = agent.step([ts], is_evaluation=True)
                env_actions.append(agent_output[0].action)
            else:
                # Terminal state handling
                if current_p == pyspiel.PlayerId.TERMINAL or ts.last():
                    action = 0  # Dummy action for terminal states
                else:
                    # Either use opponent_agent (self-play) or random
                    if opponent_agent is not None:
                        opp_output = opponent_agent.step([ts], is_evaluation=True)
                        action = opp_output[0].action
                    else:
                        legal_acts = ts.observations["legal_actions"][current_p]
                        action = random.choice(legal_acts) if legal_acts else 0
                env_actions.append(action)

        # Convert to StepOutput objects - one per environment
        step_outputs = [StepOutput(action=a, probs=None) for a in env_actions]

        # Step the vector environment with the StepOutput objects
        next_time_step, reward, done, _ = envs.step(step_outputs)
        total_eval_reward += sum(r[player_id] if r is not None else 0.0 for r in reward)
        # Count how many envs finished an episode
        episodes_done += sum(1 for d in done if d)
        time_step = next_time_step

    avg_eval_reward = total_eval_reward / num_episodes
    mode = "self-play" if opponent_agent else "vs random"
    print(f"Evaluation ({mode}), episodes={num_episodes}, avg reward={avg_eval_reward:.2f}")
    return avg_eval_reward

def main():
    run_ppo_on_quantum_cat(
        num_players=3,
        num_episodes=1000,
        steps_per_batch=16,
        player_id=0,
        seed=1234,
    )

if __name__ == "__main__":
    main()
