#!/usr/bin/env python3
"""
Training script for Quantum Cat using PPO (Proximal Policy Optimization).

This trains multiple PPO agents to play Quantum Cat, supporting 3-5 players,
uses tqdm for progress, and saves the agent after training.

We also include a function to evaluate the trained agent vs. random or self-play.
"""

import os
import random
import time
from collections import namedtuple, deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from absl import app
from absl import flags
from absl import logging

from tqdm import tqdm

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.rl_agent import StepOutput
from open_spiel.python.vector_env import SyncVectorEnv

# Import your existing PPO implementation
# (It must match your open_spiel/python/pytorch/ppo.py)
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import PPOAgent

from open_spiel.python.games import quantum_cat

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_players", 3, "Number of players in Quantum Cat (3..5).")
flags.DEFINE_integer("num_episodes", 2000, "Number of full games to train for.")
flags.DEFINE_integer("steps_per_batch", 16, "Environment steps per PPO update.")
flags.DEFINE_integer("seed", 1234, "Random seed.")
flags.DEFINE_integer("num_envs", 2, "Number of vectorized envs.")
flags.DEFINE_string("save_path", "quantum_cat_agent.pth", "Where to save the agent.")
flags.DEFINE_bool("use_tensorboard", False, "Whether to log to TensorBoard.")

# If you want to run from CLI: python quantum_cat_ppo.py --num_players=3 --num_episodes=1000 ...
# or just call main() programmatically.

def make_env(game, seed, num_players):
    """Creates an OpenSpiel RL environment for Quantum Cat."""
    # Each call returns a fresh environment instance
    return rl_environment.Environment(
        game=game, seed=seed, players=num_players
    )

def run_ppo_on_quantum_cat(
    num_players=3,
    num_episodes=1000,
    steps_per_batch=16,
    player_id=0,
    seed=1234,
    num_envs=1,
    save_path="quantum_cat_agent.pth",
    use_tensorboard=False
):
    """Trains a single PPO agent on the quantum_cat game vs. PPO opponents.

    Args:
      num_players: how many total players in the game (3..5)
      num_episodes: how many episodes (full games) of training
      steps_per_batch: environment steps before each PPO update
      player_id: which seat the 'main' PPO agent occupies
      seed: random seed
      num_envs: how many synchronous envs for parallel training
      save_path: file path to save the agent
      use_tensorboard: if True, logs training info to TensorBoard

    Returns:
      The trained PPO agent (and any opponents).
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the quantum_cat game with N players
    game = pyspiel.load_game("python_quantum_cat", {"players": num_players})

    # Create multiple vector envs
    envs = SyncVectorEnv([
        make_env(game, seed + i, num_players) for i in range(num_envs)
    ])

    # We only train the agent for 'player_id'.
    # The others are also PPO-based, or random, as you prefer.

    # Figure out state shape
    sample_ts = envs.reset()
    obs_spec = sample_ts[0].observations["info_state"][player_id]
    info_state_shape = (len(obs_spec),)
    num_actions = game.num_distinct_actions()

    # Initialize main agent
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
        device="cpu",  # or "cuda" if GPU available
        agent_fn=PPOAgent,
    )

    # Initialize opponents as PPO or random
    opponents = {}
    for opp_id in range(num_players):
        if opp_id != player_id:
            opp = PPO(
                input_shape=info_state_shape,
                num_actions=num_actions,
                num_players=num_players,
                player_id=opp_id,
                num_envs=num_envs,
                steps_per_batch=steps_per_batch,
                update_epochs=4,
                learning_rate=2.5e-4,
                gae=True,
                gamma=0.99,
                gae_lambda=0.95,
                device="cpu",
                agent_fn=PPOAgent,
            )
            opponents[opp_id] = opp

    writer = None
    if use_tensorboard:
        writer = SummaryWriter()

    episodes_done = 0
    time_step = envs.reset()

    # We'll use tqdm to track progress in terms of completed episodes
    with tqdm(total=num_episodes, desc="Training Episodes") as pbar:
        while episodes_done < num_episodes:
            # step until we fill up agent's batch
            for _ in range(steps_per_batch):
                env_actions = []
                for i in range(num_envs):
                    ts = time_step[i]
                    current_p = ts.current_player()
                    if ts.last():
                        # If terminal, no action needed (use dummy 0)
                        env_actions.append(0)
                    elif current_p == player_id:
                        # main agent
                        agent_output = agent.step([ts], is_evaluation=False)
                        env_actions.append(agent_output[0].action)
                    else:
                        # PPO opponent or random
                        opp = opponents.get(current_p, None)
                        if opp:
                            opp_out = opp.step([ts], is_evaluation=False)
                            env_actions.append(opp_out[0].action)
                        else:
                            # fallback random
                            legal_acts = ts.observations["legal_actions"][current_p]
                            env_actions.append(random.choice(legal_acts) if legal_acts else 0)

                step_outputs = [StepOutput(action=a, probs=None) for a in env_actions]
                next_time_step, reward, done, _ = envs.step(step_outputs)

                # post_step for main agent & opponents
                for pid in range(num_players):
                    # gather each player's reward from vector env
                    if pid == player_id:
                        agent_rewards = [r[pid] if r is not None else 0.0 for r in reward]
                        agent.post_step(agent_rewards, done)
                    elif pid in opponents:
                        opp_rewards = [r[pid] if r is not None else 0.0 for r in reward]
                        opponents[pid].post_step(opp_rewards, done)

                # count finished episodes
                finished_episodes = sum(1 for d in done if d)
                episodes_done += finished_episodes
                pbar.update(finished_episodes)

                if episodes_done >= num_episodes:
                    break

                time_step = next_time_step

            # Once we have a full batch, do learning
            agent_timesteps = [ts for ts in time_step]
            agent.learn(agent_timesteps)
            for opp in opponents.values():
                opp.learn(agent_timesteps)

            # Optionally log something
            if writer is not None:
                writer.add_scalar("training/episodes_done", episodes_done, episodes_done)

    # Save the agent
    torch.save(agent.state_dict(), save_path)
    print(f"Saved trained agent to {save_path}")

    if writer is not None:
        writer.close()

    return agent, opponents

def evaluate_agent(agent, envs, player_id=0, num_episodes=20, opponents=None):
    """
    Evaluates `agent` in `envs` for `num_episodes`, where `agent` is for `player_id`.
    By default, other players pick random actions (or use given `opponents`).
    """
    total_eval_reward = 0
    episodes_done = 0
    time_step = envs.reset()

    while episodes_done < num_episodes:
        env_actions = []
        for i in range(envs.num_envs):
            ts = time_step[i]
            current_p = ts.current_player()
            if ts.last():
                # terminal -> dummy action
                env_actions.append(0)
            elif current_p == player_id:
                agent_out = agent.step([ts], is_evaluation=True)
                env_actions.append(agent_out[0].action)
            else:
                # random or opponent
                if opponents and current_p in opponents:
                    opp_out = opponents[current_p].step([ts], is_evaluation=True)
                    env_actions.append(opp_out[0].action)
                else:
                    legals = ts.observations["legal_actions"][current_p]
                    env_actions.append(random.choice(legals) if legals else 0)

        step_outputs = [StepOutput(action=a, probs=None) for a in env_actions]
        next_time_step, reward, done, _ = envs.step(step_outputs)
        total_eval_reward += sum(r[player_id] if r is not None else 0.0 for r in reward)
        episodes_done += sum(1 for d in done if d)
        time_step = next_time_step

    avg_eval_reward = total_eval_reward / num_episodes
    print(f"[Eval] Player {player_id} => episodes={num_episodes}, avg reward={avg_eval_reward}")
    return avg_eval_reward

def main(_):
    num_players = FLAGS.num_players
    num_episodes = FLAGS.num_episodes
    steps_per_batch = FLAGS.steps_per_batch
    seed = FLAGS.seed
    num_envs = FLAGS.num_envs
    save_path = FLAGS.save_path
    use_tb = FLAGS.use_tensorboard

    # We'll train the agent for player_id=0 (just a default choice)
    agent, opponents = run_ppo_on_quantum_cat(
        num_players=num_players,
        num_episodes=num_episodes,
        steps_per_batch=steps_per_batch,
        player_id=0,
        seed=seed,
        num_envs=num_envs,
        save_path=save_path,
        use_tensorboard=use_tb
    )

    # Evaluate vs random or self-play:
    # reuse the same #envs and game
    game = pyspiel.load_game("python_quantum_cat", {"players": num_players})
    eval_envs = SyncVectorEnv([
        rl_environment.Environment(game=game, players=num_players)
        for _ in range(num_envs)
    ])

    # Evaluate the trained agent vs random
    evaluate_agent(agent, eval_envs, player_id=0, num_episodes=20)

    # Evaluate self-play (i.e. using same agent for all players)
    # We'll just pass a dictionary of {pid: agent} for all pids
    # so they all share the same policy
    sp_opponents = {pid: agent for pid in range(num_players) if pid != 0}
    evaluate_agent(agent, eval_envs, player_id=0, num_episodes=20, opponents=sp_opponents)

if __name__ == "__main__":
    app.run(main)
