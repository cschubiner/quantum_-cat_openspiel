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


FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("game_name", "python_quantum_cat", "Name of the game")
flags.DEFINE_integer("num_players", 5, "Number of players (3-5 supported)")
flags.DEFINE_integer("num_envs", 8, "Number of parallel environments")
flags.DEFINE_integer("num_training_steps", 10_000_000, "Number of training steps")
flags.DEFINE_integer("eval_every", 10000, "Evaluate every N steps")
flags.DEFINE_string("checkpoint_dir", "quantum_cat_ppo", "Directory for checkpoints")

# PPO specific parameters
flags.DEFINE_float("learning_rate", 2.5e-4, "Learning rate")
flags.DEFINE_integer("num_minibatches", 4, "Number of minibatches per update")
flags.DEFINE_integer("update_epochs", 4, "Number of epochs per update")
flags.DEFINE_float("gamma", 0.99, "Discount factor")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda parameter")
flags.DEFINE_float("clip_coef", 0.2, "PPO clip coefficient")
flags.DEFINE_float("ent_coef", 0.01, "Entropy coefficient")
flags.DEFINE_float("vf_coef", 0.5, "Value function coefficient")
flags.DEFINE_float("max_grad_norm", 0.5, "Maximum gradient norm")

def create_env():
    """Creates a new environment instance."""
    return rl_environment.Environment(
        FLAGS.game_name,
        players=FLAGS.num_players
    )

def main(_):
    # Set up logging
    logging.set_verbosity(logging.INFO)

    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(FLAGS.checkpoint_dir, "tensorboard"))

    # Get environment specs from a single environment instance
    test_env = create_env()
    obs_shape = test_env.observation_spec()["info_state"][0]
    num_actions = test_env.action_spec()["num_actions"]

    # Create PPO agents (one per player)
    agents = []
    for player_id in range(FLAGS.num_players):
        agent = ppo.PPO(
            input_shape=tuple([obs_shape]),  # PPO expects a tuple
            num_actions=num_actions,
            num_players=FLAGS.num_players,
            player_id=player_id,
            num_envs=FLAGS.num_envs,
            learning_rate=FLAGS.learning_rate,
            num_minibatches=FLAGS.num_minibatches,
            update_epochs=FLAGS.update_epochs,
            gamma=FLAGS.gamma,
            gae_lambda=FLAGS.gae_lambda,
            clip_coef=FLAGS.clip_coef,
            entropy_coef=FLAGS.ent_coef,
            value_coef=FLAGS.vf_coef,
            max_grad_norm=FLAGS.max_grad_norm,
            device="cuda" if torch.cuda.is_available() else "cpu",
            writer=writer,  # Use tensorboard for logging
            agent_fn=ppo.PPOAgent  # Use standard network (not Atari)
        )
        agents.append(agent)

    # Create vector environment - create actual environments, not just functions
    envs = [create_env() for _ in range(FLAGS.num_envs)]
    vector_env = SyncVectorEnv(envs)

    # Training loop
    step = 0
    while step < FLAGS.num_training_steps:
        # Collect experience
        time_steps = vector_env.reset()
        episode_returns = [0.0] * FLAGS.num_players

        # Each time_step in time_steps is actually a list containing one time step
        while not all(ts[0].last() for ts in time_steps):
            player_ids = [ts[0].observations["current_player"] for ts in time_steps]
            actions = []

            for env_idx, player_id in enumerate(player_ids):
                if player_id >= 0:  # Not a chance node
                    # The time step is already a list containing one element
                    agent_output = agents[player_id].step(time_steps[env_idx])
                    # agent_output is a list of StepOutput objects, but we only have one
                    actions.append(agent_output[0])
                else:
                    actions.append(None)  # Chance node

            next_time_steps = vector_env.step(actions)

            # Post-step updates for each agent
            for i, (ts, next_ts) in enumerate(zip(time_steps, next_time_steps)):
                if ts[0].rewards is not None:  # Some transitions might not have rewards
                    for pid, reward in enumerate(ts[0].rewards):
                        episode_returns[pid] += reward
                        agents[pid].post_step(reward, next_ts[0].last())

            time_steps = next_time_steps

        # Update all agents
        for pid, agent in enumerate(agents):
            # Each time step is already a list containing one element
            agent.learn(time_steps[0])

        step += FLAGS.num_envs

        # Evaluation and checkpointing
        if step % FLAGS.eval_every < FLAGS.num_envs:
            logging.info("Step %d: Returns %s", step, episode_returns)

            # Log to tensorboard
            for pid, returns in enumerate(episode_returns):
                writer.add_scalar(f"player_{pid}/episode_return", returns, step)

            # Save checkpoints
            checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, f"step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            for pid, agent in enumerate(agents):
                path = os.path.join(checkpoint_dir, f"agent_{pid}.pt")
                torch.save(agent.state_dict(), path)

if __name__ == "__main__":
    with spawn.main_handler():
        app.run(main)
