#!/usr/bin/env python3
"""
Script to evaluate a saved PPO agent on Quantum Cat.

Usage:
  python evaluate_quantum_cat.py --num_players=3 --num_episodes=20 \
    --agent_path=quantum_cat_agent.pth
"""

import random

import numpy as np
import torch
import pyspiel

from absl import app
from absl import flags

from open_spiel.python import rl_environment
from open_spiel.python.rl_agent import StepOutput
from open_spiel.python.vector_env import SyncVectorEnv

from open_spiel.python.pytorch.ppo import PPO, PPOAgent  # or wherever your PPO is
# Make sure quantum_cat is registered
from open_spiel.python.games import quantum_cat

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_players", 3, "Number of players.")
flags.DEFINE_integer("num_episodes", 600, "Number of episodes to evaluate.")
flags.DEFINE_string("agent_path", "quantum_cat_agent_2880001.pth", "Path to saved agent.")
flags.DEFINE_bool("self_play", False, "If True, use same agent for all players.")
flags.DEFINE_bool("random_vs_random", False, "If True, evaluate random vs random play.")


def evaluate(agent, envs, player_id=0, num_episodes=20, self_play=False, random_vs_random=False):
    """Evaluates an agent for a specified number of episodes.

    Once an environment is done, we reset it immediately (unless we've already
    finished the total desired number of episodes). This ensures that each
    environment contributes fully to the episode count.
    """
    # Number of episodes completed so far across all environments.
    episodes_done = 0

    # Sum of rewards earned by 'player_id' across all episodes.
    total_reward = 0

    # Initialize all environments.
    time_step = envs.reset()

    # Keep stepping until we've completed 'num_episodes' in total.
    while episodes_done < num_episodes:
        actions = []
        # Compute actions for all environments.
        for i, ts in enumerate(time_step):
            if ts.last():
                # If an environment is already in a terminal state, it just needs
                # a dummy action for now.
                actions.append(0)
                continue

            p = ts.current_player()
            legal = ts.observations["legal_actions"][p]

            if random_vs_random:
                # Everyone plays random in this mode.
                chosen_action = random.choice(legal) if legal else 0
            else:
                if p == player_id:
                    out = agent.step([ts], is_evaluation=True)
                    chosen_action = out[0].action
                else:
                    if self_play:
                        # Same agent for all players.
                        out = agent.step([ts], is_evaluation=True)
                        chosen_action = out[0].action
                    else:
                        # Other players (not 'player_id') act randomly.
                        chosen_action = random.choice(legal) if legal else 0

            actions.append(chosen_action)

        # Step all environments together.
        step_out = [StepOutput(action=a, probs=None) for a in actions]
        next_time_step, reward, done, _ = envs.step(step_out)

        # Process any environments that have finished.
        for i, dval in enumerate(done):
            if dval:
                # If reward[i] is not None, accumulate the agent's reward.
                if reward[i] is not None:
                    total_reward += reward[i][player_id]

                episodes_done += 1

                # If we still need more episodes, reset just this environment.
                if episodes_done < num_episodes:
                    # Reset all environments and update just the one we need
                    temp_reset = envs.reset()
                    next_time_step[i] = temp_reset[i]

        # Update time steps for the next loop iteration.
        time_step = next_time_step

    # Average reward per episode.
    avg_rew = total_reward / num_episodes

    print(f"[Evaluate] player_id={player_id}, episodes={num_episodes}, "
          f"avg reward={avg_rew:.2f}, episodes_done={episodes_done}")
    return avg_rew


def main(_):
    # Load game
    game = pyspiel.load_game("python_quantum_cat", {"players": FLAGS.num_players})

    # Setup env
    num_envs = 8  # or however many parallel envs you want
    envs = SyncVectorEnv([
        rl_environment.Environment(game=game)
        for _ in range(num_envs)
    ])

    # Build agent with the same architecture/params used in training
    sample_ts = envs.reset()
    obs_spec = sample_ts[0].observations["info_state"][0]
    info_state_shape = (len(obs_spec),)
    num_actions = game.num_distinct_actions()

    agent = PPO(
        input_shape=info_state_shape,
        num_actions=num_actions,
        num_players=FLAGS.num_players,
        player_id=0,
        num_envs=num_envs,
        steps_per_batch=16,  # not used heavily in eval
        update_epochs=4,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        agent_fn=PPOAgent,
    )

    # Load weights
    agent.load_state_dict(torch.load(FLAGS.agent_path, map_location="cpu"))
    agent.eval()

    # Evaluate
    if FLAGS.random_vs_random:
        print("Evaluating random vs random play...")
        evaluate(agent, envs, player_id=0, num_episodes=FLAGS.num_episodes,
                 self_play=False, random_vs_random=True)
    else:
        evaluate(agent, envs, player_id=0, num_episodes=FLAGS.num_episodes,
                 self_play=FLAGS.self_play, random_vs_random=False)


if __name__ == "__main__":
    app.run(main)
