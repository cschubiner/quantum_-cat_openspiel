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
# flags.DEFINE_bool("random_vs_random", True, "If True, evaluate random vs random play.")


def evaluate(agent, envs, player_id=0, num_episodes=20, self_play=False, random_vs_random=False):
    """Evaluates an agent for a specified number of episodes using synchronous evaluation.
    
    All environments run until completion before any are reset. This prevents mixing
    of stale actions with newly reset environments.
    """
    # Track completed episodes and total reward
    episodes_done = 0
    total_reward = 0
    
    # Track which envs are finished
    done_mask = [False] * len(envs.envs)
    
    # Initialize all environments together
    time_step = envs.reset()

    # Keep stepping until we've completed 'num_episodes' in total.
    while episodes_done < num_episodes:
        actions = []
        # Compute actions for all environments.
        for i, ts in enumerate(time_step):
            # If this environment is done or in terminal state,
            # just append a dummy action
            if done_mask[i] or ts.last():
                actions.append(0)
                continue

            p = ts.current_player()
            legal = ts.observations["legal_actions"][p]

            # Debug print: see the environment's current player, legal actions, and chosen action
            # print(f"[DEBUG] Env idx={i}, current_player={p}, legal={legal}", end="")

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

            # print(f" => chosen_action={chosen_action}")
            actions.append(chosen_action)

        # Step all environments together.
        step_out = [StepOutput(action=a, probs=None) for a in actions]
        next_time_step, reward, done, _ = envs.step(step_out)

        # Mark finished envs, accumulate reward, count episodes
        for i, dval in enumerate(done):
            if dval and not done_mask[i]:
                if reward[i] is not None:
                    total_reward += reward[i][player_id]
                episodes_done += 1
                done_mask[i] = True

        # Synchronous approach: if all envs are done OR we've hit the episode target,
        # reset everything (if there are still episodes left). Otherwise continue.
        if all(done_mask) or episodes_done >= num_episodes:
            if episodes_done < num_episodes:
                time_step = envs.reset()
                done_mask = [False] * len(envs.envs)
            else:
                # we've done enough episodes
                break
        else:
            # some envs still running
            time_step = next_time_step

    # Average reward per episode
    avg_rew = total_reward / num_episodes
    
    print(f"[Sync Eval] player_id={player_id}, episodes={num_episodes}, "
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

    # Make sure the environment shape matches the agent's shape
    if info_state_shape != agent.input_shape:
        raise ValueError(f"Mismatched environment shape {info_state_shape} vs "
                         f"agent shape {agent.input_shape}. Are you using an agent "
                         f"trained for {FLAGS.num_players} players?")

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
