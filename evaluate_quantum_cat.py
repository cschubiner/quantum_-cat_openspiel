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
flags.DEFINE_integer("num_episodes", 20, "Number of episodes to evaluate.")
flags.DEFINE_string("agent_path", "quantum_cat_agent.pth", "Path to saved agent.")
flags.DEFINE_bool("self_play", False, "If True, use same agent for all players.")

def evaluate(agent, envs, player_id=0, num_episodes=20, self_play=False):
    total_reward = 0
    episodes_done = 0
    time_step = envs.reset()

    while episodes_done < num_episodes:
        actions = []
        for i in range(envs.num_envs):
            ts = time_step[i]
            p = ts.current_player()
            if ts.last():
                actions.append(0)  # dummy
                continue
            if p == player_id:
                out = agent.step([ts], is_evaluation=True)
                actions.append(out[0].action)
            else:
                if self_play:
                    # same agent for other players
                    out = agent.step([ts], is_evaluation=True)
                    actions.append(out[0].action)
                else:
                    # random
                    legal = ts.observations["legal_actions"][p]
                    actions.append(random.choice(legal) if legal else 0)

        step_out = [StepOutput(action=a, probs=None) for a in actions]
        next_time_step, reward, done, _ = envs.step(step_out)
        total_reward += sum(r[player_id] if r is not None else 0 for r in reward)
        episodes_done += sum(1 for d in done if d)
        time_step = next_time_step

    avg_rew = total_reward / num_episodes
    print(f"[Evaluate] player_id={player_id}, episodes={num_episodes}, avg reward={avg_rew:.2f}")
    return avg_rew

def main(_):
    # Load game
    game = pyspiel.load_game("python_quantum_cat", {"players": FLAGS.num_players})

    # Setup env
    num_envs = 2  # or however many parallel envs
    envs = SyncVectorEnv([
        lambda: rl_environment.Environment(game=game, players=FLAGS.num_players)
        for _ in range(num_envs)
    ])

    # Build agent with same shape/params you used in training
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
        steps_per_batch=16,  # arbitrary; not used heavily in eval
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
    evaluate(agent, envs, player_id=0, num_episodes=FLAGS.num_episodes, self_play=FLAGS.self_play)

if __name__ == "__main__":
    app.run(main)
