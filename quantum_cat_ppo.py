#!/usr/bin/env python3
"""
Training script for Quantum Cat using PPO (Proximal Policy Optimization).

This trains multiple PPO agents to play Quantum Cat, supporting 3-5 players,
uses tqdm for progress, and saves the agent after training.

We also include a function to evaluate the trained agent vs. random or self-play.
"""

import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from absl import app, flags
from tqdm import tqdm
import pyspiel
from evaluate_quantum_cat import evaluate

from open_spiel.python import rl_environment
from open_spiel.python.rl_agent import StepOutput
from open_spiel.python.vector_env import SyncVectorEnv
# Import your existing PPO implementation
# (It must match your open_spiel/python/pytorch/ppo.py)
from open_spiel.python.pytorch.ppo import PPO, PPOAgent

from open_spiel.python.games import quantum_cat

def pick_opponent_type():
    """Randomly choose from PPO, random, or follow_suit."""
    return random.choice(["ppo", "random", "follow_suit"])

def pick_follow_suit_action(legal_actions, info_state, num_card_types):
    """Tries to match the led color if possible."""
    num_players = 3  # Hardcoded for now
    led_color_start = num_players + 5
    led_color = info_state[led_color_start:led_color_start + 5]
    led_color_idx = int(np.argmax(led_color))  # 0..3 => R,B,Y,G, 4 => None
    
    if led_color_idx == 4 or 999 in legal_actions:  # No led color or paradox
        return random.choice(legal_actions)
        
    # Try to find actions that follow suit
    follow_suit = []
    for action in legal_actions:
        if action == 999:  # PARADOX
            continue
        color_idx = action // num_card_types
        if color_idx == led_color_idx:
            follow_suit.append(action)
            
    if follow_suit:
        return random.choice(follow_suit)
    return random.choice(legal_actions)

EVALUATE_EVERY_X_EPISODES = 5000

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_players", 3, "Number of players in Quantum Cat (3..5).")
flags.DEFINE_integer("num_episodes", 250000000, "Number of full games to train for.")
# flags.DEFINE_integer("num_episodes", 3000, "Number of full games to train for.")
flags.DEFINE_integer("steps_per_batch", 4096, "Environment steps per PPO update.")
flags.DEFINE_integer("seed", 1234, "Random seed.")
flags.DEFINE_integer("num_envs", 12, "Number of vectorized envs.")
flags.DEFINE_string("save_path", "quantum_cat_agent_v2.pth", "Where to save the agent.")
flags.DEFINE_bool("use_tensorboard", True, "Whether to log to TensorBoard.")

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

    # Setup device - try to use MPS (M1 GPU) if available, else CUDA, else CPU
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("Using Apple M1 GPU (MPS)")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

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

    # Setup tensorboard
    writer = None
    if use_tensorboard:
        writer = SummaryWriter()

    # Initialize main agent
    agent = PPO(
        input_shape=info_state_shape,
        num_actions=num_actions,
        num_players=num_players,
        player_id=player_id,
        num_envs=num_envs,
        steps_per_batch=steps_per_batch,
        update_epochs=8,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
        agent_fn=PPOAgent,
        writer=writer,
    )

    # Initialize opponents as PPO, random, or follow_suit
    opponents = {}
    for opp_id in range(num_players):
        if opp_id == player_id:
            continue
        opp_type = pick_opponent_type()
        if opp_type == "ppo":
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
                device=device,
                agent_fn=PPOAgent,
                writer=writer,
            )
            opponents[opp_id] = opp
        else:
            # Store just the string: "random" or "follow_suit"
            opponents[opp_id] = opp_type

    episodes_done = 0
    train_steps = 0  # counts environment steps, used for tensorboard logging
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
                        opp = opponents.get(current_p, None)
                        if isinstance(opp, PPO):
                            # PPO opponent
                            opp_out = opp.step([ts], is_evaluation=False)
                            env_actions.append(opp_out[0].action)
                        elif opp == "random":
                            # Random opponent
                            legal_acts = ts.observations["legal_actions"][current_p]
                            env_actions.append(random.choice(legal_acts) if legal_acts else 0)
                        elif opp == "follow_suit":
                            # Follow suit opponent
                            legal_acts = ts.observations["legal_actions"][current_p]
                            info_st = ts.observations["info_state"][current_p]
                            num_card_types = game.num_distinct_actions() // 4
                            env_actions.append(
                                pick_follow_suit_action(legal_acts, info_st, num_card_types)
                                if legal_acts else 0
                            )
                        else:
                            # Fallback random
                            legal_acts = ts.observations["legal_actions"][current_p]
                            env_actions.append(random.choice(legal_acts) if legal_acts else 0)

                step_outputs = [StepOutput(action=a, probs=None) for a in env_actions]
                next_time_step, reward, done, _ = envs.step(step_outputs)

                # Log per-step rewards to TensorBoard
                if writer is not None:
                    for i in range(num_envs):
                        # If reward[i] is None, it means no reward returned this step
                        if reward[i] is not None:
                            # If done[i] is False, this is an incremental step reward
                            if not done[i]:
                                writer.add_scalar("rewards/incremental", reward[i][player_id], train_steps)
                            else:
                                # If done[i] is True, it includes the final "lump sum" from the environment
                                writer.add_scalar("rewards/final", reward[i][player_id], train_steps)
                train_steps += num_envs

                # post_step for main agent & opponents
                for pid in range(num_players):
                    # gather each player's reward from vector env
                    if pid == player_id:
                        agent_rewards = [r[pid] if r is not None else 0.0 for r in reward]
                        agent.post_step(agent_rewards, done)
                    elif pid in opponents and isinstance(opponents[pid], PPO):
                        opp_rewards = [r[pid] if r is not None else 0.0 for r in reward]
                        opponents[pid].post_step(opp_rewards, done)

                # count finished episodes
                finished_episodes = sum(1 for d in done if d)
                episodes_done += finished_episodes
                pbar.update(finished_episodes)

                # Checkpoint every X episodes
                # (only triggers when we actually cross a multiple of X)
                if episodes_done // EVALUATE_EVERY_X_EPISODES != (episodes_done - finished_episodes) // EVALUATE_EVERY_X_EPISODES:
                    ckpt_path = f"quantum_cat_agent_{episodes_done}.pth"
                    torch.save(agent.state_dict(), ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")
                    
                    # Evaluate vs random opponents
                    avg_rew, paradox_rate, correct_pred_rate = evaluate_checkpoint(
                        agent, 
                        num_episodes=600, 
                        num_players=num_players,
                        writer=writer,
                        global_step=episodes_done
                    )
                    print(f"[Checkpoint Eval] episodes_done={episodes_done}, "
                          f"avg reward vs random={avg_rew:.2f}, "
                          f"paradox_rate={paradox_rate:.1%}, "
                          f"correct_pred_rate={correct_pred_rate:.1%}")

                if episodes_done >= num_episodes:
                    break

                time_step = next_time_step

            # Once we have a full batch, do learning
            agent_timesteps = [ts for ts in time_step]
            agent.learn(agent_timesteps)
            for opp in opponents.values():
                if isinstance(opp, PPO):
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


def evaluate_checkpoint(agent, num_episodes=600, num_players=3, writer=None, global_step=0):
    """Evaluates current agent vs random opponents.
    
    Args:
        agent: The PPO agent to evaluate
        num_episodes: Number of episodes to evaluate
        num_players: Number of players in the game
        writer: Optional TensorBoard writer
        global_step: Current training step for TensorBoard logging
    """
    game = pyspiel.load_game("python_quantum_cat", {"players": num_players})
    num_eval_envs = 8
    envs = SyncVectorEnv([
        rl_environment.Environment(game=game)
        for _ in range(num_eval_envs)
    ])
    
    # Configure evaluation settings via flags
    from absl import flags
    FLAGS.opponent_type = "random"
    FLAGS.player0_type = "ppo"
    
    avg_rew, paradox_rate, correct_pred_rate = evaluate(
        agent, envs, game, player_id=0, num_episodes=num_episodes
    )
    
    if writer is not None:
        writer.add_scalar("evaluation/avg_reward", avg_rew, global_step)
        writer.add_scalar("evaluation/paradox_rate", paradox_rate, global_step)
        writer.add_scalar("evaluation/correct_pred_rate", correct_pred_rate, global_step)
    
    return avg_rew, paradox_rate, correct_pred_rate

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

if __name__ == "__main__":
    app.run(main)
