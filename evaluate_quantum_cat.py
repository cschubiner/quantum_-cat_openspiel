#!/usr/bin/env python3
"""
Script to evaluate different bot types on Quantum Cat (ISMCTS, PPO, Random, or follow_suit).

Usage:
  python evaluate_quantum_cat.py --num_players=3 --num_episodes=20 \
    --agent_path=quantum_cat_agent.pth \
    --player0_type=ismcts \
    --opponent_type=random
"""

import random

import numpy as np
import torch
import pyspiel

from absl import app
from absl import flags
from pyspiel import RandomRolloutEvaluator

# Include your Quantum Cat Python registration.
from open_spiel.python import rl_environment
from open_spiel.python.rl_agent import StepOutput
from open_spiel.python.vector_env import SyncVectorEnv

# PPO
from open_spiel.python.pytorch.ppo import PPO, PPOAgent

# ISMCTS imports
from open_spiel.python.algorithms.ismcts import (
    ISMCTSBot,
    ChildSelectionPolicy,
    ISMCTSFinalPolicyType,
    UNLIMITED_NUM_WORLD_SAMPLES
)

# Make sure quantum_cat is registered
from open_spiel.python.games import quantum_cat

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "player0_type",
    # "ppo",
    "ismcts",
    ["ppo", "random", "follow_suit", "ismcts"],
    "Policy for player_id=0."
)

flags.DEFINE_enum(
    "opponent_type",
    # "random",
    "follow_suit",
    ["random", "follow_suit", "ismcts"],
    "Policy for players != 0."
)

flags.DEFINE_integer("num_players", 3, "Number of players.")
flags.DEFINE_integer("num_episodes", 600, "Number of episodes to evaluate.")
flags.DEFINE_string("agent_path", "quantum_cat_agent_615001.pth", "Path to saved agent.")
flags.DEFINE_bool("self_play", False, "If True, use same agent for all players.")
flags.DEFINE_bool("random_vs_random", False, "If True, evaluate random vs random play.")


def pick_follow_suit_action(legal_actions, info_state, num_card_types):
    """Tries to find an action that follows the led color, if any."""
    # led color is a 5-dimensional one-hot: R,B,Y,G,None
    num_players = FLAGS.num_players
    led_color_start = num_players + 5  # skip current_player (num_players) + phase(5)
    led_color = info_state[led_color_start: led_color_start + 5]
    led_color_idx = int(np.argmax(led_color))  # 0..3 => R,B,Y,G, 4 => None

    if led_color_idx == 4 or 999 in legal_actions:  # No led color or PARADOX
        return random.choice(legal_actions)

    # Try to follow suit
    follow_suit = []
    for action in legal_actions:
        if action == 999:  # PARADOX action
            continue
        color_idx = action // num_card_types
        if color_idx == led_color_idx:
            follow_suit.append(action)
    if follow_suit:
        return random.choice(follow_suit)
    return random.choice(legal_actions)


def make_ismcts_bot(game, seed=1234):
    """Creates an IS-MCTS bot with default settings."""
    evaluator = RandomRolloutEvaluator(n_rollouts=2, seed=seed)
    bot = ISMCTSBot(
        game=game,
        evaluator=evaluator,
        uct_c=2.0,
        max_simulations=1000,
        max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
        random_state=np.random.RandomState(seed),
        final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
        use_observation_string=False,
        allow_inconsistent_action_sets=False,
        child_selection_policy=ChildSelectionPolicy.PUCT
    )
    return bot


def evaluate(agent, envs, game, player_id=0, num_episodes=20):
    """Evaluates an agent for a specified number of episodes using synchronous evaluation."""
    # Track completed episodes and total reward
    episodes_done = 0
    total_reward = 0

    # Track paradox and prediction accuracy
    paradox_count = 0
    correct_pred_count = 0

    # Track which envs are finished
    done_mask = [False] * len(envs.envs)

    # Initialize all environments together
    time_step = envs.reset()

    while episodes_done < num_episodes:
        actions = []
        for i, ts in enumerate(time_step):
            # If this environment is already done, just pick a dummy action
            if done_mask[i] or ts.last():
                actions.append(0)
                continue

            p = ts.current_player()
            legal = ts.observations["legal_actions"][p]

            if hasattr(agent, "step"):
                # If it's an RL-based agent or IS-MCTS (Bot API), try step()
                # Some bots implement step(), others do step_with_policy()...
                # We'll try step() if it exists.
                chosen_action = agent.step(ts)  # For IS-MCTS or uniform random
                if chosen_action is None:
                    chosen_action = 0
            else:
                # If the 'agent' is just a python function or a dict, handle special logic:
                # (We handle "random" or "follow_suit" via direct function calls)
                chosen_action = 0
                # We'll rely on a specialized function for random or follow_suit
                # if we had coded it that way.

            actions.append(chosen_action)

        # Step all environments together
        step_out = [StepOutput(action=a, probs=None) for a in actions]
        next_time_step, reward, done, _ = envs.step(step_out)

        # Mark finished envs, accumulate reward, count episodes
        for i, dval in enumerate(done):
            if dval and not done_mask[i]:
                if reward[i] is not None:
                    total_reward += reward[i][player_id]
                episodes_done += 1
                done_mask[i] = True

                # Track paradox and correct predictions
                final_state = envs.envs[i]._state
                if final_state._has_paradoxed[player_id]:
                    paradox_count += 1
                else:
                    if final_state._tricks_won[player_id] == final_state._predictions[player_id]:
                        correct_pred_count += 1

        # Synchronous approach: if all envs are done OR we've hit the episode target
        if all(done_mask) or episodes_done >= num_episodes:
            if episodes_done < num_episodes:
                time_step = envs.reset()
                done_mask = [False] * len(envs.envs)
            else:
                break
        else:
            time_step = next_time_step

    avg_rew = total_reward / num_episodes
    paradox_rate = paradox_count / num_episodes
    correct_pred_rate = correct_pred_count / num_episodes

    print(f"[Sync Eval] player_id={player_id}, episodes={num_episodes}, "
          f"avg reward={avg_rew:.2f}, episodes_done={episodes_done}, "
          f"paradox_rate={paradox_rate:.1%}, correct_pred_rate={correct_pred_rate:.1%}")
    return avg_rew, paradox_rate, correct_pred_rate


def make_policy(policy_type, game, player_id=0):
    """Helper to create a policy (bot or function) given a string."""
    if policy_type == "random":
        # Make a random bot that uses open_spiel's built-in uniform random
        return pyspiel.make_uniform_random_bot(player_id, random.Random(123 + player_id))
    elif policy_type == "follow_suit":
        # We'll create a simple python function that picks actions
        # We'll wrap it in a Bot adapter (requires some extra steps).
        # Alternatively, we can define a small custom bot class in-line.
        # For brevity, we'll do a custom bot:
        class FollowSuitBot(pyspiel.Bot):
            def __init__(self, player_id, game):
                super().__init__()
                self._player_id = player_id
                self._game = game
                self._num_card_types = game.num_distinct_actions() // 4  # rough guess

            def player_id(self):
                return self._player_id

            def step(self, state):
                # If not my turn, do nothing
                if state.current_player() != self._player_id:
                    return None
                legal_actions = state.legal_actions(self._player_id)
                # For observation, we'll just get the public obs, or info_state
                # to figure out led color. The quantum_cat obs might be big,
                # but let's do a quick approach:
                obs = state.observation_tensor(self._player_id)
                return pick_follow_suit_action(legal_actions, obs, self._num_card_types)

            def restart_at(self, state):
                pass

        return FollowSuitBot(player_id, game)

    elif policy_type == "ismcts":
        # Return an ISMCTS bot
        return make_ismcts_bot(game, seed=123 + player_id)

    elif policy_type == "ppo":
        # Build PPO. We'll assume player0 is the PPO agent.
        # The code is adapted from the original script
        sample_state = game.new_initial_state()
        obs = sample_state.observation_tensor(0)
        info_state_shape = (len(obs),)
        num_actions = game.num_distinct_actions()

        agent = PPO(
            input_shape=info_state_shape,
            num_actions=num_actions,
            num_players=FLAGS.num_players,
            player_id=player_id,
            num_envs=1,
            steps_per_batch=16,
            update_epochs=4,
            learning_rate=2.5e-4,
            gae=True,
            gamma=0.99,
            gae_lambda=0.95,
            device="cpu",
            agent_fn=PPOAgent,
        )
        agent.load_state_dict(torch.load(FLAGS.agent_path, map_location="cpu"))
        agent.eval()

        class PPOBotAdapter(pyspiel.Bot):
            def __init__(self, agent):
                super().__init__()
                self._agent = agent

            def step(self, state):
                if state.is_terminal():
                    return None
                if state.current_player() != self._agent.player_id:
                    return None
                # RL Env step
                legal_actions = state.legal_actions(self._agent.player_id)
                obs = [rl_environment.TimeStep(
                    observations={"info_state": [state.observation_tensor(self._agent.player_id)],
                                  "legal_actions": [legal_actions]},
                    rewards=None, discounts=None, step_type=None)]
                step_output = self._agent.step(obs, is_evaluation=True)
                return step_output[0].action

            def restart_at(self, state):
                pass

        return PPOBotAdapter(agent)

    else:
        raise ValueError(f"Unsupported policy_type={policy_type}")


def main(_):
    # Quick overrides:
    if FLAGS.random_vs_random:
        FLAGS.player0_type = "random"
        FLAGS.opponent_type = "random"

    game = pyspiel.load_game("python_quantum_cat", {"players": FLAGS.num_players})

    num_envs = 8
    envs = SyncVectorEnv([
        rl_environment.Environment(game=game) for _ in range(num_envs)
    ])

    # Build player0 policy
    bot0 = make_policy(FLAGS.player0_type, game, player_id=0)

    # Build other players
    bots = [None] * FLAGS.num_players
    bots[0] = bot0
    for p in range(1, FLAGS.num_players):
        if FLAGS.self_play:
            # Same agent for all
            bots[p] = make_policy(FLAGS.player0_type, game, player_id=p)
        else:
            bots[p] = make_policy(FLAGS.opponent_type, game, player_id=p)

    # Evaluate
    # We'll just evaluate from perspective of player0
    print(f"Evaluating with player0={FLAGS.player0_type}, "
          f"opponents={FLAGS.opponent_type}...")

    # We do synchronous stepping ourselves in `evaluate()`, passing in a single 'agent' if there's only 1 relevant agent.
    # But actually we have multiple bots. We'll handle it by wrapping them in a single "multi-bot environment" approach.
    # For simplicity, let's just evaluate from p=0's perspective:
    # We'll define a small wrapper that calls the right bot's step():
    class MultiBot:
        def __init__(self, bots):
            self._bots = bots

        def step(self, state):
            cp = state.current_player()
            return self._bots[cp].step(state)

    multi_bot = MultiBot(bots)

    # Evaluate for player0
    avg_rew, paradox_rate, correct_pred_rate = evaluate(
        multi_bot, envs, game, player_id=0, num_episodes=FLAGS.num_episodes
    )


if __name__ == "__main__":
    app.run(main)
