#!/usr/bin/env python3
"""
Play Quantum Cat vs. a PPO agent in a text-based loop.

Usage:
  python human_vs_quantum_cat.py --num_players=3 --agent_path=quantum_cat_agent.pth
"""

import random
import torch
import pyspiel

from absl import app
from absl import flags

from open_spiel.python.pytorch.ppo import PPO, PPOAgent
from open_spiel.python.rl_environment import TimeStep
from open_spiel.python.games import quantum_cat

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_players", 3, "Number of players in the game.")
flags.DEFINE_string("agent_path", "quantum_cat_agent_3115002.pth", "Path to saved PPO agent.")
flags.DEFINE_integer("human_player", 2, "Which seat the human controls (0..N-1).")

def main(_):
    num_players = FLAGS.num_players
    human_player = FLAGS.human_player

    game = pyspiel.load_game("python_quantum_cat", {"players": num_players})
    state = game.new_initial_state()
    observer = game.make_py_observer(
        pyspiel.IIGObservationType(perfect_recall=False)
    )

    # Create PPO agent for player 0 and random bot for player 1
    agents = {}
    
    # PPO agent always plays as player 0 (as it was trained)
    ppo_agent = make_agent(game, 0)  # player_id=0 to match training
    ppo_agent.load_state_dict(torch.load(FLAGS.agent_path, map_location="cpu"))
    ppo_agent.eval()
    agents[0] = ppo_agent
    
    # Player 1 will be a random bot (handled in the main loop)

    # Step through the game
    while not state.is_terminal():
        cur_player = state.current_player()
        print("\n---------------------------------")
        # Get observation string for current player
        obs_str = observer.string_from(state, cur_player)
        print(f"Observation for player {cur_player}:")
        print(f"  {obs_str}")
        print(f"Tricks won: {state._tricks_won}")  # Show tricks for all players
        if cur_player == pyspiel.PlayerId.CHANCE:
            # If chance node, apply uniform random outcome
            # Usually the state will do that automatically
            # but let's see if we must pass apply_action
            outcomes = state.chance_outcomes()
            # pick one randomly
            action, prob = random.choice(outcomes)
            print(f"Applying chance action: {state.action_to_string(cur_player, action)}")
            state.apply_action(action)
            continue

        if cur_player == human_player:
            print(f"--- Your turn! You are player {human_player} ---")
            legal = state.legal_actions(cur_player)
            print("Legal actions:")
            for idx, act in enumerate(legal):
                print(f"{idx}: {state.action_to_string(cur_player, act)}")

            chosen_idx = None
            while chosen_idx is None:
                choice_str = input("Choose an action index: ")
                try:
                    ci = int(choice_str)
                    if 0 <= ci < len(legal):
                        chosen_idx = ci
                    else:
                        print("Invalid index, try again.")
                except ValueError:
                    print("Not an integer, try again.")

            chosen_action = legal[chosen_idx]
            print(f"You picked: {state.action_to_string(cur_player, chosen_action)}")
            state.apply_action(chosen_action)
        else:
            if cur_player == 0:
                # PPO Agent's turn
                agent = agents[cur_player]
                obs = state.observation_tensor(cur_player)
                fake_ts = TimeStep(
                    observations={
                        "info_state": {cur_player: obs},
                        "legal_actions": {cur_player: state.legal_actions(cur_player)}
                    },
                    rewards=None,
                    discounts=None,
                    step_type=None
                )
                action_out = agent.step([fake_ts], is_evaluation=True)
                chosen_action = action_out[0].action
                print(f"PPO Agent picks: {state.action_to_string(cur_player, chosen_action)}")
                state.apply_action(chosen_action)
            else:
                # Random bot's turn (player 1)
                legal_actions = state.legal_actions(cur_player)
                chosen_action = random.choice(legal_actions)
                print(f"Random bot picks: {state.action_to_string(cur_player, chosen_action)}")
                state.apply_action(chosen_action)

    # Terminal
    print("\n=====================================")
    print("Game finished!")
    returns = state.returns()
    for pid in range(num_players):
        print(f"Player {pid} final return: {returns[pid]}")

def make_agent(game, pid):
    """Builds a PPO agent (same hyperparams as training) for seat pid."""
    obs_size = len(game.new_initial_state().observation_tensor(pid))
    info_state_shape = (obs_size,)
    agent = PPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        player_id=pid,
        num_envs=1,  # not relevant here
        steps_per_batch=16,
        update_epochs=4,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        agent_fn=PPOAgent,
    )
    return agent

if __name__ == "__main__":
    app.run(main)
