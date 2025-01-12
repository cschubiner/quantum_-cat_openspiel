#!/usr/bin/env python3
"""
Play Quantum Cat vs. an ISMCTS agent in a text-based loop.

Usage:
  python human_vs_quantum_cat.py --num_players=3
"""

import random
import pyspiel
import numpy as np

from absl import app
from absl import flags

from open_spiel.python.algorithms.ismcts import (
    ISMCTSBot,
    ChildSelectionPolicy,
    ISMCTSFinalPolicyType,
    UNLIMITED_NUM_WORLD_SAMPLES
)
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator
from open_spiel.python.games import quantum_cat

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_players", 3, "Number of players in the game.")
flags.DEFINE_integer("human_player", 2, "Which seat the human controls (0..N-1).")

def main(_):
    num_players = FLAGS.num_players
    human_player = FLAGS.human_player

    game = pyspiel.load_game("python_quantum_cat", {"players": num_players})
    state = game.new_initial_state()
    observer = game.make_py_observer(
        pyspiel.IIGObservationType(perfect_recall=False)
    )

    # Create ISMCTS agent for player 0
    evaluator = RandomRolloutEvaluator(n_rollouts=2, random_state=np.random.RandomState(1234))
    ismcts_bot = ISMCTSBot(
        game=game,
        evaluator=evaluator,
        uct_c=2.0,
        max_simulations=1000,
        max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
        random_state=np.random.RandomState(123),
        final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
        use_observation_string=False,
        allow_inconsistent_action_sets=False,
        child_selection_policy=ChildSelectionPolicy.PUCT
    )
    agents = {0: ismcts_bot}  # Only ISMCTS agent needs to be stored

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
                # ISMCTS Agent's turn
                bot = agents[cur_player]
                chosen_action = bot.step(state)
                print(f"ISMCTS Agent picks: {state.action_to_string(cur_player, chosen_action)}")
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


if __name__ == "__main__":
    app.run(main)
