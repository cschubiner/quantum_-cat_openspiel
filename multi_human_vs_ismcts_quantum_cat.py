#!/usr/bin/env python3
"""
multi_human_vs_ismcts.py

Allows multiple humans and multiple ISMCTS bots to play Quantum Cat simultaneously.
Usage example:
  python multi_human_vs_ismcts.py --num_humans=2 --num_bots=2
"""

import os
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
flags.DEFINE_integer("num_humans", 2, "Number of human players.")
flags.DEFINE_integer("num_bots", 1, "Number of ISMCTS bot players.")
flags.DEFINE_integer("max_simulations", 8000, "Max simulations for each ISMCTS bot.")
flags.DEFINE_integer("random_seed", 1234, "Random seed for reproducibility.")
flags.DEFINE_bool("clear_screen", True, "If True, prints many blank lines between turns.")

def clear_or_separate():
    """Optionally print many blank lines to hide the previous player's info."""
    if FLAGS.clear_screen:
        print("\n" * 50)
    else:
        print("\n---------------------------------\n")

def main(_):
    num_humans = FLAGS.num_humans
    num_bots = FLAGS.num_bots
    total_players = num_humans + num_bots
    if total_players < 2:
        raise ValueError("Need at least 2 total players (some combination of humans+bots).")
    if total_players > 5:
        raise ValueError("Quantum Cat only supports up to 5 total players.")

    random_state = np.random.RandomState(FLAGS.random_seed)
    game = pyspiel.load_game("python_quantum_cat", {"players": total_players})
    observer = game.make_py_observer(
        pyspiel.IIGObservationType(perfect_recall=False)
    )

    # Decide seat assignment: which seats are humans vs. bots
    # We'll do something simple: first num_humans seats are humans, next are bots.
    # If you'd rather randomize seat order, you can shuffle a list and assign.
    seat_types = ["human"] * num_humans + ["bot"] * num_bots
    # e.g. for num_humans=2, num_bots=2 => seat_types=["human","human","bot","bot"]

    # Build the ISMCTS bots
    # We'll store them in a dict: seat -> ISMCTSBot object
    # for each seat that is "bot".
    bots = {}
    for seat in range(total_players):
        if seat_types[seat] == "bot":
            evaluator = RandomRolloutEvaluator(
                n_rollouts=20, random_state=np.random.RandomState(FLAGS.random_seed + seat)
            )
            bot = ISMCTSBot(
                game=game,
                evaluator=evaluator,
                uct_c=2.0,
                max_simulations=FLAGS.max_simulations,
                max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
                random_state=np.random.RandomState(FLAGS.random_seed + seat),
                final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
                use_observation_string=False,
                allow_inconsistent_action_sets=False,
                child_selection_policy=ChildSelectionPolicy.PUCT
            )
            bots[seat] = bot

    # Create the initial state
    state = game.new_initial_state()

    while not state.is_terminal():
        current_player = state.current_player()

        if current_player == pyspiel.PlayerId.CHANCE:
            # Chance node -> pick an outcome randomly
            outcomes = state.chance_outcomes()
            print('outcomes:', outcomes)
            # Split outcomes into actions and probabilities
            actions, probabilities = zip(*outcomes)
            action = random_state.choice(actions, p=probabilities)
            state.apply_action(action)
            continue

        # Show partial info only if current seat is human
        if seat_types[current_player] == "human":
            observer_str = observer.string_from(state, current_player)
            print(f"--- Player {current_player}'s Turn ---")
            print("Observation:")
            print(observer_str)
            print(f"Tricks won so far (all players): {state._tricks_won}")
            print()
            # Human turn
            legal_moves = state.legal_actions(current_player)
            # Display action menu
            print("Legal actions:")
            for idx, act in enumerate(legal_moves):
                print(f"  {idx}: {state.action_to_string(current_player, act)}")

            # Prompt user
            choice = None
            while choice is None:
                user_input = input("Choose action index: ")
                try:
                    val = int(user_input)
                    if 0 <= val < len(legal_moves):
                        choice = legal_moves[val]
                    else:
                        print("Invalid index, try again.")
                except ValueError:
                    print("Invalid input, must be an integer index.")

            chosen_action = choice
            print(f"You chose: {state.action_to_string(current_player, chosen_action)}")
            state.apply_action(chosen_action)

            print(f"Next turn: Player {state.current_player()}")

            input("(Press ENTER to continue...)")
            clear_or_separate()
            input("(Press ENTER to hand over to the next player...)")
            clear_or_separate()

        else:
            # Bot turn
            ismcts_bot = bots[current_player]
            chosen_action = ismcts_bot.step(state)
            action_str = state.action_to_string(current_player, chosen_action)
            if action_str.startswith("Discard: rank="):
                # Hide the rank: just say "Discard a rank"
                action_str = "Discard a rank"
            print(f"ISMCTS Bot {current_player} chooses: {action_str}")
            state.apply_action(chosen_action)

            print(f"Next turn: Player {state.current_player()}")

            input("(Bot finished. Press ENTER to continue...)")
            clear_or_separate()

    # Terminal
    print("=========================================")
    print("Game finished! Final returns for each seat:")
    for seat in range(total_players):
        score = state.returns()[seat]
        seat_role = "Human" if seat_types[seat] == "human" else "ISMCTS Bot"
        print(f"Player {seat} ({seat_role}): {score}")
    print("=========================================")

if __name__ == "__main__":
    app.run(main)
