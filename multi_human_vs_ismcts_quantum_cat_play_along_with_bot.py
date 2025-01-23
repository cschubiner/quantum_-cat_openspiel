#!/usr/bin/env python3
"""
multi_human_vs_ismcts_quantum_cat_play_along_with_bot.py

Allows humans to specify *only their own* initial hands while the script
deduces the bot's hand by *eliminating* whatever the humans did not claim.

For example, if there are 4 players (3 humans + 1 bot), each player has
10 cards (for ranks 1..8, each rank has 5 copies => 40 total).

    - The 3 humans type in their distributions, e.g. "21000000" means
      2 copies of rank=1, 1 copy of rank=2, etc.
    - The script sums up the humans' usage for each rank.
    - It then computes leftover usage for the bot seat by subtracting from
      the total available (5 copies per rank).
    - The leftover distribution is assigned to the bot.

We then skip the dealing phase, jump to the discard phase in quantum_cat,
and play a single round (discard, predict, trick-taking, scoring).

This is handy if in real life you:
  - Deal physical cards to each human player,
  - Then physically set aside the bot's card set (or keep it hidden, of course!),
  - The script only needs the humans' holdings to deduce the bot's holdings.
"""

import random
import numpy as np
from absl import app
from absl import flags

import pyspiel

# Force-load our Python version of quantum_cat so the game is registered:
from open_spiel.python.games import quantum_cat  # noqa

# ISMCTS imports:
from open_spiel.python.algorithms.ismcts import (
    ISMCTSBot,
    ChildSelectionPolicy,
    ISMCTSFinalPolicyType,
    UNLIMITED_NUM_WORLD_SAMPLES
)
# Evaluator for the Bot:
from mcts_quantum_cat import TrickFollowingEvaluator

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_humans", 2, "Number of human players.")
flags.DEFINE_integer("num_bots", 1, "Number of ISMCTS bot players.")
flags.DEFINE_integer("random_seed", 42, "Random seed for the bot RNG.")
flags.DEFINE_integer("max_simulations", 8000, "Max simulations for each ISMCTS bot.")
flags.DEFINE_bool("clear_screen", True, "If True, prints many blank lines between turns.")


def clear_or_separate():
    """Optionally print many blank lines to hide the previous player's info."""
    if FLAGS.clear_screen:
        print("\n" * 50)
    else:
        print("\n------------------------\n")


def parse_hand_distribution(input_str, expected_len):
    """
    Parses a string (e.g. '2103') into a list of integers.
    - Checks length == expected_len
    - Each char must be a digit in 0..9 (realistically 0..5 for Cat in the Box).
    Returns a list of length expected_len.
    """
    if len(input_str) != expected_len:
        raise ValueError(f"Expected exactly {expected_len} digits, got {len(input_str)}")
    return [int(ch) for ch in input_str]


def main(_):
    num_humans = FLAGS.num_humans
    num_bots = FLAGS.num_bots
    total_players = num_humans + num_bots

    if not (3 <= total_players <= 5):
        raise ValueError("Total players must be between 3 and 5 for Cat in the Box.")

    # Create the game with the desired number of players
    game = pyspiel.load_game("python_quantum_cat", {"players": total_players})
    max_rank = game.max_card_value  # e.g. 9 if 5p, 8 if 4p, 7 if 3p
    cards_per_player = game.cards_per_player_initial  # e.g. 10 for 4p

    # We'll seat the humans first, then the bots, or reorder as you like:
    seat_types = ["human"] * num_humans + ["bot"] * num_bots

    print("\n--- Seats (turn order) ---")
    for seat_idx, stype in enumerate(seat_types):
        print(f"  Seat {seat_idx}: {stype}")
    print()

    # We store each seat’s distribution as a list of length=max_rank
    # For seat_type=="human", we read from user input
    # For seat_type=="bot", we’ll fill it by leftover elimination
    all_hands = [None] * total_players

    # We know that for each rank r in [0..max_rank-1], there are 5 total copies.
    # So leftover_for_rank[r] = 5 * (# valid ranks) - sum of that rank across humans
    # Actually, for each rank (1..max_rank), we have exactly 5 copies. There's no
    # partial deck in Cat in the Box, so total = 5 copies/rank * max_rank ranks = 5 * max_rank cards,
    # which should match (cards_per_player * total_players) if the game is consistent.
    # We'll track how many copies are "claimed" so far.
    claimed_so_far = [0] * max_rank

    # 1) Gather input for human seats
    for seat_idx in range(total_players):
        stype = seat_types[seat_idx]
        if stype == "human":
            print(f"\n--- Seat {seat_idx} (Human) ---")
            print(f"For ranks 1..{max_rank}, each rank has up to 5 copies.")
            print(f"You must enter a string of length {max_rank}, with digits in [0..5].")
            print(f"The sum of those digits must be exactly {cards_per_player}.")
            while True:
                try:
                    dist_str = input(f"Enter your {max_rank}-digit distribution: ")
                    dist = parse_hand_distribution(dist_str, max_rank)
                    if sum(dist) != cards_per_player:
                        print(
                            f"  ERROR: sum of digits is {sum(dist)}, but should be "
                            f"{cards_per_player}. Try again.\n"
                        )
                        continue
                    # Check that you haven't *already* claimed too many for any rank
                    # i.e. for rank r, claimed_so_far[r] + dist[r] <= 5
                    for r in range(max_rank):
                        if claimed_so_far[r] + dist[r] > 5:
                            raise ValueError(
                                f"Distribution claims {dist[r]} copies of rank {r+1}, but we "
                                f"already have {claimed_so_far[r]} claimed => total {claimed_so_far[r]+dist[r]} > 5."
                            )
                    # If all good, accept
                    all_hands[seat_idx] = dist
                    # Update claimed_so_far
                    for r in range(max_rank):
                        claimed_so_far[r] += dist[r]
                    break
                except ValueError as e:
                    print(f"  Invalid input: {e}\n")
        else:
            # We'll fill in later (bot seat)
            pass

    # 2) Deduce leftover distribution for each bot seat
    for seat_idx in range(total_players):
        stype = seat_types[seat_idx]
        if stype == "bot":
            # This seat’s distribution is leftover_for_rank
            # But we must ensure that sum leftover = cards_per_player
            # Because each seat has exactly `cards_per_player` cards
            leftover = [0] * max_rank
            for r in range(max_rank):
                # total available is 5, so leftover is 5 - claimed_so_far[r]
                # but if multiple bot seats exist, we must do them in sequence,
                # each time updating claimed_so_far.
                possible = 5 - claimed_so_far[r]
                if possible < 0:
                    raise ValueError(
                        f"Somehow rank {r+1} is overclaimed: claimed_so_far={claimed_so_far[r]} >5"
                    )
                leftover[r] = possible

            # Check the sum
            if sum(leftover) != cards_per_player:
                # If you have multiple bot seats, the distribution across them might be ambiguous:
                # For example, if leftover=15 for rank1, 5 for rank2, we can't just jam them into
                # a single seat if that seat only gets 10 cards total. The code below is a simple approach
                # for *one* bot seat. For >1 bot seats, you'd distribute the leftover among them
                # (a more advanced approach).
                # For typical usage (1 bot seat), leftover must sum to exactly cards_per_player
                # to be consistent.
                raise ValueError(
                    f"Leftover sum for seat {seat_idx} is {sum(leftover)}, but we need {cards_per_player}."
                    " This means the humans' input is inconsistent or we have multiple bot seats."
                )

            all_hands[seat_idx] = leftover
            # Mark them as claimed in claimed_so_far
            for r in range(max_rank):
                claimed_so_far[r] += leftover[r]

    # 3) Final check: no rank is over 5
    for r in range(max_rank):
        if claimed_so_far[r] != 5:
            # It's possible you never used up all copies of a rank
            # if you physically removed cards. But normally Cat in the Box uses the entire deck
            # for these 3–5 players. So we check if we have exactly 5 used for each rank:
            raise ValueError(
                f"Rank {r+1} usage is {claimed_so_far[r]} but should be exactly 5. "
                "Check your input or the standard rules!"
            )

    # We have a seat->distribution array in all_hands now. Let's see sums:
    for seat_idx in range(total_players):
        ssum = sum(all_hands[seat_idx])
        print(f"Seat {seat_idx} final distribution => sum={ssum}, {all_hands[seat_idx]}")
    print()

    # 4) Create the new initial state but skip dealing => jump to discard phase.
    state = game.new_initial_state()

    # The quantum_cat code sets the following:
    #   _phase=0 => dealing
    #   _phase=1 => discard
    # We forcibly override:
    if hasattr(state, "_phase"):
        state._phase = 1
        state._cards_dealt = total_players * cards_per_player
        state._current_player = 0
        # Overwrite the hand arrays:
        for p in range(total_players):
            for rank_idx in range(max_rank):
                state._hands[p][rank_idx] = all_hands[p][rank_idx]
    else:
        raise RuntimeError("Could not override quantum_cat state internals!")

    print("--- All distributions assigned. Entering discard phase now. ---")
    print("")

    clear_or_separate()
    input(f"(Press ENTER to hand over to the next player {state.current_player()}...)")

    # Build an observer to show partial info to humans
    iig_obs_type = pyspiel.IIGObservationType(perfect_recall=False)
    observer = game.make_py_observer(iig_obs_type)

    # Create ISMCTS bots for bot seats
    rng = np.random.RandomState(FLAGS.random_seed)
    bots = {}
    for seat_idx in range(total_players):
        if seat_types[seat_idx] == "bot":
            evaluator = TrickFollowingEvaluator(
                n_rollouts=2,
                random_state=np.random.RandomState(FLAGS.random_seed + seat_idx),
            )
            bot = ISMCTSBot(
                game=game,
                evaluator=evaluator,
                uct_c=2.0,
                max_simulations=FLAGS.max_simulations,
                max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
                random_state=np.random.RandomState(FLAGS.random_seed + seat_idx),
                final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
                use_observation_string=False,
                allow_inconsistent_action_sets=False,
                child_selection_policy=ChildSelectionPolicy.PUCT,
            )
            bots[seat_idx] = bot

    # --- Main loop ---
    while not state.is_terminal():
        current_player = state.current_player()
        if current_player == pyspiel.PlayerId.CHANCE:
            # Should not really happen now that we've forced phase=1, but just in case:
            outcomes = state.chance_outcomes()
            if outcomes:
                acts, probs = zip(*outcomes)
                choice = rng.choice(acts, p=probs)
                state.apply_action(choice)
            continue

        seat_type = seat_types[current_player]
        if seat_type == "human":
            # Show partial info
            obs_str = observer.string_from(state, current_player)
            print(f"--- Player {current_player} (Human) Turn ---")
            print("Observation:")
            print(obs_str)
            print(f"Tricks won (all players): {[int(x) for x in state._tricks_won]}")
            print()

            legal_moves = state.legal_actions(current_player)
            print("Legal actions:")
            for idx, act in enumerate(legal_moves):
                print(f"  {idx}: {state.action_to_string(current_player, act)}")

            choice = None
            while choice is None:
                val_str = input("Choose action index: ")
                try:
                    val = int(val_str)
                    if 0 <= val < len(legal_moves):
                        choice = legal_moves[val]
                    else:
                        print("  Invalid index, try again.")
                except ValueError:
                    print("  Invalid input, must be integer.")

            chosen_action = choice
            print(f"You chose: {state.action_to_string(current_player, chosen_action)}")
            state.apply_action(chosen_action)

            if not state.is_terminal():
                print(f"Next turn: Player {state.current_player()}")
                input("(Press ENTER to continue...)")
                clear_or_separate()
                input(f"(Press ENTER to hand over to the next player {state.current_player()}...)")
                clear_or_separate()

        else:
            # Bot turn
            print(f"--- Player {current_player} (Bot) Turn ---")
            obs_str = observer.string_from(state, current_player)
            print("Bot sees partial info (for debugging/human reference):")
            print(obs_str)
            clear_or_separate()
            print("\n(MCTS thinking...)\n")
            chosen_action = bots[current_player].step(state)
            print(f"Bot chooses: {state.action_to_string(current_player, chosen_action)}")
            state.apply_action(chosen_action)

            if not state.is_terminal():
                print(f"Next turn: Player {state.current_player()}")
                input("(Press ENTER to continue...)")
                clear_or_separate()
                input(f"(Press ENTER to hand over to the next player {state.current_player()}...)")
                clear_or_separate()

    # Terminal
    print("=========================================")
    print("Game finished! Final returns:")
    final_scores = state.returns()
    for seat_idx in range(total_players):
        stype = seat_types[seat_idx]
        print(f"  Player {seat_idx} ({stype}): {final_scores[seat_idx]}")
    print("=========================================")


if __name__ == "__main__":
    app.run(main)
