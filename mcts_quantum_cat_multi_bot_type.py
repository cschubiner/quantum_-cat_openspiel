#!/usr/bin/env python3
"""
Play a one-round Quantum Cat game with an ISMCTS bot that uses a
TrickFollowingEvaluator, competing against X ISMCTS bots using RandomRolloutEvaluator,
and Y bots that pick entirely random actions.

After every 5 episodes, prints out the average return & 90% confidence interval
for each of the three bot categories:
    1) ISMCTS + TrickFollowing
    2) ISMCTS + RandomRollout
    3) UniformRandom
"""

import argparse
import numpy as np
from tqdm import tqdm
from scipy import stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

import pyspiel

from mcts_quantum_cat import TrickFollowingEvaluator, TrickFollowingEvaluatorV2
from open_spiel.python.algorithms.ismcts import (
    ISMCTSBot,
    ChildSelectionPolicy,
    ISMCTSFinalPolicyType,
    UNLIMITED_NUM_WORLD_SAMPLES,
)
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator
# The following import references the quantum_cat file you provided (or is registered).
from open_spiel.python.games import quantum_cat


###############################################################################
# Main script logic
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50000,
                        help="Number of episodes to run.")
    parser.add_argument("--x_random_rollout_bots", type=int, default=0,
                        help="Number of ISMCTS bots using RandomRolloutEvaluator.")
    parser.add_argument("--y_random_bots", type=int, default=0,
                        help="Number of purely random bots.")
    parser.add_argument("--players", type=int, default=4,
                        help="Total players = 1 (TrickFollowingISMCTS) + X + Y. "
                             "If this is set, we override X+Y to match (players-1).")
    parser.add_argument("--uct_c", type=float, default=2.0,
                        help="UCT exploration constant for all MCTS bots.")
    parser.add_argument("--max_sims", type=int, default=250,
                        help="Number of simulations for each ISMCTS bot per move.")
    parser.add_argument("--seed", type=int, default=999,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    # TrickFollowingEvaluator parameters
    tf_param_sets = [
        # Baseline/conservative parameters
        dict(
            discard_frequent_prob=0.85,
            discard_infrequent_prob=0.15,
            pred_main_prob=0.70,
            pred_neighbor_prob=0.20,
            pred_uniform_prob=0.10,
            follow_suit_prob=0.60,
            deviate_prob=0.40,
            deviate_trump_ratio=0.75,
            deviate_other_ratio=0.25,
        ),
        dict(
            discard_frequent_prob=0.85,
            discard_infrequent_prob=0.15,
            pred_main_prob=0.70,
            pred_neighbor_prob=0.20,
            pred_uniform_prob=0.10,
            follow_suit_prob=0.60,
            deviate_prob=0.40,
            deviate_trump_ratio=0.75,
            deviate_other_ratio=0.25,
        ),
        dict(
            discard_frequent_prob=0.85,
            discard_infrequent_prob=0.15,
            pred_main_prob=0.70,
            pred_neighbor_prob=0.20,
            pred_uniform_prob=0.10,
            follow_suit_prob=0.60,
            deviate_prob=0.40,
            deviate_trump_ratio=0.75,
            deviate_other_ratio=0.25,
        ),
        dict(
            discard_frequent_prob=0.85,
            discard_infrequent_prob=0.15,
            pred_main_prob=0.70,
            pred_neighbor_prob=0.20,
            pred_uniform_prob=0.10,
            follow_suit_prob=0.60,
            deviate_prob=0.40,
            deviate_trump_ratio=0.75,
            deviate_other_ratio=0.25,
        ),
        # Aggressive follow-suit
        # dict(
        #     discard_frequent_prob=0.85,
        #     discard_infrequent_prob=0.15,
        #     pred_main_prob=0.70,
        #     pred_neighbor_prob=0.20,
        #     pred_uniform_prob=0.10,
        #     follow_suit_prob=0.70,  # More likely to follow suit
        #     deviate_prob=0.30,      # Less likely to deviate
        #     deviate_trump_ratio=0.80, # More likely to play trump when deviating
        #     deviate_other_ratio=0.20,
        # ),
        # # Deviate-heavy
        # dict(
        #     discard_frequent_prob=0.85,
        #     discard_infrequent_prob=0.15,
        #     pred_main_prob=0.70,
        #     pred_neighbor_prob=0.20,
        #     pred_uniform_prob=0.10,
        #     follow_suit_prob=0.50,  # Less likely to follow suit
        #     deviate_prob=0.50,      # More likely to deviate
        #     deviate_trump_ratio=0.65, # Less likely to play trump when deviating
        #     deviate_other_ratio=0.35,
        # ),
        # # Balanced approach
        # dict(
        #     discard_frequent_prob=0.85,
        #     discard_infrequent_prob=0.15,
        #     pred_main_prob=0.70,
        #     pred_neighbor_prob=0.20,
        #     pred_uniform_prob=0.10,
        #     follow_suit_prob=0.55,  # Slightly less likely to follow suit
        #     deviate_prob=0.45,      # Slightly more likely to deviate
        #     deviate_trump_ratio=0.85, # Much more likely to play trump when deviating
        #     deviate_other_ratio=0.15,
        # ),
    ]

    # ISMCTS bot parameters
    ismcts_param_sets = [
        dict(
            uct_c=2.2,
            max_simulations=845,
            final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
            child_selection_policy=ChildSelectionPolicy.PUCT,
        ),
        dict(
            uct_c=3.2,
            max_simulations=845,
            final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
            child_selection_policy=ChildSelectionPolicy.PUCT,
        ),
        dict(
            uct_c=3.6,
            max_simulations=845,
            final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
            child_selection_policy=ChildSelectionPolicy.PUCT,
        ),
        dict(
            uct_c=2.8,
            max_simulations=845,
            final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
            child_selection_policy=ChildSelectionPolicy.PUCT,
        ),
    ]

    assert len(tf_param_sets) == len(ismcts_param_sets)

    # If user sets --players, we interpret that as 1 + X + Y = players
    # So X+Y = players-1. We'll guess that X=... user might want a specific ratio
    # For now let's just enforce X+Y=players-1.
    # If user also specified x_random_rollout_bots and y_random_bots,
    # we won't break them, but if there's a mismatch we override.
    if (len(tf_param_sets) + args.x_random_rollout_bots + args.y_random_bots) != args.players:
        raise ValueError(f"Mismatch in specified players vs. X+Y bots: {args.players}, "
                         f"{args.x_random_rollout_bots}, {args.y_random_bots}")

    np.random.seed(args.seed)

    # Register or load the game
    game = pyspiel.load_game(
        "python_quantum_cat",
        {"players": args.players}
    )

    # We'll label: 0 => "TrickFollowingISMCTS"
    # Then next X => "RandomRolloutISMCTS"
    # Then next Y => "UniformRandom"
    # We'll track returns in separate lists for each category.
    trickfollow_returns = []

    # Create TrickFollowing bots with different parameter sets
    tf_bots = []
    for i, params in enumerate(tf_param_sets):
        # Create evaluator with TrickFollowing parameters
        if i <= 1:
            tf_evaluator = TrickFollowingEvaluatorV2(
                n_rollouts=2,
                random_state=np.random.RandomState(args.seed + 123 + i),
                **tf_param_sets[i]
            )
        else:
            tf_evaluator = TrickFollowingEvaluator(
                n_rollouts=2,
                random_state=np.random.RandomState(args.seed + 123 + i),
                **tf_param_sets[i]
            )

        # Get corresponding ISMCTS parameters and create bot
        bot = ISMCTSBot(
            game=game,
            evaluator=tf_evaluator,
            max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
            random_state=np.random.RandomState(args.seed + 999 + i),
            use_observation_string=False,
            allow_inconsistent_action_sets=False,
            **ismcts_param_sets[i]
        )
        tf_bots.append(bot)

    # We'll track returns by parameter set
    tf_returns_by_params = [[] for _ in tf_param_sets]
    # Track statistics by parameter set
    tf_paradox_counts = [0] * len(tf_param_sets)
    tf_game_counts = [0] * len(tf_param_sets)
    tf_adj_sums = [0.0] * len(tf_param_sets)
    tf_trick_sums = [0.0] * len(tf_param_sets)

    # Statistics for random rollout bots
    rr_paradox_count = 0
    rr_game_count = 0
    rr_adj_sum = 0.0
    rr_trick_sum = 0.0

    # Statistics for uniform random bots
    ur_paradox_count = 0
    ur_game_count = 0
    ur_adj_sum = 0.0
    ur_trick_sum = 0.0

    randomrollout_returns = []
    pure_random_returns = []

    # Next X => "RandomRolloutISMCTS"
    rr_bots = []
    for i in range(args.x_random_rollout_bots):
        rr_eval = RandomRolloutEvaluator(
            n_rollouts=2,
            random_state=np.random.RandomState(args.seed + 555 + i)
        )
        new_bot = ISMCTSBot(
            game=game,
            evaluator=rr_eval,
            uct_c=args.uct_c,
            max_simulations=args.max_sims,
            max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
            random_state=np.random.RandomState(args.seed + 1000 + i),
            final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
            use_observation_string=False,
            allow_inconsistent_action_sets=False,
            child_selection_policy=ChildSelectionPolicy.PUCT
        )
        rr_bots.append(new_bot)

    # Next Y => "UniformRandom"
    ur_bots = []
    for i in range(args.y_random_bots):
        # Assign each uniform-random bot a valid player_id, in sequence.
        # First player is 0 (TrickFollowingISMCTS), next X players are the random-rollout
        # MCTS bots, so these Y uniform-random bots start at 1 + X.
        player_id = len(tf_bots) + len(rr_bots) + i
        bot = pyspiel.make_uniform_random_bot(player_id, args.seed + 2000 + i)
        ur_bots.append(bot)

    # Construct the final bot array in order:
    # [tf_bots..., rr_bots..., ur_bots...]
    # Also keep track of which category each player belongs to.
    all_bots = tf_bots + rr_bots + ur_bots
    bot_types = []
    # First N => different TrickFollow parameter sets
    for i in range(len(tf_bots)):
        bot_types.append(f"trickfollow_{i}")  # Numbered to track which param set
    # next X => randomrollout
    for _ in range(args.x_random_rollout_bots):
        bot_types.append("randomrollout")
    # next Y => uniform
    for _ in range(args.y_random_bots):
        bot_types.append("uniform")

    if not (3 <= len(bot_types) <= 5):
        raise ValueError(f"wrong num bot types: {bot_types}")


    if len(bot_types) != args.players:
        raise ValueError("Mismatch in constructed bot types vs players")


    # We'll run episodes
    # Track how many games had at least one paradox by any player
    any_paradox_count = 0

    for episode_idx in tqdm(range(args.num_episodes), desc="Running episodes"):
        state = game.new_initial_state()

        # Re-initialize bots
        for pid, bot in enumerate(all_bots):
            # pyspiel.Bot doesn't always require reset, but if it's an ISMCTSBot we do
            if hasattr(bot, "reset"):
                bot.reset()

        while not state.is_terminal():
            current_player = state.current_player()
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                chosen = np.random.choice(actions, p=probs)
                state.apply_action(chosen)
            else:
                action = all_bots[current_player].step(state)
                # fallback if a bot doesn't provide an action
                if action is None:
                    legal_acts = state.legal_actions(current_player)
                    if not legal_acts:
                        # No moves => paradox
                        # We'll just let the engine handle it if it does so automatically
                        pass
                    else:
                        action = np.random.choice(legal_acts)
                if action is not None:
                    state.apply_action(action)

        final_returns = state.returns()

        # Get final game statistics
        has_paradoxed = state._has_paradoxed
        # Track if ANY player paradoxed this game
        if any(has_paradoxed):
            any_paradox_count += 1

        adjacency_bonuses = state._player_adjacency_bonus
        tricks_won = state._tricks_won

        # Tally them by category
        for pid, r in enumerate(final_returns):
            # Get stats for this player
            paradoxed = 1 if has_paradoxed[pid] else 0
            adj_bonus = adjacency_bonuses[pid]
            tricks = tricks_won[pid]

            if bot_types[pid].startswith("trickfollow_"):
                param_idx = int(bot_types[pid].split("_")[1])
                tf_returns_by_params[param_idx].append(r)
                tf_paradox_counts[param_idx] += paradoxed
                tf_game_counts[param_idx] += 1
                tf_adj_sums[param_idx] += adj_bonus
                tf_trick_sums[param_idx] += tricks
            elif bot_types[pid] == "randomrollout":
                randomrollout_returns.append(r)
                rr_paradox_count += paradoxed
                rr_game_count += 1
                rr_adj_sum += adj_bonus
                rr_trick_sum += tricks
            else:
                pure_random_returns.append(r)
                ur_paradox_count += paradoxed
                ur_game_count += 1
                ur_adj_sum += adj_bonus
                ur_trick_sum += tricks

        # Every 5 episodes, print out stats
        if (episode_idx + 1) % 5 == 0:
            print("=" * 60)
            print(f"Episode {episode_idx+1} completed. Intermediate stats:")
            
            # Show overall paradox rate first
            any_paradox_rate = 100.0 * any_paradox_count / (episode_idx + 1)
            print(f"  ANY player paradox rate so far: {any_paradox_rate:.1f}%")

            def safe_div(x, y):
                return x / y if y > 0 else 0.0

            def print_stats_for(label, data):
                if len(data) == 0:
                    print(f"  {label}: No data yet.")
                    return
                mean_r = np.mean(data)
                std_r = np.std(data)
                conf = stats.t.interval(
                    0.90, df=len(data)-1, loc=mean_r, scale=stats.sem(data)
                )
                halfwidth = (conf[1] - conf[0]) / 2.0
                print(f"  {label}: N={len(data)} mean={mean_r:.3f}, "
                      f"std={std_r:.3f}, 90%CI=({mean_r:.3f} ± {halfwidth:.3f})")

            # Print stats for each TrickFollowing parameter set
            for i, returns in enumerate(tf_returns_by_params):
                print_stats_for(f"TrickFollowingISMCTS_{i}", returns)
                
                # Additional statistics
                paradox_rate = safe_div(tf_paradox_counts[i], tf_game_counts[i]) * 100
                avg_adj = safe_div(tf_adj_sums[i], tf_game_counts[i])
                avg_tricks = safe_div(tf_trick_sums[i], tf_game_counts[i])
                
                print(f"    Paradox%={paradox_rate:.1f}%, AdjBonus={avg_adj:.3f}, "
                      f"AvgTricks={avg_tricks:.3f}")

            # RandomRollout stats
            print_stats_for("RandomRolloutISMCTS", randomrollout_returns)
            if rr_game_count > 0:
                paradox_rate = (rr_paradox_count / rr_game_count) * 100
                avg_adj = rr_adj_sum / rr_game_count
                avg_tricks = rr_trick_sum / rr_game_count
                print(f"    Paradox%={paradox_rate:.1f}%, AdjBonus={avg_adj:.3f}, "
                      f"AvgTricks={avg_tricks:.3f}")

            # UniformRandom stats
            print_stats_for("UniformRandom", pure_random_returns)
            if ur_game_count > 0:
                paradox_rate = (ur_paradox_count / ur_game_count) * 100
                avg_adj = ur_adj_sum / ur_game_count
                avg_tricks = ur_trick_sum / ur_game_count
                print(f"    Paradox%={paradox_rate:.1f}%, AdjBonus={avg_adj:.3f}, "
                      f"AvgTricks={avg_tricks:.3f}")

            # Statistical comparisons
            combined_data = []
            combined_groups = []

            # Gather data from all groups
            for i, ret_list in enumerate(tf_returns_by_params):
                combined_data.extend(ret_list)
                combined_groups.extend([f"TrickFollowingISMCTS_{i}"] * len(ret_list))

            if randomrollout_returns:
                combined_data.extend(randomrollout_returns)
                combined_groups.extend(["RandomRolloutISMCTS"] * len(randomrollout_returns))

            if pure_random_returns:
                combined_data.extend(pure_random_returns)
                combined_groups.extend(["UniformRandom"] * len(pure_random_returns))

            # Only proceed if we have at least 2 groups with data
            unique_groups = sorted(set(combined_groups))
            if len(unique_groups) > 1:
                print("\nPairwise Statistical Tests:")

                # Tukey HSD test
                df = pd.DataFrame({"score": combined_data, "group": combined_groups})
                tukey = pairwise_tukeyhsd(df["score"], df["group"])
                print("\nTukey HSD Results:")
                print(tukey)

                # Pairwise t-tests
                print("\nPairwise t-tests (Welch's), sorted by p-value:")
                # Collect all pairwise tests
                pairwise_results = []
                for g1, g2 in combinations(unique_groups, 2):
                    data1 = df[df["group"] == g1]["score"]
                    data2 = df[df["group"] == g2]["score"]
                    tstat, pval = stats.ttest_ind(data1, data2, equal_var=False)
                    reject_null = pval < 0.05  # Using 0.05 significance level
                    pairwise_results.append((pval, g1, g2, reject_null))
                
                # Sort by p-value and print
                for pval, g1, g2, reject_null in sorted(pairwise_results, key=lambda x: x[0]):
                    print(f"{g1} vs {g2}: p={pval:.3}, reject: {reject_null}")

    # Final summary
    print("\nFinal results across all episodes:")
    def print_stats_for(label, data):
        if len(data) == 0:
            print(f"  {label}: No data.")
            return
        mean_r = np.mean(data)
        std_r = np.std(data)
        conf = stats.t.interval(
            0.90, df=len(data)-1, loc=mean_r, scale=stats.sem(data)
        )
        halfwidth = (conf[1] - conf[0]) / 2.0
        print(f"  {label}: N={len(data)} mean={mean_r:.3f}, "
              f"std={std_r:.3f}, 90%CI=({mean_r:.3f} ± {halfwidth:.3f})")

    print_stats_for("TrickFollowingISMCTS", trickfollow_returns)
    print_stats_for("RandomRolloutISMCTS", randomrollout_returns)
    print_stats_for("UniformRandom", pure_random_returns)


if __name__ == "__main__":
    main()
