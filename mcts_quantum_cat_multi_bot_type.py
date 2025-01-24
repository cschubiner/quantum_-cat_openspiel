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

from mcts_quantum_cat import TrickFollowingEvaluator
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
        # Aggressive follow-suit
        dict(
            discard_frequent_prob=0.85,
            discard_infrequent_prob=0.15,
            pred_main_prob=0.70,
            pred_neighbor_prob=0.20,
            pred_uniform_prob=0.10,
            follow_suit_prob=0.70,  # More likely to follow suit
            deviate_prob=0.30,      # Less likely to deviate
            deviate_trump_ratio=0.80, # More likely to play trump when deviating
            deviate_other_ratio=0.20,
        ),
        # Deviate-heavy
        dict(
            discard_frequent_prob=0.85,
            discard_infrequent_prob=0.15,
            pred_main_prob=0.70,
            pred_neighbor_prob=0.20,
            pred_uniform_prob=0.10,
            follow_suit_prob=0.50,  # Less likely to follow suit
            deviate_prob=0.50,      # More likely to deviate
            deviate_trump_ratio=0.65, # Less likely to play trump when deviating
            deviate_other_ratio=0.35,
        ),
        # Balanced approach
        dict(
            discard_frequent_prob=0.85,
            discard_infrequent_prob=0.15,
            pred_main_prob=0.70,
            pred_neighbor_prob=0.20,
            pred_uniform_prob=0.10,
            follow_suit_prob=0.55,  # Slightly less likely to follow suit
            deviate_prob=0.45,      # Slightly more likely to deviate
            deviate_trump_ratio=0.85, # Much more likely to play trump when deviating
            deviate_other_ratio=0.15,
        ),
    ]

    # ISMCTS bot parameters
    ismcts_param_sets = [
        # Conservative settings
        dict(
            uct_c=2.0,
            max_simulations=300,
            final_policy_type="MAX_VISIT_COUNT",
            child_selection_policy="PUCT",
        ),
        # Exploration-focused
        dict(
            uct_c=3.0,              # More exploration
            max_simulations=400,
            final_policy_type="NORMALIZED_VISITED_COUNT",
            child_selection_policy="UCT",
        ),
        # Deep-thinking exploiter
        dict(
            uct_c=1.5,              # More exploitation
            max_simulations=600,     # More thinking time
            final_policy_type="MAX_VALUE",
            child_selection_policy="PUCT",
        ),
        # Quick decisions
        dict(
            uct_c=2.5,
            max_simulations=250,     # Faster decisions
            final_policy_type="MAX_VISIT_COUNT",
            child_selection_policy="UCT",
        ),
    ]


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
    randomrollout_returns = []
    pure_random_returns = []

    # Define different parameter sets for TrickFollowingISMCTS bots

    # Create TrickFollowing bots with different parameter sets
    tf_bots = []
    for i, params in enumerate(tf_param_sets):
        # Create evaluator with TrickFollowing parameters
        tf_evaluator = TrickFollowingEvaluator(
            n_rollouts=2,
            random_state=np.random.RandomState(args.seed + 123 + i),
            **tf_param_sets[i]
        )

        # Get corresponding ISMCTS parameters
        ismcts_params = ismcts_param_sets[i]
        
        # Convert string parameters to enums
        csp_str = ismcts_params["child_selection_policy"]
        if csp_str == "PUCT":
            csp = ChildSelectionPolicy.PUCT
        elif csp_str == "UCT":
            csp = ChildSelectionPolicy.UCT
        else:
            raise ValueError(f"Unsupported child_selection_policy: {csp_str}")

        fpt_str = ismcts_params["final_policy_type"]
        if fpt_str == "MAX_VISIT_COUNT":
            fpt = ISMCTSFinalPolicyType.MAX_VISIT_COUNT
        elif fpt_str == "NORMALIZED_VISITED_COUNT":
            fpt = ISMCTSFinalPolicyType.NORMALIZED_VISITED_COUNT
        elif fpt_str == "MAX_VALUE":
            fpt = ISMCTSFinalPolicyType.MAX_VALUE
        else:
            raise ValueError(f"Unsupported final_policy_type: {fpt_str}")

        bot = ISMCTSBot(
            game=game,
            evaluator=tf_evaluator,
            uct_c=ismcts_params["uct_c"],
            max_simulations=ismcts_params["max_simulations"],
            max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
            random_state=np.random.RandomState(args.seed + 999 + i),
            final_policy_type=fpt,
            use_observation_string=False,
            allow_inconsistent_action_sets=False,
            child_selection_policy=csp
        )
        tf_bots.append(bot)

    # Track returns by parameter set
    tf_returns_by_params = [[] for _ in tf_param_sets]

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
        # Tally them by category
        for pid, r in enumerate(final_returns):
            if bot_types[pid].startswith("trickfollow_"):
                param_idx = int(bot_types[pid].split("_")[1])
                tf_returns_by_params[param_idx].append(r)
            elif bot_types[pid] == "randomrollout":
                randomrollout_returns.append(r)
            else:
                pure_random_returns.append(r)

        # Every 5 episodes, print out stats
        if (episode_idx + 1) % 5 == 0:
            print("=" * 60)
            print(f"Episode {episode_idx+1} completed. Intermediate stats:")

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

            for i, returns in enumerate(tf_returns_by_params):
                print_stats_for(f"TrickFollowingISMCTS_{i}", returns)
            print_stats_for("RandomRolloutISMCTS", randomrollout_returns)
            print_stats_for("UniformRandom", pure_random_returns)

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
                print("\nPairwise t-tests (Welch's):")
                for g1, g2 in combinations(unique_groups, 2):
                    data1 = df[df["group"] == g1]["score"]
                    data2 = df[df["group"] == g2]["score"]
                    tstat, pval = stats.ttest_ind(data1, data2, equal_var=False)
                    print(f"{g1} vs {g2}: t={tstat:.3f}, p={pval:.3e}")

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
