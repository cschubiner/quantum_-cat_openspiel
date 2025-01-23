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
    parser.add_argument("--x_random_rollout_bots", type=int, default=2,
                        help="Number of ISMCTS bots using RandomRolloutEvaluator.")
    parser.add_argument("--y_random_bots", type=int, default=2,
                        help="Number of purely random bots.")
    parser.add_argument("--players", type=int, default=5,
                        help="Total players = 1 (TrickFollowingISMCTS) + X + Y. "
                             "If this is set, we override X+Y to match (players-1).")
    parser.add_argument("--uct_c", type=float, default=2.0,
                        help="UCT exploration constant for all MCTS bots.")
    parser.add_argument("--max_sims", type=int, default=250,
                        help="Number of simulations for each ISMCTS bot per move.")
    parser.add_argument("--seed", type=int, default=999,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    # If user sets --players, we interpret that as 1 + X + Y = players
    # So X+Y = players-1. We'll guess that X=... user might want a specific ratio
    # For now let's just enforce X+Y=players-1.
    # If user also specified x_random_rollout_bots and y_random_bots,
    # we won't break them, but if there's a mismatch we override.
    if (1 + args.x_random_rollout_bots + args.y_random_bots) != args.players:
        # override x+y:
        new_val = args.players - 1
        # We'll keep the ratio of x:y if possible
        total_xy = args.x_random_rollout_bots + args.y_random_bots
        if total_xy > 0:
            ratio_x = args.x_random_rollout_bots / total_xy
            ratio_y = args.y_random_bots / total_xy
            args.x_random_rollout_bots = int(round(ratio_x * new_val))
            args.y_random_bots = new_val - args.x_random_rollout_bots
        else:
            # If x_random_rollout_bots + y_random_bots was 0, then let's just put all to X
            args.x_random_rollout_bots = new_val
            args.y_random_bots = 0
        print(f"Adjusted X+Y to match total players. Now X={args.x_random_rollout_bots}, "
              f"Y={args.y_random_bots} for a total of {args.players} players.")

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
    tf_param_sets = [
        # Baseline/original parameters
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
        # More aggressive trump usage
        dict(
            discard_frequent_prob=0.90,
            discard_infrequent_prob=0.10,
            pred_main_prob=0.80,
            pred_neighbor_prob=0.15,
            pred_uniform_prob=0.05,
            follow_suit_prob=0.50,
            deviate_prob=0.50,
            deviate_trump_ratio=0.90,
            deviate_other_ratio=0.10,
        ),
        # More conservative, suit-following focused
        dict(
            discard_frequent_prob=0.80,
            discard_infrequent_prob=0.20,
            pred_main_prob=0.60,
            pred_neighbor_prob=0.30,
            pred_uniform_prob=0.10,
            follow_suit_prob=0.80,
            deviate_prob=0.20,
            deviate_trump_ratio=0.60,
            deviate_other_ratio=0.40,
        ),
    ]

    # Create TrickFollowing bots with different parameter sets
    tf_bots = []
    for i, params in enumerate(tf_param_sets):
        tf_evaluator = TrickFollowingEvaluator(
            n_rollouts=2,
            random_state=np.random.RandomState(args.seed + 123 + i),
            **params
        )
        bot = ISMCTSBot(
            game=game,
            evaluator=tf_evaluator,
            uct_c=args.uct_c,
            max_simulations=args.max_sims,
            max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
            random_state=np.random.RandomState(args.seed + 999 + i),
            final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
            use_observation_string=False,
            allow_inconsistent_action_sets=False,
            child_selection_policy=ChildSelectionPolicy.PUCT
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
        player_id = 1 + len(rr_bots) + i
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
