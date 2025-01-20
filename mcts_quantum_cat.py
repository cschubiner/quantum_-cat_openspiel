#!/usr/bin/env python3

import pyspiel
import numpy as np
from tqdm import tqdm
from scipy import stats
from open_spiel.python.algorithms.ismcts import (
    ISMCTSBot,
    ChildSelectionPolicy,
    ISMCTSFinalPolicyType,
    UNLIMITED_NUM_WORLD_SAMPLES
)
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator

from open_spiel.python.games import quantum_cat


NUM_RANDOM_BOTS = 2

def main():
    game = pyspiel.load_game("python_quantum_cat", {"players": 1 + NUM_RANDOM_BOTS})

    # Create an ISMCTS bot for player 0
    ismcts_evaluator = RandomRolloutEvaluator(n_rollouts=2, random_state=np.random.RandomState(42))

    # Create random bots for players 1 and 2
    USE_ISMCTS_BOT = True
    # USE_ISMCTS_BOT = False
    if USE_ISMCTS_BOT:
        bot0 = ISMCTSBot(
            game=game,
            evaluator=ismcts_evaluator,
            uct_c=2.0,
            # max_simulations=500,
            max_simulations=2200,
            max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
            random_state=np.random.RandomState(999),
            final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
            use_observation_string=False,
            allow_inconsistent_action_sets=False,
            child_selection_policy=ChildSelectionPolicy.PUCT
        )
    else:
        bot0 = pyspiel.make_uniform_random_bot(0, 77)
    random_bots = [
        pyspiel.make_uniform_random_bot(player_id, 100 + player_id*111)
        for player_id in range(1, NUM_RANDOM_BOTS + 1)
    ]

    if USE_ISMCTS_BOT:
        num_episodes = 1000
    else:
        num_episodes = 1500

    ismcts_returns = []
    for _ in tqdm(range(num_episodes), desc="Playing episodes"):
        state = game.new_initial_state()
        bots = [bot0] + random_bots
        while not state.is_terminal():
            current_player = state.current_player()
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = np.random.choice(actions, p=probs)
                state.apply_action(action)
            else:
                action = bots[current_player].step(state)
                if action is None:
                    # Fallback: pick a random legal action
                    action = np.random.choice(state.legal_actions(current_player))
                state.apply_action(action)

        final_returns = state.returns()
        ismcts_returns.append(final_returns[0])  # Track the ISMCTS player's return
        
        # Print running stats every 5 episodes in ISMCTS mode
        if USE_ISMCTS_BOT and (_ + 1) % 5 == 0:
            mean_return = np.mean(ismcts_returns)
            std_return = np.std(ismcts_returns)
            confidence_interval = stats.t.interval(
                confidence=0.90,
                df=len(ismcts_returns)-1,
                loc=mean_return,
                scale=stats.sem(ismcts_returns)
            )
            plus_minus_ci = (confidence_interval[1] - confidence_interval[0]) / 2
            print(f"ISMCTS results over {_ + 1} episodes:")
            print(f"  Average return: {mean_return:.3f} ± {std_return:.3f}")
            print(f"  90% confidence interval: {mean_return:.3f} ± {plus_minus_ci:.3f}")
            
        print(f"Game over. Returns: {final_returns}")


if __name__ == "__main__":
    main()
