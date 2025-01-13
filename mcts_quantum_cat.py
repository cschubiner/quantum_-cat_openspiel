#!/usr/bin/env python3

import pyspiel
import numpy as np
from tqdm import tqdm
from open_spiel.python.algorithms.ismcts import (
    ISMCTSBot,
    ChildSelectionPolicy,
    ISMCTSFinalPolicyType,
    UNLIMITED_NUM_WORLD_SAMPLES
)
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator

from open_spiel.python.games import quantum_cat


def main():
    game = pyspiel.load_game("python_quantum_cat", {"players": 3})

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
            max_simulations=30,
            max_world_samples=UNLIMITED_NUM_WORLD_SAMPLES,
            random_state=np.random.RandomState(999),
            final_policy_type=ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
            use_observation_string=False,
            allow_inconsistent_action_sets=False,
            child_selection_policy=ChildSelectionPolicy.PUCT
        )
    else:
        bot0 = pyspiel.make_uniform_random_bot(0, 77)
    random_bot1 = pyspiel.make_uniform_random_bot(1, 111)
    random_bot2 = pyspiel.make_uniform_random_bot(2, 222)

    if USE_ISMCTS_BOT:
        num_episodes = 7
    else:
        num_episodes = 1500

    ismcts_returns = []
    for _ in tqdm(range(num_episodes), desc="Playing episodes"):
        state = game.new_initial_state()
        bots = [bot0, random_bot1, random_bot2]
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
        print(f"Game over. Returns: {final_returns}")

    mean_return = np.mean(ismcts_returns)
    std_return = np.std(ismcts_returns)
    print(f"ISMCTS results over {num_episodes} episodes:")
    print(f"  Average return: {mean_return:.3f} ± {std_return:.3f}")


if __name__ == "__main__":
    main()
