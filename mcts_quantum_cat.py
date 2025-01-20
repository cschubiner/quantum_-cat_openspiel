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


class TrickFollowingEvaluator(RandomRolloutEvaluator):
    """
    Uses a suit-following heuristic both for prior probabilities and for
    rollouts. If a suit is led and the player still has a token for that suit,
    they follow it with high probability. Otherwise they might deviate to trump
    or another suit.
    """

    def prior(self, state):
        """Returns a list of (action, probability) pairs for the root node expansion."""
        legal_actions = state.legal_actions(state.current_player())
        if not legal_actions:
            return []

        # Compute a simple distribution for 'suit-following' logic.
        action_probs = self._compute_suit_following_distribution(state, legal_actions)
        return list(zip(legal_actions, action_probs))

    def evaluate(self, state):
        """Returns a terminal value estimate for the state + does a random(ish) simulation.

        We'll override the random step distribution with the same suit-following idea.
        """
        # If terminal, just return returns.
        if state.is_terminal():
            return state.returns()

        working_state = state.clone()
        while not working_state.is_terminal():
            current_player = working_state.current_player()
            if working_state.is_chance_node():
                outcomes = working_state.chance_outcomes()
                acts, probs = zip(*outcomes)
                chosen = self._random_state.choice(acts, p=probs)
                working_state.apply_action(chosen)
            else:
                legals = working_state.legal_actions(current_player)
                if legals:
                    action_probs = self._compute_suit_following_distribution(working_state, legals)
                    chosen = self._random_state.choice(legals, p=action_probs)
                    working_state.apply_action(chosen)
                else:
                    # No moves: must paradox, or the game might handle it automatically.
                    # Just break or let the game handle it.
                    break

        return working_state.returns()

    def _compute_suit_following_distribution(self, state, legal_actions):
        """
        A helper that gives probabilities for each legal action, encouraging:
          - If you haven't used the led color at all, always follow that color (prob=1).
          - If you've used it once already, ~60% chance to follow suit, 40% to deviate.
          - If deviating, 75% of those times pick Red (if possible), else pick from others.
        """
        # Identify led color
        led_color = state._led_color  # e.g. "R", "B", "Y", "G", or None if no lead
        current_player = state.current_player()
        color_tokens = state._color_tokens[current_player]  # bool array, shape [4]
        # color index: 0=R,1=B,2=Y,3=G
        color_map = {"R": 0, "B": 1, "Y": 2, "G": 3}

        # If no suit is led, we just pick uniform among legals
        if led_color is None:
            return np.ones(len(legal_actions)) / len(legal_actions)

        led_idx = color_map[led_color]
        # Count how many times we've used that led color so far
        # (We do it by scanning board_ownership or by seeing if color_tokens is still True.)
        # But simpler logic: if color_tokens[led_idx] is still True, that means we
        # haven't removed that color. We can interpret that as not having forced a color removal,
        # which typically means we haven't "left" that suit in a prior trick.
        # For the example requested, we'll treat "unused" as color_tokens[led_idx] == True.

        # Step 1: gather subsets of legal actions
        follow_actions = []
        trump_actions = []
        other_actions = []
        for a in legal_actions:
            c_idx = a // state._num_card_types
            if c_idx == led_idx:
                follow_actions.append(a)
            elif c_idx == 0:  # 0 means "R"
                trump_actions.append(a)
            else:
                other_actions.append(a)

        # If we cannot follow because we have no valid follow actions, then we just deviate:
        if len(follow_actions) == 0:
            # 75% trump, 25% everything else
            if len(trump_actions) == 0 and len(other_actions) == 0:
                # No choice
                return np.ones(len(legal_actions)) / len(legal_actions)

            # Weighted distribution among trump vs. others
            distribution = np.zeros(len(legal_actions), dtype=float)
            for i, a in enumerate(legal_actions):
                if a in trump_actions:
                    distribution[i] = 0.75 / len(trump_actions)
                elif a in other_actions:
                    distribution[i] = 0.25 / len(other_actions)
            return distribution

        # If color_tokens[led_idx] is still True => treat that as "never left suit"
        # => always follow suit with probability 1.0
        if color_tokens[led_idx]:
            distribution = np.zeros(len(legal_actions), dtype=float)
            for i, a in enumerate(legal_actions):
                if a in follow_actions:
                    distribution[i] = 1.0 / len(follow_actions)
            return distribution

        # Else, we must have used that color once (or removed that token).
        # We'll follow suit 60% of the time, deviate 40% of the time.
        # Among the deviate portion, 75% on trump, 25% on others.
        distribution = np.zeros(len(legal_actions), dtype=float)

        if len(trump_actions) == 0 and len(other_actions) == 0:
            # If we cannot deviate (no trump or other color),
            # then we must follow 100% of the time
            for i, a in enumerate(legal_actions):
                if a in follow_actions:
                    distribution[i] = 1.0 / len(follow_actions)
            return distribution

        # Normal scenario: some follow possible, some deviate possible
        # portion for follow
        follow_prob = 0.60
        # portion for deviate
        deviate_prob = 0.40
        # within deviate => 75% trump, 25% others
        dev_trump_prob = 0.75 * deviate_prob
        dev_other_prob = 0.25 * deviate_prob

        total_follow = len(follow_actions)
        total_trump = len(trump_actions)
        total_other = len(other_actions)

        for i, a in enumerate(legal_actions):
            if a in follow_actions:
                distribution[i] = follow_prob / total_follow
            elif a in trump_actions:
                distribution[i] = dev_trump_prob / total_trump
            else:
                distribution[i] = dev_other_prob / total_other

        return distribution


NUM_RANDOM_BOTS = 2

def main():
    game = pyspiel.load_game("python_quantum_cat", {"players": 1 + NUM_RANDOM_BOTS})

    # Create an ISMCTS bot for player 0
    ismcts_evaluator = TrickFollowingEvaluator(
        n_rollouts=2,
        random_state=np.random.RandomState(42)
    )

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
