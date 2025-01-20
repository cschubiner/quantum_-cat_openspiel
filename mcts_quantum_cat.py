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
          - If you haven't used the led color at all, follow that color at 100% prob.
          - Once you've removed that color token, follow suit 60%/deviate 40%.
          - If you deviate, 75% trump vs. 25% other (but re-scale if no trump or no other).
        """
        led_color = state._led_color  # e.g. "R", "B", "Y", "G", or None if no lead
        current_player = state.current_player()
        color_tokens = state._color_tokens[current_player]  # shape [4]
        color_map = {"R": 0, "B": 1, "Y": 2, "G": 3}

        # If nothing is led, fallback to uniform among legals.
        if led_color is None:
            return np.ones(len(legal_actions)) / len(legal_actions)

        led_idx = color_map[led_color]

        # Partition legal_actions by color
        follow_actions = []
        trump_actions = []
        other_actions = []
        for a in legal_actions:
            c_idx = a // state._num_card_types
            if c_idx == led_idx:
                follow_actions.append(a)
            elif c_idx == 0:  # 0 => "R"
                trump_actions.append(a)
            else:
                other_actions.append(a)

        # If no possible follow => must deviate (some mix of trump vs. other).
        if len(follow_actions) == 0:
            distribution = np.zeros(len(legal_actions), dtype=float)
            t_count = len(trump_actions)
            o_count = len(other_actions)

            if t_count == 0 and o_count == 0:
                # Shouldn't happen if legal_actions is nonempty, but just in case:
                return np.ones(len(legal_actions)) / len(legal_actions)

            # We'll treat the "0.75 / 0.25" as a ratio, then re-scale if one portion is missing.
            # Example approach: if no 'other_actions', all deviate-prob goes to trump (and vice versa).
            deviate_prob = 1.0  # the entire probability goes to "deviate" scenario
            # We'll keep the 75:25 ratio if both sets exist:
            ratio_trump = 0.75
            ratio_other = 0.25

            if t_count > 0 and o_count > 0:
                # normal 75/25 split
                total_weight = ratio_trump + ratio_other  # 1.0
                # portion for trump vs other:
                portion_trump = (ratio_trump / total_weight) * deviate_prob
                portion_other = (ratio_other / total_weight) * deviate_prob
                # assign them
                for i, a in enumerate(legal_actions):
                    if a in trump_actions:
                        distribution[i] = portion_trump / t_count
                    elif a in other_actions:
                        distribution[i] = portion_other / o_count
            elif t_count > 0:
                # only trump actions => entire deviate prob = 1 => all on trump
                for i, a in enumerate(legal_actions):
                    if a in trump_actions:
                        distribution[i] = deviate_prob / t_count
            else:
                # only other actions => entire deviate prob = 1 => all on other
                for i, a in enumerate(legal_actions):
                    if a in other_actions:
                        distribution[i] = deviate_prob / o_count

            return distribution

        # If color_tokens[led_idx] is still True => "never left suit", follow 100%:
        if color_tokens[led_idx]:
            distribution = np.zeros(len(legal_actions), dtype=float)
            # all legal follow actions share probability 1
            for i, a in enumerate(legal_actions):
                if a in follow_actions:
                    distribution[i] = 1.0 / len(follow_actions)
            return distribution

        # Else we have used/removed that suit. Follow ~60%, deviate ~40% (split 75/25 to trump/others).
        distribution = np.zeros(len(legal_actions), dtype=float)
        f_count = len(follow_actions)
        t_count = len(trump_actions)
        o_count = len(other_actions)

        follow_prob = 0.60
        deviate_prob = 0.40

        # among deviate portion => 75% to trump, 25% to other
        ratio_trump = 0.75
        ratio_other = 0.25
        total_weight = ratio_trump + ratio_other  # 1.0

        # if t_count>0 and o_count>0 => normal scenario
        # if t_count=0 => all deviate to other
        # if o_count=0 => all deviate to trump
        for i, a in enumerate(legal_actions):
            if a in follow_actions:
                # follow-suit portion
                distribution[i] = follow_prob / f_count
            elif t_count > 0 and o_count > 0:
                if a in trump_actions:
                    distribution[i] = (deviate_prob * ratio_trump / total_weight) / t_count
                elif a in other_actions:
                    distribution[i] = (deviate_prob * ratio_other / total_weight) / o_count
            elif t_count > 0:  # only trump
                if a in trump_actions:
                    distribution[i] = deviate_prob / t_count
            else:  # only others
                if a in other_actions:
                    distribution[i] = deviate_prob / o_count

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
            max_simulations=500,
            # max_simulations=2200,
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
