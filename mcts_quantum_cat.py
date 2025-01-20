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

        if state._phase == 1:
            distribution = self._get_discard_distribution(state, legal_actions)
            return list(zip(legal_actions, distribution))

        elif state._phase == 2:
            distribution = self._get_prediction_distribution(state, legal_actions)
            return list(zip(legal_actions, distribution))

        # else: trick-taking phase => existing fallback
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
                    if working_state._phase == 1:
                        distribution = self._get_discard_distribution(working_state, legals)
                    elif working_state._phase == 2:
                        distribution = self._get_prediction_distribution(working_state, legals)
                    else:
                        # Trick-taking
                        distribution = self._compute_suit_following_distribution(working_state, legals)

                    chosen = self._random_state.choice(legals, p=distribution)
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

    def _get_discard_distribution(self, state, legal_actions):
        """Return 85%-most-frequent-rank, 15%-other discard distribution."""
        hand_vec = state._hands[state.current_player()]
        max_count = max(hand_vec[r] for r in legal_actions)
        most_ranks = [r for r in legal_actions if hand_vec[r] == max_count]
        distribution = []
        
        # Check how many 'others' are left
        others_count = len(legal_actions) - len(most_ranks)
        
        for r in legal_actions:
            if r in most_ranks:
                if others_count == 0:
                    # If *all* ranks are "most," give them uniform probability
                    distribution.append(1.0 / len(legal_actions))
                else:
                    # 85% portion among 'most_ranks'
                    distribution.append(0.85 / len(most_ranks))
            else:
                # If there *are* others
                if others_count > 0:
                    distribution.append(0.15 / others_count)
                else:
                    distribution.append(0.0)

        # Normalize if needed
        s = sum(distribution)
        if s > 1e-12:  # guard against divide by zero
            distribution = [x / s for x in distribution]
        else:
            # fallback uniform if something went wrong
            distribution = [1.0 / len(legal_actions)] * len(legal_actions)
            
        return distribution

    def _get_prediction_distribution(self, state, legal_actions):
        """
        Return distribution for [101..104] => 1..4:
          - 70% on 'guess' = min(max(count_of_highest_rank,1),4),
          - 20% on ±1 (if valid),
          - 10% uniform among all 4 predictions.
        """
        # find highest rank in your hand, clamp how many copies to 1..4
        current_player = state.current_player()
        hand_vec = state._hands[current_player]
        best_rank_idx = max((i for i in range(len(hand_vec)) if hand_vec[i] > 0), default=0)
        best_count = hand_vec[best_rank_idx]
        guess = min(max(best_count, 1), 4)

        distribution = [0.0] * len(legal_actions)  # e.g. legal_actions = [101..104]

        # (A) 70% on 'guess'
        guess_action = 100 + guess  # e.g., guess=2 => 102
        if guess_action in legal_actions:
            i_guess = legal_actions.index(guess_action)
            distribution[i_guess] += 0.70

        # (B) 20% on ±1 if valid
        near_candidates = []
        if guess > 1: near_candidates.append(guess - 1)
        if guess < 4: near_candidates.append(guess + 1)
        if near_candidates:
            share = 0.20 / len(near_candidates)
            for c in near_candidates:
                a_val = 100 + c
                if a_val in legal_actions:
                    i_near = legal_actions.index(a_val)
                    distribution[i_near] += share

        # (C) 10% uniform among all legal predictions
        if len(legal_actions) > 0:
            each = 0.10 / len(legal_actions)
            for i in range(len(legal_actions)):
                distribution[i] += each

        # Normalize so sum(distribution) == 1
        s = sum(distribution)
        if s > 1e-12:
            distribution = [x / s for x in distribution]
        else:
            # fallback uniform
            distribution = [1.0 / len(legal_actions)] * len(legal_actions)

        return distribution
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
