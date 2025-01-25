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


import numpy as np
import collections
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator


class TrickFollowingEvaluatorV2(RandomRolloutEvaluator):
    """
    An evaluator for Cat in the Box that uses:
      1) Suit-following heuristics to decide how likely we are to follow suit,
         deviate, or trump, *plus*
      2) Adjacency-based weighting when choosing among multiple rank options
         within the same suit or color category.

    We also preserve the original logic for discarding and prediction phases,
    but you can change the parameters below to suit your preferences.
    """

    def __init__(
        self,
        n_rollouts=2,
        random_state=None,
        # Discard distribution params
        discard_frequent_prob=0.85,
        discard_infrequent_prob=0.15,
        # Prediction distribution params
        pred_main_prob=0.70,
        pred_neighbor_prob=0.20,
        pred_uniform_prob=0.10,
        # Trick-taking color logic
        follow_suit_prob=0.60,     # Probability allocated to following suit (if possible)
        deviate_prob=0.40,        # Probability allocated to deviating if we have already used that suit
        deviate_trump_ratio=0.75, # Within deviate_prob, fraction that tries trump
        deviate_other_ratio=0.25, # Within deviate_prob, fraction that tries other non-led, non-trump color
        # Adjacency weighting params
        adjacency_base=1.0,       # Baseline adjacency weight
        adjacency_gain_scale=1.0, # How much to weight an increase in largest-cluster size
    ):
        """
        Args:
          n_rollouts: number of random rollouts for state evaluation.
          random_state: an optional np.random.RandomState or None.
          discard_frequent_prob: portion to allocate to discarding the player's
             most frequent rank(s).
          discard_infrequent_prob: portion for discarding less-frequent ranks.
          pred_main_prob: portion for "predicted best guess" bid.
          pred_neighbor_prob: portion for bidding adjacent to that guess (±1).
          pred_uniform_prob: portion spread uniformly among all valid predictions.
          follow_suit_prob: portion for following suit if still valid.
          deviate_prob: portion for deviating from the led suit if we've used that suit before.
          deviate_trump_ratio: fraction of deviate_prob that tries trump over other suits.
          deviate_other_ratio: fraction of deviate_prob that tries non-trump color if not following suit.
          adjacency_base: baseline for adjacency weighting.
             If the new cluster size is the same as old, we get weight=adjacency_base.
          adjacency_gain_scale: how strongly to reward expansions in your largest adjacency cluster.
             If placing a token grows your largest cluster from old_size to new_size,
             final weight = adjacency_base + adjacency_gain_scale*(new_size - old_size).
        """
        super().__init__(n_rollouts=n_rollouts, random_state=random_state)
        # Store distribution parameters
        self._discard_frequent_prob = discard_frequent_prob
        self._discard_infrequent_prob = discard_infrequent_prob
        self._pred_main_prob = pred_main_prob
        self._pred_neighbor_prob = pred_neighbor_prob
        self._pred_uniform_prob = pred_uniform_prob

        self._follow_suit_prob = follow_suit_prob
        self._deviate_prob = deviate_prob
        self._deviate_trump_ratio = deviate_trump_ratio
        self._deviate_other_ratio = deviate_other_ratio

        # Store adjacency parameters
        self._adjacency_base = adjacency_base
        self._adjacency_gain_scale = adjacency_gain_scale


    # ----------------------------------------------------------------------
    # Public interface for MCTS: prior(...) and evaluate(...).
    # ----------------------------------------------------------------------
    def prior(self, state):
        """Returns a list of (action, probability) for expansion at the root."""
        legal_actions = state.legal_actions(state.current_player())
        if not legal_actions:
            return []

        phase = state._phase
        if phase == 1:
            # Discard
            distribution = self._get_discard_distribution(state, legal_actions)
            return list(zip(legal_actions, distribution))

        elif phase == 2:
            # Prediction
            distribution = self._get_prediction_distribution(state, legal_actions)
            return list(zip(legal_actions, distribution))

        elif phase == 3:
            # Trick-taking with adjacency weighting
            distribution = self._compute_suit_following_distribution(state, legal_actions)
            return list(zip(legal_actions, distribution))

        # Else, fallback uniform if we ever get here unexpectedly
        uniform_probs = np.ones(len(legal_actions)) / len(legal_actions)
        return list(zip(legal_actions, uniform_probs))


    def evaluate(self, state):
        """
        State evaluation by random(ish) simulation with the same logic
        for discarding, prediction, and adjacency-based trick-taking.

        If the state is terminal, just return the final returns.
        """
        if state.is_terminal():
            return state.returns()

        working_state = state.clone()
        while not working_state.is_terminal():
            current_player = working_state.current_player()
            if working_state.is_chance_node():
                outcomes = working_state.chance_outcomes()
                actions, probs = zip(*outcomes)
                chosen = self._random_state.choice(actions, p=probs)
                working_state.apply_action(chosen)
            else:
                legals = working_state.legal_actions(current_player)
                if not legals:
                    # No moves => must paradox or end
                    break

                phase = working_state._phase
                if phase == 1:
                    distribution = self._get_discard_distribution(working_state, legals)
                elif phase == 2:
                    distribution = self._get_prediction_distribution(working_state, legals)
                else:
                    # phase == 3 => adjacency-based trick logic
                    distribution = self._compute_suit_following_distribution(working_state, legals)

                chosen = self._random_state.choice(legals, p=distribution)
                working_state.apply_action(chosen)

        return working_state.returns()


    # ----------------------------------------------------------------------
    # Helper: Discard logic
    # ----------------------------------------------------------------------
    def _get_discard_distribution(self, state, legal_actions):
        """
        Weighted so we discard one of the player's most frequent ranks
        with probability discard_frequent_prob, and we discard a less
        frequent rank with discard_infrequent_prob.
        """
        hand_vec = state._hands[state.current_player()]
        # Find the maximum count in your hand
        max_count = max(hand_vec[r] for r in legal_actions)
        most_ranks = [r for r in legal_actions if hand_vec[r] == max_count]

        distribution = []
        # Count how many are "others"
        others_count = len(legal_actions) - len(most_ranks)

        for r in legal_actions:
            if r in most_ranks:
                # If *all* ranks are "most," they'd share the entire probability
                if others_count == 0:
                    distribution.append(1.0 / len(most_ranks))
                else:
                    distribution.append(self._discard_frequent_prob / len(most_ranks))
            else:
                # Remainder is allocated to "others"
                if others_count > 0:
                    distribution.append(self._discard_infrequent_prob / others_count)
                else:
                    distribution.append(0.0)

        return self._normalize(np.array(distribution))


    # ----------------------------------------------------------------------
    # Helper: Prediction logic
    # ----------------------------------------------------------------------
    def _get_prediction_distribution(self, state, legal_actions):
        """
        Weighted toward "best guess" predicted number of tricks, with
        smaller probabilities for neighbor guesses and uniform fallback.
        """
        current_player = state.current_player()
        hand_vec = state._hands[current_player]
        # For a heuristic guess, pick the highest rank in your hand
        best_rank_idx = max(
            (i for i in range(len(hand_vec)) if hand_vec[i] > 0),
            default=0
        )
        best_count = hand_vec[best_rank_idx]
        guess = min(max(best_count, 1), 4)  # clamp to 1..4

        distribution = np.zeros(len(legal_actions), dtype=float)

        # (A) main prob on "guess"
        guess_action = 100 + guess  # e.g. guess=3 => action=103
        if guess_action in legal_actions:
            i_guess = legal_actions.index(guess_action)
            distribution[i_guess] += self._pred_main_prob

        # (B) neighbor prob on ±1
        near_candidates = []
        if guess > 1:
            near_candidates.append(guess - 1)
        if guess < 4:
            near_candidates.append(guess + 1)
        if near_candidates:
            share = self._pred_neighbor_prob / len(near_candidates)
            for c in near_candidates:
                a_val = 100 + c
                if a_val in legal_actions:
                    i_near = legal_actions.index(a_val)
                    distribution[i_near] += share

        # (C) uniform remainder
        if len(legal_actions) > 0:
            each = self._pred_uniform_prob / len(legal_actions)
            for i in range(len(legal_actions)):
                distribution[i] += each

        return self._normalize(distribution)


    # ----------------------------------------------------------------------
    # Helper: Trick-taking logic with adjacency weighting
    # ----------------------------------------------------------------------
    def _compute_suit_following_distribution(self, state, legal_actions):
        """
        Suit-following logic + adjacency weighting on rank choices.

        1) We partition actions into (follow_actions, trump_actions, other_actions).
        2) We decide how much total probability to put in each group:
           - Possibly 100% if we *cannot* follow or if we haven't used that suit yet.
           - Possibly follow_suit_prob : deviate_prob if we have used that suit.
        3) Within each group, we distribute its portion proportionally to
           an adjacency-based metric, so that placing a token that grows
           your largest adjacency cluster is more favored.
        """
        if legal_actions == [999]:
            return [1.0]

        led_color = state._led_color  # e.g. "R","B","Y","G" or None
        current_player = state.current_player()

        if led_color is None:
            # If there's no led color, just adjacency-weight all legal actions uniformly
            return self._adjacency_biased_uniform(state, current_player, legal_actions, 1.0)

        # Basic color partition
        color_map = {"R": 0, "B": 1, "Y": 2, "G": 3}
        led_idx = color_map[led_color]

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

        # If we cannot follow at all => all probability is "deviate," which we break
        # into trump vs. other at deviate_trump_ratio : deviate_other_ratio
        if len(follow_actions) == 0:
            distribution = np.zeros(len(legal_actions), dtype=float)
            t_count = len(trump_actions)
            o_count = len(other_actions)
            # If neither trump nor other is available, fallback to uniform adjacency:
            if t_count + o_count == 0:
                return self._adjacency_biased_uniform(state, current_player, legal_actions, 1.0)

            deviate_prob = 1.0
            total_ratio = self._deviate_trump_ratio + self._deviate_other_ratio

            if t_count > 0 and o_count > 0:
                portion_trump = deviate_prob * self._deviate_trump_ratio / total_ratio
                portion_other = deviate_prob * self._deviate_other_ratio / total_ratio
                self._apply_adjacency_weighting(
                    state, current_player, trump_actions, portion_trump,
                    distribution, legal_actions
                )
                self._apply_adjacency_weighting(
                    state, current_player, other_actions, portion_other,
                    distribution, legal_actions
                )
            elif t_count > 0:
                # only trump
                self._apply_adjacency_weighting(
                    state, current_player, trump_actions, deviate_prob,
                    distribution, legal_actions
                )
            else:
                # only others
                self._apply_adjacency_weighting(
                    state, current_player, other_actions, deviate_prob,
                    distribution, legal_actions
                )

            return self._normalize(distribution)

        # If we *can* follow, check whether we've used that suit yet:
        # if not used => 100% follow
        has_used_led_color = np.any(state._board_ownership[led_idx] == current_player)
        if not has_used_led_color:
            distribution = np.zeros(len(legal_actions), dtype=float)
            # All probability to follow_actions
            self._apply_adjacency_weighting(
                state, current_player, follow_actions, 1.0,
                distribution, legal_actions
            )
            return self._normalize(distribution)

        # Else standard 60% to follow, 40% deviate (split among trump vs other)
        distribution = np.zeros(len(legal_actions), dtype=float)
        # 1) follow
        self._apply_adjacency_weighting(
            state, current_player, follow_actions, self._follow_suit_prob,
            distribution, legal_actions
        )
        # 2) deviate => portion between trump and other
        deviate_portion = self._deviate_prob
        total_ratio = self._deviate_trump_ratio + self._deviate_other_ratio

        t_count = len(trump_actions)
        o_count = len(other_actions)
        if t_count > 0 and o_count > 0:
            portion_trump = deviate_portion * self._deviate_trump_ratio / total_ratio
            portion_other = deviate_portion * self._deviate_other_ratio / total_ratio
            self._apply_adjacency_weighting(
                state, current_player, trump_actions, portion_trump,
                distribution, legal_actions
            )
            self._apply_adjacency_weighting(
                state, current_player, other_actions, portion_other,
                distribution, legal_actions
            )
        elif t_count > 0:
            # all deviate prob to trump
            self._apply_adjacency_weighting(
                state, current_player, trump_actions, deviate_portion,
                distribution, legal_actions
            )
        elif o_count > 0:
            # all deviate prob to others
            self._apply_adjacency_weighting(
                state, current_player, other_actions, deviate_portion,
                distribution, legal_actions
            )
        # If neither trump nor other, we've already assigned follow part; do nothing extra.

        return self._normalize(distribution)


    # ----------------------------------------------------------------------
    # Adjacency weighting subroutines
    # ----------------------------------------------------------------------
    def _apply_adjacency_weighting(self, state, player, action_subset, portion,
                                   out_distribution, all_legals):
        """
        Distribute 'portion' of probability among actions in `action_subset`
        proportionally to each action's adjacency weight.
        """
        if not action_subset or portion <= 1e-12:
            return  # nothing to do

        # 1) compute adjacency weights for each action
        weights = []
        for a in action_subset:
            w = self._adjacency_weight(state, player, a)
            weights.append(max(w, 0.0))

        total_w = sum(weights)
        if total_w < 1e-12:
            # fallback uniform if adjacency is all zero
            uniform_prob = portion / len(action_subset)
            for a in action_subset:
                idx = all_legals.index(a)
                out_distribution[idx] += uniform_prob
            return

        # 2) distribute portion in ratio to weights
        for i, a in enumerate(action_subset):
            idx = all_legals.index(a)
            out_distribution[idx] += portion * (weights[i] / total_w)

    def _adjacency_weight(self, state, player, action):
        """
        Return a numeric "preference" for placing a token at color/rank indicated by `action`,
        based on how it grows your largest adjacency cluster.

        We'll measure:
          old_size = largest cluster for 'player' now,
          new_size = largest cluster if we place this token,
          and final = adjacency_base + adjacency_gain_scale*(new_size - old_size).

        If new_size <= old_size, final = adjacency_base.
        """
        color_idx = action // state._num_card_types
        rank_idx  = action % state._num_card_types

        old_size = self._largest_cluster_for_player(state, player)

        # Make a copy of the board ownership
        board_copy = np.copy(state._board_ownership)
        # Place this token
        board_copy[color_idx, rank_idx] = player

        new_size = self._largest_cluster_for_player(state, player, board_override=board_copy)

        gain = float(new_size - old_size)
        return self._adjacency_base + self._adjacency_gain_scale * gain

    def _largest_cluster_for_player(self, state, player, board_override=None):
        """
        BFS to find the largest connected cluster of squares owned by 'player'.
        If board_override is given, use it instead of state's board_ownership.
        """
        board = board_override if board_override is not None else state._board_ownership
        num_colors, num_ranks = board.shape
        visited = np.zeros((num_colors, num_ranks), dtype=bool)
        max_cluster = 0

        def neighbors(c, r):
            for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
                cc, rr = c+dc, r+dr
                if 0 <= cc < num_colors and 0 <= rr < num_ranks:
                    yield (cc, rr)

        for c_idx in range(num_colors):
            for r_idx in range(num_ranks):
                if board[c_idx, r_idx] == player and not visited[c_idx, r_idx]:
                    # BFS from here
                    size = 0
                    queue = collections.deque([(c_idx, r_idx)])
                    visited[c_idx, r_idx] = True
                    while queue:
                        c0, r0 = queue.popleft()
                        size += 1
                        for (c1, r1) in neighbors(c0, r0):
                            if not visited[c1, r1] and board[c1, r1] == player:
                                visited[c1, r1] = True
                                queue.append((c1, r1))
                    max_cluster = max(max_cluster, size)
        return max_cluster


    def _adjacency_biased_uniform(self, state, player, actions, portion):
        """
        If you just want to spread 'portion' of probability among 'actions'
        in proportion to adjacency weighting, ignoring suit logic.
        """
        distribution = np.zeros(len(actions), dtype=float)
        weights = []
        for a in actions:
            w = self._adjacency_weight(state, player, a)
            weights.append(max(w, 0.0))

        total_w = sum(weights)
        if total_w < 1e-12:
            # fallback uniform
            for i in range(len(actions)):
                distribution[i] = portion / len(actions)
            return self._normalize(distribution)

        for i, a in enumerate(actions):
            distribution[i] = portion * (weights[i] / total_w)
        return self._normalize(distribution)


    # ----------------------------------------------------------------------
    # Utility: safe normalization
    # ----------------------------------------------------------------------
    def _normalize(self, distribution):
        total = np.sum(distribution)
        if total <= 1e-12:
            # fallback uniform
            n = len(distribution)
            return np.ones(n, dtype=float) / n
        return distribution / total


class TrickFollowingEvaluator(RandomRolloutEvaluator):
    """
    Uses a suit-following heuristic both for prior probabilities and for
    rollouts. If a suit is led and the player still has a token for that suit,
    they follow it with high probability. Otherwise they might deviate to trump
    or another suit.
    """
    
    def _normalize(self, distribution):
        """Helper to ensure probability distributions sum to 1."""
        s = np.sum(distribution)
        if s > 1e-12:
            return distribution / s
        return np.ones_like(distribution) / len(distribution)

    def __init__(
        self,
        n_rollouts=2,
        random_state=None,
        # Distribution parameters
        discard_frequent_prob=0.85,
        discard_infrequent_prob=0.15,
        pred_main_prob=0.70,
        pred_neighbor_prob=0.20,
        pred_uniform_prob=0.10,
        follow_suit_prob=0.60,      # when you *still* have suit
        deviate_prob=0.40,          # 1 - follow_suit_prob
        deviate_trump_ratio=0.75,   # portion of deviate-prob for trump
        deviate_other_ratio=0.25,   # portion of deviate-prob for off-color
    ):
        super().__init__(n_rollouts=n_rollouts, random_state=random_state)
        self._discard_frequent_prob = discard_frequent_prob
        self._discard_infrequent_prob = discard_infrequent_prob
        self._pred_main_prob = pred_main_prob
        self._pred_neighbor_prob = pred_neighbor_prob
        self._pred_uniform_prob = pred_uniform_prob
        self._follow_suit_prob = follow_suit_prob
        self._deviate_prob = deviate_prob
        self._deviate_trump_ratio = deviate_trump_ratio
        self._deviate_other_ratio = deviate_other_ratio

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
                    distribution.append(self._discard_frequent_prob / len(most_ranks))
            else:
                # If there *are* others
                if others_count > 0:
                    distribution.append(self._discard_infrequent_prob / others_count)
                else:
                    distribution.append(0.0)

        # Normalize the distribution
        return self._normalize(np.array(distribution))

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

        # (A) Main probability on 'guess'
        guess_action = 100 + guess  # e.g., guess=2 => 102
        if guess_action in legal_actions:
            i_guess = legal_actions.index(guess_action)
            distribution[i_guess] += self._pred_main_prob

        # (B) Neighbor probability on ±1 if valid
        near_candidates = []
        if guess > 1: near_candidates.append(guess - 1)
        if guess < 4: near_candidates.append(guess + 1)
        if near_candidates:
            share = self._pred_neighbor_prob / len(near_candidates)
            for c in near_candidates:
                a_val = 100 + c
                if a_val in legal_actions:
                    i_near = legal_actions.index(a_val)
                    distribution[i_near] += share

        # (C) Uniform portion among all legal predictions
        if len(legal_actions) > 0:
            each = self._pred_uniform_prob / len(legal_actions)
            for i in range(len(legal_actions)):
                distribution[i] += each

        # Normalize the distribution
        return self._normalize(np.array(distribution))

    def _compute_suit_following_distribution(self, state, legal_actions):
        """
        A helper that gives probabilities for each legal action, encouraging:
          - If you haven't used the led color at all, follow that color at 100% prob.
          - Once you've removed that color token, follow suit 60%/deviate 40%.
          - If you deviate, 75% trump vs. 25% other (but re-scale if no trump or no other).
        """

        led_color = state._led_color  # e.g. "R", "B", "Y", "G", or None if no lead
        current_player = state.current_player()
        color_map = {"R": 0, "B": 1, "Y": 2, "G": 3}

        # If nothing is led, fallback to uniform among legals.
        if led_color is None:
            return self._normalize(np.ones(len(legal_actions)))

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
                return self._normalize(np.ones(len(legal_actions)))

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

            return self._normalize(distribution)

        # Check if the player has already placed any token in the led color on the board
        has_used_led_color = np.any(state._board_ownership[led_idx] == current_player)
        # If the player has NOT used that color yet on the board => follow 100%
        if not has_used_led_color:
            distribution = np.zeros(len(legal_actions), dtype=float)
            # all legal follow actions share probability 1
            for i, a in enumerate(legal_actions):
                if a in follow_actions:
                    distribution[i] = 1.0 / len(follow_actions)
            return distribution

        # Else they have used that color before => follow ~60%, deviate ~40%
        distribution = np.zeros(len(legal_actions), dtype=float)
        f_count = len(follow_actions)
        t_count = len(trump_actions)
        o_count = len(other_actions)

        # Follow portion
        for i, a in enumerate(legal_actions):
            if a in follow_actions:
                distribution[i] = self._follow_suit_prob / f_count

        # Deviate portion => split between trump and other
        total_ratio = self._deviate_trump_ratio + self._deviate_other_ratio

        for i, a in enumerate(legal_actions):
            if t_count > 0 and o_count > 0:
                if a in trump_actions:
                    distribution[i] += (self._deviate_prob * self._deviate_trump_ratio / total_ratio) / t_count
                elif a in other_actions:
                    distribution[i] += (self._deviate_prob * self._deviate_other_ratio / total_ratio) / o_count
            elif t_count > 0:  # only trump
                if a in trump_actions:
                    distribution[i] += self._deviate_prob / t_count
            else:  # only others
                if a in other_actions:
                    distribution[i] += self._deviate_prob / o_count

        return self._normalize(distribution)


NUM_RANDOM_BOTS = 2

def main():
    game = pyspiel.load_game("python_quantum_cat", {"players": 1 + NUM_RANDOM_BOTS})

    # Create an ISMCTS bot for player 0
    # ismcts_evaluator = TrickFollowingEvaluator(
    #     n_rollouts=2,
    #     random_state=np.random.RandomState(42)
    # )
    ismcts_evaluator = RandomRolloutEvaluator(n_rollouts=2, random_state=np.random.RandomState(42))


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
