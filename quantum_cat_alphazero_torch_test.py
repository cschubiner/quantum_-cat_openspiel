#!/usr/bin/env python3
"""Tests for AlphaZero trainer helpers."""

import argparse
import json
import random
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
import torch

from quantum_cat_ai import HeuristicBot, shared_prediction_action
import quantum_cat_alphazero_torch as az_torch
import quantum_cat_action_paradox_policy_replay as risk_policy_replay
import quantum_cat_failure_miner as failure_miner
import quantum_cat_full_match_elo as full_elo
from quantum_cat_alphazero_torch import ACTION_FEATURE_ADJACENCY_GAIN_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_FOLLOWS_LED_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_IS_RED_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_RANK_NORM_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_SIZE
from quantum_cat_alphazero_torch import APPENDED_ACTION_FEATURE_INDEX
from quantum_cat_alphazero_torch import AZNet
from quantum_cat_alphazero_torch import TACTICAL_POLICY_BUCKET_NAMES
from quantum_cat_alphazero_torch import action_paradox_pairwise_ranking_loss
from quantum_cat_alphazero_torch import action_feature_matrix
from quantum_cat_alphazero_torch import apply_replay_metadata_to_args
from quantum_cat_alphazero_torch import blend_policy_with_counterfactual_paradox
from quantum_cat_alphazero_torch import blend_policy_with_counterfactual_values
from quantum_cat_alphazero_torch import configure_trainable_parameters
from quantum_cat_alphazero_torch import counterfactual_label_coverage_report
from quantum_cat_alphazero_torch import add_frozen_parameter_integrity_or_raise
from quantum_cat_alphazero_torch import frozen_parameter_integrity_report
from quantum_cat_alphazero_torch import frozen_parameter_snapshot
from quantum_cat_alphazero_torch import keep_teacher_policy_target
from quantum_cat_alphazero_torch import lane_capacity_pairwise_policy_loss
from quantum_cat_alphazero_torch import legal_policy_confidence
from quantum_cat_alphazero_torch import policy_target_bucket_masks
from quantum_cat_alphazero_torch import policy_target_bucket_weights
from quantum_cat_alphazero_torch import policy_target_pairwise_ranking_loss
from quantum_cat_alphazero_torch import sample_training_batch
from quantum_cat_alphazero_torch import maybe_write_eval_progress
from quantum_cat_alphazero_torch import maybe_write_teacher_progress
from quantum_cat_alphazero_torch import summarize_full_match_eval
from quantum_cat_alphazero_torch import terminal_search_value
from quantum_cat_alphazero_torch import value_targets_from_scores
from quantum_cat_alphazero_torch import weighted_masked_action_paradox_bce_loss
from quantum_cat_alphazero_torch import wrap_eval_result
from quantum_cat_alphazero_torch import wrap_teacher_result
from quantum_cat_alphazero_torch import write_json_artifact
from quantum_cat_full_match_elo import summarize_homogeneous_paradox_gate


def _example(labeled):
  mask = np.zeros(4, dtype=np.float32)
  if labeled:
    mask[:2] = 0.25
  return (
      np.zeros(8, dtype=np.float32),
      np.ones(4, dtype=np.float32),
      np.full(4, 0.25, dtype=np.float32),
      np.zeros(3, dtype=np.float32),
      np.zeros(3, dtype=np.float32),
      -1,
      -1,
      None,
      None,
      np.zeros(4, dtype=np.float32),
      mask,
  )


def _paradox_example(labeled):
  mask = np.zeros(4, dtype=np.float32)
  targets = np.zeros(4, dtype=np.float32)
  if labeled:
    mask[:2] = 1.0
    targets[:2] = [0.0, 1.0]
  return (
      np.zeros(8, dtype=np.float32),
      np.ones(4, dtype=np.float32),
      np.full(4, 0.25, dtype=np.float32),
      np.zeros(3, dtype=np.float32),
      np.zeros(3, dtype=np.float32),
      -1,
      -1,
      targets,
      mask,
      None,
      None,
  )


class _TerminalState:

  def __init__(self, returns, paradoxes):
    self._returns = returns
    self._has_paradoxed = paradoxes

  def returns(self):
    return self._returns


class _FeatureState:

  def __init__(self):
    self._phase = 3
    self._num_players = 3
    self._num_card_types = 3
    self._num_colors = 4
    self._num_tricks = 3
    self._cards_per_player_initial = 3
    self._hands = [
        np.array([2, 1, 0], dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
    ]
    self._board_ownership = -1 * np.ones((4, 3), dtype=np.int32)
    self._board_ownership[0, 0] = 1
    self._board_ownership[2, 0] = 2
    self._board_ownership[3, 0] = 1
    self._color_tokens = np.ones((3, 4), dtype=bool)
    self._led_color = None
    self._trump_broken = True
    self._cards_played_this_trick = [None, None, None]
    self._predictions = [1, 1, 1]
    self._tricks_won = np.zeros(3, dtype=np.int32)

  def legal_actions(self, player):
    del player
    return [3, 4]

  def num_distinct_actions(self):
    return 10


class _DiscardState:

  def __init__(self):
    self._phase = 1
    self._num_colors = 4
    self._num_card_types = 3
    self._hands = [
        np.array([1, 2, 1], dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
    ]
    self._board_ownership = -1 * np.ones((4, 3), dtype=np.int32)
    self._color_tokens = np.ones((3, 4), dtype=bool)

  def clone(self):
    clone = _DiscardState()
    clone._hands = [np.copy(hand) for hand in self._hands]
    clone._board_ownership = np.copy(self._board_ownership)
    clone._color_tokens = np.copy(self._color_tokens)
    return clone

  def apply_action(self, action):
    self._hands[0][action] -= 1

  def legal_actions(self, player):
    del player
    return [0, 1, 2]

  def num_distinct_actions(self):
    return 10


class _FeasibilityDiscardState:

  def __init__(self):
    self._phase = 1
    self._num_players = 3
    self._num_colors = 4
    self._num_card_types = 2
    self._hands = [
        np.array([2, 1], dtype=np.int32),
        np.zeros(2, dtype=np.int32),
        np.zeros(2, dtype=np.int32),
    ]
    self._board_ownership = -1 * np.ones((4, 2), dtype=np.int32)
    self._board_ownership[0, 0] = 1
    self._board_ownership[1, 0] = 2
    self._board_ownership[2, 0] = 1
    self._color_tokens = np.ones((3, 4), dtype=bool)

  def clone(self):
    clone = _FeasibilityDiscardState()
    clone._hands = [np.copy(hand) for hand in self._hands]
    clone._board_ownership = np.copy(self._board_ownership)
    clone._color_tokens = np.copy(self._color_tokens)
    return clone

  def apply_action(self, action):
    self._hands[0][int(action)] -= 1

  def legal_actions(self, player):
    del player
    return [0, 1]

  def num_distinct_actions(self):
    return 10


class _ExitLiquidityPlayState:

  def __init__(self):
    self._phase = 3
    self._num_players = 3
    self._num_colors = 4
    self._num_card_types = 3
    self._num_tricks = 3
    self._trick_number = 1
    self._current_player = 0
    self._led_color = "B"
    self._trump_broken = True
    self._hands = [
        np.array([0, 2, 0], dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
    ]
    self._board_ownership = -1 * np.ones((4, 3), dtype=np.int32)
    self._color_tokens = np.ones((3, 4), dtype=bool)
    self._cards_played_this_trick = [None, (1, "B"), (1, "B")]
    self._tricks_won = np.zeros(3, dtype=np.int32)
    self._predictions = [2, 1, 1]
    self._has_paradoxed = [False, False, False]
    self.follow_action = 1 * self._num_card_types + 1
    self.off_led_action = 3 * self._num_card_types + 1

  def clone(self):
    clone = _ExitLiquidityPlayState()
    clone._hands = [np.copy(hand) for hand in self._hands]
    clone._board_ownership = np.copy(self._board_ownership)
    clone._color_tokens = np.copy(self._color_tokens)
    clone._cards_played_this_trick = list(self._cards_played_this_trick)
    clone._tricks_won = np.copy(self._tricks_won)
    clone._predictions = list(self._predictions)
    clone._has_paradoxed = list(self._has_paradoxed)
    return clone

  def apply_action(self, action):
    color_names = ["R", "B", "Y", "G"]
    color_idx = int(action) // self._num_card_types
    rank_idx = int(action) % self._num_card_types
    color = color_names[color_idx]
    self._hands[0][rank_idx] -= 1
    self._board_ownership[color_idx, rank_idx] = 0
    self._cards_played_this_trick[0] = (rank_idx + 1, color)
    if self._led_color is not None and color != self._led_color:
      self._color_tokens[0, color_names.index(self._led_color)] = False

  def legal_actions(self, player):
    del player
    return [self.follow_action, self.off_led_action]

  def num_distinct_actions(self):
    return 20

  def _count_cards_played_by(self, player):
    return int(self._trick_number) + (
        1 if self._cards_played_this_trick[int(player)] is not None else 0
    )

  def observation_tensor(self, player):
    del player
    return np.zeros(32, dtype=np.float32)

  def _action_to_string(self, player, action):
    del player
    return f"Action {int(action)}"


class _PredictState:

  def __init__(self, hand=None, start_player=0):
    self._phase = 2
    self._hands = [
        np.array(
            [0, 0, 0, 0, 1, 1] if hand is None else hand,
            dtype=np.int32,
        ),
        np.zeros(6, dtype=np.int32),
        np.zeros(6, dtype=np.int32),
    ]
    self._num_card_types = 6
    self._round_start_player = start_player
    self._start_player = start_player

  def legal_actions(self, player):
    del player
    return [101, 102, 103, 104]

  def num_distinct_actions(self):
    return 1000


class _TokenLossPressureState:

  def __init__(self):
    self._phase = 3
    self._num_players = 3
    self._num_colors = 4
    self._num_card_types = 3
    self._num_tricks = 3
    self._hands = [
        np.array([1, 1, 0], dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
    ]
    self._board_ownership = -1 * np.ones((4, 3), dtype=np.int32)
    self._board_ownership[0, 0] = 1
    self._board_ownership[2, 0] = 1
    self._board_ownership[3, 0] = 1
    self._color_tokens = np.ones((3, 4), dtype=bool)
    self._led_color = "B"
    self._trump_broken = True
    self._cards_played_this_trick = [None, None, None]
    self._predictions = [1, 1, 1]
    self._tricks_won = np.zeros(3, dtype=np.int32)

  def clone(self):
    clone = _TokenLossPressureState()
    clone._hands = [np.copy(hand) for hand in self._hands]
    clone._board_ownership = np.copy(self._board_ownership)
    clone._color_tokens = np.copy(self._color_tokens)
    clone._cards_played_this_trick = list(self._cards_played_this_trick)
    return clone

  def apply_action(self, action):
    color_idx = action // self._num_card_types
    rank_idx = action % self._num_card_types
    self._hands[0][rank_idx] -= 1
    self._board_ownership[color_idx, rank_idx] = 0
    color = ["R", "B", "Y", "G"][color_idx]
    if color != self._led_color:
      self._color_tokens[0][1] = False


class _RankUrgencyState:

  def __init__(self):
    self._phase = 3
    self._num_players = 3
    self._num_colors = 4
    self._num_card_types = 3
    self._num_tricks = 3
    self._hands = [
        np.array([1, 1, 0], dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
    ]
    self._board_ownership = -1 * np.ones((4, 3), dtype=np.int32)
    self._board_ownership[0, 0] = 1
    self._board_ownership[2, 0] = 1
    self._board_ownership[3, 0] = 1
    self._color_tokens = np.ones((3, 4), dtype=bool)
    self._led_color = None
    self._trump_broken = True
    self._cards_played_this_trick = [None, None, None]
    self._predictions = [1, 1, 1]
    self._tricks_won = np.zeros(3, dtype=np.int32)


class _FutureTokenLossFragilityState:

  def __init__(self):
    self._phase = 3
    self._num_players = 3
    self._num_colors = 4
    self._num_card_types = 3
    self._num_tricks = 3
    self._cards_per_player_initial = 3
    self._hands = [
        np.array([1, 1, 0], dtype=np.int32),
        np.zeros(3, dtype=np.int32),
        np.zeros(3, dtype=np.int32),
    ]
    self._board_ownership = np.zeros((4, 3), dtype=np.int32)
    self._board_ownership[1, 0] = -1
    self._board_ownership[3, 1] = -1
    self._color_tokens = np.ones((3, 4), dtype=bool)
    self._led_color = None
    self._trump_broken = True
    self._cards_played_this_trick = [None, None, None]
    self._predictions = [1, 1, 1]
    self._tricks_won = np.zeros(3, dtype=np.int32)

  def legal_actions(self, player):
    del player
    return [3, 10]

  def num_distinct_actions(self):
    return 12


class _RolloutParadoxState:

  def __init__(self, paradoxes):
    self._has_paradoxed = list(paradoxes)
    self.applied_actions = []

  def clone(self):
    return _RolloutParadoxState(self._has_paradoxed)

  def apply_action(self, action):
    self.applied_actions.append(action)


class _FixedActionBot:

  def step(self, state, player):
    del state, player
    return 99


class _SurvivalParadoxState:

  def __init__(
      self,
      paradox_after_by_first_action,
      first_action=None,
      continuation_plies=0,
      terminal=False,
  ):
    self._has_paradoxed = [False, False, False]
    self._phase = 3
    self._num_players = 3
    self._hands = [
        np.ones(4, dtype=np.int32),
        np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=np.int32),
    ]
    self._first_action = first_action
    self._continuation_plies = continuation_plies
    self._terminal = terminal
    self._paradox_after_by_first_action = dict(paradox_after_by_first_action)

  def clone(self):
    clone = _SurvivalParadoxState(
        self._paradox_after_by_first_action,
        first_action=self._first_action,
        continuation_plies=self._continuation_plies,
        terminal=self._terminal,
    )
    clone._has_paradoxed = list(self._has_paradoxed)
    return clone

  def apply_action(self, action):
    if self._first_action is None:
      self._first_action = int(action)
      if self._paradox_after_by_first_action.get(self._first_action) == 0:
        self._has_paradoxed[0] = True
      return
    self._continuation_plies += 1
    paradox_after = self._paradox_after_by_first_action.get(self._first_action)
    if (
        paradox_after is not None
        and self._continuation_plies >= paradox_after
    ):
      self._has_paradoxed[0] = True
    if self._continuation_plies >= 4:
      self._terminal = True

  def is_terminal(self):
    return self._terminal

  def is_chance_node(self):
    return False

  def current_player(self):
    return 0

  def legal_actions(self, player):
    del player
    return [99]

  def num_distinct_actions(self):
    return 100


class _RolloutPolicyState:

  def legal_actions(self, player):
    del player
    return [3]

  def clone(self):
    return self


class _QPolicyState:

  def __init__(self):
    self._phase = 3

  def legal_actions(self, player):
    del player
    return [1, 2]

  def num_distinct_actions(self):
    return 4


class _ShieldState:

  def __init__(self, legal=None):
    self._phase = 3
    self._legal = list(legal or [1, 2, 3])

  def clone(self):
    return self

  def legal_actions(self, player):
    del player
    return list(self._legal)

  def num_distinct_actions(self):
    return max(max(self._legal) + 1, 1000)


class _ShieldEarlyParadoxState:

  def __init__(self, paradoxed=False):
    self._phase = 3
    self._has_paradoxed = [False, bool(paradoxed), False]
    self.applied_actions = []

  def clone(self):
    return _ShieldEarlyParadoxState(self._has_paradoxed[1])

  def apply_action(self, action):
    self.applied_actions.append(int(action))
    self._has_paradoxed[1] = True

  def is_terminal(self):
    return False

  def is_chance_node(self):
    return False

  def current_player(self):
    return 0

  def legal_actions(self, player):
    del player
    return [1]


class _ActionBot:

  def __init__(self, action):
    self.action = int(action)

  def step(self, state, player):
    del state, player
    return self.action


class _RoundGame:

  def num_distinct_actions(self):
    return 4


class _MCTSRiskState:

  def __init__(self, terminal=False, applied_action=None):
    self._terminal = terminal
    self._applied_action = applied_action

  def clone(self):
    return _MCTSRiskState(self._terminal, self._applied_action)

  def is_terminal(self):
    return self._terminal

  def current_player(self):
    return 0

  def legal_actions(self, player):
    del player
    return [1, 2]

  def num_distinct_actions(self):
    return 4

  def apply_action(self, action):
    self._applied_action = action
    self._terminal = True

  def returns(self):
    return [0.0]


class QuantumCatAlphaZeroTorchTest(absltest.TestCase):

  def _legal_action_features(self, batch_size=2, num_actions=5):
    action_features = torch.zeros(
        (batch_size, num_actions, ACTION_FEATURE_SIZE), dtype=torch.float32
    )
    action_features[:, :, 0] = 1.0
    return action_features

  def test_labeled_batch_fraction_applies_to_ranking_only_action_value_loss(self):
    replay = [_example(True), _example(True)] + [
        _example(False) for _ in range(8)
    ]
    args = type("Args", (), {
        "action_value_labeled_batch_fraction": 1.0,
        "action_value_loss_weight": 0.0,
        "action_value_ranking_loss_weight": 1.0,
    })()

    batch = sample_training_batch(replay, batch_size=4, args=args)

    self.assertLen(batch, 4)
    self.assertTrue(all(float(np.sum(example[10])) > 0.0 for example in batch))

  def test_labeled_batch_fraction_applies_to_action_paradox_loss(self):
    replay = [_paradox_example(True), _paradox_example(True)] + [
        _paradox_example(False) for _ in range(8)
    ]
    args = type("Args", (), {
        "action_value_labeled_batch_fraction": 0.0,
        "action_value_loss_weight": 0.0,
        "action_value_ranking_loss_weight": 0.0,
        "action_paradox_labeled_batch_fraction": 1.0,
        "action_paradox_loss_weight": 1.0,
    })()

    batch = sample_training_batch(replay, batch_size=4, args=args)

    self.assertLen(batch, 4)
    self.assertTrue(all(float(np.sum(example[8])) > 0.0 for example in batch))

  def test_labeled_batch_fraction_applies_to_action_paradox_ranking_loss(self):
    replay = [_paradox_example(True), _paradox_example(True)] + [
        _paradox_example(False) for _ in range(8)
    ]
    args = type("Args", (), {
        "action_value_labeled_batch_fraction": 0.0,
        "action_value_loss_weight": 0.0,
        "action_value_ranking_loss_weight": 0.0,
        "action_paradox_labeled_batch_fraction": 1.0,
        "action_paradox_loss_weight": 0.0,
        "action_paradox_ranking_loss_weight": 1.0,
    })()

    batch = sample_training_batch(replay, batch_size=4, args=args)

    self.assertLen(batch, 4)
    self.assertTrue(all(float(np.sum(example[8])) > 0.0 for example in batch))

  def test_action_paradox_negative_weight_emphasizes_safe_labels(self):
    logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    unweighted = weighted_masked_action_paradox_bce_loss(
        logits, targets, mask
    )
    weighted = weighted_masked_action_paradox_bce_loss(
        logits, targets, mask, negative_weight=3.0
    )
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    expected = (3.0 * raw[0, 0] + raw[0, 1]) / 4.0

    self.assertGreater(float(weighted), float(unweighted))
    self.assertAlmostEqual(float(weighted), float(expected), places=6)

  def test_action_paradox_pairwise_ranking_prefers_safer_ordering(self):
    targets = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    good_logits = torch.tensor([[-2.0, 2.0, 1.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[2.0, -2.0, -1.0]], dtype=torch.float32)
    args = type("Args", (), {
        "action_paradox_ranking_min_diff": 1e-6,
        "action_paradox_ranking_target_scale": 1.0,
    })()

    good_loss = action_paradox_pairwise_ranking_loss(
        good_logits, targets, mask, args
    )
    bad_loss = action_paradox_pairwise_ranking_loss(
        bad_logits, targets, mask, args
    )
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([-4.0, -3.0]), torch.tensor([0.0, 0.0]), reduction="none"
        ).mean()
    )

    self.assertLess(float(good_loss), float(bad_loss))
    self.assertAlmostEqual(float(good_loss), float(expected), places=6)

  def test_terminal_action_paradox_fallback_can_target_any_player(self):
    paradox_t = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    player_t = torch.tensor([0, 1, 0], dtype=torch.long)
    batch_indices = torch.tensor([0, 1, 2], dtype=torch.long)

    acting = az_torch.selected_terminal_paradox_targets(
        paradox_t,
        player_t,
        batch_indices,
        argparse.Namespace(action_paradox_terminal_fallback_scope="acting"),
    )
    any_player = az_torch.selected_terminal_paradox_targets(
        paradox_t,
        player_t,
        batch_indices,
        argparse.Namespace(action_paradox_terminal_fallback_scope="any"),
    )

    np.testing.assert_array_equal(acting.numpy(), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(any_player.numpy(), np.array([1.0, 0.0, 1.0]))

  def test_terminal_action_paradox_fallback_prefers_round_sidecar(self):
    paradox_t = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    round_paradox_t = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    round_mask_t = torch.tensor([1.0, 1.0], dtype=torch.float32)
    player_t = torch.tensor([0, 0], dtype=torch.long)
    batch_indices = torch.tensor([0, 1], dtype=torch.long)

    any_player = az_torch.selected_terminal_paradox_targets(
        paradox_t,
        player_t,
        batch_indices,
        argparse.Namespace(action_paradox_terminal_fallback_scope="any"),
        terminal_action_paradox_t=round_paradox_t,
        terminal_action_paradox_mask_t=round_mask_t,
    )
    acting = az_torch.selected_terminal_paradox_targets(
        paradox_t,
        player_t,
        batch_indices,
        argparse.Namespace(action_paradox_terminal_fallback_scope="acting"),
        terminal_action_paradox_t=round_paradox_t,
        terminal_action_paradox_mask_t=round_mask_t,
    )

    np.testing.assert_array_equal(any_player.numpy(), np.array([0.0, 1.0]))
    np.testing.assert_array_equal(acting.numpy(), np.array([0.0, 0.0]))

  def test_replay_round_paradox_sidecar_round_trips_zero_labels(self):
    pre_terminal = [
        (
            np.zeros(8, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.full(4, 0.25, dtype=np.float32),
            None,
            1,
            0,
            None,
            None,
            None,
            None,
            np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32),
        ),
        (
            np.ones(8, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.full(4, 0.25, dtype=np.float32),
            None,
            2,
            1,
            None,
            None,
            None,
            None,
            np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32),
        ),
    ]
    replay = az_torch.with_terminal_targets(
        pre_terminal,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
        terminal_action_paradox_targets=[
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      path = f"{tmpdir}/replay.npz"
      az_torch.save_replay(replay, path)
      loaded = az_torch.load_replay(path)

    self.assertLen(loaded, 2)
    np.testing.assert_array_equal(
        loaded[0][12], np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        loaded[1][12], np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )

  def test_action_paradox_policy_replay_prefers_low_risk_actions(self):
    action_features = np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32)
    action_features[:, 4] = 1.0  # play phase one-hot
    example = (
        np.zeros(8, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        2,
        0,
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        None,
        None,
        action_features,
    )
    args = argparse.Namespace(
        min_labeled_actions=2,
        min_safe_actions=1,
        min_risky_actions=1,
        min_spread=0.9,
        min_top_margin=0.0,
        min_original_risk_improvement=0.0,
        safe_threshold=0.34,
        risky_threshold=0.67,
        temperature=0.05,
        risk_policy_weight=1.0,
        include_non_overrides=False,
        keep_action_paradox_labels=False,
        include_unselected_originals=False,
    )

    derived, summary = risk_policy_replay.derive_policy_replay([example], args)

    self.assertLen(derived, 1)
    self.assertEqual(summary["best_top_changed_rows"], 1)
    self.assertEqual(summary["derived_top_changed_rows"], 1)
    self.assertEqual(summary["derived_rows_by_phase"], {"play": 1})
    self.assertEqual(summary["derived_top_changed_rows_by_phase"], {"play": 1})
    self.assertEqual(int(np.argmax(derived[0][2])), 0)
    self.assertEqual(float(np.sum(derived[0][8])), 0.0)

  def test_action_paradox_policy_replay_can_keep_unselected_originals(self):
    action_features = np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32)
    action_features[:, 4] = 1.0  # play phase one-hot
    unlabeled = (
        np.zeros(8, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        1,
        0,
        None,
        None,
        None,
        None,
        action_features,
    )
    labeled = (
        np.zeros(8, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        2,
        0,
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        None,
        None,
        action_features,
    )
    args = argparse.Namespace(
        min_labeled_actions=2,
        min_safe_actions=1,
        min_risky_actions=1,
        min_spread=0.9,
        min_top_margin=0.0,
        min_original_risk_improvement=0.0,
        safe_threshold=0.34,
        risky_threshold=0.67,
        temperature=0.05,
        risk_policy_weight=1.0,
        include_non_overrides=False,
        keep_action_paradox_labels=False,
        include_unselected_originals=True,
    )

    derived, summary = risk_policy_replay.derive_policy_replay(
        [unlabeled, labeled], args
    )

    self.assertLen(derived, 2)
    self.assertEqual(summary["derived_rows"], 1)
    self.assertEqual(summary["included_original_rows"], 1)
    self.assertEqual(summary["output_rows"], 2)
    self.assertEqual(summary["derived_rows_by_phase"], {"play": 1})
    self.assertEqual(summary["included_original_rows_by_phase"], {"play": 1})
    self.assertEqual(int(np.argmax(derived[0][2])), 1)
    self.assertEqual(int(np.argmax(derived[1][2])), 0)

  def test_action_paradox_policy_replay_can_reference_anchor_top(self):
    action_features = np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32)
    action_features[:, 4] = 1.0  # play phase one-hot
    example = (
        np.zeros(8, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        0,
        0,
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        None,
        None,
        action_features,
    )
    args = argparse.Namespace(
        min_labeled_actions=2,
        min_safe_actions=1,
        min_risky_actions=1,
        min_spread=0.9,
        min_top_margin=0.0,
        min_original_risk_improvement=0.5,
        safe_threshold=0.34,
        risky_threshold=0.67,
        temperature=0.05,
        risk_policy_weight=1.0,
        include_non_overrides=False,
        keep_action_paradox_labels=False,
        include_unselected_originals=False,
        reference_policy="anchor",
        anchor_checkpoint="anchor.pt",
    )
    anchor_policies = [np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)]

    derived, summary = risk_policy_replay.derive_policy_replay(
        [example], args, anchor_policies=anchor_policies
    )

    self.assertLen(derived, 1)
    self.assertEqual(summary["reference_policy"], "anchor")
    self.assertEqual(summary["reference_top_changed_rows"], 1)
    self.assertEqual(summary["derived_reference_top_changed_rows"], 1)
    self.assertEqual(summary["mean_original_risk_improvement"], 1.0)
    self.assertEqual(int(np.argmax(derived[0][2])), 0)

  def test_action_features_expose_future_token_loss_fragility(self):
    state = _FutureTokenLossFragilityState()
    action = 10  # Play the G-2, leaving B-1 as the only remaining slot.

    features = az_torch.action_feature_vector(
        state, 0, action, state.legal_actions(0)
    )

    self.assertEqual(features.shape[0], ACTION_FEATURE_SIZE)
    self.assertAlmostEqual(
        features[
            APPENDED_ACTION_FEATURE_INDEX[
                "future_token_loss_max_rank_deficit_after"
            ]
        ],
        1.0,
    )
    self.assertAlmostEqual(
        features[
            APPENDED_ACTION_FEATURE_INDEX[
                "future_token_loss_no_exit_frac_after"
            ]
        ],
        0.25,
    )
    self.assertGreater(
        features[
            APPENDED_ACTION_FEATURE_INDEX[
                "future_token_loss_safe_flex_drop_after"
            ]
        ],
        0.0,
    )

  def test_action_features_expose_public_exit_liquidity_damage(self):
    state = _ExitLiquidityPlayState()

    features = action_feature_matrix(state, player=0, num_actions=20)
    follow = features[state.follow_action]
    off_led = features[state.off_led_action]

    public_damage_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_public_slot_damage"
    ]
    own_damage_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_own_public_slot_damage"
    ]
    min_slots_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_min_player_open_slots_after"
    ]
    legal_pct_idx = APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_exit_public_slot_damage"
    ]
    self.assertAlmostEqual(follow[public_damage_idx], 3.0 / 36.0)
    self.assertAlmostEqual(follow[own_damage_idx], 1.0 / 12.0)
    self.assertAlmostEqual(follow[min_slots_idx], 11.0 / 12.0)
    self.assertGreater(off_led[public_damage_idx], follow[public_damage_idx])
    self.assertGreater(off_led[own_damage_idx], follow[own_damage_idx])
    self.assertLess(off_led[min_slots_idx], follow[min_slots_idx])
    self.assertGreater(off_led[legal_pct_idx], follow[legal_pct_idx])

  def test_action_features_expose_public_lane_surplus_pressure(self):
    state = _ExitLiquidityPlayState()

    features = action_feature_matrix(state, player=0, num_actions=20)
    follow = features[state.follow_action]
    off_led = features[state.off_led_action]

    own_surplus_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_own_lane_surplus_after"
    ]
    min_surplus_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_min_player_lane_surplus_after"
    ]
    total_surplus_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_total_player_lane_surplus_after"
    ]
    surplus_damage_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_lane_surplus_damage"
    ]
    min_surplus_damage_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_min_lane_surplus_damage"
    ]
    pressure_idx = APPENDED_ACTION_FEATURE_INDEX[
        "exit_lane_pressure_player_count_after"
    ]
    legal_pct_min_idx = APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_exit_min_player_lane_surplus_after"
    ]
    legal_pct_damage_idx = APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_exit_lane_surplus_damage"
    ]

    self.assertAlmostEqual(follow[own_surplus_idx], 10.0 / 12.0)
    self.assertAlmostEqual(follow[min_surplus_idx], 10.0 / 12.0)
    self.assertAlmostEqual(follow[total_surplus_idx], 30.0 / 36.0)
    self.assertAlmostEqual(follow[surplus_damage_idx], 2.0 / 36.0)
    self.assertAlmostEqual(follow[min_surplus_damage_idx], 0.0)
    self.assertAlmostEqual(follow[pressure_idx], 0.0)
    self.assertAlmostEqual(off_led[own_surplus_idx], 7.0 / 12.0)
    self.assertAlmostEqual(off_led[min_surplus_idx], 7.0 / 12.0)
    self.assertAlmostEqual(off_led[total_surplus_idx], 27.0 / 36.0)
    self.assertAlmostEqual(off_led[surplus_damage_idx], 5.0 / 36.0)
    self.assertAlmostEqual(off_led[min_surplus_damage_idx], 3.0 / 12.0)
    self.assertAlmostEqual(off_led[pressure_idx], 0.0)
    self.assertGreater(follow[legal_pct_min_idx], off_led[legal_pct_min_idx])
    self.assertGreater(off_led[legal_pct_damage_idx], follow[legal_pct_damage_idx])

  def test_action_features_expose_own_future_hand_feasibility(self):
    state = _ExitLiquidityPlayState()
    state._board_ownership[0, 1] = 2
    state._board_ownership[2, 1] = 1

    features = action_feature_matrix(state, player=0, num_actions=20)
    follow = features[state.follow_action]
    off_led = features[state.off_led_action]

    min_colors_idx = APPENDED_ACTION_FEATURE_INDEX[
        "own_future_min_colors_after"
    ]
    zero_exit_idx = APPENDED_ACTION_FEATURE_INDEX[
        "own_future_zero_exit_frac_after"
    ]
    one_exit_idx = APPENDED_ACTION_FEATURE_INDEX[
        "own_future_one_exit_frac_after"
    ]
    legal_pct_min_idx = APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_own_future_min_colors_after"
    ]
    legal_pct_zero_idx = APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_own_future_zero_exit_frac_after"
    ]

    self.assertAlmostEqual(follow[min_colors_idx], 1.0 / 4.0)
    self.assertAlmostEqual(follow[zero_exit_idx], 0.0)
    self.assertAlmostEqual(follow[one_exit_idx], 1.0)
    self.assertAlmostEqual(off_led[min_colors_idx], 0.0)
    self.assertAlmostEqual(off_led[zero_exit_idx], 1.0)
    self.assertAlmostEqual(off_led[one_exit_idx], 1.0)
    self.assertGreater(follow[legal_pct_min_idx], off_led[legal_pct_min_idx])
    self.assertGreater(off_led[legal_pct_zero_idx], follow[legal_pct_zero_idx])

  def test_adapt_action_features_pads_public_exit_liquidity_columns(self):
    old_width = APPENDED_ACTION_FEATURE_INDEX["exit_public_slot_damage"]
    old_features = np.ones((4, old_width), dtype=np.float32)

    adapted = az_torch.adapt_action_features(old_features, num_actions=4)

    self.assertEqual(adapted.shape, (4, ACTION_FEATURE_SIZE))
    self.assertEqual(
        adapted[0, APPENDED_ACTION_FEATURE_INDEX["exit_public_slot_damage"]],
        0.0,
    )
    self.assertEqual(adapted[0, old_width - 1], 1.0)

  def test_load_compatible_state_dict_pads_widened_action_encoder(self):
    model = AZNet(
        obs_size=8,
        num_actions=4,
        num_players=3,
        width=32,
        depth=1,
        arch="action_mlp",
    )
    current_weight = model.state_dict()["action_encoder.0.weight"]
    old_width = current_weight.shape[1] - 6
    saved_weight = torch.full(
        (current_weight.shape[0], old_width), 2.0, dtype=current_weight.dtype
    )

    az_torch.load_compatible_state_dict(
        model, {"action_encoder.0.weight": saved_weight}
    )

    loaded = model.state_dict()["action_encoder.0.weight"]
    self.assertTrue(torch.allclose(loaded[:, :old_width], saved_weight))
    self.assertTrue(torch.allclose(
        loaded[:, old_width:], torch.zeros_like(loaded[:, old_width:])
    ))

  def test_train_steps_adapts_loaded_replay_observation_width(self):
    model = AZNet(
        obs_size=4,
        num_actions=4,
        num_players=3,
        width=16,
        depth=1,
        arch="action_mlp",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args = argparse.Namespace(
        train_steps=1,
        batch_size=2,
        players=3,
        policy_loss_weight=1.0,
        value_loss_weight=0.0,
        value_loss_mode="all",
        paradox_loss_weight=0.0,
        action_paradox_loss_weight=0.0,
        action_paradox_ranking_loss_weight=0.0,
        action_value_loss_weight=0.0,
        action_value_ranking_loss_weight=0.0,
        action_value_labeled_batch_fraction=0.0,
        action_paradox_labeled_batch_fraction=0.0,
        prediction_hit_policy_loss_weight=0.0,
        future_hit_policy_loss_weight=0.0,
        dangerous_future_hit_policy_loss_weight=0.0,
        anchor_kl_weight=0.0,
        anchor_top_action_loss_weight=0.0,
        policy_target_action_type_weights="",
        policy_target_bucket_weights="",
    )

    loss = az_torch.train_steps(
        model,
        optimizer,
        [_example(False), _example(False)],
        args,
        torch.device("cpu"),
    )

    self.assertIsNotNone(loss)

  def test_led_token_loss_policy_regularizer_penalizes_lane_collapse_mass(self):
    model = AZNet(
        obs_size=8,
        num_actions=4,
        num_players=3,
        width=16,
        depth=1,
        arch="action_mlp",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    action_features = np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32)
    action_features[1, APPENDED_ACTION_FEATURE_INDEX[
        "token_loss_newly_loses_led"
    ]] = 1.0
    action_features[2, ACTION_FEATURE_FOLLOWS_LED_INDEX] = 1.0

    def row():
      return (
          np.zeros(8, dtype=np.float32),
          np.ones(4, dtype=np.float32),
          np.full(4, 0.25, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          -1,
          -1,
          None,
          None,
          np.zeros(4, dtype=np.float32),
          np.zeros(4, dtype=np.float32),
          action_features,
      )

    args = argparse.Namespace(
        train_steps=1,
        batch_size=2,
        players=3,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        value_loss_mode="all",
        paradox_loss_weight=0.0,
        action_paradox_loss_weight=0.0,
        action_paradox_ranking_loss_weight=0.0,
        action_value_loss_weight=0.0,
        action_value_ranking_loss_weight=0.0,
        action_value_labeled_batch_fraction=0.0,
        action_paradox_labeled_batch_fraction=0.0,
        prediction_hit_policy_loss_weight=0.0,
        future_hit_policy_loss_weight=0.0,
        led_token_loss_policy_loss_weight=1.0,
        led_token_loss_policy_max_mass=0.05,
        dangerous_future_hit_policy_loss_weight=0.0,
        anchor_kl_weight=0.0,
        anchor_top_action_loss_weight=0.0,
        policy_target_action_type_weights="",
        policy_target_bucket_weights="",
    )

    loss = az_torch.train_steps(
        model,
        optimizer,
        [row(), row()],
        args,
        torch.device("cpu"),
    )

    self.assertGreater(loss[11], 0.0)

  def test_lane_capacity_pairwise_policy_loss_prefers_follow_led_exits(self):
    features = torch.zeros((1, 3, ACTION_FEATURE_SIZE), dtype=torch.float32)
    features[0, 0, APPENDED_ACTION_FEATURE_INDEX[
        "token_loss_newly_loses_led"
    ]] = 1.0
    features[0, 0, APPENDED_ACTION_FEATURE_INDEX[
        "legal_z_exit_lane_surplus_damage"
    ]] = 1.0
    features[0, 1, ACTION_FEATURE_FOLLOWS_LED_INDEX] = 1.0
    features[0, 1, APPENDED_ACTION_FEATURE_INDEX[
        "legal_z_exit_min_player_lane_surplus_after"
    ]] = 1.0
    mask = torch.tensor([[True, True, True]])
    args = argparse.Namespace(
        lane_capacity_ranking_min_diff=0.25,
        lane_capacity_ranking_target_scale=2.0,
        lane_capacity_ranking_require_led_choice=True,
        lane_capacity_ranking_token_loss_penalty=3.0,
        lane_capacity_ranking_follow_led_bonus=1.0,
        lane_capacity_ranking_min_surplus_weight=1.0,
        lane_capacity_ranking_damage_penalty=0.75,
        lane_capacity_ranking_pressure_penalty=0.25,
    )

    good_logits = torch.tensor([[-2.0, 2.0, 0.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[2.0, -2.0, 0.0]], dtype=torch.float32)

    good_loss = lane_capacity_pairwise_policy_loss(
        good_logits, features, mask, args
    )
    bad_loss = lane_capacity_pairwise_policy_loss(
        bad_logits, features, mask, args
    )

    self.assertLess(float(good_loss), float(bad_loss))

  def test_policy_target_pairwise_ranking_loss_prefers_teacher_action(self):
    mask = torch.tensor([[True, True, True, True]])
    policy = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    args = argparse.Namespace(
        policy_target_ranking_min_target_prob=0.999,
        policy_target_ranking_margin=1.0,
        policy_target_ranking_max_negatives=0,
    )
    good_logits = torch.tensor([[-2.0, 0.0, 2.0, -1.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[-2.0, 2.0, -2.0, -1.0]], dtype=torch.float32)

    good_loss = policy_target_pairwise_ranking_loss(
        good_logits, policy, mask, args
    )
    bad_loss = policy_target_pairwise_ranking_loss(
        bad_logits, policy, mask, args
    )
    soft_policy = torch.tensor([[0.1, 0.1, 0.7, 0.1]], dtype=torch.float32)
    ignored_loss = policy_target_pairwise_ranking_loss(
        bad_logits, soft_policy, mask, args
    )

    self.assertLess(float(good_loss), float(bad_loss))
    self.assertEqual(float(ignored_loss), 0.0)

  def test_train_steps_reports_policy_target_pairwise_ranking_loss(self):
    model = AZNet(
        obs_size=8,
        num_actions=4,
        num_players=3,
        width=16,
        depth=1,
        arch="action_mlp",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    action_features = np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32)

    def row():
      return (
          np.zeros(8, dtype=np.float32),
          np.ones(4, dtype=np.float32),
          np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          -1,
          -1,
          None,
          None,
          np.zeros(4, dtype=np.float32),
          np.zeros(4, dtype=np.float32),
          action_features,
      )

    args = argparse.Namespace(
        train_steps=1,
        batch_size=2,
        players=3,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        value_loss_mode="all",
        paradox_loss_weight=0.0,
        action_paradox_loss_weight=0.0,
        action_paradox_ranking_loss_weight=0.0,
        action_value_loss_weight=0.0,
        action_value_ranking_loss_weight=0.0,
        action_value_labeled_batch_fraction=0.0,
        action_paradox_labeled_batch_fraction=0.0,
        prediction_hit_policy_loss_weight=0.0,
        future_hit_policy_loss_weight=0.0,
        led_token_loss_policy_loss_weight=0.0,
        dangerous_future_hit_policy_loss_weight=0.0,
        policy_target_ranking_loss_weight=1.0,
        policy_target_ranking_min_target_prob=0.999,
        policy_target_ranking_margin=1.0,
        policy_target_ranking_max_negatives=0,
        lane_capacity_ranking_policy_loss_weight=0.0,
        anchor_kl_weight=0.0,
        anchor_top_action_loss_weight=0.0,
        policy_target_action_type_weights="",
        policy_target_bucket_weights="",
    )

    loss = az_torch.train_steps(
        model,
        optimizer,
        [row(), row()],
        args,
        torch.device("cpu"),
    )

    self.assertGreater(loss[14], 0.0)
    self.assertEqual(loss[-1], 0.0)

  def test_train_steps_reports_lane_capacity_pairwise_policy_loss(self):
    model = AZNet(
        obs_size=8,
        num_actions=4,
        num_players=3,
        width=16,
        depth=1,
        arch="action_mlp",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    action_features = np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32)
    action_features[1, APPENDED_ACTION_FEATURE_INDEX[
        "token_loss_newly_loses_led"
    ]] = 1.0
    action_features[1, APPENDED_ACTION_FEATURE_INDEX[
        "legal_z_exit_lane_surplus_damage"
    ]] = 1.0
    action_features[2, ACTION_FEATURE_FOLLOWS_LED_INDEX] = 1.0
    action_features[2, APPENDED_ACTION_FEATURE_INDEX[
        "legal_z_exit_min_player_lane_surplus_after"
    ]] = 1.0

    def row():
      return (
          np.zeros(8, dtype=np.float32),
          np.ones(4, dtype=np.float32),
          np.full(4, 0.25, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          -1,
          -1,
          None,
          None,
          np.zeros(4, dtype=np.float32),
          np.zeros(4, dtype=np.float32),
          action_features,
      )

    args = argparse.Namespace(
        train_steps=1,
        batch_size=2,
        players=3,
        policy_loss_weight=0.0,
        value_loss_weight=0.0,
        value_loss_mode="all",
        paradox_loss_weight=0.0,
        action_paradox_loss_weight=0.0,
        action_paradox_ranking_loss_weight=0.0,
        action_value_loss_weight=0.0,
        action_value_ranking_loss_weight=0.0,
        action_value_labeled_batch_fraction=0.0,
        action_paradox_labeled_batch_fraction=0.0,
        prediction_hit_policy_loss_weight=0.0,
        future_hit_policy_loss_weight=0.0,
        led_token_loss_policy_loss_weight=0.0,
        dangerous_future_hit_policy_loss_weight=0.0,
        lane_capacity_ranking_policy_loss_weight=1.0,
        lane_capacity_ranking_min_diff=0.25,
        lane_capacity_ranking_target_scale=2.0,
        lane_capacity_ranking_require_led_choice=True,
        lane_capacity_ranking_token_loss_penalty=3.0,
        lane_capacity_ranking_follow_led_bonus=1.0,
        lane_capacity_ranking_min_surplus_weight=1.0,
        lane_capacity_ranking_damage_penalty=0.75,
        lane_capacity_ranking_pressure_penalty=0.25,
        anchor_kl_weight=0.0,
        anchor_top_action_loss_weight=0.0,
        policy_target_action_type_weights="",
        policy_target_bucket_weights="",
    )

    loss = az_torch.train_steps(
        model,
        optimizer,
        [row(), row()],
        args,
        torch.device("cpu"),
    )

    self.assertGreater(loss[-1], 0.0)

  def test_action_features_expose_discard_setup_safety(self):
    state = _DiscardState()
    # Rank 1 has two cards but only one future color slot, so discarding one
    # copy relieves a rank deficit. Singleton ranks remain useful exits.
    state._board_ownership[0, 1] = 1
    state._board_ownership[1, 1] = 1
    state._board_ownership[2, 1] = 1

    features = action_feature_matrix(state, player=0, num_actions=10)
    duplicate_row = features[1]
    singleton_row = features[0]

    self.assertAlmostEqual(
        duplicate_row[
            APPENDED_ACTION_FEATURE_INDEX["discard_rank_open_slots_frac"]
        ],
        0.25,
    )
    self.assertAlmostEqual(
        duplicate_row[
            APPENDED_ACTION_FEATURE_INDEX["discard_rank_slot_surplus_before"]
        ],
        -0.25,
    )
    self.assertAlmostEqual(
        duplicate_row[
            APPENDED_ACTION_FEATURE_INDEX["discard_rank_slot_surplus_after"]
        ],
        0.0,
    )
    self.assertAlmostEqual(
        duplicate_row[
            APPENDED_ACTION_FEATURE_INDEX["discard_rank_deficit_relief"]
        ],
        0.25,
    )
    self.assertGreater(
        duplicate_row[
            APPENDED_ACTION_FEATURE_INDEX["discard_safe_flex_delta"]
        ],
        0.0,
    )
    self.assertEqual(
        duplicate_row[APPENDED_ACTION_FEATURE_INDEX["discard_from_duplicate"]],
        1.0,
    )
    self.assertEqual(
        duplicate_row[APPENDED_ACTION_FEATURE_INDEX["discard_removes_singleton"]],
        0.0,
    )
    self.assertEqual(
        singleton_row[APPENDED_ACTION_FEATURE_INDEX["discard_from_duplicate"]],
        0.0,
    )
    self.assertEqual(
        singleton_row[APPENDED_ACTION_FEATURE_INDEX["discard_removes_singleton"]],
        1.0,
    )
    self.assertGreater(
        duplicate_row[
            APPENDED_ACTION_FEATURE_INDEX[
                "legal_pct_discard_rank_deficit_relief"
            ]
        ],
        singleton_row[
            APPENDED_ACTION_FEATURE_INDEX[
                "legal_pct_discard_rank_deficit_relief"
            ]
        ],
    )

  def test_separate_action_value_stack_is_not_used_by_policy_logits(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
        separate_action_value_encoder=True,
    )
    obs = torch.randn((2, 8), dtype=torch.float32)
    action_features = self._legal_action_features()

    logits, _ = model(obs, action_features)
    logits.sum().backward()

    q_prefixes = (
        "action_value.",
        "action_value_encoder.",
        "action_value_state_action_projection.",
        "action_value_attention.",
    )
    for name, param in model.named_parameters():
      if name == "action_value_attention_gate" or name.startswith(q_prefixes):
        self.assertIsNone(param.grad, msg=name)

  def test_separate_action_value_stack_does_not_backprop_to_policy_stack(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
        separate_action_value_encoder=True,
    )
    obs = torch.randn((2, 8), dtype=torch.float32)
    action_features = self._legal_action_features()

    state_embedding = model.body(obs)
    model._action_values(state_embedding, action_features).sum().backward()

    policy_prefixes = (
        "action_encoder.",
        "state_action_projection.",
        "action_attention.",
        "policy.",
    )
    for name, param in model.named_parameters():
      if name == "action_attention_gate" or name.startswith(policy_prefixes):
        self.assertIsNone(param.grad, msg=name)

  def test_separate_action_paradox_stack_is_not_used_by_policy_logits(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
        separate_action_paradox_encoder=True,
    )
    obs = torch.randn((2, 8), dtype=torch.float32)
    action_features = self._legal_action_features()

    logits, _ = model(obs, action_features)
    logits.sum().backward()

    risk_prefixes = (
        "action_paradox.",
        "action_paradox_encoder.",
        "action_paradox_state_action_projection.",
        "action_paradox_attention.",
    )
    for name, param in model.named_parameters():
      if name == "action_paradox_attention_gate" or name.startswith(
          risk_prefixes
      ):
        self.assertIsNone(param.grad, msg=name)

  def test_separate_action_paradox_stack_does_not_backprop_to_policy_stack(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
        separate_action_paradox_encoder=True,
    )
    obs = torch.randn((2, 8), dtype=torch.float32)
    action_features = self._legal_action_features()

    state_embedding = model.body(obs)
    model._action_paradox_logits(state_embedding, action_features).sum().backward()

    policy_prefixes = (
        "action_encoder.",
        "state_action_projection.",
        "action_attention.",
        "policy.",
    )
    for name, param in model.named_parameters():
      if name == "action_attention_gate" or name.startswith(policy_prefixes):
        self.assertIsNone(param.grad, msg=name)

  def test_train_action_value_stack_only_selects_dedicated_q_stack(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
        separate_action_value_encoder=True,
    )
    args = argparse.Namespace(train_action_value_stack_only=True)

    trainable = configure_trainable_parameters(model, args)
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    self.assertNotEmpty(trainable)
    self.assertIn("action_value.0.weight", trainable_names)
    self.assertIn("action_value_encoder.0.weight", trainable_names)
    self.assertIn(
        "action_value_state_action_projection.0.weight", trainable_names
    )
    self.assertIn("action_value_attention.in_proj_weight", trainable_names)
    self.assertNotIn("action_encoder.0.weight", trainable_names)

  def test_train_action_paradox_stack_only_selects_dedicated_risk_stack(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
        separate_action_paradox_encoder=True,
    )
    args = argparse.Namespace(train_action_paradox_stack_only=True)

    trainable = configure_trainable_parameters(model, args)
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    self.assertNotEmpty(trainable)
    self.assertIn("action_paradox.0.weight", trainable_names)
    self.assertIn("action_paradox_encoder.0.weight", trainable_names)
    self.assertIn(
        "action_paradox_state_action_projection.0.weight", trainable_names
    )
    self.assertIn("action_paradox_attention.in_proj_weight", trainable_names)
    self.assertNotIn("action_encoder.0.weight", trainable_names)
    self.assertNotIn("policy.0.weight", trainable_names)

  def test_train_policy_action_stack_only_excludes_aux_heads(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
        separate_action_value_encoder=True,
        separate_action_paradox_encoder=True,
    )
    args = argparse.Namespace(train_policy_action_stack_only=True)

    trainable = configure_trainable_parameters(model, args)
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    self.assertNotEmpty(trainable)
    self.assertIn("action_encoder.0.weight", trainable_names)
    self.assertIn("state_action_projection.0.weight", trainable_names)
    self.assertIn("policy.0.weight", trainable_names)
    self.assertIn("action_attention.in_proj_weight", trainable_names)
    self.assertNotIn("action_paradox.0.weight", trainable_names)
    self.assertNotIn("action_paradox_encoder.0.weight", trainable_names)
    self.assertNotIn("action_value.0.weight", trainable_names)
    self.assertNotIn("action_value_encoder.0.weight", trainable_names)

  def test_frozen_parameter_integrity_rejects_policy_only_aux_drift(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
        separate_action_value_encoder=True,
        separate_action_paradox_encoder=True,
    )
    args = argparse.Namespace(train_policy_action_stack_only=True)
    configure_trainable_parameters(model, args)
    before = frozen_parameter_snapshot(model)

    with torch.no_grad():
      model.action_encoder[0].weight.add_(1.0)
    report = frozen_parameter_integrity_report(model, before)
    self.assertTrue(report["exact_match"])
    self.assertEqual(report["changed_parameter_count"], 0)

    with torch.no_grad():
      model.action_paradox[0].weight.add_(1.0)
    row = {}
    with self.assertRaisesRegex(RuntimeError, "changed frozen parameters"):
      add_frozen_parameter_integrity_or_raise(row, model, before, args)
    self.assertFalse(row["frozen_parameter_integrity"]["exact_match"])
    self.assertIn(
        "action_paradox.0.weight",
        row["frozen_parameter_integrity"]["changed_parameter_names"],
    )

  def test_train_value_head_only_selects_state_value_head(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_mlp",
    )
    args = argparse.Namespace(train_value_head_only=True)

    trainable = configure_trainable_parameters(model, args)
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    self.assertNotEmpty(trainable)
    self.assertIn("value.0.weight", trainable_names)
    self.assertIn("value.2.weight", trainable_names)
    self.assertNotIn("body.0.weight", trainable_names)
    self.assertNotIn("action_encoder.0.weight", trainable_names)
    self.assertNotIn("state_action_projection.0.weight", trainable_names)
    self.assertNotIn("policy.0.weight", trainable_names)

  def test_loaded_replay_best_metric_supports_value_validation(self):
    row = {
        "value_prediction_validation_report": {
            "acting_player": {
                "auc": 0.73,
                "brier": 0.18,
                "corr": 0.21,
            },
            "all_players": {
                "auc": 0.69,
                "brier": 0.22,
                "corr": 0.17,
            },
        }
    }

    self.assertEqual(
        az_torch.loaded_replay_validation_score(
            row,
            argparse.Namespace(
                loaded_replay_best_metric="value_validation_acting_auc"
            ),
        ),
        0.73,
    )
    self.assertEqual(
        az_torch.loaded_replay_validation_score(
            row,
            argparse.Namespace(
                loaded_replay_best_metric="value_validation_acting_brier"
            ),
        ),
        -0.18,
    )
    self.assertEqual(
        az_torch.loaded_replay_validation_score(
            row,
            argparse.Namespace(loaded_replay_best_metric="value_validation_all_corr"),
        ),
        0.17,
    )

  def test_eval_artifact_writer_creates_parent_and_round_trips_json(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      out_path = f"{tmpdir}/nested/eval.json"

      write_json_artifact(out_path, {"complete": True, "score": 3})

      with open(out_path, encoding="utf-8") as handle:
        payload = json.load(handle)
      self.assertEqual(payload, {"complete": True, "score": 3})

  def test_eval_progress_writer_respects_interval_and_wraps_result(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      out_path = f"{tmpdir}/eval_progress.json"
      args = argparse.Namespace(
          eval_output_json=out_path,
          eval_progress_interval=3,
          eval_checkpoint="checkpoint.pt",
          eval_opponent_checkpoint="opponent.pt",
          eval_games=10,
          eval_mcts_sims=32,
          eval_belief_samples=0,
          eval_belief_sims=16,
          eval_candidate="az_search",
          action_value_selection_weight=0.5,
      )

      maybe_write_eval_progress(args, {"ratings": {"az_search": 1000}}, 2)
      with self.assertRaises(FileNotFoundError):
        open(out_path, encoding="utf-8").close()

      maybe_write_eval_progress(args, {"ratings": {"az_search": 1001}}, 3)

      with open(out_path, encoding="utf-8") as handle:
        payload = json.load(handle)
      self.assertFalse(payload["complete"])
      self.assertEqual(payload["completed_games"], 3)
      self.assertEqual(payload["checkpoint"], "checkpoint.pt")
      self.assertEqual(payload["opponent_checkpoint"], "opponent.pt")
      self.assertEqual(payload["action_value_selection_weight"], 0.5)
      self.assertEqual(payload["eval"]["ratings"]["az_search"], 1001)

  def test_wrap_eval_result_marks_final_completion(self):
    args = argparse.Namespace(
        eval_checkpoint="checkpoint.pt",
        eval_opponent_checkpoint="opponent.pt",
        eval_games=10,
        eval_mcts_sims=32,
        eval_belief_samples=0,
        eval_belief_sims=16,
        eval_candidate="az_search",
        action_value_selection_weight=0.5,
    )

    wrapped = wrap_eval_result(
        args, {"ratings": {"az_search": 1002}}, complete=True,
        completed_games=10
    )

    self.assertTrue(wrapped["complete"])
    self.assertEqual(wrapped["completed_games"], 10)
    self.assertEqual(wrapped["opponent_checkpoint"], "opponent.pt")
    self.assertEqual(wrapped["eval"]["ratings"]["az_search"], 1002)

  def test_teacher_progress_writer_respects_interval_and_wraps_result(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      out_path = f"{tmpdir}/teacher_progress.json"
      args = argparse.Namespace(
          teacher_output_json=out_path,
          teacher_progress_interval=2,
          generate_replay_progress_interval=0,
          teacher_checkpoint="teacher.pt",
          teacher_games=5,
          teacher_mode="mcts",
          teacher_sims=8,
          teacher_temperature=0.35,
          action_value_selection_weight=0.5,
          action_paradox_selection_penalty=2.0,
          action_value_rerank_phases="prediction,play",
          full_match_training=True,
      )
      row = {
          "iteration": "teacher_replay_progress",
          "completed_games": 1,
          "total_games": 5,
          "examples": 20,
      }

      maybe_write_teacher_progress(args, row)
      with self.assertRaises(FileNotFoundError):
        open(out_path, encoding="utf-8").close()

      row["completed_games"] = 2
      row["examples"] = 40
      maybe_write_teacher_progress(args, row)

      with open(out_path, encoding="utf-8") as handle:
        payload = json.load(handle)
      self.assertFalse(payload["complete"])
      self.assertEqual(payload["completed_games"], 2)
      self.assertEqual(payload["teacher"]["examples"], 40)
      self.assertEqual(payload["teacher_checkpoint"], "teacher.pt")
      self.assertEqual(payload["action_value_selection_weight"], 0.5)
      self.assertEqual(payload["action_paradox_selection_penalty"], 2.0)
      self.assertEqual(payload["action_value_rerank_phases"], "prediction,play")

  def test_wrap_teacher_result_marks_final_completion(self):
    args = argparse.Namespace(
        teacher_checkpoint="teacher.pt",
        teacher_games=5,
        teacher_mode="mcts",
        teacher_sims=8,
        teacher_temperature=0.35,
        action_value_selection_weight=0.5,
        action_paradox_selection_penalty=2.0,
        action_value_rerank_phases="prediction,play",
        full_match_training=True,
    )

    wrapped = wrap_teacher_result(
        args, {"replay_size": 100}, complete=True, completed_games=5
    )

    self.assertTrue(wrapped["complete"])
    self.assertEqual(wrapped["completed_games"], 5)
    self.assertEqual(wrapped["teacher"]["replay_size"], 100)
    self.assertEqual(wrapped["action_paradox_selection_penalty"], 2.0)
    self.assertEqual(wrapped["action_value_rerank_phases"], "prediction,play")

  def test_q_policy_risk_only_rerank_can_override_policy(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=3.0,
        action_paradox_rerank_mode="additive",
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)
    risks = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    bot = full_elo.AZQPolicyBot(
        model=object(), name="q", device="cpu", model_args=args
    )

    with mock.patch.object(
        full_elo, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        full_elo, "model_action_risks", return_value=risks
    ), mock.patch.object(
        full_elo,
        "model_action_values",
        side_effect=AssertionError("value head should be unused"),
    ):
      action = bot.step(_ShieldState([1, 2, 3]), player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["rerank_considered"], 1)
    self.assertEqual(stats["rerank_risk_used"], 1)
    self.assertEqual(stats["rerank_overrides"], 1)

  def test_q_policy_threshold_rerank_can_override_without_penalty(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="threshold",
        action_paradox_risk_threshold=0.5,
        action_paradox_min_risk_margin=0.25,
        action_paradox_max_policy_log_gap=2.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)
    risks = np.array([0.0, 0.8, 0.1, 0.0], dtype=np.float32)
    bot = full_elo.AZQPolicyBot(
        model=object(), name="q", device="cpu", model_args=args
    )

    with mock.patch.object(
        full_elo, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        full_elo, "model_action_risks", return_value=risks
    ), mock.patch.object(
        full_elo,
        "model_action_values",
        side_effect=AssertionError("value head should be unused"),
    ):
      action = bot.step(_ShieldState([1, 2, 3]), player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["rerank_risk_used"], 1)
    self.assertEqual(stats["rerank_threshold_applied"], 1)
    self.assertEqual(stats["rerank_overrides"], 1)
    self.assertAlmostEqual(stats["rerank_baseline_risk_avg"], 0.8)
    self.assertAlmostEqual(stats["rerank_selected_risk_margin_avg"], 0.7)

  def test_q_policy_threshold_rerank_respects_policy_gap(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="threshold",
        action_paradox_risk_threshold=0.5,
        action_paradox_min_risk_margin=0.25,
        action_paradox_max_policy_log_gap=1.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.95, 0.05, 0.0], dtype=np.float32)
    risks = np.array([0.0, 0.8, 0.0, 0.0], dtype=np.float32)
    bot = full_elo.AZQPolicyBot(
        model=object(), name="q", device="cpu", model_args=args
    )

    with mock.patch.object(
        full_elo, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        full_elo, "model_action_risks", return_value=risks
    ), mock.patch.object(
        full_elo,
        "model_action_values",
        side_effect=AssertionError("value head should be unused"),
    ):
      action = bot.step(_QPolicyState(), player=0)

    self.assertEqual(action, 1)
    stats = bot.decision_stats()
    self.assertEqual(stats["rerank_threshold_blocked_policy_gap"], 1)
    self.assertEqual(stats["rerank_overrides"], 0)

  def test_q_policy_relative_rerank_chooses_best_action_in_safe_set(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="relative",
        action_paradox_risk_threshold=0.0,
        action_paradox_min_risk_margin=0.05,
        action_paradox_max_policy_log_gap=-1.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.6, 0.35, 0.05], dtype=np.float32)
    risks = np.array([0.0, 0.6, 0.2, 0.22], dtype=np.float32)
    bot = full_elo.AZQPolicyBot(
        model=object(), name="q", device="cpu", model_args=args
    )

    with mock.patch.object(
        full_elo, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        full_elo, "model_action_risks", return_value=risks
    ), mock.patch.object(
        full_elo,
        "model_action_values",
        side_effect=AssertionError("value head should be unused"),
    ):
      action = bot.step(_ShieldState([1, 2, 3]), player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["rerank_relative_applied"], 1)
    self.assertEqual(stats["rerank_relative_candidates"], 2)
    self.assertEqual(stats["rerank_overrides"], 1)

  def test_q_policy_relative_rerank_can_use_feasibility_bonus(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="relative",
        action_paradox_risk_threshold=0.0,
        action_paradox_min_risk_margin=0.05,
        action_paradox_max_policy_log_gap=-1.0,
        action_feasibility_selection_weight=1.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.zeros(1000, dtype=np.float32)
    policy[1] = 0.05
    policy[2] = 0.55
    policy[3] = 0.40
    risks = np.zeros(1000, dtype=np.float32)
    risks[1] = 0.80
    risks[2] = 0.20
    risks[3] = 0.22
    features = np.zeros((1000, ACTION_FEATURE_SIZE), dtype=np.float32)
    features[2, APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_own_future_min_colors_after"
    ]] = 0.1
    features[2, APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_own_future_zero_exit_frac_after"
    ]] = 0.9
    features[3, APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_own_future_min_colors_after"
    ]] = 1.0
    features[3, APPENDED_ACTION_FEATURE_INDEX[
        "legal_pct_own_future_zero_exit_frac_after"
    ]] = 0.0
    bot = full_elo.AZQPolicyBot(
        model=object(), name="q", device="cpu", model_args=args
    )

    with mock.patch.object(
        full_elo, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        full_elo, "model_action_risks", return_value=risks
    ), mock.patch.object(
        full_elo, "action_feature_matrix", return_value=features
    ), mock.patch.object(
        full_elo,
        "model_action_values",
        side_effect=AssertionError("value head should be unused"),
    ):
      action = bot.step(_ShieldState([1, 2, 3]), player=0)

    self.assertEqual(action, 3)
    stats = bot.decision_stats()
    self.assertEqual(stats["rerank_relative_applied"], 1)
    self.assertEqual(stats["rerank_relative_candidates"], 2)
    self.assertEqual(stats["rerank_overrides"], 1)

  def test_q_policy_teacher_policy_uses_threshold_reranker(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="threshold",
        action_paradox_risk_threshold=0.5,
        action_paradox_min_risk_margin=0.25,
        action_paradox_max_policy_log_gap=2.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)
    risks = np.array([0.0, 0.8, 0.1, 0.0], dtype=np.float32)

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        az_torch, "model_action_risks", return_value=risks
    ), mock.patch.object(
        az_torch,
        "model_action_values",
        side_effect=AssertionError("value head should be unused"),
    ):
      target = az_torch.q_policy_rerank_policy(
          _QPolicyState(), player=0, model=object(), args=args, device="cpu"
      )

    self.assertEqual(float(target[2]), 1.0)
    self.assertEqual(float(target[1]), 0.0)

  def test_learner_policy_q_policy_mode_uses_threshold_reranker(self):
    args = argparse.Namespace(
        self_play_policy_mode="q_policy",
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="threshold",
        action_paradox_risk_threshold=0.5,
        action_paradox_min_risk_margin=0.25,
        action_paradox_max_policy_log_gap=2.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)
    risks = np.array([0.0, 0.8, 0.1, 0.0], dtype=np.float32)

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        az_torch, "model_action_risks", return_value=risks
    ), mock.patch.object(
        az_torch,
        "model_action_values",
        side_effect=AssertionError("value head should be unused"),
    ):
      target = az_torch.learner_policy(
          _QPolicyState(), player=0, model=object(), args=args, device="cpu"
      )

    self.assertEqual(float(target[2]), 1.0)
    self.assertEqual(float(target[1]), 0.0)

  def test_mcts_action_risk_penalty_can_override_prior(self):
    args = argparse.Namespace(
        value_scale=1.0,
        c_puct=1.0,
        sims=8,
        paradox_value_penalty=0.0,
        action_paradox_selection_penalty=2.0,
        action_paradox_root_only=False,
        action_value_selection_weight=0.0,
        action_value_root_only=False,
    )
    priors = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32)
    risks = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(priors, np.array([0.0]))
    ), mock.patch.object(
        az_torch, "model_action_risks", return_value=risks
    ):
      policy = az_torch.mcts_policy(
          _MCTSRiskState(), object(), args, device="cpu", add_noise=False
      )

    self.assertGreater(policy[2], policy[1])

  def test_q_policy_teacher_policy_uses_risk_reranker(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=3.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)
    risks = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        az_torch, "model_action_risks", return_value=risks
    ), mock.patch.object(
        az_torch,
        "model_action_values",
        side_effect=AssertionError("value head should be unused"),
    ):
      target = az_torch.q_policy_rerank_policy(
          _QPolicyState(), player=0, model=object(), args=args, device="cpu"
      )

    np.testing.assert_allclose(target, [0.0, 0.0, 1.0, 0.0])

  def test_q_policy_teacher_policy_respects_phase_filter(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=3.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="prediction",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(policy, 0.0)
    ), mock.patch.object(
        az_torch,
        "model_action_risks",
        side_effect=AssertionError("phase-filtered reranker should be unused"),
    ):
      target = az_torch.q_policy_rerank_policy(
          _QPolicyState(), player=0, model=object(), args=args, device="cpu"
      )

    np.testing.assert_allclose(target, policy)

  def test_q_policy_teacher_rollout_filter_accepts_safer_override(self):
    args = argparse.Namespace(
        players=3,
        q_policy_teacher_confirm_rollouts=1,
        q_policy_teacher_confirm_min_paradox_improvement=1e-6,
        q_policy_teacher_confirm_min_score_margin=0.0,
        counterfactual_full_match_rollout=False,
        full_match_training=False,
    )
    raw_policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)
    q_policy = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def paradox_for_action(_state, _game, _player, action, *_args):
      return 1.0 if action == 1 else 0.0

    with mock.patch.object(
        az_torch, "counterfactual_belief_states", return_value=[_QPolicyState()]
    ), mock.patch.object(
        az_torch, "rollout_paradox_after_action", side_effect=paradox_for_action
    ), mock.patch.object(
        az_torch, "rollout_score_after_action", return_value=5.0
    ):
      stats = {}
      self.assertTrue(
          az_torch.q_policy_teacher_target_rollout_confirmed(
              _QPolicyState(),
              None,
              0,
              raw_policy,
              q_policy,
              object(),
              args,
              "cpu",
              stats,
          )
      )
    self.assertEqual(stats["q_policy_teacher_confirm_considered"], 1)
    self.assertEqual(stats["q_policy_teacher_confirm_evaluated"], 1)
    self.assertEqual(stats["q_policy_teacher_confirm_accepted"], 1)
    self.assertAlmostEqual(
        stats["q_policy_teacher_confirm_paradox_improvement_sum"], 1.0
    )

  def test_q_policy_teacher_rollout_filter_rejects_score_harm(self):
    args = argparse.Namespace(
        players=3,
        q_policy_teacher_confirm_rollouts=1,
        q_policy_teacher_confirm_min_paradox_improvement=1e-6,
        q_policy_teacher_confirm_min_score_margin=0.0,
        counterfactual_full_match_rollout=False,
        full_match_training=False,
    )
    raw_policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)
    q_policy = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def paradox_for_action(_state, _game, _player, action, *_args):
      return 1.0 if action == 1 else 0.0

    def score_for_action(_state, _game, _player, action, *_args):
      return 5.0 if action == 1 else 4.0

    with mock.patch.object(
        az_torch, "counterfactual_belief_states", return_value=[_QPolicyState()]
    ), mock.patch.object(
        az_torch, "rollout_paradox_after_action", side_effect=paradox_for_action
    ), mock.patch.object(
        az_torch, "rollout_score_after_action", side_effect=score_for_action
    ):
      stats = {}
      self.assertFalse(
          az_torch.q_policy_teacher_target_rollout_confirmed(
              _QPolicyState(),
              None,
              0,
              raw_policy,
              q_policy,
              object(),
              args,
              "cpu",
              stats,
          )
      )
    self.assertEqual(stats["q_policy_teacher_confirm_considered"], 1)
    self.assertEqual(stats["q_policy_teacher_confirm_evaluated"], 1)
    self.assertEqual(stats["q_policy_teacher_confirm_rejected_score"], 1)
    self.assertNotIn("q_policy_teacher_confirm_accepted", stats)
    self.assertAlmostEqual(stats["q_policy_teacher_confirm_score_margin_sum"], -1.0)

  def test_counterfactual_action_paradox_targets_pair_rollout_seeds(self):
    args = argparse.Namespace(
        counterfactual_action_rollouts=3,
        counterfactual_action_label_phases="",
        counterfactual_action_min_policy_entropy=0.0,
        counterfactual_action_max_policy_top_prob=1.0,
        counterfactual_action_max_legal=0,
    )
    draws = []

    def paradox_for_action(_state, _game, _player, action, *_args):
      draws.append((action, random.random(), float(np.random.random())))
      return 1.0 if action == 1 else 0.0

    random.seed(123)
    np.random.seed(456)
    with mock.patch.object(
        az_torch, "counterfactual_belief_states", return_value=[_QPolicyState()]
    ), mock.patch.object(
        az_torch, "rollout_paradox_after_action", side_effect=paradox_for_action
    ):
      targets, mask = az_torch.counterfactual_action_targets(
          _QPolicyState(),
          _QPolicyState(),
          0,
          [1, 2],
          ["learner"],
          {},
          object(),
          None,
          args,
          "cpu",
      )

    self.assertLen(draws, 6)
    for offset in range(0, len(draws), 2):
      self.assertEqual(draws[offset][0], 1)
      self.assertEqual(draws[offset + 1][0], 2)
      self.assertAlmostEqual(draws[offset][1], draws[offset + 1][1])
      self.assertAlmostEqual(draws[offset][2], draws[offset + 1][2])
    self.assertEqual(targets[1], 1.0)
    self.assertEqual(targets[2], 0.0)
    self.assertEqual(mask[1], 1.0)
    self.assertEqual(mask[2], 1.0)

  def test_aggregate_numeric_result_stats_combines_teacher_stats(self):
    results = [
        {
            "teacher_stats": {
                "q_policy_teacher_confirm_considered": 2,
                "q_policy_teacher_confirm_score_margin_sum": 0.25,
            }
        },
        {
            "teacher_stats": {
                "q_policy_teacher_confirm_considered": 3,
                "q_policy_teacher_confirm_score_margin_sum": -0.5,
            }
        },
    ]

    stats = az_torch.aggregate_numeric_result_stats(results, "teacher_stats")

    self.assertEqual(stats["q_policy_teacher_confirm_considered"], 5)
    self.assertAlmostEqual(
        stats["q_policy_teacher_confirm_score_margin_sum"], -0.25
    )

  def test_rollout_select_teacher_policy_accepts_safer_override(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        rollout_select_teacher_rollouts=1,
        rollout_select_teacher_min_actions=2,
        rollout_select_teacher_min_paradox_improvement=1e-6,
        rollout_select_teacher_min_score_margin=0.0,
        rollout_select_teacher_continuation_role="learner",
        rollout_select_teacher_keep_policy_best=False,
        counterfactual_full_match_rollout=False,
        full_match_training=False,
    )
    raw_policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)

    def paradox_for_action(_state, _game, _player, action, *_args):
      return 1.0 if action == 1 else 0.0

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(raw_policy, np.zeros(3))
    ), mock.patch.object(
        az_torch, "counterfactual_belief_states", return_value=[_QPolicyState()]
    ), mock.patch.object(
        az_torch, "rollout_paradox_after_action", side_effect=paradox_for_action
    ), mock.patch.object(
        az_torch, "rollout_score_after_action", return_value=5.0
    ):
      stats = {}
      policy, keep = az_torch.rollout_select_teacher_policy(
          _QPolicyState(), None, 0, object(), args, "cpu", stats
      )

    self.assertTrue(keep)
    np.testing.assert_allclose(policy, [0.0, 0.0, 1.0, 0.0])
    self.assertEqual(stats["rollout_select_teacher_considered"], 1)
    self.assertEqual(stats["rollout_select_teacher_evaluated"], 1)
    self.assertEqual(stats["rollout_select_teacher_accepted"], 1)
    self.assertEqual(stats["rollout_select_teacher_accepted_overrides"], 1)

  def test_rollout_select_teacher_policy_rejects_score_harm(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        rollout_select_teacher_rollouts=1,
        rollout_select_teacher_min_actions=2,
        rollout_select_teacher_min_paradox_improvement=1e-6,
        rollout_select_teacher_min_score_margin=0.0,
        rollout_select_teacher_continuation_role="learner",
        rollout_select_teacher_keep_policy_best=False,
        counterfactual_full_match_rollout=False,
        full_match_training=False,
    )
    raw_policy = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)

    def paradox_for_action(_state, _game, _player, action, *_args):
      return 1.0 if action == 1 else 0.0

    def score_for_action(_state, _game, _player, action, *_args):
      return 5.0 if action == 1 else 4.0

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(raw_policy, np.zeros(3))
    ), mock.patch.object(
        az_torch, "counterfactual_belief_states", return_value=[_QPolicyState()]
    ), mock.patch.object(
        az_torch, "rollout_paradox_after_action", side_effect=paradox_for_action
    ), mock.patch.object(
        az_torch, "rollout_score_after_action", side_effect=score_for_action
    ):
      stats = {}
      policy, keep = az_torch.rollout_select_teacher_policy(
          _QPolicyState(), None, 0, object(), args, "cpu", stats
      )

    self.assertFalse(keep)
    np.testing.assert_allclose(policy, raw_policy)
    self.assertEqual(stats["rollout_select_teacher_rejected_score"], 1)
    self.assertNotIn("rollout_select_teacher_accepted", stats)

  def test_teacher_policy_target_filter_uses_legal_confidence(self):
    policy = np.array([0.70, 0.20, 0.10, 0.00], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    confidence = legal_policy_confidence(policy, legal_mask)

    self.assertAlmostEqual(confidence["top_prob"], 0.7777778, places=6)
    self.assertAlmostEqual(confidence["top_margin"], 0.5555556, places=6)
    self.assertLess(confidence["normalized_entropy"], 1.0)

    args = argparse.Namespace(
        teacher_min_target_prob=0.75,
        teacher_min_target_margin=0.50,
        teacher_max_target_entropy=0.90,
    )
    self.assertTrue(keep_teacher_policy_target(policy, legal_mask, args))

    args.teacher_min_target_margin = 0.60
    self.assertFalse(keep_teacher_policy_target(policy, legal_mask, args))

  def test_full_match_summary_reports_score_uncertainty(self):
    result = summarize_full_match_eval(
        names=["az_search", "heuristic"],
        players=3,
        ratings={"az_search": 1010.0, "heuristic": 990.0},
        totals_by_name={
            "az_search": [4.0, 6.0, 8.0],
            "heuristic": [5.0],
        },
        paradoxes_by_name={"az_search": 1, "heuristic": 0},
    )

    self.assertEqual(result["avg_match_total"]["az_search"], 6.0)
    self.assertAlmostEqual(result["avg_match_total_se"]["az_search"], 1.1547)
    self.assertEqual(result["avg_match_total_ci95"]["az_search"], [3.7368, 8.2632])
    self.assertEqual(result["avg_match_total_se"]["heuristic"], 0.0)
    self.assertEqual(result["avg_match_total_ci95"]["heuristic"], [5.0, 5.0])
    self.assertEqual(result["paradoxes_per_match"]["az_search"], 0.3333)
    self.assertEqual(result["paradoxes_per_round"]["az_search"], 0.1111)
    self.assertEqual(result["paradox_round_rate"]["az_search"], 0.1111)

  def test_value_targets_can_penalize_any_terminal_paradox(self):
    args = argparse.Namespace(
        value_scale=10.0,
        terminal_paradox_penalty=2.0,
        terminal_any_paradox_penalty=5.0,
        ordinal_value_weight=0.0,
        official_outcome_value_weight=0.0,
    )

    targets = value_targets_from_scores(
        scores=[10.0, 8.0, 6.0],
        paradox_target=[0.0, 1.0, 0.0],
        args=args,
    )

    np.testing.assert_allclose(targets, [0.5, 0.1, 0.1])

  def test_terminal_search_value_uses_any_paradox_penalty(self):
    args = argparse.Namespace(
        value_scale=10.0,
        terminal_paradox_penalty=0.0,
        terminal_any_paradox_penalty=5.0,
        ordinal_value_weight=0.0,
        official_outcome_value_weight=0.0,
    )
    state = _TerminalState(
        returns=[10.0, 8.0, 6.0],
        paradoxes=[False, True, False],
    )

    np.testing.assert_allclose(
        terminal_search_value(state, args),
        [5.0, 3.0, 1.0],
    )

  def test_counterfactual_action_paradox_scope_can_target_any_player(self):
    state = _RolloutParadoxState([False, True, False])
    args = argparse.Namespace(
        players=3,
        counterfactual_action_paradox_scope="acting",
    )
    with mock.patch.object(az_torch, "play_rollout_to_terminal"):
      self.assertEqual(
          az_torch.rollout_paradox_after_action(
              state, None, 0, 7, [], {}, None, None, args, None
          ),
          0.0,
      )

      args.counterfactual_action_paradox_scope = "any"
      self.assertEqual(
          az_torch.rollout_paradox_after_action(
              state, None, 0, 7, [], {}, None, None, args, None
          ),
          1.0,
      )

  def test_counterfactual_paradox_rollout_respects_max_plies(self):
    state = _RolloutParadoxState([False, True, False])
    args = argparse.Namespace(
        players=3,
        counterfactual_action_paradox_scope="any",
        counterfactual_rollout_max_plies=7,
    )

    with mock.patch.object(az_torch, "play_rollout_to_terminal") as rollout:
      az_torch.rollout_paradox_after_action(
          state, None, 0, 7, [], {}, None, None, args, None
      )

    self.assertEqual(rollout.call_args.kwargs["max_decision_plies"], 7)

  def test_survival_counterfactual_paradox_labels_order_late_paradox_lower(self):
    state = _SurvivalParadoxState({1: 0, 2: 2})
    args = argparse.Namespace(
        players=3,
        counterfactual_action_paradox_scope="acting",
        counterfactual_action_paradox_target_mode="survival",
        counterfactual_action_paradox_survival_weight=0.5,
        counterfactual_rollout_max_plies=4,
    )
    seat_roles = ["fixed", "fixed", "fixed"]
    fixed_bots = {0: _FixedActionBot(), 1: _FixedActionBot(), 2: _FixedActionBot()}

    immediate = az_torch.rollout_paradox_after_action(
        state, None, 0, 1, seat_roles, fixed_bots, None, None, args, None
    )
    delayed = az_torch.rollout_paradox_after_action(
        state, None, 0, 2, seat_roles, fixed_bots, None, None, args, None
    )
    safe = az_torch.rollout_paradox_after_action(
        state, None, 0, 3, seat_roles, fixed_bots, None, None, args, None
    )

    self.assertEqual(immediate, 1.0)
    self.assertAlmostEqual(delayed, 0.75)
    self.assertEqual(safe, 0.0)
    self.assertGreater(immediate, delayed)
    self.assertGreater(delayed, safe)

  def test_counterfactual_action_value_survival_objective_uses_unit_scale(self):
    args = argparse.Namespace(
        players=3,
        value_scale=10.0,
        counterfactual_action_value_rollouts=1,
        counterfactual_action_value_objective="survival",
        counterfactual_action_label_phases="",
        counterfactual_action_min_policy_entropy=0.0,
        counterfactual_action_max_policy_top_prob=1.0,
        counterfactual_action_value_advantage=False,
        counterfactual_action_value_min_spread=0.0,
        counterfactual_action_value_audit_rollouts=0,
        counterfactual_action_value_max_stderr=0.0,
        counterfactual_action_value_confidence_weight=False,
    )

    def survival_for_action(*call_args, **unused_kwargs):
      action = int(call_args[3])
      return 1.0 if action == 2 else -1.0

    with mock.patch.object(
        az_torch, "counterfactual_belief_states", return_value=[_ShieldState([1, 2])]
    ), mock.patch.object(
        az_torch, "sampled_counterfactual_legal_actions", return_value=[1, 2]
    ), mock.patch.object(
        az_torch, "rollout_survival_value_after_action",
        side_effect=survival_for_action,
    ), mock.patch.object(
        az_torch,
        "rollout_score_after_action",
        side_effect=AssertionError("score objective should be unused"),
    ):
      targets, mask = az_torch.counterfactual_action_value_targets(
          _ShieldState([1, 2]),
          _RoundGame(),
          0,
          [1, 2],
          ["learner", "learner", "learner"],
          {},
          None,
          None,
          args,
          None,
          policy=np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32),
      )

    self.assertEqual(float(mask[1]), 1.0)
    self.assertEqual(float(mask[2]), 1.0)
    self.assertEqual(float(targets[1]), -1.0)
    self.assertEqual(float(targets[2]), 1.0)

  def test_counterfactual_sampler_reserves_led_lane_collapse_pair(self):
    args = argparse.Namespace(
        counterfactual_action_max_legal=3,
        counterfactual_action_top_policy=1,
        counterfactual_action_include_bots="",
        counterfactual_action_feature_candidates=True,
    )
    state = _FeatureState()
    legal = [0, 1, 2, 3, 4]
    policy = np.array([0.80, 0.05, 0.04, 0.03, 0.08], dtype=np.float32)
    features = np.zeros(
        (state.num_distinct_actions(), ACTION_FEATURE_SIZE), dtype=np.float32
    )
    for action in legal:
      features[action, 6 + 2] = 1.0
    features[2, ACTION_FEATURE_FOLLOWS_LED_INDEX] = 1.0
    features[2, APPENDED_ACTION_FEATURE_INDEX[
        "exit_min_player_lane_surplus_after"
    ]] = 0.75
    features[3, APPENDED_ACTION_FEATURE_INDEX[
        "token_loss_newly_loses_led"
    ]] = 1.0
    features[3, APPENDED_ACTION_FEATURE_INDEX[
        "exit_lane_surplus_damage"
    ]] = 0.50

    with mock.patch.object(
        az_torch, "action_feature_matrix", return_value=features
    ):
      sampled = az_torch.sampled_counterfactual_legal_actions(
          state, 0, legal, args, policy=policy
      )

    self.assertEqual(sampled, [0, 2, 3])

  def test_counterfactual_action_value_score_objective_keeps_value_scale(self):
    args = argparse.Namespace(
        players=3,
        value_scale=10.0,
        counterfactual_action_value_rollouts=1,
        counterfactual_action_value_objective="score",
        counterfactual_action_label_phases="",
        counterfactual_action_min_policy_entropy=0.0,
        counterfactual_action_max_policy_top_prob=1.0,
        counterfactual_action_value_advantage=False,
        counterfactual_action_value_min_spread=0.0,
        counterfactual_action_value_audit_rollouts=0,
        counterfactual_action_value_max_stderr=0.0,
        counterfactual_action_value_confidence_weight=False,
        counterfactual_full_match_rollout=False,
        full_match_training=False,
    )

    def score_for_action(*call_args, **unused_kwargs):
      action = int(call_args[3])
      return 5.0 if action == 2 else -3.0

    with mock.patch.object(
        az_torch, "counterfactual_belief_states", return_value=[_ShieldState([1, 2])]
    ), mock.patch.object(
        az_torch, "sampled_counterfactual_legal_actions", return_value=[1, 2]
    ), mock.patch.object(
        az_torch, "rollout_score_after_action", side_effect=score_for_action
    ), mock.patch.object(
        az_torch,
        "rollout_survival_value_after_action",
        side_effect=AssertionError("survival objective should be unused"),
    ):
      targets, mask = az_torch.counterfactual_action_value_targets(
          _ShieldState([1, 2]),
          _RoundGame(),
          0,
          [1, 2],
          ["learner", "learner", "learner"],
          {},
          None,
          None,
          args,
          None,
          policy=np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32),
      )

    self.assertEqual(float(mask[1]), 1.0)
    self.assertEqual(float(mask[2]), 1.0)
    self.assertAlmostEqual(float(targets[1]), -0.3)
    self.assertAlmostEqual(float(targets[2]), 0.5)

  def test_replay_metadata_can_restore_survival_action_value_objective(self):
    args = argparse.Namespace(counterfactual_action_value_objective="score")

    copied = apply_replay_metadata_to_args(
        args, {"counterfactual_action_value_objective": "survival"}
    )

    self.assertEqual(copied.counterfactual_action_value_objective, "survival")

  def test_counterfactual_action_value_report_kind_names_survival_objective(self):
    survival_args = argparse.Namespace(
        counterfactual_action_value_objective="survival",
        counterfactual_full_match_rollout=True,
    )
    full_match_args = argparse.Namespace(
        counterfactual_action_value_objective="score",
        counterfactual_full_match_rollout=True,
    )
    round_args = argparse.Namespace(
        counterfactual_action_value_objective="score",
        counterfactual_full_match_rollout=False,
    )

    self.assertEqual(
        az_torch.counterfactual_action_value_report_kind(survival_args),
        "counterfactual_round_survival",
    )
    self.assertEqual(
        az_torch.counterfactual_action_value_report_kind(full_match_args),
        "counterfactual_full_match_score",
    )
    self.assertEqual(
        az_torch.counterfactual_action_value_report_kind(round_args),
        "counterfactual_round_score",
    )

  def test_value_prediction_summary_reports_survival_calibration(self):
    report = az_torch.value_prediction_summary(
        preds=np.array([-0.8, -0.2, 0.4, 0.9], dtype=np.float32),
        targets=np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32),
        bucket_count=2,
    )

    self.assertEqual(report["count"], 4)
    self.assertEqual(report["positive_rate"], 0.5)
    self.assertEqual(report["auc"], 1.0)
    self.assertLess(report["brier"], 0.2)
    self.assertLen(report["reliability_buckets"], 2)

  def test_counterfactual_rollout_learner_bot_bypasses_neural_policy(self):
    args = argparse.Namespace(counterfactual_rollout_learner_bot="random")
    with mock.patch.object(
        az_torch, "learner_policy", side_effect=AssertionError("unused")
    ):
      action = az_torch.rollout_policy_action(
          _RolloutPolicyState(), None, 0, "learner", {}, None, None, args, None
      )

    self.assertEqual(action, 3)

  def test_homogeneous_paradox_gate_aggregates_bot_rounds(self):
    neural_bots = {
        name: {
            "checkpoint": "same.pt",
            "mode": "policy",
            "overrides": {},
        }
        for name in ("c0", "c1", "c2")
    }
    result = summarize_homogeneous_paradox_gate(
        names=["c0", "c1", "c2"],
        players=3,
        games_by_bot={"c0": 10, "c1": 10, "c2": 10},
        paradoxes_by_bot={"c0": 4, "c1": 3, "c2": 2},
        neural_bots=neural_bots,
        threshold=0.40,
    )

    self.assertTrue(result["eligible"])
    self.assertTrue(result["passed"])
    self.assertEqual(result["completed_matches"], 10)
    self.assertEqual(result["hand_rounds"], 30)
    self.assertEqual(result["rounds_with_any_paradox"], 9)
    self.assertEqual(result["bot_round_opportunities"], 90)
    self.assertEqual(result["total_paradoxes"], 9)
    self.assertEqual(result["same_policy_paradox_round_rate"], 0.3)
    self.assertEqual(result["any_paradox_round_rate"], 0.3)
    self.assertEqual(result["per_seat_paradox_round_rate"], 0.1)
    self.assertEqual(result["paradoxes_per_match"], 0.3)
    self.assertEqual(result["paradox_round_rate"], 0.1)
    self.assertEqual(result["bot_spec"]["kind"], "neural")
    self.assertEqual(result["bot_spec"]["checkpoint"], "same.pt")
    self.assertFalse(result["ci95_upper_below_threshold"])
    self.assertTrue(result["per_seat_ci95_upper_below_threshold"])

  def test_homogeneous_paradox_gate_counts_rounds_with_any_paradox(self):
    neural_bots = {
        name: {
            "checkpoint": "same.pt",
            "mode": "policy",
            "overrides": {},
        }
        for name in ("c0", "c1", "c2")
    }
    match_results = [
        (
            0,
            ["c0", "c1", "c2"],
            {
                "rounds": [
                    {"paradoxed": [True, True, False]},
                    {"paradoxed": [False, False, False]},
                    {"paradoxed": [False, True, False]},
                ]
            },
        ),
        (
            1,
            ["c0", "c1", "c2"],
            {
                "rounds": [
                    {"paradoxed": [False, False, True]},
                    {"paradoxed": [False, False, False]},
                    {"paradoxed": [False, False, False]},
                ]
            },
        ),
    ]

    result = summarize_homogeneous_paradox_gate(
        names=["c0", "c1", "c2"],
        players=3,
        games_by_bot={"c0": 2, "c1": 2, "c2": 2},
        paradoxes_by_bot={"c0": 1, "c1": 2, "c2": 1},
        neural_bots=neural_bots,
        threshold=0.40,
        match_results=match_results,
    )

    self.assertEqual(result["hand_rounds"], 6)
    self.assertEqual(result["rounds_with_any_paradox"], 3)
    self.assertEqual(result["total_paradoxes"], 4)
    self.assertEqual(result["same_policy_paradox_round_rate"], 0.5)
    self.assertEqual(result["any_paradox_round_rate"], 0.5)
    self.assertEqual(result["per_seat_paradox_round_rate"], 0.2222)

  def test_first_paradox_summary_groups_trigger_context(self):
    match_results = [(
        0,
        ["a", "b", "c"],
        {
            "rounds": [{
                "start_player": 0,
                "first_paradox_trace": {
                    "trigger": {
                        "phase": "play",
                        "player": 1,
                        "action": 999,
                        "action_kind": "paradox",
                        "legal_count": 1,
                        "trick_number": 7,
                        "led_color": "R",
                        "predictions": [1, 2, 1],
                        "tricks_won": [1, 4, 1],
                    },
                    "last_decisions": [{"action": 999}],
                },
            }]
        },
    )]

    summary = full_elo.summarize_first_paradox_traces(match_results)

    self.assertEqual(summary["rounds_with_trace"], 1)
    self.assertEqual(summary["trigger_by_phase"], {"play": 1})
    self.assertEqual(summary["trigger_by_action_kind"], {"paradox": 1})
    self.assertEqual(summary["trigger_by_legal_count"], {"1": 1})
    self.assertEqual(summary["forced_triggers"], 1)
    self.assertAlmostEqual(summary["prediction_gap_avg"], -2.0)
    self.assertLen(summary["samples"], 1)

  def test_homogeneous_paradox_gate_accepts_builtin_aliases(self):
    result = summarize_homogeneous_paradox_gate(
        names=["h0=heuristic", "h1=heuristic", "h2=heuristic"],
        players=3,
        games_by_bot={
            "h0=heuristic": 8,
            "h1=heuristic": 8,
            "h2=heuristic": 8,
        },
        paradoxes_by_bot={
            "h0=heuristic": 1,
            "h1=heuristic": 2,
            "h2=heuristic": 3,
        },
        neural_bots={},
        threshold=0.40,
    )

    self.assertTrue(result["eligible"])
    self.assertEqual(result["bot_spec"], {"kind": "builtin", "name": "heuristic"})
    self.assertEqual(result["completed_matches"], 8)
    self.assertEqual(result["hand_rounds"], 24)
    self.assertEqual(result["rounds_with_any_paradox"], 6)
    self.assertEqual(result["any_paradox_round_rate"], 0.25)

  def test_homogeneous_paradox_gate_rejects_mixed_specs(self):
    neural_bots = {
        "c0": {"checkpoint": "a.pt", "mode": "policy", "overrides": {}},
        "c1": {"checkpoint": "a.pt", "mode": "policy", "overrides": {}},
        "c2": {"checkpoint": "b.pt", "mode": "policy", "overrides": {}},
    }
    result = summarize_homogeneous_paradox_gate(
        names=["c0", "c1", "c2"],
        players=3,
        games_by_bot={"c0": 10, "c1": 10, "c2": 10},
        paradoxes_by_bot={"c0": 4, "c1": 3, "c2": 2},
        neural_bots=neural_bots,
        threshold=0.40,
    )

    self.assertFalse(result["eligible"])
    self.assertIsNone(result["passed"])
    self.assertEqual(result["reason"], "alias_specs_differ")

  def test_neural_bot_overrides_allow_url_escaped_commas(self):
    payload, overrides = full_elo._split_payload_options(
        "checkpoint.pt:q_policy?action_value_rerank_phases=prediction%2Cplay,"
        "action_paradox_selection_penalty=4.0,"
        "action_paradox_root_only=true,"
        "action_paradox_rerank_mode=threshold,"
        "action_paradox_risk_threshold=0.5,"
        "action_paradox_min_risk_margin=0.25,"
        "action_paradox_max_policy_log_gap=1.5"
    )

    self.assertEqual(payload, "checkpoint.pt:q_policy")
    self.assertEqual(overrides["action_value_rerank_phases"], "prediction,play")
    self.assertEqual(overrides["action_paradox_selection_penalty"], 4.0)
    self.assertTrue(overrides["action_paradox_root_only"])
    self.assertEqual(overrides["action_paradox_rerank_mode"], "threshold")
    self.assertEqual(overrides["action_paradox_risk_threshold"], 0.5)
    self.assertEqual(overrides["action_paradox_min_risk_margin"], 0.25)
    self.assertEqual(overrides["action_paradox_max_policy_log_gap"], 1.5)

  def test_neural_bot_overrides_allow_root_rollout_paradox_objective(self):
    payload, overrides = full_elo._split_payload_options(
        "checkpoint.pt:root_rollout?root_rollout_objective=paradox_then_score,"
        "root_rollout_paradox_scope=any,"
        "root_rollout_continuation_bot=heuristic_safe3,"
        "root_rollout_continuation_mode=liveness_shield,"
        "root_rollout_include_continuation_candidate=false"
    )

    self.assertEqual(payload, "checkpoint.pt:root_rollout")
    self.assertEqual(overrides["root_rollout_objective"], "paradox_then_score")
    self.assertEqual(overrides["root_rollout_paradox_scope"], "any")
    self.assertEqual(overrides["root_rollout_continuation_bot"], "heuristic_safe3")
    self.assertEqual(
        overrides["root_rollout_continuation_mode"], "liveness_shield"
    )
    self.assertFalse(overrides["root_rollout_include_continuation_candidate"])

  def test_play_graft_overrides_allow_builtin_prediction_graft(self):
    payload, overrides = full_elo._split_payload_options(
        "checkpoint.pt:play_graft?base_mode=q_policy,"
        "graft_builtin_bot=heuristic_safe14,"
        "graft_phases=prediction,"
        "action_value_rerank_phases=play"
    )

    self.assertEqual(payload, "checkpoint.pt:play_graft")
    self.assertEqual(overrides["base_mode"], "q_policy")
    self.assertEqual(overrides["graft_builtin_bot"], "heuristic_safe14")
    self.assertEqual(overrides["graft_phases"], "prediction")
    self.assertEqual(overrides["action_value_rerank_phases"], "play")

  def test_value_shield_overrides_allow_survival_checkpoint(self):
    payload, overrides = full_elo._split_payload_options(
        "policy.pt:value_shield?survival_checkpoint=survive.pt,"
        "survival_value_threshold=0.62,"
        "survival_value_phases=play,"
        "survival_value_max_actions=5,"
        "survival_value_scope=acting"
    )

    self.assertEqual(payload, "policy.pt:value_shield")
    self.assertEqual(overrides["survival_checkpoint"], "survive.pt")
    self.assertEqual(overrides["survival_value_threshold"], 0.62)
    self.assertEqual(overrides["survival_value_phases"], "play")
    self.assertEqual(overrides["survival_value_max_actions"], 5)
    self.assertEqual(overrides["survival_value_scope"], "acting")

  def test_homogeneous_spec_distinguishes_builtin_grafts(self):
    args = argparse.Namespace(base_mode="q_policy")
    entry = {
        "checkpoint": "same.pt",
        "mode": "play_graft",
        "graft_builtin_bot": "heuristic_safe14",
        "overrides": {"graft_builtin_bot": "heuristic_safe14"},
        "args": args,
    }

    spec = full_elo._homogeneous_neural_spec(entry)

    self.assertEqual(spec["kind"], "neural")
    self.assertEqual(spec["mode"], "play_graft")
    self.assertEqual(spec["graft_builtin_bot"], "heuristic_safe14")
    self.assertEqual(spec["base_mode"], "q_policy")

  def test_homogeneous_spec_includes_value_shield_survival_checkpoint(self):
    args = argparse.Namespace(base_mode="policy")
    entry = {
        "checkpoint": "policy.pt",
        "mode": "value_shield",
        "survival_checkpoint": "survive.pt",
        "overrides": {"survival_checkpoint": "survive.pt"},
        "args": args,
    }

    spec = full_elo._homogeneous_neural_spec(entry)

    self.assertEqual(spec["kind"], "neural")
    self.assertEqual(spec["mode"], "value_shield")
    self.assertEqual(spec["checkpoint"], "policy.pt")
    self.assertEqual(spec["survival_checkpoint"], "survive.pt")

  def test_root_rollout_paradox_objective_prefers_lower_paradox(self):
    policy = np.zeros(10, dtype=np.float32)
    policy[1] = 0.8
    policy[2] = 0.2
    scored = {
        1: {"paradox": 1.0, "score": 12.0},
        2: {"paradox": 0.0, "score": 1.0},
    }

    self.assertEqual(
        full_elo.AZRootRolloutBot._select_scored_action(
            scored, policy, "paradox_then_score"
        ),
        2,
    )
    self.assertEqual(
        full_elo.AZRootRolloutBot._select_scored_action(
            scored, policy, "score"
        ),
        1,
    )

  def test_root_rollout_paradox_scope_can_target_any_player(self):
    state = argparse.Namespace(_has_paradoxed=[False, True, False])
    args = argparse.Namespace(players=3, root_rollout_paradox_scope="acting")
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)

    self.assertEqual(bot._paradox_indicator(state, player=0), 0.0)
    args.root_rollout_paradox_scope = "any"
    self.assertEqual(bot._paradox_indicator(state, player=0), 1.0)

  def test_root_rollout_can_include_continuation_candidate(self):
    args = argparse.Namespace(
        players=3,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_max_actions=2,
        root_rollout_top_policy=1,
        root_rollout_include_bots="",
        root_rollout_include_continuation_candidate=True,
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    bot._continuation_neural_bot = object()
    bot._continuation_action = lambda state, player: 3
    state = argparse.Namespace(clone=lambda: state)
    legal = [1, 2, 3, 4]
    policy = np.zeros(10, dtype=np.float32)
    policy[1] = 1.0

    self.assertEqual(bot._candidate_actions(state, 0, legal, policy), [1, 3])

  def test_root_rollout_rejects_recursive_continuation_mode(self):
    args = argparse.Namespace(
        players=3,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="root_rollout",
    )

    with self.assertRaisesRegex(ValueError, "own continuation mode"):
      full_elo.AZRootRolloutBot(None, "root", None, args)

  def test_value_shield_prefers_policy_best_action_above_threshold(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        survival_value_threshold=0.65,
        survival_value_max_actions=0,
        survival_value_phases="",
        survival_value_scope="mean",
        survival_value_chance_depth=3,
        survival_value_max_chance_outcomes=32,
        survival_value_max_policy_log_gap=-1.0,
    )
    bot = full_elo.AZValueShieldPolicyBot(None, None, "shield", None, args)
    policy = np.zeros(1000, dtype=np.float32)
    policy[1] = 0.90
    policy[2] = 0.70
    policy[3] = 0.20

    with mock.patch.object(
        full_elo,
        "model_policy_value",
        return_value=(policy, np.zeros(3, dtype=np.float32)),
    ), mock.patch.object(
        bot,
        "_score_candidates",
        return_value={1: 0.40, 2: 0.70, 3: 0.80},
    ):
      action = bot.step(_ShieldState(), player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["value_shield_base_kept"], 0)
    self.assertEqual(stats["value_shield_overrides"], 1)
    self.assertEqual(stats["value_shield_fallback_max_survival"], 0)
    self.assertAlmostEqual(stats["value_shield_selected_survival_avg"], 0.7)

  def test_value_shield_falls_back_to_max_survival_when_none_safe(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        survival_value_threshold=0.95,
        survival_value_max_actions=0,
        survival_value_phases="",
        survival_value_scope="mean",
        survival_value_chance_depth=3,
        survival_value_max_chance_outcomes=32,
        survival_value_max_policy_log_gap=-1.0,
    )
    bot = full_elo.AZValueShieldPolicyBot(None, None, "shield", None, args)
    policy = np.zeros(1000, dtype=np.float32)
    policy[1] = 0.90
    policy[2] = 0.70
    policy[3] = 0.20

    with mock.patch.object(
        full_elo,
        "model_policy_value",
        return_value=(policy, np.zeros(3, dtype=np.float32)),
    ), mock.patch.object(
        bot,
        "_score_candidates",
        return_value={1: 0.40, 2: 0.70, 3: 0.80},
    ):
      action = bot.step(_ShieldState(), player=0)

    self.assertEqual(action, 3)
    stats = bot.decision_stats()
    self.assertEqual(stats["value_shield_fallback_max_survival"], 1)
    self.assertEqual(stats["value_shield_overrides"], 1)
    self.assertAlmostEqual(stats["value_shield_selected_survival_avg"], 0.8)

  def test_own_hand_feasibility_after_action_detects_rank_deficit(self):
    state = _FeasibilityDiscardState()

    bad = full_elo.own_hand_feasibility_after_action(state, 0, 1)
    good = full_elo.own_hand_feasibility_after_action(state, 0, 0)

    self.assertFalse(bad["feasible"])
    self.assertEqual(bad["total_deficit"], 1)
    self.assertTrue(good["feasible"])
    self.assertEqual(good["total_deficit"], 0)
    self.assertEqual(good["matching_size"], good["remaining_cards"])

  def test_feasibility_shield_overrides_base_that_creates_rank_deficit(self):
    args = argparse.Namespace(
        players=3,
        feasibility_shield_base_bot="heuristic_safe14",
        feasibility_shield_min_slot_surplus=0,
        feasibility_shield_phases="",
    )
    bot = full_elo.OwnHandFeasibilityShieldBot(
        "feasibility_shield", args, seed=0
    )
    bot.base_bot = _ActionBot(1)

    action = bot.step(_FeasibilityDiscardState(), player=0)

    self.assertEqual(action, 0)
    stats = bot.decision_stats()
    self.assertEqual(stats["feasibility_shield_base_feasible"], 0)
    self.assertEqual(stats["feasibility_shield_overrides"], 1)
    self.assertEqual(stats["feasibility_shield_legal_infeasible_count"], 1)
    self.assertAlmostEqual(stats["feasibility_shield_selected_deficit_avg"], 0.0)

  def test_feasibility_shield_keeps_feasible_base_action(self):
    args = argparse.Namespace(
        players=3,
        feasibility_shield_base_bot="heuristic_safe14",
        feasibility_shield_min_slot_surplus=0,
        feasibility_shield_phases="",
    )
    bot = full_elo.OwnHandFeasibilityShieldBot(
        "feasibility_shield", args, seed=0
    )
    bot.base_bot = _ActionBot(0)

    action = bot.step(_FeasibilityDiscardState(), player=0)

    self.assertEqual(action, 0)
    stats = bot.decision_stats()
    self.assertEqual(stats["feasibility_shield_base_feasible"], 1)
    self.assertEqual(stats["feasibility_shield_base_kept"], 1)
    self.assertEqual(stats["feasibility_shield_overrides"], 0)

  def test_public_exit_liquidity_counts_off_led_token_loss_damage(self):
    state = _ExitLiquidityPlayState()

    follow = full_elo.public_exit_liquidity_after_action(
        state, 0, state.follow_action
    )
    off_led = full_elo.public_exit_liquidity_after_action(
        state, 0, state.off_led_action
    )

    self.assertTrue(follow["own_feasible"])
    self.assertTrue(off_led["own_feasible"])
    self.assertFalse(follow["lost_led_token"])
    self.assertTrue(off_led["lost_led_token"])
    self.assertEqual(follow["public_slot_damage"], 3)
    self.assertGreater(off_led["public_slot_damage"], follow["public_slot_damage"])
    self.assertGreater(
        off_led["own_public_slot_damage"], follow["own_public_slot_damage"]
    )
    self.assertGreater(
        follow["total_player_open_slots_after"],
        off_led["total_player_open_slots_after"],
    )
    self.assertGreater(
        follow["own_public_slots_after"],
        off_led["own_public_slots_after"],
    )
    self.assertGreater(
        follow["min_player_lane_surplus_after"],
        off_led["min_player_lane_surplus_after"],
    )
    self.assertGreater(
        follow["own_lane_surplus_after"],
        off_led["own_lane_surplus_after"],
    )

  def test_liveness_certificate_ranks_exit_preserving_action_higher(self):
    state = _ExitLiquidityPlayState()

    follow = full_elo.action_liveness_certificate(
        state, 0, state.follow_action, base_action=state.off_led_action
    )
    off_led = full_elo.action_liveness_certificate(
        state, 0, state.off_led_action, base_action=state.off_led_action
    )

    self.assertEqual(follow["phase"], "play")
    self.assertEqual(follow["liveness_failure_count"], 0)
    self.assertEqual(off_led["liveness_failure_count"], 0)
    self.assertGreater(
        full_elo.action_liveness_key(
            follow, state.follow_action, base_action=state.off_led_action
        ),
        full_elo.action_liveness_key(
            off_led, state.off_led_action, base_action=state.off_led_action
        ),
    )

  def test_liveness_key_prioritizes_led_token_preservation(self):
    preserves_led = {
        "is_paradox": False,
        "own_feasible": True,
        "own_total_deficit": 0,
        "own_dead_rank_count": 0,
        "lost_led_token": False,
        "min_player_open_slots_after": 1,
        "total_player_open_slots_after": 4,
        "own_public_slots_after": 1,
        "min_player_rank_slots_after": 1,
        "own_min_rank_slots_after": 1,
        "over_target_would_win": False,
        "public_slot_damage": 3,
        "own_public_slot_damage": 1,
        "board_open_cell_damage": 3,
        "own_buffer_deficit": 0,
        "own_min_choices": 1,
        "own_slot_surplus": 0,
    }
    loses_led = dict(preserves_led)
    loses_led.update({
        "lost_led_token": True,
        "min_player_open_slots_after": 3,
        "total_player_open_slots_after": 9,
        "own_public_slots_after": 3,
        "public_slot_damage": 1,
        "own_public_slot_damage": 0,
        "board_open_cell_damage": 1,
    })

    self.assertGreater(
        full_elo.action_liveness_key(preserves_led, 10),
        full_elo.action_liveness_key(loses_led, 11),
    )

  def test_liveness_key_prioritizes_tight_lane_surplus(self):
    keeps_tight_lane_alive = {
        "is_paradox": False,
        "own_feasible": True,
        "own_total_deficit": 0,
        "own_dead_rank_count": 0,
        "lost_led_token": False,
        "min_player_lane_surplus_after": 2,
        "total_player_lane_surplus_after": 8,
        "own_lane_surplus_after": 2,
        "lane_pressure_player_count_after": 0,
        "min_player_open_slots_after": 4,
        "total_player_open_slots_after": 14,
        "own_public_slots_after": 4,
        "min_player_rank_slots_after": 1,
        "own_min_rank_slots_after": 1,
        "over_target_would_win": False,
        "public_slot_damage": 3,
        "own_public_slot_damage": 1,
        "board_open_cell_damage": 1,
        "own_buffer_deficit": 0,
        "own_min_choices": 1,
        "own_slot_surplus": 0,
    }
    more_slots_but_tighter_lane = dict(keeps_tight_lane_alive)
    more_slots_but_tighter_lane.update({
        "min_player_lane_surplus_after": 0,
        "total_player_lane_surplus_after": 12,
        "own_lane_surplus_after": 4,
        "min_player_open_slots_after": 6,
        "total_player_open_slots_after": 20,
        "own_public_slots_after": 6,
        "public_slot_damage": 1,
        "own_public_slot_damage": 0,
    })

    self.assertGreater(
        full_elo.action_liveness_key(keeps_tight_lane_alive, 10),
        full_elo.action_liveness_key(more_slots_but_tighter_lane, 11),
    )

  def test_failure_miner_liveness_policy_preserves_led_token(self):
    state = _ExitLiquidityPlayState()
    policy = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    policy[state.off_led_action] = 0.9
    policy[state.follow_action] = 0.1
    args = argparse.Namespace(
        liveness_shield_phases="play",
        liveness_shield_min_open_slot_delta=1,
        liveness_shield_min_public_damage_delta=1,
        liveness_shield_max_policy_log_gap=-1.0,
    )

    action = failure_miner.liveness_policy_action_from_policy(
        state, 0, state.legal_actions(0), policy, args
    )

    self.assertEqual(action, state.follow_action)

  def test_exit_liquidity_shield_overrides_high_damage_off_led_base_action(self):
    state = _ExitLiquidityPlayState()
    args = argparse.Namespace(
        players=3,
        exit_liquidity_base_bot="heuristic_safe14",
        exit_liquidity_phases="",
        exit_liquidity_min_trick_number=0,
        exit_liquidity_min_damage_delta=1.0,
        exit_liquidity_shadow_only=False,
    )
    bot = full_elo.ExitLiquidityShieldBot(
        "exit_liquidity_shield", args, seed=0
    )
    bot.base_bot = _ActionBot(state.off_led_action)

    action = bot.step(state, player=0)

    self.assertEqual(action, state.follow_action)
    stats = bot.decision_stats()
    self.assertEqual(stats["exit_liquidity_overrides"], 1)
    self.assertEqual(stats["exit_liquidity_base_kept"], 0)
    self.assertGreater(
        stats["exit_liquidity_base_public_slot_damage_avg"],
        stats["exit_liquidity_selected_public_slot_damage_avg"],
    )

  def test_exit_liquidity_shield_does_not_avoid_win_by_burning_exits(self):
    args = argparse.Namespace(
        players=3,
        exit_liquidity_base_bot="heuristic_safe14",
        exit_liquidity_phases="",
        exit_liquidity_min_trick_number=0,
        exit_liquidity_min_damage_delta=1.0,
        exit_liquidity_shadow_only=False,
    )
    bot = full_elo.ExitLiquidityShieldBot(
        "exit_liquidity_shield", args, seed=0
    )
    base_row = {
        "is_paradox": False,
        "own_total_deficit": 0,
        "over_target_would_win": True,
        "public_slot_damage": 3,
        "own_public_slot_damage": 1,
    }
    burns_exits = {
        "is_paradox": False,
        "own_total_deficit": 0,
        "over_target_would_win": False,
        "public_slot_damage": 4,
        "own_public_slot_damage": 2,
    }
    preserves_exits = {
        "is_paradox": False,
        "own_total_deficit": 0,
        "over_target_would_win": False,
        "public_slot_damage": 3,
        "own_public_slot_damage": 1,
    }

    self.assertFalse(bot._should_override(base_row, burns_exits))
    self.assertTrue(bot._should_override(base_row, preserves_exits))

  def test_liveness_shield_overrides_high_damage_neural_base_action(self):
    state = _ExitLiquidityPlayState()
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        liveness_shield_base_mode="policy",
        liveness_shield_phases="",
        liveness_shield_min_open_slot_delta=1,
        liveness_shield_min_public_damage_delta=1,
        liveness_shield_max_policy_log_gap=-1.0,
        liveness_shield_shadow_only=False,
        liveness_shield_sample_limit=20,
    )
    bot = full_elo.AZLivenessShieldPolicyBot(
        None, "liveness_shield", "cpu", args
    )
    bot.base_bot = _ActionBot(state.off_led_action)
    bot._policy = lambda state, player: np.ones(
        state.num_distinct_actions(), dtype=np.float32
    )

    action = bot.step(state, player=0)

    self.assertEqual(action, state.follow_action)
    stats = bot.decision_stats()
    self.assertEqual(stats["liveness_shield_overrides"], 1)
    self.assertEqual(stats["liveness_shield_base_kept"], 0)
    self.assertGreater(
        stats["liveness_shield_base_public_damage_avg"],
        stats["liveness_shield_selected_public_damage_avg"],
    )

  def test_liveness_shield_keeps_neural_base_without_dominance(self):
    state = _ExitLiquidityPlayState()
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        liveness_shield_base_mode="policy",
        liveness_shield_phases="",
        liveness_shield_min_open_slot_delta=1,
        liveness_shield_min_public_damage_delta=1,
        liveness_shield_max_policy_log_gap=-1.0,
        liveness_shield_shadow_only=False,
        liveness_shield_sample_limit=20,
    )
    bot = full_elo.AZLivenessShieldPolicyBot(
        None, "liveness_shield", "cpu", args
    )
    bot.base_bot = _ActionBot(state.follow_action)
    bot._policy = lambda state, player: np.ones(
        state.num_distinct_actions(), dtype=np.float32
    )

    action = bot.step(state, player=0)

    self.assertEqual(action, state.follow_action)
    stats = bot.decision_stats()
    self.assertEqual(stats["liveness_shield_base_kept"], 1)
    self.assertEqual(stats["liveness_shield_overrides"], 0)

  def test_residual_policy_zero_clip_keeps_anchor_action(self):
    state = _ShieldState(legal=[1, 2])
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        residual_policy_phases="",
        residual_delta_clip=0.0,
        residual_delta_scale=1.0,
    )
    bot = full_elo.AZResidualPolicyBot(None, None, "residual", "cpu", args)
    anchor = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    candidate = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    anchor[1], anchor[2] = 0.8, 0.2
    candidate[1], candidate[2] = 0.1, 0.9
    bot._policies = lambda state, player, num_actions: (anchor, candidate)

    action = bot.step(state, player=0)

    self.assertEqual(action, 1)
    stats = bot.decision_stats()
    self.assertEqual(stats["residual_overrides"], 0)

  def test_residual_policy_wide_clip_allows_candidate_override(self):
    state = _ShieldState(legal=[1, 2])
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        residual_policy_phases="",
        residual_delta_clip=10.0,
        residual_delta_scale=1.0,
    )
    bot = full_elo.AZResidualPolicyBot(None, None, "residual", "cpu", args)
    anchor = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    candidate = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    anchor[1], anchor[2] = 0.8, 0.2
    candidate[1], candidate[2] = 0.1, 0.9
    bot._policies = lambda state, player, num_actions: (anchor, candidate)

    action = bot.step(state, player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["residual_overrides"], 1)
    self.assertEqual(stats["residual_overrides_by_phase"]["play"], 1)

  def test_residual_q_policy_zero_clip_keeps_q_base_action(self):
    state = _ShieldState(legal=[1, 2])
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        residual_policy_phases="",
        residual_delta_clip=0.0,
        residual_delta_scale=1.0,
        residual_q_policy_base_margin=0.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="additive",
        action_feasibility_selection_weight=0.0,
        action_value_rerank_phases="",
    )
    bot = full_elo.AZResidualQPolicyBot(None, None, "residual_q", "cpu", args)
    anchor = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    candidate = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    anchor[1], anchor[2] = 0.8, 0.2
    candidate[1], candidate[2] = 0.1, 0.9
    bot._base_action_and_policy = lambda state, player, num_actions: (2, anchor)
    bot._candidate_policy = lambda state, player, num_actions: candidate

    action = bot.step(state, player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["residual_overrides"], 0)
    self.assertEqual(stats["residual_q_policy_base_over_raw"], 1)

  def test_residual_q_policy_wide_clip_allows_candidate_override(self):
    state = _ShieldState(legal=[1, 2])
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        residual_policy_phases="",
        residual_delta_clip=10.0,
        residual_delta_scale=1.0,
        residual_q_policy_base_margin=0.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="additive",
        action_feasibility_selection_weight=0.0,
        action_value_rerank_phases="",
    )
    bot = full_elo.AZResidualQPolicyBot(None, None, "residual_q", "cpu", args)
    anchor = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    candidate = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    anchor[1], anchor[2] = 0.8, 0.2
    candidate[1], candidate[2] = 0.1, 0.9
    bot._base_action_and_policy = lambda state, player, num_actions: (1, anchor)
    bot._candidate_policy = lambda state, player, num_actions: candidate

    action = bot.step(state, player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["residual_overrides"], 1)
    self.assertEqual(stats["residual_overrides_by_phase"]["play"], 1)

  def test_residual_q_risk_policy_uses_candidate_risk_over_q_base(self):
    state = _ShieldState(legal=[1, 2])
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        residual_policy_phases="",
        residual_delta_clip=0.0,
        residual_delta_scale=1.0,
        residual_q_policy_base_margin=0.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="relative",
        action_paradox_min_risk_margin=0.0,
        action_paradox_max_policy_log_gap=-1.0,
        action_feasibility_selection_weight=0.0,
        action_value_rerank_phases="",
        action_value_rerank_clip=0.5,
        action_value_rerank_min_margin=0.0,
    )
    bot = full_elo.AZResidualQRiskPolicyBot(
        None, None, "residual_q_risk", "cpu", args
    )
    anchor = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    risks = np.ones(state.num_distinct_actions(), dtype=np.float32)
    anchor[1], anchor[2] = 0.8, 0.2
    risks[1], risks[2] = 0.9, 0.1
    bot._base_action_and_policy = lambda state, player, num_actions: (1, anchor)
    bot._candidate_action_risks = lambda state, player, num_actions: risks

    action = bot.step(state, player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["residual_overrides"], 1)
    self.assertEqual(stats["rerank_relative_applied"], 1)

  def test_residual_q_risk_policy_keeps_q_base_on_equal_risk(self):
    state = _ShieldState(legal=[1, 2])
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        residual_policy_phases="",
        residual_delta_clip=0.0,
        residual_delta_scale=1.0,
        residual_q_policy_base_margin=0.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="relative",
        action_paradox_min_risk_margin=0.0,
        action_paradox_max_policy_log_gap=-1.0,
        action_feasibility_selection_weight=0.0,
        action_value_rerank_phases="",
        action_value_rerank_clip=0.5,
        action_value_rerank_min_margin=0.0,
    )
    bot = full_elo.AZResidualQRiskPolicyBot(
        None, None, "residual_q_risk", "cpu", args
    )
    anchor = np.zeros(state.num_distinct_actions(), dtype=np.float32)
    risks = np.ones(state.num_distinct_actions(), dtype=np.float32) * 0.5
    anchor[1], anchor[2] = 0.8, 0.2
    bot._base_action_and_policy = lambda state, player, num_actions: (1, anchor)
    bot._candidate_action_risks = lambda state, player, num_actions: risks

    action = bot.step(state, player=0)

    self.assertEqual(action, 1)
    stats = bot.decision_stats()
    self.assertEqual(stats["residual_overrides"], 0)

  def test_failure_miner_dense_row_prefers_safe_action_with_liveness_metadata(self):
    state = _ExitLiquidityPlayState()
    decision = {
        "state": state,
        "player": 0,
        "phase": "play",
        "trick_number": 1,
        "legal": [state.follow_action, state.off_led_action],
        "action": state.off_led_action,
        "action_string": "off led",
    }
    baseline = {
        "any_paradox": True,
        "forced_paradox": True,
        "first_new_paradox_trigger": {"phase": "play", "legal_count": 1},
        "paradoxed": [True, False, False],
        "raw_scores": [1.0, 0.0, 0.0],
    }
    alternatives = [{
        "action": state.follow_action,
        "action_string": "follow",
        "any_paradox": False,
        "actor_paradox": False,
        "forced_paradox": False,
        "first_new_paradox_trigger": None,
        "paradoxed": [False, False, False],
        "raw_scores": [-1.0, 0.0, 0.0],
        "acting_score_delta": -2.0,
        "total_score_delta": -2.0,
    }]
    args = argparse.Namespace(
        players=3,
        dense_target_temperature=0.05,
        dense_action_value_objective="safety",
    )

    row, metadata = failure_miner.dense_label_row_for_decision(
        decision, baseline, alternatives, args, value_scale=20.0
    )

    self.assertIsNotNone(row)
    policy_target = row[2]
    action_paradox_targets = row[7]
    action_paradox_mask = row[8]
    action_value_targets = row[9]
    action_value_mask = row[10]
    self.assertEqual(int(np.argmax(policy_target)), state.follow_action)
    self.assertGreater(policy_target[state.follow_action], 0.99)
    self.assertEqual(action_paradox_targets[state.off_led_action], 1.0)
    self.assertEqual(action_paradox_targets[state.follow_action], 0.0)
    self.assertEqual(action_paradox_mask[state.off_led_action], 1.0)
    self.assertEqual(action_paradox_mask[state.follow_action], 1.0)
    self.assertGreater(
        action_value_targets[state.follow_action],
        action_value_targets[state.off_led_action],
    )
    self.assertEqual(action_value_mask[state.follow_action], 1.0)
    self.assertTrue(metadata["policy_target_changed"])
    self.assertEqual(metadata["label_best_action"], state.follow_action)
    follow_meta = [
        item for item in metadata["action_outcomes"]
        if item["action"] == state.follow_action
    ][0]
    self.assertIn("liveness", follow_meta)
    self.assertEqual(follow_meta["liveness"]["phase"], "play")

  def test_failure_miner_dense_safety_values_are_bounded_for_tanh_head(self):
    args = argparse.Namespace(
        dense_action_value_objective="safety",
        dense_safety_value_safe_target=0.9,
        dense_safety_value_paradox_target=-0.55,
        dense_safety_value_forced_penalty=0.25,
        dense_safety_value_actor_penalty=0.10,
        dense_safety_value_score_weight=0.0,
        dense_safety_value_min=-0.95,
        dense_safety_value_max=0.95,
    )
    safe = {
        "any_paradox": False,
        "forced_paradox": False,
        "actor_paradox": False,
        "acting_score_delta": 0.0,
    }
    plain_paradox = {
        "any_paradox": True,
        "forced_paradox": False,
        "actor_paradox": False,
        "acting_score_delta": 0.0,
    }
    forced_actor_paradox = {
        "any_paradox": True,
        "forced_paradox": True,
        "actor_paradox": True,
        "acting_score_delta": 0.0,
    }

    safe_value = failure_miner.dense_action_value_target(
        safe, args, value_scale=20.0
    )
    plain_value = failure_miner.dense_action_value_target(
        plain_paradox, args, value_scale=20.0
    )
    forced_actor_value = failure_miner.dense_action_value_target(
        forced_actor_paradox, args, value_scale=20.0
    )

    self.assertEqual(safe_value, 0.9)
    self.assertEqual(plain_value, -0.55)
    self.assertEqual(forced_actor_value, -0.9)
    self.assertGreaterEqual(forced_actor_value, -0.95)
    self.assertGreater(safe_value, plain_value)
    self.assertGreater(plain_value, forced_actor_value)

  def test_failure_miner_dense_safety_raw_keeps_legacy_unbounded_values(self):
    args = argparse.Namespace(dense_action_value_objective="safety_raw")
    record = {
        "any_paradox": True,
        "forced_paradox": True,
        "actor_paradox": True,
        "acting_score_delta": 0.0,
    }

    value = failure_miner.dense_action_value_target(
        record, args, value_scale=20.0
    )

    self.assertEqual(value, -1.75)

  def test_survival_shield_keeps_base_action_above_threshold(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.45,
        survival_shield_rollouts=1,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)
    bot.base_bot = _ActionBot(1)

    with mock.patch.object(
        bot,
        "_score_candidates",
        return_value={1: 0.50, 2: 0.90, 3: 0.80},
    ):
      action = bot.step(_ShieldState(), player=0)

    self.assertEqual(action, 1)
    stats = bot.decision_stats()
    self.assertEqual(stats["survival_shield_base_kept"], 1)
    self.assertEqual(stats["survival_shield_overrides"], 0)
    self.assertAlmostEqual(stats["survival_shield_base_survival_avg"], 0.5)
    self.assertAlmostEqual(stats["survival_shield_selected_survival_avg"], 0.5)

  def test_survival_shield_falls_back_to_max_survival_when_all_below_threshold(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.95,
        survival_shield_rollouts=1,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)
    bot.base_bot = _ActionBot(1)

    with mock.patch.object(
        bot,
        "_score_candidates",
        return_value={1: 0.40, 2: 0.70, 3: 0.80},
    ):
      action = bot.step(_ShieldState(), player=0)

    self.assertEqual(action, 3)
    stats = bot.decision_stats()
    self.assertEqual(stats["survival_shield_base_kept"], 0)
    self.assertEqual(stats["survival_shield_fallback_max_survival"], 1)
    self.assertEqual(stats["survival_shield_overrides"], 1)
    self.assertAlmostEqual(stats["survival_shield_selected_survival_avg"], 0.8)

  def test_survival_shield_tie_breaks_away_from_paradox_action(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.95,
        survival_shield_rollouts=1,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)
    bot.base_bot = _ActionBot(999)

    with mock.patch.object(
        bot,
        "_score_candidates",
        return_value={1: 0.0, 999: 0.0},
    ):
      action = bot.step(_ShieldState(legal=[1, 999]), player=0)

    self.assertEqual(action, 1)
    stats = bot.decision_stats()
    self.assertEqual(stats["survival_shield_fallback_max_survival"], 1)
    self.assertEqual(stats["survival_shield_overrides"], 1)

  def test_survival_shield_filters_paradox_candidate_when_alternative_legal(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.45,
        survival_shield_score_mode="mean",
        survival_shield_lcb_z=1.96,
        survival_shield_rollouts=1,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_include_bots="",
        survival_shield_feature_candidates=False,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)

    self.assertEqual(
        bot._candidate_actions(
            _ShieldState(legal=[999, 3, 1]), 0, [999, 3, 1], base_action=999
        ),
        [1, 3],
    )
    stats = bot.raw_decision_stats()
    self.assertEqual(stats["survival_shield_paradox_candidate_filtered"], 1)

  def test_survival_shield_capped_candidates_include_heuristic_siblings(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.45,
        survival_shield_score_mode="mean",
        survival_shield_lcb_z=1.96,
        survival_shield_rollouts=1,
        survival_shield_samples=1,
        survival_shield_max_actions=3,
        survival_shield_include_bots="heuristic_safe8,heuristic_target2",
        survival_shield_feature_candidates=False,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)
    state = _ShieldState(legal=[1, 2, 4, 6, 8])

    with mock.patch.object(
        full_elo,
        "make_bot",
        side_effect=[_ActionBot(8), _ActionBot(6)],
    ):
      candidates = bot._candidate_actions(
          state, 0, [1, 2, 4, 6, 8], base_action=4
      )

    self.assertEqual(candidates, [4, 6, 8])
    self.assertEqual(
        bot.raw_decision_stats()["survival_shield_candidate_bot_additions"], 2
    )

  def test_survival_shield_capped_candidates_include_feature_extremes(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.45,
        survival_shield_score_mode="mean",
        survival_shield_lcb_z=1.96,
        survival_shield_rollouts=1,
        survival_shield_samples=1,
        survival_shield_max_actions=2,
        survival_shield_include_bots="",
        survival_shield_feature_candidates=True,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)
    state = _ShieldState(legal=[1, 2, 4, 6])
    feature_count = max(full_elo.APPENDED_ACTION_FEATURE_INDEX.values()) + 1
    features = np.zeros((1000, feature_count), dtype=np.float32)
    features[6, full_elo.APPENDED_ACTION_FEATURE_INDEX["hits_prediction"]] = 1.0

    with mock.patch.object(
        full_elo, "action_feature_matrix", return_value=features
    ):
      candidates = bot._candidate_actions(state, 0, [1, 2, 4, 6], base_action=4)

    self.assertEqual(candidates, [4, 6])
    self.assertEqual(
        bot.raw_decision_stats()["survival_shield_candidate_feature_additions"],
        1,
    )

  def test_survival_shield_wilson_lcb_scores_noisy_survival_conservatively(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.45,
        survival_shield_score_mode="wilson_lcb",
        survival_shield_lcb_z=1.96,
        survival_shield_rollouts=4,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_include_bots="",
        survival_shield_feature_candidates=False,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)

    score = bot._score_rollout_values([1.0, 1.0, 1.0, 1.0])

    self.assertAlmostEqual(score["mean"], 1.0)
    self.assertLess(score["wilson_lcb"], 0.6)
    self.assertAlmostEqual(score["score"], score["wilson_lcb"])

  def test_survival_shield_structured_lcb_score_controls_thresholding(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.75,
        survival_shield_score_mode="wilson_lcb",
        survival_shield_lcb_z=1.96,
        survival_shield_rollouts=4,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)
    bot.base_bot = _ActionBot(1)

    with mock.patch.object(
        bot,
        "_score_candidates",
        return_value={
            1: {"score": 0.50, "mean": 1.00, "wilson_lcb": 0.50},
            2: {"score": 0.60, "mean": 0.75, "wilson_lcb": 0.60},
        },
    ):
      action = bot.step(_ShieldState(legal=[1, 2]), player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["survival_shield_fallback_max_survival"], 1)
    self.assertEqual(stats["survival_shield_overrides"], 1)
    self.assertAlmostEqual(stats["survival_shield_base_survival_avg"], 0.5)
    self.assertAlmostEqual(stats["survival_shield_base_survival_mean_avg"], 1.0)
    self.assertAlmostEqual(stats["survival_shield_selected_survival_lcb_avg"], 0.6)

  def test_survival_shield_dominance_overrides_only_confidently(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.75,
        survival_shield_selection_mode="dominance",
        survival_shield_override_margin=0.05,
        survival_shield_override_mean_delta=0.20,
        survival_shield_score_mode="wilson_lcb",
        survival_shield_lcb_z=1.96,
        survival_shield_rollouts=4,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)
    bot.base_bot = _ActionBot(1)

    with mock.patch.object(
        bot,
        "_score_candidates",
        return_value={
            1: {"score": 0.20, "mean": 0.40, "wilson_lcb": 0.20, "wilson_ucb": 0.45},
            2: {"score": 0.55, "mean": 0.70, "wilson_lcb": 0.55, "wilson_ucb": 0.85},
        },
    ):
      action = bot.step(_ShieldState(legal=[1, 2]), player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["survival_shield_overrides"], 1)
    self.assertEqual(stats["survival_shield_base_kept"], 0)

  def test_survival_shield_dominance_keeps_base_without_clear_margin(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.75,
        survival_shield_selection_mode="dominance",
        survival_shield_override_margin=0.05,
        survival_shield_override_mean_delta=0.20,
        survival_shield_score_mode="wilson_lcb",
        survival_shield_lcb_z=1.96,
        survival_shield_rollouts=4,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)
    bot.base_bot = _ActionBot(1)

    with mock.patch.object(
        bot,
        "_score_candidates",
        return_value={
            1: {"score": 0.20, "mean": 0.40, "wilson_lcb": 0.20, "wilson_ucb": 0.45},
            2: {"score": 0.48, "mean": 0.50, "wilson_lcb": 0.48, "wilson_ucb": 0.80},
        },
    ):
      action = bot.step(_ShieldState(legal=[1, 2]), player=0)

    self.assertEqual(action, 1)
    stats = bot.decision_stats()
    self.assertEqual(stats["survival_shield_overrides"], 0)
    self.assertEqual(stats["survival_shield_base_kept"], 1)

  def test_survival_shield_rollout_exits_when_any_player_paradoxes(self):
    args = argparse.Namespace(
        players=3,
        survival_shield_base_bot="heuristic_safe14",
        survival_shield_continuation_bot="heuristic_safe14",
        survival_shield_threshold=0.45,
        survival_shield_rollouts=1,
        survival_shield_samples=1,
        survival_shield_max_actions=0,
        survival_shield_phases="",
    )
    bot = full_elo.SurvivalShieldBot("survival_shield", args, seed=0)

    with mock.patch.object(
        bot,
        "_continuation_action",
        side_effect=AssertionError("rollout should stop after paradox"),
    ):
      survived = bot._rollout_survives(
          _ShieldEarlyParadoxState(), player=0, first_action=1
      )

    self.assertEqual(survived, 0.0)
    self.assertEqual(
        bot.raw_decision_stats()["survival_shield_early_paradox_rollouts"], 1
    )

  def test_action_features_include_future_rank_pressure(self):
    features = action_feature_matrix(_FeatureState(), player=0, num_actions=10)
    row = features[3]

    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_rank_slots_after"]],
        4.0 / 12.0,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_rank_deficit_after"]],
        0.5,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_tight_rank_count_after"]],
        1.0 / 3.0,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_rank_deficit_delta"]],
        0.0,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_safe_flex_score_after"]],
        -30.2 / 90.0,
        places=5,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_min_rank_surplus_after"]],
        -0.25,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_max_rank_deficit_after"]],
        0.5,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_buffer_deficit_after"]],
        1.0,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_no_exit_after"]],
        0.0,
    )
    self.assertAlmostEqual(
        row[APPENDED_ACTION_FEATURE_INDEX["future_dead_rank_count_after"]],
        1.0 / 3.0,
    )

  def test_safe_discard_preserves_singleton_escape_cards(self):
    bot = HeuristicBot(
        future_flex_weight=4.0,
        rank_deficit_weight=30.0,
        discard_future_flex_weight=4.0,
        discard_rank_deficit_weight=30.0,
    )

    self.assertEqual(bot._discard(_DiscardState(), 0, [0, 1, 2]), 1)

  def test_shared_prediction_counts_top_rank_and_first_player_kicker(self):
    self.assertEqual(shared_prediction_action(_PredictState([0, 0, 0, 0, 1, 1]), 0), 102)
    self.assertEqual(shared_prediction_action(_PredictState([0, 0, 0, 0, 2, 1]), 0), 102)
    self.assertEqual(shared_prediction_action(_PredictState([0, 0, 0, 0, 1, 1], start_player=1), 0), 101)
    self.assertEqual(shared_prediction_action(_PredictState([0, 0, 0, 0, 0, 3]), 0), 103)

  def test_prediction_weights_no_longer_change_shared_bid(self):
    state = _PredictState([0, 0, 0, 0, 2, 1])

    self.assertEqual(
        HeuristicBot(prediction_bias=0.0)._predict(state, 0, [101, 102, 103, 104]),
        102,
    )
    self.assertEqual(
        HeuristicBot(prediction_bias=1.0, prediction_duplicate_weight=10.0)._predict(
            state, 0, [101, 102, 103, 104]
        ),
        102,
    )

  def test_az_policy_bot_uses_shared_prediction_without_model(self):
    state = _PredictState([0, 0, 0, 0, 1, 1])
    bot = az_torch.AZPolicyBot(
        model=None, name="az_policy", device=None, value_scale=1.0
    )

    self.assertEqual(bot.step(state, 0), 102)

  def test_token_loss_pressure_penalizes_destroying_only_future_slot(self):
    bot = HeuristicBot(
        rank_deficit_weight=10.0,
        token_loss_pressure_weight=1.0,
        token_loss_no_exit_weight=5.0,
    )

    self.assertAlmostEqual(
        bot._score_token_loss_pressure(_TokenLossPressureState(), 0, 10),
        -15.0,
    )

  def test_rank_slot_urgency_prefers_scarce_rank(self):
    bot = HeuristicBot(rank_slot_urgency_weight=8.0)
    state = _RankUrgencyState()

    self.assertGreater(
        bot._score_trick_action(state, 0, 3),
        bot._score_trick_action(state, 0, 4),
    )

  def test_counterfactual_policy_blend_uses_positive_mask_weights(self):
    policy = np.array([0.70, 0.10, 0.10, 0.10], dtype=np.float32)
    value_targets = np.array([0.0, 0.2, -0.1, 0.0], dtype=np.float32)
    value_mask = np.array([0.25, 0.25, 0.25, 0.0], dtype=np.float32)
    args = type("Args", (), {
        "counterfactual_policy_target_weight": 1.0,
        "counterfactual_policy_target_min_actions": 2,
        "counterfactual_policy_target_temperature": 0.06,
        "counterfactual_policy_target_min_spread": 0.05,
    })()

    blended = blend_policy_with_counterfactual_values(
        policy, value_targets, value_mask, args
    )

    self.assertGreater(blended[1], blended[0])
    self.assertGreater(blended[1], blended[2])
    self.assertAlmostEqual(float(np.sum(blended)), 1.0, places=6)

  def test_counterfactual_paradox_policy_blend_prefers_low_risk_actions(self):
    policy = np.array([0.60, 0.20, 0.10, 0.10], dtype=np.float32)
    risk_targets = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    risk_mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    args = type("Args", (), {
        "counterfactual_paradox_policy_target_weight": 1.0,
        "counterfactual_paradox_policy_target_min_actions": 2,
        "counterfactual_paradox_policy_target_temperature": 0.06,
        "counterfactual_paradox_policy_target_min_spread": 0.5,
    })()

    blended = blend_policy_with_counterfactual_paradox(
        policy, risk_targets, risk_mask, args
    )

    self.assertGreater(blended[1], blended[0])
    self.assertGreater(blended[1], blended[2])
    self.assertAlmostEqual(float(np.sum(blended)), 1.0, places=6)

  def test_counterfactual_label_coverage_reports_policy_risk_labels(self):
    action_features = np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32)
    action_features[:, 1 + 2] = 1.0
    replay = [(
        np.zeros(8, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.full(4, 0.25, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        -1,
        -1,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        None,
        None,
        action_features,
    )]

    report = counterfactual_label_coverage_report(replay)

    self.assertEqual(report["policy_labeled_rows"], 1)
    self.assertEqual(report["policy_labeled_actions"], 3)
    self.assertAlmostEqual(report["policy_target_positive_rate"], 2.0 / 3.0, places=4)
    self.assertEqual(report["policy_by_phase"]["prediction"]["rows"], 1)

  def test_policy_target_bucket_weights_multiply_matching_buckets(self):
    action_features = np.zeros((2, 4, ACTION_FEATURE_SIZE), dtype=np.float32)
    action_features[:, :, ACTION_FEATURE_RANK_NORM_INDEX] = 1.0
    action_features[0, 1, ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX] = 1.0
    action_features[0, 1, APPENDED_ACTION_FEATURE_INDEX[
        "can_still_hit_after"
    ]] = 1.0
    action_features[1, 2, ACTION_FEATURE_ADJACENCY_GAIN_INDEX] = 0.25
    policy = np.zeros((2, 4), dtype=np.float32)
    policy[0, 1] = 1.0
    policy[1, 2] = 1.0
    args = type("Args", (), {
        "policy_target_bucket_weights": (
            "token_loss=2,prediction_feasible=3,cluster_growth=5"
        ),
    })()

    weights = policy_target_bucket_weights(
        torch.tensor(action_features),
        torch.tensor(policy),
        args,
        torch.device("cpu"),
    )

    self.assertEqual(float(weights[0]), 6.0)
    self.assertEqual(float(weights[1]), 5.0)

  def test_validation_split_can_use_action_paradox_labels(self):
    def row(mask_value):
      return (
          np.zeros(8, dtype=np.float32),
          np.ones(4, dtype=np.float32),
          np.full(4, 0.25, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          0,
          0,
          np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
          np.array([0.0, mask_value, 0.0, mask_value], dtype=np.float32),
          None,
          None,
          np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32),
      )
    replay = [row(1.0), row(1.0), row(0.0)]
    args = argparse.Namespace(
        action_value_validation_fraction=0.5,
        loaded_replay_validation_label_kind="action_paradox",
        action_value_validation_seed=7,
    )

    train_replay, validation_replay, split = (
        az_torch.split_action_value_validation_replay(replay, args)
    )

    self.assertEqual(len(train_replay), 2)
    self.assertEqual(len(validation_replay), 1)
    self.assertEqual(split["loaded_replay_validation_label_kind"], "action_paradox")
    self.assertEqual(split["loaded_replay_validation_labeled_rows"], 2)
    self.assertEqual(split["loaded_replay_validation_rows"], 1)

  def test_loaded_replay_validation_score_supports_action_paradox_brier(self):
    row = {"action_paradox_validation_report": {"brier": 0.125, "corr": 0.75}}

    brier_score = az_torch.loaded_replay_validation_score(
        row,
        argparse.Namespace(
            loaded_replay_best_metric="action_paradox_validation_brier"
        ),
    )
    corr_score = az_torch.loaded_replay_validation_score(
        row,
        argparse.Namespace(
            loaded_replay_best_metric="action_paradox_validation_corr"
        ),
    )

    self.assertEqual(brier_score, -0.125)
    self.assertEqual(corr_score, 0.75)

  def test_mid_rank_composite_bucket_masks(self):
    action_rows = torch.zeros((3, ACTION_FEATURE_SIZE), dtype=torch.float32)
    action_rows[:, ACTION_FEATURE_RANK_NORM_INDEX] = torch.tensor([
        4.0 / 6.0,
        4.0 / 6.0,
        1.0,
    ])
    action_rows[0, ACTION_FEATURE_ADJACENCY_GAIN_INDEX] = 1.0
    action_rows[1, APPENDED_ACTION_FEATURE_INDEX["hits_prediction"]] = 1.0
    action_rows[1, ACTION_FEATURE_FOLLOWS_LED_INDEX] = 1.0
    action_rows[2, ACTION_FEATURE_ADJACENCY_GAIN_INDEX] = 1.0
    self.assertIn("mid_rank_cluster_growth", TACTICAL_POLICY_BUCKET_NAMES)
    self.assertIn("mid_rank_hits_prediction", TACTICAL_POLICY_BUCKET_NAMES)
    self.assertIn("mid_rank_follows_led", TACTICAL_POLICY_BUCKET_NAMES)

    masks = policy_target_bucket_masks(action_rows)

    self.assertEqual(masks["mid_rank"].tolist(), [True, True, False])
    self.assertEqual(
        masks["mid_rank_cluster_growth"].tolist(), [True, False, False]
    )
    self.assertEqual(
        masks["mid_rank_hits_prediction"].tolist(), [False, True, False]
    )
    self.assertEqual(
        masks["mid_rank_follows_led"].tolist(), [False, True, False]
    )

  def test_red_composite_bucket_masks(self):
    action_rows = torch.zeros((4, ACTION_FEATURE_SIZE), dtype=torch.float32)
    action_rows[0, ACTION_FEATURE_IS_RED_INDEX] = 1.0
    action_rows[0, ACTION_FEATURE_ADJACENCY_GAIN_INDEX] = 1.0
    action_rows[1, ACTION_FEATURE_IS_RED_INDEX] = 1.0
    action_rows[1, APPENDED_ACTION_FEATURE_INDEX["can_still_hit_after"]] = 1.0
    action_rows[2, ACTION_FEATURE_IS_RED_INDEX] = 1.0
    action_rows[2, APPENDED_ACTION_FEATURE_INDEX["hits_prediction"]] = 1.0
    action_rows[3, ACTION_FEATURE_ADJACENCY_GAIN_INDEX] = 1.0
    action_rows[3, APPENDED_ACTION_FEATURE_INDEX["can_still_hit_after"]] = 1.0
    self.assertIn("red_cluster_growth", TACTICAL_POLICY_BUCKET_NAMES)
    self.assertIn("red_prediction_feasible", TACTICAL_POLICY_BUCKET_NAMES)
    self.assertIn("red_hits_prediction", TACTICAL_POLICY_BUCKET_NAMES)

    masks = policy_target_bucket_masks(action_rows)

    self.assertEqual(masks["red"].tolist(), [True, True, True, False])
    self.assertEqual(
        masks["red_cluster_growth"].tolist(), [True, False, False, False]
    )
    self.assertEqual(
        masks["red_prediction_feasible"].tolist(), [False, True, False, False]
    )
    self.assertEqual(
        masks["red_hits_prediction"].tolist(), [False, False, True, False]
    )

  def test_apply_replay_metadata_accepts_nested_combined_metadata(self):
    args = argparse.Namespace(counterfactual_full_match_rollout=False)
    metadata = {
        "rows": 3,
        "sources": [
            {"name": "base", "metadata": {}},
            {
                "name": "full_match",
                "metadata": {"counterfactual_full_match_rollout": True},
            },
        ],
    }

    updated = apply_replay_metadata_to_args(args, metadata)

    self.assertTrue(updated.counterfactual_full_match_rollout)
    self.assertFalse(args.counterfactual_full_match_rollout)


if __name__ == "__main__":
  absltest.main()
