#!/usr/bin/env python3
"""Tests for AlphaZero trainer helpers."""

import argparse
import json
from pathlib import Path
import random
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np
import torch

from quantum_cat_ai import (
    HeuristicBot,
    shared_discard_action,
    shared_prediction_action,
)
import quantum_cat_alphazero_torch as az_torch
import quantum_cat_action_paradox_policy_replay as risk_policy_replay
import quantum_cat_action_set_transformer_trainer as action_set_trainer
import quantum_cat_best_of_rollouts_replay as best_of_rollouts_replay
import quantum_cat_best_of_rollouts_calibration as best_of_rollouts_calibration
import quantum_cat_calibration_transfer_audit as calibration_transfer_audit
import quantum_cat_chained_best_of_rollouts_replay as chained_rollout_replay
import quantum_cat_chained_checkpoint_audit as chained_checkpoint_audit
import quantum_cat_failure_miner as failure_miner
import quantum_cat_full_match_elo as full_elo
import quantum_cat_improve_graft_miner as improve_graft_miner
import quantum_cat_legal_set_selector_trainer as legal_set_trainer
import quantum_cat_proposal_policy_checkpoint_audit as proposal_policy_audit
import quantum_cat_proposal_policy_replay_repair as proposal_replay_repair
import quantum_cat_public_history as public_history
import quantum_cat_preforced_replay as preforced_replay
import quantum_cat_chained_replay_audit as chained_replay_audit
import quantum_cat_curated_validation as curated_validation
import quantum_cat_replay_holdout_audit as replay_holdout_audit
import quantum_cat_survival_value_replay as survival_value_replay
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

  def __init__(self, hand=None, legal=None):
    self._phase = 1
    self._num_colors = 4
    hand = [1, 2, 1] if hand is None else hand
    self._num_card_types = len(hand)
    self._hands = [
        np.array(hand, dtype=np.int32),
        np.zeros(len(hand), dtype=np.int32),
        np.zeros(len(hand), dtype=np.int32),
    ]
    self._board_ownership = -1 * np.ones((4, len(hand)), dtype=np.int32)
    self._color_tokens = np.ones((3, 4), dtype=bool)
    self._legal = list(range(len(hand))) if legal is None else list(legal)

  def clone(self):
    clone = _DiscardState(self._hands[0], self._legal)
    clone._hands = [np.copy(hand) for hand in self._hands]
    clone._board_ownership = np.copy(self._board_ownership)
    clone._color_tokens = np.copy(self._color_tokens)
    return clone

  def apply_action(self, action):
    self._hands[0][action] -= 1

  def legal_actions(self, player):
    del player
    return list(self._legal)

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

  def is_chance_node(self):
    return False

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

  def _patch_full_elo_q_policy_aux(
      self, policy, risks=None, action_values=None, action_features=None
  ):
    return mock.patch.object(
        full_elo,
        "model_policy_value_action_aux",
        return_value=(policy, 0.0, action_values, risks, action_features),
    )

  def _patch_az_q_policy_aux(
      self, policy, risks=None, action_values=None, action_features=None
  ):
    return mock.patch.object(
        az_torch,
        "model_policy_value_action_aux",
        return_value=(policy, 0.0, action_values, risks, action_features),
    )

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

    derived, summary, source_indices = risk_policy_replay.derive_policy_replay(
        [example], args
    )

    self.assertLen(derived, 1)
    self.assertEqual(source_indices, [0])
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

    derived, summary, source_indices = risk_policy_replay.derive_policy_replay(
        [unlabeled, labeled], args
    )

    self.assertLen(derived, 2)
    self.assertEqual(source_indices, [0, 1])
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

    derived, summary, source_indices = risk_policy_replay.derive_policy_replay(
        [example], args, anchor_policies=anchor_policies
    )

    self.assertLen(derived, 1)
    self.assertEqual(source_indices, [0])
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

  def test_action_features_distinguish_all_three_bid_regimes_without_resize(self):
    suite = curated_validation.load_suite(
        curated_validation.DEFAULT_BID_REGIME_SUITE
    )
    cases = {
        kind: next(
            item for item in suite["scenarios"]
            if f"4p-{kind}" in item["id"]
        )
        for kind in ("below-bid-win", "at-bid-lose", "above-bid-win")
    }
    outcomes = {}
    for kind, scenario in cases.items():
      state = curated_validation.validate_scenario(scenario)["state"]
      actor = int(scenario["state"]["actingPlayer"])
      legal = state.legal_actions(actor)
      winning = az_torch.action_feature_vector(state, actor, 6, legal)
      losing = az_torch.action_feature_vector(state, actor, 1, legal)
      self.assertEqual(winning.shape, (ACTION_FEATURE_SIZE,))
      self.assertEqual(losing.shape, (ACTION_FEATURE_SIZE,))
      outcomes[kind] = {
          "wants_more": winning[
              az_torch.ACTION_FEATURE_WANTS_MORE_TRICKS_INDEX
          ],
          "at_or_above": winning[
              az_torch.ACTION_FEATURE_AT_OR_ABOVE_PREDICTION_INDEX
          ],
          "winning_aligns": winning[
              az_torch.ACTION_FEATURE_WIN_ALIGNS_TARGET_INDEX
          ],
          "losing_aligns": losing[
              az_torch.ACTION_FEATURE_WIN_ALIGNS_TARGET_INDEX
          ],
          "winner_can_hit": winning[
              APPENDED_ACTION_FEATURE_INDEX["can_still_hit_after"]
          ],
          "loser_can_hit": losing[
              APPENDED_ACTION_FEATURE_INDEX["can_still_hit_after"]
          ],
          "winner_breaks_exact": winning[
              APPENDED_ACTION_FEATURE_INDEX["exit_over_target_would_win"]
          ],
      }

    self.assertEqual(outcomes["below-bid-win"], {
        "wants_more": 1.0, "at_or_above": 0.0,
        "winning_aligns": 1.0, "losing_aligns": 0.0,
        "winner_can_hit": 1.0, "loser_can_hit": 1.0,
        "winner_breaks_exact": 0.0,
    })
    self.assertEqual(outcomes["at-bid-lose"], {
        "wants_more": 0.0, "at_or_above": 1.0,
        "winning_aligns": 0.0, "losing_aligns": 1.0,
        "winner_can_hit": 0.0, "loser_can_hit": 1.0,
        "winner_breaks_exact": 1.0,
    })
    self.assertEqual(outcomes["above-bid-win"], {
        "wants_more": 1.0, "at_or_above": 1.0,
        "winning_aligns": 1.0, "losing_aligns": 0.0,
        "winner_can_hit": 0.0, "loser_can_hit": 0.0,
        "winner_breaks_exact": 0.0,
    })

  def test_action_features_mark_unreachable_underbid_as_already_missed(self):
    suite = curated_validation.load_suite(
        curated_validation.DEFAULT_BID_REGIME_SUITE
    )
    scenario = next(
        item for item in suite["scenarios"]
        if "unreachable-underbid-exact-final" in item["id"]
    )
    state = curated_validation.validate_scenario(scenario)["state"]
    actor = int(scenario["state"]["actingPlayer"])
    for action in state.legal_actions(actor):
      features = az_torch.action_feature_vector(
          state, actor, action, state.legal_actions(actor)
      )
      self.assertEqual(
          features[az_torch.ACTION_FEATURE_WANTS_MORE_TRICKS_INDEX], 1.0
      )
      self.assertEqual(
          features[APPENDED_ACTION_FEATURE_INDEX["can_still_hit_after"]], 0.0
      )

  def test_bid_regime_auxiliary_loss_teaches_all_three_safe_trick_choices(self):
    suite = curated_validation.load_suite(
        curated_validation.DEFAULT_BID_REGIME_SUITE
    )
    rows = []
    for regime in ("below-bid-win", "at-bid-lose", "above-bid-win"):
      scenario = next(
          item for item in suite["scenarios"] if f"4p-{regime}" in item["id"]
      )
      state = curated_validation.validate_scenario(scenario)["state"]
      actor = int(scenario["state"]["actingPlayer"])
      rows.append(np.stack([
          az_torch.action_feature_vector(state, actor, action,
                                         state.legal_actions(actor))
          for action in (1, 6)
      ]))
    features = torch.tensor(np.asarray(rows), dtype=torch.float32)
    mask = torch.ones((3, 2), dtype=torch.bool)
    bad = torch.tensor(
        [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]], requires_grad=True
    )
    good = torch.tensor([[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
    args = argparse.Namespace(bid_regime_policy_target_mass=0.65)

    bad_loss = az_torch.bid_regime_policy_loss(bad, features, mask, args)
    good_loss = az_torch.bid_regime_policy_loss(good, features, mask, args)

    self.assertGreater(float(bad_loss.detach()), 0.3)
    self.assertLess(float(good_loss.detach()), 1e-8)
    bad_loss.backward()
    self.assertIsNotNone(bad.grad)
    self.assertLess(float(bad.grad[0, 1]), 0.0)
    self.assertLess(float(bad.grad[1, 0]), 0.0)
    self.assertLess(float(bad.grad[2, 1]), 0.0)

  def test_bid_regime_auxiliary_loss_never_rewards_personal_no_exit_risk(self):
    suite = curated_validation.load_suite(
        curated_validation.DEFAULT_BID_REGIME_SUITE
    )
    scenario = next(
        item for item in suite["scenarios"] if "4p-above-bid-win" in item["id"]
    )
    state = curated_validation.validate_scenario(scenario)["state"]
    actor = int(scenario["state"]["actingPlayer"])
    features = np.stack([
        az_torch.action_feature_vector(state, actor, action,
                                       state.legal_actions(actor))
        for action in (1, 6)
    ])
    features[1, APPENDED_ACTION_FEATURE_INDEX[
        "own_future_zero_exit_frac_after"
    ]] = 1.0
    loss = az_torch.bid_regime_policy_loss(
        torch.tensor([[4.0, 0.0]]),
        torch.tensor(features[None, ...]),
        torch.ones((1, 2), dtype=torch.bool),
        argparse.Namespace(bid_regime_policy_target_mass=0.65),
    )
    self.assertEqual(float(loss), 0.0)

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

  def test_prediction_policy_head_starts_as_exact_champion_no_op(self):
    champion = AZNet(
        obs_size=8, num_actions=4, num_players=3, width=32, depth=1,
        arch="action_mlp",
    )
    augmented = AZNet(
        obs_size=8, num_actions=4, num_players=3, width=32, depth=1,
        arch="action_mlp", separate_prediction_policy_head=True,
    )
    az_torch.load_compatible_state_dict(augmented, champion.state_dict())
    observations = torch.randn((2, 8), dtype=torch.float32)
    features = torch.zeros((2, 4, ACTION_FEATURE_SIZE), dtype=torch.float32)
    features[..., 0] = 1.0
    features[0, :, 7] = 1.0
    features[1, :, 8] = 1.0

    expected = champion.forward_with_all_aux(observations, features)
    actual = augmented.forward_with_all_aux(observations, features)

    for baseline, candidate in zip(expected, actual):
      self.assertTrue(torch.equal(baseline, candidate))

  def test_prediction_policy_head_cannot_change_play_or_auxiliary_outputs(self):
    model = AZNet(
        obs_size=8, num_actions=4, num_players=3, width=32, depth=1,
        arch="action_mlp", separate_prediction_policy_head=True,
    )
    observations = torch.randn((2, 8), dtype=torch.float32)
    features = torch.zeros((2, 4, ACTION_FEATURE_SIZE), dtype=torch.float32)
    features[..., 0] = 1.0
    features[0, :, 7] = 1.0
    features[1, :, 8] = 1.0
    before = model.forward_with_all_aux(observations, features)

    with torch.no_grad():
      model.prediction_policy[-1].weight.fill_(1.0)
      model.prediction_policy[-1].bias.fill_(2.0)
    after = model.forward_with_all_aux(observations, features)

    self.assertFalse(torch.equal(before[0][0], after[0][0]))
    self.assertTrue(torch.equal(before[0][1], after[0][1]))
    for baseline, candidate in zip(before[1:], after[1:]):
      self.assertTrue(torch.equal(baseline, candidate))

  def test_raw_prediction_head_reads_public_expected_trick_features(self):
    model = AZNet(
        obs_size=8, num_actions=4, num_players=3, width=32, depth=1,
        arch="action_mlp", separate_prediction_policy_head=True,
        prediction_policy_input="raw",
    )
    features = torch.zeros((1, 4, ACTION_FEATURE_SIZE), dtype=torch.float32)
    features[..., 0] = 1.0
    features[..., 7] = 1.0
    features[0, :, 156] = torch.tensor([0.6, 0.1, 0.4, 0.8])
    with torch.no_grad():
      for param in model.prediction_policy.parameters():
        param.zero_()
      model.prediction_policy[0].weight[0, 156] = 1.0
      model.prediction_policy[2].weight[0, 0] = -10.0
    observation = torch.zeros((1, 8), dtype=torch.float32)
    base = model.policy(model._pair_features(model.body(observation), features))
    actual, _ = model(observation, features)
    delta = actual - base.squeeze(-1)

    self.assertTrue(torch.allclose(
        delta,
        torch.tensor([[-6.0, -1.0, -4.0, -8.0]]),
        atol=1e-6,
    ))

  def test_prediction_head_only_freezes_entire_existing_champion(self):
    model = AZNet(
        obs_size=8, num_actions=4, num_players=3, width=32, depth=1,
        arch="action_mlp", separate_action_value_encoder=True,
        separate_action_paradox_encoder=True,
        separate_prediction_policy_head=True,
    )
    args = argparse.Namespace(train_prediction_head_only=True)

    trainable = configure_trainable_parameters(model, args)
    names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    self.assertNotEmpty(trainable)
    self.assertEqual(
        names,
        {
            "prediction_policy.0.weight", "prediction_policy.0.bias",
            "prediction_policy.2.weight", "prediction_policy.2.bias",
        },
    )

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

  def test_compact_legal_replay_batch_remaps_all_action_fields(self):
    num_actions = 8

    def row(legal, selected):
      mask = np.zeros(num_actions, dtype=np.float32)
      mask[legal] = 1.0
      policy = np.zeros(num_actions, dtype=np.float32)
      policy[selected] = 1.0
      risk_targets = np.arange(num_actions, dtype=np.float32)
      risk_mask = np.array(mask, copy=True)
      value_targets = risk_targets + 10.0
      features = np.zeros((num_actions, ACTION_FEATURE_SIZE), dtype=np.float32)
      features[legal, 0] = 1.0
      features[legal, 1] = np.asarray(legal, dtype=np.float32)
      return (
          np.zeros(4, dtype=np.float32),
          mask,
          policy,
          np.ones(3, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          selected,
          0,
          risk_targets,
          risk_mask,
          value_targets,
          risk_mask,
          features,
      )

    batch = [row([1, 4], 4), row([0, 2, 7], 7)]
    compact = az_torch.compact_legal_replay_batch(batch, num_actions)

    self.assertEqual(compact["num_actions"], 3)
    self.assertEqual(compact["actions"], [1, 2])
    np.testing.assert_array_equal(compact["masks"][0], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(compact["policies"][0], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(
        compact["action_target_vectors"][0], [1.0, 4.0, 0.0]
    )
    np.testing.assert_array_equal(
        compact["action_value_target_vectors"][1], [10.0, 12.0, 17.0]
    )
    np.testing.assert_array_equal(
        compact["action_features"][0, :2]
        if isinstance(compact["action_features"], np.ndarray)
        else compact["action_features"][0][:2],
        batch[0][11][[1, 4]],
    )

  def test_compact_legal_action_training_matches_dense_value_loss(self):
    torch.manual_seed(20260819)
    dense_model = AZNet(
        obs_size=4,
        num_actions=8,
        num_players=3,
        width=16,
        depth=1,
        arch="action_mlp",
    )
    compact_model = AZNet(
        obs_size=4,
        num_actions=8,
        num_players=3,
        width=16,
        depth=1,
        arch="action_mlp",
    )
    compact_model.load_state_dict(dense_model.state_dict())

    def row(legal, selected, target):
      mask = np.zeros(8, dtype=np.float32)
      mask[legal] = 1.0
      policy = np.zeros(8, dtype=np.float32)
      policy[selected] = 1.0
      features = np.zeros((8, ACTION_FEATURE_SIZE), dtype=np.float32)
      features[legal, 0] = 1.0
      features[legal, 1] = np.asarray(legal, dtype=np.float32) / 8.0
      return (
          np.full(4, target, dtype=np.float32),
          mask,
          policy,
          np.full(3, target, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          selected,
          0,
          None,
          None,
          None,
          None,
          features,
      )

    replay = [row([1, 4], 4, 1.0), row([0, 2, 7], 7, -1.0)]
    args = argparse.Namespace(
        train_steps=1,
        batch_size=2,
        players=3,
        policy_loss_weight=0.0,
        value_loss_weight=1.0,
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
        compact_legal_action_training=False,
    )
    dense_optimizer = torch.optim.Adam(dense_model.parameters(), lr=0.0)
    compact_optimizer = torch.optim.Adam(compact_model.parameters(), lr=0.0)

    with mock.patch.object(
        az_torch, "sample_training_batch", return_value=replay
    ):
      dense_loss = az_torch.train_steps(
          dense_model, dense_optimizer, replay, args, torch.device("cpu")
      )
      args.compact_legal_action_training = True
      compact_loss = az_torch.train_steps(
          compact_model, compact_optimizer, replay, args, torch.device("cpu")
      )

    np.testing.assert_allclose(compact_loss, dense_loss, atol=1e-6, rtol=1e-6)

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

  def test_train_policy_head_only_freezes_action_encoders(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_attn",
    )
    args = argparse.Namespace(train_policy_head_only=True)

    trainable = configure_trainable_parameters(model, args)
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    self.assertNotEmpty(trainable)
    self.assertIn("policy.0.weight", trainable_names)
    self.assertIn("policy.2.weight", trainable_names)
    self.assertNotIn("action_encoder.0.weight", trainable_names)
    self.assertNotIn("state_action_projection.0.weight", trainable_names)
    self.assertNotIn("action_attention.in_proj_weight", trainable_names)
    self.assertNotIn("body.0.weight", trainable_names)
    self.assertNotIn("action_value.0.weight", trainable_names)

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

  def test_train_value_output_only_freezes_hidden_value_representation(self):
    model = AZNet(
        obs_size=8,
        num_actions=5,
        num_players=3,
        width=32,
        depth=1,
        arch="action_mlp",
    )
    args = argparse.Namespace(train_value_output_only=True)

    trainable = configure_trainable_parameters(model, args)
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    self.assertLen(trainable, 2)
    self.assertEqual(trainable_names, {"value.2.weight", "value.2.bias"})

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

    with self._patch_full_elo_q_policy_aux(policy, risks=risks):
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

    with self._patch_full_elo_q_policy_aux(policy, risks=risks):
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

    with self._patch_full_elo_q_policy_aux(policy, risks=risks):
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

    with self._patch_full_elo_q_policy_aux(policy, risks=risks):
      action = bot.step(_ShieldState([1, 2, 3]), player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["rerank_relative_applied"], 1)
    self.assertEqual(stats["rerank_relative_candidates"], 2)
    self.assertEqual(stats["rerank_overrides"], 1)

  def test_q_policy_relative_dominance_abstains_on_small_risk_edge(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="relative_dominance",
        action_paradox_risk_threshold=0.0,
        action_paradox_min_risk_margin=0.05,
        action_paradox_max_policy_log_gap=-1.0,
        action_value_rerank_clip=0.5,
        action_value_rerank_phases="play",
        action_value_rerank_min_margin=0.0,
    )
    policy = np.array([0.0, 0.6, 0.35, 0.05], dtype=np.float32)
    risks = np.array([0.0, 0.24, 0.2, 0.22], dtype=np.float32)
    bot = full_elo.AZQPolicyBot(
        model=object(), name="q", device="cpu", model_args=args
    )

    with self._patch_full_elo_q_policy_aux(policy, risks=risks):
      action = bot.step(_ShieldState([1, 2, 3]), player=0)

    self.assertEqual(action, 1)
    stats = bot.decision_stats()
    self.assertEqual(stats["rerank_relative_blocked_risk_margin"], 1)
    self.assertEqual(stats["rerank_relative_applied"], 0)
    self.assertEqual(stats["rerank_overrides"], 0)

  def test_q_policy_relative_dominance_overrides_clear_risk_edge(self):
    args = argparse.Namespace(
        value_scale=10.0,
        action_value_selection_weight=0.0,
        action_paradox_selection_penalty=0.0,
        action_paradox_rerank_mode="relative_dominance",
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

    with self._patch_full_elo_q_policy_aux(policy, risks=risks):
      action = bot.step(_ShieldState([1, 2, 3]), player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["rerank_relative_blocked_risk_margin"], 0)
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

    with self._patch_full_elo_q_policy_aux(
        policy, risks=risks, action_features=features
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

    with self._patch_az_q_policy_aux(policy, risks=risks):
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

    with self._patch_az_q_policy_aux(policy, risks=risks):
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
        az_torch,
        "model_policy_value_action_aux",
        return_value=(
            priors,
            np.array([0.0]),
            np.zeros_like(priors),
            risks,
            None,
        ),
    ):
      policy = az_torch.mcts_policy(
          _MCTSRiskState(), object(), args, device="cpu", add_noise=False
      )

    self.assertGreater(policy[2], policy[1])

  def test_mcts_backpropagates_each_visited_node_exactly_once(self):
    class _DepthTwoState:

      def __init__(self):
        self.depth = 0

      def is_terminal(self):
        return self.depth == 2

      def current_player(self):
        return self.depth % 2

      def apply_action(self, action):
        if action != 0:
          raise ValueError(f"unexpected action {action}")
        self.depth += 1

      def returns(self):
        return [1.0, -1.0]

    root = az_torch.Node(1.0)
    intermediate = az_torch.Node(1.0)
    terminal = az_torch.Node(1.0)
    root.children[0] = intermediate
    intermediate.children[0] = terminal

    value = az_torch.run_simulation(
        root,
        _DepthTwoState(),
        model=None,
        num_actions=1,
        c_puct=1.0,
        value_scale=1.0,
        device="cpu",
    )

    np.testing.assert_array_equal(value, [1.0, -1.0])
    self.assertEqual(root.visit_count, 1)
    self.assertEqual(intermediate.visit_count, 1)
    self.assertEqual(terminal.visit_count, 1)
    self.assertEqual(intermediate.q_value(0), 1.0)
    self.assertEqual(intermediate.q_value(1), -1.0)

  def test_mcts_applies_survival_model_before_reaching_terminal_state(self):
    survival_model = object()
    counters = {"nonterminal_evaluations": 0}
    args = argparse.Namespace(
        _search_survival_model=survival_model,
        _survival_search_counters=counters,
        _survival_search_root_risk=0.0,
        survival_value_search_penalty=20.0,
    )
    priors = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)
    node = az_torch.Node(1.0)

    with mock.patch.object(
        az_torch,
        "model_policy_value_action_aux",
        return_value=(
            priors,
            np.array([10.0, 6.0], dtype=np.float32),
            None,
            None,
            None,
        ),
    ), mock.patch.object(
        az_torch,
        "model_policy_value",
        return_value=(priors, np.array([-0.6, -0.6], dtype=np.float32)),
    ) as survival_prediction:
      values = az_torch.expand(
          node,
          _MCTSRiskState(),
          model=object(),
          num_actions=4,
          value_scale=20.0,
          device="cpu",
          terminal_value_args=args,
      )

    np.testing.assert_allclose(values, [-6.0, -10.0])
    self.assertEqual(counters["nonterminal_evaluations"], 1)
    self.assertIs(survival_prediction.call_args.args[0], survival_model)

  def test_mcts_survival_root_baseline_prevents_absolute_risk_shift(self):
    counters = {"nonterminal_evaluations": 0}
    args = argparse.Namespace(
        _search_survival_model=object(),
        _survival_search_counters=counters,
        survival_value_search_penalty=20.0,
    )
    priors = np.array([0.0, 0.7, 0.3, 0.0], dtype=np.float32)

    with mock.patch.object(
        az_torch,
        "model_policy_value_action_aux",
        return_value=(
            priors,
            np.array([10.0, 6.0], dtype=np.float32),
            None,
            None,
            None,
        ),
    ), mock.patch.object(
        az_torch,
        "model_policy_value",
        side_effect=[
            (priors, np.array([-0.6, -0.6], dtype=np.float32)),
            (priors, np.array([-0.2, -0.2], dtype=np.float32)),
        ],
    ):
      root_values = az_torch.expand(
          az_torch.Node(1.0),
          _MCTSRiskState(),
          model=object(),
          num_actions=4,
          value_scale=20.0,
          device="cpu",
          terminal_value_args=args,
      )
      safer_values = az_torch.expand(
          az_torch.Node(1.0),
          _MCTSRiskState(),
          model=object(),
          num_actions=4,
          value_scale=20.0,
          device="cpu",
          terminal_value_args=args,
      )

    np.testing.assert_allclose(root_values, [10.0, 6.0])
    np.testing.assert_allclose(safer_values, [14.0, 10.0])
    self.assertEqual(counters["nonterminal_evaluations"], 2)

  def test_batched_belief_probability_honors_shared_prediction_without_model(self):
    state = _PredictState()
    legal = state.legal_actions(0)
    forced_action = shared_prediction_action(state, 0, legal)
    rejected_action = next(action for action in legal if action != forced_action)
    model = mock.Mock()

    probabilities = az_torch._model_policy_action_probs_batch(
        model,
        [state, state],
        [0, 0],
        [forced_action, rejected_action],
        argparse.Namespace(value_scale=20.0),
        torch.device("cpu"),
    )

    self.assertEqual(probabilities, [1.0, 0.0])
    model.body.assert_not_called()

  def test_batched_belief_probability_matches_scalar_for_prediction_and_play(self):
    game = az_torch.pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    while state._phase == 1:
      player = state.current_player()
      state.apply_action(shared_discard_action(state, player))
    prediction_state = state.clone()
    prediction_player = prediction_state.current_player()
    forced_action = shared_prediction_action(prediction_state, prediction_player)
    rejected_action = next(
        action
        for action in prediction_state.legal_actions(prediction_player)
        if action != forced_action
    )
    while state._phase == 2:
      player = state.current_player()
      state.apply_action(shared_prediction_action(state, player))
    play_player = state.current_player()
    play_action = state.legal_actions(play_player)[0]
    torch.manual_seed(20260819)
    model = AZNet(
        game.observation_tensor_shape()[0],
        game.num_distinct_actions(),
        game.num_players(),
        width=16,
        depth=1,
        arch="mlp",
    )
    model.eval()
    args = argparse.Namespace(
        value_scale=20.0,
        counterfactual_belief_ref_policy_mix="model:1.0",
    )
    shadows = [prediction_state, prediction_state, state]
    players = [prediction_player, prediction_player, play_player]
    actions = [forced_action, rejected_action, play_action]

    batched = az_torch._reference_policy_action_probs_batch(
        shadows, players, actions, model, args, torch.device("cpu")
    )
    scalar = [
        az_torch._reference_policy_action_prob(
            shadow, player, action, model, args, torch.device("cpu")
        )
        for shadow, player, action in zip(shadows, players, actions)
    ]

    np.testing.assert_allclose(batched, scalar, atol=1e-7, rtol=1e-6)
    self.assertEqual(batched[:2], [1.0, 0.0])

  def test_belief_likelihood_rejects_impossible_shared_prediction(self):
    game = az_torch.pyspiel.load_game("python_quantum_cat", {"players": 4})
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    while state._phase < 3:
      player = state.current_player()
      action = shared_discard_action(state, player)
      if action is None:
        action = shared_prediction_action(state, player)
      state.apply_action(action)
    impossible = state.clone()
    prediction_replay = az_torch._reset_to_prediction_replay_start(impossible)
    player = prediction_replay.current_player()
    forced = shared_prediction_action(prediction_replay, player)
    rejected = next(
        action
        for action in prediction_replay.legal_actions(player)
        if action != forced
    )
    impossible._predictions[player] = rejected - 100
    args = argparse.Namespace(
        value_scale=20.0,
        counterfactual_belief_ref_policy_mix="model:1.0",
        counterfactual_belief_logprob_floor=1e-6,
    )

    scalar = az_torch._score_observed_trick_likelihood(
        impossible, mock.Mock(), args, torch.device("cpu")
    )
    batched = az_torch._score_observed_trick_likelihood_batch(
        [impossible], mock.Mock(), args, torch.device("cpu")
    )

    self.assertEqual(scalar, -1e9)
    self.assertEqual(batched, [-1e9])

  def test_policy_weighted_belief_never_samples_impossible_worlds(self):
    impossible = mock.Mock()
    possible = mock.Mock()
    args = argparse.Namespace(
        counterfactual_belief_samples=2,
        counterfactual_belief_candidates=2,
        counterfactual_belief_uniform_mix=0.5,
    )

    with mock.patch.object(
        az_torch,
        "_uniform_counterfactual_resamples",
        return_value=[impossible, possible],
    ), mock.patch.object(
        az_torch,
        "_score_observed_trick_likelihood_batch",
        return_value=[-1e9, 0.0],
    ), mock.patch.object(
        np.random, "choice", return_value=np.array([1, 1])
    ) as choose:
      worlds = az_torch.policy_weighted_belief_states(
          mock.Mock(), 0, args, mock.Mock(), torch.device("cpu")
      )

    self.assertLen(worlds, 2)
    np.testing.assert_array_equal(choose.call_args.kwargs["p"], [0.0, 1.0])
    impossible.clone.assert_not_called()
    self.assertEqual(possible.clone.call_count, 2)

  def test_policy_weighted_belief_retries_until_world_matches_public_history(self):
    impossible = mock.Mock()
    possible = mock.Mock()
    args = argparse.Namespace(
        counterfactual_belief_samples=1,
        counterfactual_belief_candidates=1,
    )

    with mock.patch.object(
        az_torch,
        "_uniform_counterfactual_resamples",
        side_effect=[[impossible], [possible]],
    ) as sampler, mock.patch.object(
        az_torch,
        "_score_observed_trick_likelihood_batch",
        side_effect=[[-1e9], [0.0]],
    ), mock.patch.object(np.random, "choice", return_value=np.array([0])):
      worlds = az_torch.policy_weighted_belief_states(
          mock.Mock(), 0, args, mock.Mock(), torch.device("cpu")
      )

    self.assertLen(worlds, 1)
    self.assertEqual(sampler.call_count, 2)
    self.assertEqual(
        [call.args[2] for call in sampler.call_args_list], [1, 2]
    )
    impossible.clone.assert_not_called()
    possible.clone.assert_called_once()

  def test_policy_weighted_belief_fails_closed_if_every_world_is_impossible(self):
    args = argparse.Namespace(
        counterfactual_belief_samples=1,
        counterfactual_belief_candidates=1,
    )

    with mock.patch.object(
        az_torch,
        "_uniform_counterfactual_resamples",
        return_value=[mock.Mock()],
    ) as sampler, mock.patch.object(
        az_torch,
        "_score_observed_trick_likelihood_batch",
        return_value=[-1e9],
    ):
      with self.assertRaisesRegex(RuntimeError, "observed public history"):
        az_torch.policy_weighted_belief_states(
            mock.Mock(), 0, args, mock.Mock(), torch.device("cpu")
        )

    self.assertEqual(sampler.call_count, 12)

  def test_policy_weighted_belief_fails_closed_when_likelihood_cannot_be_scored(self):
    args = argparse.Namespace(
        counterfactual_belief_samples=1,
        counterfactual_belief_candidates=1,
    )

    with mock.patch.object(
        az_torch,
        "_uniform_counterfactual_resamples",
        return_value=[mock.Mock()],
    ) as sampler, mock.patch.object(
        az_torch,
        "_score_observed_trick_likelihood_batch",
        return_value=[None],
    ):
      with self.assertRaisesRegex(RuntimeError, "unverified hidden worlds"):
        az_torch.policy_weighted_belief_states(
            mock.Mock(), 0, args, mock.Mock(), torch.device("cpu")
        )

    sampler.assert_called_once()

  def test_legal_only_inference_matches_dense_outputs_for_every_action_arch(self):
    game = az_torch.pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    while state._phase < 3:
      player = state.current_player()
      action = shared_discard_action(state, player)
      if action is None:
        action = shared_prediction_action(state, player)
      state.apply_action(action)
    player = state.current_player()
    legal = state.legal_actions(player)
    dense_features = az_torch.action_feature_matrix(
        state, player, game.num_distinct_actions()
    )
    compact_features = az_torch.compact_action_feature_matrix(
        state, player, legal
    )
    np.testing.assert_allclose(compact_features, dense_features[legal], atol=0.0)
    obs_t = torch.tensor(
        state.observation_tensor(player), dtype=torch.float32
    ).unsqueeze(0)
    dense_features_t = torch.tensor(
        dense_features, dtype=torch.float32
    ).unsqueeze(0)

    for arch in az_torch.ACTION_CONDITIONED_ARCHS:
      with self.subTest(arch=arch):
        torch.manual_seed(20260819)
        model = AZNet(
            game.observation_tensor_shape()[0],
            game.num_distinct_actions(),
            game.num_players(),
            width=32,
            depth=1,
            arch=arch,
        )
        model.eval()
        with torch.no_grad():
          dense_logits, dense_values, _, dense_risks, dense_action_values = (
              model.forward_with_all_aux(obs_t, dense_features_t)
          )
        expected_policy = torch.softmax(dense_logits[0, legal], dim=0).numpy()
        expected_risks = torch.sigmoid(dense_risks[0, legal]).numpy()
        expected_action_values = dense_action_values[0, legal].numpy()

        policy, values, action_values, risks, returned_features = (
            az_torch.model_policy_value_action_aux(
                model,
                state,
                player,
                game.num_distinct_actions(),
                value_scale=20.0,
                device=torch.device("cpu"),
                need_action_values=True,
                need_action_risks=True,
                return_action_features=True,
            )
        )

        np.testing.assert_allclose(policy[legal], expected_policy, atol=2e-6)
        np.testing.assert_allclose(
            values, 20.0 * dense_values[0].numpy(), atol=2e-5
        )
        np.testing.assert_allclose(risks[legal], expected_risks, atol=2e-6)
        np.testing.assert_allclose(
            action_values[legal], expected_action_values, atol=2e-6
        )
        np.testing.assert_array_equal(returned_features, dense_features)
        np.testing.assert_allclose(
            az_torch.model_policy_probs(
                model,
                state,
                player,
                game.num_distinct_actions(),
                torch.device("cpu"),
            )[legal],
            expected_policy,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            az_torch.model_action_risks(
                model,
                state,
                player,
                game.num_distinct_actions(),
                torch.device("cpu"),
            )[legal],
            expected_risks,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            az_torch.model_action_values(
                model,
                state,
                player,
                game.num_distinct_actions(),
                torch.device("cpu"),
            )[legal],
            expected_action_values,
            atol=2e-6,
        )

  def test_batched_compact_belief_probability_handles_different_legal_sets(self):
    game = az_torch.pyspiel.load_game("python_quantum_cat", {"players": 3})
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    while state._phase < 3:
      player = state.current_player()
      action = shared_discard_action(state, player)
      if action is None:
        action = shared_prediction_action(state, player)
      state.apply_action(action)
    other = state.clone()
    other.apply_action(other.legal_actions(other.current_player())[0])
    states = [state, other]
    players = [item.current_player() for item in states]
    actions = [
        item.legal_actions(player)[0]
        for item, player in zip(states, players)
    ]
    args = argparse.Namespace(
        value_scale=20.0,
        counterfactual_belief_ref_policy_mix="model:1.0",
    )

    for arch in az_torch.ACTION_CONDITIONED_ARCHS:
      with self.subTest(arch=arch):
        torch.manual_seed(20260820)
        model = AZNet(
            game.observation_tensor_shape()[0],
            game.num_distinct_actions(),
            game.num_players(),
            width=32,
            depth=1,
            arch=arch,
        )
        model.eval()
        batched = az_torch._reference_policy_action_probs_batch(
            states, players, actions, model, args, torch.device("cpu")
        )
        scalar = [
            az_torch._reference_policy_action_prob(
                shadow, player, action, model, args, torch.device("cpu")
            )
            for shadow, player, action in zip(states, players, actions)
        ]

        np.testing.assert_allclose(batched, scalar, atol=2e-6, rtol=2e-6)

  def test_counterfactual_resampling_fails_closed_without_actual_hidden_state(self):
    state = mock.Mock()
    state.resample_from_infostate.side_effect = ValueError("sampler rejected state")

    with self.assertRaisesRegex(RuntimeError, "actual hidden state"):
      az_torch._uniform_counterfactual_resamples(state, player=1, count=2)

    state.clone.assert_not_called()

  def test_information_state_resampling_rejects_actor_observation_changes(self):
    state = mock.Mock()
    state.observation_tensor.return_value = np.array([1.0, 2.0], dtype=np.float32)
    state.legal_actions.return_value = [101, 102]
    sampled = mock.Mock()
    sampled.observation_tensor.return_value = np.array([1.0, 3.0], dtype=np.float32)
    sampled.legal_actions.return_value = [101, 102]
    state.resample_from_infostate.return_value = sampled

    with self.assertRaisesRegex(RuntimeError, "acting player's observation"):
      az_torch._uniform_counterfactual_resamples(state, player=0, count=1)

  def test_information_state_resampling_rejects_changed_legal_actions(self):
    state = mock.Mock()
    state.observation_tensor.return_value = np.array([1.0, 2.0], dtype=np.float32)
    state.legal_actions.return_value = [101, 102]
    sampled = mock.Mock()
    sampled.observation_tensor.return_value = np.array([1.0, 2.0], dtype=np.float32)
    sampled.legal_actions.return_value = [101, 103]
    state.resample_from_infostate.return_value = sampled

    with self.assertRaisesRegex(RuntimeError, "legal actions"):
      az_torch._uniform_counterfactual_resamples(state, player=0, count=1)

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

    with self._patch_az_q_policy_aux(policy, risks=risks):
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

    with self._patch_az_q_policy_aux(policy):
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

  def test_full_match_targets_keep_score_shared_but_survival_round_local(self):
    args = argparse.Namespace(
        players=3,
        value_scale=10.0,
        terminal_paradox_penalty=2.0,
        terminal_any_paradox_penalty=5.0,
        ordinal_value_weight=0.0,
        official_outcome_value_weight=0.0,
    )
    action_features = np.zeros(
        (4, ACTION_FEATURE_SIZE), dtype=np.float32
    )

    def example(value):
      return (
          np.full(8, value, dtype=np.float32),
          np.ones(4, dtype=np.float32),
          np.full(4, 0.25, dtype=np.float32),
          None,
          1,
          0,
          None,
          None,
          None,
          None,
          action_features,
      )

    rows = az_torch.with_match_round_terminal_targets(
        [
            ([example(0.0)], np.array([0.0, 0.0, 0.0])),
            ([example(1.0)], np.array([0.0, 1.0, 0.0])),
        ],
        match_totals=np.array([10.0, 8.0, 6.0]),
        args=args,
        final_round_scores=np.array([3.0, 2.0, 1.0]),
    )

    self.assertLen(rows, 2)
    np.testing.assert_allclose(rows[0][3], [1.0, 0.8, 0.6])
    np.testing.assert_array_equal(rows[0][4], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(rows[0][12], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rows[1][3], [0.5, 0.1, 0.1])
    np.testing.assert_array_equal(rows[1][4], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(rows[1][12], [0.0, 1.0, 0.0])

  def test_full_match_replay_metadata_declares_round_local_targets(self):
    metadata = az_torch.replay_metadata_from_args(
        argparse.Namespace(full_match_training=True)
    )

    self.assertTrue(metadata["full_match_training"])
    self.assertEqual(
        metadata["full_match_terminal_target_scope"], "round"
    )

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

  def test_terminal_search_uses_official_placement_and_final_round_tiebreak(self):
    args = argparse.Namespace(
        value_scale=20.0,
        terminal_paradox_penalty=0.0,
        terminal_any_paradox_penalty=0.0,
        ordinal_value_weight=0.0,
        official_outcome_value_weight=1.0,
    )
    state = _TerminalState(
        returns=[10.0, 10.0, 8.0],
        paradoxes=[False, False, False],
    )
    state.raw_round_scores = lambda: [1.0, 3.0, 2.0]

    np.testing.assert_allclose(
        terminal_search_value(state, args),
        [0.0, 20.0, -20.0],
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

  def _competitive_gate_fixture(
      self, block_score_signs, *, candidate_paradoxes=False
  ):
    names = ["challenger", "inc0", "inc1", "inc2"]
    matches = []
    deal_blocks = {}
    match_idx = 0
    for block_idx, score_sign in enumerate(block_score_signs):
      for shift in range(len(names)):
        seated = names[shift:] + names[:shift]
        candidate_seat = seated.index("challenger")
        opponent_seat = next(
            seat for seat in range(len(names)) if seat != candidate_seat
        )
        totals = [4.0] * len(names)
        totals[candidate_seat] = 8.0 if score_sign > 0 else 0.0
        paradox_seat = candidate_seat if candidate_paradoxes else opponent_seat
        paradoxed = [seat == paradox_seat for seat in range(len(names))]
        rounds = [
            {
                "paradoxed": list(paradoxed),
                "predictions": [1] * len(names),
                "tricks_won": [1] * len(names),
                "adjacency_bonus": [
                    0.0 if seat == paradox_seat else 2.0
                    for seat in range(len(names))
                ],
                "potential_prediction_bonus": [2.0] * len(names),
            }
            for _ in range(len(names))
        ]
        matches.append((
            match_idx,
            seated,
            {
                "totals": totals,
                "final_round_returns": [1.0] * len(names),
                "paradoxes": [
                    len(names) if seat == paradox_seat else 0
                    for seat in range(len(names))
                ],
                "rounds": rounds,
            },
        ))
        deal_blocks[match_idx] = block_idx
        match_idx += 1
    return names, matches, deal_blocks

  def test_cyclic_paired_schedule_uses_each_seat_once_per_deal(self):
    args = argparse.Namespace(
        players=4,
        schedule="cyclic",
        candidate=None,
        paired_deals=2,
        matches=8,
    )
    names = ["challenger", "inc0", "inc1", "inc2"]
    lineups = full_elo.ladder_lineups(args, names)
    jobs = full_elo.match_jobs(args, names, lineups, True)

    self.assertLen(lineups, 4)
    self.assertLen(jobs, 8)
    self.assertEqual(
        [lineup.index("challenger") for lineup in lineups], [0, 3, 2, 1]
    )
    self.assertEqual([job[3] for job in jobs], [0] * 4 + [1] * 4)
    self.assertEqual([job[4] for job in jobs], [0] * 4 + [1] * 4)

  def test_cyclic_paired_deal_offset_never_reuses_previous_global_deals(self):
    args = argparse.Namespace(
        players=5,
        schedule="cyclic",
        candidate=None,
        paired_deals=2,
        deal_offset=32,
        matches=10,
    )
    names = ["challenger", "inc0", "inc1", "inc2", "inc3"]
    lineups = full_elo.ladder_lineups(args, names)
    jobs = full_elo.match_jobs(args, names, lineups, True)

    self.assertLen(jobs, 10)
    self.assertEqual([job[3] for job in jobs], [32] * 5 + [33] * 5)
    self.assertEqual([job[4] for job in jobs], [2] * 5 + [3] * 5)
    self.assertEqual([job[0] for job in jobs], list(range(10)))

  def test_deal_offset_rejects_negative_and_unpaired_schedules(self):
    names = ["challenger", "inc0", "inc1", "inc2", "inc3"]
    args = argparse.Namespace(
        players=5,
        schedule="cyclic",
        candidate=None,
        paired_deals=1,
        deal_offset=-1,
        matches=5,
    )
    with self.assertRaisesRegex(ValueError, "cannot be negative"):
      full_elo.match_jobs(args, names, [names], True)

    args.paired_deals = 0
    args.deal_offset = 32
    with self.assertRaisesRegex(ValueError, "requires --paired-deals"):
      full_elo.match_jobs(args, names, [names], True)

  def test_competitive_gate_passes_wins_despite_every_round_paradoxing(self):
    names, matches, deal_blocks = self._competitive_gate_fixture([1, 1, 1, 1])
    result = full_elo.summarize_competitive_promotion_gate(
        names,
        4,
        matches,
        "challenger",
        deal_blocks_by_match=deal_blocks,
        min_independent_deals=4,
    )

    self.assertTrue(result["eligible"])
    self.assertTrue(result["passed"])
    self.assertEqual(result["inference_unit"], "independent_common_deal_block")
    self.assertEqual(result["independent_deal_blocks"], 4)
    self.assertEqual(result["matches"], 16)
    self.assertEqual(result["pairwise"]["score_rate"], 1.0)
    self.assertEqual(result["bga_style_rank_points"]["per_match"], 6.0)
    self.assertEqual(result["candidate_own_paradox"]["round_rate"], 0.0)
    self.assertEqual(result["table_paradox"]["any_paradox_round_rate"], 1.0)
    self.assertTrue(result["table_paradox"]["diagnostic_only"])
    self.assertEqual(result["candidate_prediction"]["hit_rate"], 1.0)

  def test_competitive_gate_counts_deals_not_correlated_seat_permutations(self):
    names, matches, deal_blocks = self._competitive_gate_fixture([1, -1])
    result = full_elo.summarize_competitive_promotion_gate(
        names,
        4,
        matches,
        "challenger",
        deal_blocks_by_match=deal_blocks,
        min_independent_deals=2,
    )

    self.assertEqual(result["matches"], 8)
    self.assertEqual(result["independent_deal_blocks"], 2)
    self.assertEqual(result["pairwise"]["wins"], 12)
    self.assertEqual(result["pairwise"]["losses"], 12)
    self.assertEqual(result["pairwise"]["score_rate"], 0.5)
    self.assertFalse(result["placement_advantage_passed"])
    self.assertFalse(result["passed"])

  def test_competitive_gate_rejects_significant_own_paradox_regression(self):
    names, matches, deal_blocks = self._competitive_gate_fixture(
        [1, 1, 1, 1], candidate_paradoxes=True
    )
    result = full_elo.summarize_competitive_promotion_gate(
        names,
        4,
        matches,
        "challenger",
        deal_blocks_by_match=deal_blocks,
        min_independent_deals=4,
    )

    self.assertTrue(result["placement_advantage_passed"])
    self.assertEqual(result["candidate_own_paradox"]["round_rate"], 1.0)
    self.assertEqual(result["reference_personal_paradox"]["round_rate"], 0.0)
    self.assertEqual(result["candidate_own_paradox"]["points_lost"], 256.0)
    self.assertFalse(result["own_paradox_guard_passed"])
    self.assertFalse(result["passed"])

  def test_competitive_gate_requires_each_deal_to_cover_all_seats(self):
    names, matches, deal_blocks = self._competitive_gate_fixture([1, 1, 1])
    matches = [row for row in matches if row[1].index("challenger") != 3]
    result = full_elo.summarize_competitive_promotion_gate(
        names,
        4,
        matches,
        "challenger",
        deal_blocks_by_match=deal_blocks,
        min_independent_deals=3,
    )

    self.assertFalse(result["eligible"])
    self.assertFalse(result["seat_balanced_within_each_deal"])
    self.assertFalse(result["passed"])

  def test_competitive_gate_uses_final_round_score_for_tied_match_totals(self):
    names, matches, deal_blocks = self._competitive_gate_fixture([1, 1])
    for _match_idx, seated, match in matches:
      candidate_seat = seated.index("challenger")
      match["totals"] = [4.0] * 4
      match["final_round_returns"] = [1.0] * 4
      match["final_round_returns"][candidate_seat] = 2.0
    result = full_elo.summarize_competitive_promotion_gate(
        names,
        4,
        matches,
        "challenger",
        deal_blocks_by_match=deal_blocks,
        min_independent_deals=2,
    )

    self.assertEqual(result["pairwise"]["score_rate"], 1.0)
    self.assertEqual(result["official_score_advantage"]["mean"], 0.0)
    self.assertTrue(result["official_score_guard_passed"])
    self.assertTrue(result["passed"])

  def test_bga_rank_points_use_all_opponents_not_only_reference_subset(self):
    names, matches, deal_blocks = self._competitive_gate_fixture([1, 1])
    for _match_idx, seated, match in matches:
      match["totals"] = [
          {
              "challenger": 8.0,
              "inc0": 4.0,
              "inc1": 10.0,
              "inc2": 12.0,
          }[name]
          for name in seated
      ]
    result = full_elo.summarize_competitive_promotion_gate(
        names,
        4,
        matches,
        "challenger",
        reference_names=["inc0"],
        deal_blocks_by_match=deal_blocks,
        min_independent_deals=2,
    )

    self.assertEqual(result["pairwise"]["score_rate"], 1.0)
    self.assertEqual(result["mean_official_placement"], 3.0)
    self.assertEqual(result["bga_style_rank_points"]["per_match"], 2.0)

  def _competitive_posterior_bot_and_state(self, objective="competitive"):
    args = full_elo._model_args(4)
    args.survival_value_base_mode = "policy"
    args.survival_value_analytic_posterior = True
    args.survival_value_analytic_objective = objective
    args.survival_value_analytic_min_trick_number = 2
    args.survival_value_analytic_min_advantage = 0.04
    bot = full_elo.AZValueShieldPolicyBot(None, None, "competitive", None, args)
    state = _ShieldState([1, 2, 3])
    state._trick_number = 5
    state._predictions = [1, 1, 1, 1]
    state._tricks_won = np.array([1, 0, 0, 0], dtype=np.int32)
    scores = {
        1: {
            "sum_failure_probability": 0.30,
            "max_failure_probability": 0.20,
            "min_expected_surplus": 1.0,
            "own_margin": 2.0,
        },
        2: {
            "sum_failure_probability": 0.55,
            "max_failure_probability": 0.45,
            "min_expected_surplus": 0.5,
            "own_margin": 2.0,
        },
        3: {
            "sum_failure_probability": 0.15,
            "max_failure_probability": 0.10,
            "min_expected_surplus": 2.0,
            "own_margin": 2.0,
        },
    }
    return bot, state, scores

  def test_competitive_posterior_pressures_opponents_after_own_bid(self):
    bot, state, scores = self._competitive_posterior_bot_and_state()
    with mock.patch.object(
        full_elo,
        "public_posterior_action_score",
        side_effect=lambda _state, _player, action: scores[action],
    ), mock.patch.object(
        full_elo,
        "action_liveness_certificate",
        return_value={
            "own_feasible": True,
            "lost_led_token": False,
            "over_target_would_win": False,
        },
    ):
      self.assertEqual(
          bot._analytic_posterior_action(state, 0, [1, 2, 3], 1), 2
      )
    self.assertEqual(
        bot._decision_stats["value_shield_analytic_competitive_overrides"], 1
    )

  def test_legacy_table_posterior_still_protects_opponent_exits(self):
    bot, state, scores = self._competitive_posterior_bot_and_state(
        objective="table_survival"
    )
    with mock.patch.object(
        full_elo,
        "public_posterior_action_score",
        side_effect=lambda _state, _player, action: scores[action],
    ), mock.patch.object(
        full_elo,
        "action_liveness_certificate",
        return_value={"own_feasible": True, "lost_led_token": False},
    ):
      self.assertEqual(
          bot._analytic_posterior_action(state, 0, [1, 2, 3], 1), 3
      )

  def test_competitive_posterior_requires_completed_own_prediction(self):
    bot, state, _scores = self._competitive_posterior_bot_and_state()
    state._tricks_won[0] = 0
    with mock.patch.object(
        full_elo, "public_posterior_action_score"
    ) as score_action:
      self.assertEqual(
          bot._analytic_posterior_action(state, 0, [1, 2, 3], 1), 1
      )
      score_action.assert_not_called()
    self.assertEqual(
        bot._decision_stats["value_shield_analytic_blocked_bid_incomplete"], 1
    )

  def test_competitive_posterior_skips_unsafe_stronger_attack(self):
    bot, state, scores = self._competitive_posterior_bot_and_state()
    scores[2]["own_margin"] = 1.0
    scores[3]["sum_failure_probability"] = 0.40
    scores[3]["max_failure_probability"] = 0.35
    with mock.patch.object(
        full_elo,
        "public_posterior_action_score",
        side_effect=lambda _state, _player, action: scores[action],
    ), mock.patch.object(
        full_elo,
        "action_liveness_certificate",
        return_value={
            "own_feasible": True,
            "lost_led_token": False,
            "over_target_would_win": False,
        },
    ):
      self.assertEqual(
          bot._analytic_posterior_action(state, 0, [1, 2, 3], 1), 3
      )
    self.assertEqual(
        bot._decision_stats["value_shield_analytic_blocked_own_margin"], 1
    )

  def test_competitive_posterior_rejects_bid_breaking_extra_trick(self):
    bot, state, scores = self._competitive_posterior_bot_and_state()
    with mock.patch.object(
        full_elo,
        "public_posterior_action_score",
        side_effect=lambda _state, _player, action: scores[action],
    ), mock.patch.object(
        full_elo,
        "action_liveness_certificate",
        side_effect=lambda _state, _player, action, **_kwargs: {
            "own_feasible": True,
            "lost_led_token": False,
            "over_target_would_win": action == 2,
        },
    ):
      self.assertEqual(
          bot._analytic_posterior_action(state, 0, [1, 2, 3], 1), 1
      )
    self.assertEqual(
        bot._decision_stats["value_shield_analytic_blocked_over_target"], 1
    )

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
    self.assertFalse(result["passed"])
    self.assertTrue(result["point_estimate_below_threshold"])
    self.assertEqual(result["gate_basis"], "any_paradox_round_rate_ci95_upper")
    self.assertEqual(result["ci95_method"], "wilson_score")
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

  def test_homogeneous_paradox_gate_pass_requires_ci_upper_below_threshold(self):
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
        games_by_bot={"c0": 600, "c1": 600, "c2": 600},
        paradoxes_by_bot={"c0": 180, "c1": 180, "c2": 180},
        neural_bots=neural_bots,
        threshold=0.40,
    )

    self.assertTrue(result["passed"])
    self.assertTrue(result["ci95_upper_below_threshold"])
    self.assertLess(result["any_paradox_round_rate_ci95"][1], 0.40)

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
    self.assertEqual(
        result["hand_round_keys"],
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
    )
    self.assertEqual(result["any_paradox_outcomes"], [1, 0, 1, 1, 0, 0])
    self.assertEqual(result["paradox_seat_masks"], [3, 0, 2, 4, 0, 0])

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

  def test_neural_bot_overrides_allow_model_prediction(self):
    payload, overrides = full_elo._split_payload_options(
        "checkpoint.pt:root_rollout?neural_prediction_mode=model,"
        "root_rollout_phases=prediction"
    )

    self.assertEqual(payload, "checkpoint.pt:root_rollout")
    self.assertEqual(overrides["neural_prediction_mode"], "model")
    self.assertEqual(overrides["root_rollout_phases"], "prediction")

  def test_dedicated_prediction_head_bypasses_card_play_q_reranking(self):
    game = az_torch.pyspiel.load_game(
        "python_quantum_cat", {"players": 5, "start_player": 0}
    )
    state = game.new_initial_state()
    state._phase = 2
    state._current_player = 0
    state._hands = [np.ones(9, dtype=np.int32) for _ in range(5)]
    state._predictions = [-1] * 5
    args = full_elo._model_args(players=5)
    args.neural_prediction_mode = "model"
    args.action_value_selection_weight = 1.0
    args.action_paradox_rerank_mode = "relative"
    model = mock.Mock(separate_prediction_policy_head=True)
    bot = full_elo.AZQPolicyBot(model, "candidate", "cpu", args)
    policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
    policy[101] = 0.05
    policy[102] = 0.15
    policy[103] = 0.75
    policy[104] = 0.05
    values = np.zeros_like(policy)
    values[101] = 100.0

    with mock.patch.object(
        full_elo,
        "model_policy_value_action_aux",
        return_value=(policy, None, values, np.zeros_like(policy), None),
    ):
      action = bot.step(state, 0)

    self.assertEqual(action, 103)

  def test_neural_bot_overrides_allow_terminal_search_shaping(self):
    payload, overrides = full_elo._split_payload_options(
        "checkpoint.pt:belief?terminal_paradox_penalty=7.5,"
        "terminal_any_paradox_penalty=20.0,"
        "ordinal_value_weight=0.25,"
        "paradox_value_penalty=3.0,"
        "belief_samples=16,"
        "belief_sims=64"
    )

    self.assertEqual(payload, "checkpoint.pt:belief")
    self.assertEqual(overrides["terminal_paradox_penalty"], 7.5)
    self.assertEqual(overrides["terminal_any_paradox_penalty"], 20.0)
    self.assertEqual(overrides["ordinal_value_weight"], 0.25)
    self.assertEqual(overrides["paradox_value_penalty"], 3.0)
    self.assertEqual(overrides["belief_samples"], 16)
    self.assertEqual(overrides["belief_sims"], 64)

    model_args = full_elo._model_args(players=4)
    self.assertEqual(model_args.terminal_paradox_penalty, 0.0)
    self.assertEqual(model_args.terminal_any_paradox_penalty, 0.0)
    self.assertEqual(model_args.ordinal_value_weight, 0.0)
    self.assertEqual(model_args.paradox_value_penalty, 0.0)
    self.assertEqual(model_args.belief_samples, 0)
    self.assertEqual(model_args.belief_sims, 0)

    full_elo._apply_neural_bot_overrides(model_args, overrides)
    self.assertEqual(model_args.terminal_paradox_penalty, 7.5)
    self.assertEqual(model_args.terminal_any_paradox_penalty, 20.0)
    self.assertEqual(model_args.ordinal_value_weight, 0.25)
    self.assertEqual(model_args.paradox_value_penalty, 3.0)
    self.assertEqual(model_args.belief_samples, 16)
    self.assertEqual(model_args.belief_sims, 64)

  def test_full_match_prediction_reaches_opted_in_neural_bot(self):
    class _PredictionOnlyState:
      def __init__(self):
        self._phase = 2
        self._terminal = False
        self._has_paradoxed = [False]
        self.applied_action = None

      def is_terminal(self):
        return self._terminal

      def is_chance_node(self):
        return False

      def current_player(self):
        return 0

      def legal_actions(self, player):
        del player
        return [101, 102, 103]

      def apply_action(self, action):
        self.applied_action = int(action)
        self._terminal = True

      def returns(self):
        return [0.0]

    class _PredictionOnlyGame:
      def __init__(self):
        self.state = _PredictionOnlyState()

      def new_initial_state(self):
        return self.state

    game = _PredictionOnlyGame()
    bot = _ActionBot(103)
    bot.uses_neural_prediction = True

    with mock.patch.object(
        full_elo, "shared_discard_action", return_value=None
    ), mock.patch.object(
        full_elo, "shared_prediction_action", return_value=101
    ) as shared_prediction:
      full_elo.play_round(game, [bot], seed=1)

    self.assertEqual(game.state.applied_action, 103)
    shared_prediction.assert_not_called()

  def test_improve_graft_overrides_allow_led_color_filter(self):
    payload, overrides = full_elo._split_payload_options(
        "checkpoint.pt:improve_graft?improve_model=improve.pt,"
        "improve_led_colors=B|Y|G,"
        "improve_candidate_colors=R,"
        "improve_require_base_off_led=true,"
        "improve_require_responsible_red_win=true,"
        "improve_threshold=0.97"
    )

    self.assertEqual(payload, "checkpoint.pt:improve_graft")
    self.assertEqual(overrides["improve_model"], "improve.pt")
    self.assertEqual(overrides["improve_led_colors"], "B|Y|G")
    self.assertEqual(overrides["improve_candidate_colors"], "R")
    self.assertTrue(overrides["improve_require_base_off_led"])
    self.assertTrue(overrides["improve_require_responsible_red_win"])
    self.assertEqual(overrides["improve_threshold"], 0.97)

  def test_neural_bot_overrides_allow_root_rollout_paradox_objective(self):
    payload, overrides = full_elo._split_payload_options(
        "checkpoint.pt:root_rollout?root_rollout_objective=paradox_then_score,"
        "root_rollout_paradox_scope=any,"
        "root_rollout_min_paradox_improvement=0.5,"
        "root_rollout_teacher_candidates=true,"
        "root_rollout_led_colors=None,"
        "root_rollout_min_trick_number=5,"
        "root_rollout_max_trick_number=7,"
        "root_rollout_audit_samples=4,"
        "root_rollout_audit_rollouts=2,"
        "root_rollout_audit_passes=3,"
        "root_rollout_min_audit_paradox_improvement=0.25,"
        "root_rollout_max_audit_paired_binomial_p=0.1,"
        "root_rollout_base_mode=liveness_shield,"
        "root_rollout_continuation_bot=heuristic_safe3,"
        "root_rollout_continuation_mode=liveness_shield,"
        "root_rollout_include_continuation_candidate=false"
    )

    self.assertEqual(payload, "checkpoint.pt:root_rollout")
    self.assertEqual(overrides["root_rollout_objective"], "paradox_then_score")
    self.assertEqual(overrides["root_rollout_paradox_scope"], "any")
    self.assertEqual(overrides["root_rollout_min_paradox_improvement"], 0.5)
    self.assertTrue(overrides["root_rollout_teacher_candidates"])
    self.assertEqual(overrides["root_rollout_led_colors"], "None")
    self.assertEqual(overrides["root_rollout_min_trick_number"], 5)
    self.assertEqual(overrides["root_rollout_max_trick_number"], 7)
    self.assertEqual(overrides["root_rollout_audit_samples"], 4)
    self.assertEqual(overrides["root_rollout_audit_rollouts"], 2)
    self.assertEqual(overrides["root_rollout_audit_passes"], 3)
    self.assertEqual(
        overrides["root_rollout_min_audit_paradox_improvement"], 0.25
    )
    self.assertEqual(
        overrides["root_rollout_max_audit_paired_binomial_p"], 0.1
    )
    self.assertEqual(overrides["root_rollout_base_mode"], "liveness_shield")
    self.assertEqual(overrides["root_rollout_continuation_bot"], "heuristic_safe3")
    self.assertEqual(
        overrides["root_rollout_continuation_mode"], "liveness_shield"
    )
    self.assertFalse(overrides["root_rollout_include_continuation_candidate"])

  def test_neural_bot_overrides_allow_belief_table_liveness(self):
    payload, overrides = full_elo._split_payload_options(
        "checkpoint.pt:belief_table_liveness?"
        "belief_table_liveness_base_mode=policy,"
        "belief_table_liveness_samples=16,"
        "belief_table_liveness_min_valid_samples=16,"
        "belief_table_liveness_min_trick_number=5,"
        "belief_table_liveness_max_trick_number=7,"
        "belief_table_liveness_aggregation=teacher_vote,"
        "belief_table_liveness_min_paired_lcb=0.1,"
        "belief_table_liveness_paired_lcb_z=1.0,"
        "belief_table_liveness_min_best_share=0.5"
    )

    self.assertEqual(payload, "checkpoint.pt:belief_table_liveness")
    self.assertEqual(overrides["belief_table_liveness_base_mode"], "policy")
    self.assertEqual(overrides["belief_table_liveness_samples"], 16)
    self.assertEqual(overrides["belief_table_liveness_min_valid_samples"], 16)
    self.assertEqual(overrides["belief_table_liveness_min_trick_number"], 5)
    self.assertEqual(overrides["belief_table_liveness_max_trick_number"], 7)
    self.assertEqual(
        overrides["belief_table_liveness_aggregation"], "teacher_vote"
    )
    self.assertEqual(overrides["belief_table_liveness_min_paired_lcb"], 0.1)
    self.assertEqual(overrides["belief_table_liveness_paired_lcb_z"], 1.0)
    self.assertEqual(overrides["belief_table_liveness_min_best_share"], 0.5)

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
        "survival_value_base_mode=liveness_shield,"
        "survival_value_threshold=0.62,"
        "survival_value_phases=play,"
        "survival_value_max_actions=5,"
        "survival_value_scope=acting"
    )

    self.assertEqual(payload, "policy.pt:value_shield")
    self.assertEqual(overrides["survival_checkpoint"], "survive.pt")
    self.assertEqual(overrides["survival_value_base_mode"], "liveness_shield")
    self.assertEqual(overrides["survival_value_threshold"], 0.62)
    self.assertEqual(overrides["survival_value_phases"], "play")
    self.assertEqual(overrides["survival_value_max_actions"], 5)
    self.assertEqual(overrides["survival_value_scope"], "acting")

  def test_belief_search_overrides_allow_nonterminal_survival_checkpoint(self):
    payload, overrides = full_elo._split_payload_options(
        "policy.pt:belief?survival_checkpoint=survive.pt,"
        "survival_value_search_penalty=20.0,"
        "belief_samples=4,belief_sims=64"
    )

    self.assertEqual(payload, "policy.pt:belief")
    self.assertEqual(overrides["survival_checkpoint"], "survive.pt")
    self.assertEqual(overrides["survival_value_search_penalty"], 20.0)

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

  def test_root_rollout_fresh_audit_requires_paired_improvement(self):
    scored = {
        1: {"paradox": 0.75, "score": 2.0},
        2: {"paradox": 0.50, "score": 1.0},
    }

    self.assertTrue(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            scored, 1, 2, 0.25
        )
    )
    self.assertFalse(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            scored, 1, 2, 0.50
        )
    )
    self.assertFalse(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            {1: scored[1]}, 1, 2, 0.0
        )
    )

  def test_counterfactual_official_objective_uses_final_placement_tiebreak(self):
    class FinalRound:

      def __init__(self):
        self._match_round = 4
        self._round_start_player = 4
        self._match_totals = np.array([30, 29, 8, 7, 6], dtype=np.float32)

      def clone(self):
        return FinalRound()

      def apply_action(self, unused_action):
        return None

      def is_terminal(self):
        return True

      def raw_round_scores(self):
        return np.array([0, 1, 0, 0, 0], dtype=np.float32)

    args = argparse.Namespace(
        players=5,
        value_scale=20.0,
        counterfactual_rollout_max_plies=0,
        counterfactual_action_value_objective="official",
    )
    with mock.patch.object(az_torch, "play_rollout_to_terminal"):
      official = az_torch.rollout_match_score_after_action(
          FinalRound(), None, 0, 1, [], [], None, None, args, None
      )
      args.counterfactual_action_value_objective = "score"
      raw = az_torch.rollout_match_score_after_action(
          FinalRound(), None, 0, 1, [], [], None, None, args, None
      )

    self.assertEqual(official, 10.0)
    self.assertEqual(raw, 30.0)
    args.counterfactual_action_value_objective = "official"
    self.assertEqual(
        az_torch.counterfactual_action_value_report_kind(args),
        "counterfactual_full_match_official_placement",
    )

  def test_root_rollout_fresh_audit_can_require_official_score_improvement(self):
    scored = {
        1: {"paradox": 0.0, "score": 10.0},
        2: {"paradox": 0.0, "score": -5.0},
        3: {"paradox": 0.0, "score": 10.5},
    }

    self.assertTrue(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            scored, 1, 2, 0.0
        )
    )
    self.assertFalse(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            scored, 1, 2, 0.0, 1.0, 0.0
        )
    )
    self.assertTrue(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            scored, 1, 3, 0.0, 1.0, 0.25
        )
    )
    self.assertFalse(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            scored, 1, 3, 0.0, 1.0, 0.75
        )
    )

  def test_root_rollout_fresh_audit_can_require_exact_paired_evidence(self):
    strong = {
        1: {
            "paradox": 1.0,
            "score": 2.0,
            "paradox_outcomes": [1.0, 1.0, 1.0, 1.0],
        },
        2: {
            "paradox": 0.0,
            "score": 1.0,
            "paradox_outcomes": [0.0, 0.0, 0.0, 0.0],
        },
    }
    weak = {
        1: {
            "paradox": 0.75,
            "score": 2.0,
            "paradox_outcomes": [1.0, 1.0, 1.0, 0.0],
        },
        2: {
            "paradox": 0.25,
            "score": 1.0,
            "paradox_outcomes": [0.0, 0.0, 0.0, 1.0],
        },
    }

    self.assertTrue(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            strong, 1, 2, 0.25, 0.10
        )
    )
    self.assertFalse(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            weak, 1, 2, 0.25, 0.10
        )
    )

  def test_root_rollout_exact_evidence_counts_worlds_not_nested_rollouts(self):
    duplicated_three_worlds = {
        1: {
            "paradox": 1.0,
            "score": 2.0,
            "paradox_outcomes": [1.0] * 6,
            "paradox_world_rates": [1.0] * 3,
            "rollout_worlds": 3,
            "rollouts_per_world": 2,
        },
        2: {
            "paradox": 0.0,
            "score": 1.0,
            "paradox_outcomes": [0.0] * 6,
            "paradox_world_rates": [0.0] * 3,
            "rollout_worlds": 3,
            "rollouts_per_world": 2,
        },
    }
    self.assertFalse(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            duplicated_three_worlds, 1, 2, 0.25, 0.10
        )
    )

    for result in duplicated_three_worlds.values():
      result["paradox_world_rates"].append(float(result["paradox"]))
      result["rollout_worlds"] = 4
    self.assertTrue(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            duplicated_three_worlds, 1, 2, 0.25, 0.10
        )
    )
    self.assertFalse(
        full_elo.AZRootRolloutBot._audit_override_allowed(
            {
                1: {"paradox": 1.0, "score": 2.0},
                2: {"paradox": 0.0, "score": 1.0},
            },
            1,
            2,
            0.25,
            0.10,
        )
    )

  def test_root_rollout_exact_binomial_matches_replay_teacher(self):
    for saved, harmed in ((0, 0), (1, 0), (4, 0), (3, 1), (7, 2)):
      self.assertAlmostEqual(
          full_elo.AZRootRolloutBot._one_sided_binomial_tail(saved, harmed),
          best_of_rollouts_replay._one_sided_binomial_tail(saved, harmed),
      )

  def test_root_rollout_requires_every_fresh_audit_pass(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_max_audit_paired_binomial_p=0.10,
        root_rollout_samples=4,
        root_rollouts=1,
        root_rollout_objective="paradox_then_score",
        root_rollout_paradox_scope="any",
        root_rollout_min_paradox_improvement=0.50,
        root_rollout_audit_samples=4,
        root_rollout_audit_rollouts=1,
        root_rollout_audit_passes=2,
        root_rollout_min_audit_paradox_improvement=0.25,
        root_rollout_record_replay_samples=True,
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(
        _phase=3,
        _trick_number=5,
        legal_actions=lambda player: [1, 2],
        observation_tensor=lambda player: [0.0, 1.0],
        num_distinct_actions=lambda: 3,
    )
    selection = {
        1: {"paradox": 1.0, "score": 2.0},
        2: {"paradox": 0.0, "score": 1.0},
    }
    strong_audit = {
        1: {
            "paradox": 1.0,
            "score": 2.0,
            "paradox_outcomes": [1.0, 1.0, 1.0, 1.0],
        },
        2: {
            "paradox": 0.0,
            "score": 1.0,
            "paradox_outcomes": [0.0, 0.0, 0.0, 0.0],
        },
    }
    weak_audit = {
        1: {
            "paradox": 0.75,
            "score": 2.0,
            "paradox_outcomes": [1.0, 1.0, 1.0, 0.0],
        },
        2: {
            "paradox": 0.25,
            "score": 1.0,
            "paradox_outcomes": [0.0, 0.0, 0.0, 1.0],
        },
    }
    policy = np.zeros(3, dtype=np.float32)
    policy[1] = 0.9
    policy[2] = 0.1
    with mock.patch.object(
        full_elo, "_shared_discard_decision", return_value=None
    ), mock.patch.object(
        full_elo, "model_policy_value", return_value=(policy, np.zeros(3))
    ), mock.patch.object(
        full_elo, "_record_decision"
    ), mock.patch.object(
        bot, "_enabled_for_phase", return_value=True
    ), mock.patch.object(
        bot, "_candidate_actions", return_value=[1, 2]
    ), mock.patch.object(
        bot, "_sample_infostate_worlds", return_value=[object()] * 4
    ), mock.patch.object(
        bot, "_score_actions", side_effect=[selection, strong_audit, weak_audit]
    ), mock.patch.object(
        full_elo,
        "action_feature_matrix",
        return_value=np.arange(3 * 214, dtype=np.float32).reshape(3, 214),
    ):
      selected = bot.step(state, player=0)

    self.assertEqual(selected, 1)
    self.assertEqual(bot._decision_stats["root_rollout_audit_considered"], 1)
    self.assertEqual(bot._decision_stats["root_rollout_audit_passes_passed"], 1)
    self.assertEqual(bot._decision_stats["root_rollout_audit_blocked"], 1)
    self.assertEqual(bot._decision_stats["root_rollout_audit_passed"], 0)
    self.assertEqual(bot._decision_stats["root_rollout_overrides"], 0)
    samples = bot._decision_stats["root_rollout_replay_samples"]
    self.assertEqual(len(samples), 1)
    self.assertTrue(samples[0]["information_safe"])
    self.assertFalse(samples[0]["audit_accepted"])
    self.assertEqual(samples[0]["baseline_action"], 1)
    self.assertEqual(samples[0]["proposed_action"], 2)
    self.assertEqual(samples[0]["selected_action"], 1)
    self.assertEqual(samples[0]["audit_contract"]["baseline_mode"], "policy")
    self.assertEqual(samples[0]["audit_contract"]["paradox_scope"], "any")
    self.assertEqual(len(samples[0]["legal_action_features"]), 2)
    self.assertEqual(len(samples[0]["legal_action_features"][0]), 214)
    self.assertEqual(len(samples[0]["audit_passes"]), 2)

  def test_root_rollout_replay_cap_never_drops_accepted_corrections(self):
    rejected = [
        {"audit_accepted": False, "index": index} for index in range(105)
    ]
    accepted = [
        {"audit_accepted": True, "index": 1000 + index}
        for index in range(3)
    ]
    retained = full_elo._trim_root_rollout_replay_samples(
        rejected[:50] + accepted[:1] + rejected[50:] + accepted[1:]
    )
    self.assertEqual(
        sum(not sample["audit_accepted"] for sample in retained), 100
    )
    self.assertEqual(
        [sample["index"] for sample in retained if sample["audit_accepted"]],
        [1000, 1001, 1002],
    )

    target = {"root_rollout_replay_samples": rejected[:100]}
    full_elo._merge_decision_stats(target, {
        "root_rollout_replay_samples": rejected[100:] + accepted,
    })
    merged = target["root_rollout_replay_samples"]
    self.assertEqual(sum(not row["audit_accepted"] for row in merged), 100)
    self.assertEqual(sum(row["audit_accepted"] for row in merged), 3)

  def test_root_rollout_rng_is_scoped_to_acting_information_state(self):
    args = argparse.Namespace(
        players=3,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_seed_salt=17,
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(
        observation_tensor=lambda player: [float(player), 1.0, 0.0]
    )

    np.random.seed(991)
    np.random.random(100)
    expected_global = np.random.random(3)
    np.random.seed(991)
    first_seed = bot._reset_search_rng(state, 0, [3, 1, 2])
    first_draws = bot._search_rng.random(5)
    np.random.random(100)
    second_seed = bot._reset_search_rng(state, 0, [1, 2, 3])
    second_draws = bot._search_rng.random(5)

    self.assertEqual(first_seed, second_seed)
    np.testing.assert_array_equal(first_draws, second_draws)
    np.testing.assert_array_equal(np.random.random(3), expected_global)
    args.root_rollout_seed_salt = 18
    self.assertNotEqual(
        bot._reset_search_rng(state, 0, [1, 2, 3]), first_seed
    )

  def test_root_rollout_belief_resampling_fails_closed(self):
    args = argparse.Namespace(
        players=3,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(
        legal_actions=lambda player: [1, 2],
        observation_tensor=lambda player: [0.0, 1.0],
        resample_from_infostate=lambda player, sampler: (_ for _ in ()).throw(
            ValueError("cannot resample")
        ),
    )

    self.assertEqual(bot._sample_infostate_worlds(state, 0, 4), [])
    self.assertEqual(bot._decision_stats["root_rollout_belief_sample_failures"], 1)

  def test_root_rollout_policy_weighted_beliefs_preserve_global_rng(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        belief_source="policy_weighted",
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
    )
    model = object()
    device = torch.device("cpu")
    bot = full_elo.AZRootRolloutBot(model, "root", device, args)
    state = argparse.Namespace(
        legal_actions=lambda player: [1, 2],
        observation_tensor=lambda player: [0.0, 1.0],
    )

    np.random.seed(991)
    expected_global = np.random.random(3)
    np.random.seed(991)
    with mock.patch.object(
        full_elo,
        "sampled_belief_states_for_policy",
        return_value=[state, state],
    ) as sampler:
      worlds = bot._sample_infostate_worlds(state, 0, 2)

    self.assertEqual(worlds, [state, state])
    np.testing.assert_array_equal(np.random.random(3), expected_global)
    sampler.assert_called_once_with(
        state,
        0,
        2,
        args,
        model,
        device,
        20.0,
        context="eval",
    )

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

  def test_root_rollout_can_protect_liveness_shield_baseline(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        root_rollout_base_mode="liveness_shield",
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_max_actions=2,
        root_rollout_top_policy=1,
        root_rollout_teacher_candidates=False,
        root_rollout_include_bots="",
        root_rollout_include_continuation_candidate=False,
        liveness_shield_base_mode="policy",
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(clone=lambda: state)
    legal = [1, 2, 3, 4]
    policy = np.zeros(10, dtype=np.float32)
    policy[1] = 1.0

    with mock.patch.object(bot._base_neural_bot, "step", return_value=4):
      baseline = bot._base_action(state, 0, legal, policy)

    self.assertEqual(baseline, 4)
    self.assertEqual(
        bot._candidate_actions(
            state, 0, legal, policy, baseline_action=baseline
        ),
        [1, 4],
    )
    self.assertEqual(bot._decision_stats["root_rollout_base_mode"],
                     "liveness_shield")

  def test_root_rollout_can_protect_competitive_tactics_baseline(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        root_rollout_base_mode="competitive_tactics",
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_max_actions=2,
        root_rollout_top_policy=1,
        root_rollout_teacher_candidates=False,
        root_rollout_include_bots="",
        root_rollout_include_continuation_candidate=False,
        competitive_tactics_base_mode="liveness_shield",
        liveness_shield_base_mode="policy",
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(clone=lambda: state)
    legal = [1, 2, 3, 4]
    policy = np.zeros(10, dtype=np.float32)
    policy[1] = 1.0

    with mock.patch.object(bot._base_neural_bot, "step", return_value=4):
      baseline = bot._base_action(state, 0, legal, policy)

    self.assertIsInstance(bot._base_neural_bot, full_elo.AZCompetitiveTacticsBot)
    self.assertEqual(baseline, 4)
    self.assertEqual(
        bot._candidate_actions(
            state, 0, legal, policy, baseline_action=baseline
        ),
        [1, 4],
    )
    self.assertEqual(
        bot._decision_stats["root_rollout_base_mode"], "competitive_tactics"
    )

  def test_root_rollout_preserves_competitive_tactics_prediction_handler(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        root_rollout_base_mode="competitive_tactics",
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        competitive_tactics_base_mode="policy",
        competitive_tactics_prediction_style="conservative",
    )

    bot = full_elo.AZRootRolloutBot(None, "root", None, args)

    self.assertTrue(bot._base_neural_bot.uses_neural_prediction)
    self.assertTrue(bot.uses_neural_prediction)

  def test_root_rollout_can_preserve_existing_competitive_tactical_corrections(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        root_rollout_base_mode="competitive_tactics",
        root_rollout_preserve_tactical_overrides=True,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        competitive_tactics_base_mode="policy",
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)

    bot._base_neural_bot.last_tactical_override = True
    self.assertTrue(bot._preserves_tactical_override())

    bot._base_neural_bot.last_tactical_override = False
    self.assertFalse(bot._preserves_tactical_override())

    bot._base_neural_bot.last_tactical_override = True
    args.root_rollout_preserve_tactical_overrides = False
    self.assertFalse(bot._preserves_tactical_override())

  def test_root_rollout_rejects_lower_own_exit_margin_before_search(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        root_rollout_base_mode="policy",
        root_rollout_preserve_own_liveness=True,
        root_rollout_max_baseline_risk_increase=-1.0,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace()
    rows = {
        1: {"own_feasible": True, "own_buffer_deficit": 0},
        2: {"own_feasible": True, "own_buffer_deficit": 0},
    }

    with mock.patch.object(
        full_elo, "action_liveness_certificate",
        side_effect=lambda _state, _player, action, **_kwargs: rows[action],
    ), mock.patch.object(
        full_elo.liveness_oracle, "view_from_state", return_value=object()
    ), mock.patch.object(
        full_elo.liveness_oracle, "post_play_margin",
        side_effect=lambda _view, action: 2 if action == 1 else 1,
    ):
      self.assertEqual(bot._safe_candidate_actions(state, 0, [1, 2], 1), [1])

    self.assertEqual(bot._decision_stats["root_rollout_blocked_own_liveness"], 1)

  def test_root_rollout_rejects_candidate_above_incumbent_personal_risk(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        root_rollout_base_mode="policy",
        root_rollout_preserve_own_liveness=False,
        root_rollout_max_baseline_risk_increase=0.0,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(num_distinct_actions=lambda: 4)
    risks = np.asarray([0.0, 0.10, 0.11, 0.05], dtype=np.float32)

    with mock.patch.object(
        full_elo, "model_policy_value_action_aux",
        return_value=(None, None, None, risks, None),
    ):
      self.assertEqual(
          bot._safe_candidate_actions(state, 0, [1, 2, 3], 1), [1, 3]
      )

    self.assertEqual(bot._decision_stats["root_rollout_blocked_baseline_risk"], 1)

  def test_root_rollout_can_match_teacher_candidate_buckets(self):
    args = argparse.Namespace(
        players=3,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_max_actions=4,
        root_rollout_top_policy=1,
        root_rollout_teacher_candidates=True,
        root_rollout_include_bots="",
        root_rollout_include_continuation_candidate=False,
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(_led_color="B", _num_card_types=6)
    legal = [1, 6, 12, 13, 18]
    policy = np.zeros(24, dtype=np.float32)
    policy[12] = 0.9
    policy[13] = 0.4
    policy[18] = 0.3
    policy[6] = 0.2
    policy[1] = 0.1

    self.assertEqual(
        bot._candidate_actions(state, 0, legal, policy), [1, 6, 12, 13]
    )

  def test_root_rollout_phase_filter_honors_trick_window(self):
    args = argparse.Namespace(
        players=3,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_phases="play",
        root_rollout_led_colors="",
        root_rollout_min_trick_number=5,
        root_rollout_max_trick_number=7,
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(_phase=3, _trick_number=5)

    self.assertTrue(bot._enabled_for_phase(state))
    state._trick_number = 4
    self.assertFalse(bot._enabled_for_phase(state))
    state._trick_number = 8
    self.assertFalse(bot._enabled_for_phase(state))
    state._phase = 2
    state._trick_number = 5
    self.assertFalse(bot._enabled_for_phase(state))

  def test_root_rollout_phase_filter_honors_public_match_round_window(self):
    args = argparse.Namespace(
        players=5,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_phases="play",
        root_rollout_led_colors="",
        root_rollout_min_trick_number=3,
        root_rollout_max_trick_number=7,
        root_rollout_min_match_round=4,
        root_rollout_max_match_round=4,
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(_phase=3, _trick_number=5, _match_round=4)

    self.assertTrue(bot._enabled_for_phase(state))
    state._match_round = 3
    self.assertFalse(bot._enabled_for_phase(state))
    state._match_round = 5
    self.assertFalse(bot._enabled_for_phase(state))

  def test_root_rollout_can_continue_with_actual_competitive_tactics(self):
    args = argparse.Namespace(
        players=5,
        value_scale=20.0,
        root_rollout_base_mode="competitive_tactics",
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="competitive_tactics",
        competitive_tactics_base_mode="policy",
    )

    bot = full_elo.AZRootRolloutBot(None, "root", None, args)

    self.assertIsInstance(
        bot._continuation_neural_bot, full_elo.AZCompetitiveTacticsBot
    )
    self.assertEqual(bot._continuation_mode, "competitive_tactics")

  def test_root_rollout_led_filter_can_target_leading_states_only(self):
    args = argparse.Namespace(
        players=4,
        root_rollout_continuation_bot="",
        root_rollout_continuation_mode="",
        root_rollout_phases="play",
        root_rollout_led_colors="None",
        root_rollout_min_trick_number=0,
        root_rollout_max_trick_number=2,
    )
    bot = full_elo.AZRootRolloutBot(None, "root", None, args)
    state = argparse.Namespace(_phase=3, _trick_number=1, _led_color=None)

    self.assertTrue(bot._enabled_for_phase(state))
    state._led_color = "Y"
    self.assertFalse(bot._enabled_for_phase(state))
    state._led_color = None
    state._trick_number = 3
    self.assertFalse(bot._enabled_for_phase(state))

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

  def test_value_shield_can_preserve_current_liveness_shield_base(self):
    args = argparse.Namespace(
        players=3,
        value_scale=20.0,
        survival_value_base_mode="liveness_shield",
        survival_value_threshold=0.65,
        survival_value_max_actions=0,
        survival_value_phases="play",
    )
    policy = np.zeros(1000, dtype=np.float32)
    policy[1] = 0.90
    policy[2] = 0.10

    with mock.patch.object(
        full_elo, "AZLivenessShieldPolicyBot", return_value=_ActionBot(2)
    ):
      bot = full_elo.AZValueShieldPolicyBot(None, None, "shield", None, args)
    with mock.patch.object(
        full_elo,
        "model_policy_value",
        return_value=(policy, np.zeros(3, dtype=np.float32)),
    ), mock.patch.object(
        bot, "_score_candidates", return_value={1: 0.40, 2: 0.80}
    ):
      action = bot.step(_ShieldState([1, 2]), player=0)

    self.assertEqual(action, 2)
    self.assertEqual(bot.decision_stats()["value_shield_base_kept"], 1)

  def test_value_shield_scores_only_acting_player_information_state_worlds(self):
    game = az_torch.pyspiel.load_game("python_quantum_cat", {"players": 4})
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    while state._phase < 3:
      player = state.current_player()
      action = shared_discard_action(state, player)
      if action is None:
        action = shared_prediction_action(state, player)
      state.apply_action(action)
    player = state.current_player()
    action = state.legal_actions(player)[0]
    sampled = None
    actual_child = state.clone()
    actual_child.apply_action(action)
    next_player = actual_child.current_player()
    rng = np.random.RandomState(20260819)
    for _ in range(16):
      candidate = state.resample_from_infostate(player, rng)
      if not np.array_equal(
          candidate._hands[next_player], state._hands[next_player]
      ):
        sampled = candidate
        break
    self.assertIsNotNone(sampled)
    args = argparse.Namespace(
        players=4,
        value_scale=20.0,
        survival_value_threshold=0.55,
        survival_value_belief_samples=1,
    )
    bot = full_elo.AZValueShieldPolicyBot(object(), object(), "shield", None, args)
    observed_hands = []

    def record_world(child, acting_player):
      self.assertEqual(acting_player, player)
      observed_hands.append(np.array(child._hands[next_player], copy=True))
      return 0.75

    with mock.patch.object(
        full_elo, "sampled_belief_states_for_policy", return_value=[sampled]
    ) as sampler, mock.patch.object(
        bot, "_state_survival_score", side_effect=record_world
    ):
      scores = bot._score_candidates(state, player, [action])

    self.assertEqual(scores, {action: 0.75})
    np.testing.assert_array_equal(observed_hands[0], sampled._hands[next_player])
    self.assertFalse(
        np.array_equal(observed_hands[0], state._hands[next_player])
    )
    sampler.assert_called_once()

  def test_value_shield_rejects_world_outside_acting_information_state(self):
    game = az_torch.pyspiel.load_game("python_quantum_cat", {"players": 4})
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    while state._phase < 3:
      player = state.current_player()
      action = shared_discard_action(state, player)
      if action is None:
        action = shared_prediction_action(state, player)
      state.apply_action(action)
    player = state.current_player()
    corrupted = state.clone()
    corrupted._hands[player][0] += 1
    args = argparse.Namespace(
        players=4,
        value_scale=20.0,
        survival_value_threshold=0.55,
        survival_value_belief_samples=1,
    )
    bot = full_elo.AZValueShieldPolicyBot(object(), object(), "shield", None, args)

    with mock.patch.object(
        full_elo, "sampled_belief_states_for_policy", return_value=[corrupted]
    ):
      with self.assertRaisesRegex(RuntimeError, "acting player's information state"):
        bot._score_candidates(state, player, state.legal_actions(player)[:1])

  def test_survival_replay_uses_round_local_targets_and_source_group(self):
    args = argparse.Namespace(
        players=4,
        seed=20260819,
        bot="heuristic",
        checkpoint="",
        initial_start=0,
        match_context=True,
        source_group_id="source_a",
        decision_phases="play",
    )

    result = survival_value_replay.collect_match(args, match_idx=0)

    self.assertLen(result["rounds"], 4)
    self.assertNotEmpty(result["rows"])
    self.assertEqual(len(result["row_group_ids"]), len(result["rows"]))
    self.assertEqual(set(result["row_group_ids"]), {"source_a"})
    for row in result["rows"]:
      any_paradox = bool(np.any(np.asarray(row[4]) > 0.5))
      expected_value = -1.0 if any_paradox else 1.0
      np.testing.assert_array_equal(row[3], np.full(4, expected_value))
      selected_features = row[11][int(row[5])]
      self.assertEqual(int(np.argmax(selected_features[1:6])), 3)

  def test_compact_survival_replay_retains_selected_legal_action_features(self):
    args = argparse.Namespace(
        players=4,
        seed=20260819,
        bot="heuristic",
        checkpoint="",
        initial_start=0,
        match_context=True,
        source_group_id="source_compact",
        decision_phases="play",
        compact_legal_actions=True,
    )

    result = survival_value_replay.collect_match(args, match_idx=0)

    self.assertNotEmpty(result["rows"])
    for row in result["rows"]:
      self.assertLess(len(row[1]), 1000)
      self.assertEqual(len(row[1]), len(row[11]))
      self.assertTrue(np.all(np.asarray(row[1]) == 1.0))
      selected = int(row[5])
      self.assertGreaterEqual(selected, 0)
      self.assertLess(selected, len(row[1]))
      self.assertEqual(float(row[2][selected]), 1.0)
      self.assertEqual(int(np.argmax(row[11][selected][1:6])), 3)

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

    with mock.patch.object(
        full_elo, "_shared_discard_decision", return_value=None
    ):
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

    with mock.patch.object(
        full_elo, "_shared_discard_decision", return_value=None
    ):
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

  def test_failure_miner_resolves_query_style_liveness_shield(self):
    args = argparse.Namespace(
        bot_mode=(
            "liveness_shield?liveness_shield_base_mode=policy,"
            "liveness_shield_phases=discard|prediction|play,"
            "liveness_shield_min_trick_number=5,"
            "liveness_shield_led_colors=B|Y|G,"
            "liveness_shield_min_open_slot_delta=0,"
            "liveness_shield_min_public_damage_delta=0,"
            "liveness_shield_max_policy_log_gap=-1.0"
        )
    )

    resolved = failure_miner.resolve_actor_bot_args(args)

    self.assertEqual(resolved.bot_mode, "liveness_policy")
    self.assertEqual(resolved.liveness_shield_phases, "discard|prediction|play")
    self.assertEqual(resolved.liveness_shield_min_trick_number, 5)
    self.assertEqual(resolved.liveness_shield_led_colors, "B|Y|G")
    self.assertEqual(resolved.liveness_shield_min_open_slot_delta, 0)
    self.assertEqual(resolved.liveness_shield_min_public_damage_delta, 0)
    self.assertEqual(resolved.liveness_shield_max_policy_log_gap, -1.0)

  def test_failure_miner_cli_accepts_query_style_liveness_shield(self):
    selector = (
        "liveness_shield?liveness_shield_base_mode=policy,"
        "liveness_shield_phases=play,"
        "liveness_shield_min_open_slot_delta=0"
    )
    with mock.patch(
        "sys.argv",
        [
            "quantum_cat_failure_miner.py",
            "--checkpoint=checkpoint.pt",
            "--out=failures.json",
            f"--bot-mode={selector}",
        ],
    ):
      parsed = failure_miner.parse_args()

    resolved = failure_miner.resolve_actor_bot_args(parsed)
    self.assertEqual(resolved.bot_mode, "liveness_policy")
    self.assertEqual(resolved.liveness_shield_phases, "play")
    self.assertEqual(resolved.liveness_shield_min_open_slot_delta, 0)

  def test_failure_miner_rejects_unknown_actor_bot_mode(self):
    with self.assertRaisesRegex(ValueError, "unsupported actor bot mode"):
      failure_miner.resolve_actor_bot_args(
          argparse.Namespace(bot_mode="liveness_shiled")
      )

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

  def test_belief_table_liveness_scores_only_resampled_worlds(self):
    class _BeliefWorld:
      _phase = 3
      _trick_number = 5

      def __init__(self, sampled=False, observation=1.0):
        self.sampled = sampled
        self.observation = observation

      def legal_actions(self, _player):
        return [1, 2]

      def num_distinct_actions(self):
        return 3

      def observation_tensor(self, _player):
        return [self.observation]

      def clone(self):
        return self

      def resample_from_infostate(self, _player, _rng):
        return _BeliefWorld(sampled=True, observation=self.observation)

    args = argparse.Namespace(
        players=4,
        value_scale=20.0,
        belief_table_liveness_base_mode="policy",
        belief_table_liveness_samples=8,
        belief_table_liveness_min_valid_samples=8,
        belief_table_liveness_seed_salt=7,
        belief_table_liveness_phases="play",
        belief_table_liveness_min_trick_number=5,
        belief_table_liveness_max_trick_number=7,
        belief_table_liveness_aggregation="teacher_vote",
        belief_table_liveness_min_paired_lcb=0.0,
        belief_table_liveness_paired_lcb_z=1.0,
        belief_table_liveness_min_best_share=0.0,
        belief_table_liveness_max_policy_log_gap=-1.0,
        belief_table_liveness_shadow_only=False,
        belief_table_liveness_sample_limit=20,
    )
    bot = full_elo.AZBeliefTableLivenessPolicyBot(
        None, "belief_table_liveness", "cpu", args
    )
    bot.base_bot = _ActionBot(1)
    bot._policy = lambda state, player: np.array(
        [0.0, 0.8, 0.2], dtype=np.float32
    )

    def score_sampled_world(sampled, _player, _legal, _base, _policy):
      self.assertTrue(sampled.sampled)
      return {1: 0.0, 2: 1.0}, 2

    bot._score_world = score_sampled_world
    state = _BeliefWorld(sampled=False)

    action = bot.step(state, player=0)

    self.assertEqual(action, 2)
    stats = bot.decision_stats()
    self.assertEqual(stats["belief_table_liveness_overrides"], 1)
    self.assertEqual(stats["belief_table_liveness_worlds_scored"], 8)
    self.assertEqual(stats["belief_table_liveness_infostate_mismatches"], 0)
    self.assertFalse(stats["belief_table_liveness_real_state_exact_scoring"])

  def test_belief_table_liveness_fails_closed_on_infostate_mismatch(self):
    class _MismatchedBeliefWorld:
      _phase = 3
      _trick_number = 5

      def __init__(self, observation):
        self.observation = observation

      def legal_actions(self, _player):
        return [1, 2]

      def num_distinct_actions(self):
        return 3

      def observation_tensor(self, _player):
        return [self.observation]

      def clone(self):
        return self

      def resample_from_infostate(self, _player, _rng):
        return _MismatchedBeliefWorld(observation=self.observation + 1.0)

    args = argparse.Namespace(
        players=4,
        value_scale=20.0,
        belief_table_liveness_base_mode="policy",
        belief_table_liveness_samples=4,
        belief_table_liveness_min_valid_samples=4,
        belief_table_liveness_seed_salt=0,
        belief_table_liveness_phases="play",
        belief_table_liveness_min_trick_number=5,
        belief_table_liveness_max_trick_number=7,
        belief_table_liveness_aggregation="teacher_vote",
        belief_table_liveness_min_paired_lcb=0.0,
        belief_table_liveness_paired_lcb_z=1.0,
        belief_table_liveness_min_best_share=0.0,
        belief_table_liveness_max_policy_log_gap=-1.0,
        belief_table_liveness_shadow_only=False,
        belief_table_liveness_sample_limit=20,
    )
    bot = full_elo.AZBeliefTableLivenessPolicyBot(
        None, "belief_table_liveness", "cpu", args
    )
    bot.base_bot = _ActionBot(1)
    bot._policy = lambda state, player: np.array(
        [0.0, 0.8, 0.2], dtype=np.float32
    )

    action = bot.step(_MismatchedBeliefWorld(observation=1.0), player=0)

    self.assertEqual(action, 1)
    stats = bot.decision_stats()
    self.assertEqual(stats["belief_table_liveness_overrides"], 0)
    self.assertEqual(stats["belief_table_liveness_infostate_mismatches"], 4)
    self.assertEqual(stats["belief_table_liveness_insufficient_worlds"], 1)

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

  def test_shared_discard_prefers_most_common_rank(self):
    state = _DiscardState([1, 3, 2, 0, 0, 0], legal=[0, 1, 2])

    self.assertEqual(shared_discard_action(state, 0), 1)
    self.assertEqual(
        HeuristicBot(discard_future_flex_weight=99.0)._discard(state, 0, [0, 1, 2]),
        1,
    )

  def test_shared_discard_avoids_six_only_when_tied(self):
    self.assertEqual(
        shared_discard_action(_DiscardState([0, 2, 0, 0, 0, 2], legal=[1, 5]), 0),
        1,
    )
    self.assertEqual(
        shared_discard_action(_DiscardState([0, 1, 0, 0, 0, 3], legal=[1, 5]), 0),
        5,
    )

  def test_shared_discard_tie_chooses_least_similar_rank(self):
    state = _DiscardState([1, 0, 1, 1, 0, 0], legal=[0, 2, 3])

    self.assertEqual(shared_discard_action(state, 0), 0)

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

  def test_az_policy_bot_can_use_model_prediction(self):
    state = _PredictState([0, 0, 0, 0, 1, 1])
    bot = az_torch.AZPolicyBot(
        model=object(),
        name="az_policy",
        device=None,
        value_scale=1.0,
        use_shared_prediction=False,
    )
    policy = np.zeros(1000, dtype=np.float32)
    policy[103] = 1.0

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(policy, np.zeros(3))
    ) as infer:
      action = bot.step(state, 0)

    self.assertEqual(action, 103)
    self.assertTrue(bot.uses_neural_prediction)
    self.assertFalse(infer.call_args.kwargs["use_shared_prediction"])

  def test_joint_self_play_model_prediction_bypasses_shared_bid(self):
    state = _PredictState([0, 0, 0, 0, 1, 1])
    args = argparse.Namespace(
        self_play_policy_mode="q_policy",
        self_play_prediction_mode="model",
        value_scale=20.0,
    )
    policy = np.zeros(1000, dtype=np.float32)
    policy[103] = 1.0

    with mock.patch.object(
        az_torch, "model_policy_value", return_value=(policy, np.zeros(3))
    ) as infer:
      result = az_torch.learner_policy(
          state, player=0, model=object(), args=args, device="cpu"
      )

    self.assertEqual(int(np.argmax(result)), 103)
    self.assertFalse(infer.call_args.kwargs["use_shared_prediction"])

  def test_joint_self_play_prediction_search_uses_information_state_belief(self):
    state = _PredictState([0, 0, 0, 0, 1, 1])
    args = argparse.Namespace(
        self_play_policy_mode="q_policy",
        self_play_prediction_mode="belief_search",
        value_scale=20.0,
    )
    searched = np.zeros(1000, dtype=np.float32)
    searched[103] = 1.0

    with mock.patch.object(
        az_torch, "belief_mcts_policy", return_value=searched
    ) as search:
      result = az_torch.learner_policy(
          state, player=0, model=object(), args=args, device="cpu"
      )

    self.assertEqual(int(np.argmax(result)), 103)
    search.assert_called_once()

  def test_prediction_search_prior_keeps_other_bids_available(self):
    state = _PredictState([0, 0, 0, 0, 1, 1])
    args = argparse.Namespace(
        self_play_prediction_mode="belief_search",
        prediction_shared_prior_weight=0.35,
    )
    priors = np.zeros(1000, dtype=np.float32)
    priors[101] = 1.0

    blended = az_torch.blend_prediction_search_prior(state, 0, priors, args)

    self.assertAlmostEqual(float(blended.sum()), 1.0, places=6)
    self.assertGreater(float(blended[102]), 0.0)
    self.assertGreater(float(blended[103]), 0.0)

  def test_frozen_league_opponent_can_use_actual_q_policy(self):
    state = _PredictState([0, 0, 0, 0, 1, 1])
    args = argparse.Namespace(league_opponent_mode="q_policy")
    policy = np.zeros(1000, dtype=np.float32)
    policy[103] = 1.0

    with mock.patch.object(
        az_torch, "q_policy_rerank_policy", return_value=policy
    ) as rerank:
      selected = az_torch.old_policy_action(
          state,
          player=0,
          legal=[101, 102, 103, 104],
          opponent_model=object(),
          role="old_policy",
          args=args,
          device="cpu",
      )

    self.assertEqual(selected, 103)
    rerank.assert_called_once()

  def test_az_policy_bot_uses_shared_discard_without_model(self):
    state = _DiscardState([1, 0, 1, 1, 0, 0], legal=[0, 2, 3])
    bot = az_torch.AZPolicyBot(
        model=None, name="az_policy", device=None, value_scale=1.0
    )

    self.assertEqual(bot.step(state, 0), 0)

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

  def test_grouped_validation_holds_out_every_row_from_selected_source(self):
    replay = []
    for index, labeled in enumerate([True, False, True, False]):
      row = _paradox_example(labeled)
      replay.append((np.array([index], dtype=np.float32), *row[1:]))
    group_ids = ["source_a", "source_a", "source_b", "source_b"]
    args = argparse.Namespace(
        action_value_validation_fraction=0.5,
        loaded_replay_validation_label_kind="action_paradox",
        action_value_validation_seed=7,
        action_value_validation_group_metadata_key="row_group_ids",
    )

    train_replay, validation_replay, split = (
        az_torch.split_action_value_validation_replay(
            replay, args, group_ids=group_ids
        )
    )

    train_groups = {group_ids[int(row[0][0])] for row in train_replay}
    validation_groups = {
        group_ids[int(row[0][0])] for row in validation_replay
    }
    self.assertLen(train_replay, 2)
    self.assertLen(validation_replay, 2)
    self.assertEmpty(train_groups & validation_groups)
    self.assertEqual(split["action_value_validation_split_mode"], "group")
    self.assertEqual(split["action_value_validation_group_overlap_count"], 0)
    self.assertEqual(split["action_value_validation_train_group_count"], 1)
    self.assertEqual(split["action_value_validation_group_count"], 1)
    self.assertEqual(split["action_value_validation_heldout_labeled_rows"], 1)

  def test_grouped_validation_rejects_missing_source_group_metadata(self):
    replay = [_paradox_example(True), _paradox_example(True)]
    args = argparse.Namespace(
        action_value_validation_fraction=0.5,
        loaded_replay_validation_label_kind="action_paradox",
        action_value_validation_seed=7,
        action_value_validation_group_metadata_key="row_group_ids",
    )

    with self.assertRaisesRegex(ValueError, "source-group metadata"):
      az_torch.split_action_value_validation_replay(replay, args)

    with self.assertRaisesRegex(ValueError, "one source-group id per row"):
      az_torch.split_action_value_validation_replay(
          replay, args, group_ids=["source_a"]
      )

  def test_grouped_validation_rejects_nonindependent_labeled_sources(self):
    replay = [
        _paradox_example(True),
        _paradox_example(True),
        _paradox_example(False),
    ]
    args = argparse.Namespace(
        action_value_validation_fraction=0.5,
        loaded_replay_validation_label_kind="action_paradox",
        action_value_validation_seed=7,
        action_value_validation_group_metadata_key="row_group_ids",
    )

    with self.assertRaisesRegex(ValueError, "two independent source groups"):
      az_torch.split_action_value_validation_replay(
          replay,
          args,
          group_ids=["same_source", "same_source", "unlabeled_source"],
      )

    with self.assertRaisesRegex(ValueError, "nonempty source-group id"):
      az_torch.split_action_value_validation_replay(
          replay,
          args,
          group_ids=["source_a", "source_b", ""],
      )

  def test_explicit_validation_rejects_reused_training_replay_path(self):
    args = argparse.Namespace(action_value_validation_group_metadata_key="")
    validation = {
        "action_value_validation_replay_paths": ["banks/shared.npz"],
        "action_value_validation_replay_metadata": [{}],
    }

    with self.assertRaisesRegex(ValueError, "reuse the same replay files"):
      az_torch.explicit_validation_source_group_report(
          ["banks/../banks/shared.npz"], [{}], validation, args
      )

  def test_preforced_groups_are_namespaced_by_collector_source(self):
    rows = [
        {"match_idx": 0, "round_idx": 0},
        {"match_idx": 0, "round_idx": 1},
        {"match_idx": 1, "round_idx": 0},
    ]

    training = preforced_replay.row_group_ids_from_stats(
        rows, "preforced:players=4:seed=11"
    )
    validation = preforced_replay.row_group_ids_from_stats(
        rows, "preforced:players=4:seed=22"
    )

    self.assertEmpty(set(training) & set(validation))
    self.assertEqual(training[0], "preforced:players=4:seed=11:m0:r0")
    self.assertEqual(validation[0], "preforced:players=4:seed=22:m0:r0")

  def test_explicit_validation_rejects_overlapping_collector_source_groups(self):
    args = argparse.Namespace(
        action_value_validation_group_metadata_key="row_group_ids"
    )
    validation = {
        "action_value_validation_replay_paths": ["validation.npz"],
        "action_value_validation_replay_metadata": [
            {"row_group_ids": ["shared", "validation_only"]}
        ],
    }

    with self.assertRaisesRegex(ValueError, "share collector source groups"):
      az_torch.explicit_validation_source_group_report(
          ["training.npz"],
          [{"row_group_ids": ["training_only", "shared"]}],
          validation,
          args,
      )

  def test_explicit_validation_records_disjoint_groups_and_rejects_missing_ids(self):
    args = argparse.Namespace(
        action_value_validation_group_metadata_key="row_group_ids"
    )
    validation = {
        "action_value_validation_replay_paths": ["validation.npz"],
        "action_value_validation_replay_metadata": [
            {"row_group_ids": ["validation_a", "validation_b"]}
        ],
    }

    report = az_torch.explicit_validation_source_group_report(
        ["training.npz"],
        [{"row_group_ids": ["training_a", "training_b"]}],
        validation,
        args,
    )

    self.assertEqual(report["action_value_validation_split_mode"], "explicit_group")
    self.assertEqual(report["action_value_validation_train_group_count"], 2)
    self.assertEqual(report["action_value_validation_group_count"], 2)
    self.assertEqual(report["action_value_validation_group_overlap_count"], 0)
    self.assertEqual(report["action_value_validation_replay_path_overlap_count"], 0)

    summarized = az_torch.summarized_validation_split({
        **validation,
        **report,
    })
    self.assertNotIn(
        "row_group_ids",
        summarized["action_value_validation_replay_metadata"][0],
    )
    self.assertEqual(
        summarized["action_value_validation_replay_metadata"][0][
            "source_group_ids"
        ],
        ["validation_a", "validation_b"],
    )

    with self.assertRaisesRegex(ValueError, "complete source-group metadata"):
      az_torch.explicit_validation_source_group_report(
          ["training.npz"], [{}], validation, args
      )

    with self.assertRaisesRegex(ValueError, "complete source-group metadata"):
      az_torch.explicit_validation_source_group_report(
          ["training.npz"], [], validation, args
      )

    with self.assertRaisesRegex(ValueError, "complete source-group metadata"):
      az_torch.explicit_validation_source_group_report(
          ["training.npz"],
          [{"row_group_ids": ["training_a"]}],
          {"action_value_validation_replay_paths": ["validation.npz"]},
          args,
      )

  def test_loaded_replay_validation_score_supports_action_paradox_brier(self):
    row = {
        "action_paradox_validation_report": {
            "brier": 0.125,
            "corr": 0.75,
            "top1_rate": 0.8,
            "mean_regret": 0.04,
            "mean_state_corr": 0.6,
            "contrast_top1_rate": 0.7,
            "contrast_mean_regret": 0.06,
            "contrast_mean_state_corr": 0.5,
        }
    }

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
    self.assertEqual(
        az_torch.loaded_replay_validation_score(
            row,
            argparse.Namespace(
                loaded_replay_best_metric="action_paradox_validation_top1"
            ),
        ),
        0.8,
    )
    self.assertEqual(
        az_torch.loaded_replay_validation_score(
            row,
            argparse.Namespace(
                loaded_replay_best_metric=(
                    "action_paradox_validation_mean_regret"
                )
            ),
        ),
        -0.04,
    )
    self.assertEqual(
        az_torch.loaded_replay_validation_score(
            row,
            argparse.Namespace(
                loaded_replay_best_metric=(
                    "action_paradox_validation_mean_state_corr"
                )
            ),
        ),
        0.6,
    )
    self.assertEqual(
        az_torch.loaded_replay_validation_score(
            row,
            argparse.Namespace(
                loaded_replay_best_metric=(
                    "action_paradox_validation_contrast_top1"
                )
            ),
        ),
        0.7,
    )
    self.assertEqual(
        az_torch.loaded_replay_validation_score(
            row,
            argparse.Namespace(
                loaded_replay_best_metric=(
                    "action_paradox_validation_contrast_mean_regret"
                )
            ),
        ),
        -0.06,
    )

  def test_action_paradox_pair_summary_measures_low_risk_ranking(self):
    summary = az_torch.action_paradox_pair_summary(
        np.array([
            [0.1, 0.9, 0.0],
            [0.4, 0.2, 0.7],
        ], dtype=np.float32),
        np.array([
            [0.0, 0.5, 0.0],
            [0.3, 0.1, 0.9],
        ], dtype=np.float32),
        np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32),
    )

    self.assertEqual(summary["state_count"], 2)
    self.assertEqual(summary["top1_rate"], 1.0)
    self.assertEqual(summary["mean_regret"], 0.0)
    self.assertGreater(summary["mean_state_corr"], 0.99)
    contrast = az_torch.action_paradox_pair_summary(
        np.array([[0.1, 0.9]], dtype=np.float32),
        np.array([[0.2, 0.2]], dtype=np.float32),
        np.array([[1.0, 1.0]], dtype=np.float32),
        min_target_spread=1e-6,
    )
    self.assertIsNone(contrast)

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

  def test_best_of_rollouts_requires_explicit_forced_paradox_contrast(self):
    args = argparse.Namespace(
        require_survival_contrast=True,
        min_any_paradox_rate_spread=0.25,
        require_forced_paradox_contrast=True,
        min_forced_paradox_rate_spread=0.25,
        require_policy_target_changed=True,
        min_best_vs_base_score_delta=0.0,
    )
    decision = {"action": 1}
    records = [
        {
            "action": 1,
            "any_paradox_rate": 1.0,
            "forced_paradox_rate": 1.0,
            "branch_score": 0.0,
        },
        {
            "action": 2,
            "any_paradox_rate": 0.0,
            "forced_paradox_rate": 0.0,
            "branch_score": 1.0,
        },
    ]

    accepted, reason = best_of_rollouts_replay._acceptance(
        decision, records, 2, args
    )

    self.assertTrue(accepted)
    self.assertEqual(reason, "accepted")

  def test_best_of_rollouts_samples_decisions_across_completed_round(self):
    decisions = [{"index": index} for index in range(24)]

    first = best_of_rollouts_replay._sample_round_decisions(
        decisions, max_count=3, seed=20260750
    )
    second = best_of_rollouts_replay._sample_round_decisions(
        decisions, max_count=3, seed=20260750
    )

    self.assertEqual(first, second)
    self.assertEqual(len(first), 3)
    self.assertTrue(any(item["index"] >= 3 for item in first))

  def test_best_of_rollouts_rejects_weak_forced_paradox_contrast(self):
    args = argparse.Namespace(
        require_survival_contrast=True,
        min_any_paradox_rate_spread=0.25,
        require_forced_paradox_contrast=True,
        min_forced_paradox_rate_spread=0.25,
        require_policy_target_changed=True,
        min_best_vs_base_score_delta=0.0,
    )
    decision = {"action": 1}
    records = [
        {
            "action": 1,
            "any_paradox_rate": 1.0,
            "forced_paradox_rate": 1.0,
            "branch_score": 0.0,
        },
        {
            "action": 2,
            "any_paradox_rate": 0.0,
            "forced_paradox_rate": 1.0,
            "branch_score": 1.0,
        },
    ]

    accepted, reason = best_of_rollouts_replay._acceptance(
        decision, records, 2, args
    )

    self.assertFalse(accepted)
    self.assertEqual(reason, "weak_forced_paradox_contrast")

  def test_best_of_rollouts_requires_base_protected_survival_improvement(self):
    args = argparse.Namespace(
        require_survival_contrast=True,
        min_any_paradox_rate_spread=0.5,
        require_forced_paradox_contrast=False,
        min_forced_paradox_rate_spread=0.0,
        require_policy_target_changed=True,
        min_best_vs_base_score_delta=0.0,
        min_best_vs_base_any_paradox_improvement=0.5,
    )
    decision = {"action": 1}
    records = [
        {
            "action": 1,
            "any_paradox_rate": 0.25,
            "forced_paradox_rate": 0.25,
            "branch_score": 0.0,
        },
        {
            "action": 2,
            "any_paradox_rate": 0.25,
            "forced_paradox_rate": 0.25,
            "branch_score": 1.0,
        },
        {
            "action": 3,
            "any_paradox_rate": 0.75,
            "forced_paradox_rate": 0.75,
            "branch_score": -1.0,
        },
    ]

    accepted, reason = best_of_rollouts_replay._acceptance(
        decision, records, 2, args
    )

    self.assertFalse(accepted)
    self.assertEqual(reason, "weak_best_vs_base_any_paradox_improvement")

  def test_best_of_rollouts_requires_fresh_belief_validation_improvement(self):
    args = argparse.Namespace(
        min_audit_best_vs_base_any_paradox_improvement=0.25
    )
    decision = {"action": 1}
    records = [
        {"action": 1, "any_paradox_rate": 0.75},
        {"action": 2, "any_paradox_rate": 0.50},
    ]

    accepted, reason, improvement = best_of_rollouts_replay._audit_acceptance(
        decision, records, 2, args
    )

    self.assertTrue(accepted)
    self.assertEqual(reason, "accepted")
    self.assertEqual(improvement, 0.25)

    records[1]["any_paradox_rate"] = 0.75
    accepted, reason, improvement = best_of_rollouts_replay._audit_acceptance(
        decision, records, 2, args
    )
    self.assertFalse(accepted)
    self.assertEqual(
        reason, "weak_audit_best_vs_base_any_paradox_improvement"
    )
    self.assertEqual(improvement, 0.0)

  def test_best_of_rollouts_requires_paired_binomial_audit_evidence(self):
    args = argparse.Namespace(
        min_audit_best_vs_base_any_paradox_improvement=0.0,
        max_audit_paired_binomial_p=0.10,
    )
    decision = {"action": 1}
    records = [
        {
            "action": 1,
            "any_paradox_rate": 1.0,
            "any_paradox_outcomes": [True, True, True, True],
        },
        {
            "action": 2,
            "any_paradox_rate": 0.0,
            "any_paradox_outcomes": [False, False, False, False],
        },
    ]
    accepted, reason, _ = best_of_rollouts_replay._audit_acceptance(
        decision, records, 2, args
    )
    self.assertTrue(accepted)
    self.assertEqual(reason, "accepted")
    paired = best_of_rollouts_replay._paired_any_paradox_summary(*records)
    self.assertEqual(paired["saved_outcomes"], 4)
    self.assertEqual(paired["harmed_outcomes"], 0)
    self.assertEqual(paired["one_sided_binomial_p"], 0.0625)

    records[0]["any_paradox_outcomes"] = [True, True, True, False]
    records[1]["any_paradox_outcomes"] = [False, False, False, True]
    accepted, reason, _ = best_of_rollouts_replay._audit_acceptance(
        decision, records, 2, args
    )
    self.assertFalse(accepted)
    self.assertEqual(reason, "weak_audit_paired_evidence")

  def test_best_of_rollouts_paired_evidence_uses_independent_worlds(self):
    records = [
        {
            "action": 1,
            "any_paradox_rate": 1.0,
            "any_paradox_outcomes": [True] * 6,
            "rollout_worlds": 3,
            "rollouts_per_world": 2,
            "rollouts": 6,
        },
        {
            "action": 2,
            "any_paradox_rate": 0.0,
            "any_paradox_outcomes": [False] * 6,
            "rollout_worlds": 3,
            "rollouts_per_world": 2,
            "rollouts": 6,
        },
    ]

    paired = best_of_rollouts_replay._paired_any_paradox_summary(*records)

    self.assertEqual(paired["inference_unit"], "information_state_world")
    self.assertEqual(paired["paired_worlds"], 3)
    self.assertEqual(paired["paired_outcomes"], 3)
    self.assertEqual(paired["continuation_outcomes"], 6)
    self.assertEqual(paired["saved_outcomes"], 3)
    self.assertEqual(paired["one_sided_binomial_p"], 0.125)
    args = argparse.Namespace(
        min_audit_best_vs_base_any_paradox_improvement=0.0,
        max_audit_paired_binomial_p=0.10,
    )
    accepted, reason, _ = best_of_rollouts_replay._audit_acceptance(
        {"action": 1}, records, 2, args
    )
    self.assertFalse(accepted)
    self.assertEqual(reason, "weak_audit_paired_evidence")

  def test_action_set_source_split_rejects_canonical_group_overlap(self):
    train = [{"source_group_id": "source-a"}]
    validation = [{"source_group_id": "source-b"}]

    report = action_set_trainer._source_group_split_report(
        train, validation
    )

    self.assertTrue(report["passed"])
    self.assertEmpty(report["overlap"])
    with self.assertRaisesRegex(ValueError, "cross train/validation split"):
      action_set_trainer._source_group_split_report(
          train, [{"source_group_id": "source-a"}]
      )

  def test_best_of_rollouts_audit_rejection_targets_base_action(self):
    target, action = best_of_rollouts_replay._base_keep_target(
        {"action": 7}, 12
    )

    self.assertEqual(action, 7)
    np.testing.assert_array_equal(
        target,
        np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0],
            dtype=np.float32,
        ),
    )

  def test_best_of_rollouts_calibration_maps_fresh_audit_outcomes(self):
    common = {
        "decision": {
            "policy_action": 2,
            "legal_count": 3,
            "phase": "play",
            "trick_number": 6,
        },
        "candidate_actions": [2, 3, 4],
        "information_safe_rollout_teacher": True,
        "independent_belief_validation": True,
        "audit_best_vs_base_any_paradox_improvement": 0.5,
    }
    positive = best_of_rollouts_calibration.calibration_metadata_row(
        {**common, "best_action": 3},
        3,
        "positive",
        "positive.npz",
        0,
        "train-shard-7",
    )
    negative = best_of_rollouts_calibration.calibration_metadata_row(
        {
            **common,
            "proposed_best_action": 4,
            "audit_best_vs_base_any_paradox_improvement": 0.0,
        },
        2,
        "negative",
        "negative.npz",
        0,
    )

    self.assertEqual(positive["calibration_class"], "POSITIVE")
    self.assertEqual(positive["group"]["proposal_source"], "train-shard-7")
    self.assertEqual(positive["label_best_action"], 3)
    self.assertEqual(positive["deployment_target_action"], 3)
    self.assertEqual(positive["audit_paradox_delta_vs_policy"], -0.5)
    self.assertFalse(positive["false_positive_for_positive_gate"])
    self.assertEqual(negative["calibration_class"], "NEGATIVE")
    self.assertEqual(negative["label_best_action"], 4)
    self.assertEqual(negative["deployment_target_action"], 2)
    self.assertTrue(negative["false_positive_for_positive_gate"])

  def test_calibration_explicit_validation_source_overlap_is_detected(self):
    examples = [
        {"source_key": "train-a"},
        {"source_key": "shared"},
        {"source_key": "validation-b"},
        {"source_key": "shared"},
    ]

    train, validation, overlap = (
        calibration_transfer_audit.explicit_source_overlap(
            examples, [0, 1], [2, 3]
        )
    )

    self.assertEqual(train, {"train-a", "shared"})
    self.assertEqual(validation, {"validation-b", "shared"})
    self.assertEqual(overlap, ["shared"])
    self.assertEqual(
        calibration_transfer_audit._metadata_source_key(
            {"group": {"proposal_source": "shadow"}},
            "replay.npz",
            "seed=17",
        ),
        "seed=17|shadow",
    )

  def test_calibration_expands_top_k_candidate_audits(self):
    rows = calibration_transfer_audit.candidate_calibration_metadata_rows({
        "policy_best_action": 2,
        "label_best_action": 7,
        "deployment_target_action": 9,
        "candidate_evaluations": [
            {
                "action": 7,
                "calibration_class": "NEGATIVE",
                "audit_margin_vs_policy": -1.0,
                "audit_paradox_delta_vs_policy": 0.5,
            },
            {
                "action": 9,
                "calibration_class": "POSITIVE",
                "audit_margin_vs_policy": 3.0,
                "audit_paradox_delta_vs_policy": -0.5,
            },
        ],
    })

    self.assertLen(rows, 2)
    self.assertEqual(rows[0]["label_best_action"], 7)
    self.assertEqual(rows[0]["deployment_target_action"], 2)
    self.assertTrue(rows[0]["false_positive_for_positive_gate"])
    self.assertEqual(rows[1]["label_best_action"], 9)
    self.assertEqual(rows[1]["deployment_target_action"], 9)
    self.assertFalse(rows[1]["false_positive_for_positive_gate"])

  def test_improve_miner_negative_pair_keeps_proposed_action_features(self):
    fields = improve_graft_miner.proposal_calibration_fields(
        21, "Play G4", "NEGATIVE", -1.75, 0.25
    )

    self.assertEqual(fields["label_best_action"], 21)
    self.assertEqual(fields["calibration_class"], "NEGATIVE")
    self.assertTrue(fields["false_positive_for_positive_gate"])
    self.assertEqual(fields["audit_margin_vs_policy"], -1.75)
    self.assertEqual(fields["audit_paradox_delta_vs_policy"], 0.25)

  def test_improve_miner_policy_checkpoint_proposes_actual_disagreement(self):
    proposed, reason, candidates = (
        improve_graft_miner._proposal_from_policy_disagreement(
            base_policy=np.array([0.05, 0.70, 0.20, 0.05]),
            proposal_policy=np.array([0.05, 0.20, 0.60, 0.15]),
            legal=[1, 2, 3],
            base_action=1,
            top_k=2,
        )
    )

    self.assertEqual(proposed, 2)
    self.assertEqual(reason, "proposal")
    self.assertEqual([row["action"] for row in candidates], [2, 3])
    self.assertAlmostEqual(candidates[0]["proposal_checkpoint_prob"], 0.60)
    self.assertAlmostEqual(candidates[0]["policy_prob"], 0.20)

  def test_improve_miner_policy_checkpoint_abstains_when_top_actions_agree(self):
    proposed, reason, candidates = (
        improve_graft_miner._proposal_from_policy_disagreement(
            base_policy=np.array([0.05, 0.70, 0.20, 0.05]),
            proposal_policy=np.array([0.05, 0.60, 0.30, 0.05]),
            legal=[1, 2, 3],
            base_action=1,
            top_k=2,
        )
    )

    self.assertEqual(proposed, 1)
    self.assertEqual(reason, "policy_agrees")
    self.assertEqual([row["action"] for row in candidates], [2, 3])

  def test_improve_miner_near_positive_target_fails_closed_to_base(self):
    args = argparse.Namespace(
        calibration_positive_margin=0.0,
        calibration_max_positive_paradox_delta=-0.25,
    )
    audit = {
        2: {"mean": 5.0, "mean_paradox": 1.0},
        7: {"mean": 8.0, "mean_paradox": 1.0},
    }

    target, target_class, evaluations = (
        improve_graft_miner._candidate_target_from_audit(
            audit,
            base_action=2,
            candidate_actions=[7],
            args=args,
        )
    )

    self.assertEqual(evaluations[0]["calibration_class"], "NEAR_POSITIVE")
    self.assertEqual(target, 2)
    self.assertEqual(target_class, "NEGATIVE")

  def test_improve_miner_positive_target_can_override_base(self):
    args = argparse.Namespace(
        calibration_positive_margin=0.0,
        calibration_max_positive_paradox_delta=-0.25,
    )
    audit = {
        2: {"mean": 5.0, "mean_paradox": 1.0},
        7: {"mean": 8.0, "mean_paradox": 0.5},
    }

    target, target_class, _ = improve_graft_miner._candidate_target_from_audit(
        audit,
        base_action=2,
        candidate_actions=[7],
        args=args,
    )

    self.assertEqual(target, 7)
    self.assertEqual(target_class, "POSITIVE")

  def test_improve_miner_unstable_aggregate_positive_fails_closed(self):
    args = argparse.Namespace(
        calibration_positive_margin=0.0,
        calibration_max_positive_paradox_delta=-0.25,
        stability_audit_passes=3,
        stability_min_positive_sign_rate=1.0,
        stability_min_safe_risk_rate=1.0,
        unstable_positive_as_negative=True,
    )
    audit = {
        2: {
            "mean": 5.0,
            "mean_paradox": 1.0,
            "pass_means": [5.0, 5.0, 5.0],
            "pass_paradoxes": [1.0, 1.0, 1.0],
        },
        7: {
            "mean": 20.0 / 3.0,
            "mean_paradox": 2.0 / 3.0,
            "pass_means": [8.0, 4.0, 8.0],
            "pass_paradoxes": [0.5, 1.0, 0.5],
        },
    }
    improve_graft_miner.annotate_candidate_stability(
        audit, 2, [7], args
    )

    target, target_class, evaluations = (
        improve_graft_miner._candidate_target_from_audit(
            audit, base_action=2, candidate_actions=[7], args=args
        )
    )

    self.assertEqual(target, 2)
    self.assertEqual(target_class, "NEGATIVE")
    self.assertEqual(evaluations[0]["calibration_class"], "NEGATIVE")
    self.assertFalse(evaluations[0]["stability"]["stable_positive"])

  def test_proposal_replay_repair_recomputes_class_instead_of_trusting_label(self):
    target, target_class, _ = proposal_replay_repair.safe_target_from_audit_row(
        {
            "policy_best_action": 2,
            "primary_proposal_action": 7,
            "candidate_evaluations": [{
                "action": 7,
                "calibration_class": "POSITIVE",
            }],
            "audit": {
                "2": {"mean": 5.0, "mean_paradox": 1.0},
                "7": {"mean": 8.0, "mean_paradox": 1.0},
            },
        },
        positive_margin=0.0,
        max_delta=-0.25,
    )

    self.assertEqual(target, 2)
    self.assertEqual(target_class, "NEGATIVE")

  def test_proposal_replay_repair_preserves_candidate_proposal_context(self):
    merged = proposal_replay_repair.preserve_candidate_proposal_context(
        {
            "candidate_evaluations": [{
                "action": 7,
                "proposal_improve_prob": 0.97,
                "proposal_safe_prob": 0.88,
                "proposal_policy_prob": 0.12,
            }],
        },
        [{
            "action": 7,
            "calibration_class": "POSITIVE",
            "audit_margin_vs_policy": 2.0,
            "audit_paradox_delta_vs_policy": -0.5,
        }],
    )

    self.assertEqual(merged[0]["proposal_improve_prob"], 0.97)
    self.assertEqual(merged[0]["proposal_safe_prob"], 0.88)
    self.assertEqual(merged[0]["proposal_policy_prob"], 0.12)
    self.assertEqual(merged[0]["calibration_class"], "POSITIVE")

  def test_proposal_policy_audit_separates_overrides_and_base_retention(self):
    candidate = np.array([
        [0.1, 0.8, 0.1, 0.0],
        [0.0, 0.1, 0.8, 0.1],
        [0.4, 0.6, 0.0, 0.0],
    ], dtype=np.float32)
    anchor = np.array([
        [0.8, 0.1, 0.1, 0.0],
        [0.0, 0.1, 0.8, 0.1],
        [0.7, 0.3, 0.0, 0.0],
    ], dtype=np.float32)
    policies = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    rows = [
        {
            "target_calibration_class": "POSITIVE",
            "deployment_target_action": 1,
            "policy_best_action": 0,
            "primary_proposal_action": 1,
        },
        {
            "target_calibration_class": "NEGATIVE",
            "deployment_target_action": 2,
            "policy_best_action": 2,
            "primary_proposal_action": 3,
        },
        {
            "target_calibration_class": "NEGATIVE",
            "deployment_target_action": 0,
            "policy_best_action": 0,
            "primary_proposal_action": 1,
        },
    ]
    args = argparse.Namespace(
        min_positive_rows=1,
        min_positive_top1_rate=1.0,
        min_base_keep_top1_rate=0.5,
        min_positive_target_prob_gain=0.5,
        max_unsafe_proposal_adoption_rate=0.5,
    )

    report = proposal_policy_audit.proposal_policy_audit_report(
        candidate, anchor, policies, rows, args
    )

    self.assertEqual(report["positive_override"]["count"], 1)
    self.assertEqual(
        report["positive_override"]["candidate_target_top1_rate"], 1.0
    )
    self.assertEqual(report["base_keep"]["candidate_target_top1_rate"], 0.5)
    self.assertEqual(report["unsafe_proposal_adoption"]["rate"], 0.5)
    self.assertTrue(report["deployment_gate"]["pass"])

  def test_calibration_requires_one_threshold_to_pass_every_input(self):
    aggregate = [
        {"threshold": 0.9, "risk_threshold": None, "pass_offline_gate": True},
        {"threshold": 0.99, "risk_threshold": None, "pass_offline_gate": True},
    ]
    by_input = {
        "a": [
            {"threshold": 0.9, "risk_threshold": None, "pass_offline_gate": True},
            {"threshold": 0.99, "risk_threshold": None, "pass_offline_gate": True},
        ],
        "b": [
            {"threshold": 0.9, "risk_threshold": None, "pass_offline_gate": False},
            {"threshold": 0.99, "risk_threshold": None, "pass_offline_gate": True},
        ],
    }

    passing = calibration_transfer_audit._passing_across_inputs(
        aggregate, by_input
    )

    self.assertLen(passing, 1)
    self.assertEqual(passing[0]["threshold"], 0.99)

  def test_calibration_game_context_is_fixed_size_and_deployable(self):
    features = calibration_transfer_audit.game_context_feature_vector(
        trick_number=7,
        match_round=2,
        player=1,
        players=3,
        mode="onehot",
    )

    self.assertEqual(features.shape, (23,))
    self.assertEqual(features[7 + 3], 1.0)
    self.assertEqual(features[10 + 2 + 3], 1.0)
    self.assertEqual(features[15 + 1 + 3], 1.0)

  def test_audit_rejected_base_bank_verifies_hard_keep_target(self):
    with tempfile.TemporaryDirectory() as directory:
      replay_path = Path(directory) / "base_keep.npz"
      sidecar_path = Path(directory) / "base_keep.npz.metadata.json"
      target = np.zeros(5, dtype=np.float32)
      target[2] = 1.0
      example = (
          np.zeros(4, dtype=np.float32),
          np.ones(5, dtype=np.float32),
          target,
          np.zeros(3, dtype=np.float32),
          np.zeros(3, dtype=np.float32),
          2,
          0,
          np.zeros(5, dtype=np.float32),
          np.ones(5, dtype=np.float32),
          np.zeros(5, dtype=np.float32),
          np.ones(5, dtype=np.float32),
          np.zeros((5, 1), dtype=np.float32),
      )
      az_torch.save_replay(
          [example],
          replay_path,
          metadata={"row_group_ids": ["source-1"]},
      )
      sidecar_path.write_text(json.dumps({
          "rows": [{
              "decision": {"policy_action": 2, "trick_number": 5},
              "best_action": 2,
              "proposed_best_action": 3,
              "audit_override_rejected": True,
              "audit_base_keep_validation": True,
              "independent_belief_validation": True,
              "information_safe_rollout_teacher": True,
              "selection_proposed_best_vs_base_any_paradox_improvement": 0.5,
              "audit_best_vs_base_any_paradox_improvement": 0.0,
          }]
      }))

      report = replay_holdout_audit._bank(
          replay_path,
          sidecar_path,
          min_base_improvement=0.5,
          min_audit_improvement=0.25,
          bank_kind="audit_rejected_base",
      )

      self.assertEqual(report["base_keep_target_rows"], 1)
      self.assertEqual(report["audit_rejected_base_rows"], 1)
      self.assertEqual(report["selection_proposal_protected_rows"], 1)
      self.assertEqual(report["fresh_audit_rejection_confirmed_rows"], 1)
      self.assertEqual(report["proposed_action_changed_rows"], 1)

  def test_best_of_rollouts_samples_common_actor_infostate_worlds(self):
    class _BeliefState:

      def __init__(self, marker="actual"):
        self.marker = marker

      def observation_tensor(self, player):
        del player
        return np.array([1.0, 2.0], dtype=np.float32)

      def legal_actions(self, player):
        del player
        return [1, 2]

      def resample_from_infostate(self, player, sampler):
        del player
        values = [1, 2, 3, 4]
        sampler.shuffle(values)
        return _BeliefState(tuple(values))

    state = _BeliefState()
    decision = {
        "state": state,
        "player": 0,
        "legal": [1, 2],
    }
    args = argparse.Namespace(
        rollout_state_source="infostate", belief_samples=3
    )

    first = best_of_rollouts_replay._rollout_states_for_decision(
        decision, seed_base=1234, args=args
    )
    second = best_of_rollouts_replay._rollout_states_for_decision(
        decision, seed_base=1234, args=args
    )
    held_out = best_of_rollouts_replay._rollout_states_for_decision(
        decision, seed_base=9876, args=args, belief_samples=2
    )

    self.assertLen(first, 3)
    self.assertTrue(all(world is not state for world in first))
    self.assertEqual(
        [world.marker for world in first], [world.marker for world in second]
    )
    self.assertLen(held_out, 2)

  def test_best_of_rollouts_pairs_continuation_seeds_across_actions(self):
    class _RolloutState:
      _num_card_types = 2
      _led_color = None

    worlds = [_RolloutState(), _RolloutState()]
    decision = {
        "state": worlds[0],
        "player": 0,
        "policy": np.array([0.0, 0.5, 0.5], dtype=np.float32),
    }
    args = argparse.Namespace(
        rollouts_per_action=2,
        survival_weight=10.0,
        actor_survival_weight=2.0,
        forced_paradox_penalty=1.0,
        actor_score_weight=0.2,
        total_score_weight=0.02,
    )
    calls = []

    def _outcome(model, state, action, value_scale, device, seed, call_args):
      del model, value_scale, device, call_args
      calls.append((id(state), action, seed))
      return {
          "any_paradox": False,
          "paradoxed": [False, False, False],
          "forced_paradox": False,
          "raw_scores": [1.0, 0.0, 0.0],
          "first_new_paradox_trigger": None,
      }

    with mock.patch.object(
        best_of_rollouts_replay,
        "rollout_from_action",
        side_effect=_outcome,
    ):
      for action in (1, 2):
        best_of_rollouts_replay._evaluate_action(
            None,
            decision,
            action,
            worlds,
            value_scale=20.0,
            device=torch.device("cpu"),
            seed_base=2**32 - 10,
            args=args,
        )

    action_one_seeds = [seed for _, action, seed in calls if action == 1]
    action_two_seeds = [seed for _, action, seed in calls if action == 2]
    self.assertEqual(action_one_seeds, action_two_seeds)
    self.assertLen(action_one_seeds, 4)
    self.assertTrue(all(0 <= seed < 2**32 - 1 for seed in action_one_seeds))
    self.assertTrue(any(seed < 1000000 for seed in action_one_seeds))

  def test_public_history_features_retain_public_sequence_only(self):
    class _HistoryState:
      _num_players = 3
      _num_tricks = 8
      _trick_number = 2
      _round_start_player = 0
      _start_player = 0
      _completed_tricks = [
          [(0, (2, "R")), (1, (6, "B")), (2, (4, "R"))],
          [(0, (5, "Y")), (1, (6, "G")), (2, (3, "Y"))],
      ]
      _cards_played_this_trick = [(2, "G"), (1, "R"), None]
      _hands = ["private-a", "private-b", "private-c"]
      _discarded_cards = [1, 2, 3]

    state = _HistoryState()
    features = public_history.public_history_features(state, player=2)
    index = {
        name: idx for idx, name in enumerate(public_history.PUBLIC_HISTORY_SCHEMA)
    }

    self.assertEqual(features.shape, (len(public_history.PUBLIC_HISTORY_SCHEMA),))
    self.assertAlmostEqual(features[index["public_play_progress_norm"]], 8 / 24)
    self.assertEqual(features[index["last_trick_own_leader"]], 1.0)
    self.assertEqual(features[index["last_trick_own_winner"]], 0.0)
    self.assertEqual(features[index["last_trick_led_yellow"]], 1.0)
    self.assertEqual(features[index["previous_play_red"]], 1.0)
    self.assertEqual(features[index["previous_play_own"]], 0.0)

    state._hands = ["different", "hidden", "cards"]
    state._discarded_cards = [6, 5, 4]
    np.testing.assert_array_equal(
        features,
        public_history.public_history_features(state, player=2),
    )
    action_context = public_history.public_history_action_features(
        features,
        candidate_follows_led=False,
        candidate_is_red=False,
        candidate_token_loss=True,
    )
    self.assertEqual(
        action_context.shape,
        (len(public_history.PUBLIC_HISTORY_ACTION_SCHEMA),),
    )

  def test_audit_base_keep_shapes_selector_auxiliary_labels_to_abstain(self):
    num_actions = 3
    action_features = np.zeros(
        (num_actions, ACTION_FEATURE_SIZE), dtype=np.float32
    )
    row = (
        np.zeros(8, dtype=np.float32),
        np.ones(num_actions, dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        0,
        0,
        np.array([0.8, 0.1, 0.2], dtype=np.float32),
        np.ones(num_actions, dtype=np.float32),
        np.array([0.0, 0.7, 0.4], dtype=np.float32),
        np.ones(num_actions, dtype=np.float32),
        action_features,
    )
    args = argparse.Namespace(
        allow_value_only_action_labels=False,
        base_action_source="policy_context_argmax",
        positive_value_margin=0.01,
        safe_risk_delta_max=0.0,
        force_led_lane_burn_negative=False,
        led_lane_red_exception="urgent_win",
        observation_feature_mode="none",
        policy_context_mode="none",
        target_mode="replay_action",
        soft_target_temperature=0.35,
        soft_target_risk_penalty=1.0,
        soft_target_unsafe_penalty=1.0,
    )
    metadata = [{"audit_base_keep_validation": True}]
    with mock.patch.object(
        legal_set_trainer,
        "_load_policy_context_model",
        return_value=(None, torch.device("cpu")),
    ), mock.patch.object(
        legal_set_trainer,
        "_policy_probs_from_row",
        return_value=np.array([0.8, 0.1, 0.1], dtype=np.float32),
    ):
      states, _ = legal_set_trainer._build_states(args, [row], metadata)

    state = states[0]
    self.assertEqual(state["target_action"], 0)
    for action in state["actions"]:
      if int(action["action"]) == 0:
        continue
      self.assertFalse(action["positive"])
      self.assertLessEqual(action["value_delta"], 0.0)
      self.assertGreaterEqual(action["risk_delta"], 0.0)

  def test_selector_metadata_base_action_uses_chained_policy_decision(self):
    self.assertEqual(
        legal_set_trainer._metadata_base_action({
            "decision": {"policy_action": 7},
            "best_action": 9,
            "proposed_best_action": 9,
        }),
        7,
    )
    self.assertEqual(
        legal_set_trainer._metadata_base_action({
            "policy_base_action": 3,
            "policy_best_action": 4,
            "decision": {"policy_action": 5},
        }),
        3,
    )
    self.assertEqual(
        legal_set_trainer._metadata_base_action({
            "policy_base_action": 6,
            "baseline_action": 6,
            "best_action": 9,
        }),
        6,
    )

  def test_action_set_selection_scope_filters_trick_bounds(self):
    args = argparse.Namespace(
        selection_min_trick_number=5,
        selection_max_trick_number=6,
    )
    states = [{"trick_number": value} for value in (-1, 4, 5, 6, 7)]
    self.assertEqual(
        action_set_trainer._selection_scope_states(args, states),
        [{"trick_number": 5}, {"trick_number": 6}],
    )

  def test_action_set_saved_selection_scope_fails_closed(self):
    bot = object.__new__(full_elo.AZActionSetTransformerBot)
    bot.args = argparse.Namespace(
        action_set_transformer_phases="",
        action_set_transformer_min_legal_count=0,
        action_set_transformer_max_legal_count=0,
        action_set_transformer_min_trick_number=-1,
        action_set_transformer_max_trick_number=-1,
    )
    bot.saved_args = {
        "selection_min_trick_number": 5,
        "selection_max_trick_number": 6,
    }
    state = argparse.Namespace(_trick_number=4)
    self.assertEqual(bot._enabled_for_state(state, 2), (False, "trick_number_low"))
    state._trick_number = 5
    self.assertEqual(bot._enabled_for_state(state, 2), (True, "enabled"))
    state._trick_number = 7
    self.assertEqual(bot._enabled_for_state(state, 2), (False, "trick_number_high"))

  def test_live_action_set_compact_feature_matches_trainer(self):
    num_actions = 4
    action_features = np.arange(
        num_actions * ACTION_FEATURE_SIZE, dtype=np.float32
    ).reshape(num_actions, ACTION_FEATURE_SIZE) / 100.0
    legal_mask = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    observation = np.linspace(-1.0, 1.0, 24, dtype=np.float32)
    policy = np.array([0.1, 0.5, 0.0, 0.4], dtype=np.float32)
    saved_args = {
        "pair_feature_mode": "compact_action_delta",
        "observation_feature_mode": "summary",
        "policy_context_mode": "summary",
    }
    bot = object.__new__(full_elo.AZActionSetTransformerBot)
    bot.saved_args = saved_args
    actual = bot._feature_for_action(
        action_features,
        legal_mask,
        observation,
        policy,
        3,
        1,
    )
    expected = legal_set_trainer._selector_pair_feature(
        argparse.Namespace(**saved_args),
        action_features,
        legal_mask > 0.5,
        3,
        1,
        observation,
        policy,
    )
    np.testing.assert_allclose(actual, expected)

  def test_selector_chained_audit_pair_excludes_unaudited_actions(self):
    labeled = np.zeros(10, dtype=bool)
    labeled[[1, 3, 5, 7]] = True
    actions = legal_set_trainer._chained_audit_pair_actions(
        {
            "target_kind": "audit_rejected_base_keep",
            "proposed_best_action": 7,
        },
        3,
        labeled,
    )
    self.assertEqual(actions, [3, 7])

  def test_selector_compact_pair_uses_candidate_and_base_delta(self):
    features = np.arange(20, dtype=np.float32).reshape(5, 4)
    legal = np.array([False, True, True, False, False])
    args = argparse.Namespace(
        pair_feature_mode="compact_action_delta",
        observation_feature_mode="none",
        policy_context_mode="none",
    )
    pair = legal_set_trainer._selector_pair_feature(
        args,
        features,
        legal,
        candidate_action=2,
        base_action=1,
        observation=np.zeros(1, dtype=np.float32),
        policy=np.zeros(5, dtype=np.float32),
    )
    self.assertLen(pair, 10)
    np.testing.assert_array_equal(pair[2:6], features[2])
    np.testing.assert_array_equal(pair[6:10], features[2] - features[1])

  def test_linear_action_set_masks_padded_scores(self):
    model = action_set_trainer.LinearActionSet(feature_size=4)
    x = torch.zeros((2, 3, 4), dtype=torch.float32)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    outputs = model(x, mask)
    self.assertLen(outputs, 5)
    for output in outputs:
      self.assertEqual(tuple(output.shape), (2, 3))
    self.assertLess(float(outputs[0][0, 2]), -1e8)
    self.assertLess(float(outputs[0][1, 1]), -1e8)

  def test_chained_rollout_requires_every_disjoint_audit_pass(self):
    args = argparse.Namespace(
        audit_passes=3,
        audit_belief_samples=4,
        audit_rollouts_per_action=2,
    )
    decision = {"action": 0, "player": 0}
    audit_results = [
        (True, "accepted", 0.5),
        (False, "weak", 0.0),
        (True, "accepted", 0.25),
    ]
    world_seeds = []
    continuation_seeds = []

    def _worlds(current_decision, seed_base, args, belief_samples):
      del current_decision, args
      world_seeds.append(int(seed_base))
      return [object()] * int(belief_samples)

    def _evaluate(
        model,
        current_decision,
        action,
        worlds,
        value_scale,
        device,
        seed_base,
        args,
        rollouts_per_action,
    ):
      del model, current_decision, worlds, value_scale, device, args
      del rollouts_per_action
      continuation_seeds.append(int(seed_base))
      return {"action": int(action)}

    with mock.patch.object(
        chained_rollout_replay,
        "_rollout_states_for_decision",
        side_effect=_worlds,
    ), mock.patch.object(
        chained_rollout_replay,
        "_evaluate_action",
        side_effect=_evaluate,
    ), mock.patch.object(
        chained_rollout_replay,
        "_audit_acceptance",
        side_effect=audit_results,
    ):
      result = chained_rollout_replay._audit_passes(
          None,
          decision,
          best_action=1,
          value_scale=20.0,
          device=torch.device("cpu"),
          seed_base=123,
          args=args,
      )

    self.assertFalse(result["all_passed"])
    self.assertTrue(result["error_free"])
    self.assertEqual(result["min_improvement"], 0.0)
    self.assertLen(set(world_seeds), 3)
    self.assertLen(set(continuation_seeds), 3)
    self.assertLen(result["passes"], 3)

  def test_chained_rollout_hard_targets_only_the_audited_action(self):
    selection_target = np.zeros(5, dtype=np.float32)
    selection_target[1] = 0.5
    selection_target[3] = 0.5

    audited = chained_rollout_replay._hard_audited_target(
        selection_target, best_action=3
    )

    np.testing.assert_array_equal(
        audited, np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    )

  def test_chained_checkpoint_threshold_counts_corrections_and_false_overrides(self):
    rows = [
        {
            "kind": "audited_correction",
            "base_action": 0,
            "target_action": 1,
            "proposal_action": 1,
            "top_action": 1,
            "top_vs_base_prob_margin": 0.2,
        },
        {
            "kind": "audit_rejected_base_keep",
            "base_action": 0,
            "target_action": 0,
            "proposal_action": 2,
            "top_action": 2,
            "top_vs_base_prob_margin": 0.1,
        },
        {
            "kind": "audit_rejected_base_keep",
            "base_action": 0,
            "target_action": 0,
            "proposal_action": 2,
            "top_action": 0,
            "top_vs_base_prob_margin": 0.0,
        },
    ]

    permissive = chained_checkpoint_audit._threshold_summary(rows, 0.0)
    strict = chained_checkpoint_audit._threshold_summary(rows, 0.15)

    self.assertEqual(permissive["overrides"], 2)
    self.assertEqual(permissive["correct_overrides"], 1)
    self.assertEqual(permissive["override_precision"], 0.5)
    self.assertEqual(permissive["correction_recall"], 1.0)
    self.assertEqual(permissive["base_keep_rate"], 0.5)
    self.assertEqual(strict["overrides"], 1)
    self.assertEqual(strict["override_precision"], 1.0)
    self.assertEqual(strict["base_keep_rate"], 1.0)

  def test_chained_replay_audit_checks_targets_passes_and_seed_disjointness(self):
    def _record(action, any_rate):
      return {
          "action": int(action),
          "any_paradox_rate": float(any_rate),
          "actor_paradox_rate": float(any_rate),
          "forced_paradox_rate": float(any_rate),
          "rollout_worlds": 4,
          "rollouts_per_world": 1,
          "rollouts": 4,
      }

    target = np.zeros(5, dtype=np.float32)
    target[1] = 1.0
    legal_mask = np.zeros(5, dtype=np.float32)
    legal_mask[[0, 1]] = 1.0
    example = (
        np.zeros(4, dtype=np.float32),
        legal_mask,
        target,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        1,
        0,
    )
    pass_improvements = (0.5, 0.25, 0.75)
    row = {
        "source_group_id": "source-1",
        "trajectory_id": "source-1:match=0:round=0",
        "match_index": 0,
        "round_index": 0,
        "target_kind": "audited_correction",
        "decision": {
            "index": 0,
            "player": 0,
            "trick_number": 0,
            "policy_action": 0,
            "legal_count": 2,
            "legal_actions": [0, 1],
        },
        "best_action": 1,
        "proposed_best_action": 1,
        "applied_action": 1,
        "policy_target_changed": True,
        "audit_override_rejected": False,
        "audit_base_keep_validation": False,
        "public_history": {
            "version": public_history.PUBLIC_HISTORY_VERSION,
            "schema": list(public_history.PUBLIC_HISTORY_SCHEMA),
            "features": [0.0] * len(public_history.PUBLIC_HISTORY_SCHEMA),
        },
        "public_history_trace_contract": "chained_best_of_rollouts_v3",
        "public_play_trace_length": 0,
        "public_play_trace_before": [],
        "num_players": 3,
        "num_tricks": 8,
        "num_card_types": 6,
        "start_player": 0,
        "selection_best_vs_base_any_paradox_improvement": 0.5,
        "rollout_records": [_record(0, 0.75), _record(1, 0.25)],
        "all_audit_passes_passed": True,
        "audit_min_best_vs_base_any_paradox_improvement": 0.25,
        "audit_passes": [
            {
                "pass_index": index,
                "world_seed": 100 + index,
                "continuation_seed": 200 + index,
                "passed": True,
                "error": False,
                "reason": "accepted",
                "improvement": improvement,
                "records": [
                    _record(0, 1.0),
                    _record(1, 1.0 - improvement),
                ],
            }
            for index, improvement in enumerate(pass_improvements)
        ],
        "corrections_before": 0,
        "prior_audited_corrections": [],
    }

    errors, audited = chained_replay_audit._audit_row(
        example,
        row,
        kind="audited_correction",
        source_group="source-1",
        required_passes=3,
        min_selection_improvement=0.5,
        min_audit_improvement=0.25,
        label="fixture",
    )

    self.assertEmpty(errors)
    self.assertEqual(audited["audit_improvements"], [0.5, 0.25, 0.75])

    broken = json.loads(json.dumps(row))
    broken["audit_passes"][1]["continuation_seed"] = 100
    broken["audit_passes"][1]["records"][1]["any_paradox_rate"] = 1.0
    broken_errors, _ = chained_replay_audit._audit_row(
        example,
        broken,
        kind="audited_correction",
        source_group="source-1",
        required_passes=3,
        min_selection_improvement=0.5,
        min_audit_improvement=0.25,
        label="broken",
    )

    self.assertTrue(any("seeds are not disjoint" in item for item in broken_errors))
    self.assertTrue(any("improvement disagrees" in item for item in broken_errors))
    self.assertTrue(any("passed flag disagrees" in item for item in broken_errors))


if __name__ == "__main__":
  absltest.main()
