#!/usr/bin/env python3
"""Tests for Q replay derivation utilities."""

from absl.testing import absltest
import numpy as np

from quantum_cat_alphazero_torch import ACTION_FEATURE_IS_RED_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_RANK_NORM_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_SIZE
from quantum_cat_q_replay_tools import derive_policy_replay
from quantum_cat_q_teacher_stability_audit import _stable_replay_reason
from quantum_cat_q_teacher_stability_audit import _target_bucket_candidate_actions
from quantum_cat_tactical_bucket_replay import select_bucket_balanced_replay


def _bucket_example(target_action, feature_updates):
  action_features = np.zeros((4, ACTION_FEATURE_SIZE), dtype=np.float32)
  action_features[:, ACTION_FEATURE_RANK_NORM_INDEX] = 1.0
  for feature_idx, value in feature_updates.items():
    action_features[target_action, feature_idx] = value
  policy = np.zeros(4, dtype=np.float32)
  policy[target_action] = 1.0
  return (
      np.zeros(8, dtype=np.float32),
      np.ones(4, dtype=np.float32),
      policy,
      np.zeros(3, dtype=np.float32),
      np.zeros(3, dtype=np.float32),
      -1,
      -1,
      None,
      None,
      None,
      None,
      action_features,
  )


class QuantumCatQReplayToolsTest(absltest.TestCase):

  def test_derive_policy_replay_treats_positive_mask_weights_as_labeled(self):
    num_actions = 4
    action_features = np.zeros(
        (num_actions, ACTION_FEATURE_SIZE), dtype=np.float32
    )
    action_features[:, 4] = 1.0  # phase=play in columns 1:6.
    action_features[:, 8] = 1.0  # action type=play in columns 6:11.
    example = (
        np.zeros(8, dtype=np.float32),
        np.ones(num_actions, dtype=np.float32),
        np.array([0.70, 0.10, 0.10, 0.10], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        -1,
        -1,
        None,
        None,
        np.array([0.0, 0.2, -0.1, 0.0], dtype=np.float32),
        np.array([0.25, 0.25, 0.25, 0.0], dtype=np.float32),
        action_features,
    )
    args = type("Args", (), {
        "phases": "play",
        "min_actions": 2,
        "min_spread": 0.1,
        "min_top_margin": 0.05,
        "min_original_top_regret": 0.0,
        "temperature": 0.06,
        "q_policy_weight": 0.85,
        "include_non_overrides": False,
        "keep_action_value_labels": True,
    })()

    derived, summary = derive_policy_replay([example], args)

    self.assertLen(derived, 1)
    self.assertEqual(summary["derived_rows"], 1)
    self.assertEqual(summary["q_top_changed_rows"], 1)
    self.assertEqual(summary["skipped"]["min_actions"], 0)

  def test_derive_policy_replay_uses_metadata_original_policy_top(self):
    num_actions = 4
    action_features = np.zeros(
        (num_actions, ACTION_FEATURE_SIZE), dtype=np.float32
    )
    action_features[:, 4] = 1.0  # phase=play in columns 1:6.
    action_features[:, 8] = 1.0  # action type=play in columns 6:11.
    example = (
        np.zeros(8, dtype=np.float32),
        np.ones(num_actions, dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        -1,
        -1,
        None,
        None,
        np.array([0.0, 0.4, 0.1, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        action_features,
    )
    args = type("Args", (), {
        "phases": "play",
        "min_actions": 2,
        "min_spread": 0.1,
        "min_top_margin": 0.05,
        "min_original_top_regret": 0.2,
        "temperature": 0.06,
        "q_policy_weight": 0.85,
        "include_non_overrides": False,
        "keep_action_value_labels": True,
    })()

    without_metadata, without_summary = derive_policy_replay([example], args)
    with_metadata, with_summary = derive_policy_replay(
        [example], args, [{"policy_best_action": 2}]
    )

    self.assertEmpty(without_metadata)
    self.assertEqual(without_summary["skipped"]["original_regret"], 1)
    self.assertLen(with_metadata, 1)
    self.assertEqual(with_summary["metadata_original_top_rows"], 1)
    self.assertEqual(with_summary["q_top_changed_rows"], 1)

  def test_tactical_bucket_balancer_selects_unique_multi_bucket_rows(self):
    replay = [
        _bucket_example(
            1,
            {
                ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX: 1.0,
                ACTION_FEATURE_IS_RED_INDEX: 1.0,
            },
        ),
        _bucket_example(2, {}),
    ]
    args = type("Args", (), {
        "bucket_targets": "token_loss=1,red=1",
        "phases": "",
        "min_actions": 0,
        "min_spread": 0.0,
        "min_top_margin": 0.0,
        "seed": 7,
        "include_unbucketed": 0,
    })()

    selected, summary = select_bucket_balanced_replay(replay, args)

    self.assertLen(selected, 1)
    self.assertEqual(summary["selected_bucket_counts"]["token_loss"], 1)
    self.assertEqual(summary["selected_bucket_counts"]["red"], 1)

  def test_stable_replay_can_require_target_action_bucket_match(self):
    args = type("Args", (), {
        "stable_require_high_margin": False,
        "stable_include_policy_matches": True,
        "stable_min_top1_mode_rate": 0.0,
        "stable_min_spearman": -1.0,
        "stable_min_forced_margin": -100.0,
        "stable_max_forced_paradox_delta": 10.0,
        "stable_min_legal_count": 0,
        "stable_min_q_spread": 0.0,
        "stable_require_recall_consensus": False,
        "stable_use_recall_labels": False,
        "stable_require_target_bucket_match": True,
        "value_scale": 20.0,
    })()
    row = {
        "evaluable": True,
        "high_margin": False,
        "consensus_matches_policy": False,
        "top1_mode_rate": 0.5,
        "pairwise_spearman_mean": 0.0,
        "legal_count": 4,
        "consensus_best_action": 2,
        "mean_raw_scores": {"1": 0.0, "2": 1.0},
        "forced_audit": {
            "audited": True,
            "margin_vs_policy": 0.0,
            "paradox_delta_vs_policy": 0.0,
        },
        "group": {
            "state_mix_bucket": "follows_led",
            "tactical_bucket_matches": {"follows_led": [1]},
        },
    }

    self.assertEqual(_stable_replay_reason(row, args), "target_bucket_mismatch")

    row["group"]["tactical_bucket_matches"]["follows_led"].append(2)
    self.assertIsNone(_stable_replay_reason(row, args))

  def test_recall_labeled_stable_replay_uses_all_legal_best_bucket_match(self):
    args = type("Args", (), {
        "stable_require_high_margin": True,
        "stable_include_policy_matches": False,
        "stable_min_top1_mode_rate": 0.75,
        "stable_min_spearman": 0.35,
        "stable_min_forced_margin": 0.5,
        "stable_max_forced_paradox_delta": 0.0,
        "stable_min_legal_count": 0,
        "stable_min_q_spread": 0.0,
        "stable_require_recall_consensus": False,
        "stable_use_recall_labels": True,
        "stable_require_target_bucket_match": True,
        "min_high_margin": 0.5,
        "value_scale": 20.0,
    })()
    row = {
        "evaluable": True,
        "policy_best_action": 0,
        "legal_count": 4,
        "candidate_recall_audit": {
            "audited": True,
            "all_legal_best_action": 2,
            "all_legal_top_margin": 1.0,
            "all_legal_best_margin_vs_policy": 2.0,
            "all_legal_paradox_delta_vs_policy": 0.0,
            "all_legal_raw_scores": {
                "0": 0.0,
                "1": 1.0,
                "2": 2.0,
                "3": -1.0,
            },
        },
        "group": {
            "state_mix_bucket": "red",
            "tactical_bucket_matches": {"red": [1]},
        },
    }

    self.assertEqual(_stable_replay_reason(row, args), "target_bucket_mismatch")

    row["group"]["tactical_bucket_matches"]["red"].append(2)
    self.assertIsNone(_stable_replay_reason(row, args))

  def test_target_bucket_candidate_actions_rank_by_policy(self):
    group = {
        "state_mix_bucket": "mid_rank",
        "tactical_bucket_matches": {"mid_rank": [5, 3, 9]},
    }
    policy = np.zeros(10, dtype=np.float32)
    policy[3] = 0.2
    policy[5] = 0.7
    policy[9] = 0.5

    selected = _target_bucket_candidate_actions(
        group, legal=[3, 5, 9], policy=policy, cap=2
    )

    self.assertEqual(selected, [5, 9])


if __name__ == "__main__":
  absltest.main()
