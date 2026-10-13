import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import quantum_cat_root_rollout_replay as replay
from quantum_cat_alphazero_torch import ACTION_FEATURE_SIZE
from quantum_cat_table_liveness_improve_trainer import _load_metadata_rows


def _sample(accepted):
  return {
      "schema_version": 1,
      "information_safe": True,
      "player": 1,
      "phase": "play",
      "trick_number": 6,
      "search_seed": 7,
      "observation": [0.0] * 76,
      "legal_actions": [2, 5, 7],
      "legal_action_features": np.zeros(
          (3, ACTION_FEATURE_SIZE), dtype=np.float32
      ).tolist(),
      "legal_policy": [0.5, 0.3, 0.2],
      "baseline_action": 2,
      "proposed_action": 5,
      "selected_action": 5 if accepted else 2,
      "audit_accepted": accepted,
      "audit_contract": {
          "baseline_mode": "liveness_shield",
          "samples": 32,
          "rollouts_per_world": 1,
          "required_passes": 2,
          "min_paradox_improvement": 0.125,
          "max_paired_binomial_p": 0.1,
          "inference_unit": "information_state_world",
      },
      "selection": {"valid": False},
      "audit_passes": [
          {
              "valid": True,
              "worlds": 32,
              "baseline_paradox": 0.75,
              "proposed_paradox": 0.25,
              "paradox_improvement": 0.5,
              "saved": 4,
              "harmed": 0,
              "paired_binomial_p": 0.0625,
          },
          {
              "valid": True,
              "worlds": 32,
              "baseline_paradox": 0.625,
              "proposed_paradox": 0.25,
              "paradox_improvement": 0.375,
              "saved": 4,
              "harmed": 0,
              "paired_binomial_p": 0.0625,
          },
      ],
  }


class RootRolloutReplayTest(unittest.TestCase):

  def test_accepted_sample_targets_proposal_and_carries_risk_pair(self):
    row = replay.replay_row(_sample(True), players=3)
    self.assertEqual(row[5], 5)
    self.assertEqual(np.flatnonzero(row[2]).tolist(), [5])
    self.assertEqual(np.flatnonzero(row[8]).tolist(), [2, 5])
    self.assertAlmostEqual(row[7][2], 0.6875)
    self.assertAlmostEqual(row[7][5], 0.25)
    self.assertAlmostEqual(row[9][2], -0.6875)
    self.assertAlmostEqual(row[9][5], -0.25)
    self.assertEqual(np.flatnonzero(row[10]).tolist(), [2, 5])
    self.assertEqual(row[11].shape, (1000, ACTION_FEATURE_SIZE))

  def test_rejected_sample_is_hard_base_keep(self):
    row = replay.replay_row(_sample(False), players=3)
    self.assertEqual(row[5], 2)
    self.assertEqual(np.flatnonzero(row[2]).tolist(), [2])

  def test_hidden_or_inconsistent_samples_fail_closed(self):
    sample = _sample(True)
    sample["information_safe"] = False
    with self.assertRaisesRegex(ValueError, "not information-safe"):
      replay.replay_row(sample, players=3)
    sample = _sample(True)
    sample["selected_action"] = 7
    with self.assertRaisesRegex(ValueError, "accepted/selected"):
      replay.replay_row(sample, players=3)

  def test_accepted_sample_must_independently_clear_recorded_contract(self):
    sample = _sample(True)
    sample["audit_passes"][1]["paired_binomial_p"] = 0.11
    with self.assertRaisesRegex(ValueError, "paired p threshold"):
      replay.replay_row(sample, players=3)
    sample = _sample(True)
    sample["audit_passes"][0]["worlds"] = 31
    with self.assertRaisesRegex(ValueError, "wrong world count"):
      replay.replay_row(sample, players=3)

  def test_conversion_retains_durable_selection_and_audit_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "screen.json"
      source.write_text(json.dumps({
          "players": 3,
          "decision_stats": {
              "c0": {
                  "root_rollout_overrides": 1,
                  "root_rollout_replay_samples": [_sample(True)],
              }
          },
      }), encoding="utf-8")
      out = root / "replay.npz"
      metadata_out = root / "replay.metadata.json"
      report = replay.convert([source], out, metadata_out, max_samples=0)
      metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
      self.assertEqual(report["rows"], 1)
      self.assertEqual(metadata["row_metadata"][0]["search_seed"], 7)
      self.assertEqual(metadata["row_metadata"][0]["phase"], "play")
      self.assertEqual(
          metadata["row_metadata"][0]["audit_contract"]["samples"], 32
      )
      self.assertEqual(
          metadata["row_metadata"][0]["audit_contract"]["baseline_mode"],
          "liveness_shield",
      )
      self.assertEqual(
          metadata["row_metadata"][0]["selection"], {"valid": False}
      )
      self.assertEqual(
          metadata["row_metadata"][0]["target_kind"], "audited_correction"
      )
      self.assertEqual(metadata["row_metadata"][0]["policy_base_action"], 2)
      self.assertEqual(metadata["row_metadata"][0]["proposed_best_action"], 5)
      self.assertFalse(
          metadata["row_metadata"][0]["audit_base_keep_validation"]
      )
      self.assertEqual(_load_metadata_rows(metadata_out), metadata["row_metadata"])
      self.assertTrue(out.is_file())

  def test_conversion_rejects_missing_accepted_rows(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "screen.json"
      source.write_text(json.dumps({
          "players": 3,
          "decision_stats": {
              "c0": {
                  "root_rollout_overrides": 2,
                  "root_rollout_replay_samples": [_sample(True)],
              }
          },
      }), encoding="utf-8")
      with self.assertRaisesRegex(
          ValueError, "recorded 1 accepted rows, expected 2"
      ):
        replay.convert(
            [source], root / "replay.npz", root / "metadata.json", 0
        )

  def test_conversion_repeats_corrections_without_inflating_unique_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "screen.json"
      rejected = _sample(False)
      rejected["search_seed"] = 8
      source.write_text(json.dumps({
          "players": 3,
          "decision_stats": {
              "c0": {
                  "root_rollout_overrides": 1,
                  "root_rollout_replay_samples": [_sample(True), rejected],
              }
          },
      }), encoding="utf-8")
      out = root / "replay.npz"
      metadata_out = root / "replay.metadata.json"
      report = replay.convert(
          [source], out, metadata_out, max_samples=0, accepted_repeat=3
      )
      metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
      self.assertEqual(report["rows"], 4)
      self.assertEqual(report["unique_rows"], 2)
      self.assertEqual(report["accepted_rows"], 3)
      self.assertEqual(report["unique_accepted_rows"], 1)
      self.assertEqual(report["base_keep_rows"], 1)
      self.assertEqual(report["unique_base_keep_rows"], 1)
      self.assertEqual(report["accepted_repeat"], 3)
      self.assertEqual(
          [row["unique_source_row_index"] for row in metadata["row_metadata"]],
          [0, 0, 0, 1],
      )


if __name__ == "__main__":
  unittest.main()
