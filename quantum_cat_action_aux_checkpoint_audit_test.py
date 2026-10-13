#!/usr/bin/env python3

import unittest

from quantum_cat_action_aux_checkpoint_audit import (
    _passes,
    _reports_by_source,
    _threshold_summary,
)


class ActionAuxCheckpointAuditTest(unittest.TestCase):

  def test_threshold_summary_distinguishes_corrections_and_base_keeps(self):
    rows = [
        {"kind": "audited_correction", "predicted_improvement": 0.2},
        {"kind": "audited_correction", "predicted_improvement": 0.1},
        {"kind": "audit_rejected_base_keep", "predicted_improvement": 0.03},
        {"kind": "audit_rejected_base_keep", "predicted_improvement": -0.1},
    ]
    report = _threshold_summary(rows, 0.05)
    self.assertEqual(report["overrides"], 2)
    self.assertEqual(report["correct_overrides"], 2)
    self.assertEqual(report["override_precision"], 1.0)
    self.assertEqual(report["correction_recall"], 1.0)
    self.assertEqual(report["base_keep_rate"], 1.0)

  def test_gate_requires_every_source_to_be_safe(self):
    aggregate = {
        "overrides": 10,
        "override_precision": 0.9,
        "base_keep_rate": 0.95,
    }
    by_source = {
        "a": {
            "positive_rows": 5,
            "overrides": 5,
            "override_precision": 1.0,
            "base_keep_rows": 20,
            "base_keep_rate": 1.0,
        },
        "b": {
            "positive_rows": 5,
            "overrides": 5,
            "override_precision": 0.8,
            "base_keep_rows": 20,
            "base_keep_rate": 0.9,
        },
    }
    self.assertFalse(_passes(
        aggregate,
        by_source,
        min_overrides=8,
        min_precision=0.9,
        min_base_keep_rate=0.9,
        min_source_overrides=1,
    ))

  def test_reports_use_embedded_source_groups_not_replay_container(self):
    rows = [
        {
            "kind": "audited_correction",
            "predicted_improvement": 0.2,
            "source": "calibration:0:combined.npz",
            "source_group_id": "seed-a",
        },
        {
            "kind": "audit_rejected_base_keep",
            "predicted_improvement": -0.1,
            "source": "calibration:0:combined.npz",
            "source_group_id": "seed-b",
        },
    ]
    reports = _reports_by_source(rows, 0.05)
    self.assertEqual(set(reports), {"seed-a", "seed-b"})
    self.assertEqual(reports["seed-a"]["positive_rows"], 1)
    self.assertEqual(reports["seed-b"]["base_keep_rows"], 1)

  def test_gate_accepts_common_safe_threshold(self):
    aggregate = {
        "overrides": 9,
        "override_precision": 1.0,
        "base_keep_rate": 0.95,
    }
    by_source = {
        "a": {
            "positive_rows": 4,
            "overrides": 4,
            "override_precision": 1.0,
            "base_keep_rows": 20,
            "base_keep_rate": 0.95,
        },
        "b": {
            "positive_rows": 5,
            "overrides": 5,
            "override_precision": 1.0,
            "base_keep_rows": 20,
            "base_keep_rate": 0.95,
        },
    }
    self.assertTrue(_passes(
        aggregate,
        by_source,
        min_overrides=8,
        min_precision=0.9,
        min_base_keep_rate=0.9,
        min_source_overrides=1,
    ))


if __name__ == "__main__":
  unittest.main()
