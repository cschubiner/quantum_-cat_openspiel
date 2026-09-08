import unittest

import numpy as np

from quantum_cat_root_rollout_policy_audit import policy_audit_report


def _row(kind, baseline, proposed, selected, source, trick):
  return {
      "target_kind": kind,
      "audit_accepted": kind == "audited_correction",
      "baseline_action": baseline,
      "proposed_action": proposed,
      "selected_action": selected,
      "source_group_id": source,
      "trick_number": trick,
  }


class RootRolloutPolicyAuditTest(unittest.TestCase):

  def test_report_separates_corrections_from_base_keep_safety(self):
    policies = np.zeros((4, 6), dtype=np.float32)
    policies[np.arange(4), [2, 3, 1, 4]] = 1.0
    rows = [
        _row("audited_correction", 1, 2, 2, "a", 5),
        _row("audited_correction", 1, 3, 3, "b", 6),
        _row("audit_rejected_base_keep", 1, 2, 1, "a", 5),
        _row("audit_rejected_base_keep", 4, 3, 4, "b", 7),
    ]
    anchor = np.asarray([
        [0, .7, .3, 0, 0, 0],
        [0, .7, 0, .3, 0, 0],
        [0, .7, .3, 0, 0, 0],
        [0, 0, 0, .3, .7, 0],
    ], dtype=np.float32)
    candidate = np.asarray([
        [0, .3, .7, 0, 0, 0],
        [0, .3, 0, .7, 0, 0],
        [0, .6, .4, 0, 0, 0],
        [0, 0, 0, .2, .8, 0],
    ], dtype=np.float32)
    report = policy_audit_report(
        candidate,
        anchor,
        policies,
        rows,
        min_correction_rows=2,
        min_correction_top1_rate=1.0,
        min_correction_target_prob_gain=0.3,
        max_base_keep_top1_drop=0.0,
        max_unsafe_proposal_adoption_rate=0.0,
    )
    self.assertEqual(report["correction"]["candidate_target_top1_rate"], 1.0)
    self.assertEqual(report["base_keep"]["candidate_target_top1_rate"], 1.0)
    self.assertEqual(report["unsafe_proposal_adoption"]["rate"], 0.0)
    self.assertEqual(report["source_group_count"], 2)
    self.assertTrue(report["gate"]["pass"])

  def test_inconsistent_audit_target_fails_closed(self):
    policies = np.zeros((1, 4), dtype=np.float32)
    policies[0, 2] = 1.0
    rows = [_row("audit_rejected_base_keep", 1, 2, 2, "a", 5)]
    probs = np.asarray([[0, .5, .5, 0]], dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "violates audit result"):
      policy_audit_report(probs, probs, policies, rows)


if __name__ == "__main__":
  unittest.main()
