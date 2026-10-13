#!/usr/bin/env python3
"""Tests for promotion-aligned label-oracle risk semantics."""

from types import SimpleNamespace
import unittest

import numpy as np

from quantum_cat_label_oracle_audit import scoped_full_match_paradox_count


class QuantumCatLabelOracleAuditTest(unittest.TestCase):

  def test_full_match_paradox_scope_matches_requested_gate(self):
    paradox_counts = np.array([2.0, 1.0, 0.0], dtype=np.float32)

    acting = scoped_full_match_paradox_count(
        paradox_counts,
        any_paradox_rounds=3.0,
        acting_player=1,
        args=SimpleNamespace(counterfactual_action_paradox_scope="acting"),
    )
    any_player = scoped_full_match_paradox_count(
        paradox_counts,
        any_paradox_rounds=3.0,
        acting_player=1,
        args=SimpleNamespace(counterfactual_action_paradox_scope="any"),
    )

    self.assertEqual(acting, 1.0)
    self.assertEqual(any_player, 3.0)


if __name__ == "__main__":
  unittest.main()
