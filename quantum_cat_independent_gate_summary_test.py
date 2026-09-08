import json
import tempfile
import unittest
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


SCRIPT = (
    Path(__file__).parent
    / ".codex/skills/test-model/scripts/summarize_independent_model_gates.py"
)
SPEC = spec_from_file_location("independent_gate_summary", SCRIPT)
summary = module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _gate(
    seed,
    successes=650,
    trials=1800,
    upper=0.3836,
    passed=True,
    placement_lower=0.54,
):
  return {
      "checkpoint": "candidate.pt",
      "checkpoint_sha256": "abc",
      "incumbent_checkpoint_sha256": "frozen-reference",
      "incumbent_mode": "policy",
      "mode": "policy",
      "led_mode": "policy",
      "players": 3,
      "seed": seed,
      "led_seed": seed + 101,
      "competitive_promotion_gate": {
          "eligible": True,
          "passed": passed,
          "matches": 600,
          "independent_deal_blocks": 200,
          "seat_balanced_within_each_deal": True,
          "pairwise": {"score_rate": 0.61},
          "placement_advantage": {"ci95": [placement_lower, 0.68]},
          "official_score_advantage": {"ci95": [0.2, 1.0]},
          "own_paradox_rate_difference": {"ci95": [-0.03, 0.01]},
          "max_own_paradox_rate_regression": 0.0,
          "placement_advantage_passed": passed,
          "official_score_guard_passed": True,
          "own_paradox_guard_passed": True,
      },
      "homogeneous_paradox_gate": {
          "matches": 600,
          "hand_rounds": trials,
          "rounds_with_any_paradox": successes,
          "same_policy_paradox_round_rate_ci95": [0.3392, upper],
          "threshold": 0.4,
          "passed": passed,
          "ci95_upper_below_threshold": passed,
      },
  }


class IndependentGateSummaryTest(unittest.TestCase):

  def _write(self, root, index, payload):
    path = root / f"gate_{index}" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    players = int(payload["players"])
    ids, families = summary._expected_bidding_scenarios(players)
    bidding = {
        "suiteID": "quantum-cat-curated-bidding-validation-v1",
        "mode": "candidate_bid_validation",
        "candidate": f"{payload['checkpoint']}:{payload['mode']}",
        "scenarioCount": len(ids),
        "passedCount": len(ids),
        "independentScenarioFamilyCount": families,
        "passed": True,
        "results": [
            {"id": scenario, "players": players, "passed": True}
            for scenario in sorted(ids)
        ],
    }
    (root / "curated_bidding_validation.json").write_text(
        json.dumps(bidding), encoding="utf-8"
    )

  def test_rejects_missing_locked_bidding_suite(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        self._write(root, index, _gate(100 + index))
      (root / "curated_bidding_validation.json").unlink()
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(report["passed"])
      self.assertIn("missing required curated bidding validation", report["errors"])

  def test_rejects_incomplete_or_schema_only_bidding_suite(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        self._write(root, index, _gate(100 + index))
      path = root / "curated_bidding_validation.json"
      bidding = json.loads(path.read_text(encoding="utf-8"))
      bidding["mode"] = "schema_and_state_validation_only"
      bidding["results"].pop()
      path.write_text(json.dumps(bidding), encoding="utf-8")
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(report["passed"])
      self.assertTrue(any("bidding" in error for error in report["errors"]))

  def test_requires_every_independent_gate_even_when_pool_is_good(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._write(root, 1, _gate(101))
      self._write(
          root,
          2,
          _gate(
              102, successes=710, upper=0.418, passed=False,
              placement_lower=0.47,
          ),
      )
      self._write(root, 3, _gate(103, successes=580, upper=0.344))
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(report["passed"])
      self.assertTrue(report["pooled"]["informational_only"])
      self.assertTrue(any("gate 2" in item for item in report["errors"]))

  def test_passes_three_unique_consistent_full_size_gates(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        self._write(root, index, _gate(100 + index))
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertTrue(report["passed"])
      self.assertTrue(all(item["passed"] for item in report["gates"]))

  def test_homogeneous_table_paradox_failure_is_diagnostic_only(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index, successes=1700, upper=0.98)
        payload["homogeneous_paradox_gate"]["threshold"] = 0.5
        payload["homogeneous_paradox_gate"]["passed"] = False
        payload["homogeneous_paradox_gate"]["ci95_upper_below_threshold"] = (
            False
        )
        self._write(root, index, payload)
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertTrue(report["passed"])
      self.assertFalse(report["table_paradox_is_promotion_veto"])

  def test_rejects_duplicate_led_seeds(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index)
        payload["led_seed"] = 999
        self._write(root, index, payload)
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(report["passed"])
      self.assertIn("gate led seeds are not unique", report["errors"])

  def test_rejects_mismatched_led_mode_and_player_count(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index)
        payload["led_mode"] = "q_policy"
        payload["players"] = 4 if index == 3 else 3
        self._write(root, index, payload)
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(report["passed"])
      self.assertIn(
          "gate led-audit mode does not match gameplay mode", report["errors"]
      )
      self.assertIn(
          "gate player counts are missing or inconsistent", report["errors"]
      )

  def test_rejects_mismatched_frozen_incumbent_hash_or_mode(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index)
        if index == 3:
          payload["incumbent_checkpoint_sha256"] = "different-reference"
          payload["incumbent_mode"] = "q_policy"
        self._write(root, index, payload)
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(report["passed"])
      self.assertIn(
          "gate frozen incumbent hashes are missing or inconsistent",
          report["errors"],
      )
      self.assertIn(
          "gate frozen incumbent modes are missing or inconsistent",
          report["errors"],
      )

  def test_rejects_pseudoreplicated_or_unbalanced_competitive_gates(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index)
        if index == 2:
          payload["competitive_promotion_gate"]["independent_deal_blocks"] = 20
        if index == 3:
          payload["competitive_promotion_gate"][
              "seat_balanced_within_each_deal"
          ] = False
        self._write(root, index, payload)
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(report["passed"])
      self.assertTrue(any(
          "independent deal blocks 20 < 200" in error
          for error in report["errors"]
      ))
      self.assertTrue(any(
          "seats are not balanced" in error for error in report["errors"]
      ))

  def test_rejects_relaxed_or_contradicted_personal_paradox_guard(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index)
        if index == 2:
          payload["competitive_promotion_gate"][
              "max_own_paradox_rate_regression"
          ] = 1.0
        if index == 3:
          payload["competitive_promotion_gate"][
              "own_paradox_rate_difference"
          ]["ci95"] = [0.02, 0.09]
        self._write(root, index, payload)
      report = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(report["passed"])
      self.assertTrue(any(
          "regression limit 1.0 must equal 0" in error
          for error in report["errors"]
      ))
      self.assertTrue(any(
          "regression CI lower 0.02 exceeds 0" in error
          for error in report["errors"]
      ))

  def test_rejects_relaxed_formal_match_or_deal_minimum(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        self._write(root, index, _gate(100 + index))
      report = summary.summarize(
          root,
          expected_gates=3,
          min_matches=8,
          min_independent_deals=2,
      )
      self.assertFalse(report["passed"])
      self.assertIn(
          "minimum matches 8 < formal minimum 600", report["errors"]
      )
      self.assertTrue(any(
          "required independent deal blocks 2 < formal minimum 200" in error
          for error in report["errors"]
      ))

  def test_four_player_formal_gate_requires_matching_locked_tactical_suite(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index)
        payload["players"] = 4
        self._write(root, index, payload)

      missing = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(missing["passed"])
      self.assertIn(
          "missing required four-player curated tactical validation",
          missing["errors"],
      )

      tactical = root / "curated_tactical_validation.json"
      expected_ids, family_count = summary._expected_tactical_scenarios(4)
      tactical.write_text(json.dumps({
          "suiteID": "quantum-cat-curated-tactical-validation-v1",
          "candidate": "candidate.pt:policy",
          "scenarioCount": len(expected_ids),
          "passedCount": len(expected_ids),
          "independentScenarioFamilyCount": family_count,
          "passed": True,
          "results": [
              {"id": scenario_id, "players": 4, "passed": True}
              for scenario_id in sorted(expected_ids)
          ],
      }), encoding="utf-8")
      passed = summary.summarize(
          root,
          expected_gates=3,
          min_matches=600,
          curated_tactical=tactical,
      )
      self.assertTrue(passed["passed"])
      self.assertTrue(passed["curated_tactical"]["required"])

      broken = json.loads(tactical.read_text(encoding="utf-8"))
      broken["candidate"] = "different.pt:policy"
      broken["results"][1]["passed"] = False
      tactical.write_text(json.dumps(broken), encoding="utf-8")
      rejected = summary.summarize(
          root,
          expected_gates=3,
          min_matches=600,
          curated_tactical=tactical,
      )
      self.assertFalse(rejected["passed"])
      self.assertIn(
          "curated tactical candidate does not match gated model",
          rejected["errors"],
      )
      self.assertIn(
          "one or more four-player tactical fixtures failed",
          rejected["errors"],
      )

  def test_five_player_formal_gate_requires_exact_official_scoring_cases(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index)
        payload["players"] = 5
        self._write(root, index, payload)

      missing = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(missing["passed"])
      self.assertIn(
          "missing required five-player curated tactical validation",
          missing["errors"],
      )

      expected_ids, family_count = summary._expected_tactical_scenarios(5)
      tactical = root / "curated_tactical_validation.json"
      tactical.write_text(json.dumps({
          "suiteID": "quantum-cat-curated-tactical-validation-v1",
          "candidate": "candidate.pt:policy",
          "scenarioCount": len(expected_ids),
          "passedCount": len(expected_ids),
          "independentScenarioFamilyCount": family_count,
          "passed": True,
          "results": [
              {"id": scenario_id, "players": 5, "passed": True}
              for scenario_id in sorted(expected_ids)
          ],
      }), encoding="utf-8")
      accepted = summary.summarize(
          root, expected_gates=3, min_matches=600, curated_tactical=tactical
      )
      self.assertTrue(accepted["passed"])
      self.assertEqual(accepted["curated_tactical"]["expected_scenario_count"], 15)
      self.assertEqual(accepted["curated_tactical"]["expected_family_count"], 3)

      broken = json.loads(tactical.read_text(encoding="utf-8"))
      broken["results"][0]["id"] = "forged-replacement"
      tactical.write_text(json.dumps(broken), encoding="utf-8")
      rejected = summary.summarize(
          root, expected_gates=3, min_matches=600, curated_tactical=tactical
      )
      self.assertFalse(rejected["passed"])
      self.assertIn(
          "five-player tactical fixture identities are incomplete",
          rejected["errors"],
      )

  def test_formal_bid_regime_report_requires_every_exact_matching_decision(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for index in range(1, 4):
        payload = _gate(100 + index)
        payload["curated_bid_regime_gate"] = {"required": True}
        self._write(root, index, payload)

      missing = summary.summarize(root, expected_gates=3, min_matches=600)
      self.assertFalse(missing["passed"])
      self.assertIn("missing required curated bid-regime validation", missing["errors"])

      ids, families = summary._expected_bid_regime_scenarios(3)
      path = root / "curated_bid_regime_validation.json"
      report = {
          "suiteID": "quantum-cat-curated-bid-regime-validation-v1",
          "candidate": "candidate.pt:policy",
          "scenarioCount": len(ids),
          "passedCount": len(ids),
          "independentScenarioFamilyCount": families,
          "passed": True,
          "results": [
              {"id": scenario_id, "players": 3, "passed": True}
              for scenario_id in sorted(ids)
          ],
      }
      path.write_text(json.dumps(report), encoding="utf-8")
      accepted = summary.summarize(
          root, expected_gates=3, min_matches=600, curated_bid_regime=path
      )
      self.assertTrue(accepted["passed"])
      self.assertTrue(accepted["curated_bid_regime"]["required"])
      self.assertEqual(accepted["curated_bid_regime"]["scenario_count"], 9)

      report["results"][0]["id"] = "forged-replacement"
      report["results"][1]["passed"] = False
      report["candidate"] = "different.pt:policy"
      path.write_text(json.dumps(report), encoding="utf-8")
      rejected = summary.summarize(
          root, expected_gates=3, min_matches=600, curated_bid_regime=path
      )
      self.assertFalse(rejected["passed"])
      self.assertIn(
          "curated bid-regime fixture identities are incomplete", rejected["errors"]
      )
      self.assertIn(
          "one or more curated bid-regime decisions failed", rejected["errors"]
      )
      self.assertIn(
          "curated bid-regime candidate does not match gated model", rejected["errors"]
      )


if __name__ == "__main__":
  unittest.main()
