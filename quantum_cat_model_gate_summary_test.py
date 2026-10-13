import contextlib
import io
import json
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).parent
    / ".codex/skills/test-model/scripts/summarize_model_gates.py"
)
SPEC = spec_from_file_location("model_gate_summary", SCRIPT)
summary = module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class ModelGateSummaryTest(unittest.TestCase):

  def _run(self, tactical=None, require_tactical=False,
           require_competitive=False, bidding=None, require_bidding=False,
           bid_regime=None, require_bid_regime=False):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      paradox = root / "paradox.json"
      competitive = root / "competitive.json"
      led = root / "led.json"
      report = root / "summary.json"
      paradox.write_text(json.dumps({"homogeneous_paradox_gate": {}}))
      competitive.write_text(json.dumps({
          "competitive_promotion_gate": {"passed": True},
      }))
      led.write_text(json.dumps({"stats": {}}))
      argv = [
          "summarize_model_gates.py",
          "--checkpoint", "candidate.pt",
          "--checkpoint-sha", "candidate-sha",
          "--mode", "policy",
          "--players", "4",
          "--paradox", str(paradox),
          "--competitive", str(competitive),
          "--led", str(led),
          "--out", str(report),
      ]
      if tactical is not None:
        artifact = root / "tactical.json"
        artifact.write_text(json.dumps(tactical))
        argv.extend(["--curated-tactical", str(artifact)])
      if bidding is not None:
        artifact = root / "bidding.json"
        artifact.write_text(json.dumps(bidding))
        argv.extend(["--curated-bidding", str(artifact)])
      if bid_regime is not None:
        artifact = root / "bid_regime.json"
        artifact.write_text(json.dumps(bid_regime))
        argv.extend(["--curated-bid-regime", str(artifact)])
      if require_tactical:
        argv.append("--require-tactical-pass")
      if require_competitive:
        argv.append("--require-competitive-pass")
      if require_bidding:
        argv.append("--require-bidding-pass")
      if require_bid_regime:
        argv.append("--require-bid-regime-pass")
      with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
        result = summary.main()
      return result, json.loads(report.read_text())

  @staticmethod
  def _tactical(passed=True):
    return {
        "candidate": "candidate.pt:policy",
        "scenarioCount": 8,
        "passedCount": 8 if passed else 4,
        "independentScenarioFamilyCount": 2,
        "humanReviewedFamilyCount": 2,
        "exactOfficialScoringFamilyCount": 0,
        "passed": passed,
        "families": [],
        "results": [{"players": 4, "passed": passed}],
    }

  @staticmethod
  def _bidding(passed=True):
    return {
        "suiteID": "quantum-cat-curated-bidding-validation-v1",
        "mode": "candidate_bid_validation",
        "candidate": "candidate.pt:policy",
        "scenarioCount": 12,
        "passedCount": 12 if passed else 8,
        "independentScenarioFamilyCount": 3,
        "passed": passed,
        "results": [{"players": 4, "passed": passed}],
    }

  @staticmethod
  def _bid_regime(passed=True):
    return {
        "suiteID": "quantum-cat-curated-bid-regime-validation-v1",
        "candidate": "candidate.pt:policy",
        "scenarioCount": 12,
        "passedCount": 12 if passed else 8,
        "independentScenarioFamilyCount": 3,
        "passed": passed,
        "results": [{"players": 4, "passed": passed}],
    }

  def test_required_matching_tactical_report_passes(self):
    result, report = self._run(self._tactical(), require_tactical=True)
    self.assertEqual(result, 0)
    self.assertTrue(report["curated_tactical_gate"]["required"])
    self.assertEqual(report["curated_tactical_gate"]["scenario_count"], 8)
    self.assertEqual(
        report["curated_tactical_gate"]["independent_scenario_family_count"], 2
    )

  def test_required_failed_tactical_report_blocks_promotion(self):
    result, report = self._run(self._tactical(False), require_tactical=True)
    self.assertEqual(result, 4)
    self.assertFalse(report["curated_tactical_gate"]["passed"])

  def test_explicit_exploration_retains_visible_failed_tactical_report(self):
    result, report = self._run(self._tactical(False), require_tactical=False)
    self.assertEqual(result, 0)
    self.assertFalse(report["curated_tactical_gate"]["passed"])
    self.assertEqual(report["curated_tactical_gate"]["passed_count"], 4)

  def test_mismatched_tactical_candidate_fails_even_in_exploration(self):
    tactical = self._tactical()
    tactical["candidate"] = "different.pt:policy"
    result, _ = self._run(tactical, require_tactical=False)
    self.assertEqual(result, 4)

  def test_missing_required_tactical_report_fails_closed(self):
    result, report = self._run(None, require_tactical=True)
    self.assertEqual(result, 4)
    self.assertFalse(report["curated_tactical_gate"]["available"])

  def test_competitive_promotion_cannot_omit_matching_tactical_report(self):
    result, report = self._run(
        None, require_tactical=False, require_competitive=True
    )
    self.assertEqual(result, 4)
    self.assertTrue(report["curated_tactical_gate"]["required"])

  def test_competitive_promotion_cannot_ignore_failed_tactical_report(self):
    result, report = self._run(
        self._tactical(False), require_tactical=False,
        require_competitive=True,
    )
    self.assertEqual(result, 4)
    self.assertTrue(report["curated_tactical_gate"]["required"])

  def test_required_matching_bidding_report_passes(self):
    result, report = self._run(
        bidding=self._bidding(), require_bidding=True
    )
    self.assertEqual(result, 0)
    self.assertTrue(report["curated_bidding_gate"]["required"])
    self.assertEqual(report["curated_bidding_gate"]["scenario_count"], 12)

  def test_failed_bidding_report_blocks_promotion(self):
    result, report = self._run(
        bidding=self._bidding(False), require_bidding=True
    )
    self.assertEqual(result, 5)
    self.assertFalse(report["curated_bidding_gate"]["passed"])

  def test_competitive_gate_cannot_omit_bidding_after_tactical_pass(self):
    result, report = self._run(
        tactical=self._tactical(), require_competitive=True
    )
    self.assertEqual(result, 5)
    self.assertTrue(report["curated_bidding_gate"]["required"])

  def test_mismatched_bidding_candidate_fails_even_in_exploration(self):
    bidding = self._bidding()
    bidding["candidate"] = "different.pt:policy"
    result, _ = self._run(bidding=bidding)
    self.assertEqual(result, 5)

  def test_required_matching_bid_regime_report_passes(self):
    result, report = self._run(
        bid_regime=self._bid_regime(), require_bid_regime=True
    )
    self.assertEqual(result, 0)
    self.assertTrue(report["curated_bid_regime_gate"]["required"])
    self.assertEqual(report["curated_bid_regime_gate"]["scenario_count"], 12)

  def test_failed_bid_regime_report_blocks_promotion(self):
    result, report = self._run(
        bid_regime=self._bid_regime(False), require_bid_regime=True
    )
    self.assertEqual(result, 6)
    self.assertFalse(report["curated_bid_regime_gate"]["passed"])

  def test_competitive_promotion_cannot_omit_bid_regime_after_other_gates(self):
    result, report = self._run(
        tactical=self._tactical(), bidding=self._bidding(),
        require_competitive=True,
    )
    self.assertEqual(result, 6)
    self.assertTrue(report["curated_bid_regime_gate"]["required"])

  def test_mismatched_bid_regime_candidate_fails_even_in_exploration(self):
    bid_regime = self._bid_regime()
    bid_regime["candidate"] = "different.pt:policy"
    result, _ = self._run(bid_regime=bid_regime)
    self.assertEqual(result, 6)


if __name__ == "__main__":
  unittest.main()
