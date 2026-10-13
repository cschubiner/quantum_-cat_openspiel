import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import quantum_cat_candidate_sweep as sweep


class CandidateSweepTacticalGateTest(unittest.TestCase):

  def _args(self, players=4, dry_run=False):
    return SimpleNamespace(
        players=players,
        dry_run=dry_run,
        gate_champion=True,
        gate_top_n=1,
        gate_one_per_recipe=False,
        gate_max_per_run=0,
        gate_include_best_paradox_per_recipe=False,
        gate_champion_mode="policy",
        gate_candidate_mode="policy",
        gate_belief_samples=4,
        gate_belief_sims=8,
        gate_mcts_sims=8,
        gate_repeat_count=1,
        gate_repeat_seed_stride=100000,
        gate_matches=4,
        gate_opponent="heuristic",
        gate_workers=1,
        gate_worker_torch_threads=1,
        gate_auto_worker_min_games=32,
        champion="champion.pt",
        seed=100,
    )

  @staticmethod
  def _candidate():
    return {
        "recipe": "fixture",
        "checkpoint": "candidate.pt",
        "train_steps_completed": 20,
    }

  def _write_tactical_failure(self, command, **_kwargs):
    path = Path(command[command.index("--out") + 1])
    players = int(command[command.index("--players") + 1])
    path.write_text(json.dumps({
        "candidate": "candidate.pt:policy",
        "scenarioCount": 8 if players == 4 else 15,
        "passedCount": 4 if players == 4 else 0,
        "passed": False,
        "results": [],
    }), encoding="utf-8")
    return SimpleNamespace(returncode=1, stdout="", stderr="")

  def test_four_player_sweep_rejects_before_full_match_gate(self):
    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(
          sweep.subprocess, "run", side_effect=self._write_tactical_failure
      ) as run, contextlib.redirect_stdout(io.StringIO()):
        candidate = self._candidate()
        gates = sweep.run_champion_gates(
            self._args(4), Path(directory), [candidate]
        )
      self.assertEqual(run.call_count, 1)
      self.assertIn("quantum_cat_curated_validation.py", run.call_args.args[0])
      self.assertEqual(gates[0]["skipped"], "locked_tactical_failure")
      self.assertFalse(candidate["champion_gate"]["passed"])
      self.assertEqual(gates[0]["curated_tactical"]["scenario_count"], 8)

  def test_five_player_sweep_rejects_exact_endgame_failures(self):
    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(
          sweep.subprocess, "run", side_effect=self._write_tactical_failure
      ) as run, contextlib.redirect_stdout(io.StringIO()):
        gates = sweep.run_champion_gates(
            self._args(5), Path(directory), [self._candidate()]
        )
      self.assertEqual(run.call_count, 1)
      self.assertEqual(gates[0]["curated_tactical"]["scenario_family_count"], 3)
      self.assertEqual(gates[0]["curated_tactical"]["scenario_count"], 15)
      self.assertEqual(gates[0]["skipped"], "locked_tactical_failure")

  def test_three_player_coverage_gap_is_explicit(self):
    with tempfile.TemporaryDirectory() as directory:
      report = sweep.run_candidate_tactical_preflight(
          self._args(3), self._candidate(), Path(directory) / "unused.json"
      )
    self.assertFalse(report["available"])
    self.assertEqual(report["scenario_family_count"], 0)
    self.assertIsNone(report["passed"])

  def test_dry_run_reports_exact_candidate_selector_without_running(self):
    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(sweep.subprocess, "run") as run:
        report = sweep.run_candidate_tactical_preflight(
            self._args(4, dry_run=True), self._candidate(),
            Path(directory) / "planned.json",
        )
    run.assert_not_called()
    self.assertEqual(report["candidate"], "candidate.pt:policy")
    self.assertIn("--mcts-sims", report["command"])
    self.assertEqual(report["scenario_family_count"], 2)

  def test_mismatched_preflight_candidate_fails_closed(self):
    def forged(command, **kwargs):
      result = self._write_tactical_failure(command, **kwargs)
      path = Path(command[command.index("--out") + 1])
      payload = json.loads(path.read_text(encoding="utf-8"))
      payload["candidate"] = "different.pt:policy"
      path.write_text(json.dumps(payload), encoding="utf-8")
      return result

    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(sweep.subprocess, "run", side_effect=forged):
        with self.assertRaisesRegex(ValueError, "does not match gated selector"):
          sweep.run_candidate_tactical_preflight(
              self._args(4), self._candidate(), Path(directory) / "forged.json"
          )

  def test_three_player_sweep_rejects_constant_one_bidding_before_games(self):
    def bid_failure(command, **_kwargs):
      path = Path(command[command.index("--out") + 1])
      path.write_text(json.dumps({
          "suiteID": "quantum-cat-curated-bidding-validation-v1",
          "mode": "candidate_bid_validation",
          "candidate": "candidate.pt:policy",
          "scenarioCount": 9,
          "passedCount": 6,
          "passed": False,
          "results": [],
      }), encoding="utf-8")
      return SimpleNamespace(returncode=1, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(
          sweep.subprocess, "run", side_effect=bid_failure
      ) as run, contextlib.redirect_stdout(io.StringIO()):
        candidate = self._candidate()
        gates = sweep.run_champion_gates(
            self._args(3), Path(directory), [candidate]
        )
      self.assertEqual(run.call_count, 1)
      self.assertIn("quantum_cat_curated_bidding_validation.py", run.call_args.args[0])
      self.assertEqual(gates[0]["skipped"], "locked_bidding_failure")
      self.assertEqual(gates[0]["curated_bidding"]["passed_count"], 6)

  def test_bidding_dry_run_has_all_player_count_families(self):
    for players, families in ((3, 3), (4, 3), (5, 5)):
      with tempfile.TemporaryDirectory() as directory:
        report = sweep.run_candidate_bidding_preflight(
            self._args(players, dry_run=True), self._candidate(),
            Path(directory) / "planned_bidding.json",
        )
      self.assertEqual(report["scenario_family_count"], families)

  def test_bid_regime_dry_run_has_all_player_count_families(self):
    for players, families in ((3, 3), (4, 3), (5, 4)):
      with tempfile.TemporaryDirectory() as directory:
        report = sweep.run_candidate_bid_regime_preflight(
            self._args(players, dry_run=True), self._candidate(),
            Path(directory) / "planned_bid_regime.json",
        )
      self.assertEqual(report["scenario_family_count"], families)
      self.assertIn("--suite", report["command"])
      self.assertTrue(any(
          "curated_bid_regime_validation" in str(item)
          for item in report["command"]
      ))

  def test_three_player_sweep_rejects_bid_regime_failure_before_games(self):
    def mocked_preflight(command, **_kwargs):
      path = Path(command[command.index("--out") + 1])
      is_bidding = "quantum_cat_curated_bidding_validation.py" in command
      payload = {
          "suiteID": (
              "quantum-cat-curated-bidding-validation-v1" if is_bidding
              else "quantum-cat-curated-bid-regime-validation-v1"
          ),
          "candidate": "candidate.pt:policy",
          "scenarioCount": 9,
          "passedCount": 9 if is_bidding else 3,
          "passed": is_bidding,
          "results": [{"players": 3, "passed": is_bidding}],
      }
      if is_bidding:
        payload["mode"] = "candidate_bid_validation"
      path.write_text(json.dumps(payload), encoding="utf-8")
      return SimpleNamespace(
          returncode=0 if is_bidding else 1, stdout="", stderr=""
      )

    with tempfile.TemporaryDirectory() as directory:
      with mock.patch.object(
          sweep.subprocess, "run", side_effect=mocked_preflight
      ) as run, contextlib.redirect_stdout(io.StringIO()):
        candidate = self._candidate()
        gates = sweep.run_champion_gates(
            self._args(3), Path(directory), [candidate]
        )
      self.assertEqual(run.call_count, 2)
      self.assertEqual(gates[0]["skipped"], "locked_bid_regime_failure")
      self.assertEqual(gates[0]["curated_bid_regime"]["passed_count"], 3)
      self.assertFalse(candidate["champion_gate"]["passed"])


if __name__ == "__main__":
  unittest.main()
