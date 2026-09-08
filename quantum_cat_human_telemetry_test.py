import json
import tempfile
import unittest
from pathlib import Path

import quantum_cat_human_telemetry as telemetry


def _match(match_id="m1", human_score=9.0, bot_score=4.0):
  seats = [
      {
          "index": 0,
          "participantID": "human:seat:0",
          "actor": "human",
          "botKind": None,
          "modelResource": None,
          "modelCheckpointSHA256": None,
          "liveInputContract": None,
      },
      {
          "index": 1,
          "participantID": "bot:champion:abc",
          "actor": "bot",
          "botKind": "championBeliefPolicy",
          "modelResource": "champion_belief_policy",
          "modelCheckpointSHA256": "abc",
          "liveInputContract": "swift-partial-v0-unverified",
      },
      {
          "index": 2,
          "participantID": "bot:champion:abc",
          "actor": "bot",
          "botKind": "championBeliefPolicy",
          "modelResource": "champion_belief_policy",
          "modelCheckpointSHA256": "abc",
          "liveInputContract": "swift-partial-v0-unverified",
      },
  ]
  rounds = []
  for round_index in range(3):
    rounds.append({
        "index": round_index,
        "startingPlayer": round_index,
        "startedAt": "2026-07-21T10:00:00Z",
        "completedAt": "2026-07-21T10:01:00Z",
        "scores": [human_score, bot_score, bot_score],
        "paradoxes": [False, False, False],
        "winner": 0,
        "moves": [{
            "sequence": 0,
            "roundIndex": round_index,
            "phase": "discard",
            "player": 0,
            "actor": "human",
            "action": "discard:1",
            "legalActions": ["discard:1", "discard:2"],
            "legalActionCount": 2,
            "trickNumber": 0,
            "ledSuit": None,
            "championRecommendation": "discard:1",
            "agreedWithChampion": True,
        }],
    })
  return {
      "id": match_id,
      "schemaVersion": 1,
      "createdAt": "2026-07-21T10:00:00Z",
      "completedAt": "2026-07-21T10:03:00Z",
      "appVersion": "1",
      "buildVersion": "1",
      "seats": seats,
      "targetRounds": 3,
      "rounds": rounds,
      "currentRoundIndex": 2,
      "currentRoundStartingPlayer": 2,
      "currentRoundStartedAt": "2026-07-21T10:02:00Z",
      "currentRoundMoves": [],
  }


class HumanTelemetryTest(unittest.TestCase):

  def test_shipped_three_player_anchor_uses_verified_live_contract(self):
    anchor = telemetry.load_anchor_pool(
        Path("experiments/human_elo_anchor_pool.json")
    )
    self.assertEqual(anchor["poolID"], "quantum-cat-frozen-human-elo-v2")
    participants = set(anchor["playerCounts"]["3"]["ratings"])
    self.assertEqual(len(participants), 1)
    self.assertTrue(next(iter(participants)).endswith(
        f":{telemetry.VERIFIED_LIVE_INPUT_CONTRACT}"
    ))

  def test_duplicate_bot_identity_is_aggregated_without_double_update(self):
    match = _match()
    ratings = telemetry.compute_ratings([match], k_factor=24.0)

    self.assertEqual(set(ratings), {"human:seat:0", "bot:champion:abc"})
    self.assertAlmostEqual(ratings["human:seat:0"], 1012.0)
    self.assertAlmostEqual(ratings["bot:champion:abc"], 988.0)

  def test_report_marks_unverified_live_model_contract_diagnostic_only(self):
    report = telemetry.build_report(
        [_match()], [], k_factor=24.0, bootstrap_samples=20, seed=7
    )

    self.assertEqual(report["model_evidence_status"], "diagnostic_only")
    self.assertEqual(report["confidence"], "insufficient_data")
    self.assertTrue(report["unverified_model_contracts"])
    self.assertEqual(
        report["participants"]["human:seat:0"][
            "champion_recommendation_agreement_rate"
        ],
        1.0,
    )

  def test_frozen_anchor_does_not_drift_and_connects_human_rating(self):
    match = _match()
    anchor = {"bot:champion:abc": 1000.0}
    ratings = telemetry.compute_ratings(
        [match], k_factor=24.0, anchor_ratings=anchor
    )
    self.assertEqual(ratings["bot:champion:abc"], 1000.0)
    self.assertAlmostEqual(ratings["human:seat:0"], 1012.0)

    report = telemetry.build_report(
        [match], [], k_factor=24.0, bootstrap_samples=20, seed=7,
        anchor_pool={
            "poolID": "test-pool",
            "playerCounts": {"3": {"ratings": anchor}},
        },
    )
    table = report["player_count_reports"]["3"]
    self.assertEqual(table["anchor_evidence_status"], "anchored")
    self.assertTrue(table["participants"]["human:seat:0"]["anchor_connected"])
    self.assertTrue(
        table["participants"]["bot:champion:abc"]["rating_is_frozen_anchor"]
    )
    self.assertEqual(
        table["bot_vs_human_cohort"]["bot:champion:abc"]
        ["elo_minus_human_cohort"],
        -12.0,
    )

  def test_player_counts_are_reported_in_separate_elo_pools(self):
    three_player = _match("m3")
    four_player = _match("m4")
    four_player["seats"].append({
        "index": 3,
        "participantID": "bot:heuristic:native",
        "actor": "bot",
        "botKind": "heuristicTarget",
        "modelResource": None,
        "modelCheckpointSHA256": None,
        "liveInputContract": "native-v1",
    })
    for round_record in four_player["rounds"]:
      round_record["scores"].append(3.0)
      round_record["paradoxes"].append(False)
    extra_round = dict(four_player["rounds"][-1])
    extra_round["index"] = 3
    extra_round["moves"] = []
    four_player["rounds"].append(extra_round)
    four_player["targetRounds"] = 4

    report = telemetry.build_report(
        [three_player, four_player], [], k_factor=24.0,
        bootstrap_samples=10, seed=9,
    )
    self.assertEqual(set(report["player_count_reports"]), {"3", "4"})
    self.assertNotIn("participants", report)

  def test_jsonl_loader_deduplicates_matches_and_rejects_hidden_hands(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "telemetry.jsonl"
      match = _match()
      path.write_text(
          json.dumps(match) + "\n" + json.dumps(match) + "\n",
          encoding="utf-8",
      )
      matches, sources = telemetry.load_matches([path])
      self.assertEqual(len(matches), 1)
      self.assertEqual(len(sources), 1)

      match["rounds"][0]["hand"] = [1, 2, 3]
      with self.assertRaisesRegex(ValueError, "hidden-state"):
        telemetry.validate_match(match)

      del match["rounds"][0]["hand"]
      match["rounds"][0]["discardedRanks"] = [4, 5, 6]
      with self.assertRaisesRegex(ValueError, "hidden-state"):
        telemetry.validate_match(match)


if __name__ == "__main__":
  unittest.main()
