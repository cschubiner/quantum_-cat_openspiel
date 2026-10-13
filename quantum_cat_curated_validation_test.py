import copy
import unittest

import numpy as np

import quantum_cat_curated_validation as curated


class CuratedValidationTest(unittest.TestCase):

  def setUp(self):
    self.payload = curated.load_suite()

  def test_locked_fixture_reconstructs_exact_information_safe_position(self):
    validated = curated.validate_suite(self.payload)
    self.assertEqual(len(validated), 23)
    self.assertEqual(len(self.payload["scenarios"]), 5)
    item = validated[0]
    scenario = item["scenario"]
    state = item["state"]

    self.assertEqual(scenario["id"], "iphone-4p-r2-t1-p0-y8-over-y6-20260722")
    self.assertIs(scenario["trainingEligible"], False)
    self.assertEqual(state._predictions, [3, 2, 1, 1])
    self.assertEqual(state._discarded_cards[0], 2)
    self.assertEqual(state._hands[0].tolist(), [0, 1, 1, 0, 1, 2, 2, 2])
    self.assertTrue(all(hand.sum() == 0 for hand in state._hands[1:]))
    self.assertEqual(state._cards_played_this_trick[2], (7, "Y"))
    self.assertEqual(state._cards_played_this_trick[3], (4, "Y"))
    self.assertEqual(item["legalActions"], [
        1, 2, 4, 5, 6, 7,
        9, 10, 12, 13, 14, 15,
        17, 18, 20, 21, 23,
        25, 26, 28, 29, 30, 31,
    ])
    self.assertEqual(item["observationSize"], 97)

  def test_expected_y8_and_known_bad_y6_are_stable_action_labels(self):
    scenario = self.payload["scenarios"][0]
    acceptance = scenario["acceptance"]
    accepted = curated.accepted_action_entries(acceptance)
    self.assertEqual(acceptance["kind"], "one_of_top1")
    self.assertEqual(
        curated.move_to_action(accepted[0]["move"], 8), 23
    )
    self.assertEqual(
        curated.move_to_action(acceptance["knownBadActions"][0]["move"], 8),
        21,
    )
    self.assertEqual(curated.action_to_move(23, 8), {
        "phase": "play", "suit": "Y", "rank": 8,
    })

  def test_opening_follow_accepts_y8_or_unbeatable_r8(self):
    validated = curated.validate_suite(self.payload)[:1]
    yellow = curated.evaluate_validated(
        validated, lambda _state, _actor: 23, "test-y8"
    )
    red = curated.evaluate_validated(
        validated, lambda _state, _actor: 7, "test-r8"
    )
    failing = curated.evaluate_validated(
        validated, lambda _state, _actor: 21, "test-y6"
    )
    self.assertTrue(yellow["passed"])
    self.assertTrue(red["passed"])
    self.assertFalse(failing["passed"])
    self.assertEqual(failing["results"][0]["actualMove"]["rank"], 6)

  def test_second_fixture_accepts_only_b8_or_g8(self):
    validated = curated.validate_suite(self.payload)
    item = validated[1]
    scenario = item["scenario"]
    state = item["state"]

    self.assertEqual(
        scenario["id"],
        "iphone-4p-r2-t2-p0-b8-or-g8-over-y6-20260722",
    )
    self.assertEqual(state._trick_number, 1)
    self.assertEqual(state._start_player, 0)
    self.assertIsNone(state._led_color)
    self.assertEqual(state._tricks_won.tolist(), [1, 0, 0, 0])
    self.assertEqual(state._hands[0].tolist(), [0, 1, 1, 0, 1, 2, 2, 1])
    self.assertEqual(len(item["legalActions"]), 15)
    self.assertIn(15, item["legalActions"])
    self.assertIn(31, item["legalActions"])
    self.assertIn(21, item["legalActions"])

    blue = curated.evaluate_validated([item], lambda _state, _actor: 15, "B8")
    green = curated.evaluate_validated([item], lambda _state, _actor: 31, "G8")
    yellow = curated.evaluate_validated([item], lambda _state, _actor: 21, "Y6")
    self.assertTrue(blue["passed"])
    self.assertTrue(green["passed"])
    self.assertFalse(yellow["passed"])
    self.assertEqual(blue["results"][0]["acceptedActions"], [15, 31])
    self.assertEqual(
        blue["results"][0]["knownUnavailableMoves"][0]["move"],
        {"phase": "play", "suit": "Y", "rank": 8},
    )

  def test_y8_is_the_only_winning_follow_and_b8_g8_burn_yellow(self):
    item = curated.validate_suite(self.payload)[0]
    reviews = {row["action"]: row for row in item["legalActionReview"]}

    self.assertTrue(reviews[23]["followsLedSuit"])
    self.assertTrue(reviews[23]["currentlyWinningAgainstPlayedCards"])
    self.assertFalse(reviews[23]["losesLedColorToken"])
    for action in (15, 31):
      self.assertFalse(reviews[action]["followsLedSuit"])
      self.assertFalse(reviews[action]["currentlyWinningAgainstPlayedCards"])
      self.assertTrue(reviews[action]["losesLedColorToken"])
      self.assertTrue(reviews[action]["knownBad"])
    self.assertTrue(reviews[7]["currentlyWinningAgainstPlayedCards"])
    self.assertTrue(reviews[7]["losesLedColorToken"])
    self.assertTrue(reviews[7]["unbeatableByRemainingLegalCards"])
    self.assertTrue(reviews[7]["ownRemainingHandFeasible"])
    self.assertTrue(reviews[7]["accepted"])
    self.assertFalse(reviews[23]["unbeatableByRemainingLegalCards"])

  def test_first_fixture_cannot_omit_defensible_unbeatable_trump(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][0]["acceptance"]["acceptedActions"].pop()
    with self.assertRaisesRegex(ValueError, "omits or adds strategically"):
      curated.validate_suite(payload)

  def test_r8_wins_a_possible_hidden_world_that_can_overtrump_y8(self):
    item = curated.validate_suite(self.payload)[0]
    actor = int(item["scenario"]["state"]["actingPlayer"])
    sampled = item["state"].resample_from_infostate(
        actor, np.random.RandomState(2026081937)
    )
    yellow = sampled.clone()
    yellow.apply_action(23)
    opponent = int(yellow.current_player())
    red_replies = [
        action for action in yellow.legal_actions(opponent)
        if int(action) < int(yellow._num_card_types)
    ]
    self.assertTrue(red_replies)
    yellow.apply_action(red_replies[0])
    self.assertEqual(int(yellow._tricks_won[actor]), 0)

    trump = sampled.clone()
    trump.apply_action(7)
    for action in trump.legal_actions(int(trump.current_player())):
      if int(action) == 999:
        continue
      completed = trump.clone()
      completed.apply_action(int(action))
      self.assertEqual(int(completed._tricks_won[actor]), 1)

  def test_second_fixture_cannot_omit_equally_acceptable_green_eight(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][1]["acceptance"]["acceptedActions"].pop()
    with self.assertRaisesRegex(ValueError, "omits or adds strategically"):
      curated.validate_suite(payload)

  def test_second_fixture_cannot_arbitrarily_accept_a_lower_rank(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][1]["acceptance"]["acceptedActions"].append({
        "action": 14,
        "move": {"phase": "play", "suit": "B", "rank": 7},
    })
    with self.assertRaisesRegex(ValueError, "omits or adds strategically"):
      curated.validate_suite(payload)

  def test_second_fixture_cannot_claim_a_legal_color_is_unavailable(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][1]["acceptance"]["knownUnavailableMoves"] = [{
        "action": 15,
        "move": {"phase": "play", "suit": "B", "rank": 8},
    }]
    with self.assertRaisesRegex(ValueError, "legal move is unavailable"):
      curated.validate_suite(payload)

  def test_every_locked_position_declares_its_complete_acceptance_rule(self):
    payload = copy.deepcopy(self.payload)
    del payload["scenarios"][0]["acceptance"]["reasonableActionRule"]
    with self.assertRaisesRegex(ValueError, "complete reasonable-action rule"):
      curated.validate_suite(payload)

  def test_candidate_report_records_every_legal_move_and_its_reasoning(self):
    item = curated.validate_suite(self.payload)[0]
    report = curated.evaluate_validated(
        [item], lambda _state, _actor: 15, "off-suit-blue-eight"
    )
    result = report["results"][0]
    self.assertFalse(report["passed"])
    self.assertEqual(result["decisionContext"]["ledSuit"], "Y")
    self.assertEqual(result["acceptanceRule"]["kind"],
                     "winning_follow_or_unbeatable_trump_for_needed_bid")
    self.assertEqual(result["acceptanceConfidence"],
                     "human_reviewed_public_tactical_judgment")
    self.assertEqual(len(result["legalActionReview"]), 23)
    rejected_blue = next(
        row for row in result["legalActionReview"] if row["action"] == 15
    )
    self.assertTrue(rejected_blue["losesLedColorToken"])
    self.assertFalse(rejected_blue["currentlyWinningAgainstPlayedCards"])

  def test_hidden_opponent_state_is_rejected(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][0]["state"]["opponentHands"] = [[1], [2], [3]]
    with self.assertRaisesRegex(ValueError, "opponent hidden state"):
      curated.validate_suite(payload)

  def test_training_eligibility_fails_closed(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][0]["trainingEligible"] = True
    with self.assertRaisesRegex(ValueError, "training-eligible"):
      curated.validate_suite(payload)

  def test_every_locked_position_covers_every_acting_seat(self):
    validated = curated.validate_suite(self.payload)
    families = {}
    for item in validated:
      scenario = item["scenario"]
      family = scenario.get("metamorphic", {}).get("familyID", scenario["id"])
      families.setdefault(family, set()).add(
          int(scenario["state"]["actingPlayer"])
      )
    self.assertEqual(len(families), 5)
    for scenario in self.payload["scenarios"]:
      self.assertEqual(
          families[scenario["id"]],
          set(range(int(scenario["state"]["numPlayers"]))),
      )

  def test_seat_rotation_preserves_exact_public_state_and_action(self):
    original = self.payload["scenarios"][2]
    rotated = curated.rotate_scenario_seats(original, 2)
    validated = curated.validate_scenario(rotated)
    self.assertEqual(rotated["state"]["actingPlayer"], 1)
    self.assertEqual(rotated["state"]["currentPlayer"], 1)
    self.assertEqual(rotated["state"]["matchTotals"], [8.0, 8.0, 2.0, 11.0, 11.0])
    self.assertEqual(validated["legalActions"], [25, 34])
    self.assertEqual(rotated["acceptance"]["expectedAction"], 25)
    self.assertTrue(all(hand.sum() == 0 for player, hand in enumerate(
        validated["state"]._hands
    ) if player != 1))

  def test_exact_five_player_bid_preservation_is_scoring_certified(self):
    scenario = self.payload["scenarios"][2]
    state = curated.validate_scenario(scenario)["state"]
    rows = curated._official_terminal_action_rows(state, 4)
    self.assertEqual([row["action"] for row in rows], [25, 34])
    self.assertEqual([row["rankPoints"] for row in rows], [2.5, 2.0])
    self.assertEqual([row["ownConnectedBonus"] for row in rows], [2.0, 0.0])

  def test_exact_five_player_needed_trump_is_scoring_certified(self):
    scenario = self.payload["scenarios"][3]
    state = curated.validate_scenario(scenario)["state"]
    rows = curated._official_terminal_action_rows(state, 1)
    self.assertEqual([row["action"] for row in rows], [8, 35])
    self.assertEqual([row["rankPoints"] for row in rows], [3.5, 2.0])
    self.assertEqual([row["ownTotalScore"] for row in rows], [13.0, 10.0])

  def test_exact_five_player_connected_bonus_is_scoring_certified(self):
    scenario = self.payload["scenarios"][4]
    state = curated.validate_scenario(scenario)["state"]
    rows = curated._official_terminal_action_rows(state, 2)
    self.assertEqual([row["action"] for row in rows], [34, 25])
    self.assertEqual([row["ownTricksWon"] for row in rows], [1, 1])
    self.assertEqual([row["ownPrediction"] for row in rows], [1, 1])
    self.assertEqual([row["ownConnectedBonus"] for row in rows], [2.0, 1.0])
    self.assertEqual([row["rankPoints"] for row in rows], [2.5, 2.0])

  def test_exact_terminal_proof_ignores_all_opponent_hidden_hands(self):
    scenario = self.payload["scenarios"][4]
    state = curated.validate_scenario(scenario)["state"]
    expected = curated._official_terminal_action_rows(state, 2)
    for player in range(5):
      if player != 2:
        state._hands[player][:] = player + 1
    self.assertEqual(curated._official_terminal_action_rows(state, 2), expected)

  def test_exact_proof_rejects_a_changed_expected_move(self):
    payload = copy.deepcopy(self.payload)
    scenario = payload["scenarios"][2]
    scenario["acceptance"]["expectedAction"] = 34
    scenario["acceptance"]["expectedMove"] = {
        "phase": "play", "suit": "G", "rank": 8,
    }
    scenario["acceptance"]["knownBadActions"] = []
    with self.assertRaisesRegex(ValueError, "exact official final placement"):
      curated.validate_suite(payload)

  def test_exact_proof_rejects_changed_official_scoring_evidence(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][3]["proof"]["allLegalActions"][0]["rankPoints"] = 4.0
    with self.assertRaisesRegex(ValueError, "exact proof changed rankPoints"):
      curated.validate_suite(payload)

  def test_exact_source_cannot_omit_its_scoring_proof(self):
    payload = copy.deepcopy(self.payload)
    del payload["scenarios"][2]["proof"]
    with self.assertRaisesRegex(ValueError, "claims exact scoring without proof"):
      curated.validate_suite(payload)

  def test_exact_proof_cannot_change_its_competitive_objective(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][2]["proof"]["objective"] = [
        "minimize_any_player_paradox"
    ]
    with self.assertRaisesRegex(ValueError, "changes the exact official objective"):
      curated.validate_suite(payload)

  def _equivalent_final_placement_scenario(self):
    scenario = copy.deepcopy(self.payload["scenarios"][2])
    actor = int(scenario["state"]["actingPlayer"])
    scenario["state"]["matchTotals"][actor] = 100.0
    state = curated.build_state(scenario)
    rows = curated._official_terminal_action_rows(state, actor)
    self.assertEqual({row["rankPoints"] for row in rows}, {4.0})
    self.assertNotEqual(rows[0]["ownTotalScore"], rows[1]["ownTotalScore"])
    scenario["proof"]["allLegalActions"] = [
        dict(row, move=curated.action_to_move(row["action"], 9))
        for row in rows
    ]
    scenario["acceptance"] = {
        "kind": "one_of_top1",
        "acceptedActions": [
            {
                "action": row["action"],
                "move": curated.action_to_move(row["action"], 9),
            }
            for row in rows
        ],
        "reasonableActionRule": {
            "kind": "best_official_match_placement",
        },
        "knownBadActions": [],
    }
    return scenario

  def test_all_equal_official_placements_pass_despite_different_scores(self):
    scenario = self._equivalent_final_placement_scenario()
    item = curated.validate_scenario(scenario)
    yellow = curated.evaluate_validated([item], lambda _state, _actor: 25, "Y8")
    green = curated.evaluate_validated([item], lambda _state, _actor: 34, "G8")

    self.assertTrue(yellow["passed"])
    self.assertTrue(green["passed"])
    self.assertEqual(set(yellow["results"][0]["acceptedActions"]), {25, 34})
    self.assertEqual(yellow["results"][0]["acceptanceConfidence"],
                     "exhaustive_final_match_official_scoring")

  def test_exact_proof_rejects_omitting_a_tied_official_placement(self):
    scenario = self._equivalent_final_placement_scenario()
    scenario["acceptance"]["acceptedActions"].pop()
    with self.assertRaisesRegex(ValueError, "exact official final placement"):
      curated.validate_scenario(scenario)

  def test_family_reporting_does_not_count_seat_variants_as_human_labels(self):
    validated = [
        item for item in curated.validate_suite(self.payload)
        if int(item["scenario"]["state"]["numPlayers"]) == 5
    ]
    report = curated.evaluate_validated(
        validated,
        lambda state, actor: curated._official_terminal_action_rows(
            state, actor
        )[0]["action"],
        "exact-terminal-oracle",
    )
    self.assertTrue(report["passed"])
    self.assertEqual(report["scenarioCount"], 15)
    self.assertEqual(report["independentScenarioFamilyCount"], 3)
    self.assertEqual(report["humanReviewedFamilyCount"], 0)
    self.assertEqual(report["exactOfficialScoringFamilyCount"], 3)

  def test_bid_regime_suite_covers_all_three_states_and_player_counts(self):
    payload = curated.load_suite(curated.DEFAULT_BID_REGIME_SUITE)
    validated = curated.validate_suite(payload)

    self.assertEqual(payload["suiteID"],
                     "quantum-cat-curated-bid-regime-validation-v1")
    self.assertFalse(payload["trainingPolicy"]["allowInTraining"])
    self.assertEqual(len(payload["scenarios"]), 10)
    self.assertEqual(len(validated), 41)
    self.assertEqual({
        players: sum(
            int(item["state"]._num_players) == players for item in validated
        )
        for players in (3, 4, 5)
    }, {3: 9, 4: 12, 5: 20})
    for players in (3, 4, 5):
      root = [
          scenario for scenario in payload["scenarios"]
          if int(scenario["state"]["numPlayers"]) == players
      ]
      self.assertTrue(any("below-bid-win" in item["id"] for item in root))
      self.assertTrue(any("at-bid-lose" in item["id"] for item in root))
      self.assertTrue(any("above-bid-win" in item["id"] for item in root))

  def test_bid_regime_accepts_all_equally_safe_losing_cards(self):
    payload = curated.load_suite(curated.DEFAULT_BID_REGIME_SUITE)
    scenario = next(
        item for item in payload["scenarios"]
        if "4p-at-bid-lose" in item["id"]
    )
    item = curated.validate_scenario(scenario)
    self.assertEqual({
        int(entry["action"])
        for entry in curated.accepted_action_entries(scenario["acceptance"])
    }, {1, 2})
    for action in (1, 2):
      result = curated.evaluate_validated(
          [item], lambda _state, _actor, value=action: value, str(action)
      )
      self.assertTrue(result["passed"])

  def test_bid_regime_rejects_omitting_an_equally_safe_move(self):
    payload = curated.load_suite(curated.DEFAULT_BID_REGIME_SUITE)
    scenario = copy.deepcopy(next(
        item for item in payload["scenarios"]
        if "4p-at-bid-lose" in item["id"]
    ))
    scenario["acceptance"]["acceptedActions"].pop()
    with self.assertRaisesRegex(ValueError, "strategically acceptable"):
      curated.validate_scenario(scenario)

  def test_bid_regime_labels_are_invariant_to_opponent_private_hands(self):
    payload = curated.load_suite(curated.DEFAULT_BID_REGIME_SUITE)
    scenario = next(
        item for item in payload["scenarios"]
        if "4p-above-bid-win" in item["id"]
    )
    item = curated.validate_scenario(scenario)
    state = item["state"]
    actor = int(scenario["state"]["actingPlayer"])
    expected = [
        curated._public_action_review(state, actor, action, set(), set())
        for action in state.legal_actions(actor)
    ]
    for seat in range(int(state._num_players)):
      if seat != actor:
        state._hands[seat] = np.arange(state._num_card_types, dtype=int) + seat
        state._discarded_cards[seat] = seat + 1
    actual = [
        curated._public_action_review(state, actor, action, set(), set())
        for action in state.legal_actions(actor)
    ]
    self.assertEqual(actual, expected)

  def test_unreachable_bid_final_trick_has_exact_official_scoring_proof(self):
    payload = curated.load_suite(curated.DEFAULT_BID_REGIME_SUITE)
    scenario = next(
        item for item in payload["scenarios"]
        if "unreachable-underbid-exact-final" in item["id"]
    )
    state = curated.validate_scenario(scenario)["state"]
    actor = int(scenario["state"]["actingPlayer"])
    rows = curated._official_terminal_action_rows(state, actor)
    self.assertEqual(int(state._predictions[actor]), 3)
    self.assertEqual(int(state._tricks_won[actor]), 1)
    self.assertEqual(int(state._num_tricks) - int(state._trick_number), 1)
    self.assertEqual([row["ownRoundScore"] for row in rows], [2.0, 1.0])
    self.assertEqual([row["rankPoints"] for row in rows], [2.0, 1.0])


if __name__ == "__main__":
  unittest.main()
