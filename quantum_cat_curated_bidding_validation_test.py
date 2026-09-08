import copy
import unittest

import numpy as np

import quantum_cat_curated_bidding_validation as bidding


class CuratedBiddingValidationTest(unittest.TestCase):

  def setUp(self):
    self.payload = bidding.load_suite()

  def test_root_families_expand_to_all_player_seats(self):
    rows = bidding.validate_suite(self.payload)
    self.assertEqual(len(self.payload["scenarios"]), 11)
    self.assertEqual(len(rows), 46)
    self.assertEqual(sum(item["state"]._num_players == 3 for item in rows), 9)
    self.assertEqual(sum(item["state"]._num_players == 4 for item in rows), 12)
    self.assertEqual(sum(item["state"]._num_players == 5 for item in rows), 25)

  def test_four_nines_accept_every_bid_above_one(self):
    item = bidding.validate_suite(self.payload)[0]
    self.assertEqual(item["state"]._hands[0][-1], 4)
    for bid in (2, 3, 4):
      report = bidding.evaluate_validated(
          [item], lambda _state, _actor, value=bid: 100 + value, "strong"
      )
      self.assertTrue(report["passed"], bid)
    self.assertFalse(bidding.evaluate_validated(
        [item], lambda _state, _actor: 101, "always-one"
    )["passed"])

  def test_weak_five_player_hand_accepts_only_one(self):
    item = bidding.validate_suite(self.payload)[2]
    self.assertEqual(item["state"]._hands[0][-3:].tolist(), [0, 0, 0])
    self.assertTrue(bidding.evaluate_validated(
        [item], lambda _state, _actor: 101, "minimum"
    )["passed"])
    for bid in (2, 3, 4):
      self.assertFalse(bidding.evaluate_validated(
          [item], lambda _state, _actor, value=bid: 100 + value, "overbid"
      )["passed"])

  def test_single_high_card_accepts_both_defensible_low_bids(self):
    item = bidding.validate_suite(self.payload)[4]
    for bid in (1, 2):
      self.assertTrue(bidding.evaluate_validated(
          [item], lambda _state, _actor, value=bid: 100 + value, "boundary"
      )["passed"])

  def test_fixture_cannot_silently_omit_a_reasonable_bid(self):
    payload = copy.deepcopy(self.payload)
    scenario = payload["scenarios"][0]
    scenario["acceptance"]["acceptedBids"] = [2, 3]
    scenario["acceptance"]["knownBadBids"] = [1, 4]
    with self.assertRaisesRegex(ValueError, "strategically reasonable"):
      bidding.validate_suite(payload)

  def test_fixture_rejects_opponent_or_future_information(self):
    for forbidden in ("opponentHands", "opponentDiscards", "futureTricks"):
      payload = copy.deepcopy(self.payload)
      payload["scenarios"][0]["state"][forbidden] = []
      with self.assertRaisesRegex(ValueError, "hidden or future information"):
        bidding.validate_suite(payload)

  def test_fixture_rejects_impossible_rank_copy_count(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][0]["state"]["actingHandRankCounts"][-1] = 6
    with self.assertRaisesRegex(ValueError, "physical copies"):
      bidding.validate_suite(payload)

  def test_wrong_rank_limit_and_hand_size_cannot_pass(self):
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][0]["state"]["numRanks"] = 8
    with self.assertRaises(ValueError):
      bidding.validate_suite(payload)
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][0]["state"]["actingHandRankCounts"][0] += 1
    with self.assertRaisesRegex(ValueError, "post-discard card count"):
      bidding.validate_suite(payload)

  def test_actor_observation_is_invariant_to_hidden_opponent_cards(self):
    for item in bidding.validate_suite(self.payload):
      state = item["state"]
      actor = int(state.current_player())
      before = np.asarray(state.observation_tensor(actor), dtype=np.float32)
      perturbed = state.clone()
      for opponent in range(state._num_players):
        if opponent != actor:
          perturbed._hands[opponent][:] = np.arange(state._num_card_types) % 3
          perturbed._discarded_cards[opponent] = opponent % state._num_card_types
      after = np.asarray(perturbed.observation_tensor(actor), dtype=np.float32)
      np.testing.assert_array_equal(before, after)
      self.assertEqual(state.legal_actions(actor), perturbed.legal_actions(actor))

  def test_suite_and_each_fixture_stay_excluded_from_training(self):
    payload = copy.deepcopy(self.payload)
    payload["trainingPolicy"]["allowInTraining"] = True
    with self.assertRaisesRegex(ValueError, "excluded from training"):
      bidding.validate_suite(payload)
    payload = copy.deepcopy(self.payload)
    payload["scenarios"][0]["trainingEligible"] = True
    with self.assertRaisesRegex(ValueError, "excluded from training"):
      bidding.validate_suite(payload)


if __name__ == "__main__":
  unittest.main()
