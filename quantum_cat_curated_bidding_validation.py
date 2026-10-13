"""Locked, set-valued, information-safe prediction fixtures for Quantum Cat."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyspiel

from open_spiel.python.games import quantum_cat as _quantum_cat  # pylint: disable=unused-import
from quantum_cat_curated_validation import _candidate_selector


DEFAULT_SUITE = (
    Path(__file__).resolve().parent
    / "experiments/quantum_cat_curated_bidding_validation.json"
)
SUITE_ID = "quantum-cat-curated-bidding-validation-v1"
FORBIDDEN_STATE_KEYS = {
    "allHands", "deck", "discardedRanks", "hiddenHands", "opponentDiscards",
    "opponentHands", "futureTricks", "futureOutcome", "realizedTricks",
}


def load_suite(path=DEFAULT_SUITE):
  with Path(path).open("r", encoding="utf-8") as handle:
    return json.load(handle)


def build_state(scenario):
  payload = scenario["state"]
  leaked = sorted(FORBIDDEN_STATE_KEYS.intersection(payload))
  if leaked:
    raise ValueError(
        "Bidding scenarios must not include hidden or future information: "
        + ", ".join(leaked)
    )
  players = int(payload["numPlayers"])
  if players not in (3, 4, 5):
    raise ValueError(f"Unsupported player count: {players}")
  game = pyspiel.load_game("python_quantum_cat", {
      "players": players,
      "start_player": int(payload["roundStartPlayer"]),
      "match_context": 1,
  })
  if int(payload["numRanks"]) != int(game.num_card_types):
    raise ValueError(
        f"Bidding scenario has wrong maximum rank for {players} players"
    )
  if int(payload["cardsPerPlayerInitial"]) != int(game.cards_per_player_initial):
    raise ValueError("Bidding scenario has the wrong initial hand size")
  actor = int(payload["actingPlayer"])
  if not 0 <= actor < players:
    raise ValueError("Bidding actor is outside the player count")
  if int(payload["roundStartPlayer"]) != actor:
    raise ValueError("Locked bidding fixtures must act first in their round")
  state = game.new_initial_state()
  state._phase = 2
  state._round_start_player = actor
  state._start_player = actor
  state._current_player = actor
  state._cards_dealt = state._total_cards
  state._hands = [np.zeros(game.num_card_types, dtype=int) for _ in range(players)]
  state._hands[actor] = np.asarray(payload["actingHandRankCounts"], dtype=int)
  state._has_discarded = [True] * players
  state._discarded_cards = [-1] * players
  state._discarded_cards[actor] = int(payload["actingDiscardedRank"])
  state._predictions = [int(value) for value in payload["publicPredictions"]]
  state._match_totals = np.asarray(payload["matchTotals"], dtype=np.float32)
  state._match_round = int(payload["matchRound"])
  state._game_over = False
  return state


def reasonable_bids(scenario):
  acceptance = scenario["acceptance"]
  rule = acceptance.get("reasonableBidRule")
  if not isinstance(rule, dict):
    raise ValueError("Bidding fixtures require a complete reasonable-bid rule")
  hand = [int(value) for value in scenario["state"]["actingHandRankCounts"]]
  kind = rule.get("kind")
  if kind == "multiple_max_rank_requires_above_minimum":
    threshold = int(rule.get("minimumMaxRankCopies", 3))
    if threshold < 2 or hand[-1] < threshold:
      raise ValueError("Highest-rank copies do not justify the strong-hand rule")
    return [2, 3, 4]
  if kind == "multiple_second_rank_requires_above_minimum":
    threshold = int(rule.get("minimumSecondRankCopies", 4))
    if hand[-1] != 0 or threshold < 3 or hand[-2] < threshold:
      raise ValueError("Second-highest copies do not justify the strong-hand rule")
    return [2, 3, 4]
  if kind == "no_top_three_ranks_requires_minimum":
    if any(hand[-3:]):
      raise ValueError("A weak-hand fixture unexpectedly contains a top-three rank")
    return [1]
  if kind == "single_max_rank_allows_one_or_two":
    if hand[-1] != 1 or hand[-2] != 0:
      raise ValueError("An isolated highest-rank fixture has another high card")
    return [1, 2]
  raise ValueError(f"Unsupported reasonable-bid rule: {kind!r}")


def validate_scenario(scenario):
  if scenario.get("reviewStatus") != "locked":
    raise ValueError("Bidding scenario must be locked")
  if scenario.get("trainingEligible") is not False:
    raise ValueError("Locked bidding scenarios must be excluded from training")
  payload = scenario["state"]
  players = int(payload["numPlayers"])
  ranks = int(payload["numRanks"])
  hand = [int(value) for value in payload["actingHandRankCounts"]]
  if len(hand) != ranks or any(value < 0 or value > 5 for value in hand):
    raise ValueError("Bidding hand must respect the five physical copies per rank")
  if sum(hand) != int(payload["cardsPerPlayerInitial"]) - 1:
    raise ValueError("Bidding hand has the wrong post-discard card count")
  discard = int(payload["actingDiscardedRank"])
  if not 0 <= discard < ranks or hand[discard] >= 5:
    raise ValueError("Acting discard violates the five physical copies per rank")
  predictions = [int(value) for value in payload["publicPredictions"]]
  if len(predictions) != players or any(value != -1 for value in predictions):
    raise ValueError("Opening bidding fixtures cannot include prior predictions")
  if len(payload["matchTotals"]) != players:
    raise ValueError("Public match totals do not match the player count")
  acceptance = scenario["acceptance"]
  if acceptance.get("kind") != "one_of_bid_values":
    raise ValueError("Bidding fixtures require set-valued bid acceptance")
  accepted = [int(value) for value in acceptance.get("acceptedBids", [])]
  if accepted != sorted(set(accepted)) or any(value not in (1, 2, 3, 4) for value in accepted):
    raise ValueError("Accepted bids must be unique, ordered legal predictions")
  if accepted != reasonable_bids(scenario):
    raise ValueError("Bidding fixture omits or adds a strategically reasonable bid")
  known_bad = [int(value) for value in acceptance.get("knownBadBids", [])]
  if set(known_bad) != set((1, 2, 3, 4)).difference(accepted):
    raise ValueError("Known-bad bids must exactly complement the accepted bids")
  state = build_state(scenario)
  actor = int(payload["actingPlayer"])
  legal = [int(action) for action in state.legal_actions(actor)]
  if legal != [101, 102, 103, 104]:
    raise ValueError("Bidding legal-action set differs from official predictions")
  observation = np.asarray(state.observation_tensor(actor), dtype=np.float32)
  return {
      "scenario": scenario,
      "state": state,
      "legalActions": legal,
      "observationSize": int(observation.size),
      "observationSHA256": hashlib.sha256(observation.tobytes()).hexdigest(),
  }


def rotate_scenario_seats(scenario, rotation):
  result = copy.deepcopy(scenario)
  payload = result["state"]
  players = int(payload["numPlayers"])
  rotation = int(rotation) % players
  if not rotation:
    return result
  original_id = str(result["id"])
  result["id"] = f"{original_id}--seat-r{rotation}"
  result["title"] = f"{result['title']} (seat rotation {rotation})"
  result["metamorphic"] = {
      "familyID": original_id, "kind": "cyclic_seat_relabeling",
      "rotation": rotation,
  }
  payload["actingPlayer"] = (int(payload["actingPlayer"]) + rotation) % players
  payload["roundStartPlayer"] = (
      int(payload["roundStartPlayer"]) + rotation
  ) % players
  for field in ("publicPredictions", "matchTotals"):
    original = list(payload[field])
    payload[field] = [original[(index - rotation) % players] for index in range(players)]
  return result


def validate_suite(payload):
  if int(payload.get("schemaVersion", -1)) != 1:
    raise ValueError("Unsupported curated bidding validation schema")
  if payload.get("suiteID") != SUITE_ID:
    raise ValueError("Unexpected curated bidding suite identifier")
  if payload.get("trainingPolicy", {}).get("allowInTraining") is not False:
    raise ValueError("Curated bidding suite must be excluded from training")
  scenarios = payload.get("scenarios", [])
  if not scenarios:
    raise ValueError("Curated bidding suite is empty")
  ids = [scenario.get("id") for scenario in scenarios]
  if any(not value for value in ids) or len(ids) != len(set(ids)):
    raise ValueError("Curated bidding scenario IDs must be unique")
  if payload.get("metamorphicValidation", {}).get("seatRotations") != "all":
    raise ValueError("Locked bidding scenarios must rotate through every seat")
  validated = [validate_scenario(scenario) for scenario in scenarios]
  for scenario in scenarios:
    for rotation in range(1, int(scenario["state"]["numPlayers"])):
      validated.append(validate_scenario(rotate_scenario_seats(scenario, rotation)))
  return validated


def evaluate_validated(validated, selector, candidate_label):
  results = []
  for item in validated:
    scenario = item["scenario"]
    actor = int(scenario["state"]["actingPlayer"])
    action = int(selector(item["state"].clone(), actor))
    bid = action - 100
    accepted = list(scenario["acceptance"]["acceptedBids"])
    results.append({
        "id": scenario["id"],
        "familyID": scenario.get("metamorphic", {}).get("familyID", scenario["id"]),
        "seatRotation": int(scenario.get("metamorphic", {}).get("rotation", 0)),
        "players": int(scenario["state"]["numPlayers"]),
        "handRankCounts": list(scenario["state"]["actingHandRankCounts"]),
        "highestRank": int(scenario["state"]["numRanks"]),
        "acceptedBids": accepted,
        "knownBadBids": list(scenario["acceptance"]["knownBadBids"]),
        "reasonableBidRule": copy.deepcopy(scenario["acceptance"]["reasonableBidRule"]),
        "actualAction": action,
        "actualBid": bid,
        "passed": bid in accepted,
        "observationSize": item["observationSize"],
        "observationSHA256": item["observationSHA256"],
    })
  family_ids = {item["familyID"] for item in results}
  return {
      "suiteID": SUITE_ID,
      "mode": "candidate_bid_validation",
      "candidate": candidate_label,
      "scenarioCount": len(results),
      "passedCount": sum(bool(item["passed"]) for item in results),
      "independentScenarioFamilyCount": len(family_ids),
      "passed": all(bool(item["passed"]) for item in results),
      "results": results,
  }


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
  parser.add_argument("--candidate")
  parser.add_argument("--players", type=int, choices=(3, 4, 5))
  parser.add_argument("--scenario", action="append", default=[])
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--out", type=Path)
  parser.add_argument(
      "--quiet", action="store_true",
      help="Print compact totals while retaining every fixture in the JSON artifact.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  validated = validate_suite(load_suite(args.suite))
  selected = [item for item in validated if (
      args.players is None
      or int(item["scenario"]["state"]["numPlayers"]) == args.players
  ) and (not args.scenario or item["scenario"]["id"] in args.scenario)]
  if not selected:
    raise SystemExit("No curated bidding scenarios matched the requested filters")
  if args.candidate:
    player_counts = {
        int(item["scenario"]["state"]["numPlayers"]) for item in selected
    }
    if len(player_counts) != 1:
      raise SystemExit("A bidding candidate must select exactly one player count")
    players = next(iter(player_counts))
    report = evaluate_validated(
        selected, _candidate_selector(args.candidate, players, args.device),
        args.candidate,
    )
  else:
    coverage = {}
    for item in selected:
      players = int(item["scenario"]["state"]["numPlayers"])
      row = coverage.setdefault(players, {"players": players, "scenarioCount": 0})
      row["scenarioCount"] += 1
    family_ids = {
        item["scenario"].get("metamorphic", {}).get(
            "familyID", item["scenario"]["id"]
        )
        for item in selected
    }
    report = {
        "suiteID": SUITE_ID,
        "mode": "schema_and_state_validation_only",
        "scenarioCount": len(selected),
        "independentScenarioFamilyCount": len(family_ids),
        "coverage": list(coverage.values()),
        "passed": True,
    }
  printable = (
      {key: value for key, value in report.items() if key != "results"}
      if args.quiet else report
  )
  print(json.dumps(printable, indent=2))
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  if not report["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
