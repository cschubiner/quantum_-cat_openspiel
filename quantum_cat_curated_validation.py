"""Information-safe, human-reviewed tactical validation for Quantum Cat.

This suite is deliberately separate from self-play and replay training data.
It reconstructs only the acting player's private information plus public table
state, then accepts every explicitly justified, strategically equivalent top-1
action from a candidate policy stack.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyspiel

from open_spiel.python.games import quantum_cat as _quantum_cat  # pylint: disable=unused-import
import quantum_cat_liveness_oracle as liveness_oracle


DEFAULT_SUITE = Path("experiments/quantum_cat_curated_tactical_validation.json")
DEFAULT_BID_REGIME_SUITE = Path(
    "experiments/quantum_cat_curated_bid_regime_validation.json"
)
COLORS = ("R", "B", "Y", "G")
PHASES = {"deal": 0, "discard": 1, "prediction": 2, "play": 3}
FORBIDDEN_STATE_KEYS = {
    "allHands",
    "deck",
    "discardedRanks",
    "hiddenHands",
    "opponentDiscards",
    "opponentHands",
}
EXACT_TERMINAL_PROOF = "exact_terminal_official_placement"
EXACT_TERMINAL_OBJECTIVE = [
    "maximize_official_match_rank_points",
    "accept_all_actions_tied_on_official_placement",
]


def load_suite(path=DEFAULT_SUITE):
  with Path(path).open("r", encoding="utf-8") as handle:
    return json.load(handle)


def move_to_action(move, num_ranks):
  if move.get("phase") != "play":
    raise ValueError(f"Only play moves are supported, got {move!r}")
  suit = str(move["suit"]).upper()
  rank = int(move["rank"])
  if suit not in COLORS or not 1 <= rank <= int(num_ranks):
    raise ValueError(f"Invalid play move {move!r} for {num_ranks} ranks")
  return COLORS.index(suit) * int(num_ranks) + rank - 1


def action_to_move(action, num_ranks):
  action = int(action)
  num_ranks = int(num_ranks)
  if action < 0 or action >= len(COLORS) * num_ranks:
    return {"phase": "special", "action": action}
  return {
      "phase": "play",
      "suit": COLORS[action // num_ranks],
      "rank": action % num_ranks + 1,
  }


def _card_tuple(card, expected_player=None):
  if card is None:
    return None
  player = int(card["player"])
  if expected_player is not None and player != expected_player:
    raise ValueError(
        f"Current-trick slot {expected_player} contains player {player}"
    )
  suit = str(card["suit"]).upper()
  rank = int(card["rank"])
  if suit not in COLORS:
    raise ValueError(f"Invalid card suit {suit!r}")
  return rank, suit


def _validate_information_boundary(state_payload):
  leaked = sorted(FORBIDDEN_STATE_KEYS.intersection(state_payload))
  if leaked:
    raise ValueError(
        "Curated scenarios must not contain opponent hidden state: "
        + ", ".join(leaked)
    )


def build_state(scenario):
  """Reconstruct a simulator state without inventing opponent private cards."""
  payload = scenario["state"]
  _validate_information_boundary(payload)
  players = int(payload["numPlayers"])
  ranks = int(payload["numRanks"])
  if players not in (3, 4, 5):
    raise ValueError(f"Unsupported player count in {scenario['id']}: {players}")

  game = pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": players,
          "start_player": int(payload["roundStartPlayer"]),
          "match_context": 1,
      },
  )
  if game.num_card_types != ranks:
    raise ValueError(
        f"Scenario {scenario['id']} has {ranks} ranks; game has "
        f"{game.num_card_types}"
    )
  state = game.new_initial_state()
  actor = int(payload["actingPlayer"])

  state._phase = PHASES[str(payload["phase"])]
  state._round_start_player = int(payload["roundStartPlayer"])
  state._start_player = int(payload["startPlayer"])
  state._current_player = int(payload["currentPlayer"])
  state._trick_number = int(payload["trickNumber"])
  state._led_color = payload.get("ledSuit")
  state._cards_played_this_trick = [
      _card_tuple(card, player)
      for player, card in enumerate(payload["currentTrick"])
  ]
  state._completed_tricks = [
      [
          (player, _card_tuple(card, player))
          for player, card in enumerate(trick)
      ]
      for trick in payload.get("completedTricks", [])
  ]

  # Opponent hands and discards remain deliberately unknown. Neither the live
  # observation nor information-legal action features may depend on them.
  state._hands = [np.zeros(ranks, dtype=int) for _ in range(players)]
  state._hands[actor] = np.asarray(
      payload["actingHandRankCounts"], dtype=int
  )
  state._has_discarded = [True] * players
  state._discarded_cards = [-1] * players
  state._discarded_cards[actor] = int(payload["actingDiscardedRank"])
  state._predictions = [int(value) for value in payload["predictions"]]
  state._tricks_won = np.asarray(payload["tricksWon"], dtype=int)
  state._board_ownership = np.asarray(
      payload["boardOwnership"], dtype=int
  )
  state._color_tokens = np.asarray(payload["colorTokens"], dtype=bool)
  state._has_paradoxed = [
      bool(value) for value in payload["hasParadoxed"]
  ]
  state._trump_broken = bool(payload["trumpBroken"])
  state._match_totals = np.asarray(payload["matchTotals"], dtype=np.float32)
  state._match_round = int(payload["matchRound"])
  state._cards_dealt = state._total_cards
  state._game_over = False
  return state


def accepted_action_entries(acceptance):
  kind = acceptance.get("kind")
  if kind == "exact_top1":
    return [{
        "action": int(acceptance["expectedAction"]),
        "move": acceptance["expectedMove"],
    }]
  if kind == "one_of_top1":
    entries = list(acceptance.get("acceptedActions", []))
    if not entries:
      raise ValueError("one_of_top1 acceptance must contain acceptedActions")
    return entries
  raise ValueError(f"Unsupported tactical acceptance kind {kind!r}")


def _provisional_trick_winner(state, actor, action):
  """Return the current leader without guessing any unplayed hidden cards."""
  move = action_to_move(action, state._num_card_types)
  cards = list(state._cards_played_this_trick)
  cards[int(actor)] = (int(move["rank"]), str(move["suit"]))
  played = [
      (player, card) for player, card in enumerate(cards) if card is not None
  ]
  trump = [(player, card[0]) for player, card in played if card[1] == "R"]
  if trump:
    return int(max(trump, key=lambda item: item[1])[0])
  led_color = state._led_color or move["suit"]
  following = [
      (player, card[0]) for player, card in played if card[1] == led_color
  ]
  return int(max(following, key=lambda item: item[1])[0])


def _public_action_review(state, actor, action, accepted, known_bad,
                          terminal_rows=None):
  """Describe one legal choice using only the actor's own and public state."""
  action = int(action)
  actor = int(actor)
  move = action_to_move(action, state._num_card_types)
  led_color = state._led_color
  after = state.clone()
  after.apply_action(action)
  loses_led_token = False
  if led_color is not None:
    led_index = COLORS.index(led_color)
    loses_led_token = bool(
        state._color_tokens[actor, led_index]
        and not after._color_tokens[actor, led_index]
    )
  all_other_cards_played = (
      sum(card is not None for card in state._cards_played_this_trick)
      == int(state._num_players) - 1
  )
  unplayed_opponents = [
      player for player, card in enumerate(state._cards_played_this_trick)
      if player != actor and card is None
  ]
  currently_winning = _provisional_trick_winner(state, actor, action) == actor
  unbeatable = currently_winning
  if currently_winning and unplayed_opponents:
    for opponent in unplayed_opponents:
      if bool(state._color_tokens[opponent, 0]):
        lowest_beating_red = int(move["rank"]) if move["suit"] == "R" else 0
        if np.any(after._board_ownership[0, lowest_beating_red:] == -1):
          unbeatable = False
          break
      if move["suit"] != "R":
        led_index = COLORS.index(led_color or move["suit"])
        if (
            bool(state._color_tokens[opponent, led_index])
            and np.any(
                after._board_ownership[led_index, int(move["rank"]):] == -1
            )
        ):
          unbeatable = False
          break
  own_remaining_feasible = all(
      int(count) <= sum(
          bool(after._color_tokens[actor, color])
          and int(after._board_ownership[color, rank]) == -1
          for color in range(len(COLORS))
      )
      for rank, count in enumerate(after._hands[actor])
  )
  own_buffer_deficit = sum(
      max(
          0,
          int(count) + 1 - sum(
              bool(after._color_tokens[actor, color])
              and int(after._board_ownership[color, rank]) == -1
              for color in range(len(COLORS))
          ),
      )
      for rank, count in enumerate(after._hands[actor])
      if int(count) > 0
  )
  row = {
      "action": action,
      "move": move,
      "accepted": action in accepted,
      "knownBad": action in known_bad,
      "followsLedSuit": (
          move["suit"] == led_color if led_color is not None else None
      ),
      "losesLedColorToken": loses_led_token,
      "currentlyWinningAgainstPlayedCards": currently_winning,
      "remainingUnplayedOpponentCount": len(unplayed_opponents),
      "unbeatableByRemainingLegalCards": unbeatable,
      "ownRemainingHandFeasible": own_remaining_feasible,
      "ownExitMarginAfter": int(liveness_oracle.post_play_margin(
          liveness_oracle.view_from_state(state, actor), action
      )),
      "ownFutureBufferDeficitAfter": int(own_buffer_deficit),
      "ownRemainingColorTokenCount": int(
          np.sum(after._color_tokens[actor])
      ),
      "wouldWinCompletedTrick": (
          int(after._tricks_won[actor]) > int(state._tricks_won[actor])
          if all_other_cards_played else None
      ),
      "ownConnectedRegionAfter": int(after._largest_cluster_for_player(actor)),
  }
  if terminal_rows is not None:
    result = terminal_rows[action]
    row.update({
        "officialPlacementPoints": float(result["rankPoints"]),
        "ownFinalMatchScore": float(result["ownTotalScore"]),
        "ownFinalRoundScore": float(result["ownRoundScore"]),
        "ownFinalTricksWon": int(result["ownTricksWon"]),
        "ownFinalConnectedBonus": float(result["ownConnectedBonus"]),
    })
  return row


def _bid_regime_safety_key(row):
  return (
      int(bool(row["ownRemainingHandFeasible"])),
      int(row["ownExitMarginAfter"]),
      -int(row["ownFutureBufferDeficitAfter"]),
      int(row["ownRemainingColorTokenCount"]),
  )


def _bid_regime_at_least_as_safe(candidate, reference):
  return (
      bool(candidate["ownRemainingHandFeasible"])
      and int(candidate["ownExitMarginAfter"])
      >= int(reference["ownExitMarginAfter"])
      and int(candidate["ownFutureBufferDeficitAfter"])
      <= int(reference["ownFutureBufferDeficitAfter"])
      and int(candidate["ownRemainingColorTokenCount"])
      >= int(reference["ownRemainingColorTokenCount"])
  )


def _bid_regime_reasonable_actions(scenario, state, action_reviews, kind):
  actor = int(scenario["state"]["actingPlayer"])
  prediction = int(state._predictions[actor])
  won = int(state._tricks_won[actor])
  remaining = max(
      0, int(state._num_tricks) - int(state._trick_number)
  )
  winning = [
      row for row in action_reviews
      if row["currentlyWinningAgainstPlayedCards"]
      and row["unbeatableByRemainingLegalCards"]
  ]
  losing = [
      row for row in action_reviews
      if not row["currentlyWinningAgainstPlayedCards"]
  ]
  if not winning or not losing:
    raise ValueError(
        f"Scenario {scenario['id']} needs public winning and losing choices"
    )
  if kind == "safe_losing_trick_to_preserve_exact_bid":
    if won != prediction:
      raise ValueError(f"Scenario {scenario['id']} has not exactly met its bid")
    reference = max(winning, key=_bid_regime_safety_key)
    preferred = losing
  else:
    if kind == "safe_winning_trick_for_needed_bid":
      if not (won < prediction <= won + remaining):
        raise ValueError(
            f"Scenario {scenario['id']} does not have a reachable higher bid"
        )
    elif kind == "safe_winning_trick_after_irreversible_bid_miss":
      if not (won > prediction or won + remaining < prediction):
        raise ValueError(f"Scenario {scenario['id']} has not missed its bid")
    else:
      raise ValueError(f"Unsupported bid-regime rule {kind!r}")
    reference = max(losing, key=_bid_regime_safety_key)
    preferred = winning
  return {
      row["action"] for row in preferred
      if _bid_regime_at_least_as_safe(row, reference)
  }


def _validate_reasonable_action_rule(scenario, state, accepted_actions,
                                     action_reviews):
  """Require locked labels to enumerate the full declared reasonable set."""
  rule = scenario["acceptance"].get("reasonableActionRule")
  if not isinstance(rule, dict) or not rule.get("kind"):
    raise ValueError(
        f"Scenario {scenario['id']} does not declare a complete "
        "reasonable-action rule"
    )
  actor = int(scenario["state"]["actingPlayer"])
  kind = rule["kind"]
  if kind == "winning_follow_or_unbeatable_trump_for_needed_bid":
    if state._led_color is None:
      raise ValueError(f"Scenario {scenario['id']} follow rule has no led suit")
    if int(state._predictions[actor]) <= int(state._tricks_won[actor]):
      raise ValueError(f"Scenario {scenario['id']} follow rule needs no trick")
    reasonable = {
        row["action"] for row in action_reviews
        if (
            row["followsLedSuit"]
            and row["currentlyWinningAgainstPlayedCards"]
            and not row["losesLedColorToken"]
        ) or (
            row["move"]["suit"] == "R"
            and row["remainingUnplayedOpponentCount"] > 0
            and row["unbeatableByRemainingLegalCards"]
            and row["ownRemainingHandFeasible"]
        )
    }
  elif kind == "highest_nontrump_lead_for_needed_bid":
    if state._led_color is not None:
      raise ValueError(f"Scenario {scenario['id']} lead rule already has a suit")
    if int(state._predictions[actor]) <= int(state._tricks_won[actor]):
      raise ValueError(f"Scenario {scenario['id']} lead rule needs no trick")
    nontrump = [row for row in action_reviews if row["move"]["suit"] != "R"]
    if not nontrump:
      raise ValueError(f"Scenario {scenario['id']} has no legal nontrump lead")
    highest_rank = max(int(row["move"]["rank"]) for row in nontrump)
    reasonable = {
        row["action"] for row in nontrump
        if int(row["move"]["rank"]) == highest_rank
    }
  elif kind == "best_official_match_placement":
    if any("officialPlacementPoints" not in row for row in action_reviews):
      raise ValueError(
          f"Scenario {scenario['id']} placement rule has no exact proof"
      )
    best_placement = max(row["officialPlacementPoints"] for row in action_reviews)
    reasonable = {
        row["action"] for row in action_reviews
        if np.isclose(row["officialPlacementPoints"], best_placement)
    }
  elif kind in (
      "safe_winning_trick_for_needed_bid",
      "safe_losing_trick_to_preserve_exact_bid",
      "safe_winning_trick_after_irreversible_bid_miss",
  ):
    reasonable = _bid_regime_reasonable_actions(
        scenario, state, action_reviews, kind
    )
  else:
    raise ValueError(
        f"Scenario {scenario['id']} has unsupported reasonable-action "
        f"rule {kind!r}"
    )
  if not reasonable:
    raise ValueError(f"Scenario {scenario['id']} has no reasonable legal move")
  if set(accepted_actions) != reasonable:
    raise ValueError(
        f"Scenario {scenario['id']} omits or adds strategically acceptable "
        f"moves: expected the complete set {sorted(reasonable)}"
    )


def _official_terminal_action_rows(state, actor):
  """Solve a last action from own hand and public, official final scoring."""
  actor = int(actor)
  players = int(state._num_players)
  if int(state._match_round) != players - 1:
    raise ValueError("Exact tactical proof must be in the final match round")
  if int(state._trick_number) != int(state._num_tricks) - 1:
    raise ValueError("Exact tactical proof must be in the final trick")
  if sum(card is not None for card in state._cards_played_this_trick) != players - 1:
    raise ValueError("Exact tactical proof must be the final player action")

  rows = []
  for action in state.legal_actions(actor):
    action = int(action)
    if action == 999:
      continue
    terminal = state.clone()
    terminal.apply_action(action)
    if not terminal.is_terminal():
      raise ValueError("Exact tactical action did not finish the match")
    totals = [float(value) for value in terminal.returns()]
    finals = [float(value) for value in terminal.raw_round_scores()]
    own_official = (totals[actor], finals[actor])
    rank_points = sum(
        1.0 if own_official > (totals[player], finals[player])
        else 0.5 if own_official == (totals[player], finals[player])
        else 0.0
        for player in range(players) if player != actor
    )
    rows.append({
        "action": action,
        "rankPoints": float(rank_points),
        "ownTotalScore": totals[actor],
        "ownRoundScore": finals[actor],
        "ownTricksWon": int(terminal._tricks_won[actor]),
        "ownPrediction": int(terminal._predictions[actor]),
        "ownConnectedBonus": float(terminal._player_adjacency_bonus[actor]),
    })
  if len(rows) < 2:
    raise ValueError("Exact tactical proof needs at least two legal choices")
  rows.sort(key=lambda row: (
      -row["rankPoints"], -row["ownTotalScore"], -row["ownRoundScore"],
      row["action"],
  ))
  return rows


def _validate_exact_terminal_proof(scenario, state, accepted_actions):
  proof = scenario.get("proof")
  source_kind = scenario.get("source", {}).get("kind")
  if proof is None:
    if source_kind == "exhaustive_final_match_official_scoring":
      raise ValueError(
          f"Scenario {scenario['id']} claims exact scoring without proof"
      )
    return None
  if source_kind != "exhaustive_final_match_official_scoring":
    raise ValueError(
        f"Scenario {scenario['id']} exact proof has an unrecognized source"
    )
  if proof.get("kind") != EXACT_TERMINAL_PROOF:
    raise ValueError(f"Unsupported tactical proof kind {proof.get('kind')!r}")
  if proof.get("objective") != EXACT_TERMINAL_OBJECTIVE:
    raise ValueError(
        f"Scenario {scenario['id']} changes the exact official objective"
    )
  rows = _official_terminal_action_rows(state, scenario["state"]["actingPlayer"])
  best_placement = max(float(row["rankPoints"]) for row in rows)
  proven = {
      row["action"] for row in rows
      if np.isclose(float(row["rankPoints"]), best_placement)
  }
  if set(accepted_actions) != proven:
    raise ValueError(
        f"Scenario {scenario['id']} accepted actions do not match exact "
        f"official final placement: expected {sorted(proven)}"
    )
  declared = {
      int(row["action"]): row for row in proof.get("allLegalActions", [])
  }
  if set(declared) != {row["action"] for row in rows}:
    raise ValueError(f"Scenario {scenario['id']} exact proof omits legal moves")
  for row in rows:
    declared_row = declared[row["action"]]
    for key in (
        "rankPoints", "ownTotalScore", "ownRoundScore", "ownTricksWon",
        "ownPrediction", "ownConnectedBonus",
    ):
      if not np.isclose(float(declared_row.get(key, np.nan)), float(row[key])):
        raise ValueError(
            f"Scenario {scenario['id']} exact proof changed {key} for "
            f"action {row['action']}"
        )
  return rows


def validate_scenario(scenario):
  if scenario.get("reviewStatus") != "locked":
    raise ValueError(f"Scenario {scenario.get('id')} is not locked")
  if scenario.get("trainingEligible") is not False:
    raise ValueError(f"Scenario {scenario.get('id')} is training-eligible")
  payload = scenario["state"]
  players = int(payload["numPlayers"])
  ranks = int(payload["numRanks"])
  actor = int(payload["actingPlayer"])
  acceptance = scenario["acceptance"]
  accepted_entries = accepted_action_entries(acceptance)
  if len(payload["currentTrick"]) != players:
    raise ValueError(f"Scenario {scenario['id']} has malformed current trick")
  vector_fields = (
      "playedCountByPlayer",
      "predictions",
      "tricksWon",
      "colorTokens",
      "hasParadoxed",
      "matchTotals",
  )
  for field in vector_fields:
    if len(payload[field]) != players:
      raise ValueError(f"Scenario {scenario['id']} has malformed {field}")
  if len(payload["actingHandRankCounts"]) != ranks:
    raise ValueError(f"Scenario {scenario['id']} has malformed acting hand")
  if len(payload["boardOwnership"]) != len(COLORS) or any(
      len(row) != ranks for row in payload["boardOwnership"]
  ):
    raise ValueError(f"Scenario {scenario['id']} has malformed board")
  if any(len(row) != len(COLORS) for row in payload["colorTokens"]):
    raise ValueError(f"Scenario {scenario['id']} has malformed color tokens")
  if actor != int(payload["currentPlayer"]):
    raise ValueError(f"Scenario {scenario['id']} actor is not current player")

  played_counts = [0] * players
  for trick in payload.get("completedTricks", []):
    if len(trick) != players:
      raise ValueError(f"Scenario {scenario['id']} has malformed completed trick")
    for player, card in enumerate(trick):
      if card is not None:
        _card_tuple(card, player)
        played_counts[player] += 1
  for player, card in enumerate(payload["currentTrick"]):
    if card is not None:
      _card_tuple(card, player)
      played_counts[player] += 1
  if played_counts != [int(value) for value in payload["playedCountByPlayer"]]:
    raise ValueError(f"Scenario {scenario['id']} played counts do not match")
  expected_hand_size = (
      int(payload["cardsPerPlayerInitial"]) - 1 - played_counts[actor]
  )
  if sum(int(value) for value in payload["actingHandRankCounts"]) != expected_hand_size:
    raise ValueError(f"Scenario {scenario['id']} acting hand size is inconsistent")

  state = build_state(scenario)
  legal = [int(action) for action in state.legal_actions(actor)]
  if len(legal) != int(payload["expectedLegalActionCount"]):
    raise ValueError(f"Scenario {scenario['id']} legal action count changed")
  accepted_actions = []
  for accepted in accepted_entries:
    action = int(accepted["action"])
    if move_to_action(accepted["move"], ranks) != action:
      raise ValueError(f"Scenario {scenario['id']} accepted move/action disagree")
    if action not in legal:
      raise ValueError(f"Scenario {scenario['id']} accepted action is not legal")
    accepted_actions.append(action)
  if len(accepted_actions) != len(set(accepted_actions)):
    raise ValueError(f"Scenario {scenario['id']} repeats an accepted action")
  known_bad_actions = set()
  for known_bad in acceptance.get("knownBadActions", []):
    action = int(known_bad["action"])
    if move_to_action(known_bad["move"], ranks) != action:
      raise ValueError(f"Scenario {scenario['id']} known-bad move/action disagree")
    if action not in legal:
      raise ValueError(f"Scenario {scenario['id']} known-bad action is not legal")
    if action in accepted_actions:
      raise ValueError(f"Scenario {scenario['id']} accepts a known-bad action")
    known_bad_actions.add(action)
  unavailable_actions = set()
  for unavailable in acceptance.get("knownUnavailableMoves", []):
    action = int(unavailable["action"])
    if move_to_action(unavailable["move"], ranks) != action:
      raise ValueError(
          f"Scenario {scenario['id']} unavailable move/action disagree"
      )
    if action in legal:
      raise ValueError(
          f"Scenario {scenario['id']} claims a legal move is unavailable"
      )
    if action in unavailable_actions:
      raise ValueError(
          f"Scenario {scenario['id']} repeats an unavailable move"
      )
    unavailable_actions.add(action)
  terminal_rows = _validate_exact_terminal_proof(
      scenario, state, accepted_actions
  )
  terminal_by_action = (
      {int(row["action"]): row for row in terminal_rows}
      if terminal_rows is not None else None
  )
  action_reviews = [
      _public_action_review(
          state, actor, action, set(accepted_actions), known_bad_actions,
          terminal_by_action,
      )
      for action in legal
  ]
  _validate_reasonable_action_rule(
      scenario, state, accepted_actions, action_reviews
  )
  observation = np.asarray(state.observation_tensor(actor), dtype=np.float32)
  return {
      "scenario": scenario,
      "state": state,
      "legalActions": legal,
      "legalActionReview": action_reviews,
      "observationSize": int(observation.size),
      "observationSHA256": hashlib.sha256(observation.tobytes()).hexdigest(),
  }


def _rotate_player_vector(values, rotation):
  result = [None] * len(values)
  for player, value in enumerate(values):
    result[(player + rotation) % len(values)] = copy.deepcopy(value)
  return result


def rotate_scenario_seats(scenario, rotation):
  """Relabel every seat consistently; game outcome and best move are invariant."""
  result = copy.deepcopy(scenario)
  state = result["state"]
  players = int(state["numPlayers"])
  rotation = int(rotation) % players
  if rotation == 0:
    return result
  original_id = str(result["id"])
  result["id"] = f"{original_id}--seat-r{rotation}"
  result["title"] = f"{result['title']} (seat rotation {rotation})"
  result["metamorphic"] = {
      "familyID": original_id,
      "kind": "cyclic_seat_relabeling",
      "rotation": rotation,
  }
  for field in (
      "actingPlayer", "currentPlayer", "roundStartPlayer", "startPlayer",
  ):
    state[field] = (int(state[field]) + rotation) % players
  for field in (
      "playedCountByPlayer", "predictions", "tricksWon", "colorTokens",
      "hasParadoxed", "matchTotals",
  ):
    state[field] = _rotate_player_vector(state[field], rotation)
  for trick in [state["currentTrick"], *state.get("completedTricks", [])]:
    original = list(trick)
    for player, card in enumerate(original):
      new_player = (player + rotation) % players
      if card is not None:
        card = copy.deepcopy(card)
        card["player"] = new_player
      trick[new_player] = card
  state["boardOwnership"] = [
      [
          (int(owner) + rotation) % players if int(owner) >= 0 else int(owner)
          for owner in row
      ]
      for row in state["boardOwnership"]
  ]
  for proof_row in result.get("proof", {}).get("allLegalActions", []):
    for field in ("totals", "roundScores"):
      if field in proof_row:
        proof_row[field] = _rotate_player_vector(proof_row[field], rotation)
  return result


def _seat_rotations(payload, players):
  configuration = payload.get("metamorphicValidation", {})
  requested = configuration.get("seatRotations", [0])
  if requested == "all":
    return list(range(int(players)))
  if not isinstance(requested, list) or not requested:
    raise ValueError("Tactical seat rotations must be a nonempty list or 'all'")
  rotations = [int(value) for value in requested]
  if rotations[0] != 0 or len(rotations) != len(set(rotations)):
    raise ValueError("Tactical seat rotations must start at zero and be unique")
  if any(value < 0 or value >= int(players) for value in rotations):
    raise ValueError("Tactical seat rotation is outside its player count")
  return rotations


def validate_suite(payload):
  if int(payload.get("schemaVersion", -1)) != 1:
    raise ValueError("Unsupported curated validation schema")
  if payload.get("trainingPolicy", {}).get("allowInTraining") is not False:
    raise ValueError("Curated validation suite must be excluded from training")
  scenarios = payload.get("scenarios", [])
  if not scenarios:
    raise ValueError("Curated validation suite is empty")
  ids = [scenario.get("id") for scenario in scenarios]
  if any(not value for value in ids) or len(ids) != len(set(ids)):
    raise ValueError("Curated validation scenario IDs must be unique")
  # Keep every independently reviewed/proved root first. Derived relabelings
  # increase robustness coverage, not the number of independent scenario families.
  validated = [validate_scenario(scenario) for scenario in scenarios]
  for scenario in scenarios:
    for rotation in _seat_rotations(payload, scenario["state"]["numPlayers"])[1:]:
      validated.append(validate_scenario(
          rotate_scenario_seats(scenario, rotation)
      ))
  return validated


def evaluate_validated(validated, selector, candidate_label):
  results = []
  for item in validated:
    scenario = item["scenario"]
    actor = int(scenario["state"]["actingPlayer"])
    actual = int(selector(item["state"].clone(), actor))
    accepted_entries = accepted_action_entries(scenario["acceptance"])
    accepted = [int(entry["action"]) for entry in accepted_entries]
    result = {
        "id": scenario["id"],
        "familyID": scenario.get("metamorphic", {}).get(
            "familyID", scenario["id"]
        ),
        "sourceKind": scenario.get("source", {}).get("kind"),
        "seatRotation": int(
            scenario.get("metamorphic", {}).get("rotation", 0)
        ),
        "players": int(scenario["state"]["numPlayers"]),
        "acceptanceRule": copy.deepcopy(
            scenario["acceptance"]["reasonableActionRule"]
        ),
        "acceptanceConfidence": (
            "exhaustive_final_match_official_scoring"
            if scenario.get("proof") is not None
            else "human_reviewed_public_tactical_judgment"
        ),
        "decisionContext": {
            "ledSuit": item["state"]._led_color,
            "trickNumber": int(item["state"]._trick_number),
            "ownPrediction": int(item["state"]._predictions[actor]),
            "ownTricksWon": int(item["state"]._tricks_won[actor]),
        },
        "acceptedActions": accepted,
        "acceptedMoves": [
            action_to_move(action, scenario["state"]["numRanks"])
            for action in accepted
        ],
        "knownBadMoves": copy.deepcopy(
            scenario["acceptance"].get("knownBadActions", [])
        ),
        "knownUnavailableMoves": copy.deepcopy(
            scenario["acceptance"].get("knownUnavailableMoves", [])
        ),
        "actualAction": actual,
        "actualMove": action_to_move(
            actual, scenario["state"]["numRanks"]
        ),
        "passed": actual in accepted,
        "legalActionCount": len(item["legalActions"]),
        "legalActionReview": copy.deepcopy(item["legalActionReview"]),
        "observationSize": item["observationSize"],
        "observationSHA256": item["observationSHA256"],
    }
    if len(accepted) == 1:
      result["expectedAction"] = accepted[0]
      result["expectedMove"] = result["acceptedMoves"][0]
    results.append(result)
  families = {}
  for result in results:
    family = families.setdefault(result["familyID"], {
        "id": result["familyID"],
        "sourceKind": result["sourceKind"],
        "scenarioCount": 0,
        "passedCount": 0,
        "passed": True,
    })
    family["scenarioCount"] += 1
    family["passedCount"] += int(bool(result["passed"]))
    family["passed"] = family["passed"] and bool(result["passed"])
  return {
      "suiteID": "quantum-cat-curated-tactical-validation-v1",
      "candidate": candidate_label,
      "scenarioCount": len(results),
      "passedCount": sum(bool(row["passed"]) for row in results),
      "independentScenarioFamilyCount": len(families),
      "humanReviewedFamilyCount": sum(
          family["sourceKind"] == "human_reviewed_iphone_capture"
          for family in families.values()
      ),
      "exactOfficialScoringFamilyCount": sum(
          family["sourceKind"] == "exhaustive_final_match_official_scoring"
          for family in families.values()
      ),
      "families": list(families.values()),
      "passed": all(bool(row["passed"]) for row in results),
      "results": results,
  }


def _candidate_selector(candidate_spec, players, device_name,
                        belief_samples=8, belief_sims=16, mcts_sims=16):
  import torch
  from quantum_cat_full_match_elo import load_neural_bots, make_eval_bot

  device = torch.device(device_name)
  neural = load_neural_bots(
      [f"curated_candidate={candidate_spec}"], players, device
  )
  bot = make_eval_bot(
      "curated_candidate",
      0,
      None,
      None,
      device,
      belief_samples=int(belief_samples),
      belief_sims=int(belief_sims),
      mcts_sims=int(mcts_sims),
      neural_bots=neural,
  )
  return lambda state, actor: bot.step(state, actor)


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
  parser.add_argument(
      "--candidate",
      help=(
          "Candidate checkpoint policy specification, using the same suffix "
          "and ?key=value overrides as quantum_cat_full_match_elo.py"
      ),
  )
  parser.add_argument("--players", type=int, choices=(3, 4, 5))
  parser.add_argument("--scenario", action="append", default=[])
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--belief-samples", type=int, default=8)
  parser.add_argument("--belief-sims", type=int, default=16)
  parser.add_argument("--mcts-sims", type=int, default=16)
  parser.add_argument("--out", type=Path)
  return parser.parse_args()


def main():
  args = parse_args()
  payload = load_suite(args.suite)
  validated = validate_suite(payload)
  selected = [
      item for item in validated
      if (args.players is None or
          int(item["scenario"]["state"]["numPlayers"]) == args.players)
      and (not args.scenario or item["scenario"]["id"] in args.scenario)
  ]
  if not selected:
    raise SystemExit("No curated scenarios matched the requested filters")

  if args.candidate:
    player_counts = {
        int(item["scenario"]["state"]["numPlayers"]) for item in selected
    }
    if len(player_counts) != 1:
      raise SystemExit("A candidate run must select exactly one player count")
    players = next(iter(player_counts))
    report = evaluate_validated(
        selected,
        _candidate_selector(
            args.candidate, players, args.device,
            belief_samples=args.belief_samples,
            belief_sims=args.belief_sims,
            mcts_sims=args.mcts_sims,
        ),
        args.candidate,
    )
    report["suiteID"] = payload["suiteID"]
  else:
    family_sources = {
        item["scenario"].get("metamorphic", {}).get(
            "familyID", item["scenario"]["id"]
        ): item["scenario"].get("source", {}).get("kind")
        for item in selected
    }
    coverage = {}
    for item in selected:
      players = int(item["scenario"]["state"]["numPlayers"])
      family_id = item["scenario"].get("metamorphic", {}).get(
          "familyID", item["scenario"]["id"]
      )
      row = coverage.setdefault(players, {
          "players": players, "scenarioCount": 0, "familyIDs": set(),
      })
      row["scenarioCount"] += 1
      row["familyIDs"].add(family_id)
    report = {
        "suiteID": payload["suiteID"],
        "mode": "schema_and_state_validation_only",
        "scenarioCount": len(selected),
        "independentScenarioFamilyCount": len(family_sources),
        "humanReviewedFamilyCount": sum(
            source == "human_reviewed_iphone_capture"
            for source in family_sources.values()
        ),
        "exactOfficialScoringFamilyCount": sum(
            source == "exhaustive_final_match_official_scoring"
            for source in family_sources.values()
        ),
        "coverageByPlayers": [
            {
                "players": players,
                "scenarioCount": row["scenarioCount"],
                "independentScenarioFamilyCount": len(row["familyIDs"]),
            }
            for players, row in sorted(coverage.items())
        ],
        "missingPlayerCounts": [
            players for players in (3, 4, 5) if players not in coverage
        ] if args.players is None else [],
        "passed": True,
        "scenarios": [
            {
                "id": item["scenario"]["id"],
                "players": int(item["scenario"]["state"]["numPlayers"]),
                "acceptedActions": [
                    int(entry["action"])
                    for entry in accepted_action_entries(
                        item["scenario"]["acceptance"]
                    )
                ],
                "acceptedMoves": [
                    action_to_move(
                        entry["action"], item["scenario"]["state"]["numRanks"]
                    )
                    for entry in accepted_action_entries(
                        item["scenario"]["acceptance"]
                    )
                ],
                "acceptanceRule": item["scenario"]["acceptance"][
                    "reasonableActionRule"
                ],
                "knownUnavailableMoves": item["scenario"]["acceptance"].get(
                    "knownUnavailableMoves", []
                ),
                "acceptanceConfidence": (
                    "exhaustive_final_match_official_scoring"
                    if item["scenario"].get("proof") is not None
                    else "human_reviewed_public_tactical_judgment"
                ),
                "legalActionCount": len(item["legalActions"]),
                "observationSize": item["observationSize"],
                "observationSHA256": item["observationSHA256"],
            }
            for item in selected
        ],
    }

  rendered = json.dumps(report, indent=2, sort_keys=True)
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered + "\n", encoding="utf-8")
    console_report = copy.deepcopy(report)
    for row in console_report.get("results", []):
      reviews = row.pop("legalActionReview", None)
      if reviews is not None:
        row["legalActionsAudited"] = len(reviews)
    console_report["fullActionAudit"] = str(args.out)
    print(json.dumps(console_report, indent=2, sort_keys=True))
  else:
    print(rendered)
  if not report["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
