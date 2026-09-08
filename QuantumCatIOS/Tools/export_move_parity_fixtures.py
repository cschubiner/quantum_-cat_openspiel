#!/usr/bin/env python3
"""Export deterministic PyTorch move-parity fixtures for the iOS Core ML app."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import shared_discard_action, shared_prediction_action
from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_SIZE,
    action_feature_matrix,
    adapt_observation,
    load_model_payload,
    model_input_size,
)


def checkpoint_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def checkpoint_args(path: Path, players: int) -> SimpleNamespace:
  payload = torch.load(path, map_location="cpu")
  saved = payload.get("args", {})
  return SimpleNamespace(
      players=int(saved.get("players", players)),
      arch=saved.get("arch", "action_mlp"),
      width=int(saved.get("width", 256)),
      depth=int(saved.get("depth", 3)),
      value_scale=float(saved.get("value_scale", 20.0)),
      separate_action_value_encoder=bool(
          saved.get("separate_action_value_encoder", False)
      ),
      separate_action_paradox_encoder=bool(
          saved.get("separate_action_paradox_encoder", False)
      ),
      separate_prediction_policy_head=bool(
          saved.get("separate_prediction_policy_head", False)
      ),
      prediction_policy_input=str(saved.get("prediction_policy_input", "pair")),
  )


def raw_policy_inputs(model, state, player, max_actions):
  legal = [int(action) for action in state.legal_actions(player)]
  observation = adapt_observation(
      np.asarray(state.observation_tensor(player), dtype=np.float32),
      model_input_size(model),
  )
  full_features = action_feature_matrix(
      state, player, state.num_distinct_actions()
  )
  compressed = np.zeros(
      (max_actions, ACTION_FEATURE_SIZE), dtype=np.float32
  )
  for row, action in enumerate(legal):
    compressed[row] = full_features[action]
  with torch.no_grad():
    logits, _ = model(
        torch.from_numpy(observation).unsqueeze(0),
        torch.from_numpy(compressed).unsqueeze(0),
    )
  legal_logits = logits.squeeze(0).cpu().numpy()[:len(legal)]
  order = np.argsort(legal_logits)
  margin = (
      float(legal_logits[order[-1]] - legal_logits[order[-2]])
      if len(order) > 1 else float("inf")
  )
  return legal, observation, compressed[:len(legal)], legal_logits, margin


def live_state_snapshot(state, player):
  """Information-safe native state needed to reproduce the 76/214 inputs.

  Opponent hands and face-down discards are deliberately excluded. The live
  encoder is allowed to use only the acting player's private information plus
  state that is public at the table.
  """
  current_trick = []
  for card in state._cards_played_this_trick:  # pylint: disable=protected-access
    if card is None:
      current_trick.append(None)
    else:
      rank, suit = card
      current_trick.append({"rank": int(rank), "suit": str(suit)})
  return {
      "numPlayers": int(state._num_players),  # pylint: disable=protected-access
      "numRanks": int(state._num_card_types),  # pylint: disable=protected-access
      "numColors": int(state._num_colors),  # pylint: disable=protected-access
      "numTricks": int(state._num_tricks),  # pylint: disable=protected-access
      "cardsPerPlayerInitial": int(
          state._cards_per_player_initial  # pylint: disable=protected-access
      ),
      "phase": int(state._phase),  # pylint: disable=protected-access
      "currentPlayer": int(state.current_player()),
      "actingPlayer": int(player),
      "trickNumber": int(state._trick_number),  # pylint: disable=protected-access
      "startPlayer": int(state._start_player),  # pylint: disable=protected-access
      "roundStartPlayer": int(
          state._round_start_player  # pylint: disable=protected-access
      ),
      "ledColor": state._led_color,  # pylint: disable=protected-access
      "currentTrick": current_trick,
      "playedCounts": [
          int(state._count_cards_played_by(seat))  # pylint: disable=protected-access
          for seat in range(int(state._num_players))  # pylint: disable=protected-access
      ],
      "predictions": [
          int(value) for value in state._predictions  # pylint: disable=protected-access
      ],
      "tricksWon": [
          int(value) for value in state._tricks_won  # pylint: disable=protected-access
      ],
      "board": np.asarray(
          state._board_ownership, dtype=np.int32  # pylint: disable=protected-access
      ).tolist(),
      "colorTokens": np.asarray(
          state._color_tokens, dtype=bool  # pylint: disable=protected-access
      ).tolist(),
      "actingHand": np.asarray(
          state._hands[player], dtype=np.int32  # pylint: disable=protected-access
      ).tolist(),
      "actingDiscardedRank": int(
          state._discarded_cards[player]  # pylint: disable=protected-access
      ),
      "trumpBroken": bool(state._trump_broken),  # pylint: disable=protected-access
      "matchTotals": np.asarray(
          state._match_totals, dtype=np.float32  # pylint: disable=protected-access
      ).tolist(),
      "matchRound": int(state._match_round),  # pylint: disable=protected-access
      "isTerminal": bool(state.is_terminal()),
  }


def choose_round_action(model, state, player, max_actions):
  legal = [int(action) for action in state.legal_actions(player)]
  setup = shared_discard_action(state, player, legal)
  if setup is None:
    setup = shared_prediction_action(state, player, legal)
  if setup is not None:
    return int(setup)
  legal, _, _, logits, _ = raw_policy_inputs(
      model, state, player, max_actions
  )
  return legal[int(np.argmax(logits))]


def auxiliary_outputs(model, observation, action_features, max_actions):
  """Export the actual dedicated risk/value heads, not shared policy features."""
  compressed = np.zeros(
      (max_actions, ACTION_FEATURE_SIZE), dtype=np.float32
  )
  compressed[:len(action_features)] = action_features
  with torch.no_grad():
    (_, values, _, action_paradox, action_values) = (
        model.forward_with_all_aux(
            torch.from_numpy(observation).unsqueeze(0),
            torch.from_numpy(compressed).unsqueeze(0),
        )
    )
  legal_count = len(action_features)
  return {
      "expectedStateValues": values.squeeze(0).cpu().numpy().tolist(),
      "expectedActionParadoxLogits": (
          action_paradox.squeeze(0).cpu().numpy()[:legal_count].tolist()
      ),
      "expectedActionValues": (
          action_values.squeeze(0).cpu().numpy()[:legal_count].tolist()
      ),
  }


def collect_fixtures(model, args):
  rng = np.random.default_rng(args.seed)
  fixtures = []
  above_bid_count = 0
  for match_index in range(args.max_matches):
    fixtures_this_match = 0
    start_player = match_index % args.players
    game = pyspiel.load_game(
        "python_quantum_cat",
        {
            "players": args.players,
            "start_player": start_player,
            "match_context": 1,
        },
    )
    state = game.new_initial_state()
    state.set_match_context(np.zeros(args.players, dtype=np.float32), 0)
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probabilities = zip(*outcomes)
        state.apply_action(int(rng.choice(actions, p=probabilities)))
        continue
      player = int(state.current_player())
      phase = int(getattr(state, "_phase", -1))
      legal = [int(action) for action in state.legal_actions(player)]
      if phase == 3 and len(legal) >= 2:
        (legal, observation, features, logits, margin) = raw_policy_inputs(
            model, state, player, args.max_actions
        )
        prediction = int(state._predictions[player])  # pylint: disable=protected-access
        tricks_won = int(state._tricks_won[player])  # pylint: disable=protected-access
        above_bid = prediction >= 0 and tricks_won > prediction
        reserve_above_bid = bool(
            args.require_above_bid
            and above_bid_count == 0
            and len(fixtures) >= args.fixtures - 1
            and not above_bid
        )
        if margin >= args.min_top_margin and not reserve_above_bid:
          expected_index = int(np.argmax(logits))
          fixture = {
              "id": (
                  f"{args.fixture_id_prefix}m{match_index}_"
                  f"t{int(getattr(state, '_trick_number', -1))}_p{player}"
              ),
              "legalActions": legal,
              "observation": observation.tolist(),
              "actionFeatures": features.tolist(),
              "expectedIndex": expected_index,
              "expectedAction": int(legal[expected_index]),
              "expectedLogits": logits.astype(np.float32).tolist(),
              "topMargin": margin,
              "liveState": live_state_snapshot(state, player),
          }
          if args.include_auxiliary_outputs:
            fixture.update(
                auxiliary_outputs(
                    model, observation, features, args.max_actions
                )
            )
          fixtures.append(fixture)
          above_bid_count += int(above_bid)
          fixtures_this_match += 1
          if (
              len(fixtures) >= args.fixtures
              and (not args.require_above_bid or above_bid_count > 0)
          ):
            return fixtures
      state.apply_action(
          choose_round_action(model, state, player, args.max_actions)
      )
      if (
          fixtures_this_match >= args.max_fixtures_per_match
          and (not args.require_above_bid or above_bid_count > 0)
      ):
        break
  raise RuntimeError(
      "collected only "
      f"{len(fixtures)} stable fixtures; requested {args.fixtures}; "
      f"above_bid_fixtures={above_bid_count}"
  )


def collect_prediction_fixtures(model, args):
  """Exercise the actual learned bid head on actor-public real deal states."""
  requested = int(getattr(args, "prediction_fixtures", 0))
  if requested <= 0:
    return []
  if not getattr(model, "separate_prediction_policy_head", False):
    raise ValueError("Prediction fixtures require a real trained bid head")
  rng = np.random.default_rng(int(args.seed) + 90_001)
  fixtures = []
  selected_bids = set()
  for match_index in range(args.max_matches):
    start_player = match_index % args.players
    game = pyspiel.load_game("python_quantum_cat", {
        "players": args.players, "start_player": start_player,
        "match_context": 1,
    })
    state = game.new_initial_state()
    state.set_match_context(np.zeros(args.players, dtype=np.float32), 0)
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probabilities = zip(*outcomes)
        state.apply_action(int(rng.choice(actions, p=probabilities)))
        continue
      player = int(state.current_player())
      if int(getattr(state, "_phase", -1)) == 2:
        legal, observation, features, logits, margin = raw_policy_inputs(
            model, state, player, args.max_actions
        )
        if margin >= args.min_top_margin:
          best = int(np.argmax(logits))
          fixture = {
              "id": f"{args.fixture_id_prefix}bid_m{match_index}_p{player}",
              "legalActions": legal,
              "observation": observation.tolist(),
              "actionFeatures": features.tolist(),
              "expectedIndex": best,
              "expectedAction": int(legal[best]),
              "expectedLogits": logits.astype(np.float32).tolist(),
              "topMargin": margin,
              "liveState": live_state_snapshot(state, player),
          }
          if args.include_auxiliary_outputs:
            fixture.update(auxiliary_outputs(
                model, observation, features, args.max_actions
            ))
          fixtures.append(fixture)
          selected_bids.add(int(legal[best]) - 100)
          if len(fixtures) >= requested and len(selected_bids) >= 2:
            return fixtures
      state.apply_action(
          choose_round_action(model, state, player, args.max_actions)
      )
  raise RuntimeError(
      f"collected only {len(fixtures)} learned-bid fixtures with "
      f"bids={sorted(selected_bids)}; need {requested} and diverse actions"
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--fixtures", type=int, default=16)
  parser.add_argument("--max-matches", type=int, default=100)
  parser.add_argument("--max-fixtures-per-match", type=int, default=4)
  parser.add_argument("--max-actions", type=int, default=1000)
  parser.add_argument("--min-top-margin", type=float, default=0.05)
  parser.add_argument("--logit-tolerance", type=float, default=0.02)
  parser.add_argument("--seed", type=int, default=20260721)
  parser.add_argument("--model-resource", default="champion_belief_policy")
  parser.add_argument("--policy-output-name", default="var_175")
  parser.add_argument(
      "--live-input-contract", default="python-observation76-action214-v1"
  )
  parser.add_argument("--fixture-id-prefix", default="")
  parser.add_argument("--require-above-bid", action="store_true")
  parser.add_argument("--include-auxiliary-outputs", action="store_true")
  parser.add_argument("--prediction-fixtures", type=int, default=0)
  args = parser.parse_args()

  checkpoint = Path(args.checkpoint)
  game = pyspiel.load_game(
      "python_quantum_cat",
      {"players": args.players, "start_player": 0, "match_context": 1},
  )
  model, _, _ = load_model_payload(
      str(checkpoint),
      game,
      checkpoint_args(checkpoint, args.players),
      torch.device("cpu"),
  )
  model.eval()
  fixtures = collect_fixtures(model, args)
  prediction_fixtures = collect_prediction_fixtures(model, args)
  fixtures.extend(prediction_fixtures)
  payload = {
      "schemaVersion": 2,
      "checkpointSha256": checkpoint_sha256(checkpoint),
      "modelResource": args.model_resource,
      "policyOutputName": args.policy_output_name,
      "liveInputContract": args.live_input_contract,
      "liveInputTolerance": 1e-5,
      "observationSize": model_input_size(model),
      "actionFeatureSize": ACTION_FEATURE_SIZE,
      "maxActions": args.max_actions,
      "logitTolerance": args.logit_tolerance,
      "fixtures": fixtures,
  }
  if prediction_fixtures:
    payload["learnedPredictionEnabled"] = True
    payload["predictionFixtureCount"] = len(prediction_fixtures)
  if args.include_auxiliary_outputs:
    payload.update({
        "stateValueOutputName": "state_values",
        "actionParadoxOutputName": "action_paradox_logits",
        "actionValueOutputName": "action_values",
    })
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(
      f"wrote {out} fixtures={len(fixtures)} "
      f"checkpoint_sha256={payload['checkpointSha256']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
