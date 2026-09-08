#!/usr/bin/env python3
"""Mine candidate-vs-champion policy disagreements in full-match gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import make_bot
from quantum_cat_alphazero_torch import (
    APPENDED_ACTION_FEATURE_NAMES,
    APPENDED_ACTION_FEATURE_START,
    action_feature_matrix,
    load_model_payload,
    model_policy_value,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--champion", required=True)
  parser.add_argument("--candidate", required=True)
  parser.add_argument("--candidate-name", default="candidate")
  parser.add_argument("--opponent", default="heuristic")
  parser.add_argument("--paired-deals", type=int, default=50)
  parser.add_argument("--seed", type=int, default=20270671)
  parser.add_argument("--max-records", type=int, default=200)
  parser.add_argument("--out", default="az_runs/gate_miner_latest.json")
  return parser.parse_args()


def build_lineups(players, candidate_name, opponent):
  if players < 2:
    raise ValueError("--players must be at least 2")
  lineups = []
  for champion_seat in range(players):
    for candidate_seat in range(players):
      if candidate_seat == champion_seat:
        continue
      lineup = [opponent] * players
      lineup[champion_seat] = "champion"
      lineup[candidate_seat] = candidate_name
      lineups.append(tuple(lineup))
  return lineups


def model_args(players):
  return SimpleNamespace(
      players=players,
      arch="mlp",
      width=256,
      depth=3,
      value_scale=20.0,
      c_puct=1.8,
      sims=64,
      match_context=True,
  )


def load_model(path, players, device):
  args = model_args(players)
  game = pyspiel.load_game(
      "python_quantum_cat",
      {"players": players, "start_player": 0, "match_context": 1},
  )
  model, _, saved_args = load_model_payload(path, game, args, device)
  model.eval()
  value_scale = saved_args.get("value_scale", args.value_scale)
  return model, value_scale


def policy_and_action(model, value_scale, state, player, device):
  policy, _ = model_policy_value(
      model, state, player, state.num_distinct_actions(), value_scale, device
  )
  legal = state.legal_actions(player)
  action = max(legal, key=lambda legal_action: policy[legal_action])
  return policy, int(action)


def action_string(state, player, action):
  try:
    return state._action_to_string(player, action)  # pylint: disable=protected-access
  except Exception:
    return str(action)


def action_features_for(state, player, action):
  features = action_feature_matrix(state, player, state.num_distinct_actions())
  return features[action]


def appended_summary(features):
  values = features[
      APPENDED_ACTION_FEATURE_START:
      APPENDED_ACTION_FEATURE_START + len(APPENDED_ACTION_FEATURE_NAMES)
  ]
  return {
      name: float(value)
      for name, value in zip(APPENDED_ACTION_FEATURE_NAMES, values)
  }


def update_feature_totals(totals, prefix, features):
  appended = appended_summary(features)
  for name, value in appended.items():
    totals[f"{prefix}_{name}_sum"] = totals.get(f"{prefix}_{name}_sum", 0.0) + value
    if abs(value) > 1e-9:
      totals[f"{prefix}_{name}_nonzero"] = (
          totals.get(f"{prefix}_{name}_nonzero", 0) + 1
      )


def play_mined_match(
    args,
    lineup,
    deal_idx,
    initial_start,
    champion_model,
    champion_scale,
    candidate_model,
    candidate_scale,
    device,
):
  totals = np.zeros(args.players, dtype=np.float32)
  final_round_returns = np.zeros(args.players, dtype=np.float32)
  paradoxes = np.zeros(args.players, dtype=np.int32)
  records = []
  summary = {
      "candidate_decisions": 0,
      "candidate_disagreements": 0,
      "candidate_policy_prob_sum": 0.0,
      "champion_policy_prob_sum": 0.0,
  }
  fixed_bots = {
      seat: make_bot(name, seed=args.seed + deal_idx * 1009 + seat)
      for seat, name in enumerate(lineup)
      if name not in ("champion", args.candidate_name)
  }

  for round_idx in range(args.players):
    start_player = (initial_start + round_idx) % args.players
    game = pyspiel.load_game(
        "python_quantum_cat",
        {
            "players": args.players,
            "start_player": start_player,
            "match_context": 1,
        },
    )
    state = game.new_initial_state()
    state.set_match_context(totals, round_idx)
    np.random.seed((args.seed + deal_idx * 37 + round_idx * 9973) % (2**32 - 1))

    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        state.apply_action(int(np.random.choice(actions, p=probs)))
        continue

      player = state.current_player()
      role = lineup[player]
      if role == "champion":
        _, action = policy_and_action(
            champion_model, champion_scale, state, player, device
        )
      elif role == args.candidate_name:
        candidate_policy, candidate_action = policy_and_action(
            candidate_model, candidate_scale, state, player, device
        )
        champion_policy, champion_action = policy_and_action(
            champion_model, champion_scale, state, player, device
        )
        summary["candidate_decisions"] += 1
        summary["candidate_policy_prob_sum"] += float(
            candidate_policy[candidate_action]
        )
        summary["champion_policy_prob_sum"] += float(
            champion_policy[champion_action]
        )
        if candidate_action != champion_action:
          summary["candidate_disagreements"] += 1
          candidate_features = action_features_for(state, player, candidate_action)
          champion_features = action_features_for(state, player, champion_action)
          update_feature_totals(summary, "candidate", candidate_features)
          update_feature_totals(summary, "champion", champion_features)
          if len(records) < args.max_records:
            records.append({
                "deal_idx": deal_idx,
                "round_idx": round_idx,
                "lineup": list(lineup),
                "seat": player,
                "phase": int(getattr(state, "_phase", -1)),
                "trick_number": int(getattr(state, "_trick_number", -1)),
                "prediction": int(getattr(state, "_predictions", [-1])[player]),
                "tricks_won": int(getattr(state, "_tricks_won", [0])[player]),
                "candidate_action": int(candidate_action),
                "candidate_action_string": action_string(
                    state, player, candidate_action
                ),
                "candidate_prob": float(candidate_policy[candidate_action]),
                "candidate_appended": appended_summary(candidate_features),
                "champion_action": int(champion_action),
                "champion_action_string": action_string(
                    state, player, champion_action
                ),
                "champion_prob": float(champion_policy[champion_action]),
                "champion_appended": appended_summary(champion_features),
            })
        action = candidate_action
      else:
        action = fixed_bots[player].step(state, player)
      state.apply_action(action)

    round_returns = np.array(state.returns(), dtype=np.float32) - totals
    totals += round_returns
    final_round_returns = round_returns
    paradoxes += np.array(getattr(state, "_has_paradoxed", [False] * args.players))

  summary["candidate_total"] = float(
      totals[list(lineup).index(args.candidate_name)]
  )
  summary["champion_total"] = float(totals[list(lineup).index("champion")])
  summary["candidate_paradoxes"] = int(
      paradoxes[list(lineup).index(args.candidate_name)]
  )
  summary["champion_paradoxes"] = int(paradoxes[list(lineup).index("champion")])
  summary["final_round_returns"] = [float(x) for x in final_round_returns]
  return summary, records


def finalize_summary(summary):
  decisions = max(1, int(summary.get("candidate_decisions", 0)))
  disagreements = max(1, int(summary.get("candidate_disagreements", 0)))
  summary["disagreement_rate"] = (
      float(summary.get("candidate_disagreements", 0)) / decisions
  )
  summary["candidate_mean_top_prob"] = (
      float(summary.get("candidate_policy_prob_sum", 0.0)) / decisions
  )
  summary["champion_mean_top_prob"] = (
      float(summary.get("champion_policy_prob_sum", 0.0)) / decisions
  )
  for prefix in ("candidate", "champion"):
    for name in APPENDED_ACTION_FEATURE_NAMES:
      key = f"{prefix}_{name}_sum"
      if key in summary:
        summary[f"{prefix}_{name}_mean_on_disagreements"] = (
            float(summary[key]) / disagreements
        )
  return summary


def main():
  args = parse_args()
  device = torch.device("cpu")
  champion_model, champion_scale = load_model(args.champion, args.players, device)
  candidate_model, candidate_scale = load_model(args.candidate, args.players, device)
  lineups = build_lineups(args.players, args.candidate_name, args.opponent)
  if not lineups:
    raise ValueError(f"No lineups generated for --players={args.players}")
  aggregate = {
      "matches": 0,
      "candidate_decisions": 0,
      "candidate_disagreements": 0,
      "candidate_policy_prob_sum": 0.0,
      "champion_policy_prob_sum": 0.0,
      "candidate_total_sum": 0.0,
      "champion_total_sum": 0.0,
      "candidate_paradoxes_sum": 0,
      "champion_paradoxes_sum": 0,
  }
  records = []
  for deal_idx in range(args.paired_deals):
    initial_start = deal_idx % args.players
    for lineup in lineups:
      match_summary, match_records = play_mined_match(
          args,
          lineup,
          deal_idx,
          initial_start,
          champion_model,
          champion_scale,
          candidate_model,
          candidate_scale,
          device,
      )
      aggregate["matches"] += 1
      for key, value in match_summary.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
          if key in ("candidate_total", "champion_total"):
            aggregate[f"{key}_sum"] += float(value)
          elif key in ("candidate_paradoxes", "champion_paradoxes"):
            aggregate[f"{key}_sum"] += int(value)
          else:
            aggregate[key] = aggregate.get(key, 0.0) + value
      if len(records) < args.max_records:
        remaining = args.max_records - len(records)
        records.extend(match_records[:remaining])
  aggregate = finalize_summary(aggregate)
  matches = int(aggregate["matches"])
  if matches <= 0:
    raise RuntimeError("Gate miner did not play any matches")
  aggregate["candidate_avg_total"] = aggregate["candidate_total_sum"] / matches
  aggregate["champion_avg_total"] = aggregate["champion_total_sum"] / matches
  aggregate["candidate_paradoxes_per_match"] = (
      aggregate["candidate_paradoxes_sum"] / matches
  )
  aggregate["champion_paradoxes_per_match"] = (
      aggregate["champion_paradoxes_sum"] / matches
  )
  result = {
      "champion": args.champion,
      "candidate": args.candidate,
      "candidate_name": args.candidate_name,
      "opponent": args.opponent,
      "paired_deals": args.paired_deals,
      "summary": aggregate,
      "records": records,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
