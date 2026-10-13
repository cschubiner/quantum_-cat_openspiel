#!/usr/bin/env python3
"""Evaluate Cat in the Box bots with a multiplayer Elo ladder."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean
from time import time

import pyspiel

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_ai import make_bot, multiplayer_elo_update, play_game


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--games", type=int, default=300)
  parser.add_argument("--bots", nargs="+", default=[
      "random", "heuristic", "heuristic_adj2", "heuristic_target2"
  ])
  parser.add_argument("--k-factor", type=float, default=24.0)
  parser.add_argument("--seed", type=int, default=20260602)
  parser.add_argument("--out", default="experiments/elo_latest.json")
  return parser.parse_args()


def main():
  args = parse_args()
  if args.players < 2 or args.players > 5:
    raise ValueError("--players must be 2..5")
  if len(args.bots) < args.players:
    raise ValueError("Need at least as many bot names as players.")

  game = pyspiel.load_game("python_quantum_cat", {"players": args.players})
  ratings = {name: 1000.0 for name in args.bots}
  returns_by_bot = {name: [] for name in args.bots}
  games_by_bot = {name: 0 for name in args.bots}
  paradox_by_bot = {name: 0 for name in args.bots}
  lineups = list(itertools.combinations(args.bots, args.players))
  start = time()

  for game_idx in range(args.games):
    lineup = list(lineups[game_idx % len(lineups)])
    # Rotate seats so first-player/start-player artifacts do not masquerade as skill.
    shift = game_idx % args.players
    seated = lineup[shift:] + lineup[:shift]
    bots = [make_bot(name, seed=args.seed + game_idx * 17 + seat)
            for seat, name in enumerate(seated)]
    returns, state = play_game(game, bots, seed=args.seed + game_idx)
    multiplayer_elo_update(ratings, seated, returns, args.k_factor)
    for seat, name in enumerate(seated):
      returns_by_bot[name].append(float(returns[seat]))
      games_by_bot[name] += 1
      if state._has_paradoxed[seat]:
        paradox_by_bot[name] += 1

  elapsed = time() - start
  summary = {
      "players": args.players,
      "games": args.games,
      "k_factor": args.k_factor,
      "seed": args.seed,
      "elapsed_seconds": elapsed,
      "ratings": {k: round(v, 2) for k, v in sorted(
          ratings.items(), key=lambda item: item[1], reverse=True)},
      "stats": {
          name: {
              "games": games_by_bot[name],
              "avg_return": round(mean(returns_by_bot[name]), 4)
              if returns_by_bot[name] else None,
              "paradox_rate": round(paradox_by_bot[name] / games_by_bot[name], 4)
              if games_by_bot[name] else None,
          }
          for name in args.bots
      },
  }

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
