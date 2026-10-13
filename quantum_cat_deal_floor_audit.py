"""Estimate initial-deal liveness floors for Quantum Cat.

This is a diagnostic for the homogeneous paradox-rate gate. It samples fresh
deals, stops before the first avoidable decision, and asks whether any seat is
already negative-margin under the exact own-hand liveness oracle, even after
its best possible discard.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyspiel

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
import quantum_cat_liveness_oracle as oracle


def _ci95(successes, total):
  if total <= 0:
    return [0.0, 0.0]
  z = 1.96
  p = successes / total
  denom = 1.0 + z * z / total
  center = (p + z * z / (2.0 * total)) / denom
  half = z * ((p * (1.0 - p) / total + z * z / (4.0 * total * total)) ** 0.5) / denom
  return [max(0.0, center - half), min(1.0, center + half)]


def _deal_to_first_decision(game, rng):
  state = game.new_initial_state()
  while state.is_chance_node():
    outcomes = state.chance_outcomes()
    if not outcomes:
      break
    action = int(outcomes[int(rng.integers(len(outcomes)))][0])
    state.apply_action(action)
  return state


def _seat_summary(state, player):
  view = oracle.discard_view_from_state(state, player)
  raw_margin = int(view.margin())
  legal_discards = state.legal_actions(player)
  post_discard_margins = [
      int(oracle.post_discard_margin(view, int(action)))
      for action in legal_discards
  ]
  best_post_discard_margin = max(post_discard_margins)
  return {
      "player": int(player),
      "raw_margin": raw_margin,
      "best_post_discard_margin": int(best_post_discard_margin),
      "legal_discard_count": int(len(legal_discards)),
  }


def run_audit(players, deals, seed, sample_limit):
  game = pyspiel.load_game(
      "python_quantum_cat",
      {"players": int(players), "match_context": 1},
  )
  rng = np.random.default_rng(int(seed))
  seat_count = int(players)

  raw_negative_deals = 0
  best_discard_negative_deals = 0
  raw_negative_seats = 0
  best_discard_negative_seats = 0
  min_raw_margins = []
  min_best_discard_margins = []
  samples = []

  for deal_idx in range(int(deals)):
    state = _deal_to_first_decision(game, rng)
    if int(state._phase) != 1:
      raise RuntimeError(f"Expected discard phase after deal, got {state._phase}")
    seats = [_seat_summary(state, p) for p in range(seat_count)]
    raw_neg = sum(1 for s in seats if s["raw_margin"] < 0)
    best_neg = sum(1 for s in seats if s["best_post_discard_margin"] < 0)
    raw_negative_seats += raw_neg
    best_discard_negative_seats += best_neg
    raw_negative_deals += int(raw_neg > 0)
    best_discard_negative_deals += int(best_neg > 0)
    min_raw = min(s["raw_margin"] for s in seats)
    min_best = min(s["best_post_discard_margin"] for s in seats)
    min_raw_margins.append(min_raw)
    min_best_discard_margins.append(min_best)
    if (raw_neg > 0 or best_neg > 0) and len(samples) < sample_limit:
      samples.append({
          "deal_index": deal_idx,
          "min_raw_margin": int(min_raw),
          "min_best_post_discard_margin": int(min_best),
          "seats": seats,
          "hands": [
              [int(x) for x in state._hands[p]]
              for p in range(seat_count)
          ],
      })

  total_seats = int(deals) * seat_count
  return {
      "players": int(players),
      "deals": int(deals),
      "seed": int(seed),
      "definition": {
          "raw_negative_deal_rate":
              "fraction of deals where any seat has margin < 0 before discard",
          "best_discard_negative_deal_rate":
              "fraction of deals where any seat still has margin < 0 after its best legal discard",
          "margin":
              "exact own-hand adversarial cell-deletion margin from quantum_cat_liveness_oracle",
      },
      "raw_negative_deals": int(raw_negative_deals),
      "raw_negative_deal_rate": raw_negative_deals / int(deals),
      "raw_negative_deal_rate_ci95": _ci95(raw_negative_deals, int(deals)),
      "best_discard_negative_deals": int(best_discard_negative_deals),
      "best_discard_negative_deal_rate":
          best_discard_negative_deals / int(deals),
      "best_discard_negative_deal_rate_ci95":
          _ci95(best_discard_negative_deals, int(deals)),
      "raw_negative_seats": int(raw_negative_seats),
      "raw_negative_seat_rate": raw_negative_seats / total_seats,
      "best_discard_negative_seats": int(best_discard_negative_seats),
      "best_discard_negative_seat_rate":
          best_discard_negative_seats / total_seats,
      "min_raw_margin_mean": float(np.mean(min_raw_margins)),
      "min_raw_margin_p05": float(np.percentile(min_raw_margins, 5)),
      "min_raw_margin_p50": float(np.percentile(min_raw_margins, 50)),
      "min_best_discard_margin_mean":
          float(np.mean(min_best_discard_margins)),
      "min_best_discard_margin_p05":
          float(np.percentile(min_best_discard_margins, 5)),
      "min_best_discard_margin_p50":
          float(np.percentile(min_best_discard_margins, 50)),
      "samples": samples,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--deals", type=int, default=10000)
  parser.add_argument("--seed", type=int, default=20260612)
  parser.add_argument("--sample-limit", type=int, default=12)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  result = run_audit(
      players=args.players,
      deals=args.deals,
      seed=args.seed,
      sample_limit=args.sample_limit,
  )
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  digest = hashlib.sha256(out.read_bytes()).hexdigest()
  print(json.dumps({
      "out": str(out),
      "sha256": digest,
      "players": result["players"],
      "deals": result["deals"],
      "best_discard_negative_deals":
          result["best_discard_negative_deals"],
      "best_discard_negative_deal_rate":
          result["best_discard_negative_deal_rate"],
      "best_discard_negative_deal_rate_ci95":
          result["best_discard_negative_deal_rate_ci95"],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
