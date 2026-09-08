#!/usr/bin/env python3
"""Validate iOS human-match telemetry and compute a provenance-aware Elo report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


VERIFIED_LIVE_INPUT_CONTRACT = "python-observation76-action214-v1"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+", help="Telemetry .json or .jsonl files")
  parser.add_argument("--out", default="experiments/human_telemetry_elo.json")
  parser.add_argument("--k-factor", type=float, default=24.0)
  parser.add_argument("--bootstrap-samples", type=int, default=2000)
  parser.add_argument("--seed", type=int, default=20260721)
  parser.add_argument(
      "--anchor-pool",
      default="experiments/human_elo_anchor_pool.json",
      help=(
          "Frozen, table-size-specific Elo anchor configuration. Ratings in "
          "this file are never updated by telemetry matches."
      ),
  )
  parser.add_argument(
      "--require-verified-model-contract",
      action="store_true",
      help="Fail if a Core ML participant used an unverified live input encoder.",
  )
  return parser.parse_args()


def load_anchor_pool(path: Path) -> dict:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if int(payload.get("schemaVersion", -1)) != 1:
    raise ValueError(f"Unsupported Elo anchor-pool schema: {path}")
  player_counts = payload.get("playerCounts")
  if not isinstance(player_counts, dict):
    raise ValueError(f"Elo anchor pool has no playerCounts mapping: {path}")
  for player_count, entry in player_counts.items():
    if str(player_count) not in {"2", "3", "4", "5"}:
      raise ValueError(f"Invalid Elo anchor player count: {player_count}")
    ratings = entry.get("ratings") if isinstance(entry, dict) else None
    if not isinstance(ratings, dict) or not ratings:
      raise ValueError(f"Elo anchor pool {player_count}p has no ratings")
    for participant, rating in ratings.items():
      if not str(participant) or not math.isfinite(float(rating)):
        raise ValueError(f"Invalid Elo anchor rating for {participant}")
  return payload


def _read_objects(path: Path) -> list[dict]:
  raw = path.read_text(encoding="utf-8")
  if path.suffix == ".jsonl":
    return [json.loads(line) for line in raw.splitlines() if line.strip()]
  payload = json.loads(raw)
  return payload if isinstance(payload, list) else [payload]


def load_matches(paths: Iterable[Path]) -> tuple[list[dict], list[dict]]:
  matches = []
  sources = []
  seen_ids = set()
  for path in paths:
    data = path.read_bytes()
    sources.append({
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
    })
    for match in _read_objects(path):
      validate_match(match)
      match_id = str(match["id"])
      if match_id in seen_ids:
        continue
      seen_ids.add(match_id)
      matches.append(match)
  matches.sort(key=lambda item: (str(item.get("completedAt", "")), str(item["id"])))
  if not matches:
    raise ValueError("No completed telemetry matches found")
  return matches, sources


def _walk_keys(value):
  if isinstance(value, dict):
    for key, child in value.items():
      yield str(key)
      yield from _walk_keys(child)
  elif isinstance(value, list):
    for child in value:
      yield from _walk_keys(child)


def validate_match(match: dict) -> None:
  if int(match.get("schemaVersion", -1)) != 1:
    raise ValueError(f"Unsupported telemetry schema for {match.get('id')}")
  if not match.get("completedAt"):
    raise ValueError(f"Telemetry match {match.get('id')} is incomplete")
  seats = sorted(match.get("seats", []), key=lambda seat: int(seat["index"]))
  if not 2 <= len(seats) <= 5:
    raise ValueError(f"Telemetry match {match.get('id')} must have 2..5 seats")
  if [int(seat["index"]) for seat in seats] != list(range(len(seats))):
    raise ValueError(f"Telemetry match {match.get('id')} has invalid seat indices")
  if any(not seat.get("participantID") for seat in seats):
    raise ValueError(f"Telemetry match {match.get('id')} has an unnamed participant")
  target_rounds = int(match.get("targetRounds", 0))
  rounds = match.get("rounds", [])
  if target_rounds != len(seats) or len(rounds) != target_rounds:
    raise ValueError(
        f"Telemetry match {match.get('id')} must contain one round per seat"
    )
  for round_record in rounds:
    scores = round_record.get("scores", [])
    paradoxes = round_record.get("paradoxes", [])
    if len(scores) != len(seats) or len(paradoxes) != len(seats):
      raise ValueError(f"Telemetry match {match.get('id')} has malformed results")
    for move in round_record.get("moves", []):
      legal = move.get("legalActions", [])
      if int(move.get("legalActionCount", -1)) != len(legal):
        raise ValueError(f"Telemetry match {match.get('id')} has a legal-count mismatch")
      if move.get("action") not in legal:
        raise ValueError(f"Telemetry match {match.get('id')} records an illegal action")
  # Completed telemetry is evidence about observable play, not a simulator
  # dump.  Face-down discards are hidden information too, even though the rank
  # becomes available inside the engine once the round is over.
  forbidden = {
      "hand",
      "hands",
      "deck",
      "discardedhand",
      "discardedhands",
      "discardedrank",
      "discardedranks",
  }
  leaked = forbidden.intersection(key.lower() for key in _walk_keys(match))
  if leaked:
    raise ValueError(
        f"Telemetry match {match.get('id')} contains hidden-state keys: {sorted(leaked)}"
    )


def match_scores(match: dict) -> list[float]:
  seats = match["seats"]
  totals = [0.0] * len(seats)
  for round_record in match["rounds"]:
    for seat, score in enumerate(round_record["scores"]):
      totals[seat] += float(score)
  return totals


def _elo_update(
    ratings: dict[str, float], participant_ids: list[str], scores: list[float],
    k_factor: float, anchored: set[str] | None = None,
) -> None:
  anchored = anchored or set()
  deltas = defaultdict(float)
  counts = defaultdict(int)
  for left in range(len(participant_ids)):
    for right in range(left + 1, len(participant_ids)):
      left_id = participant_ids[left]
      right_id = participant_ids[right]
      if left_id == right_id:
        continue
      expected = 1.0 / (
          1.0 + 10.0 ** ((ratings[right_id] - ratings[left_id]) / 400.0)
      )
      if scores[left] > scores[right]:
        actual = 1.0
      elif scores[left] < scores[right]:
        actual = 0.0
      else:
        actual = 0.5
      change = k_factor * (actual - expected)
      if left_id not in anchored:
        deltas[left_id] += change
        counts[left_id] += 1
      if right_id not in anchored:
        deltas[right_id] -= change
        counts[right_id] += 1
  for participant_id in counts:
    ratings[participant_id] += deltas[participant_id] / counts[participant_id]


def compute_ratings(
    matches: list[dict], k_factor: float,
    anchor_ratings: dict[str, float] | None = None,
) -> dict[str, float]:
  anchor_ratings = {
      str(participant): float(rating)
      for participant, rating in (anchor_ratings or {}).items()
  }
  participants = {
      str(seat["participantID"])
      for match in matches for seat in match["seats"]
  }
  ratings = {
      participant: anchor_ratings.get(participant, 1000.0)
      for participant in participants
  }
  for match in matches:
    seated = [str(seat["participantID"]) for seat in match["seats"]]
    _elo_update(
        ratings, seated, match_scores(match), k_factor,
        anchored=set(anchor_ratings),
    )
  return ratings


def _percentile(values: list[float], percentile: float) -> float:
  ordered = sorted(values)
  if not ordered:
    return math.nan
  position = (len(ordered) - 1) * percentile
  lower = int(math.floor(position))
  upper = int(math.ceil(position))
  if lower == upper:
    return ordered[lower]
  weight = position - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_rating_draws(
    matches: list[dict], participants: Iterable[str], k_factor: float,
    samples: int, seed: int,
    anchor_ratings: dict[str, float] | None = None,
) -> dict[str, list[float]]:
  participants = list(participants)
  anchor_ratings = anchor_ratings or {}
  if samples <= 0:
    return {participant: [] for participant in participants}
  rng = random.Random(seed)
  draws = {participant: [] for participant in participants}
  for _ in range(samples):
    sampled = [matches[rng.randrange(len(matches))] for _ in matches]
    ratings = compute_ratings(sampled, k_factor, anchor_ratings)
    for participant in participants:
      draws[participant].append(
          ratings.get(participant, anchor_ratings.get(participant, 1000.0))
      )
  return draws


def bootstrap_intervals(
    matches: list[dict], participants: Iterable[str], k_factor: float,
    samples: int, seed: int,
    anchor_ratings: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
  participants = list(participants)
  draws = bootstrap_rating_draws(
      matches, participants, k_factor, samples, seed, anchor_ratings
  )
  return {
      participant: (
          _percentile(draws.get(participant, []), 0.025),
          _percentile(draws.get(participant, []), 0.975),
      )
      for participant in participants
  }


def _anchor_connectivity(
    matches: list[dict], anchor_ratings: dict[str, float]
) -> set[str]:
  adjacency = defaultdict(set)
  present = set()
  for match in matches:
    participants = {str(seat["participantID"]) for seat in match["seats"]}
    present.update(participants)
    for participant in participants:
      adjacency[participant].update(participants - {participant})
  connected = set(present.intersection(anchor_ratings))
  frontier = list(connected)
  while frontier:
    participant = frontier.pop()
    for neighbor in adjacency[participant]:
      if neighbor not in connected:
        connected.add(neighbor)
        frontier.append(neighbor)
  return connected


def _table_report(
    matches: list[dict], sources: list[dict], k_factor: float,
    bootstrap_samples: int, seed: int,
    anchor_ratings: dict[str, float], anchor_pool_id: str | None,
) -> dict:
  ratings = compute_ratings(matches, k_factor, anchor_ratings)
  rating_draws = bootstrap_rating_draws(
      matches, ratings, k_factor, bootstrap_samples, seed, anchor_ratings
  )
  intervals = {
      participant: (
          _percentile(draws, 0.025), _percentile(draws, 0.975)
      )
      for participant, draws in rating_draws.items()
  }
  score_rows = defaultdict(list)
  appearances = defaultdict(int)
  paradox_rounds = defaultdict(int)
  round_appearances = defaultdict(int)
  human_decisions = defaultdict(int)
  human_agreements = defaultdict(int)
  unverified = set()
  human_match_count = 0
  for match in matches:
    human_match_count += int(any(
        seat.get("actor") == "human" for seat in match["seats"]
    ))
    scores = match_scores(match)
    for seat_index, seat in enumerate(match["seats"]):
      participant = str(seat["participantID"])
      score_rows[participant].append(scores[seat_index])
      appearances[participant] += 1
      contract = seat.get("liveInputContract")
      if seat.get("modelResource") and contract != VERIFIED_LIVE_INPUT_CONTRACT:
        unverified.add(f"{participant}:{contract or 'missing'}")
    for round_record in match["rounds"]:
      for seat_index, seat in enumerate(match["seats"]):
        participant = str(seat["participantID"])
        round_appearances[participant] += 1
        paradox_rounds[participant] += int(bool(round_record["paradoxes"][seat_index]))
      for move in round_record.get("moves", []):
        if move.get("actor") != "human":
          continue
        participant = str(match["seats"][int(move["player"])]["participantID"])
        if move.get("agreedWithChampion") is not None:
          human_decisions[participant] += 1
          human_agreements[participant] += int(bool(move["agreedWithChampion"]))

  connected = _anchor_connectivity(matches, anchor_ratings)
  human_participants = {
      str(seat["participantID"])
      for match in matches for seat in match["seats"]
      if seat.get("actor") == "human"
  }
  unanchored_humans = sorted(human_participants - connected)
  bot_participants = {
      str(seat["participantID"])
      for match in matches for seat in match["seats"]
      if seat.get("actor") == "bot"
  }
  confidence = (
      "measured" if human_match_count >= 200
      else "provisional" if human_match_count >= 100
      else "insufficient_data"
  )
  participants = {}
  for participant, rating in sorted(
      ratings.items(), key=lambda item: item[1], reverse=True
  ):
    low, high = intervals[participant]
    decisions = human_decisions[participant]
    participants[participant] = {
        "elo": round(rating, 2),
        "elo_bootstrap_ci95": [round(low, 2), round(high, 2)],
        "matches": appearances[participant],
        "rounds": round_appearances[participant],
        "average_match_score": round(mean(score_rows[participant]), 4),
        "paradox_round_rate": round(
            paradox_rounds[participant] / max(1, round_appearances[participant]), 4
        ),
        "champion_recommendation_decisions": decisions,
        "champion_recommendation_agreement_rate": (
            round(human_agreements[participant] / decisions, 4)
            if decisions else None
        ),
        "anchor_connected": participant in connected,
        "rating_is_frozen_anchor": participant in anchor_ratings,
    }
  human_weight_total = sum(appearances[item] for item in human_participants)
  human_cohort_elo = (
      sum(ratings[item] * appearances[item] for item in human_participants)
      / human_weight_total
      if human_weight_total else math.nan
  )
  cohort_draws = []
  if human_weight_total and rating_draws:
    draw_count = min(
        (len(rating_draws[item]) for item in human_participants), default=0
    )
    for draw_index in range(draw_count):
      cohort_draws.append(sum(
          rating_draws[item][draw_index] * appearances[item]
          for item in human_participants
      ) / human_weight_total)
  cohort_interval = [
      _percentile(cohort_draws, 0.025),
      _percentile(cohort_draws, 0.975),
  ]
  bot_vs_human = {}
  for participant in sorted(bot_participants):
    delta = ratings[participant] - human_cohort_elo
    delta_draws = [
        rating_draws[participant][index] - cohort_draws[index]
        for index in range(min(len(rating_draws[participant]), len(cohort_draws)))
    ]
    delta_interval = [
        _percentile(delta_draws, 0.025),
        _percentile(delta_draws, 0.975),
    ]
    bot_vs_human[participant] = {
        "elo_minus_human_cohort": round(delta, 2),
        "elo_difference_bootstrap_ci95": [
            round(delta_interval[0], 2), round(delta_interval[1], 2)
        ],
        "point_estimate_at_or_above_human": bool(delta >= 0.0),
        "ci95_supports_at_or_above_human": bool(delta_interval[0] >= 0.0),
        "anchor_connected": participant in connected,
    }
  if unverified:
    claim_status = "diagnostic_only_unverified_model_contract"
  elif human_match_count < 100:
    claim_status = "needs_100_human_matches"
  elif unanchored_humans:
    claim_status = "human_cohort_not_anchor_connected"
  elif not bot_vs_human:
    claim_status = "no_bot_human_comparison"
  elif any(
      row["anchor_connected"] and row["ci95_supports_at_or_above_human"]
      for row in bot_vs_human.values()
  ):
    claim_status = "bot_at_or_above_human_with_ci"
  else:
    claim_status = "no_bot_clears_human_ci"
  return {
      "player_count": len(matches[0]["seats"]),
      "match_count": len(matches),
      "human_match_count": human_match_count,
      "confidence": confidence,
      "k_factor": k_factor,
      "bootstrap_samples": bootstrap_samples,
      "seed": seed,
      "verified_live_input_contract": VERIFIED_LIVE_INPUT_CONTRACT,
      "anchor_pool_id": anchor_pool_id,
      "anchor_ratings": dict(sorted(anchor_ratings.items())),
      "anchor_evidence_status": (
          "anchored" if connected else "relative_only_no_anchor_match"
      ),
      "unanchored_human_participants": unanchored_humans,
      "unverified_model_contracts": sorted(unverified),
      "model_evidence_status": (
          "diagnostic_only" if unverified else "verified_contract"
      ),
      "human_comparison_claim_status": claim_status,
      "human_cohort": {
          "participants": sorted(human_participants),
          "participant_match_weight_total": human_weight_total,
          "elo": round(human_cohort_elo, 2),
          "elo_bootstrap_ci95": [
              round(cohort_interval[0], 2), round(cohort_interval[1], 2)
          ],
      },
      "bot_vs_human_cohort": bot_vs_human,
      "participants": participants,
      "sources": sources,
  }


def build_report(
    matches: list[dict], sources: list[dict], k_factor: float,
    bootstrap_samples: int, seed: int, anchor_pool: dict | None = None,
) -> dict:
  grouped = defaultdict(list)
  for match in matches:
    grouped[len(match["seats"])].append(match)
  player_count_reports = {}
  anchor_pool_id = (anchor_pool or {}).get("poolID")
  anchor_groups = (anchor_pool or {}).get("playerCounts", {})
  for player_count, table_matches in sorted(grouped.items()):
    entry = anchor_groups.get(str(player_count), {})
    anchor_ratings = {
        str(participant): float(rating)
        for participant, rating in entry.get("ratings", {}).items()
    }
    player_count_reports[str(player_count)] = _table_report(
        table_matches,
        sources,
        k_factor,
        bootstrap_samples,
        seed + player_count,
        anchor_ratings,
        anchor_pool_id,
    )
  report = {
      "schema_version": 2,
      "match_count": len(matches),
      "player_count_reports": player_count_reports,
      "anchor_pool_id": anchor_pool_id,
      "sources": sources,
  }
  # Preserve the convenient single-table top-level fields for existing report
  # consumers without ever mixing Elo observations across table sizes.
  if len(player_count_reports) == 1:
    only = next(iter(player_count_reports.values()))
    report.update({
        key: value for key, value in only.items()
        if key not in {"sources", "match_count"}
    })
  return report


def main() -> int:
  args = parse_args()
  matches, sources = load_matches([Path(path) for path in args.inputs])
  anchor_pool = load_anchor_pool(Path(args.anchor_pool))
  report = build_report(
      matches,
      sources,
      k_factor=args.k_factor,
      bootstrap_samples=args.bootstrap_samples,
      seed=args.seed,
      anchor_pool=anchor_pool,
  )
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  unverified = sorted({
      item
      for table in report["player_count_reports"].values()
      for item in table["unverified_model_contracts"]
  })
  if args.require_verified_model_contract and unverified:
    return 3
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
