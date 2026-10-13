#!/usr/bin/env python3
"""Census champion decisions to estimate tactical-correction opportunity.

This script is deliberately cheaper than a Q-teacher export. It rolls the
current policy through full matches and counts how often focused state slices
appear, so we can decide whether a correction niche is worth labeling/training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import (
    load_model_payload,
    model_policy_value,
    phase_name_for_state,
)
from quantum_cat_label_oracle_audit import action_string, policy_uncertainty_stats


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=200)
  parser.add_argument("--seed", type=int, default=20262197)
  parser.add_argument("--match-context", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--phases", default="play")
  parser.add_argument("--state-min-legal-count", type=int, default=3)
  parser.add_argument("--state-max-legal-count", type=int, default=3)
  parser.add_argument("--state-min-trick-number", type=int, default=4)
  parser.add_argument("--state-min-match-round", type=int, default=2)
  parser.add_argument("--sample-limit", type=int, default=12)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument("--out", required=True)
  return parser.parse_args()


def _device(name):
  if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
    return torch.device("mps")
  return torch.device("cpu")


def _shape_args(args):
  return SimpleNamespace(
      players=args.players,
      arch="mlp",
      width=256,
      depth=3,
      value_scale=20.0,
      match_context=bool(args.match_context),
  )


def _load_model(args, device):
  game = _make_game(args, start_player=0)
  model, _, saved_args = load_model_payload(
      args.checkpoint, game, _shape_args(args), device
  )
  model.eval()
  model_args = _shape_args(args)
  model_args.value_scale = float(saved_args.get("value_scale", model_args.value_scale))
  model_args.arch = saved_args.get("arch", model_args.arch)
  model_args.match_context = bool(saved_args.get("match_context", args.match_context))
  return model, model_args


def _make_game(args, start_player):
  return pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": int(args.players),
          "start_player": int(start_player),
          "match_context": int(bool(args.match_context)),
      },
  )


def _state_matches_focus(state, legal, args, wanted_phases):
  phase = phase_name_for_state(state)
  if phase not in wanted_phases:
    return False
  min_legal = int(args.state_min_legal_count)
  max_legal = int(args.state_max_legal_count)
  if min_legal > 0 and len(legal) < min_legal:
    return False
  if max_legal > 0 and len(legal) > max_legal:
    return False
  if int(args.state_min_trick_number) >= 0:
    if int(getattr(state, "_trick_number", -1)) < int(args.state_min_trick_number):
      return False
  if int(args.state_min_match_round) >= 0:
    if int(getattr(state, "_match_round", -1)) < int(args.state_min_match_round):
      return False
  return True


def _counter_key(value):
  return str(int(value))


def _round_score(state):
  raw_scores = (
      state.raw_round_scores()
      if hasattr(state, "raw_round_scores")
      else state.returns()
  )
  return np.array(raw_scores, dtype=np.float32)


def _mean_or_none(values):
  return round(float(mean(values)), 6) if values else None


def _summarize_counter(counter):
  return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def _rate(num, denom):
  return round(float(num) / float(denom), 6) if denom else None


def main():
  args = parse_args()
  if args.players < 2 or args.players > 5:
    raise ValueError("--players must be 2..5")
  np.random.seed(int(args.seed) % (2**32 - 1))
  torch.manual_seed(int(args.seed))
  device = _device(args.device)
  model, model_args = _load_model(args, device)
  wanted_phases = {part.strip() for part in args.phases.split(",") if part.strip()}
  if not wanted_phases:
    wanted_phases = {"discard", "prediction", "play"}

  started = time.perf_counter()
  counts = Counter()
  phase_counts = Counter()
  legal_counts_by_phase = defaultdict(Counter)
  focus_by_player = Counter()
  focus_by_round = Counter()
  focus_by_trick = Counter()
  focus_by_start = Counter()
  focus_action_counts = Counter()
  focus_policy_top_probs = []
  focus_policy_entropies = []
  play_policy_top_probs = []
  play_policy_entropies = []
  match_totals = []
  match_paradoxes = []
  samples = []

  for match_idx in range(int(args.matches)):
    match_seed = int(args.seed) + 100000 + match_idx * 37
    rng = np.random.default_rng(match_seed % (2**32 - 1))
    initial_start = match_idx % int(args.players)
    totals = np.zeros(int(args.players), dtype=np.float32)
    paradoxes = np.zeros(int(args.players), dtype=np.int32)
    for round_idx in range(int(args.players)):
      start_player = (initial_start + round_idx) % int(args.players)
      game = _make_game(args, start_player)
      state = game.new_initial_state()
      if args.match_context:
        state.set_match_context(np.copy(totals), round_idx)
      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(int(rng.choice(actions, p=probs)))
          continue

        player = int(state.current_player())
        phase = phase_name_for_state(state)
        legal = list(state.legal_actions(player))
        counts["decisions"] += 1
        phase_counts[phase] += 1
        legal_counts_by_phase[phase][_counter_key(len(legal))] += 1
        if len(legal) <= 1:
          counts["forced_actions"] += 1
        else:
          counts["meaningful_decisions"] += 1

        policy, _ = model_policy_value(
            model,
            state,
            player,
            state.num_distinct_actions(),
            model_args.value_scale,
            device,
        )
        policy_action = int(max(legal, key=lambda action: policy[action]))
        if phase == "play":
          counts["play_decisions"] += 1
          uncertainty = policy_uncertainty_stats(policy, legal)
          play_policy_top_probs.append(float(uncertainty["policy_top_prob"]))
          play_policy_entropies.append(float(uncertainty["policy_entropy_norm"]))

        if _state_matches_focus(state, legal, args, wanted_phases):
          counts["focused_opportunities"] += 1
          focus_by_player[_counter_key(player)] += 1
          focus_by_round[_counter_key(getattr(state, "_match_round", -1))] += 1
          focus_by_trick[_counter_key(getattr(state, "_trick_number", -1))] += 1
          focus_by_start[_counter_key(start_player)] += 1
          focus_action_counts[action_string(state, player, policy_action)] += 1
          uncertainty = policy_uncertainty_stats(policy, legal)
          focus_policy_top_probs.append(float(uncertainty["policy_top_prob"]))
          focus_policy_entropies.append(float(uncertainty["policy_entropy_norm"]))
          if len(samples) < int(args.sample_limit):
            samples.append({
                "match_index": int(match_idx),
                "round_index": int(round_idx),
                "start_player": int(start_player),
                "player": int(player),
                "match_round": int(getattr(state, "_match_round", -1)),
                "trick_number": int(getattr(state, "_trick_number", -1)),
                "legal_count": int(len(legal)),
                "policy_action": int(policy_action),
                "policy_action_string": action_string(state, player, policy_action),
                "policy_top_prob": round(float(uncertainty["policy_top_prob"]), 6),
                "policy_entropy_norm": round(
                    float(uncertainty["policy_entropy_norm"]), 6
                ),
                "legal_action_strings": [
                    action_string(state, player, int(action)) for action in legal
                ],
            })
        state.apply_action(policy_action)

      round_scores = _round_score(state)
      totals = totals + round_scores
      paradoxes += np.array(
          getattr(state, "_has_paradoxed", [False] * int(args.players)),
          dtype=np.int32,
      )
    match_totals.append([float(x) for x in totals])
    match_paradoxes.append([int(x) for x in paradoxes])

  elapsed = time.perf_counter() - started
  focused = int(counts["focused_opportunities"])
  result = {
      "checkpoint": args.checkpoint,
      "players": int(args.players),
      "matches": int(args.matches),
      "rounds": int(args.matches) * int(args.players),
      "seed": int(args.seed),
      "elapsed_sec": round(float(elapsed), 3),
      "match_context": bool(args.match_context),
      "focus_predicate": {
          "phases": sorted(wanted_phases),
          "state_min_legal_count": int(args.state_min_legal_count),
          "state_max_legal_count": int(args.state_max_legal_count),
          "state_min_trick_number": int(args.state_min_trick_number),
          "state_min_match_round": int(args.state_min_match_round),
      },
      "counts": {str(k): int(v) for k, v in sorted(counts.items())},
      "rates": {
          "focused_per_match": _rate(focused, int(args.matches)),
          "focused_per_round": _rate(focused, int(args.matches) * int(args.players)),
          "focused_per_decision": _rate(focused, counts["decisions"]),
          "focused_per_meaningful_decision": _rate(
              focused, counts["meaningful_decisions"]
          ),
          "focused_per_play_decision": _rate(focused, counts["play_decisions"]),
          "forced_action_rate": _rate(counts["forced_actions"], counts["decisions"]),
      },
      "phase_counts": _summarize_counter(phase_counts),
      "legal_counts_by_phase": {
          phase: _summarize_counter(counter)
          for phase, counter in sorted(legal_counts_by_phase.items())
      },
      "focused_by_player": _summarize_counter(focus_by_player),
      "focused_by_match_round": _summarize_counter(focus_by_round),
      "focused_by_trick_number": _summarize_counter(focus_by_trick),
      "focused_by_start_player": _summarize_counter(focus_by_start),
      "focused_policy_action_counts": _summarize_counter(focus_action_counts),
      "policy_uncertainty": {
          "play_top_prob_mean": _mean_or_none(play_policy_top_probs),
          "play_entropy_norm_mean": _mean_or_none(play_policy_entropies),
          "focused_top_prob_mean": _mean_or_none(focus_policy_top_probs),
          "focused_entropy_norm_mean": _mean_or_none(focus_policy_entropies),
      },
      "match_score_summary": {
          "avg_total_by_player": [
              round(float(np.mean([row[player] for row in match_totals])), 6)
              for player in range(int(args.players))
          ],
          "avg_paradoxes_by_player": [
              round(float(np.mean([row[player] for row in match_paradoxes])), 6)
              for player in range(int(args.players))
          ],
      },
      "samples": samples,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
