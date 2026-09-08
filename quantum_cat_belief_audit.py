#!/usr/bin/env python3
"""Audit Cat in the Box hidden-hand belief samplers against known self-play truth."""

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
from quantum_cat_alphazero_torch import load_model_payload
from quantum_cat_alphazero_torch import model_policy_value
from quantum_cat_alphazero_torch import policy_weighted_belief_states
from quantum_cat_alphazero_torch import sample_action
from quantum_cat_full_match_elo import _model_args


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--states", type=int, default=96)
  parser.add_argument("--max-games", type=int, default=64)
  parser.add_argument("--samples", type=int, default=24)
  parser.add_argument("--candidates", type=int, default=48)
  parser.add_argument("--temperature", type=float, default=0.35)
  parser.add_argument("--seed", type=int, default=20260604)
  parser.add_argument(
      "--phases",
      default="prediction,play",
      help="Comma-separated phases to audit: discard,prediction,play.",
  )
  parser.add_argument(
      "--ref-mixes",
      default=(
          "actual;"
          "uniform;"
          "model:1.0;"
          "model:0.65,heuristic:0.25,uniform:0.10"
      ),
      help=(
          "Semicolon-separated sampler specs. Use 'actual' for the true "
          "simulator state as an oracle sanity check, 'uniform' for plain "
          "information-state resampling; other values are policy-weighted "
          "reference mixtures."
      ),
  )
  parser.add_argument("--policy-temperature", type=float, default=1.0)
  parser.add_argument("--uniform-mix", type=float, default=0.15)
  parser.add_argument(
      "--ref-mix-by-phase",
      default="",
      help=(
          "Optional semicolon-separated phase overrides for policy-weighted "
          "samplers, e.g. "
          "prediction=model:0.65,heuristic:0.25,uniform:0.10;play=model:1.0."
      ),
  )
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument("--out", default="az_runs/belief_audit_latest.json")
  return parser.parse_args()


def phase_name(state):
  return {
      1: "discard",
      2: "prediction",
      3: "play",
  }.get(int(getattr(state, "_phase", -1)), "other")


def hidden_players(state, player):
  return [p for p in range(int(state._num_players)) if p != player]


def hand_rank_nll(true_counts, sampled_counts):
  total_cards = float(np.sum(true_counts))
  if total_cards <= 0:
    return None
  pred_probs = np.asarray(sampled_counts, dtype=np.float64) / total_cards
  pred_probs = np.clip(pred_probs, 1e-9, 1.0)
  return float(-np.sum(np.asarray(true_counts) * np.log(pred_probs)) / total_cards)


def discard_true_prob(state, samples, player):
  probs = []
  for p in hidden_players(state, player):
    true_rank = int(state._discarded_cards[p])
    if true_rank <= 0:
      continue
    hits = 0
    for sampled in samples:
      hits += int(int(sampled._discarded_cards[p]) == true_rank)
    probs.append(hits / float(max(1, len(samples))))
  if not probs:
    return None
  return float(np.mean(probs))


def score_samples(state, samples, player):
  if not samples:
    return {
        "hand_mae": None,
        "hand_rmse": None,
        "hand_rank_nll": None,
        "discard_true_prob": None,
        "exact_hidden_hand_sample_rate": None,
  }
  true_parts = []
  pred_parts = []
  exact = 0
  for sampled in samples:
    exact += int(
        all(
            np.array_equal(sampled._hands[p], state._hands[p])
            for p in hidden_players(state, player)
        )
    )
  nlls = []
  for p in hidden_players(state, player):
    true_counts = np.asarray(state._hands[p], dtype=np.float64)
    sampled_counts = np.mean(
        np.stack([sampled._hands[p] for sampled in samples]).astype(np.float64),
        axis=0,
    )
    true_parts.append(true_counts)
    pred_parts.append(sampled_counts)
    nll = hand_rank_nll(true_counts, sampled_counts)
    if nll is not None:
      nlls.append(nll)
  true_vec = np.concatenate(true_parts)
  pred_vec = np.concatenate(pred_parts)
  diff = pred_vec - true_vec
  return {
      "hand_mae": float(np.mean(np.abs(diff))),
      "hand_rmse": float(math.sqrt(np.mean(diff * diff))),
      "hand_rank_nll": float(np.mean(nlls)) if nlls else None,
      "discard_true_prob": discard_true_prob(state, samples, player),
      "exact_hidden_hand_sample_rate": exact / float(len(samples)),
  }


def belief_args(args, mix):
  return SimpleNamespace(
      counterfactual_belief_samples=int(args.samples),
      counterfactual_belief_candidates=max(int(args.samples), int(args.candidates)),
      counterfactual_belief_policy_temperature=float(args.policy_temperature),
      counterfactual_belief_uniform_mix=float(args.uniform_mix),
      counterfactual_belief_ref_policy_mix=mix,
      counterfactual_belief_ref_policy_mix_by_phase=str(args.ref_mix_by_phase),
      counterfactual_belief_logprob_floor=1e-6,
      value_scale=20.0,
  )


def sample_belief_states(state, player, args, model, model_args, device, spec):
  failures = 0
  if spec == "actual":
    return [state.clone() for _ in range(int(args.samples))], failures, None
  if spec == "uniform":
    samples = []
    first_error = None
    for _ in range(int(args.samples)):
      try:
        samples.append(state.resample_from_infostate(player, np.random))
      except Exception as exc:
        failures += 1
        if first_error is None:
          first_error = f"{type(exc).__name__}: {exc}"
        samples.append(state.clone())
    return samples, failures, first_error
  bargs = belief_args(args, spec)
  bargs.value_scale = model_args.value_scale
  samples = policy_weighted_belief_states(state, player, bargs, model, device)
  return samples, failures, None


def collect_states(game, model, model_args, args, device):
  wanted = {item.strip() for item in args.phases.split(",") if item.strip()}
  states = []
  games = 0
  while len(states) < args.states and games < args.max_games:
    state = game.new_initial_state()
    games += 1
    ply = 0
    while not state.is_terminal() and len(states) < args.states:
      if state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state.apply_action(int(np.random.choice(actions, p=probs)))
        continue
      player = int(state.current_player())
      if phase_name(state) in wanted:
        states.append(state.clone())
      policy, _ = model_policy_value(
          model,
          state,
          player,
          state.num_distinct_actions(),
          model_args.value_scale,
          device,
      )
      temp = args.temperature if ply < 20 else 0.0
      action = sample_action(policy, state.legal_actions(player), temp)
      state.apply_action(action)
      ply += 1
  return states, games


def summarize_flat(rows, specs):
  summary = {}
  for spec in specs:
    spec_rows = [row["samplers"][spec] for row in rows]
    summary[spec] = {}
    for key in (
        "hand_mae",
        "hand_rmse",
        "hand_rank_nll",
        "discard_true_prob",
        "exact_hidden_hand_sample_rate",
        "resample_failures",
    ):
      vals = [row[key] for row in spec_rows if row[key] is not None]
      summary[spec][key] = float(np.mean(vals)) if vals else None
  return summary


def summarize(rows, specs):
  summary = summarize_flat(rows, specs)
  by_phase = {}
  for phase in sorted({row["phase"] for row in rows}):
    phase_rows = [row for row in rows if row["phase"] == phase]
    by_phase[phase] = summarize_flat(phase_rows, specs)
  return summary, by_phase


def main():
  args = parse_args()
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model_args = _model_args(args.players)
  game = pyspiel.load_game("python_quantum_cat", {"players": args.players})
  model, _, saved_args = load_model_payload(args.checkpoint, game, model_args, device)
  model.eval()
  model_args.value_scale = saved_args.get("value_scale", model_args.value_scale)
  model_args.match_context = saved_args.get("match_context", False)
  game = pyspiel.load_game(
      "python_quantum_cat",
      {"players": args.players, "match_context": int(model_args.match_context)},
  )

  states, games = collect_states(game, model, model_args, args, device)
  specs = [item.strip() for item in args.ref_mixes.split(";") if item.strip()]
  rows = []
  for idx, state in enumerate(states):
    if idx and idx % 16 == 0:
      print(f"audited {idx}/{len(states)} states", flush=True)
    player = int(state.current_player())
    row = {
        "index": idx,
        "phase": phase_name(state),
        "player": player,
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "samplers": {},
    }
    for spec in specs:
      samples, failures, first_error = sample_belief_states(
          state, player, args, model, model_args, device, spec
      )
      row["samplers"][spec] = score_samples(state, samples, player)
      row["samplers"][spec]["resample_failures"] = int(failures)
      if first_error:
        row["samplers"][spec]["first_resample_error"] = first_error
    rows.append(row)

  summary, by_phase = summarize(rows, specs)
  result = {
      "checkpoint": args.checkpoint,
      "players": args.players,
      "seed": args.seed,
      "states_requested": args.states,
      "states_collected": len(states),
      "games_used": games,
      "samples": args.samples,
      "candidates": args.candidates,
      "ref_mixes": specs,
      "ref_mix_by_phase": args.ref_mix_by_phase,
      "summary": summary,
      "by_phase": by_phase,
      "rows": rows,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({k: result[k] for k in result if k != "rows"}, indent=2))


if __name__ == "__main__":
  main()
