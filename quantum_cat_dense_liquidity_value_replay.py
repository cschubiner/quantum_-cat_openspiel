"""Generate dense round-survival/liquidity value replay from full matches."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from quantum_cat_ai import make_bot
from quantum_cat_alphazero_torch import (
    action_feature_matrix,
    make_game,
    model_policy_value,
    save_replay,
    terminal_paradox_target,
)
from quantum_cat_full_match_elo import load_neural


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=20)
  parser.add_argument("--seed", type=int, default=20268650)
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument("--no-match-context", dest="match_context", action="store_false")
  parser.set_defaults(match_context=True)
  parser.add_argument("--bot", default="")
  parser.add_argument("--checkpoint", default="")
  parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
  parser.add_argument("--source-temperature", type=float, default=0.0)
  parser.add_argument("--paradox-half-life", type=float, default=6.0)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--progress-interval", type=int, default=0)
  return parser.parse_args()


def legal_mask(num_actions: int, legal: list[int]) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=np.float32)
  for action in legal:
    mask[int(action)] = 1.0
  return mask


def one_hot_policy(num_actions: int, action: int) -> np.ndarray:
  policy = np.zeros(num_actions, dtype=np.float32)
  if 0 <= int(action) < num_actions:
    policy[int(action)] = 1.0
  return policy


def round_raw_scores(state) -> np.ndarray:
  if hasattr(state, "raw_round_scores"):
    return np.array(state.raw_round_scores(), dtype=np.float32)
  return np.array(state.returns(), dtype=np.float32)


def sample_policy(
    policy: np.ndarray,
    legal: list[int],
    temperature: float,
    rng: np.random.Generator,
) -> int:
  if len(legal) == 1:
    return int(legal[0])
  if temperature <= 1e-9:
    return int(max(legal, key=lambda action: float(policy[int(action)])))
  probs = np.array([max(float(policy[int(action)]), 1e-12) for action in legal])
  logits = np.log(probs) / max(1e-6, float(temperature))
  logits -= float(np.max(logits))
  weights = np.exp(logits)
  total = float(np.sum(weights))
  if not math.isfinite(total) or total <= 0.0:
    weights = np.ones_like(weights)
    total = float(np.sum(weights))
  weights /= total
  return int(rng.choice(np.array(legal, dtype=np.int64), p=weights))


def survival_probability(
    row_idx: int,
    first_paradox_idx: int | None,
    legal_counts: list[int],
    half_life: float,
) -> tuple[float, int, int | None]:
  future = legal_counts[row_idx:]
  min_future_legal = int(min(future)) if future else 0
  if first_paradox_idx is None:
    prob = 0.92
    if min_future_legal <= 1:
      prob -= 0.18
    elif min_future_legal == 2:
      prob -= 0.08
    return float(np.clip(prob, 0.55, 0.98)), min_future_legal, None

  distance = max(0, int(first_paradox_idx) - int(row_idx))
  prob = 0.08 + 0.72 * (1.0 - math.exp(-distance / max(1e-6, half_life)))
  if min_future_legal <= 1:
    prob -= 0.18
  elif min_future_legal == 2:
    prob -= 0.08
  return float(np.clip(prob, 0.02, 0.86)), min_future_legal, distance


def source_policy_action(
    args: argparse.Namespace,
    state,
    player: int,
    num_actions: int,
    model,
    model_args,
    device,
    bots,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
  legal = [int(action) for action in state.legal_actions(player)]
  if model is not None:
    policy, _ = model_policy_value(
        model,
        state,
        player,
        num_actions,
        model_args.value_scale,
        device,
    )
    action = sample_policy(policy, legal, float(args.source_temperature), rng)
    return np.asarray(policy, dtype=np.float32), int(action)

  action = int(bots[player].step(state.clone(), player))
  if action not in legal:
    action = int(legal[0])
  return one_hot_policy(num_actions, action), action


def collect_round(
    args: argparse.Namespace,
    game,
    state,
    model,
    model_args,
    device,
    bots,
    rng: np.random.Generator,
) -> tuple[list[tuple], list[dict], object]:
  examples = []
  decisions = []
  num_actions = game.num_distinct_actions()
  while not state.is_terminal():
    if state.is_chance_node():
      actions, probs = zip(*state.chance_outcomes())
      state.apply_action(int(rng.choice(np.array(actions, dtype=np.int64), p=probs)))
      continue
    player = int(state.current_player())
    legal = [int(action) for action in state.legal_actions(player)]
    policy, action = source_policy_action(
        args, state, player, num_actions, model, model_args, device, bots, rng
    )
    obs = np.array(state.observation_tensor(player), dtype=np.float32)
    mask = legal_mask(num_actions, legal)
    features = action_feature_matrix(state, player, num_actions)
    before_paradox = [bool(value) for value in getattr(state, "_has_paradoxed", [])]
    state.apply_action(int(action))
    after_paradox = [bool(value) for value in getattr(state, "_has_paradoxed", [])]
    examples.append((
        obs,
        mask,
        policy,
        int(action),
        int(player),
        features,
    ))
    decisions.append({
        "player": int(player),
        "action": int(action),
        "phase": int(getattr(state, "_phase", -1)),
        "trick_number": int(getattr(state, "_trick_number", -1)),
        "legal_count": int(len(legal)),
        "before_paradox": before_paradox,
        "after_paradox": after_paradox,
        "any_after_paradox": bool(any(after_paradox)),
    })
  return examples, decisions, state


def rows_for_round(
    examples: list[tuple],
    decisions: list[dict],
    terminal_state,
    players: int,
    half_life: float,
) -> tuple[list[tuple], dict]:
  paradox_target = terminal_paradox_target(terminal_state, players)
  first_paradox_idx = None
  for idx, decision in enumerate(decisions):
    if bool(decision["any_after_paradox"]):
      first_paradox_idx = idx
      break
  legal_counts = [int(decision["legal_count"]) for decision in decisions]
  rows = []
  probs = []
  min_future_counts = []
  distances = []
  for idx, example in enumerate(examples):
    prob, min_future_legal, distance = survival_probability(
        idx, first_paradox_idx, legal_counts, half_life
    )
    value = np.full(players, 2.0 * prob - 1.0, dtype=np.float32)
    obs, mask, policy, action, player, features = example
    rows.append((
        obs,
        mask,
        policy,
        value,
        paradox_target,
        int(action),
        int(player),
        None,
        None,
        None,
        None,
        features,
        paradox_target,
    ))
    probs.append(float(prob))
    min_future_counts.append(int(min_future_legal))
    if distance is not None:
      distances.append(int(distance))
  summary = {
      "decision_rows": int(len(examples)),
      "any_paradox": bool(np.any(paradox_target > 0.5)),
      "first_paradox_idx": (
          None if first_paradox_idx is None else int(first_paradox_idx)
      ),
      "mean_survival_prob": float(np.mean(probs)) if probs else 0.0,
      "min_future_legal_min": int(min(min_future_counts)) if min_future_counts else 0,
      "distance_mean": float(np.mean(distances)) if distances else None,
      "paradoxed": [bool(value) for value in paradox_target],
  }
  return rows, summary


def main() -> None:
  args = parse_args()
  if not args.bot and not args.checkpoint:
    raise ValueError("Provide --bot or --checkpoint")
  started = time.perf_counter()
  rng = np.random.default_rng(int(args.seed))
  torch.manual_seed(int(args.seed))
  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device=mps requested but unavailable")
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model = None
  model_args = None
  if args.checkpoint:
    model, model_args = load_neural(args.checkpoint, int(args.players), device)
    if model is None or model_args is None:
      raise ValueError(f"Could not load checkpoint {args.checkpoint}")
  bots = [
      make_bot(args.bot, seed=int(args.seed) + seat * 17)
      for seat in range(int(args.players))
  ] if args.bot else []

  all_rows = []
  round_summaries = []
  match_totals = np.zeros(int(args.players), dtype=np.float32)
  for match_idx in range(int(args.matches)):
    for round_idx in range(int(args.players)):
      start_player = (
          int(args.initial_start) + int(match_idx) + int(round_idx)
      ) % int(args.players)
      game = make_game(args, start_player)
      state = game.new_initial_state()
      if bool(args.match_context):
        state.set_match_context(match_totals, round_idx)
      examples, decisions, terminal_state = collect_round(
          args, game, state, model, model_args, device, bots, rng
      )
      rows, summary = rows_for_round(
          examples,
          decisions,
          terminal_state,
          int(args.players),
          float(args.paradox_half_life),
      )
      all_rows.extend(rows)
      round_summaries.append({
          "match_idx": int(match_idx),
          "round_idx": int(round_idx),
          "start_player": int(start_player),
          **summary,
      })
      match_totals += round_raw_scores(terminal_state)
    if (
        int(args.progress_interval) > 0
        and (match_idx + 1) % int(args.progress_interval) == 0
    ):
      print(json.dumps({
          "completed_matches": int(match_idx + 1),
          "rows": int(len(all_rows)),
          "elapsed_sec": round(time.perf_counter() - started, 3),
      }), flush=True)

  total_rounds = len(round_summaries)
  paradox_rounds = sum(1 for row in round_summaries if row["any_paradox"])
  probs = np.array(
      [(np.asarray(row[3], dtype=np.float32)[0] + 1.0) / 2.0 for row in all_rows],
      dtype=np.float32,
  )
  metadata = {
      "kind": "dense_liquidity_round_survival_value",
      "players": int(args.players),
      "matches": int(args.matches),
      "bot": str(args.bot),
      "checkpoint": str(args.checkpoint),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "source_temperature": float(args.source_temperature),
      "paradox_half_life": float(args.paradox_half_life),
      "target": "dense_probability_from_distance_to_first_paradox_and_future_legal_count",
  }
  save_replay(all_rows, args.out, metadata=metadata)
  summary = {
      **metadata,
      "out": str(args.out),
      "rows": int(len(all_rows)),
      "rounds": int(total_rounds),
      "rounds_with_any_paradox": int(paradox_rounds),
      "same_policy_paradox_round_rate": (
          round(float(paradox_rounds) / float(total_rounds), 4)
          if total_rounds else None
      ),
      "mean_survival_prob": float(np.mean(probs)) if len(probs) else 0.0,
      "survival_prob_quantiles": (
          np.quantile(probs, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
          .round(4)
          .tolist()
          if len(probs) else []
      ),
      "round_samples": round_summaries[:5],
      "elapsed_sec": round(time.perf_counter() - started, 3),
  }
  summary_path = (
      Path(args.summary_out)
      if args.summary_out else Path(args.out).with_suffix(Path(args.out).suffix + ".summary.json")
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
