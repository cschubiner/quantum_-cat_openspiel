#!/usr/bin/env python3
"""Generate low-risk action calibration rows from survived same-policy rounds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import

from quantum_cat_alphazero_torch import action_feature_matrix, save_replay
from quantum_cat_failure_miner import (
    choose_policy_action,
    phase_name,
    terminal_round_result,
)
from quantum_cat_full_match_elo import load_neural


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=40)
  parser.add_argument("--seed", type=int, default=20288760)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--bot-mode", choices=("policy", "q_policy"), default="q_policy")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--max-rows", type=int, default=0)
  parser.add_argument("--risk-target", type=float, default=0.0)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  return parser.parse_args()


def _device(name: str):
  if name == "auto":
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  return torch.device(name)


def _game(players: int, start_player: int, match_context: bool):
  return pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": int(players),
          "start_player": int(start_player),
          "match_context": int(bool(match_context)),
      },
  )


def _legal_mask(num_actions: int, legal) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=np.float32)
  for action in legal:
    mask[int(action)] = 1.0
  return mask


def _row_for_decision(decision, players: int, risk_target: float):
  state = decision["state"]
  player = int(decision["player"])
  action = int(decision["action"])
  num_actions = int(state.num_distinct_actions())
  legal_mask = _legal_mask(num_actions, decision["legal"])
  policy_target = np.zeros(num_actions, dtype=np.float32)
  policy_target[action] = 1.0
  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  action_paradox_targets[action] = float(risk_target)
  action_paradox_mask[action] = 1.0
  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_value_targets[action] = 0.9
  action_value_mask[action] = 1.0
  return (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      policy_target,
      np.zeros(int(players), dtype=np.float32),
      np.zeros(int(players), dtype=np.float32),
      action,
      player,
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      action_feature_matrix(state, player, num_actions),
  )


def play_round(model, game, seed, value_scale, device, match_totals, round_idx, args):
  np.random.seed(int(seed) % (2**32 - 1))
  state = game.new_initial_state()
  if bool(args.match_context):
    state.set_match_context(match_totals, int(round_idx))
  decisions = []
  while not state.is_terminal():
    if state.is_chance_node():
      actions, probs = zip(*state.chance_outcomes())
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    player = int(state.current_player())
    legal = list(map(int, state.legal_actions(player)))
    state_before = state.clone()
    action, policy = choose_policy_action(
        model, state, player, value_scale, device, args
    )
    if int(action) not in legal:
      action = int(max(legal, key=lambda item: float(policy[int(item)])))
    decisions.append({
        "state": state_before,
        "player": player,
        "phase": phase_name(state_before),
        "trick_number": int(getattr(state_before, "_trick_number", -1)),
        "legal": legal,
        "action": int(action),
        "policy_prob": float(policy[int(action)]),
    })
    state.apply_action(int(action))
  return state, decisions


def main():
  args = parse_args()
  phases = {
      part.strip().lower()
      for part in str(args.decision_phases).split(",")
      if part.strip()
  }
  device = _device(args.device)
  model, model_args = load_neural(args.checkpoint, int(args.players), device, args)
  if model is None or model_args is None:
    raise ValueError(f"Could not load checkpoint {args.checkpoint}")
  if str(args.bot_mode) == "policy":
    model_args.action_value_selection_weight = 0.0
    model_args.action_paradox_selection_penalty = 0.0
    model_args.action_paradox_rerank_mode = "additive"
  else:
    model_args.action_paradox_rerank_mode = getattr(
        args, "action_paradox_rerank_mode", "additive"
    )
  rows = []
  metadata_rows = []
  summary = Counter()
  match_totals_samples = []
  max_rows = max(0, int(args.max_rows))
  for match_idx in range(int(args.matches)):
    match_totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if max_rows and len(rows) >= max_rows:
        break
      start_player = (match_idx + round_idx) % int(args.players)
      game = _game(int(args.players), start_player, bool(args.match_context))
      seed = int(args.seed) + match_idx * 1009 + round_idx * 9173
      terminal, decisions = play_round(
          model,
          game,
          seed,
          float(model_args.value_scale),
          device,
          match_totals,
          round_idx,
          args,
      )
      result = terminal_round_result(terminal)
      match_totals += np.array(result["raw_scores"], dtype=np.float32)
      summary["rounds"] += 1
      if bool(result["any_paradox"]):
        summary["paradox_rounds"] += 1
        continue
      summary["safe_rounds"] += 1
      for decision in decisions:
        if max_rows and len(rows) >= max_rows:
          break
        phase = str(decision["phase"]).lower()
        if phases and phase not in phases:
          continue
        if len(decision["legal"]) <= 1:
          continue
        rows.append(_row_for_decision(decision, int(args.players), args.risk_target))
        metadata_rows.append({
            "source_index": len(metadata_rows),
            "match_idx": int(match_idx),
            "round_idx": int(round_idx),
            "start_player": int(start_player),
            "phase": phase,
            "player": int(decision["player"]),
            "trick_number": int(decision["trick_number"]),
            "legal_count": int(len(decision["legal"])),
            "action": int(decision["action"]),
            "policy_prob": float(decision["policy_prob"]),
            "calibration_class": "SAFE_SELECTED_ACTION",
            "risk_target": float(args.risk_target),
        })
        summary["rows"] += 1
        summary[f"rows_phase_{phase}"] += 1
    match_totals_samples.append([float(value) for value in match_totals])
    if int(args.progress_interval) > 0 and (match_idx + 1) % int(args.progress_interval) == 0:
      print(json.dumps({
          "iteration": "safe_action_calibration_progress",
          "matches": match_idx + 1,
          "rows": len(rows),
          "safe_rounds": int(summary["safe_rounds"]),
          "paradox_rounds": int(summary["paradox_rounds"]),
      }), flush=True)
    if max_rows and len(rows) >= max_rows:
      break
  out_path = Path(args.out)
  save_replay(rows, out_path, metadata={
      "source": "quantum_cat_safe_action_calibration_replay",
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "risk_target": float(args.risk_target),
      "summary": dict(summary),
  })
  output = {
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches_requested": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "risk_target": float(args.risk_target),
      "summary": dict(summary),
      "out": str(out_path),
      "match_totals_samples": match_totals_samples[:5],
  }
  summary_path = (
      Path(args.summary_out)
      if args.summary_out else out_path.with_suffix(out_path.suffix + ".summary.json")
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(output, indent=2) + "\n")
  metadata_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
  metadata_path.write_text(json.dumps({"rows": metadata_rows}, indent=2) + "\n")
  print(json.dumps(output, indent=2))


if __name__ == "__main__":
  main()
