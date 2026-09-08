"""Generate homogeneous same-policy round-survival value replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from quantum_cat_ai import make_bot
from quantum_cat_alphazero_torch import (
    action_feature_matrix,
    compact_action_feature_matrix,
    make_game,
    save_replay,
    terminal_paradox_target,
    with_terminal_targets,
)
from quantum_cat_full_match_elo import load_neural_bots, make_eval_bot


def _one_hot_policy(num_actions: int, action: int) -> np.ndarray:
  policy = np.zeros(num_actions, dtype=np.float32)
  if 0 <= int(action) < num_actions:
    policy[int(action)] = 1.0
  return policy


def _legal_mask(num_actions: int, legal: list[int]) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=np.float32)
  for action in legal:
    if 0 <= int(action) < num_actions:
      mask[int(action)] = 1.0
  return mask


def _round_raw_scores(state) -> np.ndarray:
  if hasattr(state, "raw_round_scores"):
    return np.array(state.raw_round_scores(), dtype=np.float32)
  return np.array(state.returns(), dtype=np.float32)


def _play_round_examples(
    state,
    game,
    bots,
    players: int,
    decision_phases=None,
    compact_legal_actions: bool = False,
) -> tuple[list[tuple], object]:
  examples = []
  num_actions = game.num_distinct_actions()
  while not state.is_terminal():
    if state.is_chance_node():
      actions, probs = zip(*state.chance_outcomes())
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue
    player = int(state.current_player())
    legal = [int(action) for action in state.legal_actions(player)]
    action = int(bots[player].step(state.clone(), player))
    if action not in legal:
      action = int(legal[0])
    phase_name = {
        1: "discard", 2: "prediction", 3: "play"
    }.get(int(getattr(state, "_phase", -1)), "unknown")
    if decision_phases is None or phase_name in decision_phases:
      obs = np.array(state.observation_tensor(player), dtype=np.float32)
      if compact_legal_actions:
        selected_action = legal.index(action)
        mask = np.ones(len(legal), dtype=np.float32)
        policy = _one_hot_policy(len(legal), selected_action)
        features = compact_action_feature_matrix(state, player, legal)
      else:
        selected_action = action
        mask = _legal_mask(num_actions, legal)
        policy = _one_hot_policy(num_actions, action)
        features = action_feature_matrix(state, player, num_actions)
      examples.append((
          obs,
          mask,
          policy,
          None,
          selected_action,
          player,
          None,
          None,
          None,
          None,
          features,
      ))
    state.apply_action(action)
  return examples, state


def _collector_bots(args: argparse.Namespace, seed: int) -> list:
  checkpoint = str(getattr(args, "checkpoint", "") or "").strip()
  if not checkpoint:
    return [
        make_bot(args.bot, seed=seed + seat * 17)
        for seat in range(int(args.players))
    ]
  device = torch.device(str(getattr(args, "device", "cpu")))
  neural_bots = getattr(args, "_collector_neural_bots", None)
  if neural_bots is None:
    mode = str(getattr(args, "policy_mode", "liveness_shield"))
    overrides = ""
    if mode == "liveness_shield":
      overrides = (
          "?liveness_shield_base_mode=policy,liveness_shield_phases=play,"
          "liveness_shield_min_open_slot_delta=0,"
          "liveness_shield_min_public_damage_delta=0,"
          "liveness_shield_max_policy_log_gap=-1.0"
      )
    spec = f"survival_source={checkpoint}:{mode}{overrides}"
    neural_bots = load_neural_bots([spec], int(args.players), device)
    args._collector_neural_bots = neural_bots
  return [
      make_eval_bot(
          "survival_source",
          seed + seat * 17,
          None,
          None,
          device,
          belief_samples=4,
          belief_sims=8,
          mcts_sims=8,
          neural_bots=neural_bots,
      )
      for seat in range(int(args.players))
  ]


def _decision_phase_names(args: argparse.Namespace):
  spec = str(getattr(args, "decision_phases", "all") or "all").strip()
  if not spec or spec.lower() == "all":
    return None
  phases = {item.strip().lower() for item in spec.split(",") if item.strip()}
  unknown = phases - {"discard", "prediction", "play"}
  if unknown:
    raise ValueError(f"Unsupported survival replay phases: {sorted(unknown)}")
  return phases


def collect_match(args: argparse.Namespace, match_idx: int) -> dict:
  seed = int(args.seed) + match_idx * 1009
  np.random.seed(seed % (2**32 - 1))
  bots = _collector_bots(args, seed)
  match_totals = np.zeros(int(args.players), dtype=np.float32)
  rows = []
  row_group_ids = []
  round_summaries = []
  source_group_id = str(
      getattr(args, "source_group_id", "")
      or f"players={args.players}:seed={args.seed}"
  )
  decision_phases = _decision_phase_names(args)
  for round_idx in range(int(args.players)):
    start_player = (int(args.initial_start) + match_idx + round_idx) % int(args.players)
    game = make_game(args, start_player)
    state = game.new_initial_state()
    if bool(args.match_context):
      state.set_match_context(match_totals, round_idx)
    examples, terminal_state = _play_round_examples(
        state,
        game,
        bots,
        int(args.players),
        decision_phases=decision_phases,
        compact_legal_actions=bool(
            getattr(args, "compact_legal_actions", False)
        ),
    )
    raw_scores = _round_raw_scores(terminal_state)
    match_totals += raw_scores
    paradox_target = terminal_paradox_target(terminal_state, int(args.players))
    any_paradox = bool(np.any(paradox_target > 0.5))
    survival_value = -1.0 if any_paradox else 1.0
    value_targets = np.full(int(args.players), survival_value, dtype=np.float32)
    round_rows = with_terminal_targets(
        examples,
        value_targets,
        paradox_target,
        terminal_action_paradox_targets=[paradox_target] * len(examples),
    )
    rows.extend(round_rows)
    row_group_ids.extend([source_group_id] * len(round_rows))
    round_summaries.append({
        "source_group_id": source_group_id,
        "match_idx": int(match_idx),
        "round_idx": int(round_idx),
        "start_player": int(start_player),
        "decision_rows": int(len(examples)),
        "any_paradox": any_paradox,
        "survival_value": float(survival_value),
        "paradoxed": [bool(value) for value in paradox_target],
        "raw_scores": [float(value) for value in raw_scores],
    })
  return {
      "rows": rows,
      "row_group_ids": row_group_ids,
      "rounds": round_summaries,
      "match_totals": [float(value) for value in match_totals],
  }


def summarize_replay(rows: list[tuple], round_summaries: list[dict]) -> dict:
  total_rounds = len(round_summaries)
  bad_rounds = sum(1 for row in round_summaries if row["any_paradox"])
  value_counts: dict[str, int] = {}
  phase_counts: dict[str, int] = {}
  for item in rows:
    value = float(np.array(item[3], dtype=np.float32)[0])
    value_counts[str(value)] = value_counts.get(str(value), 0) + 1
    features = item[11]
    action = int(item[5])
    if features is not None and 0 <= action < len(features):
      phase_idx = int(np.argmax(features[action, 1:6]))
      phase_name = ["chance", "discard", "prediction", "play", "terminal"][
          phase_idx
      ]
      phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
  return {
      "rows": int(len(rows)),
      "rounds": int(total_rounds),
      "rounds_with_any_paradox": int(bad_rounds),
      "same_policy_paradox_round_rate": (
          round(float(bad_rounds) / float(total_rounds), 4)
          if total_rounds else None
      ),
      "value_target_counts": value_counts,
      "phase_counts": phase_counts,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=10)
  parser.add_argument("--bot", default="heuristic_safe14")
  parser.add_argument(
      "--checkpoint",
      default="",
      help="Optional neural checkpoint for collecting on the actual deployment policy.",
  )
  parser.add_argument(
      "--policy-mode",
      choices=("policy", "liveness_shield"),
      default="liveness_shield",
  )
  parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
  parser.add_argument(
      "--source-group-id",
      default="",
      help="Independent collector-source identifier stored on every replay row.",
  )
  parser.add_argument(
      "--decision-phases",
      default="all",
      help="Comma-separated discard,prediction,play phases, or all.",
  )
  parser.add_argument(
      "--compact-legal-actions",
      action="store_true",
      help=(
          "Store only per-row legal action features and remap the selected "
          "action locally. This keeps state-value training equivalent while "
          "avoiding 1,000-action replay tensors."
      ),
  )
  parser.add_argument("--seed", type=int, default=20266700)
  parser.add_argument("--initial-start", type=int, default=0)
  parser.add_argument("--no-match-context", dest="match_context", action="store_false")
  parser.set_defaults(match_context=True)
  parser.add_argument("--out", default="az_runs/survival_value_replay/replay_latest.npz")
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--progress-interval", type=int, default=0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.players < 2 or args.players > 5:
    raise ValueError("--players must be 2..5")
  started = time.perf_counter()
  all_rows = []
  all_row_group_ids = []
  all_rounds = []
  match_totals = []
  for match_idx in range(int(args.matches)):
    result = collect_match(args, match_idx)
    all_rows.extend(result["rows"])
    all_row_group_ids.extend(result["row_group_ids"])
    all_rounds.extend(result["rounds"])
    match_totals.append(result["match_totals"])
    if args.progress_interval > 0 and (match_idx + 1) % args.progress_interval == 0:
      print(json.dumps({
          "completed_matches": match_idx + 1,
          "rows": len(all_rows),
          "elapsed_sec": round(time.perf_counter() - started, 3),
      }), flush=True)

  metadata = {
      "kind": "homogeneous_round_survival_value",
      "players": int(args.players),
      "matches": int(args.matches),
      "bot": (
          f"checkpoint:{args.policy_mode}"
          if args.checkpoint else str(args.bot)
      ),
      "checkpoint": str(args.checkpoint),
      "policy_mode": str(args.policy_mode) if args.checkpoint else "builtin",
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "target": "round_no_any_player_paradox:+1_else:-1",
      "full_match_training": True,
      "full_match_terminal_target_scope": "round",
      "action_paradox_terminal_fallback_scope": "any",
      "source_group_id": str(
          args.source_group_id or f"players={args.players}:seed={args.seed}"
      ),
      "row_group_ids": all_row_group_ids,
      "decision_phases": str(args.decision_phases),
      "action_indexing": (
          "per_row_legal" if args.compact_legal_actions else "global"
      ),
      "original_num_distinct_actions": int(
          make_game(args).num_distinct_actions()
      ),
  }
  out_path = Path(args.out)
  save_replay(all_rows, out_path, metadata=metadata)
  summary = {
      **{key: value for key, value in metadata.items() if key != "row_group_ids"},
      "row_group_count": len(set(all_row_group_ids)),
      **summarize_replay(all_rows, all_rounds),
      "out": str(out_path),
      "elapsed_sec": round(time.perf_counter() - started, 3),
      "match_totals_mean": (
          np.mean(np.array(match_totals, dtype=np.float32), axis=0).round(4).tolist()
          if match_totals else []
      ),
      "round_samples": all_rounds[:5],
  }
  summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(
      out_path.suffix + ".summary.json"
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
