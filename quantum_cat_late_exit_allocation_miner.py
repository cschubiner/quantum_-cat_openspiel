#!/usr/bin/env python3
"""Mine late red-led/no-led action choices that preserve table exits.

Run645 showed that the current 4p policy already follows non-red led colors in
the mined states, while first paradoxes remain late forced exits, usually on
red-led or no-led trick-7 positions. This miner targets the next bottleneck:
when the round is late and red/no-led choices are available, label which legal
play keeps the table's exact non-paradox legal-action pressure highest.

Exact pressure labels inspect full simulator state and are training labels only.
Saved observations and action features remain the normal information-legal
inputs for inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyspiel
import torch

from quantum_cat_alphazero_torch import action_feature_matrix, save_replay
from quantum_cat_failure_miner import (
    action_string,
    parse_phase_filter,
    play_round_with_history,
    terminal_round_result,
)
from quantum_cat_full_match_elo import (
    action_liveness_certificate,
    exact_legal_pressure_after_action,
    load_neural,
)
import quantum_cat_liveness_oracle as liveness_oracle


COLOR_NAMES = ("R", "B", "Y", "G")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=40)
  parser.add_argument("--seed", type=int, default=20290101)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--min-trick-number", type=int, default=5)
  parser.add_argument("--max-trick-number", type=int, default=7)
  parser.add_argument(
      "--led-colors",
      default="R,None",
      help="Comma-separated led colors to mine. Use None for no led color.",
  )
  parser.add_argument("--lookback-decisions", type=int, default=24)
  parser.add_argument("--failed-rounds-only", action="store_true")
  parser.add_argument("--max-failure-rounds", type=int, default=200)
  parser.add_argument("--max-rows", type=int, default=500)
  parser.add_argument("--include-policy-aligned", action="store_true")
  parser.add_argument(
      "--rank-mode",
      choices=("pressure", "table_liveness", "hybrid"),
      default="pressure",
      help=(
          "pressure keeps the historical one-ply exact legal-count ranking; "
          "table_liveness ranks by all players' exact post-action own-hand "
          "liveness margins; hybrid uses table liveness first, then pressure."
      ),
  )
  parser.add_argument(
      "--min-table-margin-spread",
      type=int,
      default=-999,
      help="Drop rows whose best-worst table min-margin spread is below this.",
  )
  parser.add_argument(
      "--min-exact-legal-spread",
      type=int,
      default=-1,
      help="Drop rows whose best-worst exact min legal-count spread is below this.",
  )
  parser.add_argument("--safe-action-value-target", type=float, default=0.9)
  parser.add_argument("--risky-action-value-target", type=float, default=-0.9)
  parser.add_argument("--safe-risk-target", type=float, default=0.05)
  parser.add_argument("--risky-risk-target", type=float, default=0.95)
  parser.add_argument(
      "--bot-mode",
      choices=("policy", "q_policy", "liveness_policy", "q_liveness_policy"),
      default="policy",
  )
  parser.add_argument("--liveness-shield-phases", default="play")
  parser.add_argument("--liveness-shield-min-open-slot-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-min-public-damage-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument(
      "--action-paradox-rerank-mode",
      choices=("additive", "threshold", "relative"),
      default="additive",
  )
  parser.add_argument("--action-paradox-risk-threshold", type=float, default=0.0)
  parser.add_argument("--action-paradox-min-risk-margin", type=float, default=0.0)
  parser.add_argument("--action-paradox-max-policy-log-gap", type=float, default=2.0)
  parser.add_argument("--action-feasibility-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument("--action-value-rerank-phases", default="")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  parser.add_argument("--action-value-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  return parser.parse_args()


def parse_led_colors(value: str) -> set[str | None]:
  out = set()
  for raw in str(value).split(","):
    item = raw.strip()
    if not item:
      continue
    if item.lower() in ("none", "null", "no_led"):
      out.add(None)
    elif item in COLOR_NAMES:
      out.add(item)
    else:
      raise ValueError(f"Unknown led color filter: {item}")
  return out


def action_color_name(state, action: int) -> str | None:
  action = int(action)
  if action == 999:
    return None
  return COLOR_NAMES[action // int(state._num_card_types)]


def pressure_rank_key(pressure: dict, cert: dict, action: int) -> tuple:
  """Higher key is safer/better for late exit allocation."""
  # Exact table pressure dominates. Liveness certificate fields are
  # information-legal features, but here they only break exact-label ties.
  return (
      -int(pressure.get("exact_forced_paradox_players_after", 0)),
      int(pressure.get("exact_min_nonparadox_legal_after", 0)),
      int(pressure.get("exact_current_nonparadox_legal_after", 0)),
      int(pressure.get("exact_total_nonparadox_legal_after", 0)),
      int(cert.get("min_player_open_slots_after", 0)),
      int(cert.get("total_player_open_slots_after", 0)),
      0 if bool(cert.get("over_target_would_win", False)) else 1,
      1 if bool(cert.get("would_win_now", False)) else 0,
      1 if bool(pressure.get("exact_terminal_after", False)) else 0,
      -int(action),
  )


def table_liveness_after_action(state, action):
  """Exact post-action table liveness label for training-time ranking.

  This intentionally reads every player's hand after the candidate action.
  That is hidden information, so it is only used to create supervised labels;
  saved replay inputs remain the acting player's normal observation/features.
  """
  clone = state.clone()
  clone.apply_action(int(action))
  num_players = int(getattr(clone, "_num_players", 0) or 0)
  has_paradoxed = list(getattr(clone, "_has_paradoxed", [False] * num_players))
  any_paradox = any(bool(value) for value in has_paradoxed)
  if clone.is_terminal():
    good_terminal = not any_paradox
    margin = 99 if good_terminal else -99
    return {
        "table_terminal_after": True,
        "table_any_paradox_after": bool(any_paradox),
        "table_dead_margin_players_after": int(sum(1 for value in has_paradoxed if value)),
        "table_min_margin_after": int(margin),
        "table_current_margin_after": int(margin),
        "table_total_margin_after": int(margin * max(1, num_players)),
        "table_active_players_after": 0,
    }

  margins = []
  dead_count = 0
  for seat in range(num_players):
    if bool(has_paradoxed[seat]):
      dead_count += 1
      continue
    remaining_cards = int(np.sum(getattr(clone, "_hands", [])[seat]))
    if remaining_cards <= 0:
      continue
    view = liveness_oracle.view_from_state(clone, seat)
    margin = int(view.margin())
    margins.append(margin)
    if margin < 0:
      dead_count += 1

  current = int(clone.current_player())
  if 0 <= current < num_players and not bool(has_paradoxed[current]):
    current_remaining = int(np.sum(getattr(clone, "_hands", [])[current]))
    if current_remaining > 0:
      current_margin = int(liveness_oracle.view_from_state(clone, current).margin())
    else:
      current_margin = 99
  else:
    current_margin = 99

  return {
      "table_terminal_after": False,
      "table_any_paradox_after": bool(any_paradox),
      "table_dead_margin_players_after": int(dead_count),
      "table_min_margin_after": int(min(margins)) if margins else 99,
      "table_current_margin_after": int(current_margin),
      "table_total_margin_after": int(sum(margins)) if margins else 99 * max(1, num_players),
      "table_active_players_after": int(len(margins)),
  }


def table_liveness_rank_key(table: dict, action: int) -> tuple:
  return (
      0 if bool(table.get("table_any_paradox_after", False)) else 1,
      -int(table.get("table_dead_margin_players_after", 0)),
      int(table.get("table_min_margin_after", -99)),
      int(table.get("table_current_margin_after", -99)),
      int(table.get("table_total_margin_after", -99)),
      1 if bool(table.get("table_terminal_after", False)) else 0,
      -int(action),
  )


def combined_rank_key(pressure: dict, cert: dict, table: dict, action: int, args):
  if str(args.rank_mode) == "pressure":
    return pressure_rank_key(pressure, cert, action)
  table_key = table_liveness_rank_key(table, action)
  if str(args.rank_mode) == "table_liveness":
    return table_key + pressure_rank_key(pressure, cert, action)
  return table_key + pressure_rank_key(pressure, cert, action)


def ranked_targets(
    legal_actions, pressure_by_action, cert_by_action, table_by_action, args
):
  actions = [int(action) for action in legal_actions if int(action) != 999]
  ranked = sorted(
      actions,
      key=lambda action: combined_rank_key(
          pressure_by_action[action],
          cert_by_action[action],
          table_by_action[action],
          action,
          args,
      ),
  )
  if len(ranked) <= 1:
    return (
        {ranked[0]: float(args.safe_action_value_target)} if ranked else {},
        {ranked[0]: float(args.safe_risk_target)} if ranked else {},
        ranked,
    )
  denom = float(len(ranked) - 1)
  value_targets = {}
  risk_targets = {}
  low = float(args.risky_action_value_target)
  high = float(args.safe_action_value_target)
  risk_low = float(args.safe_risk_target)
  risk_high = float(args.risky_risk_target)
  for idx, action in enumerate(ranked):
    frac = float(idx) / denom
    value_targets[action] = low + frac * (high - low)
    risk_targets[action] = risk_high - frac * (risk_high - risk_low)
  return value_targets, risk_targets, ranked


def build_row(decision, best_action, value_targets, risk_targets, args):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[list(decision["legal"])] = 1.0
  policy_target = np.zeros(num_actions, dtype=np.float32)
  policy_target[int(best_action)] = 1.0
  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  for action, target in value_targets.items():
    action_value_targets[int(action)] = float(target)
    action_value_mask[int(action)] = 1.0
  for action, target in risk_targets.items():
    action_paradox_targets[int(action)] = float(target)
    action_paradox_mask[int(action)] = 1.0
  return (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      policy_target,
      np.zeros(int(args.players), dtype=np.float32),
      np.zeros(int(args.players), dtype=np.float32),
      int(best_action),
      player,
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      action_feature_matrix(state, player, num_actions),
  )


def candidate_row(decision, allowed_phases, allowed_leds, args):
  if str(decision.get("phase")) not in allowed_phases:
    return None
  trick = int(decision.get("trick_number", -1))
  if trick < int(args.min_trick_number):
    return None
  if int(args.max_trick_number) >= 0 and trick > int(args.max_trick_number):
    return None
  state = decision["state"]
  led_color = getattr(state, "_led_color", None)
  if led_color not in allowed_leds:
    return None
  legal = [int(action) for action in decision["legal"] if int(action) != 999]
  if len(legal) < 2:
    return None
  selected_action = int(decision["action"])
  player = int(decision["player"])
  pressure_by_action = {}
  cert_by_action = {}
  table_by_action = {}
  for action in legal:
    pressure_by_action[action] = exact_legal_pressure_after_action(state, action)
    cert_by_action[action] = action_liveness_certificate(
        state, player, action, base_action=selected_action
    )
    table_by_action[action] = table_liveness_after_action(state, action)
  value_targets, risk_targets, ranked = ranked_targets(
      legal, pressure_by_action, cert_by_action, table_by_action, args
  )
  if not ranked:
    return None
  best_table = table_by_action[int(ranked[-1])]
  worst_table = table_by_action[int(ranked[0])]
  table_margin_spread = int(best_table.get("table_min_margin_after", -99)) - int(
      worst_table.get("table_min_margin_after", -99)
  )
  if table_margin_spread < int(args.min_table_margin_spread):
    return None
  best_pressure = pressure_by_action[int(ranked[-1])]
  worst_pressure = pressure_by_action[int(ranked[0])]
  exact_legal_spread = int(
      best_pressure.get("exact_min_nonparadox_legal_after", 0)
  ) - int(worst_pressure.get("exact_min_nonparadox_legal_after", 0))
  if exact_legal_spread < int(args.min_exact_legal_spread):
    return None
  best_action = int(ranked[-1])
  if (
      not bool(args.include_policy_aligned)
      and int(selected_action) == int(best_action)
  ):
    return None
  return {
      "best_action": best_action,
      "pressure_by_action": pressure_by_action,
      "cert_by_action": cert_by_action,
      "table_by_action": table_by_action,
      "value_targets": value_targets,
      "risk_targets": risk_targets,
      "ranked_actions": ranked,
      "table_margin_spread": table_margin_spread,
      "exact_legal_spread": exact_legal_spread,
  }


def main():
  args = parse_args()
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model, model_args = load_neural(args.checkpoint, args.players, device, args)
  value_scale = float(model_args.value_scale)
  allowed_phases = parse_phase_filter(args.decision_phases)
  allowed_leds = parse_led_colors(args.led_colors)
  rows = []
  row_group_ids = []
  metadata_rows = []
  summary = Counter()

  for match_idx in range(int(args.matches)):
    if int(args.max_rows) > 0 and len(rows) >= int(args.max_rows):
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if int(args.max_rows) > 0 and len(rows) >= int(args.max_rows):
        break
      if (
          int(args.max_failure_rounds) > 0
          and summary["failure_rounds"] >= int(args.max_failure_rounds)
      ):
        break
      start_player = round_idx % int(args.players)
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": int(args.players),
              "start_player": int(start_player),
              "match_context": int(bool(args.match_context)),
          },
      )
      prior_totals = np.copy(totals)
      round_seed = int(args.seed) + match_idx * 1000003 + round_idx * 9973
      terminal, decisions, first_paradox_index = play_round_with_history(
          model,
          game,
          seed=round_seed,
          value_scale=value_scale,
          device=device,
          match_totals=prior_totals if args.match_context else None,
          round_index=round_idx,
          args=args,
      )
      returns = np.array(terminal.returns(), dtype=np.float32)
      totals = returns if args.match_context else totals + returns
      result = terminal_round_result(terminal)
      summary["rounds"] += 1
      if result["any_paradox"]:
        summary["failure_rounds"] += 1
      elif bool(args.failed_rounds_only):
        continue
      if first_paradox_index is None:
        decision_window = decisions
      else:
        start = max(0, int(first_paradox_index) - int(args.lookback_decisions))
        decision_window = decisions[start:int(first_paradox_index)]
      for decision in decision_window:
        if int(args.max_rows) > 0 and len(rows) >= int(args.max_rows):
          break
        mined = candidate_row(decision, allowed_phases, allowed_leds, args)
        if mined is None:
          summary["decisions_skipped"] += 1
          continue
        state = decision["state"]
        selected_action = int(decision["action"])
        best_action = int(mined["best_action"])
        selected_cert = mined["cert_by_action"][selected_action]
        best_cert = mined["cert_by_action"][best_action]
        selected_pressure = mined["pressure_by_action"][selected_action]
        best_pressure = mined["pressure_by_action"][best_action]
        selected_table = mined["table_by_action"][selected_action]
        best_table = mined["table_by_action"][best_action]
        rows.append(
            build_row(
                decision,
                best_action,
                mined["value_targets"],
                mined["risk_targets"],
                args,
            )
        )
        group_id = f"m{match_idx:06d}_r{round_idx:02d}"
        row_group_ids.append(group_id)
        summary["rows"] += 1
        led_color = getattr(state, "_led_color", None)
        summary[f"row_led_{led_color}"] += 1
        summary[f"row_trick_{int(decision.get('trick_number', -1))}"] += 1
        summary[f"best_color_{action_color_name(state, best_action)}"] += 1
        summary[f"selected_color_{action_color_name(state, selected_action)}"] += 1
        if selected_action == best_action:
          summary["selected_is_best"] += 1
        if int(mined["table_margin_spread"]) > 0:
          summary["positive_table_margin_spread"] += 1
        if int(mined["exact_legal_spread"]) > 0:
          summary["positive_exact_legal_spread"] += 1
        if bool(best_cert.get("would_win_now", False)):
          summary["best_would_win_now"] += 1
        if bool(best_cert.get("over_target_would_win", False)):
          summary["best_over_target_would_win"] += 1
        if bool(selected_cert.get("over_target_would_win", False)):
          summary["selected_over_target_would_win"] += 1
        metadata_rows.append({
            "match_index": int(match_idx),
            "round_index": int(round_idx),
            "start_player": int(start_player),
            "decision_index": int(decision["index"]),
            "player": int(decision["player"]),
            "trick_number": int(decision["trick_number"]),
            "led_color": led_color,
            "legal_count": int(len(decision["legal"])),
            "selected_action": selected_action,
            "selected_action_string": str(decision["action_string"]),
            "selected_pressure": selected_pressure,
            "selected_table_liveness": selected_table,
            "selected_liveness": selected_cert,
            "best_action": best_action,
            "best_action_string": action_string(
                state, int(decision["player"]), best_action
            ),
            "best_pressure": best_pressure,
            "best_table_liveness": best_table,
            "best_liveness": best_cert,
            "ranked_actions": [int(action) for action in mined["ranked_actions"]],
            "table_margin_spread": int(mined["table_margin_spread"]),
            "exact_legal_spread": int(mined["exact_legal_spread"]),
        })
        if (
            int(args.progress_interval) > 0
            and summary["rows"] % int(args.progress_interval) == 0
        ):
          print(json.dumps({
              "iteration": "late_exit_allocation_progress",
              "matches": int(match_idx + 1),
              "rounds": int(summary["rounds"]),
              "failure_rounds": int(summary["failure_rounds"]),
              "rows": int(summary["rows"]),
          }), flush=True)

  if not rows:
    raise SystemExit("No late-exit allocation rows mined; relax filters")

  metadata = {
      "kind": "late_exit_allocation_replay",
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "bot_mode": str(args.bot_mode),
      "min_trick_number": int(args.min_trick_number),
      "max_trick_number": int(args.max_trick_number),
      "led_colors": str(args.led_colors),
      "rank_mode": str(args.rank_mode),
      "min_table_margin_spread": int(args.min_table_margin_spread),
      "min_exact_legal_spread": int(args.min_exact_legal_spread),
      "failed_rounds_only": bool(args.failed_rounds_only),
      "include_policy_aligned": bool(args.include_policy_aligned),
      "row_group_ids": row_group_ids,
      "row_group_id_kind": "match_round",
      "row_group_count": int(len(set(row_group_ids))),
      "summary": dict(summary),
      "rows": metadata_rows,
  }
  save_replay(rows, args.out, metadata=metadata)
  summary_path = (
      Path(args.summary_out)
      if args.summary_out else Path(args.out).with_suffix(Path(args.out).suffix + ".summary.json")
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
      "out": str(args.out),
      "summary_out": str(summary_path),
      "rows": len(rows),
      "summary": dict(summary),
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
