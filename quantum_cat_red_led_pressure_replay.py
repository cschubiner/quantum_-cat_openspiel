#!/usr/bin/env python3
"""Mine late red-led exact-pressure action contrasts.

Run651 taught the non-red follow-led grammar, but the live failures still
clustered at trick-7 red-led forced exits. This miner targets those states
directly: when red is led late, label legal actions by exact post-action legal
pressure. The exact labels are training-only; replay observations/features are
the normal information-legal tensors.
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


COLOR_NAMES = ("R", "B", "Y", "G")
RED_INDEX = 0


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=40)
  parser.add_argument("--seed", type=int, default=20290751)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--min-trick-number", type=int, default=5)
  parser.add_argument("--max-trick-number", type=int, default=7)
  parser.add_argument("--lookback-decisions", type=int, default=80)
  parser.add_argument("--failed-rounds-only", action="store_true")
  parser.add_argument("--include-policy-aligned", action="store_true")
  parser.add_argument("--max-rows", type=int, default=1000)
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
  parser.add_argument("--progress-interval", type=int, default=100)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  return parser.parse_args()


def led_color_idx(state) -> int | None:
  led = getattr(state, "_led_color", None)
  if led is None:
    return None
  return COLOR_NAMES.index(str(led))


def pressure_key(row: dict) -> tuple[int, int, int, int, int]:
  return (
      -int(row.get("exact_forced_paradox_players_after", 0)),
      int(row.get("exact_min_nonparadox_legal_after", 0)),
      int(row.get("exact_current_nonparadox_legal_after", 0)),
      int(row.get("exact_total_nonparadox_legal_after", 0)),
      1 if bool(row.get("exact_terminal_after", False)) else 0,
  )


def ranked_targets(legal_actions, pressure_by_action, args):
  actions = [int(action) for action in legal_actions if int(action) != 999]
  ranked = sorted(
      actions,
      key=lambda action: (*pressure_key(pressure_by_action[action]), -int(action)),
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


def candidate_row(decision, allowed_phases, args):
  if str(decision.get("phase")) not in allowed_phases:
    return None
  trick = int(decision.get("trick_number", -1))
  if trick < int(args.min_trick_number):
    return None
  if int(args.max_trick_number) >= 0 and trick > int(args.max_trick_number):
    return None
  state = decision["state"]
  if led_color_idx(state) != RED_INDEX:
    return None
  legal = [int(action) for action in decision["legal"] if int(action) != 999]
  if len(legal) < 2:
    return None
  pressure_by_action = {
      action: exact_legal_pressure_after_action(state, action)
      for action in legal
  }
  value_targets, risk_targets, ranked = ranked_targets(
      legal, pressure_by_action, args
  )
  if not ranked:
    return None
  best_action = int(ranked[-1])
  selected_action = int(decision["action"])
  if (
      not bool(args.include_policy_aligned)
      and selected_action == best_action
  ):
    return None
  player = int(decision["player"])
  cert_by_action = {
      action: action_liveness_certificate(
          state, player, action, base_action=selected_action
      )
      for action in legal
  }
  return {
      "best_action": best_action,
      "pressure_by_action": pressure_by_action,
      "cert_by_action": cert_by_action,
      "value_targets": value_targets,
      "risk_targets": risk_targets,
      "ranked_actions": ranked,
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
  rows = []
  metadata_rows = []
  summary = Counter()

  for match_idx in range(int(args.matches)):
    if int(args.max_rows) > 0 and len(rows) >= int(args.max_rows):
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if int(args.max_rows) > 0 and len(rows) >= int(args.max_rows):
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
        mined = candidate_row(decision, allowed_phases, args)
        if mined is None:
          summary["decisions_skipped"] += 1
          continue
        state = decision["state"]
        player = int(decision["player"])
        selected_action = int(decision["action"])
        best_action = int(mined["best_action"])
        rows.append(
            build_row(
                decision,
                best_action,
                mined["value_targets"],
                mined["risk_targets"],
                args,
            )
        )
        summary["rows"] += 1
        summary[f"row_trick_{int(decision.get('trick_number', -1))}"] += 1
        best_cert = mined["cert_by_action"][best_action]
        selected_cert = mined["cert_by_action"][selected_action]
        if bool(best_cert.get("responsible_red_win_switch", False)):
          summary["best_responsible_red_win_switch"] += 1
        if bool(selected_cert.get("responsible_red_win_switch", False)):
          summary["selected_responsible_red_win_switch"] += 1
        if selected_action == best_action:
          summary["selected_is_best"] += 1
        metadata_rows.append({
            "match_index": int(match_idx),
            "round_index": int(round_idx),
            "start_player": int(start_player),
            "decision_index": int(decision["index"]),
            "player": player,
            "trick_number": int(decision["trick_number"]),
            "led_color": getattr(state, "_led_color", None),
            "legal_count": int(len(decision["legal"])),
            "selected_action": selected_action,
            "selected_action_string": str(decision["action_string"]),
            "best_action": best_action,
            "best_action_string": action_string(state, player, best_action),
            "selected_pressure": mined["pressure_by_action"][selected_action],
            "best_pressure": mined["pressure_by_action"][best_action],
            "selected_liveness": selected_cert,
            "best_liveness": best_cert,
            "ranked_actions": [int(action) for action in mined["ranked_actions"]],
        })
        if (
            int(args.progress_interval) > 0
            and summary["rows"] % int(args.progress_interval) == 0
        ):
          print(json.dumps({
              "iteration": "red_led_pressure_progress",
              "matches": int(match_idx + 1),
              "rounds": int(summary["rounds"]),
              "failure_rounds": int(summary["failure_rounds"]),
              "rows": int(summary["rows"]),
          }), flush=True)

  if not rows:
    raise SystemExit("No red-led pressure rows mined; relax filters")
  metadata = {
      "kind": "red_led_pressure_replay",
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "bot_mode": str(args.bot_mode),
      "min_trick_number": int(args.min_trick_number),
      "max_trick_number": int(args.max_trick_number),
      "failed_rounds_only": bool(args.failed_rounds_only),
      "include_policy_aligned": bool(args.include_policy_aligned),
      "summary": dict(summary),
      "row_group_ids": [
          f"m{int(row['match_index'])}:r{int(row['round_index'])}"
          for row in metadata_rows
      ],
      "row_group_count": len({
          f"m{int(row['match_index'])}:r{int(row['round_index'])}"
          for row in metadata_rows
      }),
      "row_group_id_kind": "match_round",
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
