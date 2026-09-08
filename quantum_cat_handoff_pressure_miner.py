#!/usr/bin/env python3
"""Mine actions that hand the table a low-exit bottleneck.

The low-exit miner labels decisions where the acting player is already near
forced. This miner looks one step earlier: it finds decisions in failed rounds
whose selected action immediately leaves the current/next live player or the
table with very few non-paradox legal exits, then labels legal alternatives by
the same full-state post-action pressure metric.
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


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=40)
  parser.add_argument("--seed", type=int, default=20289601)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--min-trick-number", type=int, default=4)
  parser.add_argument("--max-trick-number", type=int, default=7)
  parser.add_argument("--lookback-decisions", type=int, default=24)
  parser.add_argument("--max-failure-rounds", type=int, default=120)
  parser.add_argument("--max-rows", type=int, default=500)
  parser.add_argument("--max-selected-current-legal", type=int, default=2)
  parser.add_argument("--max-selected-min-legal", type=int, default=2)
  parser.add_argument("--min-current-legal-improvement", type=int, default=2)
  parser.add_argument("--min-min-legal-improvement", type=int, default=1)
  parser.add_argument(
      "--candidate-guard",
      choices=("none", "lane_or_responsible_red_win"),
      default="none",
      help=(
          "Optional filter for positive alternatives. "
          "lane_or_responsible_red_win keeps candidates that either preserve "
          "the led-color token or spend it on a red switch that wins a "
          "still-needed trick according to the liveness certificate."
      ),
  )
  parser.add_argument("--safe-action-value-target", type=float, default=0.9)
  parser.add_argument("--risky-action-value-target", type=float, default=-0.9)
  parser.add_argument("--safe-risk-target", type=float, default=0.05)
  parser.add_argument("--risky-risk-target", type=float, default=0.95)
  parser.add_argument(
      "--dense-pressure-labels",
      action="store_true",
      help=(
          "Label every legal action by exact post-action table pressure instead "
          "of only labeling selected-vs-best. The dense key prefers avoiding "
          "forced exits, preserving the led-color lane unless the action is a "
          "responsible red-win switch, and then higher future legal counts."
      ),
  )
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


def pressure_key(row: dict) -> tuple[int, int, int, int, int]:
  return (
      -int(row.get("exact_forced_paradox_players_after", 0)),
      int(row.get("exact_min_nonparadox_legal_after", 0)),
      int(row.get("exact_current_nonparadox_legal_after", 0)),
      int(row.get("exact_total_nonparadox_legal_after", 0)),
      1 if bool(row.get("exact_terminal_after", False)) else 0,
  )


def is_selected_pressure_warning(row: dict, args) -> bool:
  if int(row.get("exact_forced_paradox_players_after", 0)) > 0:
    return True
  if (
      int(row.get("exact_current_nonparadox_legal_after", 99))
      <= int(args.max_selected_current_legal)
  ):
    return True
  return (
      int(row.get("exact_min_nonparadox_legal_after", 99))
      <= int(args.max_selected_min_legal)
  )


def improves_pressure(candidate: dict, selected: dict, args) -> bool:
  if pressure_key(candidate) <= pressure_key(selected):
    return False
  forced_delta = (
      int(selected.get("exact_forced_paradox_players_after", 0))
      - int(candidate.get("exact_forced_paradox_players_after", 0))
  )
  current_delta = (
      int(candidate.get("exact_current_nonparadox_legal_after", 0))
      - int(selected.get("exact_current_nonparadox_legal_after", 0))
  )
  min_delta = (
      int(candidate.get("exact_min_nonparadox_legal_after", 0))
      - int(selected.get("exact_min_nonparadox_legal_after", 0))
  )
  return (
      forced_delta > 0
      or current_delta >= int(args.min_current_legal_improvement)
      or min_delta >= int(args.min_min_legal_improvement)
  )


def candidate_guard_allows(decision, action: int, args) -> tuple[bool, dict]:
  if str(args.candidate_guard) == "none":
    return True, {}
  state = decision["state"]
  player = int(decision["player"])
  selected_action = int(decision["action"])
  cert = action_liveness_certificate(
      state, player, int(action), base_action=selected_action
  )
  if str(args.candidate_guard) == "lane_or_responsible_red_win":
    allowed = (
        not bool(cert.get("lost_led_token", False))
        or bool(cert.get("responsible_red_win_switch", False))
    )
    return allowed, cert
  raise ValueError(f"Unknown candidate guard: {args.candidate_guard}")


def dense_pressure_label_key(pressure: dict, cert: dict, action: int) -> tuple:
  """Sort key for dense all-legal pressure labels.

  Exact pressure is table-wide and hidden-state-derived, so it is only a
  training label. Lane preservation is intentionally explicit because the
  recent failures show that red-switch positives need a "do not burn lanes for
  free" counterweight.
  """
  lost_led = bool(cert.get("lost_led_token", False))
  responsible_red = bool(cert.get("responsible_red_win_switch", False))
  preserves_lane_or_responsible = (not lost_led) or responsible_red
  return (
      -int(pressure.get("exact_forced_paradox_players_after", 0)),
      1 if preserves_lane_or_responsible else 0,
      1 if responsible_red else 0,
      int(pressure.get("exact_min_nonparadox_legal_after", 0)),
      int(pressure.get("exact_current_nonparadox_legal_after", 0)),
      int(pressure.get("exact_total_nonparadox_legal_after", 0)),
      1 if bool(pressure.get("exact_terminal_after", False)) else 0,
      -int(action),
  )


def dense_pressure_targets(legal, pressure_by_action, cert_by_action, args):
  ranked = sorted(
      [int(action) for action in legal if int(action) != 999],
      key=lambda action: dense_pressure_label_key(
          pressure_by_action[action], cert_by_action[action], action
      ),
  )
  if not ranked:
    return {}, {}
  if len(ranked) == 1:
    return {ranked[0]: float(args.safe_action_value_target)}, {
        ranked[0]: float(args.safe_risk_target)
    }
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
  return value_targets, risk_targets


def find_handoff_decision(decisions, first_paradox_index, allowed_phases, args):
  if first_paradox_index is None:
    return None, None
  end = max(0, int(first_paradox_index))
  start = max(0, end - int(args.lookback_decisions))
  best = None
  best_selected_pressure = None
  for decision in decisions[start:end]:
    if str(decision.get("phase")) not in allowed_phases:
      continue
    if len(decision.get("legal", [])) < 2:
      continue
    trick = int(decision.get("trick_number", -1))
    if trick < int(args.min_trick_number):
      continue
    if int(args.max_trick_number) >= 0 and trick > int(args.max_trick_number):
      continue
    selected_pressure = exact_legal_pressure_after_action(
        decision["state"], int(decision["action"])
    )
    if not is_selected_pressure_warning(selected_pressure, args):
      continue
    if best is None:
      best = decision
      best_selected_pressure = selected_pressure
      continue
    # Prefer earlier upstream warnings, then worse immediate pressure.
    key = (
        int(decision.get("trick_number", 99)),
        int(decision.get("index", 99)),
        pressure_key(selected_pressure),
    )
    best_key = (
        int(best.get("trick_number", 99)),
        int(best.get("index", 99)),
        pressure_key(best_selected_pressure or {}),
    )
    if key < best_key:
      best = decision
      best_selected_pressure = selected_pressure
  return best, best_selected_pressure


def replay_row_for_handoff(
    decision,
    selected_pressure,
    best_action,
    best_pressure,
    args,
    pressure_by_action=None,
    cert_by_action=None,
):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  selected_action = int(decision["action"])
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[list(decision["legal"])] = 1.0
  policy_target = np.zeros(num_actions, dtype=np.float32)
  policy_target[int(best_action)] = 1.0

  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  action_paradox_targets[selected_action] = float(args.risky_risk_target)
  action_paradox_targets[int(best_action)] = float(args.safe_risk_target)
  action_paradox_mask[selected_action] = 1.0
  action_paradox_mask[int(best_action)] = 1.0

  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_value_targets[selected_action] = float(args.risky_action_value_target)
  action_value_targets[int(best_action)] = float(args.safe_action_value_target)
  action_value_mask[selected_action] = 1.0
  action_value_mask[int(best_action)] = 1.0
  if bool(getattr(args, "dense_pressure_labels", False)):
    if pressure_by_action is None or cert_by_action is None:
      raise ValueError("dense pressure labels require all legal pressures")
    dense_values, dense_risks = dense_pressure_targets(
        decision["legal"], pressure_by_action, cert_by_action, args
    )
    for action, target in dense_values.items():
      action_value_targets[int(action)] = float(target)
      action_value_mask[int(action)] = 1.0
    for action, target in dense_risks.items():
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
  row_group_ids = []
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
      if not result["any_paradox"]:
        continue
      summary["failure_rounds"] += 1
      warning, selected_pressure = find_handoff_decision(
          decisions,
          first_paradox_index,
          allowed_phases,
          args,
      )
      if warning is None or selected_pressure is None:
        summary["no_handoff_warning"] += 1
        continue
      summary["handoff_warnings"] += 1

      candidates = []
      selected_cert = action_liveness_certificate(
          warning["state"],
          int(warning["player"]),
          int(warning["action"]),
          base_action=int(warning["action"]),
      )
      pressure_by_action = {int(warning["action"]): selected_pressure}
      cert_by_action = {int(warning["action"]): selected_cert}
      for action in warning["legal"]:
        action = int(action)
        if action == int(warning["action"]):
          continue
        pressure = exact_legal_pressure_after_action(warning["state"], action)
        pressure_by_action[action] = pressure
        cert_by_action[action] = action_liveness_certificate(
            warning["state"],
            int(warning["player"]),
            action,
            base_action=int(warning["action"]),
        )
        if improves_pressure(pressure, selected_pressure, args):
          allowed, cert = candidate_guard_allows(warning, action, args)
          if not allowed:
            summary["candidate_guard_rejected"] += 1
            continue
          candidates.append((action, pressure, cert))
      if not candidates:
        summary["no_improving_candidate"] += 1
        continue
      best_action, best_pressure, best_cert = max(
          candidates, key=lambda item: pressure_key(item[1])
      )
      rows.append(
          replay_row_for_handoff(
              warning,
              selected_pressure,
              best_action,
              best_pressure,
              args,
              pressure_by_action=pressure_by_action,
              cert_by_action=cert_by_action,
          )
      )
      row_group_id = f"m{int(match_idx)}_r{int(round_idx)}_s{int(start_player)}"
      row_group_ids.append(row_group_id)
      summary["rows"] += 1
      if bool(args.dense_pressure_labels):
        dense_count = sum(1 for action in warning["legal"] if int(action) != 999)
        summary["dense_pressure_rows"] += 1
        summary["dense_pressure_actions_labeled"] += int(dense_count)
      summary[f"warning_trick_{int(warning.get('trick_number', -1))}"] += 1
      summary[f"selected_current_legal_{int(selected_pressure.get('exact_current_nonparadox_legal_after', -1))}"] += 1
      summary[f"selected_min_legal_{int(selected_pressure.get('exact_min_nonparadox_legal_after', -1))}"] += 1
      if bool(selected_cert.get("lost_led_token", False)):
        summary["selected_lost_led_token"] += 1
      if bool(selected_cert.get("responsible_red_win_switch", False)):
        summary["selected_responsible_red_win_switch"] += 1
      if bool(best_cert.get("lost_led_token", False)):
        summary["best_lost_led_token"] += 1
      if bool(best_cert.get("responsible_red_win_switch", False)):
        summary["best_responsible_red_win_switch"] += 1
      dense_ranked_actions = []
      if bool(args.dense_pressure_labels):
        dense_ranked_actions = sorted(
            [int(action) for action in warning["legal"] if int(action) != 999],
            key=lambda action: dense_pressure_label_key(
                pressure_by_action[action], cert_by_action[action], action
            ),
            reverse=True,
        )
      metadata_rows.append({
          "row_group_id": row_group_id,
          "match_index": int(match_idx),
          "round_index": int(round_idx),
          "start_player": int(start_player),
          "decision_index": int(warning["index"]),
          "player": int(warning["player"]),
          "trick_number": int(warning["trick_number"]),
          "led_color": getattr(warning["state"], "_led_color", None),
          "legal_count": int(len(warning["legal"])),
          "selected_action": int(warning["action"]),
          "selected_action_string": str(warning["action_string"]),
          "best_action": int(best_action),
          "best_action_string": action_string(
              warning["state"], int(warning["player"]), int(best_action)
          ),
          "selected_pressure": selected_pressure,
          "best_pressure": best_pressure,
          "selected_liveness": selected_cert,
          "best_liveness": best_cert,
          "selected_pressure_key": list(pressure_key(selected_pressure)),
          "best_pressure_key": list(pressure_key(best_pressure)),
          "dense_pressure_labels": bool(args.dense_pressure_labels),
          "dense_ranked_actions": dense_ranked_actions,
      })
      if (
          int(args.progress_interval) > 0
          and summary["handoff_warnings"] % int(args.progress_interval) == 0
      ):
        print(json.dumps({
            "iteration": "handoff_pressure_progress",
            "matches": int(match_idx + 1),
            "rounds": int(summary["rounds"]),
            "failure_rounds": int(summary["failure_rounds"]),
            "handoff_warnings": int(summary["handoff_warnings"]),
            "rows": int(len(rows)),
        }), flush=True)

  if not rows:
    raise SystemExit("No handoff-pressure rows mined; relax warning thresholds")

  metadata = {
      "kind": "handoff_pressure_replay",
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "bot_mode": str(args.bot_mode),
      "min_trick_number": int(args.min_trick_number),
      "max_trick_number": int(args.max_trick_number),
      "lookback_decisions": int(args.lookback_decisions),
      "max_selected_current_legal": int(args.max_selected_current_legal),
      "max_selected_min_legal": int(args.max_selected_min_legal),
      "min_current_legal_improvement": int(args.min_current_legal_improvement),
      "min_min_legal_improvement": int(args.min_min_legal_improvement),
      "candidate_guard": str(args.candidate_guard),
      "dense_pressure_labels": bool(args.dense_pressure_labels),
      "row_group_ids": row_group_ids,
      "row_group_count": int(len(set(row_group_ids))),
      "row_group_id_kind": "source_match_round",
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
