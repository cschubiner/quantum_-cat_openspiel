#!/usr/bin/env python3
"""Mine direct non-red led-color follow versus lane-burn contrasts.

The 4p failure traces show many late forced paradoxes, but late rescue labels
mostly teach "switch to red and win." This miner targets the upstream grammar
directly: when a non-red color is led and the player has legal follow-led
actions, burning that led token is a cost unless the off-led red switch wins
and buys enough exact future legal pressure to justify it.

Exact post-action pressure is hidden-state-derived and is therefore a training
label only. The observation/action features saved in replay rows remain the
normal information-legal tensors used by the policy.
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
  parser.add_argument("--seed", type=int, default=20290001)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--min-trick-number", type=int, default=1)
  parser.add_argument("--max-trick-number", type=int, default=5)
  parser.add_argument("--lookback-decisions", type=int, default=40)
  parser.add_argument("--failed-rounds-only", action="store_true")
  parser.add_argument("--max-failure-rounds", type=int, default=200)
  parser.add_argument("--max-rows", type=int, default=500)
  parser.add_argument("--include-policy-aligned", action="store_true")
  parser.add_argument("--red-win-min-current-legal-advantage", type=int, default=2)
  parser.add_argument("--red-win-min-min-legal-advantage", type=int, default=1)
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


def action_color_idx(state, action: int) -> int | None:
  action = int(action)
  if action == 999:
    return None
  return action // int(state._num_card_types)


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


def pressure_advantage(candidate: dict, baseline: dict) -> tuple[int, int, int]:
  return (
      int(baseline.get("exact_forced_paradox_players_after", 0))
      - int(candidate.get("exact_forced_paradox_players_after", 0)),
      int(candidate.get("exact_min_nonparadox_legal_after", 0))
      - int(baseline.get("exact_min_nonparadox_legal_after", 0)),
      int(candidate.get("exact_current_nonparadox_legal_after", 0))
      - int(baseline.get("exact_current_nonparadox_legal_after", 0)),
  )


def red_win_is_justified(red_pressure: dict, best_follow_pressure: dict, args) -> bool:
  forced_delta, min_delta, current_delta = pressure_advantage(
      red_pressure, best_follow_pressure
  )
  return (
      forced_delta > 0
      or min_delta >= int(args.red_win_min_min_legal_advantage)
      or current_delta >= int(args.red_win_min_current_legal_advantage)
  )


def lane_label_key(action: int, pressure: dict, cert: dict,
                   best_follow_pressure: dict, args) -> tuple:
  follows_led = not bool(cert.get("lost_led_token", False))
  red_win = bool(cert.get("responsible_red_win_switch", False))
  justified_red = red_win and red_win_is_justified(
      pressure, best_follow_pressure, args
  )
  # Lane class dominates ordinary pressure counts: follow-led is the grammar
  # target. A red winning switch is a good lane burn, but it should only beat
  # follow-led when the exact forced-paradox count improves via the first key.
  if follows_led:
    lane_class = 3
  elif justified_red:
    lane_class = 2
  elif red_win:
    lane_class = 1
  else:
    lane_class = 0
  return (
      -int(pressure.get("exact_forced_paradox_players_after", 0)),
      lane_class,
      int(pressure.get("exact_min_nonparadox_legal_after", 0)),
      int(pressure.get("exact_current_nonparadox_legal_after", 0)),
      int(pressure.get("exact_total_nonparadox_legal_after", 0)),
      1 if bool(pressure.get("exact_terminal_after", False)) else 0,
      -int(action),
  )


def ranked_targets(legal_actions, pressure_by_action, cert_by_action,
                   best_follow_pressure, args):
  actions = [int(action) for action in legal_actions if int(action) != 999]
  ranked = sorted(
      actions,
      key=lambda action: lane_label_key(
          action,
          pressure_by_action[action],
          cert_by_action[action],
          best_follow_pressure,
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


def candidate_row(decision, allowed_phases, args):
  if str(decision.get("phase")) not in allowed_phases:
    return None
  trick = int(decision.get("trick_number", -1))
  if trick < int(args.min_trick_number):
    return None
  if int(args.max_trick_number) >= 0 and trick > int(args.max_trick_number):
    return None
  state = decision["state"]
  led_idx = led_color_idx(state)
  if led_idx is None or led_idx == RED_INDEX:
    return None
  legal = [int(action) for action in decision["legal"] if int(action) != 999]
  follow_actions = [
      action for action in legal if action_color_idx(state, action) == led_idx
  ]
  burn_actions = [
      action for action in legal if action_color_idx(state, action) != led_idx
  ]
  if not follow_actions or not burn_actions:
    return None

  selected_action = int(decision["action"])
  pressure_by_action = {}
  cert_by_action = {}
  player = int(decision["player"])
  for action in legal:
    pressure_by_action[action] = exact_legal_pressure_after_action(state, action)
    cert_by_action[action] = action_liveness_certificate(
        state, player, action, base_action=selected_action
    )

  best_follow = max(follow_actions, key=lambda action: pressure_key(
      pressure_by_action[action]
  ))
  best_follow_pressure = pressure_by_action[best_follow]
  value_targets, risk_targets, ranked = ranked_targets(
      legal,
      pressure_by_action,
      cert_by_action,
      best_follow_pressure,
      args,
  )
  if not ranked:
    return None
  best_action = int(ranked[-1])
  if (
      not bool(args.include_policy_aligned)
      and int(selected_action) == int(best_action)
  ):
    return None
  return {
      "best_action": best_action,
      "best_follow_action": int(best_follow),
      "best_follow_pressure": best_follow_pressure,
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
        mined = candidate_row(decision, allowed_phases, args)
        if mined is None:
          summary["decisions_skipped"] += 1
          continue
        state = decision["state"]
        selected_action = int(decision["action"])
        best_action = int(mined["best_action"])
        best_cert = mined["cert_by_action"][best_action]
        selected_cert = mined["cert_by_action"][selected_action]
        best_pressure = mined["pressure_by_action"][best_action]
        selected_pressure = mined["pressure_by_action"][selected_action]
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
        led_color = getattr(state, "_led_color", None)
        summary[f"row_led_{led_color}"] += 1
        if bool(selected_cert.get("lost_led_token", False)):
          summary["selected_lost_led_token"] += 1
        else:
          summary["selected_followed_led"] += 1
        if bool(best_cert.get("lost_led_token", False)):
          summary["best_lost_led_token"] += 1
        else:
          summary["best_followed_led"] += 1
        if bool(best_cert.get("responsible_red_win_switch", False)):
          summary["best_responsible_red_win_switch"] += 1
        if bool(selected_cert.get("responsible_red_win_switch", False)):
          summary["selected_responsible_red_win_switch"] += 1
        if best_action == int(mined["best_follow_action"]):
          summary["best_is_follow_led"] += 1
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
            "best_action": best_action,
            "best_action_string": action_string(
                state, int(decision["player"]), best_action
            ),
            "best_follow_action": int(mined["best_follow_action"]),
            "best_follow_action_string": action_string(
                state, int(decision["player"]), int(mined["best_follow_action"])
            ),
            "selected_pressure": selected_pressure,
            "best_pressure": best_pressure,
            "best_follow_pressure": mined["best_follow_pressure"],
            "selected_liveness": selected_cert,
            "best_liveness": best_cert,
            "ranked_actions": [int(action) for action in mined["ranked_actions"]],
        })
        if (
            int(args.progress_interval) > 0
            and summary["rows"] % int(args.progress_interval) == 0
        ):
          print(json.dumps({
              "iteration": "lane_burn_contrast_progress",
              "matches": int(match_idx + 1),
              "rounds": int(summary["rounds"]),
              "failure_rounds": int(summary["failure_rounds"]),
              "rows": int(summary["rows"]),
          }), flush=True)

  if not rows:
    raise SystemExit("No lane-burn contrast rows mined; relax filters")

  metadata = {
      "kind": "lane_burn_contrast_replay",
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
      "red_win_min_current_legal_advantage": int(
          args.red_win_min_current_legal_advantage
      ),
      "red_win_min_min_legal_advantage": int(
          args.red_win_min_min_legal_advantage
      ),
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
