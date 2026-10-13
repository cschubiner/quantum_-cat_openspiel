#!/usr/bin/env python3
"""Mine table-wide color/rank exit-protection labels.

Run757-759 showed that the deployed 4p bot follows led when it can, but often
arrives later with a non-red led token still available and zero playable cards
in that led color. Run759 traced those future bottlenecks to earlier plays that
claimed the exact color/rank cell another player later needed.

This miner turns that diagnosis into dense action-conditioned labels. For each
live play decision, every legal action is applied to a clone and scored by how
many target player/color pairs it would reduce from at least one potential
future action to zero while the target token remains available. The labels use
hidden hand state at training time only; the saved observations and action
features remain the normal information-legal model inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from quantum_cat_full_match_elo import load_neural


COLOR_NAMES = ("R", "B", "Y", "G")
PROTECTED_COLORS = ("B", "Y", "G")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--matches", type=int, default=40)
  parser.add_argument("--seed", type=int, default=20260616)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--min-trick-number", type=int, default=0)
  parser.add_argument("--max-trick-number", type=int, default=7)
  parser.add_argument("--max-rows", type=int, default=1000)
  parser.add_argument("--max-rows-per-round", type=int, default=0)
  parser.add_argument("--collect-policy-damage-only", action="store_true")
  parser.add_argument("--include-self-targets", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      "--anchor-policy-when-no-other-damage",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Keep the policy target on the deployed action unless that action "
          "zeros another player's future color exit. This prevents the label "
          "bank from drifting into generic red preference just because red "
          "does not claim non-red cells."
      ),
  )
  parser.add_argument("--other-damage-weight", type=float, default=1.0)
  parser.add_argument("--self-damage-weight", type=float, default=0.5)
  parser.add_argument("--paradox-action-damage", type=float, default=3.0)
  parser.add_argument("--damage-normalizer", type=float, default=2.0)
  parser.add_argument("--safe-action-value-target", type=float, default=0.85)
  parser.add_argument("--risky-action-value-target", type=float, default=-0.85)
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
  parser.add_argument("--progress-interval", type=int, default=10)
  parser.add_argument("--sample-limit", type=int, default=24)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--metadata-out", default="")
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  return parser.parse_args()


def color_index(color: str) -> int:
  return COLOR_NAMES.index(str(color))


def action_color_idx(state, action: int) -> int | None:
  action = int(action)
  if action == 999:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  if num_card_types <= 0:
    return None
  color_idx = action // num_card_types
  return int(color_idx) if 0 <= color_idx < len(COLOR_NAMES) else None


def action_color(state, action: int) -> str | None:
  idx = action_color_idx(state, action)
  return None if idx is None else COLOR_NAMES[idx]


def action_rank(state, action: int) -> int | None:
  action = int(action)
  if action == 999:
    return None
  num_card_types = int(getattr(state, "_num_card_types", 0) or 0)
  if num_card_types <= 0:
    return None
  return int(action % num_card_types) + 1


def token_available(state, player: int, color: str) -> bool:
  tokens = np.asarray(getattr(state, "_color_tokens"), dtype=bool)
  color_idx = color_index(color)
  return bool(
      0 <= int(player) < tokens.shape[0]
      and color_idx < tokens.shape[1]
      and tokens[int(player), color_idx]
  )


def potential_color_action_count(state, player: int, color: str) -> int:
  player = int(player)
  color_idx = color_index(color)
  if not token_available(state, player, color):
    return 0
  hand = np.asarray(getattr(state, "_hands")[player], dtype=int)
  board = np.asarray(getattr(state, "_board_ownership"), dtype=int)
  count = 0
  for rank_idx, qty in enumerate(hand):
    if int(qty) <= 0:
      continue
    if int(board[color_idx, rank_idx]) == -1:
      count += 1
  return int(count)


def target_had_action_rank(state, target_player: int, action: int) -> bool:
  rank = action_rank(state, action)
  if rank is None:
    return False
  rank_idx = int(rank) - 1
  hand = np.asarray(getattr(state, "_hands")[int(target_player)], dtype=int)
  return bool(0 <= rank_idx < hand.shape[0] and int(hand[rank_idx]) > 0)


def exit_damage_for_action(state, actor: int, action: int, args):
  action = int(action)
  if action == 999:
    return {
        "action": action,
        "action_color": None,
        "action_rank": None,
        "is_paradox_action": True,
        "details": [],
        "other_zero_count": 0,
        "self_zero_count": 0,
        "total_zero_count": 0,
        "weighted_damage": float(args.paradox_action_damage),
    }
  before = state
  after = state.clone()
  after.apply_action(action)
  actor = int(actor)
  details = []
  for target in range(int(args.players)):
    if target == actor and not bool(args.include_self_targets):
      continue
    for color in PROTECTED_COLORS:
      before_count = potential_color_action_count(before, target, color)
      if before_count <= 0:
        continue
      after_count = potential_color_action_count(after, target, color)
      if after_count != 0:
        continue
      if not token_available(after, target, color):
        continue
      details.append({
          "target_player": int(target),
          "target_relation": "self" if target == actor else "other",
          "color": color,
          "before_count": int(before_count),
          "after_count": int(after_count),
          "target_had_action_rank_before": bool(
              target_had_action_rank(before, target, action)
          ),
      })
  other_zero = sum(1 for item in details if item["target_relation"] == "other")
  self_zero = sum(1 for item in details if item["target_relation"] == "self")
  weighted = (
      float(other_zero) * float(args.other_damage_weight)
      + float(self_zero) * float(args.self_damage_weight)
  )
  return {
      "action": action,
      "action_color": action_color(before, action),
      "action_rank": action_rank(before, action),
      "is_paradox_action": False,
      "details": details,
      "other_zero_count": int(other_zero),
      "self_zero_count": int(self_zero),
      "total_zero_count": int(other_zero + self_zero),
      "weighted_damage": float(weighted),
  }


def target_values_for_damage(damage_by_action, legal, args):
  safe_value = float(args.safe_action_value_target)
  risky_value = float(args.risky_action_value_target)
  safe_risk = float(args.safe_risk_target)
  risky_risk = float(args.risky_risk_target)
  normalizer = max(1e-6, float(args.damage_normalizer))
  value_targets = {}
  risk_targets = {}
  for action in legal:
    action = int(action)
    damage = float(damage_by_action[action]["weighted_damage"])
    frac = min(1.0, max(0.0, damage / normalizer))
    value_targets[action] = safe_value + frac * (risky_value - safe_value)
    risk_targets[action] = safe_risk + frac * (risky_risk - safe_risk)
  return value_targets, risk_targets


def best_protection_action(decision, damage_by_action, args):
  legal = [int(action) for action in decision["legal"]]
  policy = np.asarray(decision.get("policy"), dtype=np.float32)
  selected = int(decision["action"])
  if (
      bool(args.anchor_policy_when_no_other_damage)
      and int(damage_by_action[selected]["other_zero_count"]) <= 0
  ):
    return selected
  led_color = getattr(decision["state"], "_led_color", None)

  def key(action):
    row = damage_by_action[int(action)]
    policy_prob = float(policy[int(action)]) if int(action) < policy.shape[0] else 0.0
    follows_led = bool(
        led_color in PROTECTED_COLORS
        and action_color(decision["state"], action) == led_color
    )
    is_red = bool(action_color(decision["state"], action) == "R")
    return (
        int(bool(row["is_paradox_action"])),
        int(row["other_zero_count"]),
        int(row["total_zero_count"]),
        float(row["weighted_damage"]),
        -int(follows_led),
        int(is_red),
        -policy_prob,
        int(action),
    )

  return int(min(legal, key=key))


def should_collect(decision, damage_by_action, best_action, args):
  selected = int(decision["action"])
  selected_damage = damage_by_action[selected]
  best_damage = damage_by_action[int(best_action)]
  legal = [int(action) for action in decision["legal"]]
  min_damage = min(float(damage_by_action[action]["weighted_damage"]) for action in legal)
  max_damage = max(float(damage_by_action[action]["weighted_damage"]) for action in legal)
  if max_damage <= min_damage:
    return False
  if bool(args.collect_policy_damage_only):
    return bool(selected_damage["other_zero_count"] > 0)
  return (
      bool(selected_damage["other_zero_count"] > best_damage["other_zero_count"])
      or bool(selected_damage["total_zero_count"] > best_damage["total_zero_count"])
      or bool(max_damage > 0.0 and min_damage == 0.0)
  )


def build_replay_row(decision, best_action, damage_by_action, args):
  state = decision["state"]
  player = int(decision["player"])
  num_actions = int(state.num_distinct_actions())
  legal = [int(action) for action in decision["legal"]]
  legal_mask = np.zeros(num_actions, dtype=np.float32)
  legal_mask[legal] = 1.0
  policy_target = np.zeros(num_actions, dtype=np.float32)
  policy_target[int(best_action)] = 1.0
  value_targets, risk_targets = target_values_for_damage(
      damage_by_action, legal, args
  )
  action_value_targets = np.zeros(num_actions, dtype=np.float32)
  action_value_mask = np.zeros(num_actions, dtype=np.float32)
  action_paradox_targets = np.zeros(num_actions, dtype=np.float32)
  action_paradox_mask = np.zeros(num_actions, dtype=np.float32)
  for action in legal:
    action_value_targets[action] = float(value_targets[action])
    action_value_mask[action] = 1.0
    action_paradox_targets[action] = float(risk_targets[action])
    action_paradox_mask[action] = 1.0
  return (
      np.array(state.observation_tensor(player), dtype=np.float32),
      legal_mask,
      policy_target,
      np.zeros(int(args.players), dtype=np.float32),
      np.zeros(int(args.players), dtype=np.float32),
      int(best_action),
      int(player),
      action_paradox_targets,
      action_paradox_mask,
      action_value_targets,
      action_value_mask,
      action_feature_matrix(state, player, num_actions),
  )


def row_metadata(decision, best_action, damage_by_action, match_idx, round_idx,
                 start_player):
  selected = int(decision["action"])
  best_action = int(best_action)
  legal = [int(action) for action in decision["legal"]]
  damaging_actions = [
      int(action) for action in legal
      if int(damage_by_action[action]["other_zero_count"]) > 0
  ]
  safe_actions = [
      int(action) for action in legal
      if float(damage_by_action[action]["weighted_damage"]) <= 0.0
  ]
  return {
      "phase": str(decision["phase"]),
      "player": int(decision["player"]),
      "group": {
          "proposal_source": "cell_exit_protection",
          "match_index": int(match_idx),
          "round_index": int(round_idx),
          "decision_index": int(decision["index"]),
          "start_player": int(start_player),
      },
      "match_round": int(getattr(decision["state"], "_match_round", -1)),
      "round_start_player": int(getattr(decision["state"], "_round_start_player", -1)),
      "trick_number": int(decision["trick_number"]),
      "legal_count": int(len(legal)),
      "policy_best_action": int(selected),
      "policy_best_action_string": str(decision["action_string"]),
      "label_best_action": int(best_action),
      "label_best_action_string": action_string(
          decision["state"], int(decision["player"]), best_action
      ),
      "deployment_target_action": int(best_action),
      "deployment_target_action_string": action_string(
          decision["state"], int(decision["player"]), best_action
      ),
      "policy_target_changed": bool(best_action != selected),
      "selected_damage": damage_by_action[selected],
      "best_damage": damage_by_action[best_action],
      "damaging_other_exit_actions": damaging_actions,
      "zero_damage_actions": safe_actions,
      "candidate_actions": legal,
      "action_damage": {
          str(action): damage_by_action[action] for action in legal
      },
      "calibration_class": (
          "POSITIVE" if best_action != selected else "ANCHOR_SAFE"
      ),
      "false_positive_for_positive_gate": False,
      "label_margin_vs_policy": (
          float(damage_by_action[selected]["weighted_damage"])
          - float(damage_by_action[best_action]["weighted_damage"])
      ),
      "audit_margin_vs_policy": (
          float(damage_by_action[selected]["weighted_damage"])
          - float(damage_by_action[best_action]["weighted_damage"])
      ),
      "audit_paradox_delta_vs_policy": (
          float(damage_by_action[best_action]["weighted_damage"])
          - float(damage_by_action[selected]["weighted_damage"])
      ),
      "candidate_count": int(len(legal)),
      "candidate_set_contains_label_best": True,
  }


def increment_damage_counters(summary, prefix, damage):
  summary[f"{prefix}_other_zero_sum"] += int(damage["other_zero_count"])
  summary[f"{prefix}_self_zero_sum"] += int(damage["self_zero_count"])
  summary[f"{prefix}_total_zero_sum"] += int(damage["total_zero_count"])
  if int(damage["other_zero_count"]) > 0:
    summary[f"{prefix}_other_zero_actions"] += 1
  if int(damage["total_zero_count"]) > 0:
    summary[f"{prefix}_total_zero_actions"] += 1


def inc(mapping, key, amount=1):
  mapping[str(key)] += int(amount)


def main():
  args = parse_args()
  if int(args.players) < 2 or int(args.players) > 5:
    raise ValueError("--players must be 2..5")
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
  replay_rows = []
  metadata_rows = []
  samples = []
  summary = Counter()
  by_trick = defaultdict(int)
  by_phase = defaultdict(int)

  for match_idx in range(int(args.matches)):
    if int(args.max_rows) > 0 and len(replay_rows) >= int(args.max_rows):
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_idx in range(int(args.players)):
      if int(args.max_rows) > 0 and len(replay_rows) >= int(args.max_rows):
        break
      start_player = (match_idx + round_idx) % int(args.players)
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": int(args.players),
              "start_player": int(start_player),
              "match_context": int(bool(args.match_context)),
          },
      )
      prior_totals = np.copy(totals)
      terminal, decisions, _ = play_round_with_history(
          model,
          game,
          seed=int(args.seed) + match_idx * 1000003 + round_idx * 9973,
          value_scale=value_scale,
          device=device,
          match_totals=prior_totals if bool(args.match_context) else None,
          round_index=round_idx,
          args=args,
      )
      result = terminal_round_result(terminal)
      returns = np.array(terminal.returns(), dtype=np.float32)
      totals = returns if bool(args.match_context) else totals + returns
      summary["rounds"] += 1
      if result["any_paradox"]:
        summary["paradox_rounds"] += 1
      rows_this_round = 0
      for decision in decisions:
        if int(args.max_rows) > 0 and len(replay_rows) >= int(args.max_rows):
          break
        if (
            int(args.max_rows_per_round) > 0
            and rows_this_round >= int(args.max_rows_per_round)
        ):
          break
        phase = str(decision["phase"]).lower()
        if allowed_phases and phase not in allowed_phases:
          continue
        trick = int(decision["trick_number"])
        if trick < int(args.min_trick_number):
          continue
        if int(args.max_trick_number) >= 0 and trick > int(args.max_trick_number):
          continue
        if len(decision["legal"]) <= 1:
          continue
        summary["decisions_considered"] += 1
        damage_by_action = {
            int(action): exit_damage_for_action(
                decision["state"], int(decision["player"]), int(action), args
            )
            for action in decision["legal"]
        }
        selected = int(decision["action"])
        increment_damage_counters(summary, "selected", damage_by_action[selected])
        for action, damage in damage_by_action.items():
          increment_damage_counters(summary, "legal", damage)
          if int(damage["other_zero_count"]) > 0:
            summary["legal_actions_with_other_exit_damage"] += 1
          if float(damage["weighted_damage"]) <= 0:
            summary["legal_zero_damage_actions"] += 1
        best_action = best_protection_action(decision, damage_by_action, args)
        if not should_collect(decision, damage_by_action, best_action, args):
          continue
        summary["rows_collected"] += 1
        rows_this_round += 1
        if best_action != selected:
          summary["policy_target_changed"] += 1
        if int(damage_by_action[selected]["other_zero_count"]) > 0:
          summary["selected_other_exit_damage_rows"] += 1
        if int(damage_by_action[best_action]["other_zero_count"]) == 0:
          summary["best_has_no_other_exit_damage_rows"] += 1
        replay_rows.append(
            build_replay_row(decision, best_action, damage_by_action, args)
        )
        metadata = row_metadata(
            decision,
            best_action,
            damage_by_action,
            match_idx,
            round_idx,
            start_player,
        )
        metadata["source_index"] = int(len(metadata_rows))
        metadata_rows.append(metadata)
        inc(by_trick, trick)
        inc(by_phase, phase)
        if len(samples) < int(args.sample_limit):
          samples.append(metadata)
      if (
          int(args.progress_interval) > 0
          and summary["rounds"] % int(args.progress_interval) == 0
      ):
        print(json.dumps({
            "rounds": int(summary["rounds"]),
            "rows_collected": int(summary["rows_collected"]),
            "decisions_considered": int(summary["decisions_considered"]),
        }), flush=True)

  out_path = Path(args.out)
  metadata_path = (
      Path(args.metadata_out)
      if str(args.metadata_out or "").strip()
      else out_path.with_suffix(out_path.suffix + ".metadata.json")
  )
  summary_path = (
      Path(args.summary_out)
      if str(args.summary_out or "").strip()
      else out_path.with_suffix(out_path.suffix + ".summary.json")
  )
  rates = {}
  rounds = int(summary.get("rounds", 0))
  decisions = int(summary.get("decisions_considered", 0))
  rows = int(summary.get("rows_collected", 0))
  if rounds:
    rates["same_policy_paradox_round_rate"] = round(
        float(summary.get("paradox_rounds", 0)) / rounds, 4
    )
    rates["rows_per_round"] = round(float(rows) / rounds, 4)
  if decisions:
    rates["row_collection_rate"] = round(float(rows) / decisions, 4)
    rates["selected_other_exit_damage_rate"] = round(
        float(summary.get("selected_other_exit_damage_rows", 0)) / max(1, rows),
        4,
    )
  if rows:
    rates["policy_target_changed_rate"] = round(
        float(summary.get("policy_target_changed", 0)) / rows, 4
    )
    rates["best_no_other_exit_damage_rate"] = round(
        float(summary.get("best_has_no_other_exit_damage_rows", 0)) / rows, 4
    )

  summary_json = {
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "bot_mode": str(args.bot_mode),
      "include_self_targets": bool(args.include_self_targets),
      "anchor_policy_when_no_other_damage": bool(
          args.anchor_policy_when_no_other_damage
      ),
      "summary": dict(summary),
      "rates": rates,
      "by_trick": dict(by_trick),
      "by_phase": dict(by_phase),
      "replay_out": str(out_path),
      "metadata_out": str(metadata_path),
      "row_group_ids": [
          str(row["group"]["match_index"])
          + ":"
          + str(row["group"]["round_index"])
          + ":"
          + str(row["group"]["decision_index"])
          for row in metadata_rows
      ],
      "samples": samples,
  }
  save_replay(replay_rows, out_path, metadata=summary_json)
  metadata_path.parent.mkdir(parents=True, exist_ok=True)
  metadata_path.write_text(
      json.dumps(metadata_rows, indent=2, sort_keys=True) + "\n"
  )
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(
      json.dumps(summary_json, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(summary_json, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
