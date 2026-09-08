#!/usr/bin/env python3
"""Sweep q-policy selector knobs on a cell-exit replay.

This is an offline deployment-shape audit: it uses the same q-policy selector
math as live play, but applies it to replay rows with Run760-style
per-action cell-exit damage metadata.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    APPENDED_ACTION_FEATURE_INDEX,
    adapt_observation_batch,
    load_model_payload,
    load_replay,
    make_game,
    model_input_size,
    parse_args as trainer_parse_args,
    q_policy_select_action,
)


def csv(value: str) -> list[str]:
  return [part.strip() for part in str(value).split(",") if part.strip()]


def floats(value: str) -> list[float]:
  return [float(part) for part in csv(value)]


def parse_named_paths(spec: str) -> dict[str, str]:
  paths: dict[str, str] = {}
  for item in csv(spec):
    if "=" in item:
      name, path = item.split("=", 1)
      paths[name.strip()] = path.strip()
    else:
      path = item.strip()
      paths[Path(path).stem] = path
  return paths


def trainer_defaults():
  old_argv = sys.argv[:]
  try:
    sys.argv = ["quantum_cat_cell_exit_selector_sweep"]
    return trainer_parse_args()
  finally:
    sys.argv = old_argv


def rate(count: int, total: int) -> float:
  if total <= 0:
    return 0.0
  return round(count / total, 6)


def mean(values):
  if not values:
    return None
  return round(float(np.mean(values)), 6)


def damage_for(row_meta, action: int):
  return row_meta.get("action_damage", {}).get(str(int(action)), {
      "action": int(action),
      "action_color": None,
      "other_zero_count": 0,
      "self_zero_count": 0,
      "total_zero_count": 0,
      "weighted_damage": 0.0,
      "missing_damage_metadata": True,
  })


def distribution(values):
  counts: dict[str, int] = {}
  for value in values:
    key = "none" if value is None else str(value)
    counts[key] = counts.get(key, 0) + 1
  total = max(1, sum(counts.values()))
  return {
      key: {"count": count, "rate": round(count / total, 6)}
      for key, count in sorted(counts.items())
  }


def load_row_metadata(path: str, expected_rows: int):
  data = json.loads(Path(path).read_text())
  if not isinstance(data, list):
    raise ValueError(f"{path} must contain a JSON list")
  if len(data) != expected_rows:
    raise ValueError(
        f"{path} row count {len(data)} != replay rows {expected_rows}"
    )
  return data


def model_outputs(model, replay, device, batch_size: int):
  policies = []
  action_values = []
  action_risks = []
  for start in range(0, len(replay), batch_size):
    batch = replay[start:start + batch_size]
    obs = np.asarray([item[0] for item in batch], dtype=np.float32)
    masks = np.asarray([item[1] for item in batch], dtype=bool)
    action_features = np.asarray([item[11] for item in batch], dtype=np.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    obs_t = adapt_observation_batch(obs_t, model_input_size(model))
    features_t = torch.tensor(action_features, dtype=torch.float32, device=device)
    mask_t = torch.tensor(masks, dtype=torch.bool, device=device)
    with torch.no_grad():
      logits_t, _, _, risk_logits_t, value_t = model.forward_with_all_aux(
          obs_t, features_t
      )
      logits_t = logits_t.masked_fill(~mask_t, -1e9)
      policy_t = torch.softmax(logits_t, dim=1)
      risk_t = torch.sigmoid(risk_logits_t)
    policies.append(policy_t.detach().cpu().numpy())
    action_values.append(value_t.detach().cpu().numpy())
    action_risks.append(risk_t.detach().cpu().numpy())
  return (
      np.concatenate(policies, axis=0),
      np.concatenate(action_values, axis=0),
      np.concatenate(action_risks, axis=0),
  )


def summarize_selected(name, checkpoint_name, selected_actions, baseline_actions, replay, row_metadata):
  target_actions = [
      int(np.argmax(np.asarray(item[2], dtype=np.float32)))
      for item in replay
  ]
  changed_rows = [
      bool(row.get("policy_target_changed", False))
      for row in row_metadata
  ]
  damages = [
      damage_for(row, action)
      for row, action in zip(row_metadata, selected_actions)
  ]
  action_features = [
      np.asarray(item[11], dtype=np.float32)[action]
      for item, action in zip(replay, selected_actions)
  ]
  other_damage = [int(d.get("other_zero_count", 0)) for d in damages]
  total_damage = [int(d.get("total_zero_count", 0)) for d in damages]
  weighted = [float(d.get("weighted_damage", 0.0)) for d in damages]
  colors = [d.get("action_color") for d in damages]
  follows_led = [
      float(features[ACTION_FEATURE_FOLLOWS_LED_INDEX]) > 0.5
      for features in action_features
  ]
  is_red = [
      float(features[ACTION_FEATURE_IS_RED_INDEX]) > 0.5
      for features in action_features
  ]
  off_led_loses_token = [
      float(features[ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX]) > 0.5
      for features in action_features
  ]
  token_loss_idx = APPENDED_ACTION_FEATURE_INDEX["token_loss_newly_loses_led"]
  newly_loses_led = [
      float(features[token_loss_idx]) > 0.5
      for features in action_features
  ]
  target_top1 = [
      selected == target
      for selected, target in zip(selected_actions, target_actions)
  ]
  overrides = [
      selected != baseline
      for selected, baseline in zip(selected_actions, baseline_actions)
  ]

  def subset(indices):
    return {
        "rows": len(indices),
        "target_top1_rate": rate(sum(target_top1[i] for i in indices), len(indices)),
        "override_rate": rate(sum(overrides[i] for i in indices), len(indices)),
        "other_exit_damage_rate": rate(
            sum(other_damage[i] > 0 for i in indices), len(indices)
        ),
        "total_exit_damage_rate": rate(
            sum(total_damage[i] > 0 for i in indices), len(indices)
        ),
        "mean_other_zero_count": mean([other_damage[i] for i in indices]),
        "mean_weighted_damage": mean([weighted[i] for i in indices]),
        "red_rate": rate(sum(is_red[i] for i in indices), len(indices)),
        "follows_led_rate": rate(sum(follows_led[i] for i in indices), len(indices)),
        "newly_loses_led_token_rate": rate(
            sum(newly_loses_led[i] for i in indices), len(indices)
        ),
    }

  rows = len(replay)
  changed = [idx for idx, flag in enumerate(changed_rows) if flag]
  unchanged = [idx for idx, flag in enumerate(changed_rows) if not flag]
  return {
      "name": name,
      "checkpoint": checkpoint_name,
      "rows": rows,
      "target_top1_rate": rate(sum(target_top1), rows),
      "override_rate": rate(sum(overrides), rows),
      "other_exit_damage_rate": rate(sum(x > 0 for x in other_damage), rows),
      "total_exit_damage_rate": rate(sum(x > 0 for x in total_damage), rows),
      "mean_other_zero_count": mean(other_damage),
      "mean_total_zero_count": mean(total_damage),
      "mean_weighted_damage": mean(weighted),
      "red_rate": rate(sum(is_red), rows),
      "follows_led_rate": rate(sum(follows_led), rows),
      "off_led_loses_token_rate": rate(sum(off_led_loses_token), rows),
      "newly_loses_led_token_rate": rate(sum(newly_loses_led), rows),
      "predicted_color": distribution(colors),
      "changed_rows": subset(changed),
      "unchanged_rows": subset(unchanged),
  }


def selector_args(base_args, mode, risk_penalty, risk_threshold, min_margin, max_gap, value_weight):
  return SimpleNamespace(
      action_paradox_rerank_mode=mode,
      action_paradox_selection_penalty=float(risk_penalty),
      action_paradox_risk_threshold=float(risk_threshold),
      action_paradox_min_risk_margin=float(min_margin),
      action_paradox_max_policy_log_gap=float(max_gap),
      action_value_selection_weight=float(value_weight),
      action_value_max_policy_log_gap=float(max_gap),
      action_feasibility_selection_weight=0.0,
      value_scale=float(getattr(base_args, "value_scale", 1.0)),
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--name", default="candidate")
  parser.add_argument("--replay", required=True)
  parser.add_argument("--metadata", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--match-context", action="store_true", default=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--modes", default="threshold,relative_dominance")
  parser.add_argument("--risk-penalties", default="0.0,0.5,1.0,2.0,4.0")
  parser.add_argument("--risk-thresholds", default="0.0,0.25,0.5")
  parser.add_argument("--min-risk-margins", default="0.05,0.10,0.20")
  parser.add_argument("--policy-log-gaps", default="0.5,1.0,2.0")
  parser.add_argument("--value-weights", default="0.0,0.25,0.5")
  args = parser.parse_args()

  trainer_args = trainer_defaults()
  trainer_args.players = args.players
  trainer_args.full_match_training = True
  trainer_args.match_context = bool(args.match_context)
  trainer_args.arch = "action_attn"
  trainer_args.width = 256
  trainer_args.depth = 3
  trainer_args.device = args.device

  device = torch.device(args.device)
  replay, replay_metadata = load_replay(args.replay, return_metadata=True)
  row_metadata = load_row_metadata(args.metadata, len(replay))
  game = make_game(trainer_args)
  model, _, saved_args = load_model_payload(args.checkpoint, game, trainer_args, device)
  model.eval()
  policies, values, risks = model_outputs(model, replay, device, args.batch_size)
  masks = np.asarray([item[1] for item in replay], dtype=bool)
  legal_actions = [list(np.flatnonzero(mask)) for mask in masks]
  baseline_actions = [
      max(legal, key=lambda action: float(policy[action]))
      for legal, policy in zip(legal_actions, policies)
  ]

  rows = [
      summarize_selected(
          "raw_policy",
          args.name,
          baseline_actions,
          baseline_actions,
          replay,
          row_metadata,
      )
  ]
  base_value_scale = float(getattr(trainer_args, "value_scale", 1.0))
  for mode, risk_penalty, risk_threshold, min_margin, max_gap, value_weight in itertools.product(
      csv(args.modes),
      floats(args.risk_penalties),
      floats(args.risk_thresholds),
      floats(args.min_risk_margins),
      floats(args.policy_log_gaps),
      floats(args.value_weights),
  ):
    if mode in ("threshold", "relative", "relative_dominance"):
      effective_risk_penalty = risk_penalty
    else:
      effective_risk_penalty = risk_penalty
    selector_config = selector_args(
        trainer_args,
        mode,
        effective_risk_penalty,
        risk_threshold,
        min_margin,
        max_gap,
        value_weight,
    )
    selected = []
    for legal, policy, action_value_row, action_risk_row in zip(
        legal_actions, policies, values, risks
    ):
      selected.append(q_policy_select_action(
          legal,
          policy,
          action_value_row,
          action_risk_row,
          selector_config,
          base_value_scale,
      ))
    row = summarize_selected(
        (
            f"{mode}:risk_penalty={risk_penalty:g},"
            f"threshold={risk_threshold:g},margin={min_margin:g},"
            f"gap={max_gap:g},value={value_weight:g}"
        ),
        args.name,
        selected,
        baseline_actions,
        replay,
        row_metadata,
    )
    row["selector"] = {
        "mode": mode,
        "risk_penalty": risk_penalty,
        "risk_threshold": risk_threshold,
        "min_risk_margin": min_margin,
        "policy_log_gap": max_gap,
        "value_weight": value_weight,
    }
    rows.append(row)

  output = {
      "checkpoint": args.checkpoint,
      "name": args.name,
      "saved_args": {
          "arch": saved_args.get("arch"),
          "width": saved_args.get("width"),
          "depth": saved_args.get("depth"),
      },
      "replay": args.replay,
      "metadata": args.metadata,
      "replay_metadata": replay_metadata,
      "rows": rows,
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
  leaderboard = sorted(
      rows,
      key=lambda row: (
          row["other_exit_damage_rate"],
          row["newly_loses_led_token_rate"],
          -row["target_top1_rate"],
          row["override_rate"],
      ),
  )[:20]
  print(json.dumps({
      "out": str(out_path),
      "leaderboard": [
          {
              "name": row["name"],
              "target_top1_rate": row["target_top1_rate"],
              "override_rate": row["override_rate"],
              "other_exit_damage_rate": row["other_exit_damage_rate"],
              "total_exit_damage_rate": row["total_exit_damage_rate"],
              "red_rate": row["red_rate"],
              "follows_led_rate": row["follows_led_rate"],
              "newly_loses_led_token_rate": row["newly_loses_led_token_rate"],
              "changed_other_exit_damage_rate": row["changed_rows"][
                  "other_exit_damage_rate"
              ],
              "unchanged_other_exit_damage_rate": row["unchanged_rows"][
                  "other_exit_damage_rate"
              ],
          }
          for row in leaderboard
      ],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
