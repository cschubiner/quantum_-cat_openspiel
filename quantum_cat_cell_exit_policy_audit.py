#!/usr/bin/env python3
"""Audit policy choices against cell-exit damage metadata.

Run760-style replays carry per-legal-action metadata describing whether an
action removes another player's last future exit in a still-tokened color. This
script measures what a checkpoint's masked policy argmax would choose on those
rows, then joins the choice back to that metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
)


def csv(value: str) -> list[str]:
  return [part.strip() for part in str(value).split(",") if part.strip()]


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
    sys.argv = ["quantum_cat_cell_exit_policy_audit"]
    return trainer_parse_args()
  finally:
    sys.argv = old_argv


def mean(values):
  if not values:
    return None
  return round(float(np.mean(values)), 6)


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


def rate(count: int, total: int) -> float:
  if total <= 0:
    return 0.0
  return round(count / total, 6)


def load_row_metadata(path: str, expected_rows: int):
  data = json.loads(Path(path).read_text())
  if not isinstance(data, list):
    raise ValueError(f"{path} must contain a JSON list")
  if len(data) != expected_rows:
    raise ValueError(
        f"{path} row count {len(data)} != replay rows {expected_rows}"
    )
  return data


def selected_actions(model, replay, device, batch_size: int):
  actions: list[int] = []
  probs: list[float] = []
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
      logits_t, _ = model(obs_t, features_t)
      logits_t = logits_t.masked_fill(~mask_t, -1e9)
      probs_t = torch.softmax(logits_t, dim=1)
      action_t = torch.argmax(logits_t, dim=1)
      prob_t = probs_t.gather(1, action_t[:, None]).squeeze(1)
    actions.extend(int(x) for x in action_t.detach().cpu().numpy())
    probs.extend(float(x) for x in prob_t.detach().cpu().numpy())
  return actions, probs


def damage_for(row_meta, action: int):
  damage = row_meta.get("action_damage", {}).get(str(int(action)))
  if damage is None:
    return {
        "action": int(action),
        "action_color": None,
        "other_zero_count": 0,
        "self_zero_count": 0,
        "total_zero_count": 0,
        "weighted_damage": 0.0,
        "missing_damage_metadata": True,
    }
  return damage


def summarize_checkpoint(name, checkpoint_path, model, replay, row_metadata, device, batch_size):
  predictions, prediction_probs = selected_actions(model, replay, device, batch_size)
  target_actions = [
      int(np.argmax(np.asarray(item[2], dtype=np.float32)))
      for item in replay
  ]
  anchor_actions = [
      int(row.get("policy_best_action", -1))
      for row in row_metadata
  ]
  deployed_actions = [
      int(row.get("deployment_target_action", row.get("policy_best_action", -1)))
      for row in row_metadata
  ]
  changed_rows = [
      bool(row.get("policy_target_changed", False))
      for row in row_metadata
  ]
  selected_damages = [
      damage_for(row, action)
      for row, action in zip(row_metadata, predictions)
  ]
  action_features = [
      np.asarray(item[11], dtype=np.float32)[action]
      for item, action in zip(replay, predictions)
  ]
  other_damage = [
      int(damage.get("other_zero_count", 0)) for damage in selected_damages
  ]
  total_damage = [
      int(damage.get("total_zero_count", 0)) for damage in selected_damages
  ]
  weighted_damage = [
      float(damage.get("weighted_damage", 0.0)) for damage in selected_damages
  ]
  predicted_color = [
      damage.get("action_color") for damage in selected_damages
  ]
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
  target_top1 = [pred == target for pred, target in zip(predictions, target_actions)]
  anchor_top1 = [pred == anchor for pred, anchor in zip(predictions, anchor_actions)]
  deployed_top1 = [
      pred == deployed for pred, deployed in zip(predictions, deployed_actions)
  ]

  changed_indices = [idx for idx, changed in enumerate(changed_rows) if changed]
  unchanged_indices = [idx for idx, changed in enumerate(changed_rows) if not changed]

  def subset_summary(indices):
    return {
        "rows": len(indices),
        "target_top1_rate": rate(sum(target_top1[idx] for idx in indices), len(indices)),
        "anchor_top1_rate": rate(sum(anchor_top1[idx] for idx in indices), len(indices)),
        "deployed_top1_rate": rate(sum(deployed_top1[idx] for idx in indices), len(indices)),
        "other_exit_damage_rate": rate(
            sum(other_damage[idx] > 0 for idx in indices), len(indices)
        ),
        "total_exit_damage_rate": rate(
            sum(total_damage[idx] > 0 for idx in indices), len(indices)
        ),
        "mean_other_zero_count": mean([other_damage[idx] for idx in indices]),
        "mean_weighted_damage": mean([weighted_damage[idx] for idx in indices]),
        "red_rate": rate(sum(is_red[idx] for idx in indices), len(indices)),
        "follows_led_rate": rate(sum(follows_led[idx] for idx in indices), len(indices)),
        "off_led_loses_token_rate": rate(
            sum(off_led_loses_token[idx] for idx in indices), len(indices)
        ),
        "newly_loses_led_token_rate": rate(
            sum(newly_loses_led[idx] for idx in indices), len(indices)
        ),
    }

  rows = len(replay)
  return {
      "checkpoint": name,
      "checkpoint_path": checkpoint_path,
      "rows": rows,
      "target_top1_rate": rate(sum(target_top1), rows),
      "anchor_top1_rate": rate(sum(anchor_top1), rows),
      "deployed_top1_rate": rate(sum(deployed_top1), rows),
      "other_exit_damage_rate": rate(sum(value > 0 for value in other_damage), rows),
      "total_exit_damage_rate": rate(sum(value > 0 for value in total_damage), rows),
      "mean_other_zero_count": mean(other_damage),
      "mean_total_zero_count": mean(total_damage),
      "mean_weighted_damage": mean(weighted_damage),
      "mean_selected_probability": mean(prediction_probs),
      "red_rate": rate(sum(is_red), rows),
      "follows_led_rate": rate(sum(follows_led), rows),
      "off_led_loses_token_rate": rate(sum(off_led_loses_token), rows),
      "newly_loses_led_token_rate": rate(sum(newly_loses_led), rows),
      "predicted_color": distribution(predicted_color),
      "changed_rows": subset_summary(changed_indices),
      "unchanged_rows": subset_summary(unchanged_indices),
      "sample_predictions": [
          {
              "row": idx,
              "prediction": int(predictions[idx]),
              "target": int(target_actions[idx]),
              "anchor": int(anchor_actions[idx]),
              "policy_target_changed": bool(changed_rows[idx]),
              "damage": selected_damages[idx],
          }
          for idx in range(min(10, rows))
      ],
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoints", required=True)
  parser.add_argument("--replay", required=True)
  parser.add_argument("--metadata", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--match-context", action="store_true", default=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--batch-size", type=int, default=256)
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

  rows = []
  for name, checkpoint_path in parse_named_paths(args.checkpoints).items():
    model, _, saved_args = load_model_payload(
        checkpoint_path, game, trainer_args, device
    )
    model.eval()
    summary = summarize_checkpoint(
        name, checkpoint_path, model, replay, row_metadata, device, args.batch_size
    )
    summary["saved_arch"] = saved_args.get("arch")
    summary["saved_width"] = saved_args.get("width")
    summary["saved_depth"] = saved_args.get("depth")
    rows.append(summary)

  output = {
      "replay": args.replay,
      "metadata": args.metadata,
      "replay_metadata": replay_metadata,
      "rows": rows,
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
      "out": str(out_path),
      "rows": [
          {
              "checkpoint": row["checkpoint"],
              "target_top1_rate": row["target_top1_rate"],
              "other_exit_damage_rate": row["other_exit_damage_rate"],
              "total_exit_damage_rate": row["total_exit_damage_rate"],
              "mean_other_zero_count": row["mean_other_zero_count"],
              "red_rate": row["red_rate"],
              "follows_led_rate": row["follows_led_rate"],
              "newly_loses_led_token_rate": row[
                  "newly_loses_led_token_rate"
              ],
              "changed_rows": {
                  "target_top1_rate": row["changed_rows"][
                      "target_top1_rate"
                  ],
                  "other_exit_damage_rate": row["changed_rows"][
                      "other_exit_damage_rate"
                  ],
              },
              "unchanged_rows": {
                  "target_top1_rate": row["unchanged_rows"][
                      "target_top1_rate"
                  ],
                  "other_exit_damage_rate": row["unchanged_rows"][
                      "other_exit_damage_rate"
                  ],
              },
          }
          for row in rows
      ],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
