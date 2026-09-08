#!/usr/bin/env python3
"""Evaluate checkpoints across Q-label validation families.

This is intentionally separate from the trainer.  It answers two questions:

1. Which checkpoint ranks labeled actions best on each validation family?
2. What tactical features characterize each family's Q-best actions?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_SIZE,
    APPENDED_ACTION_FEATURE_INDEX,
    APPENDED_ACTION_FEATURE_NAMES,
    APPENDED_ACTION_FEATURE_START,
    MAX_COLOR_FEATURES,
    MAX_RANK_FEATURES,
    filter_action_value_labels_for_training,
    load_model_payload,
    load_replay,
    make_game,
    maybe_add_policy_target_report,
    parse_args as trainer_parse_args,
    q_label_policy_report,
)


PHASE_NAMES = ["chance", "discard", "prediction", "play", "terminal"]
ACTION_TYPE_NAMES = ["discard", "prediction", "play", "paradox", "other"]


def trainer_defaults():
  old_argv = sys.argv[:]
  try:
    sys.argv = ["quantum_cat_validation_family_audit"]
    return trainer_parse_args()
  finally:
    sys.argv = old_argv


def base_feature_names():
  names = ["is_legal"]
  names.extend(f"phase_{name}" for name in PHASE_NAMES)
  names.extend(f"type_{name}" for name in ACTION_TYPE_NAMES)
  names.extend(f"rank_{idx + 1}" for idx in range(MAX_RANK_FEATURES))
  names.extend(f"color_{name}" for name in ["R", "B", "Y", "G"][:MAX_COLOR_FEATURES])
  names.extend(["led_none", "led_R", "led_B", "led_Y", "led_G"])
  numeric = [
      "phase_norm",
      "trick_number_norm",
      "trick_fill_norm",
      "tricks_won_norm",
      "prediction_norm",
      "has_prediction",
      "wants_more",
      "prediction_already_hit_or_over",
      "prediction_gap_before_norm",
      "rank_norm",
      "hand_count_frac",
      "hand_count_after_frac",
      "total_hand_frac",
      "duplicate_rank_frac",
      "board_free",
      "color_token_available",
      "follows_led",
      "off_led_loses_token",
      "would_complete_trick",
      "would_win",
      "win_aligns_target",
      "adjacency_gain_norm",
      "largest_before_norm",
      "largest_after_norm",
      "is_red",
      "leads_red_before_red_broken",
      "red_broken",
      "legal_count_frac",
      "same_rank_legal_frac",
      "same_color_legal_frac",
      "prediction_value_norm",
      "is_paradox",
      "lead_legal_count_frac_before",
      "lead_legal_count_frac_after",
      "lead_legal_count_delta_frac",
      "nonlead_legal_count_frac_after",
      "lead_dead_card_frac_after",
      "lead_dead_card_delta_frac",
      "lead_legal_rank_frac_after",
      "lead_legal_color_frac_after",
      "lead_no_legal_after",
  ]
  names.extend(numeric)
  names.extend(APPENDED_ACTION_FEATURE_NAMES)
  if len(names) != ACTION_FEATURE_SIZE:
    raise ValueError(
        f"feature name size drift: {len(names)} != {ACTION_FEATURE_SIZE}"
    )
  return names


FEATURE_NAMES = base_feature_names()


DEFAULT_SUMMARY_FEATURES = [
    "rank_norm",
    "would_win",
    "win_aligns_target",
    "adjacency_gain_norm",
    "largest_after_norm",
    "follows_led",
    "off_led_loses_token",
    "leads_red_before_red_broken",
    "hits_prediction",
    "overshoots_prediction",
    "can_still_hit_after",
    "hit_with_future_tricks",
    "remaining_tricks_after",
    "wins_needed_after",
    "post_hit_low_legal_lead_ratio",
    "post_hit_low_card_survival_margin",
    "post_hit_forced_card_pressure",
    "token_loss_newly_loses_led",
    "token_loss_legal_lead_damage",
    "token_loss_creates_no_exit",
    "cluster_frontier_gain",
    "cluster_connects_components",
    "cluster_dead_end_after",
]


def csv(value):
  return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_named_paths(spec):
  paths = {}
  for item in csv(spec):
    if "=" in item:
      name, path = item.split("=", 1)
      paths[name.strip()] = path.strip()
    else:
      path = item.strip()
      paths[Path(path).stem] = path
  return paths


def mean(values):
  if not values:
    return None
  return round(float(np.mean(values)), 4)


def action_type_from_features(row):
  idx = int(np.argmax(row[6:11]))
  return ACTION_TYPE_NAMES[idx] if 0 <= idx < len(ACTION_TYPE_NAMES) else "unknown"


def rank_from_features(row):
  rank_bits = row[11:11 + MAX_RANK_FEATURES]
  if float(np.max(rank_bits)) <= 0.5:
    return None
  return int(np.argmax(rank_bits)) + 1


def color_from_features(row):
  start = 11 + MAX_RANK_FEATURES
  bits = row[start:start + MAX_COLOR_FEATURES]
  if float(np.max(bits)) <= 0.5:
    return None
  names = ["R", "B", "Y", "G"]
  idx = int(np.argmax(bits))
  return names[idx] if idx < len(names) else str(idx)


def distribution(values):
  counts = {}
  for value in values:
    key = "none" if value is None else str(value)
    counts[key] = counts.get(key, 0) + 1
  total = max(1, sum(counts.values()))
  return {
      key: {"count": count, "rate": round(count / total, 4)}
      for key, count in sorted(counts.items())
  }


def selected_action_indices(replay, pred_matrix=None):
  rows = []
  for idx, example in enumerate(replay):
    if len(example) <= 10 or example[9] is None or example[10] is None:
      continue
    target_values = np.asarray(example[9], dtype=np.float32)
    target_mask = np.asarray(example[10], dtype=np.float32) > 0.0
    if int(np.sum(target_mask)) < 2:
      continue
    labeled_indices = np.flatnonzero(target_mask)
    targets = target_values[labeled_indices]
    best_idx = int(labeled_indices[int(np.argmax(targets))])
    policy_target = np.asarray(example[2], dtype=np.float32)
    policy_target_idx = int(np.argmax(policy_target * target_mask))
    pred_idx = None
    if pred_matrix is not None:
      pred_logits = pred_matrix[idx, labeled_indices]
      pred_idx = int(labeled_indices[int(np.argmax(pred_logits))])
    rows.append({
        "row": idx,
        "best": best_idx,
        "policy_target": policy_target_idx,
        "pred": pred_idx,
        "best_target": float(target_values[best_idx]),
        "policy_target_target": float(target_values[policy_target_idx]),
    })
  return rows


def feature_summary(replay, selected_rows, key, feature_names):
  action_features = [
      np.asarray(example[11], dtype=np.float32)
      for example in replay
  ]
  selected_features = []
  regrets_vs_policy_target = []
  for row in selected_rows:
    action = row.get(key)
    if action is None:
      continue
    selected_features.append(action_features[row["row"]][action])
    regrets_vs_policy_target.append(
        row["best_target"] - row["policy_target_target"]
    )
  if not selected_features:
    return {"count": 0}
  arr = np.asarray(selected_features, dtype=np.float32)
  result = {
      "count": int(len(selected_features)),
      "mean_best_minus_policy_target_top": mean(regrets_vs_policy_target),
      "action_type": distribution(action_type_from_features(row) for row in arr),
      "rank": distribution(rank_from_features(row) for row in arr),
      "color": distribution(color_from_features(row) for row in arr),
      "features": {},
  }
  for name in feature_names:
    if name not in FEATURE_NAMES:
      continue
    idx = FEATURE_NAMES.index(name)
    vals = arr[:, idx]
    result["features"][name] = {
        "mean": round(float(np.mean(vals)), 4),
        "positive_rate": round(float(np.mean(vals > 0.5)), 4),
    }
  return result


def model_logits_for_replay(model, replay, device):
  obs = np.asarray([item[0] for item in replay], dtype=np.float32)
  masks = np.asarray([item[1] for item in replay], dtype=bool)
  action_features = np.asarray([item[11] for item in replay], dtype=np.float32)
  obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
  features_t = torch.tensor(action_features, dtype=torch.float32, device=device)
  mask_t = torch.tensor(masks, dtype=torch.bool, device=device)
  with torch.no_grad():
    logits, _ = model(obs_t, features_t)
    logits = logits.masked_fill(~mask_t, -1e9)
  return logits.detach().cpu().numpy()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoints", required=True)
  parser.add_argument("--validations", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--match-context", action="store_true", default=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--filter-phases", default="play")
  parser.add_argument("--filter-min-spread", type=float, default=0.10)
  parser.add_argument("--filter-min-top-margin", type=float, default=0.08)
  parser.add_argument("--summary-features", default=",".join(DEFAULT_SUMMARY_FEATURES))
  args = parser.parse_args()

  trainer_args = trainer_defaults()
  trainer_args.players = args.players
  trainer_args.full_match_training = True
  trainer_args.match_context = bool(args.match_context)
  trainer_args.arch = "action_attn"
  trainer_args.width = 256
  trainer_args.depth = 3
  trainer_args.device = args.device
  trainer_args.action_value_filter_phases = args.filter_phases
  trainer_args.action_value_filter_min_spread = args.filter_min_spread
  trainer_args.action_value_filter_min_top_margin = args.filter_min_top_margin

  device = torch.device(args.device)
  game = make_game(trainer_args)
  checkpoints = parse_named_paths(args.checkpoints)
  validations = parse_named_paths(args.validations)
  feature_names = csv(args.summary_features)

  validation_replays = {}
  validation_summary = {}
  for name, path in validations.items():
    replay, metadata = load_replay(path, return_metadata=True)
    replay, filter_summary = filter_action_value_labels_for_training(
        replay, trainer_args
    )
    selected = selected_action_indices(replay)
    validation_replays[name] = replay
    validation_summary[name] = {
        "path": path,
        "rows": len(replay),
        "labeled_rows": len(selected),
        "metadata": metadata,
        "filter": filter_summary,
        "q_best_action_features": feature_summary(
            replay, selected, "best", feature_names
        ),
        "target_policy_top_labeled_action_features": feature_summary(
            replay, selected, "policy_target", feature_names
        ),
    }

  rows = []
  for checkpoint_name, checkpoint_path in checkpoints.items():
    model, _, saved_args = load_model_payload(
        checkpoint_path, game, trainer_args, device
    )
    model.eval()
    for validation_name, replay in validation_replays.items():
      q_report = q_label_policy_report(model, list(replay), trainer_args, device)
      report_row = {}
      maybe_add_policy_target_report(
          report_row,
          model,
          list(replay),
          trainer_args,
          device,
          name="policy_target_report",
      )
      policy_report = report_row.get("policy_target_report", {})
      pred_matrix = model_logits_for_replay(model, replay, device)
      selected = selected_action_indices(replay, pred_matrix=pred_matrix)
      rows.append({
          "checkpoint": checkpoint_name,
          "checkpoint_path": checkpoint_path,
          "validation": validation_name,
          "validation_path": validations[validation_name],
          "saved_arch": saved_args.get("arch"),
          "saved_width": saved_args.get("width"),
          "saved_depth": saved_args.get("depth"),
          "rows": len(replay),
          "q_top1": None if q_report is None else q_report.get("top1_rate"),
          "q_mean_regret": (
              None if q_report is None else q_report.get("mean_regret")
          ),
          "q_mean_state_corr": (
              None if q_report is None else q_report.get("mean_state_corr")
          ),
          "policy_top1": policy_report.get("top1_rate"),
          "policy_cross_entropy": policy_report.get("cross_entropy"),
          "predicted_action_features": feature_summary(
              replay, selected, "pred", feature_names
          ),
      })

  output = {
      "checkpoints": checkpoints,
      "validations": validations,
      "filter": {
          "phases": args.filter_phases,
          "min_spread": args.filter_min_spread,
          "min_top_margin": args.filter_min_top_margin,
      },
      "feature_names": FEATURE_NAMES,
      "summary_features": feature_names,
      "validation_summary": validation_summary,
      "rows": rows,
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(out_path), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
