#!/usr/bin/env python3
"""Build policy replay targets with action-level led-follow constraints.

This is the next step after row filtering: keep the public anchor policy as
the default target, and only imitate the hidden-info table-liveness teacher
when the teacher action is explicitly disciplined.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from quantum_cat_alphazero_torch import (
    load_model_payload,
    load_replay,
    make_game,
    parse_args as trainer_parse_args,
    save_replay,
)
from quantum_cat_validation_family_audit import FEATURE_NAMES


SUMMARY_FEATURES = [
    "follows_led",
    "token_loss_newly_loses_led",
    "off_led_loses_token",
    "is_red",
    "would_win",
    "win_aligns_target",
    "hits_prediction",
    "can_still_hit_after",
    "rank_norm",
    "post_hit_forced_card_pressure",
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--out")
  parser.add_argument("--summary-out", required=True)
  parser.add_argument("--anchor-checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument(
      "--red-rescue-q-margin",
      type=float,
      default=0.50,
      help=(
          "Minimum table-liveness Q advantage over the anchor action before "
          "imitating an off-led red rescue."
      ),
  )
  parser.add_argument(
      "--follow-q-margin",
      type=float,
      default=-1e9,
      help=(
          "Optional minimum teacher-anchor Q margin for follow-led teacher "
          "actions. The default accepts all follow-led teacher actions."
      ),
  )
  parser.add_argument(
      "--require-red-win-aligns-target",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--require-red-q-labeled-anchor",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "When true, red rescue imitation requires both teacher and anchor "
          "actions to have dense Q labels."
      ),
  )
  parser.add_argument(
      "--require-follow-q-labeled-anchor",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "When true, follow-led teacher imitation requires both teacher and "
          "anchor actions to have dense Q labels."
      ),
  )
  parser.add_argument(
      "--positive-follow-repeat",
      type=int,
      default=1,
      help=(
          "Total emitted copies for accepted teacher_follows_led rows. Values "
          ">1 oversample the disciplined positive-margin follow corrections "
          "while preserving row_group_ids for grouped validation."
      ),
  )
  return parser.parse_args()


def trainer_defaults():
  old_argv = sys.argv[:]
  try:
    sys.argv = ["quantum_cat_constrained_policy_replay"]
    return trainer_parse_args()
  finally:
    sys.argv = old_argv


def feature_idx(name):
  if name not in FEATURE_NAMES:
    raise ValueError(f"Unknown action feature {name!r}")
  return FEATURE_NAMES.index(name)


FEATURE_INDEX = {name: feature_idx(name) for name in SUMMARY_FEATURES}


def positive(action_features, action, name):
  return float(action_features[int(action), FEATURE_INDEX[name]]) > 0.5


def value(action_features, action, name):
  return float(action_features[int(action), FEATURE_INDEX[name]])


def one_hot_like(policy, action):
  result = np.zeros_like(np.asarray(policy, dtype=np.float32), dtype=np.float32)
  result[int(action)] = 1.0
  return result


def q_margin(example, teacher_action, anchor_action):
  if len(example) <= 10 or example[9] is None or example[10] is None:
    return None
  q_values = np.asarray(example[9], dtype=np.float32)
  q_mask = np.asarray(example[10], dtype=np.float32) > 0.0
  teacher_action = int(teacher_action)
  anchor_action = int(anchor_action)
  if (
      teacher_action < 0
      or teacher_action >= len(q_mask)
      or anchor_action < 0
      or anchor_action >= len(q_mask)
      or not bool(q_mask[teacher_action])
      or not bool(q_mask[anchor_action])
  ):
    return None
  return float(q_values[teacher_action] - q_values[anchor_action])


def choose_action(example, anchor_action, args):
  policy = np.asarray(example[2], dtype=np.float32)
  legal_mask = np.asarray(example[1], dtype=np.float32) > 0.0
  masked_policy = np.where(legal_mask, policy, -np.inf)
  teacher_action = int(np.argmax(masked_policy))
  anchor_action = int(anchor_action)
  action_features = np.asarray(example[11], dtype=np.float32)
  if teacher_action == anchor_action:
    return anchor_action, teacher_action, "same_as_anchor", q_margin(
        example, teacher_action, anchor_action
    )

  margin = q_margin(example, teacher_action, anchor_action)
  teacher_follows = positive(action_features, teacher_action, "follows_led")
  teacher_red_rescue = (
      positive(action_features, teacher_action, "is_red")
      and positive(action_features, teacher_action, "would_win")
      and (
          not bool(args.require_red_win_aligns_target)
          or positive(action_features, teacher_action, "win_aligns_target")
      )
  )

  if teacher_follows:
    if bool(args.require_follow_q_labeled_anchor) and margin is None:
      return anchor_action, teacher_action, "anchor_follow_q_unlabeled", margin
    if margin is not None and margin < float(args.follow_q_margin):
      return anchor_action, teacher_action, "anchor_follow_margin", margin
    return teacher_action, teacher_action, "teacher_follows_led", margin

  if teacher_red_rescue:
    if bool(args.require_red_q_labeled_anchor) and margin is None:
      return anchor_action, teacher_action, "anchor_red_q_unlabeled", margin
    if margin is None or margin < float(args.red_rescue_q_margin):
      return anchor_action, teacher_action, "anchor_red_margin", margin
    return teacher_action, teacher_action, "teacher_red_rescue", margin

  return anchor_action, teacher_action, "anchor_teacher_unsafe", margin


def model_logits_for_replay(model, replay, device, batch_size):
  logits_rows = []
  for start in range(0, len(replay), int(batch_size)):
    batch = replay[start:start + int(batch_size)]
    obs = np.asarray([item[0] for item in batch], dtype=np.float32)
    masks = np.asarray([item[1] for item in batch], dtype=bool)
    action_features = np.asarray([item[11] for item in batch], dtype=np.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    features_t = torch.tensor(action_features, dtype=torch.float32, device=device)
    mask_t = torch.tensor(masks, dtype=torch.bool, device=device)
    with torch.no_grad():
      logits, _ = model(obs_t, features_t)
      logits = logits.masked_fill(~mask_t, -1e9)
    logits_rows.append(logits.detach().cpu().numpy())
  return np.concatenate(logits_rows, axis=0)


def summarize_features(replay, actions):
  if not replay:
    return {}
  rows = []
  for example, action in zip(replay, actions):
    rows.append(np.asarray(example[11], dtype=np.float32)[int(action)])
  arr = np.asarray(rows, dtype=np.float32)
  summary = {}
  for name in SUMMARY_FEATURES:
    vals = arr[:, FEATURE_INDEX[name]]
    summary[name] = {
        "mean": round(float(np.mean(vals)), 4),
        "positive_rate": round(float(np.mean(vals > 0.5)), 4),
    }
  return summary


def summarize_margins(margins):
  values = [float(v) for v in margins if v is not None and math.isfinite(float(v))]
  if not values:
    return None
  arr = np.asarray(values, dtype=np.float32)
  return {
      "count": int(len(values)),
      "mean": round(float(np.mean(arr)), 4),
      "min": round(float(np.min(arr)), 4),
      "max": round(float(np.max(arr)), 4),
      "p10": round(float(np.quantile(arr, 0.10)), 4),
      "p50": round(float(np.quantile(arr, 0.50)), 4),
      "p90": round(float(np.quantile(arr, 0.90)), 4),
  }


def main():
  args = parse_args()
  replay, source_metadata = load_replay(args.input, return_metadata=True)

  trainer_args = trainer_defaults()
  trainer_args.players = int(args.players)
  trainer_args.full_match_training = True
  trainer_args.match_context = True
  trainer_args.arch = "action_attn"
  trainer_args.width = 256
  trainer_args.depth = 3
  trainer_args.device = args.device
  device = torch.device(args.device)
  game = make_game(trainer_args)
  anchor_model, _, _ = load_model_payload(
      args.anchor_checkpoint, game, trainer_args, device
  )
  anchor_model.eval()
  anchor_logits = model_logits_for_replay(
      anchor_model, replay, device, int(args.batch_size)
  )
  anchor_actions = np.argmax(anchor_logits, axis=1).astype(np.int64)

  source_group_ids = []
  if isinstance(source_metadata, dict):
    maybe_group_ids = source_metadata.get("row_group_ids")
    if isinstance(maybe_group_ids, list) and len(maybe_group_ids) == len(replay):
      source_group_ids = [str(item) for item in maybe_group_ids]

  derived = []
  output_chosen_actions = []
  output_teacher_actions = []
  output_anchor_actions = []
  output_group_ids = []
  source_chosen_actions = []
  source_teacher_actions = []
  reasons = Counter()
  output_reasons = Counter()
  margins_by_reason = defaultdict(list)
  repeat_by_reason = defaultdict(list)
  positive_follow_repeat = max(1, int(args.positive_follow_repeat))
  for row_idx, (example, anchor_action) in enumerate(zip(replay, anchor_actions)):
    chosen, teacher, reason, margin = choose_action(example, anchor_action, args)
    reasons[reason] += 1
    margins_by_reason[reason].append(margin)
    source_teacher_actions.append(int(teacher))
    source_chosen_actions.append(int(chosen))

    repeat = positive_follow_repeat if reason == "teacher_follows_led" else 1
    repeat_by_reason[reason].append(int(repeat))
    output_reasons[reason] += int(repeat)
    for _ in range(int(repeat)):
      items = list(example)
      items[2] = one_hot_like(items[2], int(chosen))
      derived.append(tuple(items))
      output_chosen_actions.append(int(chosen))
      output_teacher_actions.append(int(teacher))
      output_anchor_actions.append(int(anchor_action))
      if source_group_ids:
        output_group_ids.append(source_group_ids[row_idx])

  chosen_arr = np.asarray(source_chosen_actions, dtype=np.int64)
  teacher_arr = np.asarray(source_teacher_actions, dtype=np.int64)
  anchor_arr = np.asarray(anchor_actions, dtype=np.int64)
  output_chosen_arr = np.asarray(output_chosen_actions, dtype=np.int64)
  output_teacher_arr = np.asarray(output_teacher_actions, dtype=np.int64)
  output_anchor_arr = np.asarray(output_anchor_actions, dtype=np.int64)
  summary = {
      "kind": "constrained_policy_replay",
      "input": args.input,
      "out": args.out,
      "anchor_checkpoint": args.anchor_checkpoint,
      "source_rows": int(len(replay)),
      "rows": int(len(derived)),
      "red_rescue_q_margin": float(args.red_rescue_q_margin),
      "follow_q_margin": float(args.follow_q_margin),
      "positive_follow_repeat": positive_follow_repeat,
      "require_red_win_aligns_target": bool(args.require_red_win_aligns_target),
      "require_red_q_labeled_anchor": bool(args.require_red_q_labeled_anchor),
      "require_follow_q_labeled_anchor": bool(args.require_follow_q_labeled_anchor),
      "decision_counts": dict(sorted(reasons.items())),
      "output_decision_counts": dict(sorted(output_reasons.items())),
      "repeat_by_reason": {
          reason: {
              "min": int(min(repeats)),
              "max": int(max(repeats)),
              "mean": round(float(np.mean(repeats)), 4),
          }
          for reason, repeats in sorted(repeat_by_reason.items())
      },
      "teacher_chosen_rows": int(np.sum(chosen_arr == teacher_arr)),
      "anchor_chosen_rows": int(np.sum(chosen_arr == anchor_arr)),
      "changed_from_anchor_rows": int(np.sum(chosen_arr != anchor_arr)),
      "changed_from_teacher_rows": int(np.sum(chosen_arr != teacher_arr)),
      "output_teacher_chosen_rows": int(np.sum(output_chosen_arr == output_teacher_arr)),
      "output_anchor_chosen_rows": int(np.sum(output_chosen_arr == output_anchor_arr)),
      "output_changed_from_anchor_rows": int(np.sum(output_chosen_arr != output_anchor_arr)),
      "output_changed_from_teacher_rows": int(np.sum(output_chosen_arr != output_teacher_arr)),
      "chosen_features": summarize_features(derived, output_chosen_actions),
      "teacher_features": summarize_features(derived, output_teacher_actions),
      "anchor_features": summarize_features(derived, output_anchor_actions),
      "source_chosen_features": summarize_features(replay, source_chosen_actions),
      "source_teacher_features": summarize_features(replay, source_teacher_actions),
      "source_anchor_features": summarize_features(replay, anchor_actions),
      "q_margin_by_reason": {
          reason: summarize_margins(margins)
          for reason, margins in sorted(margins_by_reason.items())
      },
  }
  if output_group_ids:
    summary.update({
        "row_group_ids": output_group_ids,
        "row_group_id_kind": source_metadata.get(
            "row_group_id_kind", "source_match_round"
        ),
        "row_group_count": int(len(set(output_group_ids))),
        "source_row_group_count": int(len(set(source_group_ids))),
    })

  Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.summary_out).write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n"
  )
  if args.out:
    save_replay(derived, args.out, metadata=summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
