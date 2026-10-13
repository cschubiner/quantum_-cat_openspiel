#!/usr/bin/env python3
"""Derive policy targets from sparse per-action paradox-risk labels.

The trainer stores per-action paradox labels in replay item[7]/item[8], while
the policy head learns from item[2]. This tool converts high-confidence
avoidable-risk rows into direct policy targets by putting probability mass on
lower-risk labeled actions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from quantum_cat_alphazero_torch import adapt_action_features
from quantum_cat_alphazero_torch import adapt_observation_batch
from quantum_cat_alphazero_torch import ACTION_FEATURE_FOLLOWS_LED_INDEX
from quantum_cat_alphazero_torch import ACTION_FEATURE_IS_RED_INDEX
from quantum_cat_alphazero_torch import load_model_payload
from quantum_cat_alphazero_torch import load_replay
from quantum_cat_alphazero_torch import make_game
from quantum_cat_alphazero_torch import model_input_size
from quantum_cat_alphazero_torch import save_replay


PHASE_NAMES = ("chance", "discard", "prediction", "play", "terminal")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--arch", default="mlp")
  parser.add_argument("--width", type=int, default=256)
  parser.add_argument("--depth", type=int, default=3)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--separate-action-value-encoder", action="store_true")
  parser.add_argument("--anchor-checkpoint", default="")
  parser.add_argument("--anchor-policy-batch-size", type=int, default=256)
  parser.add_argument(
      "--reference-policy",
      choices=("replay", "anchor"),
      default="replay",
      help=(
          "Policy used as the baseline top action for override/improvement "
          "filters. 'replay' preserves historical behavior. 'anchor' uses "
          "--anchor-checkpoint and asks whether the label-best action improves "
          "on the checkpoint's actual policy top action."
      ),
  )
  parser.add_argument("--min-labeled-actions", type=int, default=2)
  parser.add_argument("--min-safe-actions", type=int, default=1)
  parser.add_argument("--min-risky-actions", type=int, default=1)
  parser.add_argument("--min-spread", type=float, default=0.0)
  parser.add_argument("--min-top-margin", type=float, default=0.0)
  parser.add_argument(
      "--min-original-risk-improvement",
      type=float,
      default=-1e9,
      help=(
          "Require original-top risk minus best labeled risk to be at least "
          "this value when the original top action is labeled."
      ),
  )
  parser.add_argument("--safe-threshold", type=float, default=0.34)
  parser.add_argument("--risky-threshold", type=float, default=0.67)
  parser.add_argument("--temperature", type=float, default=0.08)
  parser.add_argument("--risk-policy-weight", type=float, default=1.0)
  parser.add_argument(
      "--target-mode",
      choices=("softmin", "disciplined_one_hot"),
      default="softmin",
      help=(
          "How to convert risk labels into a policy target. softmin preserves "
          "the historical risk-only target. disciplined_one_hot keeps risk as "
          "the primary score, then tie-breaks safe actions toward follow-led "
          "and red discipline."
      ),
  )
  parser.add_argument("--follow-led-bonus", type=float, default=0.20)
  parser.add_argument("--red-bonus", type=float, default=0.05)
  parser.add_argument(
      "--include-non-overrides",
      action="store_true",
      help="Keep rows even when the best low-risk action is already policy-top.",
  )
  parser.add_argument(
      "--keep-action-paradox-labels",
      action="store_true",
      help="Keep item[7]/item[8] for auxiliary risk-head training.",
  )
  parser.add_argument(
      "--include-unselected-originals",
      action="store_true",
      help=(
          "Also emit replay rows that do not pass the risk-policy filter, "
          "preserving their original policy targets. This keeps broad policy "
          "coverage while still shaping high-confidence avoidable-risk rows."
      ),
  )
  parser.add_argument(
      "--include-source-metadata",
      action="store_true",
      help=(
          "Embed full source replay metadata. By default only compact source "
          "metadata is saved so derived replays do not recursively bloat."
      ),
  )
  return parser.parse_args()


def _labeled_mask(mask):
  return np.asarray(mask, dtype=np.float32) > 0.0


def action_paradox_quality(example):
  if len(example) <= 8 or example[7] is None or example[8] is None:
    return None
  targets = np.asarray(example[7], dtype=np.float32)
  mask = _labeled_mask(example[8])
  labeled = np.flatnonzero(mask)
  if labeled.size < 2:
    return None
  values = targets[labeled]
  sorted_values = np.sort(values)
  return {
      "labeled": labeled,
      "values": values,
      "spread": float(sorted_values[-1] - sorted_values[0]),
      "top_margin": float(sorted_values[1] - sorted_values[0]),
      "safe_actions": int(np.sum(values <= 0.34)),
      "risky_actions": int(np.sum(values >= 0.67)),
      "best_action": int(labeled[int(np.argmin(values))]),
  }


def _softmin_policy(targets, mask, temperature):
  selected = _labeled_mask(mask)
  if int(np.sum(selected)) == 0:
    return None
  logits = -np.asarray(targets, dtype=np.float32)[selected] / max(
      1e-6, float(temperature)
  )
  logits -= np.max(logits)
  exp = np.exp(logits)
  total = float(exp.sum())
  if total <= 0.0 or not math.isfinite(total):
    return None
  policy = np.zeros_like(np.asarray(targets, dtype=np.float32), dtype=np.float32)
  policy[selected] = exp / total
  return policy


def _disciplined_one_hot_policy(example, targets, mask, args):
  selected = _labeled_mask(mask)
  labeled = np.flatnonzero(selected)
  if labeled.size == 0:
    return None
  policy = np.zeros_like(np.asarray(targets, dtype=np.float32), dtype=np.float32)
  scores = -np.asarray(targets, dtype=np.float32).copy()
  action_features = None
  if len(example) > 11 and example[11] is not None:
    action_features = np.asarray(example[11], dtype=np.float32)
  if (
      action_features is not None
      and action_features.ndim == 2
      and action_features.shape[0] == scores.shape[0]
      and action_features.shape[1] > ACTION_FEATURE_IS_RED_INDEX
  ):
    scores += (
        float(getattr(args, "follow_led_bonus", 0.20))
        * (action_features[:, ACTION_FEATURE_FOLLOWS_LED_INDEX] > 0.5)
    )
    scores += (
        float(getattr(args, "red_bonus", 0.05))
        * (action_features[:, ACTION_FEATURE_IS_RED_INDEX] > 0.5)
    )
  scores[~selected] = -np.inf
  best = int(np.argmax(scores))
  if not np.isfinite(float(scores[best])):
    return None
  policy[best] = 1.0
  return policy


def _risk_policy_for_example(example, risk_targets, risk_mask, args):
  mode = str(getattr(args, "target_mode", "softmin") or "softmin")
  if mode == "disciplined_one_hot":
    return _disciplined_one_hot_policy(example, risk_targets, risk_mask, args)
  return _softmin_policy(risk_targets, risk_mask, args.temperature)


def _clear_action_paradox_labels(example):
  if len(example) <= 8:
    return example
  items = list(example)
  if items[7] is not None:
    items[7] = np.zeros_like(items[7], dtype=np.float32)
  if items[8] is not None:
    items[8] = np.zeros_like(items[8], dtype=np.float32)
  return tuple(items)


def _phase_name_for_example(example, labeled_actions=None):
  if len(example) <= 11 or example[11] is None:
    return "unknown"
  action_features = np.asarray(example[11], dtype=np.float32)
  if action_features.ndim != 2 or action_features.shape[1] < 6:
    return "unknown"
  action_idx = 0
  if labeled_actions is not None and len(labeled_actions) > 0:
    action_idx = int(labeled_actions[0])
  if action_idx < 0 or action_idx >= action_features.shape[0]:
    return "unknown"
  phase_idx = int(np.argmax(action_features[action_idx, 1:6]))
  if 0 <= phase_idx < len(PHASE_NAMES):
    return PHASE_NAMES[phase_idx]
  return "unknown"


def _bump(counter, key):
  counter[key] = int(counter.get(key, 0)) + 1


def anchor_policies_for_replay(replay, args):
  if not getattr(args, "anchor_checkpoint", ""):
    return None
  device = torch.device(str(getattr(args, "device", "cpu")))
  game = make_game(args, 0)
  model, _, _ = load_model_payload(args.anchor_checkpoint, game, args, device)
  model.eval()
  for param in model.parameters():
    param.requires_grad = False
  batch_size = max(1, int(getattr(args, "anchor_policy_batch_size", 256)))
  policies = []
  with torch.no_grad():
    for start in range(0, len(replay), batch_size):
      batch = replay[start:start + batch_size]
      obs = torch.tensor(
          np.array([item[0] for item in batch], dtype=np.float32),
          dtype=torch.float32,
          device=device,
      )
      obs = adapt_observation_batch(obs, model_input_size(model))
      masks = torch.tensor(
          np.array([item[1] for item in batch], dtype=np.float32) > 0.0,
          dtype=torch.bool,
          device=device,
      )
      action_features = torch.tensor(
          np.array([
              adapt_action_features(
                  item[11] if len(item) > 11 else None,
                  len(item[1]),
              )
              for item in batch
          ], dtype=np.float32),
          dtype=torch.float32,
          device=device,
      )
      logits, _ = model(obs, action_features)
      logits = logits.masked_fill(~masks, -1e9)
      policies.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())
  return policies


def derive_policy_replay(replay, args, anchor_policies=None):
  weight = min(1.0, max(0.0, float(args.risk_policy_weight)))
  reference_source = str(getattr(args, "reference_policy", "replay") or "replay")
  if reference_source == "anchor" and anchor_policies is None:
    raise ValueError("--reference-policy=anchor requires anchor_policies")
  output = []
  output_source_indices = []
  skipped = {
      "unlabeled": 0,
      "min_labeled_actions": 0,
      "min_safe_actions": 0,
      "min_risky_actions": 0,
      "quality": 0,
      "original_risk_improvement": 0,
      "non_override": 0,
      "bad_softmax": 0,
      "anchor_policy_missing": 0,
  }
  spreads = []
  margins = []
  improvements = []
  best_top_changed = 0
  derived_top_changed = 0
  labeled_rows = 0
  derived_rows = 0
  included_original_rows = 0
  labeled_rows_by_phase = {}
  derived_rows_by_phase = {}
  included_original_rows_by_phase = {}
  best_top_changed_rows_by_phase = {}
  derived_top_changed_rows_by_phase = {}

  def maybe_emit_original(example, phase_name=None):
    nonlocal included_original_rows
    if not bool(getattr(args, "include_unselected_originals", False)):
      return
    if phase_name is None:
      phase_name = _phase_name_for_example(example)
    if bool(getattr(args, "keep_action_paradox_labels", False)):
      output.append(example)
    else:
      output.append(_clear_action_paradox_labels(example))
    output_source_indices.append(row_idx)
    included_original_rows += 1
    _bump(included_original_rows_by_phase, phase_name)

  for row_idx, example in enumerate(replay):
    quality = action_paradox_quality(example)
    phase_name = _phase_name_for_example(
        example, None if quality is None else quality["labeled"]
    )
    if quality is None:
      skipped["unlabeled"] += 1
      maybe_emit_original(example, phase_name)
      continue
    labeled_rows += 1
    _bump(labeled_rows_by_phase, phase_name)
    if len(quality["labeled"]) < int(args.min_labeled_actions):
      skipped["min_labeled_actions"] += 1
      maybe_emit_original(example, phase_name)
      continue
    values = quality["values"]
    safe_count = int(np.sum(values <= float(args.safe_threshold)))
    risky_count = int(np.sum(values >= float(args.risky_threshold)))
    if safe_count < int(args.min_safe_actions):
      skipped["min_safe_actions"] += 1
      maybe_emit_original(example, phase_name)
      continue
    if risky_count < int(args.min_risky_actions):
      skipped["min_risky_actions"] += 1
      maybe_emit_original(example, phase_name)
      continue
    if (
        quality["spread"] < float(args.min_spread)
        or quality["top_margin"] < float(args.min_top_margin)
    ):
      skipped["quality"] += 1
      maybe_emit_original(example, phase_name)
      continue

    original_policy = np.asarray(example[2], dtype=np.float32)
    if reference_source == "anchor":
      if row_idx >= len(anchor_policies):
        skipped["anchor_policy_missing"] += 1
        maybe_emit_original(example, phase_name)
        continue
      reference_policy = np.asarray(anchor_policies[row_idx], dtype=np.float32)
    else:
      reference_policy = original_policy
    original_top = int(np.argmax(reference_policy))
    risk_targets = np.asarray(example[7], dtype=np.float32)
    risk_mask = _labeled_mask(example[8])
    best_action = int(quality["best_action"])
    improvement = None
    if 0 <= original_top < len(risk_mask) and bool(risk_mask[original_top]):
      improvement = float(risk_targets[original_top] - risk_targets[best_action])
      if improvement < float(args.min_original_risk_improvement):
        skipped["original_risk_improvement"] += 1
        maybe_emit_original(example, phase_name)
        continue
      improvements.append(improvement)
    if best_action != original_top:
      best_top_changed += 1
      _bump(best_top_changed_rows_by_phase, phase_name)
    elif not args.include_non_overrides:
      skipped["non_override"] += 1
      maybe_emit_original(example, phase_name)
      continue

    risk_policy = _risk_policy_for_example(example, risk_targets, risk_mask, args)
    if risk_policy is None:
      skipped["bad_softmax"] += 1
      maybe_emit_original(example, phase_name)
      continue
    derived_policy = (1.0 - weight) * original_policy + weight * risk_policy
    total = float(derived_policy.sum())
    if total <= 0.0 or not math.isfinite(total):
      skipped["bad_softmax"] += 1
      maybe_emit_original(example, phase_name)
      continue
    derived_policy = (derived_policy / total).astype(np.float32)
    if int(np.argmax(derived_policy)) != original_top:
      derived_top_changed += 1
      _bump(derived_top_changed_rows_by_phase, phase_name)
    items = list(example)
    items[2] = derived_policy
    new_example = tuple(items)
    if not args.keep_action_paradox_labels:
      new_example = _clear_action_paradox_labels(new_example)
    output.append(new_example)
    output_source_indices.append(row_idx)
    derived_rows += 1
    _bump(derived_rows_by_phase, phase_name)
    spreads.append(float(quality["spread"]))
    margins.append(float(quality["top_margin"]))

  summary = {
      "input_rows": len(replay),
      "action_paradox_labeled_rows": labeled_rows,
      "output_rows": len(output),
      "derived_rows": int(derived_rows),
      "included_original_rows": int(included_original_rows),
      "action_paradox_labeled_rows_by_phase": labeled_rows_by_phase,
      "derived_rows_by_phase": derived_rows_by_phase,
      "included_original_rows_by_phase": included_original_rows_by_phase,
      "skipped": skipped,
      "min_labeled_actions": int(args.min_labeled_actions),
      "min_safe_actions": int(args.min_safe_actions),
      "min_risky_actions": int(args.min_risky_actions),
      "min_spread": float(args.min_spread),
      "min_top_margin": float(args.min_top_margin),
      "min_original_risk_improvement": float(args.min_original_risk_improvement),
      "safe_threshold": float(args.safe_threshold),
      "risky_threshold": float(args.risky_threshold),
      "temperature": float(args.temperature),
      "risk_policy_weight": weight,
      "target_mode": str(getattr(args, "target_mode", "softmin") or "softmin"),
      "follow_led_bonus": float(getattr(args, "follow_led_bonus", 0.20)),
      "red_bonus": float(getattr(args, "red_bonus", 0.05)),
      "include_non_overrides": bool(args.include_non_overrides),
      "keep_action_paradox_labels": bool(args.keep_action_paradox_labels),
      "include_unselected_originals": bool(
          getattr(args, "include_unselected_originals", False)
      ),
      "reference_policy": reference_source,
      "anchor_checkpoint": str(getattr(args, "anchor_checkpoint", "") or ""),
      "best_top_changed_rows": int(best_top_changed),
      "derived_top_changed_rows": int(derived_top_changed),
      "reference_top_changed_rows": int(best_top_changed),
      "derived_reference_top_changed_rows": int(derived_top_changed),
      "best_top_changed_rows_by_phase": best_top_changed_rows_by_phase,
      "derived_top_changed_rows_by_phase": derived_top_changed_rows_by_phase,
      "reference_top_changed_rows_by_phase": best_top_changed_rows_by_phase,
      "derived_reference_top_changed_rows_by_phase": (
          derived_top_changed_rows_by_phase
      ),
      "mean_spread": float(np.mean(spreads)) if spreads else None,
      "mean_top_margin": float(np.mean(margins)) if margins else None,
      "mean_original_risk_improvement": (
          float(np.mean(improvements)) if improvements else None
      ),
  }
  return output, summary, output_source_indices


def _compact_metadata(metadata):
  if not isinstance(metadata, dict):
    return metadata
  compact = {}
  for key, value in metadata.items():
    if key == "row_group_ids":
      compact["row_group_count"] = int(len(set(value))) if value else 0
      compact["row_group_ids_omitted"] = True
      continue
    if key == "input_metadata":
      compact["input_metadata_omitted"] = True
      continue
    compact[key] = value
  return compact


def main():
  args = parse_args()
  if args.reference_policy == "anchor" and not args.anchor_checkpoint:
    raise SystemExit("--reference-policy=anchor requires --anchor-checkpoint")
  replay = []
  replay_group_ids = []
  input_counts = {}
  input_metadata = []
  for input_path in args.input:
    rows, metadata = load_replay(input_path, return_metadata=True)
    input_counts[str(input_path)] = len(rows)
    input_metadata.append({
        "path": str(input_path),
        "metadata": (
            metadata if args.include_source_metadata else _compact_metadata(metadata)
        ),
    })
    row_group_ids = []
    if isinstance(metadata, dict):
      row_group_ids = list(metadata.get("row_group_ids") or [])
    if row_group_ids and len(row_group_ids) != len(rows):
      row_group_ids = []
    if row_group_ids:
      replay_group_ids.extend(str(group_id) for group_id in row_group_ids)
    else:
      replay_group_ids.extend([""] * len(rows))
    replay.extend(rows)

  anchor_policies = (
      anchor_policies_for_replay(replay, args)
      if args.reference_policy == "anchor"
      else None
  )
  derived, summary, output_source_indices = derive_policy_replay(
      replay, args, anchor_policies=anchor_policies
  )
  summary["inputs"] = input_counts
  summary["input_metadata"] = input_metadata
  if (
      len(replay_group_ids) == len(replay)
      and len(output_source_indices) == len(derived)
      and any(replay_group_ids)
  ):
    output_group_ids = [
        replay_group_ids[int(source_idx)] for source_idx in output_source_indices
    ]
    if all(output_group_ids):
      summary["row_group_ids"] = output_group_ids
      summary["row_group_id_kind"] = "source_replay"
      summary["row_group_count"] = int(len(set(output_group_ids)))
  summary["out"] = args.out
  text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
  if args.summary_out:
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text)
  if not derived:
    raise SystemExit(text)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  save_replay(derived, out_path, metadata=summary)
  print(text, end="")


if __name__ == "__main__":
  main()
