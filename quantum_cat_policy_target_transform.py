#!/usr/bin/env python3
"""Transform replay policy targets toward validated action targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_metadata(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
  raw = arrays.get("replay_meta_json")
  if raw is None:
    return {}
  try:
    return json.loads(str(raw.item()))
  except Exception:
    return {}


def _renormalize(row: np.ndarray, legal: np.ndarray) -> np.ndarray:
  out = row.astype(np.float32, copy=True)
  out *= legal.astype(np.float32, copy=False)
  total = float(out.sum())
  if total <= 0.0:
    legal_count = int(np.sum(legal))
    if legal_count > 0:
      out[legal] = 1.0 / float(legal_count)
    return out
  out /= total
  return out


def _softmax_on_indices(
    values: np.ndarray,
    indices: np.ndarray,
    temperature: float,
) -> np.ndarray:
  target = np.zeros_like(values, dtype=np.float32)
  if indices.size == 0:
    return target
  logits = values[indices].astype(np.float64, copy=False)
  temp = max(1e-6, float(temperature))
  logits = logits / temp
  logits = logits - float(np.max(logits))
  weights = np.exp(logits)
  denom = float(weights.sum())
  if denom <= 0.0:
    target[indices] = 1.0 / float(indices.size)
  else:
    target[indices] = (weights / denom).astype(np.float32)
  return target


def _target_distribution(
    arrays: dict[str, np.ndarray],
    row_idx: int,
    mode: str,
    temperature: float,
    positive_threshold: float,
) -> np.ndarray:
  policies = arrays["policies"]
  masks = arrays["masks"] > 0.5
  action_targets = arrays.get("action_target_vectors")
  action_target_masks = arrays.get("action_target_masks")
  action_values = arrays.get("action_value_target_vectors")
  action_value_masks = arrays.get("action_value_target_masks")

  legal = masks[row_idx]
  old_policy = _renormalize(policies[row_idx], legal)
  if action_targets is None or action_target_masks is None:
    return old_policy

  target_mask = (action_target_masks[row_idx] > 0.5) & legal
  positive = (
      target_mask & (action_targets[row_idx] >= float(positive_threshold))
  )
  positive_indices = np.flatnonzero(positive)
  if positive_indices.size == 0:
    positive_indices = np.flatnonzero(target_mask)
  if positive_indices.size == 0:
    return old_policy

  if mode == "positive_uniform":
    target = np.zeros_like(old_policy, dtype=np.float32)
    target[positive_indices] = 1.0 / float(positive_indices.size)
    return target

  if mode == "positive_value_softmax":
    if action_values is None:
      return _target_distribution(
          arrays,
          row_idx,
          "positive_uniform",
          temperature,
          positive_threshold,
      )
    value_mask = (
        np.ones_like(legal, dtype=bool)
        if action_value_masks is None
        else action_value_masks[row_idx] > 0.5
    )
    indices = np.array(
        [idx for idx in positive_indices if value_mask[idx]],
        dtype=np.int64,
    )
    if indices.size == 0:
      indices = positive_indices
    return _softmax_on_indices(action_values[row_idx], indices, temperature)

  if mode == "positive_value_onehot":
    target = np.zeros_like(old_policy, dtype=np.float32)
    if action_values is None:
      best = int(positive_indices[0])
    else:
      value_mask = (
          np.ones_like(legal, dtype=bool)
          if action_value_masks is None
          else action_value_masks[row_idx] > 0.5
      )
      indices = np.array(
          [idx for idx in positive_indices if value_mask[idx]],
          dtype=np.int64,
      )
      if indices.size == 0:
        indices = positive_indices
      best = int(indices[int(np.argmax(action_values[row_idx, indices]))])
    target[best] = 1.0
    return target

  raise ValueError(f"unsupported mode: {mode}")


def transform(args: argparse.Namespace) -> dict[str, Any]:
  with np.load(args.input, allow_pickle=True) as data:
    arrays = {key: data[key] for key in data.files}
  required = {"policies", "masks"}
  missing = sorted(required - set(arrays))
  if missing:
    raise ValueError(f"missing required arrays: {', '.join(missing)}")

  policies = arrays["policies"].astype(np.float32, copy=True)
  masks = arrays["masks"] > 0.5
  old_top = np.argmax(policies, axis=1)
  old_target_probs = []
  new_target_probs = []
  target_top_indices = []
  target_entropies = []

  for row_idx in range(policies.shape[0]):
    old_policy = _renormalize(policies[row_idx], masks[row_idx])
    target = _target_distribution(
        arrays,
        row_idx,
        args.mode,
        args.temperature,
        args.positive_threshold,
    )
    target = _renormalize(target, masks[row_idx])
    blended = (
        (1.0 - float(args.alpha)) * old_policy
        + float(args.alpha) * target
    )
    blended = _renormalize(blended, masks[row_idx])
    target_top = int(np.argmax(target))
    old_target_probs.append(float(old_policy[target_top]))
    new_target_probs.append(float(blended[target_top]))
    target_top_indices.append(target_top)
    nz = target[target > 0.0]
    target_entropies.append(float(-np.sum(nz * np.log(np.maximum(nz, 1e-12)))))
    policies[row_idx] = blended.astype(np.float32)

  new_top = np.argmax(policies, axis=1)
  target_top_array = np.array(target_top_indices, dtype=np.int64)
  arrays["policies"] = policies

  metadata = _load_metadata(arrays)
  metadata.update({
      "policy_target_transform": "quantum_cat_policy_target_transform",
      "policy_target_transform_mode": args.mode,
      "policy_target_transform_alpha": float(args.alpha),
      "policy_target_transform_temperature": float(args.temperature),
      "policy_target_transform_positive_threshold": float(args.positive_threshold),
      "policy_target_transform_input": str(args.input),
      "policy_target_transform_rows": int(policies.shape[0]),
  })
  arrays["replay_meta_json"] = np.array(
      json.dumps(metadata, sort_keys=True), dtype=np.str_
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.out, **arrays)

  summary = {
      "kind": "policy_target_transform",
      "input": str(args.input),
      "out": str(args.out),
      "mode": args.mode,
      "alpha": float(args.alpha),
      "temperature": float(args.temperature),
      "positive_threshold": float(args.positive_threshold),
      "rows": int(policies.shape[0]),
      "old_top_equals_target_top": int(np.sum(old_top == target_top_array)),
      "new_top_equals_target_top": int(np.sum(new_top == target_top_array)),
      "old_top_changed": int(np.sum(new_top != old_top)),
      "mean_old_target_top_prob": float(np.mean(old_target_probs)),
      "mean_new_target_top_prob": float(np.mean(new_target_probs)),
      "min_new_target_top_prob": float(np.min(new_target_probs)),
      "max_new_target_top_prob": float(np.max(new_target_probs)),
      "mean_target_entropy": float(np.mean(target_entropies)),
  }
  if args.summary_out:
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--summary-out", type=Path)
  parser.add_argument(
      "--mode",
      choices=(
          "positive_uniform",
          "positive_value_softmax",
          "positive_value_onehot",
      ),
      default="positive_value_softmax",
  )
  parser.add_argument("--alpha", type=float, default=1.0)
  parser.add_argument("--temperature", type=float, default=0.25)
  parser.add_argument("--positive-threshold", type=float, default=0.5)
  args = parser.parse_args()
  if not 0.0 <= args.alpha <= 1.0:
    raise ValueError("--alpha must be in [0, 1]")
  summary = transform(args)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
