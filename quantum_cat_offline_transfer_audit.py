#!/usr/bin/env python3
"""Offline transfer audit for accepted Cat-in-the-Box action labels.

This checks whether a trained scorer can convert accepted sparse Q labels into
actual deployment decisions. It is intentionally separate from match Elo: the
question here is coverage and calibration on labeled states, not win rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quantum_cat_alphazero_torch import (
    adapt_action_features,
    adapt_observation_batch,
    load_replay,
    model_action_value_batch,
    model_input_size,
)
from quantum_cat_full_match_elo import load_neural


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--anchor-checkpoint", required=True)
  parser.add_argument("--replay", action="append", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
  parser.add_argument("--out", required=True)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--action-value-selection-weight", type=float, default=1.0)
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument(
      "--require-anchor-labeled",
      action="store_true",
      help=(
          "Only include rows where the anchor policy top action has a Q label. "
          "This makes true override-margin stats stricter."
      ),
  )
  return parser.parse_args()


def _rate(numerator, denominator):
  if denominator <= 0:
    return None
  return round(float(numerator) / float(denominator), 4)


def _mean(values):
  if not values:
    return None
  return round(float(np.mean(values)), 6)


def _ci95_mean(values):
  if len(values) <= 1:
    return None
  arr = np.array(values, dtype=np.float64)
  se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
  mean = float(np.mean(arr))
  return [round(mean - 1.96 * se, 6), round(mean + 1.96 * se, 6)]


def _load_rows(paths):
  rows = []
  counts = {}
  for path in paths:
    replay_rows = load_replay(path)
    counts[str(path)] = len(replay_rows)
    rows.extend(replay_rows)
  return rows, counts


def _valid_rows(rows, require_anchor_labeled=False):
  valid = []
  skipped = {
      "unlabeled_q": 0,
      "too_few_q_actions": 0,
      "no_legal_actions": 0,
      "anchor_not_labeled": 0,
  }
  for row in rows:
    if len(row) <= 11 or row[9] is None or row[10] is None:
      skipped["unlabeled_q"] += 1
      continue
    legal_mask = np.array(row[1], dtype=np.float32) > 0.5
    q_mask = np.array(row[10], dtype=np.float32) > 0.0
    if int(np.sum(q_mask)) < 2:
      skipped["too_few_q_actions"] += 1
      continue
    if not np.any(legal_mask):
      skipped["no_legal_actions"] += 1
      continue
    valid.append(row)
  return valid, skipped


def _predict_batches(model, rows, device, batch_size):
  obs = np.array([row[0] for row in rows], dtype=np.float32)
  legal_masks = np.array([row[1] for row in rows], dtype=np.float32) > 0.5
  action_features = np.array([
      adapt_action_features(row[11] if len(row) > 11 else None, len(row[1]))
      for row in rows
  ], dtype=np.float32)
  policy_chunks = []
  q_chunks = []
  risk_chunks = []
  with torch.no_grad():
    for start in range(0, len(rows), batch_size):
      end = min(len(rows), start + batch_size)
      obs_t = torch.tensor(obs[start:end], dtype=torch.float32, device=device)
      obs_t = adapt_observation_batch(obs_t, model_input_size(model))
      features_t = torch.tensor(
          action_features[start:end], dtype=torch.float32, device=device
      )
      mask_t = torch.tensor(legal_masks[start:end], dtype=torch.bool, device=device)
      logits, _ = model(obs_t, features_t)
      logits = logits.masked_fill(~mask_t, -1e9)
      policy_chunks.append(F.softmax(logits, dim=1).detach().cpu().numpy())
      q_chunks.append(
          model_action_value_batch(model, obs_t, features_t).detach().cpu().numpy()
      )
      if hasattr(model, "action_paradox"):
        _, _, _, risk_logits = model.forward_with_action_aux(obs_t, features_t)
        risk_chunks.append(torch.sigmoid(risk_logits).detach().cpu().numpy())
      else:
        risk_chunks.append(np.zeros_like(q_chunks[-1], dtype=np.float32))
  return (
      np.concatenate(policy_chunks, axis=0),
      np.concatenate(q_chunks, axis=0),
      np.concatenate(risk_chunks, axis=0),
      legal_masks,
      action_features,
  )


def audit(args):
  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device=mps requested, but torch MPS is not available")
  device = torch.device(args.device)
  model, model_args = load_neural(args.checkpoint, args.players, device)
  anchor_model, _ = load_neural(args.anchor_checkpoint, args.players, device)
  rows, input_counts = _load_rows(args.replay)
  valid, skipped = _valid_rows(rows)
  if not valid:
    return {
        "inputs": input_counts,
        "rows": len(rows),
        "valid_rows": 0,
        "skipped": skipped,
    }

  model_policy, model_q, model_risk, legal_masks, _ = _predict_batches(
      model, valid, device, max(1, int(args.batch_size))
  )
  anchor_policy, _, _, _, _ = _predict_batches(
      anchor_model, valid, device, max(1, int(args.batch_size))
  )

  label_best_hits = 0
  label_positive_or_tie_hits = 0
  anchor_labeled_rows = 0
  q_head_margins = []
  q_head_regrets = []
  deployed_overrides = 0
  deployed_applied = 0
  deployed_abstained_margin = 0
  deployed_selected_labeled = 0
  deployed_anchor_labeled = 0
  deployed_label_best_hits = 0
  deployed_true_margins = []
  deployed_override_true_margins = []
  deployed_override_label_supported = 0
  deployed_override_to_label_best = 0
  deployed_override_positive = 0
  audited_rows = 0

  for idx, row in enumerate(valid):
    q_targets = np.array(row[9], dtype=np.float32)
    q_mask = np.array(row[10], dtype=np.float32) > 0.0
    labeled_actions = np.flatnonzero(q_mask)
    true_best = int(labeled_actions[np.argmax(q_targets[q_mask])])
    anchor_action = int(np.argmax(np.where(legal_masks[idx], anchor_policy[idx], -np.inf)))
    if args.require_anchor_labeled and not q_mask[anchor_action]:
      skipped["anchor_not_labeled"] += 1
      continue
    audited_rows += 1
    if q_mask[anchor_action]:
      anchor_labeled_rows += 1
    pred_best = int(labeled_actions[np.argmax(model_q[idx][q_mask])])
    if pred_best == true_best:
      label_best_hits += 1
    if q_targets[pred_best] >= q_targets[true_best] - 1e-9:
      label_positive_or_tie_hits += 1
    q_head_margins.append(float(q_targets[true_best] - q_targets[pred_best]))
    sorted_pred = np.sort(model_q[idx][q_mask])
    if len(sorted_pred) >= 2:
      pred_margin = float(sorted_pred[-1] - sorted_pred[-2])
    else:
      pred_margin = 0.0
    if pred_margin < float(args.action_value_rerank_min_margin):
      deployed_abstained_margin += 1
      deployed_action = anchor_action
    else:
      deployed_applied += 1
      clipped_q = np.array(model_q[idx], dtype=np.float32)
      clip = max(0.0, float(args.action_value_rerank_clip))
      if clip > 0:
        clipped_q = np.clip(clipped_q, -clip, clip)
      score = (
          np.log(np.maximum(model_policy[idx], 1e-12))
          + (
              float(args.action_value_selection_weight)
              * clipped_q
              * float(getattr(model_args, "value_scale", 1.0))
          )
          - float(args.action_paradox_selection_penalty) * model_risk[idx]
      )
      score = np.where(legal_masks[idx], score, -np.inf)
      deployed_action = int(np.argmax(score))
    if q_mask[deployed_action]:
      deployed_selected_labeled += 1
    if q_mask[anchor_action]:
      deployed_anchor_labeled += 1
    if deployed_action == true_best:
      deployed_label_best_hits += 1
    if q_mask[deployed_action] and q_mask[anchor_action]:
      margin = float(q_targets[deployed_action] - q_targets[anchor_action])
      deployed_true_margins.append(margin)
    if deployed_action != anchor_action:
      deployed_overrides += 1
      if q_mask[deployed_action]:
        deployed_override_label_supported += 1
      if deployed_action == true_best:
        deployed_override_to_label_best += 1
      if q_mask[deployed_action] and q_mask[anchor_action]:
        margin = float(q_targets[deployed_action] - q_targets[anchor_action])
        deployed_override_true_margins.append(margin)
        if margin > 0:
          deployed_override_positive += 1
    if q_mask[anchor_action]:
      q_head_regrets.append(float(q_targets[true_best] - q_targets[anchor_action]))

  valid_count = len(valid)
  report = {
      "checkpoint": args.checkpoint,
      "anchor_checkpoint": args.anchor_checkpoint,
      "inputs": input_counts,
      "rows": len(rows),
      "valid_rows": valid_count,
      "audited_rows": audited_rows,
      "skipped": skipped,
      "action_value_selection_weight": float(args.action_value_selection_weight),
      "action_paradox_selection_penalty": float(args.action_paradox_selection_penalty),
      "action_value_rerank_min_margin": float(args.action_value_rerank_min_margin),
      "action_value_rerank_clip": float(args.action_value_rerank_clip),
      "value_scale": float(getattr(model_args, "value_scale", 1.0)),
      "label_best_top1_rate_on_labeled_actions": _rate(
          label_best_hits, audited_rows
      ),
      "label_positive_or_tie_rate_on_labeled_actions": _rate(
          label_positive_or_tie_hits, audited_rows
      ),
      "q_head_mean_regret_to_label_best": _mean(q_head_margins),
      "q_head_mean_anchor_regret_when_anchor_labeled": _mean(q_head_regrets),
      "anchor_labeled_rows": anchor_labeled_rows,
      "anchor_labeled_rate": _rate(anchor_labeled_rows, audited_rows),
      "deployed_applied_rows": deployed_applied,
      "deployed_abstained_margin_rows": deployed_abstained_margin,
      "deployed_override_rows": deployed_overrides,
      "deployed_override_rate": _rate(deployed_overrides, valid_count),
      "deployed_override_rate_per_audited_row": _rate(
          deployed_overrides, audited_rows
      ),
      "deployed_selected_labeled_rows": deployed_selected_labeled,
      "deployed_selected_labeled_rate": _rate(
          deployed_selected_labeled, audited_rows
      ),
      "deployed_anchor_labeled_rows": deployed_anchor_labeled,
      "deployed_label_best_rate": _rate(deployed_label_best_hits, audited_rows),
      "deployed_mean_true_margin_when_labeled": _mean(deployed_true_margins),
      "deployed_true_margin_ci95_when_labeled": _ci95_mean(deployed_true_margins),
      "deployed_override_label_supported_rows": deployed_override_label_supported,
      "deployed_override_label_supported_rate": _rate(
          deployed_override_label_supported, deployed_overrides
      ),
      "deployed_override_to_label_best_rows": deployed_override_to_label_best,
      "deployed_override_to_label_best_rate": _rate(
          deployed_override_to_label_best, deployed_overrides
      ),
      "deployed_override_positive_margin_rows": deployed_override_positive,
      "deployed_override_positive_margin_rate_when_comparable": _rate(
          deployed_override_positive, len(deployed_override_true_margins)
      ),
      "deployed_override_mean_true_margin_when_comparable": _mean(
          deployed_override_true_margins
      ),
      "deployed_override_true_margin_ci95_when_comparable": _ci95_mean(
          deployed_override_true_margins
      ),
      "deployed_override_comparable_rows": len(deployed_override_true_margins),
  }
  return report


def main():
  args = parse_args()
  report = audit(args)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
  main()
