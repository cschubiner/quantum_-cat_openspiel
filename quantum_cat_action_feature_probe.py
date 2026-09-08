#!/usr/bin/env python3
"""Train a feature-only action scorer on Quantum Cat Q-label replays.

This is a representation diagnostic, not a production bot. It answers a narrow
question: are the per-action tactical features sufficient to rank Q-labeled
legal actions out of sample, without the full state encoder or MCTS machinery?
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_SIZE,
    adapt_action_features,
    load_replay,
    policy_target_bucket_masks,
)


class FeatureOnlyActionScorer(nn.Module):
  """Scores each candidate action from its action-feature row only."""

  def __init__(self, feature_size: int, width: int, depth: int):
    super().__init__()
    layers = []
    in_size = feature_size
    for _ in range(depth):
      layers.append(nn.Linear(in_size, width))
      layers.append(nn.ReLU())
      in_size = width
    layers.append(nn.Linear(in_size, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, action_features):
    return self.net(action_features).squeeze(-1)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-input", action="append", required=True)
  parser.add_argument(
      "--validation",
      action="append",
      default=[],
      help="Named validation replay as name=path. May be repeated.",
  )
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--steps", type=int, default=400)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--width", type=int, default=128)
  parser.add_argument("--depth", type=int, default=2)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-3)
  parser.add_argument("--temperature", type=float, default=0.06)
  parser.add_argument("--min-actions", type=int, default=2)
  parser.add_argument("--seed", type=int, default=20262790)
  parser.add_argument(
      "--standardize",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Standardize action-feature columns using labeled train actions.",
  )
  return parser.parse_args()


def _load_rows(paths):
  rows = []
  counts = {}
  for path in paths:
    loaded = load_replay(path)
    counts[path] = len(loaded)
    rows.extend(loaded)
  return rows, counts


def _labeled_rows(rows, min_actions):
  result = []
  for row in rows:
    if len(row) <= 11 or row[9] is None or row[10] is None:
      continue
    target_mask = np.array(row[10], dtype=np.float32) > 0.0
    if int(np.sum(target_mask)) < int(min_actions):
      continue
    action_features = adapt_action_features(row[11], len(row[1]))
    result.append((
        np.array(row[1], dtype=bool),
        np.array(row[9], dtype=np.float32),
        target_mask,
        action_features,
    ))
  return result


def _feature_stats(rows):
  selected = []
  for _legal_mask, _targets, target_mask, action_features in rows:
    selected.append(action_features[target_mask])
  if not selected:
    mean = np.zeros(ACTION_FEATURE_SIZE, dtype=np.float32)
    std = np.ones(ACTION_FEATURE_SIZE, dtype=np.float32)
    return mean, std
  matrix = np.concatenate(selected, axis=0).astype(np.float32)
  mean = matrix.mean(axis=0).astype(np.float32)
  std = matrix.std(axis=0).astype(np.float32)
  std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
  return mean, std


def _normalize_features(action_features, mean, std):
  return (action_features.astype(np.float32) - mean) / std


def _batch(rows, batch_size, rng):
  if len(rows) >= batch_size:
    return rng.sample(rows, batch_size)
  return [rng.choice(rows) for _ in range(batch_size)]


def _q_soft_targets(targets_t, target_mask_t, temperature):
  logits = targets_t / max(1e-6, float(temperature))
  logits = logits.masked_fill(~target_mask_t, -1e9)
  return torch.softmax(logits, dim=1)


def _train_loss(model, batch_rows, mean, std, temperature, device):
  features = np.stack([
      _normalize_features(row[3], mean, std) for row in batch_rows
  ]).astype(np.float32)
  targets = np.stack([row[1] for row in batch_rows]).astype(np.float32)
  target_masks = np.stack([row[2] for row in batch_rows]).astype(bool)
  features_t = torch.tensor(features, dtype=torch.float32, device=device)
  targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
  target_mask_t = torch.tensor(target_masks, dtype=torch.bool, device=device)
  logits = model(features_t).masked_fill(~target_mask_t, -1e9)
  log_probs = F.log_softmax(logits, dim=1)
  q_policy = _q_soft_targets(targets_t, target_mask_t, temperature)
  return -(q_policy * log_probs).sum(dim=1).mean()


def _summary_from_logits(
    pred_matrix, target_vectors, target_masks, action_features,
    include_buckets=True,
):
  selected = target_masks > 0.0
  preds = pred_matrix[selected]
  targets = target_vectors[selected]
  if len(preds) == 0:
    return None
  corr = None
  if (
      len(preds) > 1
      and float(np.std(preds)) > 0
      and float(np.std(targets)) > 0
  ):
    corr_candidate = float(np.corrcoef(preds, targets)[0, 1])
    if math.isfinite(corr_candidate):
      corr = corr_candidate
  state_corrs = []
  top_hits = []
  chosen_targets = []
  best_targets = []
  mean_targets = []
  regrets = []
  for pred_row, target_row, mask_row in zip(
      pred_matrix, target_vectors, target_masks
  ):
    labeled = mask_row > 0.0
    if int(np.sum(labeled)) < 2:
      continue
    labeled_preds = pred_row[labeled]
    labeled_targets = target_row[labeled]
    if float(np.std(labeled_preds)) > 0 and float(np.std(labeled_targets)) > 0:
      state_corr = float(np.corrcoef(labeled_preds, labeled_targets)[0, 1])
      if math.isfinite(state_corr):
        state_corrs.append(state_corr)
    chosen_index = int(np.argmax(labeled_preds))
    best_index = int(np.argmax(labeled_targets))
    chosen_target = float(labeled_targets[chosen_index])
    best_target = float(labeled_targets[best_index])
    chosen_targets.append(chosen_target)
    best_targets.append(best_target)
    mean_targets.append(float(np.mean(labeled_targets)))
    regrets.append(best_target - chosen_target)
    top_hits.append(1.0 if chosen_index == best_index else 0.0)
  summary = {
      "count": int(len(preds)),
      "target_mean": round(float(np.mean(targets)), 4),
      "logit_mean": round(float(np.mean(preds)), 4),
      "corr": None if corr is None else round(corr, 4),
      "state_count": int(len(chosen_targets)),
      "mean_state_corr": (
          None if not state_corrs else round(float(np.mean(state_corrs)), 4)
      ),
      "top1_rate": None if not top_hits else round(float(np.mean(top_hits)), 4),
      "top_policy_target_mean": (
          None if not chosen_targets else round(float(np.mean(chosen_targets)), 4)
      ),
      "best_target_mean": (
          None if not best_targets else round(float(np.mean(best_targets)), 4)
      ),
      "mean_target_mean": (
          None if not mean_targets else round(float(np.mean(mean_targets)), 4)
      ),
      "mean_regret": None if not regrets else round(float(np.mean(regrets)), 4),
  }

  by_bucket = {}
  if include_buckets:
    masked_targets = np.where(target_masks > 0.0, target_vectors, -np.inf)
    q_best_actions = np.argmax(masked_targets, axis=1)
    q_best_rows = action_features[np.arange(len(action_features)), q_best_actions]
    for name, bucket_mask in policy_target_bucket_masks(q_best_rows).items():
      if not np.any(bucket_mask):
        continue
      bucket = _summary_from_logits(
          pred_matrix[bucket_mask],
          target_vectors[bucket_mask],
          target_masks[bucket_mask],
          action_features[bucket_mask],
          include_buckets=False,
      )
      if bucket is not None:
        by_bucket[name] = bucket
  if include_buckets:
    summary["by_q_best_tactical_bucket"] = by_bucket
  return summary


def evaluate(model, rows, mean, std, device):
  if not rows:
    return None
  features = np.stack([
      _normalize_features(row[3], mean, std) for row in rows
  ]).astype(np.float32)
  raw_features = np.stack([row[3] for row in rows]).astype(np.float32)
  targets = np.stack([row[1] for row in rows]).astype(np.float32)
  target_masks = np.stack([row[2] for row in rows]).astype(np.float32)
  with torch.no_grad():
    logits = model(
        torch.tensor(features, dtype=torch.float32, device=device)
    ).cpu().numpy()
  return _summary_from_logits(logits, targets, target_masks, raw_features)


def _parse_validation(spec):
  if "=" not in spec:
    path = spec
    name = Path(path).stem
  else:
    name, path = spec.split("=", 1)
  name = name.strip()
  path = path.strip()
  if not name or not path:
    raise ValueError(f"Invalid validation spec: {spec!r}")
  return name, path


def main():
  args = parse_args()
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  device = torch.device("cpu")

  train_raw, train_counts = _load_rows(args.train_input)
  train_rows = _labeled_rows(train_raw, args.min_actions)
  if not train_rows:
    raise SystemExit("No labeled training rows found.")
  if args.standardize:
    mean, std = _feature_stats(train_rows)
  else:
    mean = np.zeros(ACTION_FEATURE_SIZE, dtype=np.float32)
    std = np.ones(ACTION_FEATURE_SIZE, dtype=np.float32)

  model = FeatureOnlyActionScorer(
      ACTION_FEATURE_SIZE, width=args.width, depth=args.depth
  ).to(device)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
  )
  rng = random.Random(args.seed)
  loss_trace = []
  for step in range(1, args.steps + 1):
    model.train()
    loss = _train_loss(
        model,
        _batch(train_rows, args.batch_size, rng),
        mean,
        std,
        args.temperature,
        device,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
      loss_trace.append({"step": step, "loss": round(float(loss.item()), 6)})

  model.eval()
  validation_reports = {}
  validation_counts = {}
  for spec in args.validation:
    name, path = _parse_validation(spec)
    rows, counts = _load_rows([path])
    labeled = _labeled_rows(rows, args.min_actions)
    validation_counts[name] = {
        "path": path,
        "rows": len(rows),
        "labeled_rows": len(labeled),
        "input_counts": counts,
    }
    validation_reports[name] = evaluate(model, labeled, mean, std, device)

  train_report = evaluate(model, train_rows, mean, std, device)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_path = out_dir / "feature_probe.pt"
  torch.save({
      "model_state_dict": model.state_dict(),
      "feature_mean": mean,
      "feature_std": std,
      "args": vars(args),
  }, checkpoint_path)
  summary = {
      "train_inputs": train_counts,
      "train_rows": len(train_raw),
      "train_labeled_rows": len(train_rows),
      "checkpoint": str(checkpoint_path),
      "loss_trace": loss_trace,
      "train_report": train_report,
      "validation_counts": validation_counts,
      "validation_reports": validation_reports,
  }
  summary_path = out_dir / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
