#!/usr/bin/env python3
"""Offline listwise verifier for safe-improvement calibration rows.

The pairwise improve-graft MLP sees engineered candidate-vs-policy features.
This diagnostic asks whether a small learned set encoder over every legal action
can separate audited improvements from false positives better.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_cat_alphazero_torch import adapt_action_features, load_replay
from quantum_cat_calibration_transfer_audit import (
    _load_metadata,
    _load_policy_context_model,
    _metadata_group_key,
    _metadata_path_for,
    _metadata_source_key,
    _metadata_source_namespace,
    _policy_probs_from_row,
    _passing_across_inputs,
    candidate_calibration_metadata_rows,
    observation_feature_vector,
    policy_context_feature_vector,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--calibration-replay", action="append", required=True)
  parser.add_argument("--metadata", action="append", default=[])
  parser.add_argument(
      "--validation-replay",
      action="append",
      default=[],
      help=(
          "Optional held-out calibration replay(s). When set, training uses "
          "--calibration-replay only and validation uses these replay(s) only."
      ),
  )
  parser.add_argument("--validation-metadata", action="append", default=[])
  parser.add_argument(
      "--require-explicit-source-disjoint",
      action=argparse.BooleanOptionalAction,
      default=False,
  )
  parser.add_argument(
      "--require-each-validation-replay-pass",
      action=argparse.BooleanOptionalAction,
      default=False,
  )
  parser.add_argument(
      "--filter-validation-rows",
      action="store_true",
      help=(
          "Apply the same stability filters to --validation-replay rows. By "
          "default, external validation keeps all audited rows to better "
          "estimate deployment behavior."
      ),
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--model-out", default="")
  parser.add_argument("--seed", type=int, default=20262263)
  parser.add_argument("--validation-fraction", type=float, default=0.35)
  parser.add_argument(
      "--validation-split",
      choices=("row", "group", "source_group"),
      default="source_group",
  )
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--epochs", type=int, default=600)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--weight-decay", type=float, default=0.002)
  parser.add_argument("--hidden", type=int, default=96)
  parser.add_argument("--action-hidden", type=int, default=64)
  parser.add_argument("--dropout", type=float, default=0.05)
  parser.add_argument("--positive-label-margin-min", type=float, default=1.0)
  parser.add_argument(
      "--near-positive-train-label",
      type=float,
      default=0.0,
      help=(
          "Soft training target for NEAR_POSITIVE rows. Default 0.0 keeps "
          "historical strict POSITIVE-only binary training; values such as "
          "0.25-0.5 test a curriculum/ranking target without changing the "
          "deployment gate."
      ),
  )
  parser.add_argument(
      "--near-positive-label-margin-min",
      type=float,
      default=0.0,
      help="Minimum audited margin required before a NEAR_POSITIVE row gets the soft target.",
  )
  parser.add_argument(
      "--unstable-positive-train-label",
      type=float,
      default=0.0,
      help="Soft training target for UNSTABLE_POSITIVE rows. Default keeps them negative.",
  )
  parser.add_argument(
      "--include-outcome-features",
      action="store_true",
      help=(
          "Legacy/debug only: include audited margin/paradox labels as input "
          "features. Leave off for deployable diagnostics."
      ),
  )
  parser.add_argument(
      "--multi-head",
      action="store_true",
      help="Also train auxiliary heads for audited margin and paradox delta.",
  )
  parser.add_argument("--margin-loss-weight", type=float, default=0.0)
  parser.add_argument("--paradox-loss-weight", type=float, default=0.0)
  parser.add_argument(
      "--use-predicted-risk-gate",
      action="store_true",
      help=(
          "During threshold reporting, require auxiliary predictions to pass "
          "the configured margin/paradox gates before counting an override."
      ),
  )
  parser.add_argument("--predicted-margin-min", type=float, default=0.0)
  parser.add_argument("--predicted-paradox-max", type=float, default=0.0)
  parser.add_argument(
      "--min-top1-mode-rate",
      type=float,
      default=0.0,
      help="Drop rows whose repeated teacher top-1 agreement is below this value.",
  )
  parser.add_argument(
      "--min-pairwise-spearman",
      type=float,
      default=-1.0,
      help="Drop rows whose repeated teacher pairwise Spearman mean is below this value.",
  )
  parser.add_argument(
      "--min-top2-jaccard",
      type=float,
      default=0.0,
      help="Drop rows whose repeated teacher top-2 Jaccard mean is below this value.",
  )
  parser.add_argument(
      "--min-consensus-top-margin",
      type=float,
      default=0.0,
      help="Drop rows whose teacher consensus top margin is below this value.",
  )
  parser.add_argument(
      "--require-high-margin",
      action="store_true",
      help="Drop rows unless metadata marks the teacher row as high_margin.",
  )
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--policy-context-device", default="cpu")
  parser.add_argument(
      "--thresholds",
      default="0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95",
  )
  parser.add_argument("--target-coverage-min", type=float, default=0.03)
  parser.add_argument("--target-coverage-max", type=float, default=0.20)
  parser.add_argument("--target-min-overrides", type=int, default=1)
  parser.add_argument(
      "--target-min-true-margin-positive-overrides", type=int, default=1
  )
  parser.add_argument("--target-precision-min", type=float, default=0.70)
  parser.add_argument("--target-mean-margin-min", type=float, default=0.50)
  parser.add_argument("--target-false-positive-rate-max", type=float, default=0.15)
  parser.add_argument("--target-paradox-delta-max", type=float, default=0.0)
  return parser.parse_args()


class ListwiseVerifier(nn.Module):
  def __init__(self, action_size, extra_size, action_hidden, hidden, dropout):
    super().__init__()
    self.action_encoder = nn.Sequential(
        nn.Linear(action_size, action_hidden),
        nn.ReLU(),
        nn.Linear(action_hidden, action_hidden),
        nn.ReLU(),
    )
    pair_size = (
        extra_size
        + 8
        + action_hidden * 8
    )
    self.net = nn.Sequential(
        nn.Linear(pair_size, hidden),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    self.margin_head = nn.Sequential(
        nn.Linear(pair_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    self.paradox_head = nn.Sequential(
        nn.Linear(pair_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )

  def forward(self, action_features, legal_mask, candidate_actions, policy_actions, extra):
    emb = self.action_encoder(action_features)
    mask = legal_mask.bool()
    mask_f = mask.unsqueeze(-1).to(emb.dtype)
    legal_count = mask_f.sum(dim=1).clamp_min(1.0)
    mean = (emb * mask_f).sum(dim=1) / legal_count
    max_emb = emb.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
    min_emb = emb.masked_fill(~mask.unsqueeze(-1), 1e9).min(dim=1).values
    has_legal = mask.any(dim=1, keepdim=True)
    max_emb = torch.where(has_legal, max_emb, torch.zeros_like(max_emb))
    min_emb = torch.where(has_legal, min_emb, torch.zeros_like(min_emb))
    centered = (emb - mean.unsqueeze(1)) * mask_f
    std = torch.sqrt((centered * centered).sum(dim=1) / legal_count + 1e-8)
    batch = torch.arange(action_features.shape[0], device=action_features.device)
    cand = emb[batch, candidate_actions]
    policy = emb[batch, policy_actions]
    legal_scalar = torch.stack([
        legal_mask.sum(dim=1),
        candidate_actions.to(extra.dtype),
        policy_actions.to(extra.dtype),
        (candidate_actions == policy_actions).to(extra.dtype),
        torch.norm(cand - policy, dim=1),
        torch.norm(cand - mean, dim=1),
        torch.norm(policy - mean, dim=1),
        torch.sum((cand - policy) * (cand - mean), dim=1),
    ], dim=1)
    features = torch.cat([
        extra,
        legal_scalar,
        cand,
        policy,
        cand - policy,
        cand * policy,
        mean,
        max_emb,
        min_emb,
        std,
    ], dim=1)
    return (
        self.net(features).squeeze(-1),
        self.margin_head(features).squeeze(-1),
        self.paradox_head(features).squeeze(-1),
    )


def _set_seed(seed):
  random.seed(int(seed))
  np.random.seed(int(seed) % (2**32 - 1))
  torch.manual_seed(int(seed))


def _stable_filter_reason(meta, args):
  if bool(getattr(args, "require_high_margin", False)) and not bool(
      meta.get("high_margin")
  ):
    return "not_high_margin"
  checks = [
      ("top1_mode_rate", "min_top1_mode_rate"),
      ("pairwise_spearman_mean", "min_pairwise_spearman"),
      ("top2_jaccard_mean", "min_top2_jaccard"),
      ("consensus_top_margin", "min_consensus_top_margin"),
  ]
  for meta_key, arg_key in checks:
    minimum = float(getattr(args, arg_key))
    if meta_key == "pairwise_spearman_mean" and minimum <= -1.0:
      continue
    if minimum <= 0.0:
      continue
    value = meta.get(meta_key)
    if value is None or float(value) < minimum:
      return f"{meta_key}_below_min"
  return None


def _load_examples_for_paths(args, replay_paths, metadata_paths, apply_filters=True):
  replay_paths = [str(path) for path in replay_paths]
  metadata_paths = [str(path) for path in metadata_paths]
  if metadata_paths and len(metadata_paths) != len(replay_paths):
    raise ValueError("--metadata count must match --calibration-replay count")
  if not metadata_paths:
    metadata_paths = [str(_metadata_path_for(path)) for path in replay_paths]

  rows_by_source = []
  policy_model = None
  policy_device = None
  model_args = SimpleNamespace(
      policy_context_mode="summary",
      policy_context_checkpoint=args.policy_context_checkpoint,
      policy_context_device=args.policy_context_device,
      players=args.players,
  )
  input_counts = {}
  examples = []
  for replay_path, metadata_path in zip(replay_paths, metadata_paths):
    rows = load_replay(replay_path)
    metadata = _load_metadata(metadata_path)
    source_namespace = _metadata_source_namespace(metadata_path)
    input_counts[replay_path] = {
        "replay_rows": int(len(rows)),
        "metadata_rows": int(len(metadata)),
        "metadata": metadata_path,
        "stability_filters_applied": bool(apply_filters),
        "loaded_rows": 0,
        "skipped_rows": 0,
        "skipped_by_reason": {},
    }
    if len(rows) != len(metadata):
      raise ValueError(
          f"Replay/metadata row mismatch for {replay_path}: "
          f"{len(rows)} vs {len(metadata)}"
      )
    if policy_model is None and rows:
      policy_model, policy_device = _load_policy_context_model(model_args, rows[0])
    rows_by_source.append((
        replay_path, rows, metadata, source_namespace
    ))

  for replay_path, rows, metadata, source_namespace in rows_by_source:
    for idx, (row, meta) in enumerate(zip(rows, metadata)):
      filter_reason = _stable_filter_reason(meta, args) if apply_filters else None
      if filter_reason is not None:
        input_counts[replay_path]["skipped_rows"] += 1
        skipped = input_counts[replay_path]["skipped_by_reason"]
        skipped[filter_reason] = int(skipped.get(filter_reason, 0)) + 1
        continue
      source_target = int(meta["deployment_target_action"])
      replay_target = int(row[5]) if len(row) > 5 else -1
      if replay_target != source_target:
        raise ValueError(
            f"Replay target mismatch at {replay_path}:{idx}: "
            f"{replay_target} vs {source_target}"
        )
      action_features = adapt_action_features(
          row[11] if len(row) > 11 else None, len(row[1])
      )
      legal_mask = np.asarray(row[1], dtype=np.float32)
      policy_probs = _policy_probs_from_row(row, policy_model, policy_device)
      obs_features = observation_feature_vector(row[0], mode="summary")
      for candidate_meta in candidate_calibration_metadata_rows(meta):
        row_class = str(
            candidate_meta.get("calibration_class", "UNKNOWN")
        )
        if row_class == "UNAUDITED":
          input_counts[replay_path]["skipped_rows"] += 1
          skipped = input_counts[replay_path]["skipped_by_reason"]
          skipped["unaudited"] = int(skipped.get("unaudited", 0)) + 1
          continue
        label_action = int(candidate_meta["label_best_action"])
        policy_action = int(candidate_meta["policy_best_action"])
        policy_features = policy_context_feature_vector(
            policy_probs,
            legal_mask,
            label_action,
            policy_action,
            mode="summary",
        )
        audit_margin = float(
            candidate_meta.get("audit_margin_vs_policy") or 0.0
        )
        paradox_delta = float(
            candidate_meta.get("audit_paradox_delta_vs_policy") or 0.0
        )
        train_label = 0.0
        train_positive = (
            row_class == "POSITIVE"
            and audit_margin >= float(args.positive_label_margin_min)
        )
        if train_positive:
          train_label = 1.0
        elif (
            row_class == "NEAR_POSITIVE"
            and audit_margin >= float(args.near_positive_label_margin_min)
        ):
          train_label = float(args.near_positive_train_label)
        elif row_class == "UNSTABLE_POSITIVE":
          train_label = float(args.unstable_positive_train_label)
        train_label = min(1.0, max(0.0, float(train_label)))
        outcome_margin_feature = (
            audit_margin if args.include_outcome_features else 0.0
        )
        outcome_paradox_feature = (
            paradox_delta if args.include_outcome_features else 0.0
        )
        extra = np.concatenate([
            obs_features,
            policy_features,
            np.array([
                float(candidate_meta.get("legal_count") or np.sum(legal_mask)),
                float(candidate_meta.get("candidate_count") or 0.0),
                float(outcome_margin_feature),
                float(outcome_paradox_feature),
            ], dtype=np.float32),
        ]).astype(np.float32)
        examples.append({
            "input": replay_path,
            "row_index": int(idx),
            "candidate_rank": int(candidate_meta.get("candidate_rank", 0)),
            "source_key": _metadata_source_key(
                meta, replay_path, source_namespace
            ),
            "group_key": _metadata_group_key(meta, replay_path, idx),
            "action_features": action_features.astype(np.float32),
            "legal_mask": legal_mask.astype(np.float32),
            "candidate_action": label_action,
            "policy_action": policy_action,
            "extra": extra,
            "label": train_label,
            "strict_train_positive": bool(train_positive),
            "class": row_class,
            "audit_margin": audit_margin,
            "paradox_delta": paradox_delta,
            "false_positive": bool(
                candidate_meta.get("false_positive_for_positive_gate")
            ),
        })
        input_counts[replay_path]["loaded_rows"] += 1
  return examples, input_counts


def _load_examples(args):
  return _load_examples_for_paths(
      args,
      args.calibration_replay,
      getattr(args, "metadata", []),
      apply_filters=True,
  )


def _split_indices(examples, validation_fraction, seed, mode):
  rng = random.Random(int(seed))
  if mode == "row":
    by_label = {0: [], 1: []}
    for idx, example in enumerate(examples):
      by_label[int(example["label"])].append(idx)
    train, val = [], []
    for indices in by_label.values():
      rng.shuffle(indices)
      if len(indices) <= 1:
        train.extend(indices)
        continue
      val_count = max(1, min(len(indices) - 1, int(round(len(indices) * validation_fraction))))
      val.extend(indices[:val_count])
      train.extend(indices[val_count:])
    return train, val, {"split": "row"}

  group_rows = {}
  for idx, example in enumerate(examples):
    key = example["group_key"]
    if mode == "source_group":
      key = f"{example['input']}::{key}"
    group_rows.setdefault(key, []).append(idx)
  buckets = {}
  for group, rows in group_rows.items():
    source = examples[rows[0]]["input"] if mode == "source_group" else "__all__"
    has_positive = int(any(examples[idx]["label"] > 0.5 for idx in rows))
    buckets.setdefault((source, has_positive), []).append(group)
  val_groups = set()
  train_groups = set()
  for _, groups in sorted(buckets.items(), key=lambda item: str(item[0])):
    groups = list(groups)
    rng.shuffle(groups)
    if len(groups) <= 1:
      train_groups.update(groups)
      continue
    val_count = max(1, min(len(groups) - 1, int(round(len(groups) * validation_fraction))))
    val_groups.update(groups[:val_count])
    train_groups.update(groups[val_count:])
  train, val = [], []
  for group, rows in group_rows.items():
    if group in val_groups:
      val.extend(rows)
    else:
      train.extend(rows)
  return train, val, {
      "split": mode,
      "groups": int(len(group_rows)),
      "train_groups": int(len(train_groups)),
      "validation_groups": int(len(val_groups)),
      "buckets": {
          f"{source}|label={label}": int(len(groups))
          for (source, label), groups in sorted(buckets.items(), key=lambda item: str(item[0]))
      },
  }


def _stack(examples, indices, action_mean=None, action_std=None, extra_mean=None, extra_std=None):
  actions = np.array([examples[idx]["action_features"] for idx in indices], dtype=np.float32)
  masks = np.array([examples[idx]["legal_mask"] for idx in indices], dtype=np.float32)
  cand = np.array([examples[idx]["candidate_action"] for idx in indices], dtype=np.int64)
  policy = np.array([examples[idx]["policy_action"] for idx in indices], dtype=np.int64)
  extra = np.array([examples[idx]["extra"] for idx in indices], dtype=np.float32)
  if action_mean is None:
    legal = masks.astype(bool)
    legal_features = actions[legal]
    action_mean = legal_features.mean(axis=0)
    action_std = legal_features.std(axis=0)
    action_std = np.where(action_std < 1e-6, 1.0, action_std)
  if extra_mean is None:
    extra_mean = extra.mean(axis=0)
    extra_std = extra.std(axis=0)
    extra_std = np.where(extra_std < 1e-6, 1.0, extra_std)
  actions = (actions - action_mean) / action_std
  extra = (extra - extra_mean) / extra_std
  labels = np.array([examples[idx]["label"] for idx in indices], dtype=np.float32)
  margins = np.array([examples[idx]["audit_margin"] for idx in indices], dtype=np.float32)
  paradoxes = np.array([examples[idx]["paradox_delta"] for idx in indices], dtype=np.float32)
  return (
      actions,
      masks,
      cand,
      policy,
      extra,
      labels,
      margins,
      paradoxes,
  ), action_mean, action_std, extra_mean, extra_std


def _predict(model, data):
  actions, masks, cand, policy, extra, *_ = data
  with torch.no_grad():
    logits, margin_pred, paradox_pred = model(
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.float32),
        torch.tensor(cand, dtype=torch.long),
        torch.tensor(policy, dtype=torch.long),
        torch.tensor(extra, dtype=torch.float32),
    )
    return (
        torch.sigmoid(logits).cpu().numpy().astype(np.float32),
        margin_pred.cpu().numpy().astype(np.float32),
        paradox_pred.cpu().numpy().astype(np.float32),
    )


def _rate(num, den):
  return None if den <= 0 else round(float(num) / float(den), 6)


def _mean(values):
  return None if not values else round(float(np.mean(values)), 6)


def _threshold_report(
    examples,
    indices,
    probs,
    thresholds,
    args,
    margin_preds=None,
    paradox_preds=None,
):
  rows = [examples[idx] for idx in indices]
  false_positive_rows = [row for row in rows if row["false_positive"]]
  reports = []
  for threshold in thresholds:
    override_pairs = []
    for i, prob in enumerate(probs):
      if float(prob) < float(threshold):
        continue
      if bool(args.use_predicted_risk_gate):
        if margin_preds is None or paradox_preds is None:
          continue
        if float(margin_preds[i]) <= float(args.predicted_margin_min):
          continue
        if float(paradox_preds[i]) > float(args.predicted_paradox_max):
          continue
      override_pairs.append((i, rows[i]))
    overrides = [row for _, row in override_pairs]
    positives = [row for row in overrides if float(row["audit_margin"]) > 0.0]
    false_positive_overrides = [row for row in overrides if row["false_positive"]]
    coverage = _rate(len(overrides), len(rows))
    precision = _rate(len(positives), len(overrides))
    false_positive_rate = _rate(len(false_positive_overrides), len(false_positive_rows))
    mean_margin = _mean([float(row["audit_margin"]) for row in overrides])
    mean_paradox = _mean([float(row["paradox_delta"]) for row in overrides])
    mean_pred_margin = _mean(
        [float(margin_preds[i]) for i, _ in override_pairs]
        if margin_preds is not None else []
    )
    mean_pred_paradox = _mean(
        [float(paradox_preds[i]) for i, _ in override_pairs]
        if paradox_preds is not None else []
    )
    pass_gate = bool(
        coverage is not None
        and precision is not None
        and false_positive_rate is not None
        and mean_margin is not None
        and mean_paradox is not None
        and len(overrides) >= int(args.target_min_overrides)
        and len(positives)
        >= int(args.target_min_true_margin_positive_overrides)
        and coverage >= float(args.target_coverage_min)
        and coverage <= float(args.target_coverage_max)
        and precision >= float(args.target_precision_min)
        and mean_margin > float(args.target_mean_margin_min)
        and false_positive_rate <= float(args.target_false_positive_rate_max)
        and mean_paradox <= float(args.target_paradox_delta_max)
    )
    reports.append({
        "threshold": float(threshold),
        "risk_threshold": None,
        "rows": int(len(rows)),
        "overrides": int(len(overrides)),
        "true_margin_positive_overrides": int(len(positives)),
        "coverage": coverage,
        "precision_true_margin_positive": precision,
        "positive_class_override_rate": _rate(
            sum(1 for row in overrides if row["class"] == "POSITIVE"),
            sum(1 for row in rows if row["class"] == "POSITIVE"),
        ),
        "near_positive_class_override_rate": _rate(
            sum(1 for row in overrides if row["class"] == "NEAR_POSITIVE"),
            sum(1 for row in rows if row["class"] == "NEAR_POSITIVE"),
        ),
        "soft_positive_class_override_rate": _rate(
            sum(1 for row in overrides if float(row["label"]) > 0.0),
            sum(1 for row in rows if float(row["label"]) > 0.0),
        ),
        "false_positive_row_override_rate": false_positive_rate,
        "mean_true_margin": mean_margin,
        "mean_paradox_delta": mean_paradox,
        "mean_predicted_margin": mean_pred_margin,
        "mean_predicted_paradox_delta": mean_pred_paradox,
        "pass_offline_gate": pass_gate,
    })
  return reports


def main():
  args = parse_args()
  _set_seed(args.seed)
  thresholds = [float(part) for part in str(args.thresholds).split(",") if part.strip()]
  train_source_examples, input_counts = _load_examples(args)
  validation_source_examples = []
  validation_input_counts = {}
  if getattr(args, "validation_replay", None):
    validation_source_examples, validation_input_counts = _load_examples_for_paths(
        args,
        args.validation_replay,
        getattr(args, "validation_metadata", []),
        apply_filters=bool(args.filter_validation_rows),
    )
  examples = list(train_source_examples) + list(validation_source_examples)
  if not examples:
    raise ValueError("No calibration examples")
  if validation_source_examples:
    train_idx = list(range(len(train_source_examples)))
    val_idx = list(
        range(len(train_source_examples), len(examples))
    )
    split_report = {
        "split": "external_validation_replay",
        "train_rows": int(len(train_source_examples)),
        "validation_rows": int(len(validation_source_examples)),
        "validation_inputs": validation_input_counts,
    }
    train_sources = {row["source_key"] for row in train_source_examples}
    validation_sources = {
        row["source_key"] for row in validation_source_examples
    }
    overlap = sorted(train_sources & validation_sources)
    split_report.update({
        "train_source_count": int(len(train_sources)),
        "validation_source_count": int(len(validation_sources)),
        "source_overlap": overlap,
        "source_disjoint": not overlap,
    })
    if bool(args.require_explicit_source_disjoint) and overlap:
      raise ValueError(
          "Explicit validation proposal sources overlap training sources: "
          + ", ".join(overlap)
      )
  else:
    train_idx, val_idx, split_report = _split_indices(
        examples, args.validation_fraction, args.seed, args.validation_split
    )
  if not val_idx:
    val_idx = list(train_idx)
  train_data, action_mean, action_std, extra_mean, extra_std = _stack(examples, train_idx)
  val_data, _, _, _, _ = _stack(
      examples, val_idx, action_mean, action_std, extra_mean, extra_std
  )
  all_data, _, _, _, _ = _stack(
      examples, list(range(len(examples))), action_mean, action_std, extra_mean, extra_std
  )
  model = ListwiseVerifier(
      train_data[0].shape[-1],
      train_data[4].shape[-1],
      int(args.action_hidden),
      int(args.hidden),
      float(args.dropout),
  )
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
  )
  labels = torch.tensor(train_data[5], dtype=torch.float32)
  margin_targets = torch.tensor(train_data[6], dtype=torch.float32)
  paradox_targets = torch.tensor(train_data[7], dtype=torch.float32)
  positives = float(torch.sum(labels).item())
  negatives = float(len(labels) - positives)
  pos_weight = torch.tensor([
      negatives / max(1.0, positives) if positives and negatives else 1.0
  ], dtype=torch.float32)
  tensors = [
      torch.tensor(train_data[0], dtype=torch.float32),
      torch.tensor(train_data[1], dtype=torch.float32),
      torch.tensor(train_data[2], dtype=torch.long),
      torch.tensor(train_data[3], dtype=torch.long),
      torch.tensor(train_data[4], dtype=torch.float32),
  ]
  losses = []
  bce_losses = []
  margin_losses = []
  paradox_losses = []
  for _ in range(max(1, int(args.epochs))):
    logits, margin_pred, paradox_pred = model(*tensors)
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight
    )
    margin_loss = F.smooth_l1_loss(margin_pred, margin_targets)
    paradox_loss = F.smooth_l1_loss(paradox_pred, paradox_targets)
    loss = bce_loss
    if bool(args.multi_head):
      loss = (
          loss
          + float(args.margin_loss_weight) * margin_loss
          + float(args.paradox_loss_weight) * paradox_loss
      )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
    bce_losses.append(float(bce_loss.detach()))
    margin_losses.append(float(margin_loss.detach()))
    paradox_losses.append(float(paradox_loss.detach()))

  train_probs, train_margin_preds, train_paradox_preds = _predict(model, train_data)
  val_probs, val_margin_preds, val_paradox_preds = _predict(model, val_data)
  all_probs, all_margin_preds, all_paradox_preds = _predict(model, all_data)
  train_thresholds = _threshold_report(
      examples,
      train_idx,
      train_probs,
      thresholds,
      args,
      train_margin_preds,
      train_paradox_preds,
  )
  validation_thresholds = _threshold_report(
      examples,
      val_idx,
      val_probs,
      thresholds,
      args,
      val_margin_preds,
      val_paradox_preds,
  )
  all_thresholds = _threshold_report(
      examples,
      list(range(len(examples))),
      all_probs,
      thresholds,
      args,
      all_margin_preds,
      all_paradox_preds,
  )
  passing_validation = [
      row for row in validation_thresholds if row["pass_offline_gate"]
  ]
  validation_thresholds_by_input = {}
  if validation_source_examples:
    for input_name in args.validation_replay:
      positions = [
          local_index
          for local_index, example_index in enumerate(val_idx)
          if examples[example_index]["input"] == str(input_name)
      ]
      input_indices = [val_idx[pos] for pos in positions]
      validation_thresholds_by_input[str(input_name)] = _threshold_report(
          examples,
          input_indices,
          val_probs[positions],
          thresholds,
          args,
          val_margin_preds[positions],
          val_paradox_preds[positions],
      )
  passing_all_validation_inputs = _passing_across_inputs(
      validation_thresholds, validation_thresholds_by_input
  )
  best_validation = max(
      validation_thresholds,
      key=lambda row: (
          row["pass_offline_gate"],
          row["precision_true_margin_positive"] or -1.0,
          row["mean_true_margin"] or -999.0,
          -(row["false_positive_row_override_rate"] or 999.0),
      ),
  )
  result = {
      "inputs": input_counts,
      "validation_inputs": validation_input_counts,
      "rows": int(len(examples)),
      "classes": {
          key: int(sum(1 for row in examples if row["class"] == key))
          for key in sorted({row["class"] for row in examples})
      },
      "seed": int(args.seed),
      "validation_split": str(args.validation_split),
      "validation_fraction": float(args.validation_fraction),
      "split_report": split_report,
      "train_rows": int(len(train_idx)),
      "validation_rows": int(len(val_idx)),
      "train_positive_rows": int(sum(examples[idx]["label"] > 0.5 for idx in train_idx)),
      "validation_positive_rows": int(sum(examples[idx]["label"] > 0.5 for idx in val_idx)),
      "train_soft_positive_rows": int(sum(examples[idx]["label"] > 0.0 for idx in train_idx)),
      "validation_soft_positive_rows": int(sum(examples[idx]["label"] > 0.0 for idx in val_idx)),
      "train_label_sum": round(float(sum(examples[idx]["label"] for idx in train_idx)), 6),
      "validation_label_sum": round(float(sum(examples[idx]["label"] for idx in val_idx)), 6),
      "near_positive_train_label": float(args.near_positive_train_label),
      "near_positive_label_margin_min": float(args.near_positive_label_margin_min),
      "unstable_positive_train_label": float(args.unstable_positive_train_label),
      "train_report": {
          "initial_loss": round(losses[0], 6),
          "final_loss": round(losses[-1], 6),
          "initial_bce_loss": round(bce_losses[0], 6),
          "final_bce_loss": round(bce_losses[-1], 6),
          "initial_margin_loss": round(margin_losses[0], 6),
          "final_margin_loss": round(margin_losses[-1], 6),
          "initial_paradox_loss": round(paradox_losses[0], 6),
          "final_paradox_loss": round(paradox_losses[-1], 6),
          "epochs": int(args.epochs),
          "hidden": int(args.hidden),
          "action_hidden": int(args.action_hidden),
          "pos_weight": round(float(pos_weight.item()), 6),
          "multi_head": bool(args.multi_head),
          "margin_loss_weight": float(args.margin_loss_weight),
          "paradox_loss_weight": float(args.paradox_loss_weight),
          "include_outcome_features": bool(args.include_outcome_features),
          "use_predicted_risk_gate": bool(args.use_predicted_risk_gate),
          "predicted_margin_min": float(args.predicted_margin_min),
          "predicted_paradox_max": float(args.predicted_paradox_max),
          "target_min_overrides": int(args.target_min_overrides),
          "target_min_true_margin_positive_overrides": int(
              args.target_min_true_margin_positive_overrides
          ),
      },
      "train_thresholds": train_thresholds,
      "validation_thresholds": validation_thresholds,
      "validation_thresholds_by_input": validation_thresholds_by_input,
      "all_thresholds": all_thresholds,
      "best_validation_threshold": best_validation,
      "passing_validation_thresholds": passing_validation,
      "passing_all_validation_inputs": passing_all_validation_inputs,
      "require_explicit_source_disjoint": bool(
          args.require_explicit_source_disjoint
      ),
      "require_each_validation_replay_pass": bool(
          args.require_each_validation_replay_pass
      ),
      "offline_gate_pass": bool(
          passing_all_validation_inputs
          if args.require_each_validation_replay_pass
          else passing_validation
      ),
  }
  if args.model_out:
    out_model = Path(args.model_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std.astype(np.float32),
        "extra_mean": extra_mean.astype(np.float32),
        "extra_std": extra_std.astype(np.float32),
        "args": vars(args),
        "model_kind": "listwise_verifier",
        "action_size": int(train_data[0].shape[-1]),
        "extra_size": int(train_data[4].shape[-1]),
        "multi_head": bool(args.multi_head),
    }, out_model)
    result["model_out"] = str(out_model)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
