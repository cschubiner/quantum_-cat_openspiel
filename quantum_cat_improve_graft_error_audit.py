#!/usr/bin/env python3
"""Audit improve_graft false positives and missed positives on dense replay rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quantum_cat_alphazero_torch import (
    ACTION_FEATURE_FOLLOWS_LED_INDEX,
    ACTION_FEATURE_IS_RED_INDEX,
    ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    ACTION_FEATURE_PREFIX_SIZE,
    ACTION_FEATURE_RANK_NORM_INDEX,
    APPENDED_ACTION_FEATURE_INDEX,
    load_replay,
)
from quantum_cat_calibration_transfer_audit import (
    ImproveMLP,
    _load_policy_context_model,
    _policy_probs_from_row,
    pair_feature_vector_from_matrix,
)
from quantum_cat_table_liveness_improve_trainer import (
    _choose_candidates,
    _load_metadata_rows,
    _metadata_group_key,
    _metadata_path_for,
    _parse_float_list,
    _split_state_ids,
    _split_state_ids_by_group,
)


FEATURES_TO_AUDIT = {
    "rank_norm": ACTION_FEATURE_RANK_NORM_INDEX,
    "follows_led": ACTION_FEATURE_FOLLOWS_LED_INDEX,
    "off_led_loses_token": ACTION_FEATURE_OFF_LED_LOSES_TOKEN_INDEX,
    "is_red": ACTION_FEATURE_IS_RED_INDEX,
    "would_win": ACTION_FEATURE_PREFIX_SIZE + 19,
    "token_loss_newly_loses_led": APPENDED_ACTION_FEATURE_INDEX[
        "token_loss_newly_loses_led"
    ],
    "token_loss_legal_lead_damage": APPENDED_ACTION_FEATURE_INDEX[
        "token_loss_legal_lead_damage"
    ],
    "token_loss_creates_no_exit": APPENDED_ACTION_FEATURE_INDEX[
        "token_loss_creates_no_exit"
    ],
    "post_hit_low_legal_lead_ratio": APPENDED_ACTION_FEATURE_INDEX[
        "post_hit_low_legal_lead_ratio"
    ],
    "post_hit_survival_margin": APPENDED_ACTION_FEATURE_INDEX[
        "post_hit_low_card_survival_margin"
    ],
    "post_hit_forced_pressure": APPENDED_ACTION_FEATURE_INDEX[
        "post_hit_forced_card_pressure"
    ],
    "future_token_loss_rank_deficit": APPENDED_ACTION_FEATURE_INDEX[
        "future_token_loss_max_rank_deficit_after"
    ],
    "future_token_loss_buffer_deficit": APPENDED_ACTION_FEATURE_INDEX[
        "future_token_loss_max_buffer_deficit_after"
    ],
    "future_token_loss_no_exit_frac": APPENDED_ACTION_FEATURE_INDEX[
        "future_token_loss_no_exit_frac_after"
    ],
    "future_token_loss_safe_flex_drop": APPENDED_ACTION_FEATURE_INDEX[
        "future_token_loss_safe_flex_drop_after"
    ],
    "exit_public_slot_damage": APPENDED_ACTION_FEATURE_INDEX[
        "exit_public_slot_damage"
    ],
    "exit_own_public_slot_damage": APPENDED_ACTION_FEATURE_INDEX[
        "exit_own_public_slot_damage"
    ],
    "exit_min_player_open_slots_after": APPENDED_ACTION_FEATURE_INDEX[
        "exit_min_player_open_slots_after"
    ],
    "exit_min_player_lane_surplus_after": APPENDED_ACTION_FEATURE_INDEX[
        "exit_min_player_lane_surplus_after"
    ],
    "exit_lane_surplus_damage": APPENDED_ACTION_FEATURE_INDEX[
        "exit_lane_surplus_damage"
    ],
    "exit_min_lane_surplus_damage": APPENDED_ACTION_FEATURE_INDEX[
        "exit_min_lane_surplus_damage"
    ],
    "exit_lane_pressure_player_count_after": APPENDED_ACTION_FEATURE_INDEX[
        "exit_lane_pressure_player_count_after"
    ],
    "own_future_zero_exit_frac_after": APPENDED_ACTION_FEATURE_INDEX[
        "own_future_zero_exit_frac_after"
    ],
    "own_future_one_exit_frac_after": APPENDED_ACTION_FEATURE_INDEX[
        "own_future_one_exit_frac_after"
    ],
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", required=True)
  parser.add_argument("--metadata", default="")
  parser.add_argument("--model", required=True)
  parser.add_argument("--policy-context-checkpoint", required=True)
  parser.add_argument("--players", type=int, default=4)
  parser.add_argument("--seed", type=int, default=20299221)
  parser.add_argument("--validation-fraction", type=float, default=0.25)
  parser.add_argument(
      "--validation-split",
      choices=("state", "metadata_group"),
      default="metadata_group",
  )
  parser.add_argument(
      "--split",
      choices=("all", "train", "validation"),
      default="validation",
  )
  parser.add_argument("--positive-value-margin", type=float, default=0.04)
  parser.add_argument("--safe-risk-delta-max", type=float, default=0.0)
  parser.add_argument("--negative-value-margin", type=float, default=0.0)
  parser.add_argument("--max-candidates-per-state", type=int, default=0)
  parser.add_argument("--threshold", type=float, default=0.5)
  parser.add_argument(
      "--risk-threshold",
      type=float,
      default=-1.0,
      help="Negative disables the safe-prob threshold.",
  )
  parser.add_argument(
      "--threshold-grid",
      default="0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99",
  )
  parser.add_argument(
      "--risk-threshold-grid",
      default="0.5,0.7,0.85,0.9,0.95,0.97,0.99",
  )
  parser.add_argument("--sample-limit", type=int, default=12)
  parser.add_argument("--out", required=True)
  return parser.parse_args()


def _load_scorer(path: str):
  payload = torch.load(path, map_location="cpu", weights_only=False)
  saved_args = payload.get("args", {})
  feature_size = int(payload.get("feature_size", len(payload["feature_mean"])))
  hidden = int(saved_args.get("hidden", 64))
  model = ImproveMLP(feature_size, hidden)
  model.load_state_dict(payload["model_state"])
  model.eval()
  risk_model = None
  if payload.get("risk_model_state") is not None:
    risk_model = ImproveMLP(feature_size, hidden)
    risk_model.load_state_dict(payload["risk_model_state"])
    risk_model.eval()
  std = np.asarray(payload["feature_std"], dtype=np.float32)
  std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
  return {
      "model": model,
      "risk_model": risk_model,
      "mean": np.asarray(payload["feature_mean"], dtype=np.float32),
      "std": std,
      "observation_feature_mode": str(
          payload.get("observation_feature_mode")
          or saved_args.get("observation_feature_mode")
          or "none"
      ),
      "policy_context_mode": str(
          payload.get("policy_context_mode")
          or saved_args.get("policy_context_mode")
          or "none"
      ),
      "args": saved_args,
  }


def _predict(model, features):
  with torch.no_grad():
    logits = model(torch.tensor(features, dtype=torch.float32))
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


def _feature_snapshot(action_features, candidate, base):
  row = {}
  for name, idx in FEATURES_TO_AUDIT.items():
    if idx >= action_features.shape[1]:
      continue
    cand = float(action_features[candidate, idx])
    base_value = float(action_features[base, idx])
    row[f"candidate_{name}"] = cand
    row[f"base_{name}"] = base_value
    row[f"delta_{name}"] = cand - base_value
  return row


def _build_states(args, rows, metadata_rows, scorer):
  policy_args = SimpleNamespace(
      policy_context_mode=str(scorer["policy_context_mode"]),
      policy_context_checkpoint=str(args.policy_context_checkpoint),
      policy_context_device="cpu",
      players=int(args.players),
  )
  policy_model, policy_device = _load_policy_context_model(policy_args, rows[0])
  examples = []
  states = []
  for state_idx, row in enumerate(rows):
    legal_mask = np.asarray(row[1], dtype=np.float32) > 0.5
    value_mask = np.asarray(row[10], dtype=np.float32) > 0.5
    risk_mask = np.asarray(row[8], dtype=np.float32) > 0.5
    labeled_mask = legal_mask & value_mask & risk_mask
    legal = np.flatnonzero(labeled_mask)
    if len(legal) < 2:
      continue
    policy = _policy_probs_from_row(row, policy_model, policy_device)
    if policy is None or float(np.sum(policy[legal])) <= 0.0:
      continue
    meta = metadata_rows[state_idx]
    base_action = int(meta.get("policy_best_action", -1))
    if base_action not in set(int(action) for action in legal):
      continue
    values = np.asarray(row[9], dtype=np.float32)
    risks = np.asarray(row[7], dtype=np.float32)
    base_value = float(values[base_action])
    base_risk = float(risks[base_action])
    candidates = _choose_candidates(
        legal,
        base_action,
        values,
        risks,
        int(args.max_candidates_per_state),
    )
    action_features = np.asarray(row[11], dtype=np.float32)
    state_record = {
        "state_index": int(state_idx),
        "group_key": _metadata_group_key(meta, state_idx),
        "base_action": int(base_action),
        "base_value": float(base_value),
        "base_risk": float(base_risk),
        "candidates": [],
        "meta": meta,
    }
    for candidate in candidates:
      value_delta = float(values[candidate]) - base_value
      risk_delta = float(risks[candidate]) - base_risk
      positive = (
          value_delta >= float(args.positive_value_margin)
          and risk_delta <= float(args.safe_risk_delta_max)
      )
      safe = risk_delta <= float(args.safe_risk_delta_max)
      feature = pair_feature_vector_from_matrix(
          action_features,
          legal_mask.astype(np.float32),
          int(candidate),
          int(base_action),
          target_action=int(candidate),
          label_margin=0.0,
          legal_count=int(len(legal)),
          candidate_count=int(len(candidates)),
          include_label_margin=False,
          observation=row[0],
          observation_feature_mode=str(scorer["observation_feature_mode"]),
          policy_context=policy,
          policy_context_mode=str(scorer["policy_context_mode"]),
          include_target_match_features=False,
      )
      standardized = (feature.astype(np.float32) - scorer["mean"]) / scorer["std"]
      example = {
          "state_index": int(state_idx),
          "group_key": state_record["group_key"],
          "candidate_action": int(candidate),
          "base_action": int(base_action),
          "policy_prob": float(policy[candidate]),
          "value_delta": float(value_delta),
          "risk_delta": float(risk_delta),
          "positive": bool(positive),
          "safe": bool(safe),
          "feature": standardized.astype(np.float32),
          "action_snapshot": _feature_snapshot(action_features, candidate, base_action),
      }
      examples.append(example)
      state_record["candidates"].append(example)
    states.append(state_record)
  if not states:
    raise ValueError("no states built")
  features = np.asarray([example["feature"] for example in examples], dtype=np.float32)
  probs = _predict(scorer["model"], features)
  safe_probs = (
      _predict(scorer["risk_model"], features)
      if scorer["risk_model"] is not None else
      np.ones_like(probs, dtype=np.float32)
  )
  for example, prob, safe_prob in zip(examples, probs, safe_probs):
    example["prob"] = float(prob)
    example["safe_prob"] = float(safe_prob)
  return states


def _split_states(args, states):
  if str(args.validation_split) == "metadata_group":
    train_ids, validation_ids, split_report = _split_state_ids_by_group(
        states, float(args.validation_fraction), int(args.seed)
    )
  else:
    state_ids = [int(state["state_index"]) for state in states]
    train_ids, validation_ids = _split_state_ids(
        state_ids, float(args.validation_fraction), int(args.seed)
    )
    split_report = {
        "split": "state",
        "groups": None,
        "train_groups": None,
        "validation_groups": None,
    }
  wanted = set(
      int(state["state_index"])
      for state in states
  )
  if args.split == "train":
    wanted = set(train_ids)
  elif args.split == "validation":
    wanted = set(validation_ids)
  selected = [state for state in states if int(state["state_index"]) in wanted]
  return selected, split_report, {
      "train_states": int(len(train_ids)),
      "validation_states": int(len(validation_ids)),
  }


def _selected_candidate(state, threshold, risk_threshold):
  if not state["candidates"]:
    return None
  best = max(
      state["candidates"],
      key=lambda row: (float(row["prob"]), float(row["policy_prob"])),
  )
  if float(best["prob"]) < threshold:
    return None
  if risk_threshold is not None and float(best["safe_prob"]) < risk_threshold:
    return None
  return best


def _summarize_feature_rows(rows):
  if not rows:
    return {"count": 0}
  numeric_keys = sorted(
      key for row in rows for key, value in row.items()
      if isinstance(value, (int, float))
  )
  summary = {"count": int(len(rows))}
  for key in numeric_keys:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
      continue
    summary[key] = {
        "mean": round(float(np.mean(values)), 6),
        "min": round(float(np.min(values)), 6),
        "max": round(float(np.max(values)), 6),
    }
  return summary


def _compact_example(example):
  keys = [
      "state_index",
      "group_key",
      "candidate_action",
      "base_action",
      "prob",
      "safe_prob",
      "policy_prob",
      "value_delta",
      "risk_delta",
      "positive",
      "safe",
  ]
  row = {key: example[key] for key in keys if key in example}
  row.update(example.get("action_snapshot", {}))
  return row


def _threshold_summary(states, thresholds, risk_thresholds):
  reports = []
  for threshold in thresholds:
    for risk_threshold in [None] + list(risk_thresholds):
      selected = [
          _selected_candidate(state, threshold, risk_threshold)
          for state in states
      ]
      selected = [row for row in selected if row is not None]
      positives = sum(1 for row in selected if bool(row["positive"]))
      safe = sum(1 for row in selected if bool(row["safe"]))
      reports.append({
          "threshold": float(threshold),
          "risk_threshold": None if risk_threshold is None else float(risk_threshold),
          "states": int(len(states)),
          "overrides": int(len(selected)),
          "positive_precision": (
              None if not selected else round(float(positives / len(selected)), 6)
          ),
          "safe_rate": (
              None if not selected else round(float(safe / len(selected)), 6)
          ),
          "mean_value_delta": (
              None if not selected else round(
                  float(np.mean([row["value_delta"] for row in selected])), 6
              )
          ),
          "mean_risk_delta": (
              None if not selected else round(
                  float(np.mean([row["risk_delta"] for row in selected])), 6
              )
          ),
      })
  return reports


def _audit(args, states):
  risk_threshold = None if float(args.risk_threshold) < 0 else float(args.risk_threshold)
  false_positives = []
  true_positives = []
  missed_positives = []
  abstained_positive_states = 0
  false_selected_positive_states = 0
  for state in states:
    selected = _selected_candidate(state, float(args.threshold), risk_threshold)
    positives = [row for row in state["candidates"] if bool(row["positive"])]
    best_positive = (
        max(positives, key=lambda row: (float(row["value_delta"]), -float(row["risk_delta"])))
        if positives else None
    )
    if selected is None:
      if best_positive is not None:
        abstained_positive_states += 1
        missed_positives.append(best_positive)
      continue
    if bool(selected["positive"]):
      true_positives.append(selected)
    else:
      false_positives.append(selected)
      if best_positive is not None:
        false_selected_positive_states += 1
        missed_positives.append(best_positive)
  selected_count = len(true_positives) + len(false_positives)
  return {
      "states": int(len(states)),
      "threshold": float(args.threshold),
      "risk_threshold": risk_threshold,
      "selected_overrides": int(selected_count),
      "true_positives": int(len(true_positives)),
      "false_positives": int(len(false_positives)),
      "precision": (
          None if selected_count == 0 else
          round(float(len(true_positives) / selected_count), 6)
      ),
      "abstained_positive_states": int(abstained_positive_states),
      "false_selected_positive_states": int(false_selected_positive_states),
      "missed_positive_rows": int(len(missed_positives)),
      "feature_summary": {
          "true_positives": _summarize_feature_rows(
              [row["action_snapshot"] for row in true_positives]
          ),
          "false_positives": _summarize_feature_rows(
              [row["action_snapshot"] for row in false_positives]
          ),
          "missed_positives": _summarize_feature_rows(
              [row["action_snapshot"] for row in missed_positives]
          ),
      },
      "samples": {
          "false_positives": [
              _compact_example(row)
              for row in sorted(false_positives, key=lambda item: item["prob"], reverse=True)[
                  : int(args.sample_limit)
              ]
          ],
          "missed_positives": [
              _compact_example(row)
              for row in sorted(
                  missed_positives,
                  key=lambda item: (item["value_delta"], -item["risk_delta"]),
                  reverse=True,
              )[: int(args.sample_limit)]
          ],
          "true_positives": [
              _compact_example(row)
              for row in sorted(true_positives, key=lambda item: item["prob"], reverse=True)[
                  : int(args.sample_limit)
              ]
          ],
      },
  }


def main():
  args = parse_args()
  rows = load_replay(args.replay)
  metadata_path = Path(args.metadata) if args.metadata else _metadata_path_for(
      args.replay
  )
  metadata_rows = _load_metadata_rows(metadata_path)
  if len(rows) != len(metadata_rows):
    raise ValueError(f"replay/metadata row mismatch: {len(rows)} vs {len(metadata_rows)}")
  scorer = _load_scorer(args.model)
  states = _build_states(args, rows, metadata_rows, scorer)
  selected_states, split_report, split_counts = _split_states(args, states)
  result = {
      "replay": str(args.replay),
      "metadata": str(metadata_path),
      "model": str(args.model),
      "split": str(args.split),
      "split_report": split_report,
      **split_counts,
      "scorer": {
          "observation_feature_mode": scorer["observation_feature_mode"],
          "policy_context_mode": scorer["policy_context_mode"],
          "has_risk_model": bool(scorer["risk_model"] is not None),
      },
      "threshold_grid": _threshold_summary(
          selected_states,
          _parse_float_list(args.threshold_grid),
          _parse_float_list(args.risk_threshold_grid),
      ),
      "audit": _audit(args, selected_states),
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
      "out": str(out),
      "states": int(len(selected_states)),
      "selected_overrides": result["audit"]["selected_overrides"],
      "false_positives": result["audit"]["false_positives"],
      "missed_positive_rows": result["audit"]["missed_positive_rows"],
      "precision": result["audit"]["precision"],
  }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
