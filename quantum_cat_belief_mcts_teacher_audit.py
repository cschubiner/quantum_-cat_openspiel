#!/usr/bin/env python3
"""Audit repeated belief-MCTS teacher labels before distillation.

This is a Run290-style go/no-go diagnostic. It collects real policy decision
states, runs belief-averaged MCTS several times per state and budget, measures
label stability, then independently force-audits the consensus teacher action
against the checkpoint policy action with common rollout seeds.

The script intentionally does not train or promote a model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import
from quantum_cat_alphazero_torch import (
    load_model_payload,
    make_game,
    phase_name_for_state,
    save_replay,
    teacher_belief_mcts_policy,
)
from quantum_cat_label_oracle_audit import (
    accepted_replay_example,
    accepted_replay_summary,
    action_string,
    calibration_row_class,
    calibration_summary,
    calibration_target_action,
    choose_champion_action,
    evaluate_actions_independently,
)
from quantum_cat_q_teacher_stability_audit import (
    _collect_mixed_states,
    _collection_state_summary,
    _pairwise_mean,
    _spearman,
    _summarize_numbers,
    model_args_for,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--states", type=int, default=48)
  parser.add_argument("--max-matches", type=int, default=300)
  parser.add_argument("--max-states-per-match", type=int, default=2)
  parser.add_argument(
      "--state-mix",
      default="high_entropy=12,near_paradox=12,hard=12,random_play=12",
      help=(
          "Comma-separated bucket=count spec. Reuses the Q-teacher collector "
          "buckets: random_play/high_entropy/hard/loss_gap/near_paradox."
      ),
  )
  parser.add_argument(
      "--state-selection",
      choices=("first", "high_entropy", "hard"),
      default="high_entropy",
  )
  parser.add_argument("--state-pool-per-match", type=int, default=0)
  parser.add_argument("--state-min-legal-count", type=int, default=2)
  parser.add_argument("--state-max-legal-count", type=int, default=0)
  parser.add_argument("--state-min-trick-number", type=int, default=-1)
  parser.add_argument("--state-max-trick-number", type=int, default=-1)
  parser.add_argument("--state-min-match-round", type=int, default=-1)
  parser.add_argument("--state-max-match-round", type=int, default=-1)
  parser.add_argument("--hard-entropy-weight", type=float, default=1.0)
  parser.add_argument("--hard-loss-weight", type=float, default=0.25)
  parser.add_argument("--hard-paradox-weight", type=float, default=0.75)
  parser.add_argument("--phases", default="play")
  parser.add_argument(
      "--budgets",
      default="2x8,4x16",
      help="Comma-separated belief_samples x mcts_sims specs, e.g. 2x8,4x16.",
  )
  parser.add_argument("--repeat-label-passes", type=int, default=3)
  parser.add_argument("--min-policy-margin", type=float, default=0.05)
  parser.add_argument("--audit-rollouts", type=int, default=2)
  parser.add_argument("--audit-belief-samples", type=int, default=2)
  parser.add_argument("--audit-belief-candidates", type=int, default=8)
  parser.add_argument("--full-match-rollout", action="store_true")
  parser.add_argument("--rollout-max-plies", type=int, default=0)
  parser.add_argument("--match-context", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--c-puct", type=float, default=1.5)
  parser.add_argument("--teacher-belief-source", default="infostate")
  parser.add_argument("--teacher-belief-candidates", type=int, default=8)
  parser.add_argument("--teacher-belief-policy-temperature", type=float, default=1.0)
  parser.add_argument("--teacher-belief-uniform-mix", type=float, default=0.15)
  parser.add_argument("--teacher-belief-ref-policy-mix", default="model:1.0")
  parser.add_argument("--teacher-belief-ref-policy-mix-by-phase", default="")
  parser.add_argument("--pass-min-high-margin-rows", type=int, default=8)
  parser.add_argument("--pass-min-high-margin-top1-stability", type=float, default=0.60)
  parser.add_argument("--pass-min-top2-jaccard", type=float, default=0.75)
  parser.add_argument("--pass-min-spearman", type=float, default=0.35)
  parser.add_argument("--pass-min-forced-score-rate", type=float, default=0.55)
  parser.add_argument("--pass-min-forced-avg-diff", type=float, default=0.0)
  parser.add_argument("--pass-max-forced-paradox-delta", type=float, default=0.0)
  parser.add_argument("--seed", type=int, default=20262240)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument("--progress-interval", type=int, default=0)
  parser.add_argument("--collect-only", action="store_true")
  parser.add_argument("--calibration-replay-out", default="")
  parser.add_argument("--calibration-positive-margin", type=float, default=0.5)
  parser.add_argument("--calibration-max-positive-paradox-delta", type=float, default=0.0)
  parser.add_argument(
      "--calibration-target-positive-rows",
      type=int,
      default=0,
      help=(
          "If exporting calibration replay, stop auditing once at least this "
          "many POSITIVE rows have been found. Zero disables the target."
      ),
  )
  parser.add_argument(
      "--calibration-target-negative-rows",
      type=int,
      default=0,
      help=(
          "If exporting calibration replay, stop auditing once at least this "
          "many NEGATIVE rows have been found. Zero disables the target."
      ),
  )
  parser.add_argument(
      "--calibration-abort-after-states",
      type=int,
      default=0,
      help=(
          "If exporting calibration replay, stop early after this many "
          "evaluated states when the stable positive count is below "
          "--calibration-abort-min-positive-rows. Zero disables the check."
      ),
  )
  parser.add_argument(
      "--calibration-abort-min-positive-rows",
      type=int,
      default=0,
      help=(
          "Minimum stable POSITIVE rows required by "
          "--calibration-abort-after-states. This is a seed-screening guard, "
          "not a success target."
      ),
  )
  parser.add_argument(
      "--calibration-include-policy-matches",
      action="store_true",
      help="Export teacher-policy agreement rows as NEGATIVE no-override examples.",
  )
  parser.add_argument(
      "--calibration-target-require-stable",
      action="store_true",
      help=(
          "When calibration row targets are set, count only rows that pass the "
          "calibration stable-row filters toward the early-stop targets."
      ),
  )
  parser.add_argument(
      "--calibration-demote-unstable-positives",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "When stable-row filters are active, export raw POSITIVE rows that "
          "fail those filters as hard no-override UNSTABLE_POSITIVE rows."
      ),
  )
  parser.add_argument("--calibration-stable-min-top1-mode-rate", type=float, default=0.0)
  parser.add_argument("--calibration-stable-min-pairwise-spearman", type=float, default=-1.0)
  parser.add_argument("--calibration-stable-min-top2-jaccard", type=float, default=0.0)
  parser.add_argument("--calibration-stable-min-consensus-top-margin", type=float, default=0.0)
  parser.add_argument(
      "--calibration-stable-require-high-margin",
      action="store_true",
  )
  parser.add_argument("--out", required=True)
  return parser.parse_args()


def _set_seed(seed):
  seed = int(seed)
  random.seed(seed)
  np.random.seed(seed % (2**32 - 1))
  torch.manual_seed(seed)


def _device(name):
  if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
    return torch.device("mps")
  return torch.device("cpu")


def _parse_budgets(spec):
  budgets = []
  for raw in str(spec or "").split(","):
    item = raw.strip().lower()
    if not item:
      continue
    if "x" in item:
      samples, sims = item.split("x", 1)
    elif ":" in item:
      samples, sims = item.split(":", 1)
    else:
      raise ValueError(f"Bad budget spec {raw!r}; expected samplesxsims")
    budgets.append((max(1, int(samples)), max(1, int(sims))))
  if not budgets:
    raise ValueError("--budgets must include at least one samplesxsims item")
  return budgets


def _teacher_args(args, saved_args, samples, sims):
  value_scale = float(saved_args.get("value_scale", 20.0))
  return SimpleNamespace(
      players=int(args.players),
      value_scale=value_scale,
      match_context=bool(args.match_context),
      full_match_training=bool(args.match_context),
      c_puct=float(args.c_puct),
      sims=int(sims),
      teacher_sims=int(sims),
      teacher_belief_samples=int(samples),
      teacher_belief_source=str(args.teacher_belief_source),
      teacher_belief_candidates=max(int(samples), int(args.teacher_belief_candidates)),
      teacher_belief_policy_temperature=float(args.teacher_belief_policy_temperature),
      teacher_belief_uniform_mix=float(args.teacher_belief_uniform_mix),
      teacher_belief_ref_policy_mix=str(args.teacher_belief_ref_policy_mix),
      teacher_belief_ref_policy_mix_by_phase=str(
          args.teacher_belief_ref_policy_mix_by_phase
      ),
      teacher_belief_logprob_floor=1e-6,
      paradox_value_penalty=0.0,
      action_paradox_selection_penalty=0.0,
      action_paradox_root_only=False,
      action_value_selection_weight=0.0,
      action_value_root_only=False,
  )


def _audit_args(args, saved_args):
  value_scale = float(saved_args.get("value_scale", 20.0))
  return SimpleNamespace(
      players=int(args.players),
      value_scale=value_scale,
      match_context=bool(args.match_context),
      full_match_training=bool(args.match_context),
      c_puct=float(args.c_puct),
      sims=1,
      self_play_policy_mode="policy",
      self_play_belief_samples=1,
      self_play_belief_sims=1,
      league_opponent_mode="policy",
      eval_belief_samples=max(1, int(args.audit_belief_samples)),
      eval_belief_sims=1,
      counterfactual_full_match_rollout=bool(args.full_match_rollout),
      counterfactual_rollout_max_plies=max(0, int(args.rollout_max_plies)),
      counterfactual_rollout_opponent_mode="",
      counterfactual_belief_source="infostate",
      counterfactual_belief_samples=max(1, int(args.audit_belief_samples)),
      counterfactual_belief_candidates=max(
          int(args.audit_belief_samples), int(args.audit_belief_candidates)
      ),
      counterfactual_belief_policy_temperature=1.0,
      counterfactual_belief_uniform_mix=0.0,
      counterfactual_belief_ref_policy_mix="model:1.0",
      counterfactual_belief_ref_policy_mix_by_phase="",
      counterfactual_belief_logprob_floor=1e-6,
  )


def _entropy(probs):
  values = np.asarray(probs, dtype=np.float64)
  total = float(np.sum(values))
  if total <= 0:
    return 0.0
  values = values / total
  return float(-np.sum(values * np.log(np.maximum(values, 1e-12))))


def _top2_set(score_map):
  ordered = sorted(score_map, key=lambda action: score_map[action], reverse=True)
  return set(int(action) for action in ordered[: min(2, len(ordered))])


def _forced_summary(rows, high_margin_only, disagreement_only=False):
  selected = [
      row for row in rows
      if row.get("evaluable")
      and (not high_margin_only or row.get("high_margin"))
      and (not disagreement_only or not row.get("consensus_matches_policy"))
      and row.get("forced_audit", {}).get("audited")
      and row.get("forced_audit", {}).get("margin_vs_policy") is not None
  ]
  if not selected:
    return {
        "rows": 0,
        "score_rate": None,
        "mean_margin": None,
        "mean_paradox_delta": None,
        "wins": 0,
        "ties": 0,
        "losses": 0,
  }
  margins = [float(row["forced_audit"]["margin_vs_policy"]) for row in selected]
  paradoxes = [
      float(row["forced_audit"]["paradox_delta_vs_policy"])
      for row in selected
      if row["forced_audit"].get("paradox_delta_vs_policy") is not None
  ]
  wins = sum(1 for margin in margins if margin > 0)
  ties = sum(1 for margin in margins if margin == 0)
  losses = sum(1 for margin in margins if margin < 0)
  return {
      "rows": int(len(selected)),
      "score_rate": float((wins + 0.5 * ties) / len(selected)),
      "mean_margin": float(np.mean(margins)),
      "mean_paradox_delta": None if not paradoxes else float(np.mean(paradoxes)),
      "wins": int(wins),
      "ties": int(ties),
      "losses": int(losses),
  }


def _stability_summary(rows):
  evaluable = [row for row in rows if row.get("evaluable")]
  high_margin = [row for row in evaluable if row.get("high_margin")]
  return {
      "rows": int(len(rows)),
      "evaluable_rows": int(len(evaluable)),
      "high_margin_rows": int(len(high_margin)),
      "top1_mode_rate": _summarize_numbers(
          [row.get("top1_mode_rate") for row in evaluable]
      ),
      "high_margin_top1_mode_rate": _summarize_numbers(
          [row.get("top1_mode_rate") for row in high_margin]
      ),
      "pairwise_spearman": _summarize_numbers(
          [row.get("pairwise_spearman_mean") for row in evaluable]
      ),
      "high_margin_pairwise_spearman": _summarize_numbers(
          [row.get("pairwise_spearman_mean") for row in high_margin]
      ),
      "top2_jaccard": _summarize_numbers(
          [row.get("top2_jaccard_mean") for row in evaluable]
      ),
      "high_margin_top2_jaccard": _summarize_numbers(
          [row.get("top2_jaccard_mean") for row in high_margin]
      ),
      "consensus_top_margin": _summarize_numbers(
          [row.get("consensus_top_margin") for row in evaluable]
      ),
      "teacher_entropy": _summarize_numbers(
          [row.get("mean_teacher_entropy") for row in evaluable]
      ),
      "forced_audit": _forced_summary(rows, high_margin_only=False),
      "high_margin_forced_audit": _forced_summary(rows, high_margin_only=True),
      "forced_disagreement_audit": _forced_summary(
          rows, high_margin_only=False, disagreement_only=True
      ),
      "high_margin_forced_disagreement_audit": _forced_summary(
          rows, high_margin_only=True, disagreement_only=True
      ),
  }


def _bucket_summary(rows, key):
  report = {}
  for row in rows:
    if not row.get("evaluable"):
      continue
    value = row.get(key)
    if value is None and key == "state_mix_bucket":
      value = (row.get("group") or {}).get("state_mix_bucket")
    value = "unknown" if value is None else str(value)
    report.setdefault(value, []).append(row)
  return {name: _stability_summary(bucket) for name, bucket in sorted(report.items())}


def _build_calibration_replay(states, rows, audit_ns, args):
  replay = []
  metadata = []
  skipped = {}
  for idx, row in enumerate(rows):
    if idx >= len(states):
      skipped["missing_state"] = skipped.get("missing_state", 0) + 1
      continue
    if not row.get("evaluable"):
      skipped["not_evaluable"] = skipped.get("not_evaluable", 0) + 1
      continue
    forced = row.get("forced_audit") or {}
    if not forced.get("audited"):
      skipped["not_audited"] = skipped.get("not_audited", 0) + 1
      continue
    if row.get("consensus_matches_policy") and not args.calibration_include_policy_matches:
      skipped["matches_policy"] = skipped.get("matches_policy", 0) + 1
      continue
    label_action = int(row["consensus_best_action"])
    policy_action = int(row["policy_best_action"])
    audit_margin = forced.get("margin_vs_policy")
    paradox_delta = forced.get("paradox_delta_vs_policy")
    class_input = {
        "audited": True,
        "audit_margin_vs_policy": audit_margin,
        "audit_paradox_delta_vs_policy": paradox_delta,
        "label_best_action": label_action,
        "policy_best_action": policy_action,
    }
    raw_row_class, row_class, stable_reason, stability_demoted = (
        _calibration_class_details_for_row(row, args)
    )
    if row_class is None:
      skipped["unclassified"] = skipped.get("unclassified", 0) + 1
      continue
    target_action = calibration_target_action(class_input, row_class)
    raw_stats = forced.get("raw") or {}
    replay.append(
        accepted_replay_example(
            states[idx],
            int(row["player"]),
            int(target_action),
            policy_action,
            raw_stats,
            audit_ns,
            label_stats=raw_stats,
            scored_actions={label_action, policy_action},
        )
    )
    metadata.append({
        "source_index": int(idx),
        "source": "belief_mcts_teacher_audit",
        "phase": row.get("phase"),
        "player": int(row["player"]),
        "group": row.get("group") or {},
        "policy_best_action": policy_action,
        "policy_best_action_string": row.get("policy_best_action_string"),
        "label_best_action": label_action,
        "label_best_action_string": row.get("consensus_best_action_string"),
        "deployment_target_action": int(target_action),
        "deployment_target_action_string": action_string(
            states[idx], int(row["player"]), int(target_action)
        ),
        "calibration_class": row_class,
        "raw_calibration_class": raw_row_class,
        "calibration_stable_filter_reason": stable_reason,
        "stability_demoted": bool(stability_demoted),
        "false_positive_for_positive_gate": bool(row_class != "POSITIVE"),
        "label_margin_vs_policy": (
            None if audit_margin is None else float(audit_margin)
        ),
        "audit_margin_vs_policy": (
            None if audit_margin is None else float(audit_margin)
        ),
        "audit_paradox_delta_vs_policy": (
            None if paradox_delta is None else float(paradox_delta)
        ),
        "legal_count": row.get("legal_count"),
        "candidate_count": row.get("legal_count"),
        "top1_mode_rate": row.get("top1_mode_rate"),
        "pairwise_spearman_mean": row.get("pairwise_spearman_mean"),
        "top2_jaccard_mean": row.get("top2_jaccard_mean"),
        "consensus_top_margin": row.get("consensus_top_margin"),
        "high_margin": bool(row.get("high_margin")),
        "forced_audit": forced,
    })
  return replay, metadata, skipped


def _calibration_class_for_row(row, args):
  _, row_class, _, _ = _calibration_class_details_for_row(row, args)
  return row_class


def _calibration_class_details_for_row(row, args):
  if not row.get("evaluable"):
    return None, None, None, False
  forced = row.get("forced_audit") or {}
  if not forced.get("audited"):
    return None, None, None, False
  if row.get("consensus_matches_policy"):
    row_class = "NEGATIVE" if args.calibration_include_policy_matches else None
    return row_class, row_class, _calibration_stable_filter_reason(row, args), False
  class_input = {
      "audited": True,
      "audit_margin_vs_policy": forced.get("margin_vs_policy"),
      "audit_paradox_delta_vs_policy": forced.get("paradox_delta_vs_policy"),
      "label_best_action": int(row["consensus_best_action"]),
      "policy_best_action": int(row["policy_best_action"]),
  }
  raw_row_class = calibration_row_class(
      class_input,
      args.calibration_positive_margin,
      args.calibration_max_positive_paradox_delta,
  )
  stable_reason = _calibration_stable_filter_reason(row, args)
  stability_demoted = bool(
      raw_row_class == "POSITIVE"
      and stable_reason is not None
      and bool(args.calibration_demote_unstable_positives)
  )
  row_class = "UNSTABLE_POSITIVE" if stability_demoted else raw_row_class
  return raw_row_class, row_class, stable_reason, stability_demoted


def _calibration_stable_filter_reason(row, args):
  if bool(getattr(args, "calibration_stable_require_high_margin", False)) and not bool(
      row.get("high_margin")
  ):
    return "not_high_margin"
  checks = [
      ("top1_mode_rate", "calibration_stable_min_top1_mode_rate"),
      ("pairwise_spearman_mean", "calibration_stable_min_pairwise_spearman"),
      ("top2_jaccard_mean", "calibration_stable_min_top2_jaccard"),
      ("consensus_top_margin", "calibration_stable_min_consensus_top_margin"),
  ]
  for row_key, arg_key in checks:
    minimum = float(getattr(args, arg_key))
    if row_key == "pairwise_spearman_mean" and minimum <= -1.0:
      continue
    if minimum <= 0.0:
      continue
    value = row.get(row_key)
    if value is None or float(value) < minimum:
      return f"{row_key}_below_min"
  return None


def _calibration_targets_satisfied(class_counts, args):
  positive_target = max(0, int(args.calibration_target_positive_rows))
  negative_target = max(0, int(args.calibration_target_negative_rows))
  if positive_target <= 0 and negative_target <= 0:
    return False
  if positive_target > 0 and class_counts.get("POSITIVE", 0) < positive_target:
    return False
  negative_count = int(class_counts.get("NEGATIVE", 0))
  if negative_target > 0 and negative_count < negative_target:
    return False
  return True


def _calibration_abort_satisfied(class_counts, processed_states, args):
  abort_after = max(0, int(getattr(args, "calibration_abort_after_states", 0)))
  min_positive = max(
      0, int(getattr(args, "calibration_abort_min_positive_rows", 0))
  )
  if abort_after <= 0 or min_positive <= 0:
    return False
  if int(processed_states) < abort_after:
    return False
  return int(class_counts.get("POSITIVE", 0)) < min_positive


def _policy_maps_for_budget(state, player, model, args, saved_args, device, samples, sims, idx):
  legal = list(state.legal_actions(player))
  maps = []
  pass_rows = []
  for pass_idx in range(max(1, int(args.repeat_label_passes))):
    seed = int(args.seed) + idx * 1000003 + pass_idx * 104729 + samples * 1009 + sims
    _set_seed(seed)
    targs = _teacher_args(args, saved_args, samples, sims)
    policy = teacher_belief_mcts_policy(state, player, model, targs, device)
    score_map = {int(action): float(policy[action]) for action in legal}
    ordered = sorted(score_map, key=lambda action: score_map[action], reverse=True)
    top = int(ordered[0])
    second = int(ordered[1]) if len(ordered) > 1 else top
    margin = float(score_map[top] - score_map[second])
    maps.append(score_map)
    pass_rows.append({
        "pass": int(pass_idx),
        "seed": int(seed),
        "best_action": int(top),
        "best_action_string": action_string(state, player, top),
        "top_prob": float(score_map[top]),
        "top_margin": margin,
        "entropy": _entropy([score_map[action] for action in legal]),
    })
  return maps, pass_rows


def _evaluate_state(idx, state, group, model, saved_args, args, device, budgets):
  player = int(state.current_player())
  phase = phase_name_for_state(state)
  legal = list(state.legal_actions(player))
  audit_ns = _audit_args(args, saved_args)
  policy, policy_best = choose_champion_action(
      model, audit_ns, device, state, player
  )
  del policy
  row = {
      "index": int(idx),
      "phase": phase,
      "player": int(player),
      "group": group or {},
      "match_round": int(getattr(state, "_match_round", -1)),
      "round_start_player": int(getattr(state, "_round_start_player", -1)),
      "trick_number": int(getattr(state, "_trick_number", -1)),
      "legal_count": int(len(legal)),
      "legal_actions": [int(action) for action in legal],
      "legal_action_strings": {
          str(action): action_string(state, player, action) for action in legal
      },
      "policy_best_action": int(policy_best),
      "policy_best_action_string": action_string(state, player, policy_best),
  }
  if len(legal) < 2:
    row["evaluable"] = False
    row["skip_reason"] = "too_few_legal_actions"
    return row

  budget_rows = []
  all_maps = []
  all_passes = []
  for samples, sims in budgets:
    maps, passes = _policy_maps_for_budget(
        state, player, model, args, saved_args, device, samples, sims, idx
    )
    all_maps.extend(maps)
    all_passes.extend([{**item, "samples": samples, "sims": sims} for item in passes])
    bests = [item["best_action"] for item in passes]
    best_counter = Counter(bests)
    mode_best, mode_count = best_counter.most_common(1)[0]
    pair_spearmans = []
    top2_jaccards = []
    for a_idx in range(len(maps)):
      for b_idx in range(a_idx + 1, len(maps)):
        a_values = [maps[a_idx][action] for action in legal]
        b_values = [maps[b_idx][action] for action in legal]
        pair_spearmans.append(_spearman(a_values, b_values))
        a_top2 = _top2_set(maps[a_idx])
        b_top2 = _top2_set(maps[b_idx])
        union = a_top2 | b_top2
        top2_jaccards.append(1.0 if not union else len(a_top2 & b_top2) / len(union))
    mean_scores = {
        int(action): float(np.mean([score_map[action] for score_map in maps]))
        for action in legal
    }
    ordered = sorted(mean_scores, key=lambda action: mean_scores[action], reverse=True)
    best = int(ordered[0])
    second = int(ordered[1]) if len(ordered) > 1 else best
    budget_rows.append({
        "samples": int(samples),
        "sims": int(sims),
        "top1_mode_action": int(mode_best),
        "top1_mode_action_string": action_string(state, player, mode_best),
        "top1_mode_rate": float(mode_count / max(1, len(bests))),
        "pairwise_spearman_mean": _pairwise_mean(pair_spearmans),
        "top2_jaccard_mean": _pairwise_mean(top2_jaccards),
        "consensus_best_action": int(best),
        "consensus_best_action_string": action_string(state, player, best),
        "consensus_top_margin": float(mean_scores[best] - mean_scores[second]),
        "mean_teacher_entropy": float(np.mean([item["entropy"] for item in passes])),
        "passes": passes,
        "mean_policy": {str(k): round(v, 6) for k, v in sorted(mean_scores.items())},
    })

  bests = [item["best_action"] for item in all_passes]
  best_counter = Counter(bests)
  mode_best, mode_count = best_counter.most_common(1)[0]
  pair_spearmans = []
  top2_jaccards = []
  for a_idx in range(len(all_maps)):
    for b_idx in range(a_idx + 1, len(all_maps)):
      a_values = [all_maps[a_idx][action] for action in legal]
      b_values = [all_maps[b_idx][action] for action in legal]
      pair_spearmans.append(_spearman(a_values, b_values))
      a_top2 = _top2_set(all_maps[a_idx])
      b_top2 = _top2_set(all_maps[b_idx])
      union = a_top2 | b_top2
      top2_jaccards.append(1.0 if not union else len(a_top2 & b_top2) / len(union))
  mean_scores = {
      int(action): float(np.mean([score_map[action] for score_map in all_maps]))
      for action in legal
  }
  ordered = sorted(mean_scores, key=lambda action: mean_scores[action], reverse=True)
  consensus_best = int(ordered[0])
  consensus_second = int(ordered[1]) if len(ordered) > 1 else consensus_best
  consensus_margin = float(mean_scores[consensus_best] - mean_scores[consensus_second])
  row.update({
      "evaluable": True,
      "budgets": budget_rows,
      "top1_mode_action": int(mode_best),
      "top1_mode_action_string": action_string(state, player, mode_best),
      "top1_mode_rate": float(mode_count / max(1, len(bests))),
      "pairwise_spearman_mean": _pairwise_mean(pair_spearmans),
      "top2_jaccard_mean": _pairwise_mean(top2_jaccards),
      "consensus_best_action": int(consensus_best),
      "consensus_best_action_string": action_string(state, player, consensus_best),
      "consensus_top_margin": consensus_margin,
      "high_margin": bool(consensus_margin >= float(args.min_policy_margin)),
      "consensus_matches_policy": bool(consensus_best == int(policy_best)),
      "mean_teacher_entropy": float(np.mean([item["entropy"] for item in all_passes])),
      "mean_policy": {str(k): round(v, 6) for k, v in sorted(mean_scores.items())},
  })

  if consensus_best == int(policy_best):
    row["forced_audit"] = {
        "audited": True,
        "audit_seed": None,
        "margin_vs_policy": 0.0,
        "paradox_delta_vs_policy": 0.0,
        "label_mean": None,
        "policy_mean": None,
        "label_paradox": None,
        "policy_paradox": None,
        "raw": {},
    }
    return row

  audit_seed = int(args.seed) + 2000003 + idx * 104729
  _set_seed(audit_seed)
  round_game = make_game(audit_ns, int(getattr(state, "_round_start_player", 0)))
  audit = evaluate_actions_independently(
      state,
      player,
      [int(consensus_best), int(policy_best)],
      round_game,
      model,
      audit_ns,
      device,
      args.audit_rollouts,
  )
  label_stats = audit.get(int(consensus_best), {})
  policy_stats = audit.get(int(policy_best), {})
  label_mean = label_stats.get("mean")
  policy_mean = policy_stats.get("mean")
  label_paradox = label_stats.get("mean_paradox")
  policy_paradox = policy_stats.get("mean_paradox")
  row["forced_audit"] = {
      "audited": label_mean is not None and policy_mean is not None,
      "audit_seed": int(audit_seed),
      "margin_vs_policy": (
          None if label_mean is None or policy_mean is None
          else float(label_mean - policy_mean)
      ),
      "paradox_delta_vs_policy": (
          None if label_paradox is None or policy_paradox is None
          else float(label_paradox - policy_paradox)
      ),
      "label_mean": None if label_mean is None else float(label_mean),
      "policy_mean": None if policy_mean is None else float(policy_mean),
      "label_paradox": None if label_paradox is None else float(label_paradox),
      "policy_paradox": None if policy_paradox is None else float(policy_paradox),
      "raw": {str(k): v for k, v in sorted(audit.items())},
  }
  return row


def main():
  args = parse_args()
  started = time.perf_counter()
  _set_seed(args.seed)
  device = _device(args.device)
  game = make_game(SimpleNamespace(players=args.players, match_context=True), 0)
  model, _, saved_args = load_model_payload(
      args.checkpoint, game, model_args_for(args.players), device
  )
  args.value_scale = float(saved_args.get("value_scale", 20.0))
  # The shared hard-state collector was built for Q-label audits and expects
  # these names while collecting policy-driven states.
  args.belief_samples = max(1, int(args.audit_belief_samples))
  args.belief_candidates = max(
      int(args.belief_samples), int(args.audit_belief_candidates)
  )
  args.belief_policy_temperature = 1.0
  args.belief_uniform_mix = 0.0
  args.belief_ref_policy_mix = "model:1.0"
  args.belief_ref_policy_mix_by_phase = ""
  args.rollouts_per_action = 1
  args.label_max_actions = 2
  args.label_top_policy = 2
  args.label_include_bots = ""
  args.label_feature_candidates = False
  args.label_paradox_penalty = 0.0
  model.eval()
  wanted = {part.strip() for part in args.phases.split(",") if part.strip()}
  states, groups, bucket_reports = _collect_mixed_states(
      model, saved_args, args, device, wanted
  )
  collection_summary = _collection_state_summary(states, groups)
  budgets = _parse_budgets(args.budgets)
  if args.collect_only:
    rows = [
        {
            "index": int(idx),
            "phase": phase_name_for_state(state),
            "player": int(state.current_player()),
            "legal_count": int(len(state.legal_actions(state.current_player()))),
            "match_round": int(getattr(state, "_match_round", -1)),
            "round_start_player": int(getattr(state, "_round_start_player", -1)),
            "trick_number": int(getattr(state, "_trick_number", -1)),
            "group": groups[idx] if idx < len(groups) else {},
        }
        for idx, state in enumerate(states)
    ]
    result = {
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "collect_only": True,
        "states_requested": int(args.states),
        "states_collected": int(len(states)),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "phases": sorted(wanted),
        "state_mix": args.state_mix,
        "state_buckets": bucket_reports,
        "collection_state_summary": collection_summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return

  rows = []
  calibration_class_counts = Counter()
  calibration_target_class_counts = Counter()
  calibration_target_skipped = Counter()
  early_stop_reason = ""
  for idx, state in enumerate(states):
    row = _evaluate_state(
        idx,
        state,
        groups[idx] if idx < len(groups) else {},
        model,
        saved_args,
        args,
        device,
        budgets,
    )
    rows.append(row)
    if args.calibration_replay_out:
      _, row_class, stable_reason, stability_demoted = (
          _calibration_class_details_for_row(row, args)
      )
      if row_class:
        calibration_class_counts[str(row_class)] += 1
        if (
            bool(args.calibration_target_require_stable)
            and stable_reason is not None
        ):
          calibration_target_skipped[str(stable_reason)] += 1
        else:
          calibration_target_class_counts[str(row_class)] += 1
    if args.progress_interval > 0 and (
        (idx + 1) % args.progress_interval == 0 or (idx + 1) == len(states)
    ):
      print(
          json.dumps({
              "processed_states": int(idx + 1),
              "states_collected": int(len(states)),
              "evaluable": int(sum(1 for item in rows if item.get("evaluable"))),
              "high_margin": int(sum(1 for item in rows if item.get("high_margin"))),
              "calibration_class_counts": {
                  key: int(value)
                  for key, value in sorted(calibration_class_counts.items())
              },
              "calibration_target_class_counts": {
                  key: int(value)
                  for key, value in sorted(calibration_target_class_counts.items())
              },
              "calibration_target_skipped": {
                  key: int(value)
                  for key, value in sorted(calibration_target_skipped.items())
              },
              "elapsed_sec": round(time.perf_counter() - started, 3),
          }),
          flush=True,
      )
    if args.calibration_replay_out and _calibration_targets_satisfied(
        calibration_target_class_counts, args
    ):
      early_stop_reason = "calibration_targets_satisfied"
      print(
          json.dumps({
              "processed_states": int(idx + 1),
              "states_collected": int(len(states)),
              "early_stop_reason": early_stop_reason,
              "calibration_class_counts": {
                  key: int(value)
                  for key, value in sorted(calibration_class_counts.items())
              },
              "calibration_target_class_counts": {
                  key: int(value)
                  for key, value in sorted(calibration_target_class_counts.items())
              },
              "calibration_target_skipped": {
                  key: int(value)
                  for key, value in sorted(calibration_target_skipped.items())
              },
              "elapsed_sec": round(time.perf_counter() - started, 3),
          }),
          flush=True,
      )
      break
    if args.calibration_replay_out and _calibration_abort_satisfied(
        calibration_target_class_counts, idx + 1, args
    ):
      early_stop_reason = "calibration_abort_low_positive_yield"
      print(
          json.dumps({
              "processed_states": int(idx + 1),
              "states_collected": int(len(states)),
              "early_stop_reason": early_stop_reason,
              "calibration_class_counts": {
                  key: int(value)
                  for key, value in sorted(calibration_class_counts.items())
              },
              "calibration_target_class_counts": {
                  key: int(value)
                  for key, value in sorted(calibration_target_class_counts.items())
              },
              "calibration_target_skipped": {
                  key: int(value)
                  for key, value in sorted(calibration_target_skipped.items())
              },
              "elapsed_sec": round(time.perf_counter() - started, 3),
          }),
          flush=True,
      )
      break

  stability = _stability_summary(rows)
  high_stability = stability["high_margin_top1_mode_rate"]
  high_spearman = stability["high_margin_pairwise_spearman"]
  high_top2 = stability["high_margin_top2_jaccard"]
  forced = stability["high_margin_forced_disagreement_audit"]
  teacher_stability_pass = bool(
      stability["high_margin_rows"] >= int(args.pass_min_high_margin_rows)
      and high_stability is not None
      and high_spearman is not None
      and high_top2 is not None
      and forced["rows"] > 0
      and high_stability["mean"] >= float(args.pass_min_high_margin_top1_stability)
      and high_top2["mean"] >= float(args.pass_min_top2_jaccard)
      and high_spearman["mean"] >= float(args.pass_min_spearman)
      and forced["score_rate"] is not None
      and forced["score_rate"] >= float(args.pass_min_forced_score_rate)
      and forced["mean_margin"] is not None
      and forced["mean_margin"] > float(args.pass_min_forced_avg_diff)
      and forced["mean_paradox_delta"] is not None
      and forced["mean_paradox_delta"] <= float(args.pass_max_forced_paradox_delta)
  )
  calibration_replay = []
  calibration_metadata = []
  calibration_skipped = {}
  if args.calibration_replay_out:
    audit_ns = _audit_args(args, saved_args)
    calibration_replay, calibration_metadata, calibration_skipped = (
        _build_calibration_replay(states, rows, audit_ns, args)
    )
    calibration_path = Path(args.calibration_replay_out)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    save_replay(calibration_replay, calibration_path)
    metadata_path = calibration_path.with_suffix(
        calibration_path.suffix + ".metadata.json"
    )
    metadata_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "calibration_replay_out": str(calibration_path),
        "positive_margin": float(args.calibration_positive_margin),
        "max_positive_paradox_delta": float(
            args.calibration_max_positive_paradox_delta
        ),
        "include_policy_matches": bool(args.calibration_include_policy_matches),
        "target_positive_rows": int(args.calibration_target_positive_rows),
        "target_negative_rows": int(args.calibration_target_negative_rows),
        "abort_after_states": int(args.calibration_abort_after_states),
        "abort_min_positive_rows": int(
            args.calibration_abort_min_positive_rows
        ),
        "demote_unstable_positives": bool(
            args.calibration_demote_unstable_positives
        ),
        "early_stop_reason": early_stop_reason,
        "summary": calibration_summary(calibration_metadata),
        "accepted_replay_summary": accepted_replay_summary(calibration_replay),
        "skipped": calibration_skipped,
        "rows": calibration_metadata,
    }, indent=2) + "\n")
  result = {
      "checkpoint": args.checkpoint,
      "players": int(args.players),
      "seed": int(args.seed),
      "states_requested": int(args.states),
      "states_collected": int(len(states)),
      "states_evaluated": int(len(rows)),
      "early_stop_reason": early_stop_reason,
      "elapsed_sec": round(time.perf_counter() - started, 3),
      "phases": sorted(wanted),
      "state_mix": args.state_mix,
      "state_buckets": bucket_reports,
      "state_min_legal_count": int(args.state_min_legal_count),
      "state_max_legal_count": int(args.state_max_legal_count),
      "state_min_trick_number": int(args.state_min_trick_number),
      "state_max_trick_number": int(args.state_max_trick_number),
      "state_min_match_round": int(args.state_min_match_round),
      "state_max_match_round": int(args.state_max_match_round),
      "collection_state_summary": collection_summary,
      "budgets": [
          {"teacher_belief_samples": int(samples), "teacher_sims": int(sims)}
          for samples, sims in budgets
      ],
      "repeat_label_passes": int(args.repeat_label_passes),
      "min_policy_margin": float(args.min_policy_margin),
      "audit_rollouts": int(args.audit_rollouts),
      "audit_belief_samples": int(args.audit_belief_samples),
      "full_match_rollout": bool(args.full_match_rollout),
      "rollout_max_plies": int(args.rollout_max_plies),
      "teacher_belief_source": str(args.teacher_belief_source),
      "pass_min_high_margin_rows": int(args.pass_min_high_margin_rows),
      "pass_min_high_margin_top1_stability": (
          float(args.pass_min_high_margin_top1_stability)
      ),
      "pass_min_top2_jaccard": float(args.pass_min_top2_jaccard),
      "pass_min_spearman": float(args.pass_min_spearman),
      "pass_min_forced_score_rate": float(args.pass_min_forced_score_rate),
      "pass_min_forced_avg_diff": float(args.pass_min_forced_avg_diff),
      "pass_max_forced_paradox_delta": float(args.pass_max_forced_paradox_delta),
      "teacher_stability_pass": teacher_stability_pass,
      "calibration_replay_out": args.calibration_replay_out,
      "calibration_replay_rows": int(len(calibration_replay)),
      "calibration_positive_margin": float(args.calibration_positive_margin),
      "calibration_max_positive_paradox_delta": (
          float(args.calibration_max_positive_paradox_delta)
      ),
      "calibration_include_policy_matches": bool(
          args.calibration_include_policy_matches
      ),
      "calibration_target_positive_rows": int(
          args.calibration_target_positive_rows
      ),
      "calibration_target_negative_rows": int(
          args.calibration_target_negative_rows
      ),
      "calibration_abort_after_states": int(args.calibration_abort_after_states),
      "calibration_abort_min_positive_rows": int(
          args.calibration_abort_min_positive_rows
      ),
      "calibration_target_require_stable": bool(
          args.calibration_target_require_stable
      ),
      "calibration_demote_unstable_positives": bool(
          args.calibration_demote_unstable_positives
      ),
      "calibration_stable_filters": {
          "min_top1_mode_rate": float(args.calibration_stable_min_top1_mode_rate),
          "min_pairwise_spearman": float(
              args.calibration_stable_min_pairwise_spearman
          ),
          "min_top2_jaccard": float(args.calibration_stable_min_top2_jaccard),
          "min_consensus_top_margin": float(
              args.calibration_stable_min_consensus_top_margin
          ),
          "require_high_margin": bool(args.calibration_stable_require_high_margin),
      },
      "calibration_target_class_counts": {
          key: int(value)
          for key, value in sorted(calibration_target_class_counts.items())
      },
      "calibration_target_skipped": {
          key: int(value)
          for key, value in sorted(calibration_target_skipped.items())
      },
      "calibration_summary": calibration_summary(calibration_metadata),
      "calibration_replay_summary": accepted_replay_summary(calibration_replay),
      "calibration_skipped": calibration_skipped,
      "summary": stability,
      "by_phase": _bucket_summary(rows, "phase"),
      "by_state_mix_bucket": _bucket_summary(rows, "state_mix_bucket"),
      "rows": rows,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
