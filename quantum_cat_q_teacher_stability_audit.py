#!/usr/bin/env python3
"""Audit repeated counterfactual teacher labels before policy distillation.

This is a go/no-go diagnostic for sparse Q / rollout teachers. It keeps the
candidate action set fixed for each state, evaluates it several times with fresh
belief samples and rollout seeds, then asks two questions:

1. Is the teacher's ranking stable enough to learn?
2. Does the consensus teacher-best action beat Run119's policy action in an
   independent forced-action audit?

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
    TACTICAL_POLICY_BUCKET_NAMES,
    load_model_payload,
    make_game,
    phase_name_for_state,
    sampled_counterfactual_legal_actions,
    save_replay,
)
from quantum_cat_label_oracle_audit import (
    accepted_replay_example,
    accepted_replay_summary,
    action_string,
    audit_namespace,
    choose_champion_action,
    collect_states,
    evaluate_actions_independently,
    model_args_for,
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--states", type=int, default=96)
  parser.add_argument("--max-matches", type=int, default=300)
  parser.add_argument("--max-states-per-match", type=int, default=2)
  parser.add_argument(
      "--state-mix",
      default="",
      help=(
          "Comma-separated bucket=count spec. Buckets: first/random_play, "
          "high_entropy, hard, loss_gap, near_paradox, plus tactical action "
          f"buckets {','.join(TACTICAL_POLICY_BUCKET_NAMES)}. When empty, "
          "--state-selection and --states are used."
      ),
  )
  parser.add_argument(
      "--interleave-state-buckets",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "When collected states have group.state_mix_bucket labels, evaluate "
          "them round-robin across buckets. This prevents tactical generation "
          "from spending all early label budget on the first requested bucket."
      ),
  )
  parser.add_argument(
      "--state-selection",
      choices=("first", "high_entropy", "hard"),
      default="high_entropy",
  )
  parser.add_argument("--state-pool-per-match", type=int, default=0)
  parser.add_argument(
      "--state-min-legal-count",
      type=int,
      default=0,
      help="When >0, only collect candidate states with at least this many legal actions.",
  )
  parser.add_argument(
      "--state-max-legal-count",
      type=int,
      default=0,
      help="When >0, only collect candidate states with at most this many legal actions.",
  )
  parser.add_argument(
      "--state-min-trick-number",
      type=int,
      default=-1,
      help="When >=0, only collect candidate states at or after this trick number.",
  )
  parser.add_argument(
      "--state-max-trick-number",
      type=int,
      default=-1,
      help="When >=0, only collect candidate states at or before this trick number.",
  )
  parser.add_argument(
      "--state-min-match-round",
      type=int,
      default=-1,
      help="When >=0, only collect candidate states at or after this match round.",
  )
  parser.add_argument(
      "--state-max-match-round",
      type=int,
      default=-1,
      help="When >=0, only collect candidate states at or before this match round.",
  )
  parser.add_argument("--hard-entropy-weight", type=float, default=1.0)
  parser.add_argument("--hard-loss-weight", type=float, default=0.25)
  parser.add_argument("--hard-paradox-weight", type=float, default=0.75)
  parser.add_argument("--phases", default="play")
  parser.add_argument("--repeat-label-passes", type=int, default=3)
  parser.add_argument("--rollouts-per-action", type=int, default=2)
  parser.add_argument("--audit-rollouts", type=int, default=2)
  parser.add_argument("--belief-samples", type=int, default=2)
  parser.add_argument("--belief-candidates", type=int, default=8)
  parser.add_argument("--belief-policy-temperature", type=float, default=1.0)
  parser.add_argument("--belief-uniform-mix", type=float, default=0.15)
  parser.add_argument(
      "--belief-ref-policy-mix",
      default="model:0.65,heuristic:0.25,uniform:0.10",
  )
  parser.add_argument("--belief-ref-policy-mix-by-phase", default="")
  parser.add_argument("--label-max-actions", type=int, default=12)
  parser.add_argument("--label-top-policy", type=int, default=4)
  parser.add_argument(
      "--label-target-bucket-actions",
      type=int,
      default=0,
      help=(
          "When >0 and the collected row has a tactical state_mix_bucket, add "
          "up to this many policy-ranked legal actions from that bucket to the "
          "teacher candidate set. This lets target-bucket-gated replay "
          "generation actually score the action type it is trying to export."
      ),
  )
  parser.add_argument(
      "--label-include-bots",
      default="heuristic,heuristic_target2,heuristic_adj2",
  )
  parser.add_argument(
      "--label-feature-candidates",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--label-paradox-penalty",
      type=float,
      default=0.0,
      help="Rank candidates by score - penalty * paradox_count.",
  )
  parser.add_argument("--full-match-rollout", action="store_true")
  parser.add_argument("--rollout-max-plies", type=int, default=0)
  parser.add_argument("--match-context", action="store_true", default=True)
  parser.add_argument("--min-candidate-actions", type=int, default=2)
  parser.add_argument("--min-high-margin", type=float, default=0.5)
  parser.add_argument(
      "--pass-min-high-margin-rows",
      type=int,
      default=20,
      help="Minimum high-margin forced-audited rows required before pass=true.",
  )
  parser.add_argument("--pass-min-high-margin-top1-stability", type=float, default=0.60)
  parser.add_argument("--pass-min-spearman", type=float, default=0.35)
  parser.add_argument("--pass-min-forced-score-rate", type=float, default=0.55)
  parser.add_argument("--pass-min-forced-avg-diff", type=float, default=0.0)
  parser.add_argument("--pass-max-forced-paradox-delta", type=float, default=0.0)
  parser.add_argument("--stable-replay-out", default="")
  parser.add_argument("--stable-replay-metadata-out", default="")
  parser.add_argument(
      "--stable-require-high-margin",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--stable-min-top1-mode-rate", type=float, default=0.75)
  parser.add_argument("--stable-min-spearman", type=float, default=0.35)
  parser.add_argument("--stable-min-forced-margin", type=float, default=0.5)
  parser.add_argument("--stable-max-forced-paradox-delta", type=float, default=0.0)
  parser.add_argument("--stable-min-legal-count", type=int, default=0)
  parser.add_argument("--stable-min-q-spread", type=float, default=0.0)
  parser.add_argument(
      "--stable-require-recall-consensus",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "When enabled, require candidate-recall audit to agree that the "
          "consensus action is all-legal best before exporting a stable row."
      ),
  )
  parser.add_argument(
      "--stable-use-recall-labels",
      action="store_true",
      help=(
          "Export stable replay labels from all-legal candidate-recall scores "
          "instead of the capped candidate consensus. Use this for low-branching "
          "states where recall audits score every legal action."
      ),
  )
  parser.add_argument(
      "--stable-require-target-bucket-match",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "When a row came from a tactical state_mix bucket, require the "
          "consensus target action to be one of that bucket's recorded legal "
          "actions before exporting it. This prevents balanced state "
          "collection from producing imbalanced target-action labels."
      ),
  )
  parser.add_argument(
      "--stable-target-rows",
      type=int,
      default=0,
      help=(
          "When >0 and --stable-replay-out is set, stop evaluating the "
          "pre-collected state pool once this many stable-export rows have "
          "been found."
      ),
  )
  parser.add_argument(
      "--stable-target-rows-per-state-bucket",
      type=int,
      default=0,
      help=(
          "When >0 and --stable-replay-out is set, stop only after every "
          "collected state_mix bucket has at least this many stable-export "
          "rows. This is useful with tactical --state-mix collection."
      ),
  )
  parser.add_argument(
      "--stable-include-policy-matches",
      action="store_true",
      help="Also export stable rows whose consensus action matches Run119.",
  )
  parser.add_argument(
      "--candidate-recall-audit-every",
      type=int,
      default=0,
      help=(
          "When >0, every Nth evaluated state is rescored over all legal "
          "actions to estimate whether the capped candidate set missed the "
          "teacher-best action. This is expensive and diagnostic-only."
      ),
  )
  parser.add_argument(
      "--candidate-recall-max-legal",
      type=int,
      default=0,
      help=(
          "Skip candidate-recall audits when legal_count exceeds this cap. "
          "0 means no cap."
      ),
  )
  parser.add_argument(
      "--candidate-recall-rollouts",
      type=int,
      default=0,
      help="Rollouts per action for recall audits. 0 reuses --rollouts-per-action.",
  )
  parser.add_argument("--candidate-recall-seed-offset", type=int, default=3000003)
  parser.add_argument("--seed", type=int, default=20262152)
  parser.add_argument("--label-seed-offset", type=int, default=1000003)
  parser.add_argument("--audit-seed-offset", type=int, default=2000003)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument("--progress-interval", type=int, default=0)
  parser.add_argument(
      "--collect-only",
      action="store_true",
      help="Only collect and summarize states; skip teacher evaluation.",
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


def _oracle_cli_namespace(args, states, state_selection, weights, seed):
  entropy_weight, loss_weight, paradox_weight = weights
  return SimpleNamespace(
      players=args.players,
      states=max(1, int(states)),
      max_matches=max(1, int(args.max_matches)),
      max_states_per_match=max(0, int(args.max_states_per_match)),
      state_selection=state_selection,
      state_pool_per_match=max(0, int(args.state_pool_per_match)),
      state_min_legal_count=max(0, int(args.state_min_legal_count)),
      state_max_legal_count=max(0, int(args.state_max_legal_count)),
      state_min_trick_number=int(args.state_min_trick_number),
      state_max_trick_number=int(args.state_max_trick_number),
      state_min_match_round=int(args.state_min_match_round),
      state_max_match_round=int(args.state_max_match_round),
      hard_entropy_weight=float(entropy_weight),
      hard_loss_weight=float(loss_weight),
      hard_paradox_weight=float(paradox_weight),
      seed=int(seed),
      match_context=bool(args.match_context),
      belief_samples=args.belief_samples,
      belief_candidates=args.belief_candidates,
      belief_policy_temperature=args.belief_policy_temperature,
      belief_uniform_mix=args.belief_uniform_mix,
      belief_ref_policy_mix=args.belief_ref_policy_mix,
      belief_ref_policy_mix_by_phase=args.belief_ref_policy_mix_by_phase,
      label_rollouts=args.rollouts_per_action,
      audit_rollouts=args.audit_rollouts,
      phases=args.phases,
      label_max_actions=args.label_max_actions,
      label_top_policy=args.label_top_policy,
      label_include_bots=args.label_include_bots,
      label_feature_candidates=bool(args.label_feature_candidates),
      label_paradox_penalty=args.label_paradox_penalty,
      label_max_paradox_delta=None,
      min_label_margin=0.0,
      label_paradox_penalty_unused=args.label_paradox_penalty,
      min_audit_margin=0.0,
      pass_min_score_rate=args.pass_min_forced_score_rate,
      pass_min_avg_diff=args.pass_min_forced_avg_diff,
      pass_max_paradox_delta=args.pass_max_forced_paradox_delta,
      full_match_rollout=bool(args.full_match_rollout),
      rollout_max_plies=args.rollout_max_plies,
  )


def _parse_state_mix(spec, default_count, default_selection, default_weights):
  if not str(spec or "").strip():
    return [(default_selection, int(default_count), default_weights, default_selection)]
  buckets = []
  for raw_part in str(spec).split(","):
    part = raw_part.strip()
    if not part:
      continue
    if "=" in part:
      name, count = part.split("=", 1)
    elif ":" in part:
      name, count = part.split(":", 1)
    else:
      raise ValueError(f"Bad state-mix item: {part!r}")
    name = name.strip()
    count = int(count.strip())
    if count <= 0:
      continue
    if name in ("first", "random", "random_play"):
      buckets.append(("first", count, default_weights, name))
    elif name == "high_entropy":
      buckets.append(("high_entropy", count, (1.0, 0.0, 0.0), name))
    elif name == "loss_gap":
      buckets.append(("hard", count, (0.2, 1.0, 0.0), name))
    elif name == "near_paradox":
      buckets.append(("hard", count, (0.2, 0.0, 1.0), name))
    elif name == "hard":
      buckets.append(("hard", count, default_weights, name))
    elif name in TACTICAL_POLICY_BUCKET_NAMES:
      buckets.append((f"tactical:{name}", count, default_weights, name))
    elif name.startswith("tactical:"):
      bucket_name = name.split(":", 1)[1]
      if bucket_name not in TACTICAL_POLICY_BUCKET_NAMES:
        raise ValueError(
            f"Unknown tactical state-mix bucket: {bucket_name!r}"
        )
      buckets.append((name, count, default_weights, name))
    elif name.startswith("tactical_"):
      bucket_name = name.split("_", 1)[1]
      if bucket_name not in TACTICAL_POLICY_BUCKET_NAMES:
        raise ValueError(
            f"Unknown tactical state-mix bucket: {bucket_name!r}"
        )
      buckets.append((f"tactical:{bucket_name}", count, default_weights, name))
    else:
      raise ValueError(f"Unknown state-mix bucket: {name!r}")
  return buckets


def _collect_mixed_states(model, saved_args, args, device, wanted):
  default_weights = (
      float(args.hard_entropy_weight),
      float(args.hard_loss_weight),
      float(args.hard_paradox_weight),
  )
  buckets = _parse_state_mix(
      args.state_mix, args.states, args.state_selection, default_weights
  )
  all_states = []
  all_groups = []
  bucket_reports = []
  for bucket_idx, (selection, count, weights, source_name) in enumerate(buckets):
    seed = int(args.seed) + bucket_idx * 104729
    _set_seed(seed)
    if int(getattr(args, "progress_interval", 0)) > 0:
      print(
          json.dumps({
              "collection_bucket_start": source_name,
              "bucket_index": int(bucket_idx),
              "requested": int(count),
              "state_selection": selection,
              "seed": int(seed),
          }),
          flush=True,
      )
    cli_ns = _oracle_cli_namespace(args, count, selection, weights, seed)
    oracle_args = audit_namespace(cli_ns, saved_args)
    states, groups, matches_used = collect_states(
        model, oracle_args, device, wanted
    )
    collection_report = getattr(oracle_args, "_state_collection_report", {})
    for group in groups:
      group["state_mix_bucket"] = source_name
    all_states.extend(states)
    all_groups.extend(groups)
    bucket_reports.append({
        "bucket": source_name,
        "state_selection": selection,
        "requested": int(count),
        "collected": int(len(states)),
        "matches_used": int(matches_used),
        "hard_entropy_weight": float(weights[0]),
        "hard_loss_weight": float(weights[1]),
        "hard_paradox_weight": float(weights[2]),
        "state_min_legal_count": int(args.state_min_legal_count),
        "state_max_legal_count": int(args.state_max_legal_count),
        "state_min_trick_number": int(args.state_min_trick_number),
        "state_max_trick_number": int(args.state_max_trick_number),
        "state_min_match_round": int(args.state_min_match_round),
        "state_max_match_round": int(args.state_max_match_round),
        "collection": collection_report,
        "seed": int(seed),
    })
    if int(getattr(args, "progress_interval", 0)) > 0:
      print(
          json.dumps({
              "collection_bucket_done": source_name,
              "bucket_index": int(bucket_idx),
              "requested": int(count),
              "collected": int(len(states)),
              "matches_used": int(matches_used),
              "total_collected": int(len(all_states)),
          }),
          flush=True,
      )
  return all_states[: int(args.states)], all_groups[: int(args.states)], bucket_reports


def _rank_values(values):
  order = np.argsort(values)
  ranks = np.empty(len(values), dtype=np.float64)
  i = 0
  while i < len(values):
    j = i + 1
    while j < len(values) and values[order[j]] == values[order[i]]:
      j += 1
    avg_rank = 0.5 * (i + j - 1)
    ranks[order[i:j]] = avg_rank
    i = j
  return ranks


def _spearman(a, b):
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  if len(a) < 2:
    return None
  ra = _rank_values(a)
  rb = _rank_values(b)
  if float(np.std(ra)) <= 1e-12 or float(np.std(rb)) <= 1e-12:
    return None
  return float(np.corrcoef(ra, rb)[0, 1])


def _pairwise_mean(values):
  values = [v for v in values if v is not None and math.isfinite(float(v))]
  return None if not values else float(np.mean(values))


def _top2_set(score_map):
  ordered = sorted(score_map, key=lambda action: score_map[action], reverse=True)
  return set(int(action) for action in ordered[: min(2, len(ordered))])


def _summarize_numbers(values):
  values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
  if not values:
    return None
  arr = np.array(values, dtype=np.float64)
  return {
      "min": round(float(np.min(arr)), 6),
      "mean": round(float(np.mean(arr)), 6),
      "max": round(float(np.max(arr)), 6),
  }


def _collection_state_summary(states, groups):
  legal_counts = []
  trick_numbers = []
  match_rounds = []
  entropies = []
  top_probs = []
  loss_gaps = []
  paradoxes = []
  buckets = Counter()
  tactical_targets = Counter()
  tactical_available = Counter()
  for idx, state in enumerate(states):
    group = groups[idx] if idx < len(groups) else {}
    player = int(state.current_player()) if not state.is_terminal() else -1
    legal_counts.append(len(state.legal_actions(player)) if player >= 0 else 0)
    trick_numbers.append(getattr(state, "_trick_number", -1))
    match_rounds.append(getattr(state, "_match_round", -1))
    if group:
      buckets[str(group.get("state_mix_bucket", "unknown"))] += 1
      entropies.append(group.get("policy_entropy_norm"))
      top_probs.append(group.get("policy_top_prob"))
      loss_gaps.append(group.get("acting_player_loss_gap"))
      paradoxes.append(group.get("acting_player_match_paradoxes"))
      for name in group.get("tactical_target_buckets") or []:
        tactical_targets[str(name)] += 1
      for name in group.get("tactical_available_buckets") or []:
        tactical_available[str(name)] += 1
  return {
      "rows": int(len(states)),
      "legal_count": _summarize_numbers(legal_counts),
      "trick_number": _summarize_numbers(trick_numbers),
      "match_round": _summarize_numbers(match_rounds),
      "policy_entropy_norm": _summarize_numbers(entropies),
      "policy_top_prob": _summarize_numbers(top_probs),
      "acting_player_loss_gap": _summarize_numbers(loss_gaps),
      "acting_player_match_paradoxes": _summarize_numbers(paradoxes),
      "buckets": dict(sorted(buckets.items())),
      "tactical_target_buckets": dict(sorted(tactical_targets.items())),
      "tactical_available_buckets": dict(sorted(tactical_available.items())),
  }


def _interleave_states_by_bucket(states, groups):
  if not states or not groups:
    return states, groups, {}
  buckets = {}
  bucket_order = []
  for idx, state in enumerate(states):
    group = groups[idx] if idx < len(groups) else {}
    bucket = str(group.get("state_mix_bucket", "unknown"))
    if bucket not in buckets:
      buckets[bucket] = []
      bucket_order.append(bucket)
    buckets[bucket].append((state, group))
  if len(bucket_order) <= 1:
    return states, groups, {
        "enabled": False,
        "reason": "single_bucket",
        "bucket_counts": {name: len(items) for name, items in buckets.items()},
    }
  interleaved = []
  max_len = max(len(items) for items in buckets.values())
  for offset in range(max_len):
    for bucket in bucket_order:
      items = buckets[bucket]
      if offset < len(items):
        interleaved.append(items[offset])
  new_states = [state for state, _ in interleaved]
  new_groups = [group for _, group in interleaved]
  return new_states, new_groups, {
      "enabled": True,
      "bucket_order": bucket_order,
      "bucket_counts": {name: len(items) for name, items in buckets.items()},
  }


def _adjusted_score(stats, penalty):
  mean = stats.get("mean")
  if mean is None:
    return None
  paradox = stats.get("mean_paradox")
  paradox = 0.0 if paradox is None else float(paradox)
  return float(mean) - float(penalty) * paradox


def _target_bucket_candidate_actions(group, legal, policy, cap):
  cap = int(cap)
  if cap <= 0:
    return []
  group = group or {}
  bucket = group.get("state_mix_bucket")
  if bucket not in TACTICAL_POLICY_BUCKET_NAMES:
    return []
  legal_set = {int(action) for action in legal}
  matches = [
      int(action)
      for action in ((group.get("tactical_bucket_matches") or {}).get(bucket, []))
      if int(action) in legal_set
  ]
  if not matches:
    return []
  policy_arr = np.asarray(policy, dtype=np.float32) if policy is not None else None

  def score(action):
    if policy_arr is None or action < 0 or action >= len(policy_arr):
      return 0.0
    return float(policy_arr[action])

  return sorted(set(matches), key=lambda action: (-score(action), action))[:cap]


def _evaluate_state(idx, state, group, model, oracle_args, args, device):
  player = int(state.current_player())
  phase = phase_name_for_state(state)
  round_game = make_game(oracle_args, int(getattr(state, "_round_start_player", 0)))
  policy, policy_best = choose_champion_action(
      model, oracle_args, device, state, player
  )
  legal = list(state.legal_actions(player))
  candidates = sampled_counterfactual_legal_actions(
      state, player, legal, oracle_args, policy=policy
  )
  if int(policy_best) not in candidates:
    candidates.append(int(policy_best))
  target_bucket_candidates = _target_bucket_candidate_actions(
      group,
      legal,
      policy,
      int(getattr(args, "label_target_bucket_actions", 0)),
  )
  candidates.extend(target_bucket_candidates)
  candidates = sorted(set(int(action) for action in candidates if action in legal))
  row = {
      "index": int(idx),
      "phase": phase,
      "player": int(player),
      "group": group or {},
      "match_round": int(getattr(state, "_match_round", -1)),
      "round_start_player": int(getattr(state, "_round_start_player", -1)),
      "trick_number": int(getattr(state, "_trick_number", -1)),
      "legal_count": int(len(legal)),
      "candidate_count": int(len(candidates)),
      "candidate_actions": [int(action) for action in candidates],
      "target_bucket_candidate_actions": [
          int(action) for action in target_bucket_candidates
      ],
      "target_bucket_candidate_count": int(len(target_bucket_candidates)),
      "candidate_action_strings": {
          str(action): action_string(state, player, action) for action in candidates
      },
      "policy_best_action": int(policy_best),
      "policy_best_action_string": action_string(state, player, policy_best),
  }
  if len(candidates) < int(args.min_candidate_actions):
    row["evaluable"] = False
    row["skip_reason"] = "too_few_candidates"
    return row

  pass_results = []
  pass_score_maps = []
  pass_raw_score_maps = []
  pass_paradox_maps = []
  for pass_idx in range(max(1, int(args.repeat_label_passes))):
    label_seed = (
        int(args.seed)
        + int(args.label_seed_offset)
        + idx * 1000003
        + pass_idx * 104729
    )
    _set_seed(label_seed)
    stats = evaluate_actions_independently(
        state,
        player,
        candidates,
        round_game,
        model,
        oracle_args,
        device,
        args.rollouts_per_action,
    )
    adjusted = {}
    raw_scores = {}
    paradox_scores = {}
    for action in candidates:
      action_stats = stats.get(int(action), {})
      score = _adjusted_score(action_stats, args.label_paradox_penalty)
      if score is None:
        continue
      adjusted[int(action)] = float(score)
      raw_scores[int(action)] = float(action_stats.get("mean"))
      if action_stats.get("mean_paradox") is not None:
        paradox_scores[int(action)] = float(action_stats.get("mean_paradox"))
    if adjusted:
      best = int(max(adjusted, key=lambda action: adjusted[action]))
      ordered = sorted(adjusted, key=lambda action: adjusted[action], reverse=True)
      second = ordered[1] if len(ordered) > 1 else ordered[0]
      margin = float(adjusted[best] - adjusted[second])
    else:
      best = None
      margin = None
    pass_results.append({
        "pass": int(pass_idx),
        "seed": int(label_seed),
        "scored_actions": int(len(adjusted)),
        "best_action": best,
        "best_action_string": (
            None if best is None else action_string(state, player, best)
        ),
        "top_margin": margin,
    })
    pass_score_maps.append(adjusted)
    pass_raw_score_maps.append(raw_scores)
    pass_paradox_maps.append(paradox_scores)

  common_actions = set(candidates)
  for score_map in pass_score_maps:
    common_actions &= set(score_map)
  common_actions = sorted(int(action) for action in common_actions)
  if len(common_actions) < int(args.min_candidate_actions):
    row.update({
        "evaluable": False,
        "skip_reason": "too_few_common_scored_actions",
        "passes": pass_results,
        "common_scored_actions": int(len(common_actions)),
      })
    return row

  bests = [item["best_action"] for item in pass_results if item["best_action"] is not None]
  best_counter = Counter(bests)
  mode_best, mode_count = best_counter.most_common(1)[0]
  pair_spearmans = []
  top2_jaccards = []
  for a_idx in range(len(pass_score_maps)):
    for b_idx in range(a_idx + 1, len(pass_score_maps)):
      a_values = [pass_score_maps[a_idx][action] for action in common_actions]
      b_values = [pass_score_maps[b_idx][action] for action in common_actions]
      pair_spearmans.append(_spearman(a_values, b_values))
      a_top2 = _top2_set({action: pass_score_maps[a_idx][action] for action in common_actions})
      b_top2 = _top2_set({action: pass_score_maps[b_idx][action] for action in common_actions})
      union = a_top2 | b_top2
      top2_jaccards.append(1.0 if not union else len(a_top2 & b_top2) / len(union))

  mean_scores = {
      action: float(np.mean([score_map[action] for score_map in pass_score_maps]))
      for action in common_actions
  }
  mean_raw_scores = {
      action: float(np.mean([score_map[action] for score_map in pass_raw_score_maps]))
      for action in common_actions
      if all(action in score_map for score_map in pass_raw_score_maps)
  }
  mean_paradoxes = {
      action: float(np.mean([score_map[action] for score_map in pass_paradox_maps]))
      for action in common_actions
      if all(action in score_map for score_map in pass_paradox_maps)
  }
  ordered_mean = sorted(mean_scores, key=lambda action: mean_scores[action], reverse=True)
  consensus_best = int(ordered_mean[0])
  consensus_second = int(ordered_mean[1]) if len(ordered_mean) > 1 else consensus_best
  consensus_margin = float(mean_scores[consensus_best] - mean_scores[consensus_second])
  high_margin = bool(consensus_margin >= float(args.min_high_margin))

  row.update({
      "evaluable": True,
      "common_scored_actions": int(len(common_actions)),
      "passes": pass_results,
      "top1_mode_action": int(mode_best),
      "top1_mode_action_string": action_string(state, player, mode_best),
      "top1_mode_rate": float(mode_count / max(1, len(bests))),
      "pairwise_spearman_mean": _pairwise_mean(pair_spearmans),
      "top2_jaccard_mean": _pairwise_mean(top2_jaccards),
      "pass_top_margin": _summarize_numbers(
          [item.get("top_margin") for item in pass_results]
      ),
      "consensus_best_action": int(consensus_best),
      "consensus_best_action_string": action_string(state, player, consensus_best),
      "consensus_top_margin": consensus_margin,
      "high_margin": high_margin,
      "consensus_matches_policy": bool(consensus_best == int(policy_best)),
      "consensus_score": float(mean_scores[consensus_best]),
      "policy_consensus_score": (
          None if int(policy_best) not in mean_scores
          else float(mean_scores[int(policy_best)])
      ),
      "mean_scores": {str(k): round(v, 6) for k, v in sorted(mean_scores.items())},
      "mean_raw_scores": {
          str(k): round(v, 6) for k, v in sorted(mean_raw_scores.items())
      },
      "mean_paradoxes": {
          str(k): round(v, 6) for k, v in sorted(mean_paradoxes.items())
      },
  })

  recall_every = int(getattr(args, "candidate_recall_audit_every", 0))
  recall_max_legal = int(getattr(args, "candidate_recall_max_legal", 0))
  if recall_every > 0 and (idx + 1) % recall_every == 0:
    if recall_max_legal > 0 and len(legal) > recall_max_legal:
      row["candidate_recall_audit"] = {
          "audited": False,
          "skip_reason": "too_many_legal_actions",
          "legal_count": int(len(legal)),
          "max_legal": int(recall_max_legal),
      }
    else:
      recall_seed = (
          int(args.seed)
          + int(getattr(args, "candidate_recall_seed_offset", 3000003))
          + idx * 104729
      )
      _set_seed(recall_seed)
      recall_stats = evaluate_actions_independently(
          state,
          player,
          legal,
          round_game,
          model,
          oracle_args,
          device,
          (
              int(args.candidate_recall_rollouts)
              if int(getattr(args, "candidate_recall_rollouts", 0)) > 0
              else int(args.rollouts_per_action)
          ),
      )
      recall_scores = {}
      recall_raw_scores = {}
      recall_paradox_scores = {}
      for action in legal:
        action_stats = recall_stats.get(int(action), {})
        score = _adjusted_score(action_stats, args.label_paradox_penalty)
        if score is None:
          continue
        recall_scores[int(action)] = float(score)
        recall_raw_scores[int(action)] = float(action_stats.get("mean"))
        if action_stats.get("mean_paradox") is not None:
          recall_paradox_scores[int(action)] = float(
              action_stats.get("mean_paradox")
          )
      if recall_scores:
        all_best = int(max(recall_scores, key=lambda action: recall_scores[action]))
        ordered_recall = sorted(
            recall_scores, key=lambda action: recall_scores[action], reverse=True
        )
        recall_second = ordered_recall[1] if len(ordered_recall) > 1 else all_best
        candidate_scored = [
            int(action) for action in candidates if int(action) in recall_scores
        ]
        candidate_best = (
            None if not candidate_scored
            else int(max(candidate_scored, key=lambda action: recall_scores[action]))
        )
        candidate_best_score = (
            None if candidate_best is None else float(recall_scores[candidate_best])
        )
        all_best_score = float(recall_scores[all_best])
        all_best_paradox = recall_paradox_scores.get(all_best)
        consensus_recall_score = (
            None if consensus_best not in recall_scores
            else float(recall_scores[consensus_best])
        )
        policy_recall_score = (
            None if int(policy_best) not in recall_scores
            else float(recall_scores[int(policy_best)])
        )
        policy_recall_paradox = recall_paradox_scores.get(int(policy_best))
        row["candidate_recall_audit"] = {
            "audited": True,
            "seed": int(recall_seed),
            "legal_count": int(len(legal)),
            "candidate_count": int(len(candidates)),
            "scored_legal_actions": int(len(recall_scores)),
            "all_legal_best_action": int(all_best),
            "all_legal_best_action_string": action_string(state, player, all_best),
            "all_legal_best_score": all_best_score,
            "all_legal_top_margin": float(
                all_best_score - float(recall_scores[recall_second])
            ),
            "all_legal_best_paradox": (
                None if all_best_paradox is None else float(all_best_paradox)
            ),
            "candidate_best_under_all_legal_action": candidate_best,
            "candidate_best_under_all_legal_action_string": (
                None
                if candidate_best is None
                else action_string(state, player, candidate_best)
            ),
            "candidate_best_under_all_legal_score": candidate_best_score,
            "all_legal_best_in_candidates": bool(all_best in set(candidates)),
            "consensus_is_all_legal_best": bool(int(consensus_best) == int(all_best)),
            "policy_is_all_legal_best": bool(int(policy_best) == int(all_best)),
            "missed_best_gap": (
                None if candidate_best_score is None
                else float(all_best_score - candidate_best_score)
            ),
            "all_legal_best_margin_vs_consensus": (
                None if consensus_recall_score is None
                else float(all_best_score - consensus_recall_score)
            ),
            "all_legal_best_margin_vs_policy": (
                None if policy_recall_score is None
                else float(all_best_score - policy_recall_score)
            ),
            "all_legal_paradox_delta_vs_policy": (
                None
                if all_best_paradox is None or policy_recall_paradox is None
                else float(all_best_paradox - policy_recall_paradox)
            ),
            "all_legal_scores": {
                str(k): round(v, 6) for k, v in sorted(recall_scores.items())
            },
            "all_legal_raw_scores": {
                str(k): round(v, 6) for k, v in sorted(recall_raw_scores.items())
            },
            "all_legal_paradoxes": {
                str(k): round(v, 6)
                for k, v in sorted(recall_paradox_scores.items())
            },
        }
      else:
        row["candidate_recall_audit"] = {
            "audited": False,
            "skip_reason": "no_scored_legal_actions",
            "seed": int(recall_seed),
            "legal_count": int(len(legal)),
        }

  if consensus_best == int(policy_best):
    row["forced_audit"] = {
        "audited": True,
        "audit_seed": None,
        "margin_vs_policy": 0.0,
        "paradox_delta_vs_policy": 0.0,
        "label_mean": mean_raw_scores.get(consensus_best),
        "policy_mean": mean_raw_scores.get(int(policy_best)),
    }
    return row

  audit_seed = int(args.seed) + int(args.audit_seed_offset) + idx * 104729
  _set_seed(audit_seed)
  audit = evaluate_actions_independently(
      state,
      player,
      [consensus_best, int(policy_best)],
      round_game,
      model,
      oracle_args,
      device,
      args.audit_rollouts,
  )
  label_stats = audit.get(consensus_best, {})
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


def _forced_summary(rows, high_margin_only):
  selected = [
      row for row in rows
      if row.get("evaluable")
      and (not high_margin_only or row.get("high_margin"))
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


def _row_q_spread(row, value_scale=20.0):
  raw_scores = row.get("mean_raw_scores") or {}
  values = []
  for score in raw_scores.values():
    try:
      values.append(float(score))
    except (TypeError, ValueError):
      continue
  if len(values) < 2:
    return 0.0
  scale = max(1e-6, float(value_scale))
  return float((max(values) - min(values)) / scale)


def _target_action_matches_state_bucket(row):
  return _action_matches_state_bucket(row, row.get("consensus_best_action"))


def _action_matches_state_bucket(row, action):
  group = row.get("group") or {}
  bucket = group.get("state_mix_bucket")
  if bucket not in TACTICAL_POLICY_BUCKET_NAMES:
    return True
  matches = (group.get("tactical_bucket_matches") or {}).get(bucket, [])
  try:
    action = int(action)
  except (TypeError, ValueError):
    return False
  try:
    return action in {int(item) for item in matches}
  except (TypeError, ValueError):
    return False


def _recall_stats_from_row(row):
  recall = row.get("candidate_recall_audit") or {}
  raw_scores = recall.get("all_legal_raw_scores") or {}
  paradoxes = recall.get("all_legal_paradoxes") or {}
  stats = {}
  for action_key, score in raw_scores.items():
    try:
      action = int(action_key)
    except (TypeError, ValueError):
      continue
    item = {"mean": float(score)}
    if str(action) in paradoxes:
      item["mean_paradox"] = float(paradoxes[str(action)])
    stats[action] = item
  return stats


def _recall_q_spread(row, value_scale=20.0):
  stats = _recall_stats_from_row(row)
  values = []
  for item in stats.values():
    mean = item.get("mean")
    if mean is not None:
      values.append(float(mean))
  if len(values) < 2:
    return 0.0
  scale = max(1e-6, float(value_scale))
  return float((max(values) - min(values)) / scale)


def _stable_recall_replay_reason(row, args):
  if not row.get("evaluable"):
    return "not_evaluable"
  recall = row.get("candidate_recall_audit") or {}
  if not recall.get("audited"):
    return "recall_not_audited"
  try:
    best_action = int(recall.get("all_legal_best_action"))
    policy_action = int(row.get("policy_best_action"))
  except (TypeError, ValueError):
    return "recall_missing_best"
  if (
      bool(getattr(args, "stable_require_high_margin", True))
      and float(recall.get("all_legal_top_margin", -math.inf))
      < float(getattr(args, "min_high_margin", 0.5))
  ):
    return "not_high_margin"
  if best_action == policy_action and not args.stable_include_policy_matches:
    return "matches_policy"
  policy_margin = recall.get("all_legal_best_margin_vs_policy")
  if (
      policy_margin is None
      or float(policy_margin) < float(args.stable_min_forced_margin)
  ):
    return "forced_margin"
  paradox_delta = recall.get("all_legal_paradox_delta_vs_policy")
  if (
      paradox_delta is None
      or float(paradox_delta) > float(args.stable_max_forced_paradox_delta)
  ):
    return "forced_paradox_delta"
  min_legal_count = int(getattr(args, "stable_min_legal_count", 0))
  legal_count = row.get("legal_count")
  if (
      min_legal_count > 0
      and (legal_count is None or int(legal_count) < min_legal_count)
  ):
    return "low_legal_count"
  min_q_spread = float(getattr(args, "stable_min_q_spread", 0.0))
  q_spread = _recall_q_spread(row, getattr(args, "value_scale", 20.0))
  if min_q_spread > 0.0 and q_spread < min_q_spread:
    return "low_q_spread"
  if (
      bool(getattr(args, "stable_require_target_bucket_match", False))
      and not _action_matches_state_bucket(row, best_action)
  ):
    return "target_bucket_mismatch"
  return None


def _stable_replay_reason(row, args):
  if bool(getattr(args, "stable_use_recall_labels", False)):
    return _stable_recall_replay_reason(row, args)
  if not row.get("evaluable"):
    return "not_evaluable"
  if bool(args.stable_require_high_margin) and not row.get("high_margin"):
    return "not_high_margin"
  if row.get("consensus_matches_policy") and not args.stable_include_policy_matches:
    return "matches_policy"
  top1 = row.get("top1_mode_rate")
  if top1 is None or float(top1) < float(args.stable_min_top1_mode_rate):
    return "unstable_top1"
  spearman = row.get("pairwise_spearman_mean")
  if spearman is None or float(spearman) < float(args.stable_min_spearman):
    return "unstable_rank"
  forced = row.get("forced_audit") or {}
  if not forced.get("audited"):
    return "not_forced_audited"
  margin = forced.get("margin_vs_policy")
  if margin is None or float(margin) < float(args.stable_min_forced_margin):
    return "forced_margin"
  paradox_delta = forced.get("paradox_delta_vs_policy")
  if (
      paradox_delta is None
      or float(paradox_delta) > float(args.stable_max_forced_paradox_delta)
  ):
    return "forced_paradox_delta"
  min_legal_count = int(getattr(args, "stable_min_legal_count", 0))
  legal_count = row.get("legal_count")
  if (
      min_legal_count > 0
      and (legal_count is None or int(legal_count) < min_legal_count)
  ):
    return "low_legal_count"
  min_q_spread = float(getattr(args, "stable_min_q_spread", 0.0))
  q_spread = _row_q_spread(row, getattr(args, "value_scale", 20.0))
  if min_q_spread > 0.0 and q_spread < min_q_spread:
    return "low_q_spread"
  if bool(getattr(args, "stable_require_recall_consensus", False)):
    recall = row.get("candidate_recall_audit") or {}
    if not recall.get("audited"):
      return "recall_not_audited"
    if not recall.get("consensus_is_all_legal_best"):
      return "recall_disagrees"
  if (
      bool(getattr(args, "stable_require_target_bucket_match", False))
      and not _target_action_matches_state_bucket(row)
  ):
    return "target_bucket_mismatch"
  return None


def _stats_from_row(row):
  stats = {}
  raw_scores = row.get("mean_raw_scores") or {}
  paradoxes = row.get("mean_paradoxes") or {}
  for action_key, score in raw_scores.items():
    try:
      action = int(action_key)
    except (TypeError, ValueError):
      continue
    item = {"mean": float(score)}
    if str(action) in paradoxes:
      item["mean_paradox"] = float(paradoxes[str(action)])
    stats[action] = item
  forced_raw = (row.get("forced_audit") or {}).get("raw") or {}
  for action_key, item in forced_raw.items():
    try:
      action = int(action_key)
    except (TypeError, ValueError):
      continue
    if item:
      stats[action] = item
  return stats


def _build_stable_replay(states, rows, oracle_args, args):
  replay = []
  metadata = []
  skipped = {}
  for idx, row in enumerate(rows):
    reason = _stable_replay_reason(row, args)
    if reason is not None:
      skipped[reason] = skipped.get(reason, 0) + 1
      continue
    if idx >= len(states):
      skipped["missing_state"] = skipped.get("missing_state", 0) + 1
      continue
    state = states[idx]
    player = int(row["player"])
    policy_action = int(row["policy_best_action"])
    if bool(getattr(args, "stable_use_recall_labels", False)):
      recall = row.get("candidate_recall_audit") or {}
      best_action = int(recall["all_legal_best_action"])
      stats = _recall_stats_from_row(row)
      scored_actions = set(stats)
      best_action_string = recall.get("all_legal_best_action_string")
      export_label_source = "all_legal_recall"
      q_spread = _recall_q_spread(row, getattr(args, "value_scale", 20.0))
      target_matches = _action_matches_state_bucket(row, best_action)
    else:
      best_action = int(row["consensus_best_action"])
      stats = _stats_from_row(row)
      scored_actions = set(int(action) for action in row.get("candidate_actions", []))
      best_action_string = row.get("consensus_best_action_string")
      export_label_source = "candidate_consensus"
      q_spread = _row_q_spread(row, getattr(args, "value_scale", 20.0))
      target_matches = _target_action_matches_state_bucket(row)
    if best_action not in stats or policy_action not in stats:
      skipped["missing_stats"] = skipped.get("missing_stats", 0) + 1
      continue
    replay.append(
        accepted_replay_example(
            state,
            player,
            best_action,
            policy_action,
            stats,
            oracle_args,
            label_stats=stats,
            scored_actions=scored_actions,
        )
    )
    metadata.append({
        "source_index": int(idx),
        "phase": row.get("phase"),
        "player": int(player),
        "group": row.get("group") or {},
        "policy_best_action": int(policy_action),
        "policy_best_action_string": row.get("policy_best_action_string"),
        "consensus_best_action": int(best_action),
        "consensus_best_action_string": best_action_string,
        "export_label_source": export_label_source,
        "candidate_consensus_best_action": row.get("consensus_best_action"),
        "candidate_consensus_best_action_string": row.get(
            "consensus_best_action_string"
        ),
        "consensus_top_margin": row.get("consensus_top_margin"),
        "top1_mode_rate": row.get("top1_mode_rate"),
        "pairwise_spearman_mean": row.get("pairwise_spearman_mean"),
        "forced_audit": row.get("forced_audit"),
        "candidate_recall_audit": row.get("candidate_recall_audit"),
        "candidate_actions": row.get("candidate_actions"),
        "candidate_count": row.get("candidate_count"),
        "target_bucket_candidate_actions": row.get(
            "target_bucket_candidate_actions"
        ),
        "target_bucket_candidate_count": row.get(
            "target_bucket_candidate_count"
        ),
        "legal_count": row.get("legal_count"),
        "q_spread": q_spread,
        "target_action_matches_state_bucket": target_matches,
        "stable_export": True,
    })
  return replay, metadata, skipped


def _stable_row_count(rows, args):
  return sum(1 for row in rows if _stable_replay_reason(row, args) is None)


def _stable_row_counts_by_state_mix_bucket(rows, args):
  counts = Counter()
  for row in rows:
    if _stable_replay_reason(row, args) is not None:
      continue
    bucket = ((row.get("group") or {}).get("state_mix_bucket") or "unknown")
    counts[str(bucket)] += 1
  return counts


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


def _candidate_recall_summary(rows, high_margin_only):
  selected = [
      row for row in rows
      if row.get("evaluable")
      and (not high_margin_only or row.get("high_margin"))
      and (row.get("candidate_recall_audit") or {}).get("audited")
  ]
  if not selected:
    return {
        "rows": 0,
        "all_legal_best_in_candidates_rate": None,
        "consensus_is_all_legal_best_rate": None,
        "policy_is_all_legal_best_rate": None,
        "mean_missed_best_gap": None,
        "missed_best_gap_gt_0_5": 0,
        "mean_all_legal_best_margin_vs_consensus": None,
        "mean_all_legal_best_margin_vs_policy": None,
  }
  recalls = [row["candidate_recall_audit"] for row in selected]
  missed_gaps = [
      item.get("missed_best_gap")
      for item in recalls
      if item.get("missed_best_gap") is not None
  ]
  consensus_gaps = [
      item.get("all_legal_best_margin_vs_consensus")
      for item in recalls
      if item.get("all_legal_best_margin_vs_consensus") is not None
  ]
  policy_gaps = [
      item.get("all_legal_best_margin_vs_policy")
      for item in recalls
      if item.get("all_legal_best_margin_vs_policy") is not None
  ]
  return {
      "rows": int(len(selected)),
      "all_legal_best_in_candidates_rate": float(
          np.mean([1.0 if item.get("all_legal_best_in_candidates") else 0.0
                   for item in recalls])
      ),
      "consensus_is_all_legal_best_rate": float(
          np.mean([1.0 if item.get("consensus_is_all_legal_best") else 0.0
                   for item in recalls])
      ),
      "policy_is_all_legal_best_rate": float(
          np.mean([1.0 if item.get("policy_is_all_legal_best") else 0.0
                   for item in recalls])
      ),
      "mean_missed_best_gap": (
          None if not missed_gaps else float(np.mean(missed_gaps))
      ),
      "missed_best_gap_gt_0_5": int(
          sum(1 for gap in missed_gaps if float(gap) > 0.5)
      ),
      "mean_all_legal_best_margin_vs_consensus": (
          None if not consensus_gaps else float(np.mean(consensus_gaps))
      ),
      "mean_all_legal_best_margin_vs_policy": (
          None if not policy_gaps else float(np.mean(policy_gaps))
      ),
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
      "consensus_top_margin": _summarize_numbers(
          [row.get("consensus_top_margin") for row in evaluable]
      ),
      "forced_audit": _forced_summary(rows, high_margin_only=False),
      "high_margin_forced_audit": _forced_summary(rows, high_margin_only=True),
      "candidate_recall_audit": _candidate_recall_summary(
          rows, high_margin_only=False
      ),
      "high_margin_candidate_recall_audit": _candidate_recall_summary(
          rows, high_margin_only=True
      ),
  }


def main():
  args = parse_args()
  started = time.perf_counter()
  _set_seed(args.seed)
  device = _device(args.device)
  load_game = make_game(SimpleNamespace(players=args.players, match_context=True), 0)
  model, _, saved_args = load_model_payload(
      args.checkpoint, load_game, model_args_for(args.players), device
  )
  args.value_scale = float(saved_args.get("value_scale", 20.0))
  model.eval()
  wanted = {part.strip() for part in args.phases.split(",") if part.strip()}
  states, groups, bucket_reports = _collect_mixed_states(
      model, saved_args, args, device, wanted
  )
  interleave_report = {"enabled": False, "reason": "disabled"}
  if bool(getattr(args, "interleave_state_buckets", True)):
    states, groups, interleave_report = _interleave_states_by_bucket(
        states, groups
    )
  collection_summary = _collection_state_summary(states, groups)
  if args.collect_only:
    result = {
        "checkpoint": args.checkpoint,
        "players": int(args.players),
        "seed": int(args.seed),
        "collect_only": True,
        "states_requested": int(args.states),
        "states_collected": int(len(states)),
        "states_evaluated": 0,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "phases": sorted(wanted),
        "state_mix": args.state_mix,
        "state_buckets": bucket_reports,
        "state_bucket_interleave": interleave_report,
        "state_min_legal_count": int(args.state_min_legal_count),
        "state_max_legal_count": int(args.state_max_legal_count),
        "state_min_trick_number": int(args.state_min_trick_number),
        "state_max_trick_number": int(args.state_max_trick_number),
        "state_min_match_round": int(args.state_min_match_round),
        "state_max_match_round": int(args.state_max_match_round),
        "max_states_per_match": int(args.max_states_per_match),
        "state_pool_per_match": int(args.state_pool_per_match),
        "collection_state_summary": collection_summary,
        "rows": [
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
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return
  oracle_cli = _oracle_cli_namespace(
      args,
      max(1, len(states)),
      args.state_selection,
      (args.hard_entropy_weight, args.hard_loss_weight, args.hard_paradox_weight),
      args.seed,
  )
  oracle_args = audit_namespace(oracle_cli, saved_args)

  rows = []
  early_stop_reason = None
  required_stable_buckets = sorted({
      str(group.get("state_mix_bucket"))
      for group in groups
      if group.get("state_mix_bucket") is not None
  })
  for idx, state in enumerate(states):
    row = _evaluate_state(
        idx,
        state,
        groups[idx] if idx < len(groups) else {},
        model,
        oracle_args,
        args,
        device,
    )
    rows.append(row)
    stable_found = _stable_row_count(rows, args) if args.stable_replay_out else 0
    stable_by_bucket = (
        _stable_row_counts_by_state_mix_bucket(rows, args)
        if args.stable_replay_out
        else Counter()
    )
    if args.progress_interval > 0 and (
        (idx + 1) % args.progress_interval == 0 or (idx + 1) == len(states)
    ):
      print(
          json.dumps({
              "processed_states": int(idx + 1),
              "states_collected": int(len(states)),
              "evaluable": int(sum(1 for item in rows if item.get("evaluable"))),
              "high_margin": int(sum(1 for item in rows if item.get("high_margin"))),
              "stable_export_rows": int(stable_found),
              "stable_export_rows_by_state_bucket": dict(
                  sorted(stable_by_bucket.items())
              ),
              "elapsed_sec": round(time.perf_counter() - started, 3),
          }),
          flush=True,
      )
    if (
        args.stable_replay_out
        and int(args.stable_target_rows_per_state_bucket) > 0
        and required_stable_buckets
        and all(
            int(stable_by_bucket.get(bucket, 0))
            >= int(args.stable_target_rows_per_state_bucket)
            for bucket in required_stable_buckets
        )
    ):
      early_stop_reason = "stable_target_rows_per_state_bucket"
      break
    if (
        args.stable_replay_out
        and int(args.stable_target_rows) > 0
        and int(args.stable_target_rows_per_state_bucket) <= 0
        and stable_found >= int(args.stable_target_rows)
    ):
      early_stop_reason = "stable_target_rows"
      break

  stability = _stability_summary(rows)
  high_stability = stability["high_margin_top1_mode_rate"]
  high_spearman = stability["high_margin_pairwise_spearman"]
  forced = stability["high_margin_forced_audit"]
  teacher_stability_pass = bool(
      stability["high_margin_rows"] >= int(args.pass_min_high_margin_rows)
      and high_stability is not None
      and high_spearman is not None
      and forced["rows"] > 0
      and high_stability["mean"] >= float(args.pass_min_high_margin_top1_stability)
      and high_spearman["mean"] >= float(args.pass_min_spearman)
      and forced["score_rate"] is not None
      and forced["score_rate"] >= float(args.pass_min_forced_score_rate)
      and forced["mean_margin"] is not None
      and forced["mean_margin"] > float(args.pass_min_forced_avg_diff)
      and forced["mean_paradox_delta"] is not None
      and forced["mean_paradox_delta"] <= float(args.pass_max_forced_paradox_delta)
  )

  stable_replay = []
  stable_metadata = []
  stable_skipped = {}
  if args.stable_replay_out:
    stable_replay, stable_metadata, stable_skipped = _build_stable_replay(
        states, rows, oracle_args, args
    )
    stable_replay_by_state_mix_bucket = Counter(
        str((row.get("group") or {}).get("state_mix_bucket", "unknown"))
        for row in stable_metadata
    )
    stable_path = Path(args.stable_replay_out)
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    save_replay(stable_replay, stable_path)
    metadata_path = (
        Path(args.stable_replay_metadata_out)
        if args.stable_replay_metadata_out
        else stable_path.with_suffix(stable_path.suffix + ".metadata.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "stable_replay_out": str(stable_path),
        "stable_replay_rows": int(len(stable_replay)),
        "stable_replay_summary": accepted_replay_summary(stable_replay),
        "stable_replay_by_state_mix_bucket": dict(
            sorted(stable_replay_by_state_mix_bucket.items())
        ),
        "stable_export_criteria": {
            "stable_require_high_margin": bool(args.stable_require_high_margin),
            "stable_min_top1_mode_rate": float(args.stable_min_top1_mode_rate),
            "stable_min_spearman": float(args.stable_min_spearman),
            "stable_min_forced_margin": float(args.stable_min_forced_margin),
            "stable_max_forced_paradox_delta": float(
                args.stable_max_forced_paradox_delta
            ),
            "stable_min_legal_count": int(args.stable_min_legal_count),
            "stable_min_q_spread": float(args.stable_min_q_spread),
            "stable_require_recall_consensus": bool(
                args.stable_require_recall_consensus
            ),
            "stable_use_recall_labels": bool(args.stable_use_recall_labels),
            "stable_require_target_bucket_match": bool(
                args.stable_require_target_bucket_match
            ),
            "stable_include_policy_matches": bool(
                args.stable_include_policy_matches
            ),
            "stable_target_rows": int(args.stable_target_rows),
            "stable_target_rows_per_state_bucket": int(
                args.stable_target_rows_per_state_bucket
            ),
        },
        "stable_skipped": stable_skipped,
        "rows": stable_metadata,
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
      "state_bucket_interleave": interleave_report,
      "state_min_legal_count": int(args.state_min_legal_count),
      "state_max_legal_count": int(args.state_max_legal_count),
      "state_min_trick_number": int(args.state_min_trick_number),
      "state_max_trick_number": int(args.state_max_trick_number),
      "state_min_match_round": int(args.state_min_match_round),
      "state_max_match_round": int(args.state_max_match_round),
      "collection_state_summary": collection_summary,
      "repeat_label_passes": int(args.repeat_label_passes),
      "rollouts_per_action": int(args.rollouts_per_action),
      "audit_rollouts": int(args.audit_rollouts),
      "belief_samples": int(args.belief_samples),
      "belief_candidates": int(args.belief_candidates),
      "belief_ref_policy_mix": args.belief_ref_policy_mix,
      "label_max_actions": int(args.label_max_actions),
      "label_top_policy": int(args.label_top_policy),
      "label_target_bucket_actions": int(args.label_target_bucket_actions),
      "label_include_bots": args.label_include_bots,
      "label_feature_candidates": bool(args.label_feature_candidates),
      "label_paradox_penalty": float(args.label_paradox_penalty),
      "full_match_rollout": bool(args.full_match_rollout),
      "rollout_max_plies": int(args.rollout_max_plies),
      "min_high_margin": float(args.min_high_margin),
      "pass_min_high_margin_rows": int(args.pass_min_high_margin_rows),
      "pass_min_high_margin_top1_stability": (
          float(args.pass_min_high_margin_top1_stability)
      ),
      "pass_min_spearman": float(args.pass_min_spearman),
      "pass_min_forced_score_rate": float(args.pass_min_forced_score_rate),
      "pass_min_forced_avg_diff": float(args.pass_min_forced_avg_diff),
      "pass_max_forced_paradox_delta": float(args.pass_max_forced_paradox_delta),
      "teacher_stability_pass": teacher_stability_pass,
      "stable_replay_out": args.stable_replay_out,
      "stable_replay_rows": int(len(stable_replay)),
      "stable_replay_summary": accepted_replay_summary(stable_replay),
      "stable_replay_by_state_mix_bucket": dict(sorted(Counter(
          str((row.get("group") or {}).get("state_mix_bucket", "unknown"))
          for row in stable_metadata
      ).items())),
      "stable_export_criteria": {
          "stable_require_high_margin": bool(args.stable_require_high_margin),
          "stable_min_top1_mode_rate": float(args.stable_min_top1_mode_rate),
          "stable_min_spearman": float(args.stable_min_spearman),
          "stable_min_forced_margin": float(args.stable_min_forced_margin),
          "stable_max_forced_paradox_delta": float(
              args.stable_max_forced_paradox_delta
          ),
          "stable_min_legal_count": int(args.stable_min_legal_count),
          "stable_min_q_spread": float(args.stable_min_q_spread),
          "stable_require_recall_consensus": bool(
              args.stable_require_recall_consensus
          ),
          "stable_use_recall_labels": bool(args.stable_use_recall_labels),
          "stable_require_target_bucket_match": bool(
              args.stable_require_target_bucket_match
          ),
          "stable_include_policy_matches": bool(args.stable_include_policy_matches),
          "stable_target_rows": int(args.stable_target_rows),
          "stable_target_rows_per_state_bucket": int(
              args.stable_target_rows_per_state_bucket
          ),
      },
      "candidate_recall_audit_every": int(args.candidate_recall_audit_every),
      "candidate_recall_max_legal": int(args.candidate_recall_max_legal),
      "candidate_recall_rollouts": int(args.candidate_recall_rollouts),
      "stable_skipped": stable_skipped,
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
