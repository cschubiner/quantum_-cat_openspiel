#!/usr/bin/env python3
"""Collect dense, independently audited correction trajectories.

Unlike the static best-of-rollouts miner, this collector evaluates decisions
while the round is being played. A correction is applied to the live source
trajectory only when it clears selection and every fresh information-state
audit pass. Later labels therefore come from states induced by earlier audited
corrections instead of only from the frozen incumbent distribution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyspiel
import torch

from quantum_cat_alphazero_torch import save_replay
from quantum_cat_best_of_rollouts_replay import (
    _acceptance,
    _audit_acceptance,
    _base_keep_target,
    _candidate_actions,
    _decision_allowed,
    _device_from_arg,
    _evaluate_action,
    _make_replay_row,
    _paired_any_paradox_summary,
    _public_decision,
    _rollout_states_for_decision,
    _target_from_records,
)
from quantum_cat_failure_miner import (
    action_string,
    choose_policy_action,
    parse_phase_filter,
    phase_name,
    resolve_actor_bot_args,
    terminal_round_result,
)
from quantum_cat_full_match_elo import load_neural
from quantum_cat_public_history import public_history_metadata


_NUMPY_SEED_MODULUS = 2**32 - 1
_COLOR_NAMES = ("R", "B", "Y", "G")
_COLLECTION_CONTRACT_VERSION = "chained_best_of_rollouts_v3"


def _numpy_seed(value: int) -> int:
  """Map deterministic seed blocks into RandomState's accepted range."""
  return int(int(value) % _NUMPY_SEED_MODULUS)


def _hard_audited_target(selection_target, best_action: int) -> np.ndarray:
  """Keep policy mass only on the action that cleared every fresh audit.

  The selection-time soft target can contain tied or near-best alternatives.
  Those alternatives were not evaluated by the independent audit passes, so
  retaining their mass would quietly supervise unaudited actions.  Chained
  correction replay therefore uses a hard target on the one audited action.
  """
  target = np.asarray(selection_target, dtype=np.float32)
  best_action = int(best_action)
  if target.ndim != 1 or not 0 <= best_action < len(target):
    raise ValueError("invalid audited correction target")
  if not np.all(np.isfinite(target)) or float(target[best_action]) <= 0.0:
    raise ValueError("audited action is absent from the selection target")
  audited = np.zeros_like(target)
  audited[best_action] = 1.0
  return audited


def _public_play_event(
    state,
    player: int,
    action: int,
    *,
    decision_index: int,
    base_action: int,
    correction_applied: bool,
) -> dict | None:
  """Serialize one already-public card play without private deal state."""
  action = int(action)
  num_ranks = int(getattr(state, "_num_card_types", 0))
  color_index = action // max(1, num_ranks)
  rank_index = action % max(1, num_ranks)
  if (
      action == 999
      or num_ranks <= 0
      or not 0 <= color_index < len(_COLOR_NAMES)
      or not 0 <= rank_index < num_ranks
  ):
    return None
  return {
      "sequence_index": -1,
      "decision_index": int(decision_index),
      "trick_number": int(getattr(state, "_trick_number", -1)),
      "trick_start_player": int(getattr(state, "_start_player", -1)),
      "player": int(player),
      "action": action,
      "rank": int(rank_index + 1),
      "color": str(_COLOR_NAMES[color_index]),
      "base_action": int(base_action),
      "correction_applied": bool(correction_applied),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--matches", type=int, default=100)
  parser.add_argument("--seed", type=int, default=2026081001)
  parser.add_argument("--match-context", action="store_true")
  parser.add_argument("--decision-phases", default="play")
  parser.add_argument("--bot-mode", default="policy")
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
  parser.add_argument("--out", required=True)
  parser.add_argument("--metadata-out", default="")
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--base-keep-out", default="")
  parser.add_argument("--base-keep-metadata-out", default="")
  parser.add_argument("--source-group-id", default="")
  parser.add_argument("--max-evaluated-decisions", type=int, default=0)
  parser.add_argument("--max-evaluations-per-round", type=int, default=0)
  parser.add_argument("--max-positive-rows", type=int, default=0)
  parser.add_argument("--max-base-keep-rows", type=int, default=0)
  parser.add_argument("--min-legal-count", type=int, default=2)
  parser.add_argument("--max-legal-count", type=int, default=0)
  parser.add_argument("--min-trick-number", type=int, default=0)
  parser.add_argument("--max-trick-number", type=int, default=99)
  parser.add_argument("--led-contrast-only", action="store_true")
  parser.add_argument("--require-red-switch-legal", action="store_true")
  parser.add_argument("--rollout-state-source", choices=("infostate",), default="infostate")
  parser.add_argument("--belief-samples", type=int, default=4)
  parser.add_argument("--rollouts-per-action", type=int, default=1)
  parser.add_argument("--audit-belief-samples", type=int, default=4)
  parser.add_argument("--audit-rollouts-per-action", type=int, default=1)
  parser.add_argument("--audit-passes", type=int, default=3)
  parser.add_argument("--max-candidate-actions", type=int, default=5)
  parser.add_argument("--policy-top-actions", type=int, default=3)
  parser.add_argument("--include-all-legal", action="store_true")
  parser.add_argument(
      "--require-policy-target-changed",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--require-survival-contrast",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--min-any-paradox-rate-spread", type=float, default=0.5)
  parser.add_argument("--require-forced-paradox-contrast", action="store_true")
  parser.add_argument("--min-forced-paradox-rate-spread", type=float, default=0.5)
  parser.add_argument("--min-best-vs-base-score-delta", type=float, default=-1e9)
  parser.add_argument(
      "--min-best-vs-base-any-paradox-improvement", type=float, default=0.5
  )
  parser.add_argument(
      "--min-audit-best-vs-base-any-paradox-improvement",
      type=float,
      default=0.25,
  )
  parser.add_argument(
      "--max-audit-paired-binomial-p",
      type=float,
      default=1.0,
      help=(
          "Optional per-pass one-sided exact-binomial evidence threshold for "
          "paired saved-versus-harmed information-state worlds. Multiple "
          "continuations within one world are nested replicates and count as "
          "one inference unit."
      ),
  )
  parser.add_argument("--target-temperature", type=float, default=0.20)
  parser.add_argument("--target-top-score-gap", type=float, default=2.0)
  parser.add_argument("--survival-weight", type=float, default=10.0)
  parser.add_argument("--actor-survival-weight", type=float, default=2.0)
  parser.add_argument("--forced-paradox-penalty", type=float, default=1.0)
  parser.add_argument("--actor-score-weight", type=float, default=0.20)
  parser.add_argument("--total-score-weight", type=float, default=0.02)
  parser.add_argument("--action-value-safe-target", type=float, default=0.90)
  parser.add_argument("--action-value-paradox-target", type=float, default=-0.55)
  parser.add_argument("--action-value-actor-penalty", type=float, default=0.10)
  parser.add_argument("--action-value-forced-penalty", type=float, default=0.25)
  parser.add_argument("--action-value-score-weight", type=float, default=0.05)
  parser.add_argument("--action-value-min", type=float, default=-0.95)
  parser.add_argument("--action-value-max", type=float, default=0.95)
  parser.add_argument("--progress-interval", type=int, default=0)
  return parser.parse_args()


def _audit_passes(
    model,
    decision: dict,
    best_action: int,
    value_scale: float,
    device: torch.device,
    seed_base: int,
    args,
) -> dict:
  """Evaluate base/candidate on disjoint fresh worlds for every pass."""
  passes = []
  audit_actions = list(dict.fromkeys([
      int(decision["action"]),
      int(best_action),
  ]))
  for pass_index in range(int(args.audit_passes)):
    world_seed = _numpy_seed(
        int(seed_base) + 900000007 + pass_index * 1000000007
    )
    continuation_seed = _numpy_seed(
        int(seed_base) + 800000011 + pass_index * 1000000033
    )
    try:
      audit_states = _rollout_states_for_decision(
          decision,
          seed_base=world_seed,
          args=args,
          belief_samples=int(args.audit_belief_samples),
      )
      records = [
          _evaluate_action(
              model,
              decision,
              action,
              audit_states,
              value_scale,
              device,
              seed_base=continuation_seed,
              args=args,
              rollouts_per_action=int(args.audit_rollouts_per_action),
          )
          for action in audit_actions
      ]
      passed, reason, improvement = _audit_acceptance(
          decision,
          records,
          int(best_action),
          args,
      )
      by_action = {int(item["action"]): item for item in records}
      paired = _paired_any_paradox_summary(
          by_action[int(decision["action"])],
          by_action[int(best_action)],
      )
      passes.append({
          "pass_index": int(pass_index),
          "world_seed": int(world_seed),
          "continuation_seed": int(continuation_seed),
          "belief_samples": int(len(audit_states)),
          "rollouts_per_action": int(args.audit_rollouts_per_action),
          "passed": bool(passed),
          "error": False,
          "reason": str(reason),
          "improvement": float(improvement),
          "paired_any_paradox": paired,
          "records": records,
      })
    except Exception as exc:
      passes.append({
          "pass_index": int(pass_index),
          "world_seed": int(world_seed),
          "continuation_seed": int(continuation_seed),
          "passed": False,
          "error": True,
          "reason": f"{type(exc).__name__}:{exc}",
          "improvement": -2.0,
          "records": [],
      })
  return {
      "required_passes": int(args.audit_passes),
      "completed_passes": int(len(passes)),
      "all_passed": bool(
          len(passes) == int(args.audit_passes)
          and all(bool(item["passed"]) for item in passes)
      ),
      "error_free": bool(
          len(passes) == int(args.audit_passes)
          and not any(bool(item.get("error", False)) for item in passes)
      ),
      "min_improvement": float(
          min((float(item["improvement"]) for item in passes), default=-2.0)
      ),
      "passes": passes,
  }


def _trajectory_metadata(
    *,
    args,
    decision: dict,
    match_index: int,
    round_index: int,
    round_seed: int,
    start_player: int,
    records: list[dict],
    best_action: int,
    audit: dict,
    corrections_before: list[dict],
    public_play_trace_before: list[dict],
    target_kind: str,
) -> dict:
  base_action = int(decision["action"])
  best_record = next(
      item for item in records if int(item["action"]) == int(best_action)
  )
  base_record = next(
      item for item in records if int(item["action"]) == int(base_action)
  )
  applied_action = int(best_action) if target_kind == "audited_correction" else base_action
  return {
      "source": "chained_best_of_rollouts",
      "source_group_id": str(args.source_group_id),
      "trajectory_id": (
          f"{args.source_group_id or args.seed}:match={match_index}:round={round_index}"
      ),
      "match_index": int(match_index),
      "round_index": int(round_index),
      "round_seed": int(round_seed),
      "start_player": int(start_player),
      "decision": _public_decision(decision),
      "public_history": public_history_metadata(
          decision["state"], int(decision["player"])
      ),
      "public_history_trace_contract": _COLLECTION_CONTRACT_VERSION,
      "public_play_trace_length": int(len(public_play_trace_before)),
      "public_play_trace_before": [
          dict(item) for item in public_play_trace_before
      ],
      "num_players": int(getattr(decision["state"], "_num_players", args.players)),
      "num_tricks": int(getattr(decision["state"], "_num_tricks", 0)),
      "num_card_types": int(
          getattr(decision["state"], "_num_card_types", 0)
      ),
      "target_kind": str(target_kind),
      "policy_base_action": int(base_action),
      "policy_best_action": int(base_action),
      "best_action": int(applied_action),
      "best_action_string": action_string(
          decision["state"], int(decision["player"]), int(applied_action)
      ),
      "proposed_best_action": int(best_action),
      "proposed_best_action_string": action_string(
          decision["state"], int(decision["player"]), int(best_action)
      ),
      "applied_action": int(applied_action),
      "policy_target_changed": bool(int(applied_action) != int(base_action)),
      "audit_override_rejected": bool(target_kind == "audit_rejected_base_keep"),
      "audit_base_keep_validation": bool(
          target_kind == "audit_rejected_base_keep"
      ),
      "information_safe_rollout_teacher": True,
      "independent_belief_validation": True,
      "independent_audit_passes": int(audit["required_passes"]),
      "all_audit_passes_passed": bool(audit["all_passed"]),
      "audit_min_best_vs_base_any_paradox_improvement": float(
          audit["min_improvement"]
      ),
      "audit_passes": audit["passes"],
      "selection_best_vs_base_any_paradox_improvement": float(
          float(base_record["any_paradox_rate"])
          - float(best_record["any_paradox_rate"])
      ),
      "rollout_state_source": "infostate",
      "belief_samples": int(args.belief_samples),
      "rollouts_per_action": int(args.rollouts_per_action),
      "rollout_records": records,
      "corrections_before": int(len(corrections_before)),
      "prior_audited_corrections": [dict(item) for item in corrections_before],
  }


def _output_paths(args):
  out = Path(args.out)
  metadata = (
      Path(args.metadata_out)
      if str(args.metadata_out or "").strip()
      else out.with_suffix(out.suffix + ".metadata.json")
  )
  summary = (
      Path(args.summary_out)
      if str(args.summary_out or "").strip()
      else out.with_suffix(out.suffix + ".summary.json")
  )
  base_out = Path(args.base_keep_out) if str(args.base_keep_out or "").strip() else None
  base_metadata = None
  if base_out is not None:
    base_metadata = (
        Path(args.base_keep_metadata_out)
        if str(args.base_keep_metadata_out or "").strip()
        else base_out.with_suffix(base_out.suffix + ".metadata.json")
    )
  return out, metadata, summary, base_out, base_metadata


def _write_bank(path: Path, metadata_path: Path, rows, row_metadata, top: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  metadata_path.parent.mkdir(parents=True, exist_ok=True)
  embedded = {
      **top,
      "rows": int(len(rows)),
      "row_group_ids": [str(top["source_group_id"])] * len(rows),
      "row_group_id_kind": "independent_source_shard",
      "row_group_count": 1 if rows else 0,
  }
  if rows:
    save_replay(rows, path, metadata=embedded)
  elif path.exists():
    raise RuntimeError(
        f"refusing to leave a stale replay at {path} when the new bank is empty"
    )
  metadata_path.write_text(
      json.dumps({**embedded, "rows": row_metadata}, indent=2, sort_keys=True)
      + "\n"
  )


def main() -> None:
  args = parse_args()
  if int(args.audit_passes) < 2:
    raise ValueError("--audit-passes must be at least 2 for chained supervision")
  if int(args.audit_belief_samples) <= 0:
    raise ValueError("--audit-belief-samples must be positive")
  if int(args.audit_rollouts_per_action) <= 0:
    raise ValueError("--audit-rollouts-per-action must be positive")
  if not 0.0 <= float(args.max_audit_paired_binomial_p) <= 1.0:
    raise ValueError("--max-audit-paired-binomial-p must be in [0, 1]")
  source_bot_mode = str(args.bot_mode)
  args = resolve_actor_bot_args(args)
  allowed_phases = parse_phase_filter(args.decision_phases)
  device = _device_from_arg(str(args.device))
  model, model_args = load_neural(args.checkpoint, int(args.players), device, args)
  if model is None:
    raise ValueError("--checkpoint failed to load")
  value_scale = float(getattr(model_args, "value_scale", 20.0))
  np.random.seed(int(args.seed))
  torch.manual_seed(int(args.seed))

  positive_rows = []
  positive_metadata = []
  base_rows = []
  base_metadata = []
  summary = Counter()
  reject_reasons = Counter()
  audit_error_reasons = Counter()

  def evaluation_cap_reached() -> bool:
    cap = int(args.max_evaluated_decisions)
    return bool(cap > 0 and summary["decisions_evaluated"] >= cap)

  def row_caps_reached() -> bool:
    positive_cap = int(args.max_positive_rows)
    base_cap = int(args.max_base_keep_rows)
    positive_full = bool(positive_cap > 0 and len(positive_rows) >= positive_cap)
    base_full = bool(base_cap > 0 and len(base_rows) >= base_cap)
    return bool(positive_full and (base_cap <= 0 or base_full))

  for match_index in range(int(args.matches)):
    if evaluation_cap_reached() or row_caps_reached():
      break
    totals = np.zeros(int(args.players), dtype=np.float32)
    for round_index in range(int(args.players)):
      if evaluation_cap_reached() or row_caps_reached():
        break
      start_player = int(round_index % int(args.players))
      game = pyspiel.load_game(
          "python_quantum_cat",
          {
              "players": int(args.players),
              "start_player": int(start_player),
              "match_context": int(bool(args.match_context)),
          },
      )
      state = game.new_initial_state()
      if bool(args.match_context):
        state.set_match_context(np.copy(totals), round_index)
      round_seed = int(args.seed) + match_index * 1000003 + round_index * 9973
      np.random.seed(round_seed)
      decisions_in_round = 0
      evaluations_in_round = 0
      corrections = []
      public_play_trace = []
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes = state.chance_outcomes()
          actions, probs = zip(*outcomes)
          state.apply_action(int(np.random.choice(actions, p=probs)))
          continue
        player = int(state.current_player())
        legal = [int(action) for action in state.legal_actions(player)]
        state_before = state.clone()
        base_action, policy = choose_policy_action(
            model,
            state,
            player,
            value_scale,
            device,
            args,
        )
        decision = {
            "index": int(decisions_in_round),
            "state": state_before,
            "player": int(player),
            "phase": phase_name(state_before),
            "trick_number": int(getattr(state_before, "_trick_number", -1)),
            "legal": legal,
            "action": int(base_action),
            "action_string": action_string(state_before, player, base_action),
            "policy_prob": float(policy[int(base_action)]),
            "policy": np.asarray(policy, dtype=np.float32),
        }
        decisions_in_round += 1
        applied_action = int(base_action)
        eligible = _decision_allowed(decision, allowed_phases, args)
        round_cap = int(args.max_evaluations_per_round)
        if round_cap > 0 and evaluations_in_round >= round_cap:
          eligible = False
        if eligible and not evaluation_cap_reached():
          evaluations_in_round += 1
          summary["decisions_evaluated"] += 1
          summary[f"decisions_trick_{decision['trick_number']}"] += 1
          summary[f"decisions_after_{len(corrections)}_corrections"] += 1
          candidates = _candidate_actions(decision, args)
          if len(candidates) < 2:
            reject_reasons["too_few_candidates"] += 1
          else:
            seed_base = round_seed + int(decision["index"]) * 1009 + 17000019
            try:
              worlds = _rollout_states_for_decision(
                  decision,
                  seed_base=seed_base + 70000001,
                  args=args,
              )
              records = [
                  _evaluate_action(
                      model,
                      decision,
                      action,
                      worlds,
                      value_scale,
                      device,
                      seed_base=seed_base,
                      args=args,
                  )
                  for action in candidates
              ]
              target, best_action = _target_from_records(
                  records,
                  int(state.num_distinct_actions()),
                  args,
              )
              accepted, reason = _acceptance(
                  decision,
                  records,
                  int(best_action),
                  args,
              )
            except Exception as exc:
              reject_reasons[f"selection_error:{type(exc).__name__}"] += 1
              accepted = False
              reason = "selection_error"
              target = None
              best_action = int(base_action)
              records = []
            if accepted and target is not None:
              audit = _audit_passes(
                  model,
                  decision,
                  int(best_action),
                  value_scale,
                  device,
                  seed_base,
                  args,
              )
              summary["selection_positive"] += 1
              summary["audit_passes_run"] += int(audit["completed_passes"])
              if bool(audit["all_passed"]):
                if int(args.max_positive_rows) <= 0 or len(positive_rows) < int(
                    args.max_positive_rows
                ):
                  audited_target = _hard_audited_target(target, int(best_action))
                  positive_rows.append(_make_replay_row(
                      decision,
                      records,
                      audited_target,
                      int(best_action),
                      args,
                      int(args.players),
                      value_scale,
                  ))
                  metadata = _trajectory_metadata(
                      args=args,
                      decision=decision,
                      match_index=match_index,
                      round_index=round_index,
                      round_seed=round_seed,
                      start_player=start_player,
                      records=records,
                      best_action=int(best_action),
                      audit=audit,
                      corrections_before=corrections,
                      public_play_trace_before=public_play_trace,
                      target_kind="audited_correction",
                  )
                  positive_metadata.append(metadata)
                  applied_action = int(best_action)
                  corrections.append({
                      "decision_index": int(decision["index"]),
                      "trick_number": int(decision["trick_number"]),
                      "player": int(player),
                      "base_action": int(base_action),
                      "applied_action": int(best_action),
                      "audit_min_improvement": float(audit["min_improvement"]),
                  })
                  summary["audited_corrections"] += 1
                  summary[f"corrections_trick_{decision['trick_number']}"] += 1
              else:
                summary["audit_rejected"] += 1
                if not bool(audit["error_free"]):
                  reject_reasons["audit_infrastructure_error"] += 1
                  for audit_pass in audit.get("passes", []):
                    if bool(audit_pass.get("error", False)):
                      audit_error_reasons[
                          str(audit_pass.get("reason", "unknown"))
                      ] += 1
                elif int(args.max_base_keep_rows) <= 0 or len(base_rows) < int(
                    args.max_base_keep_rows
                ):
                  base_target, keep_action = _base_keep_target(
                      decision, int(state.num_distinct_actions())
                  )
                  base_rows.append(_make_replay_row(
                      decision,
                      records,
                      base_target,
                      int(keep_action),
                      args,
                      int(args.players),
                      value_scale,
                  ))
                  base_metadata.append(_trajectory_metadata(
                      args=args,
                      decision=decision,
                      match_index=match_index,
                      round_index=round_index,
                      round_seed=round_seed,
                      start_player=start_player,
                      records=records,
                      best_action=int(best_action),
                      audit=audit,
                      corrections_before=corrections,
                      public_play_trace_before=public_play_trace,
                      target_kind="audit_rejected_base_keep",
                  ))
            else:
              reject_reasons[str(reason)] += 1
        if str(decision["phase"]) == "play":
          event = _public_play_event(
              state_before,
              player,
              int(applied_action),
              decision_index=int(decision["index"]),
              base_action=int(base_action),
              correction_applied=bool(int(applied_action) != int(base_action)),
          )
          if event is not None:
            event["sequence_index"] = int(len(public_play_trace))
            public_play_trace.append(event)
        state.apply_action(int(applied_action))

      result = terminal_round_result(state)
      returns = np.asarray(state.returns(), dtype=np.float32)
      totals = returns if bool(args.match_context) else totals + returns
      summary["rounds"] += 1
      summary["paradox_rounds"] += int(bool(result["any_paradox"]))
      summary["rounds_with_audited_correction"] += int(bool(corrections))
      summary["applied_audited_corrections"] += int(len(corrections))
      if int(args.progress_interval) > 0 and (
          summary["decisions_evaluated"] % int(args.progress_interval) == 0
      ):
        print(json.dumps({
            "iteration": "chained_best_of_rollouts_progress",
            "matches": int(match_index + 1),
            "rounds": int(summary["rounds"]),
            "decisions_evaluated": int(summary["decisions_evaluated"]),
            "positive_rows": int(len(positive_rows)),
            "base_keep_rows": int(len(base_rows)),
            "applied_corrections": int(summary["applied_audited_corrections"]),
        }), flush=True)

  out, metadata_out, summary_out, base_out, base_metadata_out = _output_paths(args)
  top = {
      "source": "quantum_cat_chained_best_of_rollouts_replay",
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "matches_requested": int(args.matches),
      "seed": int(args.seed),
      "match_context": bool(args.match_context),
      "bot_mode": source_bot_mode,
      "effective_bot_mode": str(args.bot_mode),
      "source_group_id": str(args.source_group_id),
      "rollout_state_source": "infostate",
      "information_safe_rollout_teacher": True,
      "target_mode": "hard_audited_action",
      "collection_contract_version": _COLLECTION_CONTRACT_VERSION,
      "collector_config": {
          "decision_phases": str(args.decision_phases),
          "min_legal_count": int(args.min_legal_count),
          "max_legal_count": int(args.max_legal_count),
          "min_trick_number": int(args.min_trick_number),
          "max_trick_number": int(args.max_trick_number),
          "led_contrast_only": bool(args.led_contrast_only),
          "require_red_switch_legal": bool(args.require_red_switch_legal),
          "max_candidate_actions": int(args.max_candidate_actions),
          "policy_top_actions": int(args.policy_top_actions),
          "include_all_legal": bool(args.include_all_legal),
          "require_policy_target_changed": bool(
              args.require_policy_target_changed
          ),
          "require_survival_contrast": bool(args.require_survival_contrast),
          "min_any_paradox_rate_spread": float(
              args.min_any_paradox_rate_spread
          ),
          "require_forced_paradox_contrast": bool(
              args.require_forced_paradox_contrast
          ),
          "min_forced_paradox_rate_spread": float(
              args.min_forced_paradox_rate_spread
          ),
          "target_temperature": float(args.target_temperature),
          "target_top_score_gap": float(args.target_top_score_gap),
          "max_evaluated_decisions": int(args.max_evaluated_decisions),
          "max_evaluations_per_round": int(args.max_evaluations_per_round),
      },
      "belief_samples": int(args.belief_samples),
      "rollouts_per_action": int(args.rollouts_per_action),
      "audit_belief_samples": int(args.audit_belief_samples),
      "audit_rollouts_per_action": int(args.audit_rollouts_per_action),
      "independent_audit_passes": int(args.audit_passes),
      "min_best_vs_base_any_paradox_improvement": float(
          args.min_best_vs_base_any_paradox_improvement
      ),
      "min_audit_best_vs_base_any_paradox_improvement": float(
          args.min_audit_best_vs_base_any_paradox_improvement
      ),
      "max_audit_paired_binomial_p": float(
          args.max_audit_paired_binomial_p
      ),
      "summary": {key: int(value) for key, value in sorted(summary.items())},
      "reject_reasons": {
          key: int(value) for key, value in sorted(reject_reasons.items())
      },
      "audit_error_reasons": {
          key: int(value) for key, value in sorted(audit_error_reasons.items())
      },
  }
  _write_bank(out, metadata_out, positive_rows, positive_metadata, {
      **top,
      "target_kind": "audited_correction",
  })
  if base_out is not None and base_metadata_out is not None:
    _write_bank(base_out, base_metadata_out, base_rows, base_metadata, {
        **top,
        "target_kind": "audit_rejected_base_keep",
    })
  summary_out.parent.mkdir(parents=True, exist_ok=True)
  summary_out.write_text(json.dumps({
      **top,
      "positive_rows": int(len(positive_rows)),
      "base_keep_rows": int(len(base_rows)),
      "out": str(out),
      "metadata_out": str(metadata_out),
      "base_keep_out": "" if base_out is None else str(base_out),
      "base_keep_metadata_out": (
          "" if base_metadata_out is None else str(base_metadata_out)
      ),
  }, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
      "iteration": "chained_best_of_rollouts_complete",
      "positive_rows": int(len(positive_rows)),
      "base_keep_rows": int(len(base_rows)),
      "summary": str(summary_out),
  }, sort_keys=True))


if __name__ == "__main__":
  main()
