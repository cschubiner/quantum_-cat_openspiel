#!/usr/bin/env python3
"""Audit failed cell-exit selector overrides with continuation rollouts.

The cell-exit miner labels one-step table-wide exit damage. Run785/786 showed
that a legal-set selector can confuse harmless alternatives with necessary
corrections. This script reconstructs those exact source states from the
recorded match/round/decision metadata and compares base/selected/target
actions under same-policy continuations with common random seeds.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from open_spiel.python.games import quantum_cat  # pylint: disable=unused-import

from quantum_cat_failure_miner import (
    action_string,
    play_round_with_history,
    rollout_from_action,
)
from quantum_cat_full_match_elo import load_neural


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--selector-audit", required=True)
  parser.add_argument("--metadata", required=True)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--players", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20260971)
  parser.add_argument("--match-context", action="store_true", default=True)
  parser.add_argument("--rollouts-per-action", type=int, default=8)
  parser.add_argument("--audit-seed", type=int, default=20261001)
  parser.add_argument("--row-offset", type=int, default=0)
  parser.add_argument("--max-rows", type=int, default=0)
  parser.add_argument("--progress-interval", type=int, default=0)
  parser.add_argument(
      "--rows",
      choices=("selected", "false_positives"),
      default="selected",
      help="Rows from selector-audit to evaluate.",
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")

  # Source/continuation behavior knobs. Defaults match the Run779/Run785 q-policy
  # branch when explicitly passed by the caller.
  parser.add_argument(
      "--bot-mode",
      choices=("policy", "q_policy", "liveness_policy", "q_liveness_policy"),
      default="policy",
  )
  parser.add_argument("--liveness-shield-phases", default="play")
  parser.add_argument("--liveness-shield-min-open-slot-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-min-public-damage-delta", type=int, default=0)
  parser.add_argument("--liveness-shield-max-policy-log-gap", type=float, default=-1.0)
  parser.add_argument("--action-value-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-paradox-selection-penalty", type=float, default=0.0)
  parser.add_argument(
      "--action-paradox-rerank-mode",
      choices=("additive", "threshold", "relative"),
      default="additive",
  )
  parser.add_argument("--action-paradox-risk-threshold", type=float, default=0.0)
  parser.add_argument("--action-paradox-min-risk-margin", type=float, default=0.0)
  parser.add_argument("--action-paradox-max-policy-log-gap", type=float, default=2.0)
  parser.add_argument("--action-feasibility-selection-weight", type=float, default=0.0)
  parser.add_argument("--action-value-rerank-clip", type=float, default=0.5)
  parser.add_argument("--action-value-rerank-phases", default="")
  parser.add_argument("--action-value-rerank-min-margin", type=float, default=0.0)
  return parser.parse_args()


def _load_metadata(path: str) -> tuple[list[dict], dict[str, int]]:
  data = json.loads(Path(path).read_text())
  rows = data if isinstance(data, list) else data.get("rows", [])
  if not isinstance(rows, list):
    raise ValueError(f"metadata rows not found in {path}")
  source_seeds: dict[str, int] = {}
  if isinstance(data, dict):
    for source in data.get("sources", []):
      if not isinstance(source, dict):
        continue
      name = source.get("name")
      metadata = source.get("metadata")
      if name is not None and isinstance(metadata, dict) and "seed" in metadata:
        source_seeds[str(name)] = int(metadata["seed"])
  return rows, source_seeds


def _source_match_round(meta: dict) -> tuple[int, int, int]:
  group = meta.get("group")
  if not isinstance(group, dict):
    raise ValueError("metadata row is missing group dict")
  return (
      int(group["match_index"]),
      int(group["round_index"]),
      int(group["decision_index"]),
  )


def _game(players: int, start_player: int, match_context: bool):
  return pyspiel.load_game(
      "python_quantum_cat",
      {
          "players": int(players),
          "start_player": int(start_player),
          "match_context": int(bool(match_context)),
      },
  )


def _round_seed(seed: int, match_idx: int, round_idx: int) -> int:
  return int(seed) + int(match_idx) * 1000003 + int(round_idx) * 9973


def _reconstruct_round_decisions(
    cache: dict[tuple[int, int, int], tuple[object, list[dict]]],
    *,
    source_seed: int,
    match_idx: int,
    round_idx: int,
    model,
    value_scale: float,
    device,
    args: argparse.Namespace,
) -> tuple[object, list[dict]]:
  key = (int(source_seed), int(match_idx), int(round_idx))
  if key in cache:
    return cache[key]

  totals = np.zeros(int(args.players), dtype=np.float32)
  terminal = None
  decisions: list[dict] = []
  for current_round in range(int(round_idx) + 1):
    start_player = (int(match_idx) + int(current_round)) % int(args.players)
    game = _game(int(args.players), start_player, bool(args.match_context))
    prior_totals = np.copy(totals)
    terminal, decisions, _ = play_round_with_history(
        model,
        game,
        seed=_round_seed(int(source_seed), int(match_idx), int(current_round)),
        value_scale=value_scale,
        device=device,
        match_totals=prior_totals if bool(args.match_context) else None,
        round_index=int(current_round),
        args=args,
    )
    returns = np.asarray(terminal.returns(), dtype=np.float32)
    totals = returns if bool(args.match_context) else totals + returns

  if terminal is None:
    raise RuntimeError("failed to reconstruct round")
  cache[key] = (terminal, decisions)
  return cache[key]


def _eval_action(
    decision: dict,
    action: int,
    *,
    model,
    value_scale: float,
    device,
    args: argparse.Namespace,
    row_ordinal: int,
) -> dict:
  outcomes = []
  for rollout_idx in range(max(1, int(args.rollouts_per_action))):
    seed = int(args.audit_seed) + int(row_ordinal) * 1000003 + rollout_idx
    result = rollout_from_action(
        model,
        decision["state"],
        int(action),
        value_scale,
        device,
        seed,
        args,
    )
    outcomes.append(result)

  acting_player = int(decision["player"])
  any_paradox = [bool(row["any_paradox"]) for row in outcomes]
  actor_paradox = [bool(row["paradoxed"][acting_player]) for row in outcomes]
  forced = [bool(row.get("forced_paradox", False)) for row in outcomes]
  scores = np.asarray([row["raw_scores"] for row in outcomes], dtype=np.float32)
  continuation_decisions = [
      int(row.get("continuation_decisions", 0)) for row in outcomes
  ]
  return {
      "action": int(action),
      "action_string": action_string(decision["state"], acting_player, int(action)),
      "any_paradox_rate": float(np.mean(any_paradox)),
      "actor_paradox_rate": float(np.mean(actor_paradox)),
      "forced_paradox_rate": float(np.mean(forced)),
      "mean_actor_score": float(np.mean(scores[:, acting_player])),
      "mean_total_score": float(np.mean(np.sum(scores, axis=1))),
      "mean_continuation_decisions": float(np.mean(continuation_decisions)),
      "rollouts": int(len(outcomes)),
      "samples": [
          {
              "any_paradox": bool(row["any_paradox"]),
              "actor_paradox": bool(row["paradoxed"][acting_player]),
              "forced_paradox": bool(row.get("forced_paradox", False)),
              "raw_scores": [float(value) for value in row["raw_scores"]],
              "first_new_paradox_trigger": row.get("first_new_paradox_trigger"),
          }
          for row in outcomes[:4]
      ],
  }


def _rate(counter: Counter, key: str, total: int) -> float:
  if total <= 0:
    return 0.0
  return round(float(counter[key]) / float(total), 6)


def main() -> None:
  args = parse_args()
  device = torch.device(
      "mps"
      if args.device == "mps" or (
          args.device == "auto" and torch.backends.mps.is_available()
      )
      else "cpu"
  )
  model, model_args = load_neural(args.checkpoint, args.players, device, args)
  value_scale = float(model_args.value_scale)
  selector_audit = json.loads(Path(args.selector_audit).read_text())
  metadata_rows, source_seeds = _load_metadata(args.metadata)
  selected_rows = selector_audit.get(str(args.rows), [])
  if not isinstance(selected_rows, list):
    raise ValueError(f"selector audit does not contain list {args.rows!r}")
  offset = max(0, int(args.row_offset))
  if offset:
    selected_rows = selected_rows[offset:]
  if int(args.max_rows) > 0:
    selected_rows = selected_rows[:int(args.max_rows)]

  cache: dict[tuple[int, int, int], tuple[object, list[dict]]] = {}
  audited = []
  summary = Counter()
  mismatches = []
  for ordinal, row in enumerate(selected_rows):
    state_index = int(row["state_index"])
    meta = metadata_rows[state_index]
    match_idx, round_idx, decision_idx = _source_match_round(meta)
    source_name = str(row.get("source_name") or meta.get("source_name") or "")
    source_seed = int(source_seeds.get(source_name, int(args.seed)))
    _, decisions = _reconstruct_round_decisions(
        cache,
        source_seed=source_seed,
        match_idx=match_idx,
        round_idx=round_idx,
        model=model,
        value_scale=value_scale,
        device=device,
        args=args,
    )
    if decision_idx < 0 or decision_idx >= len(decisions):
      raise IndexError(
          f"decision {decision_idx} outside reconstructed decisions "
          f"for match {match_idx} round {round_idx}"
      )
    decision = decisions[decision_idx]
    base_action = int(row["base_action"])
    selected_action = int(row["selected_action"])
    target_action = int(row["target_action"])
    legal = {int(action) for action in decision["legal"]}
    if base_action not in legal or selected_action not in legal:
      mismatches.append({
          "state_index": state_index,
          "base_action": base_action,
          "selected_action": selected_action,
          "legal": sorted(legal),
      })
      continue

    actions = {
        "base": _eval_action(
            decision,
            base_action,
            model=model,
            value_scale=value_scale,
            device=device,
            args=args,
            row_ordinal=ordinal,
        ),
        "selected": _eval_action(
            decision,
            selected_action,
            model=model,
            value_scale=value_scale,
            device=device,
            args=args,
            row_ordinal=ordinal,
        ),
    }
    if target_action in legal and target_action not in (base_action, selected_action):
      actions["target"] = _eval_action(
          decision,
          target_action,
          model=model,
          value_scale=value_scale,
          device=device,
          args=args,
          row_ordinal=ordinal,
      )

    selected_delta = (
        float(actions["base"]["any_paradox_rate"])
        - float(actions["selected"]["any_paradox_rate"])
    )
    actor_delta = (
        float(actions["base"]["actor_paradox_rate"])
        - float(actions["selected"]["actor_paradox_rate"])
    )
    score_delta = (
        float(actions["selected"]["mean_actor_score"])
        - float(actions["base"]["mean_actor_score"])
    )
    supports_override = bool(
        selected_delta > 0.0
        or (selected_delta == 0.0 and score_delta > 0.0)
    )
    rejects_override = bool(
        selected_delta < 0.0
        or (selected_delta == 0.0 and score_delta <= 0.0)
    )
    summary["audited"] += 1
    summary["one_step_positive"] += int(bool(row.get("selected_positive")))
    summary["one_step_false_positive"] += int(not bool(row.get("selected_positive")))
    summary["continuation_supports_override"] += int(supports_override)
    summary["continuation_rejects_override"] += int(rejects_override)
    summary["continuation_selected_safer"] += int(selected_delta > 0.0)
    summary["continuation_selected_riskier"] += int(selected_delta < 0.0)
    summary["continuation_selected_actor_safer"] += int(actor_delta > 0.0)
    if not bool(row.get("selected_positive")):
      summary["false_positive_continuation_supported"] += int(supports_override)
      summary["false_positive_continuation_rejected"] += int(rejects_override)
    if bool(row.get("selected_positive")):
      summary["true_positive_continuation_supported"] += int(supports_override)
      summary["true_positive_continuation_rejected"] += int(rejects_override)

    audited.append({
        "state_index": state_index,
        "source_name": row.get("source_name"),
        "source_seed": int(source_seed),
        "match_index": int(match_idx),
        "round_index": int(round_idx),
        "decision_index": int(decision_idx),
        "trick_number": int(row.get("trick_number", -1)),
        "legal_count": int(row.get("legal_count", len(legal))),
        "one_step_selected_positive": bool(row.get("selected_positive")),
        "one_step_selected_safe": bool(row.get("selected_safe")),
        "base_action": int(base_action),
        "selected_action": int(selected_action),
        "target_action": int(target_action),
        "selected_delta_any_paradox_rate": float(selected_delta),
        "selected_delta_actor_paradox_rate": float(actor_delta),
        "selected_delta_actor_score": float(score_delta),
        "continuation_supports_override": bool(supports_override),
        "continuation_rejects_override": bool(rejects_override),
        "actions": actions,
    })
    if (
        int(args.progress_interval) > 0
        and int(summary["audited"]) % int(args.progress_interval) == 0
    ):
      print(json.dumps({
          "audited": int(summary["audited"]),
          "continuation_supports_override": int(
              summary["continuation_supports_override"]
          ),
          "continuation_rejects_override": int(
              summary["continuation_rejects_override"]
          ),
      }, sort_keys=True), flush=True)

  total = int(summary["audited"])
  summary_json = {
      "selector_audit": str(args.selector_audit),
      "metadata": str(args.metadata),
      "checkpoint": str(args.checkpoint),
      "players": int(args.players),
      "seed": int(args.seed),
      "audit_seed": int(args.audit_seed),
      "rollouts_per_action": int(args.rollouts_per_action),
      "row_offset": int(args.row_offset),
      "max_rows": int(args.max_rows),
      "rows": str(args.rows),
      "bot_mode": str(args.bot_mode),
      "action_value_selection_weight": float(args.action_value_selection_weight),
      "action_feasibility_selection_weight": float(
          args.action_feasibility_selection_weight
      ),
      "action_paradox_rerank_mode": str(args.action_paradox_rerank_mode),
      "action_paradox_min_risk_margin": float(args.action_paradox_min_risk_margin),
      "action_paradox_max_policy_log_gap": float(
          args.action_paradox_max_policy_log_gap
      ),
      "summary": {
          **{key: int(value) for key, value in sorted(summary.items())},
          "continuation_support_rate": _rate(
              summary, "continuation_supports_override", total
          ),
          "continuation_reject_rate": _rate(
              summary, "continuation_rejects_override", total
          ),
      },
      "mismatches": mismatches,
      "audited_rows": audited,
  }
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(summary_json, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary_json["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
