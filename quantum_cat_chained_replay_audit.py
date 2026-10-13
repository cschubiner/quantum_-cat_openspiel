#!/usr/bin/env python3
"""Fail-closed audit for chained best-of-rollouts replay shards.

The chained collector emits two complementary banks per independently seeded
source shard:

* corrections whose proposed action passed every fresh audit pass; and
* hard base-keeps whose proposal passed selection but failed at least one
  error-free fresh audit pass.

This auditor checks the replay tensors and provenance together.  It is stricter
than a row-count report: target semantics, paired improvements, seed
disjointness, public-history schema, and correction ancestry must all agree.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quantum_cat_alphazero_torch import load_replay
from quantum_cat_best_of_rollouts_replay import _paired_any_paradox_summary
import quantum_cat_public_history as public_history
from quantum_cat_public_history import PUBLIC_HISTORY_SCHEMA, PUBLIC_HISTORY_VERSION


_KINDS = ("audited_correction", "audit_rejected_base_keep")
_MAX_COLLECTOR_SEED = 2**32 - 2
_COLLECTION_CONTRACT_VERSION = "chained_best_of_rollouts_v3"
_COLOR_NAMES = ("R", "B", "Y", "G")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_sidecar(path: Path) -> tuple[dict, list[dict]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"sidecar must be a JSON object: {path}")
  rows = payload.get("rows")
  if not isinstance(rows, list):
    raise ValueError(f"sidecar rows must be a list: {path}")
  return payload, rows


def _finite_number(value) -> bool:
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


def _record_map(records) -> dict[int, dict]:
  result = {}
  for record in records if isinstance(records, list) else []:
    if not isinstance(record, dict) or record.get("action") is None:
      continue
    result[int(record["action"])] = record
  return result


def _error(errors: list[str], label: str, message: str) -> None:
  errors.append(f"{label}: {message}")


def _validate_public_trace(row: dict, errors: list[str], label: str) -> None:
  """Recompute the deployable sequence features from public play events."""
  if str(row.get("public_history_trace_contract")) != _COLLECTION_CONTRACT_VERSION:
    _error(errors, label, "missing v3 public-history trace contract")
    return
  trace = row.get("public_play_trace_before")
  if not isinstance(trace, list):
    _error(errors, label, "public_play_trace_before is not a list")
    return
  if int(row.get("public_play_trace_length", -1)) != len(trace):
    _error(errors, label, "public play trace length mismatch")
  players = int(row.get("num_players", -1))
  num_tricks = int(row.get("num_tricks", -1))
  num_ranks = int(row.get("num_card_types", -1))
  decision = row.get("decision", {})
  trick_number = int(decision.get("trick_number", -1))
  if players <= 0 or num_tricks <= 0 or num_ranks <= 0:
    _error(errors, label, "invalid public trace game dimensions")
    return
  if not 0 <= trick_number < num_tricks:
    _error(errors, label, "decision trick is outside public trace dimensions")
    return

  groups = defaultdict(list)
  for index, event in enumerate(trace):
    event_label = f"{label}/trace={index}"
    if not isinstance(event, dict):
      _error(errors, event_label, "public play event is not an object")
      continue
    if int(event.get("sequence_index", -1)) != index:
      _error(errors, event_label, "public play sequence index mismatch")
    event_trick = int(event.get("trick_number", -1))
    player = int(event.get("player", -1))
    rank = int(event.get("rank", -1))
    color = str(event.get("color", ""))
    action = int(event.get("action", -1))
    if not 0 <= event_trick <= trick_number:
      _error(errors, event_label, "public play belongs to a future/invalid trick")
    if not 0 <= player < players:
      _error(errors, event_label, "public play has invalid player")
    if not 1 <= rank <= num_ranks or color not in _COLOR_NAMES:
      _error(errors, event_label, "public play has invalid public card")
    else:
      expected_action = _COLOR_NAMES.index(color) * num_ranks + rank - 1
      if action != expected_action:
        _error(errors, event_label, "public card does not match action id")
    if bool(event.get("correction_applied")) != (
        action != int(event.get("base_action", action))
    ):
      _error(errors, event_label, "correction flag disagrees with applied/base action")
    groups[event_trick].append(event)

  completed = []
  start_player = int(row.get("start_player", -1))
  if not 0 <= start_player < players:
    _error(errors, label, "invalid round start player")
    return
  current_start = start_player
  for completed_trick in range(trick_number):
    events = groups.get(completed_trick, [])
    if len(events) != players:
      _error(errors, label, f"completed public trick {completed_trick} is incomplete")
      return
    if any(int(event.get("trick_start_player", -1)) != current_start for event in events):
      _error(errors, label, f"completed public trick {completed_trick} leader mismatch")
    public_cards = [
        (int(event["player"]), (int(event["rank"]), str(event["color"])))
        for event in events
    ]
    ordered = public_history._ordered_trick_events(
        public_cards, current_start, players
    )
    if [int(item["player"]) for item in ordered] != [
        int(event["player"]) for event in events
    ]:
      _error(errors, label, f"completed public trick {completed_trick} order mismatch")
    completed.append(public_cards)
    current_start = int(public_history._trick_winner(ordered))

  current_events = groups.get(trick_number, [])
  if len(current_events) >= players:
    _error(errors, label, "current public trick is already complete")
  if any(int(event.get("trick_start_player", -1)) != current_start for event in current_events):
    _error(errors, label, "current public trick leader mismatch")
  if current_events and int(current_events[0]["player"]) != current_start:
    _error(errors, label, "current public trick does not begin with its leader")
  current_cards = [None] * players
  for event in current_events:
    player = int(event["player"])
    if current_cards[player] is not None:
      _error(errors, label, "player appears twice in current public trick")
    current_cards[player] = (int(event["rank"]), str(event["color"]))

  state = SimpleNamespace(
      _num_players=players,
      _num_tricks=num_tricks,
      _trick_number=trick_number,
      _round_start_player=start_player,
      _start_player=current_start,
      _completed_tricks=completed,
      _cards_played_this_trick=current_cards,
  )
  expected = public_history.public_history_features(
      state, int(decision.get("player", -1))
  )
  stored = np.asarray(row.get("public_history", {}).get("features", []))
  if stored.shape == expected.shape and not np.allclose(
      stored, expected, atol=1e-7, rtol=0.0
  ):
    _error(errors, label, "public_history features disagree with public play trace")

  prior = row.get("prior_audited_corrections", [])
  trace_corrections = [event for event in trace if bool(event.get("correction_applied"))]
  if len(trace_corrections) != len(prior):
    _error(errors, label, "public trace correction count disagrees with ancestry")
  for event, correction in zip(trace_corrections, prior):
    for key in (
        "decision_index",
        "trick_number",
        "player",
        "base_action",
        "applied_action",
    ):
      event_key = "action" if key == "applied_action" else key
      if int(event.get(event_key, -1)) != int(correction.get(key, -1)):
        _error(errors, label, "public trace correction disagrees with ancestry")
        break


def _audit_row(
    example,
    row: dict,
    *,
    kind: str,
    source_group: str,
    required_passes: int,
    min_selection_improvement: float,
    min_audit_improvement: float,
    label: str,
    max_audit_paired_binomial_p: float = 1.0,
) -> tuple[list[str], dict]:
  errors: list[str] = []
  decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
  base_action = int(decision.get("policy_action", -1))
  proposed_action = int(row.get("proposed_best_action", -1))
  applied_action = int(row.get("applied_action", -1))
  best_action = int(row.get("best_action", -1))
  selected_action = int(example[5])
  selected_player = int(example[6])
  target = np.asarray(example[2], dtype=np.float64)
  legal_mask = np.asarray(example[1], dtype=np.float64)

  if str(row.get("target_kind")) != kind:
    _error(errors, label, f"target_kind {row.get('target_kind')!r} != {kind!r}")
  if str(row.get("source_group_id")) != source_group:
    _error(errors, label, "row source_group_id does not match shard")
  expected_trajectory = (
      f"{source_group}:match={int(row.get('match_index', -1))}:"
      f"round={int(row.get('round_index', -1))}"
  )
  if str(row.get("trajectory_id")) != expected_trajectory:
    _error(errors, label, "trajectory_id does not match source/match/round")
  if selected_player != int(decision.get("player", -1)):
    _error(errors, label, "replay player does not match decision player")

  legal_actions = [int(action) for action in decision.get("legal_actions", [])]
  expected_mask = np.zeros_like(legal_mask)
  if any(action < 0 or action >= len(expected_mask) for action in legal_actions):
    _error(errors, label, "decision contains an out-of-range legal action")
  else:
    expected_mask[legal_actions] = 1.0
    if not np.array_equal(legal_mask, expected_mask):
      _error(errors, label, "replay legal mask does not match decision legal actions")
  if int(decision.get("legal_count", -1)) != len(legal_actions):
    _error(errors, label, "decision legal_count does not match legal_actions")
  if base_action not in legal_actions or proposed_action not in legal_actions:
    _error(errors, label, "base or proposed action is not legal")
  if proposed_action == base_action:
    _error(errors, label, "proposal does not change the source action")

  if target.ndim != 1 or target.shape != legal_mask.shape:
    _error(errors, label, "policy target shape does not match legal mask")
  elif (
      not np.all(np.isfinite(target))
      or np.any(target < -1e-7)
      or not np.isclose(float(target.sum()), 1.0, atol=1e-6)
      or np.any(target[legal_mask <= 0.0] > 1e-7)
  ):
    _error(errors, label, "policy target is not a finite legal distribution")

  history = row.get("public_history")
  if not isinstance(history, dict):
    _error(errors, label, "missing public_history object")
  else:
    features = np.asarray(history.get("features", []), dtype=np.float64)
    if str(history.get("version")) != PUBLIC_HISTORY_VERSION:
      _error(errors, label, "public_history version mismatch")
    if list(history.get("schema", [])) != list(PUBLIC_HISTORY_SCHEMA):
      _error(errors, label, "public_history schema mismatch")
    if features.shape != (len(PUBLIC_HISTORY_SCHEMA),):
      _error(errors, label, "public_history feature width mismatch")
    elif not np.all(np.isfinite(features)) or np.any(features < 0.0) or np.any(
        features > 1.0
    ):
      _error(errors, label, "public_history features are not finite rates")
  _validate_public_trace(row, errors, label)

  selection_records = _record_map(row.get("rollout_records"))
  base_record = selection_records.get(base_action)
  proposed_record = selection_records.get(proposed_action)
  if base_record is None or proposed_record is None:
    _error(errors, label, "selection records omit base or proposed action")
    computed_selection = float("nan")
  else:
    computed_selection = float(base_record.get("any_paradox_rate", float("nan"))) - float(
        proposed_record.get("any_paradox_rate", float("nan"))
    )
    explicit_selection = row.get(
        "selection_best_vs_base_any_paradox_improvement"
    )
    if not _finite_number(computed_selection) or not np.isclose(
        computed_selection, float(explicit_selection), atol=1e-9
    ):
      _error(errors, label, "selection improvement disagrees with rollout records")
    if not _finite_number(computed_selection) or computed_selection + 1e-9 < float(
        min_selection_improvement
    ):
      _error(errors, label, "proposal is below the selection improvement gate")

  passes = row.get("audit_passes")
  if not isinstance(passes, list):
    passes = []
    _error(errors, label, "audit_passes is not a list")
  if len(passes) != int(required_passes):
    _error(errors, label, "audit pass count does not match required count")
  if [int(item.get("pass_index", -1)) for item in passes] != list(
      range(len(passes))
  ):
    _error(errors, label, "audit pass indexes are not contiguous")
  world_seeds = [int(item.get("world_seed", -1)) for item in passes]
  continuation_seeds = [int(item.get("continuation_seed", -1)) for item in passes]
  all_seeds = [*world_seeds, *continuation_seeds]
  if any(seed < 0 or seed > _MAX_COLLECTOR_SEED for seed in all_seeds):
    _error(errors, label, "audit seed is outside the collector's NumPy range")
  if len(set(all_seeds)) != len(all_seeds):
    _error(errors, label, "audit world/continuation seeds are not disjoint")

  pass_improvements = []
  pass_results = []
  paired_summaries = []
  for pass_index, audit_pass in enumerate(passes):
    pass_label = f"{label}/audit={pass_index}"
    if bool(audit_pass.get("error", False)):
      _error(errors, pass_label, "stored row contains an audit infrastructure error")
    audit_records = _record_map(audit_pass.get("records"))
    if set(audit_records) != {base_action, proposed_action}:
      _error(errors, pass_label, "audit records are not exactly base and proposal")
      improvement = float("nan")
      paired = None
    else:
      base_rate = float(audit_records[base_action].get("any_paradox_rate", float("nan")))
      proposal_rate = float(
          audit_records[proposed_action].get("any_paradox_rate", float("nan"))
      )
      improvement = base_rate - proposal_rate
      paired = _paired_any_paradox_summary(
          audit_records[base_action], audit_records[proposed_action]
      )
      for action, record in audit_records.items():
        rates = (
            record.get("any_paradox_rate"),
            record.get("actor_paradox_rate"),
            record.get("forced_paradox_rate"),
        )
        if any(not _finite_number(rate) or not 0.0 <= float(rate) <= 1.0 for rate in rates):
          _error(errors, pass_label, f"action {action} has an invalid paradox rate")
        worlds = int(record.get("rollout_worlds", -1))
        per_world = int(record.get("rollouts_per_world", -1))
        if int(record.get("rollouts", -1)) != worlds * per_world or worlds <= 0 or per_world <= 0:
          _error(errors, pass_label, f"action {action} has inconsistent rollout counts")
    stored_paired = audit_pass.get("paired_any_paradox")
    if stored_paired is not None:
      if paired is None:
        _error(errors, pass_label, "stored paired evidence cannot be reconstructed")
      else:
        if str(stored_paired.get("inference_unit")) != str(
            paired["inference_unit"]
        ):
          _error(errors, pass_label, "paired evidence inference unit mismatch")
        for key in (
            "paired_worlds",
            "paired_outcomes",
            "continuation_outcomes",
            "rollouts_per_world",
            "saved_outcomes",
            "harmed_outcomes",
            "tied_outcomes",
            "net_saved_outcomes",
        ):
          if int(stored_paired.get(key, -1)) != int(paired[key]):
            _error(errors, pass_label, f"paired evidence {key} mismatch")
        if not np.isclose(
            float(stored_paired.get("one_sided_binomial_p", float("nan"))),
            float(paired["one_sided_binomial_p"]),
            atol=1e-12,
        ):
          _error(errors, pass_label, "paired evidence p-value mismatch")
    paired_evidence_passed = bool(
        float(max_audit_paired_binomial_p) >= 1.0
        or (
            paired is not None
            and float(paired["one_sided_binomial_p"])
            <= float(max_audit_paired_binomial_p) + 1e-12
        )
    )
    explicit = audit_pass.get("improvement")
    if not _finite_number(improvement) or not _finite_number(explicit) or not np.isclose(
        improvement, float(explicit), atol=1e-9
    ):
      _error(errors, pass_label, "improvement disagrees with audit records")
    expected_passed = bool(
        _finite_number(improvement)
        and improvement + 1e-9 >= float(min_audit_improvement)
        and paired_evidence_passed
    )
    if bool(audit_pass.get("passed")) != expected_passed:
      _error(errors, pass_label, "passed flag disagrees with audit threshold")
    if expected_passed and str(audit_pass.get("reason")) != "accepted":
      _error(errors, pass_label, "passing audit does not say accepted")
    if not expected_passed:
      reason = str(audit_pass.get("reason", ""))
      improvement_failed = bool(
          not _finite_number(improvement)
          or improvement + 1e-9 < float(min_audit_improvement)
      )
      if improvement_failed and not reason.startswith(
          "weak_audit_best_vs_base_any_paradox_improvement"
      ):
        _error(errors, pass_label, "failed improvement audit has unexpected reason")
      if not improvement_failed and not paired_evidence_passed and reason not in (
          "missing_audit_paired_outcomes",
          "weak_audit_paired_evidence",
      ):
        _error(errors, pass_label, "failed paired audit has unexpected reason")
    pass_improvements.append(float(improvement))
    pass_results.append(expected_passed)
    paired_summaries.append(paired)

  expected_all_passed = bool(
      len(pass_results) == int(required_passes) and all(pass_results)
  )
  if bool(row.get("all_audit_passes_passed")) != expected_all_passed:
    _error(errors, label, "all_audit_passes_passed is inconsistent")
  explicit_min = row.get("audit_min_best_vs_base_any_paradox_improvement")
  computed_min = min(pass_improvements, default=float("nan"))
  if not _finite_number(computed_min) or not _finite_number(explicit_min) or not np.isclose(
      computed_min, float(explicit_min), atol=1e-9
  ):
    _error(errors, label, "minimum audit improvement is inconsistent")

  if kind == "audited_correction":
    if not expected_all_passed:
      _error(errors, label, "correction did not pass every audit")
    if not bool(row.get("policy_target_changed")):
      _error(errors, label, "correction is not marked as a target change")
    if bool(row.get("audit_override_rejected")) or bool(
        row.get("audit_base_keep_validation")
    ):
      _error(errors, label, "correction is marked as an audit rejection")
    if not (selected_action == best_action == applied_action == proposed_action):
      _error(errors, label, "correction replay action/proposal/application disagree")
    if target.ndim == 1 and (
        int(np.argmax(target)) != proposed_action
        or not np.isclose(float(target[proposed_action]), 1.0, atol=1e-7)
    ):
      _error(errors, label, "correction target is not hard one-hot on audited action")
  else:
    if expected_all_passed or all(pass_results):
      _error(errors, label, "base-keep lacks a failed fresh audit pass")
    if bool(row.get("policy_target_changed")):
      _error(errors, label, "base-keep is marked as a target change")
    if not bool(row.get("audit_override_rejected")) or not bool(
        row.get("audit_base_keep_validation")
    ):
      _error(errors, label, "base-keep lacks audit-rejection provenance")
    if not (selected_action == best_action == applied_action == base_action):
      _error(errors, label, "base-keep replay action is not the source action")
    if target.ndim == 1 and (
        int(np.argmax(target)) != base_action
        or not np.isclose(float(target[base_action]), 1.0, atol=1e-7)
    ):
      _error(errors, label, "base-keep target is not hard one-hot on source action")

  prior = row.get("prior_audited_corrections")
  if not isinstance(prior, list):
    prior = []
    _error(errors, label, "prior_audited_corrections is not a list")
  if int(row.get("corrections_before", -1)) != len(prior):
    _error(errors, label, "corrections_before does not match correction ancestry")
  decision_index = int(decision.get("index", -1))
  if any(int(item.get("decision_index", -1)) >= decision_index for item in prior):
    _error(errors, label, "correction ancestry contains a non-prior decision")

  return errors, {
      "source_group_id": source_group,
      "trajectory_id": str(row.get("trajectory_id")),
      "decision_index": decision_index,
      "trick_number": int(decision.get("trick_number", -1)),
      "kind": kind,
      "base_action": base_action,
      "proposed_action": proposed_action,
      "applied_action": applied_action,
      "selection_improvement": float(computed_selection),
      "audit_improvements": pass_improvements,
      "audit_pass_results": pass_results,
      "audit_paired_any_paradox": paired_summaries,
      "audit_seed_pairs": list(zip(world_seeds, continuation_seeds)),
      "corrections_before": int(row.get("corrections_before", -1)),
      "prior_audited_corrections": prior,
  }


def _audit_bank(
    replay_path: Path,
    sidecar_path: Path,
    *,
    kind: str,
    min_selection_improvement: float,
    min_audit_improvement: float,
    min_audit_passes: int,
    max_audit_paired_binomial_p: float,
) -> tuple[dict, list[dict]]:
  errors: list[str] = []
  sidecar, rows = _load_sidecar(sidecar_path)
  replay_present = replay_path.exists()
  if replay_present:
    replay, embedded = load_replay(replay_path, return_metadata=True)
  else:
    replay = []
    embedded = {
        **sidecar,
        "rows": 0,
        "row_group_ids": [],
    }
    if rows:
      errors.append(
          f"{replay_path}: replay is missing despite non-empty sidecar rows"
      )
  label = str(replay_path)
  source_group = str(sidecar.get("source_group_id", ""))
  required_passes = int(sidecar.get("independent_audit_passes", -1))
  collector_max_p = float(sidecar.get("max_audit_paired_binomial_p", 1.0))
  if not source_group:
    _error(errors, label, "empty source_group_id")
  if str(sidecar.get("collection_contract_version")) != _COLLECTION_CONTRACT_VERSION:
    _error(errors, label, "sidecar is not a v2 chained collection")
  if replay_present and str(
      embedded.get("collection_contract_version")
  ) != _COLLECTION_CONTRACT_VERSION:
    _error(errors, label, "embedded replay is not a v2 chained collection")
  collector_config = sidecar.get("collector_config")
  if not isinstance(collector_config, dict):
    _error(errors, label, "missing collector_config provenance")
  else:
    if str(collector_config.get("decision_phases")) != "play":
      _error(errors, label, "collector did not restrict labels to play decisions")
    if not bool(collector_config.get("require_policy_target_changed")):
      _error(errors, label, "collector did not require a policy target change")
    if not bool(collector_config.get("require_survival_contrast")):
      _error(errors, label, "collector did not require survival contrast")
    if float(collector_config.get("min_any_paradox_rate_spread", -math.inf)) < float(
        min_selection_improvement
    ):
      _error(errors, label, "collector survival contrast is below selection gate")
  if str(sidecar.get("target_kind")) != kind:
    _error(errors, label, "sidecar target_kind mismatch")
  if replay_present and str(embedded.get("target_kind")) != kind:
    _error(errors, label, "embedded target_kind mismatch")
  if replay_present and str(embedded.get("source_group_id")) != source_group:
    _error(errors, label, "embedded source_group_id mismatch")
  if replay_present and int(embedded.get("rows", -1)) != len(replay):
    _error(errors, label, "embedded row count mismatch")
  if len(rows) != len(replay):
    _error(errors, label, "sidecar row count mismatch")
  groups = [str(item) for item in embedded.get("row_group_ids", [])]
  if groups != [source_group] * len(replay):
    _error(errors, label, "embedded row_group_ids are inconsistent")
  if required_passes < int(min_audit_passes):
    _error(errors, label, "shard requires too few independent audit passes")
  if float(sidecar.get("min_best_vs_base_any_paradox_improvement", -math.inf)) < float(
      min_selection_improvement
  ):
    _error(errors, label, "collector selection threshold is below audit threshold")
  if float(
      sidecar.get("min_audit_best_vs_base_any_paradox_improvement", -math.inf)
  ) < float(min_audit_improvement):
    _error(errors, label, "collector fresh-audit threshold is below audit threshold")
  if collector_max_p > float(max_audit_paired_binomial_p) + 1e-12:
    _error(errors, label, "collector paired-evidence p threshold is too weak")
  for key in (
      "information_safe_rollout_teacher",
      "rollout_state_source",
      "seed",
      "players",
      "summary",
  ):
    if replay_present and embedded.get(key) != sidecar.get(key):
      _error(errors, label, f"embedded/sidecar {key} mismatch")
  if not bool(sidecar.get("information_safe_rollout_teacher")) or str(
      sidecar.get("rollout_state_source")
  ) != "infostate":
    _error(errors, label, "shard is not an information-state rollout teacher")
  if str(sidecar.get("target_mode")) != "hard_audited_action" or (
      replay_present
      and str(embedded.get("target_mode")) != "hard_audited_action"
  ):
    _error(errors, label, "shard does not hard-target only the audited action")

  audited_rows = []
  for index, (example, row) in enumerate(zip(replay, rows)):
    row_errors, audited = _audit_row(
        example,
        row,
        kind=kind,
        source_group=source_group,
        required_passes=required_passes,
        min_selection_improvement=min_selection_improvement,
        min_audit_improvement=min_audit_improvement,
        max_audit_paired_binomial_p=collector_max_p,
        label=f"{label}/row={index}",
    )
    errors.extend(row_errors)
    audited_rows.append(audited)

  return {
      "path": str(replay_path),
      "sidecar": str(sidecar_path),
      "replay_present": bool(replay_present),
      "sha256": _sha256(replay_path) if replay_present else None,
      "kind": kind,
      "rows": len(replay),
      "source_group_id": source_group,
      "seed": int(sidecar.get("seed", -1)),
      "players": int(sidecar.get("players", -1)),
      "required_audit_passes": required_passes,
      "max_audit_paired_binomial_p": collector_max_p,
      "decisions_evaluated": int(
          sidecar.get("summary", {}).get("decisions_evaluated", 0)
      ),
      "selection_positive": int(
          sidecar.get("summary", {}).get("selection_positive", 0)
      ),
      "applied_audited_corrections": int(
          sidecar.get("summary", {}).get("applied_audited_corrections", 0)
      ),
      "evaluated_after_correction": sum(
          int(value)
          for key, value in sidecar.get("summary", {}).items()
          if key.startswith("decisions_after_")
          and key != "decisions_after_0_corrections"
      ),
      "passed": not errors,
      "errors": errors,
  }, audited_rows


def audit_chained_banks(
    positive_pairs: list[tuple[Path, Path]],
    base_keep_pairs: list[tuple[Path, Path]],
    *,
    min_selection_improvement: float = 0.5,
    min_audit_improvement: float = 0.25,
    min_audit_passes: int = 3,
    max_audit_paired_binomial_p: float = 1.0,
) -> dict:
  errors: list[str] = []
  reports = []
  rows = []
  for kind, pairs in (
      ("audited_correction", positive_pairs),
      ("audit_rejected_base_keep", base_keep_pairs),
  ):
    for replay_path, sidecar_path in pairs:
      report, audited = _audit_bank(
          Path(replay_path),
          Path(sidecar_path),
          kind=kind,
          min_selection_improvement=min_selection_improvement,
          min_audit_improvement=min_audit_improvement,
          min_audit_passes=min_audit_passes,
          max_audit_paired_binomial_p=max_audit_paired_binomial_p,
      )
      reports.append(report)
      rows.extend(audited)
      errors.extend(report["errors"])

  by_kind = defaultdict(list)
  for report in reports:
    by_kind[report["kind"]].append(report)
  for kind, kind_reports in by_kind.items():
    groups = [report["source_group_id"] for report in kind_reports]
    seeds = [report["seed"] for report in kind_reports]
    if len(groups) != len(set(groups)):
      errors.append(f"{kind}: duplicate source_group_id across shards")
    if len(seeds) != len(set(seeds)):
      errors.append(f"{kind}: duplicate collector seed across shards")

  positive_by_group = {
      report["source_group_id"]: report
      for report in by_kind["audited_correction"]
  }
  base_by_group = {
      report["source_group_id"]: report
      for report in by_kind["audit_rejected_base_keep"]
  }
  if set(positive_by_group) != set(base_by_group):
    errors.append("positive and base-keep source-group sets differ")
  for group in sorted(set(positive_by_group) & set(base_by_group)):
    positive = positive_by_group[group]
    base = base_by_group[group]
    for key in ("seed", "players", "required_audit_passes", "decisions_evaluated"):
      if positive[key] != base[key]:
        errors.append(f"{group}: positive/base-keep {key} mismatch")

  seed_pair_owners = defaultdict(list)
  seed_value_owners = defaultdict(list)
  for row in rows:
    for world_seed, continuation_seed in row["audit_seed_pairs"]:
      owner = (row["source_group_id"], row["trajectory_id"], row["decision_index"])
      seed_pair_owners[(world_seed, continuation_seed)].append(
          owner
      )
      seed_value_owners[world_seed].append(("world", *owner))
      seed_value_owners[continuation_seed].append(("continuation", *owner))
  duplicate_seed_pairs = {
      str(pair): owners
      for pair, owners in seed_pair_owners.items()
      if len(owners) > 1
  }
  if duplicate_seed_pairs:
    errors.append("audit seed pair reused across decisions")
  duplicate_seed_values = {
      str(seed): owners
      for seed, owners in seed_value_owners.items()
      if len(owners) > 1
  }
  if duplicate_seed_values:
    errors.append("audit world/continuation seed reused across decisions")

  rows_by_trajectory = defaultdict(list)
  for row in rows:
    rows_by_trajectory[(row["source_group_id"], row["trajectory_id"])].append(row)
  corrected_followup_rows = 0
  for (group, trajectory), trajectory_rows in rows_by_trajectory.items():
    corrections = {
        row["decision_index"]: row
        for row in trajectory_rows
        if row["kind"] == "audited_correction"
    }
    for row in sorted(trajectory_rows, key=lambda item: item["decision_index"]):
      expected = [
          corrections[index]
          for index in sorted(corrections)
          if index < row["decision_index"]
      ]
      if row["corrections_before"] != len(expected):
        errors.append(
            f"{group}/{trajectory}/decision={row['decision_index']}: "
            "corrections_before omits or invents a prior emitted correction"
        )
      if expected:
        corrected_followup_rows += 1
      prior = row["prior_audited_corrections"]
      for expected_row, item in zip(expected, prior):
        fields = {
            "decision_index": expected_row["decision_index"],
            "trick_number": expected_row["trick_number"],
            "base_action": expected_row["base_action"],
            "applied_action": expected_row["applied_action"],
        }
        if any(int(item.get(key, -1)) != int(value) for key, value in fields.items()):
          errors.append(
              f"{group}/{trajectory}/decision={row['decision_index']}: "
              "prior correction provenance disagrees with emitted correction"
          )

  positive_rows = [row for row in rows if row["kind"] == "audited_correction"]
  base_rows = [row for row in rows if row["kind"] == "audit_rejected_base_keep"]
  trick_counts = Counter(row["trick_number"] for row in positive_rows)
  report = {
      "kind": "chained_best_of_rollouts_replay_audit",
      "passed": not errors,
      "errors": errors,
      "min_selection_improvement": float(min_selection_improvement),
      "min_audit_improvement_per_pass": float(min_audit_improvement),
      "min_independent_audit_passes": int(min_audit_passes),
      "max_audit_paired_binomial_p": float(max_audit_paired_binomial_p),
      "positive_rows": len(positive_rows),
      "base_keep_rows": len(base_rows),
      "source_groups": sorted(set(report["source_group_id"] for report in reports)),
      "source_group_count": len(
          set(report["source_group_id"] for report in reports)
      ),
      "trajectory_count": len(rows_by_trajectory),
      "positive_trick_counts": {
          str(key): int(value) for key, value in sorted(trick_counts.items())
      },
      "positive_selection_improvements": [
          row["selection_improvement"] for row in positive_rows
      ],
      "positive_audit_improvements": [
          row["audit_improvements"] for row in positive_rows
      ],
      "base_keep_failed_pass_counts": [
          sum(not passed for passed in row["audit_pass_results"])
          for row in base_rows
      ],
      "evaluated_decisions": sum(
          report["decisions_evaluated"]
          for report in by_kind["audited_correction"]
      ),
      "evaluated_after_correction": sum(
          report["evaluated_after_correction"]
          for report in by_kind["audited_correction"]
      ),
      "corrected_followup_emitted_rows": int(corrected_followup_rows),
      "duplicate_audit_seed_pairs": duplicate_seed_pairs,
      "duplicate_audit_seed_values": duplicate_seed_values,
      "banks": reports,
  }
  return report


def _pairs(replays: list[str], sidecars: list[str], label: str):
  if len(replays) != len(sidecars):
    raise ValueError(
        f"--{label} and --{label}-sidecar counts differ: "
        f"{len(replays)} != {len(sidecars)}"
    )
  return [(Path(replay), Path(sidecar)) for replay, sidecar in zip(replays, sidecars)]


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--positive", action="append", required=True)
  parser.add_argument("--positive-sidecar", action="append", required=True)
  parser.add_argument("--base-keep", action="append", required=True)
  parser.add_argument("--base-keep-sidecar", action="append", required=True)
  parser.add_argument("--min-selection-improvement", type=float, default=0.5)
  parser.add_argument("--min-audit-improvement", type=float, default=0.25)
  parser.add_argument("--min-audit-passes", type=int, default=3)
  parser.add_argument("--max-audit-paired-binomial-p", type=float, default=1.0)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = audit_chained_banks(
      _pairs(args.positive, args.positive_sidecar, "positive"),
      _pairs(args.base_keep, args.base_keep_sidecar, "base-keep"),
      min_selection_improvement=args.min_selection_improvement,
      min_audit_improvement=args.min_audit_improvement,
      min_audit_passes=args.min_audit_passes,
      max_audit_paired_binomial_p=args.max_audit_paired_binomial_p,
  )
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if report["passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
