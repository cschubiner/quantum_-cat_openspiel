#!/usr/bin/env python3
"""Convert live root-rollout audit samples into information-safe replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import ACTION_FEATURE_SIZE, save_replay


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _walk_samples(value, location="decision_stats"):
  if not isinstance(value, dict):
    return
  samples = value.get("root_rollout_replay_samples")
  if isinstance(samples, list):
    for index, sample in enumerate(samples):
      yield f"{location}.root_rollout_replay_samples[{index}]", sample
  for key, child in value.items():
    if isinstance(child, dict):
      yield from _walk_samples(child, f"{location}.{key}")


def _validate_accepted_recording(value, location="decision_stats"):
  if not isinstance(value, dict):
    return
  samples = value.get("root_rollout_replay_samples")
  if isinstance(samples, list):
    if "root_rollout_overrides" not in value:
      raise ValueError(f"{location} is missing root-rollout override count")
    expected = int(value.get("root_rollout_overrides") or 0)
    recorded = sum(
        bool(sample.get("audit_accepted"))
        for sample in samples if isinstance(sample, dict)
    )
    if recorded != expected:
      raise ValueError(
          f"{location} recorded {recorded} accepted rows, expected {expected}"
      )
  for key, child in value.items():
    if isinstance(child, dict):
      _validate_accepted_recording(child, f"{location}.{key}")


def _validated_sample(sample: dict) -> dict:
  if int(sample.get("schema_version", -1)) != 1:
    raise ValueError("unsupported root-rollout replay sample schema")
  if sample.get("information_safe") is not True:
    raise ValueError("root-rollout replay sample is not information-safe")
  observation = np.asarray(sample.get("observation"), dtype=np.float32)
  legal = np.asarray(sample.get("legal_actions"), dtype=np.int64)
  legal_features = np.asarray(
      sample.get("legal_action_features"), dtype=np.float32
  )
  if observation.ndim != 1 or observation.size <= 0:
    raise ValueError("invalid root-rollout observation")
  if legal.ndim != 1 or legal.size < 2 or len(set(legal.tolist())) != legal.size:
    raise ValueError("invalid root-rollout legal actions")
  if np.any(legal < 0) or np.any(legal >= 1000):
    raise ValueError("root-rollout legal action is outside 0..999")
  if legal_features.shape != (legal.size, ACTION_FEATURE_SIZE):
    raise ValueError("invalid root-rollout legal action features")
  baseline = int(sample.get("baseline_action", -1))
  proposed = int(sample.get("proposed_action", -1))
  selected = int(sample.get("selected_action", -1))
  if any(action not in set(legal.tolist()) for action in (baseline, proposed, selected)):
    raise ValueError("root-rollout target action is not legal")
  accepted = bool(sample.get("audit_accepted"))
  if accepted != (selected == proposed) or (not accepted and selected != baseline):
    raise ValueError("root-rollout accepted/selected target mismatch")
  audit_contract = sample.get("audit_contract")
  required_passes = (
      int(audit_contract.get("required_passes", 0))
      if isinstance(audit_contract, dict) else 0
  )
  if required_passes < 2:
    raise ValueError("root-rollout replay requires at least two audit passes")
  audit_samples = int(audit_contract.get("samples", 0))
  if audit_samples <= 0:
    raise ValueError("root-rollout replay has invalid audit sample count")
  if audit_contract.get("inference_unit") != "information_state_world":
    raise ValueError("root-rollout replay has invalid audit inference unit")
  min_improvement = float(
      audit_contract.get("min_paradox_improvement", 0.0)
  )
  max_p = float(audit_contract.get("max_paired_binomial_p", 1.0))
  if not 0.0 <= max_p <= 1.0:
    raise ValueError("root-rollout replay has invalid paired p threshold")
  audit_passes = sample.get("audit_passes")
  if not isinstance(audit_passes, list) or not audit_passes:
    raise ValueError("root-rollout sample has no independent audit evidence")
  valid_passes = [item for item in audit_passes if item.get("valid") is True]
  if not valid_passes:
    raise ValueError("root-rollout sample has no valid audit pass")
  if any(int(item.get("worlds", 0)) != audit_samples for item in valid_passes):
    raise ValueError("root-rollout audit pass has wrong world count")
  if accepted and len(valid_passes) != required_passes:
    raise ValueError("accepted root-rollout sample has incomplete audit evidence")
  if accepted and any(
      float(item.get("paradox_improvement", -1.0)) + 1e-12
      < min_improvement
      for item in valid_passes
  ):
    raise ValueError("accepted root-rollout sample misses improvement margin")
  if accepted and any(
      float(item.get("paired_binomial_p", 1.0)) > max_p + 1e-12
      for item in valid_passes
  ):
    raise ValueError("accepted root-rollout sample misses paired p threshold")
  return {
      **sample,
      "_observation": observation,
      "_legal": legal,
      "_legal_features": legal_features,
      "_baseline": baseline,
      "_proposed": proposed,
      "_selected": selected,
      "_valid_passes": valid_passes,
  }


def replay_row(sample: dict, players: int):
  sample = _validated_sample(sample)
  legal = sample["_legal"]
  baseline = sample["_baseline"]
  proposed = sample["_proposed"]
  selected = sample["_selected"]
  mask = np.zeros(1000, dtype=np.float32)
  mask[legal] = 1.0
  target = np.zeros(1000, dtype=np.float32)
  target[selected] = 1.0
  features = np.zeros((1000, ACTION_FEATURE_SIZE), dtype=np.float32)
  features[legal] = sample["_legal_features"]
  risk_targets = np.zeros(1000, dtype=np.float32)
  risk_mask = np.zeros(1000, dtype=np.float32)
  passes = sample["_valid_passes"]
  risk_targets[baseline] = float(np.mean([
      item["baseline_paradox"] for item in passes
  ]))
  risk_targets[proposed] = float(np.mean([
      item["proposed_paradox"] for item in passes
  ]))
  risk_mask[[baseline, proposed]] = 1.0
  action_values = np.zeros(1000, dtype=np.float32)
  action_value_mask = np.zeros(1000, dtype=np.float32)
  # The selector's value axis is explicitly survival value here: lower paired
  # paradox risk is better, so negative paradox probability is the value.
  action_values[[baseline, proposed]] = -risk_targets[[baseline, proposed]]
  action_value_mask[[baseline, proposed]] = 1.0
  return (
      sample["_observation"],
      mask,
      target,
      np.zeros(players, dtype=np.float32),
      np.zeros(players, dtype=np.float32),
      selected,
      int(sample["player"]),
      risk_targets,
      risk_mask,
      action_values,
      action_value_mask,
      features,
  )


def convert(
    inputs: list[Path],
    out: Path,
    metadata_out: Path,
    max_samples: int,
    accepted_repeat: int = 1,
):
  if int(accepted_repeat) < 1:
    raise ValueError("accepted_repeat must be at least 1")
  rows = []
  row_metadata = []
  row_group_ids = []
  seen = set()
  unique_rows = 0
  unique_accepted_rows = 0
  unique_base_keep_rows = 0
  input_metadata = []
  players = None
  for path in inputs:
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_players = int(payload.get("players", 0))
    if current_players not in (2, 3, 4, 5):
      raise ValueError(f"invalid player count in {path}")
    if players is None:
      players = current_players
    elif players != current_players:
      raise ValueError("root-rollout replay inputs mix player counts")
    source_group = f"root-screen:{path.stem}"
    input_metadata.append({
        "path": str(path),
        "sha256": _sha256(path),
        "source_group_id": source_group,
    })
    decision_stats = payload.get("decision_stats", {})
    _validate_accepted_recording(decision_stats)
    for location, raw_sample in _walk_samples(decision_stats):
      sample = _validated_sample(raw_sample)
      key = hashlib.sha256(
          sample["_observation"].tobytes()
          + sample["_legal"].tobytes()
          + np.asarray([
              sample["_baseline"], sample["_proposed"], sample["_selected"]
          ], dtype=np.int64).tobytes()
      ).hexdigest()
      if key in seen:
        continue
      seen.add(key)
      row = replay_row(sample, current_players)
      metadata_row = {
          "source": str(path),
          "location": location,
          "source_group_id": source_group,
          "sample_sha256": key,
          "player": int(sample["player"]),
          "phase": str(sample.get("phase") or ""),
          "trick_number": int(sample["trick_number"]),
          "search_seed": int(sample.get("search_seed", -1)),
          "baseline_action": sample["_baseline"],
          "policy_base_action": sample["_baseline"],
          "policy_best_action": sample["_baseline"],
          "proposed_action": sample["_proposed"],
          "proposed_best_action": sample["_proposed"],
          "selected_action": sample["_selected"],
          "audit_accepted": bool(sample["audit_accepted"]),
          "target_kind": (
              "audited_correction" if bool(sample["audit_accepted"])
              else "audit_rejected_base_keep"
          ),
          "audit_base_keep_validation": not bool(sample["audit_accepted"]),
          "action_value_objective": "negative_any_paradox_probability",
          "audit_contract": sample["audit_contract"],
          "selection": sample.get("selection"),
          "audit_passes": sample["audit_passes"],
      }
      source_index = unique_rows
      unique_rows += 1
      accepted = bool(sample["audit_accepted"])
      unique_accepted_rows += int(accepted)
      unique_base_keep_rows += int(not accepted)
      repeat = int(accepted_repeat) if accepted else 1
      for _ in range(repeat):
        weighted_metadata = dict(metadata_row)
        weighted_metadata["unique_source_row_index"] = int(source_index)
        weighted_metadata["weighted_row_index"] = int(len(rows))
        rows.append(row)
        row_group_ids.append(source_group)
        row_metadata.append(weighted_metadata)
      if max_samples > 0 and unique_rows >= max_samples:
        break
    if max_samples > 0 and unique_rows >= max_samples:
      break
  if not rows:
    raise ValueError("no root-rollout replay samples found")
  embedded = {
      "schema_version": 1,
      "source": "root_rollout_live_audit",
      "information_safe": True,
      "players": players,
      "rows": len(rows),
      "unique_rows": int(unique_rows),
      "accepted_rows": sum(item["audit_accepted"] for item in row_metadata),
      "base_keep_rows": sum(not item["audit_accepted"] for item in row_metadata),
      "unique_accepted_rows": int(unique_accepted_rows),
      "unique_base_keep_rows": int(unique_base_keep_rows),
      "accepted_repeat": int(accepted_repeat),
      "row_group_ids": row_group_ids,
      "row_group_id_kind": "independent_performance_seed",
      "row_group_count": len(set(row_group_ids)),
      "inputs": input_metadata,
  }
  save_replay(rows, out, metadata=embedded)
  metadata_out.parent.mkdir(parents=True, exist_ok=True)
  metadata_out.write_text(
      json.dumps({**embedded, "row_metadata": row_metadata}, indent=2,
                 sort_keys=True) + "\n",
      encoding="utf-8",
  )
  return embedded


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", nargs="+")
  parser.add_argument("--out", required=True)
  parser.add_argument("--metadata-out", default="")
  parser.add_argument("--max-samples", type=int, default=0)
  parser.add_argument(
      "--accepted-repeat",
      type=int,
      default=1,
      help=(
          "Repeat each uniquely accepted correction this many times while "
          "retaining unique evidence counts in metadata. Keep at 1 for "
          "validation replay."
      ),
  )
  args = parser.parse_args()
  out = Path(args.out)
  metadata_out = (
      Path(args.metadata_out)
      if args.metadata_out else out.with_suffix(out.suffix + ".metadata.json")
  )
  report = convert(
      [Path(path) for path in args.inputs], out, metadata_out,
      max(0, int(args.max_samples)),
      accepted_repeat=int(args.accepted_repeat),
  )
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
