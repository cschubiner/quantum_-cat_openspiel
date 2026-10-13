#!/usr/bin/env python3
"""Backfill deployable public-history metadata onto deterministic replay rows.

The replay tensor stores the current observation but not public play order.
Strict best-of-rollouts sidecars retain enough deterministic provenance to
replay the source round.  This tool reconstructs that state, requires exact
observation/legal/action agreement, and only then writes the public sequence
summary used by the history-aware selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from quantum_cat_alphazero_torch import load_replay
from quantum_cat_failure_miner import play_round_with_history
from quantum_cat_full_match_elo import load_neural
from quantum_cat_public_history import public_history_metadata


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", required=True)
  parser.add_argument("--metadata", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
  return parser.parse_args()


def _read_sidecar(path: Path) -> tuple[dict, list[dict]]:
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
    raise ValueError(f"metadata sidecar must contain a row list: {path}")
  return payload, [dict(row) for row in payload["rows"]]


def _source_sidecar(row: dict, fallback: Path) -> Path:
  source_path = str(row.get("source_path", "") or "").strip()
  if not source_path:
    return fallback
  path = Path(source_path)
  candidate = path.with_suffix(path.suffix + ".metadata.json")
  if not candidate.exists():
    raise FileNotFoundError(f"source metadata sidecar not found: {candidate}")
  return candidate


def _root_source(
    row: dict,
    fallback: Path,
    source_cache: dict[Path, tuple[dict, list[dict]]],
) -> tuple[Path, dict]:
  """Follow merge provenance until reaching the generating sidecar."""
  current_row = row
  current_fallback = fallback
  visited = set()
  while True:
    source_path = _source_sidecar(current_row, current_fallback)
    if source_path in visited:
      raise ValueError(f"cyclic replay provenance at {source_path}")
    visited.add(source_path)
    if source_path not in source_cache:
      source_cache[source_path] = _read_sidecar(source_path)
    payload, rows = source_cache[source_path]
    if "checkpoint" in payload:
      return source_path, payload
    source_row_index = current_row.get("source_row_index")
    if source_row_index is None:
      raise ValueError(
          "merged replay provenance ended before a generating checkpoint: "
          f"{source_path}"
      )
    source_row_index = int(source_row_index)
    if source_row_index < 0 or source_row_index >= len(rows):
      raise ValueError(
          f"source_row_index out of range for {source_path}: {source_row_index}"
      )
    current_row = rows[source_row_index]
    current_fallback = source_path


def _actor_args(source: dict) -> SimpleNamespace:
  return SimpleNamespace(
      bot_mode=str(source.get("bot_mode", "policy")),
      liveness_shield_base_mode="q_policy",
      liveness_shield_phases="play",
      liveness_shield_min_trick_number=-1,
      liveness_shield_led_colors="",
      liveness_shield_min_open_slot_delta=1,
      liveness_shield_min_public_damage_delta=2,
      liveness_shield_max_policy_log_gap=-1.0,
      action_value_selection_weight=0.0,
      action_paradox_selection_penalty=0.0,
      action_paradox_rerank_mode="additive",
      action_paradox_risk_threshold=0.0,
      action_paradox_min_risk_margin=0.0,
      action_feasibility_selection_weight=0.0,
      action_paradox_max_policy_log_gap=2.0,
      action_value_rerank_clip=0.5,
      action_value_rerank_phases="",
      action_value_rerank_min_margin=0.0,
      behavior_epsilon=0.0,
      behavior_nonpolicy_mode="uniform",
  )


def _round_decisions(
    source: dict,
    match_index: int,
    target_round: int,
    model,
    model_args,
    device: torch.device,
) -> list[dict]:
  players = int(source["players"])
  totals = np.zeros(players, dtype=np.float32)
  decisions = None
  for round_index in range(int(target_round) + 1):
    start_player = round_index % players
    game = pyspiel.load_game(
        "python_quantum_cat",
        {
            "players": players,
            "start_player": start_player,
            "match_context": int(bool(source.get("match_context", False))),
        },
    )
    round_seed = (
        int(source["seed"])
        + int(match_index) * 1000003
        + int(round_index) * 9973
    )
    terminal, current_decisions, _ = play_round_with_history(
        model,
        game,
        seed=round_seed,
        value_scale=float(getattr(model_args, "value_scale", 20.0)),
        device=device,
        match_totals=(
            np.copy(totals) if bool(source.get("match_context", False)) else None
        ),
        round_index=round_index,
        args=_actor_args(source),
    )
    returns = np.asarray(terminal.returns(), dtype=np.float32)
    totals = returns if bool(source.get("match_context", False)) else totals + returns
    if round_index == int(target_round):
      decisions = current_decisions
  if decisions is None:
    raise RuntimeError("failed to reconstruct target round")
  return decisions


def _verify_row(row, metadata: dict, decision: dict) -> None:
  player = int(decision["player"])
  observation = np.asarray(
      decision["state"].observation_tensor(player), dtype=np.float32
  )
  replay_observation = np.asarray(row[0], dtype=np.float32)
  if observation.shape != replay_observation.shape or not np.array_equal(
      observation, replay_observation
  ):
    max_delta = (
        float(np.max(np.abs(observation - replay_observation)))
        if observation.shape == replay_observation.shape else float("inf")
    )
    raise ValueError(f"reconstructed observation mismatch; max_delta={max_delta}")
  legal = {int(action) for action in decision["legal"]}
  replay_legal = {
      int(action)
      for action in np.flatnonzero(np.asarray(row[1], dtype=np.float32) > 0.5)
  }
  if legal != replay_legal:
    raise ValueError(f"reconstructed legal actions mismatch: {legal} vs {replay_legal}")
  public = metadata.get("decision", {})
  checks = {
      "index": int(decision["index"]),
      "player": player,
      "trick_number": int(decision.get("trick_number", -1)),
      "policy_action": int(decision["action"]),
  }
  for key, actual in checks.items():
    if int(public.get(key, -999999)) != int(actual):
      raise ValueError(
          f"reconstructed decision mismatch for {key}: "
          f"{actual} vs {public.get(key)!r}"
      )


def main() -> None:
  args = parse_args()
  replay_path = Path(args.replay)
  metadata_path = Path(args.metadata)
  output_path = Path(args.out)
  replay_rows = load_replay(replay_path)
  payload, metadata_rows = _read_sidecar(metadata_path)
  if len(replay_rows) != len(metadata_rows):
    raise ValueError(
        f"replay/metadata row mismatch: {len(replay_rows)} vs {len(metadata_rows)}"
    )

  device = torch.device(str(args.device))
  source_cache: dict[Path, tuple[dict, list[dict]]] = {}
  model_cache = {}
  round_cache = {}
  enriched = []
  for row_index, (row, metadata) in enumerate(zip(replay_rows, metadata_rows)):
    source_path, source = _root_source(
        metadata,
        metadata_path,
        source_cache,
    )
    checkpoint = str(source["checkpoint"])
    players = int(source["players"])
    model_key = (checkpoint, players)
    if model_key not in model_cache:
      model_cache[model_key] = load_neural(
          checkpoint,
          players,
          device,
          _actor_args(source),
      )
    model, model_args = model_cache[model_key]
    match_index = int(metadata["match_index"])
    round_index = int(metadata["round_index"])
    round_key = (source_path, match_index, round_index)
    if round_key not in round_cache:
      round_cache[round_key] = _round_decisions(
          source,
          match_index,
          round_index,
          model,
          model_args,
          device,
      )
    decision_index = int(metadata["decision"]["index"])
    decisions = round_cache[round_key]
    if decision_index < 0 or decision_index >= len(decisions):
      raise ValueError(
          f"decision index out of range for row {row_index}: {decision_index}"
      )
    decision = decisions[decision_index]
    try:
      _verify_row(row, metadata, decision)
    except Exception as exc:
      raise ValueError(f"row {row_index} reconstruction failed: {exc}") from exc
    metadata["public_history"] = public_history_metadata(
        decision["state"], int(decision["player"])
    )
    enriched.append(metadata)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output = {
      **payload,
      "public_history_enrichment": {
          "verified_rows": int(len(enriched)),
          "source_sidecars": [str(path) for path in sorted(source_cache)],
          "replay": str(replay_path),
      },
      "rows": enriched,
  }
  output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
      "out": str(output_path),
      "verified_rows": int(len(enriched)),
      "source_sidecars": int(len(source_cache)),
      "rounds_reconstructed": int(len(round_cache)),
  }, sort_keys=True))


if __name__ == "__main__":
  main()
