#!/usr/bin/env python3
"""Audit policy transfer on strict root-rollout corrections and base keeps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quantum_cat_alphazero_torch import (
    load_model_payload,
    load_replay,
    make_game,
)
from quantum_cat_proposal_policy_checkpoint_audit import (
    checkpoint_probabilities,
)


_TARGET_KINDS = {"audited_correction", "audit_rejected_base_keep"}


def _split_csv(value: str) -> list[str]:
  return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _rounded(value: float | None) -> float | None:
  return None if value is None else round(float(value), 6)


def _group_report(
    mask: np.ndarray,
    candidate_probs: np.ndarray,
    anchor_probs: np.ndarray,
    targets: np.ndarray,
) -> dict:
  indices = np.flatnonzero(mask)
  if not len(indices):
    return {"count": 0}
  row_indices = np.arange(len(indices))
  target = targets[indices]
  candidate_top = np.argmax(candidate_probs[indices], axis=1)
  anchor_top = np.argmax(anchor_probs[indices], axis=1)
  candidate_target_prob = candidate_probs[indices][row_indices, target]
  anchor_target_prob = anchor_probs[indices][row_indices, target]
  return {
      "count": int(len(indices)),
      "candidate_target_top1_count": int(np.sum(candidate_top == target)),
      "candidate_target_top1_rate": _rounded(np.mean(candidate_top == target)),
      "anchor_target_top1_count": int(np.sum(anchor_top == target)),
      "anchor_target_top1_rate": _rounded(np.mean(anchor_top == target)),
      "candidate_mean_target_prob": _rounded(np.mean(candidate_target_prob)),
      "anchor_mean_target_prob": _rounded(np.mean(anchor_target_prob)),
      "mean_target_prob_gain": _rounded(
          np.mean(candidate_target_prob - anchor_target_prob)
      ),
      "candidate_cross_entropy": _rounded(
          np.mean(-np.log(np.clip(candidate_target_prob, 1e-12, 1.0)))
      ),
      "anchor_cross_entropy": _rounded(
          np.mean(-np.log(np.clip(anchor_target_prob, 1e-12, 1.0)))
      ),
  }


def policy_audit_report(
    candidate_probs: np.ndarray,
    anchor_probs: np.ndarray,
    policies: np.ndarray,
    metadata_rows: list[dict],
    *,
    min_correction_rows: int = 1,
    min_correction_top1_rate: float = 0.0,
    min_correction_target_prob_gain: float = 0.0,
    max_base_keep_top1_drop: float = 1.0,
    max_unsafe_proposal_adoption_rate: float = 1.0,
) -> dict:
  if candidate_probs.shape != anchor_probs.shape:
    raise ValueError("candidate/anchor probability shape mismatch")
  if len(candidate_probs) != len(policies) or len(policies) != len(metadata_rows):
    raise ValueError("probability/replay/metadata row mismatch")
  targets = np.argmax(policies, axis=1).astype(np.int64)
  kinds = np.asarray([str(row.get("target_kind", "")) for row in metadata_rows])
  unexpected = sorted(set(kinds) - _TARGET_KINDS)
  if unexpected:
    raise ValueError(f"unexpected root-rollout target kinds: {unexpected}")
  correction_mask = kinds == "audited_correction"
  base_mask = kinds == "audit_rejected_base_keep"
  for idx, row in enumerate(metadata_rows):
    target = int(targets[idx])
    selected = int(row.get("selected_action", -1))
    baseline = int(row.get("baseline_action", -1))
    proposed = int(row.get("proposed_action", -1))
    accepted = bool(row.get("audit_accepted"))
    if target != selected:
      raise ValueError(f"row {idx} replay/metadata target mismatch")
    if accepted != bool(correction_mask[idx]):
      raise ValueError(f"row {idx} accepted/target-kind mismatch")
    expected = proposed if accepted else baseline
    if selected != expected:
      raise ValueError(f"row {idx} selected target violates audit result")

  correction = _group_report(
      correction_mask, candidate_probs, anchor_probs, targets
  )
  base_keep = _group_report(base_mask, candidate_probs, anchor_probs, targets)
  all_rows = _group_report(
      np.ones(len(metadata_rows), dtype=bool),
      candidate_probs,
      anchor_probs,
      targets,
  )
  candidate_top = np.argmax(candidate_probs, axis=1)
  eligible = np.asarray([
      bool(base_mask[idx])
      and int(row["proposed_action"]) != int(row["baseline_action"])
      for idx, row in enumerate(metadata_rows)
  ])
  proposed = np.asarray([
      int(row.get("proposed_action", -1)) for row in metadata_rows
  ])
  adopted = eligible & (candidate_top == proposed)
  unsafe_rate = (
      None if int(np.sum(eligible)) == 0
      else float(np.sum(adopted) / np.sum(eligible))
  )
  source_groups = np.asarray([
      str(row.get("source_group_id", "")) for row in metadata_rows
  ])
  tricks = np.asarray([
      int(row.get("trick_number", -1)) for row in metadata_rows
  ])
  by_source = {}
  for source in sorted(set(source_groups)):
    source_mask = source_groups == source
    by_source[source] = {
        "all": _group_report(
            source_mask, candidate_probs, anchor_probs, targets
        ),
        "correction": _group_report(
            source_mask & correction_mask,
            candidate_probs,
            anchor_probs,
            targets,
        ),
        "base_keep": _group_report(
            source_mask & base_mask,
            candidate_probs,
            anchor_probs,
            targets,
        ),
    }
  by_trick = {}
  for trick in sorted(set(tricks.tolist())):
    trick_mask = tricks == trick
    by_trick[str(trick)] = {
        "all": _group_report(
            trick_mask, candidate_probs, anchor_probs, targets
        ),
        "correction": _group_report(
            trick_mask & correction_mask,
            candidate_probs,
            anchor_probs,
            targets,
        ),
        "base_keep": _group_report(
            trick_mask & base_mask,
            candidate_probs,
            anchor_probs,
            targets,
        ),
    }
  base_drop = float(
      (base_keep.get("anchor_target_top1_rate") or 0.0)
      - (base_keep.get("candidate_target_top1_rate") or 0.0)
  )
  checks = {
      "minimum_correction_rows": (
          int(correction.get("count", 0)) >= int(min_correction_rows)
      ),
      "minimum_correction_top1_rate": (
          float(correction.get("candidate_target_top1_rate") or 0.0)
          >= float(min_correction_top1_rate)
      ),
      "minimum_correction_target_probability_gain": (
          float(correction.get("mean_target_prob_gain") or 0.0)
          >= float(min_correction_target_prob_gain)
      ),
      "maximum_base_keep_top1_drop": (
          base_drop <= float(max_base_keep_top1_drop) + 1e-12
      ),
      "maximum_unsafe_proposal_adoption_rate": (
          unsafe_rate is not None
          and unsafe_rate <= float(max_unsafe_proposal_adoption_rate)
      ),
  }
  return {
      "rows": int(len(metadata_rows)),
      "source_group_count": int(len(set(source_groups))),
      "all": all_rows,
      "correction": correction,
      "base_keep": base_keep,
      "base_keep_top1_drop": _rounded(base_drop),
      "unsafe_proposal_adoption": {
          "eligible_count": int(np.sum(eligible)),
          "adopted_count": int(np.sum(adopted)),
          "rate": _rounded(unsafe_rate),
      },
      "by_source": by_source,
      "by_trick": by_trick,
      "gate": {
          "checks": checks,
          "pass": bool(all(checks.values())),
      },
  }


def _load_banks(replay_spec: str, metadata_spec: str):
  replay_paths = [Path(path) for path in _split_csv(replay_spec)]
  metadata_paths = [Path(path) for path in _split_csv(metadata_spec)]
  if not replay_paths:
    raise ValueError("at least one replay is required")
  if not metadata_paths:
    metadata_paths = [path.with_suffix(".metadata.json") for path in replay_paths]
  if len(replay_paths) != len(metadata_paths):
    raise ValueError("replay/metadata path count mismatch")
  combined_replay = []
  combined_rows = []
  source_groups = set()
  bank_reports = []
  for replay_path, metadata_path in zip(replay_paths, metadata_paths):
    replay, embedded = load_replay(replay_path, return_metadata=True)
    sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = list(sidecar.get("row_metadata") or [])
    if len(rows) != len(replay):
      raise ValueError(
          f"replay/metadata row mismatch for {replay_path}: "
          f"{len(replay)} != {len(rows)}"
      )
    if embedded.get("source") != "root_rollout_live_audit":
      raise ValueError(f"{replay_path} is not root-rollout live-audit replay")
    if embedded.get("information_safe") is not True:
      raise ValueError(f"{replay_path} is not information-safe")
    groups = set(str(row.get("source_group_id", "")) for row in rows)
    if "" in groups:
      raise ValueError(f"{metadata_path} has a missing source group")
    overlap = source_groups & groups
    if overlap:
      raise ValueError(f"duplicate source groups across banks: {sorted(overlap)}")
    source_groups.update(groups)
    combined_replay.extend(replay)
    combined_rows.extend(rows)
    bank_reports.append({
        "replay": str(replay_path),
        "replay_sha256": _sha256(replay_path),
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "rows": len(rows),
        "source_groups": len(groups),
    })
  return combined_replay, combined_rows, bank_reports


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--anchor-checkpoint", required=True)
  parser.add_argument("--replay", required=True)
  parser.add_argument("--metadata", default="")
  parser.add_argument("--players", type=int, default=3)
  parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
  parser.add_argument("--out", required=True)
  parser.add_argument("--min-correction-rows", type=int, default=1)
  parser.add_argument("--min-correction-top1-rate", type=float, default=0.0)
  parser.add_argument(
      "--min-correction-target-prob-gain", type=float, default=0.0
  )
  parser.add_argument("--max-base-keep-top1-drop", type=float, default=1.0)
  parser.add_argument(
      "--max-unsafe-proposal-adoption-rate", type=float, default=1.0
  )
  args = parser.parse_args()
  if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("--device=mps requested, but MPS is unavailable")
  device = torch.device(args.device)
  replay, rows, banks = _load_banks(args.replay, args.metadata)
  model_args = SimpleNamespace(
      players=int(args.players),
      match_context=False,
      arch="action_mlp",
      width=256,
      depth=3,
      separate_action_value_encoder=False,
      separate_action_paradox_encoder=False,
  )
  game = make_game(model_args)
  candidate, _, _ = load_model_payload(
      args.checkpoint, game, model_args, device
  )
  anchor, _, _ = load_model_payload(
      args.anchor_checkpoint, game, model_args, device
  )
  candidate_probs = checkpoint_probabilities(candidate, replay, device)
  anchor_probs = checkpoint_probabilities(anchor, replay, device)
  policies = np.asarray([row[2] for row in replay], dtype=np.float32)
  report = policy_audit_report(
      candidate_probs,
      anchor_probs,
      policies,
      rows,
      min_correction_rows=args.min_correction_rows,
      min_correction_top1_rate=args.min_correction_top1_rate,
      min_correction_target_prob_gain=args.min_correction_target_prob_gain,
      max_base_keep_top1_drop=args.max_base_keep_top1_drop,
      max_unsafe_proposal_adoption_rate=(
          args.max_unsafe_proposal_adoption_rate
      ),
  )
  report.update({
      "checkpoint": args.checkpoint,
      "checkpoint_sha256": _sha256(Path(args.checkpoint)),
      "anchor_checkpoint": args.anchor_checkpoint,
      "anchor_checkpoint_sha256": _sha256(Path(args.anchor_checkpoint)),
      "players": int(args.players),
      "information_safe": True,
      "banks": banks,
  })
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
