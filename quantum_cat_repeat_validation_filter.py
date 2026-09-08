#!/usr/bin/env python3
"""Filter repeat-validation replay rows by source-stable label strength."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def _metadata_path(path: Path) -> Path:
  return Path(str(path) + ".metadata.json")


def _load_sidecar(path: Path) -> dict:
  metadata_path = _metadata_path(path)
  if not metadata_path.exists():
    raise FileNotFoundError(f"metadata sidecar not found: {metadata_path}")
  data = json.loads(metadata_path.read_text())
  rows = data.get("rows")
  if not isinstance(rows, list):
    raise ValueError(f"metadata sidecar has no rows array: {metadata_path}")
  return data


def _row_group_id(meta: dict, idx: int) -> str:
  if meta.get("row_group_id") is not None:
    return str(meta["row_group_id"])
  group = meta.get("group") if isinstance(meta.get("group"), dict) else {}
  parts = []
  for key in [
      "match_index",
      "round_index",
      "decision_index",
      "start_player",
      "terminal_any_paradox",
  ]:
    if key in group:
      parts.append(f"{key}={group[key]}")
  if parts:
    return "|".join(parts)
  return f"row={idx}"


def _action_deltas(row, base_action: int) -> list[dict]:
  legal_mask = np.asarray(row[1], dtype=np.float32) > 0.5
  risk_mask = np.asarray(row[8], dtype=np.float32) > 0.5
  value_mask = np.asarray(row[10], dtype=np.float32) > 0.5
  labeled_mask = legal_mask & risk_mask & value_mask
  labeled = [int(action) for action in np.flatnonzero(labeled_mask)]
  if base_action not in set(labeled):
    return []
  values = np.asarray(row[9], dtype=np.float32)
  risks = np.asarray(row[7], dtype=np.float32)
  base_value = float(values[base_action])
  base_risk = float(risks[base_action])
  actions = []
  for action in labeled:
    value_delta = float(values[action]) - base_value
    risk_delta = float(risks[action]) - base_risk
    actions.append({
        "action": int(action),
        "is_base": bool(action == base_action),
        "value_delta": float(value_delta),
        "risk_delta": float(risk_delta),
        "safe": bool(risk_delta <= 0.0),
        "positive": bool(action != base_action and value_delta >= 0.0 and risk_delta <= 0.0),
    })
  return actions


def _score_action(action: dict, risk_penalty: float, unsafe_penalty: float) -> float:
  score = float(action["value_delta"]) - risk_penalty * float(action["risk_delta"])
  if not bool(action["safe"]):
    score -= unsafe_penalty
  return float(score)


def _repeat_candidate_stats(meta: dict, target_action: int) -> dict:
  repeat = meta.get("repeat_validation")
  if not isinstance(repeat, dict):
    return {}
  candidates = repeat.get("candidates")
  if not isinstance(candidates, list):
    return {}
  matched = [
      candidate for candidate in candidates
      if int(candidate.get("candidate_action", -1)) == int(target_action)
  ]
  if not matched:
    return {}
  candidate = dict(matched[0])
  passed_scores = [
      float(item.get("mean_value_delta", 0.0)) - float(item.get("mean_risk_delta", 0.0))
      for item in candidates
      if bool(item.get("passed", False))
  ]
  passed_scores.sort(reverse=True)
  margin = None
  if len(passed_scores) >= 2:
    candidate_score = (
        float(candidate.get("mean_value_delta", 0.0))
        - float(candidate.get("mean_risk_delta", 0.0))
    )
    margin = float(candidate_score - max(
        score for score in passed_scores if score != candidate_score
    )) if any(score != candidate_score for score in passed_scores) else 0.0
  return {
      "repeat_candidate_passed": bool(candidate.get("passed", False)),
      "repeat_candidate_color": candidate.get("candidate_color"),
      "repeat_candidate_is_red_switch": bool(candidate.get("candidate_is_red_switch", False)),
      "repeat_candidate_follows_led": bool(candidate.get("candidate_follows_led", False)),
      "repeat_candidate_any_paradox_rate": float(
          candidate.get("candidate_any_paradox_rate", 1.0)
      ),
      "repeat_candidate_mean_value_delta": float(
          candidate.get("mean_value_delta", 0.0)
      ),
      "repeat_candidate_mean_risk_delta": float(
          candidate.get("mean_risk_delta", 0.0)
      ),
      "repeat_passed_candidate_count": int(len(passed_scores)),
      "repeat_passed_score_margin": margin,
  }


def _row_stats(row, meta: dict, args) -> dict:
  base_action = int(meta.get("policy_best_action", -1))
  actions = _action_deltas(row, base_action)
  if len(actions) < 2:
    return {"keep": False, "reason": "missing_labeled_actions"}
  positives = [action for action in actions if bool(action["positive"])]
  if not positives:
    return {"keep": False, "reason": "no_positive_action"}
  target = max(
      positives,
      key=lambda action: (
          float(action["value_delta"]),
          -float(action["risk_delta"]),
          float(action["action"]),
      ),
  )
  scored = [
      {
          **action,
          "score": _score_action(
              action,
              float(args.risk_penalty),
              float(args.unsafe_penalty),
          ),
      }
      for action in actions
  ]
  target_score = next(
      float(action["score"])
      for action in scored
      if int(action["action"]) == int(target["action"])
  )
  runner_up = max(
      [
          float(action["score"])
          for action in scored
          if int(action["action"]) != int(target["action"])
      ],
      default=float("-inf"),
  )
  margin = float(target_score - runner_up)
  repeat_stats = _repeat_candidate_stats(meta, int(target["action"]))
  stats = {
      "keep": True,
      "reason": "kept",
      "base_action": int(base_action),
      "target_action": int(target["action"]),
      "target_value_delta": float(target["value_delta"]),
      "target_risk_delta": float(target["risk_delta"]),
      "target_score": float(target_score),
      "runner_up_score": float(runner_up),
      "score_margin": float(margin),
      "labeled_action_count": int(len(actions)),
      "positive_count": int(len(positives)),
      **repeat_stats,
  }
  if margin < float(args.min_score_margin):
    stats.update({"keep": False, "reason": "score_margin"})
  if float(target["value_delta"]) < float(args.min_target_value_delta):
    stats.update({"keep": False, "reason": "target_value_delta"})
  if float(target["risk_delta"]) > float(args.max_target_risk_delta):
    stats.update({"keep": False, "reason": "target_risk_delta"})
  if (
      args.require_repeat_candidate_passed
      and not bool(repeat_stats.get("repeat_candidate_passed", False))
  ):
    stats.update({"keep": False, "reason": "repeat_candidate_not_passed"})
  if (
      args.require_repeat_red_switch
      and not bool(repeat_stats.get("repeat_candidate_is_red_switch", False))
  ):
    stats.update({"keep": False, "reason": "repeat_candidate_not_red_switch"})
  return stats


def _filter_one(path: Path, out_dir: Path, args) -> dict:
  replay = load_replay(path)
  sidecar = _load_sidecar(path)
  rows = sidecar["rows"]
  if len(replay) != len(rows):
    raise ValueError(f"replay/metadata row mismatch for {path}: {len(replay)} vs {len(rows)}")
  selected = []
  selected_metadata = []
  stats_rows = []
  reasons = Counter()
  target_colors = Counter()
  for idx, (row, meta) in enumerate(zip(replay, rows)):
    stats = _row_stats(row, meta, args)
    stats["source_row_index"] = int(idx)
    stats_rows.append(stats)
    reasons[str(stats.get("reason", "unknown"))] += 1
    if not bool(stats.get("keep", False)):
      continue
    selected.append(row)
    enriched = dict(meta)
    enriched["repeat_validation_filter"] = stats
    selected_metadata.append(enriched)
    color = stats.get("repeat_candidate_color")
    if color is not None:
      target_colors[str(color)] += 1
  source_summary = {
      "input": str(path),
      "input_rows": int(len(replay)),
      "selected_rows": int(len(selected)),
      "rejection_reasons": dict(sorted(reasons.items())),
      "selected_target_colors": dict(sorted(target_colors.items())),
      "selected_score_margin_mean": (
          float(np.mean([row["score_margin"] for row in stats_rows if row.get("keep")]))
          if selected else None
      ),
      "selected_score_margin_min": (
          float(np.min([row["score_margin"] for row in stats_rows if row.get("keep")]))
          if selected else None
      ),
  }
  if not selected:
    source_summary["out"] = None
    return source_summary
  out_path = out_dir / path.name
  row_group_ids = [
      _row_group_id(meta, idx)
      for idx, meta in enumerate(selected_metadata)
  ]
  replay_metadata = {
      "filter": "repeat_validation_high_margin",
      "source_replay": str(path),
      "row_group_ids": row_group_ids,
      "criteria": _criteria(args),
      "summary": source_summary,
  }
  save_replay(selected, out_path, metadata=replay_metadata)
  sidecar_out = {
      "source_replay": str(path),
      "source_metadata": str(_metadata_path(path)),
      "filter": "repeat_validation_high_margin",
      "criteria": _criteria(args),
      "summary": source_summary,
      "rows": selected_metadata,
  }
  _metadata_path(out_path).write_text(
      json.dumps(sidecar_out, indent=2, sort_keys=True) + "\n"
  )
  source_summary["out"] = str(out_path)
  source_summary["metadata_out"] = str(_metadata_path(out_path))
  return source_summary


def _criteria(args) -> dict:
  return {
      "min_score_margin": float(args.min_score_margin),
      "min_target_value_delta": float(args.min_target_value_delta),
      "max_target_risk_delta": float(args.max_target_risk_delta),
      "risk_penalty": float(args.risk_penalty),
      "unsafe_penalty": float(args.unsafe_penalty),
      "require_repeat_candidate_passed": bool(args.require_repeat_candidate_passed),
      "require_repeat_red_switch": bool(args.require_repeat_red_switch),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True)
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--summary-out", default="")
  parser.add_argument("--min-score-margin", type=float, default=0.5)
  parser.add_argument("--min-target-value-delta", type=float, default=0.0)
  parser.add_argument("--max-target-risk-delta", type=float, default=0.0)
  parser.add_argument("--risk-penalty", type=float, default=1.0)
  parser.add_argument("--unsafe-penalty", type=float, default=1.0)
  parser.add_argument("--require-repeat-candidate-passed", action="store_true")
  parser.add_argument("--require-repeat-red-switch", action="store_true")
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  sources = [
      _filter_one(Path(path), out_dir, args)
      for path in args.input
  ]
  selected_sources = [item for item in sources if int(item["selected_rows"]) > 0]
  summary = {
      "criteria": _criteria(args),
      "input_replays": int(len(sources)),
      "output_replays": int(len(selected_sources)),
      "input_rows": int(sum(int(item["input_rows"]) for item in sources)),
      "selected_rows": int(sum(int(item["selected_rows"]) for item in sources)),
      "sources": sources,
  }
  summary_path = Path(args.summary_out) if args.summary_out else out_dir / "summary.json"
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
