"""Build hard selected-vs-best replay rows from dense warning metadata.

This converts dense all-legal low-exit warning labels into a smaller contrast
replay. The intended use is to keep only rows where the source policy action
paradoxes while the dense label-best action survives, then label just those two
actions for action-value/action-paradox head training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from quantum_cat_alphazero_torch import load_replay, save_replay


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument(
      "--metadata-input",
      default="",
      help="Defaults to <input>.metadata.json.",
  )
  parser.add_argument("--out", required=True)
  parser.add_argument("--summary-out", required=True)
  parser.add_argument(
      "--metadata-out",
      default="",
      help="Defaults to <out>.metadata.json.",
  )
  parser.add_argument("--require-best-survives", action="store_true")
  parser.add_argument("--require-selected-paradoxes", action="store_true")
  parser.add_argument("--require-policy-target-changed", action="store_true")
  parser.add_argument("--min-value-gap", type=float, default=1.0)
  parser.add_argument("--positive-value", type=float, default=0.9)
  parser.add_argument("--negative-value", type=float, default=-0.9)
  parser.add_argument(
      "--use-original-dense-values",
      action="store_true",
      help="Use dense_action_value_target for selected/best instead of +/- values.",
  )
  return parser.parse_args()


def _metadata_path(input_path: str, explicit: str) -> Path:
  if explicit:
    return Path(explicit)
  return Path(input_path).with_suffix(Path(input_path).suffix + ".metadata.json")


def _outcome_by_action(row: dict) -> dict[int, dict]:
  return {
      int(outcome["action"]): outcome
      for outcome in row.get("dense", {}).get("action_outcomes", [])
  }


def _safe_bool(value) -> bool:
  return bool(value) if value is not None else False


def main() -> None:
  args = parse_args()
  replay = load_replay(args.input)
  metadata_path = _metadata_path(args.input, args.metadata_input)
  metadata = json.loads(metadata_path.read_text())
  metadata_rows = list(metadata.get("rows", []))
  if len(replay) != len(metadata_rows):
    raise ValueError(
        f"Replay row count {len(replay)} does not match metadata rows "
        f"{len(metadata_rows)}"
    )

  rows = []
  output_metadata_rows = []
  summary = Counter()
  value_gaps = []
  selected_actions = Counter()
  best_actions = Counter()
  trick_numbers = Counter()
  legal_counts = Counter()

  for source_index, (example, meta_row) in enumerate(zip(replay, metadata_rows)):
    summary["input_rows"] += 1
    dense = meta_row.get("dense", {})
    outcomes = _outcome_by_action(meta_row)
    selected_action = int(meta_row.get("selected_action", -1))
    policy_action = int(dense.get("policy_best_action", selected_action))
    best_action = int(dense.get("label_best_action", meta_row.get("best_action", -1)))
    selected_outcome = outcomes.get(selected_action)
    best_outcome = outcomes.get(best_action)

    if selected_outcome is None or best_outcome is None:
      summary["missing_outcome"] += 1
      continue
    if best_action == selected_action:
      summary["same_selected_best"] += 1
      continue
    if args.require_policy_target_changed and not bool(
        dense.get("policy_target_changed", False)
    ):
      summary["policy_target_not_changed"] += 1
      continue
    if policy_action != selected_action:
      summary["selected_not_policy_best"] += 1
      continue
    selected_paradoxes = _safe_bool(selected_outcome.get("any_paradox"))
    best_paradoxes = _safe_bool(best_outcome.get("any_paradox"))
    if args.require_selected_paradoxes and not selected_paradoxes:
      summary["selected_did_not_paradox"] += 1
      continue
    if args.require_best_survives and best_paradoxes:
      summary["best_did_not_survive"] += 1
      continue

    selected_value = float(selected_outcome.get("dense_action_value_target", 0.0))
    best_value = float(best_outcome.get("dense_action_value_target", 0.0))
    value_gap = best_value - selected_value
    if value_gap < float(args.min_value_gap):
      summary["value_gap_too_small"] += 1
      continue

    obs, mask, _policy, value, paradox, _action, player = example[:7]
    num_actions = len(mask)
    policy = np.zeros(num_actions, dtype=np.float32)
    policy[best_action] = 1.0
    action_targets = np.zeros(num_actions, dtype=np.float32)
    action_target_mask = np.zeros(num_actions, dtype=np.float32)
    action_value_targets = np.zeros(num_actions, dtype=np.float32)
    action_value_mask = np.zeros(num_actions, dtype=np.float32)

    action_targets[selected_action] = 1.0
    action_targets[best_action] = 0.0
    action_target_mask[selected_action] = 1.0
    action_target_mask[best_action] = 1.0
    if args.use_original_dense_values:
      action_value_targets[selected_action] = selected_value
      action_value_targets[best_action] = best_value
    else:
      action_value_targets[selected_action] = float(args.negative_value)
      action_value_targets[best_action] = float(args.positive_value)
    action_value_mask[selected_action] = 1.0
    action_value_mask[best_action] = 1.0

    rows.append((
        obs,
        mask,
        policy,
        value,
        paradox,
        best_action,
        player,
        action_targets,
        action_target_mask,
        action_value_targets,
        action_value_mask,
        example[11],
    ))
    value_gaps.append(value_gap)
    selected_actions[str(selected_action)] += 1
    best_actions[str(best_action)] += 1
    trick_numbers[str(meta_row.get("trick_number"))] += 1
    legal_counts[str(meta_row.get("legal_count"))] += 1
    output_metadata_rows.append({
        "source_index": int(source_index),
        "source_row_group_id": meta_row.get("row_group_id"),
        "selected_action": selected_action,
        "selected_action_string": meta_row.get("selected_action_string"),
        "best_action": best_action,
        "best_action_string": dense.get(
            "label_best_action_string", meta_row.get("best_action_string")
        ),
        "selected_any_paradox": selected_paradoxes,
        "best_any_paradox": best_paradoxes,
        "selected_forced_paradox": _safe_bool(
            selected_outcome.get("forced_paradox")
        ),
        "best_forced_paradox": _safe_bool(best_outcome.get("forced_paradox")),
        "selected_dense_value": selected_value,
        "best_dense_value": best_value,
        "value_gap": value_gap,
        "trick_number": meta_row.get("trick_number"),
        "legal_count": meta_row.get("legal_count"),
        "led_color": meta_row.get("led_color"),
    })
    summary["rows"] += 1

  if not rows:
    raise ValueError("No rows matched hard dense warning criteria")

  out_metadata = {
      "kind": "hard_dense_warning_replay",
      "input": args.input,
      "metadata_input": str(metadata_path),
      "criteria": {
          "require_best_survives": bool(args.require_best_survives),
          "require_selected_paradoxes": bool(args.require_selected_paradoxes),
          "require_policy_target_changed": bool(
              args.require_policy_target_changed
          ),
          "min_value_gap": float(args.min_value_gap),
          "positive_value": float(args.positive_value),
          "negative_value": float(args.negative_value),
          "use_original_dense_values": bool(args.use_original_dense_values),
      },
      "rows": output_metadata_rows,
  }
  save_replay(rows, args.out, metadata={
      "source": "quantum_cat_hard_dense_warning_replay",
      "input": args.input,
      "rows": len(rows),
      "criteria": out_metadata["criteria"],
  })

  metadata_out = (
      Path(args.metadata_out)
      if args.metadata_out
      else Path(args.out).with_suffix(Path(args.out).suffix + ".metadata.json")
  )
  metadata_out.parent.mkdir(parents=True, exist_ok=True)
  metadata_out.write_text(json.dumps(out_metadata, indent=2, sort_keys=True) + "\n")

  summary_out = {
      "source": "quantum_cat_hard_dense_warning_replay",
      "input": args.input,
      "metadata_input": str(metadata_path),
      "out": args.out,
      "metadata_out": str(metadata_out),
      "summary": dict(sorted(summary.items())),
      "value_gap_mean": float(np.mean(value_gaps)),
      "value_gap_min": float(np.min(value_gaps)),
      "value_gap_max": float(np.max(value_gaps)),
      "selected_actions": dict(selected_actions.most_common()),
      "best_actions": dict(best_actions.most_common()),
      "trick_numbers": dict(trick_numbers),
      "legal_counts": dict(legal_counts),
  }
  Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.summary_out).write_text(
      json.dumps(summary_out, indent=2, sort_keys=True) + "\n"
  )
  print(json.dumps(summary_out, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
