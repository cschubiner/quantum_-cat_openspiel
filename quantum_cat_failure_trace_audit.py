#!/usr/bin/env python3
"""Summarize first-paradox decision traces from full-match eval artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


COLOR_NAMES = ["R", "B", "Y", "G"]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("artifacts", nargs="+")
  parser.add_argument("--num-card-types", type=int, default=8)
  parser.add_argument("--out", default="")
  return parser.parse_args()


def action_color(action, num_card_types):
  action = int(action)
  if action == 999 or action < 0 or num_card_types <= 0:
    return None
  color_idx = action // int(num_card_types)
  if 0 <= color_idx < len(COLOR_NAMES):
    return COLOR_NAMES[color_idx]
  return None


def inc(counter, key, amount=1):
  counter[str(key)] += int(amount)


def pct(numer, denom):
  if denom <= 0:
    return None
  return round(100.0 * float(numer) / float(denom), 2)


def iter_traces(data):
  sample_matches = data.get("sample_matches", [])
  if sample_matches:
    for match in sample_matches:
      for round_row in match.get("rounds", []):
        trace = round_row.get("first_paradox_trace")
        if trace:
          yield trace
    return
  matches_detail = data.get("matches_detail", [])
  if matches_detail:
    for match in matches_detail:
      for round_row in match.get("rounds", []):
        trace = round_row.get("first_paradox_trace")
        if trace:
          yield trace
    return
  # Compact artifacts may only keep samples in first_paradox_summary.
  summary = data.get("first_paradox_summary", {})
  for sample in summary.get("samples", []):
    if sample.get("trigger"):
      yield {
          "trigger": sample.get("trigger", {}),
          "last_decisions": sample.get("last_decisions", []),
      }


def audit_artifact(path, num_card_types):
  data = json.loads(Path(path).read_text())
  traces = list(iter_traces(data))
  counters = Counter()
  by_trick = Counter()
  by_led = Counter()
  low_legal_by_trick = Counter()
  low_legal_by_led = Counter()
  pretrigger_low_legal_traces = 0
  pretrigger_legal1_traces = 0
  play_decisions = 0
  nonred_led_play = 0
  nonred_led_switch = 0
  nonred_led_red_switch = 0
  nonred_led_follow = 0
  red_led_play = 0
  red_led_legal1 = 0
  earliest_low_legal_examples = []

  for trace in traces:
    trigger = trace.get("trigger", {})
    decisions = list(trace.get("last_decisions", []))
    trigger_index = len(decisions) - 1
    low_before = []
    legal1_before = []
    for idx, row in enumerate(decisions):
      phase = row.get("phase")
      if phase != "play":
        continue
      play_decisions += 1
      legal_count = int(row.get("legal_count", -1))
      trick = int(row.get("trick_number", -1))
      led = row.get("led_color")
      action = int(row.get("action", -1))
      color = action_color(action, num_card_types)
      inc(by_trick, trick)
      inc(by_led, led)
      if legal_count <= 2:
        inc(low_legal_by_trick, trick)
        inc(low_legal_by_led, led)
        if idx < trigger_index:
          low_before.append(row)
      if legal_count <= 1 and idx < trigger_index:
        legal1_before.append(row)
      if led in ("B", "Y", "G"):
        nonred_led_play += 1
        if color == led:
          nonred_led_follow += 1
        elif color is not None:
          nonred_led_switch += 1
          if color == "R":
            nonred_led_red_switch += 1
      if led == "R":
        red_led_play += 1
        if legal_count <= 1:
          red_led_legal1 += 1
    if low_before:
      pretrigger_low_legal_traces += 1
      first = low_before[0]
      if len(earliest_low_legal_examples) < 8:
        earliest_low_legal_examples.append({
            "trigger_trick": int(trigger.get("trick_number", -1)),
            "trigger_led": trigger.get("led_color"),
            "first_low_trick": int(first.get("trick_number", -1)),
            "first_low_led": first.get("led_color"),
            "first_low_player": int(first.get("player", -1)),
            "first_low_legal_count": int(first.get("legal_count", -1)),
            "first_low_action": int(first.get("action", -1)),
            "first_low_action_color": action_color(
                int(first.get("action", -1)), num_card_types
            ),
        })
    if legal1_before:
      pretrigger_legal1_traces += 1
  rounds = int(
      data.get("homogeneous_paradox_gate", {}).get("hand_rounds")
      or data.get("rounds")
      or 0
  )
  paradox_rounds = int(
      data.get("homogeneous_paradox_gate", {}).get("rounds_with_any_paradox")
      or data.get("first_paradox_summary", {}).get("rounds_with_trace")
      or len(traces)
  )
  return {
      "artifact": str(path),
      "players": data.get("players"),
      "matches": data.get("matches"),
      "rounds": rounds,
      "paradox_rounds": paradox_rounds,
      "paradox_round_rate": (
          round(float(paradox_rounds) / float(rounds), 6) if rounds else None
      ),
      "traces_audited": len(traces),
      "trace_pretrigger_low_legal_rate": pct(pretrigger_low_legal_traces, len(traces)),
      "trace_pretrigger_legal1_rate": pct(pretrigger_legal1_traces, len(traces)),
      "play_decisions_in_traces": play_decisions,
      "low_legal_by_trick": dict(sorted(low_legal_by_trick.items())),
      "low_legal_by_led": dict(sorted(low_legal_by_led.items())),
      "all_play_by_trick": dict(sorted(by_trick.items())),
      "all_play_by_led": dict(sorted(by_led.items())),
      "nonred_led_play_decisions": nonred_led_play,
      "nonred_led_follow_rate": pct(nonred_led_follow, nonred_led_play),
      "nonred_led_switch_rate": pct(nonred_led_switch, nonred_led_play),
      "nonred_led_switch_to_red_rate": pct(
          nonred_led_red_switch, nonred_led_switch
      ),
      "red_led_play_decisions": red_led_play,
      "red_led_legal1_rate": pct(red_led_legal1, red_led_play),
      "earliest_low_legal_examples": earliest_low_legal_examples,
  }


def main():
  args = parse_args()
  result = {
      "artifacts": [
          audit_artifact(path, int(args.num_card_types))
          for path in args.artifacts
      ]
  }
  text = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
  print(text)


if __name__ == "__main__":
  main()
