#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def percent(value):
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def display_percent(value):
    return "n/a" if value is None else f"{value}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--paradox", required=True)
    parser.add_argument("--led", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paradox = load_json(args.paradox)
    led = load_json(args.led)
    gate = paradox.get("homogeneous_paradox_gate", {})
    led_stats = led.get("stats", {})
    first = paradox.get("first_paradox_summary", {})

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha,
        "mode": args.mode,
        "artifacts": {
            "homogeneous_paradox": args.paradox,
            "led_switch": args.led,
        },
        "homogeneous_paradox_gate": {
            "matches": gate.get("completed_matches"),
            "hand_rounds": gate.get("hand_rounds"),
            "rounds_with_any_paradox": gate.get("rounds_with_any_paradox"),
            "same_policy_paradox_round_rate": gate.get("same_policy_paradox_round_rate"),
            "same_policy_paradox_round_rate_pct": percent(
                gate.get("same_policy_paradox_round_rate")
            ),
            "same_policy_paradox_round_rate_ci95": gate.get(
                "same_policy_paradox_round_rate_ci95"
            ),
            "per_seat_paradox_round_rate": gate.get("per_seat_paradox_round_rate"),
            "per_seat_paradox_round_rate_pct": percent(
                gate.get("per_seat_paradox_round_rate")
            ),
            "threshold": gate.get("threshold"),
            "passed": gate.get("passed"),
        },
        "first_paradox_summary": {
            "rounds_with_trace": first.get("rounds_with_trace"),
            "trigger_by_phase": first.get("trigger_by_phase", {}),
            "trigger_by_legal_count": first.get("trigger_by_legal_count", {}),
            "trigger_by_trick_number": first.get("trigger_by_trick_number", {}),
            "trigger_by_led_color": first.get("trigger_by_led_color", {}),
            "forced_triggers": first.get("forced_triggers"),
            "prediction_gap_avg": first.get("prediction_gap_avg"),
        },
        "led_switch": {
            "matches": led_stats.get("matches"),
            "hand_rounds": led_stats.get("hand_rounds"),
            "rounds_with_any_paradox": led_stats.get("rounds_with_any_paradox"),
            "same_policy_paradox_round_rate_pct": led_stats.get(
                "same_policy_paradox_round_rate_pct"
            ),
            "nonred_led_opportunities": led_stats.get("nonred_led_opportunities"),
            "nonred_led_nonparadox_actions": led_stats.get(
                "nonred_led_nonparadox_actions"
            ),
            "follow_led": led_stats.get("follow_led"),
            "follow_rate_among_nonred_led_actions": led_stats.get(
                "follow_rate_among_nonred_led_actions"
            ),
            "switch_off_led": led_stats.get("switch_off_led"),
            "switch_rate_among_nonred_led_actions": led_stats.get(
                "switch_rate_among_nonred_led_actions"
            ),
            "switch_with_led_color_legal": led_stats.get(
                "switch_with_led_color_legal"
            ),
            "switch_to_red": led_stats.get("switch_to_red"),
            "switch_to_nonred": led_stats.get("switch_to_nonred"),
            "red_rate_among_switches": led_stats.get("red_rate_among_switches"),
            "switch_with_red_legal": led_stats.get("switch_with_red_legal"),
            "switch_with_red_legal_chose_red": led_stats.get(
                "switch_with_red_legal_chose_red"
            ),
            "red_choice_rate_when_red_legal_on_switch": led_stats.get(
                "red_choice_rate_when_red_legal_on_switch"
            ),
            "switch_win_rate_resolved": led_stats.get("switch_win_rate_resolved"),
            "red_switch_win_rate_resolved": led_stats.get(
                "red_switch_win_rate_resolved"
            ),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    gate_summary = summary["homogeneous_paradox_gate"]
    led_summary = summary["led_switch"]
    print("== Summary ==")
    print(f"checkpoint_sha256={summary['checkpoint_sha256']}")
    print(
        "homogeneous_any_paradox_rate="
        f"{gate_summary['rounds_with_any_paradox']}/{gate_summary['hand_rounds']} "
        f"({gate_summary['same_policy_paradox_round_rate_pct']}%) "
        f"passed={gate_summary['passed']}"
    )
    print(
        "follow_nonred_led_rate="
        f"{led_summary['follow_led']}/{led_summary['nonred_led_nonparadox_actions']} "
        f"({led_summary['follow_rate_among_nonred_led_actions']}%)"
    )
    print(
        "switch_off_led_rate="
        f"{led_summary['switch_off_led']}/{led_summary['nonred_led_nonparadox_actions']} "
        f"({led_summary['switch_rate_among_nonred_led_actions']}%)"
    )
    print(
        "red_when_switching="
        f"{led_summary['switch_to_red']}/{led_summary['switch_off_led']} "
        f"({display_percent(led_summary['red_rate_among_switches'])})"
    )
    print(
        "red_when_red_legal_on_switch="
        f"{led_summary['switch_with_red_legal_chose_red']}/"
        f"{led_summary['switch_with_red_legal']} "
        f"({display_percent(led_summary['red_choice_rate_when_red_legal_on_switch'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
