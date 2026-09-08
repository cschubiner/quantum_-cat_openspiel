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


def expected_tactical_family_count(players: int) -> int:
    suite = (
        Path(__file__).resolve().parents[4]
        / "experiments/quantum_cat_curated_tactical_validation.json"
    )
    payload = load_json(str(suite))
    return sum(
        int(row.get("state", {}).get("numPlayers", -1)) == int(players)
        for row in payload.get("scenarios", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--incumbent-checkpoint")
    parser.add_argument("--incumbent-checkpoint-sha")
    parser.add_argument("--incumbent-mode")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--led-mode")
    parser.add_argument("--players", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--led-seed", type=int)
    parser.add_argument("--paradox", required=True)
    parser.add_argument("--competitive")
    parser.add_argument("--curated-tactical")
    parser.add_argument("--curated-bidding")
    parser.add_argument("--curated-bid-regime")
    parser.add_argument("--led", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--require-ci-upper-below-threshold",
        action="store_true",
        help="Legacy diagnostic-only opt-in; never required for competitive promotion.",
    )
    parser.add_argument(
        "--require-competitive-pass",
        action="store_true",
        help="Exit non-zero unless the seat-balanced competitive promotion gate passes.",
    )
    parser.add_argument(
        "--require-tactical-pass",
        action="store_true",
        help="Exit non-zero unless every applicable locked tactical position passes.",
    )
    parser.add_argument(
        "--require-bidding-pass",
        action="store_true",
        help="Exit non-zero unless every locked set-valued bidding hand passes.",
    )
    parser.add_argument(
        "--require-bid-regime-pass",
        action="store_true",
        help="Require every below/exact/above bid-regime tactical scenario.",
    )
    args = parser.parse_args()

    paradox = load_json(args.paradox)
    competitive = load_json(args.competitive) if args.competitive else {}
    tactical = load_json(args.curated_tactical) if args.curated_tactical else None
    bidding = load_json(args.curated_bidding) if args.curated_bidding else None
    bid_regime = (
        load_json(args.curated_bid_regime) if args.curated_bid_regime else None
    )
    tactical_required = bool(
        args.require_tactical_pass
        or (
            args.require_competitive_pass
            and expected_tactical_family_count(args.players) > 0
        )
    )
    bidding_required = bool(args.require_bidding_pass or args.require_competitive_pass)
    bid_regime_required = bool(
        args.require_bid_regime_pass or args.require_competitive_pass
    )
    led = load_json(args.led)
    gate = paradox.get("homogeneous_paradox_gate", {})
    led_stats = led.get("stats", {})
    first = paradox.get("first_paradox_summary", {})

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha,
        "incumbent_checkpoint": args.incumbent_checkpoint,
        "incumbent_checkpoint_sha256": args.incumbent_checkpoint_sha,
        "incumbent_mode": args.incumbent_mode,
        "mode": args.mode,
        "led_mode": args.led_mode or args.mode,
        "players": args.players,
        "seed": args.seed,
        "led_seed": args.led_seed,
        "artifacts": {
            "homogeneous_paradox": args.paradox,
            "competitive_match": args.competitive,
            "led_switch": args.led,
            "curated_tactical": args.curated_tactical,
            "curated_bidding": args.curated_bidding,
            "curated_bid_regime": args.curated_bid_regime,
        },
        "curated_tactical_gate": {
            "available": tactical is not None,
            "required": tactical_required,
            "candidate": tactical.get("candidate") if tactical else None,
            "scenario_count": tactical.get("scenarioCount", 0) if tactical else 0,
            "passed_count": tactical.get("passedCount", 0) if tactical else 0,
            "independent_scenario_family_count": (
                tactical.get("independentScenarioFamilyCount", 0)
                if tactical else 0
            ),
            "human_reviewed_family_count": (
                tactical.get("humanReviewedFamilyCount", 0)
                if tactical else 0
            ),
            "exact_official_scoring_family_count": (
                tactical.get("exactOfficialScoringFamilyCount", 0)
                if tactical else 0
            ),
            "families": tactical.get("families", []) if tactical else [],
            "passed": tactical.get("passed") if tactical else None,
        },
        "curated_bidding_gate": {
            "available": bidding is not None,
            "required": bidding_required,
            "candidate": bidding.get("candidate") if bidding else None,
            "scenario_count": bidding.get("scenarioCount", 0) if bidding else 0,
            "passed_count": bidding.get("passedCount", 0) if bidding else 0,
            "independent_scenario_family_count": (
                bidding.get("independentScenarioFamilyCount", 0)
                if bidding else 0
            ),
            "passed": bidding.get("passed") if bidding else None,
        },
        "curated_bid_regime_gate": {
            "available": bid_regime is not None,
            "required": bid_regime_required,
            "candidate": bid_regime.get("candidate") if bid_regime else None,
            "scenario_count": (
                bid_regime.get("scenarioCount", 0) if bid_regime else 0
            ),
            "passed_count": (
                bid_regime.get("passedCount", 0) if bid_regime else 0
            ),
            "independent_scenario_family_count": (
                bid_regime.get("independentScenarioFamilyCount", 0)
                if bid_regime else 0
            ),
            "passed": bid_regime.get("passed") if bid_regime else None,
        },
        "competitive_promotion_gate": competitive.get(
            "competitive_promotion_gate", {}
        ),
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
            "ci95_upper_below_threshold": gate.get(
                "ci95_upper_below_threshold"
            ),
            "diagnostic_only": True,
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
    competitive_summary = summary["competitive_promotion_gate"]
    tactical_summary = summary["curated_tactical_gate"]
    bidding_summary = summary["curated_bidding_gate"]
    bid_regime_summary = summary["curated_bid_regime_gate"]
    led_summary = summary["led_switch"]
    print("== Summary ==")
    print(f"checkpoint_sha256={summary['checkpoint_sha256']}")
    print(f"mode={summary['mode']}")
    print(f"led_mode={summary['led_mode']}")
    print(
        "locked_bidding_hands="
        f"{bidding_summary['passed_count']}/{bidding_summary['scenario_count']} "
        f"families={bidding_summary['independent_scenario_family_count']} "
        f"passed={bidding_summary['passed']}"
    )
    if bid_regime_summary["available"]:
        print(
            "locked_bid_regime_moves="
            f"{bid_regime_summary['passed_count']}/"
            f"{bid_regime_summary['scenario_count']} "
            f"families={bid_regime_summary['independent_scenario_family_count']} "
            f"passed={bid_regime_summary['passed']}"
        )
    if tactical_summary["available"]:
        print(
            "locked_tactical_moves="
            f"{tactical_summary['passed_count']}/{tactical_summary['scenario_count']} "
            f"families={tactical_summary['independent_scenario_family_count']} "
            f"human_reviewed={tactical_summary['human_reviewed_family_count']} "
            f"exact_official={tactical_summary['exact_official_scoring_family_count']} "
            f"passed={tactical_summary['passed']}"
        )
    else:
        print(f"locked_tactical_moves=unavailable players={args.players}")
    if competitive_summary:
        competitive_pairwise = competitive_summary.get("pairwise", {})
        own_paradox = competitive_summary.get("candidate_own_paradox", {})
        opponent_paradox = competitive_summary.get(
            "reference_personal_paradox", {}
        )
        print(
            "competitive_placement_score="
            f"{competitive_pairwise.get('score_rate')} "
            f"ci95={competitive_pairwise.get('score_rate_ci95')} "
            f"independent_deals={competitive_summary.get('independent_deal_blocks')} "
            f"elo_difference={competitive_pairwise.get('head_to_head_elo_difference')} "
            f"passed={competitive_summary.get('passed')}"
        )
        print(
            "personal_paradox_rate="
            f"{own_paradox.get('round_rate')} "
            f"reference={opponent_paradox.get('round_rate')}"
        )
    print(
        "homogeneous_any_paradox_rate="
        f"{gate_summary['rounds_with_any_paradox']}/{gate_summary['hand_rounds']} "
        f"({gate_summary['same_policy_paradox_round_rate_pct']}%) "
        f"diagnostic_passed={gate_summary['passed']} "
        "ci95_upper_below_threshold="
        f"{gate_summary['ci95_upper_below_threshold']}"
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
    if tactical is not None and (
        tactical.get("candidate") != f"{args.checkpoint}:{args.mode}"
        or any(
            int(row.get("players") or 0) != int(args.players)
            for row in tactical.get("results", [])
        )
    ):
        print("Tactical report does not match the evaluated candidate/player count.")
        return 4
    if tactical_required and (
        tactical is None or tactical.get("passed") is not True
    ):
        return 4
    if bidding is not None and (
        bidding.get("suiteID") != "quantum-cat-curated-bidding-validation-v1"
        or bidding.get("candidate") != f"{args.checkpoint}:{args.mode}"
        or any(
            int(row.get("players") or 0) != int(args.players)
            for row in bidding.get("results", [])
        )
    ):
        print("Bidding report does not match the evaluated candidate/player count.")
        return 5
    if bidding_required and (
        bidding is None
        or bidding.get("mode") != "candidate_bid_validation"
        or bidding.get("passed") is not True
        or int(bidding.get("passedCount") or 0)
        != int(bidding.get("scenarioCount") or 0)
    ):
        return 5
    if bid_regime is not None and (
        bid_regime.get("suiteID")
        != "quantum-cat-curated-bid-regime-validation-v1"
        or bid_regime.get("candidate") != f"{args.checkpoint}:{args.mode}"
        or any(
            int(row.get("players") or 0) != int(args.players)
            for row in bid_regime.get("results", [])
        )
    ):
        print("Bid-regime report does not match the evaluated candidate/player count.")
        return 6
    if bid_regime_required and (
        bid_regime is None
        or bid_regime.get("passed") is not True
        or int(bid_regime.get("passedCount") or 0)
        != int(bid_regime.get("scenarioCount") or 0)
    ):
        return 6
    if (
        args.require_competitive_pass
        and competitive_summary.get("passed") is not True
    ):
        return 3
    if (
        args.require_ci_upper_below_threshold
        and gate_summary["ci95_upper_below_threshold"] is not True
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
