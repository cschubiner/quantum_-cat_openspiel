#!/usr/bin/env python3
"""Require independently replicated competitive placement and own safety."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TACTICAL_SUITE = (
    Path(__file__).resolve().parents[4]
    / "experiments/quantum_cat_curated_tactical_validation.json"
)
BIDDING_SUITE = (
    Path(__file__).resolve().parents[4]
    / "experiments/quantum_cat_curated_bidding_validation.json"
)
BID_REGIME_SUITE = (
    Path(__file__).resolve().parents[4]
    / "experiments/quantum_cat_curated_bid_regime_validation.json"
)


def _expected_tactical_scenarios(players: int) -> tuple[set[str], int]:
  payload = json.loads(TACTICAL_SUITE.read_text(encoding="utf-8"))
  matching = [
      row for row in payload.get("scenarios", [])
      if int(row.get("state", {}).get("numPlayers", -1)) == int(players)
  ]
  rotations = payload.get("metamorphicValidation", {}).get(
      "seatRotations", [0]
  )
  if rotations == "all":
    rotations = list(range(int(players)))
  ids = set()
  for scenario in matching:
    root = str(scenario["id"])
    for rotation in rotations:
      rotation = int(rotation)
      ids.add(root if rotation == 0 else f"{root}--seat-r{rotation}")
  return ids, len(matching)


def _expected_bidding_scenarios(players: int) -> tuple[set[str], int]:
  payload = json.loads(BIDDING_SUITE.read_text(encoding="utf-8"))
  matching = [
      row for row in payload.get("scenarios", [])
      if int(row.get("state", {}).get("numPlayers", -1)) == int(players)
  ]
  ids = {
      str(row["id"]) if rotation == 0
      else f"{row['id']}--seat-r{rotation}"
      for row in matching for rotation in range(int(players))
  }
  return ids, len(matching)


def _expected_bid_regime_scenarios(players: int) -> tuple[set[str], int]:
  payload = json.loads(BID_REGIME_SUITE.read_text(encoding="utf-8"))
  matching = [
      row for row in payload.get("scenarios", [])
      if int(row.get("state", {}).get("numPlayers", -1)) == int(players)
  ]
  ids = {
      str(row["id"]) if rotation == 0
      else f"{row['id']}--seat-r{rotation}"
      for row in matching for rotation in range(int(players))
  }
  return ids, len(matching)


def _wilson(successes: int, trials: int) -> list[float]:
  if trials <= 0:
    return [0.0, 1.0]
  z = 1.959963984540054
  estimate = successes / trials
  denominator = 1.0 + z * z / trials
  center = (estimate + z * z / (2.0 * trials)) / denominator
  half = z * math.sqrt(
      estimate * (1.0 - estimate) / trials
      + z * z / (4.0 * trials * trials)
  ) / denominator
  return [center - half, center + half]


def summarize(
    out_dir: Path,
    expected_gates: int,
    min_matches: int,
    threshold: float | None = None,
    min_independent_deals: int | None = None,
    curated_tactical: Path | None = None,
    curated_bidding: Path | None = None,
    curated_bid_regime: Path | None = None,
) -> dict:
  paths = sorted(out_dir.glob("gate_*/summary.json"))
  errors: list[str] = []
  if expected_gates < 3:
    errors.append(f"expected gates {expected_gates} < formal minimum 3")
  if int(min_matches) < 600:
    errors.append(f"minimum matches {min_matches} < formal minimum 600")
  if len(paths) != expected_gates:
    errors.append(
        f"found {len(paths)} gate summaries, expected {expected_gates}"
    )
  gates = []
  checkpoint_shas = set()
  modes = set()
  led_modes = set()
  player_counts = set()
  incumbent_shas = set()
  incumbent_modes = set()
  candidate_specs = set()
  gate_requires_bid_regime = False
  seeds = []
  led_seeds = []
  pooled_paradox = 0
  pooled_rounds = 0
  pooled_competitive_deals = 0
  weighted_placement_score = 0.0
  for index, path in enumerate(paths, start=1):
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate_requires_bid_regime = gate_requires_bid_regime or bool(
        (payload.get("curated_bid_regime_gate") or {}).get("required")
    )
    gate = payload.get("competitive_promotion_gate") or {}
    diagnostic = payload.get("homogeneous_paradox_gate") or {}
    matches = int(gate.get("matches") or 0)
    deals = int(gate.get("independent_deal_blocks") or 0)
    rounds = int(diagnostic.get("hand_rounds") or 0)
    paradox = int(diagnostic.get("rounds_with_any_paradox") or 0)
    interval = (
        (gate.get("placement_advantage") or {}).get("ci95")
        or (gate.get("pairwise") or {}).get("score_rate_ci95")
        or []
    )
    lower = float(interval[0]) if len(interval) == 2 else -math.inf
    score_rate = (gate.get("pairwise") or {}).get("score_rate")
    score_interval = (gate.get("official_score_advantage") or {}).get(
        "ci95", []
    )
    score_upper = (
        float(score_interval[1])
        if len(score_interval) == 2 and score_interval[1] is not None
        else -math.inf
    )
    own_interval = (gate.get("own_paradox_rate_difference") or {}).get(
        "ci95", []
    )
    own_lower = (
        float(own_interval[0])
        if len(own_interval) == 2 and own_interval[0] is not None
        else math.inf
    )
    own_regression_limit = float(
        gate.get("max_own_paradox_rate_regression", math.inf)
    )
    seed = payload.get("seed")
    led_seed = payload.get("led_seed")
    checkpoint_shas.add(str(payload.get("checkpoint_sha256") or ""))
    modes.add(str(payload.get("mode") or ""))
    candidate_specs.add(
        f"{payload.get('checkpoint') or ''}:{payload.get('mode') or ''}"
    )
    led_modes.add(str(payload.get("led_mode") or ""))
    player_counts.add(int(payload.get("players") or 0))
    incumbent_shas.add(str(payload.get("incumbent_checkpoint_sha256") or ""))
    incumbent_modes.add(str(payload.get("incumbent_mode") or ""))
    seeds.append(seed)
    led_seeds.append(led_seed)
    gate_errors = []
    if matches < min_matches:
      gate_errors.append(f"matches {matches} < {min_matches}")
    players = int(payload.get("players") or 0)
    required_deals = (
        int(min_independent_deals)
        if min_independent_deals is not None
        else max(1, math.ceil(int(min_matches) / max(1, players)))
    )
    hard_minimum_deals = math.ceil(600 / max(1, players))
    if required_deals < hard_minimum_deals:
      gate_errors.append(
          f"required independent deal blocks {required_deals} "
          f"< formal minimum {hard_minimum_deals}"
      )
    if deals < required_deals:
      gate_errors.append(
          f"independent deal blocks {deals} < {required_deals}"
      )
    if seed is None:
      gate_errors.append("missing seed")
    if led_seed is None:
      gate_errors.append("missing led seed")
    if gate.get("eligible") is not True:
      gate_errors.append("competitive gate is not eligible")
    if gate.get("seat_balanced_within_each_deal") is not True:
      gate_errors.append("candidate seats are not balanced within each deal")
    if gate.get("passed") is not True:
      gate_errors.append("competitive gate passed is not true")
    if gate.get("placement_advantage_passed") is not True:
      gate_errors.append("placement advantage did not pass")
    if not lower > 0.5:
      gate_errors.append(f"placement CI lower {lower} is not above 0.5")
    if gate.get("official_score_guard_passed") is not True:
      gate_errors.append("official match score guard did not pass")
    if score_upper < 0.0:
      gate_errors.append("official match score CI shows significant regression")
    if gate.get("own_paradox_guard_passed") is not True:
      gate_errors.append("candidate personal paradox guard did not pass")
    if not math.isclose(own_regression_limit, 0.0, abs_tol=1e-12):
      gate_errors.append(
          f"own paradox regression limit {own_regression_limit} must equal 0"
      )
    if own_lower > 0.0:
      gate_errors.append(
          f"own paradox regression CI lower {own_lower} exceeds 0"
      )
    if gate_errors:
      errors.extend(f"gate {index}: {item}" for item in gate_errors)
    pooled_paradox += paradox
    pooled_rounds += rounds
    if score_rate is not None:
      pooled_competitive_deals += deals
      weighted_placement_score += float(score_rate) * deals
    gates.append({
        "index": index,
        "summary": str(path),
        "seed": seed,
        "led_seed": led_seed,
        "matches": matches,
        "independent_deal_blocks": deals,
        "placement_score_rate": score_rate,
        "placement_score_ci95": interval,
        "hand_rounds": rounds,
        "rounds_with_any_paradox": paradox,
        "table_paradox_diagnostic_only": True,
        "passed": not gate_errors,
        "errors": gate_errors,
    })
  if len(checkpoint_shas) != 1 or "" in checkpoint_shas:
    errors.append("gate checkpoint hashes are missing or inconsistent")
  if len(modes) != 1 or "" in modes:
    errors.append("gate selector modes are missing or inconsistent")
  if len(led_modes) != 1 or "" in led_modes:
    errors.append("gate led-audit modes are missing or inconsistent")
  if len(player_counts) != 1 or 0 in player_counts:
    errors.append("gate player counts are missing or inconsistent")
  if len(incumbent_shas) != 1 or "" in incumbent_shas:
    errors.append("gate frozen incumbent hashes are missing or inconsistent")
  if len(incumbent_modes) != 1 or "" in incumbent_modes:
    errors.append("gate frozen incumbent modes are missing or inconsistent")
  if len(modes) == 1 and len(led_modes) == 1 and modes != led_modes:
    errors.append("gate led-audit mode does not match gameplay mode")
  bidding_report = None
  expected_bidding_ids = set()
  expected_bidding_families = 0
  if len(player_counts) == 1 and 0 not in player_counts:
    players = next(iter(player_counts))
    expected_bidding_ids, expected_bidding_families = (
        _expected_bidding_scenarios(players)
    )
    if curated_bidding is None:
      curated_bidding = Path(out_dir) / "curated_bidding_validation.json"
    if not Path(curated_bidding).is_file():
      errors.append("missing required curated bidding validation")
    else:
      bidding_report = json.loads(
          Path(curated_bidding).read_text(encoding="utf-8")
      )
      scenario_count = int(bidding_report.get("scenarioCount") or 0)
      results = bidding_report.get("results") or []
      if (
          bidding_report.get("suiteID")
          != "quantum-cat-curated-bidding-validation-v1"
          or bidding_report.get("mode") != "candidate_bid_validation"
      ):
        errors.append("curated bidding suite is not a recognized candidate run")
      if (
          bidding_report.get("passed") is not True
          or int(bidding_report.get("passedCount") or 0) != scenario_count
          or scenario_count != len(expected_bidding_ids)
          or len(results) != scenario_count
      ):
        errors.append("curated bidding fixtures failed or are incomplete")
      if any(
          row.get("passed") is not True
          or int(row.get("players") or 0) != players
          for row in results
      ):
        errors.append("one or more curated bidding hands failed")
      if {str(row.get("id") or "") for row in results} != expected_bidding_ids:
        errors.append("curated bidding fixture identities are incomplete")
      if int(bidding_report.get("independentScenarioFamilyCount") or 0) != expected_bidding_families:
        errors.append("curated bidding scenario families are incomplete")
      if (
          len(candidate_specs) != 1
          or bidding_report.get("candidate") not in candidate_specs
      ):
        errors.append("curated bidding candidate does not match gated model")
  tactical_report = None
  expected_tactical_ids = set()
  expected_tactical_families = 0
  if len(player_counts) == 1 and 0 not in player_counts:
    expected_tactical_ids, expected_tactical_families = (
        _expected_tactical_scenarios(next(iter(player_counts)))
    )
  if expected_tactical_ids:
    players = next(iter(player_counts))
    player_name = {3: "three-player", 4: "four-player", 5: "five-player"}.get(
        players, f"{players}-player"
    )
    if curated_tactical is None or not Path(curated_tactical).is_file():
      errors.append(f"missing required {player_name} curated tactical validation")
    else:
      tactical_report = json.loads(
          Path(curated_tactical).read_text(encoding="utf-8")
      )
      scenario_count = int(tactical_report.get("scenarioCount") or 0)
      passed_count = int(tactical_report.get("passedCount") or 0)
      results = tactical_report.get("results") or []
      if tactical_report.get("passed") is not True:
        errors.append(f"{player_name} curated tactical validation did not pass")
      if (
          tactical_report.get("suiteID")
          != "quantum-cat-curated-tactical-validation-v1"
      ):
        errors.append(f"{player_name} curated tactical suite is not recognized")
      if (
          scenario_count != len(expected_tactical_ids)
          or passed_count != scenario_count
      ):
        errors.append(f"{player_name} curated tactical fixtures are incomplete")
      if len(results) != scenario_count or any(
          row.get("passed") is not True
          or int(row.get("players") or 0) != players
          for row in results
      ):
        errors.append(f"one or more {player_name} tactical fixtures failed")
      if {str(row.get("id") or "") for row in results} != expected_tactical_ids:
        errors.append(f"{player_name} tactical fixture identities are incomplete")
      reported_families = tactical_report.get("independentScenarioFamilyCount")
      if (
          reported_families is not None
          and int(reported_families) != expected_tactical_families
      ):
        errors.append(f"{player_name} tactical scenario families are incomplete")
      if (
          len(candidate_specs) != 1
          or tactical_report.get("candidate") not in candidate_specs
      ):
        errors.append("curated tactical candidate does not match gated model")
  bid_regime_report = None
  expected_bid_regime_ids = set()
  expected_bid_regime_families = 0
  if len(player_counts) == 1 and 0 not in player_counts:
    players = next(iter(player_counts))
    expected_bid_regime_ids, expected_bid_regime_families = (
        _expected_bid_regime_scenarios(players)
    )
    if curated_bid_regime is not None or gate_requires_bid_regime:
      if curated_bid_regime is None:
        curated_bid_regime = Path(out_dir) / "curated_bid_regime_validation.json"
      if not Path(curated_bid_regime).is_file():
        errors.append("missing required curated bid-regime validation")
      else:
        bid_regime_report = json.loads(
            Path(curated_bid_regime).read_text(encoding="utf-8")
        )
        scenario_count = int(bid_regime_report.get("scenarioCount") or 0)
        results = bid_regime_report.get("results") or []
        if (
            bid_regime_report.get("suiteID")
            != "quantum-cat-curated-bid-regime-validation-v1"
        ):
          errors.append("curated bid-regime suite is not recognized")
        if (
            bid_regime_report.get("passed") is not True
            or int(bid_regime_report.get("passedCount") or 0) != scenario_count
            or scenario_count != len(expected_bid_regime_ids)
            or len(results) != scenario_count
        ):
          errors.append("curated bid-regime fixtures failed or are incomplete")
        if any(
            row.get("passed") is not True
            or int(row.get("players") or 0) != players
            for row in results
        ):
          errors.append("one or more curated bid-regime decisions failed")
        if (
            {str(row.get("id") or "") for row in results}
            != expected_bid_regime_ids
        ):
          errors.append("curated bid-regime fixture identities are incomplete")
        if (
            int(bid_regime_report.get("independentScenarioFamilyCount") or 0)
            != expected_bid_regime_families
        ):
          errors.append("curated bid-regime scenario families are incomplete")
        if (
            len(candidate_specs) != 1
            or bid_regime_report.get("candidate") not in candidate_specs
        ):
          errors.append("curated bid-regime candidate does not match gated model")
  concrete_seeds = [int(seed) for seed in seeds if seed is not None]
  if len(concrete_seeds) != len(set(concrete_seeds)):
    errors.append("gate seeds are not unique")
  concrete_led_seeds = [int(seed) for seed in led_seeds if seed is not None]
  if len(concrete_led_seeds) != len(set(concrete_led_seeds)):
    errors.append("gate led seeds are not unique")
  if not set(concrete_seeds).isdisjoint(concrete_led_seeds):
    errors.append("gameplay and led-audit seeds overlap")
  pooled_interval = _wilson(pooled_paradox, pooled_rounds)
  return {
      "kind": "independent_competitive_model_gate_summary_v2",
      "passed": not errors,
      "required_all_gates_pass": True,
      "expected_gates": expected_gates,
      "min_matches_per_gate": min_matches,
      "legacy_table_paradox_threshold": threshold,
      "table_paradox_is_promotion_veto": False,
      "checkpoint_sha256": next(iter(checkpoint_shas), None),
      "incumbent_checkpoint_sha256": next(iter(incumbent_shas), None),
      "incumbent_mode": next(iter(incumbent_modes), None),
      "mode": next(iter(modes), None),
      "led_mode": next(iter(led_modes), None),
      "players": next(iter(player_counts), None),
      "curated_bidding": {
          "required": bool(expected_bidding_ids),
          "artifact": str(curated_bidding) if curated_bidding else None,
          "passed": (
              bidding_report.get("passed")
              if bidding_report is not None else None
          ),
          "scenario_count": (
              bidding_report.get("scenarioCount")
              if bidding_report is not None else None
          ),
          "expected_scenario_count": len(expected_bidding_ids),
          "expected_family_count": expected_bidding_families,
      },
      "curated_tactical": {
          "required": bool(expected_tactical_ids),
          "artifact": str(curated_tactical) if curated_tactical else None,
          "passed": (
              tactical_report.get("passed")
              if tactical_report is not None else None
          ),
          "scenario_count": (
              tactical_report.get("scenarioCount")
              if tactical_report is not None else None
          ),
          "expected_scenario_count": len(expected_tactical_ids),
          "expected_family_count": expected_tactical_families,
      },
      "curated_bid_regime": {
          "required": bool(curated_bid_regime is not None or gate_requires_bid_regime),
          "artifact": str(curated_bid_regime) if curated_bid_regime else None,
          "passed": (
              bid_regime_report.get("passed")
              if bid_regime_report is not None else None
          ),
          "scenario_count": (
              bid_regime_report.get("scenarioCount")
              if bid_regime_report is not None else None
          ),
          "expected_scenario_count": len(expected_bid_regime_ids),
          "expected_family_count": expected_bid_regime_families,
      },
      "gates": gates,
      "pooled": {
          "independent_competitive_deals": pooled_competitive_deals,
          "competitive_placement_score_rate": (
              weighted_placement_score / pooled_competitive_deals
              if pooled_competitive_deals else None
          ),
          "rounds_with_any_paradox": pooled_paradox,
          "hand_rounds": pooled_rounds,
          "point_estimate": (
              pooled_paradox / pooled_rounds if pooled_rounds else None
          ),
          "wilson_ci95": pooled_interval,
          "informational_only": True,
      },
      "errors": errors,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--expected-gates", type=int, required=True)
  parser.add_argument("--min-matches", type=int, default=600)
  parser.add_argument("--min-independent-deals", type=int)
  parser.add_argument("--curated-tactical", type=Path)
  parser.add_argument("--curated-bidding", type=Path)
  parser.add_argument("--curated-bid-regime", type=Path)
  parser.add_argument(
      "--threshold",
      type=float,
      help="Legacy table-wide diagnostic threshold; never used for promotion.",
  )
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  report = summarize(
      Path(args.out_dir), int(args.expected_gates), int(args.min_matches),
      args.threshold,
      args.min_independent_deals,
      args.curated_tactical,
      args.curated_bidding,
      args.curated_bid_regime,
  )
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if report["passed"] else 3


if __name__ == "__main__":
  raise SystemExit(main())
