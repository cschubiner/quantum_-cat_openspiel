#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${CHECKPOINT:-${1:-az_runs/best_3p_policy_checkpoint.pt}}"
MODE="${MODE:-policy}"
LED_MODE="${LED_MODE:-$MODE}"
PLAYERS="${PLAYERS:-3}"
INDEPENDENT_GATES="${INDEPENDENT_GATES:-3}"
FORMAL_MIN_MATCHES="${FORMAL_MIN_MATCHES:-600}"
FORMAL_MIN_INDEPENDENT_DEALS="${FORMAL_MIN_INDEPENDENT_DEALS:-$(((FORMAL_MIN_MATCHES + PLAYERS - 1) / PLAYERS))}"
COMPETITIVE_DEALS="${COMPETITIVE_DEALS:-$FORMAL_MIN_INDEPENDENT_DEALS}"
PARADOX_MATCHES="${PARADOX_MATCHES:-60}"
PARADOX_THRESHOLD="${PARADOX_THRESHOLD:-0.40}"
BASE_SEED="${BASE_SEED:-2028072100}"
OUT_DIR="${OUT_DIR:-az_runs/independent_model_gates_$(date -u +%Y%m%dT%H%M%SZ)}"
CURATED_TACTICAL_SUITE="${CURATED_TACTICAL_SUITE:-experiments/quantum_cat_curated_tactical_validation.json}"
CURATED_BIDDING_SUITE="${CURATED_BIDDING_SUITE:-experiments/quantum_cat_curated_bidding_validation.json}"
CURATED_BID_REGIME_SUITE="${CURATED_BID_REGIME_SUITE:-experiments/quantum_cat_curated_bid_regime_validation.json}"

if (( INDEPENDENT_GATES < 3 )); then
  echo "Formal promotion requires at least 3 independent gates." >&2
  exit 2
fi
if (( FORMAL_MIN_MATCHES < 600 )); then
  echo "Formal promotion requires at least 600 seat-rotated matches per gate." >&2
  exit 2
fi
if (( FORMAL_MIN_INDEPENDENT_DEALS < (600 + PLAYERS - 1) / PLAYERS )); then
  echo "Formal promotion requires at least $(((600 + PLAYERS - 1) / PLAYERS)) independent deals." >&2
  exit 2
fi
if (( COMPETITIVE_DEALS < FORMAL_MIN_INDEPENDENT_DEALS )); then
  echo "Formal gates require COMPETITIVE_DEALS >= $FORMAL_MIN_INDEPENDENT_DEALS." >&2
  exit 2
fi
if (( COMPETITIVE_DEALS * PLAYERS < FORMAL_MIN_MATCHES )); then
  echo "Formal gates require at least $FORMAL_MIN_MATCHES seat-rotated matches." >&2
  exit 2
fi
if [[ "$LED_MODE" != "$MODE" ]]; then
  echo "Formal gates require LED_MODE to exactly match MODE." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
curated_args=()
if [[ ! -f "$CURATED_BIDDING_SUITE" ]]; then
  echo "Locked bidding validation suite not found: $CURATED_BIDDING_SUITE" >&2
  exit 2
fi
bidding_out="$OUT_DIR/curated_bidding_validation.json"
PYTHONPATH=. "${PYTHON:-.venv/bin/python}" \
  quantum_cat_curated_bidding_validation.py \
  --suite "$CURATED_BIDDING_SUITE" \
  --players "$PLAYERS" \
  --candidate "${CHECKPOINT}:${MODE}" \
  --quiet \
  --out "$bidding_out"
curated_args+=(--curated-bidding "$bidding_out")
if [[ ! -f "$CURATED_BID_REGIME_SUITE" ]]; then
  echo "Locked bid-regime validation suite not found: $CURATED_BID_REGIME_SUITE" >&2
  exit 2
fi
bid_regime_out="$OUT_DIR/curated_bid_regime_validation.json"
PYTHONPATH=. "${PYTHON:-.venv/bin/python}" \
  quantum_cat_curated_validation.py \
  --suite "$CURATED_BID_REGIME_SUITE" \
  --players "$PLAYERS" \
  --candidate "${CHECKPOINT}:${MODE}" \
  --out "$bid_regime_out"
curated_args+=(--curated-bid-regime "$bid_regime_out")
if [[ ! -f "$CURATED_TACTICAL_SUITE" ]]; then
  echo "Locked tactical validation suite not found: $CURATED_TACTICAL_SUITE" >&2
  exit 2
fi
TACTICAL_SCENARIO_COUNT="$("${PYTHON:-.venv/bin/python}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as suite:
    scenarios = json.load(suite).get("scenarios", [])
print(sum(int(row.get("state", {}).get("numPlayers", -1)) == int(sys.argv[2]) for row in scenarios))
' "$CURATED_TACTICAL_SUITE" "$PLAYERS")"
if (( TACTICAL_SCENARIO_COUNT > 0 )); then
  curated_out="$OUT_DIR/curated_tactical_validation.json"
  PYTHONPATH=. "${PYTHON:-.venv/bin/python}" \
    quantum_cat_curated_validation.py \
    --suite "$CURATED_TACTICAL_SUITE" \
    --players "$PLAYERS" \
    --candidate "${CHECKPOINT}:${MODE}" \
    --out "$curated_out"
  curated_args+=(--curated-tactical "$curated_out")
fi
for ((gate = 0; gate < INDEPENDENT_GATES; gate++)); do
  seed=$((BASE_SEED + gate * 100003))
  CHECKPOINT="$CHECKPOINT" \
  MODE="$MODE" \
  LED_MODE="$LED_MODE" \
  PLAYERS="$PLAYERS" \
  INCUMBENT_CHECKPOINT="${INCUMBENT_CHECKPOINT:-}" \
  INCUMBENT_MODE="${INCUMBENT_MODE:-}" \
  SEED="$seed" \
  LED_SEED="$((seed + 101))" \
  COMPETITIVE_DEALS="$COMPETITIVE_DEALS" \
  COMPETITIVE_MIN_INDEPENDENT_DEALS="$FORMAL_MIN_INDEPENDENT_DEALS" \
  MAX_OWN_PARADOX_REGRESSION=0.0 \
  PARADOX_MATCHES="$PARADOX_MATCHES" \
  PARADOX_THRESHOLD="$PARADOX_THRESHOLD" \
  OUT_DIR="$OUT_DIR/gate_$((gate + 1))" \
  REQUIRE_COMPETITIVE_PASS=1 \
  REQUIRE_TACTICAL_PASS=1 \
  REQUIRE_BIDDING_PASS=1 \
  REQUIRE_BID_REGIME_PASS=1 \
  REQUIRE_CI_UPPER_BELOW_THRESHOLD=0 \
    bash "$SCRIPT_DIR/run_mac_model_gates.sh"
done

"${PYTHON:-.venv/bin/python}" \
  "$SCRIPT_DIR/summarize_independent_model_gates.py" \
  --out-dir "$OUT_DIR" \
  --expected-gates "$INDEPENDENT_GATES" \
  --min-matches "$FORMAL_MIN_MATCHES" \
  --min-independent-deals "$FORMAL_MIN_INDEPENDENT_DEALS" \
  "${curated_args[@]}" \
  --out "$OUT_DIR/independent_summary.json"

echo "independent_model_gates: PASS ($INDEPENDENT_GATES/$INDEPENDENT_GATES)"
