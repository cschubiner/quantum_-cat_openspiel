#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

CHECKPOINT="${CHECKPOINT:-${1:-az_runs/best_3p_policy_checkpoint.pt}}"
MODE="${MODE:-policy}"
LED_MODE="${LED_MODE:-$MODE}"
PLAYERS="${PLAYERS:-3}"
PARADOX_MATCHES="${PARADOX_MATCHES:-200}"
LED_MATCHES="${LED_MATCHES:-60}"
case "$PLAYERS" in
  3)
    DEFAULT_INCUMBENT_CHECKPOINT="az_runs/best_3p_policy_checkpoint.pt"
    DEFAULT_INCUMBENT_MODE="policy"
    ;;
  4)
    DEFAULT_INCUMBENT_CHECKPOINT="az_runs/run730_4p_liveness_shield_behavior_distill_20260612/policy_stack_behavior_s100_lr5e5_cpu/checkpoint_0000_loaded_replay_best.pt"
    DEFAULT_INCUMBENT_MODE="liveness_shield?liveness_shield_base_mode=policy,liveness_shield_phases=play,liveness_shield_min_open_slot_delta=0,liveness_shield_min_public_damage_delta=0,liveness_shield_max_policy_log_gap=-1.0,liveness_shield_sample_limit=0"
    ;;
  5)
    DEFAULT_INCUMBENT_CHECKPOINT="az_runs/run600_explicit345_pretrain_normal_20260610/normal_qpolicy_5p_from4p_i1_g4_s8_cpu/checkpoint_0001.pt"
    DEFAULT_INCUMBENT_MODE="q_policy?action_value_selection_weight=1.0,action_feasibility_selection_weight=2.0,action_paradox_rerank_mode=relative,action_paradox_min_risk_margin=0.02,action_paradox_max_policy_log_gap=2.0"
    ;;
  *)
    echo "PLAYERS must be 3, 4, or 5 for a competitive promotion gate." >&2
    exit 2
    ;;
esac
INCUMBENT_CHECKPOINT="${INCUMBENT_CHECKPOINT:-$DEFAULT_INCUMBENT_CHECKPOINT}"
INCUMBENT_MODE="${INCUMBENT_MODE:-$DEFAULT_INCUMBENT_MODE}"
COMPETITIVE_DEALS="${COMPETITIVE_DEALS:-$(((PARADOX_MATCHES + PLAYERS - 1) / PLAYERS))}"
COMPETITIVE_MIN_INDEPENDENT_DEALS="${COMPETITIVE_MIN_INDEPENDENT_DEALS:-$COMPETITIVE_DEALS}"
MAX_OWN_PARADOX_REGRESSION="${MAX_OWN_PARADOX_REGRESSION:-0.0}"
PARADOX_THRESHOLD="${PARADOX_THRESHOLD:-0.40}"
REQUIRE_CI_UPPER_BELOW_THRESHOLD="${REQUIRE_CI_UPPER_BELOW_THRESHOLD:-0}"
REQUIRE_COMPETITIVE_PASS="${REQUIRE_COMPETITIVE_PASS:-auto}"
REQUIRE_TACTICAL_PASS="${REQUIRE_TACTICAL_PASS:-auto}"
REQUIRE_BIDDING_PASS="${REQUIRE_BIDDING_PASS:-auto}"
REQUIRE_BID_REGIME_PASS="${REQUIRE_BID_REGIME_PASS:-auto}"
CURATED_TACTICAL_SUITE="${CURATED_TACTICAL_SUITE:-experiments/quantum_cat_curated_tactical_validation.json}"
CURATED_BIDDING_SUITE="${CURATED_BIDDING_SUITE:-experiments/quantum_cat_curated_bidding_validation.json}"
CURATED_BID_REGIME_SUITE="${CURATED_BID_REGIME_SUITE:-experiments/quantum_cat_curated_bid_regime_validation.json}"
SEED="${SEED:-20280610}"
LED_SEED="${LED_SEED:-$((SEED + 101))}"
WORKERS="${WORKERS:-8}"
WORKER_TORCH_THREADS="${WORKER_TORCH_THREADS:-1}"
BACKEND="${BACKEND:-pytorch}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-az_runs/model_test_${STAMP}}"

case "$BACKEND" in
  auto|pytorch)
    ;;
  mlx)
    if ! ls quantum_cat*mlx*.py >/dev/null 2>&1; then
      echo "BACKEND=mlx requested, but no Quantum Cat MLX evaluator exists in this repo." >&2
      exit 2
    fi
    echo "BACKEND=mlx requested, but this wrapper has no wired MLX gate command yet." >&2
    exit 2
    ;;
  *)
    echo "Unsupported BACKEND=$BACKEND. Use auto, pytorch, or mlx." >&2
    exit 2
    ;;
esac

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$INCUMBENT_CHECKPOINT" ]]; then
  echo "Frozen incumbent checkpoint not found: $INCUMBENT_CHECKPOINT" >&2
  echo "Set INCUMBENT_CHECKPOINT to the player-count-specific incumbent." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
CHECKPOINT_SHA="$(shasum -a 256 "$CHECKPOINT" | awk '{print $1}')"
INCUMBENT_CHECKPOINT_SHA="$(shasum -a 256 "$INCUMBENT_CHECKPOINT" | awk '{print $1}')"
if [[ "$REQUIRE_COMPETITIVE_PASS" == "auto" ]]; then
  if [[ "$CHECKPOINT_SHA" == "$INCUMBENT_CHECKPOINT_SHA" && "$MODE" == "$INCUMBENT_MODE" ]]; then
    REQUIRE_COMPETITIVE_PASS=0
  else
    REQUIRE_COMPETITIVE_PASS=1
  fi
fi
if [[ "$REQUIRE_TACTICAL_PASS" == "auto" ]]; then
  REQUIRE_TACTICAL_PASS="$REQUIRE_COMPETITIVE_PASS"
fi
if [[ "$REQUIRE_BIDDING_PASS" == "auto" ]]; then
  REQUIRE_BIDDING_PASS="$REQUIRE_COMPETITIVE_PASS"
fi
if [[ "$REQUIRE_BID_REGIME_PASS" == "auto" ]]; then
  REQUIRE_BID_REGIME_PASS="$REQUIRE_COMPETITIVE_PASS"
fi
if [[ ! -f "$CURATED_TACTICAL_SUITE" ]]; then
  echo "Locked tactical validation suite not found: $CURATED_TACTICAL_SUITE" >&2
  exit 2
fi
if [[ ! -f "$CURATED_BIDDING_SUITE" ]]; then
  echo "Locked bidding validation suite not found: $CURATED_BIDDING_SUITE" >&2
  exit 2
fi
if [[ ! -f "$CURATED_BID_REGIME_SUITE" ]]; then
  echo "Locked bid-regime tactical suite not found: $CURATED_BID_REGIME_SUITE" >&2
  exit 2
fi
TACTICAL_SCENARIO_COUNT="$("$PYTHON" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as suite:
    scenarios = json.load(suite).get("scenarios", [])
print(sum(int(row.get("state", {}).get("numPlayers", -1)) == int(sys.argv[2]) for row in scenarios))
' "$CURATED_TACTICAL_SUITE" "$PLAYERS")"
if [[ "$REQUIRE_COMPETITIVE_PASS" != "0" && "$REQUIRE_COMPETITIVE_PASS" != "1" ]]; then
  echo "REQUIRE_COMPETITIVE_PASS must be auto, 0, or 1." >&2
  exit 2
fi
if [[ "$REQUIRE_TACTICAL_PASS" != "0" && "$REQUIRE_TACTICAL_PASS" != "1" ]]; then
  echo "REQUIRE_TACTICAL_PASS must be auto, 0, or 1." >&2
  exit 2
fi
if [[ "$REQUIRE_BIDDING_PASS" != "0" && "$REQUIRE_BIDDING_PASS" != "1" ]]; then
  echo "REQUIRE_BIDDING_PASS must be auto, 0, or 1." >&2
  exit 2
fi
if [[ "$REQUIRE_BID_REGIME_PASS" != "0" && "$REQUIRE_BID_REGIME_PASS" != "1" ]]; then
  echo "REQUIRE_BID_REGIME_PASS must be auto, 0, or 1." >&2
  exit 2
fi
if [[ "$REQUIRE_COMPETITIVE_PASS" == "1" && "$REQUIRE_TACTICAL_PASS" != "1" && "$TACTICAL_SCENARIO_COUNT" -gt 0 ]]; then
  echo "A competitive promotion screen cannot disable its locked tactical gate." >&2
  exit 2
fi
if [[ "$REQUIRE_COMPETITIVE_PASS" == "1" && "$REQUIRE_BIDDING_PASS" != "1" ]]; then
  echo "A competitive promotion screen cannot disable its locked bidding gate." >&2
  exit 2
fi
if [[ "$REQUIRE_COMPETITIVE_PASS" == "1" && "$REQUIRE_BID_REGIME_PASS" != "1" ]]; then
  echo "A competitive promotion screen cannot disable its locked bid-regime gate." >&2
  exit 2
fi

if [[ -z "$LED_MODE" ]]; then
  echo "LED_MODE must name the selector payload to audit." >&2
  exit 2
fi

bots=()
neural_args=()
for ((seat = 0; seat < PLAYERS; seat++)); do
  name="c${seat}"
  bots+=("$name")
  neural_args+=(--neural-bot "${name}=${CHECKPOINT}:${MODE}")
done

competitive_bots=("challenger")
competitive_neural_args=(
  --neural-bot "challenger=${CHECKPOINT}:${MODE}"
)
competitive_opponents_csv=""
for ((seat = 0; seat < PLAYERS - 1; seat++)); do
  name="inc${seat}"
  competitive_bots+=("$name")
  competitive_neural_args+=(
    --neural-bot "${name}=${INCUMBENT_CHECKPOINT}:${INCUMBENT_MODE}"
  )
  if [[ -n "$competitive_opponents_csv" ]]; then
    competitive_opponents_csv+=","
  fi
  competitive_opponents_csv+="$name"
done

paradox_out="$OUT_DIR/homogeneous_paradox_m${PARADOX_MATCHES}_seed${SEED}.json"
competitive_out="$OUT_DIR/competitive_d${COMPETITIVE_DEALS}_seed${SEED}.json"
led_out="$OUT_DIR/led_switch_m${LED_MATCHES}_seed${LED_SEED}.json"
summary_out="$OUT_DIR/summary.json"
tactical_out="$OUT_DIR/curated_tactical_validation.json"
bidding_out="$OUT_DIR/curated_bidding_validation.json"
bid_regime_out="$OUT_DIR/curated_bid_regime_validation.json"

echo "== Quantum Cat model test =="
echo "checkpoint=$CHECKPOINT"
echo "checkpoint_sha256=$CHECKPOINT_SHA"
echo "incumbent_checkpoint_sha256=$INCUMBENT_CHECKPOINT_SHA"
echo "incumbent_checkpoint=$INCUMBENT_CHECKPOINT incumbent_mode=$INCUMBENT_MODE"
echo "mode=$MODE led_mode=$LED_MODE players=$PLAYERS backend=pytorch"
echo "out_dir=$OUT_DIR"

echo "== Locked hand-strength bidding preflight =="
if ! PYTHONPATH=. "$PYTHON" quantum_cat_curated_bidding_validation.py \
  --suite "$CURATED_BIDDING_SUITE" \
  --players "$PLAYERS" \
  --candidate "${CHECKPOINT}:${MODE}" \
  --quiet \
  --out "$bidding_out"; then
  if [[ "$REQUIRE_BIDDING_PASS" == "1" ]]; then
    echo "Candidate failed a locked bidding hand; full-match evaluation was not started." >&2
    exit 5
  fi
  echo "Locked bidding failures retained for exploratory diagnostics; candidate is not promotable." >&2
fi

echo "== Locked below/exact/above bid-regime preflight =="
if ! PYTHONPATH=. "$PYTHON" quantum_cat_curated_validation.py \
  --suite "$CURATED_BID_REGIME_SUITE" \
  --players "$PLAYERS" \
  --candidate "${CHECKPOINT}:${MODE}" \
  --out "$bid_regime_out"; then
  if [[ "$REQUIRE_BID_REGIME_PASS" == "1" ]]; then
    echo "Candidate failed a locked bid-regime scenario; full-match evaluation was not started." >&2
    exit 6
  fi
  echo "Bid-regime failures retained for exploratory diagnostics; candidate is not promotable." >&2
fi

if (( TACTICAL_SCENARIO_COUNT > 0 )); then
  echo "== Locked strategic-move preflight =="
  if ! PYTHONPATH=. "$PYTHON" quantum_cat_curated_validation.py \
    --suite "$CURATED_TACTICAL_SUITE" \
    --players "$PLAYERS" \
    --candidate "${CHECKPOINT}:${MODE}" \
    --out "$tactical_out"; then
    if [[ "$REQUIRE_TACTICAL_PASS" == "1" ]]; then
      echo "Candidate failed a locked tactical position; full-match evaluation was not started." >&2
      exit 4
    fi
    echo "Locked tactical failures retained for exploratory diagnostics; candidate is not promotable." >&2
  fi
else
  echo "== No locked strategic-move fixtures exist for ${PLAYERS} players =="
fi

echo "== Competitive seat-balanced promotion gate =="
PYTHONPATH=. "$PYTHON" quantum_cat_full_match_elo.py \
  --players="$PLAYERS" \
  --matches="$((COMPETITIVE_DEALS * PLAYERS))" \
  --bots "${competitive_bots[@]}" \
  "${competitive_neural_args[@]}" \
  --match-context \
  --schedule=cyclic \
  --paired-deals="$COMPETITIVE_DEALS" \
  --competitive-gate-candidate=challenger \
  --competitive-gate-opponents="$competitive_opponents_csv" \
  --competitive-min-independent-deals="$COMPETITIVE_MIN_INDEPENDENT_DEALS" \
  --competitive-max-own-paradox-regression="$MAX_OWN_PARADOX_REGRESSION" \
  --sample-match-limit=0 \
  --first-paradox-summary-sample-limit=0 \
  --liveness-shield-sample-limit=0 \
  --seed="$SEED" \
  --workers="$WORKERS" \
  --worker-torch-threads="$WORKER_TORCH_THREADS" \
  --out "$competitive_out"

echo "== Homogeneous table-paradox diagnostic (not a promotion veto) =="
PYTHONPATH=. "$PYTHON" quantum_cat_full_match_elo.py \
  --players="$PLAYERS" \
  --matches="$PARADOX_MATCHES" \
  --bots "${bots[@]}" \
  "${neural_args[@]}" \
  --match-context \
  --schedule=permutations \
  --homogeneous-paradox-threshold="$PARADOX_THRESHOLD" \
  --sample-match-limit=0 \
  --first-paradox-summary-sample-limit=0 \
  --liveness-shield-sample-limit=0 \
  --seed="$SEED" \
  --workers="$WORKERS" \
  --worker-torch-threads="$WORKER_TORCH_THREADS" \
  --out "$paradox_out"

echo "== Led-suit switch audit =="
PYTHONPATH=. "$PYTHON" quantum_cat_led_switch_audit.py \
  --players "$PLAYERS" \
  --matches "$LED_MATCHES" \
  --checkpoint "$CHECKPOINT" \
  --mode "$LED_MODE" \
  --match-context \
  --seed "$LED_SEED" \
  --out "$led_out"

summary_args=(
  --checkpoint "$CHECKPOINT"
  --checkpoint-sha "$CHECKPOINT_SHA"
  --incumbent-checkpoint "$INCUMBENT_CHECKPOINT"
  --incumbent-checkpoint-sha "$INCUMBENT_CHECKPOINT_SHA"
  --incumbent-mode "$INCUMBENT_MODE"
  --mode "$MODE"
  --led-mode "$LED_MODE"
  --players "$PLAYERS"
  --seed "$SEED"
  --led-seed "$LED_SEED"
  --competitive "$competitive_out"
  --curated-bidding "$bidding_out"
  --curated-bid-regime "$bid_regime_out"
  --paradox "$paradox_out"
  --led "$led_out"
  --out "$summary_out"
)
if (( TACTICAL_SCENARIO_COUNT > 0 )); then
  summary_args+=(--curated-tactical "$tactical_out")
fi
if [[ "$REQUIRE_COMPETITIVE_PASS" == "1" ]]; then
  summary_args+=(--require-competitive-pass)
fi
if [[ "$REQUIRE_TACTICAL_PASS" == "1" && "$TACTICAL_SCENARIO_COUNT" -gt 0 ]]; then
  summary_args+=(--require-tactical-pass)
fi
if [[ "$REQUIRE_BIDDING_PASS" == "1" ]]; then
  summary_args+=(--require-bidding-pass)
fi
if [[ "$REQUIRE_BID_REGIME_PASS" == "1" ]]; then
  summary_args+=(--require-bid-regime-pass)
fi
if [[ "$REQUIRE_CI_UPPER_BELOW_THRESHOLD" == "1" ]]; then
  summary_args+=(--require-ci-upper-below-threshold)
fi
PYTHONPATH=. "$PYTHON" "$SCRIPT_DIR/summarize_model_gates.py" \
  "${summary_args[@]}"

echo "summary=$summary_out"
