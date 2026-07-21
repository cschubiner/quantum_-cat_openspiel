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
PARADOX_THRESHOLD="${PARADOX_THRESHOLD:-0.40}"
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

mkdir -p "$OUT_DIR"
CHECKPOINT_SHA="$(shasum -a 256 "$CHECKPOINT" | awk '{print $1}')"

case "$LED_MODE" in
  policy|q_policy)
    ;;
  *)
    echo "Led-suit audit supports only policy/q_policy; using LED_MODE=policy for that audit." >&2
    LED_MODE="policy"
    ;;
esac

bots=()
neural_args=()
for ((seat = 0; seat < PLAYERS; seat++)); do
  name="c${seat}"
  bots+=("$name")
  neural_args+=(--neural-bot "${name}=${CHECKPOINT}:${MODE}")
done

paradox_out="$OUT_DIR/homogeneous_paradox_m${PARADOX_MATCHES}_seed${SEED}.json"
led_out="$OUT_DIR/led_switch_m${LED_MATCHES}_seed${LED_SEED}.json"
summary_out="$OUT_DIR/summary.json"

echo "== Quantum Cat model test =="
echo "checkpoint=$CHECKPOINT"
echo "checkpoint_sha256=$CHECKPOINT_SHA"
echo "mode=$MODE led_mode=$LED_MODE players=$PLAYERS backend=pytorch"
echo "out_dir=$OUT_DIR"

echo "== Homogeneous paradox gate =="
PYTHONPATH=. "$PYTHON" quantum_cat_full_match_elo.py \
  --players="$PLAYERS" \
  --matches="$PARADOX_MATCHES" \
  --bots "${bots[@]}" \
  "${neural_args[@]}" \
  --match-context \
  --schedule=permutations \
  --homogeneous-paradox-threshold="$PARADOX_THRESHOLD" \
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

PYTHONPATH=. "$PYTHON" "$SCRIPT_DIR/summarize_model_gates.py" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha "$CHECKPOINT_SHA" \
  --mode "$MODE" \
  --led-mode "$LED_MODE" \
  --paradox "$paradox_out" \
  --led "$led_out" \
  --out "$summary_out"

echo "summary=$summary_out"
