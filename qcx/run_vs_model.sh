#!/bin/bash
# usage: run_vs_model.sh <players> <opponent_spec> <ext_bot> <deals> <deal_offset> <seed> <out.json> [workers]
P=$1; SPEC=$2; EXT=$3; DEALS=$4; OFF=$5; SEED=$6; OUT=$7; W=${8:-10}
DIR=$(cd "$(dirname "$0")" && pwd)
BOTS="$EXT"; NB=(); OPP=""
for i in $(seq 0 $((P-2))); do BOTS="$BOTS c$i"; NB+=(--neural-bot "c$i=$SPEC"); OPP="$OPP,c$i"; done
cd /Users/canal/quantum_-cat_openspiel
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python "$DIR/ext_arena.py" \
  --players $P --matches $((DEALS*P)) --paired-deals $DEALS --deal-offset $OFF --schedule cyclic \
  --bots $BOTS "${NB[@]}" \
  --competitive-gate-candidate $EXT --competitive-gate-opponents ${OPP#,} \
  --competitive-min-independent-deals $DEALS --competitive-max-own-paradox-regression 0 \
  --workers $W --worker-torch-threads 1 --sample-match-limit $((DEALS*P)) \
  --first-paradox-summary-sample-limit 0 --seed $SEED --out "$OUT"
