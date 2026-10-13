#!/bin/bash
# usage: run_vs_champ.sh <ext_bot> <deals> <deal_offset> <seed> <out.json> [champ=1051|1052] [workers]
EXT=$1; DEALS=$2; OFF=$3; SEED=$4; OUT=$5; WHICH=${6:-1051}; W=${7:-10}
CK=/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1049/real_bid_raw_grounded/checkpoint_0000_loaded_replay_best.pt
MINR=4; [ "$WHICH" = "1052" ] && MINR=0
SPEC="$CK:root_rollout?root_rollout_base_mode=competitive_tactics,root_rollout_preserve_tactical_overrides=true,root_rollout_preserve_own_liveness=true,root_rollout_continuation_mode=competitive_tactics,root_rollout_phases=play,root_rollout_min_trick_number=4,root_rollout_max_trick_number=6,root_rollout_max_plies=0,root_rollout_samples=8,root_rollouts=1,root_rollout_max_actions=2,root_rollout_top_policy=2,root_rollout_objective=paradox_then_score,root_rollout_paradox_scope=acting,root_rollout_audit_samples=8,root_rollout_audit_rollouts=1,root_rollout_audit_passes=1,belief_source=infostate,root_rollout_min_match_round=$MINR,root_rollout_max_match_round=4,root_rollout_full_match=true,competitive_tactics_base_mode=q_policy,competitive_tactics_bid_regime=true,action_value_selection_weight=1.0,action_feasibility_selection_weight=2.0,action_paradox_rerank_mode=relative,action_paradox_min_risk_margin=0.02,action_paradox_max_policy_log_gap=2.0,neural_prediction_mode=model,root_rollout_min_audit_score_improvement=0"
DIR=$(cd "$(dirname "$0")" && pwd)
cd /Users/canal/quantum_-cat_openspiel
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python "$DIR/ext_arena.py" \
  --players 5 --matches $((DEALS*5)) --paired-deals $DEALS --deal-offset $OFF --schedule cyclic \
  --bots $EXT c0 c1 c2 c3 \
  --neural-bot "c0=$SPEC" --neural-bot "c1=$SPEC" --neural-bot "c2=$SPEC" --neural-bot "c3=$SPEC" \
  --competitive-gate-candidate $EXT --competitive-gate-opponents c0,c1,c2,c3 \
  --competitive-min-independent-deals $DEALS --competitive-max-own-paradox-regression 0 \
  --match-context --workers $W --worker-torch-threads 1 --sample-match-limit $((DEALS*5)) \
  --first-paradox-summary-sample-limit 0 --seed $SEED --out "$OUT"
