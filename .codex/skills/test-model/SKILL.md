---
name: test-model
description: Use this skill when testing Quantum Cat ML checkpoints or champions in this repository, including fast Mac-side PyTorch/MLX model gates, homogeneous paradox-rate checks, led-suit follow/switch audits, red-switch rates, and iPhone install/CoreML bridge validation before claiming a model is correctly deployed.
---

# Quantum Cat Model Testing

Use this skill before promoting or installing a Quantum Cat model, and whenever the user asks for model stats such as paradox rate, follow-led behavior, off-led red-switch rate, or iPhone bridge correctness.

## Core Rules

- Prefer Mac-side fast evaluation for statistical evidence. Do not use long simulator or physical-device runs for paradox-rate estimates.
- The primary survival metric is `rounds_with_any_paradox / hand_rounds`; the strict target is fewer than `40%`.
- Keep selector mode explicit. If the promoted note says raw policy, use `MODE=policy`; do not apply q-policy/risk/value rerankers unless the user explicitly asks for that variant.
- If the promoted note says liveness shield, use the full selector string, for example `MODE='liveness_shield?liveness_shield_base_mode=policy,liveness_shield_phases=discard|prediction|play,liveness_shield_min_open_slot_delta=0,liveness_shield_min_public_damage_delta=0,liveness_shield_max_policy_log_gap=-1.0'`.
- Treat checkpoint hashes and model paths as time-sensitive. Verify the actual file hash before reporting.
- For iPhone installs, do not claim exact PyTorch-to-CoreML move parity unless an exact move-parity launch/test mode exists and passes. A CoreML success-count smoke is necessary but not sufficient for exact parity.

## Mac Model Gates

Run the bundled gate script from the repository root:

```bash
.codex/skills/test-model/scripts/run_mac_model_gates.sh
```

Useful overrides:

```bash
CHECKPOINT=az_runs/best_3p_policy_checkpoint.pt \
MODE=policy \
LED_MODE=policy \
PLAYERS=3 \
PARADOX_MATCHES=200 \
LED_MATCHES=60 \
BACKEND=pytorch \
.codex/skills/test-model/scripts/run_mac_model_gates.sh
```

The script runs:

- homogeneous same-model paradox gate via `quantum_cat_full_match_elo.py`;
- non-red led-suit follow/switch/red-choice audit via `quantum_cat_led_switch_audit.py`;
- a compact summary JSON and console report.

The led-switch audit currently supports only `LED_MODE=policy` or `LED_MODE=q_policy`. When `MODE` is a richer selector such as `liveness_shield?...`, set `LED_MODE=policy` unless a liveness-aware led audit has been added.

Artifacts are written under `az_runs/model_test_<timestamp>/` unless `OUT_DIR` is set.

Backend policy:

- `BACKEND=pytorch` is the current supported path.
- `BACKEND=auto` currently resolves to PyTorch.
- `BACKEND=mlx` must fail clearly unless a real MLX runner exists in the repo; do not silently run PyTorch while saying MLX.

## Optional Strength Gate

If the user asks whether a candidate beats an old champion, run the normal paired comparison directly with `quantum_cat_full_match_elo.py`, usually `500` or `600` matches, paired/permutation schedule, and explicit neural aliases:

```bash
PYTHONPATH=. .venv/bin/python quantum_cat_full_match_elo.py \
  --players=3 \
  --matches=500 \
  --bots old new heuristic_target2 \
  --neural-bot old=OLD_CHECKPOINT:policy \
  --neural-bot new=NEW_CHECKPOINT:policy \
  --match-context \
  --schedule=permutations \
  --seed=20274450 \
  --workers=8 \
  --worker-torch-threads=1 \
  --out az_runs/model_test_strength.json
```

Report win rate, score rate with CI, average score diff with CI, paradoxes per match, and the artifact path.

## iPhone Install / Bridge Check

When the user asks to install on the physical iPhone, after copying/exporting the model and building the signed app, run:

```bash
IPHONE_DEVICE_ID=DEVICE_ID \
CHECKPOINT=az_runs/best_3p_policy_checkpoint.pt \
.codex/skills/test-model/scripts/run_ios_bridge_check.sh
```

The script checks:

- signed app bundle exists or builds it;
- bundled `champion_belief_policy.pt` hash matches `CHECKPOINT`;
- the app installs to the selected iPhone;
- on-device CoreML benchmark launch reports zero CoreML failures and shared bid/discard checks pass;
- exact move-parity mode is present and passes when `REQUIRE_EXACT_MOVE_PARITY=1`.

If the phone is locked, `devicectl` will report a SpringBoard locked-device error. Report that as an operational blocker, not a model failure.

If exact move-parity mode is absent, report that the app has only the CoreML benchmark smoke. Do not claim the model copies all PyTorch moves on iPhone until exact parity support is added and passing.
