---
name: test-model
description: Use this skill when testing Quantum Cat ML checkpoints or champions, including seat-balanced official competitive placement gates, actor-specific paradox safety, homogeneous diagnostic checks, led-suit audits, and iPhone/CoreML bridge validation.
---

# Quantum Cat Model Testing

Use this skill before promoting or installing a Quantum Cat model, and whenever the user asks for model stats such as paradox rate, follow-led behavior, off-led red-switch rate, or iPhone bridge correctness.

## Core Rules

- Prefer Mac-side fast evaluation for statistical evidence. Do not use long simulator or physical-device runs for paradox-rate estimates.
- The primary promotion metric is official full-match placement against the
  frozen incumbent on independent common deals with every candidate seat
  represented equally. Require the deal-block placement-score CI95 lower bound
  above `0.5`, no significant official-score regression, and no significant
  increase in the candidate's personal paradox rate.
- Table-wide `rounds_with_any_paradox / hand_rounds` is diagnostic only. It must
  never veto competitive promotion: an opponent paradox can improve the
  candidate's final placement.
- Keep selector mode explicit. If the promoted note says raw policy, use `MODE=policy`; do not apply q-policy/risk/value rerankers unless the user explicitly asks for that variant.
- If the promoted note says liveness shield, use the full selector string, for example `MODE='liveness_shield?liveness_shield_base_mode=policy,liveness_shield_phases=discard|prediction|play,liveness_shield_min_open_slot_delta=0,liveness_shield_min_public_damage_delta=0,liveness_shield_max_policy_log_gap=-1.0'`.
- Treat checkpoint hashes and model paths as time-sensitive. Verify the actual file hash before reporting.
- For iPhone installs, do not claim exact PyTorch-to-CoreML move parity unless an exact move-parity launch/test mode exists and passes. A CoreML success-count smoke is necessary but not sufficient for exact parity.
- Before training or promoting from best-of-rollouts replay, require a durable
  holdout audit that proves: every teacher branch was sampled from the acting
  player's information state; candidate actions shared the same sampled worlds
  and continuation seeds; every accepted row clears the declared paired
  improvement over the source action; the selected action clears a second
  paired improvement check on fresh, independent information-state worlds;
  train and validation source groups are disjoint; and every required trick
  number appears in both splits. Exact-state hidden-hand teachers and
  selection-only rollout winners are diagnostic-only, even if their live
  screen passes.

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
INCUMBENT_CHECKPOINT=az_runs/best_3p_policy_checkpoint.pt \
INCUMBENT_MODE=policy \
COMPETITIVE_DEALS=200 \
PARADOX_MATCHES=200 \
LED_MATCHES=60 \
BACKEND=pytorch \
.codex/skills/test-model/scripts/run_mac_model_gates.sh
```

The script runs:

- all matching locked, set-valued bidding hands from every acting seat before
  any expensive matches, including several highest-rank cards, missing top
  ranks, and defensible low-bid boundaries for three, four, and five players;
- all matching locked human-reviewed and exact-official-scoring tactical moves
  from every acting seat before any expensive matches;
- all matching locked below-bid/exact-bid/already-missed bid-regime play
  decisions, accepting every equally safe reasonable winning or losing card;
- seat-balanced common-deal competitive placement gate against the frozen
  player-count-specific incumbent via `quantum_cat_full_match_elo.py`;
- candidate-versus-incumbent personal paradox safety and lost-score reporting;
- homogeneous same-model table-paradox diagnostic, never a promotion veto;
- non-red led-suit follow/switch/red-choice audit via `quantum_cat_led_switch_audit.py`;
- a compact summary JSON and console report.

The led-switch audit supports full selector payloads, including `liveness_shield?...`; keep `LED_MODE` explicit and match the selector you intend to measure.

By default a checkpoint/selector that differs from its frozen incumbent must
pass the competitive gate. Inspecting the incumbent against itself is
automatically diagnostic-only. Set `REQUIRE_COMPETITIVE_PASS=0` for an
exploratory screen; never treat that screen as promotion proof. The historical
`REQUIRE_CI_UPPER_BELOW_THRESHOLD` option is disabled by default and is not
part of formal promotion.

Bidding and bid-regime play fixtures always execute for three-, four-, and
five-player candidates. Other tactical play fixtures also execute when the
selected player count has locked cases. Changed candidates fail immediately on
any missed prediction or play by
default; explicit exploratory screens retain failed artifacts and continue only
for diagnosis. Set-valued bidding accepts all reasonable choices, such as any
bid above one for four nines, instead of requiring an arbitrary exact bid.
Summaries report independent human/proven scenario
families separately from cyclic seat-rotation coverage. Three-player
bid-regime fixtures are present; separately human-captured three-player
tactical fixtures are not yet available, and that narrower gap stays visible.

Automated `quantum_cat_candidate_sweep.py --gate-champion` runs apply the same
matching tactical, bidding, and bid-regime preflights before each selected
checkpoint's
direct champion matches. A failed candidate is marked
`locked_tactical_failure`, `locked_bidding_failure`, or
`locked_bid_regime_failure`; no expensive head-to-head
evaluation is started for it.

Frozen default references are the 3p promoted raw policy, the Run730 4p
checkpoint with its exact public-liveness selector, and the Run600 5p
checkpoint with its exact q-policy selector. Override both
`INCUMBENT_CHECKPOINT` and `INCUMBENT_MODE` together only when the actual
deployed reference changes.

For promotion, require multiple independent seeds:

```bash
CHECKPOINT=path/to/candidate.pt \
MODE='liveness_shield?...' \
LED_MODE='liveness_shield?...' \
INCUMBENT_CHECKPOINT=path/to/frozen-incumbent.pt \
INCUMBENT_MODE='actual_deployed_selector?...' \
COMPETITIVE_DEALS=200 \
INDEPENDENT_GATES=3 \
.codex/skills/test-model/scripts/run_independent_mac_model_gates.sh
```

Every constituent gate must independently prove a placement advantage; an
aggregate pass cannot hide a failed seed. The formal wrapper requires at least
three distinct deal seeds and at least `600` seat-rotated matches per gate.
For every player count it first executes the complete applicable locked
bidding and bid-regime play suites against the exact candidate selector.
Four- and five-player
candidates additionally run their locked tactical-play suite. The wrapper
fails before starting expensive matches if any reviewed prediction or play is
wrong. Four-player play cases cover both human-reviewed positions from every
seat; five-player play cases cover exact bid-preservation, bid-completion, and
connected-bonus endgames from every seat.
It validates candidate and frozen-incumbent checkpoint hashes, exact selector
modes, independent deal counts, every-seat rotation, official-score
nonregression, and the candidate's personal-paradox nonregression. Table-wide
paradoxes remain informational even above `40%`. Pooled metrics are
informational only.

Artifacts are written under `az_runs/model_test_<timestamp>/` unless `OUT_DIR` is set.

Backend policy:

- `BACKEND=pytorch` is the current supported path.
- `BACKEND=auto` currently resolves to PyTorch.
- `BACKEND=mlx` must fail clearly unless a real MLX runner exists in the repo; do not silently run PyTorch while saying MLX.

## Direct Competitive Placement Gate

Run official same-deal, seat-rotated comparisons directly with
`quantum_cat_full_match_elo.py`, using explicit frozen incumbent aliases and a
cyclic seating schedule:

```bash
PYTHONPATH=. .venv/bin/python quantum_cat_full_match_elo.py \
  --players=4 \
  --bots new inc0 inc1 inc2 \
  --neural-bot new=NEW_CHECKPOINT:NEW_MODE \
  --neural-bot inc0=OLD_CHECKPOINT:OLD_MODE \
  --neural-bot inc1=OLD_CHECKPOINT:OLD_MODE \
  --neural-bot inc2=OLD_CHECKPOINT:OLD_MODE \
  --paired-deals=150 \
  --competitive-gate-candidate=new \
  --competitive-gate-opponents=inc0,inc1,inc2 \
  --competitive-min-independent-deals=150 \
  --match-context \
  --schedule=cyclic \
  --seed=20274450 \
  --workers=8 \
  --worker-torch-threads=1 \
  --out az_runs/model_test_strength.json
```

Report independent deal blocks, official placement score with deal-block CI,
average official-score difference with CI, Elo difference, candidate and
opponent personal paradox rates, and the artifact path. Do not count multiple
opponent comparisons or duplicate incumbent permutations as independent data.

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
- on-device gameplay bridge launch reports zero CoreML failures and shared bid/discard checks pass;
- live inference either reports the verified `python-observation76-action214-v1`
  input contract and makes CoreML attempts, or fails closed with zero attempts;
- raw-tensor PyTorch-to-CoreML move-parity mode is present and passes when
  `REQUIRE_EXACT_MOVE_PARITY=1`.

If the phone is locked, `devicectl` will report a SpringBoard locked-device error. Report that as an operational blocker, not a model failure.

Raw-tensor move parity proves model conversion, not native gameplay preprocessing.
Do not claim the model is deployed for live iPhone play unless the live 76-value
observation and 214-value action-feature encoder contract is independently
verified. While it is unverified, the app must fail closed to its native
heuristic fallback and all human telemetry is diagnostic-only model evidence.
