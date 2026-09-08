# Quantum Cat curated tactical validation

Human-reviewed tactical positions and exhaustively proven official-scoring
positions are stored separately from self-play data in
`experiments/quantum_cat_curated_tactical_validation.json`. They are validation
only: neither the states nor their expected actions may be sampled into
training. A table-specific model is promotable only if its top-1 move belongs
to the complete set of defensible `acceptedActions` for every matching
scenario. A single accepted move is valid only when the position's declared
public tactical rule or exhaustive official-scoring proof rules out every
alternative. This gate is required in addition to
independent-deal official competitive-placement improvement, score
nonregression, and candidate-personal paradox safety. Table-wide paradox
frequency is diagnostic only.

Every locked root position is also evaluated after every cyclic relabeling of
player seats. Seat rotations preserve the exact game, legal moves, official
scoring, and expected answer. They exercise seat-specific model blind spots,
but are reported under their original scenario family: eight rotated checks do
not become eight independent pieces of human evidence.

The two human-reviewed four-player scenario families cover distinct positions:

- Opening follow after bids `[3, 2, 1, 1]`: yellow has already been led, and
  `Y7` is currently winning. `Y8` is the only legal yellow follow that takes the
  lead and keeps the yellow token. `R8` is also accepted: although it spends
  the yellow token, it is the highest possible trump and cannot be beaten by
  the remaining player. Those are defensible flexibility-versus-certainty
  tradeoffs. `B8` and `G8` are legal but off-suit: they lose the current trick
  and burn the yellow token. The explicit
  `winning_follow_or_unbeatable_trump_for_needed_bid` rule accepts `Y8` and
  `R8`, not just the move shown in the original capture.
- Second-trick lead after P0 won with `Y8`: P0 is now leading, so the led-suit
  constraint no longer applies. Both `B8` and `G8` pass the explicit
  `highest_nontrump_lead_for_needed_bid` rule. `Y8` is no longer legal because
  its board cell was claimed during the preceding trick.

The captured `Y6` recommendation is explicitly rejected in both cases.
These are human tactical judgments about an uncertain nonterminal game, not
claims of a solved minimax position. Their acceptance rules are recomputed over
every legal action and fail closed if an equally qualifying color is omitted.

The three exact five-player final-match scenario families cover:

- when the prediction is already satisfied, decline an unnecessary final trick
  to keep the connected-region bonus: `Y8` yields 2.5 official placement points
  and 11 total score; rejected `G8` yields 2 placement points and 10 score;
- when one trick is still needed, trump with `R9` to complete the prediction:
  `R9` yields 3.5 official placement points and 13 total score; rejected `G9`
  yields 2 placement points and 10 score.
- when the prediction is already matched under either move, choose the `G8`
  board cell that grows the connected region: `G8` yields 2.5 official
  placement points and 13 total score; rejected `Y8` yields 2 placement points
  and 12 score.

These final actions terminate the final match round immediately, so the
validator independently replays every legal move and recomputes the exact
official score, connected bonus, and final-round tiebreak from only the
acting hand and public table. **Every legal action tied for the best official
placement is accepted.** Own match score, round score, bid, and connected
bonus explain that placement but cannot arbitrarily reject another move that
produces the same final placing. A modified proof or an omitted tied move fails
closed.

Every candidate report includes the full legal-action ledger, the accepted set,
the public rule behind it, led-suit/token/trick consequences, explicitly
rejected moves, relevant unavailable moves such as the already-claimed `Y8`,
and a confidence label distinguishing human-reviewed judgment from exact
scoring.

The complete suite currently contains five independent scenario families and
23 checked positions: `2 x 4` four-player seats and `3 x 5` five-player seats.
Three-player reviewed/proven tactical positions remain an explicit coverage gap.
A separate locked, set-valued prediction suite does cover all three player
counts; see `docs/quantum_cat_curated_bidding_validation.md`. Its 11 bidding
families expand to 46 seat-rotated checks without changing the independently
reviewed five-family, 23-position tactical-play suite.

Validate fixture integrity and information safety:

```bash
.venv/bin/python quantum_cat_curated_validation.py
```

Evaluate a candidate using the same policy-stack syntax as the full-match
evaluator:

```bash
.venv/bin/python quantum_cat_curated_validation.py \
  --players 4 \
  --candidate 'CHECKPOINT.pt:liveness_shield?liveness_shield_base_mode=policy,liveness_shield_phases=play,liveness_shield_min_open_slot_delta=0,liveness_shield_min_public_damage_delta=0,liveness_shield_max_policy_log_gap=-1.0'
```

The command exits nonzero if any locked strategic answer is missed. Runtime
parity fixtures remain a separate concern: they prove Python/CoreML agreement,
not that the agreed move is strategically correct.

The standard `run_mac_model_gates.sh` candidate screen now executes the matching
tactical suite before any full-match evaluation. Changed promotion candidates
fail immediately on a missed move. Explicit exploratory screens still execute
every fixture, persist its report, and surface the failures without pretending
the candidate is promotable. Formal independent promotion runs the complete
four- or five-player suite before expensive match gates and verifies every root
and seat-rotation fixture identity in the final summary. Automated
`quantum_cat_candidate_sweep.py` champion gates also run the exact selector's
tactical preflight and mark failed candidates rejected without starting their
head-to-head match jobs.

All three workflows also require the applicable bidding suite before games.
Four nines accept bids 2, 3, or 4; a hand without nines, eights, or sevens
requires bid 1. Exact acceptance sets are checked against explicit hand-strength
rules so a reasonable alternative cannot silently disappear.

A separate locked play-decision suite lives in
`experiments/quantum_cat_curated_bid_regime_validation.json`. Its ten
independent scenario families expand to 41 cyclic-seat checks: nine for three
players, twelve for four players, and twenty for five players. Below-bid,
exact-bid, and already-above-bid scenarios reuse the same acting hand and
public trick while changing only that player's announced bid. The complete
set of reasonable safe winning moves passes below the bid; every equally safe
losing move passes exactly at the bid; and every equally safe winning move
passes after the bid has already been exceeded. A fourth five-player family
uses exact official terminal scoring to prove that, when too few playable
tricks remain to reach a higher bid, taking the extra point beats giving the
trick away. Safety comparisons use only the acting player's own feasible exits,
future buffers, and suit tokens; hidden opponent hands are never inspected.

Validate the distinct suite without conflating it with the original human
tactical positions:

```bash
.venv/bin/python quantum_cat_curated_validation.py \
  --suite experiments/quantum_cat_curated_bid_regime_validation.json \
  --players 4 \
  --candidate 'CHECKPOINT.pt:SELECTOR'
```

The regular, independent formal, and automated candidate-sweep promotion
workflows all run these locked bid-regime decisions before full matches.
Competitive promotion cannot bypass the new suite, while explicitly diagnostic
incumbent screens still record its failures without incorrectly reporting an
unpromoted exploratory model as a passing champion.

High-value candidates for additional human review include preserving the final
viable rank/color exit, following a legal led color to retain its token,
selecting the connected-region-maximizing cell, preventing a dangerous extra
trick, completing a needed bid, and pressuring an opponent only after the
acting player's own bid and exits are secure. Midgame cases should stay
review-only until their expected answer is defensible across the actor's
possible hidden-opponent worlds.
