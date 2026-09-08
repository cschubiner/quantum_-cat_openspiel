# Quantum Cat Run Metrics

Every serious Cat in the Box bot run should report official competitive match
placement, score, and player-specific paradox safety. Opponents causing a
paradox can improve the candidate's standing, so table-wide paradox frequency
is a diagnostic and must not veto a stronger information-legal candidate.

## Required Per-Run Metrics

### 1. Competitive Match Placement

The promotion objective is official full-match placement against the frozen
incumbent and strong information-legal opponents. Compare total match scores,
then final-round scores for the official tiebreak. Evaluate every candidate
seat on the same deal and count independent chance deals, not correlated seats
or pairwise comparisons, as the statistical inference unit.

Track:

| Metric | Definition | Direction |
|---|---|---|
| `independent_deal_blocks` | Distinct common deals, each with all candidate seats. | higher |
| `candidate_seat_counts` | Candidate appearances in every seat. | exactly balanced |
| `pairwise.score_rate` | Official placement wins plus half of ties against reference opponents. | above `0.5` |
| `placement_advantage.ci95` | Confidence interval calculated over independent deal blocks. | lower bound above `0.5` |
| `bga_style_rank_points.per_match` | Placement points, e.g. `6/4/2/0` for four players. | above player-count neutral |
| `official_score_advantage` | Candidate match score minus mean reference score. | no significant regression |
| `pairwise.head_to_head_elo_difference` | Elo equivalent of the placement score rate. | positive |

With three identical incumbent opponents, all `4! = 24` seating permutations
contain only four unique candidate seat assignments. Use `--schedule=cyclic`
to run those four assignments once; either way, the deal remains one
independent sample.

Command shape:

```bash
PYTHONPATH=. .venv/bin/python quantum_cat_full_match_elo.py \
  --players 4 \
  --bots challenger inc0 inc1 inc2 \
  --neural-bot challenger=CANDIDATE:MODE \
  --neural-bot inc0=INCUMBENT:MODE \
  --neural-bot inc1=INCUMBENT:MODE \
  --neural-bot inc2=INCUMBENT:MODE \
  --paired-deals 150 \
  --schedule cyclic \
  --competitive-gate-candidate challenger \
  --competitive-gate-opponents inc0,inc1,inc2 \
  --competitive-min-independent-deals 150 \
  --match-context \
  --seed SEED \
  --workers=1 \
  --out OUT.json
```

### 2. Player-Specific Paradox Safety

A paradox immediately ends a round and penalizes only the player who caused it.
The candidate's own paradox rate is therefore the safety guardrail; opponents'
paradoxes can improve competitive placement and must not count against the
candidate. Reject statistically meaningful regression in personal paradox risk
relative to the frozen reference policy.

| Metric | Definition | Direction |
|---|---|---|
| `candidate_own_paradox.round_rate` | Candidate paradoxes divided by candidate hand-round opportunities. | no significant regression |
| `reference_personal_paradox.round_rate` | Opponent paradoxes divided by opponent seat-round opportunities. | comparator |
| `own_paradox_rate_difference.ci95` | Candidate-minus-reference personal risk interval over independent deals. | lower bound not above allowed regression |
| `candidate_own_paradox.points_lost` | Own trick-point reversal plus forfeited correct-prediction bonus. | lower |
| `opponent_paradox_rounds` | Rounds where a reference opponent paradoxed. | diagnostic; can be favorable |
| `table_paradox.any_paradox_round_rate` | Rounds where any player paradoxed. | diagnostic only; never a promotion veto |
| `candidate_prediction.hit_rate` | Correct acting-player trick predictions. | higher |
| `candidate_prediction.adjacency_bonus_points` | Awarded connected-region scoring bonus. | higher |

The legacy homogeneous self-play summary and
`--homogeneous-paradox-threshold` remain available for historical comparisons,
but their `passed` field is not a competitive promotion decision. For symmetric
`n`-player self-play, average personal paradox risk is the table-wide rate
divided by `n`.

For exploratory actor-aligned posterior search, set
`survival_value_analytic_objective=competitive` on a value-shield selector.
It can pressure opponents only after the actor has satisfied their own bid and
without sacrificing personal feasibility, exit margin, or bid discipline.
This mode remains experimental unless its independent-deal placement
advantage passes the complete promotion gate.

## 3. Led-Color Discipline

The central tactical question is whether the bot burns a led-color lane
responsibly. In Cat in the Box, when a non-red color is led, playing a different
color removes that led-color token for the acting player. That immediately
shrinks future legal moves and can create later forced paradoxes.

Core rule we want the model to learn:

```text
If led color is B/Y/G and a led-color play is legal:
  follow the led color.

If no led-color play is legal:
  switch responsibly, usually to red/trump, and preferably win the trick.
```

Track these over play-phase decisions where `led_color` is `B`, `Y`, or `G`:

| Metric | Definition | Direction |
|---|---|---|
| `nonred_led_opportunities` | Decisions with a non-red led color. | denominator |
| `follow_rate_among_nonred_led_actions` | Followed `B/Y/G` led color among non-paradox actions. | high |
| `switch_rate_among_nonred_led_actions` | Chose a different color than the non-red led color. | low unless forced |
| `switch_with_led_color_legal` | Switched even though at least one led-color action was legal. | near zero |
| `switch_with_no_led_color_legal` | Forced/off-lane switches where no led-color action existed. | context |
| `red_rate_among_switches` | Switched to red among all off-led switches. | high |
| `red_choice_rate_when_red_legal_on_switch` | Chose red among switches where red was legal. | ideally very high |
| `switch_win_rate_resolved` | Switch plays that won their trick, among completed switch tricks. | majority+ |
| `red_switch_win_rate_resolved` | Red switch plays that won their trick. | high |

Useful interpretation:

- A policy that frequently has `switch_with_led_color_legal > 0` is probably
  burning future legal moves too casually.
- A policy with low `red_rate_among_switches` is often sloughing instead of
  using the switch as a trump/win plan.
- A policy with low `switch_win_rate_resolved` is taking lane damage without
  enough tactical payoff.

Command shape:

```bash
PYTHONPATH=. .venv/bin/python quantum_cat_led_switch_audit.py \
  --players 3 \
  --matches 60 \
  --checkpoint CHECKPOINT \
  --match-context \
  --seed SEED \
  --out OUT.json
```

## 4. Late Forced-Exit Signature

When paradox rate is still high, record whether failures are final forced exits
or earlier avoidable paradoxes.

Track:

| Metric | Definition | Direction |
|---|---|---|
| `trigger_by_trick_number` | First paradox trick histogram. | identify late collapse |
| `trigger_by_legal_count` | Legal count at first paradox. | `legal_count=1` means forced by then |
| `forced_triggers` | First paradoxes with `legal_count <= 1`. | lower |
| led-color audit by trick | Switch/follow behavior split by trick number. | find trick 5-7 lane collapse |

If most failures are `trick 7` and `legal_count=1`, the immediate move is often
not the real mistake. The run needs more upstream training on preserving exits
before the final trick.

## 5. Promotion Decision

Promotion requires:

```text
information-legal play and locked tactical fixtures pass
candidate seat is balanced on every independent common deal
placement score independent-deal CI95 lower bound > 0.5
official match score does not significantly regress
candidate personal paradox rate does not significantly regress
```

Table-wide paradox frequency and opponent paradox frequency are descriptive;
neither is a rejection threshold. Do not claim human-superhuman performance
until a recorded human cohort anchors the Elo scale.

Locked tactical positions must run on every ordinary candidate screen, not only
at final promotion. Report reviewed/exact-proven scenario families separately
from their all-seat metamorphic checks. Current coverage is two reviewed 4p
families (eight seat checks) and three exact-scored 5p families (15 seat checks);
the absence of locked 3p cases is an explicit coverage gap.

## Per-Run Logging Checklist

Each run entry in `EXPERIMENTS.md` should include:

- checkpoint path and SHA-256 if it is a promotion candidate;
- training data and objective summary;
- seat-rotated common-deal competitive gate artifact and independent deal count;
- official placement, paired score rate, score difference, and Elo evidence;
- candidate versus opponent personal paradox rates and own points lost;
- optional homogeneous table-paradox diagnostic, explicitly marked non-gating;
- first-paradox summary, especially trick number and legal count;
- led-switch audit artifact and the led-color discipline table;
- explicit decision: promote, continue, or reject.
