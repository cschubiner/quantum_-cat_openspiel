# qcx experiment results (running log)

All C++-arena rows are 5-player complete matches: one candidate seat against a field, common deals rotated through every seat, paired against an all-field reference table on the same deals. "Placement" is pairwise placement (0.5 = parity). CIs are 95% over independent deals. Python-evaluator rows use the original `quantum_cat_full_match_elo.py` gate with the original champion bots.

## E0 — engine
- C++ engine passes the rules-parity test against `open_spiel/python/games/quantum_cat.py`: 9,000 random rounds (3/4/5p), identical legal-action lists at every step (also when rebuilt from the information-only view) and identical scores.
- Speed (unloaded machine): heuristic rollout ~68 µs per 5p round per thread. The Python engine runs ~570 µs per *random* round.

## E1 — PIMC at every decision vs the champion (Python evaluator)

| Candidate | Opponents | Deals (matches) | Pairwise placement | First place | Score adv./match | Own paradox | Bid hit |
|---|---|---|---|---|---|---|---|
| qcx PIMC 64 worlds | 4× Run1051 champion | 32 (160) | **87.1%** | **72.5%** (par 20%) | **+8.13** | 12.6% (champ 13.6%) | 51.8% |
| qcx PIMC 128 worlds (v1) | 4× Run1052 champion (the prior confirmed champion) | 16 (80) | **86.9%** | **73.8%** (par 20%) | **+7.69** | 13.3% (champ 12.6%) | 52.0% |
| **qcx v3 (learned rollouts, 256 worlds)** | 4× Run1052 champion | 32 (160) | **96.1%** | **87.5%** (par 20%) | **+10.59** | 8.1% (champ 13.5%) | 61.6% |
| qcx PIMC 128 worlds (3p) | 2× Run604 3p champion (liveness shield) | 40 (120) | **80.0%** | **69.6%** (par 33%) | **+6.35** | 10.0% (champ 26.4%) | 36.7% |
| qcx PIMC 128 worlds (4p) | 3× Run730 best 4p model (liveness shield) | 40 (160) | **84.6%** | **70.6%** (par 25%) | **+8.24** | 11.1% (champ 22.4%) | 45.0% |

For reference, the lineage's entire confirmed improvement Run1051→Run1052 was 57.1% placement.

## E2 — search budget (C++ arena)

| Candidate | Field | Deals | Placement | 95% CI on diff |
|---|---|---|---|---|
| pimc:128 | pimc:32 | 60 | 62.6% | [+8.5, +16.7] pp |
| pimc:8 | heur | 32 | 47.4% | [−8.0, +2.9] pp (too few worlds is worse than the heuristic) |
| pimc:32 | heur | 200 | 75.1% | [+23.2, +27.0] pp |

## E3 — utility (C++ arena, 60 deals, field pimc:32)

| Candidate | Placement | 95% CI on diff |
|---|---|---|
| own score (util 1) | 49.8% | [−4.4, +4.0] pp |
| match-placement aware (util 2, σ=4) | 49.8% | [−4.4, +4.1] pp |

Neutral. The simpler score-difference objective stays.

## Locked curated suites (tactical + bid-regime + bidding), qcx PIMC 128, 3 seeds
228/330 pass. Systematic failures:
- **Bidding:** with three or four 9s (5p) or four 8s (4p), it bids 1; the user-reviewed rule requires ≥2.
- **Bid-regime endgame:** trump to win when below bid; dump the highest loser when at bid.
- **iPhone 4p position 2:** it leads B7/G7 where B8/G8 is expected.

**Diagnosis:** under the heuristic rollout policy, a four-9s hand averages 1.2 tricks and paradoxes 21% even when chasing tricks. Rollout opponents readily trump a led 9 by abandoning a colour, and rollout play of strong hands is poor. Bid values are therefore computed against a weak model of future play, so rollout-policy quality is the bottleneck (E5).

## E4 — bid-informed beliefs (C++ arena, field pimc:32)
Importance resampling of worlds by a multinomial-logistic bid model (fit on 36k PIMC bids; val NLL 0.94 vs 1.14 marginal).
60 deals: +2.5pp [−0.6, +5.6]; 140 fresh deals: +0.9pp [−1.5, +3.3]. Not significant → not adopted.

## E6 — search/rollout variants (C++ arena)
| Variant | Field | Deals | Δ placement (95% CI) |
|---|---|---|---|
| two-stage pimc2:32:3:224 | pimc:64 | 60 | +5.7 [+2.1, +9.3] |
| flat pimc:128 (similar compute) | pimc:64 | 60 | +5.8 [+2.4, +9.3] → two-stage gives no efficiency gain |
| pimc:512 | pimc:128 | 40 | +4.4 [−0.6, +9.4] (diminishing but positive) |
| nested bid search (own play by inner PIMC) | pimc:64 | 60 | −1.0 [−4.7, +2.8] |
| rollout token penalty 2.5 | pimc:32 | 60 | −7.1 [−10.7, −3.5] |
| rollout token penalty 0.0 | pimc:32 | 60 | −4.6 [−8.2, −1.0] |
| rollout token penalty 0.3 | pimc:32 | 60 | +2.9 [−1.2, +7.0] |
| **rollout token penalty 0.6** | pimc:32 | 60 | **+4.1 [+0.2, +8.0]** |
| tok 0.6 + win weight 4.5 / 2.0 | pimc:32 | 60 | +2.9 / +0.6 |
| tok 0.6 + adjacency 0.7 / 0.0 | pimc:32 | 60 | −0.6 / −2.0 |
| tok 0.6 + dead-end 3 / 12 | pimc:32 | 60 | +4.1 / +4.1 |
| rollout ε=0.1 random | pimc:32 | 60 | −0.3 |
| **confirm tok 0.6 @64 worlds, fresh deals** | pimc:64 | 100 | **+3.6 [+0.7, +6.6] → promoted** |

## iOS
Champion shipped to the iPhone (branch `claude/ios-qcx`). On-device parity 72/72 fixtures; 702 benchmark decisions, 0 illegal; 256 worlds: 5p mean 22 ms, max 91 ms.

## E5 — expert iteration: learned rollout policy (branch claude/exp-exit, integrated in claude/exp-champ3)
64 information-safe features; MLP (32 hidden) trained to imitate pimc:32 decisions (86k decisions / 950k rows). Info-safety test: 12,050 decisions × 8 resampled worlds, features identical.

| Candidate | Field | P | Deals | Δ placement (95% CI) |
|---|---|---|---|---|
| learned MLP policy alone (no search) | heur | 5 | 300 | +19.9 [+18.1, +21.7] |
| pimcl:32 MLP (old tok 1.2 recipe) | pimc:32 | 5 | 200 | +12.3 [+10.0, +14.5] |
| pimc:96 (≈ same CPU) | pimc:32 | 5 | 200 | +9.3 [+7.1, +11.5] |
| gen-2 pimcl vs gen-1 pimcl | | 5 | 200 | +0.9 [−1.2, +2.9] (second iteration saturates: noisy 32-world labels) |
| **pimcl:128 MLP vs champion v2 recipe pimc:128 (tok 0.6)** | | 5 | 60 | **+11.5 [+8.0, +15.0]** |
| **pimcl:64 MLP vs pimc:256 (4× worlds)** | | 5 | 60 | **+6.5 [+2.8, +10.2]** |
| pimcl:128 MLP vs pimc:128 | | 3 | 60 | +8.1 [+1.8, +14.4] |
| pimcl:128 MLP vs pimc:128 | | 4 | 60 | +19.1 [+14.8, +23.4] |

→ **Champion v3** = PIMC with learned MLP rollouts (g1_mlp), promoted.

## Bidding ground truth (strong play, identical deals, forced bid)
5p, seat 0 holds ≥3 nines after discard; every other decision by pimcl:32 (MLP rollouts); 400 deals.
| Bid | own score | score − mean(others) | bid hit | paradox |
|---|---|---|---|---|
| 1 | 2.30 | +0.455 | 48.5% | 7.0% |
| 2 | 2.16 | +0.338 | 32.0% | 7.5% |
| 3 | 1.40 | −0.394 | 9.0% | 7.2% |
bid 2 − bid 1 = −0.12 [−0.47, +0.23]; bid 3 − bid 1 = −0.85 [−1.16, −0.54].
→ The locked rule "three/four 9s: bid 1 is known-bad" is **not supported** under strong play: 1 and 2 are statistically equivalent; 3+ is clearly worse.

## Gen-3 rollout policy (128-world v3 labels, 158k decisions, 3/4/5p) — champion v4
| Candidate vs v3 (pimcl:128 g1) | P | Deals | Δ placement (95% CI) |
|---|---|---|---|
| g3 MLP-32 | 3 | 60 | +6.7 [+1.0, +12.4] |
| g3 MLP-32 | 4 | 60 | +4.2 [+0.1, +8.4] |
| g3 MLP-32 | 5 | 100 | +2.3 [−0.5, +5.2] |
| g3 MLP-32, fresh deals | 5 | 150 | +0.9 [−1.5, +3.3] |
| g3 MLP-64 | 5 | 100 | +1.1 [−2.2, +4.4] |
→ **v4 = g3 MLP-32** (non-inferior at 5p, better at 3p/4p, lower paradox and higher bid-hit everywhere). Shipped to iPhone.

Four-nines bidding ground truth (5p, 300 deals): bid 2 − bid 1 = −0.09 [−0.48, +0.31]; bid 3 − bid 1 = −0.86 [−1.29, −0.43].

## Learned bid model inside rollouts (fit on v4's own bids)
| P | Deals | Δ placement (95% CI) |
|---|---|---|
| 3 | 60 | +1.4 [−3.5, +6.3] |
| 4 | 60 | −1.4 [−5.7, +3.0] |
| 5 | 100 | +1.75 [−1.4, +4.9] |
Not significant → not adopted.

v4 fresh-deal confirmations vs v3: 3p 80 deals +9.0 [+4.2, +13.7]; 4p 80 deals +4.2 [+1.0, +7.3].

## E7 — policy-based belief inference (branch claude/exp-infer; details in that branch's RESULTS.md)
Worlds importance-resampled by the likelihood of opponents' past plays under the learned policy (reconstruction & info-safety tests pass).
Best setting (α=0.5, ε=0.1, M=512): 5p pooled 300 deals +1.1 [−0.6, +2.9]; 4p 160 deals +2.7 [−0.2, +5.5]; 3p −0.7. CPU-matched control (more worlds) +1.9.
→ Not adopted. Diagnosed cause: deliberate discards bias the uniform hand prior; play likelihood alone can't fix it.

## Gen-4 rollout policy (256-world v4 labels) vs v4 — not promoted
| Candidate | P | Deals | Δ placement (95% CI) |
|---|---|---|---|
| g4 MLP-32 (gen-3 + gen-4 data) | 5 | 120 | −1.3 [−4.3, +1.6] |
| g4-only MLP-32 | 5 | 120 | +1.8 [−1.0, +4.5] |
| g4 MLP-32 | 4 | 80 | +1.4 [−2.8, +5.6] |
| g4 MLP-32 | 3 | 80 | −0.3 [−6.0, +5.3] |
Expert iteration with the 64-feature MLP has saturated; v4 remains champion.
On-device v4 (iPhone 17 Pro Max, 512 worlds): parity 72/72, 297 decisions 0 illegal; 5p mean 121 ms, max 0.59 s.
## E7 — policy-based inference over opponents' past plays (branch claude/exp-infer)
`pimcli:<N>:<weights>:<M>:<alpha>:<eps>[:<T>]` (qc_infer.h): the bot gets its masked view + the round's public bid/play history (discards never recorded). Draw M uniform candidate worlds; for each, rebuild every opponent's past PLAY decision this round (hand then = candidate hand + cards played since; public state by replaying from the start of play), score log[(1−ε)·π(a|s) + ε/n] with the learned g3 policy (softmax T=1; −∞ if the play would have been illegal), weight exp(α·Σ), systematically resample the N search worlds, then PIMC with CRN as before.

Tests (`test_infer`, 5p): reconstruction exact for 2,252 / 2,530 / 4,132 opponent decisions (policy-T1 / pimcl:16 / pimcl:32 games); info-safety: 52 / 56 / 93 decisions where a *different* true deal with the same masked view + history gave bit-identical action and values.

Belief quality (posterior-mean opponent rank counts vs truth, sum of squared errors per decision, M=1024; vs uniform sampler):
| Opponents | α=1 ε=.1 | α=.5 ε=.1 | α=.3 ε=.1 | α=1 ε=.1 T=2 | truth more likely than |
|---|---|---|---|---|---|
| policy sampled at T=1 (exact model), heuristic discards | +5.7% | +1.1% | +0.4% | +7.1% | 73% of candidates |
| same, uniform-random discards (exact prior) | −2.6% (α=1) | | | | 83% |
| pimcl:32 for discard/bid/play (≈ field) | +0.5% | **−1.1%** | −1.0% | +1.9% | 66% |
The plays are informative (truth ranks well), but the likelihood favours worlds where the actor had few options; with non-uniform (searched) discards the prior is wrong in exactly that direction, so the full posterior barely improves (or worsens) hand estimates. Tempering α≈0.3–0.5 is needed.

CPU per 5p PLAY decision at N=128 (400 states, interleaved): pimcl 51.8 ms; pimci M=512 67.6 ms (×1.31); M=1024 84.0 ms (×1.62). 3p: 25.9 / 32.7 (×1.26) / 40.1 (×1.55).

Arena vs champion field pimcl:128:g3_mlp32 (candidate N=128):
| Candidate | P | Seed | Deals | Δ placement (95% CI) |
|---|---|---|---|---|
| α=1 ε=.1 M=512 | 5 | 9101 | 100 | −0.3 [−3.4, +2.9] |
| tune: α=.3 ε=.1 M=512 | 5 | 9201 | 60 | +2.4 [−0.9, +5.7] |
| tune: α=.5 ε=.1 M=512 | 5 | 9201 | 60 | +5.7 [+1.6, +9.8] |
| tune: α=1 ε=.3 M=1024 | 5 | 9201 | 60 | −6.7 [−11.2, −2.3] |
| tune: α=.5 ε=.1 M=1024 | 5 | 9201 | 60 | −1.7 [−5.6, +2.3] |
| **confirm α=.5 ε=.1 M=512** | 5 | 9301 | 150 | +3.2 [+0.7, +5.8] |
| confirm, replicate | 5 | 9401 | 150 | −1.0 [−3.4, +1.4] |
| confirm | 4 | 9302 | 80 | +3.75 [−0.5, +8.0] |
| confirm, replicate | 4 | 9402 | 80 | +1.6 [−2.2, +5.4] |
| confirm | 3 | 9303 | 80 | −0.7 [−5.8, +4.4] |
| CPU-matched control pimcl:168 (no inference) | 5 | 9301 | 150 | +1.9 [−0.6, +4.4] |
Pooled α=.5 confirmations: 5p (300 deals) +1.1 [−0.6, +2.9]; 4p (160) +2.7 [−0.2, +5.5]; 3p −0.7.
→ **Not adopted.** Untempered inference (α=1) is neutral-to-harmful; tempered α=0.5 gives at most +1–3 pp, not significant, and not distinguishable from spending the same 1.3× CPU on more worlds (the M=512 vs M=1024 pair at the same α/seed differs by 7 pp, i.e. the tuning winner was largely noise). Likely blockers: misspecified prior over post-discard hands (a discard/bid model would be needed alongside the play likelihood) and a play policy that imitates search, not the field's actual decision rule.
