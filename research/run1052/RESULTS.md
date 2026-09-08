# Run1052 experiment results

Status: **confirmed improvement**. The every-round endgame-search challenger passed all preregistered gates at the first protected 64-deal look. Its benchmark promotion is recorded in [CONFIRMED.json](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/CONFIRMED.json). Existing champion artifacts, source/app defaults, and the iPhone are unchanged.

## Final protected result

The frozen challenger played 320 complete five-round matches: 64 independent fresh deals, each rotated through all five seats, against four exact Run1051 champions. A separate 320-match champion reference used the same deals and seats. Inference is by independent deal, not by the 640 correlated seat-rotated matches.

| Metric | Prior champion | Every-round search |
|---|---:|---:|
| Pairwise placement | 50.00% | **57.11%** |
| First-place share, splitting ties | 20.00% | **26.88%** |
| Outright match wins | 63 / 320 (19.69%) | **85 / 320 (26.56%)** |
| Average final score | 9.99 | **10.77** |
| Scored bid hits | 648 / 1,600 (40.50%) | **705 / 1,600 (44.06%)** |
| Actor-owned paradoxes | 180 / 1,600 rounds (11.25%) | **169 / 1,600 (10.56%)** |

The primary paired placement gain is **+7.11 percentage points**, with the preregistered three-look-corrected **98.333% confidence interval [+4.41, +9.81] points** and Bonferroni-adjusted p = 4.78e-8. All five promotion gates passed: positive corrected placement lower bound, nonnegative official-score advantage, no increase in observed actor paradoxes, all 60 locked cases, and the frozen information-safety audit. The official-score advantage over opponents improved by +0.9883; this differs from the +0.7813 change in the actor's own average score because its actions also change opponents' scores.

First-place, bid-hit, and paradox figures are secondary observed metrics, not additional preregistered significance claims. In particular, fewer observed paradoxes passes the count-based safety guard but does not prove a lower population paradox rate; the paired paradox-rate interval includes zero.

The winning change is **planning, not new neural weights**: the exact same full network now searches the final three tricks of every round instead of just the final round. No belief plugin or fastpath prototype is included. The separate trained Q and belief variants did not demonstrate improvement in development.

Confirmation ended at **01:30:56 EDT, September 5**, and the controller stopped at the first passing look. No 128- or 256-deal continuation ran. The controller exited successfully, all known workers exited, frozen files and protected receipts passed validation, and an [independent raw-outcome readback](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/confirmation/raw_readback_0064.json) reproduced both sides' metrics and the exact protected aggregate. The completed batch contains **1,240 competitive matches**: 600 development and 640 confirmation, excluding training and separate implementation audits.

This supports a stronger five-player bot against the prior champion on the registered placement benchmark. It does not establish human/superhuman strength, other player counts, iPhone performance, or deployment. Running this exact system on the phone requires the full search path, not merely reinstalling the unchanged checkpoint.

## What was actually run

The comparator is the **complete frozen Run1051 champion**, including its information-state endgame search, not the weaker neural-only phone policy. All competitive games are five-player, complete five-round matches against four copies of that champion. Common deals are rotated through all five seats. Statistical intervals count independent deals, not correlated seats or rounds.

Three lanes ran in parallel:

1. **Search:** four candidate actions, 32 sampled worlds, or search in the final three tricks of every round. These are separate variants, not a combined change.
2. **Outcome learning:** 500 completed league matches / 2,500 rounds, producing 21,166 decision examples. Snapshots at 100, 250, and 500 matches learn action values from official final-match outcomes. The 36 non-Q tensors, including bidding, policy, and risk, remain bit-identical; ten Q tensors changed. This tests a specific Q-only recipe, not every form of self-play or full-network reinforcement learning.
3. **Hidden-hand learning:** 400 completed collection matches / 67,864 decision examples; whole-match train/calibration/test split of 320/40/40. A public-history GRU scores physically possible hidden hands. A separately labeled, post-diagnostic Metropolis sampler uses the same frozen learned weights.

The [training-data audit](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/audits/training_data.json) independently checks completed-job receipts, unique training IDs/seeds, checkpoint hashes, frozen tensors, and actual stored split identifiers.

## Initial development screens

Each row below uses the same eight independent deals, each rotated through all five seats: 40 complete matches and 200 actor-round opportunities. These are exploratory screens, **not evidence for promotion**. Pairwise placement measures how often the actor finishes ahead of another player, splitting ties; 50% means parity. It is not first-place win rate.

| Variant | Pairwise placement | Actor paradoxes / matched champion |
|---|---:|---:|
| Four candidate actions | 49.06% | 20 / 20 |
| 32 selection and audit worlds | 48.75% | 20 / 20 |
| Every-round endgame search | 56.56% | 14 / 20 |
| Q learning: 100 matches | 43.75% | 25 / 20 |
| Q learning: 250 matches | 45.31% | 29 / 20 |
| Q learning: 500 matches | 50.63% | 27 / 20 |
| History model, finite-proposal resampling | 48.44% | 19 / 20 |
| History model, Metropolis sampling | 49.06% | 21 / 20 |

All completed rows passed the same 60 locked bidding/tactical/bid-regime cases. Regression passes establish basic behavior, not champion-level strength.

## Additional-deal validation

One finalist per lane may receive 16 additional development deals, bringing it to 24 independent deals / 120 complete matches. This remains development data; only a frozen finalist evaluated on untouched confirmation deals can earn promotion.

**500-match Q model:** 49.90% placement; paired difference −0.10 percentage points, exploratory 95% interval [−5.10, +4.89] points. Its actor had 56 paradoxes in 600 rounds, exactly matching the champion. First-place share was 18.33% versus 20.00% for the matched champion. The early apparent placement gain and paradox increase did not persist. Result: inconclusive, no demonstrated improvement.

**Every-round search:** its additional-deal result remained promising. Across 24 independent deals it scored 53.85% placement; paired gain +3.85 percentage points, exploratory 95% interval [+0.42, +7.29] points. Official score advantage was +0.6604. Its actor had 47 paradoxes versus 56 in 600 rounds, and 290 scored bid hits versus 269. First-place share was 18.75% versus 20.00% (21 outright wins versus 23). This is a placement/score improvement in development, not a demonstrated first-place win-rate improvement or protected confirmation.

All eight initial candidates and both finalist extensions are complete: 480 candidate matches plus 120 shared matched-champion reference matches, 600 competitive matches in total. These counts exclude the intentionally interrupted, unscored attempt; the independent sample units remain deals.

A separate [raw-outcome readback](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/audits/raw_development_readback.json) recomputed all 600 completed development matches: official total/final-round tiebreak order, pairwise placement, actor paradox counts, scored bids, and complete five-seat deal groups agree with the reports. Five additional readback tests pass, including three-way ties. Its protected-data entry point opens outcomes only after a scheduled look has been published; it cannot launch games or change the promotion rule.

## What we learned so far

- The original history model recognizes plausible complete hands better than the no-history ablation, but its finite-proposal individual-card probabilities were worse calibrated than uniform sampling. Better predictive ranking did not translate into better play.
- Both belief samplers were evaluated with the original champion's final-round search window. Their combination with the promising every-round search was not tested; these results do not establish that learned beliefs are generally useless.
- The revised Metropolis sampler improved all four checked marginal NLL/Brier metrics against both uniform sampling and a matched unlearned sampler on 240 active-window calibration positions from 40 matches. No weights or temperatures changed, and the original final predictive test was not reused for selection. That justified its subsequent gameplay test, which did not show improvement. Finite-step sampling is approximate, not a convergence guarantee.
- On 326 matched audit positions, the 500-match model changes 9.2% of full-wrapper actions while policy and risk predictions remain unchanged. Q scale and its interaction with the fixed selector deserve attention. This is a diagnostic, not proof that Q changes caused a systematic paradox regression; the larger development set has tied paradox totals.
- Six genuine-state Mac CPU latency probes explain the cost of earlier search. Early/middle-round moves took 2.77–10.17 seconds with every-round search, versus 4.7–6.2 milliseconds for the champion; in final-round search windows both took approximately 0.21–0.83 seconds. Two probes still performed expensive rollouts after safety filtering left only the retained baseline move. No optimization was applied during these frozen experiments. These are small CPU measurements under current conditions, not iPhone benchmarks or a strength evaluation.
- A separate, isolated [fastpath prototype](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/fastpath/full_match_parity/README.md) skips those single-retained-move rollouts. Its hardened version matched all 44 real candidate actions and global RNG effects across one complete five-round audit match, including four shortcuts and six subsequent multi-action searches. Native versus prototype candidate-call time totaled 97.68 versus 57.10 seconds in that single native-first Mac run; this is not a population or iPhone speed estimate. Ten prototype tests passed. It is not integrated into the frozen challenger or app; internal diagnostics and exception behavior can differ.
- A [learning follow-up audit](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/learning_followup.md) confirms that bid-regime inputs already exist, but this Q-only run froze the policy/bidding heads. Its replay contained just 78 unreachable-bid play rows, eight last-to-act. The wrapped bot's 60 locked passes do not establish neural mastery. The note specifies a future full-network outcome-learning direction; nothing additional was trained or selected from it.

## Fairness and promotion boundary

Real and simulated actors must use only their own remembered hand/discard and public information. Actual hidden hands are offline belief-training targets, never online decision inputs. Imagined hands conserve physical card counts; sampler inconsistencies fail closed. Hidden-state-swap audits check features, policy/value outputs, full selected actions, simulated actors, and sampled-world distributions. These are substantial sampled checks, not a formal proof or iPhone verification.

A metadata-only [parent-data and prior-evaluation seed audit](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/lineage_seed_audit_run1049_run1051.md) found no initial-round seed collisions with the 1,280 planned protected round seeds. The [older ancestor audit](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/RUN1046_SEED_PROVENANCE_AUDIT.md) also found no arithmetic collisions under its reconstructed generator, but lacks historical execution receipts and stops at unresolved Run599/Run596 ancestry. These are bounded provenance checks, not exhaustive proof that no historical hand ever recurred. No protected deals were instantiated for these audits.

Neither belief sampler advanced to extra development deals: the original candidate scored 48.44% and the Metropolis candidate scored 49.06% on the initial eight-deal screen. The Metropolis candidate also recorded one additional actor paradox. Its better calibration metrics did not translate into better play in this screen.

Existing neural/game regressions passed 299 tests, and 42 new arena/aggregation/protected-receipt/controller tests passed. The [protocol](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/protocol.json) keeps development and confirmation seeds separate, permits only one distinct protected finalist, and fixes corrected statistical looks at 64, 128, and 256 independent confirmation deals. A positive placement lower bound, nonnegative score estimate, no extra actor paradoxes, all 60 locked cases, and unchanged model/code provenance are required together. Protected blocks require full pre/post model/source/scenario hashes and immutable receipts tied to raw results and locked-case outputs; an unreceipted prior attempt cannot be reused.

The unfinished four-worker search extension was authorized for one throughput-only restart after the other lanes completed: the same bot, seed, and 16 deal IDs, with 12 workers in a new directory. The incomplete attempt is unscored. An isolated orchestration snapshot changes only the worker ceiling; an exact-diff test confirms no inference change. Original development runner and model code are retained unchanged.

The canonical [candidate freeze](/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml/run1052/FROZEN_CANDIDATE.json) was created at 00:28:53 EDT on September 5. Independent confirmation began at 00:29:10 EDT, with 12 CPU workers and a 5,400-second limit per baseline/candidate block. It used the same neural weights and unchanged every-round search, with no belief plugin. Only the planned aggregate looks at 64, 128, and 256 independent deals could determine promotion; the controller stopped at its first all-gates pass, 64 deals.

No claim about human or superhuman strength, other player counts, mobile performance, or deployment is supported by this batch.
