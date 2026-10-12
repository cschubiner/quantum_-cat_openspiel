# Why ~1,050 runs of training did not produce a superhuman Quantum Cat bot

Date: 2026-10-12. Sources: `EXPERIMENTS.md`, Run1046–1052 results, code audit, new measurements.

## Diagnosis (hypotheses, with evidence)

| # | Hypothesis | Evidence |
|---|---|---|
| H1 | **Compute starvation.** The learning loop ran on a pure-Python engine with tiny data. | Python engine: ~1,750 *random* rounds/s/core. The largest self-play was ≤180 games/iteration, ≤8 iterations, and a replay buffer capped at 20k positions. The Run1052 Q lane used 500 matches; the 5p checkpoint had "four final self-play matches". AlphaZero-style learning for a 5-player stochastic imperfect-information game needs orders of magnitude more. |
| H2 | **Search was starved, and search is what actually worked.** | Every confirmed gain since Run1046 came from *more search coverage* (Run1051 +1.8pp, Run1052 +7.1pp), never from new weights. Search still covered only the last 3 tricks, used 2 candidate moves and 8 worlds, and never touched bids or discards. Full-information-set search at every decision with real budgets was never tried; the only all-phase test (Run462) used 1 world. |
| H3 | **Discard and bid were not optimized.** | Discard was a fixed rule for every bot. Neural bidding collapsed toward bidding 1, and scored-bid accuracy was 40.5% in champion self-play. A plain hand-written heuristic hits 53% in its own self-play. |
| H4 | **Wrong objective for ~650 runs.** | Runs 375–1034 optimized table-wide paradox avoidance. Run1035 conceded that this was not the official objective, since a paradox penalizes only the player who causes it. |
| H5 | **Weak, inconsistent, or leaky teachers.** | MCTS double-backup bug, rollout teachers reading opponents' hands (0/624 rows information-safe), and 68% of hidden-hand-teacher disagreements contradicting its own labels. |
| H6 | **Hard-to-learn encodings at small data.** | The board is encoded as absolute seat ids cast to floats, played cards as raw rank/colour integers, and nothing is seat-relative. This makes learning from small data harder. |
| H7 | **Measurement only against its own lineage.** | Gains of a few placement points against near-identical copies hid the absolute strength level. A 64-deal confirmation cost about 1 hour of compute (Run1052 vs Run1051 copies), so iteration was slow. |
| H8 | **Beliefs ignore bids.** | World sampling is uniform over card counts, but bids are strong evidence about hands. |
| H9 | **No match-level objective in play.** | Round-level play ignores match standings, while placement is decided on match totals. |

## Strategy

1. Build a fast, rule-exact C++ engine (parity-tested against the Python rules).
2. Use **information-set determinized search (PIMC) at every decision** (discard, bid, play) as the backbone. Learn components that make the search stronger (rollout policy, beliefs), rather than learning a policy that replaces search.
3. Run experiments in separate worktrees, screen them in the fast C++ arena, then confirm against the real Python champion on fresh deals with seat rotation.

## Experiments

| Exp | Worktree / branch | Hypothesis tested |
|---|---|---|
| E1 PIMC everywhere | `base` / `claude/superhuman-base` | H2, H3 |
| E2 budget scaling | `base` | H2 |
| E3 utility: own vs diff vs match placement | `matchutil` / `claude/exp-matchutil` | H9 |
| E4 bid-informed beliefs (importance resampling) | `belief` / `claude/exp-belief` | H8 |
| E5 expert iteration: learned rollout policy from PIMC targets | `exit` / `claude/exp-exit` | H1, H2, H5 |
| E6 rollout-policy tuning | `rollout` / `claude/exp-rollout` | H2 |

Results are recorded in `RESULTS.md`.
