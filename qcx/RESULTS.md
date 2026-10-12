# qcx experiment results (running log)

All C++-arena rows are 5-player complete matches: one candidate seat against a field, common deals rotated through every seat, paired against an all-field reference table on the same deals. "Placement" is pairwise placement (0.5 = parity). CIs are 95% over independent deals. Python-evaluator rows use the original `quantum_cat_full_match_elo.py` gate with the original champion bots.

## E0 — engine
- C++ engine passes the rules-parity test against `open_spiel/python/games/quantum_cat.py`: 9,000 random rounds (3/4/5p), identical legal-action lists at every step (also when rebuilt from the information-only view) and identical scores.
- Speed (unloaded machine): heuristic rollout ~68 µs per 5p round per thread. The Python engine runs ~570 µs per *random* round.

## E1 — PIMC at every decision vs the champion (Python evaluator)

| Candidate | Opponents | Deals (matches) | Pairwise placement | First place | Score adv./match | Own paradox | Bid hit |
|---|---|---|---|---|---|---|---|
| qcx PIMC 64 worlds | 4× Run1051 champion | 32 (160) | **87.1%** | **72.5%** (par 20%) | **+8.13** | 12.6% (champ 13.6%) | 51.8% |
| qcx PIMC 128 worlds | 4× Run1052 champion | 16 (80) | *running* | | | | |

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
