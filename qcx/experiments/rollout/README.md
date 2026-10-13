# Experiment: rollout

Rollout-policy weight sweep (pimcx spec), two-stage PIMC (pimc2), rollout counter.

**Status / result:** Partly adopted: rollout token penalty 0.6 was promoted into the champion (+3.6 [+0.7,+6.6] pp, fresh deals). Two-stage PIMC: no efficiency gain over flat worlds.

Source: branch `claude/exp-rollout` at commit `712f1579` (recorded as merged into master; exact history: `git show 712f1579`).

This directory is a full snapshot of that branch's qcx sources (plus data it added). It is NOT part of the shipped champion; master's `qcx/` remains the champion engine.

Files the branch changed or added: `qc_agents.h`, `qc_arena.cpp`

Build from this directory, e.g. `clang++ -O3 -march=native -std=c++17 -o /tmp/arena qc_arena.cpp`, and run with the working directory set to `qcx/` so `weights/...` paths resolve.
