# Experiment: bidsearch

Nested bid search: each bid valued by playing the round with an inner PIMC for the bidder's own play.

**Status / result:** Not adopted: -1.0 [-4.7,+2.8] pp vs pimc:64 (5p, 60 deals).

Source: branch `claude/exp-bidsearch` at commit `7fc8c150` (recorded as merged into master; exact history: `git show 7fc8c150`).

This directory is a full snapshot of that branch's qcx sources (plus data it added). It is NOT part of the shipped champion; master's `qcx/` remains the champion engine.

Files the branch changed or added: `.gitignore`, `dbg_nb.cpp`, `qc_agents.h`, `qc_arena.cpp`

Build from this directory, e.g. `clang++ -O3 -march=native -std=c++17 -o /tmp/arena qc_arena.cpp`, and run with the working directory set to `qcx/` so `weights/...` paths resolve.
