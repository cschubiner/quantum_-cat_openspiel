# Experiment: deepsearch

Deeper search: single-observer ISMCTS with PUCT prior from the learned policy; PIMC-2 (inner PIMC for the root player's next decision).

**Status / result:** UNFINISHED: stopped by request while arena runs were in progress. No result.

Source: branch `claude/exp-deepsearch` at commit `528ad6f4` (recorded as merged into master; exact history: `git show 528ad6f4`).

This directory is a full snapshot of that branch's qcx sources (plus data it added). It is NOT part of the shipped champion; master's `qcx/` remains the champion engine.

Files the branch changed or added: `bench_deepsearch.cpp`, `qc_arena.cpp`, `qc_bots.h`, `qc_search.h`, `test_deepsearch.cpp`

Build from this directory, e.g. `clang++ -O3 -march=native -std=c++17 -o /tmp/arena qc_arena.cpp`, and run with the working directory set to `qcx/` so `weights/...` paths resolve.
