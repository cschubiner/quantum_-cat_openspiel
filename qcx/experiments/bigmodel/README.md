# Experiment: bigmodel

Bigger policy model over a raw seat-relative state encoding + action features (C++ inference, trainer, data dump).

**Status / result:** UNFINISHED: stopped by request before data generation/evaluation completed. No result.

Source: branch `claude/exp-bigmodel` at commit `85a298f9` (recorded as merged into master; exact history: `git show 85a298f9`).

This directory is a full snapshot of that branch's qcx sources (plus data it added). It is NOT part of the shipped champion; master's `qcx/` remains the champion engine.

Files the branch changed or added: `bench_big.cpp`, `bm_score.cpp`, `qc_arena.cpp`, `qc_big.h`, `qc_gen.cpp`, `test_state.cpp`, `train_big.py`

Build from this directory, e.g. `clang++ -O3 -march=native -std=c++17 -o /tmp/arena qc_arena.cpp`, and run with the working directory set to `qcx/` so `weights/...` paths resolve.
