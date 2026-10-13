# Experiment: belief

Bid-informed beliefs: worlds importance-resampled by a multinomial-logistic bid model.

**Status / result:** Not adopted: +2.5 [-0.6,+5.6] (60 deals), +0.9 [-1.5,+3.3] (140 fresh deals) pp vs pimc:32.

Source: branch `claude/exp-belief` at commit `cc913a0e` (recorded as merged into master; exact history: `git show cc913a0e`).

This directory is a full snapshot of that branch's qcx sources (plus data it added). It is NOT part of the shipped champion; master's `qcx/` remains the champion engine.

Files the branch changed or added: `.gitignore`, `bidmodel5.txt`, `fit_bidmodel.py`, `qc_agents.h`, `qc_arena.cpp`, `qc_biddata.cpp`

Build from this directory, e.g. `clang++ -O3 -march=native -std=c++17 -o /tmp/arena qc_arena.cpp`, and run with the working directory set to `qcx/` so `weights/...` paths resolve.
