# Experiment: bidroll

Learned bid model (fit on champion v4's own bids) used for bids inside rollouts.

**Status / result:** Not adopted: 3p +1.4 [-3.5,+6.3], 4p -1.4 [-5.7,+3.0], 5p +1.75 [-1.4,+4.9] pp placement vs v4.

Source: branch `claude/exp-bidroll` at commit `536a41ed` (recorded as merged into master; exact history: `git show 536a41ed`).

This directory is a full snapshot of that branch's qcx sources (plus data it added). It is NOT part of the shipped champion; master's `qcx/` remains the champion engine.

Files the branch changed or added: `.gitignore`, `bids/bm3.txt`, `bids/bm4.txt`, `bids/bm5.txt`, `fit_bidmodel.py`, `qc_arena.cpp`, `qc_biddata2.cpp`, `qc_learn.h`

Build from this directory, e.g. `clang++ -O3 -march=native -std=c++17 -o /tmp/arena qc_arena.cpp`, and run with the working directory set to `qcx/` so `weights/...` paths resolve.
