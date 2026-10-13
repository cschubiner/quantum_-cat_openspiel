# Experiment: matchutil

Match-placement-aware utility (view extended with match totals; 90-int view).

**Status / result:** Not adopted: -0.2 [-4.4,+4.1] pp vs pimc:32 (5p, 60 deals). Note: changes the 84-int view contract used by the iOS app.

Source: branch `claude/exp-matchutil` at commit `d55abbaa` (recorded as merged into master; exact history: `git show d55abbaa`).

This directory is a full snapshot of that branch's qcx sources (plus data it added). It is NOT part of the shipped champion; master's `qcx/` remains the champion engine.

Files the branch changed or added: `.gitignore`, `ext_arena.py`, `qc_agents.h`, `qc_arena.cpp`, `qc_capi.cpp`, `qc_core.h`, `qc_py.py`

Build from this directory, e.g. `clang++ -O3 -march=native -std=c++17 -o /tmp/arena qc_arena.cpp`, and run with the working directory set to `qcx/` so `weights/...` paths resolve.
