# Unfinished work, preserved but not applied

`open_spiel_local_fixes_unverified.patch` contains local edits that sat uncommitted in the
main checkout before 2026-10-12 (fixes to open_spiel examples, `alpha_zero.py`, the Python
`atari`, `dynamic_routing`, `liars_poker` and `team_dominoes` games, their tests, and
`setup.py`). They were not written in the qcx session and were not applied to master
because, with the patch applied, tests fail that pass on master:

- `liars_poker_test.py`: 13 tests, 3 failures
- `team_dominoes_test.py`: 6 tests, 2 failures + 1 error
- `dynamic_routing_test.py`: 20 tests, 2 failures

To resume: `git apply wip/open_spiel_local_fixes_unverified.patch`, then fix the failures.
