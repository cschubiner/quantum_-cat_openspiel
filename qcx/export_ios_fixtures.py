"""Export cross-language fixtures for the iOS qcx champion.

Each fixture is one decision in a real Python (OpenSpiel rules) round:
  * the dealt hands, start player and the Python action history (so Swift can
    replay the identical round in its own engine and rebuild the view),
  * the 84-int view from qc_py.build_view (acting player's information only),
  * the sorted legal action ids,
  * qc_py.decide(state, me, worlds, util, seed) for a fixed seed.

Swift asserts: Swift-built view == fixture view, C++ legal list == fixture
legal list, and qc_decide on iOS returns the same action (the RNG is
deterministic, so the whole search is bit-identical across platforms).

usage: .venv/bin/python qcx/export_ios_fixtures.py OUT.json
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import pyspiel  # noqa: E402
from open_spiel.python.games import quantum_cat  # noqa: E402,F401
import qc_py  # noqa: E402
import json as _json
_CHAMP = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qcx_champion.json")))
if _CHAMP.get("rollout_policy"):
  qc_py.load_policy(os.path.join(os.path.dirname(os.path.abspath(__file__)), _CHAMP["rollout_policy"]))
  _DECIDE = qc_py.decide_learned
else:
  _DECIDE = qc_py.decide

FIXTURE_WORLDS = 64
FIXTURE_UTIL = 0
GAMES_PER_PLAYER_COUNT = 4


def play_round(P, game_seed):
  rng = random.Random(game_seed)
  start = rng.randrange(P)
  g = pyspiel.load_game("python_quantum_cat", {"players": P, "start_player": start})
  s = g.new_initial_state()
  while s.is_chance_node():
    s.apply_action(rng.randrange(len(s.chance_outcomes())))
  hands = [[int(x) for x in h] for h in s._hands]
  history = []
  decisions = []
  while not s.is_terminal():
    me = s.current_player()
    decisions.append((s.clone(), me, list(history)))
    # Realistic trajectories: a cheap search for every seat.
    a = int(qc_py.decide(s, me, 8, 0, game_seed * 7919 + len(history)))
    history.append(a)
    s.apply_action(a)
  return P, start, hands, decisions, rng


def pick(decisions, rng):
  by_phase = {1: [], 2: [], 3: []}
  for d in decisions:
    by_phase.setdefault(d[0]._phase, []).append(d)
  chosen = []
  if by_phase[1]:
    chosen.append(rng.choice(by_phase[1]))
  if by_phase[2]:
    chosen.append(rng.choice(by_phase[2]))
  plays = by_phase[3]
  if plays:
    for frac in (0.0, 0.35, 0.7, 0.99):
      chosen.append(plays[min(len(plays) - 1, int(frac * len(plays)))])
  return chosen


def main():
  out_path = sys.argv[1]
  fixtures = []
  for P in (3, 4, 5):
    for k in range(GAMES_PER_PLAYER_COUNT):
      game_seed = 1000 * P + k
      P_, start, hands, decisions, rng = play_round(P, game_seed)
      for state, me, history in pick(decisions, rng):
        view = [int(x) for x in qc_py.build_view(state, me)]
        seed = (game_seed * 1_000_003 + len(history) * 97 + 12345) & 0x7FFFFFFFFFFFFFFF
        action = int(_DECIDE(state, me, FIXTURE_WORLDS, FIXTURE_UTIL, seed))
        fixtures.append({
            "players": P,
            "start": start,
            "hands": hands,
            "history": history,
            "me": me,
            "phase": int(state._phase),
            "view": view,
            "legal": sorted(int(a) for a in state.legal_actions()),
            "worlds": FIXTURE_WORLDS,
            "util": FIXTURE_UTIL,
            "seed": seed,
            "action": action,
        })
  doc = {"view_len": qc_py.VIEW_LEN, "generator": "qcx/export_ios_fixtures.py", "fixtures": fixtures}
  with open(out_path, "w") as f:
    json.dump(doc, f, separators=(",", ":"))
    f.write("\n")
  print(f"wrote {len(fixtures)} fixtures to {out_path}")


if __name__ == "__main__":
  main()
