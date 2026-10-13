import ctypes, random, sys
sys.path.insert(0, '/Users/canal/quantum_-cat_openspiel')
import numpy as np, pyspiel
from open_spiel.python.games import quantum_cat  # noqa
import qc_py

L = qc_py._LIB
def run(P, seed):
  rng = random.Random(seed)
  start = rng.randrange(P)
  g = pyspiel.load_game("python_quantum_cat", {"players": P, "start_player": start})
  s = g.new_initial_state()
  while s.is_chance_node():
    s.apply_action(rng.randrange(len(s.chance_outcomes())))
  hands = np.array([h for h in s._hands], dtype=np.int32).ravel()
  acts, legals = [], []
  views_ok = True
  while not s.is_terminal():
    la = sorted(s.legal_actions())
    legals.append(la)
    me = s.current_player()
    # view-based legal list must match too
    v = qc_py.build_view(s, me)
    out = (ctypes.c_int * 40)()
    n = L.qc_legal_from_view(v.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), out)
    if sorted(out[:n]) != la: views_ok = False
    a = rng.choice(la) if rng.random() < 0.5 else la[0]
    acts.append(a); s.apply_action(a)
  n = len(acts)
  A = (ctypes.c_int * n)(*acts)
  lo = (ctypes.c_int * (n * 40))(); nl = (ctypes.c_int * n)(); sc = (ctypes.c_double * 5)()
  H = (ctypes.c_int * len(hands))(*hands)
  r = L.qc_replay_check(P, start, H, n, A, lo, nl, sc)
  assert r == n, (P, seed, r)
  for i in range(n):
    assert sorted(lo[i*40:i*40+nl[i]]) == legals[i], (P, seed, i)
  py = s.raw_round_scores()
  assert list(sc[:P]) == py, (P, seed, list(sc[:P]), py)
  assert views_ok, (P, seed, "view legal mismatch")

for P in (3, 4, 5):
  for seed in range(3000):
    run(P, seed)
print("parity OK: 9000 random rounds (3/4/5p), legal lists, view legal lists, scores")
