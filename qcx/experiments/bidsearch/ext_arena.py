"""Run the original full-match evaluator with C++ search bots as external seats.

Bot names: qcx_<worlds>_<util>_<disc>   e.g. qcx_64_0_own  (own discard search)
                                              qcx_64_0_shared (forced shared discard)
Everything else is resolved by quantum_cat_full_match_elo as usual.
"""
import os, sys
REPO = "/Users/canal/quantum_-cat_openspiel"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantum_cat_full_match_elo as elo
import qc_py


class QcxBot:
  uses_neural_prediction = True

  def __init__(self, name, seed):
    _, worlds, util, disc = name.split("_")[:4]
    self.name = name
    self.worlds = int(worlds); self.util = int(util)
    self.chooses_discard = disc == "own"
    self.seed = int(seed) & 0xFFFFFFFF
    self.calls = 0

  def step(self, state, player):
    self.calls += 1
    return int(qc_py.decide(state, player, self.worlds, self.util,
                            self.seed * 1000003 + self.calls))


_orig_make = elo.make_eval_bot
def _make(name, seed, *a, **kw):
  if str(name).startswith("qcx_"):
    return QcxBot(name, seed)
  return _orig_make(name, seed, *a, **kw)
elo.make_eval_bot = _make

_CURRENT = {"bots": None}
_orig_round = elo.play_round
def _play_round(game, bots, *a, **kw):
  _CURRENT["bots"] = bots
  return _orig_round(game, bots, *a, **kw)
elo.play_round = _play_round

_orig_discard = elo.shared_discard_action
def _discard(state, player, legal=None):
  bots = _CURRENT["bots"]
  if bots is not None and getattr(bots[player], "chooses_discard", False):
    return None
  return _orig_discard(state, player, legal)
elo.shared_discard_action = _discard

if __name__ == "__main__":
  os.chdir(REPO)
  raise SystemExit(elo.main())
