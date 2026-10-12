"""ctypes bridge to the C++ engine + information-safe view extraction."""
import ctypes, os
import numpy as np

_LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libqc.dylib"))
_LIB.qc_replay_check.restype = ctypes.c_int
_LIB.qc_decide.restype = ctypes.c_int
_LIB.qc_decide.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                           ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_double)]
COLORS = {"R": 0, "B": 1, "Y": 2, "G": 3}
VIEW_LEN = 84


def build_view(state, me):
  """Only the acting player's legal information: own hand/discard + public state."""
  P = state._num_players
  R = state._num_card_types
  v = np.zeros(VIEW_LEN, dtype=np.int32)
  v[0] = P; v[1] = state._round_start_player; v[2] = state._phase; v[3] = me
  v[4] = state._start_player
  v[5] = -1 if state._led_color is None else COLORS[state._led_color]
  v[6] = state._trick_number
  v[7] = sum(1 for c in state._cards_played_this_trick if c is not None)
  v[8:8 + R] = state._hands[me]
  v[17] = (state._discarded_cards[me] - 1) if state._discarded_cards[me] > 0 else -1
  board = np.array(state._board_ownership)
  for p in range(5):
    v[18 + p] = state._predictions[p] if p < P else -1
    v[69 + p] = -1; v[74 + p] = -1
  v[23:59] = -1
  for c in range(4):
    for r in range(R):
      v[23 + c * 9 + r] = board[c, r]
  for p in range(P):
    v[59 + p] = sum(int(bool(state._color_tokens[p][c])) << c for c in range(4))
    v[64 + p] = state._tricks_won[p]
    card = state._cards_played_this_trick[p]
    if card is not None:
      v[69 + p] = card[0] - 1; v[74 + p] = COLORS[card[1]]
    played = int(np.sum(board == p))
    disc = 1 if (state._phase != 1 or state._has_discarded[p]) else 0
    v[79 + p] = state._cards_per_player_initial - disc - played
  return v


def decide(state, me, worlds=64, util=0, seed=0):
  v = build_view(state, me)
  vals = (ctypes.c_double * 41)()
  return _LIB.qc_decide(v.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), worlds, util, seed, vals)
