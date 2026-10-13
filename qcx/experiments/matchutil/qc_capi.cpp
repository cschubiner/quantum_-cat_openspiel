// C ABI for Python (ctypes): rules-parity replay and information-set decisions.
#include "qc_agents.h"

using namespace qc;

extern "C" {

// Replays `actions` from the given dealt hands. For each step writes the
// sorted legal action list (up to 40 entries) and its length; finally the raw
// round scores. Returns number of steps applied, or -1 on an illegal action.
int qc_replay_check(int P, int start, const int* hands, int nact, const int* actions,
                    int* legal_out, int* nlegal_out, double* scores_out) {
  Round s;
  s.init(P, start);
  for (int p = 0; p < P; ++p) {
    s.hsize[p] = 0;
    for (int r = 0; r < s.R; ++r) { s.hand[p][r] = (uint8_t)hands[p * s.R + r]; s.hsize[p] += hands[p * s.R + r]; }
  }
  int acts[NC * MAXR + 1];
  for (int i = 0; i < nact; ++i) {
    if (s.done()) return i;
    int n = s.legal(acts);
    nlegal_out[i] = n;
    for (int k = 0; k < n; ++k) legal_out[i * 40 + k] = acts[k];
    bool ok = false;
    for (int k = 0; k < n; ++k) ok |= acts[k] == actions[i];
    if (!ok) return -1;
    s.apply(actions[i]);
  }
  if (s.done()) s.scores(scores_out);
  return nact;
}

// View layout (ints, fixed padding MAXP=5, MAXR=9):
//  0 P, 1 start, 2 phase, 3 me(=current player), 4 leader, 5 led, 6 trick_no, 7 nplayed
//  8..16 my hand[9], 17 my discard, 18..22 bids[5], 23..58 board[4][9],
//  59..63 tokens[5], 64..68 tricks[5], 69..73 trank[5], 74..78 tcolor[5], 79..83 hsize[5]
//  84..88 match totals before this round (x100, rounded), 89 rounds remaining after this one
static void view_to_round(const int* v, Round& s) {
  s.init(v[0], v[1]);
  s.phase = (int8_t)v[2]; s.cur = (int8_t)v[3]; s.leader = (int8_t)v[4]; s.led = (int8_t)v[5];
  s.trick_no = (int8_t)v[6]; s.nplayed = (int8_t)v[7];
  const int me = v[3];
  for (int r = 0; r < MAXR; ++r) s.hand[me][r] = (uint8_t)v[8 + r];
  s.discard[me] = (int8_t)v[17];
  s.red_count = 0;
  for (int p = 0; p < MAXP; ++p) {
    s.bid[p] = (int8_t)v[18 + p];
    s.tokens[p] = (uint8_t)v[59 + p];
    s.tricks[p] = (int8_t)v[64 + p];
    s.trank[p] = (int8_t)v[69 + p];
    s.tcolor[p] = (int8_t)v[74 + p];
    s.hsize[p] = (int8_t)v[79 + p];
    s.mtot[p] = v[84 + p] / 100.0;
  }
  s.rounds_after = (int8_t)v[89];
  for (int c = 0; c < NC; ++c)
    for (int r = 0; r < MAXR; ++r) {
      s.board[c][r] = (int8_t)v[23 + c * MAXR + r];
      if (c == RED && r < s.R && s.board[c][r] >= 0) s.red_count++;
    }
}

int qc_decide(const int* view, int worlds, int util, unsigned long long seed, double* values_out, double sigma) {
  Round s;
  view_to_round(view, s);
  SearchParams sp;
  sp.worlds = worlds; sp.util = util; sp.sigma = sigma;
  Rng rng(seed);
  return pimc_decide(s, s.cur, sp, rng, values_out);
}

int qc_legal_from_view(const int* view, int* out) {
  Round s;
  view_to_round(view, s);
  return s.legal(out);
}

}  // extern "C"
