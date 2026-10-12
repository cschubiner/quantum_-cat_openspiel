// Fast Quantum Cat ("Cat in the Box") round engine + information-set search.
//
// Rules mirror open_spiel/python/games/quantum_cat.py for 3..5 players exactly
// (action encoding included) so that decisions can be exchanged with the
// Python evaluator.  The search code only ever reads an InfoView (the acting
// player's legal information); hidden hands are re-sampled per world.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace qc {

constexpr int MAXP = 5, MAXR = 9, NC = 4;
constexpr int ACT_PARADOX = 999, PRED_OFF = 100;
constexpr int RED = 0;

struct Rng {  // xoshiro256**
  uint64_t s[4];
  explicit Rng(uint64_t seed = 1) { reseed(seed); }
  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  void reseed(uint64_t seed) {
    for (auto& v : s) v = splitmix(seed);
  }
  static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  inline uint64_t next() {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  inline uint32_t below(uint32_t n) { return (uint32_t)(((next() >> 32) * (uint64_t)n) >> 32); }
  inline double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

struct Rules {
  int P, R, H, T;
};
inline Rules rules_for(int P) {
  if (P == 3) return {3, 6, 10, 8};
  if (P == 4) return {4, 8, 10, 8};
  return {5, 9, 9, 7};
}

enum Phase : int8_t { DISCARD = 1, BID = 2, PLAY = 3, DONE = 4 };

struct Round {
  int8_t P, R, H, T;
  int8_t phase, cur, start, leader, led, trick_no, nplayed, paradox;
  uint8_t hand[MAXP][MAXR];
  int8_t hsize[MAXP];
  int8_t discard[MAXP];  // rank idx, -1 none
  int8_t bid[MAXP];      // 1..4, -1 none
  int8_t board[NC][MAXR];
  uint8_t tokens[MAXP];
  int8_t tricks[MAXP];
  int8_t trank[MAXP], tcolor[MAXP];
  int8_t red_count;
  // match context (public): totals before this round, rounds remaining after it
  double mtot[MAXP];
  int8_t rounds_after;

  void init(int P_, int start_) {
    Rules r = rules_for(P_);
    P = r.P; R = r.R; H = r.H; T = r.T;
    phase = DISCARD; cur = 0; start = start_; leader = start_; led = -1;
    trick_no = 0; nplayed = 0; paradox = -1; red_count = 0;
    std::memset(hand, 0, sizeof(hand));
    for (int p = 0; p < MAXP; ++p) {
      hsize[p] = 0; discard[p] = -1; bid[p] = -1; tokens[p] = 0xF; tricks[p] = 0;
      trank[p] = -1; tcolor[p] = -1;
    }
    std::memset(board, -1, sizeof(board));
    for (int p = 0; p < MAXP; ++p) mtot[p] = 0;
    rounds_after = 0;
  }
  // Uniform deal (equivalent in distribution to the Python chance process).
  void deal(Rng& rng) {
    int deck[5 * MAXR];
    int n = 0;
    for (int r = 0; r < R; ++r)
      for (int k = 0; k < 5; ++k) deck[n++] = r;
    for (int i = n - 1; i > 0; --i) std::swap(deck[i], deck[rng.below(i + 1)]);
    int pos = 0;
    for (int p = 0; p < P; ++p) {
      for (int k = 0; k < H; ++k) hand[p][deck[pos++]]++;
      hsize[p] = H;
    }
  }
  bool done() const { return phase == DONE; }
  bool red_open() const { return red_count > 0; }
  bool playable(int p, int c, int r) const {
    return hand[p][r] > 0 && (tokens[p] >> c & 1) && board[c][r] == -1;
  }
  // Legal actions for current player. Returns count.
  int legal(int* out) const {
    int n = 0;
    const int p = cur;
    if (phase == DISCARD) {
      for (int r = 0; r < R; ++r)
        if (hand[p][r]) out[n++] = r;
      return n;
    }
    if (phase == BID) {
      for (int b = 1; b <= 4; ++b) out[n++] = PRED_OFF + b;
      return n;
    }
    if (phase != PLAY) return 0;
    int nred = 0;
    // Python sorts by action id = c*R + r, so iterate color-major.
    for (int c = 0; c < NC; ++c) {
      if (!(tokens[p] >> c & 1)) continue;
      for (int r = 0; r < R; ++r) {
        if (hand[p][r] && board[c][r] == -1) {
          out[n++] = c * R + r;
          if (c == RED) ++nred;
        }
      }
    }
    if (led < 0 && !red_open() && nred > 0 && n > nred) {
      // drop red leads (they are first in the list)
      for (int i = nred; i < n; ++i) out[i - nred] = out[i];
      n -= nred;
    }
    if (n == 0) out[n++] = ACT_PARADOX;
    return n;
  }
  int count_legal_play(int p) const {  // ignoring red-lead restriction
    int n = 0;
    for (int c = 0; c < NC; ++c) {
      if (!(tokens[p] >> c & 1)) continue;
      for (int r = 0; r < R; ++r)
        if (hand[p][r] && board[c][r] == -1) ++n;
    }
    return n;
  }
  int trick_winner() const {
    int best = -1, bestr = -1;
    for (int q = 0; q < P; ++q)
      if (tcolor[q] == RED && trank[q] > bestr) { bestr = trank[q]; best = q; }
    if (best >= 0) return best;
    for (int q = 0; q < P; ++q)
      if (tcolor[q] == led && trank[q] > bestr) { bestr = trank[q]; best = q; }
    return best;
  }
  // Who would currently win if trick ended now (requires >=1 card played).
  int current_winner() const {
    int best = -1, bestr = -1;
    for (int q = 0; q < P; ++q)
      if (tcolor[q] == RED && trank[q] > bestr) { bestr = trank[q]; best = q; }
    if (best >= 0) return best;
    for (int q = 0; q < P; ++q)
      if (tcolor[q] == led && trank[q] > bestr) { bestr = trank[q]; best = q; }
    return best;
  }
  void apply(int a) {
    const int p = cur;
    if (phase == DISCARD) {
      hand[p][a]--; hsize[p]--; discard[p] = (int8_t)a;
      if (p == P - 1) { phase = BID; cur = start; }
      else cur = p + 1;
      return;
    }
    if (phase == BID) {
      bid[p] = (int8_t)(a - PRED_OFF);
      int nx = (p + 1) % P;
      if (nx == start) { phase = PLAY; cur = start; leader = start; led = -1; nplayed = 0; }
      else cur = nx;
      return;
    }
    // PLAY
    if (a == ACT_PARADOX) { paradox = p; phase = DONE; return; }
    const int c = a / R, r = a % R;
    hand[p][r]--; hsize[p]--;
    board[c][r] = (int8_t)p;
    if (c == RED) ++red_count;
    trank[p] = (int8_t)r; tcolor[p] = (int8_t)c;
    if (led < 0) led = (int8_t)c;
    else if (c != led) tokens[p] &= (uint8_t)~(1u << led);
    ++nplayed;
    int nx = (p + 1) % P;
    if (nplayed == P) {
      int w = trick_winner();
      tricks[w]++;
      trick_no++;
      leader = (int8_t)w; cur = (int8_t)w; led = -1; nplayed = 0;
      for (int q = 0; q < P; ++q) { trank[q] = -1; tcolor[q] = -1; }
      if (trick_no >= T) phase = DONE;
    } else {
      cur = (int8_t)nx;
    }
  }
  int largest_cluster(int p) const {
    uint64_t seen = 0;  // NC*MAXR = 36 bits
    int best = 0;
    int stack[NC * MAXR];
    for (int c = 0; c < NC; ++c)
      for (int r = 0; r < R; ++r) {
        int id = c * MAXR + r;
        if (board[c][r] != p || (seen >> id & 1)) continue;
        int sp = 0, size = 0;
        stack[sp++] = id; seen |= 1ULL << id;
        while (sp) {
          int cur_id = stack[--sp]; ++size;
          int cc = cur_id / MAXR, rr = cur_id % MAXR;
          const int dc[4] = {1, -1, 0, 0}, dr[4] = {0, 0, 1, -1};
          for (int k = 0; k < 4; ++k) {
            int c2 = cc + dc[k], r2 = rr + dr[k];
            if (c2 < 0 || c2 >= NC || r2 < 0 || r2 >= R) continue;
            int id2 = c2 * MAXR + r2;
            if (board[c2][r2] == p && !(seen >> id2 & 1)) { seen |= 1ULL << id2; stack[sp++] = id2; }
          }
        }
        best = std::max(best, size);
      }
    return best;
  }
  void scores(double* out) const {
    for (int p = 0; p < P; ++p) {
      if (paradox == p) out[p] = -tricks[p];
      else if (bid[p] == tricks[p]) out[p] = tricks[p] + largest_cluster(p);
      else out[p] = tricks[p];
    }
  }
  int cells_owned(int p) const {
    int n = 0;
    for (int c = 0; c < NC; ++c)
      for (int r = 0; r < R; ++r) n += board[c][r] == p;
    return n;
  }
};

}  // namespace qc
