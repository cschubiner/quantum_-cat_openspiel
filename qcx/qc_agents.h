// Agents: heuristic policy, information-set world sampler, determinized
// Monte-Carlo search (PIMC).  Every agent decision takes a *masked* Round
// (opponents' hands/discards zeroed) plus the acting seat; nothing here can
// read hidden information because it is never handed any.
#pragma once
#include "qc_core.h"

namespace qc {

// ---------------------------------------------------------------- masking
inline bool has_discarded(const Round& s, int q) {
  return s.phase != DISCARD || q < s.cur;
}

inline Round mask_for(const Round& s, int me) {
  Round v = s;
  for (int q = 0; q < s.P; ++q) {
    if (q == me) continue;
    std::memset(v.hand[q], 0, sizeof(v.hand[q]));
    v.discard[q] = -1;
  }
  return v;
}

// Uniformly sample hidden hands/discards consistent with `me`'s information.
inline void sample_world(const Round& view, int me, Rng& rng, Round& out) {
  out = view;
  int pool[5 * MAXR];
  int cnt[MAXR];
  for (int r = 0; r < view.R; ++r) cnt[r] = 5 - view.hand[me][r];
  if (view.discard[me] >= 0) cnt[view.discard[me]]--;
  for (int c = 0; c < NC; ++c)
    for (int r = 0; r < view.R; ++r)
      if (view.board[c][r] >= 0) cnt[r]--;
  int n = 0;
  for (int r = 0; r < view.R; ++r)
    for (int k = 0; k < cnt[r]; ++k) pool[n++] = r;
  for (int i = n - 1; i > 0; --i) std::swap(pool[i], pool[rng.below(i + 1)]);
  int pos = 0;
  for (int q = 0; q < view.P; ++q) {
    if (q == me) continue;
    std::memset(out.hand[q], 0, sizeof(out.hand[q]));
    if (has_discarded(view, q)) out.discard[q] = (int8_t)pool[pos++];
    for (int k = 0; k < view.hsize[q]; ++k) out.hand[q][pool[pos++]]++;
  }
}

// ---------------------------------------------------------------- heuristic
struct HeurParams {
  double eps = 0.0;        // probability of uniformly random legal action
  double temp = 0.0;       // softmax temperature (0 => argmax)
  double w_win = 3.0, w_token = 1.2, w_dead = 6.0, w_liq = 0.08, w_adj = 0.35;
};

inline double bid_scale(const Round& s) {
  double mean = 0;
  for (int r = 0; r < s.R; ++r) mean += std::pow((r + 1.0) / s.R, 3.0);
  mean /= s.R;
  return (double)s.T / s.P / ((s.H - 1) * mean);
}

inline double hand_strength(const Round& s, int p) {
  double e = 0;
  for (int r = 0; r < s.R; ++r) e += s.hand[p][r] * std::pow((r + 1.0) / s.R, 3.0);
  return e * bid_scale(s);
}

inline int heuristic_bid(const Round& s, int p) {
  int b = (int)std::lround(hand_strength(s, p));
  return PRED_OFF + std::min(4, std::max(1, b));
}

inline int heuristic_discard(const Round& s, int p, Rng& rng) {
  int best = -1; double bv = -1e9;
  for (int r = 0; r < s.R; ++r) {
    if (!s.hand[p][r]) continue;
    double v = s.hand[p][r] * 1.0 - std::fabs(r - (s.R - 1) / 2.0) * 0.1 + rng.uniform() * 0.05;
    if (v > bv) { bv = v; best = r; }
  }
  return best;
}

// Does playing (c,r) make p the current trick winner?
inline bool wins_now(const Round& s, int c, int r) {
  if (s.nplayed == 0) return true;
  int lc = s.led;
  int best_red = -1, best_led = -1;
  for (int q = 0; q < s.P; ++q) {
    if (s.tcolor[q] == RED) best_red = std::max<int>(best_red, s.trank[q]);
    if (s.tcolor[q] == lc) best_led = std::max<int>(best_led, s.trank[q]);
  }
  if (c == RED) return r > best_red;
  if (best_red >= 0) return false;
  return c == lc && r > best_led;
}

inline double heuristic_move_score(const Round& s, int a, const HeurParams& hp) {
  const int p = s.cur;
  const int c = a / s.R, r = a % s.R;
  const int b = s.bid[p], t = s.tricks[p];
  const int rem = s.T - s.trick_no;
  bool want = (t != b);  // below bid or already busted -> tricks are good
  if (t < b && b - t > rem) want = true;
  const bool last = s.nplayed == s.P - 1;
  double pwin = 0;
  if (wins_now(s, c, r)) {
    if (last) pwin = 1;
    else {
      double rel = (r + 1.0) / s.R;
      pwin = (c == RED) ? 0.35 + 0.65 * rel * rel : 0.6 * rel * rel * rel;
      // how many unclaimed higher cells in same color (crude)
      int higher_free = 0;
      for (int r2 = r + 1; r2 < s.R; ++r2) higher_free += s.board[c][r2] == -1;
      pwin *= 1.0 / (1.0 + 0.5 * higher_free * (s.P - 1 - s.nplayed) / (double)(s.P - 1));
    }
  }
  double score = want ? hp.w_win * pwin : -hp.w_win * pwin;
  bool token_loss = s.led >= 0 && c != s.led;
  if (token_loss) score -= hp.w_token;
  // liquidity after the move (computed without copying the state)
  int plays_left = s.T - s.trick_no - 1;  // my future plays this round
  if (plays_left > 0) {
    uint8_t tok = s.tokens[p];
    if (token_loss) tok &= (uint8_t)~(1u << s.led);
    int liq = 0;
    for (int c2 = 0; c2 < NC; ++c2) {
      if (!(tok >> c2 & 1)) continue;
      for (int r2 = 0; r2 < s.R; ++r2) {
        int h = s.hand[p][r2] - (r2 == r);
        if (h > 0 && s.board[c2][r2] == -1 && !(c2 == c && r2 == r)) ++liq;
      }
    }
    if (liq == 0) score -= hp.w_dead;
    else score += hp.w_liq * std::min(liq, 8);
  }
  // adjacency (matters when on track for the bid)
  if (!want || t < b) {
    int adj = 0;
    if (c > 0 && s.board[c - 1][r] == p) ++adj;
    if (c < NC - 1 && s.board[c + 1][r] == p) ++adj;
    if (r > 0 && s.board[c][r - 1] == p) ++adj;
    if (r < s.R - 1 && s.board[c][r + 1] == p) ++adj;
    score += hp.w_adj * adj;
  }
  double rel = (double)r / s.R;
  if (pwin == 0) score += want ? -0.15 * rel : 0.25 * rel;  // dump low when chasing, high when ducking
  else score += want ? -0.1 * rel : 0;
  return score;
}

inline int heuristic_play(const Round& s, Rng& rng, const HeurParams& hp) {
  int acts[NC * MAXR + 1];
  int n = s.legal(acts);
  if (n == 1) return acts[0];
  if (hp.eps > 0 && rng.uniform() < hp.eps) return acts[rng.below(n)];
  double sc[NC * MAXR];
  double best = -1e18; int bi = 0;
  for (int i = 0; i < n; ++i) {
    sc[i] = heuristic_move_score(s, acts[i], hp) + rng.uniform() * 1e-6;
    if (sc[i] > best) { best = sc[i]; bi = i; }
  }
  if (hp.temp <= 0) return acts[bi];
  double z = 0;
  for (int i = 0; i < n; ++i) { sc[i] = std::exp((sc[i] - best) / hp.temp); z += sc[i]; }
  double u = rng.uniform() * z;
  for (int i = 0; i < n; ++i) { u -= sc[i]; if (u <= 0) return acts[i]; }
  return acts[n - 1];
}

inline int heuristic_action(const Round& s, Rng& rng, const HeurParams& hp) {
  if (s.phase == DISCARD) return heuristic_discard(s, s.cur, rng);
  if (s.phase == BID) return heuristic_bid(s, s.cur);
  return heuristic_play(s, rng, hp);
}

inline void rollout(Round& s, Rng& rng, const HeurParams& hp) {
  while (!s.done()) s.apply(heuristic_action(s, rng, hp));
}

// ---------------------------------------------------------------- search
struct SearchParams {
  int worlds = 64;
  int util = 0;  // 0: own - mean(others); 1: own score
  HeurParams rollout;
  // match context (optional): totals before this round, rounds remaining after this one
};

inline double utility(const Round& s, int me, int mode) {
  double sc[MAXP];
  s.scores(sc);
  if (mode == 1) return sc[me];
  double o = 0;
  for (int q = 0; q < s.P; ++q) if (q != me) o += sc[q];
  return sc[me] - o / (s.P - 1);
}

// PIMC with common random numbers across actions.
inline int pimc_decide(const Round& view, int me, const SearchParams& sp, Rng& rng,
                       double* out_values = nullptr) {
  int acts[NC * MAXR + 1];
  int n = view.legal(acts);
  if (n == 1) return acts[0];
  double sum[NC * MAXR + 1] = {0};
  Round world, s;
  for (int w = 0; w < sp.worlds; ++w) {
    sample_world(view, me, rng, world);
    uint64_t seed = rng.next();
    for (int i = 0; i < n; ++i) {
      s = world;
      s.apply(acts[i]);
      Rng rr(seed);
      rollout(s, rr, sp.rollout);
      sum[i] += utility(s, me, sp.util);
    }
  }
  int bi = 0;
  for (int i = 1; i < n; ++i) if (sum[i] > sum[bi]) bi = i;
  if (out_values) for (int i = 0; i < n; ++i) out_values[i] = sum[i] / sp.worlds;
  return acts[bi];
}

}  // namespace qc
