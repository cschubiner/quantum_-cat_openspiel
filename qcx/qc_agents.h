// Agents: heuristic policy, information-set world sampler, determinized
// Monte-Carlo search (PIMC).  Every agent decision takes a *masked* Round
// (opponents' hands/discards zeroed) plus the acting seat; nothing here can
// read hidden information because it is never handed any.
#pragma once
#include <cstdio>
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

// ---------------------------------------------------------------- bid model
// Multinomial logistic P(bid | hand at bid time, bid position, earlier bids),
// fitted offline on self-play.  Feature layout must match fit_bidmodel.py.
struct BidModel {
  int P = 0, F = 0;
  std::vector<double> W;  // 4 x F
  bool ok() const { return F > 0; }
  bool load(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    if (std::fscanf(f, "%d %d", &P, &F) != 2) { std::fclose(f); return false; }
    W.resize(4 * F);
    for (auto& w : W) if (std::fscanf(f, "%lf", &w) != 1) { std::fclose(f); F = 0; return false; }
    std::fclose(f);
    return true;
  }
};

inline int bid_features(int P, int R, const int* hand, int pos, int nprev, int sumprev, double* x) {
  int n = 0;
  x[n++] = 1.0;
  int cards = 0; double strength = 0;
  for (int r = 0; r < R; ++r) { x[n++] = hand[r]; cards += hand[r]; strength += hand[r] * (r + 1.0) / R; }
  for (int r = 0; r < R; ++r) x[n++] = hand[r] >= 2 ? 1.0 : 0.0;
  // top-k counts
  int top1 = hand[R - 1], top2 = top1 + hand[R - 2], top3 = top2 + hand[R - 3];
  x[n++] = top1; x[n++] = top2; x[n++] = top3; x[n++] = strength; x[n++] = strength * strength / 10.0;
  for (int k = 0; k < MAXP; ++k) x[n++] = (k == pos) ? 1.0 : 0.0;
  x[n++] = nprev ? (double)sumprev / nprev : 0.0;
  x[n++] = sumprev;
  return n;
}

inline void bid_logprobs(const BidModel& m, int R, const int* hand, int pos, int nprev, int sumprev, double* lp) {
  double x[64];
  int F = bid_features(m.P, R, hand, pos, nprev, sumprev, x);
  double mx = -1e300, z = 0;
  for (int b = 0; b < 4; ++b) {
    double v = 0;
    for (int i = 0; i < F; ++i) v += m.W[b * m.F + i] * x[i];
    lp[b] = v; mx = std::max(mx, v);
  }
  for (int b = 0; b < 4; ++b) z += std::exp(lp[b] - mx);
  for (int b = 0; b < 4; ++b) lp[b] = lp[b] - mx - std::log(z);
}

// log-likelihood of all opponents' public bids under world `w` (alpha-tempered)
inline double bid_loglik(const Round& w, int me, const BidModel& m) {
  double ll = 0;
  int hand[MAXR];
  for (int q = 0; q < w.P; ++q) {
    if (q == me || w.bid[q] < 1) continue;
    for (int r = 0; r < w.R; ++r) hand[r] = w.hand[q][r];
    for (int c = 0; c < NC; ++c)
      for (int r = 0; r < w.R; ++r)
        if (w.board[c][r] == q) hand[r]++;
    int pos = (q - w.start + w.P) % w.P, sum = 0;
    for (int k = 0; k < pos; ++k) sum += w.bid[(w.start + k) % w.P];
    double lp[4];
    bid_logprobs(m, w.R, hand, pos, pos, sum, lp);
    ll += lp[w.bid[q] - 1];
  }
  return ll;
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
  const BidModel* bidmodel = nullptr;  // if set: bid-informed world resampling
  int cand = 16;                       // candidates per world
  double alpha = 1.0;                  // likelihood tempering
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
  bool use_bids = false;
  if (sp.bidmodel && sp.bidmodel->ok())
    for (int q = 0; q < view.P; ++q) use_bids |= (q != me && view.bid[q] >= 1);
  Round cands[64];
  double lw[64];
  for (int w = 0; w < sp.worlds; ++w) {
    if (use_bids) {
      int K = std::min(sp.cand, 64);
      double mx = -1e300;
      for (int k = 0; k < K; ++k) {
        sample_world(view, me, rng, cands[k]);
        lw[k] = sp.alpha * bid_loglik(cands[k], me, *sp.bidmodel);
        mx = std::max(mx, lw[k]);
      }
      double z = 0;
      for (int k = 0; k < K; ++k) { lw[k] = std::exp(lw[k] - mx); z += lw[k]; }
      double u = rng.uniform() * z; int pick = K - 1;
      for (int k = 0; k < K; ++k) { u -= lw[k]; if (u <= 0) { pick = k; break; } }
      world = cands[pick];
    } else {
      sample_world(view, me, rng, world);
    }
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
