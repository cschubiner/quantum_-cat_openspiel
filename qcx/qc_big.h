// "Big" learned PLAY policy: a seat-relative raw state encoding of the actor's
// information (shared by all candidate actions, embedded once per decision)
// combined with the 64 hand-crafted per-action features of qc_learn.h.
//
// INFORMATION DISCIPLINE: state_features() reads only the acting player's
// (s.cur) own hand and discard plus public state (board, tokens, bids, tricks,
// hand sizes, current trick, trick number).  test_state.cpp checks that the
// encoding is identical on the true state, the masked view and re-sampled
// worlds.
//
// Network (weights file "big", standardisation folded into the first layers):
//   h  = relu(Ws1 s + bs1)                       state, sparse input   (H1)
//   h  = relu(Ws2 h + bs2)        [if H1b > 0]   state, dense          (H1b)
//   u  = Wp h + bp                               state -> action space (H2)
//   per legal action a, x = [phi(s,a) (64), color one-hot (4), rank one-hot (9)]:
//   z  = relu(u + Wa x)                          sparse input          (H2)
//   y  = relu(W3 z + b3)          [if H3 > 0]                          (H3)
//   score = w4 . (H3 ? y : z)
#pragma once
#include "qc_learn.h"

namespace qc {

constexpr int NSEATF = 33;
constexpr int NSTATE = 3 + 9 + 9 + 9 + NC * MAXR * 6 + MAXP * NSEATF + 5 + 8 + 1 + 5 + 1;  // 431
constexpr int NAEXTRA = NC + MAXR;
constexpr int NACT = NFEAT + NAEXTRA;  // 77

inline int largest_cluster_of(const Round& s, int p) { return s.largest_cluster(p); }

// Fills f[0..NSTATE).  Seat k is relative to the actor (k = 0 is the actor,
// k = 1 the next player in turn order, ...).
inline void state_features(const Round& s, float* f) {
  std::memset(f, 0, sizeof(float) * NSTATE);
  const int p = s.cur, P = s.P, R = s.R;
  int i = 0;
  f[i + P - 3] = 1; i += 3;                                     // player count one-hot
  for (int r = 0; r < R; ++r) f[i + r] = s.hand[p][r] / 5.f;   // own hand counts
  i += MAXR;
  if (s.discard[p] >= 0) f[i + s.discard[p]] = 1;              // own discard
  i += MAXR;
  {
    int unseen[MAXR];
    for (int r = 0; r < R; ++r) unseen[r] = 5 - s.hand[p][r];
    if (s.discard[p] >= 0) unseen[s.discard[p]]--;
    for (int c = 0; c < NC; ++c)
      for (int r = 0; r < R; ++r)
        if (s.board[c][r] >= 0) unseen[r]--;
    for (int r = 0; r < R; ++r) f[i + r] = unseen[r] / 5.f;    // copies not visible to me
    i += MAXR;
  }
  for (int c = 0; c < NC; ++c)                                  // board, relative owner / empty
    for (int r = 0; r < R; ++r) {
      const int o = s.board[c][r];
      const int k = o < 0 ? 5 : (o - p + P) % P;
      f[i + (c * MAXR + r) * 6 + k] = 1;
    }
  i += NC * MAXR * 6;
  const int w = s.nplayed > 0 ? s.current_winner() : -1;
  const int rem = s.T - s.trick_no;
  for (int k = 0; k < MAXP; ++k) {
    float* g = f + i + k * NSEATF;
    if (k >= P) continue;
    const int q = (p + k) % P;
    int j = 0;
    g[j++] = 1;                                                 // present
    for (int c = 0; c < NC; ++c) g[j++] = (s.tokens[q] >> c) & 1;
    if (s.bid[q] >= 1) g[j + s.bid[q] - 1] = 1;                 // bid one-hot
    j += 4;
    g[j++] = s.tricks[q] / 4.f;
    const int need = s.bid[q] >= 1 ? s.bid[q] - s.tricks[q] : 0;
    g[j++] = need / 4.f;
    if (s.bid[q] >= 1) {
      const int st = need > 0 && need <= rem ? 0 : (need == 0 ? 1 : 2);  // chase / on bid / free
      g[j + st] = 1;
    }
    j += 3;
    g[j++] = s.hsize[q] / 10.f;
    const bool played = s.tcolor[q] >= 0;
    g[j++] = played;
    if (played) {
      g[j + s.tcolor[q]] = 1;
      g[j + NC + s.trank[q]] = 1;
    }
    j += NC + MAXR;
    g[j++] = q == w;                                            // current trick winner
    g[j++] = s.largest_cluster(q) / 8.f;
    g[j++] = q == s.leader && s.nplayed > 0;                    // led this trick
    g[j++] = s.cells_owned(q) / 12.f;
  }
  i += MAXP * NSEATF;
  f[i + (s.led >= 0 ? s.led : 4)] = 1; i += 5;                  // led color / none
  if (s.trick_no < 8) f[i + s.trick_no] = 1;
  i += 8;
  f[i++] = (float)rem / s.T;
  f[i + std::min(s.nplayed, (int8_t)4)] = 1; i += 5;            // position in trick
  f[i++] = s.red_open();
#ifdef QC_CHECK
  if (i != NSTATE) { std::fprintf(stderr, "state size %d != %d\n", i, NSTATE); std::abort(); }
#endif
}

inline void action_extra(const Round& s, int a, float* f) {
  std::memset(f, 0, sizeof(float) * NAEXTRA);
  if (a == ACT_PARADOX) return;
  f[a / s.R] = 1;
  f[NC + a % s.R] = 1;
}

struct BigPolicy {
  int S = NSTATE, A = NACT, H1 = 0, H1b = 0, H2 = 0, H3 = 0;
  std::vector<float> Ws1T, bs1;   // S x H1 (input-major), H1
  std::vector<float> Ws2, bs2;    // H1b x H1, H1b
  std::vector<float> Wp, bp;      // H2 x Hs, H2  (bp includes the action bias)
  std::vector<float> WaT;         // A x H2 (input-major)
  std::vector<float> W3, b3;      // H3 x H2, H3
  std::vector<float> w4;          // H3 or H2

  bool load(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) return false;
    char kind[16];
    bool ok = std::fscanf(fp, "%15s %d %d %d %d %d %d", kind, &S, &A, &H1, &H1b, &H2, &H3) == 7 &&
              std::string(kind) == "big" && S == NSTATE && A == NACT && H1 > 0 && H1 <= 512 &&
              H1b >= 0 && H1b <= 512 && H2 > 0 && H2 <= 512 && H3 >= 0 && H3 <= 512;
    auto rd = [&](std::vector<float>& v, size_t n) {
      v.resize(n);
      for (size_t j = 0; j < n; ++j) if (std::fscanf(fp, "%f", &v[j]) != 1) return false;
      return true;
    };
    const int Hs = H1b > 0 ? H1b : H1;
    ok = ok && rd(Ws1T, (size_t)S * H1) && rd(bs1, H1);
    if (ok && H1b > 0) ok = rd(Ws2, (size_t)H1b * H1) && rd(bs2, H1b);
    ok = ok && rd(Wp, (size_t)H2 * Hs) && rd(bp, H2) && rd(WaT, (size_t)A * H2);
    if (ok && H3 > 0) ok = rd(W3, (size_t)H3 * H2) && rd(b3, H3);
    ok = ok && rd(w4, H3 > 0 ? H3 : H2);
    std::fclose(fp);
    if (!ok) std::fprintf(stderr, "bad big weights %s\n", path.c_str());
    else finalize();
    return ok;
  }

  // Transposed (input-major) copies of the dense matrices so every layer is a
  // sequence of independent-lane axpys (vectorizes without -ffast-math and
  // skips zero inputs, e.g. inactive ReLUs).
  std::vector<float> Ws2T, WpT, W3T;
  void finalize() {
    const int Hs = H1b > 0 ? H1b : H1;
    auto tr = [](const std::vector<float>& W, int rows, int cols) {
      std::vector<float> T((size_t)rows * cols);
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) T[(size_t)c * rows + r] = W[(size_t)r * cols + c];
      return T;
    };
    if (H1b > 0) Ws2T = tr(Ws2, H1b, H1);
    WpT = tr(Wp, H2, Hs);
    if (H3 > 0) W3T = tr(W3, H3, H2);
  }

  // State part: u[H2] from the raw state encoding.
  inline void embed(const float* st, float* u) const {
    alignas(64) float h[512], h2[512];
    for (int k = 0; k < H1; ++k) h[k] = bs1[k];
    for (int j = 0; j < S; ++j) {
      const float x = st[j];
      if (x == 0.f) continue;
      const float* col = &Ws1T[(size_t)j * H1];
      for (int k = 0; k < H1; ++k) h[k] += col[k] * x;
    }
    for (int k = 0; k < H1; ++k) h[k] = h[k] > 0 ? h[k] : 0;
    const float* hs = h;
    int Hs = H1;
    if (H1b > 0) {
      for (int m = 0; m < H1b; ++m) h2[m] = bs2[m];
      for (int k = 0; k < H1; ++k) {
        const float x = h[k];
        if (x == 0.f) continue;
        const float* col = &Ws2T[(size_t)k * H1b];
        for (int m = 0; m < H1b; ++m) h2[m] += col[m] * x;
      }
      for (int m = 0; m < H1b; ++m) h2[m] = h2[m] > 0 ? h2[m] : 0;
      hs = h2; Hs = H1b;
    }
    for (int m = 0; m < H2; ++m) u[m] = bp[m];
    for (int k = 0; k < Hs; ++k) {
      const float x = hs[k];
      if (x == 0.f) continue;
      const float* col = &WpT[(size_t)k * H2];
      for (int m = 0; m < H2; ++m) u[m] += col[m] * x;
    }
  }

  inline float score_action(const float* u, const float* x) const {
    alignas(64) float z[512], y[512];
    for (int m = 0; m < H2; ++m) z[m] = u[m];
    for (int j = 0; j < A; ++j) {
      const float xj = x[j];
      if (xj == 0.f) continue;
      const float* col = &WaT[(size_t)j * H2];
      for (int m = 0; m < H2; ++m) z[m] += col[m] * xj;
    }
    for (int m = 0; m < H2; ++m) z[m] = z[m] > 0 ? z[m] : 0;
    if (H3 == 0) {
      float o = 0;
      for (int m = 0; m < H2; ++m) o += w4[m] * z[m];
      return o;
    }
    for (int n = 0; n < H3; ++n) y[n] = b3[n];
    for (int m = 0; m < H2; ++m) {
      const float zm = z[m];
      if (zm == 0.f) continue;
      const float* col = &W3T[(size_t)m * H3];
      for (int n = 0; n < H3; ++n) y[n] += col[n] * zm;
    }
    float o = 0;
    for (int n = 0; n < H3; ++n) o += w4[n] * (y[n] > 0 ? y[n] : 0);
    return o;
  }
};

// Scores for all legal actions acts[0..n) of s (n >= 2).
inline void big_scores(const Round& s, const BigPolicy& pol, const int* acts, int n, float* out) {
  float st[NSTATE], u[512], x[NACT];
  state_features(s, st);
  pol.embed(st, u);
  PlayCtx cx;
  play_context(s, cx, n);
  for (int i = 0; i < n; ++i) {
    play_features(s, cx, acts[i], x);
    action_extra(s, acts[i], x + NFEAT);
    out[i] = pol.score_action(u, x);
  }
}

inline int big_play(const Round& s, Rng& rng, const BigPolicy& pol) {
  int acts[NC * MAXR + 1];
  int n = s.legal(acts);
  if (n == 1) return acts[0];
  float sc[NC * MAXR + 1];
  big_scores(s, pol, acts, n, sc);
  double best = -1e18; int bi = 0;
  for (int i = 0; i < n; ++i) {
    const double v = sc[i] + rng.uniform() * 1e-6;
    if (v > best) { best = v; bi = i; }
  }
  return acts[bi];
}

// Rollout policy: heuristic discard/bid; PLAY by the big model for every seat
// (big_seat < 0) or only for big_seat, the small learned policy otherwise.
struct BigRolloutPolicy {
  const BigPolicy* big = nullptr;
  const LearnedPolicy* small = nullptr;
  int big_seat = -1;
  HeurParams hp;
  inline int act(const Round& s, Rng& rng) const {
    if (s.phase == PLAY) {
      if (big && (big_seat < 0 || s.cur == big_seat)) return big_play(s, rng, *big);
      if (small) return learned_play(s, rng, *small, 0.0);
    }
    return heuristic_action(s, rng, hp);
  }
};

// pimc_decide_rp with a generic rollout policy type.
template <class RP>
inline int pimc_decide_t(const Round& view, int me, int worlds, int util, const RP& rp, Rng& rng,
                         double* out_values = nullptr) {
  int acts[NC * MAXR + 1];
  int n = view.legal(acts);
  if (n == 1) return acts[0];
  double sum[NC * MAXR + 1] = {0};
  Round world, s;
  for (int w = 0; w < worlds; ++w) {
    sample_world(view, me, rng, world);
    uint64_t seed = rng.next();
    for (int i = 0; i < n; ++i) {
      s = world;
      s.apply(acts[i]);
      Rng rr(seed);
      while (!s.done()) s.apply(rp.act(s, rr));
      sum[i] += utility(s, me, util);
    }
  }
  int bi = 0;
  for (int i = 1; i < n; ++i) if (sum[i] > sum[bi]) bi = i;
  if (out_values) for (int i = 0; i < n; ++i) out_values[i] = sum[i] / worlds;
  return acts[bi];
}

}  // namespace qc
