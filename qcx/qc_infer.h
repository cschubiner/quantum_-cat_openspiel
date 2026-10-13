// Policy-based inference for world sampling (cf. Solinas, Rebstock, Buro 2019,
// "Improving Search with Supervised Learning in Trick-Based Card Games").
//
// The deciding agent receives its masked view plus the PUBLIC action history
// of the current round (every bid and every card played, in order; discards
// are hidden and never recorded).  Candidate worlds drawn uniformly by
// sample_world() are weighted by how likely the opponents' actual past plays
// are under the learned play policy evaluated on that candidate world, and the
// N search worlds are resampled from the pool (systematic resampling).
//
// INFORMATION DISCIPLINE: everything here is computed from (view, me, hist,
// candidate world).  The view is mask_for(truth, me); the candidate world is
// produced by sample_world(view, ...).  Nothing reads the true hidden state.
// test_infer.cpp checks that decisions are identical for two different true
// deals that share the same masked view and public history.
#pragma once
#include <cmath>
#include <vector>

#include "qc_learn.h"

namespace qc {

// ---------------------------------------------------------------- history
struct PubHist {
  static constexpr int CAP = 64;  // <= 5 bids + 45 plays + paradox
  int n = 0;
  int8_t player[CAP];
  int16_t action[CAP];
  void clear() { n = 0; }
  void push(int p, int a) { player[n] = (int8_t)p; action[n] = (int16_t)a; ++n; }
};

// Record an action about to be applied to `s` iff it is public (bid / play).
inline void record_public(PubHist& h, const Round& s, int a) {
  if (s.phase == BID || s.phase == PLAY) h.push(s.cur, a);
}

// ---------------------------------------------------------------- evidence
// One past PLAY decision of an opponent: the public state at that moment (the
// actor's hand left empty) and, per rank, how many cards the actor played from
// that decision on (inclusive) -> actor's hand then = current hand + from.
struct OppDecision {
  Round snap;
  int q, a;
  int8_t from[MAXR];
};

struct InferParams {
  int pool = 512;      // M candidate worlds
  double alpha = 1.0;  // tempering of the log-likelihood
  double eps = 0.1;    // mix with uniform over legal actions
  double temp = 1.0;   // policy softmax temperature
};

// Rebuilds the opponents' past play decisions of this round from public
// information plus my own hand.  Returns false (and leaves `out` empty) when
// there is nothing to infer from (not in PLAY, or no opponent play yet).
inline bool build_evidence(const Round& view, int me, const PubHist& h, std::vector<OppDecision>& out) {
  out.clear();
  if (view.phase != PLAY) return false;
  const int P = view.P;
  int first = 0;
  while (first < h.n && h.action[first] >= PRED_OFF && h.action[first] != ACT_PARADOX) ++first;
  if (first != P) { std::fprintf(stderr, "build_evidence: expected %d bids, got %d\n", P, first); std::abort(); }
  // per-player plays, and my play-start hand
  int nplays[MAXP] = {0};
  Round s;
  s.init(P, view.start);
  s.phase = PLAY; s.cur = view.start; s.leader = view.start; s.led = -1; s.nplayed = 0;
  for (int q = 0; q < P; ++q) s.bid[q] = view.bid[q];
  s.discard[me] = view.discard[me];
  for (int r = 0; r < view.R; ++r) s.hand[me][r] = view.hand[me][r];
  for (int k = first; k < h.n; ++k) {
    const int p = h.player[k], a = h.action[k];
    ++nplays[p];
    if (p == me) s.hand[me][a % view.R]++;
  }
  for (int q = 0; q < P; ++q) s.hsize[q] = (int8_t)(view.hsize[q] + nplays[q]);
  for (int k = first; k < h.n; ++k) {
    const int p = h.player[k], a = h.action[k];
    if (s.cur != p || a == ACT_PARADOX) { std::fprintf(stderr, "build_evidence: inconsistent history\n"); std::abort(); }
    const int r = a % view.R;
    if (p != me) {
      OppDecision d;
      d.snap = s;  // opponents' hands are all zero here
      d.q = p; d.a = a;
      std::memset(d.from, 0, sizeof(d.from));
      for (int j = k; j < h.n; ++j)
        if (h.player[j] == p) d.from[h.action[j] % view.R]++;
      out.push_back(d);
      s.hand[p][r]++;  // placeholder so apply() decrements back to 0
    }
    s.apply(a);
  }
  // the replayed public state must coincide with the view's public state
  bool ok = s.cur == view.cur && s.trick_no == view.trick_no && s.nplayed == view.nplayed && s.led == view.led &&
            s.leader == view.leader && s.red_count == view.red_count && s.phase == view.phase &&
            std::memcmp(s.board, view.board, sizeof(s.board)) == 0 && std::memcmp(s.hand[me], view.hand[me], MAXR) == 0;
  for (int q = 0; q < P; ++q)
    ok = ok && s.tokens[q] == view.tokens[q] && s.tricks[q] == view.tricks[q] && s.hsize[q] == view.hsize[q] &&
         s.trank[q] == view.trank[q] && s.tcolor[q] == view.tcolor[q];
  if (!ok) { std::fprintf(stderr, "build_evidence: replay does not match the view\n"); std::abort(); }
  return !out.empty();
}

// Fill the actor's hand/discard of an evidence snapshot from a candidate world.
inline void fill_snapshot(const OppDecision& d, const Round& cand, int R, Round& s) {
  s = d.snap;
  for (int r = 0; r < R; ++r) s.hand[d.q][r] = (uint8_t)(cand.hand[d.q][r] + d.from[r]);
  s.discard[d.q] = cand.discard[d.q];
}

// log P(opponents' actual plays | candidate world) under the learned policy
// (softmax at ip.temp, mixed with eps-uniform).  -inf if a play would have been
// illegal in that world (e.g. a forced red lead while holding other options).
inline double evidence_loglik(const std::vector<OppDecision>& ev, const Round& cand, const LearnedPolicy& pol,
                              const InferParams& ip) {
  double ll = 0;
  Round s;
  int acts[NC * MAXR + 1];
  float f[NFEAT];
  double sc[NC * MAXR + 1];
  for (const OppDecision& d : ev) {
    fill_snapshot(d, cand, cand.R, s);
    const int n = s.legal(acts);
    int idx = -1;
    for (int i = 0; i < n; ++i) if (acts[i] == d.a) { idx = i; break; }
    if (idx < 0) return -INFINITY;
    if (n == 1) continue;
    PlayCtx x;
    play_context(s, x, n);
    double best = -1e300;
    for (int i = 0; i < n; ++i) {
      play_features(s, x, acts[i], f);
      sc[i] = pol.score(f) / ip.temp;
      if (sc[i] > best) best = sc[i];
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += std::exp(sc[i] - best);
    const double pa = std::exp(sc[idx] - best) / z;
    ll += std::log((1.0 - ip.eps) * pa + ip.eps / n);
  }
  return ll;
}

struct InferStats {
  long decisions = 0;
  double ess_sum = 0, uniq_sum = 0, rejected_sum = 0;
};

// Draw M candidates, weight by exp(alpha * loglik), systematically resample N.
inline void infer_worlds(const Round& view, int me, const std::vector<OppDecision>& ev, const LearnedPolicy& pol,
                         const InferParams& ip, int N, Rng& rng, std::vector<Round>& worlds,
                         InferStats* st = nullptr) {
  const int M = std::max(ip.pool, 1);
  std::vector<Round> pool(M);
  std::vector<double> lw(M);
  double mx = -INFINITY;
  int rej = 0;
  for (int i = 0; i < M; ++i) {
    sample_world(view, me, rng, pool[i]);
    const double ll = evidence_loglik(ev, pool[i], pol, ip);
    lw[i] = std::isinf(ll) ? -INFINITY : ip.alpha * ll;
    rej += std::isinf(ll);
    if (lw[i] > mx) mx = lw[i];
  }
  std::vector<double> cw(M);
  double tot = 0, tot2 = 0;
  for (int i = 0; i < M; ++i) {
    const double w = std::isinf(mx) ? 1.0 : (std::isinf(lw[i]) ? 0.0 : std::exp(lw[i] - mx));
    tot += w; tot2 += w * w;
    cw[i] = tot;
  }
  worlds.resize(N);
  const double step = tot / N;
  double u = rng.uniform() * step;
  int j = 0, uniq = 0, last = -1;
  for (int k = 0; k < N; ++k, u += step) {
    while (j < M - 1 && cw[j] <= u) ++j;
    worlds[k] = pool[j];
    if (j != last) { ++uniq; last = j; }
  }
  if (st) {
    st->decisions++;
    st->ess_sum += tot * tot / tot2;
    st->uniq_sum += uniq;
    st->rejected_sum += (double)rej / M;
  }
}

// PIMC over inferred worlds; common random numbers across actions as in
// pimc_decide_rp.  Falls back to pimc_decide_rp when there is no evidence.
inline int pimci_decide(const Round& view, int me, const PubHist& h, int N, int util, const RolloutPolicy& rp,
                        const InferParams& ip, Rng& rng, double* out_values = nullptr, InferStats* st = nullptr) {
  int acts[NC * MAXR + 1];
  const int n = view.legal(acts);
  if (n == 1) return acts[0];
  std::vector<OppDecision> ev;
  if (!rp.pol || !build_evidence(view, me, h, ev)) return pimc_decide_rp(view, me, N, util, rp, rng, out_values);
  std::vector<Round> worlds;
  infer_worlds(view, me, ev, *rp.pol, ip, N, rng, worlds, st);
  double sum[NC * MAXR + 1] = {0};
  Round s;
  for (int w = 0; w < N; ++w) {
    const uint64_t seed = rng.next();
    for (int i = 0; i < n; ++i) {
      s = worlds[w];
      s.apply(acts[i]);
      Rng rr(seed);
      while (!s.done()) s.apply(rp.act(s, rr));
      sum[i] += utility(s, me, util);
    }
  }
  int bi = 0;
  for (int i = 1; i < n; ++i) if (sum[i] > sum[bi]) bi = i;
  if (out_values) for (int i = 0; i < n; ++i) out_values[i] = sum[i] / N;
  return acts[bi];
}

}  // namespace qc
