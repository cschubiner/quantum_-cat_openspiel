// Learned PLAY-phase policy: per-(state, action) features phi(s,a), a linear
// or tiny-MLP scorer pi(a|s) = softmax(score(phi(s,a))), and PIMC whose
// rollouts use that policy for every seat.
//
// INFORMATION DISCIPLINE: play_context()/play_features() read only the acting
// player's (s.cur) own hand and discard plus public state (board, tokens, bids,
// tricks, hand sizes, current trick).  They never read another seat's hand or
// discard.  test_features.cpp checks this by re-sampling hidden hands.
#pragma once
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "qc_agents.h"

namespace qc {

constexpr int NFEAT = 64;

// Per-decision context, computed once and shared by all candidate actions.
struct PlayCtx {
  int p, b, t, rem, need, plays_left, pos, last, lead;
  int g_chase, g_onbid, g_free;  // mode gates (exactly one is 1)
  double urgency;                // need / rem when chasing
  int best_red, best_led;
  int w;              // current trick winner (-1 when leading)
  int w_status;       // -1 under bid, 0 on bid, 1 over/unreachable (meaningful if w>=0)
  int opp_after;      // opponents still to play this trick
  int opp_after_void_led;  // of those, how many have lost the led-color token
  int opp_after_nored;     // of those, how many have lost the red token
  int unseen[MAXR];   // copies of each rank not visible to me (opp hands + opp discards)
  int8_t comp[NC][MAXR];  // my board component id (-1 none)
  int comp_size[NC * MAXR];
  int largest;
  int nlegal;
  int opp_tok_missing[NC];  // # opponents lacking token of color c
};

inline void play_context(const Round& s, PlayCtx& x, int nlegal) {
  const int p = s.cur;
  x.p = p; x.b = s.bid[p]; x.t = s.tricks[p];
  x.rem = s.T - s.trick_no;
  x.need = x.b - x.t;
  x.plays_left = x.rem - 1;
  x.pos = s.nplayed; x.last = s.nplayed == s.P - 1; x.lead = s.nplayed == 0;
  x.g_chase = x.need > 0 && x.need <= x.rem;
  x.g_onbid = x.need == 0;
  x.g_free = !x.g_chase && !x.g_onbid;
  x.urgency = x.g_chase ? (double)x.need / x.rem : 0.0;
  x.best_red = -1; x.best_led = -1;
  for (int q = 0; q < s.P; ++q) {
    if (s.tcolor[q] == RED) x.best_red = std::max<int>(x.best_red, s.trank[q]);
    if (s.led >= 0 && s.tcolor[q] == s.led) x.best_led = std::max<int>(x.best_led, s.trank[q]);
  }
  x.w = s.nplayed > 0 ? s.current_winner() : -1;
  x.w_status = 0;
  if (x.w >= 0) {
    int nd = s.bid[x.w] - s.tricks[x.w];
    int wr = s.T - s.trick_no;
    x.w_status = nd > 0 && nd <= wr ? -1 : (nd == 0 ? 0 : 1);
  }
  x.opp_after = s.P - 1 - s.nplayed;
  x.opp_after_void_led = 0; x.opp_after_nored = 0;
  for (int k = 1; k <= x.opp_after; ++k) {
    int q = (p + k) % s.P;
    if (s.led >= 0 && !(s.tokens[q] >> s.led & 1)) ++x.opp_after_void_led;
    if (!(s.tokens[q] & 1)) ++x.opp_after_nored;
  }
  for (int c = 0; c < NC; ++c) {
    x.opp_tok_missing[c] = 0;
    for (int q = 0; q < s.P; ++q)
      if (q != p && !(s.tokens[q] >> c & 1)) ++x.opp_tok_missing[c];
  }
  for (int r = 0; r < s.R; ++r) x.unseen[r] = 5 - s.hand[p][r];
  if (s.discard[p] >= 0) x.unseen[s.discard[p]]--;
  for (int c = 0; c < NC; ++c)
    for (int r = 0; r < s.R; ++r)
      if (s.board[c][r] >= 0) x.unseen[r]--;
  // my board components
  std::memset(x.comp, -1, sizeof(x.comp));
  int ncomp = 0; x.largest = 0;
  int stack[NC * MAXR];
  for (int c = 0; c < NC; ++c)
    for (int r = 0; r < s.R; ++r) {
      if (s.board[c][r] != p || x.comp[c][r] >= 0) continue;
      int sp = 0, size = 0, id = ncomp++;
      stack[sp++] = c * MAXR + r; x.comp[c][r] = (int8_t)id;
      while (sp) {
        int cid = stack[--sp]; ++size;
        int cc = cid / MAXR, rr = cid % MAXR;
        const int dc[4] = {1, -1, 0, 0}, dr[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
          int c2 = cc + dc[k], r2 = rr + dr[k];
          if (c2 < 0 || c2 >= NC || r2 < 0 || r2 >= s.R) continue;
          if (s.board[c2][r2] == p && x.comp[c2][r2] < 0) { x.comp[c2][r2] = (int8_t)id; stack[sp++] = c2 * MAXR + r2; }
        }
      }
      x.comp_size[id] = size;
      x.largest = std::max(x.largest, size);
    }
  x.nlegal = nlegal;
}

// Fills f[0..NFEAT).  Values are roughly in [-1, 1] (hscore excepted).
inline void play_features(const Round& s, const PlayCtx& x, int a, float* f) {
  std::memset(f, 0, sizeof(float) * NFEAT);
  const int p = x.p;
  const int c = a / s.R, r = a % s.R;
  const double rel = (r + 1.0) / s.R;
  const bool wn = wins_now(s, c, r);
  const bool token_loss = s.led >= 0 && c != s.led;
  // threats: free cells that could beat this card later in the trick, holdable by unseen cards
  int thr_same = 0, thr_red = 0;
  if (wn && !x.last) {
    for (int r2 = r + 1; r2 < s.R; ++r2) thr_same += s.board[c][r2] == -1 && x.unseen[r2] > 0;
    if (c != RED)
      for (int r2 = 0; r2 < s.R; ++r2) thr_red += s.board[RED][r2] == -1 && x.unseen[r2] > 0;
  }
  const double after_frac = (double)x.opp_after / (s.P - 1);
  // crude win probability (same as heuristic) and a refined variant
  double pwin = 0, pwin2 = 0;
  if (wn) {
    if (x.last) { pwin = 1; pwin2 = 1; }
    else {
      pwin = (c == RED) ? 0.35 + 0.65 * rel * rel : 0.6 * rel * rel * rel;
      int higher_free = 0;
      for (int r2 = r + 1; r2 < s.R; ++r2) higher_free += s.board[c][r2] == -1;
      pwin *= 1.0 / (1.0 + 0.5 * higher_free * (s.P - 1 - s.nplayed) / (double)(s.P - 1));
      // refined: following players each threaten with same-color higher cells; non-red
      // also threatened by trumping (more likely from players void in led/this color).
      double lam = thr_same * 0.12 * x.opp_after;
      if (c != RED) {
        int voidc = 0;  // opps after me who can't follow color c (if I'm leading c) / led
        if (x.lead) {
          for (int k = 1; k <= x.opp_after; ++k) voidc += !(s.tokens[(p + k) % s.P] >> c & 1);
        } else voidc = x.opp_after_void_led;
        lam += thr_red * (0.02 * x.opp_after + 0.10 * voidc) * (x.opp_after - x.opp_after_nored > 0);
      }
      pwin2 = std::exp(-lam);
    }
  }
  // liquidity after the move
  uint8_t tok = s.tokens[p];
  if (token_loss) tok &= (uint8_t)~(1u << s.led);
  int liq = 0, liq_cards = 0;
  for (int r2 = 0; r2 < s.R; ++r2) {
    int h = s.hand[p][r2] - (r2 == r);
    if (h <= 0) continue;
    int cells = 0;
    for (int c2 = 0; c2 < NC; ++c2)
      if ((tok >> c2 & 1) && s.board[c2][r2] == -1 && !(c2 == c && r2 == r)) ++cells;
    liq += cells;
    liq_cards += std::min(h, cells);  // cards that still have somewhere to go
  }
  const int pl = x.plays_left;
  const bool dead = pl > 0 && liq == 0;
  // stranded copies of this rank after the move
  int free_r = 0;
  for (int c2 = 0; c2 < NC; ++c2)
    if ((tok >> c2 & 1) && s.board[c2][r] == -1 && c2 != c) ++free_r;
  const int copies_left = s.hand[p][r] - 1;
  const int stranded = std::max(0, copies_left - free_r);
  // adjacency / cluster growth
  int adj = 0, adj_opp = 0, gain_sz = 1;
  int seen_ids[4], ns = 0;
  const int dc[4] = {1, -1, 0, 0}, dr[4] = {0, 0, 1, -1};
  for (int k = 0; k < 4; ++k) {
    int c2 = c + dc[k], r2 = r + dr[k];
    if (c2 < 0 || c2 >= NC || r2 < 0 || r2 >= s.R) continue;
    int o = s.board[c2][r2];
    if (o == p) {
      ++adj;
      int id = x.comp[c2][r2];
      bool dup = false;
      for (int j = 0; j < ns; ++j) dup |= seen_ids[j] == id;
      if (!dup) { seen_ids[ns++] = id; gain_sz += x.comp_size[id]; }
    } else if (o >= 0) ++adj_opp;
  }
  const int cgain = std::max(0, gain_sz - x.largest);
  const bool lastwin = wn && x.last;
  // heuristic_move_score(s, a, HeurParams()) recomputed from the shared pieces above
  // (same expressions / order => bit-identical; checked under QC_CHECK).
  double hs;
  {
    // Pinned to the parameters the learned weights were trained with
    // (rollout defaults at training time: w_token = 1.2).
    HeurParams hp;
    hp.w_token = 1.2;
    const bool want = x.t != x.b;
    hs = want ? hp.w_win * pwin : -hp.w_win * pwin;
    if (token_loss) hs -= hp.w_token;
    if (pl > 0) {
      if (liq == 0) hs -= hp.w_dead;
      else hs += hp.w_liq * std::min(liq, 8);
    }
    if (!want || x.t < x.b) hs += hp.w_adj * adj;
    const double rel0 = (double)r / s.R;
    if (pwin == 0) hs += want ? -0.15 * rel0 : 0.25 * rel0;
    else hs += want ? -0.1 * rel0 : 0;
  }
#ifdef QC_CHECK
  {
    HeurParams chk; chk.w_token = 1.2; HeurCtx cx; heur_ctx(s, cx);
    if (hs != heuristic_move_score(s, cx, a, chk)) { std::fprintf(stderr, "inline hs mismatch\n"); std::abort(); }
  }
#endif

  int i = 0;
  // ---- base action features
  f[i++] = (float)hs;                                   // 0 heuristic score
  f[i++] = wn;                                          // 1
  f[i++] = (float)pwin;                                 // 2
  f[i++] = (float)pwin2;                                // 3
  f[i++] = lastwin;                                     // 4
  f[i++] = (float)rel;                                  // 5
  f[i++] = (float)(rel * rel);                          // 6
  f[i++] = c == 0; f[i++] = c == 1; f[i++] = c == 2; f[i++] = c == 3;  // 7..10 color one-hot (0 = red)
  f[i++] = token_loss;                                  // 11
  f[i++] = token_loss && s.led == RED;                  // 12 lose the trump token
  f[i++] = dead;                                        // 13
  f[i++] = (float)(std::min(liq, 8) / 8.0);             // 14
  f[i++] = pl > 0 && liq_cards < pl;                    // 15 fewer playable cards than plays left
  f[i++] = (float)(pl > 0 ? std::min(1.0, (double)liq_cards / pl) : 1.0);  // 16
  f[i++] = (float)(stranded / 4.0);                     // 17
  f[i++] = (float)(s.hand[p][r] / 5.0);                 // 18 copies held of this rank
  f[i++] = (float)(adj / 4.0);                          // 19
  f[i++] = (float)(cgain / 4.0);                        // 20
  f[i++] = (float)(adj_opp / 4.0);                      // 21
  f[i++] = (float)(thr_same / 8.0);                     // 22
  f[i++] = (float)(thr_red / 9.0 * after_frac);         // 23
  f[i++] = wn && x.w >= 0 && x.w_status == 0;           // 24 steal trick from opp sitting on bid
  f[i++] = wn && x.w >= 0 && x.w_status < 0;            // 25 steal trick from opp who needs it
  f[i++] = (float)(wn * x.opp_after_void_led / 4.0);    // 26
  f[i++] = !wn && token_loss && c != RED;               // 27 off-color dump
  f[i++] = (float)(x.lead ? x.opp_tok_missing[c] / 4.0 : 0.0);  // 28 lead into opps void in c
  f[i++] = x.lead && c == RED;                          // 29
  f[i++] = (float)(x.lead * rel);                       // 30
  f[i++] = (float)(x.last * rel);                       // 31
  f[i++] = (float)(x.unseen[r] / 5.0);                  // 32
  f[i++] = (float)(wn && pl > 0 ? std::min(liq, 8) / 8.0 : 0.0);  // 33 winning => I lead next
  // ---- mode-gated interactions (chase / on-bid / free)
  const int gates[3] = {x.g_chase, x.g_onbid, x.g_free};
  for (int g = 0; g < 3; ++g) {
    const float G = (float)gates[g];
    f[i++] = G * wn;                                    // 34,42,50
    f[i++] = G * (float)pwin2;
    f[i++] = G * lastwin;
    f[i++] = G * (float)rel;
    f[i++] = G * (c == RED);
    f[i++] = G * (float)(cgain / 4.0);
    f[i++] = G * (float)(adj / 4.0);
    f[i++] = G * token_loss;
  }
  // 58..63 urgency / endgame interactions
  f[i++] = (float)(x.urgency * pwin2);
  f[i++] = (float)(x.urgency * wn);
  f[i++] = (float)((x.need == 1) * pwin2);              // last trick needed
  f[i++] = (float)(x.g_onbid * (x.rem <= 2) * pwin2);  // late ducking
  f[i++] = (float)(x.g_chase * (x.need == x.rem) * wn); // must win all remaining
  f[i++] = (float)(dead * (x.t > 0 ? 1.0 : 0.5));       // paradox severity scales with tricks
#ifdef QC_CHECK
  if (i != NFEAT) { std::fprintf(stderr, "feature count %d != %d\n", i, NFEAT); std::abort(); }
#endif
}

// ---------------------------------------------------------------- policy
struct LearnedPolicy {
  int kind = 0;  // 0 linear, 1 mlp
  int F = NFEAT, H = 0;
  std::vector<float> w;          // linear: F
  std::vector<float> W1, b1, w2; // mlp: H*F (row h = hidden unit), H, H
  std::vector<float> W1T;        // F*H transpose (feature-major) for sparse accumulation
  void finalize() {  // build the transposed MLP matrix (call after filling W1 by hand)
    W1T.resize((size_t)NFEAT * H);
    for (int h = 0; h < H; ++h)
      for (int j = 0; j < NFEAT; ++j) W1T[(size_t)j * H + h] = W1[(size_t)h * NFEAT + j];
  }
  bool load(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) return false;
    char kindbuf[16];
    if (std::fscanf(fp, "%15s %d", kindbuf, &F) != 2) { std::fclose(fp); return false; }
    if (F != NFEAT) { std::fprintf(stderr, "weights F=%d but NFEAT=%d\n", F, NFEAT); std::fclose(fp); return false; }
    std::string k(kindbuf);
    auto rd = [&](std::vector<float>& v, int n) {
      v.resize(n);
      for (int j = 0; j < n; ++j) if (std::fscanf(fp, "%f", &v[j]) != 1) return false;
      return true;
    };
    bool ok;
    if (k == "linear") { kind = 0; ok = rd(w, F); }
    else if (k == "mlp") {
      kind = 1;
      ok = std::fscanf(fp, "%d", &H) == 1 && H > 0 && H <= 256 && rd(W1, H * F) && rd(b1, H) && rd(w2, H);
      if (ok) finalize();
    } else ok = false;
    std::fclose(fp);
    return ok;
  }
  inline float score(const float* f) const {
    if (kind == 0) {
      float z = 0;
      for (int j = 0; j < NFEAT; ++j) z += w[j] * f[j];
      return z;
    }
    // hidden = b1 + sum_j f_j * W1[:, j]; many features are exactly 0 (gates), skip them.
    float a[256];
    const int Hn = H;
    for (int h = 0; h < Hn; ++h) a[h] = b1[h];
    for (int j = 0; j < NFEAT; ++j) {
      const float fj = f[j];
      if (fj == 0.f) continue;
      const float* col = &W1T[(size_t)j * Hn];
      for (int h = 0; h < Hn; ++h) a[h] += col[h] * fj;  // independent lanes -> vectorizes
    }
    float z = 0;
    for (int h = 0; h < Hn; ++h) z += w2[h] * (a[h] > 0 ? a[h] : 0);
    return z;
  }
};

// temp <= 0 : greedy; else sample from softmax(score / temp)
inline int learned_play(const Round& s, Rng& rng, const LearnedPolicy& pol, double temp) {
  int acts[NC * MAXR + 1];
  int n = s.legal(acts);
  if (n == 1) return acts[0];
  PlayCtx x;
  play_context(s, x, n);
  float f[NFEAT];
  double sc[NC * MAXR];
  double best = -1e18; int bi = 0;
  for (int i = 0; i < n; ++i) {
    play_features(s, x, acts[i], f);
    sc[i] = pol.score(f) + rng.uniform() * 1e-6;
    if (sc[i] > best) { best = sc[i]; bi = i; }
  }
  if (temp <= 0) return acts[bi];
  double z = 0;
  for (int i = 0; i < n; ++i) { sc[i] = std::exp((sc[i] - best) / temp); z += sc[i]; }
  double u = rng.uniform() * z;
  for (int i = 0; i < n; ++i) { u -= sc[i]; if (u <= 0) return acts[i]; }
  return acts[n - 1];
}

// Rollout policy: heuristic discard/bid; learned play (or heuristic play if pol==nullptr).
struct RolloutPolicy {
  const LearnedPolicy* pol = nullptr;
  double temp = 0.0;
  HeurParams hp;
  inline int act(const Round& s, Rng& rng) const {
    if (s.phase == PLAY && pol) return learned_play(s, rng, *pol, temp);
    return heuristic_action(s, rng, hp);
  }
};

// PIMC identical to pimc_decide but with a pluggable rollout policy.
inline int pimc_decide_rp(const Round& view, int me, int worlds, int util, const RolloutPolicy& rp,
                          Rng& rng, double* out_values = nullptr) {
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
