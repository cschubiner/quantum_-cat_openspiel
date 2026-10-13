// Deeper information-set search on top of the learned rollout policy (exp-deepsearch).
//
//  A. ismcts_decide : single-observer information-set MCTS (Cowling, Powley &
//     Whitehouse 2012).  One tree over the actor's (= root player's)
//     information: every iteration samples a determinization with sample_world,
//     descends with PUCT (learned-policy softmax prior of whoever is to move,
//     computed from that seat's own information in the determinization),
//     expands one node and finishes the round with a learned-policy rollout.
//     Subset-armed bandit: only children legal in the current determinization
//     are eligible, and the PUCT exploration term uses per-child availability
//     counts instead of the parent visit count.  Values are per-player vectors
//     (own score - mean of others for every seat); each node maximizes the
//     value of the seat to move (max-n).  Opponents' hidden discards collapse to
//     one edge (the actor can't observe them); opponents' discards/bids follow
//     the rollout policy (chance-like nodes keyed by the observed action).
//     opp_mode: 0 = opponents' PLAY nodes are PUCT/max-n too, 1 = opponents play
//     the greedy rollout policy (only the actor's own decisions are searched).
//
//  B. pimc2_decide : PIMC at the root; inside each rollout the root player's
//     NEXT own decision is chosen by an inner PIMC (inner worlds sampled from
//     the root player's information in the rollout state), opponents and later
//     own decisions use the rollout policy.
//
//  C. pimce_decide : PIMC at the root; rollout policy until `k` tricks remain,
//     then an exact max-n search of the remaining tricks in the determinization
//     (every seat maximizes its own score - mean of others).
//
// INFORMATION DISCIPLINE: like pimc_decide_rp these functions read only what
// sample_world reads from `view` (the actor's hand/discard and public state);
// opponents' hands/discards in `view` are overwritten in every determinization.
// test_deepsearch.cpp checks this by changing the true hidden hands.
#pragma once
#include <cmath>
#include <vector>

#include "qc_learn.h"

namespace qc {

// ------------------------------------------------------------ helpers
inline void utilities_all(const Round& s, double* u) {
  double sc[MAXP], tot = 0;
  s.scores(sc);
  for (int q = 0; q < s.P; ++q) tot += sc[q];
  for (int q = 0; q < s.P; ++q) u[q] = sc[q] - (tot - sc[q]) / (s.P - 1);
}

// Softmax of the learned PLAY scores / temp over acts[0..n).
inline void policy_prior(const Round& s, const LearnedPolicy& pol, const int* acts, int n, double temp,
                         double* pr) {
  if (n == 1) { pr[0] = 1; return; }
  PlayCtx x;
  play_context(s, x, n);
  float f[NFEAT];
  double best = -1e18;
  for (int i = 0; i < n; ++i) {
    if (acts[i] == ACT_PARADOX) { pr[i] = -1e9; continue; }
    play_features(s, x, acts[i], f);
    pr[i] = pol.score(f) / temp;
    best = std::max(best, pr[i]);
  }
  double z = 0;
  for (int i = 0; i < n; ++i) { pr[i] = std::exp(pr[i] - best); z += pr[i]; }
  for (int i = 0; i < n; ++i) pr[i] /= z;
}

// Edge key: 0..35 play / discard rank, 40..43 bids, 62 hidden discard, 63 paradox.
inline int edge_key(int a) {
  if (a == ACT_PARADOX) return 63;
  if (a >= PRED_OFF) return 40 + (a - PRED_OFF);
  return a;
}
constexpr int KEY_HIDDEN = 62, NKEY = 64;

// ------------------------------------------------------------ A. SO-ISMCTS
struct IsmctsParams {
  int mult = 128;        // iterations = mult * #legal root actions (PIMC uses worlds * #legal rollouts)
  int iters = 0;         // if > 0, overrides mult
  double cpuct = 2.0;    // in utility units (points)
  double prior_temp = 1.0;
  int opp_mode = 0;      // 0 max-n PUCT for opponents' plays, 1 opponents follow the rollout policy
  double self_mix = 0.5; // prior for the actor's own discard/bid: mix * onehot(heuristic) + (1-mix) uniform
};

struct IsEdge {
  int child;
  int n, avail;
  float w[MAXP];
};
struct IsNode {
  int edge[NKEY];  // index into edge pool, -1 none
  int n;
  float w[MAXP];
};

struct IsTree {
  std::vector<IsNode> nodes;
  std::vector<IsEdge> edges;
  int new_node() {
    nodes.emplace_back();
    IsNode& nd = nodes.back();
    std::fill(nd.edge, nd.edge + NKEY, -1);
    nd.n = 0;
    std::fill(nd.w, nd.w + MAXP, 0.f);
    return (int)nodes.size() - 1;
  }
  int get_edge(int node, int key) {
    int e = nodes[node].edge[key];
    if (e >= 0) return e;
    edges.push_back(IsEdge{-1, 0, 0, {0, 0, 0, 0, 0}});
    e = (int)edges.size() - 1;
    nodes[node].edge[key] = e;
    return e;
  }
};

inline int ismcts_decide(const Round& view, int me, const IsmctsParams& ip, const RolloutPolicy& rp, Rng& rng,
                         double* out_visits = nullptr) {
  int racts[NC * MAXR + 1];
  const int rn = view.legal(racts);
  if (rn == 1) return racts[0];
  thread_local IsTree T;
  T.nodes.clear(); T.edges.clear();
  const int iters = ip.iters > 0 ? ip.iters : ip.mult * rn;
  T.nodes.reserve(iters + 2);
  T.edges.reserve((size_t)iters * 8);
  const int root = T.new_node();
  Round s;
  int path_e[64], path_n[64];
  int acts[NC * MAXR + 1], ekeys[NC * MAXR + 1];
  double pr[NC * MAXR + 1];
  for (int it = 0; it < iters; ++it) {
    sample_world(view, me, rng, s);
    Rng rr(rng.next());
    int node = root, depth = 0;
    while (!s.done()) {
      const int p = s.cur;
      int n = s.legal(acts);
      int a, e;
      const bool searched = n > 1 && (p == me || (s.phase == PLAY && ip.opp_mode == 0));
      if (!searched) {
        a = rp.act(s, rr);
        int key = (s.phase == DISCARD && p != me) ? KEY_HIDDEN : edge_key(a);
        e = T.get_edge(node, key);
        T.edges[e].avail++;
      } else {
        if (s.phase == PLAY) policy_prior(s, *rp.pol, acts, n, ip.prior_temp, pr);
        else {
          int h = rp.act(s, rr);
          for (int i = 0; i < n; ++i) pr[i] = (1 - ip.self_mix) / n + (acts[i] == h ? ip.self_mix : 0);
        }
        const IsNode& nd = T.nodes[node];
        const double fpu = nd.n > 0 ? nd.w[p] / nd.n : 0.0;
        double best = -1e18; int bi = 0;
        for (int i = 0; i < n; ++i) {
          ekeys[i] = T.get_edge(node, edge_key(acts[i]));
          IsEdge& ed = T.edges[ekeys[i]];
          ed.avail++;
          const double q = ed.n > 0 ? ed.w[p] / ed.n : fpu;
          const double u = q + ip.cpuct * pr[i] * std::sqrt((double)ed.avail) / (1.0 + ed.n);
          if (u > best) { best = u; bi = i; }
        }
        a = acts[bi]; e = ekeys[bi];
      }
      path_n[depth] = node; path_e[depth] = e; ++depth;
      s.apply(a);
      if (T.edges[e].child < 0) {  // expand one node, then roll out
        const int c = T.new_node();
        T.edges[e].child = c;
        break;
      }
      node = T.edges[e].child;
    }
    while (!s.done()) s.apply(rp.act(s, rr));
    double u[MAXP];
    utilities_all(s, u);
    T.nodes[root].n++;
    for (int q = 0; q < s.P; ++q) T.nodes[root].w[q] += (float)u[q];
    for (int d = 0; d < depth; ++d) {
      IsEdge& ed = T.edges[path_e[d]];
      ed.n++;
      for (int q = 0; q < s.P; ++q) ed.w[q] += (float)u[q];
      if (ed.child >= 0) {
        IsNode& cn = T.nodes[ed.child];
        cn.n++;
        for (int q = 0; q < s.P; ++q) cn.w[q] += (float)u[q];
      }
    }
  }
  int bi = 0; double bv = -1e18;
  for (int i = 0; i < rn; ++i) {
    const int e = T.nodes[root].edge[edge_key(racts[i])];
    const double nv = e >= 0 ? T.edges[e].n : 0;
    const double qv = (e >= 0 && T.edges[e].n > 0) ? T.edges[e].w[me] / T.edges[e].n : -1e9;
    const double v = nv + 1e-6 * qv;
    if (out_visits) out_visits[i] = nv;
    if (v > bv) { bv = v; bi = i; }
  }
  return racts[bi];
}

// ------------------------------------------------------------ B. PIMC-2
inline int pimc2_decide(const Round& view, int me, int worlds, int inner, const RolloutPolicy& rp, Rng& rng,
                        double* out_values = nullptr) {
  int acts[NC * MAXR + 1];
  const int n = view.legal(acts);
  if (n == 1) return acts[0];
  double sum[NC * MAXR + 1] = {0};
  Round world, s;
  for (int w = 0; w < worlds; ++w) {
    sample_world(view, me, rng, world);
    const uint64_t seed = rng.next();
    for (int i = 0; i < n; ++i) {
      s = world;
      s.apply(acts[i]);
      Rng rr(seed), ri(seed ^ 0x5bd1e995ULL);
      bool used = false;
      while (!s.done()) {
        int a;
        if (!used && s.cur == me) {
          used = true;
          // Inner search sees only `me`'s information in the rollout state
          // (sample_world re-draws the other hands).
          a = pimc_decide_rp(s, me, inner, 0, rp, ri);
        } else a = rp.act(s, rr);
        s.apply(a);
      }
      sum[i] += utility(s, me, 0);
    }
  }
  int bi = 0;
  for (int i = 1; i < n; ++i) if (sum[i] > sum[bi]) bi = i;
  if (out_values) for (int i = 0; i < n; ++i) out_values[i] = sum[i] / worlds;
  return acts[bi];
}

// ------------------------------------------------------------ C. PIMC + exact endgame
// Max-n over the rest of the round (perfect information within the determinization).
inline void maxn(const Round& s, double* u, long& nodes) {
  ++nodes;
  if (s.done()) { utilities_all(s, u); return; }
  int acts[NC * MAXR + 1];
  const int n = s.legal(acts);
  const int p = s.cur;
  double best[MAXP], tmp[MAXP];
  double bv = -1e18;
  Round c;
  for (int i = 0; i < n; ++i) {
    c = s;
    c.apply(acts[i]);
    maxn(c, tmp, nodes);
    if (tmp[p] > bv + 1e-12) { bv = tmp[p]; std::copy(tmp, tmp + MAXP, best); }
  }
  std::copy(best, best + MAXP, u);
}

inline int pimce_decide(const Round& view, int me, int worlds, int k, const RolloutPolicy& rp, Rng& rng,
                        double* out_values = nullptr) {
  int acts[NC * MAXR + 1];
  const int n = view.legal(acts);
  if (n == 1) return acts[0];
  double sum[NC * MAXR + 1] = {0};
  Round world, s;
  long nodes = 0;
  for (int w = 0; w < worlds; ++w) {
    sample_world(view, me, rng, world);
    const uint64_t seed = rng.next();
    for (int i = 0; i < n; ++i) {
      s = world;
      s.apply(acts[i]);
      Rng rr(seed);
      while (!s.done() && !(s.phase == PLAY && s.T - s.trick_no <= k)) s.apply(rp.act(s, rr));
      double u[MAXP];
      if (s.done()) utilities_all(s, u);
      else maxn(s, u, nodes);
      sum[i] += u[me];
    }
  }
  int bi = 0;
  for (int i = 1; i < n; ++i) if (sum[i] > sum[bi]) bi = i;
  if (out_values) for (int i = 0; i < n; ++i) out_values[i] = sum[i] / worlds;
  return acts[bi];
}

}  // namespace qc
