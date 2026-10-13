// Tests + diagnostics for policy-based world inference (qc_infer.h).
//  1. reconstruction: with the TRUE world as candidate, every rebuilt opponent
//     decision state equals the recorded true state at that time (actor hand,
//     actor discard, public state), and the true world is never rejected.
//  2. information safety: two different true deals with the same masked view
//     and public history give bit-identical pimci decisions/values (fixed seed).
//  3. belief quality: squared error of the posterior-mean opponent hands vs the
//     truth (importance-weighted pool vs uniform sampler), ESS, rank of the truth.
//     QC_MODES selects opponent behaviour: 0 policy sampled at T=1 (exact model),
//     1 pimcl:16 play, 2 pimcl:32 for discard/bid/play.  QC_RANDOM_DISCARD=1 makes
//     discards uniform in modes 0/1 (exact prior).
//  4. CPU per PLAY decision: pimcl (N worlds) vs pimci (N worlds, pool M).
// build: clang++ -O3 -march=native -std=c++17 test_infer.cpp -o test_infer
// usage: test_infer weights [P=5] [games=60] [bench_states=120]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include "qc_infer.h"

using namespace qc;

static bool same_public(const Round& a, const Round& b) {
  bool ok = a.cur == b.cur && a.trick_no == b.trick_no && a.nplayed == b.nplayed && a.led == b.led &&
            a.leader == b.leader && a.red_count == b.red_count && a.phase == b.phase &&
            std::memcmp(a.board, b.board, sizeof(a.board)) == 0;
  for (int q = 0; q < a.P; ++q)
    ok = ok && a.tokens[q] == b.tokens[q] && a.tricks[q] == b.tricks[q] && a.hsize[q] == b.hsize[q] &&
         a.trank[q] == b.trank[q] && a.tcolor[q] == b.tcolor[q] && a.bid[q] == b.bid[q];
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: test_infer weights [P] [games] [bench_states]\n"); return 2; }
  LearnedPolicy pol;
  if (!pol.load(argv[1])) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 2; }
  const int P = argc > 2 ? std::atoi(argv[2]) : 5;
  const int games = argc > 3 ? std::atoi(argv[3]) : 60;
  const int bench_states = argc > 4 ? std::atoi(argv[4]) : 120;
  RolloutPolicy greedy; greedy.pol = &pol;  // rollouts as in the champion
  InferParams ip; ip.pool = 64; ip.alpha = 1.0; ip.eps = 0.1;

  // --- 1 + 2 + 3: games where every seat samples from the learned policy at T=1
  // (opponent model is exact) and games where every seat is pimcl:16 (realistic)
  const char* mname[3] = {"policy-T1", "pimcl:16", "pimcl:32-all-phases"};
  InferParams sets[16];
  int NSET = 0;
  for (double al : {1.0, 0.5, 0.3})
    for (double ep : {0.1, 0.3}) { sets[NSET].alpha = al; sets[NSET].eps = ep; sets[NSET].temp = 1.0; ++NSET; }
  sets[NSET].alpha = 1.0; sets[NSET].eps = 0.1; sets[NSET].temp = 2.0; ++NSET;
  const char* modes_env = std::getenv("QC_MODES");
  std::string modes = modes_env ? modes_env : "01";
  for (char mc : modes) {
    const int mode = mc - '0';
    Rng rng(1000 + mode);
    long recon_checks = 0, safety_checks = 0, safety_alt_found = 0;
    long diag_n = 0;
    double mse_uni = 0, mse_set[16] = {0}, ess_set[16] = {0}, rank_set[16] = {0};
    for (int g = 0; g < games; ++g) {
      Round s; s.init(P, g % P); s.deal(rng);
      PubHist h;
      std::vector<Round> at;  // true state before each history entry
      while (!s.done()) {
        int a;
        if (s.phase == PLAY && s.trick_no >= 1) {
          const int me = s.cur;
          Round view = mask_for(s, me);
          std::vector<OppDecision> ev;
          if (build_evidence(view, me, h, ev)) {
            // (1) reconstruction against recorded truth
            int k = 0;
            for (int i = P; i < h.n; ++i) {
              if (h.player[i] == me) continue;
              const OppDecision& d = ev[k++];
              Round rs; fill_snapshot(d, s, s.R, rs);
              const Round& tr = at[i];
              if (d.q != h.player[i] || d.a != h.action[i] || !same_public(rs, tr) ||
                  std::memcmp(rs.hand[d.q], tr.hand[d.q], MAXR) != 0 || rs.discard[d.q] != tr.discard[d.q]) {
                std::printf("FAIL reconstruction (mode %d game %d hist %d)\n", mode, g, i); return 1;
              }
              ++recon_checks;
            }
            if (k != (int)ev.size()) { std::printf("FAIL evidence count\n"); return 1; }
            const double llt = evidence_loglik(ev, s, pol, ip);
            if (std::isinf(llt)) { std::printf("FAIL true world rejected (mode %d game %d)\n", mode, g); return 1; }
            // (3) diagnostics: weighted posterior over a uniform pool, several settings
            {
              const int M = 1024;
              Rng dr(g * 977 + h.n);
              std::vector<Round> pool(M);
              for (auto& w : pool) sample_world(view, me, dr, w);
              double mu[MAXP][MAXR] = {{0}};
              for (auto& w : pool) for (int q = 0; q < P; ++q) for (int r = 0; r < s.R; ++r) mu[q][r] += w.hand[q][r];
              for (int q = 0; q < P; ++q) if (q != me) for (int r = 0; r < s.R; ++r) {
                const double e = mu[q][r] / M - s.hand[q][r]; mse_uni += e * e;
              }
              for (int k = 0; k < NSET; ++k) {
                std::vector<double> lw(M);
                double mx = -INFINITY;
                for (int i = 0; i < M; ++i) {
                  lw[i] = sets[k].alpha * evidence_loglik(ev, pool[i], pol, sets[k]);
                  mx = std::max(mx, lw[i]);
                }
                const double llt = sets[k].alpha * evidence_loglik(ev, s, pol, sets[k]);
                double tot = 0, tot2 = 0, mi[MAXP][MAXR] = {{0}};
                int better = 0;
                for (int i = 0; i < M; ++i) {
                  better += lw[i] > llt;
                  const double w = std::isinf(lw[i]) ? 0.0 : std::exp(lw[i] - mx);
                  tot += w; tot2 += w * w;
                  for (int q = 0; q < P; ++q) for (int r = 0; r < s.R; ++r) mi[q][r] += w * pool[i].hand[q][r];
                }
                for (int q = 0; q < P; ++q) if (q != me) for (int r = 0; r < s.R; ++r) {
                  const double e = mi[q][r] / tot - s.hand[q][r]; mse_set[k] += e * e;
                }
                ess_set[k] += tot * tot / tot2;
                rank_set[k] += (double)better / M;
              }
              ++diag_n;
            }
            // (2) information safety: a different true deal, same view + history
            if ((h.n + g) % 3 == 0) {
              Rng ar(g * 31 + h.n);
              Round alt;
              bool found = false;
              for (int tries = 0; tries < 200 && !found; ++tries) {
                sample_world(view, me, ar, alt);
                bool differs = false;
                for (int q = 0; q < P; ++q)
                  if (q != me) differs |= std::memcmp(alt.hand[q], s.hand[q], MAXR) != 0 || alt.discard[q] != s.discard[q];
                found = differs && !std::isinf(evidence_loglik(ev, alt, pol, ip));
              }
              if (found) {
                ++safety_alt_found;
                Round v1 = mask_for(s, me), v2 = mask_for(alt, me);
                if (std::memcmp(&v1, &v2, sizeof(Round)) != 0) { std::printf("FAIL masked views differ\n"); return 1; }
                double va[64], vb[64];
                Rng ra(777 + g), rb(777 + g);
                int a1 = pimci_decide(v1, me, h, 16, 0, greedy, ip, ra, va);
                int a2 = pimci_decide(v2, me, h, 16, 0, greedy, ip, rb, vb);
                int acts[64]; int n = v1.legal(acts);
                bool same = a1 == a2;
                if (n > 1) for (int i = 0; i < n; ++i) same = same && va[i] == vb[i];
                if (!same) { std::printf("FAIL info safety: decision depends on hidden hands (game %d)\n", g); return 1; }
                ++safety_checks;
              }
            }
          }
        }
        if (s.phase == PLAY) {
          if (mode == 0) a = learned_play(s, rng, pol, 1.0);
          else if (mode == 2) a = pimc_decide_rp(mask_for(s, s.cur), s.cur, 32, 0, greedy, rng);
          else {
            Round v = mask_for(s, s.cur);
            a = pimc_decide_rp(v, s.cur, 16, 0, greedy, rng);
          }
        } else if (mode == 2) {
          a = pimc_decide_rp(mask_for(s, s.cur), s.cur, 32, 0, greedy, rng);
        } else if (s.phase == DISCARD && std::getenv("QC_RANDOM_DISCARD")) {
          int acts[64]; int n = s.legal(acts); a = acts[rng.below(n)];
        } else a = heuristic_action(s, rng, HeurParams());
        if (s.phase == BID || s.phase == PLAY) at.push_back(s);
        record_public(h, s, a);
        s.apply(a);
        if (s.phase == DISCARD) at.clear();
      }
    }
    std::printf("[%s opponents] reconstruction OK: %ld opponent decisions rebuilt exactly\n",
                mname[mode], recon_checks);
    std::printf("[%s opponents] info-safety OK: %ld decisions, identical action+values with a different true deal "
                "(%ld alternates found)\n", mname[mode], safety_checks, safety_alt_found);
    std::printf("[%s opponents] belief diagnostics over %ld decisions (trick>=1), pool M=1024: belief MSE (sum over "
                "opponent x rank cells of (posterior mean count - true count)^2), uniform sampler %.4f\n",
                mname[mode], diag_n, mse_uni / diag_n);
    for (int k = 0; k < NSET; ++k)
      std::printf("    alpha %.2f eps %.2f T %.2f: MSE %.4f (%+.2f%%)  ESS %.1f  frac(candidates more likely than truth) %.3f\n",
                  sets[k].alpha, sets[k].eps, sets[k].temp, mse_set[k] / diag_n, 100.0 * (mse_set[k] / mse_uni - 1),
                  ess_set[k] / diag_n, rank_set[k] / diag_n);
    std::fflush(stdout);
  }

  // --- 4: CPU cost per PLAY decision (decisions with >= 2 legal actions), interleaved
  if (bench_states > 0) {
    const int N = 128;
    const int Ms[3] = {0, 512, 1024};
    double cpu[3] = {0, 0, 0};
    double cpu_ev[3] = {0, 0, 0};
    int got = 0, got_ev = 0;
    Rng rng(4242);
    while (got < bench_states) {
      Round s; s.init(P, got % P); s.deal(rng);
      PubHist h;
      while (!s.done() && got < bench_states) {
        if (s.phase == PLAY) {
          const int me = s.cur;
          Round v = mask_for(s, me);
          int acts[64];
          if (v.legal(acts) >= 2) {
            std::vector<OppDecision> ev;
            const bool has_ev = build_evidence(v, me, h, ev);
            for (int k = 0; k < 3; ++k) {
              Rng r(got);
              InferParams bp = ip; bp.pool = Ms[k];
              std::clock_t c0 = std::clock();
              if (k == 0) pimc_decide_rp(v, me, N, 0, greedy, r);
              else pimci_decide(v, me, h, N, 0, greedy, bp, r);
              const double dt = (double)(std::clock() - c0) / CLOCKS_PER_SEC;
              cpu[k] += dt;
              if (has_ev) cpu_ev[k] += dt;
            }
            ++got; got_ev += has_ev;
          }
        }
        int a;
        if (s.phase == PLAY) a = learned_play(s, rng, pol, 0.0);
        else a = heuristic_action(s, rng, HeurParams());
        record_public(h, s, a);
        s.apply(a);
      }
    }
    std::printf("{\"bench\":\"cpu_per_play_decision\",\"P\":%d,\"N\":%d,\"states\":%d,\"states_with_evidence\":%d,"
                "\"ms\":{\"pimcl\":%.2f,\"pimci_M512\":%.2f,\"pimci_M1024\":%.2f},"
                "\"ms_with_evidence\":{\"pimcl\":%.2f,\"pimci_M512\":%.2f,\"pimci_M1024\":%.2f},"
                "\"ratio_M512\":%.2f,\"ratio_M1024\":%.2f}\n",
                P, N, got, got_ev, cpu[0] / got * 1e3, cpu[1] / got * 1e3, cpu[2] / got * 1e3,
                cpu_ev[0] / got_ev * 1e3, cpu_ev[1] / got_ev * 1e3, cpu_ev[2] / got_ev * 1e3, cpu[1] / cpu[0],
                cpu[2] / cpu[0]);
  }
  return 0;
}
