// Sanity tests for the learned-policy feature function:
//  1. phi(s,a) identical on the true state, the masked view, and many re-sampled
//     worlds (=> phi never reads opponents' hands/discards).
//  2. pimc_decide_rp with heuristic rollouts == pimc_decide (bit-exact values).
//  3. rollout timing: heuristic vs learned (linear / mlp with random weights).
// build: clang++ -O3 -march=native -std=c++17 -DQC_CHECK test_features.cpp -o test_features
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "qc_learn.h"

using namespace qc;

static void phi_all(const Round& s, std::vector<float>& out) {
  int acts[NC * MAXR + 1];
  int n = s.legal(acts);
  PlayCtx x;
  play_context(s, x, n);
  out.assign((size_t)n * NFEAT, 0.f);
  for (int i = 0; i < n; ++i)
    if (acts[i] != ACT_PARADOX) play_features(s, x, acts[i], &out[(size_t)i * NFEAT]);
}

int main(int argc, char** argv) {
  int P = argc > 1 ? std::atoi(argv[1]) : 5;
  Rng rng(12345);
  HeurParams hp; hp.temp = 0.3;  // some diversity in states
  long checks = 0, decisions = 0;
  for (int g = 0; g < 400; ++g) {
    Round s; s.init(P, g % P); s.deal(rng);
    while (!s.done()) {
      if (s.phase == PLAY) {
        const int me = s.cur;
        std::vector<float> f0, f1;
        phi_all(s, f0);
        Round v = mask_for(s, me);
        phi_all(v, f1);
        if (f0 != f1) { std::printf("FAIL masked view differs (game %d)\n", g); return 1; }
        for (int k = 0; k < 8; ++k) {
          Round w;
          sample_world(v, me, rng, w);
          phi_all(w, f1);
          if (f0 != f1) { std::printf("FAIL resampled world differs (game %d)\n", g); return 1; }
          ++checks;
        }
        ++decisions;
      }
      s.apply(heuristic_action(s, rng, hp));
    }
  }
  std::fflush(stdout);
  std::printf("information test OK: %ld decisions, %ld resampled-world checks\n", decisions, checks);

  // 2. pimc_decide_rp(heuristic) == pimc_decide
  {
    Rng r0(7);
    int mism = 0, tested = 0;
    for (int g = 0; g < 20; ++g) {
      Round s; s.init(P, g % P); s.deal(r0);
      while (!s.done()) {
        Round v = mask_for(s, s.cur);
        if (s.phase == PLAY) {
          SearchParams sp; sp.worlds = 8;
          RolloutPolicy rp;
          double va[64], vb[64];
          Rng ra(g * 1000 + s.trick_no * 10 + s.nplayed), rb = ra;
          int a1 = pimc_decide(v, s.cur, sp, ra, va);
          int a2 = pimc_decide_rp(v, s.cur, 8, 0, rp, rb, vb);
          int acts[64]; int n = v.legal(acts);
          ++tested;
          if (a1 != a2) ++mism;
          if (n > 1) for (int i = 0; i < n; ++i) if (va[i] != vb[i]) ++mism;
        }
        s.apply(heuristic_action(s, r0, hp));
      }
    }
    std::printf("pimc_decide_rp(heur) vs pimc_decide: %d decisions, %d mismatches\n", tested, mism);
    if (mism) return 1;
  }

  // 3. timing
  LearnedPolicy lin; lin.kind = 0; lin.w.resize(NFEAT);
  for (auto& x : lin.w) x = (float)(rng.uniform() - 0.5);
  lin.w[0] = 2.0f;
  LearnedPolicy mlp; mlp.kind = 1; mlp.H = 32;
  mlp.W1.resize(32 * NFEAT); mlp.b1.resize(32); mlp.w2.resize(32);
  for (auto& x : mlp.W1) x = (float)(rng.uniform() - 0.5);
  for (auto& x : mlp.b1) x = (float)(rng.uniform() - 0.5);
  for (auto& x : mlp.w2) x = (float)(rng.uniform() - 0.5);
  mlp.finalize();
  const int N = 20000;
  for (int which = 0; which < 3; ++which) {
    RolloutPolicy rp;
    if (which == 1) rp.pol = &lin;
    if (which == 2) rp.pol = &mlp;
    Rng r1(99);
    auto t0 = std::chrono::steady_clock::now();
    long par = 0;
    for (int i = 0; i < N; ++i) {
      Round s; s.init(P, i % P); s.deal(r1);
      while (!s.done()) s.apply(rp.act(s, r1));
      par += s.paradox >= 0;
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%-6s rollouts: %.2f us/round (paradox rounds %.3f)\n",
                which == 0 ? "heur" : which == 1 ? "linear" : "mlp32", dt / N * 1e6, (double)par / N);
  }
  return 0;
}
