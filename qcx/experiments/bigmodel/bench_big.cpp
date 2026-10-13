// CPU cost of a PIMC PLAY decision with g3 (small MLP) rollouts vs big-model
// rollouts (every seat, or root seat only), interleaved on identical states so
// machine load affects all variants equally.  Also times raw policy calls.
// usage: bench_big P worlds small_weights big_weights [states]
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "qc_big.h"
using namespace qc;
int main(int argc, char** argv) {
  int P = std::atoi(argv[1]), worlds = std::atoi(argv[2]);
  LearnedPolicy small; BigPolicy big;
  if (!small.load(argv[3]) || !big.load(argv[4])) return 2;
  int nstates = argc > 5 ? std::atoi(argv[5]) : 100;
  RolloutPolicy rs; rs.pol = &small;
  BigRolloutPolicy rb; rb.big = &big; rb.small = &small;
  double cpu[3] = {0, 0, 0};
  double pol_small = 0, pol_big = 0; long calls = 0;
  Rng rng(4242);
  HeurParams hp; hp.temp = 0.3;
  int got = 0;
  while (got < nstates) {
    Round s; s.init(P, got % P); s.deal(rng);
    while (!s.done() && got < nstates) {
      if (s.phase == PLAY) {
        Round v = mask_for(s, s.cur);
        int acts[64];
        if (v.legal(acts) >= 2) {
          for (int k = 0; k < 3; ++k) {
            Rng r(got);
            std::clock_t c0 = std::clock();
            if (k == 0) pimc_decide_rp(v, s.cur, worlds, 0, rs, r);
            else { BigRolloutPolicy x = rb; if (k == 2) x.big_seat = s.cur; pimc_decide_t(v, s.cur, worlds, 0, x, r); }
            cpu[k] += (double)(std::clock() - c0) / CLOCKS_PER_SEC;
          }
          Rng r(got);
          std::clock_t c0 = std::clock();
          for (int i = 0; i < 200; ++i) learned_play(v, r, small, 0.0);
          std::clock_t c1 = std::clock();
          for (int i = 0; i < 200; ++i) big_play(v, r, big);
          std::clock_t c2 = std::clock();
          pol_small += (double)(c1 - c0) / CLOCKS_PER_SEC; pol_big += (double)(c2 - c1) / CLOCKS_PER_SEC;
          calls += 200;
          ++got;
        }
      }
      s.apply(heuristic_action(s, rng, hp));
    }
  }
  std::printf("{\"P\":%d,\"worlds\":%d,\"states\":%d,\"big\":\"%s\",\"ms_per_decision\":{\"g3\":%.2f,\"big_all\":%.2f,"
              "\"big_root\":%.2f},\"ratio_all\":%.2f,\"ratio_root\":%.2f,\"us_per_policy_call\":{\"g3\":%.2f,\"big\":%.2f}}\n",
              P, worlds, nstates, argv[4], cpu[0] / nstates * 1e3, cpu[1] / nstates * 1e3, cpu[2] / nstates * 1e3,
              cpu[1] / cpu[0], cpu[2] / cpu[0], pol_small / calls * 1e6, pol_big / calls * 1e6);
}
