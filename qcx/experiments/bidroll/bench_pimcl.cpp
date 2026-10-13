// CPU cost of a pimc:<worlds> PLAY decision with heuristic vs learned rollouts,
// interleaved on identical states so machine load affects all variants equally.
// usage: bench_pimcl worlds lin_weights mlp_weights [states]
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "qc_learn.h"
using namespace qc;
int main(int argc, char** argv) {
  int worlds = std::atoi(argv[1]);
  LearnedPolicy lin, mlp;
  if (!lin.load(argv[2]) || !mlp.load(argv[3])) return 2;
  int nstates = argc > 4 ? std::atoi(argv[4]) : 150;
  RolloutPolicy rps[3];
  rps[1].pol = &lin; rps[2].pol = &mlp;
  double cpu[3] = {0, 0, 0};
  Rng rng(4242);
  HeurParams hp; hp.temp = 0.3;
  int got = 0;
  while (got < nstates) {
    Round s; s.init(5, got % 5); s.deal(rng);
    while (!s.done() && got < nstates) {
      if (s.phase == PLAY) {
        Round v = mask_for(s, s.cur);
        int acts[64];
        if (v.legal(acts) >= 2) {
          for (int k = 0; k < 3; ++k) {
            Rng r(got);
            std::clock_t c0 = std::clock();
            pimc_decide_rp(v, s.cur, worlds, 0, rps[k], r);
            cpu[k] += (double)(std::clock() - c0) / CLOCKS_PER_SEC;
          }
          ++got;
        }
      }
      s.apply(heuristic_action(s, rng, hp));
    }
  }
  std::printf("{\"worlds\":%d,\"states\":%d,\"ms_per_decision\":{\"heur\":%.2f,\"lin\":%.2f,\"mlp\":%.2f},"
              "\"ratio_lin\":%.2f,\"ratio_mlp\":%.2f}\n", worlds, nstates, cpu[0] / nstates * 1e3,
              cpu[1] / nstates * 1e3, cpu[2] / nstates * 1e3, cpu[1] / cpu[0], cpu[2] / cpu[0]);
}
