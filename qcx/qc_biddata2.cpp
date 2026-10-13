// Bid dataset from the learned-rollout champion's own bids (all phases by
// pimc_decide_rp with the given rollout weights).
// usage: qc_biddata2 P deals threads seed worlds weights out.csv
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include "qc_learn.h"
using namespace qc;
int main(int argc, char** argv) {
  int P = atoi(argv[1]), deals = atoi(argv[2]), threads = atoi(argv[3]);
  uint64_t seed = strtoull(argv[4], nullptr, 10);
  int worlds = atoi(argv[5]);
  LearnedPolicy pol; if (!pol.load(argv[6])) { fprintf(stderr, "bad weights\n"); return 2; }
  RolloutPolicy rp; rp.pol = &pol;
  FILE* f = fopen(argv[7], "w");
  std::mutex mu; std::atomic<int> next{0};
  auto work = [&]() {
    for (;;) {
      int d = next++; if (d >= deals) return;
      Rng rng(seed * 1000003ULL + d);
      Round s; s.init(P, d % P); s.deal(rng);
      while (s.phase != PLAY) {
        int me = s.cur;
        Round view = mask_for(s, me);
        int a = pimc_decide_rp(view, me, worlds, 0, rp, rng);
        if (s.phase == BID) {
          char line[256]; int n = 0;
          int pos = (me - s.start + P) % P;
          n += snprintf(line + n, sizeof(line) - n, "%d,%d", P, pos);
          for (int r = 0; r < MAXR; ++r) n += snprintf(line + n, sizeof(line) - n, ",%d", r < s.R ? s.hand[me][r] : 0);
          for (int k = 1; k < MAXP; ++k) { int q = (s.start + k - 1) % P; n += snprintf(line + n, sizeof(line) - n, ",%d", (k <= pos && k < P) ? s.bid[q] : 0); }
          n += snprintf(line + n, sizeof(line) - n, ",%d\n", a - PRED_OFF);
          std::lock_guard<std::mutex> g(mu); fputs(line, f);
        }
        s.apply(a);
      }
    }
  };
  std::vector<std::thread> pool; for (int t = 0; t < threads; ++t) pool.emplace_back(work);
  for (auto& t : pool) t.join(); fclose(f);
}
