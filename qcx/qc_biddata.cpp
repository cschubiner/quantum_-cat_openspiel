// Generates (hand-at-bid, position, earlier bids) -> bid rows from PIMC self-play.
// usage: qc_biddata P deals threads seed worlds out.csv
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include "qc_agents.h"
using namespace qc;
int main(int argc, char** argv) {
  int P = std::atoi(argv[1]), deals = std::atoi(argv[2]), threads = std::atoi(argv[3]);
  uint64_t seed = std::strtoull(argv[4], nullptr, 10);
  int worlds = std::atoi(argv[5]);
  FILE* f = std::fopen(argv[6], "w");
  std::mutex mu; std::atomic<int> next{0};
  auto work = [&]() {
    SearchParams sp; sp.worlds = worlds;
    for (;;) {
      int d = next++; if (d >= deals) return;
      Rng rng(seed * 1000003ULL + d);
      Round s; s.init(P, d % P); s.deal(rng);
      while (s.phase != PLAY) {
        int me = s.cur;
        Round view = mask_for(s, me);
        int a = pimc_decide(view, me, sp, rng);
        if (s.phase == BID) {
          char line[256]; int n = 0;
          int pos = (me - s.start + P) % P;
          n += std::snprintf(line + n, sizeof(line) - n, "%d,%d", P, pos);
          for (int r = 0; r < MAXR; ++r) n += std::snprintf(line + n, sizeof(line) - n, ",%d", r < s.R ? s.hand[me][r] : 0);
          for (int k = 1; k < P; ++k) { int q = (s.start + k - 1) % P; n += std::snprintf(line + n, sizeof(line) - n, ",%d", k <= pos ? s.bid[q] : 0); }
          for (int k = P; k < MAXP; ++k) n += std::snprintf(line + n, sizeof(line) - n, ",0");
          n += std::snprintf(line + n, sizeof(line) - n, ",%d\n", a - PRED_OFF);
          std::lock_guard<std::mutex> g(mu); std::fputs(line, f);
        }
        s.apply(a);
      }
    }
  };
  std::vector<std::thread> pool; for (int t = 0; t < threads; ++t) pool.emplace_back(work);
  for (auto& t : pool) t.join(); std::fclose(f);
}
