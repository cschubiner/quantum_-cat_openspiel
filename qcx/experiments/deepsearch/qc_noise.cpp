// Label-noise ceiling: at PLAY decisions of pimc:32 self-play, run PIMC twice more
// with independent RNG and measure how often run A's argmax is also optimal under
// run B (ties count), and the regret of A's choice measured by B.  Also scores the
// heuristic and (optionally) a learned policy against B on the same states.
// usage: qc_noise P rounds threads seed worlds [weights]
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "qc_learn.h"

using namespace qc;

int main(int argc, char** argv) {
  int P = std::atoi(argv[1]), rounds = std::atoi(argv[2]), threads = std::atoi(argv[3]);
  uint64_t seed = std::strtoull(argv[4], nullptr, 10);
  int worlds = std::atoi(argv[5]);
  LearnedPolicy pol;
  bool have = argc > 6 && pol.load(argv[6]);
  RolloutPolicy heur;
  std::mutex mu;
  long n = 0;
  double agreeA = 0, regA = 0, agreeH = 0, regH = 0, agreeL = 0, regL = 0;
  std::atomic<int> next{0};
  auto worker = [&]() {
    for (;;) {
      int g = next++;
      if (g >= rounds) return;
      Rng rng(seed * 1000003ULL + g);
      Round s; s.init(P, g % P); s.deal(rng);
      while (!s.done()) {
        const int me = s.cur;
        Round view = mask_for(s, me);
        int acts[NC * MAXR + 1];
        int na = view.legal(acts);
        int a = na == 1 ? acts[0] : pimc_decide_rp(view, me, worlds, 0, heur, rng);
        if (s.phase == PLAY && na >= 2) {
          double va[64], vb[64];
          pimc_decide_rp(view, me, worlds, 0, heur, rng, va);
          pimc_decide_rp(view, me, worlds, 0, heur, rng, vb);
          int ia = 0; for (int i = 1; i < na; ++i) if (va[i] > va[ia]) ia = i;
          double mb = vb[0]; for (int i = 1; i < na; ++i) mb = std::max(mb, vb[i]);
          // heuristic / learned greedy choice
          HeurParams hp; int ih = 0; double bh = -1e18;
          for (int i = 0; i < na; ++i) { double v = heuristic_move_score(view, acts[i], hp) + rng.uniform() * 1e-6; if (v > bh) { bh = v; ih = i; } }
          int il = 0;
          if (have) {
            PlayCtx x; play_context(view, x, na); float f[NFEAT]; double bl = -1e18;
            for (int i = 0; i < na; ++i) { play_features(view, x, acts[i], f); double v = pol.score(f) + rng.uniform() * 1e-6; if (v > bl) { bl = v; il = i; } }
          }
          std::lock_guard<std::mutex> lk(mu);
          ++n;
          agreeA += vb[ia] >= mb - 1e-9; regA += mb - vb[ia];
          agreeH += vb[ih] >= mb - 1e-9; regH += mb - vb[ih];
          agreeL += vb[il] >= mb - 1e-9; regL += mb - vb[il];
        }
        s.apply(a);
      }
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  std::printf("{\"decisions\":%ld,\"pimcA_vs_pimcB_agree\":%.4f,\"pimcA_regret\":%.4f,"
              "\"heur_agree\":%.4f,\"heur_regret\":%.4f,\"learned_agree\":%.4f,\"learned_regret\":%.4f}\n",
              n, agreeA / n, regA / n, agreeH / n, regH / n, have ? agreeL / n : -1.0, have ? regL / n : -1.0);
}
