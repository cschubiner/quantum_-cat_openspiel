// Ground-truth check of a bidding rule under strong play.
// Deals rounds (start seat 0) until seat 0 holds >= K copies of the top rank
// after its discard; then plays the identical deal with seat 0's bid forced to
// 1, 2, 3 while every decision (other bids, all play) is made by PIMC with the
// learned rollout policy. Reports seat-0 mean score and score-difference utility.
// usage: bid_truth P K rounds worlds threads seed weights
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include "qc_agents.h"
#include "qc_learn.h"
using namespace qc;
int main(int argc, char** argv) {
  int P = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]), W = atoi(argv[4]), T = atoi(argv[5]);
  uint64_t seed = strtoull(argv[6], nullptr, 10);
  LearnedPolicy pol; if (!pol.load(argv[7])) { fprintf(stderr, "bad weights\n"); return 2; }
  RolloutPolicy rp; rp.pol = &pol;
  double sc[3] = {0}, ut[3] = {0}, sc2[3] = {0}, dd[3][3] = {{0}}; int hits[3] = {0}, par[3] = {0};
  std::mutex mu; std::atomic<int> next{0};
  auto work = [&]() {
    for (;;) {
      int i = next++; if (i >= N) return;
      Rng rng(seed * 7777 + i);
      Round s;
      for (;;) {  // rejection sample the condition
        s.init(P, 0); s.deal(rng);
        while (s.phase == DISCARD) s.apply(heuristic_action(s, rng, rp.hp));
        if (s.hand[0][s.R - 1] >= K) break;
      }
      double u[3], v[3]; int h[3], pa[3];
      for (int b = 0; b < 3; ++b) {
        Round t = s; Rng rr(seed * 31 + i);
        t.apply(PRED_OFF + b + 1);
        while (!t.done()) {
          Round view = mask_for(t, t.cur);
          t.apply(pimc_decide_rp(view, t.cur, W, 0, rp, rr));
        }
        double s5[MAXP]; t.scores(s5);
        v[b] = s5[0]; u[b] = utility(t, 0, 0); h[b] = (t.paradox != 0 && t.bid[0] == t.tricks[0]); pa[b] = t.paradox == 0;
      }
      std::lock_guard<std::mutex> g(mu);
      for (int b = 0; b < 3; ++b) { sc[b] += v[b]; ut[b] += u[b]; sc2[b] += u[b] * u[b]; hits[b] += h[b]; par[b] += pa[b];
        for (int c = 0; c < 3; ++c) dd[b][c] += (u[b] - u[c]) * (u[b] - u[c]); }
    }
  };
  std::vector<std::thread> th; for (int t = 0; t < T; ++t) th.emplace_back(work); for (auto& t : th) t.join();
  for (int b = 0; b < 3; ++b)
    printf("P=%d K=%d bid %d: own score %.3f  util(own-mean others) %.3f  bid-hit %.3f  paradox %.3f\n", P, K, b + 1,
           sc[b] / N, ut[b] / N, (double)hits[b] / N, (double)par[b] / N);
  for (int b = 1; b < 3; ++b) {
    double m = (ut[b] - ut[0]) / N, sd = std::sqrt(dd[b][0] / N - m * m);
    printf("util(bid %d) - util(bid 1) = %+.3f  (95%% CI %+.3f..%+.3f)\n", b + 1, m, m - 1.96 * sd / std::sqrt(N), m + 1.96 * sd / std::sqrt(N));
  }
}
