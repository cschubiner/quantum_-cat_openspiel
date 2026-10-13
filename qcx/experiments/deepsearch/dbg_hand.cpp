#include <cstdio>
#include "qc_agents.h"
using namespace qc;
int main() {
  Round v; v.init(5, 0);
  int hand[9] = {1,1,1,1,0,0,0,0,4};
  for (int r = 0; r < 9; ++r) v.hand[0][r] = hand[r];
  v.discard[0] = 0; v.phase = BID; v.cur = 0;
  for (int q = 0; q < 5; ++q) v.hsize[q] = 8;
  Rng rng(3); HeurParams hp; Round w;
  for (int b = 1; b <= 4; ++b) {
    double sc = 0, tr = 0, par = 0, hit = 0; int hist[10] = {0}; int N = 20000;
    for (int i = 0; i < N; ++i) {
      sample_world(v, 0, rng, w); w.apply(PRED_OFF + b); rollout(w, rng, hp);
      double s[5]; w.scores(s); sc += s[0]; tr += w.tricks[0]; par += w.paradox == 0; hit += (w.paradox != 0 && w.tricks[0] == b);
      hist[w.tricks[0]]++;
    }
    std::printf("bid %d: score %.2f tricks %.2f paradox %.3f hit %.3f  hist", b, sc/N, tr/N, par/N, hit/N);
    for (int t = 0; t < 8; ++t) std::printf(" %d", hist[t]); std::printf("\n");
  }
}
