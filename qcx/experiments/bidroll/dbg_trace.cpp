#include <cstdio>
#include "qc_agents.h"
using namespace qc;
const char* C = "RBYG";
int main(int argc, char** argv) {
  Round v; v.init(5, 0);
  int hand[9] = {1,1,1,1,0,0,0,0,4};
  for (int r = 0; r < 9; ++r) v.hand[0][r] = hand[r];
  v.discard[0] = 0; v.phase = BID; v.cur = 0;
  for (int q = 0; q < 5; ++q) v.hsize[q] = 8;
  HeurParams hp; Round w;
  for (int g = 0; g < 4; ++g) {
    Rng rng(100 + g);
    sample_world(v, 0, rng, w); w.apply(PRED_OFF + 2);
    while (w.phase == BID) w.apply(heuristic_action(w, rng, hp));
    std::printf("game %d bids:", g); for (int q = 0; q < 5; ++q) std::printf(" %d", w.bid[q]); std::printf("\n");
    while (!w.done()) {
      int p = w.cur; int a = heuristic_action(w, rng, hp);
      if (a == ACT_PARADOX) { std::printf("  P%d PARADOX\n", p); w.apply(a); break; }
      int c = a / w.R, r = a % w.R;
      if (p == 0) { std::printf("  t%d P0 hand:", w.trick_no); for (int k = 0; k < 9; ++k) for (int z = 0; z < w.hand[0][k]; ++z) std::printf("%d", k + 1);
        std::printf(" tok:%x led:%c tricks:%d plays %c%d\n", w.tokens[0], w.led < 0 ? '-' : C[w.led], w.tricks[0], C[c], r + 1); }
      w.apply(a);
    }
    double s[5]; w.scores(s); std::printf("  result tricks %d score %.0f\n", w.tricks[0], s[0]);
  }
}
