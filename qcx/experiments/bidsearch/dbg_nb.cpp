#include <cstdio>
#include "qc_agents.h"
using namespace qc;
int main() {
  int hands[2][9] = {{1,1,1,1,0,0,0,0,4},{1,1,1,1,0,0,0,1,3}};
  for (auto& hand : hands) {
    Round v; v.init(5, 0);
    for (int r = 0; r < 9; ++r) v.hand[0][r] = hand[r];
    v.discard[0] = 0; v.phase = BID; v.cur = 0;
    for (int q = 0; q < 5; ++q) v.hsize[q] = 8;
    SearchParams sp; Rng rng(7); double vals[4];
    int a = nested_bid_decide(v, 0, sp, 128, 16, rng, vals);
    std::printf("nested bid %d values %.2f %.2f %.2f %.2f\\n", a - PRED_OFF, vals[0], vals[1], vals[2], vals[3]);
  }
}
