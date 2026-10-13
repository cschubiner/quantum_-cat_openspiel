// CPU ms per decision of several bot specs on identical states (interleaved so
// machine load affects all specs equally).  States: learned-policy self-play
// (temp 0.5), every decision with >= 2 legal actions, all phases.
// usage: bench_deepsearch P nstates seed spec1 [spec2 ...]
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include "qc_bots.h"
using namespace qc;
int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: bench_deepsearch P nstates seed spec...\n"); return 2; }
  const int P = std::atoi(argv[1]), nstates = std::atoi(argv[2]);
  const uint64_t seed = std::strtoull(argv[3], nullptr, 10);
  std::vector<Bot> bots;
  std::vector<std::string> names;
  for (int i = 4; i < argc; ++i) { bots.push_back(parse_bot(argv[i])); names.push_back(argv[i]); }
  const int B = (int)bots.size();
  LearnedPolicy pol;
  if (!pol.load("weights/g3_mlp32.txt")) return 2;
  RolloutPolicy gen; gen.pol = &pol; gen.temp = 0.5;
  std::vector<double> tot(B, 0), mx(B, 0), ph[4];
  for (auto& v : ph) v.assign(B, 0);
  int nph[4] = {0};
  std::vector<int> agree(B, 0);
  Rng rng(seed);
  int got = 0;
  for (int g = 0; got < nstates; ++g) {
    Round s; s.init(P, g % P); s.deal(rng);
    while (!s.done() && got < nstates) {
      int acts[64];
      if (s.legal(acts) >= 2) {
        Round v = mask_for(s, s.cur);
        int a0 = -1;
        for (int b = 0; b < B; ++b) {
          Rng r(seed * 1000 + got);
          std::clock_t c0 = std::clock();
          int a = bots[b].decide(v, s.cur, r);
          double dt = (double)(std::clock() - c0) / CLOCKS_PER_SEC * 1e3;
          if (b == 0) a0 = a;
          agree[b] += a == a0;
          tot[b] += dt; mx[b] = std::max(mx[b], dt); ph[s.phase][b] += dt;
        }
        ++nph[s.phase]; ++got;
      }
      s.apply(gen.act(s, rng));
    }
  }
  for (int b = 0; b < B; ++b)
    std::printf("{\"P\":%d,\"states\":%d,\"spec\":\"%s\",\"ms_mean\":%.2f,\"ms_max\":%.1f,\"ms_discard\":%.2f,"
                "\"ms_bid\":%.2f,\"ms_play\":%.2f,\"rel_to_first\":%.2f,\"agree_first\":%.3f}\n",
                P, got, names[b].c_str(), tot[b] / got, mx[b], ph[DISCARD][b] / std::max(1, nph[DISCARD]),
                ph[BID][b] / std::max(1, nph[BID]), ph[PLAY][b] / std::max(1, nph[PLAY]), tot[b] / tot[0],
                (double)agree[b] / got);
}
