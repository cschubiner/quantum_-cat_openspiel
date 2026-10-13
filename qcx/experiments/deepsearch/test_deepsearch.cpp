// Information-safety test for the deep-search agents (qc_search.h).
// For random states from learned-policy self-play (all phases), each search is
// run with the same seed on (a) the true unmasked state, (b) the masked view,
// (c) a different true state whose opponents' hands/discards were re-dealt
// consistently with the actor's information.  The chosen action and the full
// per-action statistics must be bit-identical, i.e. the search never reads the
// opponents' hidden cards.  A deliberately leaky control must be caught.
// build: clang++ -O3 -march=native -std=c++17 test_deepsearch.cpp -o /tmp/claude-501/bin/test_deepsearch
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "qc_search.h"
using namespace qc;

struct Out { int a; double v[NC * MAXR + 1]; };

static bool same(const Out& x, const Out& y, int n) {
  if (x.a != y.a) return false;
  return std::memcmp(x.v, y.v, sizeof(double) * n) == 0;
}

int main(int argc, char** argv) {
  const char* wpath = argc > 1 ? argv[1] : "weights/g3_mlp32.txt";
  int nstates = argc > 2 ? std::atoi(argv[2]) : 60;
  LearnedPolicy pol;
  if (!pol.load(wpath)) { std::printf("cannot load %s\n", wpath); return 2; }
  RolloutPolicy rp; rp.pol = &pol;
  RolloutPolicy gen; gen.pol = &pol; gen.temp = 0.5;
  IsmctsParams ip0; ip0.mult = 6; ip0.cpuct = 2.0; ip0.opp_mode = 0;
  IsmctsParams ip1 = ip0; ip1.opp_mode = 1;
  Rng rng(777);
  int checked = 0, diff_hidden = 0, fails = 0, control_caught = 0, by_phase[5] = {0};
  for (int g = 0; checked < nstates; ++g) {
    const int P = 3 + g % 3;
    Round s; s.init(P, g % P); s.deal(rng);
    while (!s.done() && checked < nstates) {
      const int me = s.cur;
      int acts[64];
      const int n = s.legal(acts);
      if (n >= 2 && rng.uniform() < 0.25) {
        Round view = mask_for(s, me);
        Round alt;
        sample_world(view, me, rng, alt);  // another truth consistent with me's information
        bool differs = false;
        for (int q = 0; q < s.P; ++q)
          if (q != me && (std::memcmp(alt.hand[q], s.hand[q], sizeof(s.hand[q])) || alt.discard[q] != s.discard[q])) differs = true;
        diff_hidden += differs;
        const Round* in[3] = {&s, &view, &alt};
        for (int alg = 0; alg < 5; ++alg) {
          Out o[3];
          for (int k = 0; k < 3; ++k) {
            std::memset(o[k].v, 0, sizeof(o[k].v));
            Rng r(1000 + checked);
            switch (alg) {
              case 0: o[k].a = pimc_decide_rp(*in[k], me, 4, 0, rp, r, o[k].v); break;
              case 1: o[k].a = ismcts_decide(*in[k], me, ip0, rp, r, o[k].v); break;
              case 2: o[k].a = ismcts_decide(*in[k], me, ip1, rp, r, o[k].v); break;
              case 3: o[k].a = pimc2_decide(*in[k], me, 2, 3, rp, r, o[k].v); break;
              case 4: o[k].a = pimce_decide(*in[k], me, 3, 1, rp, r, o[k].v); break;
            }
          }
          if (!same(o[0], o[1], n) || !same(o[0], o[2], n)) {
            ++fails;
            std::printf("FAIL alg %d state %d phase %d\n", alg, checked, s.phase);
          }
        }
        {  // negative control: a deliberately leaky search (rolls out the given state
           // itself, no re-sampling) must be caught by the same comparison.
          Out o[3];
          for (int k = 0; k < 3; ++k) {
            std::memset(o[k].v, 0, sizeof(o[k].v));
            for (int i = 0; i < n; ++i) {
              Round c = *in[k]; c.apply(acts[i]);
              Rng r(1000 + checked);
              while (!c.done()) c.apply(rp.act(c, r));
              o[k].v[i] = utility(c, me, 0);
            }
            o[k].a = 0;
          }
          control_caught += !same(o[0], o[2], n);
        }
        ++checked; ++by_phase[s.phase];
      }
      s.apply(gen.act(s, rng));
    }
  }
  std::printf("{\"states\":%d,\"discard\":%d,\"bid\":%d,\"play\":%d,\"hidden_hands_changed\":%d,"
              "\"algorithms\":5,\"fails\":%d,\"leaky_control_caught\":%d}\n", checked, by_phase[DISCARD],
              by_phase[BID], by_phase[PLAY], diff_hidden, fails, control_caught);
  return fails ? 1 : 0;
}
