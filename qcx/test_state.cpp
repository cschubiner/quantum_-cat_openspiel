// Information-safety test for the big model's raw state encoding:
// state_features(s) (and the big-model action inputs) must be identical on the
// true state, the actor's masked view, and many worlds re-sampled from that
// view (=> the encoding never reads opponents' hands/discards).  Also checks
// that the encoding changes when the actor's own hand differs (non-trivial).
// build: clang++ -O3 -march=native -std=c++17 -DQC_CHECK test_state.cpp -o /tmp/claude-501/bin/test_state
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "qc_big.h"

using namespace qc;

static void enc_all(const Round& s, std::vector<float>& out) {
  int acts[NC * MAXR + 1];
  int n = s.legal(acts);
  out.assign(NSTATE + (size_t)n * NACT, 0.f);
  state_features(s, out.data());
  PlayCtx x;
  play_context(s, x, n);
  for (int i = 0; i < n; ++i) {
    float* a = &out[NSTATE + (size_t)i * NACT];
    if (acts[i] != ACT_PARADOX) play_features(s, x, acts[i], a);
    action_extra(s, acts[i], a + NFEAT);
  }
}

int main(int argc, char** argv) {
  const int games = argc > 1 ? std::atoi(argv[1]) : 300;
  long checks = 0, decisions = 0, nontriv = 0;
  for (int P = 3; P <= 5; ++P) {
    Rng rng(777 + P);
    HeurParams hp; hp.temp = 0.3;
    for (int g = 0; g < games; ++g) {
      Round s; s.init(P, g % P); s.deal(rng);
      while (!s.done()) {
        if (s.phase == PLAY) {
          const int me = s.cur;
          std::vector<float> f0, f1;
          enc_all(s, f0);
          Round v = mask_for(s, me);
          enc_all(v, f1);
          if (f0 != f1) { std::printf("FAIL masked view differs (P %d game %d)\n", P, g); return 1; }
          for (int k = 0; k < 8; ++k) {
            Round w;
            sample_world(v, me, rng, w);
            enc_all(w, f1);
            if (f0 != f1) { std::printf("FAIL resampled world differs (P %d game %d)\n", P, g); return 1; }
            ++checks;
          }
          // swap one of my cards with an opponent's: encoding must change
          for (int r = 0; r < s.R; ++r) {
            if (!s.hand[me][r]) continue;
            int q = (me + 1) % P, r2 = -1;
            for (int t = 0; t < s.R; ++t) if (t != r && s.hand[q][t]) { r2 = t; break; }
            if (r2 < 0) break;
            Round s2 = s;
            s2.hand[me][r]--; s2.hand[me][r2]++; s2.hand[q][r2]--; s2.hand[q][r]++;
            float a[NSTATE], b[NSTATE];
            state_features(s, a); state_features(s2, b);
            nontriv += std::memcmp(a, b, sizeof(a)) != 0;
            break;
          }
          ++decisions;
        }
        s.apply(heuristic_action(s, rng, hp));
      }
    }
  }
  std::printf("state information test OK: %ld decisions (3/4/5p), %ld resampled-world checks; "
              "own-hand swap changed the encoding in %ld cases; NSTATE=%d NACT=%d\n",
              decisions, checks, nontriv, NSTATE, NACT);
  return 0;
}
