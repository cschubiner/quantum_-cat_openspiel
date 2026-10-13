#include <chrono>
#include <cstdio>
#include "qc_agents.h"
using namespace qc;
int main() {
  Rng rng(1); HeurParams hp; long n = 0; auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 200000; ++i) { Round s; s.init(5, i % 5); s.deal(rng); rollout(s, rng, hp); n++; }
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("heuristic rollouts: %.0f rounds/s/thread (%.2f us each)\n", n / dt, dt / n * 1e6);
}
