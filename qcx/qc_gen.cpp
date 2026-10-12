// PIMC self-play data generator for the learned PLAY-phase rollout policy.
//
// usage: qc_gen P rounds threads seed worlds rollout out.bin [rollout_temp]
//   rollout: "heur" or a weights file (learned play in rollouts for all seats)
// Every seat is the same PIMC bot (all phases).  At each PLAY decision with >=2
// legal actions we record, for every legal action, one float32 row:
//   [game_id, decision_id, q (PIMC mean utility), heur_score, phi_0..phi_{F-1}]
// File header: int32 magic 0x51435847 ('QCXG'), int32 F, int32 ncols.
// Features are computed on the acting player's masked view.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#include "qc_learn.h"

using namespace qc;

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: qc_gen P rounds threads seed worlds rollout(heur|weights) out.bin [rollout_temp]\n");
    return 2;
  }
  int P = std::atoi(argv[1]), rounds = std::atoi(argv[2]), threads = std::atoi(argv[3]);
  uint64_t seed = std::strtoull(argv[4], nullptr, 10);
  int worlds = std::atoi(argv[5]);
  std::string rspec = argv[6];
  const char* outp = argv[7];
  double rtemp = argc > 8 ? std::atof(argv[8]) : 0.0;
  LearnedPolicy pol;
  RolloutPolicy rp;
  rp.temp = rtemp;
  if (rspec != "heur") {
    if (!pol.load(rspec)) { std::fprintf(stderr, "cannot load %s\n", rspec.c_str()); return 2; }
    rp.pol = &pol;
  }
  FILE* fp = std::fopen(outp, "wb");
  if (!fp) { std::fprintf(stderr, "cannot open %s\n", outp); return 2; }
  const int ncols = 4 + NFEAT;
  int32_t hdr[3] = {0x51435847, NFEAT, ncols};
  std::fwrite(hdr, sizeof(hdr), 1, fp);
  std::mutex mu;
  std::atomic<int> next{0};
  std::atomic<long> ndec{0}, nrows{0};
  auto t0 = std::chrono::steady_clock::now();
  auto worker = [&]() {
    std::vector<float> buf;
    for (;;) {
      int g = next++;
      if (g >= rounds) break;
      Rng rng(seed * 1000003ULL + g);
      Round s;
      s.init(P, g % P);
      s.deal(rng);
      int dec = 0;
      while (!s.done()) {
        const int me = s.cur;
        Round view = mask_for(s, me);
        int acts[NC * MAXR + 1];
        int n = view.legal(acts);
        int a;
        if (n == 1) a = acts[0];
        else {
          double vals[NC * MAXR + 1];
          a = pimc_decide_rp(view, me, worlds, 0, rp, rng, vals);
          if (s.phase == PLAY) {
            PlayCtx x;
            play_context(view, x, n);
            for (int i = 0; i < n; ++i) {
              size_t off = buf.size();
              buf.resize(off + ncols);
              float* row = &buf[off];
              row[0] = (float)g; row[1] = (float)dec; row[2] = (float)vals[i];
              row[3] = (float)heuristic_move_score(view, acts[i], HeurParams());
              play_features(view, x, acts[i], row + 4);
            }
            ++dec; ++ndec; nrows += n;
          }
        }
        s.apply(a);
      }
      if (buf.size() > (1u << 20)) {
        std::lock_guard<std::mutex> lk(mu);
        std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
        buf.clear();
      }
      if ((g + 1) % 200 == 0) {
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "rounds %d decisions %ld rows %ld  %.0fs\n", g + 1, ndec.load(), nrows.load(), dt);
      }
    }
    std::lock_guard<std::mutex> lk(mu);
    std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  std::fclose(fp);
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("{\"rounds\":%d,\"decisions\":%ld,\"rows\":%ld,\"seconds\":%.1f,\"worlds\":%d,\"rollout\":\"%s\",\"rollout_temp\":%.3f}\n",
              rounds, ndec.load(), nrows.load(), dt, worlds, rspec.c_str(), rtemp);
  return 0;
}
