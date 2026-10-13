// PIMC self-play data generator for the learned PLAY-phase rollout policy.
//
// usage: qc_gen P rounds threads seed worlds rollout out.bin [rollout_temp]
//   rollout: "heur" or a weights file (learned play in rollouts for all seats)
//   Creating the file <out.bin>.stop makes workers finish their current round and exit.
// Every seat is the same PIMC bot (all phases).  At each PLAY decision with >=2
// legal actions we record, for every legal action, one float32 row:
//   [game_id, decision_id, q (PIMC mean utility), heur_score, phi_0..phi_{F-1}]
// File header: int32 magic 0x51435847 ('QCXG'), int32 F, int32 ncols.
// Features are computed on the acting player's masked view.
// Optional 9th argument <state.bin>: per recorded decision one float32 row
//   [game_id, decision_id, P, outcome, n, action_0..action_{36} (-1 pad), state_0..state_{NSTATE-1}]
// (outcome = the actor's final round utility, score - mean(others)); header
// int32 magic 0x51435853 ('QCXS'), int32 NSTATE, int32 ncols.  Rows of the main
// file for a decision are in the same order as action_0..action_{n-1}.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

#include "qc_big.h"

using namespace qc;

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: qc_gen P rounds threads seed worlds rollout(heur|weights) out.bin [rollout_temp] [state.bin]\n");
    return 2;
  }
  int P = std::atoi(argv[1]), rounds = std::atoi(argv[2]), threads = std::atoi(argv[3]);
  uint64_t seed = std::strtoull(argv[4], nullptr, 10);
  int worlds = std::atoi(argv[5]);
  std::string rspec = argv[6];
  const char* outp = argv[7];
  double rtemp = argc > 8 ? std::atof(argv[8]) : 0.0;
  const char* statep = argc > 9 ? argv[9] : nullptr;
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
  FILE* sfp = nullptr;
  const int scols = 5 + NC * MAXR + 1 + NSTATE;
  if (statep) {
    sfp = std::fopen(statep, "wb");
    if (!sfp) { std::fprintf(stderr, "cannot open %s\n", statep); return 2; }
    int32_t sh[3] = {0x51435853, NSTATE, scols};
    std::fwrite(sh, sizeof(sh), 1, sfp);
  }
  std::mutex mu;
  std::atomic<int> next{0};
  std::atomic<long> ndec{0}, nrows{0};
  auto t0 = std::chrono::steady_clock::now();
  const std::string stopf = std::string(outp) + ".stop";
  std::atomic<int> done_rounds{0};
  auto worker = [&]() {
    std::vector<float> buf, sbuf;
    for (;;) {
      if (access(stopf.c_str(), F_OK) == 0) break;
      int g = next++;
      if (g >= rounds) break;
      Rng rng(seed * 1000003ULL + g);
      Round s;
      s.init(P, g % P);
      s.deal(rng);
      int dec = 0;
      std::vector<std::pair<size_t, int>> pend;  // (sbuf offset, actor) awaiting the outcome
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
              { HeurParams hp0; hp0.w_token = 1.2; HeurCtx cx; heur_ctx(view, cx); row[3] = (float)heuristic_move_score(view, cx, acts[i], hp0); }
              play_features(view, x, acts[i], row + 4);
            }
            if (sfp) {
              size_t off = sbuf.size();
              sbuf.resize(off + scols);
              float* row = &sbuf[off];
              row[0] = (float)g; row[1] = (float)dec; row[2] = (float)P; row[3] = 0; row[4] = (float)n;
              for (int i = 0; i < NC * MAXR + 1; ++i) row[5 + i] = i < n ? (float)acts[i] : -1.f;
              state_features(view, row + 5 + NC * MAXR + 1);
              pend.emplace_back(off, me);
            }
            ++dec; ++ndec; nrows += n;
          }
        }
        s.apply(a);
      }
      for (auto& pr : pend) sbuf[pr.first + 3] = (float)utility(s, pr.second, 0);
      if (buf.size() > (1u << 20)) {
        std::lock_guard<std::mutex> lk(mu);
        std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
        buf.clear();
        if (sfp) { std::fwrite(sbuf.data(), sizeof(float), sbuf.size(), sfp); sbuf.clear(); }
      }
      int dr = ++done_rounds;
      if (dr % 200 == 0) {
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "rounds %d decisions %ld rows %ld  %.0fs\n", dr, ndec.load(), nrows.load(), dt);
      }
    }
    std::lock_guard<std::mutex> lk(mu);
    std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
    if (sfp) std::fwrite(sbuf.data(), sizeof(float), sbuf.size(), sfp);
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  std::fclose(fp);
  if (sfp) std::fclose(sfp);
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("{\"rounds\":%d,\"decisions\":%ld,\"rows\":%ld,\"seconds\":%.1f,\"worlds\":%d,\"rollout\":\"%s\",\"rollout_temp\":%.3f}\n",
              done_rounds.load(), ndec.load(), nrows.load(), dt, worlds, rspec.c_str(), rtemp);
  return 0;
}
