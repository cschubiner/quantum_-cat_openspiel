// Bot specs shared by qc_arena and bench_deepsearch.
//   random | heur | pimc:<worlds>[:util]
//   | lin:<weights>[:temp]                      learned PLAY policy; heuristic discard/bid
//   | pimcl:<worlds>:<weights>[:temp]           PIMC, learned-policy rollouts (champion v4 = pimcl:128:weights/g3_mlp32.txt)
//   | ismcts:<mult>:<cpuct>:<opp>:<weights>[:prior_temp]   SO-ISMCTS, mult*#legal iterations
//   | pimc2:<worlds>:<inner>:<weights>          PIMC, actor's next decision in each rollout by inner PIMC
//   | pimce:<worlds>:<k>:<weights>              PIMC, exact max-n over the last k tricks
#pragma once
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "qc_search.h"

namespace qc {

struct Bot {
  int type = 0;  // 0 random 1 heur 2 pimc 3 lin 4 pimcl 5 ismcts 6 pimc2 7 pimce
  SearchParams sp;
  IsmctsParams ip;
  int inner = 8, k = 1;
  std::shared_ptr<LearnedPolicy> pol;
  RolloutPolicy rp;
  // Decide on an already-masked view (the only entry point that searches).
  int decide(const Round& view, int me, Rng& rng) const {
    int acts[NC * MAXR + 1];
    int n = view.legal(acts);
    if (n == 1) return acts[0];
    switch (type) {
      case 0: return acts[rng.below(n)];
      case 1: return heuristic_action(view, rng, sp.rollout);
      case 3: return rp.act(view, rng);
      case 4: return pimc_decide_rp(view, me, sp.worlds, sp.util, rp, rng);
      case 5: return ismcts_decide(view, me, ip, rp, rng);
      case 6: return pimc2_decide(view, me, sp.worlds, inner, rp, rng);
      case 7: return pimce_decide(view, me, sp.worlds, k, rp, rng);
      default: return pimc_decide(view, me, sp, rng);
    }
  }
  int act(const Round& truth, Rng& rng) const {
    const int me = truth.cur;
    Round view = mask_for(truth, me);  // agent sees only its information
    return decide(view, me, rng);
  }
};

inline std::vector<std::string> split_spec(const std::string& spec) {
  std::vector<std::string> parts;
  size_t st = 0;
  for (;;) {
    size_t e = spec.find(':', st);
    parts.push_back(spec.substr(st, e == std::string::npos ? std::string::npos : e - st));
    if (e == std::string::npos) break;
    st = e + 1;
  }
  return parts;
}

inline Bot parse_bot(const std::string& spec) {
  Bot b;
  auto parts = split_spec(spec);
  auto need = [&](size_t k) {
    if (parts.size() < k) { std::fprintf(stderr, "bad bot %s\n", spec.c_str()); std::exit(2); }
  };
  auto load = [&](const std::string& path) {
    b.pol = std::make_shared<LearnedPolicy>();
    if (!b.pol->load(path)) { std::fprintf(stderr, "cannot load weights %s\n", path.c_str()); std::exit(2); }
    b.rp.pol = b.pol.get();
  };
  const std::string& t = parts[0];
  if (spec == "random") b.type = 0;
  else if (spec == "heur") b.type = 1;
  else if (t == "lin" || t == "pimcl") {
    size_t wi;
    if (t == "lin") { b.type = 3; wi = 1; need(2); }
    else { b.type = 4; wi = 2; need(3); b.sp.worlds = std::atoi(parts[1].c_str()); }
    load(parts[wi]);
    b.rp.temp = parts.size() > wi + 1 ? std::atof(parts[wi + 1].c_str()) : 0.0;
  } else if (t == "ismcts") {
    need(5);
    b.type = 5;
    b.ip.mult = std::atoi(parts[1].c_str());
    b.ip.cpuct = std::atof(parts[2].c_str());
    b.ip.opp_mode = std::atoi(parts[3].c_str());
    load(parts[4]);
    if (parts.size() > 5) b.ip.prior_temp = std::atof(parts[5].c_str());
  } else if (t == "pimc2" || t == "pimce") {
    need(4);
    b.type = t == "pimc2" ? 6 : 7;
    b.sp.worlds = std::atoi(parts[1].c_str());
    (t == "pimc2" ? b.inner : b.k) = std::atoi(parts[2].c_str());
    load(parts[3]);
  } else if (t == "pimc") {
    b.type = 2;
    int w = 64, u = 0;
    std::sscanf(spec.c_str(), "pimc:%d:%d", &w, &u);
    b.sp.worlds = w; b.sp.util = u;
  } else { std::fprintf(stderr, "bad bot %s\n", spec.c_str()); std::exit(2); }
  std::fprintf(stderr, "bot %s -> type %d worlds %d temp %.3f weights %s\n", spec.c_str(), b.type, b.sp.worlds,
               b.rp.temp, b.pol ? (b.pol->kind ? "mlp" : "linear") : "-");
  return b;
}

}  // namespace qc
