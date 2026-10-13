// Full-match arena: P rounds, rotating start seat, placement by total score
// (final-round score tiebreak), common deals rotated through every seat.
//
// usage: qc_arena P deals threads seed botA botB
//   candidate botA sits at one seat, botB fills the rest; every deal is played
//   P times (candidate at each seat) and the reference lineup (all botB) is
//   played on the same deals so the paired placement difference is reported.
// bot spec: random | heur | pimc:<worlds>[:util]
//   | lin:<weights>[:temp]           learned PLAY policy (greedy unless temp>0); heuristic discard/bid
//   | pimcl:<worlds>:<weights>[:temp] PIMC whose rollouts use the learned play policy for all seats
//   | ismcts:<mult>:<cpuct>:<opp>:<weights>[:prior_temp]  SO-ISMCTS (qc_search.h)
//   | pimc2:<worlds>:<inner>:<weights>   PIMC whose rollouts make the actor's next decision by inner PIMC
//   | pimce:<worlds>:<k>:<weights>       PIMC with exact max-n over the last k tricks of each world
#include <atomic>
#include <chrono>
#include <ctime>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "qc_bots.h"

using namespace qc;

struct MatchResult {
  double totals[MAXP], last[MAXP];
  int paradox[MAXP], bidhit[MAXP];
};

MatchResult play_match(int P, const Bot* seats[], uint64_t deal_seed, uint64_t play_seed, int initial_start) {
  MatchResult m{};
  Rng prng(play_seed);
  for (int rd = 0; rd < P; ++rd) {
    Round s;
    s.init(P, (initial_start + rd) % P);
    Rng drng(deal_seed * 1000003ULL + rd);
    s.deal(drng);
    while (!s.done()) s.apply(seats[s.cur]->act(s, prng));
    double sc[MAXP];
    s.scores(sc);
    for (int p = 0; p < P; ++p) {
      m.totals[p] += sc[p];
      m.last[p] = sc[p];
      m.paradox[p] += s.paradox == p;
      m.bidhit[p] += (s.paradox != p && s.bid[p] == s.tricks[p]);
    }
  }
  return m;
}

// pairwise placement of seat i: fraction of others it beats (ties 0.5)
double pairwise(const MatchResult& m, int P, int i) {
  double w = 0;
  for (int j = 0; j < P; ++j) {
    if (j == i) continue;
    if (m.totals[i] > m.totals[j] || (m.totals[i] == m.totals[j] && m.last[i] > m.last[j])) w += 1;
    else if (m.totals[i] == m.totals[j] && m.last[i] == m.last[j]) w += 0.5;
  }
  return w / (P - 1);
}

int main(int argc, char** argv) {
  if (argc < 7) { std::fprintf(stderr, "usage: qc_arena P deals threads seed botA botB\n"); return 2; }
  int P = std::atoi(argv[1]), deals = std::atoi(argv[2]), threads = std::atoi(argv[3]);
  uint64_t seed = std::strtoull(argv[4], nullptr, 10);
  Bot A = parse_bot(argv[5]), B = parse_bot(argv[6]);
  // per deal: candidate placement averaged over seats, and reference (all B) per seat
  std::vector<double> cand(deals), ref(deals), cscore(deals), rscore(deals);
  std::vector<int> cpar(deals), rpar(deals), chit(deals), rhit(deals), cwin(deals), rwin(deals);
  std::atomic<int> next{0};
  auto wall0 = std::chrono::steady_clock::now();
  std::clock_t cpu0 = std::clock();
  auto worker = [&]() {
    for (;;) {
      int d = next++;
      if (d >= deals) return;
      uint64_t ds = seed * 7919 + d;
      double cp = 0, rp = 0, cs = 0, rs = 0;
      int cpa = 0, rpa = 0, ch = 0, rh = 0, cw = 0, rw = 0;
      const Bot* refseats[MAXP];
      for (int p = 0; p < P; ++p) refseats[p] = &B;
      MatchResult rm = play_match(P, refseats, ds, ds ^ 0xabcdefULL, d % P);
      for (int seat = 0; seat < P; ++seat) {
        const Bot* seats[MAXP];
        for (int p = 0; p < P; ++p) seats[p] = (p == seat) ? &A : &B;
        MatchResult m = play_match(P, seats, ds, ds ^ (0x1234567ULL + seat), d % P);
        cp += pairwise(m, P, seat); rp += pairwise(rm, P, seat);
        cs += m.totals[seat]; rs += rm.totals[seat];
        cpa += m.paradox[seat]; rpa += rm.paradox[seat];
        ch += m.bidhit[seat]; rh += rm.bidhit[seat];
        bool w = true, wr = true;
        for (int j = 0; j < P; ++j) if (j != seat) {
          if (pairwise(m, P, seat) < 1.0) w = false;
          if (pairwise(rm, P, seat) < 1.0) wr = false;
        }
        cw += w; rw += wr;
      }
      cand[d] = cp / P; ref[d] = rp / P; cscore[d] = cs / P; rscore[d] = rs / P;
      cpar[d] = cpa; rpar[d] = rpa; chit[d] = ch; rhit[d] = rh; cwin[d] = cw; rwin[d] = rw;
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  double cpu = (double)(std::clock() - cpu0) / CLOCKS_PER_SEC;
  double mc = 0, mr = 0, md = 0, sd = 0, sc = 0, sr = 0;
  long tp = 0, rp = 0, th = 0, rh = 0, tw = 0, rw = 0;
  for (int d = 0; d < deals; ++d) {
    mc += cand[d]; mr += ref[d]; md += cand[d] - ref[d]; sc += cscore[d]; sr += rscore[d];
    tp += cpar[d]; rp += rpar[d]; th += chit[d]; rh += rhit[d]; tw += cwin[d]; rw += rwin[d];
  }
  mc /= deals; mr /= deals; md /= deals;
  for (int d = 0; d < deals; ++d) { double x = cand[d] - ref[d] - md; sd += x * x; }
  sd = std::sqrt(sd / std::max(1, deals - 1));
  double se = sd / std::sqrt((double)deals);
  long rounds = (long)deals * P * P;
  std::printf("{\"P\":%d,\"deals\":%d,\"A\":\"%s\",\"B\":\"%s\",\"placement\":%.4f,\"ref_placement\":%.4f,"
              "\"diff\":%.4f,\"diff_ci95\":[%.4f,%.4f],\"score\":%.3f,\"ref_score\":%.3f,"
              "\"paradox_rate\":%.4f,\"ref_paradox_rate\":%.4f,\"bidhit_rate\":%.4f,\"ref_bidhit_rate\":%.4f,"
              "\"first_place\":%.4f,\"ref_first_place\":%.4f,\"wall_s\":%.1f,\"cpu_s\":%.1f}\n",
              P, deals, argv[5], argv[6], mc, mr, md, md - 1.96 * se, md + 1.96 * se, sc / deals, sr / deals,
              (double)tp / rounds, (double)rp / rounds, (double)th / rounds, (double)rh / rounds,
              (double)tw / (deals * P), (double)rw / (deals * P), wall, cpu);
  return 0;
}
