// Parity check of the C++ big-model inference against the trainer: scores the
// first N decisions of a qc_gen state sidecar (state + rows file) and writes
// one float32 score per row to out.f32.
// usage: bm_score big_weights prefix N out.f32
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "qc_big.h"
using namespace qc;
int main(int argc, char** argv) {
  BigPolicy big;
  if (argc < 5 || !big.load(argv[1])) return 2;
  std::string pre = argv[2];
  int N = std::atoi(argv[3]);
  FILE* fs = std::fopen((pre + ".state.bin").c_str(), "rb");
  FILE* fr = std::fopen((pre + ".bin").c_str(), "rb");
  int32_t hs[3], hr[3];
  if (!fs || !fr || std::fread(hs, 4, 3, fs) != 3 || std::fread(hr, 4, 3, fr) != 3) return 2;
  FILE* fo = std::fopen(argv[4], "wb");
  std::vector<float> st(hs[2]), row(hr[2]);
  long rows = 0;
  for (int d = 0; d < N; ++d) {
    if (std::fread(st.data(), 4, hs[2], fs) != (size_t)hs[2]) break;
    const int P = (int)st[2], n = (int)st[4];
    Round dummy; dummy.init(P, 0);
    float u[512], x[NACT];
    big.embed(&st[5 + NC * MAXR + 1], u);
    for (int i = 0; i < n; ++i) {
      if (std::fread(row.data(), 4, hr[2], fr) != (size_t)hr[2]) return 3;
      if (row[0] != st[0] || row[1] != st[1]) { std::fprintf(stderr, "row/state mismatch at %d\n", d); return 3; }
      std::memcpy(x, &row[4], sizeof(float) * NFEAT);
      action_extra(dummy, (int)st[5 + i], x + NFEAT);
      float sc = big.score_action(u, x);
      std::fwrite(&sc, 4, 1, fo);
      ++rows;
    }
  }
  std::fclose(fo);
  std::printf("scored %ld rows\n", rows);
}
