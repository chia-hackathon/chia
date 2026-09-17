// See LICENSE for license details.
//
//**************************************************************************
// llama-rmsnorm — fp32 RMSNorm in the shape of a Llama-3.2-1B decoder layer
//--------------------------------------------------------------------------
//
// SEALED HARNESS. main(), the pseudo-random data generator, the scalar golden
// model and the self-check. Every build is reassembled from the pristine copy
// of this file plus the optimizer's `llama_rmsnorm.c`.
//
// Wall-clock discipline (Verilator runs this SoC at ~1.8k cycles/s):
//   * no big static arrays  -> the TSI ELF load stays ~1 s
//   * vector data fill      -> the 24 KiB of buffers cost ~1k cycles
//   * FULL scalar check     -> only 2048 elements, ~20k cycles, affordable

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_rmsnorm.h"

// --------------------------------------------------------------------------
// Deterministic pseudo-random data (fixed seed -> reproducible every run)
// --------------------------------------------------------------------------
static uint32_t lcg_state = 0x13579BDFu;

static inline uint32_t lcg_next(void) {
  lcg_state = lcg_state * 1103515245u + 12345u;
  return lcg_state >> 16;
}

// uniform in [-1, 1)
static inline float lcg_f32(void) {
  return ((float)(int32_t)(lcg_next() & 0xFFFFu) - 32768.0f) * (1.0f / 32768.0f);
}

// 64 floats = one e32/LMUL=8 vector at VLEN=256. Small enough to cost nothing
// in .bss; re-rolled every REFRESH chunks so the data is not one repeated tile.
#define SEED_ELEMS 64
#define REFRESH    16
static float seed_block[SEED_ELEMS];

static void fill_f32(float *p, size_t n, float amp) {
  const size_t vl = __riscv_vsetvlmax_e32m8();
  size_t chunk = 0;
  vfloat32m8_t v0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
  for (size_t off = 0; off < n; off += vl, chunk++) {
    if ((chunk % REFRESH) == 0) {
      for (int i = 0; i < SEED_ELEMS; i++) seed_block[i] = lcg_f32();
      v0 = __riscv_vle32_v_f32m8(seed_block, vl);
    }
    size_t k = (n - off) < vl ? (n - off) : vl;
    vfloat32m8_t v = __riscv_vfadd_vf_f32m8(v0, lcg_f32(), vl);
    v = __riscv_vfmul_vf_f32m8(v, amp, vl);
    __riscv_vse32_v_f32m8(p + off, v, k);
  }
}

// --------------------------------------------------------------------------
// Tolerant float compare. The kernel is free to reassociate its reduction, so
// an exact match is not required — 1e-4 relative is far tighter than any
// plausible wrong answer and far looser than fp32 reassociation noise.
// --------------------------------------------------------------------------
#define REL_TOL 1e-4f
#define ABS_TOL 1e-5f

static inline float fabsf_(float a) { return a < 0.0f ? -a : a; }

static int close_enough(float got, float want) {
  float d = fabsf_(got - want);
  return d <= ABS_TOL + REL_TOL * fabsf_(want);
}

static inline uint32_t bits(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4); return u;
}

// --------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  float *x = LRN_X;
  float *w = LRN_W;
  float *y = LRN_Y;

  printf("llama-rmsnorm N = %d\n", LRN_N);

  fill_f32(x, LRN_N, 1.0f);
  fill_f32(w, LRN_N, 1.0f);
  for (size_t i = 0; i < LRN_N; i++) y[i] = 0.0f;

  // mcycle/minstret around exactly this call is the objective.
  setStats(1);
  llama_rmsnorm(LRN_N, x, w, y, LRN_EPS);
  setStats(0);

  // --- full scalar golden model + check ----------------------------------
  double ss = 0.0;
  for (size_t i = 0; i < LRN_N; i++) ss += (double)x[i] * (double)x[i];
  float scale = 1.0f / __builtin_sqrtf((float)(ss / (double)LRN_N) + LRN_EPS);

  int bad = 0;
  for (size_t i = 0; i < LRN_N && bad < 8; i++) {
    float want = x[i] * scale * w[i];
    if (!close_enough(y[i], want)) {
      printf("MISMATCH i=%d: got %x want %x\n",
             (int)i, (unsigned)bits(y[i]), (unsigned)bits(want));
      bad++;
    }
  }

  double sum = 0.0;
  for (size_t i = 0; i < LRN_N; i++) sum += (double)y[i];
  printf("y checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED\n"); return 1; }
  printf("PASSED (all %d elements)\n", LRN_N);
  return 0;
}
