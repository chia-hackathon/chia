// See LICENSE for license details.
//
//**************************************************************************
// llama-add — fp32 residual add, hidden_size = 2048
//--------------------------------------------------------------------------
// SEALED HARNESS. Every build is reassembled from the pristine copy of this
// file plus the optimizer's `llama_add.c`.
//
// The check is FULL and EXACT: a single fp32 add per element has exactly one
// correct answer, so no tolerance is needed or allowed.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_add.h"

static uint32_t lcg_state = 0x13579BDFu;
static inline uint32_t lcg_next(void) {
  lcg_state = lcg_state * 1103515245u + 12345u;
  return lcg_state >> 16;
}
static inline float lcg_f32(void) {
  return ((float)(int32_t)(lcg_next() & 0xFFFFu) - 32768.0f) * (1.0f / 32768.0f);
}

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

static inline uint32_t bits(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4); return u;
}

int main(int argc, char *argv[])
{
  float *x = LAD_X, *y = LAD_Y, *x0 = LAD_X0;

  printf("llama-add N = %d\n", LAD_N);

  fill_f32(x, LAD_N, 1.0f);
  fill_f32(y, LAD_N, 1.0f);
  for (size_t i = 0; i < LAD_N; i++) x0[i] = x[i];

  setStats(1);
  llama_add(LAD_N, x, y);
  setStats(0);

  int bad = 0;
  for (size_t i = 0; i < LAD_N && bad < 8; i++) {
    float want = x0[i] + y[i];
    if (bits(x[i]) != bits(want)) {
      printf("MISMATCH i=%d: got %x want %x\n",
             (int)i, (unsigned)bits(x[i]), (unsigned)bits(want));
      bad++;
    }
  }

  double sum = 0.0;
  for (size_t i = 0; i < LAD_N; i++) sum += (double)x[i];
  printf("x checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED\n"); return 1; }
  printf("PASSED (all %d elements, bit-exact)\n", LAD_N);
  return 0;
}
