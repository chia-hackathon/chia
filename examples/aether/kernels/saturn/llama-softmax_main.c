// See LICENSE for license details.
//
//**************************************************************************
// llama-softmax — softmax over one 512-long fp32 attention score row
//--------------------------------------------------------------------------
// SEALED HARNESS. Every build is reassembled from the pristine copy of this
// file plus the optimizer's `llama_softmax.c`.
//
// The check is FULL: all 512 outputs against a scalar libm reference. The
// tolerance is 1e-4 relative, which the kernel's polynomial exp clears by
// three orders of magnitude but a wrong reduction does not.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_softmax.h"

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

#define REL_TOL 1e-4f
#define ABS_TOL 1e-9f
static inline float fabsf_(float a) { return a < 0.0f ? -a : a; }
static int close_enough(float got, float want) {
  float d = fabsf_(got - want);
  return d <= ABS_TOL + REL_TOL * fabsf_(want);
}
static inline uint32_t bits(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4); return u;
}

int main(int argc, char *argv[])
{
  float *x = LSM_X, *y = LSM_Y;

  printf("llama-softmax N = %d\n", LSM_N);

  // Attention scores after the 1/sqrt(d) scale land in roughly [-8, 8]; the
  // amplitude here (x4 of [-2,2)) reproduces that spread, so the max
  // subtraction is load-bearing.
  fill_f32(x, LSM_N, 4.0f);
  for (size_t i = 0; i < LSM_N; i++) y[i] = 0.0f;

  setStats(1);
  llama_softmax(LSM_N, x, y);
  setStats(0);

  // --- scalar golden model ------------------------------------------------
  float m = x[0];
  for (size_t i = 1; i < LSM_N; i++) if (x[i] > m) m = x[i];
  double s = 0.0;
  for (size_t i = 0; i < LSM_N; i++) s += (double)expf(x[i] - m);

  int bad = 0;
  for (size_t i = 0; i < LSM_N && bad < 8; i++) {
    float want = (float)((double)expf(x[i] - m) / s);
    if (!close_enough(y[i], want)) {
      printf("MISMATCH i=%d: got %x want %x\n",
             (int)i, (unsigned)bits(y[i]), (unsigned)bits(want));
      bad++;
    }
  }

  double sum = 0.0;
  for (size_t i = 0; i < LSM_N; i++) sum += (double)y[i];
  printf("y sum x1e6 = %ld (must be ~1000000)\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED\n"); return 1; }
  printf("PASSED (all %d elements)\n", LSM_N);
  return 0;
}
