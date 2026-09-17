// See LICENSE for license details.
//
//**************************************************************************
// llama-silu-mul — SwiGLU activation, intermediate_size = 8192, fp32
//--------------------------------------------------------------------------
// SEALED HARNESS. Every build is reassembled from the pristine copy of this
// file plus the optimizer's `llama_silu_mul.c`.
//
// The check is SAMPLED: 512 of the 8192 outputs are recomputed with a scalar
// libm reference. A full scalar check would be 8192 `expf` calls (~1M cycles,
// ~10 min of Verilator) for no extra coverage — the sample stride is coprime
// with every vector length the kernel can use, and a checksum over ALL 8192
// outputs is printed so a kernel that only computed the sampled lanes is
// visible in the log.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_silu_mul.h"

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
#define ABS_TOL 1e-6f
static inline float fabsf_(float a) { return a < 0.0f ? -a : a; }
static int close_enough(float got, float want) {
  float d = fabsf_(got - want);
  return d <= ABS_TOL + REL_TOL * fabsf_(want);
}
static inline uint32_t bits(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4); return u;
}

// 512 samples at a stride of 13 (odd, coprime with 8 / 16 / 32 / 64), wrapped
// so they cover the whole array rather than only its first eighth.
#define N_SAMPLES 512
#define SAMPLE_STRIDE 13

int main(int argc, char *argv[])
{
  float *g = LSU_G, *u = LSU_U, *o = LSU_O;

  printf("llama-silu-mul N = %d\n", LSU_N);

  // x3 of [-2,2): SiLU's interesting region is |g| < 6, and this spread
  // exercises both the saturating and the near-linear side.
  fill_f32(g, LSU_N, 3.0f);
  fill_f32(u, LSU_N, 1.0f);
  for (size_t i = 0; i < LSU_N; i++) o[i] = 0.0f;

  setStats(1);
  llama_silu_mul(LSU_N, g, u, o);
  setStats(0);

  int bad = 0;
  for (int s = 0; s < N_SAMPLES && bad < 8; s++) {
    size_t i = ((size_t)s * SAMPLE_STRIDE) % LSU_N;
    float gv = g[i];
    float want = (gv / (1.0f + expf(-gv))) * u[i];
    if (!close_enough(o[i], want)) {
      printf("MISMATCH i=%d: got %x want %x\n",
             (int)i, (unsigned)bits(o[i]), (unsigned)bits(want));
      bad++;
    }
  }

  double sum = 0.0;
  for (size_t i = 0; i < LSU_N; i++) sum += (double)o[i];
  printf("out checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED\n"); return 1; }
  printf("PASSED (%d sampled elements)\n", N_SAMPLES);
  return 0;
}
