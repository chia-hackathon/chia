// See LICENSE for license details.
//
//**************************************************************************
// llama-attn-pv-q8kv — probs @ INT8 V for one decode query head, S=512, d=64
//--------------------------------------------------------------------------
// SEALED HARNESS. Every build is reassembled from the pristine copy of this
// file plus the optimizer's `llama_attn_pv_q8kv.c`.
//
// The check is FULL: all 64 outputs recomputed with a scalar dequantise +
// reduction over the 512 rows. `P` is a real softmax output (non-negative,
// sums to 1), so the reference sum is well conditioned.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_attn_pv_q8kv.h"

static uint32_t lcg_state = 0x13579BDFu;
static inline uint32_t lcg_next(void) {
  lcg_state = lcg_state * 1103515245u + 12345u;
  return lcg_state >> 16;
}
static inline float lcg_f32(void) {
  return ((float)(int32_t)(lcg_next() & 0xFFFFu) - 32768.0f) * (1.0f / 32768.0f);
}

#define I8_SEED 256
static int8_t seed_i8[I8_SEED];

static void fill_i8(int8_t *p, size_t n) {
  for (int i = 0; i < I8_SEED; i++) seed_i8[i] = (int8_t)lcg_next();
  const size_t vl = __riscv_vsetvl_e8m8(I8_SEED);
  vint8m8_t v = __riscv_vle8_v_i8m8(seed_i8, vl);
  for (size_t off = 0; off < n; off += vl) {
    size_t k = (n - off) < vl ? (n - off) : vl;
    __riscv_vse8_v_i8m8(p + off, v, k);
    v = __riscv_vadd_vx_i8m8(v, (int8_t)lcg_next(), vl);
  }
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

int main(int argc, char *argv[])
{
  float  *P = LQP_P, *vs = LQP_VS, *o = LQP_O;
  int8_t *V8 = LQP_V8;

  printf("llama-attn-pv-q8kv S,d = %d,%d (int8 V, per-row scale)\n",
         LQP_S, LQP_D);

  fill_f32(P, LQP_S, 1.0f);
  fill_i8(V8, (size_t)LQP_S * LQP_D);
  fill_f32(vs, LQP_S, 1.0f);
  for (size_t s = 0; s < LQP_S; s++) vs[s] = 0.005f + 0.004f * fabsf_(vs[s]);

  // Turn P into a real probability vector.
  float psum = 0.0f;
  for (size_t s = 0; s < LQP_S; s++) { P[s] = expf(P[s]); psum += P[s]; }
  for (size_t s = 0; s < LQP_S; s++) P[s] /= psum;

  for (size_t j = 0; j < LQP_D; j++) o[j] = 0.0f;

  setStats(1);
  llama_attn_pv_q8kv(LQP_S, LQP_D, P, V8, vs, o);
  setStats(0);

  int bad = 0;
  for (size_t j = 0; j < LQP_D; j++) {
    float want = 0.0f;
    for (size_t s = 0; s < LQP_S; s++)
      want += P[s] * vs[s] * (float)V8[s * LQP_D + j];
    if (!close_enough(o[j], want)) {
      printf("MISMATCH j=%d: got %x want %x\n",
             (int)j, (unsigned)bits(o[j]), (unsigned)bits(want));
      if (++bad >= 8) break;
    }
  }

  double sum = 0.0;
  for (size_t j = 0; j < LQP_D; j++) sum += (double)o[j];
  printf("out checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED\n"); return 1; }
  printf("PASSED (all %d outputs)\n", LQP_D);
  return 0;
}
