// See LICENSE for license details.
//
//**************************************************************************
// llama-attn-scores — one decode query head against a 512-long fp32 KV cache
//--------------------------------------------------------------------------
// SEALED HARNESS. Every build is reassembled from the pristine copy of this
// file plus the optimizer's `llama_attn_scores.c`.
//
// The check is FULL (all 512 scores recomputed with a scalar dot product,
// 32,768 MACs ~ 150k cycles ~ 90 s of Verilator) — affordable at this size.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_attn_scores.h"

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
#define ABS_TOL 1e-5f
static inline float fabsf_(float a) { return a < 0.0f ? -a : a; }
static int close_enough(float got, float want) {
  float d = fabsf_(got - want);
  return d <= ABS_TOL + REL_TOL * fabsf_(want);
}
static inline uint32_t bits(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4); return u;
}

static float ref_score(const float *K, const float *q, size_t d, size_t s) {
  const float *kr = K + s * d;
  float acc = 0.0f;
  for (size_t j = 0; j < d; j++) acc += kr[j] * q[j];
  return acc * LAS_SCALE;
}

int main(int argc, char *argv[])
{
  float *K = LAS_K, *q = LAS_Q, *o = LAS_O;

  printf("llama-attn-scores S,d = %d,%d\n", LAS_S, LAS_D);

  fill_f32(K, (size_t)LAS_S * LAS_D, 1.0f);
  fill_f32(q, LAS_D, 1.0f);
  for (size_t s = 0; s < LAS_S; s++) o[s] = 0.0f;

  setStats(1);
  llama_attn_scores(LAS_S, LAS_D, K, q, o, LAS_SCALE);
  setStats(0);

  int bad = 0;
  for (size_t s = 0; s < LAS_S && bad < 8; s++) {
    float want = ref_score(K, q, LAS_D, s);
    if (!close_enough(o[s], want)) {
      printf("MISMATCH s=%d: got %x want %x\n",
             (int)s, (unsigned)bits(o[s]), (unsigned)bits(want));
      bad++;
    }
  }

  double sum = 0.0;
  for (size_t s = 0; s < LAS_S; s++) sum += (double)o[s];
  printf("scores checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED\n"); return 1; }
  printf("PASSED (all %d scores)\n", LAS_S);
  return 0;
}
