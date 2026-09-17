// See LICENSE for license details.
//
//**************************************************************************
// llama-attn-scores-q8kv — one decode query head against a 512-long INT8 KV
//--------------------------------------------------------------------------
// SEALED HARNESS. Every build is reassembled from the pristine copy of this
// file plus the optimizer's `llama_attn_scores_q8kv.c`.
//
// The check is FULL: all 512 scores recomputed with a scalar dequantise +
// dot product. Because the reference dequantises in exactly the same order
// the tolerance can stay tight (1e-4 relative).

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_attn_scores_q8kv.h"

static uint32_t lcg_state = 0x13579BDFu;
static inline uint32_t lcg_next(void) {
  lcg_state = lcg_state * 1103515245u + 12345u;
  return lcg_state >> 16;
}
static inline float lcg_f32(void) {
  return ((float)(int32_t)(lcg_next() & 0xFFFFu) - 32768.0f) * (1.0f / 32768.0f);
}

// --- int8 fill: 256-byte seed block broadcast with a per-chunk offset, the
// --- same trick `llama-q8-gemv_main.c` uses to keep the fill off the scalar
// --- path (a 32 KiB scalar fill would be ~200k cycles).
#define I8_SEED 256
static int8_t seed_i8[I8_SEED];

static void fill_i8(int8_t *p, size_t n) {
  for (int i = 0; i < I8_SEED; i++) seed_i8[i] = (int8_t)lcg_next();
  const size_t vl = __riscv_vsetvl_e8m8(I8_SEED);   // VLEN=256, LMUL=8 -> 256
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
#define ABS_TOL 1e-5f
static inline float fabsf_(float a) { return a < 0.0f ? -a : a; }
static int close_enough(float got, float want) {
  float d = fabsf_(got - want);
  return d <= ABS_TOL + REL_TOL * fabsf_(want);
}
static inline uint32_t bits(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4); return u;
}

static float ref_score(const int8_t *K8, const float *ks, const float *q,
                       size_t d, size_t s) {
  const int8_t *kr = K8 + s * d;
  float acc = 0.0f;
  for (size_t j = 0; j < d; j++) acc += (float)kr[j] * q[j];
  return acc * ks[s] * LQS_SCALE;
}

int main(int argc, char *argv[])
{
  int8_t *K8 = LQS_K8;
  float  *ks = LQS_KS, *q = LQS_Q, *o = LQS_O;

  printf("llama-attn-scores-q8kv S,d = %d,%d (int8 KV, per-row scale)\n",
         LQS_S, LQS_D);

  fill_i8(K8, (size_t)LQS_S * LQS_D);
  // Dequant scales: a realistic per-row symmetric scale is |max|/127, i.e.
  // O(1e-2) for activations of unit scale. Kept strictly positive.
  fill_f32(ks, LQS_S, 1.0f);
  for (size_t s = 0; s < LQS_S; s++) ks[s] = 0.005f + 0.004f * fabsf_(ks[s]);
  fill_f32(q, LQS_D, 1.0f);
  for (size_t s = 0; s < LQS_S; s++) o[s] = 0.0f;

  setStats(1);
  llama_attn_scores_q8kv(LQS_S, LQS_D, K8, ks, q, o, LQS_SCALE);
  setStats(0);

  int bad = 0;
  for (size_t s = 0; s < LQS_S && bad < 8; s++) {
    float want = ref_score(K8, ks, q, LQS_D, s);
    if (!close_enough(o[s], want)) {
      printf("MISMATCH s=%d: got %x want %x\n",
             (int)s, (unsigned)bits(o[s]), (unsigned)bits(want));
      bad++;
    }
  }

  double sum = 0.0;
  for (size_t s = 0; s < LQS_S; s++) sum += (double)o[s];
  printf("scores checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED\n"); return 1; }
  printf("PASSED (all %d scores)\n", LQS_S);
  return 0;
}
