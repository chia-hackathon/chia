// See LICENSE for license details.
//
//**************************************************************************
// llama-rope — fp32 RoPE for one Llama-3.2-1B decode step
//--------------------------------------------------------------------------
//
// SEALED HARNESS. main(), the data generator, the scalar golden model and the
// self-check. Every build is reassembled from the pristine copy of this file
// plus the optimizer's `llama_rope.c`.
//
// The objective (`setStats`) brackets BOTH calls: Q (32 heads) and K (8 heads)
// at the same position, i.e. the whole `rope` work of one decoder layer,
// 2560 fp32 elements.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_rope.h"

// --------------------------------------------------------------------------
// Deterministic pseudo-random data
// --------------------------------------------------------------------------
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
// Scalar golden model — ONE head, from the pristine copy.
// --------------------------------------------------------------------------
static void ref_head(const float *x0, float *out, size_t head_dim,
                     const float *cs, const float *sn) {
  size_t H = head_dim / 2;
  for (size_t i = 0; i < H; i++) {
    out[i]     = x0[i] * cs[i] - x0[i + H] * sn[i];
    out[i + H] = x0[i + H] * cs[i] + x0[i] * sn[i];
  }
}

int main(int argc, char *argv[])
{
  float *q = LRP_Q, *k = LRP_K;
  float *cs = LRP_COS, *sn = LRP_SIN;
  float *q0 = LRP_Q0, *k0 = LRP_K0;

  printf("llama-rope heads=%d/%d head_dim=%d pos=%d\n",
         LRP_Q_HEADS, LRP_KV_HEADS, LRP_HEAD_DIM, LRP_POS);

  fill_f32(q, LRP_Q_ELEMS, 1.0f);
  fill_f32(k, LRP_K_ELEMS, 1.0f);
  for (size_t i = 0; i < LRP_Q_ELEMS; i++) q0[i] = q[i];
  for (size_t i = 0; i < LRP_K_ELEMS; i++) k0[i] = k[i];

  // cos/sin table for position LRP_POS. inv_freq[i] = theta^(-2i/head_dim).
  const float logt = logf(LRP_THETA);
  for (int i = 0; i < LRP_HALF; i++) {
    float inv_freq = expf(-logt * (2.0f * (float)i) / (float)LRP_HEAD_DIM);
    float ang = (float)LRP_POS * inv_freq;
    cs[i] = cosf(ang);
    sn[i] = sinf(ang);
  }

  // mcycle/minstret around exactly these two calls is the objective.
  setStats(1);
  llama_rope(LRP_Q_HEADS,  LRP_HEAD_DIM, q, cs, sn);
  llama_rope(LRP_KV_HEADS, LRP_HEAD_DIM, k, cs, sn);
  setStats(0);

  // --- full scalar golden model + check ----------------------------------
  float ref[LRP_HEAD_DIM];
  int bad = 0;

  for (int h = 0; h < LRP_Q_HEADS && bad < 8; h++) {
    ref_head(q0 + h * LRP_HEAD_DIM, ref, LRP_HEAD_DIM, cs, sn);
    for (int i = 0; i < LRP_HEAD_DIM; i++) {
      float got = q[h * LRP_HEAD_DIM + i];
      if (!close_enough(got, ref[i])) {
        printf("MISMATCH q head %d i %d: got %x want %x\n",
               h, i, (unsigned)bits(got), (unsigned)bits(ref[i]));
        if (++bad >= 8) break;
      }
    }
  }
  for (int h = 0; h < LRP_KV_HEADS && bad < 8; h++) {
    ref_head(k0 + h * LRP_HEAD_DIM, ref, LRP_HEAD_DIM, cs, sn);
    for (int i = 0; i < LRP_HEAD_DIM; i++) {
      float got = k[h * LRP_HEAD_DIM + i];
      if (!close_enough(got, ref[i])) {
        printf("MISMATCH k head %d i %d: got %x want %x\n",
               h, i, (unsigned)bits(got), (unsigned)bits(ref[i]));
        if (++bad >= 8) break;
      }
    }
  }

  double sum = 0.0;
  for (size_t i = 0; i < LRP_Q_ELEMS; i++) sum += (double)q[i];
  for (size_t i = 0; i < LRP_K_ELEMS; i++) sum += (double)k[i];
  printf("qk checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED\n"); return 1; }
  printf("PASSED (all %d elements)\n", LRP_Q_ELEMS + LRP_K_ELEMS);
  return 0;
}
