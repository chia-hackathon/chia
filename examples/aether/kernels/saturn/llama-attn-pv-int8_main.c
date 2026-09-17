// See LICENSE for license details.
//
//**************************************************************************
// llama-attn-pv-int8 — probs@V for one decode head, INT8 V AND INT8 probs
//--------------------------------------------------------------------------
// SEALED HARNESS. Every build is reassembled from the pristine copy of this
// file plus the optimizer's `llama_attn_pv_int8.c`.
//
// TWO checks (see llama-attn-scores-int8_main.c for the full rationale):
//
//   GATE A (tight, 1e-4 relative) — the KERNEL check. A scalar int32
//     accumulation over exactly the same W8/V8 bytes, then the same rescale.
//     The integer sum has ONE correct answer, so this is as strict as the
//     fp32 benchmark and it is what scores the attempt.
//
//   GATE B (relative L2) — the QUANTISATION check, against a true fp32
//     reference computed from the original fp32 P and V.
//
// QUANTISATION ERROR BUDGET (why 3e-2 is the right gate here)
//   Two independent sources.
//   (1) V, per-row symmetric int8: RMS round-off max|v|/(254*sqrt(3)); for
//       roughly-uniform data max|v| ~ 2*rms|v|, so ~0.45% per element, and it
//       stays ~0.45% relative on a 512-term sum because signal and noise both
//       grow as sqrt(512).
//   (2) w = P*vscale, ONE scale for all 512 values. Here the spread matters:
//       the error of out[j] is ~sqrt(S)*(wmax/440)*rms(V) against a signal of
//       ~||w||_2 * rms(V) (V has random signs, so the sum does not grow
//       linearly), giving a relative error of (wmax/rms(w))/440. A softmax
//       over 512 logits spread across ~4 nats has wmax/rms(w) ~ 3-4, so this
//       term is ~0.7-0.9% — the DOMINANT one, and the reason a single-scale
//       probability quantisation is the risky half of this scheme.
//   Combined in quadrature: ~1%. A 3e-2 gate leaves ~3x of margin while still
//   failing loudly on a broken scheme (wrong fold, saturation, lost sign),
//   which lands at 10% or worse. The measured value is PRINTED every run, so
//   the margin is never a guess.
//
//   NOTE the asymmetry worth remembering: for `attn_scores` the per-row scale
//   factors out of the reduction exactly and costs NOTHING in accuracy; for
//   `attn_pv` the reduction is across rows, so the per-row V scale has to be
//   folded into P and re-quantised with a single scale, and THAT fold is where
//   this kernel's error actually comes from.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_attn_pv_int8.h"

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

static inline float fabsf_(float a) { return a < 0.0f ? -a : a; }
static inline uint32_t bits(float f) {
  uint32_t u; __builtin_memcpy(&u, &f, 4); return u;
}

static float quant_vec(const float *src, int8_t *dst, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; i++) { float a = fabsf_(src[i]); if (a > m) m = a; }
  if (m == 0.0f) { for (size_t i = 0; i < n; i++) dst[i] = 0; return 1.0f; }
  const float inv = 127.0f / m;
  for (size_t i = 0; i < n; i++) {
    float v = src[i] * inv;
    int   r = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
    if (r >  127) r =  127;
    if (r < -127) r = -127;
    dst[i] = (int8_t)r;
  }
  return m / 127.0f;
}

#define GATE_A_REL 1e-4f
#define GATE_A_ABS 1e-9f
#define GATE_B_L2  3e-2

static int32_t ref_i32[LPI_D];
static float   ref_f32[LPI_D];

int main(int argc, char *argv[])
{
  int8_t  *W8 = LPI_W8, *V8 = LPI_V8;
  int32_t *acc = LPI_A;
  float   *o = LPI_O, *Vf = LPI_VF, *Pf = LPI_PF, *vs = LPI_VS, *wf = LPI_WF;

  printf("llama-attn-pv-int8 S,d = %d,%d (int8 V per-row folded into int8 P)\n",
         LPI_S, LPI_D);

  // fp32 source data.
  fill_f32(Vf, (size_t)LPI_S * LPI_D, 1.0f);
  fill_f32(Pf, LPI_S, 1.0f);
  // Make P a real probability vector (this is what sets wmax/rms(w) above).
  float psum = 0.0f;
  for (size_t s = 0; s < LPI_S; s++) { Pf[s] = expf(Pf[s]); psum += Pf[s]; }
  for (size_t s = 0; s < LPI_S; s++) Pf[s] /= psum;

  // Quantise V per row, then FOLD the row scales into P and quantise that.
  for (size_t s = 0; s < LPI_S; s++)
    vs[s] = quant_vec(Vf + s * LPI_D, V8 + s * LPI_D, LPI_D);
  for (size_t s = 0; s < LPI_S; s++) wf[s] = Pf[s] * vs[s];
  const float wscale = quant_vec(wf, W8, LPI_S);

  for (size_t j = 0; j < LPI_D; j++) { acc[j] = 0; o[j] = 0.0f; }

  setStats(1);
  llama_attn_pv_int8(LPI_S, LPI_D, W8, V8, acc, o, wscale);
  setStats(0);

  // --- GATE A: scalar int32 reference over the same quantised bytes -------
  for (size_t j = 0; j < LPI_D; j++) ref_i32[j] = 0;
  for (size_t s = 0; s < LPI_S; s++) {
    const int32_t w = (int32_t)W8[s];
    const int8_t *vr = V8 + s * LPI_D;
    for (size_t j = 0; j < LPI_D; j++) ref_i32[j] += (int32_t)vr[j] * w;
  }
  int bad = 0;
  for (size_t j = 0; j < LPI_D && bad < 8; j++) {
    float want = (float)ref_i32[j] * wscale;
    if (fabsf_(o[j] - want) > GATE_A_ABS + GATE_A_REL * fabsf_(want)) {
      printf("MISMATCH j=%d: got %x want %x\n",
             (int)j, (unsigned)bits(o[j]), (unsigned)bits(want));
      bad++;
    }
  }

  // --- GATE B: relative L2 error against the true fp32 reference ----------
  for (size_t j = 0; j < LPI_D; j++) ref_f32[j] = 0.0f;
  for (size_t s = 0; s < LPI_S; s++) {
    const float p = Pf[s];
    const float *vr = Vf + s * LPI_D;
    for (size_t j = 0; j < LPI_D; j++) ref_f32[j] += p * vr[j];
  }
  double num = 0.0, den = 0.0;
  for (size_t j = 0; j < LPI_D; j++) {
    double dif = (double)o[j] - (double)ref_f32[j];
    num += dif * dif;
    den += (double)ref_f32[j] * (double)ref_f32[j];
  }
  double rel_l2 = (den > 0.0) ? __builtin_sqrt(num / den) : 0.0;
  printf("quant rel-L2 error x1e6 = %ld (gate %ld)\n",
         (long)(rel_l2 * 1e6), (long)(GATE_B_L2 * 1e6));

  double sum = 0.0;
  for (size_t j = 0; j < LPI_D; j++) sum += (double)o[j];
  printf("out checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED (gate A: kernel disagrees with the int32 reference)\n"); return 1; }
  if (rel_l2 > GATE_B_L2) { printf("FAILED (gate B: quantisation error too large)\n"); return 1; }
  printf("PASSED (all %d outputs exact vs int32 ref; quantisation within budget)\n", LPI_D);
  return 0;
}
