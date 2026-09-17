// See LICENSE for license details.
//
//**************************************************************************
// llama-attn-scores-int8 — one decode query head, INT8 K cache AND INT8 query
//--------------------------------------------------------------------------
// SEALED HARNESS. Every build is reassembled from the pristine copy of this
// file plus the optimizer's `llama_attn_scores_int8.c`.
//
// TWO checks, because there are two different things that can be wrong:
//
//   GATE A (tight, 1e-4 relative) — the KERNEL check. A scalar int32 dot
//     product over exactly the same K8/q8 bytes the kernel saw, then the same
//     fp32 rescale. Nothing about quantisation enters here: the integer dot
//     has ONE correct answer, so this is as strict as the fp32 benchmarks and
//     it is what scores the attempt.
//
//   GATE B (loose, relative L2) — the QUANTISATION check. The harness
//     generates fp32 K and q, quantises them, and compares the kernel's
//     scores against a true fp32 reference computed from the ORIGINAL data.
//     This cannot be tight and should not be: it measures the scheme, not the
//     code. Reported as ||got - ref||_2 / ||ref||_2, which is the standard way
//     to state quantisation error and is robust to individual scores landing
//     near zero through cancellation (a per-element relative tolerance is
//     meaningless there). Budget below.
//
// QUANTISATION ERROR BUDGET (why 3e-2 is the right gate)
//   Symmetric int8 with scale = max|x|/127 has a uniform round-off of at most
//   half a step, i.e. |dx| <= max|x|/254, so per element the RMS error is
//   max|x| / (254 * sqrt(3)) = max|x| / 440.
//   For the roughly-uniform data here max|x| ~ 2 * rms|x|, giving a per-element
//   relative RMS error of about 1/220 = 0.45% on EACH of K and q.
//   A dot product of d = 64 terms sums 64 independent such errors against a
//   signal that also grows as sqrt(64), so the relative error of the dot is
//   the same order, not sqrt(64) times worse: ~0.45% from K, ~0.45% from q,
//   ~0.65% combined in quadrature. Doubling that for the non-Gaussian tail of
//   a 512-point max gives ~1.3%, so a 3e-2 gate has better than 2x of margin
//   while still failing loudly on a broken scheme (a wrong scale, a dropped
//   sign, saturation at 127 instead of rounding) which lands at 10% or worse.
//   The actual measured value is PRINTED, so the margin is never a guess.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "riscv_vector.h"
#include "util.h"

#include "llama_attn_scores_int8.h"

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

// Symmetric int8 quantisation of one vector. Returns the dequant scale.
static float quant_vec(const float *src, int8_t *dst, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; i++) { float a = fabsf_(src[i]); if (a > m) m = a; }
  if (m == 0.0f) { for (size_t i = 0; i < n; i++) dst[i] = 0; return 1.0f; }
  const float inv = 127.0f / m;
  for (size_t i = 0; i < n; i++) {
    float v = src[i] * inv;
    int   r = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);   // round half away from 0
    if (r >  127) r =  127;
    if (r < -127) r = -127;
    dst[i] = (int8_t)r;
  }
  return m / 127.0f;
}

#define GATE_A_REL 1e-4f
#define GATE_A_ABS 1e-6f
#define GATE_B_L2  3e-2

int main(int argc, char *argv[])
{
  int8_t *K8 = LSI_K8, *q8 = LSI_Q8;
  float  *ks = LSI_KS, *o = LSI_O, *Kf = LSI_KF, *qf = LSI_QF;

  printf("llama-attn-scores-int8 S,d = %d,%d (int8 K per-row + int8 q per-vector)\n",
         LSI_S, LSI_D);

  // fp32 source data, then quantise it. Keeping the fp32 around is what makes
  // the quantisation-error number meaningful.
  fill_f32(Kf, (size_t)LSI_S * LSI_D, 1.0f);
  fill_f32(qf, LSI_D, 1.0f);
  for (size_t s = 0; s < LSI_S; s++)
    ks[s] = quant_vec(Kf + s * LSI_D, K8 + s * LSI_D, LSI_D);
  const float qscale = quant_vec(qf, q8, LSI_D);

  for (size_t s = 0; s < LSI_S; s++) o[s] = 0.0f;

  setStats(1);
  llama_attn_scores_int8(LSI_S, LSI_D, K8, ks, q8, o, qscale, LSI_SCALE);
  setStats(0);

  // --- GATE A: scalar int32 reference over the same quantised bytes -------
  int bad = 0;
  for (size_t s = 0; s < LSI_S && bad < 8; s++) {
    const int8_t *kr = K8 + s * LSI_D;
    int32_t dot = 0;
    for (size_t j = 0; j < LSI_D; j++) dot += (int32_t)kr[j] * (int32_t)q8[j];
    float want = (float)dot * ks[s] * qscale * LSI_SCALE;
    float d = fabsf_(o[s] - want);
    if (d > GATE_A_ABS + GATE_A_REL * fabsf_(want)) {
      printf("MISMATCH s=%d: got %x want %x\n",
             (int)s, (unsigned)bits(o[s]), (unsigned)bits(want));
      bad++;
    }
  }

  // --- GATE B: relative L2 error against the true fp32 reference ----------
  double num = 0.0, den = 0.0;
  for (size_t s = 0; s < LSI_S; s++) {
    const float *kr = Kf + s * LSI_D;
    float acc = 0.0f;
    for (size_t j = 0; j < LSI_D; j++) acc += kr[j] * qf[j];
    double ref = (double)acc * (double)LSI_SCALE;
    double dif = (double)o[s] - ref;
    num += dif * dif;
    den += ref * ref;
  }
  double rel_l2 = (den > 0.0) ? __builtin_sqrt(num / den) : 0.0;
  printf("quant rel-L2 error x1e6 = %ld (gate %ld)\n",
         (long)(rel_l2 * 1e6), (long)(GATE_B_L2 * 1e6));

  double sum = 0.0;
  for (size_t s = 0; s < LSI_S; s++) sum += (double)o[s];
  printf("scores checksum x1e6 = %ld\n", (long)(sum * 1e6));

  if (bad) { printf("FAILED (gate A: kernel disagrees with the int32 reference)\n"); return 1; }
  if (rel_l2 > GATE_B_L2) { printf("FAILED (gate B: quantisation error too large)\n"); return 1; }
  printf("PASSED (all %d scores exact vs int32 ref; quantisation within budget)\n", LSI_S);
  return 0;
}
