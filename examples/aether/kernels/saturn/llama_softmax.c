// See LICENSE for license details.
//
// llama-softmax — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// THREE separate passes over the row (max, exp+sum, scale), LMUL=1, a
// straight-line Cephes exp with no strength reduction and a `vfdiv` reciprocal
// that is computed once but applied with a full-precision divide.

#include "riscv_vector.h"
#include "llama_softmax.h"

// ---------------------------------------------------------------------------
// Vector exp, Cephes single-precision (the same algorithm as
// `common/ara/exp.h`, written out here so the kernel is self-contained and the
// optimizer can change it). Accurate to ~1 ulp over the argument range a
// max-subtracted softmax produces, i.e. (-inf, 0].
// ---------------------------------------------------------------------------
static inline vfloat32m1_t vexp_f32m1(vfloat32m1_t x, size_t vl)
{
  const float EXP_HI = 88.3762626647949f, EXP_LO = -88.3762626647949f;
  const float LOG2EF = 1.44269504088896341f;
  const float C1 = 0.693359375f, C2 = -2.12194440e-4f;
  const float p0 = 1.9875691500e-4f, p1 = 1.3981999507e-3f,
              p2 = 8.3334519073e-3f, p3 = 4.1665795894e-2f,
              p4 = 1.6666665459e-1f, p5 = 5.0000001201e-1f;

  x = __riscv_vfmin_vf_f32m1(x, EXP_HI, vl);
  x = __riscv_vfmax_vf_f32m1(x, EXP_LO, vl);

  // n = floor(x*log2(e) + 0.5)
  vfloat32m1_t fx = __riscv_vfmv_v_f_f32m1(0.5f, vl);
  fx = __riscv_vfmacc_vf_f32m1(fx, LOG2EF, x, vl);
  vint32m1_t   ni  = __riscv_vfcvt_x_f_v_i32m1(fx, vl);
  vfloat32m1_t nf  = __riscv_vfcvt_f_x_v_f32m1(ni, vl);
  vbool32_t    up  = __riscv_vmflt_vv_f32m1_b32(fx, nf, vl);   // rounded up?
  vfloat32m1_t one  = __riscv_vfmv_v_f_f32m1(1.0f, vl);
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t corr = __riscv_vmerge_vvm_f32m1(zero, one, up, vl);
  nf = __riscv_vfsub_vv_f32m1(nf, corr, vl);                   // nf = floor()
  ni = __riscv_vfcvt_x_f_v_i32m1(nf, vl);

  // r = x - n*ln2, in two pieces for accuracy
  x = __riscv_vfnmsac_vf_f32m1(x, C1, nf, vl);
  x = __riscv_vfnmsac_vf_f32m1(x, C2, nf, vl);

  // Horner: y = p0 x^5 + ... + p5
  vfloat32m1_t y = __riscv_vfmv_v_f_f32m1(p0, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p1, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p2, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p3, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p4, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p5, vl);
  vfloat32m1_t z = __riscv_vfmul_vv_f32m1(x, x, vl);
  y = __riscv_vfmul_vv_f32m1(y, z, vl);
  y = __riscv_vfadd_vv_f32m1(y, x, vl);
  y = __riscv_vfadd_vf_f32m1(y, 1.0f, vl);

  // scale by 2^n
  vint32m1_t   e   = __riscv_vadd_vx_i32m1(ni, 127, vl);
  e = __riscv_vsll_vx_i32m1(e, 23, vl);
  vfloat32m1_t p2n = __riscv_vreinterpret_v_i32m1_f32m1(e);
  return __riscv_vfmul_vv_f32m1(y, p2n, vl);
}

void llama_softmax(size_t n, const float *x, float *y)
{
  const size_t vl    = __riscv_vsetvlmax_e32m1();   // 8 fp32 lanes at VLEN=256
  const size_t nmain = n - (n % vl);

  // --- pass 1: row max ---------------------------------------------------
  vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-3.4028235e38f, vl);
  for (size_t i = 0; i < nmain; i += vl)
    vmax = __riscv_vfmax_vv_f32m1(vmax, __riscv_vle32_v_f32m1(x + i, vl), vl);
  vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(-3.4028235e38f, 1);
  float m = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredmax_vs_f32m1_f32m1(vmax, seed, vl));
  for (size_t i = nmain; i < n; i++) if (x[i] > m) m = x[i];

  // --- pass 2: exp(x - m), and its sum -----------------------------------
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  for (size_t i = 0; i < nmain; i += vl) {
    vfloat32m1_t v = __riscv_vle32_v_f32m1(x + i, vl);
    v = __riscv_vfsub_vf_f32m1(v, m, vl);
    v = vexp_f32m1(v, vl);
    __riscv_vse32_v_f32m1(y + i, v, vl);
    acc = __riscv_vfadd_vv_f32m1(acc, v, vl);
  }
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  float s = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl));
  for (size_t i = nmain; i < n; i++) {
    float e = __builtin_expf(x[i] - m);
    y[i] = e;
    s += e;
  }

  // --- pass 3: normalise -------------------------------------------------
  for (size_t i = 0; i < nmain; i += vl) {
    vfloat32m1_t v = __riscv_vle32_v_f32m1(y + i, vl);
    v = __riscv_vfdiv_vf_f32m1(v, s, vl);
    __riscv_vse32_v_f32m1(y + i, v, vl);
  }
  for (size_t i = nmain; i < n; i++) y[i] /= s;
}
