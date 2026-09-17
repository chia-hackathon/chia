// See LICENSE for license details.
//
// llama-silu-mul — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// LMUL=1, one element strip at a time, a straight-line Cephes exp and a
// full-precision `vfdiv` per strip (no reciprocal estimate, no Newton step,
// no unrolling, no overlap of the load stream with the polynomial).

#include "riscv_vector.h"
#include "llama_silu_mul.h"

// Vector exp, Cephes single precision (same algorithm as common/ara/exp.h,
// written out here so the kernel is self-contained and the optimizer owns it).
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

  vfloat32m1_t fx = __riscv_vfmv_v_f_f32m1(0.5f, vl);
  fx = __riscv_vfmacc_vf_f32m1(fx, LOG2EF, x, vl);
  vint32m1_t   ni = __riscv_vfcvt_x_f_v_i32m1(fx, vl);
  vfloat32m1_t nf = __riscv_vfcvt_f_x_v_f32m1(ni, vl);
  vbool32_t    up = __riscv_vmflt_vv_f32m1_b32(fx, nf, vl);
  vfloat32m1_t one  = __riscv_vfmv_v_f_f32m1(1.0f, vl);
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t corr = __riscv_vmerge_vvm_f32m1(zero, one, up, vl);
  nf = __riscv_vfsub_vv_f32m1(nf, corr, vl);                   // floor
  ni = __riscv_vfcvt_x_f_v_i32m1(nf, vl);

  x = __riscv_vfnmsac_vf_f32m1(x, C1, nf, vl);
  x = __riscv_vfnmsac_vf_f32m1(x, C2, nf, vl);

  vfloat32m1_t y = __riscv_vfmv_v_f_f32m1(p0, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p1, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p2, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p3, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p4, vl);
  y = __riscv_vfmul_vv_f32m1(y, x, vl); y = __riscv_vfadd_vf_f32m1(y, p5, vl);
  vfloat32m1_t z = __riscv_vfmul_vv_f32m1(x, x, vl);
  y = __riscv_vfmul_vv_f32m1(y, z, vl);
  y = __riscv_vfadd_vv_f32m1(y, x, vl);
  y = __riscv_vfadd_vv_f32m1(y, one, vl);

  vint32m1_t e = __riscv_vadd_vx_i32m1(ni, 127, vl);
  e = __riscv_vsll_vx_i32m1(e, 23, vl);
  return __riscv_vfmul_vv_f32m1(y, __riscv_vreinterpret_v_i32m1_f32m1(e), vl);
}

void llama_silu_mul(size_t n, const float *gate, const float *up, float *out)
{
  for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e32m1(n - i);

    vfloat32m1_t g = __riscv_vle32_v_f32m1(gate + i, vl);
    vfloat32m1_t u = __riscv_vle32_v_f32m1(up   + i, vl);

    // silu(g) = g / (1 + exp(-g))
    vfloat32m1_t e = vexp_f32m1(__riscv_vfneg_v_f32m1(g, vl), vl);
    vfloat32m1_t d = __riscv_vfadd_vf_f32m1(e, 1.0f, vl);
    vfloat32m1_t s = __riscv_vfdiv_vv_f32m1(g, d, vl);

    __riscv_vse32_v_f32m1(out + i, __riscv_vfmul_vv_f32m1(s, u, vl), vl);
    i += vl;
  }
}
