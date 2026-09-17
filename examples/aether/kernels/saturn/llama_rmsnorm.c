// See LICENSE for license details.
//
// llama-rmsnorm — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// Two separate passes over `x` (one to reduce, one to scale), LMUL=1, one
// accumulator, no unrolling, no software pipelining.

#include "riscv_vector.h"
#include "llama_rmsnorm.h"

void llama_rmsnorm(size_t n, const float *x, const float *w, float *y,
                   float eps)
{
  const size_t vl    = __riscv_vsetvlmax_e32m1();   // VLEN=256 -> 8 fp32 lanes
  const size_t nmain = n - (n % vl);

  // --- pass 1: sum of squares -------------------------------------------
  vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  for (size_t i = 0; i < nmain; i += vl) {
    vfloat32m1_t v = __riscv_vle32_v_f32m1(x + i, vl);
    acc = __riscv_vfmacc_vv_f32m1(acc, v, v, vl);
  }
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t red  = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl);
  float ss = __riscv_vfmv_f_s_f32m1_f32(red);
  for (size_t i = nmain; i < n; i++)
    ss += x[i] * x[i];

  const float scale = 1.0f / __builtin_sqrtf(ss / (float)n + eps);

  // --- pass 2: scale by `scale` and by the per-channel weight ------------
  for (size_t i = 0; i < nmain; i += vl) {
    vfloat32m1_t v  = __riscv_vle32_v_f32m1(x + i, vl);
    vfloat32m1_t vw = __riscv_vle32_v_f32m1(w + i, vl);
    vfloat32m1_t o  = __riscv_vfmul_vv_f32m1(v, vw, vl);
    o = __riscv_vfmul_vf_f32m1(o, scale, vl);
    __riscv_vse32_v_f32m1(y + i, o, vl);
  }
  for (size_t i = nmain; i < n; i++)
    y[i] = x[i] * scale * w[i];
}
