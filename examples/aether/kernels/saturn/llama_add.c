// See LICENSE for license details.
//
// llama-add — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// LMUL=1, one strip at a time, `vsetvl` re-issued every iteration, no
// unrolling and no software pipelining of the two load streams against the
// store stream.

#include "riscv_vector.h"
#include "llama_add.h"

void llama_add(size_t n, float *x, const float *y)
{
  for (size_t i = 0; i < n; ) {
    size_t vl = __riscv_vsetvl_e32m1(n - i);
    vfloat32m1_t a = __riscv_vle32_v_f32m1(x + i, vl);
    vfloat32m1_t b = __riscv_vle32_v_f32m1(y + i, vl);
    __riscv_vse32_v_f32m1(x + i, __riscv_vfadd_vv_f32m1(a, b, vl), vl);
    i += vl;
  }
}
