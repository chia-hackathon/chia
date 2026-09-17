// See LICENSE for license details.
//
// llama-rope — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// One head at a time, LMUL=1, the cos/sin table re-loaded from memory for
// every head, no unrolling, no fusion of the two halves.

#include "riscv_vector.h"
#include "llama_rope.h"

void llama_rope(size_t nheads, size_t head_dim, float *x,
                const float *cs, const float *sn)
{
  const size_t half = head_dim / 2;

  for (size_t h = 0; h < nheads; h++) {
    float *p = x + h * head_dim;

    for (size_t i = 0; i < half; ) {
      size_t vl = __riscv_vsetvl_e32m1(half - i);

      vfloat32m1_t x1 = __riscv_vle32_v_f32m1(p + i,        vl);
      vfloat32m1_t x2 = __riscv_vle32_v_f32m1(p + half + i, vl);
      vfloat32m1_t c  = __riscv_vle32_v_f32m1(cs + i,       vl);
      vfloat32m1_t s  = __riscv_vle32_v_f32m1(sn + i,       vl);

      // o1 = x1*c - x2*s ; o2 = x2*c + x1*s
      vfloat32m1_t o1 = __riscv_vfmul_vv_f32m1(x1, c, vl);
      o1 = __riscv_vfnmsac_vv_f32m1(o1, x2, s, vl);
      vfloat32m1_t o2 = __riscv_vfmul_vv_f32m1(x2, c, vl);
      o2 = __riscv_vfmacc_vv_f32m1(o2, x1, s, vl);

      __riscv_vse32_v_f32m1(p + i,        o1, vl);
      __riscv_vse32_v_f32m1(p + half + i, o2, vl);

      i += vl;
    }
  }
}
