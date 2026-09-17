// See LICENSE for license details.
//
// llama-attn-pv — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// The output accumulator is kept in MEMORY and round-tripped once per V row:
// LMUL=1, load out, `vfmacc.vf` with the scalar probability, store out. 512
// rows x 8 chunks of load+store that never had to leave the register file.

#include "riscv_vector.h"
#include "llama_attn_pv.h"

void llama_attn_pv(size_t S, size_t d, const float *P, const float *V,
                   float *out)
{
  const size_t vl    = __riscv_vsetvlmax_e32m1();   // 8 fp32 lanes at VLEN=256
  const size_t dmain = d - (d % vl);

  for (size_t s = 0; s < S; s++) {
    const float  p  = P[s];
    const float *vr = V + s * d;

    for (size_t j = 0; j < dmain; j += vl) {
      vfloat32m1_t acc = __riscv_vle32_v_f32m1(out + j, vl);
      vfloat32m1_t vv  = __riscv_vle32_v_f32m1(vr  + j, vl);
      acc = __riscv_vfmacc_vf_f32m1(acc, p, vv, vl);
      __riscv_vse32_v_f32m1(out + j, acc, vl);
    }
    for (size_t j = dmain; j < d; j++)
      out[j] += p * vr[j];
  }
}
