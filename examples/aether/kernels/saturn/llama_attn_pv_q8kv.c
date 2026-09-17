// See LICENSE for license details.
//
// llama-attn-pv-q8kv — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// Like the fp32 `llama-attn-pv` baseline it keeps the 64-wide output
// accumulator in MEMORY and round-trips it once per V row, and it widens the
// int8 row to fp32 the obvious way (`vle8` -> `vsext.vf4` -> `vfcvt.f.x`).

#include "riscv_vector.h"
#include "llama_attn_pv_q8kv.h"

void llama_attn_pv_q8kv(size_t S, size_t d, const float *P, const int8_t *V8,
                        const float *vscale, float *out)
{
  const size_t vl    = __riscv_vsetvlmax_e32m1();   // 8 fp32 lanes at VLEN=256
  const size_t dmain = d - (d % vl);

  for (size_t s = 0; s < S; s++) {
    // One scalar multiply per ROW folds the dequant scale in for free.
    const float   p  = P[s] * vscale[s];
    const int8_t *vr = V8 + s * d;

    for (size_t j = 0; j < dmain; j += vl) {
      vint8mf4_t   v8  = __riscv_vle8_v_i8mf4(vr + j, vl);
      vint32m1_t   v32 = __riscv_vsext_vf4_i32m1(v8, vl);
      vfloat32m1_t vf  = __riscv_vfcvt_f_x_v_f32m1(v32, vl);
      vfloat32m1_t acc = __riscv_vle32_v_f32m1(out + j, vl);
      acc = __riscv_vfmacc_vf_f32m1(acc, p, vf, vl);
      __riscv_vse32_v_f32m1(out + j, acc, vl);
    }
    for (size_t j = dmain; j < d; j++)
      out[j] += p * (float)vr[j];
  }
}
