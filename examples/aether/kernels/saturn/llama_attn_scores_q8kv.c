// See LICENSE for license details.
//
// llama-attn-scores-q8kv — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// One K row at a time, LMUL=1, the int8 row widened to fp32 the obvious way
// (`vle8` -> `vsext.vf4` -> `vfcvt.f.x`), one accumulator group, one
// `vfredusum` per row on the critical path, `q` re-loaded for every row.

#include "riscv_vector.h"
#include "llama_attn_scores_q8kv.h"

void llama_attn_scores_q8kv(size_t S, size_t d, const int8_t *K8,
                            const float *kscale, const float *q,
                            float *scores, float scale)
{
  const size_t vl    = __riscv_vsetvlmax_e32m1();   // 8 fp32 lanes at VLEN=256
  const size_t dmain = d - (d % vl);

  for (size_t s = 0; s < S; s++) {
    const int8_t *kr = K8 + s * d;

    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (size_t j = 0; j < dmain; j += vl) {
      // int8 -> int32 -> fp32. The e32m1 group holds 8 elements, so the int8
      // source is an e8mf4 fractional group of the same 8.
      vint8mf4_t   k8  = __riscv_vle8_v_i8mf4(kr + j, vl);
      vint32m1_t   k32 = __riscv_vsext_vf4_i32m1(k8, vl);
      vfloat32m1_t kf  = __riscv_vfcvt_f_x_v_f32m1(k32, vl);
      vfloat32m1_t vq  = __riscv_vle32_v_f32m1(q + j, vl);
      acc = __riscv_vfmacc_vv_f32m1(acc, kf, vq, vl);
    }

    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t red  = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl);
    float dot = __riscv_vfmv_f_s_f32m1_f32(red);

    for (size_t j = dmain; j < d; j++)
      dot += (float)kr[j] * q[j];

    // The dequant scale is applied ONCE to the finished dot product, not per
    // element — the sum is linear in it.
    scores[s] = dot * kscale[s] * scale;
  }
}
