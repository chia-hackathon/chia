// See LICENSE for license details.
//
// llama-attn-scores — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// One K row at a time, LMUL=1, one accumulator group, one `vfredusum` per row
// on the critical path, `q` re-loaded from memory for every row.

#include "riscv_vector.h"
#include "llama_attn_scores.h"

void llama_attn_scores(size_t S, size_t d, const float *K, const float *q,
                       float *scores, float scale)
{
  const size_t vl    = __riscv_vsetvlmax_e32m1();   // 8 fp32 lanes at VLEN=256
  const size_t dmain = d - (d % vl);

  for (size_t s = 0; s < S; s++) {
    const float *kr = K + s * d;

    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (size_t j = 0; j < dmain; j += vl) {
      vfloat32m1_t vk = __riscv_vle32_v_f32m1(kr + j, vl);
      vfloat32m1_t vq = __riscv_vle32_v_f32m1(q  + j, vl);
      acc = __riscv_vfmacc_vv_f32m1(acc, vk, vq, vl);
    }

    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t red  = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl);
    float dot = __riscv_vfmv_f_s_f32m1_f32(red);

    for (size_t j = dmain; j < d; j++)
      dot += kr[j] * q[j];

    scores[s] = dot * scale;
  }
}
