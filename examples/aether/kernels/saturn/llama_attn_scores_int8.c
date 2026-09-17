// See LICENSE for license details.
//
// llama-attn-scores-int8 — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// The same shape as the `llama-q8-gemv` inner loop: one K row at a time,
// `e8m1` strips, `vwmul.vv` -> int16, `vwadd.wv` -> int32, one `vredsum` per
// row on the critical path, `q8` re-loaded from memory for every row, and a
// single fp32 rescale at the end of each row.

#include "riscv_vector.h"
#include "llama_attn_scores_int8.h"

void llama_attn_scores_int8(size_t S, size_t d, const int8_t *K8,
                            const float *kscale, const int8_t *q8,
                            float *scores, float qscale, float scale)
{
  // e8/LMUL=1 strip: VLEN=256 -> 32 int8 lanes. The widened int32 accumulator
  // therefore needs LMUL=4 (32 lanes x 32 bit = 4 * VLEN).
  const size_t vl    = __riscv_vsetvlmax_e8m1();
  const size_t dmain = d - (d % vl);
  const float  fs    = qscale * scale;

  for (size_t s = 0; s < S; s++) {
    const int8_t *kr = K8 + s * d;

    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
    for (size_t j = 0; j < dmain; j += vl) {
      vint8m1_t  vk = __riscv_vle8_v_i8m1(kr + j, vl);
      vint8m1_t  vq = __riscv_vle8_v_i8m1(q8 + j, vl);
      // int8 x int8 -> int16 products, widened again into the int32 running
      // sum. int16 alone is not safe in general and is not worth the risk.
      vint16m2_t vp = __riscv_vwmul_vv_i16m2(vk, vq, vl);
      acc = __riscv_vwadd_wv_i32m4(acc, vp, vl);
    }

    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t red  = __riscv_vredsum_vs_i32m4_i32m1(acc, zero, vl);
    int32_t dot = __riscv_vmv_x_s_i32m1_i32(red);

    for (size_t j = dmain; j < d; j++)
      dot += (int32_t)kr[j] * (int32_t)q8[j];

    // ONE floating-point multiply per output. Everything above was integer.
    scores[s] = (float)dot * kscale[s] * fs;
  }
}
