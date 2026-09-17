// See LICENSE for license details.
//
// llama-attn-pv-int8 — THE FILE UNDER OPTIMIZATION.
//
// Baseline: correct, readable, deliberately unoptimized RVV 1.0 intrinsics.
// It keeps the int32 accumulator in MEMORY and round-trips it once per V row —
// the same deliberate flaw as the fp32 and q8kv baselines, so the three
// measurements of this shape differ only in dtype, not in how naive they are.

#include "riscv_vector.h"
#include "llama_attn_pv_int8.h"

void llama_attn_pv_int8(size_t S, size_t d, const int8_t *W8, const int8_t *V8,
                        int32_t *acc, float *out, float wscale)
{
  // e8/LMUL=1 strip: VLEN=256 -> 32 int8 lanes, so the int32 accumulator is
  // LMUL=4 over the same 32 elements.
  const size_t vl    = __riscv_vsetvlmax_e8m1();
  const size_t dmain = d - (d % vl);

  for (size_t s = 0; s < S; s++) {
    const int8_t  w  = W8[s];
    const int8_t *vr = V8 + s * d;

    for (size_t j = 0; j < dmain; j += vl) {
      vint8m1_t  vv = __riscv_vle8_v_i8m1(vr + j, vl);
      vint32m4_t a  = __riscv_vle32_v_i32m4(acc + j, vl);
      vint16m2_t vp = __riscv_vwmul_vx_i16m2(vv, w, vl);
      a = __riscv_vwadd_wv_i32m4(a, vp, vl);
      __riscv_vse32_v_i32m4(acc + j, a, vl);
    }
    for (size_t j = dmain; j < d; j++)
      acc[j] += (int32_t)vr[j] * (int32_t)w;
  }

  // ONE floating-point multiply per output. Everything above was integer.
  for (size_t j = 0; j < d; j++)
    out[j] = (float)acc[j] * wscale;
}
