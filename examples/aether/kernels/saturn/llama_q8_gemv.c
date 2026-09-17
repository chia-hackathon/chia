// See LICENSE for license details.
//
// llama-q8-gemv — THE FILE UNDER OPTIMIZATION.
//
// This is the only file whose bytes reach the compiler from your workspace.
// Everything else (`llama-q8-gemv_main.c`, `llama_q8_gemv.h`, `common/`) is
// reassembled from the pristine tree at every build.
//
// Baseline: correct, readable, and deliberately unoptimized RVV 1.0
// intrinsics — one row at a time, one int32 accumulator group, no unrolling,
// no reuse of the loaded activation vector across rows.

#include "riscv_vector.h"
#include "llama_q8_gemv.h"

void llama_q8_gemv(size_t M, size_t K,
                   const int8_t *W, const int8_t *x, int32_t *y)
{
  // e8/LMUL=1 strip length. VLEN = 256 -> 32 int8 lanes; the widened int32
  // accumulator therefore needs LMUL=4 (32 lanes x 32 bit = 4 * VLEN).
  const size_t vl   = __riscv_vsetvlmax_e8m1();
  const size_t kmain = K - (K % vl);

  for (size_t m = 0; m < M; m++) {
    const int8_t *w = W + m * K;

    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);

    for (size_t k = 0; k < kmain; k += vl) {
      vint8m1_t  vw = __riscv_vle8_v_i8m1(w + k, vl);
      vint8m1_t  vx = __riscv_vle8_v_i8m1(x + k, vl);
      // int8 x int8 -> int16 products, widened again into the int32 running
      // sum. int16 alone would overflow: 2048 * 127 * 127 > 2^15.
      vint16m2_t vp = __riscv_vwmul_vv_i16m2(vw, vx, vl);
      acc = __riscv_vwadd_wv_i32m4(acc, vp, vl);
    }

    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t red  = __riscv_vredsum_vs_i32m4_i32m1(acc, zero, vl);
    int32_t s = __riscv_vmv_x_s_i32m1_i32(red);

    for (size_t k = kmain; k < K; k++)
      s += (int32_t)w[k] * (int32_t)x[k];

    y[m] = s;
  }
}
