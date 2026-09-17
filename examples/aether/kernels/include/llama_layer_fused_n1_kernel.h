// See LICENSE for license details.
//
// llama-layer-fused (N = 1) — THE FILE UNDER OPTIMIZATION.
//
// This is the only file whose bytes reach the compiler from your workspace.
// `bareMetalC/llama_layer_fused_n1.c`, `include/llama_layer_fused_body.h`
// (main, data generator, L2 flush, golden models, self-checks) and
// `include/gemmini.h` / `include/gemmini_params.h` are reassembled from the
// pristine, hardware-derived tree at every build.
//
// WHAT THIS BASELINE IS
// ---------------------
// The four best-known kernels for the four pieces of work, run STRICTLY BACK
// TO BACK, with Gemmini fully fenced before the first vector instruction:
//
//   1. lf_gemv_gemmini   — the `llama-q8-gemv-gemmini-n1` Round-3 winner
//                          (132,424 cycles warm; expect ~149k cold here)
//   2. lf_attn_scores    — the `llama-attn-scores-int8` Round-2 winner (8,233)
//   3. lf_softmax        — the `llama-softmax` Round-3 winner (1,555)
//   4. lf_attn_pv        — the `llama-attn-pv-int8` Round-3 winner (4,814)
//
// Each of the four is individually at or near its own floor. **There is
// nothing left to win inside any of them.** The entire opportunity is the
// BOUNDARY between 1 and 2-4: today it is a `fence` that costs the full Saturn
// time on top of the full Gemmini time, and the notes in the registry explain
// why the hardware does not require that.
//
// Do not rewrite the four kernels' inner loops. Change the SCHEDULE.

#ifndef LLAMA_LAYER_FUSED_N1_KERNEL_H
#define LLAMA_LAYER_FUSED_N1_KERNEL_H

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "include/gemmini.h"

#define LF_MIN(a, b) ((a) < (b) ? (a) : (b))

// ===========================================================================
// 1. Gemmini decode GEMV — `llama-q8-gemv-gemmini-n1` best (132,424 warm)
// ===========================================================================
// Explicit weight-stationary tiling over the hardware loop
// (`sp_tiled_matmul_ws` -> `gemmini_loop_ws`) instead of `tiled_matmul_auto`:
//   * tile_I = 1, tile_J = whole M (M/DIM tiles, exactly ACC_ROWS/2 acc rows),
//     tile_K as large as half the scratchpad allows. Every 512 B row of B is
//     moved in ONCE, so the weight stream is a single sequential sweep of B.
//     The whole 1 x 512 int32 output lives in the accumulator across the
//     entire K reduction and is moved out once. 9 `loop_ws` calls.
//   * K tiles are processed in DESCENDING order. Under the WARM-tail harnesses
//     that harvested ~200-260 KiB of L2 residue and was worth 17,104 cycles.
//     **This harness flushes the L2 first, so that credit is GONE** — whether
//     descending still helps here is an open question and a legitimate
//     one-line experiment.
//   * the Zicbop `prefetch.r` burst between `loop_ws` calls was worth -229
//     cycles warm. It spends HOST cycles — the same host cycles the Saturn
//     work needs. It is the first thing to consider deleting.

// Zicbop `prefetch.r 0(rs1)`: ORI-encoded (OP-IMM, funct3 = 110, rd = x0,
// imm[4:0] = 00001). On a core that implements it the dcache forwards a
// TileLink Hint to the L2; on a core without Zicbop it is `ori x0, rs1, 1`,
// a NOP.
static inline void lf_prefetch_r(const void *p)
{
    asm volatile(".insn i 0x13, 0x6, x0, %0, 0x1" :: "r"(p) : "memory");
}

static inline uint64_t lf_rdcycle(void)
{
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

#define LF_PREFETCH_BUDGET_CYCLES 2500
#define LF_LINE_BYTES 64

// C[N x M] (acc_t, row-major, stride M)
//   = A[N x K] (elem_t, row-major, stride K) * B[K x M] (elem_t, stride M)
// No bias, no activation, no output scaling: exact int32 result.
// NOTE: ends with `gemmini_fence()`. That fence is the thing to move.
static void lf_gemv_gemmini(size_t N, size_t K, size_t M,
                            const elem_t *A, const elem_t *B, acc_t *C)
{
    const size_t dim_I = N, dim_J = M, dim_K = K;
    const size_t stride_A = K, stride_B = M, stride_C = M;

    const size_t I_t = (dim_I + DIM - 1) / DIM;
    const size_t J_t = (dim_J + DIM - 1) / DIM;
    const size_t K_t = (dim_K + DIM - 1) / DIM;
    const size_t padding_I = I_t * DIM - dim_I;
    const size_t padding_J = J_t * DIM - dim_J;
    const size_t padding_K = K_t * DIM - dim_K;

    // The hardware loop double-buffers across consecutive loop_ws
    // invocations: each instance owns half the scratchpad and half the acc.
    const size_t max_spad_rows = (size_t)BANK_NUM * BANK_ROWS / 2;
    const size_t max_acc_rows  = (size_t)ACC_ROWS / 2;

    size_t tile_I = LF_MIN(I_t, max_acc_rows / DIM);
    size_t tile_J = LF_MIN(J_t, (max_acc_rows / DIM) / tile_I);
    size_t tile_K = LF_MIN(K_t, max_spad_rows / ((tile_I + tile_J) * DIM));
    if (tile_K == 0) tile_K = 1;

    const size_t I0 = (I_t + tile_I - 1) / tile_I;
    const size_t J0 = (J_t + tile_J - 1) / tile_J;
    const size_t K0 = (K_t + tile_K - 1) / tile_K;
    const size_t last_I = I_t % tile_I == 0 ? tile_I : I_t % tile_I;
    const size_t last_J = J_t % tile_J == 0 ? tile_J : J_t % tile_J;
    const size_t last_K = K_t % tile_K == 0 ? tile_K : K_t % tile_K;

    gemmini_extended_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION & 3, 0, 1, false, false);
    gemmini_extended_config_st(stride_C * sizeof(acc_t), NO_ACTIVATION & 3, ACC_SCALE_IDENTITY);
    gemmini_extended3_config_ld(stride_A * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(stride_B * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
    gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);

    // Dummy non-NULL "D" selects accumulator-overwrite on the first K tile
    // (no bias is ever moved in because no_bias == true).
    const void *const D_dummy = (const void *)1;

    for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
    for (size_t kk = 0; kk < K0; kk++) {
        const size_t k0 = K0 - 1 - kk;          // descending K order
        const bool first = (kk == 0);
        const bool last  = (kk == K0 - 1);

        const size_t I = i0 < I0 - 1 ? tile_I : last_I;
        const size_t J = j0 < J0 - 1 ? tile_J : last_J;
        const size_t Kt = k0 < K0 - 1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0 - 1 ? padding_I : 0;
        const size_t pad_J = j0 == J0 - 1 ? padding_J : 0;
        const size_t pad_K = k0 == K0 - 1 ? padding_K : 0;

        const elem_t *a = A + i0 * tile_I * DIM * stride_A + k0 * tile_K * DIM;
        const elem_t *b = B + k0 * tile_K * DIM * stride_B + j0 * tile_J * DIM;
        void *out = last ? (void *)(C + i0 * tile_I * DIM * stride_C + j0 * tile_J * DIM) : NULL;
        const void *pre = first ? D_dummy : NULL;

        sp_tiled_matmul_ws(a, b, pre, out,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            I, J, Kt, pad_I, pad_J, pad_K,
            stride_A, stride_B, /* D_row_stride */ 0, stride_C,
            /* a_transpose */ false, /* b_transpose */ false,
            /* full_C */ true, /* low_D */ false,
            /* no_bias */ true, /* repeating_bias */ false,
            NO_ACTIVATION, /* a_spad_id */ 0, /* b_spad_id */ 0);

        // The issue above returns once loop kk-2 has retired, i.e. while loop
        // kk-1 is still streaming and kk's rows are ~one tile away. This is
        // EXACTLY the host slack the Saturn work wants.
        if (kk > 0 && J == J_t) {
            const uint64_t t0 = lf_rdcycle();
            const size_t bytes = Kt * DIM * stride_B;
            for (size_t off = 0; off < bytes; off += 8 * LF_LINE_BYTES) {
                lf_prefetch_r(b + off + 0 * LF_LINE_BYTES);
                lf_prefetch_r(b + off + 1 * LF_LINE_BYTES);
                lf_prefetch_r(b + off + 2 * LF_LINE_BYTES);
                lf_prefetch_r(b + off + 3 * LF_LINE_BYTES);
                lf_prefetch_r(b + off + 4 * LF_LINE_BYTES);
                lf_prefetch_r(b + off + 5 * LF_LINE_BYTES);
                lf_prefetch_r(b + off + 6 * LF_LINE_BYTES);
                lf_prefetch_r(b + off + 7 * LF_LINE_BYTES);
                if (lf_rdcycle() - t0 > LF_PREFETCH_BUDGET_CYCLES) break;
            }
        }
    }

    gemmini_fence();
}

// ===========================================================================
// 2. Saturn QK^T — `llama-attn-scores-int8` best (8,233)
// ===========================================================================
// Strategy (Saturn VLEN=256 / DLEN=128):
//   * Unit-stride SEGMENTED loads (vlseg8e64, nf=8, e64m1 -> vl=4) fetch four
//     64-byte K rows at once: segment i IS row i, field f is its 8-byte
//     chunk f. Viewed as int8 (vl=32), field f lane j holds
//     K[row j/8][8*f + j%8], so each lane accumulates 8 columns of ONE row
//     -> the 64-lane per-row reduction shrinks to an 8-partial sum.
//   * K blocks are DOUBLE-BUFFERED in registers (v16-v23 / v0-v7).
//   * q8 is loaded ONCE into 8 registers with the same lane mapping.
//   * vwmul.vv + vwmacc.vv keep TWO int8*int8 products in int16
//     (|2*127*127| = 32258 < 32767, exact); pair-sums widen into i32m4.
//   * every 8 rows a vlseg8e32 (vl=8) reloads the partials transposed and
//     7 vadd.vv give the 8 exact int32 dot products. No vredsum.
static int8_t  lf_qrep[256]        __attribute__((aligned(64)));
static int32_t lf_scratch[2][64]   __attribute__((aligned(64)));

#define LF_LOADK(K0, kp)                                                      \
  __asm__ volatile(                                                           \
    "vsetivli x0, 4, e64, m1, ta, ma\n\t"                                     \
    "vlseg8e64.v " K0 ", (%[k])\n\t"                                          \
    :                                                                         \
    : [k] "r"(kp)                                                             \
    : "memory")

#define LF_ARITH4(K0,K1,K2,K3,K4,K5,K6,K7, sp, c32)                           \
  __asm__ volatile(                                                           \
    "vsetvli x0, %[vl], e8, m1, ta, ma\n\t"                                   \
    "vwmul.vv  v24, " K0 ", v8\n\t"                                           \
    "vwmul.vv  v26, " K2 ", v10\n\t"                                          \
    "vwmacc.vv v24, " K1 ", v9\n\t"                                           \
    "vwmacc.vv v26, " K3 ", v11\n\t"                                          \
    "vwmul.vv  " K0 ", " K4 ", v12\n\t"                                       \
    "vwmul.vv  " K2 ", " K6 ", v14\n\t"                                       \
    "vwmacc.vv " K0 ", " K5 ", v13\n\t"                                       \
    "vwmacc.vv " K2 ", " K7 ", v15\n\t"                                       \
    "vsetvli x0, %[vl], e16, m2, ta, ma\n\t"                                  \
    "vwadd.vv  v28, v24, v26\n\t"                                             \
    "vwadd.wv  v28, v28, " K0 "\n\t"                                          \
    "vwadd.wv  v28, v28, " K2 "\n\t"                                          \
    "vsetvli x0, %[vl], e32, m4, ta, ma\n\t"                                  \
    "vse32.v   v28, (%[s])\n\t"                                               \
    :                                                                         \
    : [s] "r"(sp), [vl] "r"(c32)                                              \
    : "v24","v25","v26","v27","v28","v29","v30","v31","memory")

#define LF_ARITH_A(sp, c32) \
  LF_ARITH4("v16","v17","v18","v19","v20","v21","v22","v23", sp, c32)
#define LF_ARITH_B(sp, c32) \
  LF_ARITH4("v0","v1","v2","v3","v4","v5","v6","v7", sp, c32)

#define LF_REDUCE8(sp, ksp, out, fsv)                                         \
  __asm__ volatile(                                                           \
    "vsetivli x0, 8, e32, m1, ta, ma\n\t"                                     \
    "vlseg8e32.v v24, (%[s])\n\t"                                             \
    "vadd.vv  v24, v24, v25\n\t"                                              \
    "vadd.vv  v26, v26, v27\n\t"                                              \
    "vadd.vv  v28, v28, v29\n\t"                                              \
    "vadd.vv  v30, v30, v31\n\t"                                              \
    "vle32.v  v25, (%[ks])\n\t"                                               \
    "vadd.vv  v24, v24, v26\n\t"                                              \
    "vadd.vv  v28, v28, v30\n\t"                                              \
    "vadd.vv  v24, v24, v28\n\t"                                              \
    "vfcvt.f.x.v v24, v24\n\t"                                                \
    "vfmul.vv v24, v24, v25\n\t"                                              \
    "vfmul.vf v24, v24, %[fs]\n\t"                                            \
    "vse32.v  v24, (%[o])\n\t"                                                \
    :                                                                         \
    : [s] "r"(sp), [ks] "r"(ksp), [o] "r"(out), [fs] "f"(fsv)                 \
    : "v24","v25","v26","v27","v28","v29","v30","v31","memory")

static void lf_attn_scores(size_t S, size_t d, const int8_t *K8,
                           const float *kscale, const int8_t *q8,
                           float *scores, float qscale, float scale)
{
  const float fs = qscale * scale;

  if (d != 64 || (S % 8) != 0) {
    for (size_t s = 0; s < S; s++) {
      int32_t dot = 0;
      for (size_t j = 0; j < d; j++)
        dot += (int32_t)K8[s * d + j] * (int32_t)q8[j];
      scores[s] = (float)dot * kscale[s] * fs;
    }
    return;
  }

  // lf_qrep = q8 replicated 4x, so the SAME vlseg8e64 used for K yields, in
  // field f int8-lane j, q8[8*f + j%8] -- exactly the column K's field f lane
  // j holds.
  {
    size_t vl = __riscv_vsetvl_e8m2(64);
    vint8m2_t vq = __riscv_vle8_v_i8m2(q8, vl);
    __riscv_vse8_v_i8m2(lf_qrep +   0, vq, vl);
    __riscv_vse8_v_i8m2(lf_qrep +  64, vq, vl);
    __riscv_vse8_v_i8m2(lf_qrep + 128, vq, vl);
    __riscv_vse8_v_i8m2(lf_qrep + 192, vq, vl);
  }

  const size_t c32 = 32;
  __asm__ volatile(
    "vsetivli x0, 4, e64, m1, ta, ma\n\t"
    "vlseg8e64.v v8, (%[q])\n\t"
    :
    : [q] "r"(lf_qrep)
    : "v8","v9","v10","v11","v12","v13","v14","v15","memory");

  const size_t npair = S / 8;
  const int8_t *kp = K8;

  LF_LOADK("v16", kp);                           // block A of pair 0
  for (size_t n = 0; n < npair; n++) {
    int32_t *cur = lf_scratch[n & 1];
    LF_LOADK("v0", kp + 256);                    // block B of pair n
    LF_ARITH_A(cur, c32);
    if (n + 1 < npair)
      LF_LOADK("v16", kp + 512);                 // block A of pair n+1
    LF_ARITH_B(cur + 32, c32);
    kp += 512;
    if (n) {
      const size_t r0 = (n - 1) * 8;
      LF_REDUCE8(lf_scratch[(n - 1) & 1], kscale + r0, scores + r0, fs);
    }
  }
  {
    const size_t r0 = (npair - 1) * 8;
    LF_REDUCE8(lf_scratch[(npair - 1) & 1], kscale + r0, scores + r0, fs);
  }
}

// ===========================================================================
// 3. Saturn softmax — `llama-softmax` best (1,555)
// ===========================================================================
//   pass 1: row max, e32m8 (16-beat chime, load chained into vfmax).
//   pass 2: exp(x - k*ln2) + sum, e32m4. Softmax is shift-invariant, so
//           instead of subtracting m we shift by k*ln2, k = round(m*log2e):
//           a pure exponent offset that folds into the magic constant.
//   pass 3: y *= 1/s, e32m8, one scalar reciprocal instead of 64 vfdivs.
#define LF_R32(v) v,v,v,v,v,v,v,v, v,v,v,v,v,v,v,v, v,v,v,v,v,v,v,v, v,v,v,v,v,v,v,v
static const float lf_c3_tab[32] __attribute__((aligned(16))) = { LF_R32(0.16753436f) };
static const float lf_k_tab[3][32] __attribute__((aligned(16))) = {
  { LF_R32(0.50005112f) },   // c2
  { LF_R32(1.0f) },          // one
  { LF_R32(0.0f) } };        // accumulator seed

static void lf_softmax(size_t n, const float *x, float *y)
{
  const float LOG2EF = 1.44269504088896341f;
  const float LN2F   = 0.69314718055994531f;
  const float MAGIC  = 12582912.0f;               // 1.5 * 2^23
  const float c4 = 0.04127729f;

  size_t vl8 = __riscv_vsetvl_e32m8(n);
  vfloat32m8_t vmax = __riscv_vle32_v_f32m8(x, vl8);
  for (size_t i = vl8; i < n; i += vl8) {
    vl8 = __riscv_vsetvl_e32m8(n - i);
    vmax = __riscv_vfmax_vv_f32m8_tu(vmax, vmax, __riscv_vle32_v_f32m8(x + i, vl8), vl8);
  }

  // Pass-2 vector constants: issue the loads NOW, while the reduction and the
  // scalar k-chain below leave the (in-order) load pipe idle for ~40 cycles.
  size_t vl = __riscv_vsetvlmax_e32m4();
  const float *kp = lf_k_tab[0];
  __asm__("" : "+r"(kp));                         // keep these as loads
  const vfloat32m4_t C2  = __riscv_vle32_v_f32m4(kp,      vl);
  const vfloat32m4_t ONE = __riscv_vle32_v_f32m4(kp + 32, vl);
  vfloat32m4_t       acc = __riscv_vle32_v_f32m4(kp + 64, vl);
  __asm__ volatile("" ::: "memory");              // ... and pin them here

  vl8 = __riscv_vsetvlmax_e32m8();
  vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(-3.4028235e38f, 1);
  const float m = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredmax_vs_f32m8_f32m1(vmax, seed, vl8));
  float tmp;
  __asm__("fmadd.s %0, %1, %2, %3" : "=f"(tmp) : "f"(m), "f"(LOG2EF), "f"(MAGIC));
  const float MAGICK = 2.0f * MAGIC - tmp;

  vl = __riscv_vsetvlmax_e32m4();
  const vfloat32m4_t MK = __riscv_vfmv_v_f_f32m4(MAGICK, vl);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m4(n - i);
    const float *xa = x + i, *xb = x + i, *cp = lf_c3_tab;
    __asm__("" : "+r"(xb));                       // keep the 2nd load a load
    __asm__("" : "+r"(cp));                       // no hoisting -> no vmv copy
    vfloat32m4_t v  = __riscv_vle32_v_f32m4(xa, vl);
    vfloat32m4_t v2 = __riscv_vle32_v_f32m4(xb, vl);
    vfloat32m4_t C3 = __riscv_vle32_v_f32m4(cp, vl);
    vfloat32m4_t fx = __riscv_vfmadd_vf_f32m4(v, LOG2EF, MK, vl);
    vfloat32m4_t nf = __riscv_vfsub_vf_f32m4(fx, MAGICK, vl);   // = n
    vfloat32m4_t r  = __riscv_vfnmsac_vf_f32m4(v2, LN2F, nf, vl);
    vfloat32m4_t p  = __riscv_vfmacc_vf_f32m4(C3, c4, r, vl);   // c4 r + c3
    p = __riscv_vfmadd_vv_f32m4(p, r, C2, vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, ONE, vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, ONE, vl);
    vint32m4_t e = __riscv_vsll_vx_i32m4(__riscv_vreinterpret_v_f32m4_i32m4(fx), 23, vl);
    e = __riscv_vadd_vv_i32m4(__riscv_vreinterpret_v_f32m4_i32m4(p), e, vl);
    vfloat32m4_t o = __riscv_vreinterpret_v_i32m4_f32m4(e);
    __riscv_vse32_v_f32m4(y + i, o, vl);
    acc = __riscv_vfadd_vv_f32m4_tu(acc, acc, o, vl);
  }
  vl = __riscv_vsetvlmax_e32m4();
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  const float s = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m4_f32m1(acc, zero, vl));
  const float rs = 1.0f / s;

  for (size_t i = 0; i < n; i += vl8) {
    vl8 = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t v = __riscv_vle32_v_f32m8(y + i, vl8);
    __riscv_vse32_v_f32m8(y + i, __riscv_vfmul_vf_f32m8(v, rs, vl8), vl8);
  }
}

// ===========================================================================
// 4. Saturn probs@V — `llama-attn-pv-int8` best (4,814)
// ===========================================================================
// Register-resident accumulators: d=64 int32 is exactly one i32m8 group at
// VLEN=256. TWO such groups (v24..v31 and v16..v23) alternate and merge once
// at the end, so consecutive 16-beat vwadd.wv never depend on each other.
// `acc` is written exactly once at the end; the caller's zeros are NOT read
// back (v24 is zeroed with vmv.v.i instead).
static void lf_attn_pv(size_t S, size_t d, const int8_t *W8, const int8_t *V8,
                       int32_t *acc, float *out, float wscale)
{
  for (size_t j = 0; j < d; ) {
    size_t vl;
    __asm__ volatile("vsetvli %0, %1, e8, m2, ta, ma" : "=r"(vl) : "r"(d - j));

    const int8_t *vp = V8 + j;
    const int8_t *wp = W8;
    int32_t *ap = acc + j;
    float   *op = out + j;
    size_t   nb  = S >> 2;         // 4-row blocks
    size_t   rem = S & 3;          // leftover rows (0..3)
    size_t   w0, w1, w2, w3;

    __asm__ volatile(
      "vsetvli  zero, %[vl], e32, m8, ta, ma\n\t"
      "vmv.v.i  v24, 0\n\t"
      "vmv.v.i  v16, 0\n\t"
      "vsetvli  zero, %[vl], e8, m2, ta, ma\n\t"
      "beqz     %[nb], 3f\n"
      "1:\n\t"
      "lbu      %[w0], 0(%[wp])\n\t"
      "lbu      %[w1], 1(%[wp])\n\t"
      "lbu      %[w2], 2(%[wp])\n\t"
      "lbu      %[w3], 3(%[wp])\n\t"
      "vle8.v   v0, (%[vp])\n\t"
      "add      %[vp], %[vp], %[d]\n\t"
      "vle8.v   v2, (%[vp])\n\t"
      "add      %[vp], %[vp], %[d]\n\t"
      "vwmul.vx  v8, v0, %[w0]\n\t"
      "vwmacc.vx v8, %[w1], v2\n\t"
      "vle8.v   v4, (%[vp])\n\t"
      "add      %[vp], %[vp], %[d]\n\t"
      "vle8.v   v6, (%[vp])\n\t"
      "add      %[vp], %[vp], %[d]\n\t"
      "vwmul.vx  v12, v4, %[w2]\n\t"
      "vwmacc.vx v12, %[w3], v6\n\t"
      "addi     %[wp], %[wp], 4\n\t"
      "addi     %[nb], %[nb], -1\n\t"
      "vsetvli  zero, %[vl], e16, m4, ta, ma\n\t"
      "vwadd.wv v24, v24, v8\n\t"
      "vwadd.wv v16, v16, v12\n\t"
      "vsetvli  zero, %[vl], e8, m2, ta, ma\n\t"
      "bnez     %[nb], 1b\n"
      "3:\n\t"
      "beqz     %[rem], 2f\n"
      "5:\n\t"
      "lbu      %[w0], 0(%[wp])\n\t"
      "vle8.v   v0, (%[vp])\n\t"
      "addi     %[wp], %[wp], 1\n\t"
      "add      %[vp], %[vp], %[d]\n\t"
      "addi     %[rem], %[rem], -1\n\t"
      "vwmul.vx v8, v0, %[w0]\n\t"
      "vsetvli  zero, %[vl], e16, m4, ta, ma\n\t"
      "vwadd.wv v24, v24, v8\n\t"
      "vsetvli  zero, %[vl], e8, m2, ta, ma\n\t"
      "bnez     %[rem], 5b\n"
      "2:\n\t"
      "vsetvli  zero, %[vl], e32, m8, ta, ma\n\t"
      "vadd.vv  v24, v24, v16\n\t"
      "vse32.v  v24, (%[ap])\n\t"
      "vfcvt.f.x.v v24, v24\n\t"
      "vfmul.vf v24, v24, %[ws]\n\t"
      "vse32.v  v24, (%[op])\n\t"
      : [wp] "+r"(wp), [vp] "+r"(vp), [nb] "+r"(nb), [rem] "+r"(rem),
        [w0] "=&r"(w0), [w1] "=&r"(w1), [w2] "=&r"(w2), [w3] "=&r"(w3)
      : [vl] "r"(vl), [d] "r"(d), [ap] "r"(ap), [op] "r"(op), [ws] "f"(wscale)
      : "memory",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

    j += vl;
  }
}

// ===========================================================================
// THE ENTRY POINT — this is what the timed region measures
// ===========================================================================
// BASELINE SCHEDULE: strictly sequential. `lf_gemv_gemmini` ends in
// `gemmini_fence()`, so not one vector instruction issues until the last
// weight byte has landed and the accumulator has been moved out.
//
// Required outputs and the ONLY dependency between them:
//     C[0..M)          from A, B                       (Gemmini)
//     scores[0..S)     from K8, kscale, q8             (Saturn)
//     probs[0..S)      from scores          <-- DEPENDS on scores
//     pvacc[0..d), pvout[0..d)  from W8, V8            (Saturn)
// Everything else is free to be reordered, interleaved or split.
static void llama_layer_fused(size_t N, size_t Kdim, size_t M,
                              const elem_t *A, const elem_t *B, acc_t *C,
                              size_t S, size_t d,
                              const elem_t *K8, const float *kscale,
                              const elem_t *q8, float *scores, float *probs,
                              const elem_t *W8, const elem_t *V8,
                              int32_t *pvacc, float *pvout,
                              float qscale, float scale, float wscale)
{
    lf_gemv_gemmini(N, Kdim, M, A, B, C);
    lf_attn_scores(S, d, K8, kscale, q8, scores, qscale, scale);
    lf_softmax(S, scores, probs);
    lf_attn_pv(S, d, W8, V8, pvacc, pvout, wscale);
}

#undef LF_MIN
#undef LF_PREFETCH_BUDGET_CYCLES
#undef LF_LINE_BYTES

#endif  // LLAMA_LAYER_FUSED_N1_KERNEL_H
