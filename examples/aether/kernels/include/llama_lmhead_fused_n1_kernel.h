// See LICENSE for license details.
//
// llama-lmhead-fused (N = 1, M = 2048) — THE FILE UNDER OPTIMIZATION.
//
// This is the only file whose bytes reach the compiler from your workspace.
// `bareMetalC/llama_lmhead_fused_n1.c`, `include/llama_lmhead_fused_body.h`
// (main, data generator, L2 flush, golden models, self-checks) and
// `include/gemmini.h` / `include/gemmini_params.h` are reassembled from the
// pristine, hardware-derived tree at every build.
//
// WHAT THIS BASELINE IS
// ---------------------
// The best-known kernel for each piece of work, run STRICTLY BACK TO BACK,
// with Gemmini fully fenced before the first vector instruction:
//
//   1. lhk_gemv_gemmini — the `llama-q8-gemv-gemmini-lmhead` Round-4 winner
//                         (634,507 cycles with a WARM tail; this harness
//                         flushes the L2 first, so expect MORE here)
//   2. lhk_rmsnorm      — the `llama-rmsnorm` incumbent (5,454 cycles warm,
//                         RVV LMUL=1, never beaten in its own round)
//   3. lhk_quantize     — fp32 -> int8, RNE, clamp to +-127
//   4. lhk_argmax       — running argmax over 2048 int32 logits
//
// The objective is the SCHEDULE, not the inner loops. Pieces 2-4 are a small
// fraction of the total and there is some headroom in them (rmsnorm's own
// roofline is 1,536 against 5,454), but shaving them is NOT what this entry
// measures: it measures how much of their cost can be made to disappear into
// the shadow of a 4 MiB weight DMA. If you rewrite an inner loop, the
// serialised reference moves too, and the exposure ratio this entry exists to
// report becomes incomparable with `llama-layer-fused-n1`'s 0.611. So: change
// the schedule, keep the work.
//
// Read the registry notes before editing. In particular: the exposure ratio,
// not the cycle count, is the deliverable.

#ifndef LLAMA_LMHEAD_FUSED_N1_KERNEL_H
#define LLAMA_LMHEAD_FUSED_N1_KERNEL_H

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "include/gemmini.h"

#define LHK_MIN(a, b) ((a) < (b) ? (a) : (b))

// ===========================================================================
// 1. Gemmini LM-head GEMV — `llama-q8-gemv-gemmini-lmhead` best (634,507)
// ===========================================================================
// Hand-issued, software-pipelined Gemmini command stream, ONE output strip,
// K reduction in REVERSE order.
//
//   * Each output tile is 1 x DIM int32 and needs ONE accumulator row, so all
//     M/DIM = 128 tiles fit in the accumulator at once: the whole width of B
//     is consumed in a single strip with K outermost. No strip boundaries ->
//     no prologue / epilogue / mvout bubbles.
//   * B is streamed in as 4-row x 64 B mvins. That request shape is the best
//     measured one on this SoC by a wide margin: 8- and 16-row commands pile
//     16 requests on one L2 bank and measure 690k / 695k. This was confirmed
//     twice (lmhead's own curve, and `llama-layer-fused-n1` iter5 vs iter7).
//     DO NOT re-test 8 or 16 rows.
//   * the chunk visiting order within a row group is permuted by a coprime
//     step, c_i = (i * c_step) mod n_chunks, so consecutive commands are not
//     512 B apart and therefore rotate over L2 banks.
//   * K blocks are processed from the LAST one down. In the lmhead harness
//     that harvested the warm tail B's fill left in the L2. **This harness
//     flushes the L2 first, so that credit is GONE** — whether descending
//     still helps is an open question and a legitimate one-line experiment.
//   * loads for K block s are interleaved with the executes of block s-1 in
//     batches of 8 so neither queue starves.
//
// NOTE: ends with `gemmini_fence()`. That fence is the thing to move.

// Rows of B per mvin command: 4 rows x 64 B.
#define LHK_B_ROWS 4
// Loads issued between two batches of executes.
#define LHK_LD_BATCH 8

static void lhk_gemv_fallback(size_t N, size_t K, size_t M,
                              const elem_t *A, const elem_t *B, acc_t *C)
{
    tiled_matmul_auto(N, M, K, A, B, NULL, (void *)C,
        K, M, M, M,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false,
        true, false, 0, WS);
}

static size_t lhk_gcd(size_t a, size_t b)
{
    while (b) { size_t t = a % b; a = b; b = t; }
    return a;
}

// C[N x M] (acc_t, row-major, stride M)
//   = A[N x K] (elem_t, row-major, stride K) * B[K x M] (elem_t, stride M)
// No bias, no activation, no output scaling: exact int32 result.
static void lhk_gemv_gemmini(size_t N, size_t K, size_t M,
                             const elem_t *A, const elem_t *B, acc_t *C)
{
    if (N != 1 || K % DIM != 0 || M % DIM != 0 || DIM % LHK_B_ROWS != 0
        || M % (MAX_BLOCK_LEN * DIM) != 0) {
        lhk_gemv_fallback(N, K, M, A, B, C);
        gemmini_fence();
        return;
    }

    const size_t stride_A = K, stride_B = M, stride_C = M;
    const size_t K_t = K / DIM;                 // K blocks
    const size_t J_t = M / DIM;                 // output tiles (one acc row each)

    // Scratchpad / accumulator geometry (all from gemmini_params.h).
    const size_t spad_rows = (size_t)BANK_NUM * BANK_ROWS;
    const size_t slot_rows = J_t * DIM;                      // spad rows per K block
    const size_t a_rows    = K_t * DIM;                      // A: block k at row k*DIM
    const uint32_t A_base  = 0;
    const uint32_t B_base  = (uint32_t)a_rows;
    const size_t n_slots   = (spad_rows > a_rows) ? (spad_rows - a_rows) / slot_rows : 0;
    if (n_slots < 2 || J_t > (size_t)ACC_ROWS) {
        lhk_gemv_fallback(N, K, M, A, B, C);
        gemmini_fence();
        return;
    }

    const size_t groups_per_block = DIM / LHK_B_ROWS;                    // 4
    const size_t n_chunks = J_t / MAX_BLOCK_LEN;                         // 32
    const size_t n_ld     = groups_per_block * n_chunks;                 // mvins per K block

    size_t c_step = (n_chunks / 4) | 1;
    if (lhk_gcd(c_step, n_chunks) != 1) c_step = 1;

    // Accumulator addresses: acc | accumulate | full-width-read flags.
    const uint32_t C_acc_start = (3u << (ADDR_LEN - 2)) | (1u << (ADDR_LEN - 3));

    gemmini_extended_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION & 3, 0, 1, false, false);
    gemmini_extended_config_st(stride_C * sizeof(acc_t), NO_ACTIVATION & 3, ACC_SCALE_IDENTITY);
    gemmini_extended3_config_ld(stride_A * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(stride_B * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);

    // A once: row 0 of block k lands at spad row A_base + k*DIM.
    for (size_t k = 0; k < K_t; k += MAX_BLOCK_LEN) {
        const size_t blocks = LHK_MIN((size_t)MAX_BLOCK_LEN, K_t - k);
        gemmini_extended_mvin(A + k * DIM, A_base + k * DIM, blocks * DIM, 1);
    }

    // Sequence step s handles K block kblk = K_t-1-s in slot s % n_slots.
#define LHK_KBLK(s)  (K_t - 1 - (s))
#define LHK_SLOT(s)  ((s) % n_slots)

#define LHK_LOAD(s, l)                                                          \
    do {                                                                        \
        const size_t g_ = (l) / n_chunks;                                       \
        const size_t c_ = (((l) % n_chunks) * c_step) % n_chunks;               \
        const elem_t *dram_ = B + (LHK_KBLK(s) * DIM + g_ * LHK_B_ROWS) * stride_B \
                                + c_ * MAX_BLOCK_LEN * DIM;                     \
        const uint32_t sp_ = B_base + (uint32_t)(LHK_SLOT(s) * slot_rows        \
                                    + c_ * MAX_BLOCK_LEN * DIM + g_ * LHK_B_ROWS); \
        gemmini_extended_mvin2(dram_, sp_, MAX_BLOCK_LEN * DIM, LHK_B_ROWS);    \
    } while (0)

#define LHK_EX(s, j)                                                            \
    do {                                                                        \
        const uint32_t bsp_ = B_base + (uint32_t)(LHK_SLOT(s) * slot_rows + (j) * DIM); \
        const uint32_t asp_ = A_base + (uint32_t)(LHK_KBLK(s) * DIM);           \
        uint32_t out_ = C_acc_start + (uint32_t)(j);                            \
        if ((s) == 0) out_ &= ~(1u << (ADDR_LEN - 2)); /* overwrite */          \
        gemmini_extended_preload(bsp_, out_, DIM, DIM, DIM, 1);                 \
        gemmini_extended_compute_preloaded(asp_, GARBAGE_ADDR, DIM, 1, DIM, DIM); \
    } while (0)

    // Prologue: loads of the first K block.
    for (size_t l = 0; l < n_ld; l++) LHK_LOAD(0, l);

    // Steady state: loads of step s interleaved with executes of step s-1.
    for (size_t s = 1; s < K_t; s++) {
        size_t t = 0;
        for (size_t l = 0; l < n_ld; l++) {
            LHK_LOAD(s, l);
            if ((l + 1) % LHK_LD_BATCH == 0 || l + 1 == n_ld) {
                const size_t te = ((l + 1) * J_t) / n_ld;
                for (; t < te; t++) LHK_EX(s - 1, t);
            }
        }
    }
    // Epilogue: executes of the last step.
    for (size_t t = 0; t < J_t; t++) LHK_EX(K_t - 1, t);

    // Move the finished 1 x M int32 result out: acc row j -> C[j*DIM .. +DIM).
    for (size_t j = 0; j < J_t; j++)
        gemmini_extended_mvout(C + j * DIM, C_acc_start + (uint32_t)j, DIM, 1);

#undef LHK_KBLK
#undef LHK_SLOT
#undef LHK_LOAD
#undef LHK_EX

    gemmini_fence();
}

// ===========================================================================
// 2. Saturn final RMSNorm — `llama-rmsnorm` incumbent (5,454)
// ===========================================================================
// ss = (1/n) * sum_i x[i]^2 ; y[i] = x[i] * rsqrt(ss + eps) * w[i].
// No mean subtraction. Two separate passes over `x`, LMUL=1, one accumulator.
// The reduction may be reassociated (the harness gate is 1e-4 relative) but
// must cover all n terms.
static void lhk_rmsnorm(size_t n, const float *x, const float *w, float *y,
                        float eps)
{
    const size_t vl    = __riscv_vsetvlmax_e32m1();   // VLEN=256 -> 8 fp32
    const size_t nmain = n - (n % vl);

    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (size_t i = 0; i < nmain; i += vl) {
        vfloat32m1_t v = __riscv_vle32_v_f32m1(x + i, vl);
        acc = __riscv_vfmacc_vv_f32m1(acc, v, v, vl);
    }
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t red  = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vl);
    float ss = __riscv_vfmv_f_s_f32m1_f32(red);
    for (size_t i = nmain; i < n; i++)
        ss += x[i] * x[i];

    const float scale = 1.0f / __builtin_sqrtf(ss / (float)n + eps);

    for (size_t i = 0; i < nmain; i += vl) {
        vfloat32m1_t v  = __riscv_vle32_v_f32m1(x + i, vl);
        vfloat32m1_t vw = __riscv_vle32_v_f32m1(w + i, vl);
        vfloat32m1_t o  = __riscv_vfmul_vv_f32m1(v, vw, vl);
        o = __riscv_vfmul_vf_f32m1(o, scale, vl);
        __riscv_vse32_v_f32m1(y + i, o, vl);
    }
    for (size_t i = nmain; i < n; i++)
        y[i] = x[i] * scale * w[i];
}

// ===========================================================================
// 3. Saturn fp32 -> int8 quantiser (the NEXT token's LM-head input)
// ===========================================================================
// q[i] = clamp(rne(y[i] * qscale), -127, +127).
// e32m2 (vl = 16 at VLEN 256), RNE via `vfcvt.x.f.v` under the default FRM,
// saturation with signed vmax/vmin, then two `vnsrl.wi ..., 0` narrowing
// steps. The narrowing is a plain low-byte truncation and that is exact:
// the values are already inside [-127, 127] after the clamp.
// Written as asm rather than intrinsics so the narrowing spelling does not
// depend on which revision of the intrinsics the toolchain implements.
static void lhk_quantize(size_t n, const float *y, int8_t *q, float qscale)
{
    const long lo = -127, hi = 127;
    size_t i = 0;
    while (i < n) {
        size_t vl;
        __asm__ volatile(
            "vsetvli %[vl], %[avl], e32, m2, ta, ma\n\t"
            "vle32.v    v8, (%[yp])\n\t"
            "vfmul.vf   v8, v8, %[qs]\n\t"
            "vfcvt.x.f.v v8, v8\n\t"
            "vmax.vx    v8, v8, %[lo]\n\t"
            "vmin.vx    v8, v8, %[hi]\n\t"
            "vsetvli    x0, %[avl], e16, m1, ta, ma\n\t"
            "vnsrl.wi   v12, v8, 0\n\t"
            "vsetvli    x0, %[avl], e8, mf2, ta, ma\n\t"
            "vnsrl.wi   v13, v12, 0\n\t"
            "vse8.v     v13, (%[qp])\n\t"
            : [vl] "=&r"(vl)
            : [avl] "r"(n - i), [yp] "r"(y + i), [qp] "r"(q + i),
              [qs] "f"(qscale), [lo] "r"(lo), [hi] "r"(hi)
            : "v8","v9","v10","v11","v12","v13","memory");
        i += vl;
    }
}

// ===========================================================================
// 4. Saturn running argmax over the previous tile's logits
// ===========================================================================
// Two passes over `lg`. Pass 1 keeps an elementwise running max in an i32m4
// group and reduces it with one `vredmax.vs`. Pass 2 finds the FIRST index
// whose value equals that max, with `vmseq.vx` + `vfirst.m`, so the tie-break
// is lowest-index and matches the harness reference exactly. `amax[0]` is the
// incoming running value and `amax[1]` the incoming running index, so this
// composes across tiles.
static void lhk_argmax(size_t n, const int32_t *lg, int32_t *amax)
{
    if (n == 0) return;

    long best;
    {
        const long seed = -2147483647L - 1L;
        size_t i = 0, vl;
        __asm__ volatile("vsetvli %[vl], %[avl], e32, m4, ta, ma\n\t"
                         "vmv.v.x v8, %[sd]\n\t"
                         : [vl] "=&r"(vl)
                         : [avl] "r"(n), [sd] "r"(seed)
                         : "v8","v9","v10","v11");
        while (i < n) {
            __asm__ volatile(
                "vsetvli %[vl], %[avl], e32, m4, tu, ma\n\t"
                "vle32.v v12, (%[p])\n\t"
                "vmax.vv v8, v8, v12\n\t"
                : [vl] "=&r"(vl)
                : [avl] "r"(n - i), [p] "r"(lg + i)
                : "v8","v9","v10","v11","v12","v13","v14","v15","memory");
            i += vl;
        }
        __asm__ volatile(
            "vsetvli x0, %[avl], e32, m4, ta, ma\n\t"
            "vmv.s.x v17, %[sd]\n\t"
            "vredmax.vs v16, v8, v17\n\t"
            "vmv.x.s %[out], v16\n\t"
            : [out] "=r"(best)
            : [avl] "r"(n), [sd] "r"(seed)
            : "v8","v9","v10","v11","v16","v17");
    }
    const int32_t bestv = (int32_t)best;

    size_t idx = 0;
    {
        size_t i = 0, vl;
        long f = -1;
        while (i < n) {
            __asm__ volatile(
                "vsetvli %[vl], %[avl], e32, m4, ta, ma\n\t"
                "vle32.v v12, (%[p])\n\t"
                "vmseq.vx v0, v12, %[b]\n\t"
                "vfirst.m %[f], v0\n\t"
                : [vl] "=&r"(vl), [f] "=&r"(f)
                : [avl] "r"(n - i), [p] "r"(lg + i), [b] "r"(best)
                : "v0","v12","v13","v14","v15","memory");
            if (f >= 0) { idx = i + (size_t)f; break; }
            i += vl;
        }
    }

    if (bestv > amax[0]) { amax[0] = bestv; amax[1] = (int32_t)idx; }
}

// ===========================================================================
// THE ENTRY POINT — this is what the timed region measures
// ===========================================================================
// BASELINE SCHEDULE: strictly sequential. `lhk_gemv_gemmini` ends in
// `gemmini_fence()`, so not one vector instruction issues until the last
// weight byte has landed and the accumulator has been moved out.
//
// Required outputs and the ONLY dependency between them:
//     C[0..M)        from A, B                        (Gemmini)
//     xn[0..H)       from X, G, eps                   (Saturn)
//     xq[0..H)       from xn, qscale   <-- DEPENDS on xn
//     amax[0..2)     from LG, amax                    (Saturn)
// Nothing in the Saturn half reads C, and nothing in the Gemmini half reads
// xn / xq / amax. Everything except xn -> xq is free to be reordered,
// interleaved or split.
static void llama_lmhead_fused(size_t N, size_t Kdim, size_t M,
                               const elem_t *A, const elem_t *B, acc_t *C,
                               size_t H, const float *X, const float *G,
                               float *xn, elem_t *xq, float eps, float qscale,
                               size_t L, const acc_t *LG, int32_t *amax)
{
    lhk_gemv_gemmini(N, Kdim, M, A, B, C);
    lhk_rmsnorm(H, X, G, xn, eps);
    lhk_quantize(H, xn, (int8_t *)xq, qscale);
    lhk_argmax(L, (const int32_t *)LG, amax);
}

#undef LHK_MIN
#undef LHK_B_ROWS
#undef LHK_LD_BATCH

#endif  // LLAMA_LMHEAD_FUSED_N1_KERNEL_H
