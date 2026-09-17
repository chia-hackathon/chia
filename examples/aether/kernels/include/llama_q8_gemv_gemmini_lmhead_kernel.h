// See LICENSE for license details.
//
// llama-q8-gemv-gemmini-lmhead — THE FILE UNDER OPTIMIZATION.
//
// This is the only file whose bytes reach the compiler from your workspace.
// `bareMetalC/llama_q8_gemv_gemmini_lmhead.c`,
// `include/llama_q8_gemv_gemmini_lmhead_body.h` (main, data generator, golden
// model, self-check) and `include/gemmini.h` / `include/gemmini_params.h` are
// reassembled from the pristine, hardware-derived tree at every build.
//
// Shape: N = 1, K = 2048, M = 2048 — one tile of the Llama-3.2-1B LM head.
// Four times the weight footprint of `llama-q8-gemv-gemmini-n1` (4 MiB vs
// 1 MiB) against the same single valid array row, so the fixed per-call and
// per-tile overheads are amortized much further and what is left is almost
// pure weight-DMA behaviour.
//
// Baseline: the stock `tiled_matmul_auto` call, weight-stationary, no bias,
// int32 (`full_C`) output.

#ifndef LLAMA_Q8_GEMV_GEMMINI_LMHEAD_KERNEL_H
#define LLAMA_Q8_GEMV_GEMMINI_LMHEAD_KERNEL_H

#include "include/gemmini.h"

// C[N x M] (acc_t, row-major, stride M)
//   = A[N x K] (elem_t, row-major, stride K)
//   * B[K x M] (elem_t, row-major, stride M)
//
// No bias, no activation, no output scaling: the accumulator value itself is
// what lands in C, so the result is exact int32 and the self-check compares it
// against a plain scalar dot product.
//
// `dim_I = 1` is NOT a multiple of DIM = 16. `tiled_matmul_auto` rounds it up
// internally (`dim_I_padded`) and zero-pads; the padded rows are computed and
// discarded, which is exactly the inefficiency this kernel measures.
static void llama_q8_gemv_gemmini(size_t N, size_t K, size_t M,
                                  const elem_t *A, const elem_t *B, acc_t *C)
{
    tiled_matmul_auto(
        /* dim_I */ N, /* dim_J */ M, /* dim_K */ K,
        A, B, /* D */ NULL, /* C */ (void *)C,
        /* stride_A */ K, /* stride_B */ M, /* stride_D */ M, /* stride_C */ M,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, /* repeating_bias */ false,
        /* transpose_A */ false, /* transpose_B */ false,
        /* full_C */ true, /* low_D */ false,
        /* weightA */ 0,
        WS);
}

#endif  // LLAMA_Q8_GEMV_GEMMINI_LMHEAD_KERNEL_H
