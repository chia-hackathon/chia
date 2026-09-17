// See LICENSE for license details.
//
// llama-attn-pv-q8kv — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_ATTN_PV_Q8KV_H
#define LLAMA_ATTN_PV_Q8KV_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — probs @ V for ONE query head of a Llama-3.2-1B decode step, with an
// **int8 V cache**. Same arithmetic as `llama-attn-pv`, but V is stored
// quantised, which is what loop/llama_model.py's ModelSpec assumes
// (`kv_bytes = 1`). The V slab is 32 KiB here instead of 128 KiB.
//
// Quantisation: PER-ROW (per KV position) symmetric int8 with an fp32 scale,
// the same layout as the K cache — a row is one token's value vector, written
// once at append time.
//
//   out[j] = sum_s P[s] * vscale[s] * (float)V8[s*d + j]
//
// The scale is per ROW and the reduction is over rows, so it does NOT factor
// out of the sum the way it does in attn-scores: it folds into the scalar
// multiplier of each row instead (`P[s] * vscale[s]`), which costs one scalar
// multiply per row and nothing per element.
//
// `P` stays fp32: it is a fresh softmax output, 2 KiB, and quantising it buys
// nothing in bandwidth.
// ---------------------------------------------------------------------------
#define LQP_S 512
#define LQP_D 64

// ---------------------------------------------------------------------------
// Memory map — outside the ELF.
//   0x81000000  P      : LQP_S float                    (2 KiB)
//   0x81010000  V8     : LQP_S x LQP_D int8, stride d   (32 KiB)
//   0x81020000  vscale : LQP_S float
//   0x81030000  out    : LQP_D float
// ---------------------------------------------------------------------------
#define LQP_P_ADDR  0x81000000UL
#define LQP_V_ADDR  0x81010000UL
#define LQP_VS_ADDR 0x81020000UL
#define LQP_O_ADDR  0x81030000UL

#define LQP_P  ((float  *)LQP_P_ADDR)
#define LQP_V8 ((int8_t *)LQP_V_ADDR)
#define LQP_VS ((float  *)LQP_VS_ADDR)
#define LQP_O  ((float  *)LQP_O_ADDR)

// `out` is zeroed by the caller.
void llama_attn_pv_q8kv(size_t S, size_t d, const float *P, const int8_t *V8,
                        const float *vscale, float *out);

#endif  // LLAMA_ATTN_PV_Q8KV_H
