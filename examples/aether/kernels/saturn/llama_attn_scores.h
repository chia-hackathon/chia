// See LICENSE for license details.
//
// llama-attn-scores — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_ATTN_SCORES_H
#define LLAMA_ATTN_SCORES_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — QK^T/sqrt(d) for ONE query head of a Llama-3.2-1B decode step.
//
// head_dim 64, KV-cache length S = 512 (the projection scenario's --S). A
// decode step runs 32 query heads per layer; because of GQA (8 KV heads, group
// size 4) four query heads share one K-cache slab, so the per-head cost
// measured here is the unit the projection multiplies by `calls = 32`.
//
//   scores[s] = (sum_j K[s*d + j] * q[j]) / sqrt(d)      s in [0, S)
//
// sqrt(64) = 8 exactly, so the scale is 0.125 and is exact in fp32.
//
// dtype: fp32 for BOTH the K cache and the query. loop/llama_model.py's
// ModelSpec assumes an int8 KV cache (kv_bytes = 1); an fp32 cache moves 4x
// the bytes, so this measurement is the CONSERVATIVE end of the range. It is
// the honest one to use while the rest of the elementwise path is fp32.
// ---------------------------------------------------------------------------
#define LAS_S     512           // KV-cache length
#define LAS_D     64            // head_dim
#define LAS_SCALE 0.125f        // 1 / sqrt(64)

// ---------------------------------------------------------------------------
// Memory map — outside the ELF.
//   0x81000000  K      : LAS_S x LAS_D float, row-major, row stride LAS_D
//                        (128 KiB — bigger than the 512 KiB L2? no, but it is
//                         a cold stream on the first and only pass)
//   0x81040000  q      : LAS_D float
//   0x81050000  scores : LAS_S float
// ---------------------------------------------------------------------------
#define LAS_K_ADDR 0x81000000UL
#define LAS_Q_ADDR 0x81040000UL
#define LAS_S_ADDR 0x81050000UL

#define LAS_K ((float *)LAS_K_ADDR)
#define LAS_Q ((float *)LAS_Q_ADDR)
#define LAS_O ((float *)LAS_S_ADDR)

void llama_attn_scores(size_t S, size_t d, const float *K, const float *q,
                       float *scores, float scale);

#endif  // LLAMA_ATTN_SCORES_H
