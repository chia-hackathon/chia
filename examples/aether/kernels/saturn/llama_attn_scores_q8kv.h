// See LICENSE for license details.
//
// llama-attn-scores-q8kv — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_ATTN_SCORES_Q8KV_H
#define LLAMA_ATTN_SCORES_Q8KV_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — QK^T/sqrt(d) for ONE query head of a Llama-3.2-1B decode step, with
// an **int8 KV cache**. Same arithmetic as `llama-attn-scores`, but K is
// stored quantised, which is what loop/llama_model.py's ModelSpec assumes
// (`kv_bytes = 1`). The K slab is 32 KiB here instead of 128 KiB.
//
// Quantisation: PER-ROW (per KV position) symmetric int8 with an fp32 scale.
// That is the standard layout for a KV cache — a row is one token's key
// vector, it is written once when the token is appended and never rewritten,
// so its scale can be computed at append time and stored beside it.
//
//   scores[s] = kscale[s] * (sum_j (float)K8[s*d + j] * q[j]) * scale
//
// `q` stays fp32: the query is produced fresh by the q_proj GEMV each step and
// is 256 bytes, so quantising it buys nothing in bandwidth. This means the
// kernel MUST convert every K byte to fp32 — see the notes in kernels.py for
// why that makes this version compute-bound where the fp32 one was balanced.
// ---------------------------------------------------------------------------
#define LQS_S     512           // KV-cache length
#define LQS_D     64            // head_dim
#define LQS_SCALE 0.125f        // 1 / sqrt(64)

// ---------------------------------------------------------------------------
// Memory map — outside the ELF.
//   0x81000000  K8     : LQS_S x LQS_D int8, row-major, row stride LQS_D (32 KiB)
//   0x81010000  kscale : LQS_S float   (one dequant scale per KV position)
//   0x81020000  q      : LQS_D float
//   0x81030000  scores : LQS_S float
// ---------------------------------------------------------------------------
#define LQS_K_ADDR  0x81000000UL
#define LQS_KS_ADDR 0x81010000UL
#define LQS_Q_ADDR  0x81020000UL
#define LQS_O_ADDR  0x81030000UL

#define LQS_K8 ((int8_t *)LQS_K_ADDR)
#define LQS_KS ((float  *)LQS_KS_ADDR)
#define LQS_Q  ((float  *)LQS_Q_ADDR)
#define LQS_O  ((float  *)LQS_O_ADDR)

void llama_attn_scores_q8kv(size_t S, size_t d, const int8_t *K8,
                            const float *kscale, const float *q,
                            float *scores, float scale);

#endif  // LLAMA_ATTN_SCORES_Q8KV_H
