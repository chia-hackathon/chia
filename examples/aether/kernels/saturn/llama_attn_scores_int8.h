// See LICENSE for license details.
//
// llama-attn-scores-int8 — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_ATTN_SCORES_INT8_H
#define LLAMA_ATTN_SCORES_INT8_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — QK^T/sqrt(d) for ONE query head of a Llama-3.2-1B decode step, FULLY
// QUANTISED: int8 K cache AND int8 query, int32 accumulation, one fp32 rescale
// per output.
//
// This is the third of three measurements of the same 512x64 shape:
//   llama-attn-scores        fp32 K, fp32 q   (128 KiB cache, no conversion)
//   llama-attn-scores-q8kv   int8 K, fp32 q   ( 32 KiB cache, int8->fp32 per elem)
//   llama-attn-scores-int8   int8 K, int8 q   ( 32 KiB cache, NO conversion)  <-- this
//
// The point of this one is that quantising BOTH sides removes the per-element
// `vsext`+`vfcvt` that made the q8kv version compute-bound: the dot product
// stays in the integer pipe (`vwmul` + `vwadd.wv` into int32, the
// `llama-q8-gemv` inner loop) and the only floating point left is one multiply
// per output row.
//
//   scores[s] = (float)(sum_j K8[s*d + j] * q8[j]) * kscale[s] * qscale * scale
//
// Quantisation: K is PER-ROW symmetric int8 (one scale per KV position, the
// natural KV-cache layout); q is PER-VECTOR symmetric int8 (one scale for the
// whole 64-long query, because it is produced fresh by the q_proj GEMV each
// step). int32 is wide enough: |sum| <= 64 * 127 * 127 = 1,032,256 < 2^31.
// ---------------------------------------------------------------------------
#define LSI_S     512           // KV-cache length
#define LSI_D     64            // head_dim
#define LSI_SCALE 0.125f        // 1 / sqrt(64)

// ---------------------------------------------------------------------------
// Memory map — outside the ELF. The `Kf`/`qf` buffers are the HARNESS's fp32
// source data: it generates fp32, quantises it, and keeps the fp32 around so
// it can report the quantisation error against a true fp32 reference. The
// kernel never sees them.
//   0x81000000  K8     : LSI_S x LSI_D int8, row-major, stride LSI_D (32 KiB)
//   0x81010000  kscale : LSI_S float
//   0x81020000  q8     : LSI_D int8
//   0x81030000  scores : LSI_S float
//   0x81040000  Kf     : LSI_S x LSI_D float (128 KiB)  [harness only]
//   0x81080000  qf     : LSI_D float                    [harness only]
// ---------------------------------------------------------------------------
#define LSI_K_ADDR  0x81000000UL
#define LSI_KS_ADDR 0x81010000UL
#define LSI_Q_ADDR  0x81020000UL
#define LSI_O_ADDR  0x81030000UL
#define LSI_KF_ADDR 0x81040000UL
#define LSI_QF_ADDR 0x81080000UL

#define LSI_K8 ((int8_t *)LSI_K_ADDR)
#define LSI_KS ((float  *)LSI_KS_ADDR)
#define LSI_Q8 ((int8_t *)LSI_Q_ADDR)
#define LSI_O  ((float  *)LSI_O_ADDR)
#define LSI_KF ((float  *)LSI_KF_ADDR)
#define LSI_QF ((float  *)LSI_QF_ADDR)

// The int32 accumulation must NOT saturate or round.
void llama_attn_scores_int8(size_t S, size_t d, const int8_t *K8,
                            const float *kscale, const int8_t *q8,
                            float *scores, float qscale, float scale);

#endif  // LLAMA_ATTN_SCORES_INT8_H
