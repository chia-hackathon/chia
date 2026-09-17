// See LICENSE for license details.
//
// llama-attn-pv — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_ATTN_PV_H
#define LLAMA_ATTN_PV_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — probs @ V for ONE query head of a Llama-3.2-1B decode step.
//
//   out[j] = sum_{s=0}^{S-1} P[s] * V[s*d + j]        j in [0, d)
//
// S = 512 (KV-cache length), d = 64 (head_dim). 32 query heads per layer, so
// the projection multiplies this per-head cost by `calls = 32`.
//
// dtype: fp32 for P and V (see llama_attn_scores.h for why fp32 is the
// conservative, honest choice while the elementwise path is fp32).
// ---------------------------------------------------------------------------
#define LPV_S 512
#define LPV_D 64

// ---------------------------------------------------------------------------
// Memory map — outside the ELF.
//   0x81000000  P   : LPV_S float             (2 KiB)
//   0x81010000  V   : LPV_S x LPV_D float     (128 KiB), row-major, stride d
//   0x81040000  out : LPV_D float
// ---------------------------------------------------------------------------
#define LPV_P_ADDR 0x81000000UL
#define LPV_V_ADDR 0x81010000UL
#define LPV_O_ADDR 0x81040000UL

#define LPV_P ((float *)LPV_P_ADDR)
#define LPV_V ((float *)LPV_V_ADDR)
#define LPV_O ((float *)LPV_O_ADDR)

// `out` is zeroed by the caller.
void llama_attn_pv(size_t S, size_t d, const float *P, const float *V,
                   float *out);

#endif  // LLAMA_ATTN_PV_H
