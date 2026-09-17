// See LICENSE for license details.
//
// llama-attn-pv-int8 — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_ATTN_PV_INT8_H
#define LLAMA_ATTN_PV_INT8_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — probs@V for ONE query head of a Llama-3.2-1B decode step, FULLY
// QUANTISED: int8 V cache AND int8 probabilities, int32 accumulation, one
// fp32 rescale per output.
//
// Third of three measurements of the same 512x64 shape:
//   llama-attn-pv        fp32 V, fp32 P   (128 KiB cache, no conversion)
//   llama-attn-pv-q8kv   int8 V, fp32 P   ( 32 KiB cache, int8->fp32 per elem)
//   llama-attn-pv-int8   int8 V, int8 P   ( 32 KiB cache, NO conversion)  <-- this
//
//   out[j] = (float)(sum_s W8[s] * V8[s*d + j]) * wscale
//
// WHERE DID THE PER-ROW V SCALE GO?
//   `attn_pv` reduces ACROSS rows, so a per-row dequant scale does not factor
//   out of the sum the way `attn_scores`'s does — an int32 accumulation over
//   rows is only possible if every row shares one scale. The standard fix, and
//   the one used here, is to FOLD the per-row V scale into the probability
//   vector before quantising it:
//
//       w[s]  = P[s] * vscale[s]            (fp32, 512 values, done once)
//       W8[s] = round(w[s] / wscale)        (int8, ONE scale for the vector)
//
//   That is cheap and correct: `w` is 512 values against V's 32,768, so the
//   fold costs 512 scalar multiplies outside the kernel and buys a pure
//   integer inner loop. This is why the kernel below never sees `vscale`.
//
// int32 is wide enough: |sum| <= 512 * 127 * 127 = 8,258,048 < 2^31.
// ---------------------------------------------------------------------------
#define LPI_S 512
#define LPI_D 64

// ---------------------------------------------------------------------------
// Memory map — outside the ELF. `Vf`/`Pf`/`vscale`/`wf` are the HARNESS's fp32
// source data, kept so it can report quantisation error against a true fp32
// reference. The kernel never sees them.
//   0x81000000  W8     : LPI_S int8            (folded, single scale)
//   0x81001000  V8     : LPI_S x LPI_D int8    (32 KiB), row-major, stride d
//   0x81010000  acc    : LPI_D int32           int32 scratch, zeroed by caller
//   0x81011000  out    : LPI_D float
//   0x81020000  Vf     : LPI_S x LPI_D float   (128 KiB) [harness only]
//   0x81060000  Pf     : LPI_S float                     [harness only]
//   0x81061000  vscale : LPI_S float                     [harness only]
//   0x81062000  wf     : LPI_S float                     [harness only]
// ---------------------------------------------------------------------------
#define LPI_W_ADDR  0x81000000UL
#define LPI_V_ADDR  0x81001000UL
#define LPI_A_ADDR  0x81010000UL
#define LPI_O_ADDR  0x81011000UL
#define LPI_VF_ADDR 0x81020000UL
#define LPI_PF_ADDR 0x81060000UL
#define LPI_VS_ADDR 0x81061000UL
#define LPI_WF_ADDR 0x81062000UL

#define LPI_W8 ((int8_t  *)LPI_W_ADDR)
#define LPI_V8 ((int8_t  *)LPI_V_ADDR)
#define LPI_A  ((int32_t *)LPI_A_ADDR)
#define LPI_O  ((float   *)LPI_O_ADDR)
#define LPI_VF ((float   *)LPI_VF_ADDR)
#define LPI_PF ((float   *)LPI_PF_ADDR)
#define LPI_VS ((float   *)LPI_VS_ADDR)
#define LPI_WF ((float   *)LPI_WF_ADDR)

// `acc` is a d-long int32 scratch buffer, zeroed by the caller (exactly as
// `out` is in the fp32 and q8kv variants). The int32 accumulation must NOT
// saturate or round.
void llama_attn_pv_int8(size_t S, size_t d, const int8_t *W8, const int8_t *V8,
                        int32_t *acc, float *out, float wscale);

#endif  // LLAMA_ATTN_PV_INT8_H
