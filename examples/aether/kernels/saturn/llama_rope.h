// See LICENSE for license details.
//
// llama-rope — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_ROPE_H
#define LLAMA_ROPE_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — RoPE for ONE decode step of Llama-3.2-1B.
//
// head_dim 64, num_attention_heads 32, num_key_value_heads 8 (GQA),
// rope_theta 500000 — all from the official config.json. A decode step rotates
// the 32 query heads and the 8 key heads at the SAME position, so the harness
// times one call for Q and one call for K back to back: 32*64 + 8*64 = 2560
// fp32 elements, which is exactly the `rope` work of one decoder layer
// (rope_q elems 2048 + rope_k elems 512 in loop/llama_model.py).
//
// Convention: HuggingFace Llama `rotate_half`, i.e. the head vector is split
// in HALVES (not interleaved pairs):
//     out[i]      = x[i]      * cos[i] - x[i+H] * sin[i]
//     out[i + H]  = x[i + H]  * cos[i] + x[i]   * sin[i]      H = head_dim/2
// The cos/sin table is INPUT: real implementations precompute it once per
// position, so generating it is not part of the kernel's cost.
//
// dtype: fp32 (loop/llama_model.py costs this kernel at acc_bytes = 4).
// ---------------------------------------------------------------------------
#define LRP_HEAD_DIM 64
#define LRP_HALF     (LRP_HEAD_DIM / 2)
#define LRP_Q_HEADS  32
#define LRP_KV_HEADS 8
#define LRP_POS      511            // decode position (S = 512 keys incl. self)
#define LRP_THETA    500000.0f

#define LRP_Q_ELEMS  (LRP_Q_HEADS  * LRP_HEAD_DIM)   // 2048
#define LRP_K_ELEMS  (LRP_KV_HEADS * LRP_HEAD_DIM)   //  512

// ---------------------------------------------------------------------------
// Memory map — outside the ELF (see llama-q8-gemv for why).
//   0x81000000  q    : 2048 float (8 KiB), rotated IN PLACE
//   0x81010000  k    :  512 float (2 KiB), rotated IN PLACE
//   0x81020000  cos  :   32 float
//   0x81021000  sin  :   32 float
//   0x81030000  q0   : pristine copy of q (harness only, for the golden model)
//   0x81040000  k0   : pristine copy of k (harness only)
// ---------------------------------------------------------------------------
#define LRP_Q_ADDR   0x81000000UL
#define LRP_K_ADDR   0x81010000UL
#define LRP_COS_ADDR 0x81020000UL
#define LRP_SIN_ADDR 0x81021000UL
#define LRP_Q0_ADDR  0x81030000UL
#define LRP_K0_ADDR  0x81040000UL

#define LRP_Q   ((float *)LRP_Q_ADDR)
#define LRP_K   ((float *)LRP_K_ADDR)
#define LRP_COS ((float *)LRP_COS_ADDR)
#define LRP_SIN ((float *)LRP_SIN_ADDR)
#define LRP_Q0  ((float *)LRP_Q0_ADDR)
#define LRP_K0  ((float *)LRP_K0_ADDR)

// ---------------------------------------------------------------------------
// The kernel under optimization. In place: `x` is both input and output.
// `cs`/`sn` are head_dim/2 long and are SHARED by every head.
// ---------------------------------------------------------------------------
void llama_rope(size_t nheads, size_t head_dim, float *x,
                const float *cs, const float *sn);

#endif  // LLAMA_ROPE_H
