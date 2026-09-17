// See LICENSE for license details.
//
// llama-q8-gemv — shared shape / memory-map declarations.
//
// SEALED: the harness and the kernel both include this. The optimizer may not
// change it (its bytes are re-read from the pristine tree at every build).

#ifndef LLAMA_Q8_GEMV_H
#define LLAMA_Q8_GEMV_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — one Llama-3.2-1B decode-step GEMV tile.
//
// Llama-3.2-1B has hidden size 2048; a decode step multiplies a [K=2048]
// activation vector by a weight matrix with M in {2048 (q/o_proj), 8192
// (gate/up_proj)}. LQ8_M is one 512-row *tile* of that matrix, so a single
// Verilator run is minutes rather than hours; the arithmetic per output row is
// identical to the full model.
// ---------------------------------------------------------------------------
#define LQ8_K 2048          // reduction length  = Llama-3.2-1B hidden size
#define LQ8_M 512           // output rows per tile

// ---------------------------------------------------------------------------
// Memory map — DELIBERATELY OUTSIDE THE ELF.
//
// The Verilator harness loads the ELF over TSI and zeroes .bss byte by byte;
// measured at roughly 12 s per KiB, so a 1 MiB weight array declared as
// `static` would cost hours of wall time before main() even runs. Instead the
// buffers live at fixed physical addresses in DRAM (memory@80000000, 256 MiB)
// past the end of the image, and the harness fills them at run time from a
// fixed seed. Nothing here is in the ELF, so the load stays ~1 s.
//
//   0x80000000 .. ~0x80010000   the ELF itself (text + tiny .data/.bss)
//   0x81000000                  W : LQ8_M x LQ8_K int8, row-major   (1 MiB)
//   0x81200000                  x : LQ8_K int8                      (2 KiB)
//   0x81210000                  y : LQ8_M int32                     (2 KiB)
// ---------------------------------------------------------------------------
#define LQ8_W_ADDR 0x81000000UL
#define LQ8_X_ADDR 0x81200000UL
#define LQ8_Y_ADDR 0x81210000UL

#define LQ8_W ((int8_t  *)LQ8_W_ADDR)
#define LQ8_X ((int8_t  *)LQ8_X_ADDR)
#define LQ8_Y ((int32_t *)LQ8_Y_ADDR)

// ---------------------------------------------------------------------------
// The kernel under optimization.
//
//   y[m] = sum_{k=0}^{K-1} W[m*K + k] * x[k]     for m in [0, M)
//
// W is row-major with row stride exactly K. All products are int8 x int8;
// the accumulation and the result are int32 and must NOT saturate or round —
// int32 is wide enough for K = 2048 (|sum| <= 2048*128*128 = 2^25).
// ---------------------------------------------------------------------------
void llama_q8_gemv(size_t M, size_t K,
                   const int8_t *W, const int8_t *x, int32_t *y);

#endif  // LLAMA_Q8_GEMV_H
