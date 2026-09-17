// See LICENSE for license details.
//
// llama-rmsnorm — shared shape / memory-map declarations.
//
// SEALED: the harness and the kernel both include this. The optimizer may not
// change it (its bytes are re-read from the pristine tree at every build).

#ifndef LLAMA_RMSNORM_H
#define LLAMA_RMSNORM_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — one Llama-3.2-1B RMSNorm over the hidden state.
//
// hidden_size = 2048, rms_norm_eps = 1e-5 (both from the official
// meta-llama/Llama-3.2-1B config.json). Decode does 2 of these per decoder
// layer (input_layernorm, post_attention_layernorm) plus one final norm.
//
// dtype: fp32 activations. The projections around this op are int8, but the
// norm's sum-of-squares reduction is done in fp32 in every real
// implementation, which is what `ModelSpec.acc_bytes = 4` in
// loop/llama_model.py already assumes for this kernel.
// ---------------------------------------------------------------------------
#define LRN_N   2048            // hidden size
#define LRN_EPS 1e-5f           // rms_norm_eps

// ---------------------------------------------------------------------------
// Memory map — DELIBERATELY OUTSIDE THE ELF.
//
// The Verilator harness loads the ELF over TSI and zeroes .bss byte by byte at
// roughly 12 s per KiB, so every array lives at a fixed physical address in
// DRAM (memory@80000000, 256 MiB) past the end of the image and is filled at
// run time from a fixed seed.
//
//   0x80000000 .. ~0x80010000   the ELF itself (text + tiny .data/.bss)
//   0x81000000                  x : LRN_N float   (8 KiB)  input
//   0x81010000                  w : LRN_N float   (8 KiB)  norm weight
//   0x81020000                  y : LRN_N float   (8 KiB)  output
// ---------------------------------------------------------------------------
#define LRN_X_ADDR 0x81000000UL
#define LRN_W_ADDR 0x81010000UL
#define LRN_Y_ADDR 0x81020000UL

#define LRN_X ((float *)LRN_X_ADDR)
#define LRN_W ((float *)LRN_W_ADDR)
#define LRN_Y ((float *)LRN_Y_ADDR)

// ---------------------------------------------------------------------------
// The kernel under optimization.
//
//   ss   = (1/n) * sum_i x[i]*x[i]
//   y[i] = x[i] * (1 / sqrt(ss + eps)) * w[i]
//
// No mean subtraction (this is RMSNorm, not LayerNorm).
// ---------------------------------------------------------------------------
void llama_rmsnorm(size_t n, const float *x, const float *w, float *y,
                   float eps);

#endif  // LLAMA_RMSNORM_H
