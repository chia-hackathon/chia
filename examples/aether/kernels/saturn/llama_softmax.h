// See LICENSE for license details.
//
// llama-softmax — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_SOFTMAX_H
#define LLAMA_SOFTMAX_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — the attention softmax of ONE query head of a Llama-3.2-1B decode
// step: one row of length S = 512. A decode step runs 32 of these per layer.
//
//   m    = max_i x[i]
//   e[i] = exp(x[i] - m)
//   y[i] = e[i] / sum_i e[i]
//
// The max subtraction is REQUIRED (it is what makes the exp finite); a kernel
// that drops it is not this kernel.
//
// dtype: fp32 (loop/llama_model.py costs `softmax` at acc_bytes = 4).
// ---------------------------------------------------------------------------
#define LSM_N 512

// ---------------------------------------------------------------------------
// Memory map — outside the ELF.
//   0x81000000  x : LSM_N float
//   0x81010000  y : LSM_N float
// ---------------------------------------------------------------------------
#define LSM_X_ADDR 0x81000000UL
#define LSM_Y_ADDR 0x81010000UL

#define LSM_X ((float *)LSM_X_ADDR)
#define LSM_Y ((float *)LSM_Y_ADDR)

void llama_softmax(size_t n, const float *x, float *y);

#endif  // LLAMA_SOFTMAX_H
