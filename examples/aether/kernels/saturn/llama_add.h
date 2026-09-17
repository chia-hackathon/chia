// See LICENSE for license details.
//
// llama-add — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_ADD_H
#define LLAMA_ADD_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — the residual add of a Llama-3.2-1B decode step: hidden_size = 2048,
// twice per decoder layer (after attn.o_proj and after mlp.down_proj).
//
//   x[i] += y[i]        in place
//
// dtype: fp32 (loop/llama_model.py costs `add` at acc_bytes = 4). This is the
// cheapest kernel in the model and is pure memory traffic: 2 reads + 1 write
// per element, no arithmetic worth the name.
// ---------------------------------------------------------------------------
#define LAD_N 2048

// ---------------------------------------------------------------------------
// Memory map — outside the ELF.
//   0x81000000  x  : LAD_N float, updated IN PLACE
//   0x81010000  y  : LAD_N float
//   0x81020000  x0 : pristine copy of x (harness only, for the golden model)
// ---------------------------------------------------------------------------
#define LAD_X_ADDR  0x81000000UL
#define LAD_Y_ADDR  0x81010000UL
#define LAD_X0_ADDR 0x81020000UL

#define LAD_X  ((float *)LAD_X_ADDR)
#define LAD_Y  ((float *)LAD_Y_ADDR)
#define LAD_X0 ((float *)LAD_X0_ADDR)

void llama_add(size_t n, float *x, const float *y);

#endif  // LLAMA_ADD_H
