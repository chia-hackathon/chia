// See LICENSE for license details.
//
// llama-silu-mul — shared shape / memory-map declarations.  SEALED.

#ifndef LLAMA_SILU_MUL_H
#define LLAMA_SILU_MUL_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shape — the SwiGLU activation of one Llama-3.2-1B decode step.
//
// intermediate_size = 8192 (official config.json). Once per decoder layer:
//
//   out[i] = silu(gate[i]) * up[i],   silu(g) = g / (1 + exp(-g))
//
// dtype: fp32 (loop/llama_model.py costs `silu_mul` at acc_bytes = 4).
// ---------------------------------------------------------------------------
#define LSU_N 8192

// ---------------------------------------------------------------------------
// Memory map — outside the ELF.
//   0x81000000  gate : LSU_N float (32 KiB)
//   0x81010000  up   : LSU_N float (32 KiB)
//   0x81020000  out  : LSU_N float (32 KiB)
// ---------------------------------------------------------------------------
#define LSU_G_ADDR 0x81000000UL
#define LSU_U_ADDR 0x81010000UL
#define LSU_O_ADDR 0x81020000UL

#define LSU_G ((float *)LSU_G_ADDR)
#define LSU_U ((float *)LSU_U_ADDR)
#define LSU_O ((float *)LSU_O_ADDR)

void llama_silu_mul(size_t n, const float *gate, const float *up, float *out);

#endif  // LLAMA_SILU_MUL_H
