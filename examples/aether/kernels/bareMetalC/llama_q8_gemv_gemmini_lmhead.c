// See LICENSE for license details.
//
// SEALED. The registry's `main_src` for llama-q8-gemv-gemmini-lmhead: it only
// pins N and M and pulls in the agent-owned compute header plus the sealed
// harness body. Everything real is in
// `include/llama_q8_gemv_gemmini_lmhead_body.h`.
//
// Shape: N = 1 (one decoded token), K = 2048 (hidden size),
// M = 2048 (one tile of the 2048 x 128256 LM-head / vocab projection).

#define LQ8_GV_N 1
#define LQ8_GV_M 2048

#include "include/llama_q8_gemv_gemmini_lmhead_kernel.h"
#include "include/llama_q8_gemv_gemmini_lmhead_body.h"
