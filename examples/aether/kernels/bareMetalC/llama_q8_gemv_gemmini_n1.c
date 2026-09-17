// See LICENSE for license details.
//
// SEALED. The registry's `main_src` for llama-q8-gemv-gemmini-n1: it only
// pins N and pulls in the agent-owned compute header plus the shared sealed
// harness body. Everything real is in
// `include/llama_q8_gemv_gemmini_body.h`.

#define LQ8_GV_N 1

#include "include/llama_q8_gemv_gemmini_n1_kernel.h"
#include "include/llama_q8_gemv_gemmini_body.h"
