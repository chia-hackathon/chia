// See LICENSE for license details.
//
// SEALED. The registry's `main_src` for llama-lmhead-fused-n1: it only pins N
// and M and pulls in the agent-owned compute header plus the sealed harness
// body. Everything real is in `include/llama_lmhead_fused_body.h`.

#define LH_N 1
#define LH_M 2048

#include "include/llama_lmhead_fused_n1_kernel.h"
#include "include/llama_lmhead_fused_body.h"
