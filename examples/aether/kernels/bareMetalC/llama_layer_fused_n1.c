// See LICENSE for license details.
//
// SEALED. The registry's `main_src` for llama-layer-fused-n1: it only pins N
// and pulls in the agent-owned compute header plus the sealed harness body.
// Everything real is in `include/llama_layer_fused_body.h`.

#define LF_N 1

#include "include/llama_layer_fused_n1_kernel.h"
#include "include/llama_layer_fused_body.h"
