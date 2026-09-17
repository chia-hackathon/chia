#!/usr/bin/env bash
# 把 repos/gemmini 下自己加的 kernel 檔案裝進 chisel-build container 的 chipyard 樹。
#
# 為什麼需要這支：loop/nodes.py 的 gemmini_collateral() 是從
# GEMMINI_TESTS_DIR = /home/ray/chipyard/generators/gemmini/software/gemmini-rocc-tests
# 讀檔的，而那棵樹在 chisel-build container 內部，**沒有** bind-mount 到
# /share1/.../aether/repos/gemmini。所以 chia up 重建 container（或換機器）之後，
# 自訂 kernel 會消失，llama-q8-gemm 就會 build 失敗。跑這支補回去。
#
# 用法：bash cluster/install-gemmini-kernels.sh [container_name]
#   env: AETHER_GEMMINI_SRC (source tree), AETHER_BUILD_CONTAINER (container name)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
AETHER_DIR="$(dirname -- "$SCRIPT_DIR")"
# Where the project's own Gemmini kernel sources live. In the original working
# tree that was a full gemmini checkout under repos/; in this published snapshot
# the same files are kept flat under kernels/ (bareMetalC/ + include/). Prefer
# the checkout when present, fall back to kernels/, and let AETHER_GEMMINI_SRC
# override both.
if [ -n "${AETHER_GEMMINI_SRC:-}" ]; then
    SRC="$AETHER_GEMMINI_SRC"
elif [ -d "$AETHER_DIR/repos/gemmini/software/gemmini-rocc-tests" ]; then
    SRC="$AETHER_DIR/repos/gemmini/software/gemmini-rocc-tests"
else
    SRC="$AETHER_DIR/kernels"
fi
CONTAINER="${1:-${AETHER_BUILD_CONTAINER:-aether-chisel-build-${USER}-0}}"
DST="/home/ray/chipyard/generators/gemmini/software/gemmini-rocc-tests"

# 每支 kernel 兩個檔：sealed harness（main + golden model + self-check）與
# agent 可改的 compute header。header 放在 include/ 是因為 GEMMINI_SEALED_GLOBS
# 已經包含 include/*.h，且 Gemmini 的編譯線只編 main_src 一個 TU。
FILES=(
    # llama-q8-gemm (prefill GEMM)
    "bareMetalC/llama_q8_gemm.c"
    "include/llama_q8_gemm_kernel.h"
    # llama-q8-gemv-gemmini-n1 / -n16 (decode GEMV on Gemmini).
    # 一支 sealed body 被兩個薄 main 包起來；每個 N 各有自己的可改 header，
    # 因為 loop 的 baseline cache 是用「可改檔案的內容」當 key，
    # 只靠 -D 區分的兩個變體會撞在同一筆 cache。
    "include/llama_q8_gemv_gemmini_body.h"
    "bareMetalC/llama_q8_gemv_gemmini_n1.c"
    "include/llama_q8_gemv_gemmini_n1_kernel.h"
    "bareMetalC/llama_q8_gemv_gemmini_n16.c"
    "include/llama_q8_gemv_gemmini_n16_kernel.h"
    # llama-q8-gemv-gemmini-lmhead (LM head / vocab projection tile, N=1,
    # K=2048, M=2048). 自己的 sealed body，因為 n1/n16 那支 body 把 M 寫死成 512。
    "include/llama_q8_gemv_gemmini_lmhead_body.h"
    "bareMetalC/llama_q8_gemv_gemmini_lmhead.c"
    "include/llama_q8_gemv_gemmini_lmhead_kernel.h"
    # llama-layer-fused-n1 (Round 7): 一個 decoder layer 在 decode N=1 的切片 —
    # Gemmini 的 1 MiB 權重 DMA 與 Saturn 的 attention/softmax 放在同一段計時區
    # 間，冷態（計時前用 Gemmini DMA 沖掉 L2）。baseline 是序列執行；最佳化空間
    # 就是把 Saturn 的工作藏進權重 DMA 的影子裡。
    "include/llama_layer_fused_body.h"
    "bareMetalC/llama_layer_fused_n1.c"
    "include/llama_layer_fused_n1_kernel.h"
    # llama-lmhead-fused-n1 (Round 10): 一個 LM-head tile 在 decode N=1 的切片 —
    # Gemmini 的 4 MiB 權重 DMA 與 Saturn 的 final rmsnorm / int8 量化 / running
    # argmax 放在同一段計時區間，冷態。存在的理由是量 lm_head 形狀的 exposure
    # ratio，用來檢驗 llama-layer-fused-n1 的 0.611 能不能外推到整個 decode。
    "include/llama_lmhead_fused_body.h"
    "bareMetalC/llama_lmhead_fused_n1.c"
    "include/llama_lmhead_fused_n1_kernel.h"
)

# docker cp 在這台機器上會踩到 chisel-build container 的 /ssh-agent bind-mount
# （docker cp 會 pause/重掛容器，而那個 mount source 是 socket 不是目錄），
# 所以用 exec + stdin 重導向，不要用 docker cp。
for rel in "${FILES[@]}"; do
    [ -f "$SRC/$rel" ] || { echo "[install] missing source file: $SRC/$rel" >&2; exit 1; }
done

for rel in "${FILES[@]}"; do
    echo "[install] $rel -> $CONTAINER:$DST/$rel"
    docker exec -i "$CONTAINER" bash -c "cat > $DST/$rel" < "$SRC/$rel"
done

docker exec "$CONTAINER" bash -c "cd $DST && ls -la ${FILES[*]}"
echo "[install] done"
