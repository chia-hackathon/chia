#!/bin/bash
# Round 10 launcher. ONE kernel: llama-lmhead-fused-n1 (new this round).
#
# 為什麼只有一個 kernel：llama-layer-fused-n1 在 Round 9 之後已經是
# effectively solved（179,033，理論剩餘空間 9.7%，沒有活的機制），它剩下的價值
# 是那個 exposure ratio 0.611。Round 10 要回答的是「0.611 能不能外推到 lm_head
# 的形狀」——4 MiB 權重串流、沒有 attention 可以藏、只有 final rmsnorm / int8
# 量化 / running argmax 這種伴生工作。所以這一輪的全部預算都投在新 kernel 上。
#
# 數字比 best 重要：iteration 1 必須先量出 PROBE stream / rms / quant / argmax
# 四個數字（exposure ratio 的分母），即使總 cycle 一動也不動。
BASE=/share1/saves/max410011/hackathon/aether
PY=/home/max410011_l/.conda/envs/chia_env/bin/python
cd "$BASE" || exit 1
export RAY_ADDRESS=140.113.241.75:6379   # 本機有兩個 Ray cluster（titan 6380），明確指定
{
echo "# round10 launched $(date -Is)  (--budget-usd 1000; llama-lmhead-fused-n1 是新 kernel，不加 --seed)"
} >> out/loop/round10-runs.txt
# launch() 的第三個參數是 seed 策略：
#   ""      -> 不加 --seed（新 kernel，沒有可繼承的最佳版本）
#   "best"  -> --seed best（DB 裡該 kernel+config 的最佳通過版本）
launch() {
  kern=$1; iters=$2; seed=$3
  log="$BASE/out/loop/round10-$kern.log"
  seedargs=()
  [[ -n "$seed" ]] && seedargs=(--seed "$seed")
  setsid nohup "$PY" loop/loop.py --kernel "$kern" --iters "$iters" \
      --budget-usd 1000 "${seedargs[@]}" > "$log" 2>&1 < /dev/null &
  echo "$(date -Is) LAUNCHED $kern iters=$iters pid=$! cmd='$PY loop/loop.py --kernel $kern --iters $iters --budget-usd 1000 ${seedargs[*]}'" \
      >> out/loop/round10-runs.txt
}
# 新 kernel：lm_head 形狀的融合（Gemmini 4 MiB 權重 DMA 與 Saturn 的 final
# rmsnorm / int8 量化 / running argmax 放同一段計時區間，冷態）。SEED 留空。
launch llama-lmhead-fused-n1 4 ""
