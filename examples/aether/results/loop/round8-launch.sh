#!/bin/bash
BASE=/share1/saves/max410011/hackathon/aether
PY=/home/max410011_l/.conda/envs/chia_env/bin/python
cd "$BASE" || exit 1
export RAY_ADDRESS=140.113.241.75:6379   # 本機有兩個 Ray cluster（titan 6380），明確指定
{
echo "# round8 launched $(date -Is)  (--budget-usd 1000; llama-layer-fused-n1 是新 kernel，不加 --seed)"
} >> out/loop/round8-runs.txt
# round8 起，每個 kernel 的 seed 策略不同，所以 launch() 多吃一個參數：
#   ""      -> 不加 --seed（新 kernel，沒有可繼承的最佳版本）
#   "best"  -> --seed best（DB 裡該 kernel+config 的最佳通過版本）
launch() {
  kern=$1; iters=$2; seed=$3
  log="$BASE/out/loop/round8-$kern.log"
  seedargs=()
  [[ -n "$seed" ]] && seedargs=(--seed "$seed")
  setsid nohup "$PY" loop/loop.py --kernel "$kern" --iters "$iters" \
      --budget-usd 1000 "${seedargs[@]}" > "$log" 2>&1 < /dev/null &
  echo "$(date -Is) LAUNCHED $kern iters=$iters pid=$! cmd='$PY loop/loop.py --kernel $kern --iters $iters --budget-usd 1000 ${seedargs[*]}'" \
      >> out/loop/round8-runs.txt
}
# 新 kernel：層級融合（Gemmini 權重 DMA 與 Saturn attention 放同一段計時區間，冷態）
launch llama-layer-fused-n1 4 best   # Round 7 已完成 4/8，接續剩下 4 個，從 189,851 起跳
sleep 60
# 探針 iteration：in-flight 6/8/10/12/16 掃描 + 冷態量測。數字比 best 重要。
