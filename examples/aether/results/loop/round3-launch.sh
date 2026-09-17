#!/bin/bash
BASE=/share1/saves/max410011/hackathon/aether
PY=/home/max410011_l/.conda/envs/chia_env/bin/python
cd "$BASE" || exit 1
{
echo "# round3 launched $(date -Is)  (all --seed best, --budget-usd 1000)"
} >> out/loop/round3-runs.txt
launch() {
  kern=$1; iters=$2
  log="$BASE/out/loop/round3-$kern.log"
  setsid nohup "$PY" loop/loop.py --kernel "$kern" --iters "$iters" \
      --budget-usd 1000 --seed best > "$log" 2>&1 < /dev/null &
  echo "$(date -Is) LAUNCHED $kern iters=$iters pid=$! cmd='$PY loop/loop.py --kernel $kern --iters $iters --budget-usd 1000 --seed best'" \
      >> out/loop/round3-runs.txt
}
launch llama-q8-gemv-gemmini-n1 12
sleep 15
launch llama-q8-gemv-gemmini-lmhead 8
sleep 15
launch llama-q8-gemm 6
sleep 15
launch llama-attn-scores-int8 4
sleep 15
launch llama-attn-pv-int8 3
sleep 15
launch llama-silu-mul 4
sleep 15
launch llama-softmax 3
