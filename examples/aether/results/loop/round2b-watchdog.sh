#!/bin/bash
# Round 2b watchdog. Every 5 min: check each round2b-<kernel>.log mtime.
# If a log hasn't been touched in >=20 min AND its loop.py process is still
# alive -> log a STUCK line (does not kill anything).
# If the loop.py process for a kernel is gone -> log EXITED with the log's
# last 3 lines (to surface "session limit"/429 messages).
BASE=/share1/saves/max410011/hackathon/aether
OUT=$BASE/out/loop/round2b-watchdog.log
KERNELS=(
  llama-q8-gemv-gemmini-n1
  llama-q8-gemv-gemmini-lmhead
  llama-attn-scores-int8
  llama-attn-pv-int8
  llama-silu-mul
  llama-softmax
)
echo "=== watchdog start $(date -Is) ===" >> "$OUT"
while true; do
  NOW=$(date +%s)
  ALIVE=0
  for kern in "${KERNELS[@]}"; do
    log="$BASE/out/loop/round2b-$kern.log"
    alive_pid=$(pgrep -f "loop.py --kernel $kern " | head -1)
    if [[ -n "$alive_pid" ]]; then
      ALIVE=1
      lm=0; [[ -f "$log" ]] && lm=$(stat -c %Y "$log")
      idle=$(( (NOW - lm) / 60 ))
      if (( idle >= 20 )); then
        echo "$(date -Is) STUCK $kern ${idle}m" >> "$OUT"
      fi
    else
      last=""
      [[ -f "$log" ]] && last=$(tail -3 "$log" | tr '\n' ' | ')
      echo "$(date -Is) EXITED $kern :: $last" >> "$OUT"
    fi
  done
  if (( ALIVE == 0 )); then
    echo "$(date -Is) all round2b runs exited, watchdog stopping" >> "$OUT"
    break
  fi
  sleep 300
done
