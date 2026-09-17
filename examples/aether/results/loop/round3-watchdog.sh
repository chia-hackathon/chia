#!/bin/bash
# Round 3 watchdog. Every 5 min:
#   - log mtime >= 20 min AND loop.py alive        -> STUCK
#   - loop.py gone                                 -> EXITED + last 3 log lines
#   - EXITED with a session-limit / 429 / rate-limit message AND iterations
#     remaining -> wait until the reset time (parsed from "resets <time>",
#     else 30 min) and RELAUNCH the remaining iters with the same command
#     (--seed best), recording RESTARTED + the new run_id in round3-runs.txt.
BASE=/share1/saves/max410011/hackathon/aether
PY=/home/max410011_l/.conda/envs/chia_env/bin/python
OUT=$BASE/out/loop/round3-watchdog.log
RUNS=$BASE/out/loop/round3-runs.txt
cd "$BASE" || exit 1

KERNELS=(
  llama-q8-gemv-gemmini-n1
  llama-q8-gemv-gemmini-lmhead
  llama-q8-gemm
  llama-attn-scores-int8
  llama-attn-pv-int8
  llama-silu-mul
  llama-softmax
)
declare -A PLANNED=(
  [llama-q8-gemv-gemmini-n1]=12
  [llama-q8-gemv-gemmini-lmhead]=8
  [llama-q8-gemm]=6
  [llama-attn-scores-int8]=4
  [llama-attn-pv-int8]=3
  [llama-silu-mul]=4
  [llama-softmax]=3
)
# remaining iterations per kernel, decremented as restarts consume the budget
declare -A REMAIN
for k in "${KERNELS[@]}"; do REMAIN[$k]=${PLANNED[$k]}; done
declare -A DONE_RESTART   # kernel -> 'final' once it is done for good
declare -A SKIP_UNTIL     # kernel -> epoch until which a restart is pending

log() { echo "$(date -Is) $*" >> "$OUT"; }

# how many iterations completed: loop.py logs "iteration N: <result>" once
# per finished iteration.
completed_iters() {
  local log=$1 n
  [[ -f "$log" ]] || { echo 0; return; }
  n=$(grep -oE 'aether\.loop:iteration [0-9]+:' "$log" \
        | grep -oE '[0-9]+' | sort -n | tail -1)
  echo "${n:-0}"
}

# seconds to wait before a restart: parse "resets <HH:MM(am|pm)>" from the log,
# else 30 minutes.
wait_seconds() {
  local log=$1 t now target
  t=$(grep -oiE 'resets[[:space:]]+(at[[:space:]]+)?[0-9]{1,2}(:[0-9]{2})?[[:space:]]*(am|pm)?' "$log" \
        | tail -1 | grep -oiE '[0-9]{1,2}(:[0-9]{2})?[[:space:]]*(am|pm)?$')
  if [[ -n "$t" ]]; then
    target=$(date -d "$t" +%s 2>/dev/null)
    now=$(date +%s)
    if [[ -n "$target" ]]; then
      (( target <= now )) && target=$(( target + 86400 ))
      echo $(( target - now + 60 ))
      return
    fi
  fi
  echo 1800
}

relaunch() {
  local kern=$1 iters=$2 log=$BASE/out/loop/round3-$kern.log
  mv "$log" "$log.$(date +%Y%m%d-%H%M%S).bak" 2>/dev/null
  setsid nohup "$PY" loop/loop.py --kernel "$kern" --iters "$iters" \
      --budget-usd 1000 --seed best > "$log" 2>&1 < /dev/null &
  local pid=$!
  log "RESTARTED $kern iters=$iters pid=$pid"
  # give loop.py a moment to print its run id
  for _ in $(seq 1 60); do
    sleep 5
    rid=$(grep -oE 'Run [0-9]{8}-[0-9]{6}-[0-9a-f]{4}' "$log" | head -1 | awk '{print $2}')
    [[ -n "$rid" ]] && break
  done
  {
    echo "$(date -Is) RESTARTED $kern iters=$iters pid=$pid run_id=${rid:-unknown} cmd='$PY loop/loop.py --kernel $kern --iters $iters --budget-usd 1000 --seed best'"
  } >> "$RUNS"
  log "RESTARTED $kern run_id=${rid:-unknown}"
}

log "=== round3 watchdog start ==="
# let the staggered launcher finish before the first sweep
sleep 180

while true; do
  NOW=$(date +%s)
  ALIVE=0
  for kern in "${KERNELS[@]}"; do
    log_f="$BASE/out/loop/round3-$kern.log"
    pid=$(pgrep -f "loop.py --kernel $kern " | head -1)
    if [[ -n "$pid" ]]; then
      ALIVE=1
      lm=0; [[ -f "$log_f" ]] && lm=$(stat -c %Y "$log_f")
      idle=$(( (NOW - lm) / 60 ))
      (( idle >= 20 )) && log "STUCK $kern ${idle}m pid=$pid"
      continue
    fi

    # process is gone
    if [[ -n "${SKIP_UNTIL[$kern]}" ]] && (( NOW < ${SKIP_UNTIL[$kern]} )); then
      ALIVE=1          # a restart is scheduled; keep the watchdog running
      continue
    fi
    [[ "${DONE_RESTART[$kern]}" == final ]] && continue
    last=""
    [[ -f "$log_f" ]] && last=$(tail -3 "$log_f" | tr '\n' '|')
    log "EXITED $kern :: $last"

    done_n=$(completed_iters "$log_f")
    left=$(( ${REMAIN[$kern]} - done_n ))
    if grep -qiE 'session limit|rate limit|429|rate_limit' "$log_f" 2>/dev/null \
       && (( left > 0 )); then
      w=$(wait_seconds "$log_f")
      log "RATE-LIMITED $kern: completed=$done_n remaining=$left, sleeping ${w}s then restarting"
      ALIVE=1
      REMAIN[$kern]=$left
      # suppress further EXITED handling for this kernel until the restarted
      # run has had time to come up (wait + 10 min of ray/sim startup)
      SKIP_UNTIL[$kern]=$(( NOW + w + 600 ))
      ( sleep "$w"; relaunch "$kern" "$left" ) &
    else
      log "FINISHED $kern: completed=$done_n of ${PLANNED[$kern]} (no restart)"
      DONE_RESTART[$kern]=final
    fi
  done
  if (( ALIVE == 0 )); then
    log "all round3 runs done, watchdog stopping"
    break
  fi
  sleep 300
done
