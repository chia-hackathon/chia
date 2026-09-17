#!/bin/bash
# Round 10 watchdog. Every 5 min:
#   - log mtime >= 20 min AND loop.py alive        -> STUCK
#   - loop.py gone                                 -> EXITED + last 3 log lines
#   - EXITED with a session-limit / 429 / rate-limit message AND iterations
#     remaining -> wait until the reset time (parsed from "resets <time>
#     (<tz>)" as a timezone-local time, or from a bare ISO 8601 timestamp
#     with an explicit offset; else 30 min) and RELAUNCH the remaining
#     iters with the same command (seed 策略見下面的 SEED 陣列), recording RESTARTED + the
#     new run_id in round10-runs.txt.
BASE=/share1/saves/max410011/hackathon/aether
PY=/home/max410011_l/.conda/envs/chia_env/bin/python
OUT=$BASE/out/loop/round10-watchdog.log
RUNS=$BASE/out/loop/round10-runs.txt
cd "$BASE" || exit 1
export RAY_ADDRESS=140.113.241.75:6379   # 本機有兩個 Ray cluster（titan 6380），明確指定

KERNELS=(
  llama-lmhead-fused-n1
)
declare -A PLANNED=(
  [llama-lmhead-fused-n1]=4
)
# seed 策略跟 round10-launch.sh 必須一致。llama-lmhead-fused-n1 是 Round 10 的
# 新 kernel，沒有可繼承的最佳版本，所以 SEED 留空；重啟時若誤加 --seed best，
# 會把這一輪前面幾個 iteration 的產物當成起點，baseline 分母不變但實驗語意就
# 變了（exposure ratio 的分母會漂掉）。
declare -A SEED=(
  [llama-lmhead-fused-n1]=""
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

# seconds to wait before a restart. Two "resets ..." message shapes are
# recognized:
#   1. "resets <time> (<tz>)"      - <time> is local to the named zone
#                                     (e.g. "resets 5pm (America/Los_Angeles)"),
#                                     so it must be parsed with TZ="<tz>",
#                                     NOT the watchdog host's timezone.
#   2. "resets <ISO8601 w/ offset>" - e.g. "resets 2026-09-09T00:00:00+00:00",
#                                     already unambiguous; date -d parses it
#                                     directly.
# Anything else falls back to 1800s (30 min).
source "$BASE/out/loop/watchdog-lib.sh"
wait_seconds() { reset_wait_seconds "$1"; }

relaunch() {
  local kern=$1 iters=$2 log=$BASE/out/loop/round10-$kern.log
  local -a seedargs=()
  [[ -n "${SEED[$kern]}" ]] && seedargs=(--seed "${SEED[$kern]}")
  mv "$log" "$log.$(date +%Y%m%d-%H%M%S).bak" 2>/dev/null
  setsid nohup "$PY" loop/loop.py --kernel "$kern" --iters "$iters" \
      --budget-usd 1000 "${seedargs[@]}" > "$log" 2>&1 < /dev/null &
  local pid=$!
  log "RESTARTED $kern iters=$iters pid=$pid"
  # give loop.py a moment to print its run id
  for _ in $(seq 1 60); do
    sleep 5
    rid=$(grep -oE 'Run [0-9]{8}-[0-9]{6}-[0-9a-f]{4}' "$log" | head -1 | awk '{print $2}')
    [[ -n "$rid" ]] && break
  done
  {
    echo "$(date -Is) RESTARTED $kern iters=$iters pid=$pid run_id=${rid:-unknown} cmd='$PY loop/loop.py --kernel $kern --iters $iters --budget-usd 1000 ${seedargs[*]}'"
  } >> "$RUNS"
  log "RESTARTED $kern run_id=${rid:-unknown}"
}

log "=== round10 watchdog start ==="
# let the staggered launcher finish before the first sweep
sleep 180

while true; do
  NOW=$(date +%s)
  ALIVE=0
  for kern in "${KERNELS[@]}"; do
    log_f="$BASE/out/loop/round10-$kern.log"
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
    if is_rate_limited "$log_f" && (( left > 0 )); then
      w=$(wait_seconds "$log_f")
      if (( w < 0 )); then
        log "WEEKLY-LIMIT $kern: completed=$done_n remaining=$left — usage limit resets in >6h, NOT restarting. Operator must relaunch."
        DONE_RESTART[$kern]=final
        continue
      fi
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
    log "all round10 runs done, watchdog stopping"
    break
  fi
  sleep 300
done
