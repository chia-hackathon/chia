#!/bin/bash
# Round 2 watchdog. Polls every 5 min: per-run iteration count from the db and
# log mtime. 20 min with no new iteration AND no simulator process running ->
# logged as STUCK. Never kills anything.
PY=/home/max410011_l/.conda/envs/chia_env/bin/python
BASE=/share1/saves/max410011/hackathon/aether
RUNS=$BASE/out/loop/round2-runs.txt
OUT=$BASE/out/loop/round2-watchdog.log
declare -A LASTN LASTT
echo "=== watchdog start $(date -Is) ===" >> "$OUT"
while true; do
  NOW=$(date +%s)
  ALIVE=0
  while read -r rid kern; do
    [[ -z "$rid" || "$rid" == \#* ]] && continue
    n=$($PY $BASE/loop/db.py --sql "select count(*) as c from iters where run_id='$rid'" 2>/dev/null | tail -1 | tr -dc '0-9')
    [[ -z "$n" ]] && n=0
    log=$BASE/out/loop/round2-$kern.log
    lm=0; [[ -f "$log" ]] && lm=$(stat -c %Y "$log")
    done_flag=$($PY $BASE/loop/db.py --sql "select count(*) as c from runs where run_id='$rid' and finished is not null" 2>/dev/null | tail -1 | tr -dc '0-9')
    [[ -z "$done_flag" ]] && done_flag=0
    prev=${LASTN[$rid]:-}
    if [[ "$n" != "$prev" ]]; then LASTN[$rid]=$n; LASTT[$rid]=$NOW; fi
    idle=$(( (NOW - ${LASTT[$rid]:-$NOW}) / 60 ))
    logidle=$(( (NOW - lm) / 60 ))
    sims=$(pgrep -fc "simulator-chipyard" 2>/dev/null || echo 0)
    state=running
    if [[ "$done_flag" != "0" ]]; then state=finished; fi
    if [[ "$state" == running ]]; then
      ALIVE=1
      if (( idle >= 20 && logidle >= 20 && sims == 0 )); then state="STUCK(idle=${idle}m,log=${logidle}m,sims=$sims)"; fi
    fi
    echo "$(date -Is) $rid $kern iters=$n log_idle=${logidle}m $state" >> "$OUT"
  done < "$RUNS"
  if (( ALIVE == 0 )); then echo "$(date -Is) all runs finished, watchdog exiting" >> "$OUT"; break; fi
  sleep 300
done
