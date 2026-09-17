#!/bin/bash
source /home/ray/chipyard/env.sh
: > /work-out/build.summary
for CFG in GENV256D128GemminiShuttleConfig GENV256D128GemminiShuttleCosimConfig; do
  s=$(date +%s)
  make -C /home/ray/chipyard/sims/verilator CONFIG=$CFG -j16 > /work-out/build.$CFG.log 2>&1
  rc=$?; e=$(date +%s)
  echo "$CFG rc=$rc secs=$((e-s))" >> /work-out/build.summary
done
touch /work-out/build.done
