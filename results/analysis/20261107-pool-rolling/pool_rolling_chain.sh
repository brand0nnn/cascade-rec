#!/bin/zsh
# Detached chain for 20261107-pool-rolling (Val only, exploratory).
#  1. (02:20 lead decision) no wait on the fresh-tt chain, which was stopped
#  2. resource gate before the build: df /Users >= 19.5 GB free AND vm_stat free+inactive >= 3 GB
#     (re-checked each minute for up to 20 min while swap is released, then abort with a log line)
#  3. prepare (CPU, 3 threads; ~0.96 GB float32 features: natural B queries + Val natural-hit rows, two arms)
#  4. fits on MPS: bce_coreview seeds 0/1/2; then bce_text seeds 0/1/2 (H3) only if co-review finished before 05:30;
#     no fit starts after 06:30 (if the first fit cannot start by 06:30, the chain stops after prepare and logs why)
#  5. CPU evaluate + analyse (writes results/analysis/20261107-pool-rolling/report.md)
# Logs: results/logs/20261107/pool-rolling-*.log
cd /Users/brandonkang/nus/cp4285/project || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
LOG=results/logs/20261107
STUDY=results/local_experiments/20261107-pool-rolling
PY=.venv/bin/python
note() { echo "$(date '+%F %T') $*" >> $LOG/pool-rolling-chain.log; }
hm() { echo $(( $(date +%H)*100 + $(date +%M) )); }
before() { (( $(hm) < $1 )); }  # chain runs after midnight only
mem_gb() { vm_stat | awk '/Pages free|Pages inactive/{gsub("\\.","",$NF); s+=$NF} END{printf "%.2f", s*16384/1e9}'; }
disk_gb() { df -k /Users | tail -1 | awk '{printf "%.2f", $4*1024/1e9}'; }
gpu_busy() { ps axo args | grep -E "ython.* -m cascade" | grep -v grep | grep -q -- "--device mps"; }

note "chain armed (pid $$); no fresh-tt wait (stopped by lead); disk $(disk_gb) GB, free+inactive $(mem_gb) GB"

if [[ ! -f $STUDY/features/manifest.json ]]; then
  tries=0
  until (( $(disk_gb) >= 19.5 && $(mem_gb) >= 3 )); do
    if (( tries >= 20 )); then
      note "ABORT before build: resource gate failed after 20 min (disk $(disk_gb) GB < 19.5 or free+inactive $(mem_gb) GB < 3)"
      exit 2
    fi
    sleep 60; tries=$((tries+1))
  done
  if ! before 0600; then note "ABORT before build: 06:00 reached, fits could not start before 06:30 after a ~20 min build"; exit 3; fi
  note "resource gate passed (disk $(disk_gb) GB, free+inactive $(mem_gb) GB); prepare start"
  caffeinate -i $PY -m cascade.pipelines.pool_rolling --phase prepare > $LOG/pool-rolling-prepare.log 2>&1
  rc=$?
  note "prepare rc=$rc; disk $(disk_gb) GB"
  (( rc == 0 )) || { note "ABORT: prepare failed (see pool-rolling-prepare.log)"; exit 4; }
fi

run_fit() {
  local method=$1 seed=$2
  [[ -f $STUDY/runs/$method/seed$seed/metrics.json ]] && return 0
  local free=0
  while (( free < 2 )); do if gpu_busy; then free=0; else free=$((free+1)); fi; sleep 30; done
  before 0630 || { note "06:30 gate: not starting $method seed$seed"; return 3; }
  (( $(disk_gb) >= 18.5 )) || { note "ABORT: disk $(disk_gb) GB < 18.5: not starting $method seed$seed"; return 2; }
  note "fit $method seed$seed start (free+inactive $(mem_gb) GB)"
  caffeinate -i $PY -m cascade.pipelines.pool_rolling --phase fit --method $method --seed $seed --device mps \
    > $LOG/pool-rolling-fit-$method-seed$seed.log 2>&1
  local rc=$?
  note "fit $method seed$seed rc=$rc"
  return $rc
}

ok=1
for s in 0 1 2; do run_fit bce_coreview $s || { ok=0; break; }; done
if (( ok )) && before 0530; then
  note "H3 gate passed ($(hm)): bce_text control"
  for s in 0 1 2; do run_fit bce_text $s || break; done
else
  note "H3 gate not passed (co-review ok=$ok, time $(hm)); no text control"
fi

if [[ -f $STUDY/runs/bce_coreview/seed2/metrics.json ]] && ! (( $(disk_gb) >= 18.5 )); then
  note "ABORT: disk $(disk_gb) GB < 18.5 before evaluate"
elif [[ -f $STUDY/runs/bce_coreview/seed2/metrics.json ]]; then
  note "evaluate start"
  caffeinate -i $PY -m cascade.pipelines.pool_rolling --phase evaluate > $LOG/pool-rolling-evaluate.log 2>&1
  rc=$?; note "evaluate rc=$rc"
  if (( rc == 0 )); then
    caffeinate -i $PY -m cascade.pipelines.pool_rolling --phase analyse > $LOG/pool-rolling-analyse.log 2>&1
    note "analyse rc=$? (report: results/analysis/20261107-pool-rolling/report.md)"
  fi
else
  note "co-review fits incomplete: no evaluation"
fi
note "chain end"
