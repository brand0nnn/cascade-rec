#!/bin/bash
# Detached chain: fresh Two-tower (text) all-pre-T1 seeds 1 then 2 on MPS, then the seed comparison (Val only).
set -euo pipefail
cd /Users/brandonkang/nus/cp4285/project
export PYTHONPATH=src
LOG=results/logs/20261107
for s in 1 2; do
  echo "$(date '+%F %T') start seed $s" >> $LOG/fresh-tt-chain.log
  .venv/bin/python -m cascade.pipelines.fresh_two_tower_seeds --phase train --seed $s --device mps > $LOG/fresh-tt-seed$s.log 2>&1
  echo "$(date '+%F %T') done seed $s" >> $LOG/fresh-tt-chain.log
done
.venv/bin/python -m cascade.pipelines.fresh_two_tower_seeds --phase compare > $LOG/fresh-tt-compare.log 2>&1
echo "$(date '+%F %T') compare done" >> $LOG/fresh-tt-chain.log
