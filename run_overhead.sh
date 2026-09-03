#!/usr/bin/env bash
# Wait for the pyramid tile fetch, then fetch (test tiles) -> train -> build -> evaluate -> evaluate-scale.
set -u
cd /home/omen/projects/geoloc-tr
P=.venv/bin/python
C=configs/ankara_overhead.yaml
true
echo "FETCH_PYRAMID_EXITED $(date +%H:%M:%S)"; grep -v Warning logs_09_fetch_pyramid.txt | tail -20
for s in "fetch" "train" "build" "evaluate --tta 4" "evaluate-scale --tta 4"; do
  n=${s%% *}
  echo "######## $n $(date +%H:%M:%S)"
  timeout 21600 $P scripts/09_overhead.py $s -c $C > logs_09_$n.txt 2>&1
  rc=$?
  echo "[$n] exit=$rc"
  tail -c 800 logs_09_$n.txt | tr '\r' '\n' | grep -v '^$' | tail -5
  [ $rc -ne 0 ] && { echo "OVERHEAD_STOPPED at $n"; exit $rc; }
done
echo "OVERHEAD_DONE $(date)"
