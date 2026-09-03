#!/usr/bin/env bash
# Wait for the tile fetch, then train -> build -> evaluate the overhead-query model. Stops at the first failure.
set -u
cd /home/omen/projects/geoloc-tr
P=.venv/bin/python
C=configs/ankara_overhead.yaml
while pgrep -f '09_overhead.py fetch' > /dev/null; do sleep 15; done
echo "FETCH_DONE $(date +%H:%M:%S)"; grep -E "^z|^wayback" logs_09_fetch.txt
for s in train build "evaluate --tta 4"; do
  n=${s%% *}
  echo "######## $n $(date +%H:%M:%S)"
  timeout 21600 $P scripts/09_overhead.py $s -c $C > logs_09_$n.txt 2>&1
  rc=$?
  echo "[$n] exit=$rc"
  tail -c 600 logs_09_$n.txt | tr '\r' '\n' | grep -v '^$' | tail -4
  [ $rc -ne 0 ] && { echo "OVERHEAD_STOPPED at $n"; exit $rc; }
done
echo "OVERHEAD_DONE $(date)"
