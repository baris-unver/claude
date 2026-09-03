#!/usr/bin/env bash
cd /home/omen/projects/geoloc-tr
while pgrep -f '[0]9_overhead.py evaluate-scale' > /dev/null; do sleep 15; done
echo "SWEEP_DONE $(date +%H:%M:%S)"
echo "######## build-fine $(date +%H:%M:%S)"
.venv/bin/python /tmp/claude-1000/-home-omen-projects-geoloc-tr/5d3f3a65-2f84-480e-aab2-292441d92cb7/scratchpad/build_fine.py > logs_09_build-fine.txt 2>&1; echo "[build-fine] exit=$?"; grep "fine database" logs_09_build-fine.txt
echo "######## evaluate-scale-small $(date +%H:%M:%S)"
.venv/bin/python scripts/09_overhead.py evaluate-scale --tta 4 --extents z18,z19 -c configs/ankara_overhead.yaml > logs_09_evaluate-scale-small.txt 2>&1; echo "[evaluate-scale-small] exit=$?"
grep -E "^(current|wayback)" logs_09_evaluate-scale-small.txt
echo "FINE_DONE $(date)"
