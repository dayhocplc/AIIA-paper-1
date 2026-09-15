#!/bin/bash
cd <GIAMIA_ROOT> || exit 1
P=ov/envs/ov/bin/python
for m in gdino-base owlv2-large; do
  d="ov/work_dirs/preds/${m}_neg"
  [ -f "$d/_meta.json" ] && { echo "SKIP $m"; continue; }
  echo "START $m/neg $(date +%T)"
  $P ov/scripts/01_infer.py "$m" --set neg --groups A,C2,D > ov/work_dirs/logs/${m}_neg.log 2>&1
  echo "DONE  $m/neg $(date +%T) rc=$?"
done
echo "NEG ALL DONE $(date +%T)"
