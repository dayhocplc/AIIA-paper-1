#!/bin/bash
cd <GIAMIA_ROOT> || exit 1
P=ov/envs/ov/bin/python
for s in all panel; do
  d="ov/work_dirs/preds/sam3_${s}"
  [ -f "$d/_meta.json" ] && { echo "SKIP sam3/$s"; continue; }
  echo "START sam3/$s $(date +%T)"
  $P ov/scripts/01_infer.py sam3 --set "$s" > ov/work_dirs/logs/sam3_${s}.log 2>&1
  echo "DONE  sam3/$s $(date +%T) rc=$?"
done
echo "SAM3 ALL DONE $(date +%T)"
