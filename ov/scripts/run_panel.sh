#!/bin/bash
cd <GIAMIA_ROOT> || exit 1
P=ov/envs/ov/bin/python
for m in owlv2-base gdino-tiny gdino-base owlv2-large; do
  d="ov/work_dirs/preds/${m}_panel"
  [ -f "$d/_meta.json" ] && { echo "SKIP $m"; continue; }
  echo "START $m/panel $(date +%T)"
  $P ov/scripts/01_infer.py "$m" --set panel > ov/work_dirs/logs/${m}_panel.log 2>&1
  echo "DONE  $m/panel $(date +%T) rc=$?"
done
echo "PANEL ALL DONE $(date +%T)"
