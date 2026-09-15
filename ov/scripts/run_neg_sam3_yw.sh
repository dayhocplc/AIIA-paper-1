#!/bin/bash
# Complete the confirmed-negative arm: sam3 and yoloworld-s were never run on the
# 869 weed-free images, so Instrument 3 covered only four of the six detectors.
# Groups A, C2 (weed -> false alarms) and D (crop -> true positives), matching
# run_neg_rest.sh.
cd <GIAMIA_ROOT> || exit 1
P=ov/envs/ov/bin/python
mkdir -p ov/work_dirs/logs

d=ov/work_dirs/preds/sam3_neg
if [ -f "$d/_meta.json" ]; then
  echo "SKIP sam3/neg"
else
  echo "START sam3/neg $(date +%T)"
  $P ov/scripts/01_infer.py sam3 --set neg --groups A,C2,D \
     > ov/work_dirs/logs/sam3_neg.log 2>&1
  echo "DONE  sam3/neg $(date +%T) rc=$?"
fi

d=ov/work_dirs/preds/yoloworld-s_neg
if [ -f "$d/_meta.json" ]; then
  echo "SKIP yoloworld-s/neg"
else
  echo "START yoloworld-s/neg $(date +%T)"
  $P ov/scripts/10_yoloworld.py --set neg --groups A,C2,D \
     > ov/work_dirs/logs/yoloworld_neg.log 2>&1
  echo "DONE  yoloworld-s/neg $(date +%T) rc=$?"
fi

echo "NEG SAM3+YW DONE $(date +%T)"
