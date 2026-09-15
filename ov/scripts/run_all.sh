#!/bin/bash
# Sequential OV inference matrix. Resumable: a run whose _meta.json exists is skipped.
cd <GIAMIA_ROOT> || exit 1
P=ov/envs/ov/bin/python
LOG=ov/work_dirs/logs; mkdir -p "$LOG"
run () {  # run <model> <set> [extra args]
  local m=$1 s=$2; shift 2
  local d="ov/work_dirs/preds/${m}_${s}"
  if [ -f "$d/_meta.json" ]; then echo "SKIP $m/$s (already done)"; return; fi
  echo "START $m/$s $(date +%T)"
  $P ov/scripts/01_infer.py "$m" --set "$s" "$@" > "$LOG/${m}_${s}.log" 2>&1
  echo "DONE  $m/$s $(date +%T) rc=$?  $(grep -c . "$LOG/${m}_${s}.log") log lines"
}
for m in owlv2-base gdino-tiny gdino-base owlv2-large; do run "$m" all; done
for m in owlv2-base gdino-tiny; do run "$m" neg --groups A,C2,D; done
echo "ALL DONE $(date +%T)"
