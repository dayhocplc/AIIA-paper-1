#!/bin/bash
# Waits for run_neg_sam3_yw.sh, then refreshes every derived artefact that the
# two new negative-set runs change, and finally benchmarks model cost on the now
# idle GPU (latency must not be measured while inference is running).
cd <GIAMIA_ROOT> || exit 1
P=ov/envs/ov/bin/python
until grep -q "NEG SAM3+YW DONE" ov/work_dirs/driver_neg2.log 2>/dev/null; do sleep 30; done

echo "=== 06_negatives $(date +%T) ==="
$P ov/scripts/06_negatives.py || exit 1
echo "=== 08_tables $(date +%T) ==="
$P ov/scripts/08_tables.py > /dev/null || exit 1
echo "=== 05_figures $(date +%T) ==="
$P ov/scripts/05_figures.py || exit 1
echo "=== 20_qualitative_figures $(date +%T) ==="
$P ov/scripts/20_qualitative_figures.py || exit 1
echo "=== 21_model_costs (GPU now idle) $(date +%T) ==="
$P ov/scripts/21_model_costs.py --device cuda --iters 30 --size 640
echo "ALL REFRESH DONE $(date +%T)"
