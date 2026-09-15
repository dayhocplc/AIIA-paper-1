#!/bin/bash
# Rebuild the manuscript's data-derived content from the result JSONs.
#
#   run_manuscript_build.sh <in.docx> <out.docx>
#
# 21 and 23 need the GPU idle (latency must not be measured against a running
# job); 22 and 24 are pure document surgery and need nothing.
set -e
cd <GIAMIA_ROOT>
P=ov/envs/ov/bin/python
IN=${1:?usage: run_manuscript_build.sh <in.docx> <out.docx>}
OUT=${2:?usage: run_manuscript_build.sh <in.docx> <out.docx>}
TMP=$(mktemp -d)

$P ov/scripts/08_tables.py > /dev/null          # tables.md, for the markdown draft
$P ov/scripts/05_figures.py                     # fig1-fig6
$P ov/scripts/20_qualitative_figures.py         # figQ1-figQ3
$P ov/scripts/21_model_costs.py --device cuda --iters 30 --size 640
PYTHONPATH=<GIAMIA_ROOT>/repro/mmdet_mit \
  repro/envs/mmdet/bin/python ov/scripts/21_model_costs.py \
  --device cuda --iters 30 --only rtmdet        # supervised arm, other virtualenv
$P ov/scripts/23_repro_record.py > /dev/null
$P ov/scripts/22_fill_docx_tables.py "$IN" "$TMP/stage1.docx"
$P ov/scripts/24_fill_appendices.py "$TMP/stage1.docx" "$OUT"
rm -rf "$TMP"
echo "built $OUT"
