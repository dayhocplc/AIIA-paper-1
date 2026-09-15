#!/bin/bash
cd <GIAMIA_ROOT> || exit 1
P=ov/envs/ov/bin/python
echo "START stage1 $(date +%T)"
$P ov/scripts/16_sam3_agent_stage1.py --sets all,panel > ov/work_dirs/logs/agent_stage1.log 2>&1
echo "STAGE1 DONE $(date +%T) rc=$?"
$P ov/scripts/17_sam3_agent_stage2.py > ov/work_dirs/logs/agent_stage2.log 2>&1
echo "STAGE2 DONE $(date +%T) rc=$?"
echo "AGENT ALL DONE $(date +%T)"
