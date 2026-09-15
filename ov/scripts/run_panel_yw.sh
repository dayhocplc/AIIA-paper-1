#!/bin/bash
cd <GIAMIA_ROOT> || exit 1
until grep -q "PANEL ALL DONE" ov/work_dirs/driver_panel.log 2>/dev/null; do sleep 20; done
echo "START yoloworld/panel $(date +%T)"
ov/envs/ov/bin/python ov/scripts/10_yoloworld.py --set panel > ov/work_dirs/logs/yoloworld_panel.log 2>&1
echo "DONE  yoloworld/panel $(date +%T) rc=$?"
echo "PANEL+YW ALL DONE $(date +%T)"
