#!/usr/bin/env python3
"""Render the manuscript's result tables from ov/results/*.json."""
import json, os
import numpy as np

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
RES = f'{ROOT}/ov/results'
OUT = f'{ROOT}/ov/paper/tables.md'

GORDER = ['A', 'B', 'E', 'C1', 'C2', 'D']
GDESC = {'A': 'abstract noun', 'B': 'morphological noun',
         'E': 'long, non-relational', 'C1': 'relational, negated',
         'C2': 'relational, affirmative', 'D': 'nominal control (sugarcane)'}

lines = []
W = lines.append

# ---------------------------------------------------------------- Table 1
ap = json.load(open(f'{RES}/annotation_protocol.json'))
w, s = ap['per_class']['weed'], ap['per_class']['sugarcane']
W('### Table 1 — Annotation-protocol asymmetry between the two classes\n')
W('| Quantity | `weed` | `sugarcane` |')
W('|---|---|---|')
for k, lab, f in [('n_boxes', 'Boxes', '{:.0f}'),
                  ('n_images', 'Images containing the class', '{:.0f}'),
                  ('boxes_per_image_median', 'Boxes per image (median)', '{:.1f}'),
                  ('box_area_pct_median', 'Box area, % of frame (median)', '{:.1f}'),
                  ('box_area_pct_p90', 'Box area, % of frame (p90)', '{:.1f}'),
                  ('frame_coverage_pct_median', 'Frame coverage per image (median, %)', '{:.1f}')]:
    W(f'| {lab} | {f.format(w[k])} | {f.format(s[k])} |')
W(f'\nMedian box-area ratio (cane/weed): **{ap["granularity_ratio_area"]:.1f}×**. '
  f'Images carrying both classes: **{ap["n_images_with_both"]}/{ap["n_images_total"]}**.\n')

# ---------------------------------------------------------------- Table 2
ev = json.load(open(f'{RES}/eval_main.json'))
models = sorted({k.split('|')[0] for k in ev['runs']})
W('\n### Table 2 — Union-mask IoU by prompt group (best prompt in each group)\n')
W(f'Evaluation set: {ev["n_images"]} images '
  f'({ev["n_with_cane"]} with annotated sugarcane, {ev["n_without_cane"]} without). '
  'Threshold swept per run; 95 % image-level bootstrap CI.\n')
W('| Group | Description | ' + ' | '.join(models) + ' |')
W('|---|---|' + '---|' * len(models))
for g in GORDER:
    row = [f'**{g}**', GDESC[g]]
    for m in models:
        vals = [v for k, v in ev['runs'].items()
                if k.startswith(m + '|') and v['group'] == g]
        if not vals:
            row.append('—'); continue
        b = max(vals, key=lambda v: v['union_iou']['best'])
        row.append(f'{b["union_iou"]["best"]:.3f} [{b["union_iou"]["lo"]:.3f}–'
                   f'{b["union_iou"]["hi"]:.3f}]')
    W('| ' + ' | '.join(row) + ' |')

# ---------------------------------------------------------------- Table 3
W('\n\n### Table 3 — Best $F_1$ at IoU 0.50, against the measured label-noise ceiling\n')
W('| Group | ' + ' | '.join(models) + ' |')
W('|---|' + '---|' * len(models))
for g in GORDER:
    row = [f'**{g}** {GDESC[g]}']
    for m in models:
        vals = [v for k, v in ev['runs'].items()
                if k.startswith(m + '|') and v['group'] == g]
        if not vals:
            row.append('—'); continue
        b = max(vals, key=lambda v: v['bestF1@0.5']['best'])
        row.append(f'{b["bestF1@0.5"]["best"]:.3f} [{b["bestF1@0.5"]["lo"]:.3f}–'
                   f'{b["bestF1@0.5"]["hi"]:.3f}]')
    W('| ' + ' | '.join(row) + ' |')
W('| *expert–expert agreement* | ' + ' | '.join(['0.391 [0.295–0.466]'] * len(models)) + ' |')
W('| *simulated label-noise ceiling* | ' + ' | '.join(['0.399 [0.342–0.454]'] * len(models)) + ' |')

# ---------------------------------------------------------------- Table 4
c = json.load(open(f'{RES}/contrasts.json'))
W('\n\n### Table 4 — Paired contrasts (union-mask IoU difference, 5 000 bootstrap replicates)\n')
W('| Contrast | Isolates | ' + ' | '.join(models) + ' |')
W('|---|---|' + '---|' * len(models))
order = [('C2-E', 'relationality'), ('C1-C2', 'negation'),
         ('E-A', 'prompt length'), ('D-A', 'category type'),
         ('B-A', 'morphology')]
for k, lab in order:
    row = [f'**{k}**', lab]
    for m in models:
        # sam3agent has a single AGENT run, no prompt groups, hence no contrasts
        r = c['models'].get(m, {}).get('contrasts', {}).get(k)
        if not r:
            row.append('—'); continue
        star = ' \\*' if r['significant'] else ''
        row.append(f'{r["diff"]:+.3f} [{r["lo"]:+.3f}, {r["hi"]:+.3f}]{star}')
    W('| ' + ' | '.join(row) + ' |')
W('\n\\* 95 % CI excludes zero.')

# ---------------------------------------------------------------- Table 5
p = f'{RES}/negatives.json'
if os.path.exists(p):
    neg = json.load(open(p))
    W(f'\n\n### Table 5 — False alarms on {neg["n_images"]} expert-confirmed weed-free images\n')
    W('Threshold fixed at each run\'s $F_1$-optimal operating point on the annotated set.\n')
    W('| Model | Group | Prompt | Operating thr. | Images firing | Detections/image |')
    W('|---|---|---|---|---|---|')
    for key, r in sorted(neg['runs'].items()):
        if r['operating_threshold'] is None:
            continue
        m, g, pr = key.split('|')
        W(f'| {m} | {g} | `{pr}` | {r["operating_threshold"]:.2f} | '
          f'{r["frac_at_operating"]*100:.1f} % | '
          f'{r["detections_per_image_at_operating"]:.2f} |')

# ---------------------------------------------------------------- Table 6
p = f'{RES}/vs_supervised.json'
if os.path.exists(p):
    v = json.load(open(p))
    W('\n\n### Table 6 — Best open-vocabulary configuration vs supervised RTMDet\n')
    W(f'Clean test split only (n = {v["n_images"]} images; minimum detectable '
      f'paired difference {v["min_detectable_ap_points"]:.1f} AP points).\n')
    W(f'Metric key in `vs_supervised.json`: `{v["metric"]}`. This is the same run '
      'as the best `sam3` cell in Table 3, scored on a different subset: Table 3 '
      'reports the 222-image set (0.188), this table the 54-image Clean test '
      'split (0.194), which is the only split on which the supervised comparison '
      'is fair.\n')
    b = v['best_vs_supervised']
    # The metric here is micro-F1 @ IoU 0.50, as declared in vs_supervised.json;
    # the header used to read "Union-mask IoU", which is a different quantity
    # (the supervised model's union-mask IoU on this split is 0.535, not 0.417).
    W('| System | Micro-$F_1$ @ IoU 0.50 |')
    W('|---|---|')
    W(f'| Best OV: `{b["run"]}` | {b["a"]:.3f} |')
    W(f'| Supervised RTMDet (AP50 36.2 % [28.4–42.6]) | {b["b"]:.3f} |')
    W(f'| **Paired difference** | **{b["diff"]:+.3f} [{b["lo"]:+.3f}, {b["hi"]:+.3f}]** '
      f'— {"distinguishable" if b["significant"] else "not distinguishable from zero"} |')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\n--- wrote {OUT}')
