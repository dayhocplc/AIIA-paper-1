#!/usr/bin/env python3
"""Inter-annotator agreement on the 80-image panel.

Four VIA projects were returned. Two carry the project name `panel_<annotator>`, so the
first thing this script establishes is whether they are one person labelling
twice (intra-annotator) or a mis-named fourth person (inter-annotator).

Matching mirrors the procedure used for the existing expert-expert figure
(F1 = 0.391 [0.295-0.466] at IoU 0.50, n = 15 images), so the numbers here are
directly comparable to it and to the label-noise ceiling of 0.399.
"""
import json, os, itertools
import numpy as np

SRC = {
    'A_output':   '<PANEL_SUBMISSIONS_DIR>/output/via_project_<annotator>.json',
    'B_output-1': '<PANEL_SUBMISSIONS_DIR>/output-1/output/via_project_<annotator>.json',
    'C_output-2': '<PANEL_SUBMISSIONS_DIR>/output-2/output/via_project_<annotator>.json',
    'D_output-3': '<PANEL_SUBMISSIONS_DIR>/output-3/output/via_project_<annotator>.json',
}
TIMES = {k: os.path.join(os.path.dirname(v), 'thoi_gian.csv') for k, v in SRC.items()}
OUT = '<GIAMIA_ROOT>/ov/results/panel_agreement.json'


def load(path):
    j = json.load(open(path))
    name = j['_via_settings']['project'].get('name')
    boxes, flags = {}, {}
    for v in j['_via_img_metadata'].values():
        f = v['filename']
        boxes[f] = [[r['shape_attributes'][k] for k in ('x', 'y', 'width', 'height')]
                    for r in v['regions'] if r['shape_attributes']['name'] == 'rect']
        flags[f] = bool(v['file_attributes'].get('khong_phan_dinh_duoc'))
    return name, boxes, flags


def iou_mat(P, G):
    if not len(P) or not len(G):
        return np.zeros((len(P), len(G)))
    P, G = np.asarray(P, float), np.asarray(G, float)
    px1, py1 = P[:, 0:1], P[:, 1:2]
    px2, py2 = px1 + P[:, 2:3], py1 + P[:, 3:4]
    gx1, gy1 = G[None, :, 0], G[None, :, 1]
    gx2, gy2 = gx1 + G[None, :, 2], gy1 + G[None, :, 3]
    iw = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    ih = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    it = iw * ih
    un = P[:, 2:3] * P[:, 3:4] + G[None, :, 2] * G[None, :, 3] - it
    return np.where(un > 0, it / np.maximum(un, 1e-9), 0.0)


def match(a, b, thr):
    """Greedy one-to-one matching -> (n_match, n_a, n_b). Symmetric by construction
    here because neither side has confidence scores; ties broken by best IoU."""
    M = iou_mat(a, b)
    used_b = np.zeros(len(b), bool)
    n = 0
    order = np.argsort(-M.max(1)) if len(a) and len(b) else range(len(a))
    for i in order:
        best, bj = thr, -1
        for j in range(len(b)):
            if not used_b[j] and M[i, j] >= best:
                best, bj = M[i, j], j
        if bj >= 0:
            used_b[bj] = True; n += 1
    return n, len(a), len(b)


def pair_f1(bx1, bx2, files, thr, idx=None):
    m = na = nb = 0
    sel = files if idx is None else [files[i] for i in idx]
    for f in sel:
        x, y, z = match(bx1[f], bx2[f], thr)
        m += x; na += y; nb += z
    return (2 * m / (na + nb)) if (na + nb) else 0.0


def boot_ci(bx1, bx2, files, thr, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(files), size=(n, len(files)))
    vals = [pair_f1(bx1, bx2, files, thr, ix) for ix in idx[:800]]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


data = {k: load(v) for k, v in SRC.items()}
files = sorted(set.intersection(*[set(d[1]) for d in data.values()]))
print(f'{len(files)} images common to all {len(data)} annotation sets\n')

print('=== per annotator ===')
summ = {}
for k, (name, bx, fl) in data.items():
    nb = sum(len(bx[f]) for f in files)
    areas = [b[2] * b[3] for f in files for b in bx[f]]
    t = {}
    for line in open(TIMES[k]).read().strip().splitlines()[1:]:
        fn, sec, _ = line.split(',')
        t[fn] = int(sec)
    tot = sum(t.get(f, 0) for f in files)
    summ[k] = {'project_name': name, 'n_boxes': nb,
               'boxes_per_image_median': float(np.median([len(bx[f]) for f in files])),
               'n_empty': sum(1 for f in files if not bx[f]),
               'n_undecidable': sum(1 for f in files if fl[f]),
               'median_box_area_px': float(np.median(areas)) if areas else 0,
               'total_seconds': tot, 'median_seconds': float(np.median(list(t.values())))}
    print(f'{k:11s} name={name:12s} boxes={nb:4d}  median/img={summ[k]["boxes_per_image_median"]:.1f}  '
          f'empty={summ[k]["n_empty"]:2d}  undecidable={summ[k]["n_undecidable"]}  '
          f'time={tot/60:.0f}min (median {summ[k]["median_seconds"]:.0f}s/img)')

print('\n=== pairwise agreement (greedy one-to-one matching) ===')
pairs = {}
for k1, k2 in itertools.combinations(data, 2):
    row = {}
    for thr in (0.50, 0.25, 0.10):
        f1 = pair_f1(data[k1][1], data[k2][1], files, thr)
        row[f'f1@{thr:.2f}'] = f1
    lo, hi = boot_ci(data[k1][1], data[k2][1], files, 0.50)
    row['f1@0.50_lo'], row['f1@0.50_hi'] = lo, hi
    pairs[f'{k1}|{k2}'] = row
    print(f'{k1:11s} vs {k2:11s}  F1@.50 {row["f1@0.50"]:.3f} [{lo:.3f}-{hi:.3f}]   '
          f'F1@.25 {row["f1@0.25"]:.3f}   F1@.10 {row["f1@0.10"]:.3f}')

json.dump({'n_images': len(files), 'per_annotator': summ, 'pairwise': pairs,
           'reference': {'existing_expert_expert_f1@0.50': 0.391,
                         'existing_ci': [0.295, 0.466], 'existing_n_images': 15,
                         'label_noise_ceiling': 0.399, 'ceiling_ci': [0.342, 0.454]}},
          open(OUT, 'w'), indent=1)
print(f'\nwrote {OUT}')
