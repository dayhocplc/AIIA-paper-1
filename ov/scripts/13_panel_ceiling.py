#!/usr/bin/env python3
"""Label-noise ceiling from the 4-way annotated 80-image panel, and the
supervised detector scored against it.

The panel is disjoint from the 222-image labelled set (0 overlap) and previously
carried no annotations at all. Four independent annotators have now labelled it,
which replaces the 15-image / 2-annotator basis of the existing ceiling estimate
with 80 images and 6 annotator pairs.
"""
import json, os, itertools
import numpy as np

SRC = {'A': '<PANEL_SUBMISSIONS_DIR>/output/via_project_<annotator>.json',
       'B': '<PANEL_SUBMISSIONS_DIR>/output-1/output/via_project_<annotator>.json',
       'C': '<PANEL_SUBMISSIONS_DIR>/output-2/output/via_project_<annotator>.json',
       'D': '<PANEL_SUBMISSIONS_DIR>/output-3/output/via_project_<annotator>.json'}
PANEL_COCO = 'repro/mmdet_mit/data/COCO_Panel/panel_coco.json'
SUP = 'repro/work_dirs/panel_preds/preds.bbox.json'
OUT = 'ov/results/panel_ceiling.json'


def load(p):
    j = json.load(open(p)); out = {}
    for v in j['_via_img_metadata'].values():
        out[v['filename']] = [[r['shape_attributes'][k] for k in ('x', 'y', 'width', 'height')]
                              for r in v['regions']]
    return out


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


def counts(pred, gt, thr):
    """(tp, n_pred, n_gt) with greedy one-to-one matching, best-IoU first."""
    M = iou_mat(pred, gt)
    if not len(pred) or not len(gt):
        return 0, len(pred), len(gt)
    used = np.zeros(len(gt), bool); tp = 0
    for i in np.argsort(-M.max(1)):
        best, bj = thr, -1
        for j in range(len(gt)):
            if not used[j] and M[i, j] >= best:
                best, bj = M[i, j], j
        if bj >= 0:
            used[bj] = True; tp += 1
    return tp, len(pred), len(gt)


def f1_over(files, get_p, get_g, thr):
    t = p = g = 0
    for f in files:
        a, b, c = counts(get_p(f), get_g(f), thr)
        t += a; p += b; g += c
    return (2 * t / (p + g)) if (p + g) else 0.0


def boot(files, get_p, get_g, thr, n=800, seed=0):
    rng = np.random.default_rng(seed)
    v = [f1_over([files[i] for i in rng.integers(0, len(files), len(files))],
                 get_p, get_g, thr) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


ANN = {k: load(v) for k, v in SRC.items()}
files = sorted(ANN['A'])
res = {'n_images': len(files), 'n_annotators': len(ANN)}

# ---- 1. pairwise human agreement
print('=== 1. Dong thuan tung cap (n=80 anh) ===')
pw = {}
for a, b in itertools.combinations(ANN, 2):
    row = {f'f1@{t:.2f}': f1_over(files, lambda f, a=a: ANN[a][f],
                                  lambda f, b=b: ANN[b][f], t) for t in (0.50, 0.25)}
    pw[f'{a}-{b}'] = row
    print(f'  {a}-{b}  F1@.50 {row["f1@0.50"]:.3f}   F1@.25 {row["f1@0.25"]:.3f}')
m50 = float(np.mean([v['f1@0.50'] for v in pw.values()]))
m25 = float(np.mean([v['f1@0.25'] for v in pw.values()]))
no_bd50 = float(np.mean([v['f1@0.50'] for k, v in pw.items() if k != 'B-D']))
print(f'  TRUNG BINH 6 cap : F1@.50 {m50:.3f}   F1@.25 {m25:.3f}')
print(f'  bo cap B-D ngoai le: F1@.50 {no_bd50:.3f}')
res['pairwise'] = pw
res['mean_pairwise'] = {'f1@0.50': m50, 'f1@0.25': m25, 'f1@0.50_excl_BD': no_bd50}

# ---- 2. leave-one-out: each annotator vs the pooled others
print('\n=== 2. Moi nguoi so voi 3 nguoi con lai (gop hop) ===')
loo = {}
for k in ANN:
    others = [o for o in ANN if o != k]
    pool = lambda f: [b for o in others for b in ANN[o][f]]
    v50 = f1_over(files, lambda f: ANN[k][f], pool, 0.50)
    lo, hi = boot(files, lambda f: ANN[k][f], pool, 0.50)
    loo[k] = {'f1@0.50': v50, 'lo': lo, 'hi': hi,
              'f1@0.25': f1_over(files, lambda f: ANN[k][f], pool, 0.25)}
    print(f'  {k} vs (3 nguoi kia)  F1@.50 {v50:.3f} [{lo:.3f}-{hi:.3f}]   F1@.25 {loo[k]["f1@0.25"]:.3f}')
res['leave_one_out'] = loo

# ---- 3. supervised model against each annotator
print('\n=== 3. Mo hinh co giam sat (RTMDet panel_preds) so voi tung nguoi ===')
cp = json.load(open(PANEL_COCO))
id2f = {i['id']: i['file_name'].split('/')[-1] for i in cp['images']}
sup_all = json.load(open(SUP))
by = {}
for p in sup_all:
    by.setdefault(id2f.get(p['image_id']), []).append(p)
print(f'  {len(sup_all)} du doan tren {len(by)} anh; quet nguong de lay F1 tot nhat')
sup = {}
for k in ANN:
    best, bt = -1, None
    for t in np.arange(0.02, 0.95, 0.02):
        v = f1_over(files, lambda f, t=t: [q['bbox'] for q in by.get(f, []) if q['score'] >= t],
                    lambda f: ANN[k][f], 0.50)
        if v > best:
            best, bt = v, round(float(t), 2)
    sup[k] = {'best_f1@0.50': best, 'thr': bt}
    print(f'  vs {k}: F1@.50 {best:.3f} (nguong {bt})')
sup_mean = float(np.mean([v['best_f1@0.50'] for v in sup.values()]))
print(f'  TRUNG BINH: {sup_mean:.3f}')
res['supervised_vs_annotator'] = sup
res['supervised_mean'] = sup_mean

res['comparison'] = {
    'panel_mean_human_f1@0.50': m50,
    'existing_estimate_15img_2ann': 0.391,
    'existing_ceiling': 0.399,
    'supervised_on_222set': 0.417,
    'best_OV_on_222set': 0.176}
os.makedirs('ov/results', exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1)
print(f'\nwrote {OUT}')
