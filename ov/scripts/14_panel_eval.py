#!/usr/bin/env python3
"""Score open-vocabulary detectors on the 80-image panel against each of the four
annotators separately.

Scoring against each annotator in turn, rather than against a manufactured
consensus, avoids inventing a merge rule and yields the quantity that matters:
how a detector compares with the spread of human opinion. The same procedure was
applied to the supervised baseline in 13_panel_ceiling.py, so the numbers are
directly comparable.

Reference values on this same panel:
    human-human, mean of 6 pairs   F1@0.50 = 0.303   (range 0.232-0.548)
    supervised RTMDet, mean of 4   F1@0.50 = 0.281   (range 0.246-0.301)
"""
import json, os, glob
import numpy as np

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
WD = f'{ROOT}/ov/work_dirs'
RES = f'{ROOT}/ov/results'
SRC = {'A': '<PANEL_SUBMISSIONS_DIR>/output/via_project_<annotator>.json',
       'B': '<PANEL_SUBMISSIONS_DIR>/output-1/output/via_project_<annotator>.json',
       'C': '<PANEL_SUBMISSIONS_DIR>/output-2/output/via_project_<annotator>.json',
       'D': '<PANEL_SUBMISSIONS_DIR>/output-3/output/via_project_<annotator>.json'}
SWEEP = [round(x, 3) for x in np.arange(0.02, 0.95, 0.01)]


def load_ann(p):
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


def counts_curve(preds_sorted, scores, gt, thr_iou=0.5):
    """tp/fp/fn at every sweep threshold for one image (score-sorted prefix)."""
    M = iou_mat([p['bbox'] for p in preds_sorted], gt)
    T = np.zeros(len(SWEEP), int); F = np.zeros_like(T); N = np.zeros_like(T)
    for c, t in enumerate(SWEEP):
        nk = int(np.searchsorted(-scores, -t, side='right')) if len(scores) else 0
        used = np.zeros(len(gt), bool); tp = 0
        for r in range(nk):
            best, bj = thr_iou, -1
            for j in range(len(gt)):
                if not used[j] and M[r, j] >= best:
                    best, bj = M[r, j], j
            if bj >= 0:
                used[bj] = True; tp += 1
        T[c], F[c], N[c] = tp, nk - tp, len(gt) - tp
    return T, F, N


def micro(T, F, N):
    d = 2 * T + F + N
    return np.where(d > 0, 2 * T / np.maximum(d, 1e-9), 0.0)


ANN = {k: load_ann(v) for k, v in SRC.items()}
ev = json.load(open(f'{WD}/eval2c_panel.json'))
id2base = {im['id']: im['file_name'].split('/')[-1] for im in ev['images']}
files = [id2base[i['id']] for i in ev['images']]

out = {'n_images': len(files), 'annotators': list(ANN),
       'reference': {'human_human_mean': 0.303, 'human_human_range': [0.232, 0.548],
                     'supervised_mean': 0.281, 'supervised_range': [0.246, 0.301]},
       'runs': {}}

rows = []
for md in sorted(glob.glob(f'{WD}/preds/*_panel')):
    mkey = os.path.basename(md).rsplit('_', 1)[0]
    for pf in sorted(glob.glob(f'{md}/*.json')):
        base = os.path.basename(pf)
        if base.startswith('_'):
            continue
        grp, prompt = base[:-5].split('__', 1)
        prompt = prompt.replace('_', ' ')
        if grp == 'D':          # sugarcane target: no annotator labelled the crop
            continue
        by = {}
        for p in json.load(open(pf)):
            by.setdefault(id2base[p['image_id']], []).append(p)

        per_ann = {}
        for k, ann in ANN.items():
            T = np.zeros(len(SWEEP), int); F = np.zeros_like(T); N = np.zeros_like(T)
            for f in files:
                ps = sorted(by.get(f, []), key=lambda q: -q['score'])
                sc = np.array([q['score'] for q in ps]) if ps else np.zeros(0)
                a, b, c = counts_curve(ps, sc, ann.get(f, []))
                T += a; F += b; N += c
            f1 = micro(T, F, N)
            i = int(np.argmax(f1))
            per_ann[k] = {'best_f1@0.50': float(f1[i]), 'thr': SWEEP[i]}
        vals = [v['best_f1@0.50'] for v in per_ann.values()]
        rec = {'group': grp, 'prompt': prompt, 'per_annotator': per_ann,
               'mean': float(np.mean(vals)), 'min': float(min(vals)), 'max': float(max(vals))}
        out['runs'][f'{mkey}|{grp}|{prompt}'] = rec
        rows.append((rec['mean'], mkey, grp, prompt, rec['min'], rec['max']))
        print(f'{mkey:12s} {grp:2s} {prompt[:32]:33s} mean {rec["mean"]:.3f} '
              f'[{rec["min"]:.3f}-{rec["max"]:.3f}]', flush=True)

rows.sort(reverse=True)
print('\n=== TOP 5 cau hinh OV tren panel (trung binh 4 nguoi) ===')
for m, mk, g, p, lo, hi in rows[:5]:
    print(f'  {m:.3f} [{lo:.3f}-{hi:.3f}]  {mk} | {g} | {p}')
print(f'\n  nguoi-nguoi  : 0.303 [0.232-0.548]')
print(f'  co giam sat  : 0.281 [0.246-0.301]')
if rows:
    print(f'  OV tot nhat  : {rows[0][0]:.3f}  = {rows[0][0]/0.303*100:.0f}% muc dong thuan nguoi')

os.makedirs(RES, exist_ok=True)
json.dump(out, open(f'{RES}/panel_ov.json', 'w'), indent=1)
print(f'\nwrote {RES}/panel_ov.json')
