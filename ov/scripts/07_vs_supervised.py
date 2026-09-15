#!/usr/bin/env python3
"""Compare the best open-vocabulary configuration against the supervised RTMDet
baseline, and both against the measured label-noise ceiling.

Restricted to the 54-image Clean test split: RTMDet saw the training partition,
so it is the only subset on which the comparison is fair.  At n = 54 the
minimum detectable paired difference is ~8.3 AP points, which is stated
alongside every result here rather than buried.
"""
import json, os, sys, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
ev2 = import_module('02_eval') if False else None   # metrics re-implemented below

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
WD = f'{ROOT}/ov/work_dirs'
RES = f'{ROOT}/ov/results'
SWEEP = [round(x, 3) for x in np.arange(0.02, 0.95, 0.01)]
GRID = 24
SUPERVISED = f'{ROOT}/repro/work_dirs/eval_clahe/preds.bbox.json'


def boxes_to_mask(boxes, W, H):
    gw, gh = max(1, W // GRID), max(1, H // GRID)
    m = np.zeros((gh, gw), dtype=bool)
    for x, y, w, h in boxes:
        x0, y0 = int(np.floor(x / GRID)), int(np.floor(y / GRID))
        x1, y1 = int(np.ceil((x + w) / GRID)), int(np.ceil((y + h) / GRID))
        m[max(0, y0):min(gh, y1), max(0, x0):min(gw, x1)] = True
    return m


def uiou_curve(gt_by_img, pred_by_img, imgs, cat=0):
    """[n_img, n_thr] per-image union IoU."""
    out = np.zeros((len(imgs), len(SWEEP)))
    for r, im in enumerate(imgs):
        i = im['id']
        g = [a['bbox'] for a in gt_by_img.get(i, []) if a['category_id'] == cat]
        gm = boxes_to_mask(g, im['width'], im['height'])
        ps = pred_by_img.get(i, [])
        for c, t in enumerate(SWEEP):
            pm = boxes_to_mask([p['bbox'] for p in ps if p['score'] >= t],
                               im['width'], im['height'])
            u = (gm | pm).sum()
            out[r, c] = ((gm & pm).sum() / u) if u else 1.0
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


def f1_counts(gt_by_img, pred_by_img, imgs, cat=0, thr_iou=0.5):
    """tp/fp/fn arrays [n_img, n_thr] — the quantity comparable to the
    inter-expert F1 = 0.391 and the label-noise ceiling 0.399."""
    T = np.zeros((len(imgs), len(SWEEP)), int)
    F = np.zeros_like(T); N = np.zeros_like(T)
    for r, im in enumerate(imgs):
        i = im['id']
        g = [a['bbox'] for a in gt_by_img.get(i, []) if a['category_id'] == cat]
        p = sorted(pred_by_img.get(i, []), key=lambda q: -q['score'])
        sc = np.array([q['score'] for q in p]) if p else np.zeros(0)
        M = iou_mat([q['bbox'] for q in p], g)
        for c, t in enumerate(SWEEP):
            nk = int(np.searchsorted(-sc, -t, side='right')) if len(sc) else 0
            used = np.zeros(len(g), dtype=bool); tp = 0
            for row in range(nk):
                best, bi = thr_iou, -1
                for j in range(len(g)):
                    if not used[j] and M[row, j] >= best:
                        best, bi = M[row, j], j
                if bi >= 0:
                    used[bi] = True; tp += 1
            T[r, c], F[r, c], N[r, c] = tp, nk - tp, len(g) - tp
    return T, F, N


def micro(T, F, N, axis=0):
    t, f, n = T.sum(axis), F.sum(axis), N.sum(axis)
    d = 2 * t + f + n
    return np.where(d > 0, 2 * t / np.maximum(d, 1e-9), 0.0)


def paired_f1(A, B, n_boot=5000, seed=0):
    ta, tb = int(np.argmax(micro(*A))), int(np.argmax(micro(*B)))
    a = [x[:, ta] for x in A]; b = [x[:, tb] for x in B]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a[0]), size=(n_boot, len(a[0])))
    d = micro(*[x[idx] for x in a], axis=1) - micro(*[x[idx] for x in b], axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    fa, fb = float(micro(*A)[ta]), float(micro(*B)[tb])
    return {'a': fa, 'b': fb, 'a_thr': SWEEP[ta], 'b_thr': SWEEP[tb],
            'diff': fa - fb, 'lo': float(lo), 'hi': float(hi),
            'significant': bool(lo > 0 or hi < 0)}


def paired(a, b, n_boot=5000, seed=0):
    ta, tb = int(np.argmax(a.mean(0))), int(np.argmax(b.mean(0)))
    va, vb = a[:, ta], b[:, tb]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(va), size=(n_boot, len(va)))
    d = va[idx].mean(1) - vb[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'a': float(va.mean()), 'b': float(vb.mean()),
            'a_thr': SWEEP[ta], 'b_thr': SWEEP[tb],
            'diff': float(va.mean() - vb.mean()), 'lo': float(lo), 'hi': float(hi),
            'significant': bool(lo > 0 or hi < 0)}


def main():
    ev = json.load(open(f'{WD}/eval2c_test.json'))
    imgs = ev['images']
    gt_by_img = {}
    for a in ev['annotations']:
        gt_by_img.setdefault(a['image_id'], []).append(a)
    test_ids = {im['id'] for im in imgs}
    orig2new = {im['orig_id']: im['id'] for im in imgs}

    # supervised baseline, remapped into the shared id space
    sup = json.load(open(SUPERVISED))
    sup_by = {}
    for p in sup:
        nid = orig2new.get(p['image_id'])
        if nid is not None and p['category_id'] == 0:
            sup_by.setdefault(nid, []).append(p)
    sup_curve = uiou_curve(gt_by_img, sup_by, imgs)
    sup_f1 = f1_counts(gt_by_img, sup_by, imgs)
    print(f'supervised RTMDet  microF1 {micro(*sup_f1).max():.4f}   '
          f'uIoU {sup_curve[:, int(np.argmax(sup_curve.mean(0)))].mean():.4f}'
          f'  (n={len(sup)} preds over {len(sup_by)} images)')

    # every OV run, restricted to the test images
    rows = []
    for md in sorted(glob.glob(f'{WD}/preds/*_all')):
        mkey = os.path.basename(md).rsplit('_', 1)[0]
        for pf in sorted(glob.glob(f'{md}/*.json')):
            base = os.path.basename(pf)
            if base.startswith('_'):
                continue
            grp, prompt = base[:-5].split('__', 1)
            if grp == 'D':          # sugarcane target, not comparable to a weed detector
                continue
            prompt = prompt.replace('_', ' ')
            pb = {}
            for p in json.load(open(pf)):
                if p['image_id'] in test_ids:
                    pb.setdefault(p['image_id'], []).append(p)
            fc = f1_counts(gt_by_img, pb, imgs)
            rows.append((f'{mkey}|{grp}|{prompt}', float(micro(*fc).max()), fc))

    rows.sort(key=lambda r: -r[1])
    print('\ntop-5 OV configurations on the 54-image test split (micro-F1 @ IoU 0.50):')
    for k, v, _ in rows[:5]:
        print(f'  {v:.4f}  {k}')

    best_key, best_v, best_c = rows[0]
    r = paired_f1(best_c, sup_f1)
    print(f'\nbest OV ({best_key})  vs  supervised RTMDet')
    print(f'  {r["a"]:.4f} vs {r["b"]:.4f}   diff {r["diff"]:+.4f} '
          f'[{r["lo"]:+.4f}, {r["hi"]:+.4f}]  '
          f'{"DISTINGUISHABLE" if r["significant"] else "NOT distinguishable"}')
    print(f'  label-noise ceiling F1 = 0.399 [0.342-0.454]; expert-expert = 0.391')

    out = {'n_images': len(imgs), 'metric': 'microF1@IoU0.50',
           'min_detectable_ap_points': float((3692 / len(imgs)) ** 0.5),
           'supervised': {'preds': SUPERVISED,
                          'microF1': float(micro(*sup_f1).max()),
                          'uiou': float(sup_curve[:, int(np.argmax(sup_curve.mean(0)))].mean()),
                          'uiou_trivial_floor': 0.398,
                          'ap50_reported': 36.2, 'ap50_ci': [28.4, 42.6]},
           'ov_ranked': [{'run': k, 'microF1': v} for k, v, _ in rows],
           'best_vs_supervised': dict(r, run=best_key),
           'label_ceiling_f1': {'value': 0.399, 'lo': 0.342, 'hi': 0.454,
                                'human_human_f1': 0.391, 'n_images': 15}}
    json.dump(out, open(f'{RES}/vs_supervised.json', 'w'), indent=1)
    print(f'\nwrote {RES}/vs_supervised.json')


if __name__ == '__main__':
    main()
