#!/usr/bin/env python3
"""Size-matched cross-class control: nominal vs relational on comparable objects.

The two classes were annotated at different granularity (weed at instance level,
median 4.0 % of frame; sugarcane at region level, median 30.7 %), which makes a
raw instance-level comparison invalid.  Cohen's kappa on the pixel grid is
prevalence-corrected but turns out to be too insensitive to resolve the
contrast.

This script instead removes the confound at source: both classes are restricted
to ground-truth boxes occupying 5-40 % of the frame, a band in which both are
well populated (363 weed boxes, 203 sugarcane boxes).  Predictions are filtered
to the same band, so the detector is asked for objects of the same size regime
in both conditions, and the sensitive instance-level metric becomes valid.
This is the COCO area-range convention applied to make two classes comparable.

Comparison is restricted further to images carrying at least one in-band box of
BOTH classes, so every image contributes to both sides of the contrast.
"""
import json, os, glob
import numpy as np

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
WD = f'{ROOT}/ov/work_dirs'
RES = f'{ROOT}/ov/results'
SWEEP = [round(x, 3) for x in np.arange(0.02, 0.95, 0.01)]
LO, HI = 5.0, 40.0          # % of frame
N_BOOT = 5000


def area_pct(b, W, H):
    return b[2] * b[3] / (W * H) * 100


def iou_matrix(P, G):
    if not len(P) or not len(G):
        return np.zeros((len(P), len(G)))
    P = np.asarray(P, float); G = np.asarray(G, float)
    px1, py1 = P[:, 0:1], P[:, 1:2]
    px2, py2 = px1 + P[:, 2:3], py1 + P[:, 3:4]
    gx1, gy1 = G[None, :, 0], G[None, :, 1]
    gx2, gy2 = gx1 + G[None, :, 2], gy1 + G[None, :, 3]
    iw = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    ih = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    inter = iw * ih
    uni = P[:, 2:3] * P[:, 3:4] + G[None, :, 2] * G[None, :, 3] - inter
    return np.where(uni > 0, inter / np.maximum(uni, 1e-9), 0.0)


def greedy(M, n_keep, n_gt, thr=0.5):
    used = np.zeros(n_gt, dtype=bool); tp = 0
    for r in range(n_keep):
        row = M[r]; best, bi = thr, -1
        for i in range(n_gt):
            if not used[i] and row[i] >= best:
                best, bi = row[i], i
        if bi >= 0:
            used[bi] = True; tp += 1
    return tp, n_keep - tp, n_gt - tp


def counts_for(preds_by_img, gt_by_img, img_ids, dims, band=True):
    """-> tp/fp/fn arrays [n_img, n_thr].

    band=True  : restrict GT and predictions to the 5-40 % size band, so the two
                 classes are compared on objects of the same size regime.
    band=False : the uncontrolled comparison, same images and same metric, so
                 that the only difference between the two conditions is whether
                 annotation granularity was controlled.
    """
    tp = np.zeros((len(img_ids), len(SWEEP)), int)
    fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    keep = (lambda b, W, H: LO <= area_pct(b, W, H) <= HI) if band else (lambda b, W, H: True)
    for r, i in enumerate(img_ids):
        W, H = dims[i]
        g = [a['bbox'] for a in gt_by_img.get(i, []) if keep(a['bbox'], W, H)]
        p = [q for q in preds_by_img.get(i, []) if keep(q['bbox'], W, H)]
        p.sort(key=lambda q: -q['score'])
        sc = np.array([q['score'] for q in p]) if p else np.zeros(0)
        M = iou_matrix([q['bbox'] for q in p], g)
        for c, t in enumerate(SWEEP):
            nk = int(np.searchsorted(-sc, -t, side='right')) if len(sc) else 0
            a, b, cc = greedy(M, nk, len(g))
            tp[r, c], fp[r, c], fn[r, c] = a, b, cc
    return tp, fp, fn


def micro_f1(tp, fp, fn, axis=0):
    T, F, N = tp.sum(axis), fp.sum(axis), fn.sum(axis)
    den = 2 * T + F + N
    return np.where(den > 0, 2 * T / np.maximum(den, 1e-9), 0.0)


def main():
    gt = json.load(open(f'{WD}/eval2c_all.json'))
    dims = {i['id']: (i['width'], i['height']) for i in gt['images']}
    by = {0: {}, 1: {}}
    for a in gt['annotations']:
        by[a['category_id']].setdefault(a['image_id'], []).append(a)

    # images with at least one IN-BAND box of both classes
    def inband_imgs(c):
        return {i for i, anns in by[c].items()
                if any(LO <= area_pct(a['bbox'], *dims[i]) <= HI for a in anns)}
    ids = sorted(inband_imgs(0) & inband_imgs(1))
    nb_w = sum(1 for a in gt['annotations'] if a['category_id'] == 0
               and LO <= area_pct(a['bbox'], *dims[a['image_id']]) <= HI)
    nb_s = sum(1 for a in gt['annotations'] if a['category_id'] == 1
               and LO <= area_pct(a['bbox'], *dims[a['image_id']]) <= HI)
    print(f'size band {LO}-{HI} % of frame: {nb_w} weed boxes, {nb_s} sugarcane boxes')
    print(f'images with in-band boxes of BOTH classes: {len(ids)}\n')

    out = {'band_pct': [LO, HI], 'n_images': len(ids),
           'n_weed_boxes': nb_w, 'n_cane_boxes': nb_s, 'models': {}}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))

    for md in sorted(glob.glob(f'{WD}/preds/*_all')):
        mkey = os.path.basename(md).rsplit('_', 1)[0]
        grp_counts = {}
        for pf in sorted(glob.glob(f'{md}/*.json')):
            base = os.path.basename(pf)
            if base.startswith('_'):
                continue
            g, prompt = base[:-5].split('__', 1)
            if g not in ('A', 'D'):          # relational target vs nominal control
                continue
            pb = {}
            for q in json.load(open(pf)):
                pb.setdefault(q['image_id'], []).append(q)
            cat = 1 if g == 'D' else 0
            for band in (True, False):
                grp_counts.setdefault((g, band), []).append(
                    counts_for(pb, by[cat], ids, dims, band=band))

        if ('A', True) not in grp_counts or ('D', True) not in grp_counts:
            continue

        res = {}
        for band, name in [(True, 'size_matched'), (False, 'uncontrolled')]:
            agg = {g: tuple(np.mean([c[k] for c in grp_counts[(g, band)]], axis=0)
                            for k in range(3)) for g in ('A', 'D')}
            fA, fD = micro_f1(*agg['A']), micro_f1(*agg['D'])
            tA, tD = int(np.argmax(fA)), int(np.argmax(fD))
            vA = [x[:, tA] for x in agg['A']]
            vD = [x[:, tD] for x in agg['D']]
            d = (micro_f1(*[x[idx] for x in vD], axis=1)
                 - micro_f1(*[x[idx] for x in vA], axis=1))
            lo, hi = np.percentile(d, [2.5, 97.5])
            res[name] = {'A_microF1': float(fA[tA]), 'D_microF1': float(fD[tD]),
                         'A_thr': SWEEP[tA], 'D_thr': SWEEP[tD],
                         'diff': float(fD[tD] - fA[tA]),
                         'lo': float(lo), 'hi': float(hi),
                         'p_two_sided': float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                         'significant': bool(lo > 0 or hi < 0)}
        # keep the flat keys for backward compatibility with the figure script
        out['models'][mkey] = dict(res['size_matched'], **{'both_conditions': res})
        for name in ('uncontrolled', 'size_matched'):
            r = res[name]
            print(f'{mkey:12s} {name:13s} A {r["A_microF1"]:.4f}  D {r["D_microF1"]:.4f}  '
                  f'D-A {r["diff"]:+.4f} [{r["lo"]:+.4f},{r["hi"]:+.4f}] '
                  f'p={r["p_two_sided"]:.4f} {"*" if r["significant"] else ""}')

    json.dump(out, open(f'{RES}/size_matched.json', 'w'), indent=1)
    print(f'\nwrote {RES}/size_matched.json')


if __name__ == '__main__':
    main()
