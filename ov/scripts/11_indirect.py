#!/usr/bin/env python3
"""Indirect open-vocabulary weed detection: prompt for the CROP, take the complement.

"Weed" is defined relationally - a plant that is not the crop - so the natural
computable form of the definition is not to ask a detector for "weed" at all, but
to ask it for the crop and subtract. This is the open-vocabulary version of the
`indirect weed detection` idea already present in the weed-management literature
(label only the crop; treat non-crop vegetation as weed).

Pipeline, per image:
    vegetation  = ExG (excess green) mask, Otsu-thresholded
    crop        = union of `sugarcane` boxes above confidence t  (group D preds,
                  reused - no new inference needed)
    proposals   = connected components of (vegetation AND NOT crop)
                  -> boxes, ranked by area

The decisive ablation is the middle row of the output: if `vegetation only`
scores the same as `vegetation minus crop`, the open-vocabulary step contributes
nothing and the method is just a greenness filter.

    ABL-1  vegetation only              no open-vocabulary model at all
    ABL-2  vegetation minus crop        the indirect OV method
    REF    direct prompt "weed"         group A, already measured
"""
import json, os, glob, sys
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
IMGDIR = f'{ROOT}/repro/mmdet_mit/data/images'
WD = f'{ROOT}/ov/work_dirs'
RES = f'{ROOT}/ov/results'

SCALE = 8                     # work at 1/8 resolution; boxes scaled back up
MIN_AREA_FRAC = 0.004         # drop components below 0.4 % of frame
CLOSE_ITERS = 2
CROP_THRS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
MAX_PROPOSALS = 20


def veg_mask(path):
    """Excess-green vegetation mask at 1/8 scale, Otsu threshold."""
    im = Image.open(path)
    im.draft('RGB', (im.width // SCALE, im.height // SCALE))
    a = np.asarray(im.convert('RGB'), dtype=np.float32)
    s = a.sum(2) + 1e-6
    r, g, b = a[..., 0] / s, a[..., 1] / s, a[..., 2] / s
    exg = 2 * g - r - b
    # Otsu on the ExG histogram
    h, edges = np.histogram(exg, bins=256)
    p = h.astype(float) / h.sum()
    om = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    m0 = np.cumsum(p * om) / np.maximum(w0, 1e-9)
    m1 = (np.cumsum((p * om)[::-1])[::-1]) / np.maximum(w1, 1e-9)
    var = w0 * w1 * (m0 - m1) ** 2
    thr = om[int(np.nanargmax(var))]
    return exg > thr, a.shape[1], a.shape[0]


def boxes_from_mask(m, W8, H8):
    """Connected components -> xywh boxes in ORIGINAL image coordinates."""
    m = ndimage.binary_closing(m, np.ones((3, 3)), iterations=CLOSE_ITERS)
    lab, n = ndimage.label(m)
    if n == 0:
        return []
    out = []
    min_px = MIN_AREA_FRAC * W8 * H8
    for sl, i in zip(ndimage.find_objects(lab), range(1, n + 1)):
        area = int((lab[sl] == i).sum())
        if area < min_px:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        out.append(([x0 * SCALE, y0 * SCALE, (x1 - x0) * SCALE, (y1 - y0) * SCALE], area))
    out.sort(key=lambda t: -t[1])
    out = out[:MAX_PROPOSALS]
    if not out:
        return []
    amax = out[0][1]
    return [{'bbox': [float(v) for v in b], 'score': a / amax} for b, a in out]


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


def counts(preds, gt, thr_iou=0.5):
    preds = sorted(preds, key=lambda p: -p['score'])
    M = iou_mat([p['bbox'] for p in preds], gt)
    used = np.zeros(len(gt), bool); tp = 0
    for r in range(len(preds)):
        best, bi = thr_iou, -1
        for j in range(len(gt)):
            if not used[j] and M[r, j] >= best:
                best, bi = M[r, j], j
        if bi >= 0:
            used[bi] = True; tp += 1
    return tp, len(preds) - tp, len(gt) - tp


def micro(T, F, N, axis=0):
    t, f, n = T.sum(axis), F.sum(axis), N.sum(axis)
    d = 2 * t + f + n
    return np.where(d > 0, 2 * t / np.maximum(d, 1e-9), 0.0)


def main():
    ev = json.load(open(f'{WD}/eval2c_all.json'))
    imgs = ev['images']
    gt = {}
    for a in ev['annotations']:
        if a['category_id'] == 0:
            gt.setdefault(a['image_id'], []).append(a['bbox'])

    print(f'computing vegetation masks for {len(imgs)} images...', flush=True)
    veg, dims = {}, {}
    for k, im in enumerate(imgs):
        m, W8, H8 = veg_mask(os.path.join(IMGDIR, im['file_name']))
        veg[im['id']] = m; dims[im['id']] = (W8, H8)
        if (k + 1) % 50 == 0:
            print(f'  {k+1}/{len(imgs)}', flush=True)

    # ABL-1: vegetation only, no open-vocabulary model involved
    T = np.zeros(len(imgs), int); F = np.zeros_like(T); N = np.zeros_like(T)
    for r, im in enumerate(imgs):
        pr = boxes_from_mask(veg[im['id']], *dims[im['id']])
        T[r], F[r], N[r] = counts(pr, gt.get(im['id'], []))
    veg_only = float(micro(T, F, N))
    print(f'\nABL-1  vegetation only (no OV model)      microF1 = {veg_only:.4f}')

    out = {'vegetation_only': veg_only, 'scale': SCALE,
           'min_area_frac': MIN_AREA_FRAC, 'models': {}}

    # ABL-2: vegetation minus crop detections, per detector, sweeping crop threshold
    for md in sorted(glob.glob(f'{WD}/preds/*_all')):
        mkey = os.path.basename(md).rsplit('_', 1)[0]
        dfiles = sorted(glob.glob(f'{md}/D__*.json'))
        if not dfiles:
            continue
        crop = {}
        for f in dfiles:
            for p in json.load(open(f)):
                crop.setdefault(p['image_id'], []).append(p)

        best = {'f1': -1}
        per_thr = {}
        for t in CROP_THRS:
            T = np.zeros(len(imgs), int); F = np.zeros_like(T); N = np.zeros_like(T)
            for r, im in enumerate(imgs):
                i = im['id']; W8, H8 = dims[i]
                cm = np.zeros((H8, W8), bool)
                for p in crop.get(i, []):
                    if p['score'] < t:
                        continue
                    x, y, w, h = [v / SCALE for v in p['bbox']]
                    cm[max(0, int(y)):min(H8, int(y + h)),
                       max(0, int(x)):min(W8, int(x + w))] = True
                pr = boxes_from_mask(veg[i] & ~cm, W8, H8)
                T[r], F[r], N[r] = counts(pr, gt.get(i, []))
            f1 = float(micro(T, F, N))
            per_thr[t] = f1
            if f1 > best['f1']:
                best = {'f1': f1, 'thr': t, 'T': T.copy(), 'F': F.copy(), 'N': N.copy()}

        # paired bootstrap against the direct prompt "weed" (group A)
        st = np.load(f'{WD}/stats/all/{mkey}__A__weed.npz') if os.path.exists(
            f'{WD}/stats/all/{mkey}__A__weed.npz') else None
        rec = {'best_f1': best['f1'], 'best_crop_thr': best['thr'],
               'per_threshold': per_thr}
        if st is not None:
            fa = micro(st['tp'], st['fp'], st['fn'])
            ta = int(np.argmax(fa))
            A = [st['tp'][:, ta], st['fp'][:, ta], st['fn'][:, ta]]
            B = [best['T'], best['F'], best['N']]
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(imgs), size=(5000, len(imgs)))
            d = (micro(*[x[idx] for x in B], axis=1)
                 - micro(*[x[idx] for x in A], axis=1))
            lo, hi = np.percentile(d, [2.5, 97.5])
            rec['direct_f1'] = float(fa[ta])
            rec['indirect_minus_direct'] = {
                'diff': best['f1'] - float(fa[ta]), 'lo': float(lo), 'hi': float(hi),
                'significant': bool(lo > 0 or hi < 0)}
        out['models'][mkey] = rec
        d = rec.get('indirect_minus_direct')
        print(f'ABL-2  {mkey:12s} indirect {best["f1"]:.4f} (crop thr {best["thr"]:.2f})'
              + (f'   direct {rec["direct_f1"]:.4f}   diff {d["diff"]:+.4f} '
                 f'[{d["lo"]:+.4f},{d["hi"]:+.4f}] {"*" if d["significant"] else ""}'
                 if d else ''))

    json.dump(out, open(f'{RES}/indirect.json', 'w'), indent=1)
    print(f'\nwrote {RES}/indirect.json')


if __name__ == '__main__':
    main()
