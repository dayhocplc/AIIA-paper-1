#!/usr/bin/env python3
"""Evaluate open-vocabulary predictions against the 2-class sugarcane/weed GT.

Three metric families are reported, deliberately:

1. Instance AP  (AP50, AP30, AP25 via pycocotools).  The conventional number.
   For the `sugarcane` category it is NOT comparable to `weed`, because the two
   classes were annotated under different protocols -- weed at instance level
   (median 3.3% of frame, 4 boxes/image), sugarcane at region level (median
   37.6% of frame, 2 boxes covering ~88% of the frame).  Reported for
   completeness and explicitly flagged.

2. Union-mask IoU.  Granularity-invariant: per image, rasterise the union of GT
   boxes of a class and the union of predicted boxes above a threshold, then
   take IoU of the two unions.  Identical protocol for both classes, so this is
   the metric that carries the nominal-vs-relational comparison (G1).

3. Best-F1 over a confidence sweep, at IoU 0.50 and 0.25, matched to the
   human-human agreement protocol used in repro/exp_phase1.py so the numbers
   are directly comparable to the F1 = 0.391 inter-expert figure and the
   F1 = 0.399 label-noise ceiling.

Image-level bootstrap CIs throughout; paired bootstrap lives in 03_compare.py.
"""
import argparse, json, os, glob, sys
import numpy as np

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
WD = f'{ROOT}/ov/work_dirs'
GRID = 24            # raster downsample factor for union-IoU
SWEEP = [round(x, 3) for x in np.arange(0.02, 0.95, 0.01)]


# ------------------------------------------------------------------ utilities
def boxes_to_mask(boxes, W, H):
    """Rasterise a union of xywh boxes onto a coarse boolean grid."""
    gw, gh = max(1, W // GRID), max(1, H // GRID)
    m = np.zeros((gh, gw), dtype=bool)
    for x, y, w, h in boxes:
        x0 = int(np.floor(x / GRID)); y0 = int(np.floor(y / GRID))
        x1 = int(np.ceil((x + w) / GRID)); y1 = int(np.ceil((y + h) / GRID))
        m[max(0, y0):min(gh, y1), max(0, x0):min(gw, x1)] = True
    return m


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def iou_matrix(preds_sorted, gt):
    """[n_pred, n_gt] IoU, computed once per image and reused across thresholds."""
    if not preds_sorted or not gt:
        return np.zeros((len(preds_sorted), len(gt)))
    P = np.array([p['bbox'] for p in preds_sorted], dtype=float)
    G = np.array([g['bbox'] for g in gt], dtype=float)
    px1, py1 = P[:, 0:1], P[:, 1:2]
    px2, py2 = px1 + P[:, 2:3], py1 + P[:, 3:4]
    gx1, gy1 = G[None, :, 0], G[None, :, 1]
    gx2, gy2 = gx1 + G[None, :, 2], gy1 + G[None, :, 3]
    iw = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    ih = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    inter = iw * ih
    union = (P[:, 2:3] * P[:, 3:4]) + (G[None, :, 2] * G[None, :, 3]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_f1_from_matrix(M, n_keep, n_gt, thr_iou):
    """Greedy score-ordered matching on the first n_keep rows of a precomputed
    IoU matrix (rows already sorted by descending score). Mirrors the matching
    used for the inter-expert agreement in repro/exp_phase1.py."""
    used = np.zeros(n_gt, dtype=bool)
    tp = 0
    for r in range(n_keep):
        row = M[r]
        best, bi = thr_iou, -1
        for i in range(n_gt):
            if not used[i] and row[i] >= best:
                best, bi = row[i], i
        if bi >= 0:
            used[bi] = True; tp += 1
    return tp, n_keep - tp, n_gt - tp


def f1_from(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


# ------------------------------------------------------------------ per-image
def per_image_stats(gt_by_img, pred_by_img, imgs, cat):
    """Precompute everything each bootstrap replicate needs, per image."""
    out = {}
    for im in imgs:
        i = im['id']
        g = [a for a in gt_by_img.get(i, []) if a['category_id'] == cat]
        p = [q for q in pred_by_img.get(i, []) if q['category_id'] == cat]
        gm = boxes_to_mask([a['bbox'] for a in g], im['width'], im['height'])
        rec = {'n_gt': len(g), 'gt_mask_area': int(gm.sum()), 'sweep': {}}

        # sort once by descending score; a threshold is then just a row prefix
        p = sorted(p, key=lambda q: -q['score'])
        scores = np.array([q['score'] for q in p]) if p else np.zeros(0)
        M = iou_matrix(p, g)

        # Trivial floor: predicting the WHOLE FRAME scores union-IoU equal to the
        # ground-truth coverage fraction.  Because `sugarcane` covers 80% of the
        # frame and `weed` only 34%, raw union-IoU is confounded with class
        # prevalence and cannot carry a cross-class comparison.  We therefore
        # also record Cohen's kappa on the pixel grid, which is prevalence
        # corrected, and the trivial floor itself, for transparency.
        n_cells = gm.size
        gt_pos = int(gm.sum())
        rec['uiou_trivial'] = gt_pos / n_cells if n_cells else 0.0

        for t in SWEEP:
            n_keep = int(np.searchsorted(-scores, -t, side='right')) if len(scores) else 0
            pm = boxes_to_mask([q['bbox'] for q in p[:n_keep]], im['width'], im['height'])
            inter = int((gm & pm).sum()); union = int((gm | pm).sum())
            pr_pos = int(pm.sum())
            po = (inter + int((~gm & ~pm).sum())) / n_cells
            pe = (gt_pos * pr_pos + (n_cells - gt_pos) * (n_cells - pr_pos)) / n_cells ** 2
            r = {'uiou': inter / union if union > 0 else (1.0 if rec['n_gt'] == 0 else 0.0),
                 'kappa': (po - pe) / (1 - pe) if pe < 1 else 0.0,
                 'n_pred': n_keep}
            for ti in (0.50, 0.25):
                tp, fp, fn = greedy_f1_from_matrix(M, n_keep, len(g), ti)
                r[f'tp{ti}'] = tp; r[f'fp{ti}'] = fp; r[f'fn{ti}'] = fn
            rec['sweep'][t] = r
        out[i] = rec
    return out


def agg(stats, ids, thr):
    """Aggregate per-image stats over an image subset at one threshold."""
    n = len(ids)
    if n == 0:
        return None
    res = {'uiou': float(np.mean([stats[i]['sweep'][thr]['uiou'] for i in ids])),
           'kappa': float(np.mean([stats[i]['sweep'][thr]['kappa'] for i in ids]))}
    for ti in (0.50, 0.25):
        tp = sum(stats[i]['sweep'][thr][f'tp{ti}'] for i in ids)
        fp = sum(stats[i]['sweep'][thr][f'fp{ti}'] for i in ids)
        fn = sum(stats[i]['sweep'][thr][f'fn{ti}'] for i in ids)
        res[f'f1@{ti}'] = f1_from(tp, fp, fn)
    return res


def best_over_sweep(stats, ids, key):
    best, bt = -1, None
    for t in SWEEP:
        v = agg(stats, ids, t)[key]
        if v > best:
            best, bt = v, t
    return best, bt


def boot_ci(stats, ids, key, thr, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(ids)
    vals = []
    for _ in range(n_boot):
        s = rng.choice(ids, size=len(ids), replace=True)
        vals.append(agg(stats, list(s), thr)[key])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def coco_ap(gt_json, preds, cat, img_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io, contextlib
    sub = {'images': [im for im in gt_json['images'] if im['id'] in img_ids],
           'annotations': [a for a in gt_json['annotations']
                           if a['image_id'] in img_ids and a['category_id'] == cat],
           'categories': gt_json['categories']}
    if not sub['annotations']:
        return {}
    pr = [p for p in preds if p['image_id'] in img_ids and p['category_id'] == cat]
    if not pr:
        return {'AP50': 0.0, 'AP30': 0.0, 'AP25': 0.0}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = COCO(); c.dataset = sub; c.createIndex()
        d = c.loadRes(pr)
        e = COCOeval(c, d, 'bbox')
        e.params.iouThrs = np.array([0.25, 0.30, 0.50])
        e.params.catIds = [cat]
        e.params.maxDets = [1, 10, 100]
        e.evaluate(); e.accumulate()
        prec = e.eval['precision']            # [T, R, K, A, M]
    out = {}
    for k, name in [(0, 'AP25'), (1, 'AP30'), (2, 'AP50')]:
        p = prec[k, :, 0, 0, 2]
        out[name] = float(np.mean(p[p > -1])) * 100 if (p > -1).any() else 0.0
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='all')
    ap.add_argument('--preds-set', default='',
                    help='read predictions from a different set directory; the '
                         'image id space is shared, so "all" predictions can be '
                         'scored on the "test" subset without re-running inference')
    ap.add_argument('--models', default='')
    ap.add_argument('--out', default=f'{ROOT}/ov/results/eval_main.json')
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()

    gt = json.load(open(f'{WD}/eval2c_{args.set}.json'))
    imgs = gt['images']
    gt_by_img = {}
    for a in gt['annotations']:
        gt_by_img.setdefault(a['image_id'], []).append(a)
    all_ids = [im['id'] for im in imgs]
    cane_ids = [im['id'] for im in imgs if im['has_cane']]
    nocane_ids = [im['id'] for im in imgs if not im['has_cane']]

    pset = args.preds_set or args.set
    model_dirs = sorted(glob.glob(f'{WD}/preds/*_{pset}'))
    if args.models:
        keep = args.models.split(',')
        model_dirs = [d for d in model_dirs
                      if os.path.basename(d).rsplit('_', 1)[0] in keep]

    results = {'set': args.set, 'n_images': len(imgs),
               'n_with_cane': len(cane_ids), 'n_without_cane': len(nocane_ids),
               'runs': {}}

    for md in model_dirs:
        mkey = os.path.basename(md).rsplit('_', 1)[0]
        for pf in sorted(glob.glob(f'{md}/*.json')):
            base = os.path.basename(pf)
            if base.startswith('_'):
                continue
            grp, prompt = base[:-5].split('__', 1)
            prompt = prompt.replace('_', ' ')
            preds = json.load(open(pf))
            cat = 1 if grp == 'D' else 0
            pred_by_img = {}
            for p in preds:
                pred_by_img.setdefault(p['image_id'], []).append(p)

            st = per_image_stats(gt_by_img, pred_by_img, imgs, cat)
            rec = {'group': grp, 'prompt': prompt, 'category': cat,
                   'n_preds': len(preds)}

            # region-level metrics.  union_iou is reported but is confounded with
            # class prevalence (see uiou_trivial); kappa is the prevalence-
            # corrected version used for the cross-class D contrast.
            u, ut = best_over_sweep(st, all_ids, 'uiou')
            lo, hi = boot_ci(st, all_ids, 'uiou', ut, args.boot)
            rec['union_iou'] = {'best': u, 'thr': ut, 'lo': lo, 'hi': hi}
            rec['uiou_trivial'] = float(np.mean([st[i]['uiou_trivial'] for i in all_ids]))
            kv, kt = best_over_sweep(st, all_ids, 'kappa')
            klo, khi = boot_ci(st, all_ids, 'kappa', kt, args.boot)
            rec['kappa'] = {'best': kv, 'thr': kt, 'lo': klo, 'hi': khi}

            # best-F1 sweeps, comparable to the human-human / ceiling numbers
            for ti in (0.50, 0.25):
                k = f'f1@{ti}'
                v, t = best_over_sweep(st, all_ids, k)
                lo, hi = boot_ci(st, all_ids, k, t, args.boot)
                rec[f'bestF1@{ti}'] = {'best': v, 'thr': t, 'lo': lo, 'hi': hi}

            # stratified union IoU: relational prompts only make sense with cane
            for name, ids in [('with_cane', cane_ids), ('without_cane', nocane_ids)]:
                if ids:
                    v, t = best_over_sweep(st, ids, 'uiou')
                    rec[f'union_iou_{name}'] = {'best': v, 'thr': t}

            rec['coco'] = coco_ap(gt, preds, cat, set(all_ids))
            results['runs'][f'{mkey}|{grp}|{prompt}'] = rec
            print(f'{mkey:12s} {grp:2s} {prompt[:34]:34s} '
                  f'uIoU {u:.3f} [{rec["union_iou"]["lo"]:.3f}-{rec["union_iou"]["hi"]:.3f}] '
                  f'F1@.5 {rec["bestF1@0.5"]["best"]:.3f} '
                  f'AP50 {rec["coco"].get("AP50", 0):.1f}', flush=True)

            # Cache per-image quantities for the paired bootstrap stage.  F1 must
            # be recomputed from resampled tp/fp/fn counts (micro-averaged), not
            # averaged over per-image F1 values, so the counts are stored too.
            cd = f'{WD}/stats/{args.set}'
            os.makedirs(cd, exist_ok=True)
            np.savez_compressed(
                f'{cd}/{mkey}__{grp}__{prompt.replace(" ", "_")}.npz',
                uiou=np.array([[st[i]['sweep'][t]['uiou'] for t in SWEEP] for i in all_ids]),
                kappa=np.array([[st[i]['sweep'][t]['kappa'] for t in SWEEP] for i in all_ids]),
                tp=np.array([[st[i]['sweep'][t]['tp0.5'] for t in SWEEP] for i in all_ids]),
                fp=np.array([[st[i]['sweep'][t]['fp0.5'] for t in SWEEP] for i in all_ids]),
                fn=np.array([[st[i]['sweep'][t]['fn0.5'] for t in SWEEP] for i in all_ids]))

    # Image ordering and per-class GT presence, needed by the paired stage.
    # An image with no GT of a class scores union-IoU = 1.0 when the model also
    # predicts nothing there, which would hand group D (sugarcane) 42 free
    # perfect scores.  The D contrasts are therefore restricted downstream to
    # the images that carry BOTH classes.
    has = {c: {a['image_id'] for a in gt['annotations'] if a['category_id'] == c}
           for c in (0, 1)}
    json.dump({'image_ids': all_ids,
               'has_weed': [i in has[0] for i in all_ids],
               'has_cane_gt': [i in has[1] for i in all_ids],
               'both': [(i in has[0]) and (i in has[1]) for i in all_ids],
               'sweep': SWEEP},
              open(f'{WD}/stats/{args.set}/_index.json', 'w'))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, 'w'), indent=1)
    print(f'\nwrote {args.out}  ({len(results["runs"])} runs)')


if __name__ == '__main__':
    main()
