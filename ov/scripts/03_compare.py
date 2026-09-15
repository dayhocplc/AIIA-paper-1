#!/usr/bin/env python3
"""Paired image-level bootstrap on the DIFFERENCE between prompt groups.

Metric choice is not cosmetic here:

* **Within the weed class** (A, B, C1, C2, E) every group is scored against
  identical ground truth, so instance-level micro-F1 at IoU 0.50 is valid and
  is what discriminates.  It is also the quantity directly comparable to the
  inter-expert agreement (F1 = 0.391) and the label-noise ceiling (0.399).
* **Across classes** (any contrast involving D = `sugarcane`) instance F1 is
  invalid, because the two classes were annotated at different granularity.
  Raw union-mask IoU is invalid too: predicting the whole frame scores the
  ground-truth coverage fraction, which is 0.34 for weed and 0.80 for
  sugarcane, so union-IoU is confounded with class prevalence.  Cross-class
  contrasts therefore use Cohen's kappa on the pixel grid, which is prevalence
  corrected, and are restricted to the images carrying both classes.

F1 is micro-averaged: each bootstrap replicate re-sums tp/fp/fn over the
resampled images and recomputes F1, rather than averaging per-image F1.
Each group's threshold is frozen from the full sample before resampling, so the
interval does not inherit the selection optimism of the sweep.
"""
import argparse, json, os, glob
import numpy as np

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
WD = f'{ROOT}/ov/work_dirs'
SWEEP = [round(x, 3) for x in np.arange(0.02, 0.95, 0.01)]

# (a, b, what it isolates, metric)
CONTRASTS = [('C2', 'E',  'relationality (length held constant)', 'f1'),
             ('C1', 'C2', 'negation',                             'f1'),
             ('E',  'A',  'prompt length / compositionality',     'f1'),
             ('B',  'A',  'morphological vs abstract noun',       'f1'),
             ('D',  'A',  'category type (nominal vs relational)', 'kappa')]


def load_group(model, group, subset):
    """Average the per-image curves of every prompt in a group."""
    fs = sorted(glob.glob(f'{WD}/stats/{subset}/{model}__{group}__*.npz'))
    if not fs:
        return None
    ds = [np.load(f) for f in fs]
    return {k: np.mean([d[k] for d in ds], axis=0)
            for k in ('uiou', 'kappa', 'tp', 'fp', 'fn')}


def f1_curve(g):
    tp, fp, fn = g['tp'].sum(0), g['fp'].sum(0), g['fn'].sum(0)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-9), 0.0)


def paired_f1(a, b, sub, n_boot, seed=0):
    ta = int(np.argmax(f1_curve({k: v[sub] for k, v in a.items()})))
    tb = int(np.argmax(f1_curve({k: v[sub] for k, v in b.items()})))
    A = {k: a[k][sub][:, ta] for k in ('tp', 'fp', 'fn')}
    B = {k: b[k][sub][:, tb] for k in ('tp', 'fp', 'fn')}

    def mf1(d, idx):
        tp, fp, fn = d['tp'][idx].sum(1), d['fp'][idx].sum(1), d['fn'][idx].sum(1)
        den = 2 * tp + fp + fn
        return np.where(den > 0, 2 * tp / np.maximum(den, 1e-9), 0.0)

    n = int(sub.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = mf1(A, idx) - mf1(B, idx)
    fa = 2 * A['tp'].sum() / max(2 * A['tp'].sum() + A['fp'].sum() + A['fn'].sum(), 1e-9)
    fb = 2 * B['tp'].sum() / max(2 * B['tp'].sum() + B['fp'].sum() + B['fn'].sum(), 1e-9)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(metric='microF1@0.50', a_mean=float(fa), b_mean=float(fb),
                a_thr=SWEEP[ta], b_thr=SWEEP[tb], diff=float(fa - fb),
                lo=float(lo), hi=float(hi), n_images=n,
                p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                significant=bool(lo > 0 or hi < 0))


def paired_mean(a, b, sub, key, n_boot, seed=0):
    ca, cb = a[key][sub], b[key][sub]
    ta, tb = int(np.argmax(ca.mean(0))), int(np.argmax(cb.mean(0)))
    va, vb = ca[:, ta], cb[:, tb]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(va), size=(n_boot, len(va)))
    d = va[idx].mean(1) - vb[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(metric=f'{key}(pixel grid)', a_mean=float(va.mean()),
                b_mean=float(vb.mean()), a_thr=SWEEP[ta], b_thr=SWEEP[tb],
                diff=float(va.mean() - vb.mean()), lo=float(lo), hi=float(hi),
                n_images=int(sub.sum()),
                p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                significant=bool(lo > 0 or hi < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='all')
    ap.add_argument('--out', default=f'{ROOT}/ov/results/contrasts.json')
    ap.add_argument('--boot', type=int, default=5000)
    args = ap.parse_args()

    idx = json.load(open(f'{WD}/stats/{args.set}/_index.json'))
    both = np.array(idx['both'])
    n_all = len(both)
    allm = np.ones(n_all, dtype=bool)
    print(f'{n_all} images; paired same-pixel subset for cross-class contrasts: '
          f'{int(both.sum())}')

    models = sorted({os.path.basename(f).split('__')[0]
                     for f in glob.glob(f'{WD}/stats/{args.set}/*.npz')})
    res = {'set': args.set, 'n_boot': args.boot, 'n_images': n_all,
           'n_images_both': int(both.sum()), 'models': {}}

    for m in models:
        G = {g: load_group(m, g, args.set) for g in ['A', 'B', 'C1', 'C2', 'D', 'E']}
        G = {k: v for k, v in G.items() if v is not None}
        summ = {}
        for g, v in G.items():
            sub = both if g == 'D' else allm
            f1 = f1_curve({k: x[sub] for k, x in v.items()})
            summ[g] = {'microF1@0.50': float(f1.max()),
                       'kappa': float(v['kappa'][sub].mean(0).max()),
                       'uiou': float(v['uiou'][sub].mean(0).max())}
        res['models'][m] = {'group_scores': summ, 'contrasts': {}}

        print(f'\n=== {m} ===')
        for g in ['A', 'B', 'E', 'C1', 'C2', 'D']:
            if g in summ:
                print(f'   {g:2s}  microF1 {summ[g]["microF1@0.50"]:.4f}   '
                      f'kappa {summ[g]["kappa"]:.4f}')

        for ga, gb, label, met in CONTRASTS:
            if ga not in G or gb not in G:
                continue
            sub = both if 'D' in (ga, gb) else allm
            r = (paired_f1(G[ga], G[gb], sub, args.boot) if met == 'f1'
                 else paired_mean(G[ga], G[gb], sub, 'kappa', args.boot))
            r['label'] = label
            res['models'][m]['contrasts'][f'{ga}-{gb}'] = r
            print(f'   {ga:2s}-{gb:2s} {label:38s} {r["diff"]:+.4f} '
                  f'[{r["lo"]:+.4f},{r["hi"]:+.4f}] p={r["p_two_sided"]:.4f} '
                  f'{"*" if r["significant"] else " "}  ({r["metric"]})')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
