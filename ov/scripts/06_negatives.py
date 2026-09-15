#!/usr/bin/env python3
"""False-alarm rate on the 869 expert-confirmed weed-free images.

There is no ground truth to match against on this set, so the quantity of
interest is simply how often a weed prompt fires where no weed exists.

To avoid selection effects the headline number is taken at each run's
*operating threshold*: the confidence that maximised best-F1 on the annotated
positive set, fixed before this set is scored.  The full threshold curve is
reported alongside, because a single operating point can hide a cliff.
"""
import json, os, glob
import numpy as np

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
WD = f'{ROOT}/ov/work_dirs'
RES = f'{ROOT}/ov/results'
THRS = [round(x, 3) for x in np.arange(0.02, 0.95, 0.02)]

main = json.load(open(f'{RES}/eval_main.json'))
n_neg = len(json.load(open(f'{WD}/negatives.json'))['files'])

out = {'n_images': n_neg, 'thresholds': THRS, 'runs': {}}

for md in sorted(glob.glob(f'{WD}/preds/*_neg')):
    mkey = os.path.basename(md).rsplit('_', 1)[0]
    for pf in sorted(glob.glob(f'{md}/*.json')):
        base = os.path.basename(pf)
        if base.startswith('_'):
            continue
        grp, prompt = base[:-5].split('__', 1)
        prompt = prompt.replace('_', ' ')
        preds = json.load(open(pf))

        by_img = {}
        for p in preds:
            by_img.setdefault(p['image_id'], []).append(p['score'])

        frac, per_img = [], []
        for t in THRS:
            hits = sum(1 for v in by_img.values() if any(s >= t for s in v))
            ndet = sum(sum(1 for s in v if s >= t) for v in by_img.values())
            frac.append(hits / n_neg)
            per_img.append(ndet / n_neg)

        # operating threshold carried over from the positive set
        key = f'{mkey}|{grp}|{prompt}'
        pos = main['runs'].get(key)
        op = pos['bestF1@0.5']['thr'] if pos else None
        fo = do = None
        if op is not None:
            hits = sum(1 for v in by_img.values() if any(s >= op for s in v))
            ndet = sum(sum(1 for s in v if s >= op) for v in by_img.values())
            fo, do = hits / n_neg, ndet / n_neg

        # On these images sugarcane IS present, so a group-D detection is a
        # TRUE positive, not a false alarm.  Groups A and C2 target weed, which
        # experts confirmed absent, so any detection there is a false alarm.
        out['runs'][key] = {
            'group': grp, 'prompt': prompt,
            'role': 'true_positive_rate' if grp == 'D' else 'false_alarm_rate',
            'thresholds': THRS,
            'frac_images_with_detection': frac,
            'detections_per_image': per_img,
            'operating_threshold': op,
            'frac_at_operating': fo,
            'detections_per_image_at_operating': do,
        }
        if op is not None:
            role = 'TPR ' if grp == 'D' else 'FALSE'
            print(f'{mkey:12s} {grp:2s} {prompt[:31]:31s} thr={op:.2f}  '
                  f'{role} {fo*100:5.1f}% of images fire  ({do:.2f} det/img)')

os.makedirs(RES, exist_ok=True)
json.dump(out, open(f'{RES}/negatives.json', 'w'))
print(f'\nwrote {RES}/negatives.json  ({len(out["runs"])} runs, n={n_neg} images)')
