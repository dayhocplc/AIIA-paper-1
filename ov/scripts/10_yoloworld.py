#!/usr/bin/env python3
"""YOLO-World arm — a third architecture family.

Two deviations from the main protocol, both forced and both declared:

1. **Score floor.** The main runs use a floor of 0.02. YOLO-World's maximum
   confidence on this imagery is ~0.0015, i.e. more than an order of magnitude
   below the *floor* of the other detectors, so a 0.02 floor yields zero
   predictions. We use 1e-5 here. This is consistent with the study's stated
   principle that confidence thresholds are swept rather than fixed, because
   open-vocabulary detectors are calibrated differently; it is nonetheless a
   per-model deviation and is reported as such.

2. **Model re-instantiation per prompt.** Calling `set_classes()` a second time
   on a CUDA-resident YOLOWorld in this ultralytics version raises a device
   mismatch (the CLIP text encoder stays on CPU). We construct a fresh model for
   each prompt, which is slower but avoids the bug.
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
IMGDIR = f'{ROOT}/repro/mmdet_mit/data/images'
WD = f'{ROOT}/ov/work_dirs'
CKPT = 'yolov8s-worldv2.pt'
FLOOR = 1e-5
TOPK = 100

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROMPTS = {
    'A':  (0, ['weed', 'weeds']),
    'B':  (0, ['grass', 'broadleaf plant', 'green plant', 'vine']),
    'C1': (0, ['plant that is not sugarcane', 'vegetation other than the crop']),
    'C2': (0, ['unwanted plant among sugarcane', 'volunteer plant in a cane field',
               'plant growing between cane rows']),
    'D':  (1, ['sugarcane', 'sugarcane leaf']),
    'E':  (0, ['tall green leafy plant in a field', 'broad flat green foliage on soil']),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='all', choices=['all', 'panel', 'neg'])
    ap.add_argument('--groups', default='A,B,C1,C2,D,E')
    args = ap.parse_args()
    from ultralytics import YOLOWorld
    if args.set == 'neg':
        # Same id convention as 01_infer.py: negatives carry negative ids so they
        # can never collide with the annotated set.
        negs = json.load(open(f'{WD}/negatives.json'))
        items = [{'id': -(i + 1), 'file_name': f"{negs['dir']}/{f}"}
                 for i, f in enumerate(negs['files'])]
    else:
        ev = json.load(open(f'{WD}/eval2c_{args.set}.json'))
        items = ev['images']
    outdir = f'{WD}/preds/yoloworld-s_{args.set}'
    os.makedirs(outdir, exist_ok=True)

    groups = args.groups.split(',')
    todo = [(g, p) for g, (_, ps) in PROMPTS.items() for p in ps if g in groups]
    t0 = time.time()
    maxconf = {}

    for k, (g, prompt) in enumerate(todo):
        model = YOLOWorld(CKPT)            # fresh instance: see docstring note 2
        model.set_classes([prompt])
        cat = PROMPTS[g][0]
        preds, mx = [], 0.0
        for im in items:
            r = model.predict(os.path.join(IMGDIR, im['file_name']),
                              conf=FLOOR, imgsz=640, verbose=False)[0]
            if not len(r.boxes):
                continue
            xyxy = r.boxes.xyxy.tolist(); conf = r.boxes.conf.tolist()
            mx = max(mx, max(conf))
            order = sorted(range(len(conf)), key=lambda i: -conf[i])[:TOPK]
            for i in order:
                x1, y1, x2, y2 = xyxy[i]
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                preds.append({'image_id': im['id'], 'category_id': cat,
                              'bbox': [round(x1, 1), round(y1, 1),
                                       round(x2 - x1, 1), round(y2 - y1, 1)],
                              'score': round(float(conf[i]), 8)})
        maxconf[f'{g}|{prompt}'] = mx
        json.dump(preds, open(f'{outdir}/{g}__{prompt.replace(" ", "_")}.json', 'w'))
        print(f'[{k+1}/{len(todo)}] {g:2s} {prompt[:34]:34s} '
              f'{len(preds):6d} preds  maxconf={mx:.6f}  {time.time()-t0:.0f}s', flush=True)
        del model

    json.dump({'model': 'yoloworld-s', 'ckpt': CKPT, 'set': args.set,
               'groups': groups,
               'n_images': len(items), 'score_floor': FLOOR, 'topk': TOPK,
               'max_confidence_per_prompt': maxconf,
               'protocol_deviation': 'score floor 1e-5 instead of 0.02; model '
                                     're-instantiated per prompt (device bug)',
               'seconds': round(time.time() - t0, 1)},
              open(f'{outdir}/_meta.json', 'w'), indent=2)
    print(f'\ndone in {time.time()-t0:.0f}s; global max confidence = '
          f'{max(maxconf.values()):.6f}')


if __name__ == '__main__':
    main()
