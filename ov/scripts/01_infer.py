#!/usr/bin/env python3
"""Open-vocabulary inference over the sugarcane/weed evaluation sets.

Usage:  01_infer.py <model_key> [--set all|test|neg] [--limit N]

Design notes that matter for correctness
----------------------------------------
* OWLv2 pads every image to a SQUARE (bottom/right, image anchored top-left)
  before resizing.  Predicted boxes are therefore normalised against
  max(H, W), not against (H, W).  We pass target_sizes = (S, S) with
  S = max(H, W) so that boxes come back in ORIGINAL pixel coordinates, then
  clip to the true frame.  Using (H, W) here silently squashes every box by
  H/W = 0.667 on this dataset.
* Grounding DINO uses an ordinary aspect-preserving resize, so (H, W) is right.
* JPEG decoding of 4608x3072 dominates runtime, so each image is decoded once
  (with PIL `draft` half-scale DCT decoding, which still leaves the short side
  at 1536 >= the 800 the models need) and reused across every prompt.
* One prompt per forward pass ("single-category oracle" mode) so that prompts
  cannot compete with one another inside the model.  For OWLv2 the per-query
  scores are independent by construction, but we keep the protocol identical
  across models rather than exploit that.
"""
import argparse, json, os, sys, time
import torch
from PIL import Image

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
IMGDIR = f'{ROOT}/repro/mmdet_mit/data/images'
WD = f'{ROOT}/ov/work_dirs'

# ---------------------------------------------------------------- prompt table
# group -> (target category id, [prompt, ...])
#   A   abstract noun               weed
#   B   morphological noun          weed
#   C1  relational, NEGATED form    weed   (confounded with the NegBench result)
#   C2  relational, AFFIRMATIVE     weed   (the actual experimental variable)
#   D   nominal positive control    sugarcane
#   E   long/compositional, NON-relational  weed  (controls prompt length, so that
#                                                  C2 - E isolates relationality)
PROMPTS = {
    'A':  (0, ['weed', 'weeds']),
    'B':  (0, ['grass', 'broadleaf plant', 'green plant', 'vine']),
    'C1': (0, ['plant that is not sugarcane', 'vegetation other than the crop']),
    'C2': (0, ['unwanted plant among sugarcane', 'volunteer plant in a cane field',
               'plant growing between cane rows']),
    'D':  (1, ['sugarcane', 'sugarcane leaf']),
    'E':  (0, ['tall green leafy plant in a field', 'broad flat green foliage on soil']),
}

MODELS = {
    'owlv2-base':  ('owlv2', 'google/owlv2-base-patch16-ensemble'),
    'owlv2-large': ('owlv2', 'google/owlv2-large-patch14-ensemble'),
    'gdino-tiny':  ('gdino', 'IDEA-Research/grounding-dino-tiny'),
    'gdino-base':  ('gdino', 'IDEA-Research/grounding-dino-base'),
    'sam3':        ('sam3',  'facebook/sam3'),
}

SCORE_FLOOR = 0.02   # keep low: the threshold sweep happens at evaluation time
TOPK = 100           # COCO convention


class Owlv2Runner:
    def __init__(self, mid):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.proc = Owlv2Processor.from_pretrained(mid)
        self.model = Owlv2ForObjectDetection.from_pretrained(mid).eval().cuda()

    def infer(self, img, prompt, W, H):
        inputs = self.proc(text=[[prompt]], images=img, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = self.model(**inputs)
        S = max(W, H)                      # square-padded reference frame
        r = self.proc.post_process_grounded_object_detection(
            out, threshold=SCORE_FLOOR, target_sizes=torch.tensor([[S, S]]))[0]
        return r['boxes'].tolist(), r['scores'].tolist()


class GdinoRunner:
    def __init__(self, mid):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = GroundingDinoForObjectDetection.from_pretrained(mid).eval().cuda()

    def infer(self, img, prompt, W, H):
        text = prompt.lower().strip()
        if not text.endswith('.'):
            text += '.'
        inputs = self.proc(images=img, text=text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = self.model(**inputs)
        r = self.proc.post_process_grounded_object_detection(
            out, inputs['input_ids'], threshold=SCORE_FLOOR,
            target_sizes=[(H, W)])[0]
        return r['boxes'].tolist(), r['scores'].tolist()


class Sam3Runner:
    """SAM 3 promptable concept segmentation, used box-only.

    SAM 3 defines its task over *simple noun phrases* and its authors state it is
    "not designed for long referring expressions or queries requiring reasoning".
    We nonetheless issue the full prompt battery, including the relational groups,
    because whether that architectural restriction actually bites is precisely the
    question. Masks are discarded: `post_process_object_detection` avoids
    materialising one 3072x4608 mask per instance.
    """
    def __init__(self, mid):
        from transformers import Sam3Processor, Sam3Model
        self.proc = Sam3Processor.from_pretrained(mid)
        self.model = Sam3Model.from_pretrained(mid).eval().cuda()

    def infer(self, img, prompt, W, H):
        inputs = self.proc(images=img, text=prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = self.model(**inputs)
        r = self.proc.post_process_object_detection(
            out, threshold=SCORE_FLOOR, target_sizes=[(H, W)])[0]
        return r['boxes'].tolist(), r['scores'].tolist()


def load_image(path):
    img = Image.open(path)
    img.draft('RGB', (img.width // 2, img.height // 2))   # fast DCT half-decode
    return img.convert('RGB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_key', choices=list(MODELS))
    ap.add_argument('--set', default='all', choices=['all', 'test', 'neg', 'panel'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--groups', default='A,B,C1,C2,D,E')
    args = ap.parse_args()

    kind, mid = MODELS[args.model_key]
    groups = args.groups.split(',')

    if args.set == 'neg':
        negs = json.load(open(f'{WD}/negatives.json'))
        items = [{'id': -(i + 1), 'file_name': f"sugarcane/{f}"}
                 for i, f in enumerate(negs['files'])]
    else:
        ev = json.load(open(f'{WD}/eval2c_{args.set}.json'))
        items = ev['images']
    if args.limit:
        items = items[:args.limit]

    runner = {'owlv2': Owlv2Runner, 'gdino': GdinoRunner,
              'sam3': Sam3Runner}[kind](mid)
    print(f'[{args.model_key}] loaded, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB, '
          f'{len(items)} images, groups {groups}', flush=True)

    todo = [(g, p) for g in groups for p in PROMPTS[g][1]]
    preds = {f'{g}|{p}': [] for g, p in todo}
    t0 = time.time()

    for n, im in enumerate(items):
        img = load_image(os.path.join(IMGDIR, im['file_name']))
        W, H = im.get('width', img.width * 2), im.get('height', img.height * 2)
        if args.set == 'neg':
            W, H = img.width * 2, img.height * 2
        for g, p in todo:
            boxes, scores = runner.infer(img, p, W, H)
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOPK]
            cat = PROMPTS[g][0]
            for i in order:
                x1, y1, x2, y2 = boxes[i]
                x1 = max(0.0, min(x1, W)); x2 = max(0.0, min(x2, W))
                y1 = max(0.0, min(y1, H)); y2 = max(0.0, min(y2, H))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                preds[f'{g}|{p}'].append({
                    'image_id': im['id'], 'category_id': cat,
                    'bbox': [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    'score': round(float(scores[i]), 5)})
        if (n + 1) % 25 == 0 or n + 1 == len(items):
            el = time.time() - t0
            print(f'  {n+1}/{len(items)}  {el:.0f}s  eta {el/(n+1)*(len(items)-n-1):.0f}s',
                  flush=True)

    outdir = f'{WD}/preds/{args.model_key}_{args.set}'
    os.makedirs(outdir, exist_ok=True)
    for key, plist in preds.items():
        g, p = key.split('|', 1)
        fn = f"{g}__{p.replace(' ', '_')}.json"
        json.dump(plist, open(os.path.join(outdir, fn), 'w'))
    meta = {'model': args.model_key, 'hf_id': mid, 'set': args.set,
            'n_images': len(items), 'score_floor': SCORE_FLOOR, 'topk': TOPK,
            'prompts': {g: PROMPTS[g] for g in groups},
            'seconds': round(time.time() - t0, 1)}
    json.dump(meta, open(f'{outdir}/_meta.json', 'w'), indent=2)
    print(f'wrote {len(preds)} prompt files to {outdir}  ({meta["seconds"]}s)')


if __name__ == '__main__':
    main()
