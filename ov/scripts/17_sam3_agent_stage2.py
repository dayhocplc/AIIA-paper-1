#!/usr/bin/env python3
"""SAM 3 Agent, stage 2: prompt SAM 3 with the MLLM-proposed noun phrases.

Each image gets its own 1-3 noun phrases from stage 1. SAM 3 is run once per
phrase and the detections are pooled, which is the whole point of the Agent
design: the complex relational goal never reaches SAM 3, only simple noun
phrases do.

Fallback: an image for which the MLLM produced no usable phrase falls back to
the bare noun "weed", so the arm is never credited with skipping hard images.
"""
import json, os, time
import torch
from PIL import Image

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
IMGDIR = f'{ROOT}/repro/mmdet_mit/data/images'
WD = f'{ROOT}/ov/work_dirs'
NPS = f'{WD}/sam3_agent_nps.json'
FLOOR = 0.02
TOPK = 100
FALLBACK = 'weed'


def load_image(path):
    img = Image.open(path)
    img.draft('RGB', (img.width // 2, img.height // 2))
    return img.convert('RGB')


def main():
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained('facebook/sam3')
    model = Sam3Model.from_pretrained('facebook/sam3').eval().cuda()
    nps = json.load(open(NPS))
    print(f'SAM3 loaded, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    for setname in ('all', 'panel'):
        ev = json.load(open(f'{WD}/eval2c_{setname}.json'))
        outdir = f'{WD}/preds/sam3agent_{setname}'
        os.makedirs(outdir, exist_ok=True)
        preds, used_fallback, t0 = [], 0, time.time()

        for n, im in enumerate(ev['images']):
            rec = nps.get(f'{setname}|{im["id"]}', {})
            phrases = rec.get('nps') or []
            if not phrases:
                phrases = [FALLBACK]; used_fallback += 1
            img = load_image(os.path.join(IMGDIR, im['file_name']))
            W, H = im['width'], im['height']
            pool = []
            for ph in phrases:
                inputs = proc(images=img, text=ph, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    out = model(**inputs)
                r = proc.post_process_object_detection(
                    out, threshold=FLOOR, target_sizes=[(H, W)])[0]
                pool += list(zip(r['boxes'].tolist(), r['scores'].tolist()))
            pool.sort(key=lambda t: -t[1])
            for (x1, y1, x2, y2), s in pool[:TOPK]:
                x1 = max(0.0, min(x1, W)); x2 = max(0.0, min(x2, W))
                y1 = max(0.0, min(y1, H)); y2 = max(0.0, min(y2, H))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                preds.append({'image_id': im['id'], 'category_id': 0,
                              'bbox': [round(x1, 1), round(y1, 1),
                                       round(x2 - x1, 1), round(y2 - y1, 1)],
                              'score': round(float(s), 5)})
            if (n + 1) % 50 == 0 or n + 1 == len(ev['images']):
                el = time.time() - t0
                print(f'  {setname} {n+1}/{len(ev["images"])}  {el:.0f}s', flush=True)

        json.dump(preds, open(f'{outdir}/AGENT__mllm_noun_phrases.json', 'w'))
        json.dump({'model': 'sam3agent', 'set': setname,
                   'n_images': len(ev['images']), 'score_floor': FLOOR,
                   'topk': TOPK, 'fallback_images': used_fallback,
                   'fallback_prompt': FALLBACK,
                   'stage1_model': 'Qwen/Qwen2.5-VL-3B-Instruct',
                   'seconds': round(time.time() - t0, 1)},
                  open(f'{outdir}/_meta.json', 'w'), indent=2)
        print(f'{setname}: {len(preds)} du doan, {used_fallback} anh dung fallback')


if __name__ == '__main__':
    main()
