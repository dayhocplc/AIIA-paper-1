#!/usr/bin/env python3
"""Parameter count, peak VRAM and per-image latency for every detector in the study.

These are the numbers in the model-inventory table.  They are a *benchmark*
condition — a single 640 x 640 RGB frame through each model's own preprocessing
and post-processing, one prompt, batch size 1 — and deliberately not the study's
inference protocol, which runs at the native 4608 x 3072 with model-specific
resizing.  End-to-end throughput under the real protocol is recorded separately
in each run's `_meta.json` and is reported alongside.

Usage:  21_model_costs.py [--device cuda|cpu] [--iters 30] [--size 640]
        --device cpu counts parameters only (no VRAM or latency).

Writes ov/results/model_costs.json.
"""
import argparse, json, os, time
import numpy as np
from PIL import Image

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
WD = f'{ROOT}/ov/work_dirs'
RES = f'{ROOT}/ov/results'

HF = {
    'owlv2-base':  ('owlv2', 'google/owlv2-base-patch16-ensemble'),
    'owlv2-large': ('owlv2', 'google/owlv2-large-patch14-ensemble'),
    'gdino-tiny':  ('gdino', 'IDEA-Research/grounding-dino-tiny'),
    'gdino-base':  ('gdino', 'IDEA-Research/grounding-dino-base'),
    'yoloworld-s': ('yolo',  'yolov8s-worldv2.pt'),
    'sam3':        ('sam3',  'facebook/sam3'),
    'qwen2.5-vl-3b': ('qwen', 'Qwen/Qwen2.5-VL-3B-Instruct'),
}
RTMDET_CFG = f'{ROOT}/repro/work_dirs/convnext_ciou_clahe/mit_convnext_ciou_clahe.py'
RTMDET_CKPT = f'{ROOT}/repro/work_dirs/convnext_ciou_clahe/best_coco_bbox_mAP_50_epoch_113.pth'
PROMPT = 'weed'


def nparams(module):
    return sum(p.numel() for p in module.parameters())


def timeit(fn, iters, device):
    import torch
    for _ in range(5):                      # warm-up: cuDNN autotune, allocator
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 5)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    import torch

    dev = args.device
    img = Image.fromarray(
        (np.random.default_rng(0).random((args.size, args.size, 3)) * 255).astype('uint8'))
    # merge into any previous run: RTMDet needs a different environment, so the
    # table is assembled from more than one invocation
    prev = {}
    if os.path.exists(f'{RES}/model_costs.json'):
        prev = json.load(open(f'{RES}/model_costs.json')).get('models', {})
    out = {'device': dev, 'input_size': [args.size, args.size], 'iters': args.iters,
           'batch_size': 1, 'prompt': PROMPT, 'models': dict(prev)}
    if dev == 'cuda':
        out['gpu'] = torch.cuda.get_device_name(0)

    keys = [k for k in list(HF) + ['rtmdet'] if not args.only or k in args.only.split(',')]

    for key in keys:
        try:
            if dev == 'cuda':
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() if dev == 'cuda' else 0

            if key == 'rtmdet':
                from mmdet.apis import init_detector, inference_detector
                model = init_detector(RTMDET_CFG, RTMDET_CKPT, device=dev)
                p = nparams(model)
                arr = np.array(img)[:, :, ::-1]          # mmdet expects BGR
                run = lambda: inference_detector(model, arr)
            else:
                kind, mid = HF[key]
                if kind == 'owlv2':
                    from transformers import Owlv2Processor, Owlv2ForObjectDetection
                    pr = Owlv2Processor.from_pretrained(mid)
                    model = Owlv2ForObjectDetection.from_pretrained(mid).to(dev).eval()
                    p = nparams(model)

                    def run():
                        inp = pr(text=[[PROMPT]], images=img, return_tensors='pt').to(dev)
                        with torch.no_grad():
                            o = model(**inp)
                        pr.post_process_grounded_object_detection(
                            o, threshold=0.02,
                            target_sizes=torch.tensor([[args.size, args.size]]).to(dev))
                elif kind == 'gdino':
                    from transformers import AutoProcessor, GroundingDinoForObjectDetection
                    pr = AutoProcessor.from_pretrained(mid)
                    model = GroundingDinoForObjectDetection.from_pretrained(mid).to(dev).eval()
                    p = nparams(model)

                    def run():
                        inp = pr(images=img, text=PROMPT + '.', return_tensors='pt').to(dev)
                        with torch.no_grad():
                            o = model(**inp)
                        pr.post_process_grounded_object_detection(
                            o, inp['input_ids'], threshold=0.02,
                            target_sizes=[(args.size, args.size)])
                elif kind == 'sam3':
                    from transformers import Sam3Processor, Sam3Model
                    pr = Sam3Processor.from_pretrained(mid)
                    model = Sam3Model.from_pretrained(mid).to(dev).eval()
                    p = nparams(model)

                    def run():
                        inp = pr(images=img, text=PROMPT, return_tensors='pt').to(dev)
                        with torch.no_grad():
                            o = model(**inp)
                        pr.post_process_object_detection(
                            o, threshold=0.02, target_sizes=[(args.size, args.size)])
                elif kind == 'qwen':
                    # Stage 1 of SAM 3 Agent: bfloat16, greedy, 40 new tokens —
                    # exactly the configuration 16_sam3_agent_stage1.py runs.
                    from transformers import (Qwen2_5_VLForConditionalGeneration,
                                              AutoProcessor)
                    import importlib.util as _il
                    spec = _il.spec_from_file_location(
                        's1', os.path.join(ROOT, 'ov/scripts/16_sam3_agent_stage1.py'))
                    s1 = _il.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(s1); goal = s1.GOAL
                    except Exception:
                        goal = 'Name the weed species visible in this field photograph.'
                    pr = AutoProcessor.from_pretrained(mid)
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        mid, dtype=torch.bfloat16 if dev == 'cuda' else torch.float32,
                        device_map=dev).eval()
                    p = nparams(model)
                    msgs = [{'role': 'user', 'content': [{'type': 'image'},
                                                         {'type': 'text', 'text': goal}]}]
                    chat = pr.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)

                    def run():
                        inp = pr(text=[chat], images=[img], return_tensors='pt').to(dev)
                        with torch.no_grad():
                            model.generate(**inp, max_new_tokens=40, do_sample=False)
                elif kind == 'yolo':
                    from ultralytics import YOLOWorld
                    model = YOLOWorld(os.path.join(ROOT, mid))
                    model.set_classes([PROMPT])
                    p = nparams(model.model)
                    run = lambda: model.predict(img, conf=1e-5, imgsz=args.size,
                                                device=dev, verbose=False)

            rec = {'params_M': round(p / 1e6, 1)}
            if dev == 'cuda' and run is not None:
                med, lo, hi = timeit(run, args.iters, dev)
                rec['latency_ms_median'] = round(med, 1)
                rec['latency_ms_p5_p95'] = [round(lo, 1), round(hi, 1)]
                rec['peak_vram_GB'] = round(
                    (torch.cuda.max_memory_allocated() - base) / 1024 ** 3, 2)
                rec['peak_vram_reserved_GB'] = round(
                    torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            out['models'][key] = rec
            print(f'{key:14s} {rec}', flush=True)

            del model
            if dev == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:                       # one model failing must not lose the rest
            out['models'][key] = {'error': f'{type(e).__name__}: {e}'}
            print(f'{key:14s} FAILED {type(e).__name__}: {e}', flush=True)

    # end-to-end throughput actually observed in the study, for context
    obs = {}
    for d in sorted(os.listdir(f'{WD}/preds')):
        f = f'{WD}/preds/{d}/_meta.json'
        if not os.path.exists(f):
            continue
        m = json.load(open(f))
        npr = sum(len(v[1]) for v in m.get('prompts', {}).values()) or 1
        if m.get('seconds') and m.get('n_images'):
            obs[d] = {'n_images': m['n_images'], 'n_prompts': npr,
                      'seconds': m['seconds'],
                      's_per_image_per_prompt': round(
                          m['seconds'] / m['n_images'] / npr, 3)}
    out['observed_study_protocol'] = obs

    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(f'{RES}/model_costs.json', 'w'), indent=1)
    print(f'\nwrote {RES}/model_costs.json')


if __name__ == '__main__':
    main()
