#!/usr/bin/env python3
"""Gate test: does OWLv2 load, run on one real image, and give sane boxes?"""
import json, os, sys, torch
from PIL import Image

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
IMGDIR = f'{ROOT}/repro/mmdet_mit/data/images'
ev = json.load(open(f'{ROOT}/ov/work_dirs/eval2c_test.json'))
im = ev['images'][0]
print('image:', im['file_name'], im['width'], 'x', im['height'])

from transformers import Owlv2Processor, Owlv2ForObjectDetection
mid = 'google/owlv2-base-patch16-ensemble'
proc = Owlv2Processor.from_pretrained(mid)
model = Owlv2ForObjectDetection.from_pretrained(mid).eval().cuda()
print('model loaded, VRAM %.2f GB' % (torch.cuda.memory_allocated() / 1e9))

img = Image.open(os.path.join(IMGDIR, im['file_name'])).convert('RGB')
texts = [['weed', 'sugarcane']]
inputs = proc(text=texts, images=img, return_tensors='pt').to('cuda')
with torch.no_grad():
    out = model(**inputs)
res = proc.post_process_grounded_object_detection(
    out, threshold=0.05, target_sizes=torch.tensor([[img.height, img.width]]))[0]
print('n detections >0.05:', len(res['scores']))
for s, l, b in list(zip(res['scores'].tolist(), res['labels'].tolist(), res['boxes'].tolist()))[:5]:
    print(f'  {texts[0][l]:10s} {s:.3f} [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]')
gt = [a for a in ev['annotations'] if a['image_id'] == im['id']]
print('GT boxes:', [(a['category_id'], [round(v) for v in a['bbox']]) for a in gt])
