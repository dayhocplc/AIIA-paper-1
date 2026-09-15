#!/usr/bin/env python3
"""SAM 3 Agent, stage 1: an MLLM proposes simple noun phrases per image.

SAM 3's authors restrict its task to simple noun phrases and offer "SAM 3 Agent"
-- an MLLM wrapper that decomposes complex language into such phrases -- for
anything more complex. This reproduces the idea in its single-pass,
image-conditioned form: the MLLM sees the image and the relational goal, and
names the specific plants that satisfy it. Stage 2 then prompts SAM 3 with those
phrases.

Deviation from Meta's version, stated for the record: theirs iterates (propose ->
segment -> inspect masks -> refine); this is one pass. Image conditioning is the
part that matters for the comparison, because a decomposition that ignores the
image would be a fixed prompt rewrite and is already covered by prompt group B.

Run before stage 2: the MLLM is unloaded before SAM 3 is loaded, so the two
never contend for VRAM.
"""
import json, os, re, time
import torch
from PIL import Image

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
IMGDIR = f'{ROOT}/repro/mmdet_mit/data/images'
WD = f'{ROOT}/ov/work_dirs'
MID = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAXSIDE = 1024
OUT = f'{WD}/sam3_agent_nps.json'

# Prompt chosen after a documented diagnostic (see EXPERIMENT_LOG.md, F7):
#   * a first version showed three plant names as a format example; the model
#     reproduced that exact list for ~95 % of images -> not image-conditioned.
#   * a second version added "if none is visible, answer exactly: none"; the model
#     then answered "none" for 20/20 images that plainly contain weeds.
# This version states the relational definition, gives no in-domain example to
# copy, and offers no escape hatch. Its species names are frequently implausible
# for nadir photographs of grass -- that unreliability is part of what this arm
# measures, so it is reported, not corrected.
GOAL = ("A weed is any plant that is not the sugarcane crop. "
        "Name the weed plants visible in THIS photo. "
        "Short noun phrases, commas, nothing else.")

STOP = {'weed', 'weeds', 'plant', 'plants', 'none', 'no weeds', 'nothing',
        'sugarcane', 'grass field', 'n/a'}


def clean(text):
    text = text.strip().split('\n')[0]
    text = re.sub(r'^[\s\-\*\d\.\)]+', '', text)
    out = []
    for p in re.split(r'[,;]', text):
        p = p.strip().strip('."\'').lower()
        p = re.sub(r'\s+', ' ', p)
        if 2 <= len(p) <= 40 and p not in STOP and not p.startswith('i '):
            out.append(p)
    return out[:3]


def report_conditioning(out):
    """Guard against the failure that killed the first attempt: if the MLLM
    returns the same phrase set for most images, the decomposition is not
    image-conditioned and the arm is meaningless."""
    from collections import Counter
    sets = Counter(tuple(sorted(v['nps'])) for v in out.values())
    n = len(out)
    top, topn = sets.most_common(1)[0]
    print(f'\n--- kiem tra dieu kien theo anh ---')
    print(f'  {len(sets)} bo cum danh tu khac nhau tren {n} anh')
    print(f'  bo pho bien nhat chiem {topn}/{n} = {topn/n*100:.1f}%  -> {list(top)}')
    if topn / n > 0.5:
        print('  ⚠️  CANH BAO: >50% anh cung mot bo -> nhieu kha nang KHONG nhin anh')
    else:
        print('  OK: dau ra thay doi theo anh')
    return {'n_distinct_sets': len(sets), 'top_set_share': topn / n,
            'top_set': list(top)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sets', default='all,panel')
    args = ap.parse_args()
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(MID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map='cuda').eval()
    print(f'MLLM loaded, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    out = {}
    for setname in args.sets.split(','):
        ev = json.load(open(f'{WD}/eval2c_{setname}.json'))
        items = ev['images'][:args.limit] if args.limit else ev['images']
        t0 = time.time()
        for n, im in enumerate(items):
            p = os.path.join(IMGDIR, im['file_name'])
            img = Image.open(p)
            img.draft('RGB', (img.width // 4, img.height // 4))
            img = img.convert('RGB')
            img.thumbnail((MAXSIDE, MAXSIDE))
            msgs = [{'role': 'user', 'content': [{'type': 'image'},
                                                 {'type': 'text', 'text': GOAL}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[img], return_tensors='pt').to('cuda')
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
            ans = proc.batch_decode(gen[:, inputs['input_ids'].shape[1]:],
                                    skip_special_tokens=True)[0]
            nps = clean(ans)
            out[f'{setname}|{im["id"]}'] = {'file_name': im['file_name'],
                                            'raw': ans.strip()[:200], 'nps': nps}
            if (n + 1) % 25 == 0 or n + 1 == len(items):
                el = time.time() - t0
                print(f'  {setname} {n+1}/{len(items)}  {el:.0f}s '
                      f'eta {el/(n+1)*(len(items)-n-1):.0f}s', flush=True)

    diag = report_conditioning(out)
    out['_diagnostics'] = diag
    json.dump(out, open(OUT, 'w'), indent=1)
    empty = sum(1 for k, v in out.items() if k != '_diagnostics' and not v['nps'])
    from collections import Counter
    c = Counter(p for k, v in out.items() if k != '_diagnostics' for p in v['nps'])
    print(f'\nwrote {OUT}  ({len(out)} anh, {empty} anh khong ra NP nao)')
    print('20 cum danh tu pho bien nhat:')
    for k, v in c.most_common(20):
        print(f'   {v:4d}  {k}')


if __name__ == '__main__':
    main()
