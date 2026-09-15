#!/usr/bin/env python3
"""Collect the reproducibility record: checkpoint revisions, environment versions,
seeds, wall-clock inference cost, and the hashes of the annotator instruments.

Writes ov/results/repro_record.json, which 24_fill_appendices.py renders into
Appendix B and Appendix C of the manuscript.

The supervised arm lives in a second virtualenv (repro/envs/mmdet), so its package
versions are read by invoking that interpreter rather than importing here.
"""
import hashlib, json, os, re, subprocess, sys, zipfile

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
RES = f'{ROOT}/ov/results'
WD = f'{ROOT}/ov/work_dirs'
HUB = os.path.expanduser('~/.cache/huggingface/hub')
MMDET_PY = f'{ROOT}/repro/envs/mmdet/bin/python'

# key -> (hugging-face repo id or None, local weight file or None)
CONFIGS = [
    ('owlv2-base',  'google/owlv2-base-patch16-ensemble', None),
    ('owlv2-large', 'google/owlv2-large-patch14-ensemble', None),
    ('gdino-tiny',  'IDEA-Research/grounding-dino-tiny', None),
    ('gdino-base',  'IDEA-Research/grounding-dino-base', None),
    ('yoloworld-s', None, f'{ROOT}/yolov8s-worldv2.pt'),
    ('sam3',        'facebook/sam3', None),
    ('sam3agent (stage 1)', 'Qwen/Qwen2.5-VL-3B-Instruct', None),
    ('RTMDet (supervised reference)', None,
     f'{ROOT}/repro/work_dirs/convnext_ciou_clahe/best_coco_bbox_mAP_50_epoch_113.pth'),
]

PANEL_PKGS = f'{ROOT}/panel_app/dist_windows'
ANNOTATORS = ['binh', 'chi', 'hai']


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def hf_revision(repo):
    d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'refs', 'main')
    return open(d).read().strip() if os.path.exists(d) else None


def ov_env():
    import torch
    import importlib.metadata as md
    out = {'python': sys.version.split()[0],
           'torch': torch.__version__,
           'cuda': torch.version.cuda,
           'cudnn': str(torch.backends.cudnn.version())}
    for p in ['transformers', 'ultralytics', 'pycocotools', 'numpy', 'pillow',
              'accelerate', 'tokenizers', 'safetensors', 'scipy']:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    # SAM 3 has no standalone distribution: it ships inside transformers.
    import importlib.util
    out['_sam3_note'] = (
        'transformers.models.sam3'
        if importlib.util.find_spec('transformers.models.sam3') else 'not found')
    return out


def mmdet_env():
    code = ('import importlib.metadata as md, sys, json, torch;'
            "d={'python': sys.version.split()[0], 'torch': torch.__version__};"
            "[d.__setitem__(p, md.version(p)) for p in "
            "['mmdet','mmcv','mmengine','mmpretrain','numpy','pycocotools']];"
            'print(json.dumps(d))')
    try:
        return json.loads(subprocess.run([MMDET_PY, '-c', code],
                                         capture_output=True, text=True,
                                         check=True).stdout)
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def seeds():
    """Every seeded call site in the analysis, read from the scripts themselves."""
    hits = {}
    for fn in sorted(os.listdir(f'{ROOT}/ov/scripts')):
        if not fn.endswith('.py'):
            continue
        txt = open(f'{ROOT}/ov/scripts/{fn}').read()
        vals = set(re.findall(r'default_rng\((?:seed=)?(\w+)\)', txt))
        vals |= set(re.findall(r'seed\s*=\s*(\d+)', txt))
        vals.discard('seed')
        if vals:
            hits[fn] = sorted(vals)
    return hits


def timings():
    """Wall-clock per model over every set it was run on, from the run metadata."""
    rows = {}
    for d in sorted(os.listdir(f'{WD}/preds')):
        p = f'{WD}/preds/{d}/_meta.json'
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        model, setname = d.rsplit('_', 1)
        # 01_infer.py records `prompts` as {group: [cat, [prompt, ...]]};
        # 10_yoloworld.py records one entry per prompt in `max_confidence_per_prompt`.
        nprompt = (sum(len(v[1]) for v in m.get('prompts', {}).values())
                   or len(m.get('max_confidence_per_prompt', {})) or 1)
        r = rows.setdefault(model, {'passes': 0, 'seconds': 0.0, 'sets': {}})
        passes = m['n_images'] * nprompt
        r['passes'] += passes
        r['seconds'] += m['seconds']
        r['sets'][setname] = {'n_images': m['n_images'], 'n_prompts': nprompt,
                              'seconds': round(m['seconds'], 1)}
    # SAM 3 Agent stage 1 is a separate script, timed in its own log
    log = f'{WD}/logs/agent_stage1.log'
    if os.path.exists(log):
        secs = [int(m) for m in
                re.findall(r'\b(\d+)s eta 0s', open(log, errors='ignore').read())]
        if secs and 'sam3agent' in rows:
            rows['sam3agent']['stage1_seconds'] = sum(secs)
            rows['sam3agent']['seconds'] += sum(secs)
    for r in rows.values():
        r['seconds'] = round(r['seconds'], 1)
        r['s_per_pass'] = round(r['seconds'] / r['passes'], 3) if r['passes'] else None
    return rows


def annotator_instruments():
    """Hashes of the two instruction sets, as actually shipped to each annotator."""
    out = {'files': {}, 'packages': {}}
    for name, p in [('round1', f'{ROOT}/panel_app/sop/round1.txt'),
                    ('round2', f'{ROOT}/panel_app/sop/round2.txt')]:
        if os.path.exists(p):
            txt = open(p, encoding='utf8').read()
            out['files'][name] = {'sha256': sha256_file(p),
                                  'bytes': os.path.getsize(p),
                                  'lines': txt.count('\n') + (0 if txt.endswith('\n') else 1)}
    for a in ANNOTATORS:
        for rnd in (1, 2):
            z = f'{PANEL_PKGS}/GanNhanPanel_{a}_round{rnd}.zip'
            if not os.path.exists(z):
                continue
            with zipfile.ZipFile(z) as zf:
                base = f'GanNhanPanel_{a}_round{rnd}'
                inner = f'{base}/sop/round{rnd}.txt'
                if inner not in zf.namelist():
                    continue
                info = zf.getinfo(inner)
                out['packages'][f'{a}/round{rnd}'] = {
                    'sop_sha256': hashlib.sha256(zf.read(inner)).hexdigest(),
                    'sop_mtime': '%04d-%02d-%02d %02d:%02d' % info.date_time[:5],
                    'guide_sha256': hashlib.sha256(
                        zf.read(f'{base}/HUONG_DAN.txt')).hexdigest()
                    if f'{base}/HUONG_DAN.txt' in zf.namelist() else None}
    # earliest annotator submission, to establish that the SOP predates the data
    subs = []
    for d, _, fs in os.walk(os.path.expanduser('~/Downloads')):
        for f in fs:
            if re.fullmatch(r'via_project_\w+\.json', f):
                subs.append((os.path.getmtime(os.path.join(d, f)), os.path.join(d, f)))
    if subs:
        import datetime
        t, p = min(subs)
        out['earliest_submission'] = {
            'file': os.path.basename(p),
            'mtime': datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')}
    return out


def main():
    rec = {
        'checkpoints': [
            {'key': k,
             'repo': repo,
             'revision': hf_revision(repo) if repo else None,
             'weight_file': os.path.basename(f) if f else None,
             'weight_sha256': sha256_file(f) if f and os.path.exists(f) else None}
            for k, repo, f in CONFIGS],
        'env_openvocab': ov_env(),
        'env_supervised': mmdet_env(),
        'gpu': json.load(open(f'{RES}/model_costs.json')).get(
            'gpu', 'NVIDIA GeForce RTX 3080 Ti (12 GB)'),
        'model_costs': json.load(open(f'{RES}/model_costs.json')),
        'seeds': seeds(),
        'timings': timings(),
        'annotator_instruments': annotator_instruments(),
    }
    json.dump(rec, open(f'{RES}/repro_record.json', 'w'), indent=1)
    print(json.dumps(rec, indent=1)[:3000])
    print(f'\nwrote {RES}/repro_record.json')


if __name__ == '__main__':
    main()
