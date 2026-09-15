#!/usr/bin/env python3
"""Export the four VIA panel projects to COCO format.

Written into ov/panel/ only; the frozen panel/ and repro/ trees are untouched.
One file per annotator plus a manifest recording provenance, because the two
`output/` and `output-1/` projects both carry the VIA project name `panel_<annotator>`
(a fourth annotator reused an existing project template) and must not be
conflated.
"""
import json, os, hashlib

SRC = {'ann1': ('<PANEL_SUBMISSIONS_DIR>/output/via_project_<annotator>.json', 'output'),
       'ann2': ('<PANEL_SUBMISSIONS_DIR>/output-1/output/via_project_<annotator>.json', 'output-1'),
       'ann3': ('<PANEL_SUBMISSIONS_DIR>/output-2/output/via_project_<annotator>.json', 'output-2'),
       'ann4': ('<PANEL_SUBMISSIONS_DIR>/output-3/output/via_project_<annotator>.json', 'output-3')}
PANEL = 'repro/mmdet_mit/data/COCO_Panel/panel_coco.json'
OUTDIR = 'ov/panel'
os.makedirs(OUTDIR, exist_ok=True)

base = json.load(open(PANEL))
by_name = {im['file_name'].split('/')[-1]: im for im in base['images']}
manifest = {'source_panel': PANEL, 'n_images': len(base['images']), 'annotators': {}}

for key, (path, folder) in SRC.items():
    j = json.load(open(path))
    vname = j['_via_settings']['project'].get('name')
    imgs, anns = [], []
    aid = 0
    undecidable = []
    for v in j['_via_img_metadata'].values():
        fn = v['filename']
        src = by_name.get(fn)
        if src is None:
            continue
        imgs.append({'id': src['id'], 'file_name': src['file_name'],
                     'width': src['width'], 'height': src['height']})
        if v['file_attributes'].get('khong_phan_dinh_duoc'):
            undecidable.append(fn)
        for r in v['regions']:
            sa = r['shape_attributes']
            if sa.get('name') != 'rect':
                continue
            w, h = sa['width'], sa['height']
            anns.append({'id': aid, 'image_id': src['id'], 'category_id': 0,
                         'bbox': [sa['x'], sa['y'], w, h], 'area': w * h, 'iscrowd': 0})
            aid += 1
    coco = {'images': sorted(imgs, key=lambda x: x['id']), 'annotations': anns,
            'categories': [{'id': 0, 'name': 'weed'}]}
    out = f'{OUTDIR}/panel_{key}_coco.json'
    json.dump(coco, open(out, 'w'))
    manifest['annotators'][key] = {
        'source_file': path, 'source_folder': folder,
        'via_project_name': vname,
        'sha256_of_source': hashlib.sha256(open(path, 'rb').read()).hexdigest(),
        'n_images': len(imgs), 'n_boxes': len(anns),
        'n_undecidable': len(undecidable), 'undecidable_files': sorted(undecidable),
        'coco_out': out}
    print(f'{key}  via_name={vname:12s} folder={folder:9s} '
          f'{len(imgs)} anh, {len(anns):4d} hop, {len(undecidable)} anh khong phan dinh duoc')

manifest['note'] = ('Both `output` and `output-1` carry the VIA project name '
                    '`panel_<annotator>`: a fourth annotator was added and reused an '
                    'existing project template. They are different people - '
                    'their mutual agreement is F1=0.232, the same level as any '
                    'other pair. Annotator identity for `output` is unconfirmed.')
json.dump(manifest, open(f'{OUTDIR}/manifest.json', 'w'), indent=1)
print(f'\nwrote {OUTDIR}/  ({len(SRC)} file COCO + manifest.json)')
