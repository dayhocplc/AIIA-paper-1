#!/usr/bin/env python3
"""Quantify the annotation-protocol asymmetry between the two classes.

This is the measurement that justifies using a granularity-invariant metric for
the nominal-vs-relational comparison.  `weed` was annotated at instance level,
`sugarcane` at region level; comparing instance AP across the two would compare
annotation protocols rather than category types.
"""
import json, os
import numpy as np

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
gt = json.load(open(f'{ROOT}/ov/work_dirs/eval2c_all.json'))
W = {i['id']: (i['width'], i['height']) for i in gt['images']}
NAMES = {0: 'weed', 1: 'sugarcane'}

by = {0: {}, 1: {}}
for a in gt['annotations']:
    by[a['category_id']].setdefault(a['image_id'], []).append(a)

rows = {}
for c, name in NAMES.items():
    counts, areas, cover = [], [], []
    for i, anns in by[c].items():
        w, h = W[i]
        counts.append(len(anns))
        for a in anns:
            areas.append(a['bbox'][2] * a['bbox'][3] / (w * h) * 100)
        cover.append(sum(a['bbox'][2] * a['bbox'][3] for a in anns) / (w * h) * 100)
    rows[name] = {
        'n_boxes': sum(counts), 'n_images': len(counts),
        'boxes_per_image_median': float(np.median(counts)),
        'boxes_per_image_max': int(np.max(counts)),
        'box_area_pct_median': float(np.median(areas)),
        'box_area_pct_p10': float(np.percentile(areas, 10)),
        'box_area_pct_p90': float(np.percentile(areas, 90)),
        'frame_coverage_pct_median': float(np.median(cover)),
        'frame_coverage_pct_max': float(np.max(cover)),
    }

# co-occurrence: how many images support the paired same-pixel control
imgs_w = set(by[0]); imgs_s = set(by[1])
paired = imgs_w & imgs_s
summary = {
    'per_class': rows,
    'n_images_total': len(gt['images']),
    'n_images_with_weed': len(imgs_w),
    'n_images_with_sugarcane': len(imgs_s),
    'n_images_with_both': len(paired),
    'granularity_ratio_area': rows['sugarcane']['box_area_pct_median'] /
                              rows['weed']['box_area_pct_median'],
}

print('=== annotation protocol asymmetry (eval2c_all, 222 images) ===')
hdr = f'{"":28s} {"weed":>12s} {"sugarcane":>12s}'
print(hdr); print('-' * len(hdr))
for k in ['n_boxes', 'n_images', 'boxes_per_image_median', 'box_area_pct_median',
          'box_area_pct_p90', 'frame_coverage_pct_median', 'frame_coverage_pct_max']:
    print(f'{k:28s} {rows["weed"][k]:12.1f} {rows["sugarcane"][k]:12.1f}')
print(f'\nmedian box area ratio (cane/weed): {summary["granularity_ratio_area"]:.1f}x')
print(f'images supporting the paired control: {len(paired)}/{len(gt["images"])}')

os.makedirs(f'{ROOT}/ov/results', exist_ok=True)
json.dump(summary, open(f'{ROOT}/ov/results/annotation_protocol.json', 'w'), indent=1)
print(f'\nwrote ov/results/annotation_protocol.json')
