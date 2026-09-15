#!/usr/bin/env python3
"""Build evaluation sets for the open-vocabulary experiments.

Outputs (all under ov/work_dirs/):
  eval2c_all.json   COCO, 222 images, 2 classes (weed, sugarcane), merged
                    train+val+test of COCO_2class with globally unique image ids.
                    This is the MAIN set for OV-vs-OV comparisons: no model is
                    trained, so there is no reason to restrict to the test split.
  eval2c_test.json  the 54-image Clean test split only, same id space.
                    Used ONLY for comparisons against the supervised RTMDet
                    baseline, which did see the train split.
  negatives.json    list of expert-confirmed weed-free images (sugarcane/ folder).

Each image record carries `split` and `has_cane` for stratified reporting.
"""
import json, os, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D2C = os.path.join(ROOT, 'repro/mmdet_mit/data/COCO_2class')
IMGDIR = os.path.join(ROOT, 'repro/mmdet_mit/data/images')
OUT = os.path.join(ROOT, 'ov/work_dirs')
os.makedirs(OUT, exist_ok=True)

CATS = [{'id': 0, 'name': 'weed'}, {'id': 1, 'name': 'sugarcane'}]

images, anns = [], []
next_img, next_ann = 0, 0
test_ids = set()

for split in ['train', 'val', 'test']:
    j = json.load(open(f'{D2C}/weed_dataset_detection_{split}_coco_format.json'))
    remap = {}
    for im in j['images']:
        rec = {'id': next_img, 'file_name': im['file_name'],
               'width': im['width'], 'height': im['height'],
               'split': split, 'orig_id': im['id']}
        remap[im['id']] = next_img
        if split == 'test':
            test_ids.add(next_img)
        images.append(rec)
        next_img += 1
    for a in j['annotations']:
        anns.append({'id': next_ann, 'image_id': remap[a['image_id']],
                     'category_id': a['category_id'], 'bbox': a['bbox'],
                     'area': a['bbox'][2] * a['bbox'][3], 'iscrowd': 0})
        next_ann += 1

# stratification flag: does this image contain any annotated sugarcane?
cane = {a['image_id'] for a in anns if a['category_id'] == 1}
for im in images:
    im['has_cane'] = im['id'] in cane

# sanity: every file must exist
missing = [im['file_name'] for im in images if not os.path.exists(os.path.join(IMGDIR, im['file_name']))]
assert not missing, f'missing images: {missing[:5]}'

full = {'images': images, 'annotations': anns, 'categories': CATS}
json.dump(full, open(f'{OUT}/eval2c_all.json', 'w'))

sub_im = [im for im in images if im['id'] in test_ids]
sub_an = [a for a in anns if a['image_id'] in test_ids]
json.dump({'images': sub_im, 'annotations': sub_an, 'categories': CATS},
          open(f'{OUT}/eval2c_test.json', 'w'))

# expert-confirmed weed-free images
negs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(IMGDIR, 'sugarcane', '*.jpg')))
json.dump({'dir': 'sugarcane', 'files': negs}, open(f'{OUT}/negatives.json', 'w'))


def summarize(ims, ans, name):
    nw = sum(1 for a in ans if a['category_id'] == 0)
    ns = sum(1 for a in ans if a['category_id'] == 1)
    hc = sum(1 for i in ims if i['has_cane'])
    print(f'{name:16s} {len(ims):4d} images | weed {nw:5d} | sugarcane {ns:4d} | '
          f'with cane {hc:4d} | without cane {len(ims)-hc:3d}')


print('=== evaluation sets ===')
summarize(images, anns, 'eval2c_all')
summarize(sub_im, sub_an, 'eval2c_test')
print(f'{"negatives":16s} {len(negs):4d} images | expert-confirmed weed-free')
print(f'\nminimum detectable AP50 difference (paired, from sd_diff=0.0295):')
for n, nm in [(len(images), 'eval2c_all'), (len(sub_im), 'eval2c_test')]:
    print(f'  {nm:16s} n={n:4d} -> d = {(3692/n)**0.5:.2f} points')
