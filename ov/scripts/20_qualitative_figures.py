#!/usr/bin/env python3
"""Qualitative figures: green-on-green + annotation granularity, the six-detector
comparison panel, and false alarms on the confirmed-negative set.

Outputs (ov/figures/):
  figQ1_green_on_green.png   Instrument 1, seen rather than tabulated
  figQ2_qualitative.png      six detectors, best configuration each, same images
  figQ3_false_alarms.png     four detectors on expert-certified weed-free images

Every box drawn here comes from the same JSON the tables are computed from; no
prediction is re-run and no threshold is chosen for the picture.  Detector
configurations are each detector's own best-F1 prompt, at its own best-F1
threshold, both fixed by 02_eval.py on the annotated positive set.
"""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
IMGDIR = f'{ROOT}/repro/mmdet_mit/data/images'
WD = f'{ROOT}/ov/work_dirs'
RES = f'{ROOT}/ov/results'
FIG = f'{ROOT}/ov/figures'
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 8, 'figure.dpi': 300, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})

# Okabe-Ito.  Never a red/green pair on the same panel.
C_WEED = '#0072B2'   # blue    - weed ground truth, and matched predictions
C_CANE = '#E69F00'   # orange  - sugarcane ground truth
C_FP = '#D55E00'     # vermillion - unmatched prediction (false positive)
C_MISS = '#999999'   # grey    - ground-truth box the detector missed

DETECTORS = ['gdino-tiny', 'gdino-base', 'owlv2-base', 'owlv2-large',
             'sam3', 'yoloworld-s']


# ------------------------------------------------------------------ utilities
def load_disp(file_name, long_side=1200):
    """Decode at reduced scale; boxes stay in original-pixel coordinates."""
    im = Image.open(os.path.join(IMGDIR, file_name))
    im.draft('RGB', (long_side, long_side))
    return im.convert('RGB')


def iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def greedy_match(preds, gts, thr=0.5):
    """COCO-style greedy score-ordered matching.  Returns (tp_flags, matched_gt)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]['score'])
    used, tp = set(), [False] * len(preds)
    for i in order:
        best, bi = thr, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(preds[i]['bbox'], g)
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            used.add(bi); tp[i] = True
    return tp, used


def draw_box(ax, bbox, color, lw=1.4, ls='-', alpha=1.0):
    x, y, w, h = bbox
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                           linewidth=lw, linestyle=ls, alpha=alpha))


def show(ax, img, W, H, title=None, title_size=7.5, title_color='k'):
    ax.imshow(img, extent=(0, W, H, 0))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color('0.3')
    if title:
        ax.set_title(title, fontsize=title_size, pad=2.5, color=title_color)


def panel_tag(ax, tag):
    ax.text(0.012, 0.978, tag, transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5, fontweight='bold', color='w',
            bbox=dict(boxstyle='square,pad=0.22', fc='k', ec='none', alpha=0.62))


def corner_note(ax, text, loc='lower left'):
    x, ha = (0.012, 'left') if 'left' in loc else (0.988, 'right')
    y, va = (0.02, 'bottom') if 'lower' in loc else (0.98, 'top')
    ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=6.4,
            color='w', linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.28', fc='k', ec='none', alpha=0.58))


def load_gt():
    ev = json.load(open(f'{WD}/eval2c_all.json'))
    ims = {im['id']: im for im in ev['images']}
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for a in ev['annotations']:
        by[a['image_id']][a['category_id']].append(a['bbox'])
    return ims, by


def best_config():
    """Each detector's best weed prompt and its best-F1 threshold, from eval_main."""
    ev = json.load(open(f'{RES}/eval_main.json'))['runs']
    best = {}
    for k, v in ev.items():
        m, grp, prompt = k.split('|', 2)
        if v.get('category') != 0:       # weed-targeting runs only
            continue
        if m not in best or v['bestF1@0.5']['best'] > best[m][3]:
            best[m] = (grp, prompt, v['bestF1@0.5']['thr'], v['bestF1@0.5']['best'])
    return best


def read_preds(model, subset, grp, prompt):
    fn = f"{WD}/preds/{model}_{subset}/{grp}__{prompt.replace(' ', '_')}.json"
    p = json.load(open(fn))
    by = collections.defaultdict(list)
    for d in p:
        by[d['image_id']].append(d)
    return by


# ------------------------------------------------- Fig Q1: green-on-green + I1
def _window(bbox, W, H, pad=1.35, aspect=2 / 3):
    """A frame-aspect window centred on bbox, at least pad x its longer side."""
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    w = max(bbox[2], bbox[3] / aspect) * pad
    w = min(w, W, H / aspect)
    h = w * aspect
    x = min(max(0.0, cx - w / 2), W - w)
    y = min(max(0.0, cy - h / 2), H - h)
    return [x, y, w, h]


def figQ1(image_id=146):
    """One mixed image, its two annotation protocols on the same pixels, and the
    median box geometry of the whole dataset drawn to scale."""
    ims, by = load_gt()
    im = ims[image_id]
    W, H = im['width'], im['height']
    asp = H / W
    img = load_disp(im['file_name'], 1600)
    weeds = sorted(by[image_id][0], key=lambda b: -b[2] * b[3])
    canes = sorted(by[image_id][1], key=lambda b: -b[2] * b[3])
    A = W * H
    w_med = 100 * np.median([b[2] * b[3] / A for b in weeds])
    c_cov = 100 * min(1.0, sum(b[2] * b[3] for b in canes) / A)

    fig = plt.figure(figsize=(10.2, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.05)

    # (a) the scene
    ax = fig.add_subplot(gs[0, 0])
    show(ax, img, W, H)
    panel_tag(ax, 'a')

    # (b) weed: instance-level protocol
    ax = fig.add_subplot(gs[0, 1])
    show(ax, img, W, H, f'`weed` · {len(weeds)} instance boxes')
    for b in weeds:
        draw_box(ax, b, C_WEED, lw=1.6)
    corner_note(ax, f'{len(weeds)} boxes · median {w_med:.1f} % of frame')
    panel_tag(ax, 'b')

    # (c) sugarcane: region-level protocol, identical pixels
    ax = fig.add_subplot(gs[0, 2])
    show(ax, img, W, H, f'`sugarcane` · {len(canes)} region boxes')
    for b in canes:
        draw_box(ax, b, C_CANE, lw=2.0)
    corner_note(ax, f'{len(canes)} boxes · {c_cov:.0f} % of the frame covered')
    panel_tag(ax, 'c')

    # (d) a weed crop, and (e) a cane-only crop at the identical magnification.
    # The weed box shown is the median-area one, i.e. a typical instance.
    wb = sorted(weeds, key=lambda b: b[2] * b[3])[len(weeds) // 2]
    win_w = _window(wb, W, H, pad=1.9, aspect=asp)
    ww, wh = win_w[2], win_w[3]

    cb = canes[0]
    cand, best = None, (-1.0, -1.0)
    for fx in np.linspace(0.05, 0.95, 19):
        for fy in np.linspace(0.05, 0.95, 19):
            cx, cy = cb[0] + fx * cb[2], cb[1] + fy * cb[3]
            x = min(max(0.0, cx - ww / 2), W - ww)
            y = min(max(0.0, cy - wh / 2), H - wh)
            win = [x, y, ww, wh]
            ov = max([iou(win, b) for b in weeds] or [0.0])
            far = min(abs(cx - (wb[0] + wb[2] / 2)), abs(cy - (wb[1] + wb[3] / 2)))
            if (-ov, far) > best:
                best, cand = (-ov, far), win
    cand = cand or [0.0, 0.0, ww, wh]

    for gi, (win, col, ttl, tag, box) in enumerate([
            (win_w, C_WEED, '`weed`, median-sized box', 'd', wb),
            (cand, C_CANE, '`sugarcane`, same magnification', 'e', None)]):
        ax = fig.add_subplot(gs[1, gi])
        show(ax, img, W, H, ttl, title_color=col)
        ax.set_xlim(win[0], win[0] + win[2])
        ax.set_ylim(win[1] + win[3], win[1])
        if box is not None:
            draw_box(ax, box, col, lw=2.0)
        panel_tag(ax, tag)

    # (f) dataset-level box geometry, to scale
    ax = fig.add_subplot(gs[1, 2])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_xlim(0, 1); ax.set_ylim(0, asp)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.add_patch(Rectangle((0.002, 0.002), 0.996, asp - 0.004, fill=False,
                           ec='0.35', lw=1.0))
    s_c = np.sqrt(0.307)                      # linear scale of a 30.7 % box
    ax.add_patch(Rectangle((0.012, 0.012), s_c, s_c * asp, fc=C_CANE,
                           alpha=0.20, ec=C_CANE, lw=1.8))
    s_w = np.sqrt(0.040)                      # linear scale of a 4.0 % box
    ax.add_patch(Rectangle((0.012, 0.012), s_w, s_w * asp, fc=C_WEED,
                           alpha=0.32, ec=C_WEED, lw=1.8))
    ax.text(0.5, asp - 0.022, 'one frame, 4608 × 3072', ha='center', va='top',
            fontsize=6.8, color='0.4')
    ax.annotate('median `sugarcane` box\n30.7 % of frame',
                xy=(0.012 + s_c * 0.82, 0.012 + s_c * asp * 0.88),
                xytext=(0.615, 0.30), fontsize=7, color=C_CANE, va='center',
                arrowprops=dict(arrowstyle='-', color=C_CANE, lw=0.7))
    ax.annotate('median `weed` box\n4.0 % of frame',
                xy=(0.012 + s_w * 0.9, 0.012 + s_w * asp * 0.55),
                xytext=(0.30, 0.055), fontsize=7, color=C_WEED, va='center',
                arrowprops=dict(arrowstyle='-', color=C_WEED, lw=0.7))
    ax.text(0.5, -0.055,
            'Whole dataset:  998 `weed` boxes vs 384 `sugarcane`\n'
            'boxes per image (median)  4.0  vs  2.0\n'
            'frame coverage (median)  33.6 %  vs  80.3 %\n'
            r'$\bf{median\ box\ area\ ratio\ =\ 7.8\times}$',
            transform=ax.transAxes, ha='center', va='top', fontsize=7.2,
            linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='#f5f5f5', ec='0.75', lw=0.6))
    panel_tag(ax, 'f')

    fig.savefig(f'{FIG}/figQ1_green_on_green.png')
    fig.savefig(f'{FIG}/figQ1_green_on_green.pdf')
    plt.close(fig)
    print(f'figQ1  {im["file_name"]}  weeds={len(weeds)} canes={len(canes)}')


# ------------------------------------------- Fig Q2: six detectors, same images
def figQ2(image_ids=(168, 178, 183), max_draw=25):
    """Rows: three test-split images.  Columns: ground truth, then each detector
    in its own best configuration (best-F1 prompt at its best-F1 threshold)."""
    ims, by = load_gt()
    best = best_config()
    nrow, ncol = len(image_ids), 1 + len(DETECTORS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.62 * ncol, 1.16 * nrow))

    for r, iid in enumerate(image_ids):
        im = ims[iid]
        W, H = im['width'], im['height']
        img = load_disp(im['file_name'], 900)
        gts = by[iid][0]

        ax = axes[r, 0]
        show(ax, img, W, H)
        for g in gts:
            draw_box(ax, g, C_WEED, lw=1.4)
        corner_note(ax, f'{len(gts)} annotated')
        ax.set_ylabel(f'{os.path.basename(im["file_name"])}\n({im["split"]} split)',
                      fontsize=6.6)
        if r == 0:
            ax.set_title('ground truth', fontsize=7.4, fontweight='bold',
                         color=C_WEED, pad=13)

        for c, m in enumerate(DETECTORS, start=1):
            grp, prompt, thr, f1 = best[m]
            preds = read_preds(m, 'all', grp, prompt).get(iid, [])
            keep = sorted([p for p in preds if p['score'] >= thr],
                          key=lambda p: -p['score'])
            tp, matched = greedy_match(keep, gts)
            ax = axes[r, c]
            show(ax, img, W, H)
            for j, g in enumerate(gts):
                if j not in matched:
                    draw_box(ax, g, C_MISS, lw=0.8, ls=(0, (3, 2)), alpha=0.9)
            for p, is_tp in zip(keep[:max_draw], tp[:max_draw]):
                draw_box(ax, p['bbox'], C_WEED if is_tp else C_FP,
                         lw=1.4 if is_tp else 0.9, alpha=1.0 if is_tp else 0.9)
            star = '*' if len(keep) > max_draw else ''
            corner_note(ax, f'{len(keep)}{star} det · {sum(tp)} hit')
            if r == 0:
                pr = prompt if len(prompt) <= 15 else prompt[:14] + '…'
                ax.set_title(f'{m}   $F_1$={f1:.3f}\n"{pr}"  thr {thr:.2f}',
                             fontsize=6.1, pad=4, linespacing=1.5)

    handles = [plt.Line2D([], [], color=C_WEED, lw=2,
                          label='weed ground truth / matched detection (IoU ≥ 0.50)'),
               plt.Line2D([], [], color=C_FP, lw=2,
                          label='unmatched detection (false positive)'),
               plt.Line2D([], [], color=C_MISS, lw=1.4, ls=(0, (3, 2)),
                          label='missed ground truth')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(wspace=0.035, hspace=0.045, bottom=0.06)
    fig.savefig(f'{FIG}/figQ2_qualitative.png')
    fig.savefig(f'{FIG}/figQ2_qualitative.pdf')
    plt.close(fig)
    print('figQ2  ' + ', '.join(f'{m}:{best[m][1]}@{best[m][2]}' for m in DETECTORS))


# ----------------------------------- Fig Q3: false alarms on weed-free imagery
def _fmt_thr(t):
    """YOLO-World scores are ~1e-3, so a two-decimal threshold prints as 0.00."""
    return f'{t:.2f}' if t >= 0.01 else f'{t:.1e}'


def figQ3(neg_idx=(600, 240, 720), grp='A', prompt='weed',
          ctrl=('D', 'sugarcane'), ctrl_idx=240, max_draw=25):
    """Rows: three expert-certified weed-free images under a weed prompt, then the
    same imagery under the crop prompt as a positive control.  Every detector sits
    at the operating threshold fixed beforehand on the annotated positive set."""
    negs = json.load(open(f'{WD}/negatives.json'))
    files = negs['files']
    negres = json.load(open(f'{RES}/negatives.json'))['runs']
    models = [m for m in DETECTORS if f'{m}|{grp}|{prompt}' in negres]
    ncol, nrow = len(models), len(neg_idx) + 1

    fig = plt.figure(figsize=(1.78 * ncol, 1.27 * nrow))
    outer = fig.add_gridspec(2, 1, height_ratios=[len(neg_idx), 1], hspace=0.22)
    gs_fa = outer[0].subgridspec(len(neg_idx), ncol, wspace=0.03, hspace=0.05)
    gs_ct = outer[1].subgridspec(1, ncol, wspace=0.03)

    imgs = {}
    for i in set(neg_idx) | {ctrl_idx}:
        im = load_disp(f'{negs["dir"]}/{files[i]}', 900)
        imgs[i] = (im, im.width * 2, im.height * 2)

    preds = {m: read_preds(m, 'neg', grp, prompt) for m in models}
    cpreds = {m: read_preds(m, 'neg', ctrl[0], ctrl[1]) for m in models}
    trunc = False

    def paint(ax, i, keep, colour, note):
        img, W, H = imgs[i]
        show(ax, img, W, H)
        for p in keep[:max_draw]:
            draw_box(ax, p['bbox'], colour, lw=1.1, alpha=0.9)
        corner_note(ax, note)

    for r, i in enumerate(neg_idx):
        for c, m in enumerate(models):
            run = negres[f'{m}|{grp}|{prompt}']
            thr = run['operating_threshold']
            keep = sorted([p for p in preds[m].get(-(i + 1), []) if p['score'] >= thr],
                          key=lambda p: -p['score'])
            trunc = trunc or len(keep) > max_draw
            star = '*' if len(keep) > max_draw else ''
            noun = 'false alarm' if len(keep) == 1 else 'false alarms'
            top = f'{keep[0]["score"]:.2f}' if keep else '—'
            ax = fig.add_subplot(gs_fa[r, c])
            paint(ax, i, keep, C_FP,
                  f'{len(keep)}{star} {noun}\nmax score {top}')
            if c == 0:
                ax.set_ylabel(f'{files[i]}\nweed-free', fontsize=6.6)
            if r == 0:
                ax.set_title(f'{m}\n"{prompt}"  thr {_fmt_thr(thr)} · '
                             f'{100 * run["frac_at_operating"]:.0f} % fire',
                             fontsize=6.1, pad=4, linespacing=1.5)

    # positive control: on these same images the crop IS present
    for c, m in enumerate(models):
        key = f'{m}|{ctrl[0]}|{ctrl[1]}'
        run = negres.get(key)
        ax = fig.add_subplot(gs_ct[0, c])
        if run is None:
            paint(ax, ctrl_idx, [], C_CANE, 'not run')
            continue
        thr = run['operating_threshold']
        keep = sorted([p for p in cpreds[m].get(-(ctrl_idx + 1), [])
                       if p['score'] >= thr], key=lambda p: -p['score'])
        trunc = trunc or len(keep) > max_draw
        star = '*' if len(keep) > max_draw else ''
        top = f'{keep[0]["score"]:.2f}' if keep else '—'
        noun = 'detection' if len(keep) == 1 else 'detections'
        paint(ax, ctrl_idx, keep, C_CANE,
              f'{len(keep)}{star} {noun}\nmax score {top}')
        ax.set_title(f'thr {_fmt_thr(thr)} · '
                     f'{100 * run["frac_at_operating"]:.0f} % fire',
                     fontsize=6.1, pad=3, color=C_CANE)
        if c == 0:
            ax.set_ylabel(f'{files[ctrl_idx]}\n"{ctrl[1]}", crop present',
                          fontsize=6.6, color=C_CANE)

    if trunc:
        print(f'  note: some panels exceed {max_draw} detections (marked *)')
    fig.savefig(f'{FIG}/figQ3_false_alarms.png')
    fig.savefig(f'{FIG}/figQ3_false_alarms.pdf')
    plt.close(fig)
    print('figQ3  ' + ', '.join(
        f'{m}@{_fmt_thr(negres[f"{m}|{grp}|{prompt}"]["operating_threshold"])}'
        for m in models))


if __name__ == '__main__':
    figQ1()
    figQ2()
    figQ3()
    print('\nwrote figQ1/figQ2/figQ3 to ov/figures/')
