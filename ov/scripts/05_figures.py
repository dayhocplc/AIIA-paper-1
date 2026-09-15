#!/usr/bin/env python3
"""Figures for the manuscript. Colour-blind-safe palette, no red/green pairing."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
RES = f'{ROOT}/ov/results'
FIG = f'{ROOT}/ov/figures'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})

# Okabe-Ito, colour-blind safe
CB = {'A': '#0072B2', 'B': '#56B4E9', 'C1': '#E69F00', 'C2': '#D55E00',
      'D': '#009E73', 'E': '#CC79A7'}
GORDER = ['A', 'B', 'E', 'C1', 'C2', 'D']
GLABEL = {'A': 'A\nabstract\nnoun', 'B': 'B\nmorph.\nnoun',
          'E': 'E\nlong, non-\nrelational', 'C1': 'C1\nrelational\nnegated',
          'C2': 'C2\nrelational\naffirm.', 'D': 'D\nnominal\ncontrol'}

CEIL_LO, CEIL_HI, CEIL = 0.342, 0.454, 0.399
HUMAN = 0.391


def fig1_groups():
    """Union-IoU and best-F1 by prompt group, per model, with the label ceiling."""
    ev = json.load(open(f'{RES}/eval_main.json'))
    # sam3agent has no prompt groups (one MLLM-decomposed run); it appears in Fig. 6
    models = sorted({k.split('|')[0] for k in ev['runs']} - {'sam3agent'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    w = 0.8 / len(models)

    for ax, key, ylab in [(axes[0], 'kappa', "Cohen's $\\kappa$, pixel grid\n(prevalence-corrected)"),
                          (axes[1], 'bestF1@0.5', 'Best $F_1$ @ IoU 0.50')]:
        for mi, m in enumerate(models):
            xs, ys, los, his = [], [], [], []
            for gi, g in enumerate(GORDER):
                vals = [v for k, v in ev['runs'].items()
                        if k.startswith(m + '|') and v['group'] == g]
                if not vals:
                    continue
                best = max(vals, key=lambda v: v[key]['best'])
                xs.append(gi + mi * w - 0.4 + w / 2)
                ys.append(best[key]['best'])
                los.append(best[key]['best'] - best[key]['lo'])
                his.append(best[key]['hi'] - best[key]['best'])
            ax.bar(xs, ys, width=w * 0.9, label=m,
                   color=plt.cm.tab20(mi * 2 / 20), edgecolor='k', linewidth=0.4)
            ax.errorbar(xs, ys, yerr=[los, his], fmt='none', ecolor='k',
                        elinewidth=0.7, capsize=1.8)
        ax.set_xticks(range(len(GORDER)))
        ax.set_xticklabels([GLABEL[g] for g in GORDER], fontsize=6.5)
        ax.set_ylabel(ylab)
        for gi, g in enumerate(GORDER):
            ax.axvspan(gi - 0.5, gi + 0.5, color=CB[g], alpha=0.055, lw=0)

    axes[1].axhspan(CEIL_LO, CEIL_HI, color='0.45', alpha=0.22, lw=0,
                    label='label-noise ceiling')
    axes[1].axhline(HUMAN, color='0.15', ls='--', lw=1.1, label='expert–expert $F_1$')
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2, loc='upper left')
    axes[1].legend(fontsize=6, frameon=False, loc='upper center', ncol=2)
    axes[1].set_ylim(0, 0.56)
    # group D here is NOT size-matched; see fig5 for the controlled comparison
    axes[1].annotate('D not size-matched\n(see Fig. 5)', xy=(5, 0.35),
                     xytext=(4.05, 0.47), fontsize=6, ha='center',
                     arrowprops=dict(arrowstyle='->', lw=0.7))
    axes[0].set_title('(a) region agreement, prevalence-corrected',
                      fontsize=9, loc='left')
    axes[1].set_title('(b) instance matching, vs. human ceiling', fontsize=9, loc='left')
    fig.savefig(f'{FIG}/fig1_prompt_groups.png')
    plt.close(fig)
    print('fig1_prompt_groups.png')


def fig2_contrasts():
    """Forest plot of the paired contrasts."""
    c = json.load(open(f'{RES}/contrasts.json'))
    models = sorted(set(c['models']) - {'sam3agent'})
    keys = ['C2-E', 'C1-C2', 'E-A', 'D-A', 'B-A']
    labels = {'C2-E': 'C2 − E   relationality\n(length held constant)',
              'C1-C2': 'C1 − C2   negation',
              'E-A': 'E − A   prompt length',
              'D-A': 'D − A   category type\n(nominal vs relational)',
              'B-A': 'B − A   morphology'}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    yt, ytl = [], []
    row = 0
    for k in keys:
        for mi, m in enumerate(models):
            r = c['models'][m]['contrasts'].get(k)
            if not r:
                continue
            y = row
            ax.errorbar(r['diff'], y, xerr=[[r['diff'] - r['lo']], [r['hi'] - r['diff']]],
                        fmt='o', ms=4, color=plt.cm.tab20(mi * 2 / 20),
                        elinewidth=1.3, capsize=2.5,
                        label=m if k == keys[0] else None)
            row += 1
        yt.append(row - len(models) / 2 - 0.5)
        ytl.append(labels[k])
        row += 1
    ax.axvline(0, color='k', lw=1)
    ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('paired difference, 95 % bootstrap CI\n'
                  '(micro-$F_1$ within the weed class; $\\kappa$ for D − A)')
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    fig.savefig(f'{FIG}/fig2_contrasts.png')
    plt.close(fig)
    print('fig2_contrasts.png')


def fig3_protocol():
    """Annotation-protocol asymmetry: why instance AP cannot carry the comparison."""
    gt = json.load(open(f'{ROOT}/ov/work_dirs/eval2c_all.json'))
    W = {i['id']: (i['width'], i['height']) for i in gt['images']}
    ar = {0: [], 1: []}
    for a in gt['annotations']:
        w, h = W[a['image_id']]
        ar[a['category_id']].append(a['bbox'][2] * a['bbox'][3] / (w * h) * 100)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.logspace(-1, 2.1, 40)
    ax.hist(ar[0], bins=bins, alpha=0.72, label=f'weed  (n={len(ar[0])})', color='#0072B2')
    ax.hist(ar[1], bins=bins, alpha=0.72, label=f'sugarcane  (n={len(ar[1])})', color='#009E73')
    ax.set_xscale('log')
    ax.axvline(np.median(ar[0]), color='#0072B2', ls='--', lw=1.2)
    ax.axvline(np.median(ar[1]), color='#009E73', ls='--', lw=1.2)
    ax.set_xlabel('box area (% of frame, log scale)')
    ax.set_ylabel('number of boxes')
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title('Instance-level vs region-level annotation', fontsize=9, loc='left')
    fig.savefig(f'{FIG}/fig3_annotation_protocol.png')
    plt.close(fig)
    print('fig3_annotation_protocol.png')


def fig4_negatives():
    p = f'{RES}/negatives.json'
    if not os.path.exists(p):
        print('fig4 skipped (no negatives.json yet)'); return
    neg = json.load(open(p))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    # On these images sugarcane IS present and weed is NOT: group D curves are a
    # true-positive rate, groups A/C2 a false-alarm rate.  Plotted together
    # because the comparison is the point.
    # 28 individual curves is unreadable; average within (detector, group) and
    # show the weed groups (false alarms) against the crop group (true positives)
    agg = {}
    for key, r in neg['runs'].items():
        m, g, _ = key.split('|')
        gg = 'weed' if g in ('A', 'C2') else 'crop'
        agg.setdefault((m, gg), []).append(r)
    models = sorted({k[0] for k in agg})
    for ci, m in enumerate(models):
        for gg, ls in [('weed', '-'), ('crop', '--')]:
            rs = agg.get((m, gg))
            if not rs:
                continue
            y = np.mean([r['frac_images_with_detection'] for r in rs], axis=0)
            ax.plot(rs[0]['thresholds'], y, lw=1.7, ls=ls,
                    color=plt.cm.tab10(ci / 10),
                    label=f"{'✗' if gg == 'weed' else '✓'} {m}  ({gg})")
            ops = [(r['operating_threshold'], r['frac_at_operating']) for r in rs
                   if r.get('operating_threshold') is not None]
            if ops:
                ax.plot([np.mean([o[0] for o in ops])],
                        [np.mean([o[1] for o in ops])], 'o',
                        ms=6, mec='k', mew=0.7, color=plt.cm.tab10(ci / 10))
    ax.set_xlabel('confidence threshold  (● = $F_1$-optimal operating point,\n'
                  'fixed on the annotated set before scoring these images)')
    ax.set_ylabel('fraction of the 869 images\nwith ≥ 1 detection')
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=6.5, frameon=False, loc='lower left', ncol=1,
              title='✗ weed prompts (A, C2) = FALSE alarms\n✓ crop prompt (D) = true positives',
              title_fontsize=6.5, alignment='left')
    ax.set_title('869 expert-confirmed weed-free sugarcane images',
                 fontsize=9, loc='left')
    fig.savefig(f'{FIG}/fig4_false_alarms.png')
    plt.close(fig)
    print('fig4_false_alarms.png')


def fig5_reversal():
    """The paper's methodological core: the crop-vs-weed contrast changes SIGN
    depending on whether annotation granularity is controlled."""
    sm = json.load(open(f'{RES}/size_matched.json'))
    models = sorted(set(sm['models']) - {'sam3agent'})
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    h = 0.34
    for i, m in enumerate(models):
        # identical metric, identical images: the ONLY difference between the two
        # bars is whether the annotation-granularity confound was controlled
        u = sm['models'][m]['both_conditions']['uncontrolled']
        s = sm['models'][m]['both_conditions']['size_matched']
        ax.barh(i + h / 2, u['diff'], height=h, color='#E69F00', edgecolor='k',
                lw=0.4, label='uncontrolled' if i == 0 else None)
        ax.errorbar(u['diff'], i + h / 2,
                    xerr=[[u['diff'] - u['lo']], [u['hi'] - u['diff']]],
                    fmt='none', ecolor='k', elinewidth=0.9, capsize=2)
        ax.barh(i - h / 2, s['diff'], height=h, color='#0072B2', edgecolor='k',
                lw=0.4, label='size-matched' if i == 0 else None)
        ax.errorbar(s['diff'], i - h / 2,
                    xerr=[[s['diff'] - s['lo']], [s['hi'] - s['diff']]],
                    fmt='none', ecolor='k', elinewidth=0.9, capsize=2)
    ax.axvline(0, color='k', lw=1.1)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel('D − A: nominal control minus relational target\n'
                  'micro-$F_1$ @ IoU 0.50, same 95 images, same metric\n(positive = the crop noun is easier)')
    ax.legend(fontsize=7.5, frameon=False, loc='lower right')
    ax.set_title('The crop\u2013weed gap reverses sign in 3 of 4 detectors\n'
                 'once box size is matched',
                 fontsize=9, loc='left')
    fig.savefig(f'{FIG}/fig5_confound_reversal.png')
    plt.close(fig)
    print('fig5_confound_reversal.png')


def fig6_sam3_agent():
    """SAM 3 versus its own MLLM wrapper, on both evaluation sets."""
    ev = json.load(open(f'{RES}/eval_main.json'))
    po = json.load(open(f'{RES}/panel_ov.json'))

    def best(runs, model, group, key):
        v = [r for k, r in runs.items() if k.startswith(model + '|') and r['group'] == group]
        return max(x[key]['best'] if isinstance(x[key], dict) else x[key] for x in v) if v else 0.0

    labels = ['A\nbare noun\n"weed"', 'B\nmorph.\n"green plant"',
              'C2\nrelational\n(affirmative)', 'Agent\nMLLM-decomposed\nnoun phrases']
    main = [best(ev['runs'], 'sam3', g, 'bestF1@0.5') for g in ('A', 'B', 'C2')]
    main.append(max([r['bestF1@0.5']['best'] for k, r in ev['runs'].items()
                     if k.startswith('sam3agent')] or [0]))
    pan = [best(po['runs'], 'sam3', g, 'mean') for g in ('A', 'B', 'C2')]
    pan.append(max([r['mean'] for k, r in po['runs'].items()
                    if k.startswith('sam3agent')] or [0]))

    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    cols = ['#0072B2'] * 3 + ['#D55E00']
    ax.bar(x - w/2, main, w, color=cols, edgecolor='k', lw=0.5)
    ax.bar(x + w/2, pan, w, color=cols, edgecolor='k', lw=0.5, hatch='///')
    from matplotlib.patches import Patch
    proxies = [Patch(fc='0.8', ec='k', lw=0.5, label='222-image set'),
               Patch(fc='0.8', ec='k', lw=0.5, hatch='///',
                     label='80-image panel (4 annotators)'),
               Patch(fc='#0072B2', ec='k', lw=0.5, label='SAM 3, prompted directly'),
               Patch(fc='#D55E00', ec='k', lw=0.5, label='SAM 3 Agent')]
    for xi, (a, b) in enumerate(zip(main, pan)):
        ax.text(xi - w/2, a + .004, f'{a:.3f}', ha='center', fontsize=6.5)
        ax.text(xi + w/2, b + .004, f'{b:.3f}', ha='center', fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('best micro-$F_1$ @ IoU 0.50')
    ax.set_ylim(0, max(main + pan) * 1.28)
    ax.axvline(2.5, color='0.6', lw=0.9, ls=':')
    ax.legend(handles=proxies, fontsize=6.5, frameon=False, loc='upper right', ncol=2)
    ax.set_title("SAM 3 prompted directly vs SAM 3 Agent\n"
                 "(the wrapper its authors propose for non-noun-phrase queries)",
                 fontsize=8.5, loc='left')
    fig.savefig(f'{FIG}/fig6_sam3_agent.png')
    plt.close(fig)
    print('fig6_sam3_agent.png')


if __name__ == '__main__':
    fig3_protocol()
    if os.path.exists(f'{RES}/eval_main.json'):
        fig1_groups()
    if os.path.exists(f'{RES}/contrasts.json'):
        fig2_contrasts()
    fig4_negatives()
    if os.path.exists(f"{RES}/size_matched.json"):
        fig5_reversal()
    if os.path.exists(f'{RES}/panel_ov.json'):
        fig6_sam3_agent()
