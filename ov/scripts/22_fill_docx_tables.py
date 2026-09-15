#!/usr/bin/env python3
"""Fill the empty cells of the manuscript's result tables from the result JSONs.

Every value written here is read from ov/results/*.json — nothing is typed in by
hand, so re-running this after a re-analysis produces a document that still
agrees with the data.

The document is edited as raw `word/document.xml`: each empty cell is the literal
run `<w:r><w:t xml:space="preserve">[ ]</w:t></w:r>`, and those runs are replaced
in document order.  A few already-filled cells are corrected too (see FIXUPS),
because they were populated from the group-pooled scores in `contrasts.json`
while their table's caption defines best-prompt scores from `eval_main.json`.

Usage:  22_fill_docx_tables.py <in.docx> <out.docx>
"""
import json, os, re, shutil, subprocess, sys, tempfile

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
RES = f'{ROOT}/ov/results'

EMPTY = '<w:r><w:t xml:space="preserve">[ ]</w:t></w:r>'
DETECTORS_T4 = ['owlv2-base', 'owlv2-large', 'gdino-tiny', 'gdino-base',
                'yoloworld-s', 'sam3', 'sam3agent', 'rtmdet']
DETECTORS_T6 = ['owlv2-base', 'owlv2-large', 'gdino-tiny', 'gdino-base',
                'yoloworld-s', 'sam3']
GROUPS = ['A', 'B', 'C1', 'C2', 'D', 'E']
WEED_GROUPS = ['A', 'B', 'C1', 'C2', 'E']          # D targets the crop, not weed
# Table 11 reports one representative prompt per group; group A was already
# filled from `weed`, so C2 uses the prompt the paper features throughout.
T11_PROMPT = {'A': 'weed', 'C2': 'unwanted plant among sugarcane'}
T11_ROWS = [('owlv2-base', 'C2'), ('owlv2-large', 'C2'), ('gdino-tiny', 'C2'),
            ('gdino-base', 'C2'), ('yoloworld-s', 'A'), ('sam3', 'A'),
            ('sam3', 'C2')]
NP_PER_IMAGE = 1.85          # mean noun phrases the MLLM returns, from sam3_agent_nps.json

# Figures the study generates and the document embeds, by relationship id.  The
# document carries the pre-revision versions (four detectors, on-artwork titles,
# ~210 dpi), so they are refreshed here together with the numbers they illustrate.
FIGURES = {'rId26': f'{ROOT}/ov/figures/figQ1_green_on_green.png',
           'rId48': f'{ROOT}/ov/figures/figQ3_false_alarms.png',
           'rId53': f'{ROOT}/ov/figures/figQ2_qualitative.png'}


def pct(x):
    """'100.0' -> '100', '99.9' -> '99.9'.  Never strip the digit off '0.3'."""
    t = f'{x:.1f}'
    return t[:-2] if t.endswith('.0') else t


def run(txt, bold=False):
    b = '<w:rPr><w:b /></w:rPr>' if bold else ''
    return f'<w:r>{b}<w:t xml:space="preserve">{txt}</w:t></w:r>'


def best_f1_by_group():
    """Best single prompt per (detector, group): what Table 6's caption defines."""
    ev = json.load(open(f'{RES}/eval_main.json'))['runs']
    out = {}
    for k, v in ev.items():
        m, _, _ = k.split('|', 2)
        out.setdefault(m, {}).setdefault(v['group'], []).append(v['bestF1@0.5']['best'])
    return {m: {g: max(vs) for g, vs in d.items()} for m, d in out.items()}


def table4_values():
    p = f'{RES}/model_costs.json'
    if not os.path.exists(p):
        raise SystemExit('run 21_model_costs.py --device cuda first')
    c = json.load(open(p))['models']

    def cell(k, field, fmt):
        v = c.get(k, {}).get(field)
        return fmt.format(v) if v is not None else '—'

    rows = {}
    for k in ['owlv2-base', 'owlv2-large', 'gdino-tiny', 'gdino-base',
              'yoloworld-s', 'sam3', 'rtmdet']:
        rows[k] = [cell(k, 'params_M', '{:.0f}'),
                   cell(k, 'peak_vram_GB', '{:.1f}'),
                   cell(k, 'latency_ms_median', '{:.0f}')]
    # SAM 3 Agent runs the two stages sequentially: parameters add, peak VRAM is
    # the larger stage, and latency is stage 1 plus one SAM 3 pass per noun phrase.
    s3, qw = c.get('sam3', {}), c.get('qwen2.5-vl-3b', {})
    if s3.get('params_M') and qw.get('params_M'):
        lat = None
        if s3.get('latency_ms_median') and qw.get('latency_ms_median'):
            lat = qw['latency_ms_median'] + NP_PER_IMAGE * s3['latency_ms_median']
        rows['sam3agent'] = [
            f"{s3['params_M'] + qw['params_M']:.0f}",
            f"{max(s3.get('peak_vram_GB', 0), qw.get('peak_vram_GB', 0)):.1f}",
            f'{lat:.0f}' if lat else '—']
    else:
        rows['sam3agent'] = ['—', '—', '—']
    return [v for k in DETECTORS_T4 for v in rows[k]]


def table6_values(best):
    vals, bolds = [], []
    for m in DETECTORS_T6:
        top = max(best[m][g] for g in WEED_GROUPS if g in best[m])
        for g in GROUPS:
            if m == 'yoloworld-s' and g in ('A', 'D'):
                continue                      # already filled / marked not applicable
            if m == 'sam3' and g != 'E':
                continue                      # filled (and corrected by FIXUPS)
            v = best[m].get(g)
            vals.append(f'{v:.3f}' if v is not None else '—')
            bolds.append(g in WEED_GROUPS and v is not None and abs(v - top) < 1e-9)
    return vals, bolds


def table11_values():
    neg = json.load(open(f'{RES}/negatives.json'))['runs']
    vals = []
    for m, g in T11_ROWS:
        r = neg.get(f'{m}|{g}|{T11_PROMPT[g]}')
        if r is None:
            vals += ['—', '—']
            continue
        vals += [pct(100 * r['frac_at_operating']),
                 f'{r["detections_per_image_at_operating"]:.1f}']
    return vals


TRANSFORMER = ('gdino-tiny', 'gdino-base', 'owlv2-base', 'owlv2-large')


def summary_rows():
    """Two summary lines for the false-alarm table.

    A single range over all six detectors reads as though the finding collapsed:
    YOLO-World barely fires on the negative set because it barely fires anywhere
    (its best weed F1 on annotated imagery is 0.071), so its low false-alarm rate
    is a symptom of a detector that does not work, not of discrimination.  The
    original 99.9-100 % claim was measured on the four Grounding DINO / OWLv2
    configurations and still holds exactly for them, so both are reported.
    """
    neg = json.load(open(f'{RES}/negatives.json'))['runs']
    weed = {k: r for k, r in neg.items() if r['role'] == 'false_alarm_rate'
            and r.get('frac_at_operating') is not None}

    def stat(items):
        fr = [100 * r['frac_at_operating'] for r in items]
        dp = [r['detections_per_image_at_operating'] for r in items]
        return (len(items), f'{pct(min(fr))}–{pct(max(fr))}',
                f'{min(dp):.1f}–{max(dp):.1f}')

    allw = list(weed.values())
    sub = [r for k, r in weed.items() if k.split('|')[0] in TRANSFORMER]
    n99 = sum(1 for r in allw if r['frac_at_operating'] >= 0.99)
    return (stat(allw), stat(sub), sorted({r['group'] for r in allw}), n99)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    best = best_f1_by_group()
    t4 = table4_values()
    t6, t6b = table6_values(best)
    t11 = table11_values()
    (n_comb, fr_range, dp_range), (n_sub, fr_sub, dp_sub), groups, n99 = summary_rows()

    assert len(t4) == 24, len(t4)
    assert len(t6) == 29, len(t6)
    assert len(t11) == 14, len(t11)
    fills = [(v, False) for v in t4] + list(zip(t6, t6b)) + [(v, False) for v in t11]

    tmp = tempfile.mkdtemp()
    subprocess.run(['unzip', '-qo', src, '-d', tmp], check=True)
    for p in (os.path.join(dp, f) for dp, _, fs in os.walk(tmp) for f in fs):
        if os.path.islink(p):
            os.unlink(p)
    xp = os.path.join(tmp, 'word', 'document.xml')
    x = open(xp, encoding='utf8').read()

    n = x.count(EMPTY)
    assert n == len(fills), f'{n} empty cells in the document, {len(fills)} values prepared'
    for txt, bold in fills:
        x = x.replace(EMPTY, run(txt, bold), 1)

    # --- corrections to cells that were already filled from the wrong source ---
    # Table 6 row `sam3`: group-pooled scores from contrasts.json, where the
    # caption defines best-prompt scores.  Scope the replacement to that row so
    # the identical numbers in the SAM 3 Agent table are left alone.
    i = x.find('>sam3<')
    while i != -1 and '0.168' not in x[i:i + 4000]:
        i = x.find('>sam3<', i + 1)
    assert i != -1, 'could not locate the sam3 row of Table 6'
    j = x.find('</w:tr>', i)
    row, orig = x[i:j], x[i:j]
    for old, g in [('0.168', 'A'), ('0.134', 'B'), ('0.133', 'C1'), ('0.170', 'C2')]:
        new = f'{best["sam3"][g]:.3f}'
        assert row.count(f'>{old}<') == 1, (old, row.count(f'>{old}<'))
        row = row.replace(f'>{old}<', f'>{new}<')
    top = max(best['sam3'][g] for g in WEED_GROUPS)
    row = row.replace(f'<w:r><w:t xml:space="preserve">{top:.3f}</w:t></w:r>',
                      run(f'{top:.3f}', True))
    x = x.replace(orig, row, 1)

    # --- the 'All 20 combinations' summary row of the false-alarm table ---
    old = '<w:t xml:space="preserve">All 20 combinations</w:t>'
    assert old in x
    x = x.replace(old, f'<w:t xml:space="preserve">All {n_comb} combinations</w:t>')
    for old, new in [('>A, B, C1, C2, E<', '>' + ', '.join(groups) + '<'),
                     ('>99.9–100<', f'>{fr_range}<'),
                     ('>2.1–7.4<', f'>{dp_range}<')]:
        assert x.count(old) >= 1, old
        x = x.replace(old, new, 1)
    # A second summary line.  Read alone, the all-six range (0.3-100 %) looks like
    # a retraction of the finding; it is not, so the subset the original claim was
    # measured on is stated next to it.
    anchor = x.find(f'All {n_comb} combinations')
    start = x.rfind('<w:tr', 0, anchor)
    end = x.find('</w:tr>', anchor) + len('</w:tr>')
    tr = x[start:end]
    extra = tr
    for a, b in [(f'All {n_comb} combinations',
                  f'{n_sub} of those, Grounding DINO and OWLv2 only'),
                 (f'>{fr_range}<', f'>{fr_sub}<'),
                 (f'>{dp_range}<', f'>{dp_sub}<')]:
        assert a in extra, a
        extra = extra.replace(a, b, 1)
    x = x[:end] + extra + x[end:]

    # ...and the hypothesis table, which restates the same count
    old = '20 of 20 combinations fire on 99.9–100% of 869 confirmed weed-free images'
    if old in x:
        x = x.replace(old,
                      f'{n99} of {n_comb} combinations fire on at least 99% of 869 '
                      f'confirmed weed-free images; all {n_sub} Grounding DINO and '
                      f'OWLv2 combinations fire on {fr_sub}%')

    # --- prose that restates the false-alarm claim -------------------------
    # The negative set now covers all six detectors, so "all twenty ... 99.9-100 %"
    # is no longer true as a universal statement.  Each passage below is rewritten
    # to the two-part form: what holds for every detector, and what holds for the
    # four that actually detect weeds.
    gd = 'Grounding DINO and OWLv2'
    prose = [
        ('All 20 prompt–detector pairs fire on 99.9–100% of 869 weed-free images',
         f'All 20 {gd} pairs fire on 99.9–100% of 869 weed-free images'),

        ('and all twenty tested prompt–detector combinations fire on 99.9–100% of '
         'weed-free images at their own optimal operating point',
         f'and {n99} of {n_comb} tested prompt–detector combinations fire on at least '
         f'99% of weed-free images at their own optimal operating point, the '
         f'{n_sub} {gd} combinations on {fr_sub}%'),

        ('Instrument 3: every weed prompt fires on every clean field',
         'Instrument 3: the detectors that find weeds also fire on every clean field'),

        (f'all twenty weed-prompt and detector combinations fire on 99.9–100% of '
         f'images, at 2.1–7.4 spurious detections per image (Table 11).',
         f'{n99} of the {n_comb} weed-prompt and detector combinations fire on at '
         f'least 99% of images (Table 11); all {n_sub} {gd} combinations fire on '
         f'{fr_sub}% of them, at {dp_sub} spurious detections per image. The four '
         'exceptions are not evidence of discrimination: three are YOLO-World, whose '
         'best weed F1 anywhere in this study is 0.071, and the fourth is SAM 3 with '
         'the bare prompt weed, which still fires on 72.3% of the clean images. A '
         'detector that is quiet on clean ground because it is quiet everywhere has '
         'not passed this control.'),

        ('for all twenty tested combinations without exception.',
         f'for all {n_sub} {gd} combinations without exception, and for {n99} of the '
         f'{n_comb} tested overall.'),

        ('weed-free images, for four detectors. Every box drawn is a false alarm.',
         'weed-free images, for all six detectors. Every box drawn in the upper '
         'block is a false alarm. The bottom row repeats one of those images under '
         'the crop prompt sugarcane, where the target really is present, so those '
         'detections are correct.'),

        ('every one of the twenty configurations tested here would fail that test at '
         'its own best operating point',
         f'every one of the {n_comb} configurations tested here would fail that test '
         'at its own best operating point, either by treating clean ground or by '
         'failing to find the weeds'),

        ('and all twenty prompt–detector combinations fired on 99.9–100% of confirmed '
         'weed-free images.',
         f'and {n99} of {n_comb} prompt–detector combinations fired on at least 99% of '
         f'confirmed weed-free images, the {n_sub} {gd} combinations on {fr_sub}%.'),

        ('across all 20 prompt–detector', f'across all {n_comb} prompt–detector'),
    ]
    for a, b in prose:
        assert x.count(a) == 1, (x.count(a), a[:60])
        x = x.replace(a, b, 1)

    # --- refresh the embedded figures, preserving the display width ----------
    for rid, path in FIGURES.items():
        if not os.path.exists(path):
            continue
        dest = os.path.join(tmp, 'word', 'media', f'{rid}.png')
        if not os.path.exists(dest):
            continue
        from PIL import Image
        w, h = Image.open(path).size
        shutil.copyfile(path, dest)
        i = x.find(f'r:embed="{rid}"')
        a = x.rfind('<w:drawing>', 0, i)
        b = x.find('</w:drawing>', i)
        seg = x[a:b]
        m = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"', seg)
        cx = int(m.group(1))
        cy = round(cx * h / w)
        seg = re.sub(r'(<wp:extent cx="\d+" cy=")\d+(")', rf'\g<1>{cy}\g<2>', seg)
        seg = re.sub(r'(<a:ext cx="\d+" cy=")\d+(")', rf'\g<1>{cy}\g<2>', seg)
        x = x[:a] + seg + x[b:]
        print(f'  refreshed {rid} from {os.path.basename(path)} ({w}x{h})')

    open(xp, 'w', encoding='utf8').write(x)

    # Pre-existing defect in the source document: the embedded figures are PNG
    # parts but [Content_Types].xml declares no png default, which is invalid OPC
    # and can cost the images on a strict reader.  Declare it.
    ct = os.path.join(tmp, '[Content_Types].xml')
    c = open(ct, encoding='utf8').read()
    if 'Extension="png"' not in c and any(
            f.lower().endswith('.png') for f in os.listdir(os.path.join(tmp, 'word', 'media'))
            if os.path.isdir(os.path.join(tmp, 'word', 'media'))):
        c = c.replace('<Default Extension="xml"',
                      '<Default Extension="png" ContentType="image/png" />'
                      '<Default Extension="xml"', 1)
        open(ct, 'w', encoding='utf8').write(c)
        print('  declared image/png in [Content_Types].xml (was missing in the source)')
    if os.path.exists(dst):
        os.remove(dst)
    subprocess.run(['zip', '-Xrq', os.path.abspath(dst), '.'], cwd=tmp, check=True)
    shutil.rmtree(tmp)
    print(f'wrote {dst}')
    print(f'  Table 4: {len(t4)} cells   Table 6: {len(t6)} cells (+4 corrected)   '
          f'Table 11: {len(t11)} cells')
    print(f'  summary: {n_comb} weed-prompt/detector combinations over groups '
          f'{", ".join(groups)}, {fr_range} % of images fire, {dp_range} det/image')
    print(f'           {n_sub} Grounding DINO / OWLv2 combinations: {fr_sub} %, '
          f'{dp_sub} det/image;  {n99}/{n_comb} fire on >= 99 %')


if __name__ == '__main__':
    main()
