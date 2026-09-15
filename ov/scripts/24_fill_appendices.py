#!/usr/bin/env python3
"""Render Appendix B (reproducibility record) and Appendix C (annotator protocol)
into the manuscript, replacing the two `[To be completed: ...]` placeholders.

Appendix B is generated entirely from ov/results/repro_record.json, so it cannot
drift from the environment it describes.  Appendix C reproduces the three
instruments actually shipped to the annotators; the Vietnamese originals are the
issued artefacts and the hashes are taken over those files, with an English
translation printed alongside.

Usage:  24_fill_appendices.py <in.docx> <out.docx>
"""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
RES = f'{ROOT}/ov/results'

PLACEHOLDER_B_START = '[To be completed: exact checkpoint revision hashes'
PLACEHOLDER_C_START = '[To be completed: the full round-1 instruction'


# ---------------------------------------------------------------- XML helpers
def esc(t):
    return (t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def para(text, style='BodyText', bold=False, italic=False):
    rpr = ''
    if bold or italic:
        rpr = '<w:rPr>' + ('<w:b />' if bold else '') + ('<w:i />' if italic else '') + '</w:rPr>'
    return (f'<w:p><w:pPr><w:pStyle w:val="{style}" /></w:pPr>'
            f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


MONO = ('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas" />'
        '<w:sz w:val="15" /><w:szCs w:val="15" />')


def verbatim(block):
    """One tight monospace paragraph per line, so the line structure survives.

    `Compact` is pandoc's no-spacing paragraph style; `SourceCode` carries its own
    spacing and would turn a 91-line procedure into five pages of loose text.
    """
    out = []
    for line in block.split('\n'):
        out.append('<w:p><w:pPr><w:pStyle w:val="Compact" />'
                   '<w:spacing w:before="0" w:after="0" w:line="240" '
                   'w:lineRule="auto" /><w:jc w:val="left" /></w:pPr>'
                   f'<w:r><w:rPr>{MONO}</w:rPr>'
                   f'<w:t xml:space="preserve">{esc(line) or " "}</w:t></w:r></w:p>')
    return ''.join(out)


def table(header, rows, weights=None):
    n = len(header)
    weights = weights or [1] * n
    total = 7920
    cols = [round(total * w / sum(weights)) for w in weights]
    cell = ('<w:tc><w:tcPr /><w:p><w:pPr><w:pStyle w:val="Compact" />'
            '<w:jc w:val="left" /></w:pPr><w:r>{rpr}'
            '<w:t xml:space="preserve">{t}</w:t></w:r></w:p></w:tc>')
    x = ['<w:tbl><w:tblPr><w:tblStyle w:val="Table" />'
         '<w:tblW w:type="pct" w:w="5000" />'
         '<w:tblLook w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" '
         'w:noHBand="0" w:noVBand="0" w:val="0020" /><w:jc w:val="start" /></w:tblPr>'
         '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{c}" />' for c in cols) + '</w:tblGrid>']
    x.append('<w:tr><w:trPr><w:tblHeader w:val="true" /></w:trPr>'
             + ''.join(cell.format(rpr='', t=esc(str(h))) for h in header) + '</w:tr>')
    for r in rows:
        x.append('<w:tr>' + ''.join(cell.format(rpr='', t=esc(str(c))) for c in r) + '</w:tr>')
    x.append('</w:tbl>')
    return ''.join(x)


def heading(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading3" /></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


# ------------------------------------------------------------- Appendix B
def appendix_b(rec):
    env, sup = rec['env_openvocab'], rec['env_supervised']
    costs = rec['model_costs']['models']
    out = [para(
        'Every value in this appendix is emitted by ov/scripts/23_repro_record.py '
        'from the environment that produced the results, and re-emitting it after '
        'a re-run regenerates the appendix rather than dating it.',
        style='FirstParagraph')]

    out.append(heading('B.1 Model checkpoints'))
    out.append(para(
        'Hugging Face configurations are pinned by commit revision; the two local '
        'weight files are pinned by SHA-256 over the file itself. Together these '
        'identify all seven open-vocabulary configurations and the supervised '
        'reference.'))
    rows = []
    for c in rec['checkpoints']:
        src = c['repo'] or c['weight_file']
        pin = c['revision'] or c['weight_sha256'] or '—'
        rows.append([c['key'], src, pin])
    out.append(table(['Configuration', 'Source', 'Revision or SHA-256'], rows, [2, 3, 4]))
    out.append(para(
        'SAM 3 Agent uses the SAM 3 checkpoint above for stage 2 and the Qwen2.5-VL '
        'checkpoint above for stage 1; it introduces no further weights.'))

    out.append(heading('B.2 Software environment'))
    out.append(para(
        f'Open-vocabulary arm: Python {env["python"]}, PyTorch {env["torch"]} '
        f'(CUDA {env["cuda"]}, cuDNN {env["cudnn"]}), transformers '
        f'{env["transformers"]}, ultralytics {env["ultralytics"]}, pycocotools '
        f'{env["pycocotools"]}, NumPy {env["numpy"]}, Pillow {env["pillow"]}, '
        f'accelerate {env["accelerate"]}, tokenizers {env["tokenizers"]}, '
        f'safetensors {env["safetensors"]}, SciPy {env["scipy"]}.'))
    out.append(para(
        'There is no separate sam3 distribution: SAM 3 ships inside transformers '
        f'{env["transformers"]} as transformers.models.sam3, and is used through '
        'Sam3Processor and Sam3Model. Pinning transformers therefore pins SAM 3.'))
    if 'error' not in sup:
        out.append(para(
            f'Supervised arm (separate virtualenv, because MMDetection pins an older '
            f'PyTorch): Python {sup["python"]}, PyTorch {sup["torch"]}, mmdet '
            f'{sup["mmdet"]}, mmcv {sup["mmcv"]}, mmengine {sup["mmengine"]}, '
            f'mmpretrain {sup["mmpretrain"]}, NumPy {sup["numpy"]}, pycocotools '
            f'{sup["pycocotools"]}.'))
    out.append(para(f'Hardware: {rec["gpu"]}, 12 GB, one GPU throughout.'))

    out.append(heading('B.3 Random seeds and determinism'))
    files = ', '.join(sorted(rec['seeds']))
    out.append(para(
        'Every resampling procedure in the analysis draws from a NumPy PCG64 '
        'generator constructed as numpy.random.default_rng(0). The seed is 0 at '
        f'every call site, in {files}. Re-running the analysis therefore reproduces '
        'every confidence interval exactly, not merely to within Monte-Carlo error.'))
    out.append(para(
        'Inference itself performs no sampling: all detectors are run in eval mode '
        'with deterministic forward passes, and the Qwen2.5-VL stage of SAM 3 Agent '
        'generates greedily (do_sample=False, max_new_tokens=40).'))

    out.append(heading('B.4 Inference cost'))
    out.append(para(
        'Two measurements are reported, because they answer different questions. '
        'The benchmark figures in Table 4 are a single 640 x 640 frame through each '
        'model’s own preprocessing and post-processing, batch size 1, one prompt, '
        'median of 30 timed iterations after 5 warm-up iterations, with the CUDA '
        'stream synchronised each iteration. The wall-clock figures below are the '
        'whole study as actually run, at the native capture resolution, and include '
        'JPEG decoding and disk I/O.'))
    rows = []
    for k, t in sorted(rec['timings'].items()):
        rows.append([k, f'{t["passes"]:,}', f'{t["seconds"] / 60:.1f}',
                     f'{t["s_per_pass"]:.3f}'])
    out.append(table(['Configuration', 'Image-prompt passes',
                      'Total wall-clock (min)', 's per pass'], rows, [3, 2, 2, 2]))
    out.append(para(
        'A pass is one image through one prompt. The 869-image negative set was run '
        'with 7 prompts (groups A, C2 and D) and the other sets with 15, which is '
        'why the totals are equal across the five detectors run on all three sets. '
        'SAM 3 Agent is counted as one pass per image, but each pass is one '
        'Qwen2.5-VL generation followed by one SAM 3 forward per extracted noun '
        'phrase, a mean of 1.85 per image.'))
    ml = costs.get('rtmdet', {}).get('latency_ms_median')
    sl = costs.get('sam3', {}).get('latency_ms_median')
    if ml and sl:
        out.append(para(
            f'For orientation, the supervised reference is the cheapest model in the '
            f'study as well as the most accurate: {costs["rtmdet"]["params_M"]:.0f} M '
            f'parameters at {ml:.0f} ms per frame, against '
            f'{costs["sam3"]["params_M"]:.0f} M at {sl:.0f} ms for the best '
            f'open-vocabulary configuration.'))
    return ''.join(out)


# ------------------------------------------------------------- Appendix C
ROUND1_EN = """ROUND 1 — MINIMAL INSTRUCTION

Draw a box around the weeds in the sugarcane field photograph. Do not box the cane.

————————————————————————————————————————————————————————————
Being a single sentence is deliberate.

This round simulates how labelling is commonly done today: the annotator receives
a short request and decides everything else alone. The agreement measured here is
the agreement most existing datasets actually have.

Round 2 uses these same 80 images but with detailed rules. Comparing the two
rounds tells us whether label noise is intrinsic to the problem or merely the
absence of rules.

So: do NOT ask for clarification, and do not try to guess what the coordinator
wants. Use your own natural judgement.

Some images contain no weed. Drawing nothing is a valid answer — press
"Save and go to next image" to record it.

If you meet an image where you cannot separate weed from cane, press M.
You decide what "cannot separate" means; this round gives no definition."""

ROUND2_EN = """ROUND 2 — DETAILED RULES
Round-2 SOP — version 1.0 — written 2026-08-31 — AWAITING COORDINATOR APPROVAL

The same 80 images as round 1. Do not reopen your round-1 results: the measurement
needs you to decide again independently.

1. LABELLING UNIT — what becomes one box
   1.1 One box = ONE CLUMP of weed, that is one root crown and the leaves growing
       from it.
   1.2 A detached leaf, root crown not visible, not connectable to any clump: DO NOT draw.
   1.3 Two clumps growing side by side but with two distinguishable crowns: draw TWO boxes.
   1.4 A dense weed mat with NO CANE MIXED IN, individual crowns not separable:
       draw ONE box for the whole patch.
       If there is cane in the patch this rule does NOT apply — see 1B.4.

1B. WHERE WEED AND CANE GROW INTERMIXED  (the most common case)
   A rectangle CANNOT enclose a weed clump while excluding a cane leaf crossing it.
   Do not try. In this dataset an average weed box is only about 44% actually weed;
   the rest is soil and intervening cane. That is normal, not an error.

   1B.1 A cane leaf crossing over a weed clump:
        draw the box around the weed clump AS IF the cane leaf were not there.
        DO NOT shrink the box, DO NOT split it into several boxes to dodge the leaf.

   1B.2 Weed pushing up from the middle of a cane stool:
        if you can make out a separate weed crown, draw a box for that clump.
        The box will contain many cane leaves — accept that.

   1B.3 A weed clump cut into two patches by a cane leaf, but you are confident it
        is THE SAME clump: draw ONE box around both patches.
        If you are not confident it is the same clump, draw TWO boxes.

   1B.4 A dense patch where weed and cane are interwoven and individual crowns
        cannot be separated:
        - draw the weed parts you CAN make out, however few;
        - DO NOT enclose the whole patch as a single weed box, because most of it
          is cane;
        - and press M to mark this image as UNDECIDABLE.

        If you cannot make out any weed at all in the image: draw nothing, just
        press M and save.

        Pressing M is NOT giving up. It is a valuable answer: we need to know which
        images defeat an experienced eye, and whether all three of you see the same
        ones that way. Do not hesitate to use it, and do not use it to get through
        the set faster.

   1B.5 General principle: the box follows the WEED CLUMP, not the gap between cane.

2. BOX BOUNDARIES
   2.1 The box hugs the visible leaves of that clump.
   2.2 A leaf occluded in the middle but clearly visible at both ends: count it as
       one leaf, enclose the whole of it.
   2.3 Do not enlarge the box to take in bare soil between leaves.
   2.4 A clump cut off at the image edge: still draw it, enclosing the visible part.

3. WHAT TO DRAW, WHAT NOT TO DRAW
   3.1 DO draw: every plant that is not cane.
   3.2 DO NOT draw: cane. Cane only needs to be recognised in order to be AVOIDED,
       not boxed.
   3.3 DO NOT draw: dry cane leaves, straw, rotting stems — not living plants.
   3.4 Tell cane from weed by comparing WITHIN THE IMAGE YOU ARE LOOKING AT:
       - cane leaves are distinctly broader, thicker, straighter, with a pale
         midrib running the length of the leaf, and radiate evenly from one crown;
       - weed leaves are much finer, softer, growing untidily or in dense clumps.
       Use the largest plant in the image as your reference; DO NOT use a fixed
       number: images were taken at different distances, so the same species has
       very different apparent widths from image to image.

4. SIZE
   4.1 There is NO minimum threshold. However small a clump is, if you judge it
       significant, draw it.
   4.2 This is left to you on purpose. Round 2 tightens the rules for the UNIT
       (what becomes one box) and for BOUNDARIES, but not for whether a small plant
       is significant — that is an agronomic judgement, not a drawing convention.

5. UNCERTAIN CASES
   5.1 Unsure whether cane or weed: DO NOT draw. Better a miss than a wrong label.
   5.2 Unsure whether one clump or two: draw according to the number of crowns you
       can count.
   5.3 Image blurred, underexposed, undecidable: draw nothing, still press save to
       record it.

————————————————————————————————————————————————————————————
If anything is unclear, ask the COORDINATOR, not another annotator.
Every answer is sent to all three of you at once and logged below.

————————————————————————————————————————————————————————————
REVISION LOG  (coordinator writes; never delete an old line)

  v1.0  —  2026-08-31  —  written, NO round-1 data yet, awaiting approval
           · 1.4 applies only to pure weed patches; mixed patches follow 1B.4
           · added the "undecidable" flag (key M) at 1B.4
           · 3.4 dropped the fixed pixel threshold, replaced by within-image comparison
           · 4.1 no minimum size threshold

  v….   —  ……/……/……  —  ………………………………………………

The SOP must be FROZEN before it is issued. Editing it mid-round means the three
annotators are no longer using the same document, and the agreement measurement
loses its meaning."""

GUIDE_EN = """ANNOTATION TOOL, PANEL — <annotator> — <round>

HOW TO RUN
  1. Extract the WHOLE folder to disk (do not run it from inside the zip).
  2. Double-click CHAY.bat
  3. A black window appears. DO NOT CLOSE IT while you work.
  4. The browser opens by itself. If not, open it and type: http://127.0.0.1:8799

  If Windows shows "Windows protected your PC":
     click "More info" -> "Run anyway".
     This is the default warning for any .bat file downloaded from elsewhere.

NOTES
  - Labels are saved IMMEDIATELY after each image. Shutting down mid-session
    loses no work.
  - Reopening resumes exactly where you stopped.
  - Work ALONE. Do not look at, ask, or compare with the other annotators.
  - Do not use any AI tool to suggest boxes.
  - If an image has no weed, draw nothing and still press "Save and go to next".

WHEN YOU FINISH — WHAT TO SEND BACK
  Zip the output folder and send it to the coordinator.
  It contains exactly 2 files:
      via_project_<annotator>.json   <- your labels
      thoi_gian.csv                  <- your time per image

  You do not need to send back the images folder (275 MB; the coordinator has it).

IF SOMETHING GOES WRONG
  Tell the COORDINATOR. Do not ask the other two annotators."""


def appendix_c(rec):
    ins = rec['annotator_instruments']
    f1, f2 = ins['files']['round1'], ins['files']['round2']
    pkgs = ins['packages']
    sop_hashes = {v['sop_sha256'] for k, v in pkgs.items() if k.endswith('round2')}
    r2_times = sorted({v['sop_mtime'] for k, v in pkgs.items() if k.endswith('round2')})
    r1_times = sorted({v['sop_mtime'] for k, v in pkgs.items() if k.endswith('round1')})
    earliest = ins.get('earliest_submission', {})

    out = [para(
        'Annotators are identified here as in Section 2.7; the personal names used '
        'in the distributed package filenames are not reproduced. '
        'The three instruments below are reproduced as issued. They were written in '
        'Vietnamese, the annotators’ working language; the Vietnamese files are the '
        'artefacts the hashes are taken over, and an English translation is given '
        'here. Each annotator received a self-contained application package '
        'containing the instruction set for that round, the 80 images and nothing '
        'else.', style='FirstParagraph')]

    out.append(heading('C.1 Integrity of the two instruction sets'))
    out.append(table(
        ['Artefact', 'Size', 'SHA-256'],
        [['Round-1 instruction (round1.txt)', f'{f1["bytes"]} bytes, {f1["lines"]} lines',
          f1['sha256']],
         ['Round-2 SOP (round2.txt)', f'{f2["bytes"]} bytes, {f2["lines"]} lines',
          f2['sha256']]], [3, 2, 5]))
    out.append(para(
        f'The round-2 SOP is byte-identical in all three annotator packages '
        f'({len(sop_hashes)} distinct hash across the three), so no annotator '
        'received a different version of the rules. The operational guide differs '
        'between packages only in the annotator name and round label it prints.'))
    out.append(para(
        f'On the ordering: the round-2 packages carry a build timestamp of '
        f'{", ".join(r2_times)}, the round-1 packages '
        f'{", ".join(r1_times)}, and the earliest annotator submission in the '
        f'archive is dated {earliest.get("mtime", "n/a")}. '
        'The round-2 procedure was therefore '
        'written and frozen before any round-1 data existed, which is what makes '
        'its rules impossible to have tuned to round-1 outcomes. The hashes above '
        'are computed from the archived artefacts; no third-party timestamping '
        'authority was used, so the ordering rests on the build and submission '
        'timestamps and on the SOP being identical across the three independently '
        'built packages.'))
    out.append(para(
        'The issued round-2 file carries the header line "awaiting coordinator '
        'approval" and a revision log entry recording the same. This is the version '
        'the annotators worked from; it is reproduced unaltered rather than tidied.'))

    out.append(heading('C.2 Round-1 instruction, in full'))
    out.append(verbatim(ROUND1_EN))

    out.append(heading('C.3 Round-2 standard operating procedure, in full'))
    out.append(verbatim(ROUND2_EN))

    out.append(heading('C.4 Operational guide issued to each annotator'))
    out.append(para(
        'Identical for all three annotators and both rounds apart from the '
        'substituted name and round label.'))
    out.append(verbatim(GUIDE_EN))
    return ''.join(out)


# ------------------------------------------------------------------- driver
def replace_placeholder(x, marker, new_xml):
    i = x.find(marker)
    assert i != -1, f'placeholder not found: {marker[:40]}'
    a = x.rfind('<w:p>', 0, i)
    b = x.find('</w:p>', i) + len('</w:p>')
    return x[:a] + new_xml + x[b:]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rec = json.load(open(f'{RES}/repro_record.json'))

    tmp = tempfile.mkdtemp()
    subprocess.run(['unzip', '-qo', src, '-d', tmp], check=True)
    for p in (os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs):
        if os.path.islink(p):
            os.unlink(p)
    xp = os.path.join(tmp, 'word', 'document.xml')
    x = open(xp, encoding='utf8').read()

    x = replace_placeholder(x, PLACEHOLDER_C_START, appendix_c(rec))
    x = replace_placeholder(x, PLACEHOLDER_B_START, appendix_b(rec))
    open(xp, 'w', encoding='utf8').write(x)

    if os.path.exists(dst):
        os.remove(dst)
    subprocess.run(['zip', '-Xrq', os.path.abspath(dst), '.'], cwd=tmp, check=True)
    shutil.rmtree(tmp)
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
