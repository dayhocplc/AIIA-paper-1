#!/usr/bin/env python3
"""Assemble the code-and-data archive that accompanies the manuscript.

What goes in: every script that produced a number in the paper, every result
JSON those scripts emit, the per-image sufficient statistics the bootstrap runs
on, all model predictions, the figures, the panel annotations this group drew,
the annotator instruments, and the run logs.

What stays out, and why:
  * the source images and the dataset owner's bounding boxes — the MIT
    SugarcaneWeedDataset carries no licence, so it grants no redistribution
    right (see DATA_LICENSING.md); an image *index* is shipped instead, which is
    enough to interpret every prediction id
  * model weights — obtainable by the pinned revisions in Appendix B
  * virtualenvs, caches

Text files are de-identified on the way in: absolute home paths become a
configurable root and annotator first names in filenames become <annotator>.
No logic is altered; the rewrite is a textual substitution on paths and names.

Two variants are produced, because journals and data repositories have different
size limits:
  full   everything, including the 378 MB of raw model predictions
  light  the same minus the predictions and the figures, small enough to upload
         as supplementary material; the per-image sufficient statistics that the
         bootstrap actually resamples are kept, so every interval in the paper
         still recomputes

Usage:  25_build_submission_archive.py [out.zip] [--light]
"""
import csv, hashlib, io, json, os, re, shutil, subprocess, sys, tempfile

ROOT = os.environ.get('GIAMIA_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..')))
NAME = 'gia-mia-ov-audit-code-and-data'
TEXT_EXT = {'.py', '.sh', '.md', '.txt', '.json', '.csv', '.log', '.bib'}

PORTABLE_ROOT = (
    "os.environ.get('GIAMIA_ROOT',\n"
    "                      os.path.abspath(os.path.join(os.path.dirname(__file__),\n"
    "                                                   '..', '..')))")

SUBS = [
    # annotator first names appear only inside filenames; the paper calls them A1-A3
    (re.compile(r'via_project_(binh|chi|hai)'), 'via_project_<annotator>'),
    (re.compile(r'panel_(binh|chi|hai)\b'), 'panel_<annotator>'),
    (re.compile(r'GanNhanPanel_(binh|chi|hai)_'), 'GanNhanPanel_<annotator>_'),
    (re.compile(r'<HOME>/\s\'"]+/Downloads'), '<PANEL_SUBMISSIONS_DIR>'),
    (re.compile(r'<HOME>/\s\'"]+/Gia-mia'), '<GIAMIA_ROOT>'),
    (re.compile(r'<HOME>/\s\'"]+'), '<HOME>'),
]


def deidentify(text, is_python):
    if is_python:
        text = text.replace("", '')      # no-op guard
        text = re.sub(r"ROOT = '<HOME>/\s']+/Gia-mia'", 'ROOT = ' + PORTABLE_ROOT, text)
    for pat, rep in SUBS:
        text = pat.sub(rep, text)
    return text


def copy_tree(src, dst, deid=True, skip=None):
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for f in filenames:
            if skip and skip(f):
                continue
            s = os.path.join(dirpath, f)
            d = os.path.join(dst, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            ext = os.path.splitext(f)[1].lower()
            if deid and ext in TEXT_EXT:
                try:
                    t = open(s, encoding='utf8').read()
                except UnicodeDecodeError:
                    shutil.copyfile(s, d); n += 1; continue
                open(d, 'w', encoding='utf8').write(deidentify(t, ext == '.py'))
            else:
                shutil.copyfile(s, d)
            n += 1
    return n


def image_index(stage):
    """id -> file name, size and split, with the dataset owner's boxes left out."""
    out = {}
    for setname in ['all', 'test', 'panel']:
        p = f'{ROOT}/ov/work_dirs/eval2c_{setname}.json'
        if not os.path.exists(p):
            continue
        ev = json.load(open(p))
        dst = f'{stage}/ov/work_dirs/image_index_{setname}.csv'
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['image_id', 'file_name', 'width', 'height', 'split', 'has_cane'])
            for im in ev['images']:
                w.writerow([im['id'], im['file_name'], im['width'], im['height'],
                            im.get('split', ''), im.get('has_cane', '')])
        out[setname] = len(ev['images'])
    return out


def panel_manifest(stage):
    """Ship the panel provenance without the annotators' names or local paths."""
    p = f'{ROOT}/ov/panel/manifest.json'
    if not os.path.exists(p):
        return
    m = json.load(open(p))
    for k, v in m.get('annotators', {}).items():
        v.pop('source_file', None)
        v.pop('via_project_name', None)
        v['source_folder'] = k        # ann1..ann4, not the submission folder name
    m['note'] = (
        'Four round-2 submissions were collected. Two carried the same VIA project '
        'template name, which is a template artefact and not a duplicate: their '
        'mutual F1 is 0.232, the same level as any other pair. Annotator identities '
        'are not published; see Section 2.7 for how the manuscript refers to them.')
    json.dump(m, open(f'{stage}/ov/panel/manifest.json', 'w'), indent=1)


def requirements(stage):
    os.makedirs(f'{stage}/env', exist_ok=True)
    for label, py in [('openvocab', f'{ROOT}/ov/envs/ov/bin/python'),
                      ('supervised', f'{ROOT}/repro/envs/mmdet/bin/python')]:
        try:
            r = subprocess.run([py, '-m', 'pip', 'freeze'], capture_output=True,
                               text=True, check=True).stdout
        except Exception as e:
            r = f'# could not be captured: {type(e).__name__}: {e}\n'
        open(f'{stage}/env/requirements-{label}.txt', 'w').write(
            deidentify(r, False))
    # the reproducibility record names the submission file it read a timestamp from
    rec = open(f'{ROOT}/ov/results/repro_record.json', encoding='utf8').read()
    open(f'{stage}/env/environment.json', 'w', encoding='utf8').write(
        deidentify(rec, False))


README = """# Code and data for the open-vocabulary weed-detection audit

This archive contains the analysis code, the model predictions and every
intermediate result behind the tables and figures of the manuscript. It does
**not** contain the source imagery or the dataset owner's annotations: see
`DATA_LICENSING.md`.

## Layout

    ov/scripts/            every script that produced a number in the paper
    ov/results/            the JSON each script emits; all tables read from here
    ov/figures/            the figures, PNG and PDF
    ov/work_dirs/preds/    model predictions, one JSON per (detector, prompt, set)
    ov/work_dirs/stats/    per-image tp/fp/fn over the threshold sweep (.npz),
                           the sufficient statistics the bootstrap resamples
    ov/work_dirs/logs/     inference and driver logs, as run
    ov/work_dirs/image_index_*.csv
                           image id -> file name, size, split (see below)
    ov/work_dirs/negatives.json
                           the 869 expert-confirmed weed-free file names
    ov/work_dirs/sam3_agent_nps*.json
                           stage-1 noun phrases from the SAM 3 Agent arm
    ov/panel/              the panel annotations this group drew, COCO format
    panel_app/sop/         the two annotator instruction sets, as issued
    env/                   pip freeze of both environments, and the full
                           reproducibility record (also Appendix B)

## Reproducing

1. Obtain the MIT SugarcaneWeedDataset from its own repository (cited in the
   manuscript) and place it at `repro/mmdet_mit/data/`, keeping the upstream
   layout. This supplies both the images and the owner's COCO annotations.
2. Create the two environments from `env/requirements-*.txt`. The supervised arm
   needs its own virtualenv because MMDetection pins an older PyTorch.
3. Set `GIAMIA_ROOT` to wherever you unpacked this archive, or edit the `ROOT`
   constant at the top of the scripts.
4. `python ov/scripts/00_build_evalsets.py` rebuilds the evaluation sets from the
   dataset you obtained in step 1. This regenerates the `eval2c_*.json` files
   that are omitted here, and their image ids match `image_index_*.csv`, so the
   shipped predictions line up without re-running inference.
5. `ov/scripts/02_eval.py`, `03_compare.py`, `06_negatives.py`, `07_vs_supervised.py`,
   `09_size_matched.py` and `11_indirect.py` regenerate `ov/results/` from the
   shipped predictions. `08_tables.py`, `05_figures.py` and `20_qualitative_figures.py`
   regenerate the tables and figures from those results.
6. Re-running inference (`01_infer.py`, `10_yoloworld.py`) is only needed to
   reproduce the predictions themselves; the checkpoint revisions are pinned in
   `env/environment.json`.

Every resampling procedure uses `numpy.random.default_rng(0)`, so re-running the
analysis reproduces each confidence interval exactly rather than to within
Monte-Carlo error.

## De-identification

Text files have had absolute home paths replaced by a configurable root, and the
annotators' first names, which appeared only inside submission filenames,
replaced by `<annotator>`. No logic was altered.

Twenty-five of the twenty-eight scripts now resolve their root automatically from
their own location, so they run unmodified once the dataset is in place. The
exceptions are `12_panel_agreement.py`, `13_panel_ceiling.py`, `14_panel_eval.py`
and `15_export_panel_coco.py`, which read the annotators' raw submissions from a
local directory that is not part of this archive, and `18_md_to_docx.py`,
`19_build_references.py` and `check_highlights.py`, which are manuscript
production utilities rather than analysis. In those files the path appears as
`<PANEL_SUBMISSIONS_DIR>` or `<GIAMIA_ROOT>` and must be set before use. The
panel results they produce are shipped in `ov/results/` and `ov/panel/`.

## Known gap

`ov/results/panel_agreement.json`, `panel_ceiling.json` and `panel_ov.json` are
the four-annotator, round-2-only panel analysis. The three-annotator round-1
versus round-2 comparison reported in Section 3.3 and in the last three rows of
Table 10 was produced by a later analysis that is not part of this archive; the
round-1 submissions themselves are likewise not included here. Everything else
in the manuscript — Sections 3.1, 3.2, 3.4, 3.5, 3.6 and Tables 4-9 and 11-13 —
is fully covered.

## Integrity

`MANIFEST.sha256` lists a SHA-256 for every file in this archive. Verify with

    sha256sum -c MANIFEST.sha256
"""

LICENSING = """# Data licensing and what is not in this archive

The imagery and the ground-truth boxes analysed in this study come from the MIT
SugarcaneWeedDataset (Papa et al. 2026, npj Artificial Intelligence 2:40,
doi:10.1038/s44387-026-00096-0). That repository carries **no licence file**.
Absent a licence, no redistribution right is granted, so this archive contains:

  * **no source images**, and
  * **none of the dataset owner's annotation files**.

What is shipped in their place is an image *index* (`image_index_*.csv`: id,
file name, pixel dimensions, split) and the script that rebuilds the evaluation
sets from the dataset once you have obtained it from its own repository. The
index is sufficient to interpret every prediction id in `ov/work_dirs/preds/`.

Everything else in this archive is this group's own work and is released with the
manuscript: the analysis code, the model predictions, the derived statistics, the
figures, and the 80-image panel annotations in `ov/panel/`.

The panel annotations were drawn by three annotators recruited for this study.
They are identified as A1-A3 in the manuscript; their names are not published
here.
"""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    light = '--light' in sys.argv
    out = args[0] if args else f'{ROOT}/{NAME}{"-light" if light else ""}.zip'
    tmp = tempfile.mkdtemp()
    stage = os.path.join(tmp, NAME + ('-light' if light else ''))
    os.makedirs(stage)

    counts = {}
    counts['scripts'] = copy_tree(f'{ROOT}/ov/scripts', f'{stage}/ov/scripts')
    counts['results'] = copy_tree(f'{ROOT}/ov/results', f'{stage}/ov/results')
    if not light:
        counts['figures'] = copy_tree(f'{ROOT}/ov/figures', f'{stage}/ov/figures',
                                      deid=False)
        counts['predictions'] = copy_tree(f'{ROOT}/ov/work_dirs/preds',
                                          f'{stage}/ov/work_dirs/preds')
    counts['stats'] = copy_tree(f'{ROOT}/ov/work_dirs/stats',
                                f'{stage}/ov/work_dirs/stats', deid=False)
    counts['logs'] = copy_tree(f'{ROOT}/ov/work_dirs/logs',
                               f'{stage}/ov/work_dirs/logs')
    counts['panel'] = copy_tree(f'{ROOT}/ov/panel', f'{stage}/ov/panel')
    counts['protocol'] = copy_tree(f'{ROOT}/panel_app/sop', f'{stage}/panel_app/sop')

    for f in ['negatives.json', 'sam3_agent_nps.json',
              'sam3_agent_nps_HONG_chep_vi_du.json']:
        s = f'{ROOT}/ov/work_dirs/{f}'
        if os.path.exists(s):
            shutil.copyfile(s, f'{stage}/ov/work_dirs/{f}')
    for f in sorted(os.listdir(f'{ROOT}/ov/work_dirs')):
        if f.startswith('driver') and f.endswith('.log'):
            t = open(f'{ROOT}/ov/work_dirs/{f}', encoding='utf8', errors='replace').read()
            open(f'{stage}/ov/work_dirs/{f}', 'w', encoding='utf8').write(
                deidentify(t, False))
    os.makedirs(f'{stage}/ov/paper', exist_ok=True)
    shutil.copyfile(f'{ROOT}/ov/paper/tables.md', f'{stage}/ov/paper/tables.md')

    counts['image_index'] = image_index(stage)
    panel_manifest(stage)
    requirements(stage)
    readme = README
    if light:
        readme += (
            '\n## This is the light variant\n\n'
            'The raw model predictions (`ov/work_dirs/preds/`, 378 MB) and the '
            'figure files are omitted here and are in the full archive. What '
            'remains is sufficient to recompute every confidence interval in the '
            'paper: `ov/work_dirs/stats/` holds the per-image tp/fp/fn counts over '
            'the whole threshold sweep, which is what the bootstrap resamples. '
            'Re-deriving those statistics from scratch, rather than trusting them, '
            'needs the predictions from the full archive or a re-run of '
            '`01_infer.py`.\n')
    open(f'{stage}/README.md', 'w').write(readme)
    open(f'{stage}/DATA_LICENSING.md', 'w').write(LICENSING)

    # manifest last, over everything else
    lines, total = [], 0
    for dirpath, _, files in os.walk(stage):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, stage)
            h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            lines.append(f'{h}  {rel}')
            total += os.path.getsize(p)
    lines.sort(key=lambda l: l.split('  ', 1)[1])
    open(f'{stage}/MANIFEST.sha256', 'w').write('\n'.join(lines) + '\n')

    # A public archive must not carry the paths or names it was built from.
    leaks = []
    for line in lines:
        rel = line.split('  ', 1)[1]
        if os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        try:
            t = open(os.path.join(stage, rel), encoding='utf8').read()
        except UnicodeDecodeError:
            continue
        for pat in [r'<HOME>', r'via_project_(?:binh|chi|hai)',
                    r'GanNhanPanel_(?:binh|chi|hai)']:
            if re.search(pat, t):
                leaks.append((rel, pat))
    if leaks:
        for rel, pat in leaks[:20]:
            print(f'  LEAK {pat} in {rel}')
        raise SystemExit(f'{len(leaks)} de-identification leaks; archive not written')

    if os.path.exists(out):
        os.remove(out)
    subprocess.run(['zip', '-Xrq', '-9', os.path.abspath(out),
                    os.path.basename(stage)], cwd=tmp, check=True)
    shutil.rmtree(tmp)

    print(f'wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB zipped, '
          f'{total / 1e6:.1f} MB unpacked, {len(lines)} files)')
    for k, v in counts.items():
        print(f'  {k:12s} {v}')
    return out


if __name__ == '__main__':
    main()
