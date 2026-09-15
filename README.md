# Code and data for the open-vocabulary weed-detection audit

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
