# Data licensing and what is not in this archive

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
