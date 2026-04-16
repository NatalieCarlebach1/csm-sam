# CSM-SAM NeurIPS paper

LaTeX source for the CSM-SAM submission.

## Layout

```
paper/
  main.tex               top-level document (abstract + \input'd sections)
  references.bib         BibTeX entries for every method cited in results
  Makefile               make / make clean / make view
  sections/
    intro.tex
    related.tex
    method.tex
    results.tex          main deliverable (fully populated, numbers TBD)
    ablations.tex
    discussion.tex
    conclusion.tex
  tables/
    hnts_mrg.tex         HNTS-MRG 2024 Task 2 (primary)
    brats_gli.tex        BraTS-GLI 2024 post-treatment (secondary)
    levir_cd.tex         LEVIR-CD binary building CD
    s2looking.tex        S2Looking binary building CD
    second.tex           SECOND semantic CD
    xbd.tex              xBD / xView2 damage CD
    ablation.tex         Single-axis ablation on HNTS-MRG
```

## Compiling

```bash
cd paper
make        # pdflatex + bibtex + 2x pdflatex
make view   # open the PDF
make clean  # remove build artifacts
```

## Swapping in the NeurIPS 2026 style file

The NeurIPS 2026 style file is not public yet. `main.tex` currently uses
`\documentclass{article}` with a `% TODO` comment at the top. When the
2026 style drops:

1. Drop `neurips_2026.sty` into `paper/`.
2. In `main.tex` replace

   ```latex
   \documentclass{article}
   ```

   with the NeurIPS preamble, typically

   ```latex
   \documentclass{article}
   \usepackage[final]{neurips_2026}  % or \usepackage{neurips_2026}
   ```

3. Re-run `make`.

## Where to paste trained numbers

Numbers are marked with `--` or `TBD-$\pm$--` in every table cell that
depends on a training run, and each such line is tagged with a
`% to fill` or `% filled in after training` comment for grepability.
After a training run, edit the corresponding `tables/*.tex` file and
replace the placeholder with the measured value. The prose in
`sections/results.tex` references each table by `\ref{tab:*}` and does
not need to change when numbers land.

Ablation rows live in `tables/ablation.tex`; the CSM-SAM row in each
benchmark table is labelled `\textbf{CSM-SAM (Ours)}` for search.
