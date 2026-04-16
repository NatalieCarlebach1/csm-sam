# CSM-SAM NeurIPS paper

LaTeX source for the CSM-SAM submission.

## Layout

```
paper/
  main.tex                  top-level document (abstract + \input'd sections)
  neurips_2024.sty          NeurIPS 2024 style file (placeholder until 2026 drops)
  references.bib            BibTeX entries for every method cited
  Makefile                  make / make clean / make view
  sections/
    intro.tex
    related.tex             6 subsections: SAM family, longitudinal medical,
                            BraTS, CD remote sensing, semantic CD, positioning
    method.tex
    experiments.tex         4 subsections: datasets, baselines, metrics, impl
    results.tex             opening paragraph + per-dataset tables + takeaways
                            + cross-dataset generalization subsection
    ablations.tex
    discussion.tex
    conclusion.tex
  tables/
    hnts_mrg.tex            HNTS-MRG 2024 Task 2 (primary)
    brats_gli.tex           BraTS-GLI 2024 post-treatment (secondary)
    levir_cd.tex            LEVIR-CD binary building CD
    s2looking.tex           S2Looking binary building CD
    second.tex              SECOND semantic CD
    xbd.tex                 xBD / xView2 damage CD
    ablation.tex            Single-axis ablation on HNTS-MRG
```

## Compiling

```bash
cd paper
make        # pdflatex + bibtex + 2x pdflatex
make view   # open the PDF
make clean  # remove build artifacts
```

## NeurIPS format

`main.tex` uses `\usepackage[preprint]{neurips_2024}` as a placeholder until `neurips_2026.sty` is released. The style file `neurips_2024.sty` ships with this directory (fetched from `https://media.neurips.cc/Conferences/NeurIPS2024/Styles.zip`).

Three mode switches:

```latex
\usepackage[preprint]{neurips_2024}  % DRAFT — author block shown, no page numbers
\usepackage{neurips_2024}            % SUBMISSION — anonymised, page numbers removed
\usepackage[final]{neurips_2024}     % CAMERA-READY — author block shown, paginated
```

When `neurips_2026.sty` is released:
1. Download it into `paper/`.
2. Replace `neurips_2024` with `neurips_2026` in `main.tex`.
3. Re-run `make`.

Citations are numeric/compressed via `\bibliographystyle{plainnat}` + `\PassOptionsToPackage{numbers,compress}{natbib}`.

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
