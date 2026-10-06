RESUME HERE: skeleton compiles (63 pp of stubs); next = write chapter 01-setup with code/ml/c01_*.py.

# Classical Machine Learning — progress log

Author on title page / running head / PDF metadata: **Philospher** (user's spelling, kept verbatim).

## Decisions (made without asking, per the brief)
- Template: `latex/preamble.tex`, adapted from Napkin's `tex/preamble.tex` + `tex/macros.tex`
  (GPL v3 notice kept in the file header, `latex/NOTICE.md`, `latex/COPYING-GPL-3.txt`).
  ML/DL macros are in the clearly marked block at the bottom; book-only macros in `latex/ml-macros.tex`.
- Chili icons are drawn in TikZ (Napkin uses a PNG); no raster images anywhere.
- Every generated number: `code/ml/*.py` → `gen/ml/<script>.tex` with `\gvset{script/key}{value}`;
  the text uses `\gv{script/key}` (undefined → red `??[key]`, caught by the read-through grep).
  Plot data → `gen/ml/data/*.dat`; captured output → `gen/ml/out/*.txt` (`\outputfile`);
  code shown in the book → `# [[name]] … # [[/name]]` regions extracted to `gen/ml/snip/` (`\codefile`).
- Library cross-checks use `out.check(...)` = `assert np.allclose` + print of both numbers,
  logged to `gen/ml/checks.log` (committed). `build.sh` fails on any mismatch.
- Generated files are committed, so the PDF also builds with `SKIP_CODE=1`.
- PyTorch: download.pytorch.org is blocked by the egress proxy, so the PyPI wheel (CUDA build)
  is used; all scripts force CPU (`CUDA_VISIBLE_DEVICES=""`).
- Problem numbering: chapter-end problems `3A, 3B, …` (Napkin); the big problem chapter uses `P1, P2, …`.
- Hints and solutions are collected into two appendices (Napkin style), with links both ways.
- Bibliography contains only entries whose details I am sure of (`latex/refs-ml.bib`).

## Gap plan (additions beyond the brief's chapter list)
- Ch1: no-free-lunch in one paragraph; data-generating distribution picture; empirical vs true risk numerics.
- Ch2: information theory block (entropy, cross-entropy, KL) because trees, logistic regression and
  naive Bayes all need it; numerically stable log-sum-exp; condition number.
- Ch3: QR vs normal equations (numerical stability), Huber loss / robust regression mention,
  heteroscedasticity and residual plots, coefficient interpretation questions.
- Ch4: effective degrees of freedom of ridge; standardisation before penalising; lasso path figure.
- Ch5: decision threshold vs cost; calibration pointer; separable data → infinite weights.
- Ch6: why NB probabilities are over-confident; log-space computation; Gaussian NB variance smoothing.
- Ch7: weighted kNN, scaling sensitivity, kNN regression.
- Ch8: hinge vs log-loss comparison figure; Platt scaling; why SVMs don't give probabilities.
- Ch9: ID3 / C4.5 gain ratio / CART naming; why trees don't need scaling; instability demo.
- Ch10: OOB error vs CV; why boosting overfits noise; learning-rate × n_estimators trade.
- Ch11: silhouette by hand; k-means as coordinate descent on inertia (why it terminates).
- Ch13: cost-sensitive thresholds; AUC = Mann–Whitney probability; a fraud-detection evaluation case.
- Ch14: bootstrap confidence interval; A/B test sample-size arithmetic.
- Ch15: feature hashing; leakage via scaling before split; pipeline code.
- Ch16: condition number ⇔ zig-zag; SGD noise; convergence-rate table.

## Chapter checklist
- [ ] 00 Preface (how to use, what placements ask, sources, reproducibility)
- [ ] 01 Setup ★
- [ ] 02 Maths ★
- [ ] 03 Linear regression ★
- [ ] 04 Bias–variance and regularisation ★
- [ ] 05 Logistic regression ★
- [ ] 06 Generative classifiers
- [ ] 07 kNN
- [ ] 08 SVM ★
- [ ] 09 Trees ★
- [ ] 10 Ensembles ★
- [ ] 11 Unsupervised ★
- [ ] 12 Semi-/self-supervised
- [ ] 13 Evaluation ★
- [ ] 14 Statistics
- [ ] 15 Data prep ★
- [ ] 16 Optimisation
- [ ] 17 Implement it
- [ ] 18 Rapid-fire (80–100)
- [ ] 19 Problems (≥60)
- [ ] Read-through (pdftotext grep; overfull ≤ 2pt), fix list, final build

## Log
- Session 1: template + test book built; ML skeleton compiles.
