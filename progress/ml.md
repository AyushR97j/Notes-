RESUME HERE: ch18 done (97 items); next = ch19 Problems (>=60) with code/ml/c19_problems.py, then preface, read-through.

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
- [x] 01 Setup ★ (c01_setup.py: Bayes error, degree sweep, selection optimism)
- [x] 02 Maths ★ (c02_maths.py: SVD, projections, matrix-calculus checks, Bayes, r², CLT, MLE/MAP, convexity, KKT, info theory)
- [x] 03 Linear regression ★ (c03_linreg.py, c03_problems.py also holds ch1–3 problem numbers)
- [x] 04 Bias–variance and regularisation ★ (c04_biasvar.py, c04_problems.py)
- [x] 05 Logistic regression ★ (c05_logistic.py, c05_problems.py)
- [x] 06 Generative classifiers (c06_generative.py, c06_problems.py)
- [x] 07 kNN (c07_knn.py incl. a KD-tree with visit counter, c07_problems.py)
- [x] 08 SVM ★ (c08_svm.py: hand hard margin, dual QP, primal QP, kernels, SVR; c08_problems.py)
- [x] 09 Trees ★ (c09_trees.py, c09_problems.py)
- [x] 10 Ensembles ★ (c10_bagging.py, c10_boosting.py: AdaBoost/GBM/XGBoost checked vs libraries; c10_problems.py)
- [x] 11 Unsupervised ★ (c11_unsup.py, c11_problems.py)
- [x] 12 Semi-/self-supervised (c12_semisup.py, c12_problems.py)
- [x] 13 Evaluation ★ (c13_eval.py, c13_problems.py)
- [x] 14 Statistics (c14_stats.py, c14_problems.py)
- [x] 15 Data prep ★ (c15_dataprep.py incl. SMOTE + target-encoding leakage; c15_problems.py)
- [x] 16 Optimisation (c16_optim.py, c16_problems.py)
- [x] 17 Implement it (c17_implement.py: 11 implementations asserted vs sklearn + 7 in earlier chapters)
- [x] 18 Rapid-fire (97 items)
- [ ] 19 Problems (≥60)
- [ ] Read-through (pdftotext grep; overfull ≤ 2pt), fix list, final build

## Log
- Session 1: template + test book built; ML skeleton compiles.
- Lesson: never put [ ] inside a proof's optional title; bold-vector macros are brace-wrapped so \nabla_\x works; use \script{name} for file paths.
