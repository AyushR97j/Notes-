"""Chapter 13: confusion-matrix metrics, ROC/PR by hand, CV schemes, leakage, nested CV, AIC/BIC, calibration."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, fbeta_score, log_loss, matthews_corrcoef, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- binary confusion matrix
TP, FP, FN, TN = 40, 10, 20, 930
y_true = np.r_[np.ones(TP), np.zeros(FP), np.ones(FN), np.zeros(TN)]
y_pred = np.r_[np.ones(TP), np.ones(FP), np.zeros(FN), np.zeros(TN)]
P = TP / (TP + FP); R = TP / (TP + FN); F1 = 2 * P * R / (P + R)
F2 = 5 * P * R / (4 * P + R)
acc = (TP + TN) / len(y_true); spec = TN / (TN + FP); bal = (R + spec) / 2
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
for name, ours, ref in [("accuracy", acc, accuracy_score(y_true, y_pred)), ("precision", P, precision_score(y_true, y_pred)),
                        ("recall", R, recall_score(y_true, y_pred)), ("F1", F1, f1_score(y_true, y_pred)),
                        ("F2", F2, fbeta_score(y_true, y_pred, beta=2)), ("balanced accuracy", bal, balanced_accuracy_score(y_true, y_pred)),
                        ("MCC", mcc, matthews_corrcoef(y_true, y_pred))]:
    out.check(f"{name} by formula vs sklearn", ours, ref)
for k, v in dict(acc=acc, P=P, R=R, F1=F1, F2=F2, spec=spec, fpr=1 - spec, bal=bal, mcc=mcc).items():
    out.val(k, v, 4)
out.val("F05", 1.25 * P * R / (0.25 * P + R), 4)
out.val("always0_acc", (FP + TN) / len(y_true), 3)

# ---------------------------------------------------------------- macro / micro / weighted
C = np.array([[50, 3, 2], [10, 25, 5], [2, 3, 10]])  # rows: true class, cols: predicted
yt, yp = [], []
for i in range(3):
    for j in range(3):
        yt += [i] * C[i, j]; yp += [j] * C[i, j]
yt, yp = np.array(yt), np.array(yp)
prec_c = np.diag(C) / C.sum(0); rec_c = np.diag(C) / C.sum(1); f1_c = 2 * prec_c * rec_c / (prec_c + rec_c)
macro = f1_c.mean(); micro = np.trace(C) / C.sum(); weighted = (f1_c * C.sum(1)).sum() / C.sum()
out.check("macro F1 vs sklearn", macro, f1_score(yt, yp, average="macro"))
out.check("micro F1 (= accuracy) vs sklearn", micro, f1_score(yt, yp, average="micro"))
out.check("weighted F1 vs sklearn", weighted, f1_score(yt, yp, average="weighted"))
out.tex("mc_f1", ", ".join(f"{v:.3f}" for v in f1_c))
out.val("macro", macro, 4); out.val("micro", micro, 4); out.val("weighted", weighted, 4)

# ---------------------------------------------------------------- ROC and AUC by hand
s = np.array([0.95, 0.85, 0.80, 0.70, 0.55, 0.45, 0.40, 0.20])
l = np.array([1, 1, 0, 1, 0, 1, 0, 0])
rows, pts = [], [(0.0, 0.0)]
Pn, Nn = l.sum(), (1 - l).sum()
for t in s:
    pred = s >= t
    tpr = (pred & (l == 1)).sum() / Pn; fpr = (pred & (l == 0)).sum() / Nn
    prec = (pred & (l == 1)).sum() / pred.sum()
    rows.append(f"{t:.2f} & {int(l[s == t][0])} & {tpr:.2f} & {fpr:.2f} & {prec:.3f} \\\\")
    pts.append((fpr, tpr))
pts = np.array(pts)
auc_trap = np.trapezoid(pts[:, 1], pts[:, 0])
pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for a in s[l == 1] for b in s[l == 0]]
auc_mw = np.mean(pairs)
out.check("AUC by trapezoid vs sklearn roc_auc_score", auc_trap, roc_auc_score(l, s))
out.check("AUC = fraction of correctly ordered (pos, neg) pairs", auc_mw, roc_auc_score(l, s))
out.tex("roc_rows", "\n".join(rows))
out.val("auc", auc_trap, 4); out.val("auc_pairs", int(sum(pairs))), out.val("n_pairs", len(pairs))
out.dat("roc_hand", {"fpr": pts[:, 0], "tpr": pts[:, 1]})
# average precision by hand: sum over positives of precision at their rank / P
order = np.argsort(-s); hits = 0; ap = 0.0
for k, i in enumerate(order, 1):
    if l[i] == 1:
        hits += 1; ap += hits / k
ap /= Pn
out.check("average precision by hand vs sklearn", ap, average_precision_score(l, s))
out.val("ap", ap, 4)

# ---------------------------------------------------------------- ROC vs PR under imbalance
y = (rng.random(20000) < 0.01).astype(int)          # 1% positives
X = rng.normal(size=(20000, 2)) + 1.4 * y[:, None]    # positives shifted by (1.4, 1.4)
tr, te = slice(0, 10000), slice(10000, None)
m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
p = m.predict_proba(X[te])[:, 1]
fpr, tpr, _ = roc_curve(y[te], p); pr, rc, _ = precision_recall_curve(y[te], p)
k = np.linspace(0, len(fpr) - 1, min(len(fpr), 150)).astype(int)
out.dat("roc_imb", {"fpr": fpr[k], "tpr": tpr[k]})
k2 = np.linspace(0, len(pr) - 1, min(len(pr), 150)).astype(int)
out.dat("pr_imb", {"rec": rc[k2], "prec": pr[k2]})
out.val("imb_auc", roc_auc_score(y[te], p), 3); out.val("imb_ap", average_precision_score(y[te], p), 3)
out.val("imb_base", y[te].mean(), 3)
out.val("imb_ll", log_loss(y[te], p), 4); out.val("imb_brier", brier_score_loss(y[te], p), 4)
out.val("imb_ll_const", log_loss(y[te], np.full(len(p), y[tr].mean())), 4)

# ---------------------------------------------------------------- regression metrics
yr = np.array([3.0, 5.0, 2.5, 7.0, 10.0]); pr_ = np.array([2.5, 5.0, 4.0, 8.0, 9.0])
mae = np.mean(np.abs(yr - pr_)); mse = np.mean((yr - pr_) ** 2); mape = np.mean(np.abs((yr - pr_) / yr))
out.check("MAE", mae, mean_absolute_error(yr, pr_)); out.check("MSE", mse, mean_squared_error(yr, pr_))
out.check("MAPE", mape, mean_absolute_percentage_error(yr, pr_))
out.check("R^2", 1 - mse * len(yr) / ((yr - yr.mean()) ** 2).sum(), r2_score(yr, pr_))
out.val("mae", mae, 3); out.val("mse", mse, 3); out.val("rmse", np.sqrt(mse), 4); out.val("mape", 100 * mape, 2)
out.val("r2", r2_score(yr, pr_), 4)

# ---------------------------------------------------------------- CV schemes: time series leakage
t = np.arange(400.0)
series = 0.05 * t + np.sin(t / 10) + rng.normal(0, 0.3, 400)
Xt = t[:, None]
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rand = -cross_val_score(rf, Xt, series, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_absolute_error").mean()
fwd = -cross_val_score(rf, Xt, series, cv=TimeSeriesSplit(5), scoring="neg_mean_absolute_error").mean()
out.val("ts_random", rand, 3); out.val("ts_forward", fwd, 3)

# ---------------------------------------------------------------- leakage: feature selection before CV
Xn = rng.normal(size=(60, 5000)); yn = rng.integers(0, 2, 60)
sel = SelectKBest(f_classif, k=20).fit(Xn, yn)
leaky = cross_val_score(LogisticRegression(max_iter=1000), sel.transform(Xn), yn, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
proper = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)), Xn, yn,
                         cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
out.val("leak_acc", leaky, 3); out.val("proper_acc", proper, 3)

# ---------------------------------------------------------------- nested CV vs the inner best score
Xbig, ybig = make_classification(n_samples=20120, n_features=20, n_informative=3, flip_y=0.2, random_state=1)
Xs, ys = Xbig[:120], ybig[:120]          # small training set; the other 20000 rows are a fresh test set
grid = {"C": np.logspace(-2, 3, 6), "gamma": np.logspace(-4, 1, 6)}
inner = StratifiedKFold(5, shuffle=True, random_state=0); outer = StratifiedKFold(5, shuffle=True, random_state=1)
gsb, nst, hold = [], [], []
for rep in range(6):
    idx = slice(120 * rep, 120 * (rep + 1))
    Xs, ys = Xbig[idx], ybig[idx]
    gs = GridSearchCV(SVC(), grid, cv=inner).fit(Xs, ys)
    gsb.append(gs.best_score_)
    nst.append(cross_val_score(GridSearchCV(SVC(), grid, cv=inner), Xs, ys, cv=outer).mean())
    hold.append(gs.score(Xbig[-10000:], ybig[-10000:]))
out.val("gs_best", np.mean(gsb), 3); out.val("nested", np.mean(nst), 3); out.val("holdout", np.mean(hold), 3)
out.val("grid_size", 36)

# ---------------------------------------------------------------- AIC / BIC for polynomial degree
xs = np.sort(rng.uniform(-1, 1, 60)); ys2 = 1 - 2 * xs + 1.5 * xs ** 3 + rng.normal(0, 0.3, 60)
n = len(xs); rows = []
best = {}
for d in range(1, 9):
    A = PolynomialFeatures(d).fit_transform(xs[:, None])
    res = ys2 - LinearRegression(fit_intercept=False).fit(A, ys2).predict(A)
    s2 = np.mean(res ** 2)
    ll = -n / 2 * (np.log(2 * np.pi * s2) + 1)
    kpar = d + 2           # d+1 coefficients + noise variance
    aic, bic = 2 * kpar - 2 * ll, kpar * np.log(n) - 2 * ll
    best[d] = (aic, bic)
    rows.append(f"{d} & {kpar} & {ll:.2f} & {aic:.2f} & {bic:.2f} \\\\")
out.tex("ic_rows", "\n".join(rows))
out.val("aic_best", min(best, key=lambda d: best[d][0])); out.val("bic_best", min(best, key=lambda d: best[d][1]))
out.val("ln_n", np.log(n), 3)
# BIC of a GMM vs the formula
Xg = np.r_[rng.normal(0, 1, (200, 2)), rng.normal(4, 1, (200, 2))]
g = GaussianMixture(2, random_state=0).fit(Xg)
kg = 2 * 2 + 2 * 3 + 1                  # means + full covariances + free weights
out.check("GaussianMixture.bic = -2 log L + k ln n", g.bic(Xg), -2 * g.score(Xg) * len(Xg) + kg * np.log(len(Xg)))

# ---------------------------------------------------------------- calibration
Xc, yc = make_classification(n_samples=6000, n_features=20, n_informative=5, n_redundant=10, random_state=3)
a, b = slice(0, 4000), slice(4000, None)
nb = GaussianNB().fit(Xc[a], yc[a])
cal = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5).fit(Xc[a], yc[a])
pn, pc = nb.predict_proba(Xc[b])[:, 1], cal.predict_proba(Xc[b])[:, 1]
for name, pp in [("nb", pn), ("cal", pc)]:
    fr, mp = calibration_curve(yc[b], pp, n_bins=10, strategy="quantile")
    out.dat(f"rel_{name}", {"mp": mp, "fr": fr})
    out.val(f"brier_{name}", brier_score_loss(yc[b], pp), 4)
    out.val(f"ll_{name}", log_loss(yc[b], pp), 4)
    # expected calibration error with equal-width bins
    bins = np.minimum((pp * 10).astype(int), 9)
    ece = sum(np.abs(yc[b][bins == k].mean() - pp[bins == k].mean()) * np.mean(bins == k) for k in range(10) if np.any(bins == k))
    out.val(f"ece_{name}", ece, 4)
out.val("auc_nb", roc_auc_score(yc[b], pn), 4); out.val("auc_cal", roc_auc_score(yc[b], pc), 4)
