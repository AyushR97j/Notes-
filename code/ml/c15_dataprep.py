"""Chapter 15: scaling, encodings and target-encoding leakage, missingness, selection, SMOTE leakage, TF-IDF, MF."""
import warnings

import numpy as np
from sklearn.datasets import load_wine, make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, TargetEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from common import setup

out = setup(__file__)
warnings.filterwarnings("ignore")   # unscaled logistic regression hits max_iter: that is the point
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- scalers on one column with an outlier
v = np.array([10.0, 12, 11, 13, 12, 100])[:, None]
for name, sc in [("std", StandardScaler()), ("mm", MinMaxScaler()), ("rob", RobustScaler())]:
    out.tex(f"sc_{name}", ", ".join(f"{t:.2f}" for t in sc.fit_transform(v).ravel()))
out.check("standard scaler by formula (population sd)", (v.ravel() - v.mean()) / v.std(), StandardScaler().fit_transform(v).ravel())
q1, q3 = np.percentile(v, [25, 75])
out.val("iqr_q1", q1, 2); out.val("iqr_q3", q3, 2); out.val("iqr_hi", q3 + 1.5 * (q3 - q1), 2)

# ---------------------------------------------------------------- which models care about scaling (wine)
X, y = load_wine(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows = []
for name, model in [("$k$-NN ($k=5$)", KNeighborsClassifier()), ("SVM (RBF)", SVC()),
                    ("logistic regression (L2)", LogisticRegression(max_iter=300)),
                    ("PCA(2) + logistic", make_pipeline(PCA(2), LogisticRegression(max_iter=300))),
                    ("decision tree", DecisionTreeClassifier(random_state=0)),
                    ("random forest", RandomForestClassifier(random_state=0))]:
    raw = cross_val_score(model, X, y, cv=cv).mean()
    scl = cross_val_score(make_pipeline(StandardScaler(), model), X, y, cv=cv).mean()
    rows.append(f"{name} & {raw:.3f} & {scl:.3f} \\\\")
out.tex("scale_rows", "\n".join(rows))

# ---------------------------------------------------------------- target encoding: smoothing by hand and leakage
cats = np.array(["a"] * 3 + ["b"] * 20 + ["c"] * 77)
yy = np.r_[np.ones(3), (rng.random(20) < 0.3).astype(float), (rng.random(77) < 0.3).astype(float)]
gmean = yy.mean(); m = 10.0
enc_a = (yy[cats == "a"].sum() + m * gmean) / ((cats == "a").sum() + m)
te = TargetEncoder(smooth=m, target_type="continuous").fit(cats[:, None], yy)
out.check("smoothed target encoding of a 3-row category vs sklearn TargetEncoder", enc_a, te.encodings_[0][0])
out.val("te_gmean", gmean, 3); out.val("te_a", enc_a, 4); out.val("te_m", m, 0)
# leakage: a pure-noise high-cardinality ID-like feature
n = 4000
idc = rng.integers(0, 1500, n).astype(str)        # ~2.7 rows per level
xs = rng.normal(size=n)
yb = (rng.random(n) < 1 / (1 + np.exp(-xs))).astype(int)   # only xs matters
tr, te_ = train_test_split(np.arange(n), test_size=0.5, random_state=0)
# naive: encode with full-train means (including each row's own label)
lev_mean = {}
for lev in np.unique(idc[tr]):
    msk = idc[tr] == lev
    lev_mean[lev] = yb[tr][msk].mean()
naive_tr = np.array([lev_mean[l] for l in idc[tr]])
naive_te = np.array([lev_mean.get(l, yb[tr].mean()) for l in idc[te_]])
lr = LogisticRegression().fit(np.c_[xs[tr], naive_tr], yb[tr])
out.val("leak_tr_auc", roc_auc_score(yb[tr], lr.predict_proba(np.c_[xs[tr], naive_tr])[:, 1]), 3)
out.val("leak_te_auc", roc_auc_score(yb[te_], lr.predict_proba(np.c_[xs[te_], naive_te])[:, 1]), 3)
out.val("leak_coef", lr.coef_[0][1], 2)
# cross-fitted (out-of-fold) encoding
tenc = TargetEncoder(target_type="binary", cv=5, random_state=0)
oof_tr = tenc.fit_transform(idc[tr][:, None], yb[tr]).ravel()
oof_te = tenc.transform(idc[te_][:, None]).ravel()
lr2 = LogisticRegression().fit(np.c_[xs[tr], oof_tr], yb[tr])
out.val("oof_tr_auc", roc_auc_score(yb[tr], lr2.predict_proba(np.c_[xs[tr], oof_tr])[:, 1]), 3)
out.val("oof_te_auc", roc_auc_score(yb[te_], lr2.predict_proba(np.c_[xs[te_], oof_te])[:, 1]), 3)
out.val("oof_coef", lr2.coef_[0][1], 2)
lr3 = LogisticRegression().fit(xs[tr][:, None], yb[tr])
out.val("base_te_auc", roc_auc_score(yb[te_], lr3.predict_proba(xs[te_][:, None])[:, 1]), 3)

# ---------------------------------------------------------------- informative missingness
n = 3000
inc = rng.normal(50, 15, n)
miss = rng.random(n) < 1 / (1 + np.exp(-(inc - 60) / 5))      # high earners skip the question
yb2 = (rng.random(n) < 1 / (1 + np.exp(-(inc - 50) / 10))).astype(int)
obs = np.where(miss, np.nan, inc)
mean_imp = np.where(miss, np.nanmean(obs), obs)
Xa = mean_imp[:, None]; Xb = np.c_[mean_imp, miss]
tr, te_ = train_test_split(np.arange(n), test_size=0.5, random_state=0)
a1 = roc_auc_score(yb2[te_], LogisticRegression().fit(Xa[tr], yb2[tr]).predict_proba(Xa[te_])[:, 1])
a2 = roc_auc_score(yb2[te_], LogisticRegression().fit(Xb[tr], yb2[tr]).predict_proba(Xb[te_])[:, 1])
out.val("miss_frac", miss.mean(), 3); out.val("miss_auc_mean", a1, 3); out.val("miss_auc_ind", a2, 3)
out.val("miss_sd_before", inc.std(), 2); out.val("miss_sd_after", mean_imp.std(), 2)

# ---------------------------------------------------------------- feature selection: filter, wrapper, embedded
Xf, yf = make_classification(n_samples=600, n_features=50, n_informative=5, n_redundant=0, shuffle=False, random_state=1)
true = set(range(5))
mi = SelectKBest(mutual_info_classif, k=5).fit(Xf, yf)
rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=5).fit(Xf, yf)
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.05).fit(Xf, yf)
for name, sel in [("mi", set(np.where(mi.get_support())[0])), ("rfe", set(np.where(rfe.support_)[0])),
                  ("l1", set(np.where(np.abs(l1.coef_[0]) > 1e-8)[0]))]:
    out.val(f"fs_{name}_hit", len(sel & true)); out.val(f"fs_{name}_n", len(sel))

# ---------------------------------------------------------------- SMOTE (from scratch) and its leakage
# [[smote]]
def smote(X, y, k=5, ratio=1.0, seed=0):
    """Add synthetic minority points on segments between a minority point and one of its k minority neighbours."""
    r = np.random.default_rng(seed)
    Xm = X[y == 1]
    n_new = int(ratio * ((y == 0).sum() - len(Xm)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    i = r.integers(0, len(Xm), n_new)
    j = nn[i, r.integers(0, k, n_new)]
    lam = r.random((n_new, 1))
    Xs = Xm[i] + lam * (Xm[j] - Xm[i])
    return np.vstack([X, Xs]), np.r_[y, np.ones(n_new, int)]
# [[/smote]]
Xi, yi = make_classification(n_samples=1000, n_features=10, n_informative=3, weights=[0.95, 0.05], flip_y=0.02, random_state=3)
Xs_, ys_ = smote(Xi, yi)
out.check("SMOTE balances the classes", ys_.mean(), 0.5)
wrong = cross_val_score(KNeighborsClassifier(3), Xs_, ys_, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1").mean()
right = []
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xi, yi):
    Xa_, ya_ = smote(Xi[a], yi[a])
    pr = KNeighborsClassifier(3).fit(Xa_, ya_).predict(Xi[b])
    tp = ((pr == 1) & (yi[b] == 1)).sum(); fp = ((pr == 1) & (yi[b] == 0)).sum(); fn = ((pr == 0) & (yi[b] == 1)).sum()
    right.append(2 * tp / (2 * tp + fp + fn))
out.val("smote_wrong", wrong, 3); out.val("smote_right", np.mean(right), 3)

# ---------------------------------------------------------------- TF-IDF by hand vs sklearn
docs = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs"]
cvz = CountVectorizer().fit(docs)
C = cvz.transform(docs).toarray(); vocab = list(cvz.get_feature_names_out())
N = len(docs); df = (C > 0).sum(0)
idf_s = np.log((1 + N) / (1 + df)) + 1          # sklearn default (smooth_idf=True)
idf_t = np.log(N / df)                          # textbook
tfv = TfidfVectorizer(norm=None).fit(docs)
out.check("smoothed idf by hand vs TfidfVectorizer.idf_", idf_s, tfv.idf_)
W = C * idf_s; Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
out.check("L2-normalised tf-idf matrix vs TfidfVectorizer", Wn, TfidfVectorizer().fit_transform(docs).toarray())
for w in ("the", "sat", "cat"):
    j = vocab.index(w)
    out.val(f"df_{w}", int(df[j])); out.val(f"idft_{w}", idf_t[j], 4); out.val(f"idfs_{w}", idf_s[j], 4)
    out.val(f"tfidf0_{w}", W[0, j], 4)
out.val("idf_log10_cat", np.log10(3 / 1), 4)

# ---------------------------------------------------------------- matrix factorisation by ALS
R = np.array([[5, 4, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [0, 1, 5, 4], [0, 0, 4, 0]], float)
M = R > 0
k, lam = 2, 0.1
r2 = np.random.default_rng(0)
U = r2.normal(0, 0.5, (5, k)); V = r2.normal(0, 0.5, (4, k))
rmses, objs = [], []
for it in range(30):
    for u in range(5):
        Vi = V[M[u]]; U[u] = np.linalg.solve(Vi.T @ Vi + lam * np.eye(k), Vi.T @ R[u, M[u]])
    for i in range(4):
        Ui = U[M[:, i]]; V[i] = np.linalg.solve(Ui.T @ Ui + lam * np.eye(k), Ui.T @ R[M[:, i], i])
    rmses.append(np.sqrt(np.mean((R - U @ V.T)[M] ** 2)))
    objs.append(((R - U @ V.T)[M] ** 2).sum() + lam * ((U ** 2).sum() + (V ** 2).sum()))
u = 1; Vi = V[M[u]]
rid = Ridge(alpha=lam, fit_intercept=False).fit(Vi, R[u, M[u]]).coef_
out.check("one ALS user update = ridge regression on that user's ratings", np.linalg.solve(Vi.T @ Vi + lam * np.eye(k), Vi.T @ R[u, M[u]]), rid)
out.check("ALS regularised objective never increases", float(np.all(np.diff(objs) <= 1e-10)), 1.0)
out.val("als_rmse", rmses[-1], 3)
P = U @ V.T
out.val("als_pred_02", P[0, 2], 2); out.val("als_pred_11", P[1, 1], 2)
