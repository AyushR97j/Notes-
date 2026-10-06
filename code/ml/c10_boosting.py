"""Chapter 10b: AdaBoost by hand, gradient boosting from scratch, XGBoost leaf weights and gains."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.special import expit
from sklearn.datasets import load_breast_cancer, make_classification, make_friedman1
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- AdaBoost by hand (1-D stumps)
x = np.arange(1, 11, dtype=float)
y = np.array([1, 1, -1, -1, -1, 1, 1, 1, -1, -1])

def best_stump(x, y, w):
    best = None
    for t in np.r_[x[0] - 0.5, (x[:-1] + x[1:]) / 2, x[-1] + 0.5]:
        for s in (1, -1):                     # predict s if x <= t else -s
            pred = np.where(x <= t, s, -s)
            err = w[pred != y].sum()
            if best is None or err < best[0] - 1e-12:
                best = (err, t, s, pred)
    return best

w = np.full(10, 0.1)
F = np.zeros(10)
rows, eps_list, alpha_list = [], [], []
for m in range(3):
    err, t, s, pred = best_stump(x, y, w)
    alpha = 0.5 * np.log((1 - err) / err)
    F += alpha * pred
    rule = f"$x\\le{t:g}\\Rightarrow{'+' if s == 1 else '-'}1$"
    wrong = ", ".join(str(int(v)) for v in x[pred != y])
    rows.append(f"{m+1} & {rule} & {wrong} & {err:.4f} & {alpha:.4f} & {np.mean(np.sign(F) == y):.1f} \\\\")
    eps_list.append(err); alpha_list.append(alpha)
    w = w * np.exp(-alpha * y * pred)
    if m == 0:
        out.tex("w1_wrong", f"{w[pred != y][0] / w.sum():.4f}")
        out.tex("w1_right", f"{w[pred == y][0] / w.sum():.4f}")
        out.val("Z1", w.sum(), 4)
    w = w / w.sum()
out.tex("ada_rows", "\n".join(rows))
out.val("ada_alpha1", alpha_list[0], 4); out.val("ada_eps1", eps_list[0], 4)
# sklearn's SAMME uses estimator weight log((1-err)/err) = 2 * alpha for two classes
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=3,
                         learning_rate=1.0, random_state=0).fit(x[:, None], y)
out.check("AdaBoost weighted errors vs sklearn estimator_errors_", eps_list, ada.estimator_errors_)
out.check("AdaBoost alphas (x2) vs sklearn estimator_weights_", 2 * np.array(alpha_list), ada.estimator_weights_)
out.check("AdaBoost training predictions vs sklearn", np.sign(F), ada.predict(x[:, None]))

# ---------------------------------------------------------------- gradient boosting by hand (4 points)
xg = np.array([1.0, 2, 3, 4]); yg = np.array([1.0, 3, 6, 8])
nu = 0.5
F0 = yg.mean(); Fh = np.full(4, F0); hand = []
for m in range(2):
    r = yg - Fh
    st = DecisionTreeRegressor(max_depth=1).fit(xg[:, None], r)
    upd = st.predict(xg[:, None])
    hand.append((r.copy(), st.tree_.threshold[0], upd.copy()))
    Fh = Fh + nu * upd
out.val("gb_F0", F0, 2)
out.tex("gb_r1", ", ".join(f"{v:g}" for v in hand[0][0]))
out.val("gb_t1", hand[0][1], 1)
out.tex("gb_u1", ", ".join(f"{v:g}" for v in hand[0][2]))
out.tex("gb_r2", ", ".join(f"{v:g}" for v in hand[1][0]))
out.tex("gb_u2", ", ".join(f"{v:g}" for v in hand[1][2]))
out.tex("gb_F2", ", ".join(f"{v:g}" for v in Fh))
gbr = GradientBoostingRegressor(n_estimators=2, learning_rate=nu, max_depth=1).fit(xg[:, None], yg)
out.check("2-round boosting by hand vs sklearn GradientBoostingRegressor", Fh, gbr.predict(xg[:, None]))

# ---------------------------------------------------------------- gradient boosting from scratch (regression and log-loss)
# [[gbr]]
def gb_regress(X, y, M=100, nu=0.1, depth=3, seed=0):
    rs = np.random.RandomState(seed)               # one RNG shared by all trees, as sklearn does
    X = X.astype(np.float32)                       # sklearn grows its trees on float32 inputs
    F0 = y.mean()
    F = np.full(len(y), F0)
    trees = []
    for _ in range(M):
        r = y - F                                  # negative gradient of 1/2 (y - F)^2
        t = DecisionTreeRegressor(max_depth=depth, random_state=rs).fit(X, r)
        F += nu * t.predict(X)
        trees.append(t)
    return lambda Z: F0 + nu * sum(t.predict(Z.astype(np.float32)) for t in trees)
# [[/gbr]]
Xf, yf = make_friedman1(n_samples=500, noise=1.0, random_state=0)
Xa, Xb, ya, yb = train_test_split(Xf, yf, random_state=0)
ours = gb_regress(Xa, ya)
sk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                               random_state=0).fit(Xa, ya)
out.check("gradient boosting regressor from scratch vs sklearn (test predictions)", ours(Xb), sk.predict(Xb), atol=1e-8)
out.val("gbr_mse", np.mean((ours(Xb) - yb) ** 2), 3)

# [[gbc]]
def gb_classify(X, y, M=100, nu=0.1, depth=3, seed=0, min_leaf=20):
    """Binary log-loss boosting; each leaf takes one Newton step: sum(y - p) / sum(p (1 - p))."""
    rs = np.random.RandomState(seed)
    X = X.astype(np.float32)
    F0 = np.log(y.mean() / (1 - y.mean()))
    F = np.full(len(y), F0)
    stages = []
    for _ in range(M):
        p = expit(F)
        r = y - p                                  # negative gradient of the log-loss
        t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=rs).fit(X, r)
        leaf = t.apply(X)
        vals = {}
        for l in np.unique(leaf):
            m = leaf == l
            vals[l] = r[m].sum() / (p[m] * (1 - p[m])).sum()
        F += nu * np.array([vals[l] for l in leaf])
        stages.append((t, vals))
    def decision(Z):
        out_ = np.full(len(Z), F0)
        for t, vals in stages:
            out_ += nu * np.array([vals[l] for l in t.apply(Z.astype(np.float32))])
        return out_
    return decision
# [[/gbc]]
Xc, yc = make_classification(n_samples=600, n_features=8, n_informative=5, random_state=0)
Xca, Xcb, yca, ycb = train_test_split(Xc, yc, random_state=0)
dec = gb_classify(Xca, yca)
skc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=20,
                                 random_state=0).fit(Xca, yca)
out.check("log-loss boosting from scratch vs sklearn GradientBoostingClassifier (decision function)",
          dec(Xcb), skc.decision_function(Xcb), atol=1e-6)
out.val("gbc_acc", np.mean((dec(Xcb) > 0) == ycb), 3)

# ---------------------------------------------------------------- XGBoost: one tree by hand vs the library
xx = np.array([1.0, 2, 3, 4, 5, 6]); yy = np.array([1.0, 2, 2, 6, 7, 9])
lam, base = 1.0, 0.5
g = base - yy; h = np.ones(6)                      # squared error: g = yhat - y, h = 1
G, Hs = g.sum(), h.sum()
splits = []
for t in (xx[:-1] + xx[1:]) / 2:
    L = xx < t
    gain = (g[L].sum() ** 2 / (h[L].sum() + lam) + g[~L].sum() ** 2 / (h[~L].sum() + lam) - G ** 2 / (Hs + lam))
    splits.append((gain, t, L))
splits.sort(key=lambda z: -z[0])
gain, t_best, L = splits[0]
wl, wr = -g[L].sum() / (h[L].sum() + lam), -g[~L].sum() / (h[~L].sum() + lam)
out.tex("xg_g", ", ".join(f"{v:g}" for v in g))
out.val("xg_G", G, 1); out.val("xg_t", t_best, 1)
out.val("xg_GL", g[L].sum(), 1); out.val("xg_GR", g[~L].sum(), 1)
out.val("xg_HL", h[L].sum(), 0); out.val("xg_HR", h[~L].sum(), 0)
out.val("xg_wl", wl, 4); out.val("xg_wr", wr, 4)
out.val("xg_gain", gain, 4); out.val("xg_half", gain / 2, 4)
out.val("xg_w_root", -G / (Hs + lam), 4)
out.tex("xg_all", ", ".join(f"{tt:g}: {gg:.2f}" for gg, tt, _ in sorted(splits, key=lambda z: z[1])))
bst = xgb.train({"max_depth": 1, "eta": 1.0, "lambda": lam, "base_score": base, "min_child_weight": 0,
                 "tree_method": "exact", "objective": "reg:squarederror"}, xgb.DMatrix(xx[:, None], label=yy),
                num_boost_round=1)
df = bst.trees_to_dataframe()
root = df[df.Node == 0].iloc[0]
leaves = df[df.Feature == "Leaf"].sort_values("Node")
out.check("XGBoost split threshold = our best midpoint", root.Split, t_best)
out.check("XGBoost leaf weights = -G/(H+lambda)", leaves.Gain.values, [wl, wr])
out.check("XGBoost reported Gain = G_L^2/(H_L+l) + G_R^2/(H_R+l) - G^2/(H+l) (no 1/2)", root.Gain, gain, rtol=1e-5)
out.check("XGBoost Cover = sum of hessians", root.Cover, Hs)
pred_lib = bst.predict(xgb.DMatrix(xx[:, None]))
out.check("XGBoost predictions = base + leaf weight", pred_lib, base + np.where(L, wl, wr), atol=1e-6)

# logistic objective: g = p - y, h = p(1 - p)
yb2 = np.array([0, 1, 0, 0, 1, 1.0])
p0 = 0.5
g2 = p0 - yb2; h2 = np.full(6, p0 * (1 - p0))
best2 = None
for t in (xx[:-1] + xx[1:]) / 2:
    L2 = xx < t
    gn = g2[L2].sum() ** 2 / (h2[L2].sum() + lam) + g2[~L2].sum() ** 2 / (h2[~L2].sum() + lam) - g2.sum() ** 2 / (h2.sum() + lam)
    if best2 is None or gn > best2[0] + 1e-12:
        best2 = (gn, t, L2)
gn, t2, L2 = best2
w2l, w2r = -g2[L2].sum() / (h2[L2].sum() + lam), -g2[~L2].sum() / (h2[~L2].sum() + lam)
bst2 = xgb.train({"max_depth": 1, "eta": 1.0, "lambda": lam, "base_score": p0, "min_child_weight": 0,
                  "tree_method": "exact", "objective": "binary:logistic"}, xgb.DMatrix(xx[:, None], label=yb2),
                 num_boost_round=1)
d2 = bst2.trees_to_dataframe()
out.check("XGBoost logistic leaf weights = -G/(H+lambda)", d2[d2.Feature == "Leaf"].sort_values("Node").Gain.values, [w2l, w2r])
out.check("XGBoost logistic predicted probabilities = sigmoid(0 + w)", bst2.predict(xgb.DMatrix(xx[:, None])),
          expit(np.where(L2, w2l, w2r)), atol=1e-6)
out.val("xl_t", t2, 1); out.val("xl_gain", gn, 4)
out.val("xl_wl", w2l, 4); out.val("xl_wr", w2r, 4)
out.val("xl_GL", g2[L2].sum(), 1); out.val("xl_GR", g2[~L2].sum(), 1)
out.val("xl_HL", h2[L2].sum(), 2); out.val("xl_HR", h2[~L2].sum(), 2)
out.val("xl_pl", expit(w2l), 4); out.val("xl_pr", expit(w2r), 4)
# gamma prunes splits whose gain (as XGBoost reports it, without the 1/2) is below gamma... check behaviour
for gamma in (gn - 0.01, gn + 0.01):
    b3 = xgb.train({"max_depth": 1, "eta": 1.0, "lambda": lam, "base_score": p0, "min_child_weight": 0,
                    "gamma": gamma, "tree_method": "exact", "objective": "binary:logistic"},
                   xgb.DMatrix(xx[:, None], label=yb2), num_boost_round=1)
    n_leaves = (b3.trees_to_dataframe().Feature == "Leaf").sum()
    out.val(f"gamma_{'below' if gamma < gn else 'above'}_leaves", int(n_leaves))
out.check("gamma just below the gain keeps the split (2 leaves)", float(n_leaves == 1), 1.0)

# ---------------------------------------------------------------- shrinkage: learning-rate curves
Xs, ys = load_breast_cancer(return_X_y=True)
Xs_a, Xs_b, ys_a, ys_b = train_test_split(Xs, ys, test_size=0.4, random_state=0, stratify=ys)
curves = {}
for eta in (1.0, 0.3, 0.05):
    res = {}
    xgb.train({"eta": eta, "max_depth": 3, "objective": "binary:logistic", "eval_metric": "logloss",
               "seed": 0, "nthread": 1}, xgb.DMatrix(Xs_a, label=ys_a), num_boost_round=400,
              evals=[(xgb.DMatrix(Xs_b, label=ys_b), "test")], evals_result=res, verbose_eval=False)
    curves[eta] = np.array(res["test"]["logloss"])
    out.val(f"eta{eta}_best", curves[eta].min(), 4)
    out.val(f"eta{eta}_arg", int(curves[eta].argmin()) + 1)
    out.val(f"eta{eta}_final", curves[eta][-1], 4)
it = np.arange(1, 401)
out.dat("eta", {"it": it[::4], "e1": curves[1.0][::4], "e03": curves[0.3][::4], "e005": curves[0.05][::4]})

# ---------------------------------------------------------------- sparsity-aware split: default direction for missing
Xmiss = Xs_a.copy(); Xmiss[rng.random(Xmiss.shape) < 0.2] = np.nan
bm = xgb.train({"max_depth": 2, "objective": "binary:logistic", "seed": 0, "nthread": 1},
               xgb.DMatrix(Xmiss, label=ys_a), num_boost_round=1)
dfm = bm.trees_to_dataframe()
r0 = dfm[dfm.Node == 0].iloc[0]
out.val("miss_dir", "yes" if r0.Missing == r0.Yes else "no")
out.val("miss_feat", r0.Feature)
out.text("dump", bm.get_dump(with_stats=False)[0])

# ---------------------------------------------------------------- LightGBM and stacking
Xk, yk = make_classification(n_samples=3000, n_features=20, n_informative=8, random_state=1)
Xka, Xkb, yka, ykb = train_test_split(Xk, yk, random_state=0)
lg = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, max_bin=255, random_state=0,
                        verbose=-1).fit(Xka, yka)
out.val("lgb_acc", lg.score(Xkb, ykb), 3)
xg2 = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, tree_method="hist",
                        random_state=0, n_jobs=1).fit(Xka, yka)
out.val("xgb_acc", xg2.score(Xkb, ykb), 3)
base_models = [("lr", LogisticRegression(max_iter=1000)), ("rf", RandomForestClassifier(n_estimators=200, random_state=0)),
               ("knn", KNeighborsClassifier(15))]
srows = []
for name, m in base_models:
    srows.append(f"{name} & {m.fit(Xka, yka).score(Xkb, ykb):.3f} \\\\")
stk = StackingClassifier(base_models, final_estimator=LogisticRegression(), cv=5).fit(Xka, yka)
srows.append(f"stack (LR on out-of-fold predictions) & {stk.score(Xkb, ykb):.3f} \\\\")
out.tex("stack_rows", "\n".join(srows))
