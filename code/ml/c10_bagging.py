"""Chapter 10a: bootstrap and 63.2%, variance of averages, bagging, random forests, OOB, importances."""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 63.2%
ns = np.array([1, 2, 3, 5, 10, 20, 50, 100, 1000, 10000])
p_out = (1 - 1 / ns) ** ns
out.dat("boot", {"n": ns, "pin": 1 - p_out})
for n_ in (2, 5, 10, 100, 1000):
    out.val(f"pin_{n_}", 1 - (1 - 1 / n_) ** n_, 4)
out.val("lim", 1 - np.exp(-1), 4); out.val("inv_e", np.exp(-1), 4)
fr = [len(np.unique(rng.integers(0, 1000, 1000))) / 1000 for _ in range(2000)]
out.val("sim_unique", np.mean(fr), 4)
out.check("fraction of unique rows in a bootstrap of 1000 vs 1-(1-1/n)^n", np.mean(fr), 1 - (1 - 1e-3) ** 1000, atol=1e-3)
# expected count of a given row in a bootstrap ~ Poisson(1): P(0)=e^-1, P(1)=e^-1, P(2)=e^-1/2
out.val("p_twice", np.exp(-1) / 2, 4)

# ---------------------------------------------------------------- variance of an average of correlated models
sigma2 = 1.0
vrows = []
for rho in (0.0, 0.3, 0.7):
    for B in (10, 100):
        C = rho * np.ones((B, B)) + (1 - rho) * np.eye(B)
        Z = rng.multivariate_normal(np.zeros(B), C * sigma2, size=20000)
        sim = Z.mean(1).var()
        th = rho * sigma2 + (1 - rho) * sigma2 / B
        out.check(f"Var(mean of {B}) with rho={rho}", sim, th, rtol=0.05, atol=2e-3)
        vrows.append(f"{rho:g} & {B} & {th:.3f} & {sim:.3f} \\\\")
out.tex("var_rows", "\n".join(vrows))

# ---------------------------------------------------------------- bagging reduces variance (regression bias-variance)
def f_true(X):
    return np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2
Xte = rng.uniform(-1, 1, (300, 2)); fte = f_true(Xte)
models = {
    "stump": lambda: DecisionTreeRegressor(max_depth=1),
    "deep tree": lambda: DecisionTreeRegressor(),
    "bagged deep trees (100)": lambda: BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=0),
    "random forest (100)": lambda: RandomForestRegressor(n_estimators=100, max_features=1, random_state=0),
    "boosted stumps (300, $\\nu=0.1$)": lambda: GradientBoostingRegressor(max_depth=1, n_estimators=300, learning_rate=0.1, random_state=0),
}
reps = 20
brows = []
bv = {}
for name, mk in models.items():
    P = np.empty((reps, len(Xte)))
    for r in range(reps):
        Xtr = rng.uniform(-1, 1, (200, 2)); ytr = f_true(Xtr) + rng.normal(0, 0.3, 200)
        P[r] = mk().fit(Xtr, ytr).predict(Xte)
    b2 = np.mean((P.mean(0) - fte) ** 2); v = np.mean(P.var(0))
    bv[name] = (b2, v)
    brows.append(f"{name} & {b2:.4f} & {v:.4f} & {b2 + v + 0.09:.4f} \\\\")
out.tex("bv_rows", "\n".join(brows))

# ---------------------------------------------------------------- random forest decorrelation and OOB
X, y = make_classification(n_samples=1500, n_features=20, n_informative=6, n_redundant=4,
                           random_state=0)
Xtr, Xte2, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
corr = {}
for mf, name in [(None, "bag"), ("sqrt", "rf")]:
    rf = RandomForestClassifier(n_estimators=200, max_features=mf, oob_score=True, random_state=0).fit(Xtr, ytr)
    preds = np.array([t.predict_proba(Xte2)[:, 1] for t in rf.estimators_])
    cc = np.corrcoef(preds)
    corr[name] = cc[np.triu_indices(len(cc), 1)].mean()
    out.val(f"{name}_corr", corr[name], 3)
    out.val(f"{name}_test", rf.score(Xte2, yte), 3)
    out.val(f"{name}_oob", rf.oob_score_, 3)
    if name == "rf":
        # OOB accuracy by hand from the bootstrap indices
        votes = np.zeros((len(Xtr), 2))
        for t, samp in zip(rf.estimators_, rf.estimators_samples_):
            mask = np.ones(len(Xtr), bool); mask[samp] = False
            votes[mask] += t.predict_proba(Xtr[mask])
        ok = votes.sum(1) > 0
        oob_hand = np.mean(votes[ok].argmax(1) == ytr[ok])
        out.check("OOB accuracy by hand vs RandomForestClassifier.oob_score_", oob_hand, rf.oob_score_)
        out.val("rf_cv", cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0), Xtr, ytr, cv=5).mean(), 3)
# test accuracy vs number of trees
accs = []
Bs = [1, 2, 5, 10, 20, 50, 100, 200, 400]
for B in Bs:
    accs.append(RandomForestClassifier(n_estimators=B, random_state=0).fit(Xtr, ytr).score(Xte2, yte))
out.dat("ntrees", {"B": Bs, "acc": accs})

# ---------------------------------------------------------------- MDI pitfalls
n = 1000
x_inf = rng.normal(size=n)                      # informative, continuous
x_bin = rng.integers(0, 2, n)                    # informative, binary
x_noise_cont = rng.normal(size=n)                # pure noise, high cardinality
x_noise_bin = rng.integers(0, 2, n)              # pure noise, binary
logit = 1.2 * x_inf + 1.5 * x_bin - 0.75
yy = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
Xm = np.column_stack([x_inf, x_bin, x_noise_cont, x_noise_bin])
names = ["informative (cont.)", "informative (binary)", "noise (cont.)", "noise (binary)"]
Xa, Xb, ya, yb = train_test_split(Xm, yy, test_size=0.5, random_state=0)
rfm = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xa, ya)
pim = permutation_importance(rfm, Xb, yb, n_repeats=30, random_state=0)
# permutation importance by hand (mean accuracy drop on held-out data)
base = rfm.score(Xb, yb)
mine = []
for j in range(4):
    drops = []
    for _ in range(30):
        Xp = Xb.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        drops.append(base - rfm.score(Xp, yb))
    mine.append(np.mean(drops))
out.check("permutation importance by hand vs sklearn (30 repeats)", mine, pim.importances_mean, atol=0.01)
mrows = [f"{nm} & {m:.3f} & {p:.3f} \\\\" for nm, m, p in zip(names, rfm.feature_importances_, pim.importances_mean)]
out.tex("mdi_rows", "\n".join(mrows))
out.val("mdi_noise_cont", rfm.feature_importances_[2], 3)
out.val("mdi_inf_bin", rfm.feature_importances_[1], 3)
# correlated copies split the credit
Xc = np.column_stack([x_inf, x_inf + 0.01 * rng.normal(size=n), x_bin])
rfc = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xc, yy)
rf1 = RandomForestClassifier(n_estimators=300, random_state=0).fit(np.column_stack([x_inf, x_bin]), yy)
out.val("copy_a", rfc.feature_importances_[0], 3); out.val("copy_b", rfc.feature_importances_[1], 3)
out.val("single_inf", rf1.feature_importances_[0], 3)
