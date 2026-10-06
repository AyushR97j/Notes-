"""Chapter 9: decision trees — impurity by hand, splits, continuous thresholds, regression trees, pruning."""
import numpy as np
from scipy.stats import entropy as H
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- the offer dataset (14 candidates)
# features: CTC (High/Low), City (Metro/Tier2), Role (Dev/Data/Ops); label: accepted offer?
rows = [
    ("High", "Metro", "Dev", 1), ("High", "Metro", "Data", 1), ("High", "Tier2", "Dev", 1),
    ("High", "Tier2", "Ops", 0), ("High", "Metro", "Ops", 1), ("High", "Tier2", "Data", 1),
    ("Low", "Metro", "Dev", 1), ("Low", "Metro", "Data", 1), ("Low", "Tier2", "Dev", 0),
    ("Low", "Tier2", "Data", 0), ("Low", "Metro", "Ops", 0), ("Low", "Tier2", "Ops", 0),
    ("High", "Metro", "Dev", 1), ("Low", "Metro", "Dev", 0),
]
F = {"CTC": [r[0] for r in rows], "City": [r[1] for r in rows], "Role": [r[2] for r in rows]}
y = np.array([r[3] for r in rows])
out.tex("table", "\n".join(f"{i+1} & {a} & {b} & {c} & {'yes' if d else 'no'} \\\\" for i, (a, b, c, d) in enumerate(rows)))

def ent(lbl):
    p = np.bincount(lbl, minlength=2) / len(lbl)
    return H(p, base=2)

def gini(lbl):
    p = np.bincount(lbl, minlength=2) / len(lbl)
    return 1 - (p ** 2).sum()

npos, nneg = int(y.sum()), int(len(y) - y.sum())
out.val("npos", npos); out.val("nneg", nneg)
Hp = ent(y); Gp = gini(y)
out.check("parent entropy vs -p log p by hand", Hp, -(npos / 14 * np.log2(npos / 14) + nneg / 14 * np.log2(nneg / 14)))
out.val("Hp", Hp, 4); out.val("Gp", Gp, 4)
best = None
for f, vals in F.items():
    vals = np.array(vals)
    cond_H, cond_G, split_info, parts = 0.0, 0.0, 0.0, []
    for v in sorted(set(vals)):
        m = vals == v
        w = m.mean()
        cond_H += w * ent(y[m]); cond_G += w * gini(y[m]); split_info -= w * np.log2(w)
        parts.append(f"{v}: {int(y[m].sum())}+/{int((1 - y[m]).sum())}-")
        out.val(f"{f}_{v}_pos", int(y[m].sum())); out.val(f"{f}_{v}_neg", int((1 - y[m]).sum()))
        out.val(f"{f}_{v}_H", ent(y[m]), 4)
    ig = Hp - cond_H
    out.val(f"{f}_condH", cond_H, 4); out.val(f"{f}_IG", ig, 4)
    out.val(f"{f}_Gdec", Gp - cond_G, 4); out.val(f"{f}_SI", split_info, 4)
    out.val(f"{f}_GR", ig / split_info, 4)
    out.tex(f"{f}_parts", "; ".join(parts))
    if best is None or ig > best[1]:
        best = (f, ig)
out.val("best", best[0])

# check the binary-feature numbers with sklearn's impurity bookkeeping
Xb = np.column_stack([(np.array(F["CTC"]) == "High").astype(int)])
st = DecisionTreeClassifier(criterion="entropy", max_depth=1).fit(Xb, y)
imp = st.tree_.impurity; ns = st.tree_.n_node_samples
ig_sk = imp[0] - (ns[1] * imp[1] + ns[2] * imp[2]) / ns[0]
out.check("IG(CTC) by hand vs sklearn entropy stump", ig_sk, Hp - sum(
    (np.array(F["CTC"]) == v).mean() * ent(y[np.array(F["CTC"]) == v]) for v in ("High", "Low")))
stg = DecisionTreeClassifier(criterion="gini", max_depth=1).fit(Xb, y)
out.check("parent Gini by hand vs sklearn", stg.tree_.impurity[0], Gp)

# an ID-like feature: every row its own value
out.val("id_IG", Hp, 4); out.val("id_SI", np.log2(14), 4); out.val("id_GR", Hp / np.log2(14), 4)

# ---------------------------------------------------------------- continuous feature: threshold search
x = np.array([2.0, 3.0, 4.5, 5.0, 6.5, 7.0, 8.0, 9.5])
yc = np.array([0, 0, 0, 1, 0, 1, 1, 1])
order = np.argsort(x)
cands = (x[order][:-1] + x[order][1:]) / 2
trows, best_t = [], None
for t in cands:
    L, R = yc[x <= t], yc[x > t]
    wg = (len(L) * gini(L) + len(R) * gini(R)) / len(yc)
    trows.append(f"{t:g} & {int(L.sum())}/{len(L)} & {int(R.sum())}/{len(R)} & {wg:.4f} \\\\")
    if best_t is None or wg < best_t[1] - 1e-12:
        best_t = (t, wg)
out.tex("thr_rows", "\n".join(trows))
out.val("thr_best", best_t[0], 2); out.val("thr_best_g", best_t[1], 4); out.val("thr_parent", gini(yc), 4)
sk_t = DecisionTreeClassifier(max_depth=1).fit(x[:, None], yc).tree_.threshold[0]
out.check("best Gini threshold vs sklearn stump", best_t[0], sk_t)

# ---------------------------------------------------------------- regression tree split
xr = np.array([1.0, 2, 3, 4, 5, 6])
yr = np.array([5.0, 6, 5, 20, 22, 21])
best_r = None
for t in (xr[:-1] + xr[1:]) / 2:
    L, R = yr[xr <= t], yr[xr > t]
    sse = ((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum()
    if best_r is None or sse < best_r[1]:
        best_r = (t, sse, L.mean(), R.mean())
rt = DecisionTreeRegressor(max_depth=1).fit(xr[:, None], yr)
out.check("regression stump threshold/leaf means vs sklearn", [best_r[0], best_r[2], best_r[3]],
          [rt.tree_.threshold[0], rt.tree_.value[1, 0, 0], rt.tree_.value[2, 0, 0]])
out.val("r_t", best_r[0], 1); out.val("r_sse", best_r[1], 2)
out.val("r_l", best_r[2], 2); out.val("r_r", best_r[3], 2)
out.val("r_sst", ((yr - yr.mean()) ** 2).sum(), 2)

# ---------------------------------------------------------------- depth sweep and pruning
X, Y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.4, random_state=0, stratify=Y)
dep, tra, tea = [], [], []
for d in range(1, 16):
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    dep.append(d); tra.append(m.score(Xtr, ytr)); tea.append(m.score(Xte, yte))
out.dat("depth", {"d": dep, "train": tra, "test": tea})
full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
out.val("full_depth", full.get_depth()); out.val("full_leaves", full.get_n_leaves())
out.val("full_test", full.score(Xte, yte), 3)
path = full.cost_complexity_pruning_path(Xtr, ytr)
alphas = path.ccp_alphas[:-1]
leaves, acc = [], []
for a in alphas:
    m = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(Xtr, ytr)
    leaves.append(m.get_n_leaves()); acc.append(m.score(Xte, yte))
out.dat("ccp", {"alpha": alphas + 1e-5, "leaves": leaves, "test": acc})
k = int(np.argmax(acc))
out.val("ccp_best_alpha", alphas[k], 4); out.val("ccp_best_leaves", leaves[k]); out.val("ccp_best_test", acc[k], 3)

# ---------------------------------------------------------------- instability of the root split
roots = []
for _ in range(200):
    idx = rng.integers(0, len(Xtr), len(Xtr))
    roots.append(DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xtr[idx], ytr[idx]).tree_.feature[0])
vals, cnts = np.unique(roots, return_counts=True)
out.val("root_distinct", len(vals)); out.val("root_top_share", cnts.max() / 200, 2)

# ---------------------------------------------------------------- invariance to monotone transforms
m1 = DecisionTreeClassifier(random_state=0, max_depth=4).fit(Xtr, ytr)
m2 = DecisionTreeClassifier(random_state=0, max_depth=4).fit(np.log1p(Xtr), ytr)
out.check("tree predictions unchanged by log-transforming every feature",
          m1.predict(Xte), m2.predict(np.log1p(Xte)))

# ---------------------------------------------------------------- missing values (native NaN support)
Xn = Xtr.copy(); mask = rng.random(Xn.shape) < 0.1; Xn[mask] = np.nan
mn = DecisionTreeClassifier(random_state=0, max_depth=5).fit(Xn, ytr)
Xtn = Xte.copy(); Xtn[rng.random(Xtn.shape) < 0.1] = np.nan
out.val("nan_acc", mn.score(Xtn, yte), 3)

# ---------------------------------------------------------------- axis-aligned regions for a figure
Xm, ym = make_moons(n_samples=200, noise=0.25, random_state=0)
for d in (2, 12):
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xm, ym)
    gx, gy = np.meshgrid(np.linspace(-1.5, 2.5, 81), np.linspace(-1.1, 1.6, 55))
    Z = m.predict(np.column_stack([gx.ravel(), gy.ravel()]))
    out.dat(f"region{d}", {"x": gx.ravel(), "y": gy.ravel(), "c": Z})
    out.val(f"moon_leaves{d}", m.get_n_leaves())
out.dat("m0", {"x": Xm[ym == 0, 0], "y": Xm[ym == 0, 1]})
out.dat("m1", {"x": Xm[ym == 1, 0], "y": Xm[ym == 1, 1]})
