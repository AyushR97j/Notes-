"""Chapter 7: kNN by hand, distances, scaling, choice of k, curse of dimensionality, KD-tree pruning."""
import numpy as np
from scipy.spatial import distance as D
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KDTree, KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- kNN by hand
P = np.array([[1, 1], [2, 1], [4, 3], [6, 4], [3, 5], [0, 3]], float)
lab = np.array(["A", "A", "B", "B", "B", "A"])
q = np.array([3.0, 3.0])
d2 = np.sqrt(((P - q) ** 2).sum(1))
d1 = np.abs(P - q).sum(1)
out.tex("hand_rows", "\n".join(f"$({int(p[0])},{int(p[1])})$ & {l} & {a:.3f} & {b:.0f} \\\\"
                               for p, l, a, b in zip(P, lab, d2, d1)))
order = np.argsort(d2, kind="stable")
agree = []
for k in (1, 3, 5):
    votes = lab[order[:k]]
    ours = max(set(votes), key=list(votes).count)
    sk = KNeighborsClassifier(n_neighbors=k).fit(P, lab).predict([q])[0]
    agree.append(float(ours == sk))
    out.val(f"pred_k{k}", ours)
    out.val(f"votes_k{k}", "".join(sorted(votes)))
out.check("hand kNN vote == sklearn prediction for k=1,3,5", agree, [1.0, 1.0, 1.0])
# regression version: targets
t = np.array([10.0, 12, 20, 24, 22, 14])
out.val("reg_k3", t[order[:3]].mean(), 2)
out.check("kNN regression k=3 vs sklearn", t[order[:3]].mean(),
          KNeighborsRegressor(n_neighbors=3).fit(P, t).predict([q])[0])
w = 1 / d2[order[:3]]
out.val("reg_k3w", (w * t[order[:3]]).sum() / w.sum(), 2)
out.check("distance-weighted kNN regression vs sklearn", (w * t[order[:3]]).sum() / w.sum(),
          KNeighborsRegressor(n_neighbors=3, weights="distance").fit(P, t).predict([q])[0])

# ---------------------------------------------------------------- distances between two vectors
a = np.array([1.0, 2.0, 3.0]); b = np.array([4.0, 0.0, 3.0])
vals = {"euc": D.euclidean(a, b), "man": D.cityblock(a, b), "cheb": D.chebyshev(a, b),
        "mink3": D.minkowski(a, b, 3), "cos": D.cosine(a, b)}
out.check("Euclidean by formula", np.sqrt(((a - b) ** 2).sum()), vals["euc"])
out.check("cosine distance by formula", 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b), vals["cos"])
for k_, v in vals.items():
    out.val(k_, v, 4)
out.val("cos_sim", 1 - vals["cos"], 4)
h1, h2 = np.array([1, 0, 1, 1, 0, 1]), np.array([1, 1, 0, 1, 0, 0])
out.val("ham", int((h1 != h2).sum()))
out.check("Hamming (scipy returns a fraction)", (h1 != h2).mean(), D.hamming(h1, h2))
# Mahalanobis
S = np.array([[4.0, 1.5], [1.5, 1.0]])
u, v = np.array([2.0, 1.0]), np.array([0.0, 0.0])
mah = D.mahalanobis(u, v, np.linalg.inv(S))
out.check("Mahalanobis by formula", np.sqrt((u - v) @ np.linalg.inv(S) @ (u - v)), mah)
out.val("mah", mah, 4); out.val("mah_euc", np.linalg.norm(u - v), 4)

# ---------------------------------------------------------------- scaling changes the neighbour
# customers: (income in rupees per month, age in years)
C = np.array([[50_000, 58], [52_000, 25]], float)
cq = np.array([51_800, 57.0])
raw_nn = int(np.argmin(((C - cq) ** 2).sum(1)))
sc = StandardScaler().fit(np.vstack([C, rng.normal([60_000, 40], [15_000, 12], (200, 2))]))
std_nn = int(np.argmin(((sc.transform(C) - sc.transform([cq])) ** 2).sum(1)))
out.val("raw_nn", raw_nn + 1); out.val("std_nn", std_nn + 1)
out.val("raw_d1", np.linalg.norm(C[0] - cq), 1); out.val("raw_d2", np.linalg.norm(C[1] - cq), 1)

# ---------------------------------------------------------------- choice of k (moons)
X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
ks = np.array([1, 2, 3, 5, 7, 9, 13, 17, 21, 31, 41, 61, 81, 121, 161, 201])
cv_err, tr_err = [], []
for k in ks:
    m = KNeighborsClassifier(n_neighbors=k)
    cv_err.append(1 - cross_val_score(m, X, y, cv=5).mean())
    tr_err.append(1 - m.fit(X, y).score(X, y))
out.dat("k", {"k": ks, "cv": cv_err, "train": tr_err})
kb = int(ks[np.argmin(cv_err)])
out.val("k_best", kb); out.val("k_best_err", min(cv_err), 3)
out.val("k1_cv", cv_err[0], 3); out.val("k1_tr", tr_err[0], 3); out.val("k201_cv", cv_err[-1], 3)

# ---------------------------------------------------------------- curse of dimensionality
dims = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000])
ratio = []
for d in dims:
    Z = rng.uniform(size=(500, d)); z0 = rng.uniform(size=d)
    dist = np.sqrt(((Z - z0) ** 2).sum(1))
    ratio.append((dist.max() - dist.min()) / dist.min())
out.dat("contrast", {"d": dims, "ratio": ratio})
out.val("contrast_2", ratio[1], 2); out.val("contrast_1000", ratio[-1], 3)
edge = 0.1 ** (1 / dims)
out.dat("edge", {"d": dims, "edge": edge})
out.val("edge_10", 0.1 ** (1 / 10), 3); out.val("edge_100", 0.1 ** (1 / 100), 3)
# fraction of a unit ball's volume within the outer 1% shell: 1 - 0.99^d
out.val("shell_100", 1 - 0.99 ** 100, 3); out.val("shell_1000", 1 - 0.99 ** 1000, 5)
# kNN accuracy as irrelevant features are added to moons
noise_dims = [0, 2, 5, 10, 20, 50, 100]
accs = []
Xtr, ytr = make_moons(n_samples=400, noise=0.2, random_state=1)
Xte, yte = make_moons(n_samples=1000, noise=0.2, random_state=2)
for nd in noise_dims:
    A = np.column_stack([Xtr, rng.uniform(-1.5, 1.5, (len(Xtr), nd))])
    B = np.column_stack([Xte, rng.uniform(-1.5, 1.5, (len(Xte), nd))])
    accs.append(KNeighborsClassifier(n_neighbors=7).fit(A, ytr).score(B, yte))
out.dat("irrel", {"nd": noise_dims, "acc": accs})
out.val("irr_0", accs[0], 3); out.val("irr_100", accs[-1], 3)

# ---------------------------------------------------------------- a KD-tree with a pruning counter
class Node:
    __slots__ = ("idx", "axis", "left", "right")

def build(points, idx, depth=0):
    if len(idx) == 0:
        return None
    axis = depth % points.shape[1]
    idx = idx[np.argsort(points[idx, axis], kind="stable")]
    mid = len(idx) // 2
    nd = Node(); nd.idx = idx[mid]; nd.axis = axis
    nd.left = build(points, idx[:mid], depth + 1)
    nd.right = build(points, idx[mid + 1:], depth + 1)
    return nd

def nearest(node, points, q, best, count):
    if node is None:
        return best
    count[0] += 1
    p = points[node.idx]
    dd = np.sum((p - q) ** 2)
    if dd < best[0]:
        best = (dd, node.idx)
    diff = q[node.axis] - p[node.axis]
    near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
    best = nearest(near, points, q, best, count)
    if diff ** 2 < best[0]:                # the other side can still hold a closer point
        best = nearest(far, points, q, best, count)
    return best

rows, ours_i, ref_i = [], [], []
for d in (2, 5, 10, 20):
    pts = rng.uniform(size=(4000, d)); qs = rng.uniform(size=(50, d))
    root = build(pts, np.arange(len(pts)))
    visits = []
    for qq in qs:
        cnt = [0]
        _, i = nearest(root, pts, qq, (np.inf, -1), cnt)
        visits.append(cnt[0])
        _, ref = KDTree(pts).query([qq], k=1)
        ours_i.append(i); ref_i.append(ref[0, 0])
    rows.append(f"{d} & {np.mean(visits):.0f} & {100 * np.mean(visits) / len(pts):.1f}\\% \\\\")
out.check("our KD-tree nearest-neighbour indices == sklearn KDTree (200 queries)", ours_i, ref_i)
out.tex("kd_rows", "\n".join(rows))
