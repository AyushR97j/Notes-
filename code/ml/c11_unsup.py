"""Chapter 11: k-means, hierarchical, DBSCAN, GMM-EM, PCA, ICA, isolation forest, Apriori."""
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.stats import norm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_wine, make_blobs, make_moons
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- k-means: one Lloyd iteration by hand
P = np.array([[1, 1], [1.5, 2], [2.8, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]])
C0 = np.array([[1.0, 1.0], [5.0, 7.0]])
D = np.sqrt(((P[:, None] - C0[None]) ** 2).sum(-1))
lab = D.argmin(1)
C1 = np.array([P[lab == k].mean(0) for k in range(2)])
out.tex("km_rows", "\n".join(f"({p[0]:g}, {p[1]:g}) & {d[0]:.2f} & {d[1]:.2f} & {l+1} \\\\" for p, d, l in zip(P, D, lab)))
out.tex("km_c1", "; ".join(f"({c[0]:.3g}, {c[1]:.3g})" for c in C1))
sk1 = KMeans(n_clusters=2, init=C0, n_init=1, max_iter=1, algorithm="lloyd").fit(P)
out.check("one Lloyd step by hand vs sklearn KMeans(max_iter=1) centroids", C1, sk1.cluster_centers_)
inertia0 = (D.min(1) ** 2).sum()
# run to convergence, recording inertia
C = C0.copy(); hist = []
for it in range(20):
    D = ((P[:, None] - C[None]) ** 2).sum(-1); lab = D.argmin(1)
    hist.append(D.min(1).sum())
    Cn = np.array([P[lab == k].mean(0) for k in range(2)])
    if np.allclose(Cn, C):
        break
    C = Cn
out.tex("km_inertia", ", ".join(f"{v:.2f}" for v in hist))
out.val("km_iters", len(hist))
skc = KMeans(n_clusters=2, init=C0, n_init=1, algorithm="lloyd", tol=0).fit(P)
out.check("converged centroids vs sklearn", C, skc.cluster_centers_)

# trajectories on blobs (bad init) for a figure
Xb, yb = make_blobs(n_samples=300, centers=[[0, 0], [4, 1], [2, 4]], cluster_std=0.8, random_state=2)
C = np.array([[1.6, 4.4], [2.4, 4.6], [2.0, 3.8]])
traj = [C.copy()]; inert = []
for it in range(15):
    D = ((Xb[:, None] - C[None]) ** 2).sum(-1); lab = D.argmin(1)
    inert.append(D.min(1).sum())
    C = np.array([Xb[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(3)])
    traj.append(C.copy())
traj = np.array(traj)
for k in range(3):
    out.dat(f"traj{k}", {"x": traj[:, k, 0], "y": traj[:, k, 1]})
out.dat("blobs", {"x": Xb[:, 0], "y": Xb[:, 1], "c": lab})
out.check("inertia is non-increasing under Lloyd iterations", float(np.all(np.diff(inert) <= 1e-9)), 1.0)

# k-means++ vs random init
rand_in, pp_in = [], []
Xk, _ = make_blobs(n_samples=600, centers=8, cluster_std=0.6, random_state=7)
for s in range(50):
    rand_in.append(KMeans(8, init="random", n_init=1, random_state=s).fit(Xk).inertia_)
    pp_in.append(KMeans(8, init="k-means++", n_init=1, random_state=s).fit(Xk).inertia_)
best = min(min(rand_in), min(pp_in))
out.val("rand_bad", np.mean(np.array(rand_in) > 1.05 * best), 2)
out.val("pp_bad", np.mean(np.array(pp_in) > 1.05 * best), 2)
out.val("rand_worst", max(rand_in) / best, 2); out.val("pp_worst", max(pp_in) / best, 2)

# elbow and silhouette
Xe, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.9, random_state=3)
ks = range(2, 9); inert_k, sil_k = [], []
for k in ks:
    m = KMeans(k, n_init=10, random_state=0).fit(Xe)
    inert_k.append(m.inertia_); sil_k.append(silhouette_score(Xe, m.labels_))
out.dat("elbow", {"k": list(ks), "inertia": inert_k, "sil": sil_k})
out.val("sil_best_k", list(ks)[int(np.argmax(sil_k))])
# silhouette by hand for one point
Xs = np.array([[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]], float); ls = np.array([0, 0, 0, 1, 1, 1])
i = 0
a = np.mean([np.linalg.norm(Xs[i] - Xs[j]) for j in (1, 2)])
b = np.mean([np.linalg.norm(Xs[i] - Xs[j]) for j in (3, 4, 5)])
s_i = (b - a) / max(a, b)
out.check("silhouette of point 0 by hand vs sklearn", s_i, silhouette_samples(Xs, ls)[0])
out.val("sil_a", a, 3); out.val("sil_b", b, 3); out.val("sil_s", s_i, 4)

# ---------------------------------------------------------------- hierarchical clustering by hand (1-D)
x1 = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
for meth in ("single", "complete", "average"):
    Z = linkage(x1[:, None], method=meth)
    out.tex(f"hc_{meth}", ", ".join(f"{h:g}" for h in Z[:, 2]))
# by-hand single linkage merge heights: 1, 2, 3, 4
out.check("single-linkage merge heights vs scipy", linkage(x1[:, None], "single")[:, 2], [1, 2, 3, 4])
out.check("complete-linkage merge heights vs scipy", linkage(x1[:, None], "complete")[:, 2], [1, 3, 4, 10])

# ---------------------------------------------------------------- DBSCAN on moons vs k-means
Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)
km = KMeans(2, n_init=10, random_state=0).fit(Xm)
db = DBSCAN(eps=0.2, min_samples=5).fit(Xm)
out.val("ari_km", adjusted_rand_score(ym, km.labels_), 3)
out.val("ari_db", adjusted_rand_score(ym, db.labels_), 3)
out.val("db_noise", int(np.sum(db.labels_ == -1)))
out.val("db_ncl", len(set(db.labels_) - {-1}))
out.dat("moons_km", {"x": Xm[:, 0], "y": Xm[:, 1], "c": km.labels_})
out.dat("moons_db", {"x": Xm[:, 0], "y": Xm[:, 1], "c": db.labels_})
# tiny DBSCAN example: core / border / noise
T = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2.2, 1], [5, 5]], float)
dbt = DBSCAN(eps=1.5, min_samples=4).fit(T)
core = np.zeros(len(T), bool); core[dbt.core_sample_indices_] = True
kinds = ["core" if c else ("noise" if l == -1 else "border") for c, l in zip(core, dbt.labels_)]
out.tex("db_kinds", ", ".join(kinds))
# by hand: neighbour counts within eps (including itself)
cnt = (np.sqrt(((T[:, None] - T[None]) ** 2).sum(-1)) <= 1.5).sum(1)
out.tex("db_counts", ", ".join(str(c) for c in cnt))
out.check("core points by hand (count >= min_samples) vs sklearn", (cnt >= 4).astype(float), core.astype(float))

# ---------------------------------------------------------------- GMM via EM (1-D) vs sklearn
xg = np.r_[rng.normal(-2, 0.8, 300), rng.normal(2.5, 1.2, 200)]
pi, mu, var = np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])
# [[em]]
def em_step(x, pi, mu, var):
    # E-step: responsibilities r_ik = pi_k N(x_i|mu_k,var_k) / sum_j (...)
    dens = pi * norm.pdf(x[:, None], mu, np.sqrt(var))
    r = dens / dens.sum(1, keepdims=True)
    # M-step: weighted MLEs
    Nk = r.sum(0)
    pi = Nk / len(x)
    mu = (r * x[:, None]).sum(0) / Nk
    var = (r * (x[:, None] - mu) ** 2).sum(0) / Nk
    return pi, mu, var, np.log(dens.sum(1)).sum()
# [[/em]]
lls = []
for it in range(200):
    pi, mu, var, ll = em_step(xg, pi, mu, var)
    lls.append(ll)
    if it > 1 and abs(lls[-1] - lls[-2]) < 1e-12:
        break
out.check("EM log-likelihood never decreases", float(np.all(np.diff(lls) >= -1e-9)), 1.0)
gm = GaussianMixture(2, means_init=[[-1.0], [1.0]], weights_init=[0.5, 0.5], precisions_init=[[[1.0]], [[1.0]]],
                     tol=1e-12, max_iter=1000, reg_covar=0).fit(xg[:, None])
order = np.argsort(gm.means_.ravel())
out.check("EM from scratch vs sklearn GaussianMixture (weights, means, variances)",
          np.r_[pi, mu, var], np.r_[gm.weights_[order], gm.means_.ravel()[order], gm.covariances_.ravel()[order]], atol=1e-5)
out.tex("em_params", f"\\pi=({pi[0]:.3f},{pi[1]:.3f}),\\ \\mu=({mu[0]:.3f},{mu[1]:.3f}),\\ \\sigma^2=({var[0]:.3f},{var[1]:.3f})")
out.val("em_iters", len(lls))
out.dat("em_ll", {"it": np.arange(1, len(lls) + 1)[:40], "ll": np.array(lls)[:40]})
# one E-step by hand on three points
x3 = np.array([-1.0, 0.0, 2.0])
d3 = np.array([0.5, 0.5]) * norm.pdf(x3[:, None], [-1.0, 1.0], 1.0)
r3 = d3 / d3.sum(1, keepdims=True)
out.tex("e_resp", ", ".join(f"{v:.3f}" for v in r3[:, 0]))

# ---------------------------------------------------------------- PCA by hand (2-D) and the scaling trap
Xp = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]])
Xc = Xp - Xp.mean(0)
S = np.cov(Xc.T)
lam, V = np.linalg.eigh(S); lam, V = lam[::-1], V[:, ::-1]
v1 = V[:, 0] * np.sign(V[0, 0])
pca = PCA(2).fit(Xp)
out.check("PCA eigenvalues vs sklearn explained_variance_", lam, pca.explained_variance_)
out.check("PC1 direction vs sklearn (up to sign)", v1, pca.components_[0] * np.sign(pca.components_[0, 0]))
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
out.check("eigenvalues = singular values^2/(n-1)", lam, s ** 2 / (len(Xp) - 1))
out.tex("pca_S", f"{S[0,0]:.4f} & {S[0,1]:.4f} \\\\ {S[1,0]:.4f} & {S[1,1]:.4f}")
out.val("pca_l1", lam[0], 4); out.val("pca_l2", lam[1], 4)
out.tex("pca_v1", f"{v1[0]:.4f}, {v1[1]:.4f}")
out.val("pca_evr", lam[0] / lam.sum(), 4)
out.tex("pca_mean", f"{Xp.mean(0)[0]:.2f}, {Xp.mean(0)[1]:.2f}")
z = (Xp[0] - Xp.mean(0)) @ v1
out.val("pca_z", z, 4)
out.check("projection of the first point vs sklearn transform", abs(z), abs(pca.transform(Xp[:1])[0, 0]))
Xw, yw = load_wine(return_X_y=True)
r_raw = PCA(2).fit(Xw); r_std = PCA(2).fit(StandardScaler().fit_transform(Xw))
names = load_wine().feature_names
top_raw = names[int(np.argmax(np.abs(r_raw.components_[0])))]
out.val("wine_evr_raw", r_raw.explained_variance_ratio_[0], 4)
out.val("wine_evr_std", r_std.explained_variance_ratio_[0], 4)
out.val("wine_top_raw", top_raw.replace("_", "\\_"))
out.val("wine_load_raw", abs(r_raw.components_[0]).max(), 4)
out.val("wine_var_proline", Xw[:, names.index("proline")].var(ddof=1), 0)
out.val("wine_var_hue", Xw[:, names.index("hue")].var(ddof=1), 3)
km_raw = KMeans(3, n_init=10, random_state=0).fit(PCA(2).fit_transform(Xw))
km_std = KMeans(3, n_init=10, random_state=0).fit(PCA(2).fit_transform(StandardScaler().fit_transform(Xw)))
out.val("wine_ari_raw", adjusted_rand_score(yw, km_raw.labels_), 3)
out.val("wine_ari_std", adjusted_rand_score(yw, km_std.labels_), 3)
evr = PCA().fit(StandardScaler().fit_transform(Xw)).explained_variance_ratio_
out.dat("scree", {"k": np.arange(1, 14), "evr": evr, "cum": np.cumsum(evr)})
out.val("wine_k90", int(np.argmax(np.cumsum(evr) >= 0.9)) + 1)

# ---------------------------------------------------------------- ICA vs PCA on mixed signals
t = np.linspace(0, 8, 2000)
S_true = np.c_[np.sign(np.sin(3 * t)), np.sin(5 * t + 1)]
A = np.array([[1.0, 0.6], [0.5, 1.0]])
Xmix = S_true @ A.T
Sica = FastICA(2, random_state=0, whiten="unit-variance").fit_transform(Xmix)
Spca = PCA(2).fit_transform(Xmix)
def best_corr(Sh):
    c = np.abs(np.corrcoef(np.c_[S_true, Sh].T)[:2, 2:])
    return min(c.max(1))
out.val("ica_corr", best_corr(Sica), 3); out.val("pca_corr", best_corr(Spca), 3)

# ---------------------------------------------------------------- isolation forest
Xi = np.r_[rng.normal(0, 1, (300, 2)), rng.uniform(-6, 6, (10, 2))]
iso = IsolationForest(n_estimators=200, random_state=0).fit(Xi)
sc = -iso.score_samples(Xi)
out.val("iso_out_mean", sc[300:].mean(), 3); out.val("iso_in_mean", sc[:300].mean(), 3)
top10 = np.argsort(-sc)[:10]
out.val("iso_top10_hits", int(np.sum(top10 >= 300)))
H = lambda i: np.log(i) + 0.5772156649
cn = 2 * H(255) - 2 * 255 / 256
out.val("iso_c256", cn, 3)

# ---------------------------------------------------------------- Apriori: support, confidence, lift
tx = [{"bread", "milk"}, {"bread", "butter", "milk"}, {"bread", "butter"}, {"milk", "tea"},
      {"bread", "butter", "milk"}, {"bread", "tea"}]
N = len(tx)
sup = lambda s: sum(set(s) <= t for t in tx) / N
out.val("sup_bread", sup({"bread"}), 4); out.val("sup_butter", sup({"butter"}), 4)
out.val("sup_bb", sup({"bread", "butter"}), 4)
out.val("conf_butter_bread", sup({"bread", "butter"}) / sup({"butter"}), 4)
out.val("conf_bread_butter", sup({"bread", "butter"}) / sup({"bread"}), 4)
out.val("lift_bb", sup({"bread", "butter"}) / (sup({"bread"}) * sup({"butter"})), 4)
out.val("lift_mt", sup({"milk", "tea"}) / (sup({"milk"}) * sup({"tea"})), 4)
# [[apriori]]
def apriori(tx, min_sup):
    items = sorted(set().union(*tx))
    level = [frozenset([i]) for i in items if sup({i}) >= min_sup]
    frequent = list(level)
    while level:
        # join step: unions of two frequent k-sets that differ in one item
        cand = {a | b for a in level for b in level if len(a | b) == len(a) + 1}
        # prune step: every k-subset of a candidate must be frequent (anti-monotonicity)
        cand = {c for c in cand if all(frozenset(s) in set(level) for s in combinations(c, len(c) - 1))}
        level = [c for c in cand if sup(c) >= min_sup]
        frequent += level
    return frequent
# [[/apriori]]
fq = apriori(tx, 1 / 3)
brute = [frozenset(c) for k in range(1, 5) for c in combinations(sorted(set().union(*tx)), k) if sup(c) >= 1 / 3]
out.check("Apriori frequent itemsets == brute-force enumeration", float(set(fq) == set(brute)), 1.0)
out.tex("freq_sets", "; ".join("\\{" + ", ".join(sorted(f)) + "\\}" for f in sorted(fq, key=lambda f: (len(f), sorted(f)))))
