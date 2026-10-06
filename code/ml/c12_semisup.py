"""Chapter 12: when unlabelled data helps and hurts; label propagation; self-training; InfoNCE."""
import numpy as np
import torch
import torch.nn.functional as Fnn
from sklearn.datasets import make_blobs, make_moons
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

def experiment(make, n_lab, seeds, gamma):
    rows = {"lr": [], "sup": [], "self": [], "spread": []}
    for s in seeds:
        X, y = make(s)
        Xt, yt = make(1000 + s)
        # pick n_lab labelled points, balanced
        idx = np.r_[rng.choice(np.where(y == 0)[0], n_lab // 2, replace=False),
                    rng.choice(np.where(y == 1)[0], n_lab // 2, replace=False)]
        y_semi = np.full(len(y), -1); y_semi[idx] = y[idx]
        rows["lr"].append(LogisticRegression().fit(X[idx], y[idx]).score(Xt, yt))
        sup = SVC(kernel="rbf", gamma=gamma, probability=True, random_state=0).fit(X[idx], y[idx])
        rows["sup"].append(sup.score(Xt, yt))
        st = SelfTrainingClassifier(SVC(kernel="rbf", gamma=gamma, probability=True, random_state=0),
                                    threshold=0.9).fit(X, y_semi)
        rows["self"].append(st.score(Xt, yt))
        ls = LabelSpreading(kernel="rbf", gamma=20, alpha=0.2, max_iter=1000).fit(X, y_semi)
        rows["spread"].append(ls.score(Xt, yt))
    return {k: (np.mean(v), np.std(v)) for k, v in rows.items()}

seeds = range(20)
# helps: two moons, classes separated by a low-density gap (cluster assumption holds)
moons = lambda s: make_moons(n_samples=600, noise=0.08, random_state=s)
r1 = experiment(moons, 6, seeds, gamma=1.0)
# hurts: one Gaussian cloud split by a line through its densest part (cluster assumption violated)
def cloud(s):
    X, _ = make_blobs(n_samples=600, centers=[[0, 0], [3, 0]], cluster_std=1.0, random_state=s)
    y = (X[:, 1] > 0).astype(int)        # label depends on x2, while the clusters are separated along x1
    return X, y
r2 = experiment(cloud, 6, seeds, gamma=1.0)
rows = []
for name, r in [("two moons (gap between classes)", r1), ("blobs labelled across the clusters", r2)]:
    rows.append(f"{name} & {r['lr'][0]:.3f} & {r['sup'][0]:.3f} & {r['self'][0]:.3f} & {r['spread'][0]:.3f} \\\\")
out.tex("ss_rows", "\n".join(rows))
out.val("moon_sup", r1["sup"][0], 3); out.val("moon_spread", r1["spread"][0], 3)
out.val("cloud_sup", r2["sup"][0], 3); out.val("cloud_spread", r2["spread"][0], 3)
out.val("cloud_lr", r2["lr"][0], 3); out.val("moon_lr", r1["lr"][0], 3)
out.val("moon_self", r1["self"][0], 3)

# ---------------------------------------------------------------- label propagation: closed form vs sklearn
X, y = make_moons(n_samples=120, noise=0.08, random_state=3)
lab = np.r_[np.where(y == 0)[0][:2], np.where(y == 1)[0][:2]]
y_semi = np.full(len(y), -1); y_semi[lab] = y[lab]
g = 10.0
# [[harmonic]]
def harmonic(X, y_semi, gamma):
    """Label propagation fixed point: f_u = (D_uu - W_uu)^{-1} W_ul Y_l (labels clamped)."""
    W = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(-1))
    L = y_semi >= 0
    Yl = np.eye(2)[y_semi[L]]
    D = np.diag(W.sum(1))
    Fu = np.linalg.solve((D - W)[np.ix_(~L, ~L)], W[np.ix_(~L, L)] @ Yl)
    F = np.zeros((len(X), 2)); F[L] = Yl; F[~L] = Fu
    return F
# [[/harmonic]]
F = harmonic(X, y_semi, g)
lp = LabelPropagation(kernel="rbf", gamma=g, max_iter=100000, tol=1e-12).fit(X, y_semi)
out.check("harmonic closed form vs sklearn LabelPropagation (label distributions)", F, lp.label_distributions_, atol=1e-4)
out.val("lp_acc", np.mean(F.argmax(1) == y), 3)

# ---------------------------------------------------------------- InfoNCE on a tiny batch
z1 = np.array([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.1]])     # view 1 embeddings (rows)
z2 = np.array([[0.9, 0.1], [0.5, 0.9], [-0.9, -0.2]])    # view 2 of the same three items
tau = 0.5
n1 = z1 / np.linalg.norm(z1, axis=1, keepdims=True); n2 = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
S = n1 @ n2.T / tau
lse = np.log(np.exp(S).sum(1))
loss = np.mean(lse - np.diag(S))
ref = Fnn.cross_entropy(torch.tensor(S), torch.arange(3)).item()
out.check("InfoNCE by hand vs torch cross_entropy on the similarity matrix", loss, ref)
out.tex("nce_S", " \\\\ ".join(" & ".join(f"{v:.2f}" for v in row) for row in S))
out.val("nce_loss", loss, 4)
out.val("nce_p00", np.exp(S[0, 0]) / np.exp(S[0]).sum(), 4)
