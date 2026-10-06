"""Chapter 17: from-scratch NumPy implementations, each asserted against scikit-learn."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs, make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)
results = []

def record(name, diff):
    results.append((name, diff))

# ---------------------------------------------------------------- linear and ridge regression
# [[linreg]]
def fit_linear(X, y, lam=0.0):
    """OLS (lam=0) or ridge; the intercept is not penalised (fit on centred data)."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, ym - xm @ w
# [[/linreg]]
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=0)
w, b = fit_linear(X, y)
sk = LinearRegression().fit(X, y)
record("linear regression (normal equations)", out.check("OLS vs LinearRegression", np.r_[w, b], np.r_[sk.coef_, sk.intercept_]))
w, b = fit_linear(X, y, 10.0)
sk = Ridge(alpha=10.0).fit(X, y)
record("ridge regression (closed form)", out.check("ridge vs Ridge", np.r_[w, b], np.r_[sk.coef_, sk.intercept_]))

# ---------------------------------------------------------------- logistic regression by gradient descent
# [[logreg]]
def fit_logistic(X, y, lr=0.5, iters=5000):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * Xb.T @ (p - y) / len(y)        # gradient of the mean log-loss
    return w
# [[/logreg]]
Xc, yc = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0, flip_y=0.15, random_state=0)
Xc = StandardScaler().fit_transform(Xc)
w = fit_logistic(Xc, yc, iters=20000)
sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000).fit(Xc, yc)
record("logistic regression (gradient descent)", out.check("logistic GD vs LogisticRegression(C=inf)", w, np.r_[sk.intercept_, sk.coef_[0]], atol=1e-4))

# ---------------------------------------------------------------- k-NN
# [[knn]]
def knn_predict(Xtr, ytr, Xte, k=5):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)      # (n_test, n_train) squared distances
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.array([np.bincount(ytr[r]).argmax() for r in idx])
# [[/knn]]
Xi, yi = load_iris(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(Xi, yi, random_state=0)
p_ours = knn_predict(Xa, ya, Xb, 5)
record("$k$-nearest neighbours", out.check("kNN vs KNeighborsClassifier", p_ours, KNeighborsClassifier(5).fit(Xa, ya).predict(Xb)))

# ---------------------------------------------------------------- Gaussian naive Bayes
# [[gnb]]
def gnb_fit_predict_proba(Xtr, ytr, Xte, eps=1e-9):
    classes = np.unique(ytr)
    eps = eps * Xtr.var(0).max()                                  # sklearn's var_smoothing convention
    logp = []
    for c in classes:
        Xk = Xtr[ytr == c]
        mu, var = Xk.mean(0), Xk.var(0) + eps
        ll = -0.5 * (np.log(2 * np.pi * var) + (Xte - mu) ** 2 / var).sum(1)
        logp.append(np.log(len(Xk) / len(Xtr)) + ll)
    L = np.array(logp).T
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)
# [[/gnb]]
P = gnb_fit_predict_proba(Xa, ya, Xb)
record("Gaussian naive Bayes", out.check("GNB vs GaussianNB", P, GaussianNB().fit(Xa, ya).predict_proba(Xb), atol=1e-8))

# ---------------------------------------------------------------- CART decision tree (Gini)
# [[tree]]
def gini(counts):
    p = counts / counts.sum()
    return 1 - (p ** 2).sum()

def build_tree(X, y, w, depth, max_depth, n_classes):
    """w = sample weights (bootstrap counts for forests). Leaves store class weights."""
    counts = np.bincount(y, weights=w, minlength=n_classes)
    if depth == max_depth or gini(counts) == 0:
        return {"leaf": counts / counts.sum()}
    best = None
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, ys, ws = X[order, j], y[order], w[order]
        left = np.cumsum(np.eye(n_classes)[ys] * ws[:, None], axis=0)   # class weights left of each cut
        total = left[-1]
        for i in np.where(xs[1:] > xs[:-1])[0]:                         # cut between distinct values only
            L, R = left[i], total - left[i]
            score = (L.sum() * gini(L) + R.sum() * gini(R)) / total.sum()
            if best is None or score < best[0] - 1e-12:
                best = (score, j, (xs[i] + xs[i + 1]) / 2)
    if best is None or best[0] >= gini(counts) - 1e-12:
        return {"leaf": counts / counts.sum()}
    _, j, t = best
    m = X[:, j] <= t
    return {"j": j, "t": t,
            "l": build_tree(X[m], y[m], w[m], depth + 1, max_depth, n_classes),
            "r": build_tree(X[~m], y[~m], w[~m], depth + 1, max_depth, n_classes)}

def tree_proba(node, x):
    while "leaf" not in node:
        node = node["l"] if x[node["j"]] <= node["t"] else node["r"]
    return node["leaf"]
# [[/tree]]
Xd, yd = make_classification(n_samples=500, n_features=8, n_informative=5, random_state=4)
Xd = Xd.astype(np.float32).astype(float)                          # sklearn thresholds in float32
Xa, Xb, ya, yb = train_test_split(Xd, yd, random_state=0)
T = build_tree(Xa, ya, np.ones(len(ya)), 0, 4, 2)
P = np.array([tree_proba(T, x) for x in Xb])
sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xa, ya)
record("decision tree (CART, Gini, depth 4)", out.check("our tree vs DecisionTreeClassifier (test probabilities)", P, sk.predict_proba(Xb)))

# ---------------------------------------------------------------- random forest (bagging mode) using sklearn's bootstrap draws
# [[forest]]
def forest_proba(Xtr, ytr, Xte, bootstrap_indices, max_depth):
    P = 0
    for idx in bootstrap_indices:
        w = np.bincount(idx, minlength=len(ytr)).astype(float)       # bootstrap = integer sample weights
        keep = w > 0
        T = build_tree(Xtr[keep], ytr[keep], w[keep], 0, max_depth, 2)
        P = P + np.array([tree_proba(T, x) for x in Xte])
    return P / len(bootstrap_indices)
# [[/forest]]
rf = RandomForestClassifier(n_estimators=10, max_features=None, max_depth=2, random_state=0).fit(Xa, ya)
P = forest_proba(Xa, ya, Xb, rf.estimators_samples_, 2)
record("random forest (bagged trees, same bootstrap draws)", out.check("our forest vs RandomForestClassifier(max_features=None)", P, rf.predict_proba(Xb)))

# ---------------------------------------------------------------- AdaBoost with stumps
# [[adaboost]]
def adaboost(X, y, M):
    """y in {-1, +1}. Stumps are our CART trees of depth 1 fitted with sample weights."""
    w = np.full(len(y), 1 / len(y)); stumps, alphas = [], []
    yy = (y > 0).astype(int)
    for _ in range(M):
        T = build_tree(X, yy, w, 0, 1, 2)
        pred = np.where([tree_proba(T, x)[1] > 0.5 for x in X], 1, -1)
        err = w[pred != y].sum() / w.sum()
        a = np.log((1 - err) / err)                      # SAMME weight (= 2 x the 1/2-log convention)
        w = w * np.exp(a * (pred != y)); w /= w.sum()
        stumps.append(T); alphas.append(a)
    return stumps, np.array(alphas)
# [[/adaboost]]
ys = np.where(ya == 1, 1, -1)
stumps, alphas = adaboost(Xa, ys, 10)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=10, random_state=0).fit(Xa, ys)
record("AdaBoost (10 stumps)", out.check("AdaBoost weights vs AdaBoostClassifier.estimator_weights_", alphas, ada.estimator_weights_))
score = sum(a * np.where([tree_proba(T, x)[1] > 0.5 for x in Xb], 1, -1) for T, a in zip(stumps, alphas))
out.check("AdaBoost test predictions", np.sign(score), ada.predict(Xb))

# ---------------------------------------------------------------- k-means
# [[kmeans]]
def kmeans(X, C, iters=100):
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        Cn = np.array([X[lab == k].mean(0) for k in range(len(C))])
        if np.allclose(Cn, C):
            break
        C = Cn
    return C, lab
# [[/kmeans]]
Xk, _ = make_blobs(n_samples=300, centers=4, random_state=1)
C0 = Xk[:4].copy()
C, lab = kmeans(Xk, C0)
skm = KMeans(4, init=C0, n_init=1, tol=0, algorithm="lloyd").fit(Xk)
record("$k$-means (Lloyd)", out.check("k-means centroids vs KMeans", C, skm.cluster_centers_))

# ---------------------------------------------------------------- PCA via SVD
# [[pca]]
def pca(X, k):
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k] * np.sign(Vt[:k, [0]])            # fix the sign convention
    return Xc @ comps.T, comps, s[:k] ** 2 / (len(X) - 1)
# [[/pca]]
Z, comps, ev = pca(Xi, 2)
skp = PCA(2).fit(Xi)
record("PCA (SVD)", out.check("PCA explained variance and |components| vs PCA", np.r_[ev, np.abs(comps).ravel()],
                              np.r_[skp.explained_variance_, np.abs(skp.components_).ravel()]))

# ---------------------------------------------------------------- soft-margin linear SVM by SMO (max-violating pair)
# [[smo]]
def svm_smo(X, y, C=1.0, tol=1e-6, max_iter=100000):
    """Dual coordinate ascent on pairs (the working-set rule of LIBSVM's first version)."""
    n = len(y); K = X @ X.T; a = np.zeros(n); g = -np.ones(n)      # g = gradient of 1/2 a^T Q a - sum a
    Q = (y[:, None] * y[None, :]) * K
    for _ in range(max_iter):
        up = ((y > 0) & (a < C)) | ((y < 0) & (a > 0))            # can move y_i a_i up
        dn = ((y > 0) & (a > 0)) | ((y < 0) & (a < C))
        i = np.where(up)[0][np.argmax(-y[up] * g[up])]
        j = np.where(dn)[0][np.argmin(-y[dn] * g[dn])]
        if -y[i] * g[i] + y[j] * g[j] < tol:                      # KKT satisfied within tol
            break
        # move along y_i d_i = -y_j d_j that decreases the objective, then clip to the box
        quad = max(Q[i, i] + Q[j, j] - 2 * y[i] * y[j] * Q[i, j], 1e-12)
        step = (-y[i] * g[i] + y[j] * g[j]) / quad
        lo = max(-a[i] if y[i] > 0 else a[i] - C, a[j] - C if y[j] > 0 else -a[j])
        hi = min(C - a[i] if y[i] > 0 else a[i], a[j] if y[j] > 0 else C - a[j])
        step = min(max(step, lo), hi)
        di, dj = y[i] * step, -y[j] * step
        a[i] += di; a[j] += dj
        g += Q[:, i] * di + Q[:, j] * dj
    w = (a * y) @ X
    free = (a > 1e-8) & (a < C - 1e-8)
    b = np.mean(y[free] - X[free] @ w)
    return w, b, a
# [[/smo]]
Xs, ysv = make_blobs(n_samples=80, centers=[[0, 0], [2.5, 2.5]], cluster_std=1.1, random_state=2)
ysv = np.where(ysv == 1, 1.0, -1.0)
w, b, a = svm_smo(Xs, ysv, C=1.0)
svc = SVC(kernel="linear", C=1.0, tol=1e-10).fit(Xs, ysv)
record("soft-margin linear SVM (SMO)", out.check("SMO (w, b) vs SVC(kernel='linear')", np.r_[w, b], np.r_[svc.coef_[0], svc.intercept_], atol=1e-4))

# also referenced from earlier chapters
for name, where in [("lasso (coordinate descent)", "Ch. 4"), ("logistic regression (Newton/IRLS)", "Ch. 5"),
                    ("softmax regression", "Ch. 5"), ("gradient boosting (squared and log loss)", "Ch. 10"),
                    ("Gaussian mixture (EM)", "Ch. 11"), ("label propagation", "Ch. 12"), ("SMOTE", "Ch. 15")]:
    results.append((name + f" [{where}]", None))
rows = []
for name, diff in results:
    rows.append(f"{name} & {'see chapter' if diff is None else f'{diff:.1e}'} \\\\")
out.tex("check_rows", "\n".join(rows).replace("e-", "e$-$"))
out.val("n_impl", len(results))
