"""Chapter 5: logistic regression, Newton/IRLS, softmax, OvR/OvO, imbalance, perceptron and XOR."""
import numpy as np
from scipy.special import expit, softmax
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression, Perceptron, PoissonRegressor
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- sigmoid / odds numerics
for z in [-2, 0, 0.5, 2, 4]:
    out.val(f"sig_{z}".replace(".", "p").replace("-", "m"), expit(z), 4)
out.val("or_07", np.exp(0.7), 3)
out.val("or_m05", np.exp(-0.5), 3)
# a scored applicant: z = -3 + 0.8*income + 1.2*has_job, income=2, job=1
z = -3 + 0.8 * 2 + 1.2 * 1
out.val("app_z", z, 2)
out.val("app_p", expit(z), 4)
out.val("app_odds", np.exp(z), 4)

# ---------------------------------------------------------------- one gradient step by hand
X4 = np.array([[1, 0.5], [1, 1.5], [1, 2.5], [1, 3.5]])  # intercept + x
y4 = np.array([0, 0, 1, 1])
w = np.array([0.0, 0.0])
p = expit(X4 @ w)
loss0 = -np.mean(y4 * np.log(p) + (1 - y4) * np.log(1 - p))
g = X4.T @ (p - y4) / len(y4)
w1 = w - 1.0 * g
p1 = expit(X4 @ w1)
loss1 = -np.mean(y4 * np.log(p1) + (1 - y4) * np.log(1 - p1))
out.val("h_loss0", loss0, 4)
out.tex("h_g", ", ".join(f"{v:.3f}" for v in g))
out.tex("h_w1", ", ".join(f"{v:.3f}" for v in w1))
out.tex("h_p1", ", ".join(f"{v:.3f}" for v in p1))
out.val("h_loss1", loss1, 4)
out.check("hand log-loss vs sklearn log_loss", loss1, log_loss(y4, p1))
# finite-difference check of the gradient formula
def L(wv):
    pp = expit(X4 @ wv)
    return -np.mean(y4 * np.log(pp) + (1 - y4) * np.log(1 - pp))
wt = np.array([0.3, -0.2]); e = 1e-6
num = np.array([(L(wt + e * np.eye(2)[i]) - L(wt - e * np.eye(2)[i])) / (2 * e) for i in range(2)])
out.check("logistic gradient X^T(p-y)/n vs finite differences", X4.T @ (expit(X4 @ wt) - y4) / 4, num, atol=1e-7)

# ---------------------------------------------------------------- Newton / IRLS vs sklearn
Xc, yc = make_classification(n_samples=300, n_features=3, n_informative=3, n_redundant=0,
                             class_sep=0.8, flip_y=0.1, random_state=0)
Xb = np.column_stack([np.ones(len(Xc)), Xc])
# [[irls]]
def logistic_newton(X, y, iters=10):
    w = np.zeros(X.shape[1])
    hist = []
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y)                       # gradient of the negative log-likelihood
        H = X.T @ (X * (p * (1 - p))[:, None])     # Hessian  X^T S X,  S = diag(p(1-p))
        w = w - np.linalg.solve(H, grad)
        p = 1 / (1 + np.exp(-X @ w))
        hist.append(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return w, hist
# [[/irls]]
w_nt, hist = logistic_newton(Xb, yc)
sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000).fit(Xc, yc)
out.check("Newton/IRLS logistic vs sklearn (no penalty)", w_nt, np.concatenate([sk.intercept_, sk.coef_[0]]), atol=1e-5)
out.tex("nt_hist", " \\\\\n".join(f"{i+1} & {h:.6f}" for i, h in enumerate(hist[:7])))
out.tex("nt_w", ", ".join(f"{v:.3f}" for v in w_nt))
# gradient descent for comparison: iterations to reach the same NLL within 1e-6
wg = np.zeros(4); it = 0
target = hist[-1]
while True:
    pg = expit(Xb @ wg)
    nll = -np.sum(yc * np.log(pg) + (1 - yc) * np.log(1 - pg))
    if nll - target < 1e-6 or it > 200000:
        break
    wg -= 0.5 * Xb.T @ (pg - yc) / len(yc)
    it += 1
out.val("gd_iters", it)
out.check("gradient descent logistic vs Newton", wg, w_nt, atol=1e-3)

# ---------------------------------------------------------------- 2D boundary for a figure
X2, y2 = make_classification(n_samples=120, n_features=2, n_informative=2, n_redundant=0,
                             n_clusters_per_class=1, class_sep=1.2, random_state=3)
m2 = LogisticRegression().fit(X2, y2)
out.dat("pts0", {"x": X2[y2 == 0, 0], "y": X2[y2 == 0, 1]})
out.dat("pts1", {"x": X2[y2 == 1, 0], "y": X2[y2 == 1, 1]})
b0, (b1, b2) = m2.intercept_[0], m2.coef_[0]
xs = np.linspace(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, 50)
ys = np.linspace(X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5, 50)
for level, name in [(0.5, "p50"), (0.1, "p10"), (0.9, "p90")]:
    zz = np.log(level / (1 - level))
    if abs(b1) > abs(b2):   # boundary closer to vertical: parametrise by x2
        out.dat(name, {"x": (zz - b0 - b2 * ys) / b1, "y": ys})
    else:
        out.dat(name, {"x": xs, "y": (zz - b0 - b1 * xs) / b2})
out.val("fig_xmin", xs.min(), 2); out.val("fig_xmax", xs.max(), 2)
out.val("fig_ymin", X2[:, 1].min() - 0.5, 2); out.val("fig_ymax", X2[:, 1].max() + 0.5, 2)

# ---------------------------------------------------------------- separable data -> divergence
Xs_ = np.array([[1, -2.0], [1, -1.0], [1, 1.0], [1, 2.0]]); ys_ = np.array([0, 0, 1, 1])
ws = np.zeros(2); norms = []
for t in range(1, 10001):
    ws -= 1.0 * Xs_.T @ (expit(Xs_ @ ws) - ys_) / 4
    if t in (10, 100, 1000, 10000):
        norms.append(np.linalg.norm(ws))
out.tex("sep_norms", ", ".join(f"{v:.2f}" for v in norms))
for C in (1.0, 1e4):
    out.val(f"sep_C{int(C)}", LogisticRegression(C=C, max_iter=10000).fit(Xs_[:, 1:], ys_).coef_[0, 0], 2)

# ---------------------------------------------------------------- why not MSE: gradients at z = -5 for y = 1
zq = -5.0
s = expit(zq)
out.val("ce_grad", s - 1, 4)
out.val("mse_grad", 2 * (s - 1) * s * (1 - s), 5)
zg = np.linspace(-8, 8, 161)
sg = expit(zg)
out.dat("msece", {"z": zg, "ce": -np.log(sg), "mse": (1 - sg) ** 2,
                  "gce": np.abs(sg - 1), "gmse": np.abs(2 * (sg - 1) * sg * (1 - sg))})
# non-convexity of MSE∘sigmoid in w: second derivative sign changes
d2 = np.gradient(np.gradient((1 - sg) ** 2, zg), zg)
out.val("mse_nonconvex", bool(d2.min() < -1e-3))

# ---------------------------------------------------------------- softmax regression from scratch vs sklearn
Xi, yi = load_iris(return_X_y=True)
Xi = StandardScaler().fit_transform(Xi)
K, n, d = 3, *Xi.shape
Xib = np.column_stack([np.ones(n), Xi])
Y = np.eye(K)[yi]
# [[softmax]]
def softmax_regression(X, Y, lam, lr=0.5, iters=20000):
    """min  -1/n sum log p_{i,y_i} + lam/2 ||W[1:]||^2  by gradient descent."""
    n, d = X.shape
    W = np.zeros((d, Y.shape[1]))
    for _ in range(iters):
        Z = X @ W
        Z -= Z.max(axis=1, keepdims=True)           # stable softmax
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        G = X.T @ (P - Y) / n                       # the famous (P - Y)
        G[1:] += lam * W[1:]                        # do not penalise the intercept row
        W -= lr * G
    return W
# [[/softmax]]
lam = 0.1
W = softmax_regression(Xib, Y, lam)
skm = LogisticRegression(C=1 / (lam * n), tol=1e-12, max_iter=100000).fit(Xi, yi)
P_ours = softmax(Xib @ W, axis=1)
out.check("softmax regression probabilities vs sklearn multinomial", P_ours, skm.predict_proba(Xi), atol=2e-4)
out.val("iris_acc", np.mean(P_ours.argmax(1) == yi), 3)

# OvR vs OvO
ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(Xi, yi)
ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000)).fit(Xi, yi)
out.val("ovr_n", len(ovr.estimators_)); out.val("ovo_n", len(ovo.estimators_))
out.val("ovr_acc", ovr.score(Xi, yi), 3); out.val("ovo_acc", ovo.score(Xi, yi), 3)
out.val("ovo_10", 10 * 9 // 2)

# ---------------------------------------------------------------- imbalance and thresholds
Xm, ym = make_classification(n_samples=4000, n_features=5, n_informative=3, weights=[0.95, 0.05],
                             class_sep=0.8, random_state=1)
tr, te = np.arange(0, 3000), np.arange(3000, 4000)
plain = LogisticRegression(max_iter=1000).fit(Xm[tr], ym[tr])
bal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xm[tr], ym[tr])
pp = plain.predict_proba(Xm[te])[:, 1]
rows = []
for name, pred in [("plain, $t=0.5$", pp > 0.5), ("plain, $t=0.2$", pp > 0.2),
                   ("plain, $t=0.1$", pp > 0.1), ("balanced weights, $t=0.5$", bal.predict(Xm[te]) == 1),
                   ("always predict 0", np.zeros(len(te), bool))]:
    acc = np.mean(pred == ym[te])
    rec = recall_score(ym[te], pred, zero_division=0)
    prec = precision_score(ym[te], pred, zero_division=0)
    rows.append(f"{name} & {acc:.3f} & {prec:.3f} & {rec:.3f} \\\\")
out.tex("imb_rows", "\n".join(rows))
out.val("imb_pos", ym[te].mean(), 3)

# ---------------------------------------------------------------- perceptron on AND and XOR
# [[perceptron]]
def perceptron(X, y, epochs=20, lr=1.0):
    """y in {-1,+1}; X includes a bias column. Returns weights and #mistakes per epoch."""
    w = np.zeros(X.shape[1])
    mistakes = []
    for _ in range(epochs):
        m = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi) <= 0:      # misclassified (or on the boundary)
                w += lr * yi * xi
                m += 1
        mistakes.append(m)
        if m == 0:
            break
    return w, mistakes
# [[/perceptron]]
B = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], float)
y_and = np.array([-1, -1, -1, 1]); y_xor = np.array([-1, 1, 1, -1])
w_and, m_and = perceptron(B, y_and)
w_xor, m_xor = perceptron(B, y_xor, epochs=50)
out.tex("and_w", ", ".join(f"{v:g}" for v in w_and))
out.val("and_epochs", len(m_and))
out.tex("and_m", ", ".join(str(v) for v in m_and))
out.val("xor_min_m", min(m_xor))
out.val("xor_epochs", len(m_xor))
out.check("our perceptron separates AND", np.sign(B @ w_and), y_and)
skp = Perceptron(max_iter=1000, tol=None, random_state=0).fit(B[:, 1:], y_and)
out.check("sklearn Perceptron separates AND too", skp.predict(B[:, 1:]), y_and)
skx = Perceptron(max_iter=1000, tol=None, random_state=0).fit(B[:, 1:], y_xor)
out.val("xor_sk_acc", skx.score(B[:, 1:], y_xor), 2)
# brute force: best accuracy of ANY linear classifier on XOR
ang = np.linspace(0, 2 * np.pi, 721)
best = 0
for a in ang:
    for c in np.linspace(-2, 2, 81):
        pred = np.sign(np.cos(a) * B[:, 1] + np.sin(a) * B[:, 2] + c)
        best = max(best, np.mean(pred == y_xor))
out.val("xor_best_lin", best, 2)
# with the product feature x1*x2 it becomes separable
Bx = np.column_stack([B, B[:, 1] * B[:, 2]])
w_x2, m_x2 = perceptron(Bx, y_xor, epochs=50)
out.check("perceptron with x1*x2 feature separates XOR", np.sign(Bx @ w_x2), y_xor)
out.tex("xor2_w", ", ".join(f"{v:g}" for v in w_x2))

# ---------------------------------------------------------------- Poisson GLM via IRLS vs sklearn
Xp = rng.normal(size=(400, 2)) * 0.5
lam_true = np.exp(0.3 + 0.8 * Xp[:, 0] - 0.5 * Xp[:, 1])
yp = rng.poisson(lam_true)
Xpb = np.column_stack([np.ones(400), Xp])
wp = np.zeros(3)
for _ in range(25):
    mu = np.exp(Xpb @ wp)
    wp -= np.linalg.solve(Xpb.T @ (Xpb * mu[:, None]), Xpb.T @ (mu - yp))
pr = PoissonRegressor(alpha=0, tol=1e-10, max_iter=1000).fit(Xp, yp)
out.check("Poisson regression Newton vs sklearn PoissonRegressor", wp, np.concatenate([[pr.intercept_], pr.coef_]), atol=1e-5)
out.tex("pois_w", ", ".join(f"{v:.3f}" for v in wp))
