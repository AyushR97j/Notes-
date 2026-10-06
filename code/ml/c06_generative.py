"""Chapter 6: naive Bayes (multinomial, Bernoulli, Gaussian), Laplace smoothing, LDA/QDA."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- multinomial NB by hand
docs = ["free offer win money", "win free prize now", "meeting schedule today",
        "project meeting notes", "free lunch meeting today"]
labels = np.array([1, 1, 0, 0, 0])  # 1 = spam
test = "free meeting money"
cv = CountVectorizer()
Xc = cv.fit_transform(docs).toarray()
vocab = cv.get_feature_names_out()
V = len(vocab)
xt = cv.transform([test]).toarray()[0]
out.val("V", V)
prior = np.array([np.mean(labels == 0), np.mean(labels == 1)])
counts = np.array([Xc[labels == c].sum(0) for c in (0, 1)])
tot = counts.sum(1)
out.val("tot0", tot[0]); out.val("tot1", tot[1])
words = ["free", "meeting", "money"]
idx = [list(vocab).index(w) for w in words]
for c in (0, 1):
    for w, j in zip(words, idx):
        out.val(f"cnt_{c}_{w}", counts[c, j])
        out.val(f"p_{c}_{w}", (counts[c, j] + 1) / (tot[c] + V), 4)
logpost = np.log(prior) + (xt * np.log((counts + 1) / (tot[:, None] + V))).sum(1)
post = np.exp(logpost - logpost.max()); post /= post.sum()
sk = MultinomialNB(alpha=1.0).fit(Xc, labels)
out.check("multinomial NB posterior by hand vs sklearn MultinomialNB", post, sk.predict_proba([xt])[0])
out.val("post_spam", post[1], 4)
out.val("score0", np.exp(logpost[0]), 3 + 4)
out.val("score1", np.exp(logpost[1]), 3 + 4)
out.val("prior1", prior[1], 1); out.val("prior0", prior[0], 1)
# without smoothing: "money" never in ham -> P(ham)=0
raw = counts / tot[:, None]
out.val("nosmooth_ham", prior[0] * np.prod(raw[0] ** xt), 1)

# Bernoulli NB on the same documents
Xb = (Xc > 0).astype(int); xb = (xt > 0).astype(int)
pw = np.array([(Xb[labels == c].sum(0) + 1) / ((labels == c).sum() + 2) for c in (0, 1)])
lpb = np.log(prior) + (xb * np.log(pw) + (1 - xb) * np.log(1 - pw)).sum(1)
pb = np.exp(lpb - lpb.max()); pb /= pb.sum()
out.check("Bernoulli NB by hand vs sklearn BernoulliNB", pb, BernoulliNB(alpha=1.0).fit(Xb, labels).predict_proba([xb])[0])
out.val("bern_spam", pb[1], 4)

# ---------------------------------------------------------------- Gaussian NB by hand
Xg = np.array([[6.0, 180], [5.9, 190], [5.6, 170], [5.8, 165],
               [5.0, 100], [5.5, 150], [5.4, 130], [5.7, 150]])
yg = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 1 = male (classic height/weight toy)
xq = np.array([5.8, 160.0])
mu = np.array([Xg[yg == c].mean(0) for c in (0, 1)])
var = np.array([Xg[yg == c].var(0) for c in (0, 1)])  # MLE variances (sklearn uses ddof=0)
lik = np.array([np.prod(np.exp(-(xq - mu[c]) ** 2 / (2 * var[c])) / np.sqrt(2 * np.pi * var[c])) for c in (0, 1)])
pg = 0.5 * lik / (0.5 * lik).sum()
gnb = GaussianNB(var_smoothing=1e-12).fit(Xg, yg)
out.check("Gaussian NB by hand vs sklearn GaussianNB", pg, gnb.predict_proba([xq])[0])
out.tex("gnb_mu1", ", ".join(f"{v:.3g}" for v in mu[1]))
out.tex("gnb_mu0", ", ".join(f"{v:.3g}" for v in mu[0]))
out.tex("gnb_var1", ", ".join(f"{v:.4g}" for v in var[1]))
out.tex("gnb_var0", ", ".join(f"{v:.4g}" for v in var[0]))
out.val("gnb_p1", pg[1], 4)

# ---------------------------------------------------------------- double counting: duplicated feature
m0, m1, s = 0.0, 1.0, 1.0
xv = 0.8
rows = []
for k in [1, 2, 5, 10]:
    l1 = k * multivariate_normal(m1, s).logpdf(xv)
    l0 = k * multivariate_normal(m0, s).logpdf(xv)
    rows.append(f"{k} & {1 / (1 + np.exp(l0 - l1)):.4f} \\\\")
out.tex("dup_rows", "\n".join(rows))
true_post = 1 / (1 + np.exp(multivariate_normal(m0, s).logpdf(xv) - multivariate_normal(m1, s).logpdf(xv)))
out.val("dup_true", true_post, 4)

# ---------------------------------------------------------------- LDA / QDA from scratch vs sklearn
n = 150
X0 = rng.multivariate_normal([0, 0], [[1.0, 0.0], [0.0, 1.0]], n)
X1 = rng.multivariate_normal([2, 1.5], [[2.5, 1.2], [1.2, 1.0]], n)
Xl = np.vstack([X0, X1]); yl = np.r_[np.zeros(n), np.ones(n)]
mus = [Xl[yl == c].mean(0) for c in (0, 1)]
Sw = sum((Xl[yl == c] - mus[c]).T @ (Xl[yl == c] - mus[c]) for c in (0, 1)) / len(yl)  # sklearn pools with 1/n
pri = [0.5, 0.5]
# [[lda]]
def lda_scores(X, mus, Sigma, priors):
    """Linear discriminant: delta_k(x) = x^T S^-1 mu_k - mu_k^T S^-1 mu_k / 2 + log pi_k."""
    Si = np.linalg.inv(Sigma)
    return np.column_stack([X @ Si @ m - 0.5 * m @ Si @ m + np.log(p) for m, p in zip(mus, priors)])
# [[/lda]]
S = lda_scores(Xl, mus, Sw, pri)
P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
skl = LinearDiscriminantAnalysis(store_covariance=True).fit(Xl, yl)
out.check("LDA posteriors from scratch vs sklearn", P, skl.predict_proba(Xl), atol=1e-8)
covs = [np.cov(Xl[yl == c].T, bias=True) for c in (0, 1)]  # sklearn QDA: MLE covariances
Sq = np.column_stack([multivariate_normal(mus[c], covs[c]).logpdf(Xl) + np.log(0.5) for c in (0, 1)])
Pq = np.exp(Sq - Sq.max(1, keepdims=True)); Pq /= Pq.sum(1, keepdims=True)
skq = QuadraticDiscriminantAnalysis().fit(Xl, yl)
out.check("QDA posteriors from scratch vs sklearn", Pq, skq.predict_proba(Xl), atol=1e-8)
out.val("lda_acc", skl.score(Xl, yl), 3); out.val("qda_acc", skq.score(Xl, yl), 3)
out.val("lda_params", 2 * 2 + 3 + 1)   # two means, one shared cov (3), one prior
out.val("qda_params", 2 * 2 + 2 * 3 + 1)
# parameter counts in d dims
for d in (10, 100):
    out.val(f"lda_p{d}", 2 * d + d * (d + 1) // 2)
    out.val(f"qda_p{d}", 2 * d + 2 * d * (d + 1) // 2)
    out.val(f"gnb_p{d}", 2 * d + 2 * d)
# boundaries for the figure via contour extraction
gx, gy = np.meshgrid(np.linspace(-3.5, 6, 300), np.linspace(-3.5, 5, 300))
G = np.column_stack([gx.ravel(), gy.ravel()])
for name, model in [("ldab", skl), ("qdab", skq)]:
    Z = model.predict_proba(G)[:, 1].reshape(gx.shape)
    cs = plt.contour(gx, gy, Z, levels=[0.5])
    seg = max(cs.allsegs[0], key=len)
    out.dat(name, {"x": seg[::3, 0], "y": seg[::3, 1]})
    plt.close("all")
out.dat("cls0", {"x": X0[:80, 0], "y": X0[:80, 1]})
out.dat("cls1", {"x": X1[:80, 0], "y": X1[:80, 1]})
