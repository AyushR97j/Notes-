"""Chapter 2: linear algebra, probability, MLE/MAP, convexity, Lagrange — every number."""
import numpy as np
from scipy import linalg, optimize, stats
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- norms
v = np.array([3.0, -4.0, 12.0])
out.val("l1", np.abs(v).sum(), 0)
out.val("l2", np.linalg.norm(v), 0)
out.val("linf", np.abs(v).max(), 0)
out.check("L2 norm via numpy.linalg", np.sqrt((v ** 2).sum()), np.linalg.norm(v))

# ---------------------------------------------------------------- eigen / SVD of a 2x2 symmetric
S = np.array([[4.0, 2.0], [2.0, 1.0]])
lam, Q = np.linalg.eigh(S)
out.val("eig_lo", lam[0], 3)
out.val("eig_hi", lam[1], 3)
out.val("detS", np.linalg.det(S), 3)
out.val("rankS", np.linalg.matrix_rank(S))
# a PSD check matrix
P = np.array([[2.0, -1.0], [-1.0, 2.0]])
lp = np.linalg.eigvalsh(P)
out.val("P_eig1", lp[0], 3)
out.val("P_eig2", lp[1], 3)
# SVD of a 3x2
A = np.array([[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
out.val("svd1", s[0], 4)
out.val("svd2", s[1], 4)
ev = np.linalg.eigvalsh(A.T @ A)[::-1]
out.check("singular values^2 = eigenvalues of A^T A", s ** 2, ev)
out.val("ata_e1", ev[0], 3)
out.val("ata_e2", ev[1], 3)
# rank-1 truncation error = s2 (spectral) and sqrt(s2^2) frobenius
A1 = s[0] * np.outer(U[:, 0], Vt[0])
out.val("trunc_fro", np.linalg.norm(A - A1), 4)
out.check("Eckart-Young: ||A-A1||_F = sigma_2", np.linalg.norm(A - A1), s[1])

# ---------------------------------------------------------------- projection
X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
y = np.array([1.0, 2.0, 2.0])
H = X @ np.linalg.inv(X.T @ X) @ X.T
yhat = H @ y
out.tex("proj_yhat", ", ".join(f"{t:.3f}" for t in yhat))
out.val("trH", np.trace(H), 3)
out.check("hat matrix idempotent H^2 = H", H @ H, H)
out.val("resid_dot", float(X.T @ (y - yhat) @ np.ones(2)), 3)

# ---------------------------------------------------------------- gradient checks of identities
n = 4
Am = rng.normal(size=(n, n))
b = rng.normal(size=n)
x0 = rng.normal(size=n)
f = lambda x: x @ Am @ x
num = optimize.approx_fprime(x0, f, 1e-6)
out.check("grad x^T A x = (A + A^T) x", (Am + Am.T) @ x0, num, atol=1e-4)
g2 = lambda x: np.sum((Am @ x - b) ** 2)
num2 = optimize.approx_fprime(x0, g2, 1e-6)
out.check("grad ||Ax-b||^2 = 2 A^T (Ax-b)", 2 * Am.T @ (Am @ x0 - b), num2, atol=1e-4)
# log-det gradient
M = Am @ Am.T + n * np.eye(n)
eps = 1e-6
num3 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        E = np.zeros((n, n)); E[i, j] = eps
        num3[i, j] = (np.log(np.linalg.det(M + E)) - np.log(np.linalg.det(M - E))) / (2 * eps)
out.check("grad log det M = M^{-T}", np.linalg.inv(M).T, num3, atol=1e-5)

# Hessian of a 2-variable function at a point: f = x^2 + 3xy + y^2
Hf = np.array([[2.0, 3.0], [3.0, 2.0]])
he = np.linalg.eigvalsh(Hf)
out.val("saddle_e1", he[0], 0)
out.val("saddle_e2", he[1], 0)

# ---------------------------------------------------------------- probability
# Bayes: disease test
prev, sens, spec = 0.01, 0.95, 0.90
ppos = sens * prev + (1 - spec) * (1 - prev)
post = sens * prev / ppos
out.val("bayes_ppos", ppos, 4)
out.val("bayes_post", post, 4)
# simulate
N = 2_000_000
d = rng.random(N) < prev
t = np.where(d, rng.random(N) < sens, rng.random(N) < 1 - spec)
out.check("P(disease | +) simulation vs Bayes", d[t].mean(), post, atol=3e-3)

# covariance and correlation of 5 points; r^2
xs = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
ys = np.array([3.0, 7.0, 5.0, 11.0, 14.0])
cov = np.cov(xs, ys, ddof=1)[0, 1]
r = np.corrcoef(xs, ys)[0, 1]
r_s = stats.pearsonr(xs, ys).statistic
out.check("Pearson r numpy vs scipy", r, r_s)
out.val("cov", cov, 2)
out.val("sx", xs.std(ddof=1), 4)
out.val("sy", ys.std(ddof=1), 4)
out.val("r", r, 4)
out.val("r2", r ** 2, 4)
# r^2 from r for an OA: r = -0.8
out.val("oa_r", -0.8, 1)
out.val("oa_r2", 0.64, 2)
# scaling invariance of r, not of cov
out.val("cov_scaled", np.cov(100 * xs, ys)[0, 1], 0)
out.val("r_scaled", np.corrcoef(100 * xs + 7, ys)[0, 1], 4)
# zero correlation but dependence: y = x^2 on symmetric x
z = np.linspace(-1, 1, 201)
out.val("r_parabola", abs(np.corrcoef(z, z ** 2)[0, 1]), 4)

# expectations: variance of a sum, linearity
out.val("dice_mean", np.mean(np.arange(1, 7)), 1)
out.val("dice_var", np.var(np.arange(1, 7)), 4)

# LLN / CLT simulation: sample means of Exponential(1)
ns = np.array([1, 2, 5, 10, 30, 100, 1000])
sd_means = []
for m in ns:
    means = rng.exponential(1.0, size=(20000, m)).mean(axis=1)
    sd_means.append(means.std())
out.dat("clt_sd", {"n": ns, "sd": sd_means, "theory": 1 / np.sqrt(ns)})
means30 = rng.exponential(1.0, size=(50000, 30)).mean(axis=1)
hist, edges = np.histogram(means30, bins=40, range=(0.3, 1.9), density=True)
mid = (edges[:-1] + edges[1:]) / 2
out.dat("clt_hist", {"x": mid, "dens": hist, "normal": stats.norm.pdf(mid, 1, 1 / np.sqrt(30))})
out.val("clt_skew1", stats.skew(rng.exponential(1.0, 200000)), 2)
out.val("clt_skew30", stats.skew(means30), 2)

# ---------------------------------------------------------------- MLE / MAP
# Bernoulli: 7 heads in 10
h, nn = 7, 10
out.val("mle_p", h / nn, 2)
a, bb = 3, 3  # Beta(3,3) prior
out.val("map_p", (h + a - 1) / (nn + a + bb - 2), 4)
out.val("post_mean", (h + a) / (nn + a + bb), 4)
res = optimize.minimize_scalar(lambda p: -(h * np.log(p) + (nn - h) * np.log(1 - p)),
                               bounds=(1e-6, 1 - 1e-6), method="bounded")
out.check("Bernoulli MLE numeric vs h/n", res.x, h / nn, atol=1e-5)
res2 = optimize.minimize_scalar(lambda p: -((h + a - 1) * np.log(p) + (nn - h + bb - 1) * np.log(1 - p)),
                                bounds=(1e-6, 1 - 1e-6), method="bounded")
out.check("Beta-Bernoulli MAP numeric vs closed form", res2.x, (h + a - 1) / (nn + a + bb - 2), atol=1e-5)
# Gaussian MLE variance is biased: n=5 sample
samp = np.array([4.0, 7.0, 5.0, 9.0, 5.0])
out.val("g_mean", samp.mean(), 1)
out.val("g_var_mle", samp.var(ddof=0), 2)
out.val("g_var_unb", samp.var(ddof=1), 2)
mu_hat, sig_hat = stats.norm.fit(samp)
out.check("scipy norm.fit sigma^2 = MLE (ddof=0)", sig_hat ** 2, samp.var(ddof=0))

# ---------------------------------------------------------------- convexity
xx = np.linspace(-3, 3, 121)
out.dat("convex", {"x": xx, "abs": np.abs(xx), "sq": xx ** 2 / 3,
                    "maxf": np.maximum(np.abs(xx), xx ** 2 / 3),
                    "minf": np.minimum(np.abs(xx), (xx - 1.5) ** 2 / 2)})
xs2 = np.linspace(-2 * np.pi, 2 * np.pi, 161)
out.dat("sin", {"x": xs2, "y": np.sin(xs2)})
# 0-1 loss and surrogates as function of margin m = y f(x)
m = np.linspace(-2, 2.5, 181)
out.dat("losses", {"m": m, "zeroone": (m <= 0).astype(float), "hinge": np.maximum(0, 1 - m),
                    "logistic": np.log2(1 + np.exp(-m)), "sq": (1 - m) ** 2, "expo": np.exp(-m)})
# Jensen check with numbers: E[x^2] >= (E x)^2 for x in {1,2,6}
vals = np.array([1.0, 2.0, 6.0])
out.val("jensen_lhs", np.mean(vals ** 2), 2)
out.val("jensen_rhs", np.mean(vals) ** 2, 0)
# chord test for sin on [0, pi]: midpoint of chord between 0 and pi: f(pi/2)=1 > chord 0 => not convex
out.val("sin_mid", np.sin(np.pi / 2), 0)
# 0-1 loss chord violation: points m=-0.5 (loss 1) and m=0.5 (loss 0); at m=0 loss 1 > 0.5
out.val("zo_chord", 0.5, 1)

# ---------------------------------------------------------------- Lagrange: maximise xy s.t. x+y=10
res = optimize.minimize(lambda z: -z[0] * z[1], x0=[1.0, 1.0],
                        constraints=[{"type": "eq", "fun": lambda z: z[0] + z[1] - 10}])
out.check("max xy s.t. x+y=10 at (5,5)", res.x, [5.0, 5.0], atol=1e-4)
# max entropy distribution over 3 outcomes with mean constraint: E[X]=2.5 on {1,2,3}
vals3 = np.array([1.0, 2.0, 3.0])
def neg_ent(p):
    p = np.clip(p, 1e-12, 1)
    return np.sum(p * np.log(p))
res = optimize.minimize(neg_ent, x0=np.ones(3) / 3, method="SLSQP",
                        constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1},
                                     {"type": "eq", "fun": lambda p: p @ vals3 - 2.5}],
                        bounds=[(0, 1)] * 3)
# closed form: p_i ∝ exp(t x_i); solve for t
tt = optimize.brentq(lambda t: (np.exp(t * vals3) @ vals3) / np.exp(t * vals3).sum() - 2.5, -10, 10)
pcf = np.exp(tt * vals3) / np.exp(tt * vals3).sum()
out.check("max-entropy p (SLSQP vs exponential-family closed form)", res.x, pcf, atol=1e-4)
out.tex("maxent_p", ", ".join(f"{t:.3f}" for t in pcf))

# KKT example: min (x-2)^2 s.t. x <= 1  -> x*=1, mu=2
res = optimize.minimize(lambda z: (z[0] - 2) ** 2, x0=[0.0],
                        constraints=[{"type": "ineq", "fun": lambda z: 1 - z[0]}])
out.check("KKT: argmin (x-2)^2 s.t. x<=1", res.x, [1.0], atol=1e-6)

# ---------------------------------------------------------------- information theory
p = np.array([0.5, 0.25, 0.25])
q = np.array([0.4, 0.4, 0.2])
H_p = -(p * np.log2(p)).sum()
CE = -(p * np.log2(q)).sum()
KL = (p * np.log2(p / q)).sum()
out.val("Hp", H_p, 4)
out.val("CEpq", CE, 4)
out.val("KLpq", KL, 4)
out.val("KLqp", (q * np.log2(q / p)).sum(), 4)
out.check("KL(p||q) vs scipy.stats.entropy", KL, stats.entropy(p, q, base=2))
out.check("H(p) vs scipy.stats.entropy", H_p, stats.entropy(p, base=2))
# log-sum-exp stability
zz = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore"):
    naive = np.log(np.sum(np.exp(zz)))
mx = zz.max()
lse = mx + np.log(np.sum(np.exp(zz - mx)))
from scipy.special import logsumexp
out.check("log-sum-exp shift trick vs scipy", lse, logsumexp(zz))
out.val("lse", lse, 4)
out.val("lse_naive", str(naive))

# condition number
Cm = np.array([[1.0, 0.0], [0.0, 1e-4]])
out.val("cond", np.linalg.cond(Cm), 0)
