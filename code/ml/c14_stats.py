"""Chapter 14: distributions, CIs, z/t tests, power, chi-square, ANOVA, multiple testing, Simpson."""
import numpy as np
from scipy import stats
from common import setup

out = setup(__file__)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- distributions
out.val("binom3", stats.binom.pmf(3, 10, 0.2), 4)
out.check("binomial pmf by formula", 120 * 0.2 ** 3 * 0.8 ** 7, stats.binom.pmf(3, 10, 0.2))
out.val("binom_ge1", 1 - 0.8 ** 10, 4)
out.val("pois0", stats.poisson.pmf(0, 2), 4); out.val("pois_le2", stats.poisson.cdf(2, 2), 4)
out.val("norm196", stats.norm.cdf(1.96) - stats.norm.cdf(-1.96), 4)
out.val("norm1", stats.norm.cdf(1) - stats.norm.cdf(-1), 4)
out.val("expo_mem", np.exp(-0.5 * 2), 4)   # P(T > s + 2 | T > s) for rate 0.5
out.val("geom_mean", 1 / 0.2, 0)
out.val("t_crit9", stats.t.ppf(0.975, 9), 3); out.val("z_crit", stats.norm.ppf(0.975), 3)

# ---------------------------------------------------------------- confidence intervals
x = np.array([12.1, 11.4, 13.0, 12.6, 11.8, 12.9, 12.2, 11.6, 12.4, 13.1])
n = len(x); m = x.mean(); s = x.std(ddof=1); se = s / np.sqrt(n)
tc = stats.t.ppf(0.975, n - 1)
lo, hi = m - tc * se, m + tc * se
ref = stats.t.interval(0.95, n - 1, loc=m, scale=se)
out.check("t confidence interval by hand vs scipy", [lo, hi], ref)
out.val("ci_m", m, 3); out.val("ci_s", s, 4); out.val("ci_se", se, 4); out.val("ci_lo", lo, 3); out.val("ci_hi", hi, 3)
# proportion: 120 of 400 clicked
ph = 120 / 400; sep = np.sqrt(ph * (1 - ph) / 400)
out.val("pr_p", ph, 2); out.val("pr_se", sep, 4)
out.val("pr_lo", ph - 1.96 * sep, 4); out.val("pr_hi", ph + 1.96 * sep, 4)
# bootstrap percentile CI for a median
data = rng.exponential(10, 60)
res = stats.bootstrap((data,), np.median, n_resamples=4000, method="percentile", random_state=1)
boots = []
r2 = np.random.default_rng(2)
for _ in range(4000):
    boots.append(np.median(r2.choice(data, len(data))))
mine = np.percentile(boots, [2.5, 97.5])
out.check("bootstrap percentile CI (ours) vs scipy.stats.bootstrap", mine,
          [res.confidence_interval.low, res.confidence_interval.high], atol=0.6)
out.val("boot_med", np.median(data), 2); out.val("boot_lo", mine[0], 2); out.val("boot_hi", mine[1], 2)

# ---------------------------------------------------------------- z test and t tests
z = (52 - 50) / (8 / np.sqrt(64))
out.val("z", z, 2); out.val("z_p", 2 * stats.norm.sf(abs(z)), 4)
t1 = (m - 12.0) / se
r = stats.ttest_1samp(x, 12.0)
out.check("one-sample t by hand vs scipy", [t1, 2 * stats.t.sf(abs(t1), n - 1)], [r.statistic, r.pvalue])
out.val("t1", t1, 3); out.val("t1_p", r.pvalue, 4)
a = np.array([78.0, 85, 82, 90, 88, 76, 84]); b = np.array([72.0, 80, 79, 74, 81, 70, 77, 75])
va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
tw = (a.mean() - b.mean()) / np.sqrt(va + vb)
dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
rw = stats.ttest_ind(a, b, equal_var=False)
out.check("Welch t and p by hand vs scipy", [tw, 2 * stats.t.sf(abs(tw), dfw)], [rw.statistic, rw.pvalue])
out.val("tw", tw, 3); out.val("dfw", dfw, 2); out.val("tw_p", rw.pvalue, 4)
out.val("ma", a.mean(), 2); out.val("mb", b.mean(), 2)
before = np.array([70.0, 82, 65, 90, 75, 68]); after = np.array([73.0, 85, 66, 94, 79, 70])
dd = after - before
tp = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
rp = stats.ttest_rel(after, before)
out.check("paired t by hand vs scipy", tp, rp.statistic)
out.val("tp", tp, 3); out.val("tp_p", rp.pvalue, 5)
ri = stats.ttest_ind(after, before)
out.val("tp_wrong_p", ri.pvalue, 4)

# ---------------------------------------------------------------- power and sample size (two proportions)
p1, p2, alpha, power = 0.10, 0.12, 0.05, 0.80
za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
pbar = (p1 + p2) / 2
n_per = ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1)) ** 2
n_per = int(np.ceil(n_per))
out.val("ab_n", n_per)
# simulate power at that n
hits = 0; sims = 4000
for _ in range(sims):
    c1 = rng.binomial(n_per, p1); c2 = rng.binomial(n_per, p2)
    pp = (c1 + c2) / (2 * n_per)
    zz = (c2 / n_per - c1 / n_per) / np.sqrt(pp * (1 - pp) * 2 / n_per)
    hits += abs(zz) > za
out.check("simulated power at the formula's n is about 0.80", hits / sims, 0.80, atol=0.03)
out.val("ab_power_sim", hits / sims, 3)
# one-sample z-test power for effect 2, sigma 8, n 64
d = 2 / (8 / np.sqrt(64))
out.val("pow_z", stats.norm.sf(za - d) + stats.norm.cdf(-za - d), 4)

# ---------------------------------------------------------------- chi-square test of independence (2x2) and GOF
T = np.array([[30, 70], [45, 55]])   # rows: app version A/B; cols: converted yes/no
E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
chi = ((T - E) ** 2 / E).sum()
cr = stats.chi2_contingency(T, correction=False)
out.check("chi-square statistic by hand vs scipy (no continuity correction)", chi, cr.statistic)
out.tex("chi_E", ", ".join(f"{v:g}" for v in E.ravel()))
out.val("chi", chi, 4); out.val("chi_p", cr.pvalue, 4)
out.val("chi_yates_p", stats.chi2_contingency(T).pvalue, 4)
obs = np.array([8, 12, 9, 11, 6, 14]); gof = stats.chisquare(obs)
out.check("die goodness of fit by hand vs scipy", ((obs - 10) ** 2 / 10).sum(), gof.statistic)
out.val("gof", gof.statistic, 2); out.val("gof_p", gof.pvalue, 4)

# ---------------------------------------------------------------- one-way ANOVA by hand
g1 = np.array([5.0, 6, 7, 6]); g2 = np.array([8.0, 9, 7, 8]); g3 = np.array([6.0, 5, 6, 7])
allv = np.r_[g1, g2, g3]; gm = allv.mean()
ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in (g1, g2, g3))
ssw = sum(((g - g.mean()) ** 2).sum() for g in (g1, g2, g3))
F = (ssb / 2) / (ssw / 9)
fr = stats.f_oneway(g1, g2, g3)
out.check("one-way ANOVA F by hand vs scipy", F, fr.statistic)
out.val("ssb", ssb, 3); out.val("ssw", ssw, 3); out.val("F", F, 3); out.val("F_p", fr.pvalue, 4)

# ---------------------------------------------------------------- multiple testing
out.val("fwer20", 1 - 0.95 ** 20, 4)
pv = np.array([0.001, 0.008, 0.012, 0.030, 0.041, 0.20, 0.35, 0.60])
bonf = pv < 0.05 / len(pv)
order = np.argsort(pv); m_ = len(pv); kmax = 0
for rank, i in enumerate(order, 1):
    if pv[i] <= rank / m_ * 0.05:
        kmax = rank
bh = np.zeros(m_, bool); bh[order[:kmax]] = True
adj = stats.false_discovery_control(pv, method="bh")
out.check("Benjamini-Hochberg rejections by hand vs scipy adjusted p < 0.05", bh.astype(float), (adj < 0.05).astype(float))
out.val("n_raw", int((pv < 0.05).sum())); out.val("n_bonf", int(bonf.sum())); out.val("n_bh", int(bh.sum()))
out.tex("bh_thr", ", ".join(f"{(k + 1) / m_ * 0.05:.5f}".rstrip("0") for k in range(m_)))

# ---------------------------------------------------------------- Pearson vs Spearman with an outlier
u = np.arange(1.0, 11); v = u + rng.normal(0, 0.5, 10); v[-1] = -20
out.val("pear_out", stats.pearsonr(u, v).statistic, 3); out.val("spear_out", stats.spearmanr(u, v).statistic, 3)

# ---------------------------------------------------------------- Simpson's paradox
# treatment success counts (success, total) for small and large kidney stones
A = {"small": (81, 87), "large": (192, 263)}; B = {"small": (234, 270), "large": (55, 80)}
for g in ("small", "large"):
    out.val(f"simA_{g}", A[g][0] / A[g][1], 3); out.val(f"simB_{g}", B[g][0] / B[g][1], 3)
out.val("simA_all", sum(v[0] for v in A.values()) / sum(v[1] for v in A.values()), 3)
out.val("simB_all", sum(v[0] for v in B.values()) / sum(v[1] for v in B.values()), 3)
