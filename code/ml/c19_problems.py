"""Every number in the end-of-book problem set (Chapter 19), with library cross-checks where one exists."""
from math import comb

import numpy as np
from scipy import stats
from scipy.special import expit, softmax
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from common import setup

out = setup(__file__)
V = out.val

# P1 fit a line
x = np.array([2.0, 4, 6, 8, 10]); y = np.array([3.0, 5, 6, 9, 12])
m = LinearRegression().fit(x[:, None], y)
b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum(); b0 = y.mean() - b1 * x.mean()
out.check("P1 slope/intercept vs sklearn", [b1, b0], [m.coef_[0], m.intercept_])
V("p1_sxy", ((x - x.mean()) * (y - y.mean())).sum(), 0); V("p1_sxx", ((x - x.mean()) ** 2).sum(), 0)
V("p1_b1", b1, 3); V("p1_b0", b0, 2); V("p1_ybar", y.mean(), 1); V("p1_pred", b0 + b1 * 7, 2)
V("p1_r2", m.score(x[:, None], y), 4)
# P2 r from slope
V("p2_r", 1.5 * 4 / 10, 2); V("p2_r2", (1.5 * 4 / 10) ** 2, 2)
# P3 MSE of a line
yh = 1 + 2 * np.array([0.0, 1, 2, 3]); yt = np.array([1.5, 2.5, 5.5, 6.5])
V("p3_mse", np.mean((yt - yh) ** 2), 4); V("p3_mae", np.mean(np.abs(yt - yh)), 2)
# P4 ridge 1-D
V("p4_ols", 12 / 8, 2); V("p4_ridge", 12 / (8 + 4), 2)
# P5 soft threshold
z = np.array([1.8, -0.3, 0.6]); st = np.sign(z) * np.maximum(np.abs(z) - 0.5, 0)
out.tex("p5_out", ", ".join(f"{v + 0.0:g}" for v in st)); out.tex("p5_ridge", ", ".join(f"{v:.2f}" for v in z / 1.5))
# P6 adjusted R^2
V("p6_a", 1 - (1 - 0.62) * 29 / 27, 4); V("p6_b", 1 - (1 - 0.66) * 29 / 23, 4)
# P8 neuron
w = np.array([1.2, -0.7, 0.5]); xx = np.array([1.0, 2.0, 3.0]); zz = w @ xx - 0.4
V("p8_z", zz, 2); V("p8_p", expit(zz), 4); V("p8_relu", max(zz, 0), 2); V("p8_tanh", np.tanh(zz), 4)
# P9 log-loss
yl = np.array([1, 0, 1, 0]); pl = np.array([0.9, 0.2, 0.6, 0.7])
ll = -np.mean(yl * np.log(pl) + (1 - yl) * np.log(1 - pl))
out.check("P9 log-loss vs sklearn", ll, log_loss(yl, pl)); V("p9_ll", ll, 4)
out.tex("p9_terms", ", ".join(f"{v:.4f}" for v in -(yl * np.log(pl) + (1 - yl) * np.log(1 - pl))))
# P10 odds ratio
odds = 0.1 / 0.9 * np.exp(-0.35 * 2); V("p10_or", np.exp(-0.35 * 2), 4); V("p10_p", odds / (1 + odds), 4)
# P11 logistic GD step
X11 = np.array([[1, 2.0], [1, -1.0]]); y11 = np.array([1, 0]); w11 = np.array([0.0, 0.5])
p11 = expit(X11 @ w11); g11 = X11.T @ (p11 - y11) / 2; n11 = w11 - 1.0 * g11
out.tex("p11_p", ", ".join(f"{v:.4f}" for v in p11)); out.tex("p11_g", ", ".join(f"{v:.4f}" for v in g11))
out.tex("p11_w", ", ".join(f"{v:.4f}" for v in n11))
# P12 softmax
lg = np.array([2.0, 1.0, -1.0]); sm = softmax(lg)
out.tex("p12_sm", ", ".join(f"{v:.4f}" for v in sm)); V("p12_ce", -np.log(sm[1]), 4); V("p12_ce0", -np.log(sm[0]), 4)
# P13 perceptron update
w13 = np.array([0.5, -1.0, 0.2]); x13 = np.array([1.0, 1.0, 2.0]); y13 = 1
V("p13_m", y13 * (w13 @ x13), 2); out.tex("p13_w", ", ".join(f"{v:g}" for v in w13 + y13 * x13))
# P15 naive Bayes with smoothing: counts
cnt = {"pos": {"great": 8, "boring": 1, "plot": 3}, "neg": {"great": 2, "boring": 6, "plot": 4}}
tot = {"pos": 40, "neg": 50}; Vv = 30; prior = {"pos": 0.5, "neg": 0.5}
sc = {c: prior[c] * np.prod([(cnt[c][w_] + 1) / (tot[c] + Vv) for w_ in ("great", "plot")]) for c in cnt}
V("p15_pos", sc["pos"], 6); V("p15_neg", sc["neg"], 6); V("p15_post", sc["pos"] / (sc["pos"] + sc["neg"]), 4)
for c in cnt:
    for w_ in ("great", "plot"):
        V(f"p15_{c}_{w_}", (cnt[c][w_] + 1) / (tot[c] + Vv), 4)
# P16 Gaussian NB 1-D
d1 = 0.6 * stats.norm.pdf(5, 4, 1); d2 = 0.4 * stats.norm.pdf(5, 7, 2)
V("p16_d1", d1, 4); V("p16_d2", d2, 4); V("p16_post", d1 / (d1 + d2), 4)
# P17 kNN
P17 = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [5, 6]], float); L17 = np.array([0, 0, 0, 1, 1, 1])
q = np.array([3.5, 3.8]); d17 = np.sqrt(((P17 - q) ** 2).sum(1))
out.tex("p17_d", ", ".join(f"{v:.3f}" for v in d17))
for k in (1, 3, 5):
    V(f"p17_k{k}", int(KNeighborsClassifier(k).fit(P17, L17).predict([q])[0]))
# P18 distances
a, b = np.array([2.0, -1, 3]), np.array([0.0, 1, 4])
V("p18_e", np.linalg.norm(a - b), 4); V("p18_m", np.abs(a - b).sum(), 0)
V("p18_cos", a @ b / np.linalg.norm(a) / np.linalg.norm(b), 4); V("p18_ch", np.abs(a - b).max(), 0)
# P19 curse
V("p19_e", 0.05 ** (1 / 50), 3)
# P20 SVM margin
w20 = np.array([1.0, -2.0, 2.0]); V("p20_w", 2 / np.linalg.norm(w20), 4); V("p20_f", w20 @ np.array([1, 1, 1.0]) + 0.5, 1)
# P21 kernel
V("p21_k", (np.array([1, 2.0]) @ np.array([2, -1.0]) + 2) ** 2, 0); V("p21_rbf", np.exp(-0.5 * 10), 5)
# P22 hinge
f22 = np.array([2.0, 0.5, -0.3, -1.5]); y22 = np.array([1, 1, 1, -1])
h22 = np.maximum(0, 1 - y22 * f22); out.tex("p22_h", ", ".join(f"{v:g}" for v in h22)); V("p22_mean", h22.mean(), 3)
# P23 information gain (counts)
H = lambda c: stats.entropy(np.array(c) / sum(c), base=2)
par = [9, 7]; L, R = [7, 1], [2, 6]
V("p23_H", H(par), 4); V("p23_HL", H(L), 4); V("p23_HR", H(R), 4)
V("p23_IG", H(par) - 8 / 16 * H(L) - 8 / 16 * H(R), 4)
# P24 Gini
G = lambda c: 1 - ((np.array(c) / sum(c)) ** 2).sum()
V("p24_par", G([10, 10]), 3); V("p24_split", (12 * G([9, 3]) + 8 * G([1, 7])) / 20, 4)
V("p24_dec", G([10, 10]) - (12 * G([9, 3]) + 8 * G([1, 7])) / 20, 4)
# P25 best threshold (Gini) on 6 points
x25 = np.array([1.0, 2, 3, 4, 5, 6]); y25 = np.array([0, 0, 0, 1, 0, 1])
best = min(((len(y25[x25 <= t]) * G(np.bincount(y25[x25 <= t], minlength=2)) + len(y25[x25 > t]) * G(np.bincount(y25[x25 > t], minlength=2))) / 6, t)
           for t in (x25[:-1] + x25[1:]) / 2)
V("p25_t", best[1], 1); V("p25_g", best[0], 4)
# P26 regression split
x26 = np.array([1.0, 2, 3, 4, 5]); y26 = np.array([3.0, 4, 3, 10, 11])
rt = DecisionTreeRegressor(max_depth=1).fit(x26[:, None], y26)
V("p26_t", rt.tree_.threshold[0], 1); V("p26_l", rt.tree_.value[1, 0, 0], 3); V("p26_r", rt.tree_.value[2, 0, 0], 2)
L26, R26 = y26[x26 <= 3.5], y26[x26 > 3.5]
V("p26_sse", ((L26 - L26.mean()) ** 2).sum() + ((R26 - R26.mean()) ** 2).sum(), 3)
# P27 bootstrap n=10
V("p27_abs", 0.9 ** 10, 4); V("p27_uniq", 10 * (1 - 0.9 ** 10), 3)
# P28 exactly twice, n=5
V("p28", comb(5, 2) * 0.2 ** 2 * 0.8 ** 3, 4)
# P29 AdaBoost: 8 points, 2 misclassified
e29 = 2 / 8; a29 = 0.5 * np.log((1 - e29) / e29)
V("p29_a", a29, 4); V("p29_wrong", 1 / 4, 3); V("p29_right", 1 / 12, 4)
# P30 alpha values
V("p30_a01", 0.5 * np.log(0.9 / 0.1), 4)
# P31 gradient boosting by hand
y31 = np.array([10.0, 14, 18, 30]); F0 = y31.mean(); r31 = y31 - F0
V("p31_F0", F0, 0); out.tex("p31_r", ", ".join(f"{v:g}" for v in r31))
st31 = DecisionTreeRegressor(max_depth=1).fit(np.arange(4.0)[:, None], r31)
upd = st31.predict(np.arange(4.0)[:, None])
out.tex("p31_u", ", ".join(f"{v:g}" for v in upd)); out.tex("p31_F1", ", ".join(f"{v:g}" for v in F0 + 0.1 * upd))
# P32 XGBoost leaf and gain, squared loss
gL, gR, hL, hR, lam = np.array([-3.0, -2, -4]).sum(), np.array([1.0, 2]).sum(), 3, 2, 1
V("p32_wL", -gL / (hL + lam), 3); V("p32_wR", -gR / (hR + lam), 3)
V("p32_gain", 0.5 * (gL ** 2 / (hL + lam) + gR ** 2 / (hR + lam) - (gL + gR) ** 2 / (hL + hR + lam)), 4)
# P33 XGBoost logistic leaf
p33 = np.array([0.2, 0.3, 0.6]); y33 = np.array([1, 1, 0])
g33 = (p33 - y33).sum(); h33 = (p33 * (1 - p33)).sum()
V("p33_G", g33, 2); V("p33_H", h33, 2); V("p33_w", -g33 / (h33 + 1), 4)
# P34 forest variance
V("p34_100", 0.4 * 9 + 0.6 * 9 / 100, 3); V("p34_inf", 0.4 * 9, 1)
# P37 k-means 1-D
x37 = np.array([1.0, 3, 4, 10, 11, 15]); C0 = np.array([[3.0], [11.0]])
km = KMeans(2, init=C0, n_init=1, max_iter=1).fit(x37[:, None])
c37 = np.sort(km.cluster_centers_.ravel()); V("p37_c1", c37[0], 3); V("p37_c2", c37[1], 2)
inert0 = ((x37[:, None] - C0.T) ** 2).min(1).sum(); V("p37_i0", inert0, 0)
lab = np.abs(x37[:, None] - c37[None]).argmin(1); V("p37_i1", ((x37 - c37[lab]) ** 2).sum(), 3)
# P39 silhouette
Xs = np.array([[0.0, 0], [0, 2], [4, 0], [4, 2]]); ls = np.array([0, 0, 1, 1])
V("p39_s", silhouette_samples(Xs, ls)[0], 4); V("p39_b", (4 + np.sqrt(20)) / 2, 4)
# P40 hierarchical
x40 = np.array([2.0, 3, 7, 8, 15])
out.tex("p40_single", ", ".join(f"{h:g}" for h in linkage(x40[:, None], "single")[:, 2]))
out.tex("p40_complete", ", ".join(f"{h:g}" for h in linkage(x40[:, None], "complete")[:, 2]))
# P41/42 PCA
S = np.array([[5.0, 2], [2, 2]]); ev, evec = np.linalg.eigh(S)
V("p41_l1", ev[1], 0); V("p41_l2", ev[0], 0); V("p41_evr", ev[1] / ev.sum(), 4)
v1 = evec[:, 1] * np.sign(evec[0, 1]); out.tex("p41_v", ", ".join(f"{v:.4f}" for v in v1))
V("p42_z", np.array([3.0, 1.0]) @ v1, 4)
# P44 GMM responsibility
d44 = np.array([0.5, 0.5]) * stats.norm.pdf(2.0, [0.0, 3.0], [1.0, 2.0]); V("p44_r1", d44[0] / d44.sum(), 4)
# P45 association
V("p45_sup", 60 / 500, 2); V("p45_conf", 60 / 150, 2); V("p45_lift", (60 / 150) / (200 / 500), 2)
# P46 confusion matrix
TP, FP, FN, TN = 70, 30, 10, 890
yt46 = np.r_[np.ones(TP + FN), np.zeros(FP + TN)]; yp46 = np.r_[np.ones(TP), np.zeros(FN), np.ones(FP), np.zeros(TN)]
V("p46_acc", (TP + TN) / 1000, 3); V("p46_p", precision_score(yt46, yp46), 3); V("p46_r", recall_score(yt46, yp46), 3)
V("p46_f1", f1_score(yt46, yp46), 4); V("p46_spec", TN / (TN + FP), 4)
# P47 macro vs micro: per-class (TP, FP, FN)
cl = [(90, 10, 10), (5, 5, 15)]
f1s = [2 * t / (2 * t + f + n) for t, f, n in cl]
V("p47_f1a", f1s[0], 3); V("p47_f1b", f1s[1], 3); V("p47_macro", np.mean(f1s), 4)
T, Fp, Fn = (sum(c[i] for c in cl) for i in range(3)); V("p47_micro", 2 * T / (2 * T + Fp + Fn), 4)
# P48 AUC
s48 = [0.8, 0.7, 0.4, 0.6, 0.3, 0.2]; l48 = [1, 1, 1, 0, 0, 0]
V("p48_auc", roc_auc_score(l48, s48), 4)
# P50 precision vs prevalence
pr = lambda pi: 0.8 * pi / (0.8 * pi + 0.1 * (1 - pi)); V("p50_a", pr(0.2), 4); V("p50_b", pr(0.01), 4)
# P51 AIC/BIC
for nm, k, l in [("a", 4, -210.0), ("b", 9, -201.0)]:
    V(f"p51_aic_{nm}", 2 * k - 2 * l, 1); V(f"p51_bic_{nm}", k * np.log(200) - 2 * l, 2)
# P52 fits count
V("p52_fits", 4 * 3 * 5 + 5); V("p52_nested", 3 * (4 * 3 * 5 + 5))
# P54 TF-IDF
V("p54_idf", np.log(500 / 25), 4); V("p54_tfidf", 4 * np.log(500 / 25), 3); V("p54_sk", np.log(501 / 26) + 1, 4)
# P55 target encoding
V("p55_a", (4 * 0.75 + 10 * 0.2) / 14, 4)
# P56 scaling
V("p56_z", (82 - 70) / 8, 2); V("p56_mm", (82 - 40) / (100 - 40), 2)
# P57 CI for a proportion
ph = 0.18; se = np.sqrt(ph * (1 - ph) / 900); V("p57_se", se, 4); V("p57_lo", ph - 1.96 * se, 4); V("p57_hi", ph + 1.96 * se, 4)
# P58 z-test
z58 = (103 - 100) / (15 / np.sqrt(100)); V("p58_z", z58, 1); V("p58_p", 2 * stats.norm.sf(z58), 4)
# P59 chi-square
T59 = np.array([[40, 60], [20, 80]]); cr = stats.chi2_contingency(T59, correction=False)
V("p59_chi", cr.statistic, 3); V("p59_p", cr.pvalue, 4)
out.tex("p59_E", ", ".join(f"{v:g}" for v in cr.expected_freq.ravel()))
# P60 FWER
V("p60_fwer", 1 - 0.95 ** 10, 4); V("p60_bonf", 0.05 / 10, 3)
# P61 GD bound
V("p61_eta", 2 / 6, 4)
# P63 limit
V("p63_100", 1 - 0.99 ** 100, 4)
