"""Numbers used in chapter 13's problems."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from common import setup

out = setup(__file__)
TP, FN, FP, TN = 45, 5, 30, 420
yt = np.r_[np.ones(TP + FN), np.zeros(FP + TN)]; yp = np.r_[np.ones(TP), np.zeros(FN), np.ones(FP), np.zeros(TN)]
out.val("p13a_acc", (TP + TN) / 500, 3); out.val("p13a_p", precision_score(yt, yp), 3)
out.val("p13a_r", recall_score(yt, yp), 3); out.val("p13a_s", TN / (TN + FP), 3); out.val("p13a_f1", f1_score(yt, yp), 3)
auc = roc_auc_score([1, 1, 1, 0, 0], [0.9, 0.6, 0.4, 0.7, 0.3])
out.check("13B AUC = 4/6", auc, 4 / 6); out.val("p13b_auc", auc, 3)
prec = lambda pi: 0.9 * pi / (0.9 * pi + 0.05 * (1 - pi))
out.val("p13c_1", prec(0.1), 3); out.val("p13c_2", prec(0.005), 3)
out.val("p13d_ba", 3 * np.log(100) + 240, 2); out.val("p13d_bb", 6 * np.log(100) + 228, 2)
