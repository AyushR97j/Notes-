"""Numbers used in chapter 9's problems."""
import numpy as np
from scipy.stats import entropy
from sklearn.tree import DecisionTreeRegressor
from common import setup

out = setup(__file__)
Hb = lambda c: entropy(np.array(c) / sum(c), base=2)
G = lambda c: 1 - ((np.array(c) / sum(c)) ** 2).sum()
out.val("p9a_H", Hb([10, 10, 10]), 4); out.val("p9a_G", G([10, 10, 10]), 4)
out.val("p9a_IG", Hb([10, 10, 10]) - 20 / 30 * Hb([10, 10]), 4)
P = Hb([6, 6]); PG = G([6, 6])
out.val("p9b_hA", Hb([5, 1]), 3); out.val("p9b_igA", P - Hb([5, 1]), 3)
out.val("p9b_gA", PG - G([5, 1]), 4)
out.val("p9b_hB", Hb([3, 6]), 3); out.val("p9b_igB", P - 9 / 12 * Hb([3, 6]), 3)
out.val("p9b_gB", PG - 9 / 12 * G([3, 6]), 4)
x = np.array([1.0, 2, 3, 4]); y = np.array([2.0, 4, 10, 12])
sse = []
for t in (1.5, 2.5, 3.5):
    L, R = y[x <= t], y[x > t]
    sse.append(((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum())
for i, v in enumerate(sse):
    out.val(f"p9c_s{i+1}", v, 2)
m = DecisionTreeRegressor(max_depth=1).fit(x[:, None], y)
out.check("9C best threshold and leaves vs sklearn", [m.tree_.threshold[0], m.tree_.value[1, 0, 0], m.tree_.value[2, 0, 0]], [2.5, 3, 11])
