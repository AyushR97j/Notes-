"""Numbers used in chapter 15's problems."""
import numpy as np
from common import setup

out = setup(__file__)
out.val("p15a_r", np.log(100), 4); out.val("p15a_t", np.log(1000 / 990), 4); out.val("p15a_tf", 3 * np.log(100), 3)
out.val("p15b_a", (1 + 20 * 0.05) / 22, 4); out.val("p15b_b", (60 + 20 * 0.05) / 420, 4)
