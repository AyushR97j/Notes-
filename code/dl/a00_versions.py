"""Record library versions for the reproducibility note in the preface."""
import scipy, sklearn  # noqa: F401
import torch  # noqa: F401
from common import setup, versions

out = setup(__file__)
for k, v in versions().items():
    out.val(k, v)
