import copy

import numpy as np
import pytest

from neuralprobe.backends import Backend, make_backend, native_lib_path
from neuralprobe.config import DEFAULTS


@pytest.fixture
def cfg():
    c = copy.deepcopy(DEFAULTS)
    c["fuzz"].update({"cases_per_op": 12, "backends": ["ort-cpu"], "shrink": False, "triage": False,
                      "time_budget_s": 120})
    return c


needs_native = pytest.mark.skipif(native_lib_path() is None, reason="native kernels not built")


class MutantBackend(Backend):
    """Wraps ort-cpu and injects a specific, realistic kernel bug into its output."""

    def __init__(self, name, mutate):
        self.name = name
        self.inner = make_backend("ort-cpu")
        self.mutate = mutate

    def prepare(self, case, mat):
        run = self.inner.prepare(case, mat)
        return lambda: self.mutate(case, np.array(run(), copy=True))
