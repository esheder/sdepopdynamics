import numpy as np
import pytest

from biopop_closure.matispop import (
    badgers_params,
    badgers_params_multiple_births,
    badgers_multiplicity_vector,
    foxes_multiplicity_vector,
    _foxes_params_multiple_births_paper,
    foxes_params_multiple_births,
)


def test_badgers_params_with_paper():
    _, _, b1, _ = badgers_params(2.0)
    assert np.isclose(b1, 0.004556, rtol=1e-4)
    _, _, b1, _ = badgers_params(6.0)
    assert np.isclose(b1, 0.005667, rtol=1e-4)


def test_badgers_params_mul_with_paper():
    a1, _, b1, _ = badgers_params(2)
    a1m, _, b1m, _, m = badgers_params_multiple_births(2)
    pop = np.arange(5) + 1
    prob = badgers_multiplicity_vector
    mt = tuple(np.sum((pop**e) * prob) for e in (1, 2, 3))
    assert np.isclose(a1m, a1 / mt[0], rtol=1e-2)
    assert np.allclose(m, mt, rtol=1e-2)
    assert np.isclose(b1m, 0.004556 / mt[0], rtol=1e-2)
    _, _, b1m, _, _ = badgers_params_multiple_births(6)
    assert np.isclose(b1m, 2.37e-3, rtol=1e-2)


@pytest.mark.xfail(reason="Paper doesn't give enough precision and it comes out super wrong?")
def test_foxes_multiplicity_from_paper_meets_its_reported_values():
    pop = np.arange(10) + 1
    prob = foxes_multiplicity_vector
    _, _, _, _, m = _foxes_params_multiple_births_paper(0)
    _, _, _, _, mt = foxes_params_multiple_births(0)
    assert np.allclose(m, mt, rtol=1e-2)


def test_foxes_params_mul_with_paper():
    a1m, _, _, _, m = _foxes_params_multiple_births_paper(2)
    assert np.isclose(a1m, 0.2247, rtol=1e-4)
