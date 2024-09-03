from functools import partial

import numpy as np
import pytest

from biopop_closure.moment_closures import moment_closure, dkdt
from biopop_closure.multiple_births import moment_closure as multiple_closure, dkdt as multiple_dkdt


def badgers(i: float, t: np.ndarray, p0: float, func=partial(moment_closure, dkdt)):
    a1, a2, b1, b2, _ = badgers_params(i)
    return func(i, a1, a2, b1, b2, t, p0)


def badgers_mul(i: float, t: np.ndarray, p0: float, func=partial(multiple_closure, multiple_dkdt)):
    a1, a2, b1, b2, m = badgers_params_multiple_births(i)
    return func(i, a1, a2, b1, b2, m, t, p0)


def badgers_params(i):
    k = 60
    a1 = 0.61
    a2 = 0.37
    a = a1 - a2
    b2 = 0.
    b1 = (i + a * k) / (k ** 2)
    return a1, a2, b1, b2


def test_badgers_params_with_paper():
    _, _, b1, _ = badgers_params(2.)
    assert np.isclose(b1, 0.004556, rtol=1e-4)
    _, _, b1, _ = badgers_params(6.)
    assert np.isclose(b1, 0.005667, rtol=1e-4)


def badgers_params_multiple_births(i):
    k = 60
    m = (2.39, 6.45, 19.37)
    a1 = 0.255
    a2 = 0.37
    b2 = 0.
    a = 0.61 - a2
    b1 = (i + a * k) / (k ** 2) / m[0]
    return a1, a2, b1, b2, m


def test_badgers_params_mul_with_paper():
    a1, _, b1, _ = badgers_params(2)
    a1m, _, b1m, _, m = badgers_params_multiple_births(2)
    pop = np.arange(5) + 1
    prob = np.array((0.11, 0.51, 0.28, 0.08, 0.02))
    mt = tuple(np.sum((pop ** e) * prob) for e in (1, 2, 3))
    assert np.isclose(a1m, a1 / mt[0], rtol=1e-2)
    assert np.allclose(m, mt, rtol=1e-2)
    assert np.isclose(b1m, 0.004556 / mt[0], rtol=1e-2)
    _, _, b1m, _, _ = badgers_params_multiple_births(6)
    assert np.isclose(b1m, 2.37e-3, rtol=1e-2)


def foxes(i: float, t: np.ndarray, p0: float, func=partial(moment_closure, dkdt)):
    a1, a2, b1, b2 = foxes_params(i)
    return func(i, a1, a2, b1, b2, t, p0)


def foxes_params(i):
    a1 = 1.
    a2 = 0.5
    a = a1 - a2
    k = 100.
    b1 = 0.
    b2 = (i + a * k) / (k ** 2)
    return a1, a2, b1, b2


def foxes_params_multiple_births(i):
    a1, a2, b1, b2 = foxes_params(i)
    m = (4.45, 22.55, 126.982)
    return a1/m[0], a2, b1, b2, m


@pytest.mark.xfail(reason="Paper doesn't give enough precision and it comes out super wrong?")
def test_foxes_multiplicity():
    pop = np.arange(10) + 1
    prob = np.array((0.035, 0.07, 0.176, 0.211, 0.317, 0.087, 0.007, 0.017, 0., 0.017))
    mt = tuple(np.sum((pop ** e) * prob) for e in (1, 2, 3))
    assert np.allclose(m, mt, rtol=1e-2)


def test_foxes_params_mul_with_paper():
    a1m, _, _, _, m = foxes_params_multiple_births(2)
    assert np.isclose(a1m, 0.2247, rtol=1e-4)

