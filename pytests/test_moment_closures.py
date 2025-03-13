import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from biopop_closure.kolmogorov import kolmogorov_moments
from biopop_closure.matispop import badgers_params
from biopop_closure.moment_closures import moment_closure, dkdt, gaussian_distribution, saddlepoint_distribution

posfloat = st.floats(min_value=1e-3, max_value=1)
smallfloat = st.floats(min_value=1e-7, max_value=1e-5)
smallint = st.integers(min_value=1, max_value=5)


@given(*(3 * [posfloat]), *(2 * [smallfloat]), smallint)
def test_moment_closure_solve_is_positive(i, a1, a2, b1, b2, p0):
    ts = np.linspace(0, 20, 100)
    sol = moment_closure(dkdt, i, a1, a2, b1, b2, ts, p0)
    assert len(sol[0, :]) == len(ts)
    assert np.all(sol[0, :] > 0)


@given(st.floats(min_value=2, max_value=6))
def test_moment_closure_close_in_mean_to_true_solution_for_badgers(i):
    ts = np.linspace(0, 20, 100)
    a1, a2, b1, b2 = badgers_params(i)
    sol = moment_closure(dkdt, i, a1, a2, b1, b2, ts, 10)
    ref = kolmogorov_moments(i, a1, a2, b1, b2, ts, 10, nmax=100)
    assert np.allclose(sol[0, :], ref[0, :], rtol=5e-2, atol=1e-1), (sol[0, :] - ref[0, :])


@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@settings(deadline=None)
@given(st.floats(min_value=50, max_value=70))
def test_saddle_is_gaussian_no_skew_for_m2_5(m1):
    _no_skew_test(m1, 5.)


@pytest.mark.xfail(reason="Not sure why saddlepoint is so fiddly here, but it is")
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@given(st.floats(min_value=50, max_value=70),
       st.floats(min_value=5, max_value=7)
       )
def test_saddle_is_gaussian_no_skew(m1, m2):
    _no_skew_test(m1, m2)


def _no_skew_test(m1, m2):
    mom = np.array([[m1], [m2], [0.]])
    g = gaussian_distribution(mom)
    s = saddlepoint_distribution(mom)
    legal = (~np.isnan(s)) & (-np.inf < g) & (g < np.inf) & (-np.inf < s) & (s < np.inf)
    gg, ss = g[legal], s[legal]
    assert np.allclose(gg, ss, rtol=1e-2, atol=1e-10), (np.max(np.abs(gg - ss)), np.argmax(np.abs(gg - ss)))


@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@given(st.floats(min_value=50, max_value=70),
       st.floats(min_value=5, max_value=5),
       st.floats(min_value=1, max_value=2))
def test_saddle_not_gaussian_with_skew(m1, m2, m3):
    mom = np.array([[m1], [m2], [m3]])
    g = gaussian_distribution(mom)
    s = saddlepoint_distribution(mom)
    legal = ~np.isnan(s)
    g, s = g[legal], s[legal]
    assert not np.allclose(g, s)
