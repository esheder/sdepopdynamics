import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings

from biopop_closure.multiple_births import moments_from_vector, moment_closure, dkdt
from biopop_closure.moment_closures import moment_closure as single_birth, dkdt as single_dkdt

ifloat = st.floats(min_value=2, max_value=6)
posfloat = st.floats(min_value=1e-3, max_value=1)
smallfloat = st.floats(min_value=1e-7, max_value=1e-5)
smallint = st.integers(min_value=1, max_value=5)
multiplicity = st.lists(posfloat, min_size=1, max_size=10).map(lambda lst: np.array(lst) / np.sum(lst))
nontrivial_multiplicity = st.lists(posfloat, min_size=2, max_size=10).map(lambda lst: np.array(lst) / np.sum(lst))


@settings(deadline=None)
@given(ifloat, *(2 * [posfloat]), *(2 * [smallfloat]), nontrivial_multiplicity, smallint)
def test_moment_closure_with_multiple_births_solve_is_positive(i, a1, a2, b1, b2, m, p0):
    ts = np.linspace(0, 20, 100)
    m: tuple[float, float, float] = moments_from_vector(m)
    sol = moment_closure(dkdt, i, a1, a2, b1, b2, m, ts, p0)
    assert len(sol[0, :]) == len(ts)
    assert np.all(sol[0, :] > 0)


@given(*(3 * [posfloat]), *(2 * [smallfloat]), smallint)
def test_multiple_births_are_single_births_if_multiplicity_is_one(i, a1, a2, b1, b2, p0):
    ts = np.linspace(0, 20, 100)
    m = (1., 1., 1.)
    msol = moment_closure(dkdt, i, a1, a2, b1, b2, m, ts, p0)
    esol = single_birth(single_dkdt, i, a1, a2, b1, b2, ts, p0)
    assert np.allclose(msol, esol, rtol=1e-7)
