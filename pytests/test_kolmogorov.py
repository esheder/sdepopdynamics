import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings, assume

from biopop_closure.kolmogorov import kolmogorov, first3moments, kolmogorov_multiplicity, kolmogorov_matrix, \
    kolmogorov_multiplicity_matrix

posfloat = st.floats(min_value=1e-3, max_value=1)
smallint = st.integers(min_value=1, max_value=5)
largeint = st.integers(min_value=30, max_value=100)
regular = st.tuples(posfloat, posfloat, largeint)
brlaw = regular.flatmap(lambda tpl: st.tuples(
    st.just(tpl[0]),
    st.just(tpl[1]),
    st.floats(min_value=tpl[0] / tpl[-1] / 100, max_value=tpl[0] / tpl[-1]),
    st.floats(min_value=tpl[1] / tpl[-1] / 100, max_value=tpl[1] / tpl[-1]),
    st.just(tpl[-1])
))
multiplicity = st.lists(posfloat, min_size=1, max_size=10).map(lambda lst: np.array(lst) / np.sum(lst))
nontrivial_multiplicity = st.lists(posfloat, min_size=2, max_size=10).map(lambda lst: np.array(lst) / np.sum(lst))

EPSILON = 1e-4


@given(smallint, largeint)
def test_kolmogorov_no_change(p0, nmax):
    pop = kolmogorov(0, 0, 0, 0, 0, np.linspace(0, 1, 20), p0, nmax)
    moo = first3moments(pop)
    assert np.all(moo[0, :] == p0)
    assert np.all(moo[1:, :] == 0.), (moo[1:, :].shape, moo[1:, :])


@settings(deadline=None)
@given(*(3 * [posfloat] + [smallint, multiplicity, largeint]))
def test_kolmogorov_multiplicity_is_positive_all_time(i, a1, a2, p0, m, nmax):
    pop = kolmogorov_multiplicity(i, a1, a2, 0, 0, m, np.linspace(0, 1, 20), p0, nmax)
    assert np.all(pop >= -EPSILON)
    moments = first3moments(pop)
    assert np.all(moments[0, :] >= -EPSILON)
    assert np.all(moments[1, :] >= -EPSILON)


@settings(deadline=None)
@given(posfloat, brlaw, smallint)
def test_kolmogorov_sums_to_1(i, law, p0):
    a1, a2, b1, b2, nmax = law
    pop = kolmogorov(i, a1, a2, b1, b2, np.linspace(0, 1, 20), p0, nmax)
    assert np.allclose(np.sum(pop, axis=0), 1, rtol=1e-3)


@settings(deadline=None)
@given(posfloat, brlaw, multiplicity, smallint)
def test_kolmogorov_multiplicity_sums_to_1(i, law, m, p0):
    a1, a2, b1, b2, nmax = law
    pop = kolmogorov_multiplicity(i, a1, a2, b1, b2, m, np.linspace(0, 1, 20), p0, nmax)
    assert np.allclose(np.sum(pop, axis=0), 1, rtol=1e-3)


@settings(deadline=None)
@given(posfloat, brlaw, smallint)
def test_kolmogorov_multiplicity_is_kolmogorov_for_delta(i, law, p0):
    a1, a2, b1, b2, nmax = law
    pop0 = kolmogorov(i, a1, a2, b1, b2, np.linspace(0, 1, 20), p0, nmax)
    pop1 = kolmogorov_multiplicity(i, a1, a2, b1, b2, [1.], np.linspace(0, 1, 20), p0, nmax)
    assert np.allclose(pop0, pop1, rtol=1e-3, atol=1e-3)


def test_kolmogorov_matrix_small_example():
    i, a1, a2, = 1, 10, 5
    mat = kolmogorov_matrix(i, a1, a2, 0, 0, nmax=3).todense()
    handsol = np.array([[-i, a2, 0, 0],
                        [i, -i - a1 - a2, a2 * 2, 0],
                        [0, i + a1, -i - 2 * (a1 + a2), a2 * 3],
                        [0, 0, i + 2 * a1, -a2 * 3]
                        ])
    assert np.allclose(mat, handsol, rtol=1e-3, atol=1e-3), (mat - handsol, mat, handsol)


def test_kolmogorov_multiplicity_matrix_small_example():
    i, a1, a2, = 1, 10, 5
    mu1 = 0.5
    mat = kolmogorov_multiplicity_matrix(i, a1, a2, 0, 0, [mu1, 1 - mu1], nmax=3).todense()
    handsol = np.array([[-i, a2, 0, 0],
                        [i, -i - a1 - a2, a2 * 2, 0],
                        [0, i + a1 * mu1, -i - 2 * (a1 * mu1 + a2), a2 * 3],
                        [0, a1 * (1 - mu1), i + 2 * a1 * mu1, -a2 * 3]
                        ])
    assert np.allclose(mat, handsol, rtol=1e-3, atol=1e-3), (mat - handsol, mat, handsol)


@settings(deadline=None, max_examples=50)
@given(posfloat, brlaw, nontrivial_multiplicity)
def test_kolmogorov_multiplicity_differs_by_matrix_for_non_delta_known_way(i, law, m):
    a1, a2, b1, b2, nmax = law
    assume(m[0] < 0.8)
    mat0 = kolmogorov_matrix(i, a1, a2, b1, b2, nmax=nmax).todense()
    mat1 = kolmogorov_multiplicity_matrix(i, a1, a2, b1, b2, m, nmax=nmax).todense()
    for j in range(mat0.shape[0]):
        if j < mat0.shape[0] - 1:
            assert np.allclose(mat0[j, j + 1:], mat1[j, j + 1:], rtol=1e-3, atol=1e-6), (j, mat1[j, j:])
        if j > 1:
            ivec = np.zeros_like(mat0[j, :j])
            ivec[-1] = i
            assert (not np.allclose(mat0[j, :j] - ivec, mat1[j, :j] - ivec, rtol=1e-3, atol=1e-6)), (j, mat1[j, :j])
