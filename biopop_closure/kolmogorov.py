from typing import Sequence

import numpy as np
import scipy
from scipy.integrate import solve_ivp


def _solve_matrix(mat, t, p0: int, nmax: int) -> np.ndarray:
    pt0 = np.zeros(nmax+1, dtype=float)
    pt0[p0] = 1.

    def _dpdt(_, x):
        return mat @ x

    sol = solve_ivp(_dpdt, (t[0], t[-1]), pt0, t_eval=t)
    if not sol.success:
        raise RuntimeError(sol.message)
    psol = sol.y
    return psol


def kolmogorov(i, a1, a2, b1, b2, t, p0: int, nmax: int = 200) -> np.ndarray:
    popsize = np.arange(nmax + 1)
    ld = a1 * popsize - b1 * popsize ** 2
    ld[ld < 0] = 0.
    mu = a2 * popsize + b2 * popsize ** 2
    dm1 = ld[:-1] + i
    dm0 = -(ld + mu + i)
    dm0[0] = -i
    dm0[-1] = -mu[-1]
    dmm1 = mu[1:]
    mat = scipy.sparse.diags((dm1, dm0, dmm1), (-1, 0, 1), format='csr')
    return _solve_matrix(mat, t, p0, nmax)



def kolmogorov_multiplicity(i, a1, a2, b1, b2, m: Sequence[float], t, p0: int, nmax: int = 200) -> np.ndarray:
    popsize = np.arange(nmax+1)
    ld = a1 * popsize - b1 * popsize ** 2
    ld[ld < 0] = 0.
    mu = a2 * popsize + b2 * popsize ** 2
    dmm1 = mu[1:]
    dm0 = -(ld + mu + i)
    dm0[0] = -i
    dm0[-1] = -mu[-1]
    mlen = len(m)
    dms = [np.empty(nmax-j, dtype=float) for j in range(mlen)]
    for j, v in enumerate(dms):
        v[0] = 0
        v[1:] = ld[1:nmax-j] * m[j]
    dms[0] += i
    mat = scipy.sparse.diags((*dms, dm0, dmm1), (*[-j for j in range(1, mlen+1)], 0, 1), format='csr')
    return _solve_matrix(mat, t, p0, nmax)


def first3moments(pop: np.ndarray) -> np.ndarray:
    popsize = np.arange(pop.shape[0])
    m1 = np.sum(popsize[:, np.newaxis] * pop, axis=0)
    x2 = np.sum((popsize[:, np.newaxis] ** 2) * pop, axis=0)
    x3 = np.sum((popsize[:, np.newaxis] ** 3) * pop, axis=0)
    m2 = x2 - (m1 ** 2)
    m3 = x3 - (3 * m1 * x2) + 2 * (m1 ** 3)
    return np.vstack((m1, m2, m3))


def kolmogorov_moments(*args, func=kolmogorov, **kwargs) -> np.ndarray:
    return first3moments(func(*args, **kwargs))


EPSILON = 1e-4


def test_kolmogorov_no_change():
    pop = kolmogorov(0, 0, 0, 0, 0, np.linspace(0, 1, 20), 5, 8)
    moo = first3moments(pop)
    assert np.all(moo[0, :] == 5)
    assert np.all(moo[1:, :] == 0.), (moo[1:, :].shape, moo[1:, :])


def test_kolmogorov_is_positive_all_time():
    pop = kolmogorov(1e-3, 1., 1., 1e-3, 0., np.linspace(0, 1, 20), 5, 100)
    assert np.all(pop >= -EPSILON)


def test_kolmogorov_moments_positive_all_time():
    pop = kolmogorov(1e-3, 1., 1., 1e-3, 0., np.linspace(0, 1, 20), 5, 100)
    moments = first3moments(pop)
    assert np.all(moments[0,:] >= -EPSILON)
    assert np.all(moments[1,:] >= -EPSILON)


def test_kolmogorov_sums_to_1():
    pop = kolmogorov(1e-3, 1., 1., 1e-3, 0., np.linspace(0, 1, 20), 5, 100)
    assert np.allclose(np.sum(pop, axis=0), 1, rtol=1e-3)


def test_kolmogorov_multiplicity_is_kolmogorov_for_delta():
    pop0 = kolmogorov(1e-3, 1., 1., 1e-3, 0., np.linspace(0, 1, 20), 5, 100)
    pop1 = kolmogorov_multiplicity(1e-3, 1., 1., 1e-3, 0., [1.], np.linspace(0, 1, 20), 5, 100)
    assert np.allclose(pop0, pop1, rtol=1e-3, atol=1e-3)


def test_kolmogorov_multiplicity_differs_for_non_delta():
    pop0 = kolmogorov(1e-3, 1., 1., 1e-3, 0., np.linspace(0, 1, 20), 5, 100)
    pop1 = kolmogorov_multiplicity(1e-3, 1., 1., 1e-3, 0., [0.5, 0.5], np.linspace(0, 1, 20), 5, 100)
    assert not np.allclose(pop0, pop1, rtol=1e-3, atol=1e-3)


def test_kolmogorov_multiplicity_positive_all_time_for_example():
    pop = kolmogorov_multiplicity(1e-3, 1., 1., 1e-3, 0., [0.5, 0.5], np.linspace(0, 1, 20), 5, 100)
    assert np.all(pop >= -EPSILON)


def test_kolmogorov_multiplicity_sums_to_1():
    pop = kolmogorov_multiplicity(1e-3, 1., 1., 1e-3, 0., [0.5, 0.5], np.linspace(0, 1, 20), 5, 100)
    assert np.allclose(np.sum(pop, axis=0), 1, rtol=1e-3)


def _test_kolmogorov_multiplicity_plot():
    pop = kolmogorov_multiplicity(1e-3, 1., 1., 1e-3, 0., [0.5, 0.5], np.linspace(0, 1, 20), 5, 100)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.stairs(pop[:, -1], np.arange(pop.shape[0]+1), fill=True)
    plt.grid()
    plt.show()
    

def test_kolmogorov_multiplicity_positive_moments_all_time_for_example():
    pop = kolmogorov_multiplicity(1e-3, 1., 1., 1e-3, 0., [0.5, 0.5], np.linspace(0, 1, 20), 5, 100)
    moments = first3moments(pop)
    m1 = moments[0, :]
    m2 = moments[1, :]
    assert np.all(m1 >= 0)
    assert np.all(m2 >= 0)


