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
    dmm1 = mu[1:]
    mat = scipy.sparse.diags((dm1, dm0, dmm1), (-1, 0, 1), format='csr')
    return _solve_matrix(mat, t, p0, nmax)



def kolmogorov_multiplicity(i, a1, a2, b1, b2, m: Sequence[float], t, p0: int, nmax: int = 200) -> np.ndarray:
    popsize = np.arange(nmax+1)
    ld = a1 * popsize - b1 * popsize ** 2
    ld[ld < 0] = 0.
    mu = a2 * popsize + b2 * popsize ** 2
    dm1 = ld[:-1] + i
    dm0 = -(ld + mu + i)
    dm0[0] = -i
    dms = [np.empty(nmax-i, dtype=float) for i in range(len(m))]
    for i, v in enumerate(dms):
        v[:] = mu[i+1:] * m[i]
    mat = scipy.sparse.diags((dm1, dm0, *dms), (-1, 0, *list(range(1, len(dms)+1))), format='csr')
    return _solve_matrix(mat, t, p0, nmax)


def first3moments(pop: np.ndarray) -> np.ndarray:
    popsize = np.arange(pop.shape[0])
    m1 = np.sum(popsize[:, np.newaxis] * pop, axis=0)
    x2 = np.sum((popsize[:, np.newaxis] ** 2) * pop, axis=0)
    x3 = np.sum((popsize[:, np.newaxis] ** 3) * pop, axis=0)
    m2 = x2 - m1 ** 2
    m3 = x3 - (3 * m1 * x2) + 2 * (m1 ** 3)
    return np.vstack((m1, m2, m3))


def kolmogorov_moments(*args, func=kolmogorov, **kwargs) -> np.ndarray:
    return first3moments(func(*args, **kwargs))


def test_kolmogorov_no_change():
    pop = kolmogorov(0, 0, 0, 0, 0, np.linspace(0, 1, 20), 5, 8)
    moo = first3moments(pop)
    assert np.all(moo[0, :] == 5)
    assert np.all(moo[1:, :] == 0.), (moo[1:, :].shape, moo[1:, :])


def test_kolmogorov_multiplicity_is_kolmogorov_for_delta():
    pop0 = kolmogorov(1e-3, 1., 1., 1e-3, 0., np.linspace(0, 1, 20), 5, 100)
    pop1 = kolmogorov_multiplicity(1e-3, 1., 1., 1e-3, 0., [1.], np.linspace(0, 1, 20), 5, 100)
    assert np.allclose(pop0, pop1, rtol=1e-3, atol=1e-3)


def test_kolmogorov_multiplicity_differs_for_non_delta():
    pop0 = kolmogorov(1e-3, 1., 1., 1e-3, 0., np.linspace(0, 1, 20), 5, 100)
    pop1 = kolmogorov_multiplicity(1e-3, 1., 1., 1e-3, 0., [0.5, 0.5], np.linspace(0, 1, 20), 5, 100)
    assert not np.allclose(pop0, pop1, rtol=1e-3, atol=1e-3)


