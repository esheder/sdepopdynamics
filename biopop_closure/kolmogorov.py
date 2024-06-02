import numpy as np
import scipy
from scipy.integrate import solve_ivp


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
    pt0 = np.zeros_like(popsize, dtype=float)
    pt0[p0] = 1.

    def _dpdt(_, x):
        return mat @ x

    sol = solve_ivp(_dpdt, (t[0], t[-1]), pt0, t_eval=t)
    if not sol.success:
        raise RuntimeError(sol.message)
    psol = sol.y
    return psol


def first3moments(pop: np.ndarray) -> np.ndarray:
    popsize = np.arange(pop.shape[0])
    m1 = np.sum(popsize[:, np.newaxis] * pop, axis=0)
    x2 = np.sum((popsize[:, np.newaxis] ** 2) * pop, axis=0)
    x3 = np.sum((popsize[:, np.newaxis] ** 3) * pop, axis=0)
    m2 = x2 - m1 ** 2
    m3 = x3 - (3 * m1 * x2) + 2 * (m1 ** 3)
    return np.vstack((m1, m2, m3))


def kolmogorov_moments(*args, **kwargs) -> np.ndarray:
    return first3moments(kolmogorov(*args, **kwargs))


def test_kolmogorov_no_change():
    pop = kolmogorov(0, 0, 0, 0, 0, np.linspace(0, 1, 20), 5, 8)
    moo = first3moments(pop)
    assert np.all(moo[0, :] == 5)
    assert np.all(moo[1:, :] == 0.), (moo[1:, :].shape, moo[1:, :])
