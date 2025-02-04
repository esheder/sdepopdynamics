from typing import Sequence

import numpy as np
import scipy
from scipy.integrate import solve_ivp


def _solve_matrix(mat, t, p0: int, nmax: int) -> np.ndarray:
    pt0 = np.zeros(nmax + 1, dtype=float)
    pt0[p0] = 1.

    def _dpdt(_, x):
        return mat @ x

    sol = solve_ivp(_dpdt, (t[0], t[-1]), pt0, t_eval=t)
    if not sol.success:
        raise RuntimeError(sol.message)
    psol = sol.y
    return psol


def kolmogorov_matrix(i, a1, a2, b1, b2, nmax) -> scipy.sparse.csr_array:
    popsize = np.arange(nmax + 1)
    ld = a1 * popsize - b1 * popsize ** 2
    ld[ld < 0] = 0.
    mu = a2 * popsize + b2 * popsize ** 2
    dm1 = ld[:-1] + i
    dm0 = -(ld + mu + i)
    dm0[0] = -i
    dm0[-1] = -mu[-1]
    dmm1 = mu[1:]
    return scipy.sparse.diags((dm1, dm0, dmm1), (-1, 0, 1), format='csr')


def kolmogorov(i, a1, a2, b1, b2, t, p0: int, nmax: int = 200) -> np.ndarray:
    mat = kolmogorov_matrix(i, a1, a2, b1, b2, nmax=nmax)
    return _solve_matrix(mat, t, p0, nmax)


def kolmogorov_multiplicity_matrix(i, a1, a2, b1, b2, m, nmax):
    popsize = np.arange(nmax + 1)
    ld = a1 * popsize - b1 * popsize ** 2
    ld[ld < 0] = 0.
    mu = a2 * popsize + b2 * popsize ** 2
    dmm1 = mu[1:]
    mlen = len(m)
    dms = [np.empty(nmax - j, dtype=float) for j in range(mlen)]
    for j, v in enumerate(dms):
        v[0] = 0
        v[1:] = ld[1:nmax - j] * m[j]
    dms[0] += i
    mat = scipy.sparse.diags((*dms, dmm1), (*[-j for j in range(1, mlen + 1)], 1), format='lil')
    for j in range(mat.shape[0]):
        mat[j, j] = -np.sum(mat[:, j])
    return mat.tocsr()


def kolmogorov_multiplicity(i, a1, a2, b1, b2, m: Sequence[float], t, p0: int, nmax: int = 200) -> np.ndarray:
    mat = kolmogorov_multiplicity_matrix(i, a1, a2, b1, b2, m, nmax=nmax)
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


