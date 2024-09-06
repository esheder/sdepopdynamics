from functools import partial

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp


def dk1dt(i, a, b, k):
    return i + (a - b * k[0]) * k[0] - b * k[1]


def dk2dt(i, a, b, c, d, k):
    return i + (c - d * k[0]) * k[0] + (2 * a - d - 4 * b * k[0]) * k[1] - 2 * b * k[2]


def dk3dt(i, a, b, c, d, k):
    return (i + (a - b * k[0]) * k[0]
            + (3 * c - b - 6 * d * k[0] - 6 * b * k[1]) * k[1]
            + 3 * (a - d - 2 * b * k[0]) * k[2]
            )


def dkdt(i, a, b, c, d, _, k):
    return np.array((dk1dt(i, a, b, k),
                     dk2dt(i, a, b, c, d, k),
                     dk3dt(i, a, b, c, d, k)
                     ))


def gaussian(*args):
    args[-1][-1] = 0
    der = dkdt(*args)
    der[-1] = 0
    return der


def moment_closure(func,
                   i: float, a1: float, a2: float, b1: float, b2: float,
                   t: np.ndarray, p0: float) -> np.ndarray:
    a = a1 - a2
    b = b1 + b2
    c = a1 + a2
    d = b1 - b2
    f = partial(func, i, a, b, c, d)
    p0 = np.array([p0, 0., 0.])
    sol = solve_ivp(f, (t[0], t[-1]), p0, t_eval=t, rtol=1e-5)
    if sol.success:
        return sol.y
    else:
        raise RuntimeError(sol.message)


def gaussian_distribution(moments: np.ndarray, maxpop: int = 200) -> np.ndarray:
    tlen = moments.shape[-1]
    res = np.empty((maxpop + 1, tlen), dtype=float)
    x = np.linspace(0, maxpop, maxpop + 1)
    m1, m2 = moments[0, :], moments[1, :]
    for i, (k1, k2) in enumerate(zip(m1, m2)):
        res[:, i] = sp.stats.norm.pdf(x, loc=k1, scale=np.sqrt(k2))
    return res


def _psi(x: np.ndarray, k1: float, k2: float, k3: float) -> np.ndarray:
    return k2 ** 2 + 2 * k3 * (x - k1)


def saddlepoint_distribution(moments: np.ndarray, maxpop: int = 200) -> np.ndarray:
    tlen = moments.shape[-1]
    res = np.empty((maxpop + 1, tlen), dtype=float)
    x = np.linspace(0, maxpop, maxpop + 1)
    m1, m2, m3 = moments[0, :], moments[1, :], moments[2, :]
    for i, (k1, k2, k3) in enumerate(zip(m1, m2, m3)):
        psi = _psi(x, k1, k2, k3)
        psi = np.where(psi >= 0, psi, 0.)
        f = ((((4 * np.pi ** 2) * psi) ** (-0.25))
             * np.exp(-(k2 ** 3 - 3 * k2 * psi + 2 * psi ** 1.5) / (6 * k3 ** 2)))
        res[:, i] = np.where(psi >= 0, f, np.nan)
    return res


def test_moment_closure_solve_is_positive():
    ts = np.linspace(0, 20, 100)
    a1 = 0.61
    a2 = 0.37
    b2 = 0.
    b1 = 0.004
    i = 0.
    p0 = 6.
    sol = moment_closure(dkdt, i, a1, a2, b1, b2, ts, p0)
    assert len(sol[0, :]) == len(ts)
    assert np.all(sol[0, :] > 0)


def test_saddle_is_gaussian_no_skew():
    mom = np.array([[50.], [5.], [0.]])
    g = gaussian_distribution(mom)
    s = saddlepoint_distribution(mom)
    legal = ~np.isnan(s)
    g, s = g[legal], s[legal]
    assert np.allclose(g, s), (np.max(np.abs(g - s)), np.argmax(np.abs(g - s)))


def test_saddle_not_gaussian_with_skew():
    mom = np.array([[50.], [5.], [3.]])
    g = gaussian_distribution(mom)
    s = saddlepoint_distribution(mom)
    legal = ~np.isnan(s)
    g, s = g[legal], s[legal]
    assert not np.allclose(g, s)


def relative_skewness(vec: np.ndarray) -> np.ndarray:
    return vec[2, :] / (vec[1, :] ** 1.5)
