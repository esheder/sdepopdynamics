from functools import partial
import warnings

import numpy as np
import scipy
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


def moment_closure(func, i, a1, a2, b1, b2, t, p0):
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


def relative_skewness(vec: np.ndarray) -> np.ndarray:
    return vec[2, :] / (vec[1, :] ** 1.5)