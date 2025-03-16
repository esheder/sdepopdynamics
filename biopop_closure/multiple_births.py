from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


def dk1dt(i, a1, a2, b1, b2, m, k):
    return i + (a1 * m[0] - a2 - b1 * k[0] * m[0] - b2 * k[0]) * k[0] - (b1 * m[0] + b2) * k[1]


def dk2dt(i, a1, a2, b1, b2, m, k):
    return (
        i
        + (a1 * m[1] + a2 - b1 * m[1] * k[0] + b2 * k[0]) * k[0]
        + (2 * a1 * m[0] - 2 * a2 - b1 * m[1] + b2 - 4 * b1 * m[0] * k[0] - 4 * b2 * k[0]) * k[1]
        - 2 * (b1 * m[0] + b2) * k[2]
    )


def dk3dt(i, a1, a2, b1, b2, m, k):
    return (
        i
        + (a1 * m[2] - a2 - b1 * m[2] * k[0] - b2 * k[0]) * k[0]
        + (
            3 * a1 * m[1]
            + 3 * a2
            - b1 * m[2]
            - b2
            - 6 * b1 * m[1] * k[0]
            + 6 * b2 * k[0]
            - 6 * b1 * m[0] * k[1]
            - 6 * b2 * k[1]
        )
        * k[1]
        + 3 * (a1 * m[0] - a2 - b1 * m[1] + b2 - 2 * b1 * m[0] * k[0] - 2 * b2 * k[0]) * k[2]
    )


def dkdt(i, a1, a2, b1, b2, m, _, k):
    return np.array(
        (
            dk1dt(i, a1, a2, b1, b2, m, k),
            dk2dt(i, a1, a2, b1, b2, m, k),
            dk3dt(i, a1, a2, b1, b2, m, k),
        )
    )


def moment_closure(
    func,
    i: float,
    a1: float,
    a2: float,
    b1: float,
    b2: float,
    m: tuple[float, float, float],
    t: np.ndarray,
    p0: float,
) -> np.ndarray:
    f = partial(func, i, a1, a2, b1, b2, m)
    p0 = np.array([p0, 0.0, 0.0])
    sol = solve_ivp(f, (t[0], t[-1]), p0, t_eval=t, rtol=1e-5)
    if sol.success:
        return sol.y
    else:
        raise RuntimeError(sol.message)


def moments_from_vector(v):
    pop = np.arange(len(v)) + 1
    return tuple(np.sum((pop**e) * v) for e in range(1, 4))
