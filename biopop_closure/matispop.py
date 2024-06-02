from functools import partial

import numpy as np

from biopop_closure.moment_closures import moment_closure, dkdt


def badgers(i: float, t: np.ndarray, p0: float, func=partial(moment_closure, dkdt)):
    a1, a2, b1, b2 = badgers_params(i)
    return func(i, a1, a2, b1, b2, t, p0)


def badgers_params(i):
    k = 60
    a1 = 0.61
    a2 = 0.37
    a = a1 - a2
    b2 = 0.
    b1 = (i + a * k) / (k ** 2)
    return a1, a2, b1, b2


def foxes(i: float, t: np.ndarray, p0: float, func=partial(moment_closure, dkdt)):
    a1, a2, b1, b2 = foxes_params(i)
    return func(i, a1, a2, b1, b2, t, p0)


def foxes_params(i):
    a1 = 1.
    a2 = 0.5
    a = a1 - a2
    k = 100.
    b1 = 0.
    b2 = (i + a * k) / (k ** 2)
    return a1, a2, b1, b2
