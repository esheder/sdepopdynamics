#!/usr/bin/env python3
from functools import partial

import numpy as np

from biopop_closure.moment_closures import moment_closure, dkdt
from biopop_closure.multiple_births import (
    moment_closure as multiple_closure,
    dkdt as multiple_dkdt,
)


def badgers(i: float, t: np.ndarray, p0: float, func=partial(moment_closure, dkdt)):
    a1, a2, b1, b2 = badgers_params(i)
    return func(i, a1, a2, b1, b2, t, p0)


def badgers_mul(i: float, t: np.ndarray, p0: float, func=partial(multiple_closure, multiple_dkdt)):
    a1, a2, b1, b2, m = badgers_params_multiple_births(i)
    return func(i, a1, a2, b1, b2, m, t, p0)


def badgers_params(i):
    k = 60
    a1 = 0.61
    a2 = 0.37
    a = a1 - a2
    b2 = 0.0
    b1 = (i + a * k) / (k**2)
    return a1, a2, b1, b2


def badgers_params_multiple_births(i):
    k = 60
    m = (2.39, 6.45, 19.37)
    a1 = 0.255
    a2 = 0.37
    b2 = 0.0
    a = 0.61 - a2
    b1 = (i + a * k) / (k**2) / m[0]
    return a1, a2, b1, b2, m


badgers_multiplicity_vector = np.array((0.11, 0.51, 0.28, 0.08, 0.02))


def foxes(i: float, t: np.ndarray, p0: float, func=partial(moment_closure, dkdt)):
    a1, a2, b1, b2 = foxes_params(i)
    return func(i, a1, a2, b1, b2, t, p0)


def foxes_params(i):
    a1 = 1.0
    a2 = 0.5
    a = a1 - a2
    k = 100.0
    b1 = 0.0
    b2 = (i + a * k) / (k**2)
    return a1, a2, b1, b2


def _foxes_params_multiple_births_paper(i):
    a1, a2, b1, b2 = foxes_params(i)
    m = (4.45, 22.55, 126.982)
    return a1 / m[0], a2, b1, b2, m


foxes_multiplicity_vector = np.array((0.035, 0.07, 0.176, 0.211, 0.317, 0.087, 0.007, 0.017, 0.0, 0.017))


def foxes_params_multiple_births(i):
    a1, a2, b1, b2 = foxes_params(i)
    mvec = foxes_multiplicity_vector
    pop = np.arange(1, 11)
    m = tuple(float(np.sum((pop**e) * mvec)) for e in range(1, 4))
    return a1 / m[0], a2, b1 / m[0], b2, m


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json
    from pathlib import Path

    populations = {
        "fox": foxes_params_multiple_births,
        "badger": badgers_params_multiple_births,
    }
    p = ArgumentParser(description="Makes a JSON file for the population")
    p.add_argument("pop", choices=list(populations.keys()), help="Population type")
    p.add_argument("immigration", type=float, help="Rate of immigration")
    p.add_argument("-o", type=Path, default=None, help="Output file. Defaults to screen output")
    args = p.parse_args()

    f = populations[args.pop]
    popargs = ["a1", "a2", "b1", "b2"]
    mulvec = {
        key: vec for key, vec in zip(populations.keys(), (foxes_multiplicity_vector, badgers_multiplicity_vector))
    }[args.pop]
    pop = {key: value for key, value in zip(popargs, f(args.immigration))} | {
        "I": args.immigration,
        "multiplicity": list(mulvec),
    }
    if args.o is None:
        print(json.dumps(pop))
    else:
        with args.o.open("w") as f:
            json.dump(pop, f)
