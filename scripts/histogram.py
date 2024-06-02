from pathlib import Path
from typing import Literal
import itertools as it

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from biopop_closure.kolmogorov import kolmogorov
from biopop_closure.matispop import foxes, badgers, foxes_params, badgers_params
from biopop_closure.moment_closures import (
    gaussian_distribution, saddlepoint_distribution, gaussian, dkdt
)

t = np.linspace(0, 100, 500)
pops = {'foxes': {'low': foxes(3, t, 6, func=kolmogorov), 'high': foxes(10, t, 6, func=kolmogorov)},
        'badgers': {'low': badgers(2, t, 6, func=kolmogorov), 'high': badgers(6, t, 6, func=kolmogorov)}}

fnames = {'foxes': foxes, 'badgers': badgers}
resolvers = {'gaussian': gaussian_distribution, 'saddle': saddlepoint_distribution}
immigrations = {'foxes': {'low': 3, 'high': 10}, 'badgers': {'low': 2, 'high': 6}}
solvefuncs = {"gaussian": gaussian, "saddle": dkdt}
pop_params = {"foxes": foxes_params, "badgers": badgers_params}


def pop_distribution(pop: Literal["foxes", "badgers"],
                     immigration: Literal["low", "high"],
                     approx: Literal["gaussian", "saddle", "exact"],
                     t: np.ndarray,
                     p0: float,
                     ) -> np.ndarray:
    i = immigrations[pop][immigration]
    match approx:
        case "exact":
            pop = kolmogorov(i, *pop_params[pop](i), t, p0)
        case _:
            popmom = fnames[pop](i, t, p0)
            pop = resolvers[approx](popmom)
    return pop


animals = ("foxes", "badgers")
imms = ("low", "high")
approxes = ("gaussian", "saddle", "exact")
t = np.linspace(0, 50, 51)
pops = {(animal, imm, approx): pop_distribution(animal, imm, approx, t, p0=6)
        for animal, imm, approx in it.product(animals, imms, approxes)}

df_low, df_high = [pd.read_parquet(Path(
    "/home/eshedm/workspace/popfeedback/runs/european_badgers/time_develop/sde/") / fn)
                   for fn in ["low_sde.parquet", "high_sde.parquet"]]
sde = {('badgers', 'low'): df_low[df_low.time == df_low.time.max()].population,
       ('badgers', 'high'): df_high[df_high.time == df_high.time.max()].population
       }

for animal, imm in it.product(animals, imms):
    g, s, k = [pops[(animal, imm, a)][:, -1] for a in approxes]
    x = np.linspace(0, len(g) - 1, len(g))
    plt.figure()
    plt.plot(x, g, 'b--', label="Gaussian")
    plt.plot(x, s, 'r-.', label="Saddlepoint")
    if (animal, imm) in sde:
        sde[(animal, imm)].hist(bins=x, density=True, color='m', label="SDE")
    plt.plot(x, k, 'k-', label="Reference")
    plt.legend()
    plt.grid()
    plt.xlabel("Population size [1]")
    plt.ylabel("Probability [1]")
    plt.title(f"Histogram at large time for {animal} with {imm} immigration")

plt.show()
