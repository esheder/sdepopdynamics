#!/usr/bin/env python3
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


def relerr(v, ref): return 100 * (v-ref)/ref


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description="Tool to show histograms at equilibrium")
    for a, i in it.product(animals, imms):
        p.add_argument(f"--{a}_{i}", type=Path, help=f"Parquet file of {i} immigration {a}")
    p.add_argument('--save', action="store_true", help="Flag to save plots")
    p.add_argument('--ext', default="svg", help="Plot type for saved figures. Defaults to svg")
    args = p.parse_args()
    dfs = {(anim, st): pd.read_parquet(getattr(args, f"{anim}_{st}")) for anim, st in it.product(animals, imms)}
    t = np.linspace(0, 10, 500)
    pops = {'foxes': {'low': foxes(3, t, 6, func=kolmogorov), 'high': foxes(10, t, 6, func=kolmogorov)},
            'badgers': {'low': badgers(2, t, 6, func=kolmogorov), 'high': badgers(6, t, 6, func=kolmogorov)}}
    fnames = {'foxes': foxes, 'badgers': badgers}
    resolvers = {'gaussian': gaussian_distribution, 'saddle': saddlepoint_distribution}
    immigrations = {'foxes': {'low': 3, 'high': 10}, 'badgers': {'low': 2, 'high': 6}}
    solvefuncs = {"gaussian": gaussian, "saddle": dkdt}
    pop_params = {"foxes": foxes_params, "badgers": badgers_params}
    t = np.linspace(0, 50, 51)
    pops = {(animal, imm, approx): pop_distribution(animal, imm, approx, t, p0=6)
            for animal, imm, approx in it.product(animals, imms, approxes)}

    for animal, imm in it.product(animals, imms):
        g, s, k = [pops[(animal, imm, a)][:, -1] for a in approxes]
        x = np.linspace(0, len(g) - 1, len(g))
        mask = k > (np.max(k)/100)
        g, s, k, x = map(lambda v: v[mask], (g, s, k, x))
        xbins = np.concatenate(([np.min(x)-1], x, [np.max(x)+1]))
        plt.figure()
        plt.plot(x, g, 'b--', label="Gaussian")
        plt.plot(x, s, 'r-.', label="Saddlepoint")
        if (animal, imm) in dfs:
            dfs[(animal, imm)].Population.hist(bins=xbins, density=True, color='m', label="SDE")
        plt.plot(x, k, 'k-', label="Reference")
        plt.legend()
        plt.grid()
        plt.xlabel("Population size [1]")
        plt.ylabel("Probability [1]")
        plt.title(f"Histogram at large time for {animal} with {imm} immigration")
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{animal}_{imm}_pdf.{args.ext}")

        plt.figure()
        gcdf, scdf, kcdf = map(np.cumsum, (g, s, k))
        plt.plot(x, gcdf, 'b--', label="Gaussian")
        plt.plot(x, scdf, 'r--', label="Saddlepoint")
        plt.plot(x, kcdf, 'k-', label="Reference")
        if (animal, imm) in dfs:
            nv = dfs[(animal, imm)].Population.values
            h, edges = np.histogram(nv, bins=xbins, density=True)
            hcdf = np.cumsum(h)
            plt.stairs(hcdf, edges, color="m", label="SDE")
        plt.legend()
        plt.grid()
        plt.xlabel("Population size [1]")
        plt.ylabel("CDF [1]")
        plt.title(f"CDF at large time for {animal} with {imm} immigration")
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{animal}_{imm}_cdf.{args.ext}")

        plt.figure()
        gerr, serr = map(lambda v: relerr(v, kcdf), (gcdf, scdf))
        plt.plot(x, gerr, 'b--', label="Gaussian")
        plt.plot(x, serr, 'r--', label="Saddlepoint")
        if (animal, imm) in dfs:
            centers = (edges[:-1] + edges[1:]) / 2
            kcdf_i = np.interp(centers, x, kcdf)
            plt.stairs(relerr(hcdf, kcdf_i), edges, color="m", label="SDE")
        plt.legend()
        plt.grid()
        plt.xlabel("Population size [1]")
        plt.ylabel("CDF Error [%]")
        plt.title(f"CDF Error at large time for {animal} with {imm} immigration")
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{animal}_{imm}_hist.{args.ext}")

    if not args.save:
        plt.show()
