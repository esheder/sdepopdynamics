#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial
from biopop_closure.kolmogorov import kolmogorov_multiplicity
from biopop_closure.multiple_births import dkdt, moment_closure, moments_from_vector
from biopop_closure.moment_closures import gaussian, gaussian_distribution, saddlepoint_distribution


EPS = 1e-3


def cut_df(df, t):
    return df[df.Time.between(t-EPS, t+EPS)]

renames = {'I': 'i', 'multiplicity': 'm'}
def rename(d):
    return {renames[key] if key in renames else key: value for key, value in d.items()}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Tool for histogram plots")
    parser.add_argument("par", type=Path, help="Parameter JSON file used")
    parser.add_argument("sde", type=Path, help="Parquet file for SDE data")
    parser.add_argument("--p0", type=int, default=10, help="Initial population to use")
    parser.add_argument("--nmax", type=int, default=200, help="Max population in Kolmogorov solve")
    parser.add_argument("--range", type=int, nargs=2, default=(0, None), help="Range to cut the population in")
    parser.add_argument("-t", type=float, default=None, help="Time to use. Defaults to last SDE time")
    parser.add_argument("--save", action="store_true", help="Flag to save figures")
    parser.add_argument("--prefix", default="moo", help="Figure filenames prefix")
    args = parser.parse_args()

    with args.par.open('r') as f:
        par = json.load(f)
        par = rename(par)
    df = pd.read_parquet(args.sde)
    t = args.t if args.t is not None else float(df.Time.max())
    df = cut_df(df, t)

    cut = slice(args.range[0], args.range[1] if args.range[1] else args.nmax)
    xcut = slice(cut.start, cut.stop+1)

    tvec = np.linspace(0, t, 20)
    kpop = kolmogorov_multiplicity(**par, p0=args.p0, nmax=args.nmax, t=tvec)
    par["m"] = moments_from_vector(par["m"])
    m3 = moment_closure(dkdt, **par, p0=args.p0, t=tvec)
    k3pop = saddlepoint_distribution(m3, maxpop=args.nmax)
    k2pop = gaussian_distribution(moment_closure(partial(gaussian, func=dkdt), **par, p0=args.p0, t=tvec), maxpop=args.nmax)
    kpop = kpop[:, -1].flatten()
    k3pop = k3pop[:, -1].flatten()
    k2pop = k2pop[:, -1].flatten()
    xbins = np.arange(kpop.size + 1)

    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 20})

    plt.figure()
    hvr, hbr = np.histogram(df.Population.values, bins=xbins, density=True)
    kpp, k3p, k2p, hv = map(lambda x: x[cut], (kpop, k3pop, k2pop, hvr))
    x, hb = map(lambda x: x[xcut], (xbins, hbr))
    plt.stairs(kpp, x, color='k', linewidth=2, label="Reference")
    plt.stairs(hv, hb, color='b', linewidth=2, label="SDE")
    plt.stairs(k3p, x, color="r", linewidth=2, label="Saddlepoint")
    plt.stairs(k2p, x, color="m", linewidth=2, label="Gaussian")
    plt.xlabel("Population")
    plt.ylabel("Probability Density")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if args.save:
        plt.savefig(f"{args.prefix}_pdf.jpg", dpi=500)

   
    plt.figure()
    rcdf, k3cdf, k2cdf, scdf = map(lambda x: np.cumsum(x)[cut], (kpop, k3pop, k2pop, hvr))
    plt.stairs(rcdf, x, color="k", linewidth=2, label="Reference")
    plt.stairs(scdf, hb, color="b", linewidth=2, label="SDE")
    if np.any(np.isfinite(k3cdf)):
        plt.stairs(k3cdf, x, color="r", linewidth=2, label="Saddlepoint")
    plt.stairs(k2cdf, x, color="m", linewidth=2, label="Gaussian")
    plt.xlabel("Population")
    plt.ylabel("Probability")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if args.save:
        plt.savefig(f"{args.prefix}_cdf.jpg", dpi=500)


    plt.figure()
    plt.stairs(scdf-rcdf, x, color="b", linewidth=2, label="SDE")
    if np.any(np.isfinite(k3cdf)):
        plt.stairs(k3cdf-rcdf, x, color="r", linewidth=2, label="Saddlepoint")
    plt.stairs(k2cdf-rcdf, x, color="m", linewidth=2, label="Gaussian")
    plt.xlabel("Population")
    plt.ylabel("CDF Error")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if args.save:
        plt.savefig(f"{args.prefix}_cdf_error.jpg", dpi=500)
    
  
    if not args.save:
        plt.show()
