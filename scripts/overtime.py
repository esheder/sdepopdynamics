#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial
from biopop_closure.kolmogorov import kolmogorov_moments, kolmogorov_multiplicity
from biopop_closure.multiple_births import dkdt, moment_closure, moments_from_vector
from biopop_closure.moment_closures import gaussian



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
    parser.add_argument("-t", type=float, default=None, help="Time to use. Defaults to last SDE time")
    parser.add_argument("--save", action="store_true", help="Flag to save figures")
    parser.add_argument("--prefix", default="moo", help="Figure filenames prefix")
    args = parser.parse_args()

    with args.par.open('r') as f:
        par = json.load(f)
        par = rename(par)
    df = pd.read_parquet(args.sde)
    t = args.t if args.t is not None else float(df.time.max())


    tvec = np.linspace(0, t, 100)
    kmom = kolmogorov_moments(func=kolmogorov_multiplicity, **par, p0=args.p0, nmax=args.nmax, t=tvec)
    par["m"] = moments_from_vector(par["m"])
    k3m = moment_closure(dkdt, **par, p0=args.p0, t=tvec)
    k2m = moment_closure(partial(gaussian, func=dkdt), **par, p0=args.p0, t=tvec)

    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 20})
    tsde = df.time.values

    for i in range(1, 4):
        kpp, k3p, k2p = map(lambda x: x[i-1, :], (kmom, k3m, k2m))
        kv = df[f"k{i}"].values
        plt.figure()
        plt.plot(tvec, kpp, color="k", linewidth=2, label="Reference")
        plt.plot(tvec, k3p, color="r", linewidth=2, label="Saddlepoint")
        plt.plot(tvec, k2p, color="m", linewidth=2, label="Gaussian")
        plt.plot(tsde, kv,  color="b", linewidth=2, label="SDE")
        plt.xlabel("Time")
        plt.ylabel(f"{dict([(1, 'First'), (2, 'Second'), (3, 'Third')])[i]} Moment")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{args.prefix}_m{i}.jpg", dpi=500)
        if i >= 3:
            continue
        plt.figure()
        def err(x): return 100*(x-kpp)/kpp
        plt.plot(tvec, err(k3p), color="r", linewidth=2, label="Saddlepoint")
        plt.plot(tvec, err(k2p), color="m", linewidth=2, label="Gaussian")
        kvc = np.interp(tvec, tsde, kv)
        plt.plot(tvec, err(kvc), color="b", linewidth=2, label="SDE")
        plt.xlabel("Time")
        plt.ylabel(f"{dict([(1, 'First'), (2, 'Second'), (3, 'Third')])[i]} Moment Error [%]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{args.prefix}_m{i}_error.jpg", dpi=500)

    if not args.save:
        plt.show()
