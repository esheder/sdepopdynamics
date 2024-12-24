#!/usr/bin/env python3
"""Tools to model the effects of the moment closures

"""

import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable
import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biopop_closure.kolmogorov import kolmogorov_moments, kolmogorov_multiplicity
from biopop_closure.moment_closures import (
        relative_skewness, gaussian
)
from biopop_closure.multiple_births import moment_closure, dkdt, moments_from_vector

from matplotlib import rc_params

rc_params()


kolmogorov = partial(kolmogorov_moments, func=kolmogorov_multiplicity)
k3 = partial(moment_closure, dkdt)
gaussian = partial(moment_closure, partial(gaussian, func=dkdt))


def moments(df):
    v = np.empty((3, df.time.values.size), dtype=float)
    v[0, :] = df.k1.values
    v[1, :] = df.k2.values
    v[2, :] = df.k3.values
    return v


def change_m(d):
    d = {key: value for key, value in d.items()}
    d["m"] = moments_from_vector(d["m"])
    return d


def plot_pop(plow, phigh, ilow, ihigh, dflow, dfhigh, p0: int, save: str = None):
    t = np.linspace(0, 50, 500)
    reference_low = kolmogorov(**plow, i=ilow, t=t, p0=p0)
    reference_high = kolmogorov(**phigh, i=ihigh, t=t, p0=p0)
    plow, phigh = map(change_m, (plow, phigh))
    k3_low = k3(**plow, i=ilow, t=t, p0=p0)
    k3_high = k3(**phigh, i=ihigh, t=t, p0=p0)
    gaussian_low = gaussian(**plow, i=ilow, t=t, p0=p0)
    gaussian_high = gaussian(**phigh, i=ihigh, t=t, p0=p0)
    refcolors = {"reference": "k", "k3": "m", "gaussian": "r", "SDE": "b"}
    refmarks = {ilow: "-", ihigh: "--"}
    labels = {(i, cas): f"{cas}: I={i:.0f}" for i, cas in product((ilow, ihigh), refcolors.keys())}
    markers = {(i, cas): f"{refmarks[i]}{refcolors[cas]}" for i, cas in labels.keys()}
    values = {
            (ilow, "reference"): (t, reference_low),
            (ihigh, "reference"): (t, reference_high),
            (ilow, "k3"): (t, k3_low),
            (ihigh, "k3"): (t, k3_high),
            (ilow, "gaussian"): (t, gaussian_low),
            (ihigh, "gaussian"): (t, gaussian_high),
            (ilow, "SDE"): (dflow.time.values, moments(dflow)),
            (ihigh, "SDE"): (dfhigh.time.values, moments(dfhigh)),
            }
    plt.figure()
    for key, (t, v) in values.items():
        plt.plot(t, v[0, :], markers[key], label=labels[key])
    plt.xlabel("Time [yrs]")
    plt.ylabel("k1 [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_expectation.jpg", dpi=300)

    plt.figure()
    for key, (t, v) in values.items():
        plt.plot(t, v[1, :], markers[key], label=labels[key])
    plt.xlabel("Time [yrs]")
    plt.ylabel("k2 [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_variance.jpg", dpi=300)

    plt.figure()
    for key, (t, v) in values.items():
        plt.plot(t, v[2, :], markers[key], label=labels[key])
    plt.xlabel("Time [yrs]")
    plt.ylabel("k3 [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_skewness.jpg", dpi=300)

    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for key, (t, v) in values.items():
            plt.plot(t, relative_skewness(v), markers[key], label=labels[key])
    plt.xlabel("Time [yrs]")
    plt.ylabel(r"Relative Skewness ($\gamma$) [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_relative_skewness.jpg", dpi=300)

    if not save:
        plt.show()


def load_json_file(path):
    with path.open('r') as f:
        return json.load(f)


def load_pandas(path):
    return pd.read_parquet(path)


def comparable(d, v):
    return dict((key if len(key) < 3 else key[0], value) for key, value in d.items() if key != v)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Tool to replicate the paper by Matis")
    parser.add_argument('pop', type=Path, nargs=2, help="Population parameters JSON file")
    parser.add_argument('sde', type=Path, nargs=2, help="Path to SDE parquets to compare")
    parser.add_argument('pt0', type=int, help="Initial population")
    parser.add_argument("--save", default=None, help="Prefix for figure files to save. Defaults to screen output")
    args = parser.parse_args()
    p0, p1 = map(load_json_file, args.pop)
    df0, df1 = map(load_pandas, args.sde)
    d0, d1 = map(lambda x: comparable(x, "I"), (p0, p1))
    plot_pop(d0, d1, p0["I"], p1["I"], df0, df1, p0=args.pt0, save=args.save)

