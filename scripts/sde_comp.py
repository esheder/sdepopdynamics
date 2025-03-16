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


def plot_pop(p, i, df, p0: int, moment: int, nmax: int, save: str = None):
    t = np.linspace(0, 50, 500)
    reference = kolmogorov(**p, i=i, t=t, p0=p0, nmax=nmax)
    p = change_m(p)
    k3v = k3(**p, i=i, t=t, p0=p0)
    gauss = gaussian(**p, i=i, t=t, p0=p0)
    refcolors = {"reference": "k", "k3": "m", "gaussian": "r", "SDE": "b"}
    labels = {key: key for key in refcolors}
    markers = refcolors
    values = {
            "reference": (t, reference),
            "k3": (t, k3v),
            "gaussian": (t, gauss),
            "SDE": (df.time.values, moments(df)),
            }

    func = {1: plot_moment, 2: plot_moment, 3: plot_k3}[moment]
    func(refcolors, labels, markers, values, moment=moment, save=save)


def plot_moment(refcolors, labels, markers, values, moment, save, *, can_show=True):
    try:
        name = {1: 'Expectation', 2: 'Variance', 3: 'Skewness'}[moment]
    except KeyError:
        raise ValueError(f"Plot moment does not work for moment {moment}")
    plt.figure()
    for key, (t, v) in values.items():
        plt.plot(t, v[moment-1, :], markers[key], label=labels[key])
    plt.xlabel("Time [yrs]")
    plt.ylabel(f"{name} [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_{name}.jpg", dpi=300)

    plt.figure()
    for key, (t, v) in values.items():
        if key == "reference":
            continue
        reference_t, reference_data = values["reference"]
        reference_y = reference_data[moment-1, :]
        reference = np.interp(t, reference_t, reference_y)
        plt.plot(t, 100*(v[moment-1, :]-reference)/reference, markers[key], label=labels[key])
    plt.xlabel("Time [yrs]")
    plt.ylabel(f"{name} Error [%]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_{name}_error.jpg", dpi=300)

    if not save and can_show:
        plt.show()


def plot_k3(refcolors, labels, markers, values, save, **_):
    plot_moment(refcolors, labels, markers, values, save=save, moment=3, can_show=False)
    
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

    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for key, (t, v) in values.items():
            if key == "reference":
                continue
            reference_t, reference_data = values["reference"]
            reference = np.empty_like(v)
            for i in range(reference_data.shape[0]):
                reference_y = reference_data[i, :]
                reference[i, :] = np.interp(t, reference_t, reference_y)
            plt.plot(t, relative_skewness(v)-relative_skewness(reference), markers[key], label=labels[key])
    plt.xlabel("Time [yrs]")
    plt.ylabel(r"Relative Skewness Error ($\gamma$) [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_relative_skewness_error.jpg", dpi=300)
    
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
    parser.add_argument('pop', type=Path, help="Population parameters JSON file")
    parser.add_argument('sde', type=Path, help="Path to SDE parquets to compare")
    parser.add_argument('pt0', type=int, help="Initial population")
    parser.add_argument('moment', type=int, help="Moment to plot for")
    parser.add_argument("--save", default=None, help="Prefix for figure files to save. Defaults to screen output")
    args = parser.parse_args()
    p = load_json_file(args.pop)
    df = load_pandas(args.sde)
    nmax = int(df.k1.max() + 3*np.sqrt(df.k2.max()))
    d = comparable(p, "I")
    plot_pop(d, p["I"], df, p0=args.pt0, nmax=nmax, save=args.save, moment=args.moment)

