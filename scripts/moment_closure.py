#!/usr/bin/env python3
"""Tools to model the effects of the moment closures

"""

import pickle
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from biopop_closure.kolmogorov import kolmogorov
from biopop_closure.moment_closures import (
    moment_closure, dkdt, relative_skewness, gaussian
)

from matplotlib import rc_params

rc_params()


def badgers(i, t, p0, func=partial(moment_closure, dkdt)):
    k = 60
    a1 = 0.61
    a2 = 0.37
    a = a1 - a2
    b2 = 0.
    b1 = (i + a * k) / (k ** 2)
    return func(i, a1, a2, b1, b2, t, p0)


def foxes(i, t, p0, func=partial(moment_closure, dkdt)):
    a1 = 1.
    a2 = 0.5
    a = a1 - a2
    k = 100.
    b1 = 0.
    b2 = (i + a * k) / (k ** 2)
    return func(i, a1, a2, b1, b2, t, p0)


def plot_pop(popfunc, ilow, ihigh, save: str = None):
    t = np.linspace(0, 50, 500)
    p0 = 6
    low = popfunc(ilow, t, p0)
    high = popfunc(ihigh, t, p0)
    labelhigh = f"I={ihigh:.0f}"
    labellow = f"I={ilow:.0f}"
    highmark = '--r'
    lowmark = '-b'
    plt.figure()
    plt.plot(t, low[0, :], lowmark, label=labellow)
    plt.plot(t, high[0, :], highmark, label=labelhigh)
    plt.xlabel("Time [yrs]")
    plt.ylabel("Mean [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_expectation.jpg", dpi=300)

    plt.figure()
    plt.plot(t, low[1, :], lowmark, label=labellow)
    plt.plot(t, high[1, :], highmark, label=labelhigh)
    plt.xlabel("Time [yrs]")
    plt.ylabel("Variance [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_variance.jpg", dpi=300)

    plt.figure()
    plt.plot(t, low[2, :], lowmark, label=labellow)
    plt.plot(t, high[2, :], highmark, label=labelhigh)
    plt.xlabel("Time [yrs]")
    plt.ylabel("Skewness [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_skewness.jpg", dpi=300)

    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plt.plot(t, relative_skewness(low), lowmark, label=labellow)
        plt.plot(t, relative_skewness(high), highmark, label=labelhigh)
    plt.xlabel("Time [yrs]")
    plt.ylabel("Relative Skewness [1]")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"{save}_relative_skewness.jpg", dpi=300)

    if not save:
        plt.show()


def parse_sde_to_moments(df: pd.DataFrame,
                         resamples: int,
                         ) -> tuple[np.ndarray, np.ndarray]:
    grp = df.groupby('time').population
    rng = np.random.default_rng(seed=42)

    def knstat(n: int, x: np.ndarray, axis=0) -> float: return st.kstat(x, n=n, nan_policy='omit', axis=axis)

    def _relsk(x: np.ndarray, axis=0) -> float: return knstat(3, x, axis) / (knstat(2, x, axis) ** 1.5)

    def btsrp(stat: Callable[[np.ndarray], float],
              x: np.ndarray
              ) -> tuple[float, float]:
        res = st.bootstrap(
            data=(x,), statistic=stat,
            n_resamples=resamples,
            method='basic',
            confidence_level=0.95,
            random_state=rng,
            vectorized=True,
        )
        return stat(x), res.standard_error

    def _tpl_unpack(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return zip(*vec)

    stats = [partial(knstat, x) for x in range(1, 4)] + [_relsk]
    bootstraps = map(lambda x: partial(btsrp, x), stats)
    (k1, k1err), (k2, k2err), (k3, k3err), (rk, rkerr) = map(
        lambda x: _tpl_unpack(grp.apply(x).values),
        bootstraps
    )
    return np.vstack((k1, k2, k3, rk)), np.vstack((k1err, k2err, k3err, rkerr))


@dataclass
class PopVec:
    vec: np.ndarray
    fmt: str
    label: str
    s: slice = slice(None)
    err: np.ndarray | None = None

    def __getitem__(self, key):
        return self.vec[key]


def plot_vecs(*vecs: PopVec,
              t: np.ndarray,
              ylabel: str,
              save: Path | None = None,
              xlabel: str = 'Time [yrs]',
              show: bool = False,
              ) -> None:
    plt.figure()
    for vec in vecs:
        if vec.err is not None:
            plt.errorbar(t[vec.s], vec[vec.s], vec.err[vec.s], fmt=vec.fmt, label=vec.label)
        else:
            plt.plot(t[vec.s], vec[vec.s], vec.fmt, label=vec.label)
    plt.grid()
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(save, dpi=600)
    if show:
        plt.show()


def compare(pop, ilow, ihigh,
            sdefh: pd.DataFrame = None,
            sdefl: pd.DataFrame = None,
            sdesol: Path | None = None,
            force: bool = False,
            p0: int = 6,
            t: np.ndarray = np.linspace(0, 50, 501),
            save: Path = None,
            **kwargs,
            ):
    k2low = pop(ilow, t, p0, func=partial(moment_closure, gaussian))
    k3low = pop(ilow, t, p0, func=partial(moment_closure, dkdt))
    kollow = pop(ilow, t, p0, func=kolmogorov)
    k2high = pop(ihigh, t, p0, func=partial(moment_closure, gaussian))
    k3high = pop(ihigh, t, p0, func=partial(moment_closure, dkdt))
    kolhigh = pop(ihigh, t, p0, func=kolmogorov)
    if sdesol is not None and sdesol.exists() and not force:
        with sdesol.open('rb') as f:
            (sdehigh, sdehigherr), (sdelow, sdelowerr) = pickle.load(f)
    elif sdefh is not None and sdefl is not None:
        sdeparse = partial(parse_sde_to_moments, **kwargs)
        (sdehigh, sdehigherr), (sdelow, sdelowerr) = map(sdeparse, (sdefh, sdefl))
        if sdesol:
            with sdesol.open('wb') as f:
                pickle.dump(((sdehigh, sdehigherr), (sdelow, sdelowerr)), f)
    else:
        sdehigh = sdehigherr = sdelow = sdelowerr = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skew_low = relative_skewness(kollow)
        skew_low_nan = skew_low.copy()
        skew_low[np.isnan(skew_low)] = 0.
        skew_high = relative_skewness(kolhigh)
        skew_high_nan = skew_high.copy()
        skew_high[np.isnan(skew_high)] = 0.
        k2_skew_low = relative_skewness(k2low)
        k2_skew_high = relative_skewness(k2high)
        k3_skew_low = relative_skewness(k3low)
        k3_skew_high = relative_skewness(k3high)
    tmax_low = t[np.abs(skew_low).argmax()]
    tmax_high = t[np.abs(skew_high).argmax()]

    def relerr(x, y):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vals = 1e2 * (x - y) / y
        return vals

    def _at_t(v, tv):
        return v[t == tv][0]

    means = [PopVec(*a) for a in [
        (k2low[0, :], 'b-', f"K2 I={ilow}"),
        (k3low[0, :], 'r-', f"K3 I={ilow}"),
        (kollow[0, :], 'k-', f"Kolmogorov I={ilow}"),
        (k2high[0, :], 'b--', f"K2 I={ihigh}"),
        (k3high[0, :], 'r--', f"K3 I={ihigh}"),
        (kolhigh[0, :], 'k--', f"Kolmogorov I={ihigh}"),
    ]]
    if sdelow is not None:
        means.append(PopVec(sdelow[0, :], 'gx', f"SDE I={ilow}",
                            slice(None, None, 5),
                            err=sdelowerr[0, :]))
    if sdehigh is not None:
        means.append(PopVec(sdehigh[0, :], 'g^', f"SDE I={ihigh}",
                            slice(None, None, 5),
                            err=sdehigherr[0, :]))
    # plot_vecs(*means, t=t, ylabel="Population Mean")
    variances = [PopVec(*a) for a in [
        (k2low[1, :], 'b-', f"K2 I={ilow}"),
        (k3low[1, :], 'r-', f"K3 I={ilow}"),
        (kollow[1, :], 'k-', f"Kolmogorov I={ilow}"),
        (k2high[1, :], 'b--', f"K2 I={ihigh}"),
        (k3high[1, :], 'r--', f"K3 I={ihigh}"),
        (kolhigh[1, :], 'k--', f"Kolmogorov I={ihigh}"),
    ]]
    if sdelow is not None:
        variances.append(PopVec(sdelow[1, :], 'gx', f"SDE I={ilow}",
                                slice(None, None, 5),
                                err=sdelowerr[1, :]))
    if sdehigh is not None:
        variances.append(PopVec(sdehigh[1, :], 'g^', f"SDE I={ihigh}",
                                slice(None, None, 5),
                                err=sdehigherr[1, :]))
    # plot_vecs(*variances, t=t, ylabel="Population Variance", show=True)
    errvecs = [PopVec(*a) for a in [
        (relerr(k2low, kollow)[0, :], 'b-', f"K2 Mean"),
        (relerr(k2low, kollow)[1, :], 'b--', f"K2 Variance"),
        (relerr(k3low, kollow)[0, :], 'r-', f"K3 Mean"),
        (relerr(k3low, kollow)[1, :], 'r--', f"K3 Variance"),
        # (relerr(k2high, kolhigh)[0, :], 'm-', f"K2 Mean I={ihigh}"),
        # (relerr(k2high, kolhigh)[1, :], 'm--', f"K2 Variance I={ihigh}"),
        # (relerr(k3high, kolhigh)[0, :], 'k-', f"K3 Mean I={ihigh}"),
        # (relerr(k3high, kolhigh)[1, :], 'k--', f"K3 Variance I={ihigh}"),
    ]]
    if sdelow is not None:
        errvecs.append(PopVec(relerr(sdelow[0, :], kollow[0, :]),
                              'gx',
                              f"SDE Mean",
                              slice(None, None, 10),
                              err=np.abs(1e2 * sdelowerr[0, :] / kollow[0, :]),
                              ))
        errvecs.append(PopVec(relerr(sdelow[1, :], kollow[1, :]),
                              'cx',
                              f"SDE Variance",
                              slice(None, None, 10),
                              err=np.abs(1e2 * sdelowerr[1, :] / kollow[1, :]),
                              ))
    if sdehigh is not None and False:
        errvecs.append(PopVec(relerr(sdehigh[0, :], kolhigh[0, :]),
                              'g^',
                              f"SDE Mean I={ihigh}",
                              slice(None, None, 5),
                              err=np.abs(1e2 * sdehigherr[0, :] / kolhigh[0, :]),
                              ))
        errvecs.append(PopVec(relerr(sdehigh[1, :], kolhigh[1, :]),
                              'c^',
                              f"SDE Variance I={ihigh}",
                              slice(None, None, 5),
                              err=np.abs(1e2 * sdehigherr[1, :] / kolhigh[1, :]),
                              ))
    _save = save.with_name("RelativeErrors.jpg") if save else save
    plot_vecs(*errvecs, t=t, ylabel="Relative Error [%]", save=_save)
    k3s = [PopVec(*a) for a in [
        (k3_skew_low, 'b-', f"K3"),
        (skew_low_nan, 'k-', f"Reference"),
        # (k3_skew_high, "b--", f"K3 I={ihigh}"),
        # (skew_high_nan, "k--", f"Reference I={ihigh}"),
    ]]
    if sdelow is not None:
        k3s.append(PopVec(sdelow[3, :], "rx", f"SDE",
                          slice(None, None, 10),
                          err=sdelowerr[3, :]))
    if sdehigh is not None and False:
        k3s.append(PopVec(sdehigh[3, :], "r^", f"SDE I={ihigh}",
                          slice(None, None, 10),
                          err=sdehigherr[3, :]))
    _save = save.with_name("K3Stat.jpg") if save else save
    plot_vecs(*k3s, t=t, ylabel="K3 Statistic", save=_save)
    plt.figure()
    plt.plot(t, relerr(k3_skew_low, skew_low_nan), 'b-', label=f"K3 I={ilow}")
    plt.plot(t, relerr(k3_skew_high, skew_high_nan), 'r-', label=f"K3 I={ihigh}")
    if sdelow is not None:
        plt.errorbar(t[::5], relerr(sdelow[3, :], skew_low_nan)[::5],
                     yerr=1e2 * np.abs(sdelowerr[3, :] / skew_low_nan)[::5],
                     fmt='kx', label=f"SDE I={ilow}"
                     )
    if sdehigh is not None:
        plt.errorbar(t[::5], relerr(sdehigh[3, :], skew_high_nan)[::5],
                     yerr=1e2 * np.abs(sdehigherr[3, :] / skew_high_nan)[::5],
                     fmt='k^', label=f"SDE I={ihigh}"
                     )
    plt.grid()
    plt.legend()
    plt.xlabel("Time [yrs]")
    plt.ylabel("Relative Error [%]")
    plt.ylim(-15, 7.5)
    if save:
        plt.savefig(save.with_name("RelativeK3.jpg"), dpi=600)
    plt.show()
    _at_low = partial(_at_t, tv=tmax_low)
    _at_high = partial(_at_t, tv=tmax_high)
    print(f"Skewness at worst skew for low immigration (at t={tmax_low:.2f}): "
          f"k2={_at_low(k2_skew_low):.3e}, k3={_at_low(k3_skew_low):.3e}, actual={_at_low(skew_low):.3e}")
    print(f"Skewness at worst skew for high immigration (at t={tmax_high:.2f}): "
          f"k2={_at_high(k2_skew_high):.3e}, k3={_at_high(k3_skew_high):.3e}, actual={_at_high(skew_high):.3e}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Tool to replicate the paper by Matis")
    parser.add_argument('pop', choices={'badgers', 'foxes'}, help="Population type")
    parser.add_argument('order', choices={'plot', 'compare'}, help="What to do")
    parser.add_argument('--sde', type=Path, nargs=2, default=None, help="Path to SDE parquet to compare")
    parser.add_argument('--closure', choices={'gaussian', 'k3', 'exact'}, default='k3', help="Closure type")
    parser.add_argument('--save', type=Path, default=None, help="Prefix for the files to save things as.")
    parser.add_argument('--resamples', type=int, default=5, help="Resamples for the SDE bootstrap")
    parser.add_argument('--force', action='store_true', help="Flag to force SDE bootstrap recomputation")
    parser.add_argument('--sdesol', type=Path, help="Path to store SDE bootstrap results")
    args = parser.parse_args()
    pop = {'badgers': (badgers, 2, 6),
           'foxes': (foxes, 3, 10)
           }[args.pop]
    closure = {'gaussian': partial(moment_closure, gaussian),
               'k3': partial(moment_closure, dkdt),
               'exact': kolmogorov}[args.closure]
    if args.order == 'plot':
        plot_pop(partial(pop[0], func=closure),
                 *pop[1:], save=args.save)
    elif args.order == 'compare':
        dfhigh, dflow = map(pd.read_parquet, args.sde) if args.sde else (None, None)
        compare(*pop,
                sdefh=dfhigh,
                sdefl=dflow,
                resamples=args.resamples,
                force=args.force,
                sdesol=args.sdesol,
                save=args.save,
                )
