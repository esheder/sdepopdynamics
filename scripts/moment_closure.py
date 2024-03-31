"""Tools to model the effects of the moment closures

"""

import warnings
from functools import partial

import numpy as np

from biopop_closure.kolmogorov import kolmogorov
from biopop_closure.moment_closures import (
    moment_closure, dkdt, relative_skewness, gaussian
)


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
    import matplotlib.pyplot as plt
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


def compare(pop, ilow, ihigh):
    import matplotlib.pyplot as plt
    t = np.linspace(0, 50, 500)
    p0 = 6
    k2low = pop(ilow, t, p0, func=partial(moment_closure, gaussian))
    k3low = pop(ilow, t, p0, func=partial(moment_closure, dkdt))
    kollow = pop(ilow, t, p0, func=kolmogorov)
    k2high = pop(ihigh, t, p0, func=partial(moment_closure, gaussian))
    k3high = pop(ihigh, t, p0, func=partial(moment_closure, dkdt))
    kolhigh = pop(ihigh, t, p0, func=kolmogorov)
    def relerr(x, y):
        vals = 1e2 * (x - y) / y
        vals[y<1] = np.nan
        return vals
    plt.figure()
    plt.plot(t, relerr(k2low, kollow)[0, :], 'b-', label=f"K2 Mean I={ilow}")
    plt.plot(t, relerr(k2low, kollow)[1, :], 'b--', label=f"K2 Variance I={ilow}")
    plt.plot(t, relerr(k3low, kollow)[0, :], 'r-', label=f"K3 Mean I={ilow}")
    plt.plot(t, relerr(k3low, kollow)[1, :], 'r--', label=f"K3 Variance I={ilow}")
    plt.plot(t, relerr(k2high, kolhigh)[0, :], 'm-', label=f"K2 Mean I={ihigh}")
    plt.plot(t, relerr(k2high, kolhigh)[1, :], 'm--', label=f"K2 Variance I={ihigh}")
    plt.plot(t, relerr(k3high, kolhigh)[0, :], 'k-', label=f"K3 Mean I={ihigh}")
    plt.plot(t, relerr(k3high, kolhigh)[1, :], 'k--', label=f"K3 Variance I={ihigh}")
    plt.xlabel("Time [yrs]")
    plt.ylabel("Relative Error [%]")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Tool to replicate the paper by Matis")
    parser.add_argument('pop', choices={'badgers', 'foxes'}, help="Population type")
    parser.add_argument('order', choices={'plot', 'compare'}, help="What to do")
    parser.add_argument('--closure', choices={'gaussian', 'k3', 'exact'}, default='k3', help="Closure type")
    parser.add_argument('--save', default='', help="Prefix for the files to save things as.")
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
        compare(*pop)
