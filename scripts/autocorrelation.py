#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def autocorrelation(df, t1, t2):
    xt1, xt2 = df[df.Time == t1], df[df.Time == t2]
    rtt = np.mean(xt1.Population.values * xt2.Population.values)
    means = float(xt1.Population.mean()) * float(xt2.Population.mean())
    return rtt - means


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Tool to compute autocorrelations")
    parser.add_argument("sde", help="SDE parquet path")
    parser.add_argument("ref", help="Branching parquet path")
    parser.add_argument("-o", default=None, help="Output path for figure")
    args = parser.parse_args()

    dfsde = pd.read_parquet(args.sde)
    dfref = pd.read_parquet(args.ref)

    times = list(sorted(dfsde[dfsde.Time >= 10.0].Time.unique()))
    autosde, autoref = map(np.array, ([autocorrelation(df, times[0], t) for t in times[1:]] for df in (dfsde, dfref)))
    dt = [t-times[0] for t in times[1:]]
   
    plt.rcParams.update({'font.size': 20})
    plt.figure()
    plt.plot(dt, autosde, 'bo-', linewidth=2, label="SDE")
    plt.plot(dt, autoref, 'kx-', linewidth=2, label="Branching")
    plt.xlabel("Time interval")
    plt.ylabel("Autocovariance")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if args.o is not None:
        plt.savefig(args.o, dpi=300)

    plt.figure()
    plt.plot(dt, 100*(autosde - autoref) / autoref, 'ko-')
    plt.xlabel("Time interval")
    plt.ylabel("Relative Error [%]")
    plt.grid()
    plt.tight_layout()
    if args.o is not None:
        spath = Path(args.o)
        plt.savefig(spath.with_stem(spath.stem + '_error'), dpi=300)
    else:
        plt.show()
    
