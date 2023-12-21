#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser(description="Plots a histogram of given samples")
    parser.add_argument('output', type=Path, help="File to save the histogram in")
    parser.add_argument('bins', type=int, nargs=2, help="Bin range")
    parser.add_argument('samples', type=Path, nargs='+', help="Samples to take")
    args = parser.parse_args()

    dfs = map(pd.read_parquet, args.samples)
    pops = tuple(map(lambda x: x.population, dfs))
    binrange = range(*args.bins)
    stems = tuple(v.stem for v in args.samples)
    df = pd.concat(pops, axis=1, keys=stems)
    alpha = 1./len(pops)
    ax0 = pops[0].hist(bins=binrange, alpha=alpha, label=stems[0])
    for pop, label in zip(pops[1:], stems[1:]):
        pop.hist(ax=ax0, bins=binrange, alpha=alpha, label=label)
    plt.legend()
    plt.savefig(args.output, dpi=600)


