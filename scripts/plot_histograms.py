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
    df = pd.concat(pops, axis=1, keys=map(lambda x: x.stem, args.samples))
    ax = df.hist(bins=binrange, alpha=1./len(args.samples))
    plt.savefig(args.output, dpi=600)


