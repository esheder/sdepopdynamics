#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kstat


EPS = 1e-5


def get_df(p, t):
    df = pd.read_parquet(p)
    return df[df.Time.between(t-EPS, t+EPS)]

def get_moments(paths, t):
    pop = pd.concat([get_df(p, t) for p in paths], axis=0).reset_index(drop=True).Population.dropna().values
    m1 = kstat(pop, n=1)
    m2 = kstat(pop, n=2)
    m3 = kstat(pop, n=3)
    return {'time': t, 'k1': m1, 'k2': m2, 'k3': m3}


def get_moments_single(df, t):
    df = df[df.Time.between(t-EPS, t+EPS)]
    pop = df.reset_index(drop=True).Population.dropna().values
    m1 = kstat(pop, n=1)
    m2 = kstat(pop, n=2)
    m3 = kstat(pop, n=3)
    return {'time': t, 'k1': m1, 'k2': m2, 'k3': m3}


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description="Tool to generate the moments at each time, given a bunch of dataframes")
    p.add_argument('dfs', type=Path, nargs='+', help="Dataframe parquet files to use")
    p.add_argument('-o', type=Path, default=None, help="Output file for results")
    args = p.parse_args()

    df = pd.read_parquet(args.dfs[0])
    times = np.array(sorted(df.Time.unique()))
    m1, m2, m3 = [], [], []
    if len(args.dfs) > 1:
        records = [get_moments(args.dfs, time) for time in times]
    else:
        df = pd.read_parquet(args.dfs[0])
        records = [get_moments_single(df, time) for time in times]
    df = pd.DataFrame.from_records(records)
    if args.o is not None:
        df.to_parquet(args.o)
    else:
        print(df)


