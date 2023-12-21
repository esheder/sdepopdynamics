#!/usr/bin/env python3
import pandas as pd
from scipy.stats import ks_2samp
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser(description="KS test parquet file samples")
    parser.add_argument('samples', nargs=2, type=Path, help="Parquet Sample files")
    args = parser.parse_args()
    df1, df2 = map(pd.read_parquet, args.samples)
    print(ks_2samp(df1.population, df2.population))

