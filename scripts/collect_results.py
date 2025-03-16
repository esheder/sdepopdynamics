#!/usr/bin/env python3
import argparse
from typing import Iterable
from os import listdir
from pathlib import Path
import pandas as pd
import numpy as np


full_range = (-np.inf, np.inf)


def combine(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(dfs), axis=0).reset_index(drop=True)


def load_df(path: Path, tframe: tuple[float, float] = full_range):
    df = pd.read_parquet(path)
    ti, te = tframe
    return df[df.Time.between(ti, te)].reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tool to generate single csv file with data")
    p.add_argument("fn", type=Path, nargs="*", default=None, help="Parquet files to collect")
    p.add_argument("-o", type=Path, default=None, help="Output path")
    p.add_argument("-d", type=Path, default=None, help="Directory to take all parquet files in")
    p.add_argument(
        "-t",
        type=float,
        nargs=2,
        default=full_range,
        help="Time to look in. When data is too large, we can drop all but the relevant times",
    )
    args = p.parse_args()
    if args.fn is None and args.d is None:
        raise ValueError("Must supply either a list of files (positional arguments) or a directory to look at (-d)")
    if args.d is not None:
        fs = listdir(args.d)
        parquets = (path for path in fs if path.ext == "parquet")
    else:
        parquets = args.fn
    dfs = (load_df(path, args.t) for path in parquets)
    total = combine(dfs)
    if args.o is None:
        print(total)
    else:
        total.to_parquet(args.o)
