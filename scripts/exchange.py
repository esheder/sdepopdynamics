#!/usr/bin/env python3
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser('Turn CSV to parquet')
    parser.add_argument('i', type=Path, help="CSV file")
    parser.add_argument('o', type=Path, help="Parquet output")
    args = parser.parse_args()
    df = pd.read_csv(args.i)
    df.to_parquet(args.o)
