#!/usr/bin/env python3
import pandas as pd
from argparse import ArgumentParser
from scipy.stats import shapiro, anderson
from pathlib import Path
from pprint import pprint

if __name__ == '__main__':
    parser = ArgumentParser(description="Test sample for normality with Shapiro-Wilks")
    parser.add_argument('sample', type=Path, help="Sample parquet file")
    args = parser.parse_args()
    sample = pd.read_parquet(args.sample).population
    pprint(shapiro(sample.head(4500)))
    pprint(anderson(sample, dist="norm"))

