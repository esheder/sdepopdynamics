#!/usr/bin/env python3

import pandas as pd
import scipy.stats as stat
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser(description="Finds the skewness of the data and prints the time at which it is highest")
parser.add_argument("parquet", type=str, help="data to find skewness for")
parser.add_argument("-out", type=str, help="Output figure file prefix")
args = parser.parse_args()
df = pd.read_parquet(args.parquet)
s1 = df.groupby("time").population.apply(lambda x: stat.kstat(x, n=1))
s2 = df.groupby("time").population.apply(lambda x: stat.kstat(x, n=2))
s3 = df.groupby("time").population.apply(lambda x: stat.kstat(x, n=3))
relskew = s3.abs() / (s2**1.5)

if args.out:
    plt.figure()
    s1.plot()
    plt.savefig(f"{args.out}.k1.jpg", dpi=300)

    plt.figure()
    s2.plot()
    plt.savefig(f"{args.out}.k2.jpg", dpi=300)

    plt.figure()
    s3.plot()
    plt.savefig(f"{args.out}.k3.jpg", dpi=300)

    plt.figure()
    relskew.plot()
    plt.savefig(f"{args.out}.relskew.jpg", dpi=300)

print(f"Maximal Relative Skewness at time={relskew.idxmax()}")
