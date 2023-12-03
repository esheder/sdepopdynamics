#!/usr/bin/env python3
import argparse
from os import listdir
from pathlib import Path


def parse_file(s: str) -> list[float]:
    return [float(v.strip()) for v in s.strip()[1:-1].split(',')]


def get_seed_off_fn(fn: str) -> int:
    return int(fn.split('_')[-1][:-4])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Tool to generate single csv file with data")
    p.add_argument("d", type=Path, help="Directory where files are at")
    p.add_argument('o', type=Path, help="Output path")
    p.add_argument("t", type=float, nargs="+", help="Times list")
    args = p.parse_args()
    with args.o.open('w') as of:
        of.write("seed,time,population\n")
        for fn in map(lambda x: args.d / x, listdir(args.d)):
            with fn.open('r') as f:
                vec = parse_file(f.read())
            seed = get_seed_off_fn(str(fn))
            for t, v in zip(args.t, vec):
                of.write(f"{seed}, {t:.2f}, {v}\n")
