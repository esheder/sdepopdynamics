#!/usr/bin/env python3
import argparse
from os import listdir
from pathlib import Path


def parse_file(s: str) -> list[float]:
    return [float(v.strip()) for v in s.strip()[1:-1].split(',')]


def get_seed_off_fn(fn: str) -> int:
    return int(fn.split('_')[-1][:-4])

def get_number_of_lines(fn: Path) -> int:
    with fn.open('rb') as f:
        return sum(1 for _ in f)

def get_seeds(argseeds: list[int], fn: Path) -> list[int]:
    if len(argseeds) != 2:
        return argseeds
    lines = get_number_of_lines(fn)
    if lines == 2:
        return argseeds
    return list(range(*argseeds))

def csv_format(s, t, v): 
    return f"{s}, {t:.2f}, {v}"


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Tool to generate single csv file with data")
    p.add_argument("--directory", type=Path, default=None, help="Directory where files are at")
    p.add_argument('--file', type=Path, help="File that includes lots of result lines")
    p.add_argument('o', type=Path, help="Output path")
    p.add_argument("t", type=float, nargs="+", help="Times list")
    p.add_argument("--seeds", required=True, type=int, nargs="+", 
            help="Seeds. If 2 are given, assumes a list between the two, if applicable.")
    args = p.parse_args()
    with args.o.open('w') as of:
        of.write("seed,time,population\n")
        if args.directory:
            for fn in map(lambda x: args.d / x, listdir(args.d)):
                with fn.open('r') as f:
                    vec = parse_file(f.read())
                seed = get_seed_off_fn(str(fn))
                of.write('\n'.join(csv_format(seed, t, v) for t, v in zip(args.t, vec)))
        if args.file:
            seeds = get_seeds(args.seeds, args.file)
            with args.file.open('r') as f:
                vecs = (parse_file(line) for line in f)
                of.write("\n".join(csv_format(seed, t, v) 
                    for seed, vec in zip(seeds, vecs) 
                    for t, v in zip(args.t, vec)))


