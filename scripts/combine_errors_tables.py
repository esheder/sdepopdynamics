#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser(description="Combine CSVs to latex table")
parser.add_argument('i', nargs='+', type=Path, help="CSV files")
args = parser.parse_args()
idfunc = {p.name.split('_error')[0]: p for p in args.i}
idrep = {"poph": ("Badger-like", "Medium"), 
         "huge": ("Badger-like", "High"),
         "popl": ("Badger-like", "Low"),
         "high_immigration_multiple_birth": ("European badgers", "High immigration"),
         "low_immigration_multiple_birth": ("European badgers", "Low immigration"),
         }


dfs = {key: pd.read_csv(path).assign(case=idrep[key][0], caseid=idrep[key][1]) 
       for key, path in idfunc.items() if key in idrep}
combined = pd.concat(dfs.values(), ignore_index=True)
def column_order(s):
    a = 0 if 'SDE' in s else 1 if 'K3' in s else 2
    b = 0 if s == 'case' else 1 if s == 'caseid' else 2 if 'sum' in s else 3
    return (b, a)
combined = combined[sorted(combined.columns, key=column_order)]

def formfunc(v):
    s = f"${v:.2e}".replace("e",r"\cdot10^{") + "}$"
    return s.replace("{-0","{-")

print(combined.to_latex(float_format=formfunc, index=False))
