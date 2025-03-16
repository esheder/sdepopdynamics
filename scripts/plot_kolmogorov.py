#!/usr/bin/env python3

import json
import argparse
import numpy as np
from pathlib import Path
from biopop_closure.kolmogorov import kolmogorov_multiplicity
from biopop_closure.moment_closures import moment
import matplotlib.pyplot as plt


def _key_replacement(d, dr):
    return {dr[key] if key in dr else key: val for key, val in d.items()}


parser = argparse.ArgumentParser(description="Tool to visualize json parameter results")
parser.add_argument("json", type=Path, help="Path to json file with parameters")
parser.add_argument("p0", type=int, help="Initial population")
parser.add_argument("--nmax", type=int, default=100, help="Maximal population size to cut off at")
parser.add_argument("--tmax", type=float, default=20.0, help="Maximal time to look at")
parser.add_argument("--tnum", type=int, default=40, help="Time points in [0, tmax] to use")

args = parser.parse_args()
with args.json.open("r") as f:
    data = json.load(f)
data = _key_replacement(data, {"I": "i", "multiplicity": "m"})


t = np.linspace(0, args.tmax, args.tnum)
kpop = kolmogorov_multiplicity(**data, t=t, p0=args.p0, nmax=args.nmax)
m = moment(kpop, 1)
m2 = moment(kpop, 2)
v = m2 - m**2

plt.figure()
plt.plot(t, m, "b.-", label="Expectation")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Expectation")
plt.tight_layout()

plt.figure()
plt.plot(t, v, "b.-", label="Variance")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Variance")
plt.tight_layout()

plt.figure()
klast = kpop[:, -1]
plt.stairs(klast, np.arange(klast.size + 1), fill=True, color="b")
plt.grid()
plt.xlabel("Population")
plt.ylabel("Probability")
plt.tight_layout()


plt.show()
