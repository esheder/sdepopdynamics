"""Tests for specific data because they seem to malfunction"""

import json
from pathlib import Path
import numpy as np
from biopop_closure.kolmogorov import kolmogorov_multiplicity
from biopop_closure.moment_closures import moment
from itertools import product
import pytest

cpath = Path(__file__).parent.parent / "runs"
fnames = [
    f"{s}_immigration{'_multiple_birth' if multi else ''}.json" for s, multi in product(("high", "low"), (True, False))
]
dnames = [cpath / f"european_{animal}" for animal in ("badgers", "foxes")]


def _load_data(path):
    with path.open("r") as f:
        return json.load(f)


def _key_replacement(d, dr):
    return {dr[key] if key in dr else key: val for key, val in d.items()}


@pytest.mark.parametrize("path", [dname / fname for dname, fname in product(dnames, fnames)])
def test_kolmogorov_for_badgers_rises(path):
    data = _key_replacement(_load_data(path), {"I": "i", "multiplicity": "m"})
    pop = kolmogorov_multiplicity(**data, t=np.linspace(0, 20, 20), p0=5, nmax=300)
    mpop = moment(pop, k=1)
    assert np.all(np.diff(mpop))
