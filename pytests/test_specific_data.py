"""Tests for specific data because they seem to malfunction

"""

import json
from pathlib import Path
import numpy as np
from biopop_closure.kolmogorov import kolmogorov_multiplicity
from biopop_closure.moment_closures import moment

cpath = Path(__file__).parent.parent / 'runs' / 'european_badgers' / 'multiple_birth_high_immigration.json'
cpath = cpath.parent / 'high_immigration.json'


def _load_data(path):
    with path.open('r') as f:
        return json.load(f)


def _key_replacement(d, dr):
    return {dr[key] if key in dr else key: val for key, val in d.items()}


def test_kolmogorov_for_badgers_rises():
    data = _key_replacement(_load_data(cpath), {'I': 'i', 'multiplicity': 'm'})
    pop = kolmogorov_multiplicity(**data,
                                  t=np.linspace(0, 10, 5),
                                  p0=5,
                                  nmax=300
                                  )
    mpop = moment(pop, k=1)
    assert np.all(np.diff(mpop))
