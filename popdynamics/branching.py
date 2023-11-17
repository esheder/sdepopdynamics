"""We have a stochastic branching process with immigration.

We are interested in the asymptotic stationary distribution this tends to.
We want to sample this population size distribution.

Therefore, we must generate population sizes from this distribution somehow.
Solving the equations is probably difficult, but we can numerically follow a long
path and sample it every once in a while. With enough luck, far samples will be
independent.

"""
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Parameters:
    r"""Problem definition information.

    .. math::
        \mu(n) = a_2 + b_2n

    .. math::
        \lambda(n) = a_1 - b_1n

    Parameters
    ----------
    a1: float
        Constant per-individual birth rate.
    a2: float
        Constant per-individual death rate.
    b1: float
        Birth feedback coefficient.
    b2: float
        Death feedback coefficient
    I: float
        Immigration rate.
    multiplicity: Sequence[float]
        Vector of probabilities for population increase in a birth event.

    """
    a1: float
    a2: float
    b1: float
    b2: float
    I: float
    multiplicity: Sequence[float] = field(hash=False, compare=False)
    barnu: float = field(init=False, repr=False, hash=False, compare=False)
    barnu2: float = field(init=False, repr=False, hash=False, compare=False)
    _cdf_multiplicity: np.ndarray = field(init=False, repr=False, hash=True, compare=True)

    def __post_init__(self):
        self._cdf_multiplicity = np.cumsum(np.asarray(self.multiplicity))
        if not np.isclose(self._cdf_multiplicity[-1], 1.):
            raise ValueError("The multiplicity vector should sum up to 1!")
        _range = np.arange(1, len(self.multiplicity)+1)
        self.barnu = float(np.sum(_range * self.multiplicity))
        self.barnu2 = float(np.sum((_range ** 2) * self.multiplicity))

    def population_rate(self, n: int) -> float:
        return self.birth(n) + self.death(n)

    def death(self, n: int) -> float:
        return self.a2 + self.b2*n

    def birth(self, n: int) -> float:
        return self.a1 - self.b1*n

    def rate(self, n: int) -> float:
        return n*self.population_rate(n) + self.I

    def pvec(self, n: int) -> np.ndarray:
        pdf_vector = np.array([n*self.death(n), n*self.birth(n), self.I]) / self.rate(n)
        return np.cumsum(pdf_vector)

    def choose_children(self, rgen: np.random.Generator) -> int:
        uniform = rgen.uniform()
        index = np.argmin(self._cdf_multiplicity <= uniform)
        return index + 1

    def population_diff(self, case: int, rgen: np.random.Generator) -> int:
        match case:
            case 0:  # Death
                return -1
            case 1:
                return self.choose_children(rgen)
            case 2:
                return 1
            case _:
                raise ValueError("We only allow for 3 cases: Death, Birth and Immigration")


def event(n: int, params: Parameters, rgen: np.random.Generator):
    p_vec = params.pvec(n)
    rv = rgen.uniform()
    return params.population_diff(np.argmin(p_vec <= rv), rgen)


def sample_at_time(n0: int,
                   params: Parameters,
                   t: float,
                   rgen: np.random.Generator) -> int:
    now = 0.
    n = n0
    while True:
        lifetime = 1. / params.rate(n)
        min_time = now + rgen.exponential(lifetime)
        next_event = min(min_time, t)
        if next_event >= t:
            return n
        n += event(n, params, rgen)
        now = next_event


if __name__ == '__main__':
    system = Parameters(1.5, 2., 0., 0.01, 1., [0., 1.])
    rnd = np.random.Generator(np.random.SFC64(48))
    n0 = 100
    n = sample_at_time(n0, system, 50., rnd)
    print(n)
