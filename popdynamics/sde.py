"""SDE model for the population dynamics.

"""
import numpy as np

from popdynamics.branching import Parameters


def sample_at_time_forward_euler(
        n0: int,
        params: Parameters,
        t: float,
        rgen: np.random.Generator,
        dt: float,
) -> int:
    n = n0
    tsteps: int = int(t // dt) + 1
    times = np.linspace(0, t, tsteps)
    dt = np.diff(times)[0]
    randoms = rgen.normal(0., np.sqrt(dt), size=len(times) - 1)
    for dw in randoms:
        mu = n * params.birth(n) * params.barnu + params.I - n * params.death(n)
        sigma = np.sqrt(params.I + n * params.death(n) + n * params.barnu2 * params.birth(n))
        n = n + mu * dt + sigma * dw
    return n


if __name__ == '__main__':
    system = Parameters(1.5, 2., 0., 0.001, 1., [0., 0.5, 0.5])
    rnd = np.random.Generator(np.random.SFC64(48))
    n0 = 10000
    n = sample_at_time_forward_euler(n0, system, 20., rnd, 1e-4)
    print(n)
