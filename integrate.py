import numpy as np


def generate_grid(T, freq, method='trapezoidal'):
    # maybe implements simpson's method

    T = T
    freq = freq
    dt = 1 / freq
    t = np.arange(0, T + dt, dt)
    n = len(t) - 1

    weight = np.ones_like(t) * T / n
    weight[0] = weight[0] / 2
    weight[1] = weight[1] / 2
    return t, weight

