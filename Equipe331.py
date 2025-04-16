import numpy as np
import matplotlib.pyplot as plt
from problimite import solve_boundary_problem


def exact(x):
    c = 0.4
    d = 0.81
    return (c - 0.4 / x**2) - (c - 0.4 / d) * np.log(x) / np.log(0.9)

def simulation(h):
    a, b = 0.9, 1.0

    N = int(round((b - a) / h))
    x = np.linspace(a, b, N + 1)
    P = -1.0 / x
    Q = np.zeros_like(x)
    R = -1.6 / x**4
    y = solve_boundary_problem((b - a) / N, P, Q, R, a, b, alpha=0.0, beta=0.0)
    return x, y


