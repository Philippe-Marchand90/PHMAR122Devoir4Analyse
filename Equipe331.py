import numpy as np
import matplotlib.pyplot as plt
from problimite import solve_boundary_problem


def y_exact(x):
    c = 0.4
    d = 0.81
    return (c - 0.4 / x**2) - (c - 0.4 / d) * np.log(x) / np.log(0.9)