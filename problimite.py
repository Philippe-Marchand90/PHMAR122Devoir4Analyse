import numpy as np
from tridiagonal import tridiagonal

def solve_boundary_problem(h, P, Q, R, a, b, alpha, beta):
    N = len(P) - 2
    D = np.zeros(N)
    I = np.zeros(N-1)
    S = np.zeros(N-1)
    b_vec = np.zeros(N)

    for i in range(N):
        pi = P[i+1]
        qi = Q[i+1]
        ri = R[i+1]
        D[i] = 2 + qi * h**2
        b_vec[i] = - ri * h**2
        if i > 0:
            I[i-1] = -1 - pi * h / 2
        if i < N-1:
            S[i] = -1 + pi * h / 2
    
    b_vec[0]     += (1 + P[1] * h / 2) * alpha
    b_vec[-1]    += (1 - P[N] * h / 2) * beta

    y_inner = tridiagonal(D, I, S, b_vec)

    y = np.empty(N+2)
    y[0]       = alpha
    y[1:-1]    = y_inner
    y[-1]      = beta
    return y