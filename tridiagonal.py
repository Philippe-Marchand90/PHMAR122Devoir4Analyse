import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

def tridiagonal(D, I, S, b):
    N = len(D)
    A = sp.diags([I, D, S], offsets=[-1, 0, 1], shape=(N, N), format='csr')
    return spla.spsolve(A, b)

def dense_tridiagonal(D, I, S, b):
    N = len(D)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = D[i]
        if i > 0:
            A[i, i-1] = I[i-1]
        if i < N-1:
            A[i, i+1] = S[i]
    return np.linalg.solve(A, b)

if __name__ == '__main__':

    N = 15000
    D = 4.0 * np.ones(N)
    I = np.ones(N-1)
    S = np.ones(N-1)
    b = np.random.rand(N)

    t0 = time.time()
    x_dense = dense_tridiagonal(D, I, S, b)
    t_dense = time.time() - t0

    t1 = time.time()
    x_sparse = tridiagonal(D, I, S, b)
    t_sparse = time.time() - t1
    print(f"Dense solve time: {t_dense:.4f} s")
    print(f"Sparse solve time: {t_sparse:.4f} s")