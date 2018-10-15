import numpy as np
from numba import cuda

@cuda.jit
def matmul(a, b, c, m, n, k):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    stride_row = cuda.gridDim.y * cuda.blockDim.y
    stride_col = cuda.gridDim.x * cuda.blockDim.x

    while row < m and col < k:
        summ = 0
        for i in range(n):
            summ += a[row, i] * b[i, col]
        c[row, col] = summ
        row += stride_row
        col += stride_col


if __name__ == '__main__':
    M = 324
    N = 432
    K = 412
    a = np.random.randn(M, N)
    b = np.random.randn(N, K)
    c = np.zeros(shape=(M, K))

    threadsperblock = (32, 32)
    blockspergrid = ((K // threadsperblock[0]) + 1, (M // threadsperblock[1]) + 1)
    matmul[blockspergrid, threadsperblock](a, b, c, M, N, K)
    assert np.allclose(np.dot(a, b), c), 'matmul op is buggy'
    assert not np.isnan(np.sum(c)), 'matmul op is buggy'