"""
CUDA PARALLEL PROGRAMMING: parallel_numba.py
*  Purpose: Python code for performing matrix operations on the GPU using Numba CUDA.JIT
*  @author Prabhsimran Singh
*  @version 1.0 15/10/18
"""
import numpy as np
from numba import cuda

NUM_THREADS = 32

def get_cuda_execution_config(m, n):
    gridBlock = (NUM_THREADS, NUM_THREADS)
    gridDim = ((n // gridBlock[0]) + 1, (m // gridBlock[1]) + 1)
    return gridDim, gridBlock

@cuda.jit
def matmul(a, b, c, m, n, k):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < k:
        summ = 0
        for i in range(n):
            summ += a[row, i] * b[i, col]
        c[row, col] = summ

@cuda.jit
def matsum(a, b, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] + b[row, col]

@cuda.jit
def matprod(a, b, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] * b[row, col]

@cuda.jit
def elemwise_sum(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] + value

@cuda.jit
def elemwise_prod(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] * value

@cuda.jit
def elemwise_max(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] if a[row, col] > value else value

if __name__ == '__main__':
    # matmul
    M = 324
    N = 432
    K = 412
    a = np.random.randn(M, N)
    b = np.random.randn(N, K)
    c = np.zeros(shape=(M, K))
    dimGrid, dimBlock = get_cuda_execution_config(M, K)
    matmul[dimGrid, dimBlock](a, b, c, M, N, K)
    assert np.allclose(np.dot(a, b), c), 'matmul op is buggy'
    assert not np.isnan(np.sum(c)), 'matmul op is buggy'

    M = 5425
    N = 4123
    a = np.random.randn(M, N)
    b = np.random.randn(M, N)
    c = np.zeros_like(a)
    value = 546.432
    dimGrid, dimBlock = get_cuda_execution_config(M, N)

    # other ops
    matsum[dimGrid, dimBlock](a, b, c, M, N)
    assert np.all((a + b) == c), 'matsum op is buggy'
    matprod[dimGrid, dimBlock](a, b, c, M, N)
    assert np.all((a * b) == c), 'matprod op is buggy'
    elemwise_sum[dimGrid, dimBlock](a, value, c, M, N)
    assert np.all((a + value) == c), 'elem-wise sum op is buggy'
    elemwise_prod[dimGrid, dimBlock](a, value, c, M, N)
    assert np.all((a * value) == c), 'elem-wise prod op is buggy'
    elemwise_max[dimGrid, dimBlock](a, 0., c, M, N)
    assert np.all(np.maximum(a, 0.) == c), 'elem-wise max op is buggy'

    print('Passed all tests!')


