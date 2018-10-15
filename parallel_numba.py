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
    """ Calculates the execution configuration optimal for maximum 
    occupancy of the grid for launching a CUDA kernel.

    Args:
        m (int): number of rows of matrix.
        n (int): number of cols of matrix.

    Returns:
        tuple of `int`: grid dimensions for launching kernel, equivalent to `dim3` type.
        tuple of `int`: block dimensions for launching kernel, equivalent to `dim3` type.
    """
    dimBlock = (NUM_THREADS, NUM_THREADS)
    dimGrid = ((n // dimBlock[0]) + 1, (m // dimBlock[1]) + 1)
    return dimGrid, dimBlock

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
def sum(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] + value

@cuda.jit
def prod(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] * value

@cuda.jit
def maximum(a, value, c, m, n):
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
    sum[dimGrid, dimBlock](a, value, c, M, N)
    assert np.all((a + value) == c), 'elem-wise sum op is buggy'
    prod[dimGrid, dimBlock](a, value, c, M, N)
    assert np.all((a * value) == c), 'elem-wise prod op is buggy'
    maximum[dimGrid, dimBlock](a, 0., c, M, N)
    assert np.all(np.maximum(a, 0.) == c), 'elem-wise max op is buggy'

    print('Passed all tests!')


