import numpy as np
import ctypes

from ctypes import *

# Build cuda lib using: nvcc -Xcompiler -fPIC -shared -o lib/cuda_mat_mul.so matmul.cu

# extract cuda_mat_mul function pointer in the shared object cuda_mat_mul.so
def get_cuda_mat_mul():
    dll = ctypes.CDLL('./ops/lib/cuda_mat_mul.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_mat_mul
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int]
    return func

# create __cuda_mat_mul function with get_cuda_mat_mul()
__cuda_mat_mul = get_cuda_mat_mul()

# convenient python wrapper for __cuda_mat_mul
# it does all type convertions from python ones to C++ ones
def cuda_mat_mul(a, b, c, size):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_mat_mul(a_p, b_p, c_p, size)

def main():
    size = int(16)

    a = np.array([2.0] * (size * size)).astype('float32')
    b = np.array([3.0] * (size * size)).astype('float32')

    c = np.zeros(size * size).astype('float32')

    cuda_mat_mul(a, b, c, size)
    print(c)

# testing, matrix multiplication of two matrices
if __name__ == '__main__':
    main()