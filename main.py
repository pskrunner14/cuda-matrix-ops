import numpy as np
import ctypes

from ctypes import POINTER, c_float, c_int

# Build cuda lib using: nvcc -Xcompiler -fPIC -shared -o lib/cuda_mat_ops.so ops/matrix_ops.cu
# extract cuda function pointers in the shared object cuda_mat_ops.so
dll = ctypes.CDLL('./lib/cuda_mat_ops.so', mode=ctypes.RTLD_GLOBAL)

def get_cuda_mat_mul(dll):
    func = dll.cuda_mat_mul
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
    return func

def get_cuda_mat_sum(dll):
    func = dll.cuda_mat_sum
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int]
    return func


# create __cuda_mat_mul function with get_cuda_mat_mul()
__cuda_mat_mul = get_cuda_mat_mul(dll)
__cuda_mat_sum = get_cuda_mat_sum(dll)

# convenient python wrapper for cuda functions as it does 
# all type convertions for ex. from python ones to C++ ones

def cuda_mat_mul(a, b, c, m, n, k):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_mat_mul(a_p, b_p, c_p, m, n, k)

def cuda_mat_sum(a, b, c, size):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_mat_sum(a_p, b_p, c_p, size)

def get_test_params():
    size = int(16)
    a = np.array([2.0] * (size * size)).astype('float32')
    b = np.array([3.0] * (size * size)).astype('float32')
    c = np.zeros(size * size).astype('float32')
    return a, b, c, size

def main():
    # Matrix Multiplication
    a, b, c, size = get_test_params()
    cuda_mat_mul(a, b, c, size, size, size)
    assert np.all(c == 96.0), "Matrix multiplication is buggy"

    # Matrix Elementwise Addition
    a, b, c, size = get_test_params()
    cuda_mat_sum(a, b, c, size * size)
    assert np.all(c == 5.0), "Matrix addition is buggy"

    print("Passed all tests!")

if __name__ == '__main__':
    main()