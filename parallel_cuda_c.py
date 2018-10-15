"""
CUDA PARALLEL PROGRAMMING: cuda_c_ops.py
*  Purpose: Python interface for performing matrix operations using CUDA C/C++
*  @author Prabhsimran Singh
*  @version 2.2 15/10/18
*  Build shared object library using:
    nvcc -Xcompiler -fPIC -shared -o lib/cuda_mat_ops.so ops/matrix_ops.cu
"""
import ctypes
import numpy as np

from ctypes import POINTER, c_double, c_int

# extract cuda function pointers in the shared object cuda_c.so
dll = ctypes.CDLL('./lib/cuda_mat_ops.so', mode=ctypes.RTLD_GLOBAL)

def get_cuda_device_info(dll):
    func = dll.cuda_device_info
    return func

def get_cuda_matmul(dll):
    func = dll.cuda_matmul
    func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int]
    return func

def get_cuda_matsum(dll):
    func = dll.cuda_matsum
    func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
    return func

def get_cuda_matprod(dll):
    func = dll.cuda_matprod
    func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
    return func

def get_cuda_elemwise_sum(dll):
    func = dll.cuda_elemwise_sum
    func.argtypes = [POINTER(c_double), c_double, POINTER(c_double), c_int, c_int]
    return func

def get_cuda_elemwise_prod(dll):
    func = dll.cuda_elemwise_prod
    func.argtypes = [POINTER(c_double), c_double, POINTER(c_double), c_int, c_int]
    return func

def get_cuda_elemwise_max(dll):
    func = dll.cuda_elemwise_max
    func.argtypes = [POINTER(c_double), c_double, POINTER(c_double), c_int, c_int]
    return func

__cuda_device_info = get_cuda_device_info(dll)
__cuda_matmul = get_cuda_matmul(dll)
__cuda_matsum = get_cuda_matsum(dll)
__cuda_matprod = get_cuda_matprod(dll)
__cuda_elemwise_sum = get_cuda_elemwise_sum(dll)
__cuda_elemwise_prod = get_cuda_elemwise_prod(dll)
__cuda_elemwise_max = get_cuda_elemwise_max(dll)

# convenient python wrappers for cuda functions
def cuda_device_info():
    __cuda_device_info()

def cuda_matmul(a, b, c, m, n, k):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_p = b.ctypes.data_as(POINTER(c_double))
    c_p = c.ctypes.data_as(POINTER(c_double))
    __cuda_matmul(a_p, b_p, c_p, m, n, k)

def cuda_matsum(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_p = b.ctypes.data_as(POINTER(c_double))
    c_p = c.ctypes.data_as(POINTER(c_double))
    __cuda_matsum(a_p, b_p, c_p, m, n)

def cuda_matprod(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_p = b.ctypes.data_as(POINTER(c_double))
    c_p = c.ctypes.data_as(POINTER(c_double))
    __cuda_matprod(a_p, b_p, c_p, m, n)

def cuda_elemwise_sum(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_f = ctypes.c_double(b)
    c_p = c.ctypes.data_as(POINTER(c_double))
    __cuda_elemwise_sum(a_p, b_f, c_p, m, n)

def cuda_elemwise_prod(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_f = ctypes.c_double(b)
    c_p = c.ctypes.data_as(POINTER(c_double))
    __cuda_elemwise_prod(a_p, b_f, c_p, m, n)

def cuda_elemwise_max(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_double))
    b_f = ctypes.c_double(b)
    c_p = c.ctypes.data_as(POINTER(c_double))
    __cuda_elemwise_max(a_p, b_f, c_p, m, n)

def get_test_params():
    size = int(16)
    a = np.array([3.0] * (size * size))
    b = np.array([3.0] * (size * size))
    c = np.zeros(shape=(size * size))
    return a, b, c, size

if __name__ == '__main__':
    cuda_device_info()

    a, b, c, size = get_test_params()
    # basic checks for all ops
    cuda_matmul(a, b, c, size, size, size)
    assert np.all(c==144.0), "Matrix dot-product operation is buggy"
    cuda_matsum(a, b, c, size, size)
    assert np.all(c==6.0), "Matrix sum operation is buggy"
    cuda_matprod(a, b, c, size, size)
    assert np.all(c==9.0), "Matrix product operation is buggy"
    cuda_elemwise_sum(a, 5.0, c, size, size)
    assert np.all(c==8.0), "Element-wise sum operation is buggy"
    cuda_elemwise_prod(a, 2.5, c, size, size)
    assert np.all(c==7.5), "Element-wise product operation is buggy"
    cuda_elemwise_max(a, 4.0, c, size, size)
    assert np.all(c==4.0), "Element-wise max operation is buggy"

    # robust check for matmul
    a = np.random.randn(205, 510)
    b = np.random.randn(510, 340)
    c = np.zeros(205 * 340)
    cuda_matmul(a.flatten(), b.flatten(), c, 205, 510, 340)
    actual_dot = np.dot(a, b)
    c = c.reshape(205, 340)
    assert np.allclose(actual_dot, c), "Matrix dot-product operation is buggy"

    # robust checks for other ops
    a = np.random.randn(100 * 200)
    b = np.random.randn(100 * 200)
    c = np.zeros_like(a)
    cuda_matsum(a, b, c, 100, 200)
    assert np.all(a + b == c), "Matrix sum operation is buggy"
    cuda_matprod(a, b, c, 100, 200)
    assert np.all(a * b == c), "Matrix product operation is buggy"
    cuda_elemwise_sum(a, 5.3, c, 100, 200)
    assert np.all(a + 5.3 == c), "Element-wise sum operation is buggy"
    cuda_elemwise_prod(a, 6, c, 100, 200)
    assert np.all(a * 6 == c), "Element-wise product operation is buggy"
    cuda_elemwise_max(a, 0, c, 100, 200)
    assert np.all(np.maximum(0, a) == c), "Element-wise max operation is buggy"

    print('Passed all tests!')