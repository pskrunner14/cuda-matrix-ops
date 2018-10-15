#include <iostream>
#include <stdio.h>
#include <assert.h>

void init(int *a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = i;
    }
}

__global__ void double_elements(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        a[i] *= 2;
    }
}

bool checkElementsAreDoubled(int *a, int n) {
    for (int i = 0; i < n; ++i) {
        if (a[i] != i*2) 
            return false;
    }
    return true;
}

int main() {

    int n = 10000;
    int *a;

    size_t size = n * sizeof(int);
    cudaMallocManaged(&a, size);

    init(a, n);

    // Do not attempt to launch the kernel with more than 
    // the maximum number of threads per block, which is 1024.
    size_t threads_per_block = 1024;
    size_t number_of_blocks = 32;

    cudaError_t syncErr, asyncErr;
    double_elements<<<number_of_blocks, threads_per_block>>>(a, n);

    // Catch errors for both the kernel launch above and any
    // errors that occur during the asynchronous `double_elements`
    // kernel execution.
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();

    if (syncErr != cudaSuccess) 
        std::cout << "CUDA Error: " << cudaGetErrorString(syncErr) << std::endl;
    if (asyncErr != cudaSuccess) 
        std::cout << "CUDA Error: " << cudaGetErrorString(asyncErr) << std::endl;

    bool areDoubled = checkElementsAreDoubled(a, n);
    std::cout << "All elements were doubled? " << (areDoubled ? "TRUE" : "FALSE") << std::endl;

    cudaFree(a);
}

// Macro that wraps CUDA function calls for checking errors
// ex.: checkCuda(cudaDeviceSynchronize());
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}