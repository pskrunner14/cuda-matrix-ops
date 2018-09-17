#include <iostream>
#include <math.h>
#include "utils/devices.cu"

// Execute with: nvcc add_multi.cu -o add_multi_cuda
// Profile with: nvprof ./add_multi_cuda (takes around 0.003 secs with 256 parallel threads on NVIDIA GTX 1050ti - compute cap 6.1)

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int n, float *x, float *y) { // device code - runs on GPU
    int index = threadIdx.x; // modify code to spread computation across parallel threads
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] += x[i];
}

int main() {
    int N = 1<<20; // 1M elements

    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // init x and y arrays
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernel on 1M elements on the GPU
    add<<<1, 256>>>(N, x, y); // execution config (no. of parallel threads to use for the launch of the GPU)
    // <<< , no. of threads in a thread block>>>

    // wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // check for errors (all vals should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max Error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}