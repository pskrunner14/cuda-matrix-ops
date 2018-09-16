#include <iostream>
#include <math.h>
#include "devices.cu"

// Execute with: nvcc add_1.cu -o add_1_cuda
// Profile with: nvprof ./add_1_cuda (takes around 0.2 secs on NVIDIA GTX 1050ti - compute cap 6.1)

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__    // device code - runs on GPU
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
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
    add<<<1, 1>>>(N, x, y);

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