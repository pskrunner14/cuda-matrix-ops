#include <iostream>
#include <math.h>
#include "../ops/utils/devices.cu"

// Execute with: nvcc add_multi_block.cu -o add_multi_block_cuda
// Profile with: nvprof ./add_multi_block_cuda (takes around 0.0025 secs with 256 parallel threads on multiple thread blocks on NVIDIA GTX 1050ti - compute cap 6.1)

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void addMatrix(float *x, float *y, int n) { // device code - runs on GPU
    int index = blockIdx.x * blockDim.x + threadIdx.x; // modify code to spread computation across parallel thread block
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] += x[i];
}

int main() {

    getCudaDeviceInfo();

    int N = 1<<20; // 1M elements

    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // init x and y arrays
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::cout << numBlocks << std::endl;

    // run kernel on 1M elements on the GPU
    add<<<numBlocks, blockSize>>>(N, x, y); // execution config (no. of parallel threads to use for the launch of the GPU)
    // <<< no.of thread blocks, no. of threads in a thread block>>>
    // grid(4096 blocks) is blocks(256 threads) of threads

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