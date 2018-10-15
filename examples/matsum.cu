#include <iostream>
#include "../ops/utils/devices.cu"
#include "../ops/utils/utils.cpp"

#define NUM_THREADS 32

__global__ void matsum(float* a, float* b, float* c, int m, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread computes one element of the block sub-matrix
    if (row < m && col < n) {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}

int main() {
    getCudaDeviceInfo();

    // Define matrix dimensions.
    int m = 20000;
    int n = 16327;

    // Allocate memory on the host.
    float *a, *b, *c;

    // Allocate memory on the device.
    cudaMallocManaged(&a, (m * n) * sizeof(float));
    cudaMallocManaged(&b, (m * n) * sizeof(float));
    cudaMallocManaged(&c, (m * n) * sizeof(float));

    // Dummy values.
    float x = 343.5534f;
    float y = 7254.1543f;

    // Initialize matrices on the host.
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = x;
            b[i * n + j] = y;
        }
    }

    // Define grid and block dimensions for the execution 
    // configuration with which to launch the CUDA kernel.
    dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
    dim3 dimGrid((n / dimBlock.x) + 1, (m / dimBlock.y) + 1, 1);

    // Launch kernel with specified exec config.
    matsum<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    // Wait for device to sync back with host.
    cudaDeviceSynchronize();

    // check for errors (all vals should be equal to (x + y))
    float maxError = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            maxError = fmax(maxError, fabs(c[i * n + j] - (x + y)));
        }
    }
    // Log errors if any.
    std::cout << "Max Error: " << maxError << std::endl;

    // Free memory allocated on the device.
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}