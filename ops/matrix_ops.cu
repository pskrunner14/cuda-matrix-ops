#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "utils/devices.cu"
#include "utils/utils.cpp"

#define BLOCK_SIZE 256

// Build using: nvcc -Xcompiler -fPIC -shared -o lib/cuda_mat_mul.so matmul.cu

__global__ void matSum(float *a, float *b, float *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        c[i] = a[i] + b[i];
}

__global__ void matMul(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    if (row < m && col < k) {
        for (int i = 0; i < n; i++)
            sum += a[row * n + i] * b[i * n + col];
        c[row * k + col] = sum;
    }
}

extern "C" {

    void cuda_mat_sum(float *a, float *b, float *c, int n) {
        float *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, n * sizeof(float));
        cudaMallocManaged(&d_b, n * sizeof(float));
        cudaMallocManaged(&d_c, n * sizeof(float));

        cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

        matSum<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void cuda_mat_mul(float *a, float *b, float *c, int m, int n, int k) {
        float *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(float));
        cudaMallocManaged(&d_b, (n * k) * sizeof(float));
        cudaMallocManaged(&d_c, (m * k) * sizeof(float));

        cudaMemcpy(d_a, a, (m * n) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (n * k) * sizeof(float), cudaMemcpyHostToDevice);

        unsigned int grid_rows = sqrt(BLOCK_SIZE);
        unsigned int grid_cols = m / grid_rows;

        dim3 dimGrid(grid_cols, grid_cols, 1);
        dim3 dimBlock(grid_rows, grid_rows, 1);

        matMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
        cudaDeviceSynchronize();

        cudaMemcpy(c, d_c, (m * k) * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
}