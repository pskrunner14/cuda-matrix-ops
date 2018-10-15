#include <iostream>
#include <math.h>
#include "../ops/utils/devices.cu"
#include "../ops/utils/utils.cpp"

// compile and execute with
// nvcc -o out matsum.cu -run --gpu-architecture=compute_61 --gpu-code=sm_61,compute_61

#define BLOCK_SIZE 16

__global__ void matSum(float* A, float* B, float* C, int m, int n) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread computes one element of the block sub-matrix
    if (ROW < m && COL < n) {
        C[ROW * n + COL] = A[ROW * n + COL] + B[ROW * n + COL];
    }
}

int main() {
    getCudaDeviceInfo();

    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int M = 20000;
    int N = 16327;

    // Allocate memory on the host
    float *A, *B, *C;

    cudaMallocManaged(&A, (M * N) * sizeof(float));
    cudaMallocManaged(&B, (M * N) * sizeof(float));
    cudaMallocManaged(&C, (M * N) * sizeof(float));

    float a = 343.5534f;
    float b = 7254.1543f;

    // Initialize matrices on the host
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = a;
            B[i * N + j] = b;
        }
    }

    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matSum<<<dimGrid, dimBlock>>>(A, B, C, M, N);
    cudaDeviceSynchronize();

    // printMatrix(C, M, N);

    // check for errors (all vals should be 10.6f)
    float maxError = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            maxError = fmax(maxError, fabs(C[i * N + j] - (a + b)));
    }
    std::cout << "Max Error: " << maxError << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}