#include <iostream>
#include <math.h>
#include "../ops/utils/devices.cu"
#include "../ops/utils/utils.cpp"

// execute with
// nvcc -o out matmul.cu -run --gpu-architecture=compute_61 --gpu-code=sm_61,compute_61

#define BLOCK_SIZE 16

/**
Inside a CUDA kernel:

gridDim.x - number of blocks in the grid
blockIdx.x - index of current block within the grid
blockDim.x - number of threads in the block
threadIdx.x - index of current thread inside the block

note: 
    Threads are grouped into thread blocks, 
    blocks are grouped into a grid, which is 
    the highest entity in the CUDA thread hierarchy. 
    In summary, CUDA kernels are executed in a grid 
    of 1 or more blocks, with each block containing 
    the same number of 1 or more threads.
*/

__global__ void matMul(float* A, float* B, float* C, int m, int n, int k) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread computes one element of the block sub-matrix
    if (ROW < m && COL < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += A[ROW * n + i] * B[i * k + COL];
        C[ROW * k + COL] = sum;
    }
}

int main() {
    getCudaDeviceInfo();

    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int M = 10;
    int N = 16;
    int K = 20;

    // Allocate memory on the host
    float *A, *B, *C;

    cudaMallocManaged(&A, (M * N) * sizeof(float));
    cudaMallocManaged(&B, (N * K) * sizeof(float));
    cudaMallocManaged(&C, (M * K) * sizeof(float));

    // Initialize matrices on the host
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 3.0f;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B[i * K + j] = 2.0f;
        }
    }

    dim3 dimGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matMul<<<dimGrid, dimBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();

    printMatrix(C, M, K);

    // check for errors (all vals should be 96.0f)
    float maxError = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++)
            maxError = fmax(maxError, fabs(C[i * K + j] - 96.0f));
    }
    std::cout << "Max Error: " << maxError << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}