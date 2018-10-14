#include <iostream>
#include <math.h>
#include "../ops/utils/devices.cu"
#include "../ops/utils/utils.cpp"

// execute with
// nvcc -o out matmul.cu -run --gpu-architecture=compute_61 --gpu-code=sm_61,compute_61

#define BLOCK_SIZE 256

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

    printf("%d-%d\n", ROW, COL);

    float sum = 0;

    // each thread computes one element of the block sub-matrix
    if (ROW < m && COL < k) {
        for (int i = 0; i < n; i++)
            sum += A[ROW * n + i] * B[i * n + COL];
        C[ROW * k + COL] = sum;
    }
}

int main() {
    getCudaDeviceInfo();

    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 16;
    int SIZE = N * N;

    // Allocate memory on the host
    float *A, *B, *C;

    cudaMallocManaged(&A, SIZE * sizeof(float));
    cudaMallocManaged(&B, SIZE * sizeof(float));
    cudaMallocManaged(&C, SIZE * sizeof(float));

    // Initialize matrices on the host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 2.5f;
            B[i * N + j] = 2.0f;
        }
    }

    unsigned int grid_rows = sqrt(BLOCK_SIZE);
    unsigned int grid_cols = N / grid_rows;

    dim3 dimGrid(grid_cols, grid_cols, 1);
    dim3 dimBlock(grid_rows, grid_rows, 1);

    matMul<<<dimGrid, dimBlock>>>(A, B, C, N, N, N);
    cudaDeviceSynchronize();

    // check for errors (all vals should be 80.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            maxError = fmax(maxError, fabs(C[i * N + j] - 80.0f));
    }
    std::cout << "Max Error: " << maxError << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}