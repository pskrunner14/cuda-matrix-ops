#include <iostream>
#include <math.h>
#include "devices.cu"

#define BLOCK_SIZE 16

void printMatrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << matrix[i * N + j] << " ";
        std::cout << std::endl;
    }
}

__global__ void matMul(float* A, float* B, float* C, int m, int n, int k) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    if (ROW < m && COL < k) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < n; i++) {
            sum += A[ROW * n + i] * B[i * n + COL];
        }
        C[ROW * k + COL] = sum;
    }
}

int main() {
    cout << "GPU Device Info:" << endl;
    getCudaDeviceInfo();
    cout << endl;

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
            A[i * N + j] = 2.0f;
            B[i * N + j] = 3.0f;
        }
    }

    unsigned int grid_rows = sqrt(BLOCK_SIZE);
    unsigned int grid_cols = N / grid_rows;

    dim3 dimGrid(grid_cols, grid_cols, 1);
    dim3 dimBlock(grid_rows, grid_rows, 1);

    matMul<<<dimGrid, dimBlock>>>(A, B, C, N, N, N);
    cudaDeviceSynchronize();

    // check for errors (all vals should be 96.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            maxError = fmax(maxError, fabs(C[i * N + j] - 96.0f));
    }
    std::cout << "Max Error: " << maxError << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}