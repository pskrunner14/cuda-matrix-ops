#include <iostream>
#include <math.h>
#include "devices.cu"

__global__
void matMul(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
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

    int blockSize = 256;
    int numBlocks = (SIZE + blockSize - 1) / blockSize;
    matMul<<<numBlocks, blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << C[i * N + j];
        cout << endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}