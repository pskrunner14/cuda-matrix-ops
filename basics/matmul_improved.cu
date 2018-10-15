#include <iostream>
#include <stdio.h>

#define NUM_THREADS 32

__global__ void matrixMulGPU(int * a, int * b, int * c, int m, int n, int k) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int stride_row = gridDim.y * blockDim.y;
    int stride_col = gridDim.x * blockDim.x;
    
    for (; row < m && col < k; row += stride_row, col += stride_col) {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

void matrixMulCPU(int * a, int * b, int * c, int m, int n, int k) {
    int val = 0;

    for(int row = 0; row < m; ++row) {
        for(int col = 0; col < k; ++col) {
            val = 0;
            for (int i = 0; i < n; ++i)
                val += a[row * n + i] * b[i * k + col];
            c[row * k + col] = val;
        }
    }
}

int main() {
    int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

    int M, N, K;
    std::cout << "Please enter M, N and K: " << std::endl;
    std::cin >> M >> N >> K;
    // int M = 2340, N = 5674, K = 9758;

    int x = 456;
    int y = 567;

    // Allocate memory
    cudaMallocManaged (&a, M * N * sizeof (int));
    cudaMallocManaged (&b, N * K * sizeof (int));
    cudaMallocManaged (&c_cpu, M * K * sizeof (int));
    cudaMallocManaged (&c_gpu, M * K * sizeof (int));

    // Initialize memory; create 2D matrices
    for( int row = 0; row < M; ++row ) {
        for( int col = 0; col < N; ++col ) {
            a[row * N + col] = x;
        }
        for( int col = 0; col < K; ++col ) {
            c_cpu[row * K + col] = 0;
            c_gpu[row * K + col] = 0;
        }
    }
    for( int row = 0; row < N; ++row ) {
        for( int col = 0; col < K; ++col ) {
            b[row * K + col] = y;
        }
    }

    /*
    * Assign `threads_per_block` and `number_of_blocks` 2D values
    * that can be used in matrixMulGPU above.
    */
    dim3 threads_per_block(NUM_THREADS, NUM_THREADS, 1);
    dim3 number_of_blocks((K / threads_per_block.x) + 1, (M / threads_per_block.y) + 1, 1);

    matrixMulGPU<<<number_of_blocks, threads_per_block>>>(a, b, c_gpu, M, N, K);
    cudaDeviceSynchronize();

    // Call the CPU version to check our work
    // matrixMulCPU(a, b, c_cpu, M, N, K);

    // Compare the two answers to make sure they are equal
    bool error = false;
    for( int row = 0; row < M && !error; row++) {
        for( int col = 0; col < K && !error; col++) {
            if (c_gpu[row * K + col] != ((x * y) * N)) {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
        }
    }
    if (!error)
        printf("Success!\n");

    // Free all our allocated memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);
}
