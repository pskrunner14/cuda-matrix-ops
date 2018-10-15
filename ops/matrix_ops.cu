/**
 *  CUDA PARALLEL PROGRAMMING: matrix_ops.cu
 *  Purpose: Matrix Operations using CUDA C/C++
 *  @author Prabhsimran Singh
 *  @version 2.3 16/10/18
 *
 *  Build using: nvcc -Xcompiler -fPIC -shared -o lib/cuda_mat_ops.so ops/matrix_ops.cu 
 *              (optional) --gpu-architecture=compute_61 --gpu-code=sm_61,compute_61
 */
#include <iostream>
#include "utils/devices.cu"
#include "utils/utils.cpp"

#define NUM_THREADS 32

/**
* Computes dot-product of two matrices (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double pointer to second input array
* @param c the double pointer to output array
* @param m the no. rows in a(m x n) and c(m x k)
* @param n the no. cols in a(m x n) and rows in b(n x k)
* @param k the no. cols in b(n x k) and c(m x k)
* @return void
*/
__global__ void matmul(double *a, double *b, double *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // strides are unnecessary if you're not using variable sized blocks
    // however I'm going to leave it here since the focus of this is not 
    // optimality or performance but readability, understandability and 
    // overall coherence.
    int stride_row = gridDim.y * blockDim.y;
    int stride_col = gridDim.x * blockDim.x;
    
    for (; row < m && col < k; row += stride_row, col += stride_col) {
        double sum = 0;
        #pragma unroll // unrolls the for loop (optimization - cuts on exec time)
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

/**
* Calculates element-wise sum of two matrices (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double pointer to second input array
* @param c the double pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void matsum(double *a, double *b, double *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_row = gridDim.y * blockDim.y;
    int stride_col = gridDim.x * blockDim.x;
    
    for (; row < m && col < n; row += stride_row, col += stride_col) {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}

/**
* Calculates element-wise product of two matrices (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double pointer to second input array
* @param c the double pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void matprod(double *a, double *b, double *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_row = gridDim.y * blockDim.y;
    int stride_col = gridDim.x * blockDim.x;
    
    for (; row < m && col < n; row += stride_row, col += stride_col) {
        c[row * n + col] = a[row * n + col] * b[row * n + col];
    }
}

/**
* Adds a value element-wise to the matrix (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double value to add to the array
* @param c the double pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void sum(double *a, double b, double *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_row = gridDim.y * blockDim.y;
    int stride_col = gridDim.x * blockDim.x;
    
    for (; row < m && col < n; row += stride_row, col += stride_col) {
        c[row * n + col] = a[row * n + col] + b;
    }
}

/**
* Multiplies a value element-wise to matrix (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double value to multiply the array with
* @param c the double pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void prod(double *a, double b, double *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_row = gridDim.y * blockDim.y;
    int stride_col = gridDim.x * blockDim.x;
    
    for (; row < m && col < n; row += stride_row, col += stride_col) {
        c[row * n + col] = a[row * n + col] * b;
    }
}

/**
* Computes the element-wise maximum ofs a matrix and a value (using parallel threads on CUDA capable device)
*
* @param a the double pointer to first input array
* @param b the double value to check maximum against
* @param c the double pointer to output array
* @param m the no. of rows in the arrays
* @param n the no. of cols in the arrays
* @return void
*/
__global__ void maximum(double *a, double b, double *c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_row = gridDim.y * blockDim.y;
    int stride_col = gridDim.x * blockDim.x;
    
    for (; row < m && col < n; row += stride_row, col += stride_col) {
        c[row * n + col] = (a[row * n + col] > b) ? a[row * n + col] : b;
    }
}

extern "C" {

    void cuda_device_info() {
        getCudaDeviceInfo();
    }

    void cuda_matmul(double *a, double *b, double *c, int m, int n, int k) {
        double *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_b, (n * k) * sizeof(double));
        cudaMallocManaged(&d_c, (m * k) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (n * k) * sizeof(double), cudaMemcpyHostToDevice);

        dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
        dim3 dimGrid((k / dimBlock.x) + 1, (m / dimBlock.y) + 1, 1);

        cudaError_t syncErr, asyncErr;
        matmul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);

        syncErr = cudaGetLastError();
        asyncErr = cudaDeviceSynchronize();

        if (syncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(syncErr) << endl;
        if (asyncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(asyncErr) << endl;
    
        cudaMemcpy(c, d_c, (m * k) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void cuda_matsum(double *a, double *b, double *c, int m, int n) {
        double *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_b, (m * n) * sizeof(double));
        cudaMallocManaged(&d_c, (m * n) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (m * n) * sizeof(double), cudaMemcpyHostToDevice);

        dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
        dim3 dimGrid((n / dimBlock.x) + 1, (m / dimBlock.y) + 1, 1);

        cudaError_t syncErr, asyncErr;
        matsum<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n);
        
        syncErr = cudaGetLastError();
        asyncErr = cudaDeviceSynchronize();

        if (syncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(syncErr) << endl;
        if (asyncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(asyncErr) << endl;

        cudaMemcpy(c, d_c, (m * n) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void cuda_matprod(double *a, double *b, double *c, int m, int n) {
        double *d_a, *d_b, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_b, (m * n) * sizeof(double));
        cudaMallocManaged(&d_c, (m * n) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, (m * n) * sizeof(double), cudaMemcpyHostToDevice);

        dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
        dim3 dimGrid((n / dimBlock.x) + 1, (m / dimBlock.y) + 1, 1);

        cudaError_t syncErr, asyncErr;
        matprod<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n);
        
        syncErr = cudaGetLastError();
        asyncErr = cudaDeviceSynchronize();

        if (syncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(syncErr) << endl;
        if (asyncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(asyncErr) << endl;

        cudaMemcpy(c, d_c, (m * n) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void cuda_sum(double *a, double b, double *c, int m, int n) {
        double *d_a, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_c, (m * n) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);

        dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
        dim3 dimGrid((n / dimBlock.x) + 1, (m / dimBlock.y) + 1, 1);

        cudaError_t syncErr, asyncErr;
        sum<<<dimGrid, dimBlock>>>(d_a, b, d_c, m, n);
        
        syncErr = cudaGetLastError();
        asyncErr = cudaDeviceSynchronize();

        if (syncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(syncErr) << endl;
        if (asyncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(asyncErr) << endl;

        cudaMemcpy(c, d_c, (m * n) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_c);
    }

    void cuda_prod(double *a, double b, double *c, int m, int n) {
        double *d_a, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_c, (m * n) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);

        dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
        dim3 dimGrid((n / dimBlock.x) + 1, (m / dimBlock.y) + 1, 1);

        cudaError_t syncErr, asyncErr;
        prod<<<dimGrid, dimBlock>>>(d_a, b, d_c, m, n);
        
        syncErr = cudaGetLastError();
        asyncErr = cudaDeviceSynchronize();

        if (syncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(syncErr) << endl;
        if (asyncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(asyncErr) << endl;

        cudaMemcpy(c, d_c, (m * n) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_c);
    }

    void cuda_maximum(double *a, double b, double *c, int m, int n) {
        double *d_a, *d_c;

        cudaMallocManaged(&d_a, (m * n) * sizeof(double));
        cudaMallocManaged(&d_c, (m * n) * sizeof(double));

        cudaMemcpy(d_a, a, (m * n) * sizeof(double), cudaMemcpyHostToDevice);

        dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
        dim3 dimGrid((n / dimBlock.x) + 1, (m / dimBlock.y) + 1, 1);

        cudaError_t syncErr, asyncErr;
        maximum<<<dimGrid, dimBlock>>>(d_a, b, d_c, m, n);
        
        syncErr = cudaGetLastError();
        asyncErr = cudaDeviceSynchronize();

        if (syncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(syncErr) << endl;
        if (asyncErr != cudaSuccess) 
            cout << "CUDA Error: " << cudaGetErrorString(asyncErr) << endl;

        cudaMemcpy(c, d_c, (m * n) * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_c);
    }
}