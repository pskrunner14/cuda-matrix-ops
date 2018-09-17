#include <iostream>

void printMatrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << matrix[i * N + j] << " ";
        std::cout << std::endl;
    }
}