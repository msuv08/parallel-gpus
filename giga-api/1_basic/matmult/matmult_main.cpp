#include <iostream>
#include <vector>
#include <iomanip>
#include <ctime>
#include "MegaGPU.h"

void initializeMatrix(std::vector<float>& matrix, int rows, int cols, float value) {
    matrix.resize(rows * cols, value);
}

void printMatrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void benchmark(int start_exp, int end_exp) {
    MegaGPU mega;
    for (int exp = start_exp; exp <= end_exp; ++exp) {
        int size = (1 << exp); // 2^exp
        std::vector<float> A, B, C;
        initializeMatrix(A, size, size, 3.0); // Initialize A with a constant value of 3.0
        initializeMatrix(B, size, size, 1.0); // Initialize B with a constant value of 1.0
        initializeMatrix(C, size, size, 0.0); // Initialize C with zeros

        // Create clock for timing
        // clock_t start = clock();

        // Perform matrix multiplication
        mega.performMatrixMultiplication(A.data(), B.data(), C.data(), size, size, size);

        // Calculate elapsed time
        // clock_t end = clock();
        // double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0; // convert to milliseconds

        // Output the matrix size and the elapsed time
        // std::cout << "Matrix size: " << size << "x" << size << ", Time: " << elapsed_time << " ms" << std::endl;

        // Optionally print the first 10x10 elements of matrix C
        // if (size <= 10) {
        //     std::cout << "First 10x10 elements of matrix C are:" << std::endl;
        //     printMatrix(C, size, size);
        // }
    }
}

int main() {
    benchmark(1, 15);  // Start at 2^1 (2) and end at 2^15 (32768)
    return 0;
}
