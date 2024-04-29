#include <iostream>
#include <vector>
#include <iomanip>
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

int main() {
    int A_rows = 1024, A_cols = 1024, B_cols = 1024;
    std::vector<float> A, B, C;
    initializeMatrix(A, A_rows, A_cols, 3.0);
    initializeMatrix(B, A_cols, B_cols, 1.0);
    initializeMatrix(C, A_rows, B_cols, 0.0);
    MegaGPU mega;
    std::cout << "Performing matrix multiplication..." << std::endl;
    mega.performMatrixMultiplication(A.data(), B.data(), C.data(), A_rows, A_cols, B_cols);
    std::cout << "Matrix multiplication completed. First 10x10 elements of C are:" << std::endl;
    printMatrix(C, 10, 10);
    return 0;
}
