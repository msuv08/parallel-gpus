#include <iostream>
#include <vector>
#include "MegaGPU.h"

int main() {
    int A_rows = 1024, A_cols = 1024, B_cols = 1024;
    std::vector<float> A(A_rows * A_cols), B(A_cols * B_cols), C(A_rows * B_cols);

    // Initialize matrices A and B
    for (int i = 0; i < A_rows * A_cols; ++i) {
        A[i] = 1.0f; // Example initialization
    }
    for (int i = 0; i < A_cols * B_cols; ++i) {
        B[i] = 2.0f; // Example initialization
    }

    MegaGPU mega;
    std::cout << "Performing matrix multiplication..." << std::endl;
    mega.performMatrixMultiplication(A.data(), B.data(), C.data(), A_rows, A_cols, B_cols);

    std::cout << "Matrix multiplication completed." << std::endl;
    // Optionally, print some elements of matrix C
    return 0;
}