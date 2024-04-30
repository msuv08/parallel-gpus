#include "MegaGPU.h"
#include <iostream>
#include <vector>

int main() {
    MegaGPU mega;

    // vectors for dot and cross product
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f};

    // making containers for results
    float dotProductResult = 0.0f;
    std::vector<float> crossProductResult(3, 0.0f);
    float l2NormResultA = 0.0f;
    float l2NormResultB = 0.0f;

    // Compute the dot product
    mega.computeDotProduct(a.data(), b.data(), dotProductResult, a.size());
    std::cout << "Dot Product Result: " << dotProductResult << std::endl;

    // Compute the L2 norm of vector a
    mega.computeL2Norm(a.data(), l2NormResultA, a.size());
    std::cout << "L2 Norm of Vector A: " << l2NormResultA << std::endl;

    // Compute the L2 norm of vector b
    mega.computeL2Norm(b.data(), l2NormResultB, b.size());
    std::cout << "L2 Norm of Vector B: " << l2NormResultB << std::endl;
    return 0;
}
