#include "MegaGPU.h"
#include <iostream>
#include <vector>

int main() {
    MegaGPU mega;

    // Example vectors for dot and cross product
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    // making containers for results
    float dotProductResult = 0.0f;
    std::vector<float> crossProductResult(3, 0.0f);
    float l2NormResult = 0.0f;

    // Compute the dot product
    mega.computeDotProduct(a.data(), b.data(), dotProductResult, a.size());
    std::cout << "Dot Product Result: " << dotProductResult << std::endl;

    // Compute the cross product
    mega.computeCrossProduct(a.data(), b.data(), crossProductResult.data());
    std::cout << "Cross Product Result: ["
              << crossProductResult[0] << ", "
              << crossProductResult[1] << ", "
              << crossProductResult[2] << "]" << std::endl;

    // Compute the L2 norm of vector a
    mega.computeL2Norm(a.data(), l2NormResult, a.size());
    std::cout << "L2 Norm of Vector A: " << l2NormResult << std::endl;

    return 0;
}
