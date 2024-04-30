#include "MegaGPU.h"
#include <iostream>
#include <vector>
#include <random>

// Function to generate a large vector with random values
std::vector<float> generate_large_vector(size_t size) {
    std::vector<float> v(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    for (size_t i = 0; i < size; ++i) {
        v[i] = dis(gen);
    }
    return v;
}


int main() {
    MegaGPU mega;

    // vectors for dot and cross product
    const size_t vector_size = 1000000;  // Increase the vector size to 1 million elements

    std::vector<float> a = generate_large_vector(vector_size);
    std::vector<float> b = generate_large_vector(vector_size);

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
