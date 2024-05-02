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

    // Start with a vector size of 2 and increase exponentially
    for (size_t vector_size = 2; vector_size <= 100000000; vector_size *= 2) {
        std::vector<float> a = generate_large_vector(vector_size);
        std::vector<float> b = generate_large_vector(vector_size);

        // Containers for results
        float dotProductResult = 0.0f;
        float l2NormResultA = 0.0f;
        float l2NormResultB = 0.0f;

        // Compute the dot product
        mega.computeDotProduct(a.data(), b.data(), dotProductResult, a.size());
        std::cout << "Vector Size: " << vector_size << " - Dot Product Result: " << dotProductResult << std::endl;
        std::cout << std::endl;

        // Compute the L2 norm of vector a
        mega.computeL2Norm(a.data(), l2NormResultA, a.size());
        std::cout << "Vector Size: " << vector_size << " - L2 Norm of Vector A: " << l2NormResultA << std::endl;
        std::cout << std::endl;

        // Compute the L2 norm of vector b
        // mega.computeL2Norm(b.data(), l2NormResultB, b.size());
        // std::cout << "Vector Size: " << vector_size << " - L2 Norm of Vector B: " << l2NormResultB << std::endl;
        // std::cout << std::endl;
    }

    return 0;
}
