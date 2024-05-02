#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// Function to fill a vector with random values
void fill_random(std::vector<float>& v) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    for (auto& element : v) {
        element = dis(gen);
    }
}

int main() {
    const size_t maxSize = 100000000;  // Maximum size
    float *d_x, *d_y;
    float result;
    cublasHandle_t handle;
    cudaEvent_t startDot, stopDot, startNorm, stopNorm;

    // Initialize cuBLAS
    cublasCreate(&handle);

    // Create CUDA events for dot product
    cudaEventCreate(&startDot);
    cudaEventCreate(&stopDot);

    // Create CUDA events for L2 norm
    cudaEventCreate(&startNorm);
    cudaEventCreate(&stopNorm);

    // Allocate memory for the maximum size to avoid repeated allocations
    cudaMalloc(&d_x, maxSize * sizeof(float));
    cudaMalloc(&d_y, maxSize * sizeof(float));

    // Loop over exponentially increasing vector sizes
    for (size_t size = 2; size <= maxSize; size *= 2) {
        std::vector<float> h_x(size), h_y(size);

        // Fill vectors with random data
        fill_random(h_x);
        fill_random(h_y);

        // Copy data to device
        cudaMemcpy(d_x, h_x.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        // Record the start time for dot product
        cudaEventRecord(startDot);

        // Compute dot product
        cublasSdot(handle, size, d_x, 1, d_y, 1, &result);
        std::cout << "Vector Size: " << size << " - Dot Product: " << result << std::endl;

        // Record the stop time for dot product
        cudaEventRecord(stopDot);
        cudaEventSynchronize(stopDot);

        // Compute the elapsed time for dot product in milliseconds
        float millisecondsDot = 0;
        cudaEventElapsedTime(&millisecondsDot, startDot, stopDot);
        std::cout << "Dot Product Time: " << millisecondsDot << " ms" << std::endl;

        // Record the start time for L2 norm
        cudaEventRecord(startNorm);

        // Compute L2 norm for vector x
        cublasSnrm2(handle, size, d_x, 1, &result);
        std::cout << "Vector Size: " << size << " - L2 Norm of x: " << result << std::endl;

        // Record the stop time for L2 norm
        cudaEventRecord(stopNorm);
        cudaEventSynchronize(stopNorm);

        // Compute the elapsed time for L2 norm in milliseconds
        float millisecondsNorm = 0;
        cudaEventElapsedTime(&millisecondsNorm, startNorm, stopNorm);
        std::cout << "L2 Norm Time: " << millisecondsNorm << " ms\n\n";

    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(startDot);
    cudaEventDestroy(stopDot);
    cudaEventDestroy(startNorm);
    cudaEventDestroy(stopNorm);
    cublasDestroy(handle);

    return 0;
}
