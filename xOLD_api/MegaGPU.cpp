#include "MegaGPU.h"
#include <iostream>
#include <cuda_runtime.h>

extern "C" void launchAddArrays(int *a, int *b, int *c, int n, int blocksPerGrid, int threadsPerBlock, cudaStream_t stream = 0);

MegaGPU::MegaGPU(int numElements) : n(numElements) {
    size = (n / 2) * sizeof(int);

    cudaSetDevice(0);
    cudaMalloc(&d_a0, size);
    cudaMalloc(&d_b0, size);
    cudaMalloc(&d_c0, size);
    std::cout << "Allocated memory on GPU 0" << std::endl;

    cudaSetDevice(1);
    cudaMalloc(&d_a1, size);
    cudaMalloc(&d_b1, size);
    cudaMalloc(&d_c1, size);
    std::cout << "Allocated memory on GPU 1" << std::endl;
}

MegaGPU::~MegaGPU() {
    cudaSetDevice(0);
    cudaFree(d_a0);
    cudaFree(d_b0);
    cudaFree(d_c0);
    std::cout << "Freed memory on GPU 0" << std::endl;

    cudaSetDevice(1);
    cudaFree(d_a1);
    cudaFree(d_b1);
    cudaFree(d_c1);
    std::cout << "Freed memory on GPU 1" << std::endl;
}

void MegaGPU::addArrays(const int *a, const int *b, int *c) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;

    cudaSetDevice(0);
    cudaMemcpy(d_a0, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b0, b, size, cudaMemcpyHostToDevice);
    launchAddArrays(d_a0, d_b0, d_c0, n / 2, blocksPerGrid, threadsPerBlock);
    cudaMemcpy(c, d_c0, size, cudaMemcpyDeviceToHost);
    std::cout << "GPU 0 - Computation and data transfer complete." << std::endl;

    cudaSetDevice(1);
    cudaMemcpy(d_a1, a + n / 2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b + n / 2, size, cudaMemcpyHostToDevice);
    launchAddArrays(d_a1, d_b1, d_c1, n / 2, blocksPerGrid, threadsPerBlock);
    cudaMemcpy(c + n / 2, d_c1, size, cudaMemcpyDeviceToHost);
    std::cout << "GPU 1 - Computation and data transfer complete." << std::endl;

    cudaDeviceSynchronize();
    std::cout << "Synchronization complete across all GPUs." << std::endl;
}
