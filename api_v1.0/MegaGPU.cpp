#include "MegaGPU.h"
#include <iostream>

extern "C" void launchGrayscaleKernel(unsigned char *input, unsigned char *output, int width, int height, cudaStream_t stream);

MegaGPU::MegaGPU() {
    imageWidth = imageHeight = sizePerGPU = 0;
    // Initialize GPUs
    cudaSetDevice(0);
    cudaMalloc(&d_input0, sizePerGPU);
    cudaMalloc(&d_output0, sizePerGPU);
    std::cout << "GPU 0 memory allocated." << std::endl;

    cudaSetDevice(1);
    cudaMalloc(&d_input1, sizePerGPU);
    cudaMalloc(&d_output1, sizePerGPU);
    std::cout << "GPU 1 memory allocated." << std::endl;
}

MegaGPU::~MegaGPU() {
    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);
    std::cout << "GPU 0 memory freed." << std::endl;

    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    std::cout << "GPU 1 memory freed." << std::endl;
}

void MegaGPU::convertToGrayscale(const unsigned char *input, unsigned char *output, int width, int height) {
    imageWidth = width;
    imageHeight = height;
    sizePerGPU = (imageWidth * imageHeight / 2) * 3; // Each pixel RGB

    int halfHeight = imageHeight / 2;

    // Debugging output to verify the input data before transfer to GPU
    std::cout << "Sample Input Data Before Transfer to GPU 0 (first 10 bytes): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(input[i]) << " ";
    }
    std::cout << std::endl;

    // Copy to GPU 0
    cudaSetDevice(0);
    cudaMemcpy(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice);
    std::cout << "Data transferred to GPU 0." << std::endl;
    launchGrayscaleKernel(d_input0, d_output0, imageWidth, halfHeight, 0);
    std::cout << "Grayscale conversion started on GPU 0." << std::endl;

    // Debugging output to verify the input data before transfer to GPU 1
    std::cout << "Sample Input Data Before Transfer to GPU 1 (first 10 bytes of second half): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(input[sizePerGPU + i]) << " ";
    }
    std::cout << std::endl;

    // Copy to GPU 1
    cudaSetDevice(1);
    cudaMemcpy(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice);
    std::cout << "Data transferred to GPU 1." << std::endl;
    launchGrayscaleKernel(d_input1, d_output1, imageWidth, halfHeight, 0);
    std::cout << "Grayscale conversion started on GPU 1." << std::endl;

    // Copy results back
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output0, sizePerGPU / 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(output + sizePerGPU / 3, d_output1, sizePerGPU / 3, cudaMemcpyDeviceToHost);
    std::cout << "Data copied back to host. Synchronization complete." << std::endl;
}

