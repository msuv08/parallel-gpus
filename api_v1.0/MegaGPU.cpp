#include "MegaGPU.h"
#include <iostream>
#include <cmath>
#include <fstream>

extern "C" void launchGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream);
extern "C" void performFFTKernel(float* input, cufftComplex* output, int width, int height, cudaStream_t stream);
extern "C" void launchMatrixMulKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols, cudaStream_t stream);

MegaGPU::MegaGPU() {
    d_input0 = d_output0 = nullptr;
    d_input1 = d_output1 = nullptr;
    imageWidth = imageHeight = sizePerGPU = 0;
    d_inputA0 = d_inputB0 = d_outputC0 = nullptr;
    d_inputA1 = d_inputB1 = d_outputC1 = nullptr;
}

MegaGPU::~MegaGPU() {
    if (d_input0) cudaFree(d_input0);
    if (d_output0) cudaFree(d_output0);
    if (d_input1) cudaFree(d_input1);
    if (d_output1) cudaFree(d_output1);
    if (d_inputA0) cudaFree(d_inputA0);
    if (d_inputB0) cudaFree(d_inputB0);
    if (d_outputC0) cudaFree(d_outputC0);
    if (d_inputA1) cudaFree(d_inputA1);
    if (d_inputB1) cudaFree(d_inputB1);
    if (d_outputC1) cudaFree(d_outputC1);
}

void MegaGPU::convertToGrayscale(const unsigned char* input, unsigned char* output, int width, int height) {
    imageWidth = width;
    imageHeight = height;
    sizePerGPU = imageWidth * (imageHeight / 2) * 3; // Each pixel RGB

    // Allocate memory on GPUs
    cudaSetDevice(0);
    cudaMalloc(&d_input0, sizePerGPU);
    cudaMalloc(&d_output0, imageWidth * (imageHeight / 2));
    std::cout << "GPU 0 memory allocated." << std::endl;

    cudaSetDevice(1);
    cudaMalloc(&d_input1, sizePerGPU);
    cudaMalloc(&d_output1, imageWidth * (imageHeight / 2));
    std::cout << "GPU 1 memory allocated." << std::endl;

    int halfHeight = imageHeight / 2;

    // Copy to GPU 0
    cudaSetDevice(0);
    cudaMemcpy(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice);
    std::cout << "Data transferred to GPU 0." << std::endl;

    launchGrayscaleKernel(d_input0, d_output0, imageWidth, halfHeight, 0);
    std::cout << "Grayscale conversion started on GPU 0." << std::endl;

    // Copy to GPU 1
    cudaSetDevice(1);
    cudaMemcpy(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice);
    std::cout << "Data transferred to GPU 1." << std::endl;

    launchGrayscaleKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, 0);
    std::cout << "Grayscale conversion started on GPU 1." << std::endl;

    // Copy results back
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output0, imageWidth * halfHeight, cudaMemcpyDeviceToHost);
    cudaMemcpy(output + imageWidth * halfHeight, d_output1, imageWidth * (imageHeight - halfHeight), cudaMemcpyDeviceToHost);
    std::cout << "Data copied back to host. Synchronization complete." << std::endl;

    // Free memory on GPUs
    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);
    std::cout << "GPU 0 memory freed." << std::endl;

    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    std::cout << "GPU 1 memory freed." << std::endl;
}

void MegaGPU::prepareData(float* input, int size) {
    std::ifstream inputFile("input_signal.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file." << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        if (!(inputFile >> input[i])) {
            std::cerr << "Insufficient samples in the input file." << std::endl;
            break;
        }
    }

    inputFile.close();
}

void MegaGPU::performFFT(float* input, cufftComplex* output, int width, int height) {
    float* d_input;
    cufftComplex* d_output;
    cudaMalloc((void**)&d_input, width * sizeof(float));
    cudaMalloc((void**)&d_output, (width / 2 + 1) * height * sizeof(cufftComplex));
    cudaMemcpy(d_input, input, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // TODO: potentially do time based operations here on kernel call. - omeed
    performFFTKernel(d_input, d_output, width, height, stream);
    
    cudaMemcpy(output, d_output, (width / 2 + 1) * height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

void MegaGPU::performMatrixMultiplication(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int sizeA = A_rows * A_cols * sizeof(float);
    int sizeB = A_cols * B_cols * sizeof(float);
    int sizeC = A_rows * B_cols * sizeof(float);
    int halfRows = A_rows / 2; // Rows handled by the first GPU
    int remainingRows = A_rows - halfRows; // Rows handled by the second GPU (important if A_rows is odd)

    // Adjust size calculations for precise memory allocations
    int sizeA_first_half = halfRows * A_cols * sizeof(float);
    int sizeA_second_half = remainingRows * A_cols * sizeof(float);
    int sizeC_first_half = halfRows * B_cols * sizeof(float);
    int sizeC_second_half = remainingRows * B_cols * sizeof(float);

    cudaSetDevice(0);
    cudaMalloc(&d_inputA0, sizeA_first_half);
    cudaMalloc(&d_inputB0, sizeB); // Full B
    cudaMalloc(&d_outputC0, sizeC_first_half);
    cudaMemcpy(d_inputA0, A, sizeA_first_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputB0, B, sizeB, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMalloc(&d_inputA1, sizeA_second_half);
    cudaMalloc(&d_inputB1, sizeB); // Full B
    cudaMalloc(&d_outputC1, sizeC_second_half);
    cudaMemcpy(d_inputA1, A + halfRows * A_cols, sizeA_second_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputB1, B, sizeB, cudaMemcpyHostToDevice);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Launch kernels using streams
    std::cout << "Launching matrix multiplication on GPU 0..." << std::endl;
    launchMatrixMulKernel(d_inputA0, d_inputB0, d_outputC0, halfRows, A_cols, B_cols, stream0);
    std::cout << "Launching matrix multiplication on GPU 1..." << std::endl;
    launchMatrixMulKernel(d_inputA1, d_inputB1, d_outputC1, remainingRows, A_cols, B_cols, stream1);

    // Synchronize and copy back results
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_outputC0, sizeC_first_half, cudaMemcpyDeviceToHost);
    cudaMemcpy(C + halfRows * B_cols, d_outputC1, sizeC_second_half, cudaMemcpyDeviceToHost);

    // Free resources
    cudaFree(d_inputA0);
    cudaFree(d_inputB0);
    cudaFree(d_outputC0);
    cudaFree(d_inputA1);
    cudaFree(d_inputB1);
    cudaFree(d_outputC1);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}