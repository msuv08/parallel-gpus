#include "MegaGPU.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <thread>
#include <filesystem>
#include <string>

// externs are used to call the kernel code 
extern "C" void launchGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream);
extern "C" void performFFTKernel(float* input, cufftComplex* output, int width, int height, cudaStream_t stream);
extern "C" void launchUpsampleKernel(unsigned char* input, unsigned char* output, int width, int height, int scaleFactor, cudaStream_t stream);
extern "C" void launchSharpenKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream);
extern "C" void launchMatrixMulKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols, cudaStream_t stream);
extern "C" void launchVectorDotKernel(const float* a, const float* b, float* result, int n, cudaStream_t stream);
extern "C" void launchVectorL2NormKernel(const float* a, float* result, int n, cudaStream_t stream);

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

    cudaSetDevice(1);
    cudaMalloc(&d_input1, sizePerGPU);
    cudaMalloc(&d_output1, imageWidth * (imageHeight / 2));

    int halfHeight = imageHeight / 2;

    // Copy to GPU 0
    cudaSetDevice(0);
    cudaMemcpy(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice);
    launchGrayscaleKernel(d_input0, d_output0, imageWidth, halfHeight, 0);
    std::cout << "Launching GPU Kernel #0: " << std::endl;

    // Copy to GPU 1
    cudaSetDevice(1);
    cudaMemcpy(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice);
    launchGrayscaleKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, 0);
    std::cout << "Launching GPU Kernel #1: " << std::endl;

    // Copy results back
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output0, imageWidth * halfHeight, cudaMemcpyDeviceToHost);
    cudaMemcpy(output + imageWidth * halfHeight, d_output1, imageWidth * (imageHeight - halfHeight), cudaMemcpyDeviceToHost);

    // Free memory on GPUs
    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);

    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    std::cout << "End grayscale conversion..." << std::endl;
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

void MegaGPU::upsampleImage(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor) {
    imageWidth = width;
    imageHeight = height;
    sizePerGPU = imageWidth * (imageHeight / 2) * 3; // Each pixel RGB
    int outputWidth = imageWidth * scaleFactor;
    int outputHeight = imageHeight * scaleFactor;
    int outputSizePerGPU = outputWidth * (outputHeight / 2) * 3;
    std::cout << "Begin image upsampling..." << std::endl;

    // Allocate memory on GPUs
    cudaSetDevice(0);
    cudaMalloc(&d_input0, sizePerGPU);
    cudaMalloc(&d_output0, outputSizePerGPU);

    cudaSetDevice(1);
    cudaMalloc(&d_input1, sizePerGPU);
    cudaMalloc(&d_output1, outputSizePerGPU);

    int halfHeight = imageHeight / 2;

    cudaSetDevice(0);
    cudaMemcpy(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice);
    launchUpsampleKernel(d_input0, d_output0, imageWidth, halfHeight, scaleFactor, 0);
    std::cout << "Launching GPU Kernel #0: " << std::endl;

    // Copy to GPU 1
    cudaSetDevice(1);
    cudaMemcpy(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice);
    launchUpsampleKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, scaleFactor, 0);
    std::cout << "Launching GPU Kernel #1: " << std::endl;

    // Copy results back
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output0, outputSizePerGPU, cudaMemcpyDeviceToHost);
    cudaMemcpy(output + outputSizePerGPU, d_output1, outputSizePerGPU, cudaMemcpyDeviceToHost);

    // Free memory on GPUs
    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);

    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    std::cout << "End image upsampling..." << std::endl;
}

void MegaGPU::upsampleAllImages(const std::vector<std::string>& imagePaths, int scaleFactor) {
    // put half the image paths in the list on one GPU for processing
    std::vector<std::string> gpu0Images(imagePaths.begin(), imagePaths.begin() + imagePaths.size() / 2);

    // put the other half of the image paths on the other GPU for processing
    std::vector<std::string> gpu1Images(imagePaths.begin() + imagePaths.size() / 2, imagePaths.end());

    // create a thread for each GPU to process the images?? idk bro
    
}

// very similar to before! 
void MegaGPU::sharpenImage(const unsigned char* input, unsigned char* output, int width, int height) {
    std::cout << "Begin sharpening ..." << std::endl;
    imageWidth = width;
    imageHeight = height;
    sizePerGPU = imageWidth * (imageHeight / 2) * 3;


    cudaSetDevice(0);
    cudaMalloc(&d_input0, sizePerGPU);
    cudaMalloc(&d_output0, sizePerGPU);
    cudaSetDevice(1);
    cudaMalloc(&d_input1, sizePerGPU);
    cudaMalloc(&d_output1, sizePerGPU);

    int halfHeight = imageHeight / 2;


    cudaSetDevice(0);
    cudaMemcpy(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice);
    launchSharpenKernel(d_input0, d_output0, imageWidth, halfHeight, 0);
    std::cout << "Launching GPU Kernel #0: " << std::endl;


    cudaSetDevice(1);
    cudaMemcpy(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice);
    launchSharpenKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, 0);
    std::cout << "Launching GPU Kernel #1: " << std::endl;


    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output0, sizePerGPU, cudaMemcpyDeviceToHost);
    cudaMemcpy(output + sizePerGPU, d_output1, sizePerGPU, cudaMemcpyDeviceToHost);


    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);
    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);

    std::cout << "End sharpening ..." << std::endl;
}

void MegaGPU::performMatrixMultiplication(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int sizeA = A_rows * A_cols * sizeof(float);
    int sizeB = A_cols * B_cols * sizeof(float);
    int sizeC = A_rows * B_cols * sizeof(float);

    int halfRows = A_rows / 2;
    int remainingRows = A_rows - halfRows;

    int sizeA_first_half = halfRows * A_cols * sizeof(float);
    int sizeA_second_half = remainingRows * A_cols * sizeof(float);
    int sizeC_first_half = halfRows * B_cols * sizeof(float);
    int sizeC_second_half = remainingRows * B_cols * sizeof(float);
    
    // Allocate memory on GPUs
    cudaSetDevice(0);
    cudaMalloc(&d_inputA0, sizeA_first_half);
    cudaMalloc(&d_inputB0, sizeB); 
    cudaMalloc(&d_outputC0, sizeC_first_half);
    cudaMemcpy(d_inputA0, A, sizeA_first_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputB0, B, sizeB, cudaMemcpyHostToDevice);

    // Allocate memory on GPUs
    cudaSetDevice(1);
    cudaMalloc(&d_inputA1, sizeA_second_half);
    cudaMalloc(&d_inputB1, sizeB); 
    cudaMalloc(&d_outputC1, sizeC_second_half);
    cudaMemcpy(d_inputA1, A + halfRows * A_cols, sizeA_second_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputB1, B, sizeB, cudaMemcpyHostToDevice);

    cudaStream_t stream0, stream1;
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);

    std::cout << "Launching matrix multiplication on GPU 0..." << std::endl;
    cudaSetDevice(0);
    launchMatrixMulKernel(d_inputA0, d_inputB0, d_outputC0, halfRows, A_cols, B_cols, stream0);

    std::cout << "Launching matrix multiplication on GPU 1..." << std::endl;
    cudaSetDevice(1);
    launchMatrixMulKernel(d_inputA1, d_inputB1, d_outputC1, remainingRows, A_cols, B_cols, stream1);

    // Synchronize and copy back results
    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    cudaMemcpy(C, d_outputC0, sizeC_first_half, cudaMemcpyDeviceToHost);

    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);
    cudaMemcpy(C + halfRows * B_cols, d_outputC1, sizeC_second_half, cudaMemcpyDeviceToHost);

    // Free resources
    cudaSetDevice(0);
    cudaFree(d_inputA0);
    cudaFree(d_inputB0);
    cudaFree(d_outputC0);
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    cudaFree(d_inputA1);
    cudaFree(d_inputB1);
    cudaFree(d_outputC1);
    cudaStreamDestroy(stream1);
}

void MegaGPU::computeDotProduct(const float* a, const float* b, float& result, int n) {
    int sizePerGPU = n / 2;
    int remainder = n % 2;
    float partialResult[2] = {0.0f, 0.0f};

    cudaStream_t streams[2];
    float* d_vectorA[2];
    float* d_vectorB[2];
    float* d_scalarResult[2];

    // Handle GPU 0
    cudaSetDevice(0);
    cudaStreamCreate(&streams[0]);
    cudaMalloc((void**)&d_vectorA[0], (sizePerGPU + (remainder > 0 ? 1 : 0)) * sizeof(float));
    cudaMalloc((void**)&d_vectorB[0], (sizePerGPU + (remainder > 0 ? 1 : 0)) * sizeof(float));
    cudaMalloc((void**)&d_scalarResult[0], sizeof(float));

    cudaMemcpy(d_vectorA[0], a, (sizePerGPU + (remainder > 0 ? 1 : 0)) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectorB[0], b, (sizePerGPU + (remainder > 0 ? 1 : 0)) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_scalarResult[0], 0, sizeof(float));

    launchVectorDotKernel(d_vectorA[0], d_vectorB[0], d_scalarResult[0], sizePerGPU + (remainder > 0 ? 1 : 0), streams[0]);
    cudaMemcpyAsync(&partialResult[0], d_scalarResult[0], sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
    cudaStreamSynchronize(streams[0]);

    // Handle GPU 1
    cudaSetDevice(1);
    cudaStreamCreate(&streams[1]);
    cudaMalloc((void**)&d_vectorA[1], sizePerGPU * sizeof(float));
    cudaMalloc((void**)&d_vectorB[1], sizePerGPU * sizeof(float));
    cudaMalloc((void**)&d_scalarResult[1], sizeof(float));

    cudaMemcpy(d_vectorA[1], a + sizePerGPU + (remainder > 0 ? 1 : 0), sizePerGPU * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectorB[1], b + sizePerGPU + (remainder > 0 ? 1 : 0), sizePerGPU * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_scalarResult[1], 0, sizeof(float));

    launchVectorDotKernel(d_vectorA[1], d_vectorB[1], d_scalarResult[1], sizePerGPU, streams[1]);
    cudaMemcpyAsync(&partialResult[1], d_scalarResult[1], sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
    cudaStreamSynchronize(streams[1]);

    // Cleanup

    cudaSetDevice(0);
    cudaFree(d_vectorA[0]);
    cudaFree(d_vectorB[0]);
    cudaFree(d_scalarResult[0]);
    cudaStreamDestroy(streams[0]);

    cudaSetDevice(1);
    cudaFree(d_vectorA[1]);
    cudaFree(d_vectorB[1]);
    cudaFree(d_scalarResult[1]);
    cudaStreamDestroy(streams[1]);

    result = partialResult[0] + partialResult[1];
}

void MegaGPU::computeL2Norm(const float* a, float& result, int n) {
    int sizePerGPU = n / 2;
    int remainder = n % 2;
    float sumOfSquares[2] = {0.0f, 0.0f};

    cudaStream_t streams[2];
    float* d_vectorA[2];
    float* d_scalarResult[2];

    // GPU 0
    cudaSetDevice(0);
    cudaStreamCreate(&streams[0]);
    cudaMalloc((void**)&d_vectorA[0], (sizePerGPU + (remainder > 0 ? 1 : 0)) * sizeof(float));
    cudaMalloc((void**)&d_scalarResult[0], sizeof(float));

    cudaMemcpy(d_vectorA[0], a, (sizePerGPU + (remainder > 0 ? 1 : 0)) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_scalarResult[0], 0, sizeof(float));

    launchVectorL2NormKernel(d_vectorA[0], d_scalarResult[0], sizePerGPU + (remainder > 0 ? 1 : 0), streams[0]);
    cudaMemcpyAsync(&sumOfSquares[0], d_scalarResult[0], sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
    cudaStreamSynchronize(streams[0]);

    // GPU 1
    cudaSetDevice(1);
    cudaStreamCreate(&streams[1]);
    cudaMalloc((void**)&d_vectorA[1], sizePerGPU * sizeof(float));
    cudaMalloc((void**)&d_scalarResult[1], sizeof(float));

    cudaMemcpy(d_vectorA[1], a + sizePerGPU + (remainder > 0 ? 1 : 0), sizePerGPU * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_scalarResult[1], 0, sizeof(float));

    launchVectorL2NormKernel(d_vectorA[1], d_scalarResult[1], sizePerGPU, streams[1]);
    cudaMemcpyAsync(&sumOfSquares[1], d_scalarResult[1], sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
    cudaStreamSynchronize(streams[1]);

    // Cleanup
    cudaSetDevice(0);
    cudaFree(d_vectorA[0]);
    cudaFree(d_scalarResult[0]);
    cudaStreamDestroy(streams[0]);

    cudaSetDevice(1);
    cudaFree(d_vectorA[1]);
    cudaFree(d_scalarResult[1]);
    cudaStreamDestroy(streams[1]);

    result = sqrt(sumOfSquares[0] + sumOfSquares[1]);
}


// 
// void MegaGPU::antiAlias(const unsigned char* input, unsigned char* output, int width, int height) {
//     std::cout << "Begin sharpening ..." << std::endl;
//     imageWidth = width;
//     imageHeight = height;
//     sizePerGPU = imageWidth * (imageHeight / 2) * 3;


//     cudaSetDevice(0);
//     cudaMalloc(&d_input0, sizePerGPU);
//     cudaMalloc(&d_output0, sizePerGPU);
//     cudaSetDevice(1);
//     cudaMalloc(&d_input1, sizePerGPU);
//     cudaMalloc(&d_output1, sizePerGPU);

//     int halfHeight = imageHeight / 2;


//     cudaSetDevice(0);
//     cudaMemcpy(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice);
//     launchAntiAliasKernel(d_input0, d_output0, imageWidth, halfHeight, 0);
//     std::cout << "Launching GPU Kernel #0: " << std::endl;


//     cudaSetDevice(1);
//     cudaMemcpy(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice);
//     launchAntiAliasKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, 0);
//     std::cout << "Launching GPU Kernel #1: " << std::endl;


//     cudaDeviceSynchronize();
//     cudaMemcpy(output, d_output0, sizePerGPU, cudaMemcpyDeviceToHost);
//     cudaMemcpy(output + sizePerGPU, d_output1, sizePerGPU, cudaMemcpyDeviceToHost);


//     cudaSetDevice(0);
//     cudaFree(d_input0);
//     cudaFree(d_output0);
//     cudaSetDevice(1);
//     cudaFree(d_input1);
//     cudaFree(d_output1);

//     std::cout << "End sharpening ..." << std::endl;
// }