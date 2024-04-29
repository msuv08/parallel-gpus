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

extern "C" void launchGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream);
extern "C" void performFFTKernel(float* input, cufftComplex* output, int width, int height, cudaStream_t stream);
extern "C" void launchUpsampleKernel(unsigned char* input, unsigned char* output, int width, int height, int scaleFactor, cudaStream_t stream);

MegaGPU::MegaGPU() {
    d_input0 = d_output0 = nullptr;
    d_input1 = d_output1 = nullptr;
    imageWidth = imageHeight = sizePerGPU = 0;
}

MegaGPU::~MegaGPU() {
    if (d_input0) cudaFree(d_input0);
    if (d_output0) cudaFree(d_output0);
    if (d_input1) cudaFree(d_input1);
    if (d_output1) cudaFree(d_output1);
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

    // Copy to GPU 1
    cudaSetDevice(1);
    cudaMemcpy(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice);
    std::cout << "Launching GPU Kernel: " << std::endl;

    launchGrayscaleKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, 0);

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


// mega gpu follows a very similar pattern to the grayscale conversion, but with a few key differences.
// (1): The image is split into two parts, and each part is sent to a different GPU.
// (2): The output image is scaled by a factor of 2, so the output size is calculated accordingly.
// (3): The upsample kernel is launched on both GPUs, and the results are copied back to the host.
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

    // Copy to GPU 1
    cudaSetDevice(1);
    cudaMemcpy(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice);
    launchUpsampleKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, scaleFactor, 0);

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

    // create a thread for each GPU to process the images??
    
}