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

// externs are used to declare functions within the correct scope.
extern "C" void launchGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream);
extern "C" void performFFTKernel(float* input, cufftComplex* output, int width, int height, cudaStream_t stream);
extern "C" void launchUpsampleKernel(unsigned char* input, unsigned char* output, int width, int height, int scaleFactor, cudaStream_t stream);
extern "C" void launchSharpenKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream);
extern "C" void launchMatrixMulKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols, cudaStream_t stream);
extern "C" void launchMiningKernel(char* miningData, int numLines, int lineSize, char* results, const char* target);

MegaGPU::MegaGPU() {
    d_input0 = d_output0 = nullptr;
    d_input1 = d_output1 = nullptr;
    imageWidth = imageHeight = sizePerGPU = 0;
    d_inputA0 = d_inputB0 = d_outputC0 = nullptr;
    d_inputA1 = d_inputB1 = d_outputC1 = nullptr;
    // reset mining pointers
    d_miningData = nullptr;
    d_results = nullptr;
}

MegaGPU::~MegaGPU() {
    // this is all to simply free the gpu memory! 
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
    if (d_miningData) cudaFree(d_miningData);
    if (d_results) cudaFree(d_results);
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

// void MegaGPU::performFFT(float* input, cufftComplex* output, int width, int height) {
//     float* d_input;
//     cufftComplex* d_output;
//     cudaMalloc((void**)&d_input, width * sizeof(float));
//     cudaMalloc((void**)&d_output, (width / 2 + 1) * height * sizeof(cufftComplex));
//     cudaMemcpy(d_input, input, width * sizeof(float), cudaMemcpyHostToDevice);
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     // TODO: potentially do time based operations here on kernel call. - omeed
//     performFFTKernel(d_input, d_output, width, height, stream);
    
//     cudaMemcpy(output, d_output, (width / 2 + 1) * height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
//     cudaFree(d_input);
//     cudaFree(d_output);
//     cudaStreamDestroy(stream);
// }

void MegaGPU::performFFT(float* input, cufftComplex* output, int width, int height, int numGPUs) {
    int chunkHeight = height / numGPUs;
    int remainingHeight = height % numGPUs;

    std::vector<float*> d_inputChunks(numGPUs);
    std::vector<cufftComplex*> d_outputChunks(numGPUs);
    std::vector<cudaStream_t> streams(numGPUs);

    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);

        int chunkSize = chunkHeight;
        if (i == numGPUs - 1)
            chunkSize += remainingHeight;

        cudaMalloc((void**)&d_inputChunks[i], width * chunkSize * sizeof(float));
        cudaMalloc((void**)&d_outputChunks[i], (width / 2 + 1) * chunkSize * sizeof(cufftComplex));

        int offset = i * chunkHeight;
        cudaMemcpy(d_inputChunks[i], input + offset * width, width * chunkSize * sizeof(float), cudaMemcpyHostToDevice);

        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);

        int chunkSize = chunkHeight;
        if (i == numGPUs - 1)
            chunkSize += remainingHeight;

        performFFTKernel(d_inputChunks[i], d_outputChunks[i], width, chunkSize, streams[i]);
    }

    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);

        int chunkSize = chunkHeight;
        if (i == numGPUs - 1)
            chunkSize += remainingHeight;

        int offset = i * chunkHeight;
        cudaMemcpy(output + offset * (width / 2 + 1), d_outputChunks[i], (width / 2 + 1) * chunkSize * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

        cudaFree(d_inputChunks[i]);
        cudaFree(d_outputChunks[i]);
        cudaStreamDestroy(streams[i]);
    }
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

// i dont know why its not working, i added an insane amount of print statements, even generated
// matching hashes but it isnt working.
std::string MegaGPU::parallelMining(const std::string& blockData, const std::string& target) {

    std::vector<std::string> miningData;
    std::cout << "Starting mining process." << std::endl;

    for (unsigned long long nonce = 0; nonce <= 200000; ++nonce) {
        std::string nonceStr = std::to_string(nonce);
        std::string paddedNonce = std::string(7 - nonceStr.length(), '0') + nonceStr; 
        std::string data = blockData + paddedNonce;
        miningData.push_back(data);
        std::cout << "Generated data with nonce " << nonce << ": " << data << std::endl;  
    }

    int numLines = miningData.size();
    int lineSize = miningData[0].length();
    
    std::cout << "Total data lines: " << numLines << ", Line size: " << lineSize << std::endl;

    char* d_miningData;
    char* d_results;

    cudaMalloc(&d_miningData, numLines * lineSize * sizeof(char));
    std::cout << "Allocated memory for mining data on GPU." << std::endl;
    
    cudaMalloc(&d_results, numLines * sizeof(char));
    std::cout << "Allocated memory for results on GPU." << std::endl;

    for (int i = 0; i < numLines; i++) {
        cudaMemcpy(d_miningData + i * lineSize, miningData[i].c_str(), lineSize * sizeof(char), cudaMemcpyHostToDevice);
        std::cout << "Copied line " << i << " to GPU." << std::endl;
    }
    
    std::cout << "Launching mining kernel." << std::endl;
    launchMiningKernel(d_miningData, numLines, lineSize, d_results, target.c_str());

    std::string result;
    for (int i = 0; i < numLines; i++) {
        char found;
        cudaError_t err = cudaMemcpy(&found, d_results + i, sizeof(char), cudaMemcpyDeviceToHost);
        if (found == 1) {
            result = miningData[i];
            std::cout << "Match found in line " << i << "." << std::endl;
            break;
        } else {
            std::cout << "No match found in line " << i << "." << std::endl;
        }
    }
    return result;
}
