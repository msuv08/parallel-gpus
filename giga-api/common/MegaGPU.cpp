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
extern "C" void launchSharpenKernel(unsigned char* input, unsigned char* output, int width, int height, int scaleFactor, cudaStream_t stream);
extern "C" void launchMatrixMulKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols, cudaStream_t stream);
extern "C" void launchMiningKernel(char* miningData, int numLines, int lineSize, char* results, const char* target);
extern "C" void launchVectorDotKernel(const float* a, const float* b, float* result, int n, cudaStream_t stream);
extern "C" void launchVectorL2NormKernel(const float* a, float* result, int n, cudaStream_t stream);

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

    cudaStream_t stream0, stream1;
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);

    // Copy to GPU 0
    cudaSetDevice(0);
    cudaMemcpyAsync(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice, stream0);
    launchGrayscaleKernel(d_input0, d_output0, imageWidth, halfHeight, stream0);
    std::cout << "Launching GPU Kernel #0: " << std::endl;

    // Copy to GPU 1
    cudaSetDevice(1);
    cudaMemcpyAsync(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice, stream1);
    launchGrayscaleKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, stream1);
    std::cout << "Launching GPU Kernel #1: " << std::endl;

    // Copy results back
    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    cudaMemcpyAsync(output, d_output0, imageWidth * halfHeight, cudaMemcpyDeviceToHost, stream0);

    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(output + imageWidth * halfHeight, d_output1, imageWidth * (imageHeight - halfHeight), cudaMemcpyDeviceToHost, stream1);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    // Free memory on GPUs
    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    cudaStreamDestroy(stream1);
    std::cout << "End grayscale conversion..." << std::endl;
}

void MegaGPU::prepareData(float* input, int size) {
    std::ifstream inputFile("params/input_signal.txt");
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
    cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEventRecord(startEvent, 0);

    for (int i = 0; i < numGPUs; ++i) {
        cudaSetDevice(i);

        int chunkSize = chunkHeight;
        if (i == numGPUs - 1)
            chunkSize += remainingHeight;
            
    
        performFFTKernel(d_inputChunks[i], d_outputChunks[i], width, chunkSize, streams[i]);

    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << "Total GPU time: " << elapsedTime << " ms" << std::endl;

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

    // std::cout << "Begin image upsampling..." << std::endl;

    cudaSetDevice(0);
    cudaMalloc(&d_input0, sizePerGPU);
    cudaMalloc(&d_output0, outputSizePerGPU);

    cudaSetDevice(1);
    cudaMalloc(&d_input1, sizePerGPU);
    cudaMalloc(&d_output1, outputSizePerGPU);

    int halfHeight = imageHeight / 2;

    //create streams for each GPU
    cudaStream_t stream0, stream1;
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);


    cudaEventRecord(startEvent, 0);

    cudaSetDevice(0);
    cudaMemcpyAsync(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice, stream0);
    launchUpsampleKernel(d_input0, d_output0, imageWidth, halfHeight, scaleFactor, stream0);
    // std::cout << "Launching GPU Kernel #0" << std::endl;

    cudaSetDevice(1);
    cudaMemcpyAsync(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice, stream1);
    launchUpsampleKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, scaleFactor, stream1);
    // std::cout << "Launching GPU Kernel #1" << std::endl;

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);    

    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    cudaMemcpyAsync(output, d_output0, outputSizePerGPU, cudaMemcpyDeviceToHost, stream0);

    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(output + outputSizePerGPU, d_output1, outputSizePerGPU, cudaMemcpyDeviceToHost, stream1);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();


    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << "Total GPU time: " << elapsedTime << " ms" << std::endl;

    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    cudaStreamDestroy(stream1);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);



    // std::cout << "End image upsampling..." << std::endl;
}

void MegaGPU::upsampleAllImages(const std::vector<std::string>& imagePaths, int scaleFactor) {
    // put half the image paths in the list on one GPU for processing
    std::vector<std::string> gpu0Images(imagePaths.begin(), imagePaths.begin() + imagePaths.size() / 2);

    // put the other half of the image paths on the other GPU for processing
    std::vector<std::string> gpu1Images(imagePaths.begin() + imagePaths.size() / 2, imagePaths.end());

    // create a thread for each GPU to process the images?? idk bro
    
}

// very similar to before! 
void MegaGPU::sharpenImage(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor) {
    imageWidth = width;
    imageHeight = height;
    sizePerGPU = imageWidth * (imageHeight / 2) * 3; // Each pixel RGB
    int outputWidth = imageWidth;
    int outputHeight = imageHeight;
    int outputSizePerGPU = outputWidth * (outputHeight / 2) * 3;

    // std::cout << "Begin image upsampling..." << std::endl;

    cudaSetDevice(0);
    cudaMalloc(&d_input0, sizePerGPU);
    cudaMalloc(&d_output0, outputSizePerGPU);

    cudaSetDevice(1);
    cudaMalloc(&d_input1, sizePerGPU);
    cudaMalloc(&d_output1, outputSizePerGPU);

    int halfHeight = imageHeight / 2;

    //create streams for each GPU
    cudaStream_t stream0, stream1;
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    cudaSetDevice(0);
    cudaMemcpyAsync(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice, stream0);

    launchSharpenKernel(d_input0, d_output0, imageWidth, halfHeight, scaleFactor, stream0);
    // std::cout << "Launching GPU Kernel #0" << std::endl;

    cudaSetDevice(1);
    cudaMemcpyAsync(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice, stream1);
    launchSharpenKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, scaleFactor, stream1);
    // std::cout << "Launching GPU Kernel #1" << std::endl;

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    cudaMemcpyAsync(output, d_output0, outputSizePerGPU, cudaMemcpyDeviceToHost, stream0);

    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(output + outputSizePerGPU, d_output1, outputSizePerGPU, cudaMemcpyDeviceToHost, stream1);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "Total GPU time: " << elapsedTime << " ms" << std::endl;

    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    cudaStreamDestroy(stream1);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // std::cout << "End image upsampling..." << std::endl;
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

void MegaGPU::singleGPU_upsampling(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor) {

    imageWidth = width;
    imageHeight = height;
    int imageSize = imageWidth * imageHeight * 3; 
    int outputWidth = imageWidth * scaleFactor;
    int outputHeight = imageHeight * scaleFactor;
    int outputSize = outputWidth * outputHeight * 3;

    // std::cout << "Begin image upsampling..." << std::endl;

    cudaSetDevice(0);
    cudaMalloc(&d_input0, imageSize);
    cudaMalloc(&d_output0, outputSize);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    cudaMemcpy(d_input0, input, imageSize, cudaMemcpyHostToDevice);
    launchUpsampleKernel(d_input0, d_output0, imageWidth, imageHeight, scaleFactor, 0);
    // std::cout << "Launching GPU Kernel" << std::endl;
    cudaDeviceSynchronize();

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << "Total GPU time: " << elapsedTime << " ms" << std::endl;
    
    cudaMemcpy(output, d_output0, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input0);
    cudaFree(d_output0);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

        // std::cout << "End image upsampling..." << std::endl;
}

void MegaGPU::singleGPU_sharpening(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor) {
    imageWidth = width;
    imageHeight = height;
    int imageSize = imageWidth * imageHeight * 3;
    int outputWidth = imageWidth;
    int outputHeight = imageHeight;
    int outputSize = outputWidth * outputHeight * 3;

    cudaSetDevice(0);
    cudaMalloc(&d_input0, imageSize);
    cudaMalloc(&d_output0, outputSize);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    cudaMemcpy(d_input0, input, imageSize, cudaMemcpyHostToDevice);
    launchSharpenKernel(d_input0, d_output0, imageWidth, imageHeight, scaleFactor, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "Total GPU time: " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(output, d_output0, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input0);
    cudaFree(d_output0);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void MegaGPU::singleGPU_performFFT(float* input, cufftComplex* output, int width, int height, int numGPUs) {
    float* d_input;
    cufftComplex* d_output;
    cudaMalloc((void**)&d_input, width * sizeof(float));
    cudaMalloc((void**)&d_output, (width / 2 + 1) * height * sizeof(cufftComplex));
    cudaMemcpy(d_input, input, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    // TODO: potentially do time based operations here on kernel call. - omeed
    performFFTKernel(d_input, d_output, width, height, stream);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "Total GPU time: " << elapsedTime << " ms" << std::endl;


    cudaMemcpy(output, d_output, (width / 2 + 1) * height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}
