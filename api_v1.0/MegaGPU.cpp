#include "MegaGPU.h"
#include <iostream>
#include <cmath>

extern "C" void launchGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream);
extern "C" void performFFTKernel(float* input, cufftComplex* output, int width, int height, cudaStream_t stream);

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
    if (d_fftInput0) cudaFree(d_fftInput0);
    if (d_fftInput1) cudaFree(d_fftInput1);
    if (d_fftOutput0) cudaFree(d_fftOutput0);
    if (d_fftOutput1) cudaFree(d_fftOutput1);
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

void MegaGPU::prepareData(float* data, int size, float frequency, float sampleRate) {
    for (int i = 0; i < size; i++) {
        float t = i / sampleRate;
        data[i] = sin(2 * M_PI * frequency * t);  // Generating a sinusoidal wave
    }
}

void MegaGPU::performFFT(const float* input, cufftComplex* output, int width, int height) {
    int totalSize = width * height;
    int sizePerGPU = totalSize / 2; // Splitting data for two GPUs

    // Allocating memory for FFT input and output on each GPU
    cudaSetDevice(0);
    cudaMalloc(&d_fftInput0, sizeof(float) * sizePerGPU);
    cudaMalloc(&d_fftOutput0, sizeof(cufftComplex) * sizePerGPU);
    cudaMemcpy(d_fftInput0, input, sizeof(float) * sizePerGPU, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMalloc(&d_fftInput1, sizeof(float) * sizePerGPU);
    cudaMalloc(&d_fftOutput1, sizeof(cufftComplex) * sizePerGPU);
    cudaMemcpy(d_fftInput1, input + sizePerGPU, sizeof(float) * sizePerGPU, cudaMemcpyHostToDevice);

    // Performing FFT on each device using separate streams
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    performFFTKernel(d_fftInput0, d_fftOutput0, width, height / 2, stream0);
    performFFTKernel(d_fftInput1, d_fftOutput1, width, height / 2, stream1);

    // Synchronizing and copying back the results
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaMemcpy(output, d_fftOutput0, sizeof(cufftComplex) * sizePerGPU, cudaMemcpyDeviceToHost);
    cudaMemcpy(output + sizePerGPU, d_fftOutput1, sizeof(cufftComplex) * sizePerGPU, cudaMemcpyDeviceToHost);

    // Cleaning up
    cudaFree(d_fftInput0);
    cudaFree(d_fftOutput0);
    cudaFree(d_fftInput1);
    cudaFree(d_fftOutput1);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}