#ifndef MEGAGPU_H
#define MEGAGPU_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <string>


class MegaGPU {
public:
    MegaGPU();
    ~MegaGPU(); 
    void convertToGrayscale(const unsigned char* input, unsigned char* output, int width, int height);
    void prepareData(float* input, int size);
    // void performFFT(float* input, cufftComplex* output, int width, int height);
    // void MegaGPU::performFFT(float* input, cufftComplex* output, int width, int height, int numGPUs) {
    void performFFT(float* input, cufftComplex* output, int width, int height, int numGPUs);
    void upsampleImage(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor);
    void upsampleAllImages(const std::vector<std::string>& imagePaths, int scaleFactor);
    void sharpenImage(const unsigned char* input, unsigned char* output, int width, int height);
    void performMatrixMultiplication(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols);
    std::string parallelMining(const std::string& blockData, const std::string& target);
// void MegaGPU::singleGPU_upsampling(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor) {
    void singleGPU_upsampling(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor);
    

private:
    // RGB to Grayscale variables
    unsigned char* d_input0, * d_output0;
    unsigned char* d_input1, * d_output1;
    // FFT variables
    float* d_fftInput0, * d_fftInput1;
    cufftComplex* d_fftOutput0, * d_fftOutput1;
    int imageWidth, imageHeight, sizePerGPU;
    int scaleFactor;
    // Matrix multiplication variables
    float* d_inputA0, *d_inputB0, *d_outputC0;
    float* d_inputA1, *d_inputB1, *d_outputC1;
    int sizePerGPU_A, sizePerGPU_B, sizePerGPU_C;
    // mining pointers
    char* d_miningData;
    char* d_results;
};

#endif