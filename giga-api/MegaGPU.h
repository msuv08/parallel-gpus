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
    void performFFT(float* input, cufftComplex* output, int width, int height);
    void upsampleImage(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor);
    void upsampleAllImages(const std::vector<std::string>& imagePaths, int scaleFactor);
    void sharpenImage(const unsigned char* input, unsigned char* output, int width, int height);
    void performMatrixMultiplication(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols);
    void computeDotProduct(const float* a, const float* b, float& result, int n);
    void computeL2Norm(const float* a, float& result, int n);
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

    // Vector operations
    float* d_vectorA, *d_vectorB, *d_vectorC;
    float* d_scalarResult;
};

#endif