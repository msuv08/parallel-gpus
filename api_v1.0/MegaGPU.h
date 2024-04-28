#ifndef MEGAGPU_H
#define MEGAGPU_H

#include <cuda_runtime.h>
#include <cufft.h>

class MegaGPU {
public:
    MegaGPU();
    ~MegaGPU(); 
    void convertToGrayscale(const unsigned char* input, unsigned char* output, int width, int height);
    void prepareData(float* input, int size);
    void performFFT(float* input, cufftComplex* output, int width, int height);
    
private:
    unsigned char* d_input0, * d_output0;
    unsigned char* d_input1, * d_output1;
    float* d_fftInput0, * d_fftInput1;
    cufftComplex* d_fftOutput0, * d_fftOutput1;
    int imageWidth, imageHeight, sizePerGPU;
};

#endif