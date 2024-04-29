#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        output[y * width + x] = gray;
    }
}

__global__ void upsampleKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int scaleFactor) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < inputWidth * scaleFactor && y < inputHeight * scaleFactor) {
        int srcX = x / scaleFactor;
        int srcY = y / scaleFactor;

        for (int c = 0; c < 3; c++) {
            output[(y * inputWidth * scaleFactor + x) * 3 + c] = input[(srcY * inputWidth + srcX) * 3 + c];
        }
    }
}

extern "C" void launchGrayscaleKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    grayscaleKernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height);
    cudaDeviceSynchronize();
    std::cout << "Grayscale kernel execution complete." << std::endl;
}

extern "C" void performFFTKernel(float* input, cufftComplex* output, int width, int height, cudaStream_t stream) {
    cufftHandle plan;
    cufftResult result;
    std::cout << "FFT KERNEL START..." << std::endl;
    result = cufftPlan2d(&plan, width, height, CUFFT_R2C);
    result = cufftSetStream(plan, stream);
    result = cufftExecR2C(plan, input, output);
    result = cufftDestroy(plan);
    cudaError_t cudaResult = cudaDeviceSynchronize();
    std::cout << "FFT KERNEL END..." << std::endl;
}

extern "C" void launchUpsampleKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int scaleFactor, cudaStream_t stream) {
    // purely for warp of 32 alignment? -im not sure if this is right, the output is losing quality. 
    dim3 blockSize(16, 16);
    dim3 gridSize((inputWidth * scaleFactor + 15) / 16, (inputHeight * scaleFactor + 15) / 16);

    upsampleKernel<<<gridSize, blockSize, 0, stream>>>(input, output, inputWidth, inputHeight, scaleFactor);
    // do we need this?
    cudaDeviceSynchronize();
    std::cout << "Upsampling kernel execution complete." << std::endl;
}
