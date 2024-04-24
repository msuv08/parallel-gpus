#include <cuda_runtime.h>
#include <iostream>

__global__ void grayscaleKernel(unsigned char *input, unsigned char *output, int width, int height) {
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

extern "C" void launchGrayscaleKernel(unsigned char *input, unsigned char *output, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    grayscaleKernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height);
    cudaDeviceSynchronize();
    std::cout << "Grayscale kernel execution complete." << std::endl;
}
