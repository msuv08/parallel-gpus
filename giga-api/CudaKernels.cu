// THIS FILE ENCOMPASSES ALL OF OUR CUDA KERNELS!

#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

__constant__ float laplacianFilter[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

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

// https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm -> 8 worked best here (omeed)
// this version used -4 -> https://github.com/KhosroBahrami/ImageFiltering_CUDA/blob/master/LaplacianFilter/laplacianFilter.cu
__global__ void sharpenKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = (y * width + x) * 3; 

    // for each color channel
    for (int c = 0; c < 3; c++) {
        float sum = 0.0;
        // indexing with i and j was off originally, neighbor indexing is now correct.
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int neighborIndex = (ny * width + nx) * 3 + c; 
                        int filterIndex = (dy + 1) * 3 + (dx + 1);
                        float pixelValue = input[neighborIndex]; 
                        float filterValue = laplacianFilter[filterIndex]; 
                        sum += pixelValue * filterValue; 
                } else {
                    // add some weight to the sum if the pixel is out of bounds! 
                    sum += 5;
                }
            }
        }
        float sharpened = sum + input[idx + c];
        sharpened = max(sharpened, 0.0);
        sharpened = min(sharpened, 255.0);
        output[idx + c] = sharpened;
    }
}

__global__ void matrixMulKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < A_rows && col < B_cols) {
        float sum = 0.0;
        for (int k = 0; k < A_cols; ++k) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
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

extern "C" void launchSharpenKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    sharpenKernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height);
    cudaDeviceSynchronize();
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

extern "C" void launchSharpenKernel(unsigned char* input, unsigned char* output, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    sharpenKernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height);
    cudaDeviceSynchronize();
}

extern "C" void launchMatrixMulKernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((B_cols + blockSize.x - 1) / blockSize.x, (A_rows + blockSize.y - 1) / blockSize.y);
    matrixMulKernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, A_rows, A_cols, B_cols);
    cudaDeviceSynchronize();
    std::cout << "Matrix multiplication kernel execution complete." << std::endl;
}
