#include <cuda_runtime.h>

__global__ void addArraysKernel(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

extern "C" void launchAddArrays(int *a, int *b, int *c, int n, int blocksPerGrid, int threadsPerBlock, cudaStream_t stream = 0) {
    addArraysKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, b, c, n);
}
