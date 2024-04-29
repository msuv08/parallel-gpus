#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
        c[index] = a[index] + b[index];
    if (index == 0) {
        printf("Kernel running on GPU, first element calculated: %d + %d = %d\n", a[index], b[index], c[index]);
    }
}

int main() {
    const int N = 1024;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; // Device pointers for GPU 1
    int *d2_a, *d2_b, *d2_c; // Device pointers for GPU 2
    int size = N * sizeof(int);

    std::cout << "Initializing GPUs and allocating memory...\n";

    cudaSetDevice(0); // Set to GPU 0
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    std::cout << "Memory allocated on GPU 0\n";

    cudaSetDevice(1); // Set to GPU 1
    cudaMalloc(&d2_a, size);
    cudaMalloc(&d2_b, size);
    cudaMalloc(&d2_c, size);
    std::cout << "Memory allocated on GPU 1\n";

    // Initialize arrays and copy to GPU 1
    a = new int[N]; std::fill(a, a + N, 1); // Fill array with 1s
    b = new int[N]; std::fill(b, b + N, 2); // Fill array with 2s
    c = new int[N];
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    std::cout << "Data copied to GPU 0\n";

    // Copy data from GPU 1 to GPU 2
    cudaMemcpyPeer(d2_a, 1, d_a, 0, size);
    cudaMemcpyPeer(d2_b, 1, d_b, 0, size);
    std::cout << "Data copied from GPU 0 to GPU 1\n";

    // Launch kernel on both GPUs
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaSetDevice(0);
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    std::cout << "Kernel launched on GPU 0\n";

    cudaSetDevice(1);
    add<<<blocksPerGrid, threadsPerBlock>>>(d2_a, d2_b, d2_c, N);
    std::cout << "Kernel launched on GPU 1\n";

    // Copy result back to host from both GPUs
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    int *c2 = new int[N];
    cudaMemcpy(c2, d2_c, size, cudaMemcpyDeviceToHost);
    std::cout << "Results copied back to host from both GPUs\n";

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d2_a); cudaFree(d2_b); cudaFree(d2_c);
    delete[] a; delete[] b; delete[] c; delete[] c2;

    cudaDeviceReset();
    std::cout << "Cleanup complete, device reset.\n";
    return 0;
}

