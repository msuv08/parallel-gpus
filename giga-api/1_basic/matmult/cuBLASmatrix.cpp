#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <vector>
#include <iomanip>

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
    curandDestroyGenerator(prng);
}

// Multiply the arrays A and B on GPU and save the result in C
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m, ldb=k, ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, A, lda, B, ldb, beta, C, ldc);
    cublasDestroy(handle);
}

void benchmark(int start_exp, int end_exp) {
    for (int exp = start_exp; exp <= end_exp; ++exp) {
        int size = (1 << exp); // 2^exp

        // Allocate memory on the host
        float *h_A = (float *)malloc(size * size * sizeof(float));
        float *h_B = (float *)malloc(size * size * sizeof(float));
        float *h_C = (float *)malloc(size * size * sizeof(float));

        // Allocate memory on the device
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size * size * sizeof(float));
        cudaMalloc(&d_B, size * size * sizeof(float));
        cudaMalloc(&d_C, size * size * sizeof(float));

        // Fill the arrays A and B on GPU with random numbers
        GPU_fill_rand(d_A, size, size);
        GPU_fill_rand(d_B, size, size);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record the start event
        cudaEventRecord(start, 0);

        // Multiply A and B on GPU
        gpu_blas_mmul(d_A, d_B, d_C, size, size, size);

        // Record the stop event
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate the elapsed time
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);

        // Output the matrix size and the elapsed time
        std::cout << "Matrix size: " << size << "x" << size << ", Time: " << elapsed_time << " ms" << std::endl;

        // Destroy CUDA events and free memory
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
    }
}

int main() {
    benchmark(1, 15);  // Start at 2^1 (2) and end at 2^15 (32768)
    return 0;
}
