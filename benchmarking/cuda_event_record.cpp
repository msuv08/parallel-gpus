// file to just hold some basic code for cuda event recording which we might
// need to use in the future.

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    
    // DO SOME KERNEL STUFF HERE? 

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // cudaFree(d_data);
}
