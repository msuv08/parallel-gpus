#ifndef MEGAGPU_H
#define MEGAGPU_H

#include <cuda_runtime.h>

class MegaGPU {
public:
    MegaGPU();  // Constructor to initialize GPUs
    ~MegaGPU(); // Destructor to free resources

    void convertToGrayscale(const unsigned char *input, unsigned char *output, int width, int height);

private:
    unsigned char *d_input0, *d_output0;
    unsigned char *d_input1, *d_output1;
    int imageWidth, imageHeight, sizePerGPU;
};

#endif // MEGAGPU_H
