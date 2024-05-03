#include "ImageProcessor.h"
#include <cstring>

ImageProcessor::ImageProcessor() {

}

ImageProcessor::~ImageProcessor() {

}

void ImageProcessor::prepareImage(const std::string& imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    width = img.cols;
    height = img.rows;

    input.resize(img.total() * img.channels());
    output.resize(img.total() * img.channels() * 4);  

    std::memcpy(input.data(), img.data, img.total() * img.channels());
    

}
int ImageProcessor::getWidth() const {
    return width;
}

int ImageProcessor::getHeight() const {
    return height;
}

// function to return input data
std::vector<unsigned char> ImageProcessor::getInput() {
    return input;
}

// function to return output data
std::vector<unsigned char> ImageProcessor::getOutput() {
    return output;
}



void upsampling(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor) {
    imageWidth = width;
    imageHeight = height;
    sizePerGPU = imageWidth * (imageHeight / 2) * 3; // Each pixel RGB
    int outputWidth = imageWidth * scaleFactor;
    int outputHeight = imageHeight * scaleFactor;
    int outputSizePerGPU = outputWidth * (outputHeight / 2) * 3;

    // std::cout << "Begin image upsampling..." << std::endl;

    cudaSetDevice(0);
    cudaMalloc(&d_input0, sizePerGPU);
    cudaMalloc(&d_output0, outputSizePerGPU);

    cudaSetDevice(1);
    cudaMalloc(&d_input1, sizePerGPU);
    cudaMalloc(&d_output1, outputSizePerGPU);

    int halfHeight = imageHeight / 2;

    //create streams for each GPU
    cudaStream_t stream0, stream1;
    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaSetDevice(1);
    cudaStreamCreate(&stream1);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    cudaSetDevice(0);
    cudaMemcpyAsync(d_input0, input, sizePerGPU, cudaMemcpyHostToDevice, stream0);
    launchUpsampleKernel(d_input0, d_output0, imageWidth, halfHeight, scaleFactor, stream0);
    // std::cout << "Launching GPU Kernel #0" << std::endl;

    cudaSetDevice(1);
    cudaMemcpyAsync(d_input1, input + sizePerGPU, sizePerGPU, cudaMemcpyHostToDevice, stream1);
    launchUpsampleKernel(d_input1, d_output1, imageWidth, imageHeight - halfHeight, scaleFactor, stream1);
    // std::cout << "Launching GPU Kernel #1" << std::endl;

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    cudaMemcpyAsync(output, d_output0, outputSizePerGPU, cudaMemcpyDeviceToHost, stream0);

    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(output + outputSizePerGPU, d_output1, outputSizePerGPU, cudaMemcpyDeviceToHost, stream1);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();


    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << "Total GPU time: " << elapsedTime << " ms" << std::endl;

    cudaSetDevice(0);
    cudaFree(d_input0);
    cudaFree(d_output0);
    cudaStreamDestroy(stream0);

    cudaSetDevice(1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    cudaStreamDestroy(stream1);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}