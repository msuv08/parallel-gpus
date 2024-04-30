// CPU UPSAMPLING FOR BENCHMARKING

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat inputImage = cv::imread("image.png");

    if (inputImage.empty()) {
        std::cout << "Failed to load the input image." << std::endl;
        return -1;
    }

    double upsamplingFactor = 30.0;

    int newWidth = static_cast<int>(inputImage.cols * upsamplingFactor);
    int newHeight = static_cast<int>(inputImage.rows * upsamplingFactor);

    cv::Mat upsampledImage;

    cv::resize(inputImage, upsampledImage, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    cv::imwrite("upsampled.png", upsampledImage);

    std::cout << "Upsampled image saved as 'upsampled.png'" << std::endl;

    return 0;
}