#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    void prepareImage(const std::string& imagePath);
    int getWidth() const;
    int getHeight() const;
    std::vector<unsigned char> getInput();
    std::vector<unsigned char> getOutput();

    std::vector<unsigned char> input;
    std::vector<unsigned char> output;

private:
    int width;
    int height;
};

#endif

#include "ImageProcessor.h"
#include "MegaGPU.h"

// int main() {
//     std::string imagePath = "image.png";
//     ImageProcessor processor;
//     processor.prepareImage(imagePath);
//     MegaGPU mega;
//     int scaleFactor = 2;
//     std::vector<unsigned char> input = processor.getInput();
//     std::vector<unsigned char> output = processor.getOutput();
//     mega.upsampleImage(input.data(), output.data(), processor.getWidth(), processor.getHeight(), scaleFactor);
//     cv::Mat resultImg(processor.getHeight() * scaleFactor, processor.getWidth() * scaleFactor, CV_8UC3, output.data());
//     cv::imwrite("upsampled_image.png", resultImg);
//     return 0;
// }