#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <filesystem> // Make sure this is supported
#include "MegaGPU.h"

void runSingleImageSharpening(const std::string& imagePath, int scaleFactor) {
    // same logic as upsampling!
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    int width = img.cols;
    int height = img.rows;
    std::vector<unsigned char> input(img.total() * img.channels());
    std::vector<unsigned char> output(img.total() * img.channels() * scaleFactor * scaleFactor);
    std::memcpy(input.data(), img.data, img.total() * img.channels());

    MegaGPU mega;
    mega.sharpenImage(input.data(), output.data(), width, height, scaleFactor);
    cv::Mat resultImg(height * scaleFactor, width * scaleFactor, CV_8UC3, output.data());
    cv::imwrite("sharpened_image.png", resultImg);
    std::cout << "sharpened image was saved to sharpened_image.png" << std::endl;
}

// same as before
void processSingleImage() {
    std::string imagePath;
    // integer for the scale factor
    int scaleFactor;
    std::cout << "Please enter the exact path to your image: ";
    std::cin >> imagePath;
    std::cout << "What factor would you like to upscale your image by: ";
    std::cin >> scaleFactor;
    runSingleImageSharpening(imagePath, scaleFactor);
}

int main() {
    processSingleImage();
    return 0;
}
