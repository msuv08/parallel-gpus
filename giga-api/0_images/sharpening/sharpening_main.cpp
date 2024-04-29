#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <filesystem>
#include "MegaGPU.h"

void runSingleImageSharpening(const std::string& imagePath, int scaleFactor) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    int width = img.cols;
    int height = img.rows;
    std::vector<unsigned char> input(img.total() * img.channels());
    std::vector<unsigned char> output(img.total() * img.channels());
    std::memcpy(input.data(), img.data, img.total() * img.channels());
    MegaGPU mega;
    mega.sharpenImage(input.data(), output.data(), width, height);
    cv::Mat sharpenedImg(height, width, CV_8UC3, output.data());
    cv::imwrite("sharpened_image.png", sharpenedImg);
    std::cout << "Sharpened image saved to sharpened_image.png" << std::endl;
}

void processSingleImage() {
    std::string imagePath;
    std::cout << "Please enter the exact path to your image: ";
    std::cin >> imagePath;
    runSingleImageSharpening(imagePath, 1);
}

int main() {
    processSingleImage();
    return 0;
}