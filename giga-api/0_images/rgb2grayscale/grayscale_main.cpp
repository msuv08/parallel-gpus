#include "MegaGPU.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    // Load image
    std::string imagePath = "images/cybertruck.jpeg";
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        return -1;
    }
    int width = img.cols;
    int height = img.rows;
    std::vector<unsigned char> input(img.total() * img.channels());
    std::vector<unsigned char> output(img.total());
    std::memcpy(input.data(), img.data, img.total() * img.channels());
    MegaGPU mega;
    mega.convertToGrayscale(input.data(), output.data(), width, height);
    cv::Mat resultImg(height, width, CV_8UC1, output.data());
    cv::imwrite("images/cybertruck_grayscale.jpeg", resultImg);
    std::cout << "Grayscale image saved to cybertruck_grayscale.jpeg" << std::endl;
    return 0;
}
