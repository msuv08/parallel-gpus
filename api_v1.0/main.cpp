#include "MegaGPU.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// int main() {
//     std::string imagePath = "cybertruck.jpeg";
//     cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
//     if (img.empty()) {
//         std::cerr << "Error: Image not found." << std::endl;
//         return -1;
//     }

//     int width = img.cols;
//     int height = img.rows;
//     std::vector<unsigned char> input(img.total() * img.channels());
//     std::vector<unsigned char> output(img.total());

//     // Copying image data to input vector
//     std::memcpy(input.data(), img.data, img.total() * img.channels());

//     MegaGPU mega;
//     mega.convertToGrayscale(input.data(), output.data(), width, height);

//     // Creating output image
//     cv::Mat resultImg(height, width, CV_8UC1, output.data());
//     cv::imwrite("cybertruck_grayscale.jpeg", resultImg);

//     std::cout << "Grayscale image saved to cybertruck_grayscale.jpeg" << std::endl;

//     return 0;
// }

int main() {
    int width = 1024;  // width represents total samples
    int height = 1;    // height used just for compatibility with existing 2D structures
    std::vector<float> input(width * height);
    std::vector<cufftComplex> output(width * height);

    MegaGPU mega;
    mega.prepareData(input.data(), width * height, 1.0, width);  // 1 Hz frequency, sampled at 1024 Hz
    mega.performFFT(input.data(), output.data(), width, height);

    std::cout << "FFT processing complete. Output ready for further processing." << std::endl;

    return 0;
}
