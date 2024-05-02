#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <filesystem> // Make sure this is supported
#include "MegaGPU.h"

void runSingleImageUpsampling(const std::string& imagePath, int scaleFactor) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    int width = img.cols;
    int height = img.rows;
    std::vector<unsigned char> input(img.total() * img.channels());
    std::vector<unsigned char> output(img.total() * img.channels() * scaleFactor * scaleFactor);
    std::memcpy(input.data(), img.data, img.total() * img.channels());

    MegaGPU mega;
    mega.upsampleImage(input.data(), output.data(), width, height, scaleFactor);
    cv::Mat resultImg(height * scaleFactor, width * scaleFactor, CV_8UC3, output.data());
    cv::imwrite("upsampled_image.png", resultImg);
    std::cout << "Upsampled image saved to upsampled_image.png" << std::endl;
}

void processSingleImage() {
    std::string imagePath="img.png";
    // integer for the scale factor
    int scaleFactor=4;
    // std::cout << "Please enter the exact path to your image: ";
    // std::cin >> imagePath;
    // std::cout << "What factor would you like to upscale your image by: ";
    // std::cin >> scaleFactor;
    runSingleImageUpsampling(imagePath, scaleFactor);
}

// Function to print basic GPU stats
void printGPUStats() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "  Device name: " << prop.name << std::endl;
        std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
        std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): "
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
    }
}

// Method to benchmark scaling on a single image
void benchmarkScaling(const std::string& imagePath) {
    float milliseconds = 0;

    for (int scaleFactor = 2; scaleFactor <= 40; scaleFactor += 1) {
        std::cout << "FOR SCALE FACTOR: " << scaleFactor << std::endl;
        runSingleImageUpsampling(imagePath, scaleFactor);
        std::cout << std::endl;
        // std::cout << "Time taken for scale factor " << scaleFactor << ": " << milliseconds << " ms" << std::endl;
    }
}


void runMultipleImageUpsampling(const std::string& folderPath, int scaleFactor) {

    MegaGPU mega;
    // first, we should get a list of all the image names in the directory
    std::vector<std::string> imageNames;
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        imageNames.push_back(entry.path().string());
    }

    // by this point, we should have the direct paths to all the images
    // we need to preprocess the images prior to passing it into a new 
    // function to upsample them
    mega.upsampleAllImages(imageNames, scaleFactor);
    std::cout << "All images have been upscaled by a factor of " << scaleFactor << std::endl;
}


void processPhotoDatabase() {
    std::string folderPath;
    int scaleFactor;
    std::cout << "Please enter the exact path to your photo database: ";
    std::cin >> folderPath;
    std::cout << "What factor would you like to upscale your images by: ";
    std::cin >> scaleFactor;
    runMultipleImageUpsampling(folderPath, scaleFactor);

}

int main() {
    // int choice;
    // std::cout << "-=-=-=-=-=-=-=-" << std::endl;
    // std::cout << "Parallization can be advantageous on large factor upscaling or when processing a large database of images." << std::endl;
    // std::cout << "Press 0 to upsample a singular image or 1 to upscale a database of images: ";
    // std::cin >> choice;
    benchmarkScaling("img.png");
    // printGPUStats();
    // processSingleImage();

    // if (choice == 0) {
    //     processSingleImage();
    // } else if (choice == 1) {
    //     std::string folderPath = "photo_database";
    //         processPhotoDatabase();
    // } else {
    //     std::cerr << "Invalid input. Please enter 0 or 1." << std::endl;
    //     return -1;
    // }
    // return 0;
}
