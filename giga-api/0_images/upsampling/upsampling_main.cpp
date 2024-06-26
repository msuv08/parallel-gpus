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
    cv::imwrite("images/upsampled_image.png", resultImg);
    std::cout << "Upsampled image saved to upsampled_image.png" << std::endl;
}

void processSingleImage() {
    std::string imagePath="images/img.png";
    // integer for the scale factor
    int scaleFactor=4;
    // std::cout << "Please enter the exact path to your image: ";
    // std::cin >> imagePath;
    // std::cout << "What factor would you like to upscale your image by: ";
    // std::cin >> scaleFactor;
    runSingleImageUpsampling(imagePath, scaleFactor);
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
    int choice;
    std::cout << "-=-=-=-=-=-=-=-" << std::endl;
    std::cout << "Parallization can be advantageous on large factor upscaling or when processing a large database of images." << std::endl;
    std::cout << "Press 0 to upsample a singular image or 1 to upscale a database of images: ";
    std::cin >> choice;

    if (choice == 0) {
        processSingleImage();
    } else if (choice == 1) {
        std::string folderPath = "photo_database";
            processPhotoDatabase();
    } else {
        std::cerr << "Invalid input. Please enter 0 or 1." << std::endl;
        return -1;
    }
    return 0;
}
