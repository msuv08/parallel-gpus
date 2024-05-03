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
