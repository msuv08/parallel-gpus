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
