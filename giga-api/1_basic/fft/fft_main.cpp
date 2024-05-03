#include <iostream>
#include <vector>
#include <fstream>
#include <cufft.h>
#include "MegaGPU.h"

int main() {
    int width = 1024;
    std::vector<float> input(width);
    std::vector<cufftComplex> output(width);
    MegaGPU mega;
    // input data comes from the signal generation! :)
    mega.prepareData(input.data(), width);
    // mega.performFFT(input.data(), output.data(), width, 1);
    // make a new call with 2 gpus
    mega.performFFT(input.data(), output.data(), width, 1, 2);


    // Save the FFT output to a file, potentially for analysis! 
    // not used anywhere.
    std::ofstream outputFile("params/fft_output.txt");
    for (int i = 0; i < output.size(); ++i) {
        outputFile << output[i].x << " " << output[i].y << "\n";
    }
    outputFile.close();
    return 0;
}