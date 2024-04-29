#include "MegaGPU.h"
#include <iostream>
#include <fstream>
#include <string>

std::string readBlockDataFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return "";
    }

    std::string blockData((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    return blockData;
}

int main() {
    std::string blockDataFile;
    std::cout << "Enter the path to the block data file: ";
    std::getline(std::cin, blockDataFile);

    std::string blockData = readBlockDataFromFile(blockDataFile);
    if (blockData.empty()) {
        return 1;
    }

    std::string target;
    std::cout << "Enter the target hash: ";
    std::getline(std::cin, target);

    MegaGPU mega;
    std::string result = mega.parallelMining(blockData, target);

    if (!result.empty()) {
        std::cout << "Block mined successfully!" << std::endl;
        std::cout << "Nonce: " << result.substr(blockData.length()) << std::endl;
        std::cout << "Hash: " << result.substr(0, 64) << std::endl;
    } else {
        std::cout << "Mining failed. No valid hash found." << std::endl;
    }

    return 0;
}