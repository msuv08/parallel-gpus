#include <string> 
#include <iostream>

bool isHashValid(const char* hash, const char* target) {
    for (int i = 0; i < 64; ++i) {
        if (hash[i] > target[i])
            return false;
        if (hash[i] < target[i])
            return true;
    }
    return true;  
}


void simpleHash(const char* data, char* hash, int lineSize) {
    unsigned long hashValue = 5381;
    // Hash each character in the data input
    for (int i = 0; i < lineSize; ++i) {
        char c = data[i];
        hashValue = ((hashValue << 5) + hashValue) + c; // hashValue * 33 + c
    }

    // Properly convert the hashValue to a 64 character hexadecimal string
    unsigned long mask = 15; // Mask for hexadecimal digits (0x0F)
    // Initialize the hash array to zero before filling
    for (int i = 0; i < 64; ++i) {
        hash[i] = '0';
    }
    // Fill the hash array with hexadecimal digits from hashValue
    for (int i = 63; i >= 0 && hashValue != 0; --i) {
        unsigned int digit = hashValue & mask; // Extract the lowest 4 bits
        if (digit <= 9) {
            hash[i] = '0' + digit;
        } else {
            hash[i] = 'A' + (digit - 10); // Convert 10-15 to 'A'-'F'
        }
        hashValue >>= 4; // Shift right by 4 bits to process the next hexadecimal digit
    }
    hash[64] = '\0'; // Null-terminate the string
}


int main() {
std::string inputData = "0000000000000000000000000000000000000000000000000000000000141880";
std::string result = "000000000000000000000000000000000000000000000000E6E3B39F1403431B";


// Prepare for hashing
char hash[65];
simpleHash(inputData.c_str(), hash, inputData.length());

// Output the hash
std::cout << "Hash: ";
for (int i = 0; i < 64; ++i) {
    std::cout << hash[i];
}
std::cout << std::endl;
}