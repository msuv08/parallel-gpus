#include <iostream>
#include <vector>
#include "MegaGPU.h"

int main() {
    int n = 10; // number of elements in each array
    std::vector<int> a(n, 1), b(n, 2), c(n);

    MegaGPU mega(n);
    mega.addArrays(a.data(), b.data(), c.data());

    for (int i = 0; i < n; i++) {
        std::cout << "c[" << i << "] = a[" << i << "] + b[" << i << "] :: " 
                    << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }
    return 0;
}

