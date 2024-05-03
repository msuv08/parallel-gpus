## Performing FFT using MegaGPU

To perform the Fast Fourier Transform (FFT) using the MegaGPU library, follow these steps:

1. Include the `MegaGPU.h` header file in your C++ code:
   ```cpp
   #include "MegaGPU.h"
   ```

2. Instantiate the MegaGPU object in your code:
   ```cpp
   MegaGPU mega;
   ```

3. Prepare the input data:
   - Create a vector of float to store the input signal data.
   - The size of the input vector should be the desired width of the FFT.

4. Prepare the output data:
   - Create a vector of `cufftComplex` to store the output FFT data.
   - The size of the output vector should be the same as the input vector.

5. Call the `prepareData` function of the MegaGPU object to prepare the input data:
   ```cpp
   mega.prepareData(input.data(), width);
   ```
   - `input.data()`: Pointer to the input signal data.
   - `width`: Width of the FFT.

6. Call the `performFFT` function of the MegaGPU object to perform the FFT:
   ```cpp
   mega.performFFT(input.data(), output.data(), width, 1, numGPUs);
   ```
   - `input.data()`: Pointer to the input signal data.
   - `output.data()`: Pointer to the output FFT data.
   - `width`: Width of the FFT.
   - `1`: Batch size (set to 1 for a single FFT).
   - `numGPUs`: Number of GPUs to use for the FFT computation (optional, default is 1).

   The `performFFT` function will perform the FFT on the input data using the MegaGPU and store the resulting FFT data in the output vector.

7. After the FFT computation, the FFT data will be available in the output vector, which you can use for further processing or saving to a file.

Note: Make sure to have the MegaGPU library properly linked and the necessary dependencies (CUDA and cuFFT) installed before using the `performFFT` function.

## Code Explanation