## Converting RGB to Grayscale using MegaGPU

To convert an RGB image to grayscale using the MegaGPU library, follow these steps:

1. Include the `MegaGPU.h` header file in your C++ code:
   ```cpp
   #include "MegaGPU.h"
   ```

2. Instantiate the MegaGPU object in your code:
   ```cpp
   MegaGPU mega;
   ```

3. Prepare the input and output data:
   - Create a vector of unsigned char to store the input RGB image data.
   - Create another vector of unsigned char to store the output grayscale image data.
   - The size of the input vector should be `width * height * num_channels`, where `num_channels` is typically 3 for RGB images.
   - The size of the output vector should be `width * height`, as grayscale images have only one channel.

4. Call the `convertToGrayscale` function of the MegaGPU object:
   ```cpp
   mega.convertToGrayscale(input.data(), output.data(), width, height);
   ```
   - `input.data()`: Pointer to the input RGB image data.
   - `output.data()`: Pointer to the output grayscale image data.
   - `width`: Width of the image.
   - `height`: Height of the image.

   The `convertToGrayscale` function will process the input RGB image using the MegaGPU and store the resulting grayscale image in the output vector.

5. After the conversion, the grayscale image data will be available in the output vector, which you can use for further processing or saving to a file.

Note: Make sure to have the MegaGPU library properly linked and the necessary dependencies (OpenCV and CUDA) installed before using the `convertToGrayscale` function.

## Makefile

The Makefile provided in the project directory contains the necessary rules to compile the project. It specifies the include and library directories for OpenCV and CUDA, as well as the compiler flags.

To compile the project, simply run `make` in the project directory.

To clean up the compiled files, run `make clean`.