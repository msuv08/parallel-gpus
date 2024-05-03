## Image Upsampling using MegaGPU

To upsample an image using the MegaGPU library, follow these steps:

1. Include the `MegaGPU.h` header file in your C++ code:
   ```cpp
   #include "MegaGPU.h"
   ```

2. Instantiate the MegaGPU object in your code:
   ```cpp
   MegaGPU mega;
   ```

3. Prepare the input and output data:
   - Create a vector of unsigned char to store the input image data.
   - Create another vector of unsigned char to store the output upsampled image data.
   - The size of the input vector should be `width * height * num_channels`, where `num_channels` is typically 3 for color images.
   - The size of the output vector should be `width * height * num_channels * scaleFactor * scaleFactor`, where `scaleFactor` is the desired upsampling factor.

4. Call the `upsampleImage` function of the MegaGPU object:
   ```cpp
   mega.upsampleImage(input.data(), output.data(), width, height, scaleFactor);
   ```
   - `input.data()`: Pointer to the input image data.
   - `output.data()`: Pointer to the output upsampled image data.
   - `width`: Width of the input image.
   - `height`: Height of the input image.
   - `scaleFactor`: Upsampling scale factor.

   The `upsampleImage` function will process the input image using the MegaGPU and store the resulting upsampled image in the output vector.

5. After the upsampling process, the upsampled image data will be available in the output vector, which you can use for further processing or saving to a file.

Note: Make sure to have the MegaGPU library properly linked and the necessary dependencies (OpenCV and CUDA) installed before using the `upsampleImage` function.

## Makefile

The Makefile provided in the project directory contains the necessary rules to compile the project. It specifies the include and library directories for OpenCV and CUDA, as well as the compiler flags.

To compile the project, simply run `make` in the project directory.

To clean up the compiled files, run `make clean`.