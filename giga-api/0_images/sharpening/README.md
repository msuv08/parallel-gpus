## Image Sharpening using MegaGPU

To sharpen an image using the MegaGPU library, follow these steps:

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
   - Create another vector of unsigned char to store the output sharpened image data.
   - The size of both vectors should be `width * height * num_channels`, where `num_channels` is typically 3 for color images.

4. Call the `sharpenImage` function of the MegaGPU object:
   ```cpp
   mega.sharpenImage(input.data(), output.data(), width, height);
   ```
   - `input.data()`: Pointer to the input image data.
   - `output.data()`: Pointer to the output sharpened image data.
   - `width`: Width of the image.
   - `height`: Height of the image.

   The `sharpenImage` function will process the input image using the MegaGPU and store the resulting sharpened image in the output vector.

5. After the sharpening process, the sharpened image data will be available in the output vector, which you can use for further processing or saving to a file.

Note: Make sure to have the MegaGPU library properly linked and the necessary dependencies (OpenCV and CUDA) installed before using the `sharpenImage` function.

## Code Explanation

1. The `runSingleImageSharpening` function takes the path to an image and a scale factor as input. It loads the image using OpenCV, prepares the input and output data, calls the `sharpenImage` function of the MegaGPU object, and saves the sharpened image.

2. The `processSingleImage` function prompts the user to enter the path to an image and calls the `runSingleImageSharpening` function with the provided image path.

3. The `main` function serves as the entry point of the program. It calls the `processSingleImage` function to process a single image.

## Makefile

The Makefile provided in the project directory contains the necessary rules to compile the project. It specifies the include and library directories for OpenCV and CUDA, as well as the compiler flags.

To compile the project, simply run `make` in the project directory.

To clean up the compiled files, run `make clean`.