# GigaAPI for GPU Parallelization: The Documentation

### By: Omeed Tehrani and Mihir Suvarna

![image](https://github.com/msuv08/parallel-gpus/assets/61725820/56c7c852-fdf4-4d9e-833a-d61648ddad51)

## What is the GigaAPI?

The user-space API presented in this project aims to simplify the process of utilizing multiple GPUs as a single, powerful computing resource. By abstracting the complexities associated with multi-GPU programming and CUDA, this API allows developers to leverage the benefits of parallel computing without the need for extensive knowledge in these areas.

The primary goal of this project is to demonstrate the potential advantages of a generalizable API for parallelism, making it more accessible to developers. By providing a proof of concept, this project encourages future researchers to further develop and refine user-space APIs that streamline multi-GPU programming.

The API is designed to be user-friendly and intuitive, enabling developers to harness the power of multiple GPUs with minimal effort. This approach can lead to improved performance in applications without the need for developers to delve into the intricacies of traditional multi-GPU programming or CUDA.

## 2 Styles of Parallelism

### Basic Operations (3 subtasks)

1. **Vector Operations**: The `computeDotProduct` function in the GigaAPI allows you to compute the dot product of two vectors using multiple GPUs in parallel. It takes the input vectors, the result variable, and the size of the vectors as parameters. The `computeL2Norm` function enables you to calculate the L2 norm of a vector using multiple GPUs concurrently. It accepts the input vector, the result variable, and the size of the vector as arguments.

2. **Fast Fourier Transform**: The `performFFT` function in the GigaAPI is used to perform the Fast Fourier Transform (FFT) on input data using multiple GPUs. It requires the input data array, the output FFT data array, the dimensions of the FFT, and the number of GPUs to use for the computation as parameters.

3. **Matrix Multiplication**: The `performMatrixMultiplication` function performs matrix multiplication using multiple GPUs. It takes the input matrices A and B, the output matrix C, and their respective dimensions as parameters.

### Images (3 subtasks)

1. **RGB to Grayscale Conversion**: The `convertToGrayscale` function converts an RGB image to grayscale using multiple GPUs. It requires the input RGB image data, the output grayscale image data, and the dimensions of the image as parameters.

2. **Image Upsampling**: The `upsampleImage` function performs image upsampling using multiple GPUs. It takes the input image data, the output upsampled image data, the dimensions of the image, and the upsampling scale factor as arguments. Additionally, the `upsampleAllImages` function allows you to upsample multiple images using multiple GPUs.

3. **Image Sharpening**: The `sharpenImage` function applies image sharpening using multiple GPUs. It accepts the input image data, the output sharpened image data, and the dimensions of the image as parameters.

### Theoretical Works

1. **Parallel Mining**: The `parallelMining` function in the GigaAPI enables parallel mining using multiple GPUs. It accepts the block data to be mined and the target hash value as parameters, and returns the mined result as a string.

## Installation and Usage

To use the GigaAPI, follow these steps:

1. Clone the GigaAPI repository.
2. Set the appropriate paths for OpenCV and CUDA in the Makefile.
3. Run `make` to build the GigaAPI library in `giga-api/common`.
4. Include the `MegaGPU.h` header file in your C++ code and create a relevant Makefile.
5. Instantiate the MegaGPU object and call the desired functions based on your requirements.

Please refer to the API documentation for detailed information on each function's parameters and usage (see the `README` inside `giga-api/common`).

## Base Makefile

The provided Makefile in `common` is used to build the GigaAPI library. It specifies the necessary compiler flags, include directories, and library directories for OpenCV and CUDA.

To build the library, simply run `make` in the project directory. This will compile the source files and generate the `libmegagpu.a` library file.

To clean the build artifacts, run `make clean`.

Note: If any changes are made to the CUDA kernels or the GigaAPI source files, you need to run `make` again to rebuild the library.

#### Sources
- [Nearest-Neighbor Interpolation](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
- [Upsampling](https://mriquestions.com/upsampling.html)
- [Grayscale Image Algorithm](https://tannerhelland.com/2011/10/01/grayscale-image-algorithm-vb6.html)
- [Discrete Approximations to the Laplacian Filter](https://www.researchgate.net/figure/Two-commonly-used-discrete-approximations-to-the-Laplacian-filter_fig1_261459927)
- [Image Sharpening](https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm)
- [Processing Image with CUDA Kernel](https://forums.developer.nvidia.com/t/processing-image-with-a-cuda-kernel-gives-me-different-result-than-a-seemingly-equivalent-cpu-function/282033)
- [CUFFT Library](https://developer.download.nvidia.com/compute/cuda/1.0/CUFFT_Library_1.0.pdf)