# MegaGPU API

The MegaGPU API is a multi-GPU library that provides functions for various image processing and computational tasks. It utilizes CUDA and cuFFT libraries for efficient parallel processing on multiple GPUs.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [API Functions](#api-functions)
  - [convertToGrayscale](#converttograyscale)
  - [prepareData](#preparedata)
  - [performFFT](#performfft)
  - [upsampleImage](#upsampleimage)
  - [upsampleAllImages](#upsampleallimages)
  - [sharpenImage](#sharpenimage)
  - [performMatrixMultiplication](#performmatrixmultiplication)
  - [computeDotProduct](#computedotproduct)
  - [computeL2Norm](#computel2norm)
  - [parallelMining](#parallelmining)
  - [singleGPU_upsampling](#singlegpu_upsampling)
- [Makefile](#makefile)

## Prerequisites
- CUDA (version 12.0)
- cuFFT library
- OpenCV (version 4.x)

## Installation

1. Clone the MegaGPU repository.
2. Set the appropriate paths for OpenCV and CUDA in the Makefile:
   ```makefile
   OPENCV_INCLUDE_DIR=/path/to/opencv/include
   OPENCV_LIB_DIR=/path/to/opencv/lib
   CUDA_LIB_DIR=/path/to/cuda/lib64
   ```
3. Run `make` to build the MegaGPU library.

## API Functions

### convertToGrayscale
```cpp
void convertToGrayscale(const unsigned char* input, unsigned char* output, int width, int height)
```
Converts an RGB image to grayscale using multiple GPUs.
- `input`: Pointer to the input RGB image data.
- `output`: Pointer to the output grayscale image data.
- `width`: Width of the image.
- `height`: Height of the image.

### prepareData
```cpp
void prepareData(float* input, int size)
```
Prepares the input data for FFT by reading samples from a file.
- `input`: Pointer to the input data array.
- `size`: Size of the input data array.

### performFFT
```cpp
void performFFT(float* input, cufftComplex* output, int width, int height, int numGPUs)
```
Performs the Fast Fourier Transform (FFT) on the input data using multiple GPUs.
- `input`: Pointer to the input data array.
- `output`: Pointer to the output FFT data array.
- `width`: Width of the FFT.
- `height`: Height of the FFT.
- `numGPUs`: Number of GPUs to use for the FFT computation.

### upsampleImage
```cpp
void upsampleImage(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor)
```
Upsamples an image using multiple GPUs.
- `input`: Pointer to the input image data.
- `output`: Pointer to the output upsampled image data.
- `width`: Width of the input image.
- `height`: Height of the input image.
- `scaleFactor`: Upsampling scale factor.

### upsampleAllImages
```cpp
void upsampleAllImages(const std::vector<std::string>& imagePaths, int scaleFactor)
```
Upsamples multiple images using multiple GPUs.
- `imagePaths`: Vector of image file paths.
- `scaleFactor`: Upsampling scale factor.

### sharpenImage
```cpp
void sharpenImage(const unsigned char* input, unsigned char* output, int width, int height)
```
Applies sharpening to an image using multiple GPUs.
- `input`: Pointer to the input image data.
- `output`: Pointer to the output sharpened image data.
- `width`: Width of the image.
- `height`: Height of the image.

### performMatrixMultiplication
```cpp
void performMatrixMultiplication(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols)
```
Performs matrix multiplication using multiple GPUs.
- `A`: Pointer to the input matrix A.
- `B`: Pointer to the input matrix B.
- `C`: Pointer to the output matrix C.
- `A_rows`: Number of rows in matrix A.
- `A_cols`: Number of columns in matrix A.
- `B_cols`: Number of columns in matrix B.

### computeDotProduct
```cpp
void computeDotProduct(const float* a, const float* b, float& result, int n)
```
Computes the dot product of two vectors using multiple GPUs.
- `a`: Pointer to the first input vector.
- `b`: Pointer to the second input vector.
- `result`: Reference to the variable to store the dot product result.
- `n`: Size of the vectors.

### computeL2Norm
```cpp
void computeL2Norm(const float* a, float& result, int n)
```
Computes the L2 norm of a vector using multiple GPUs.
- `a`: Pointer to the input vector.
- `result`: Reference to the variable to store the L2 norm result.
- `n`: Size of the vector.

### parallelMining
```cpp
std::string parallelMining(const std::string& blockData, const std::string& target)
```
Performs parallel mining using multiple GPUs.
- `blockData`: Block data to be mined.
- `target`: Target hash value.

Returns the mined result as a string.

### singleGPU_upsampling
```cpp
void singleGPU_upsampling(const unsigned char* input, unsigned char* output, int width, int height, int scaleFactor)
```
Performs image upsampling using a single GPU.
- `input`: Pointer to the input image data.
- `output`: Pointer to the output upsampled image data.
- `width`: Width of the input image.
- `height`: Height of the input image.
- `scaleFactor`: Upsampling scale factor.

## Makefile

The provided Makefile is used to build the MegaGPU library. It specifies the necessary compiler flags, include directories, and library directories for OpenCV and CUDA.

To build the library, simply run `make` in the project directory. This will compile the source files and generate the `libmegagpu.a` library file.

To clean the build artifacts, run `make clean`.

Note: If any changes are made to the CUDA kernels or the MegaGPU source files, you need to run `make` again to rebuild the library.
```