# Include and lib directories
OPENCV_INCLUDE_DIR=/u/msuvarna/opt/opencv/include/opencv4
OPENCV_LIB_DIR=/u/msuvarna/opt/opencv/lib
CUDA_LIB_DIR=/u/msuvarna/cuda-12.0/lib64
COMMON_DIR=../../common

# Compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS=-std=c++17 -I$(OPENCV_INCLUDE_DIR) -I$(COMMON_DIR) \
           -L$(OPENCV_LIB_DIR) -L$(CUDA_LIB_DIR) -L$(COMMON_DIR) \
           -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lcufft -lmegagpu

# Main target
all: sharpening

# Main compilation rule
sharpening: sharpening_main.cpp
	$(NVCC) -o sharpening sharpening_main.cpp $(NVCC_FLAGS)

# Clean up
clean:
	rm -f sharpening
