OPENCV_INCLUDE_DIR=/u/msuvarna/opt/opencv/include/opencv4
OPENCV_LIB_DIR=/u/msuvarna/opt/opencv/lib
CUDA_LIB_DIR=/u/msuvarna/cuda-12.0/lib64
# OPENCV_INCLUDE_DIR=/u/omeed/opt/opencv/include/opencv4
# OPENCV_LIB_DIR=/u/omeed/opt/opencv/lib
# CUDA_LIB_DIR=/u/omeed/cuda-12.0/lib64

PARENT_DIR=../../

# Compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS=-std=c++17 -I$(OPENCV_INCLUDE_DIR) -I$(PARENT_DIR) \
           -L$(OPENCV_LIB_DIR) -L$(CUDA_LIB_DIR) \
           -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lcufft

# Main target
all: matmult

# Main compilation rule
matmult: matmult_main.cpp
	$(NVCC) -o matmult matmult_main.cpp $(PARENT_DIR)/MegaGPU.cpp $(PARENT_DIR)/CudaKernels.cu $(NVCC_FLAGS)

# Clean up
clean:
	rm -f matmult
	