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
all: vectorops vectorbaseline

# Main compilation rule
vectorops: vectorops_main.cpp
	$(NVCC) -o vectorops vectorops_main.cpp $(PARENT_DIR)/MegaGPU.cpp $(PARENT_DIR)/CudaKernels.cu $(NVCC_FLAGS)

vectorbaseline: vectorbaseline.cpp
	$(NVCC) -o vectorbaseline vectorbaseline.cpp $(PARENT_DIR)/MegaGPU.cpp $(PARENT_DIR)/CudaKernels.cu $(NVCC_FLAGS) -lcublas
# Clean up
clean:
	rm -f vectorops vectorbaseline
	