# OPENCV_INCLUDE_DIR=/u/omeed/opt/opencv/include/opencv4
# OPENCV_LIB_DIR=/u/omeed/opt/opencv/lib
# CUDA_LIB_DIR=/u/omeed/cuda-12.0/lib64
# PARENT_DIR=../../
# OPENCV_INCLUDE_DIR=/u/msuvarna/opt/opencv/include/opencv4
# OPENCV_LIB_DIR=/u/msuvarna/opt/opencv/lib
# CUDA_LIB_DIR=/u/msuvarna/cuda-12.0/lib64
# PARENT_DIR=../../

# # Compiler
# CXX=g++
# NVCC=nvcc

# # Compiler flags
# CXXFLAGS=-std=c++17 -I$(OPENCV_INCLUDE_DIR) -I$(PARENT_DIR)
# LDFLAGS=-L$(OPENCV_LIB_DIR) -L$(CUDA_LIB_DIR)
# LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
# NVCC_FLAGS=-std=c++17 -I$(OPENCV_INCLUDE_DIR) -I$(PARENT_DIR)

# # Main target
# all: upsampling basic_upsampling

# # Main compilation rule
# upsampling: upsampling_main.cpp $(PARENT_DIR)/MegaGPU.cpp $(PARENT_DIR)/CudaKernels.cu
# 	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) -lcufft

# basic_upsampling: basic_upsampling.cpp
# 	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# # Clean up
# clean:
# 	rm -f upsampling basic_upsampling

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
all: upsampling basic_upsampling

# Main compilation rule
upsampling: upsampling_main.cpp
	$(NVCC) -o upsampling upsampling_main.cpp $(NVCC_FLAGS)

basic_upsampling: cpu_upsampling/basic_upsampling.cpp
	$(NVCC) -o cpu_upsampling/basic_upsampling cpu_upsampling/basic_upsampling.cpp $(NVCC_FLAGS)

# Clean up
clean:
	rm -f upsampling cpu_upsampling/basic_upsampling
