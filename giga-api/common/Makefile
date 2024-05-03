OPENCV_INCLUDE_DIR=/u/msuvarna/opt/opencv/include/opencv4
OPENCV_LIB_DIR=/u/msuvarna/opt/opencv/lib
CUDA_LIB_DIR=/u/msuvarna/cuda-12.0/lib64
# OPENCV_INCLUDE_DIR=/u/omeed/opt/opencv/include/opencv4
# OPENCV_LIB_DIR=/u/omeed/opt/opencv/lib
# CUDA_LIB_DIR=/u/omeed/cuda-12.0/lib64

# Compiler
NVCC=nvcc

# Compiler flags
NVCC_FLAGS=-std=c++17 -I$(OPENCV_INCLUDE_DIR)

# Output
LIBRARY=libmegagpu.a

# Build the library
$(LIBRARY): MegaGPU.o CudaKernels.o
	ar rcs $@ $^

MegaGPU.o: MegaGPU.cpp MegaGPU.h
	$(NVCC) -c -o $@ MegaGPU.cpp $(NVCC_FLAGS)

CudaKernels.o: CudaKernels.cu
	$(NVCC) -c -o $@ CudaKernels.cu $(NVCC_FLAGS)

clean:
	rm -f *.o $(LIBRARY)
