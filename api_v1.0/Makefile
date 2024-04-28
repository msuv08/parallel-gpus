# OPENCV_INCLUDE_DIR=/u/msuvarna/opt/opencv/include/opencv4
# OPENCV_LIB_DIR=/u/msuvarna/opt/opencv/lib
# CUDA_LIB_DIR=/u/msuvarna/cuda-12.0/lib64
OPENCV_INCLUDE_DIR=/u/omeed/opt/opencv/include/opencv4
OPENCV_LIB_DIR=/u/omeed/opt/opencv/lib
CUDA_LIB_DIR=/u/omeed/cuda-12.0/lib64

all: main

main: main.cpp MegaGPU.cpp CudaKernels.cu
	nvcc -o main main.cpp MegaGPU.cpp CudaKernels.cu -I$(OPENCV_INCLUDE_DIR) -L$(OPENCV_LIB_DIR) -L$(CUDA_LIB_DIR) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lcufft -std=c++11

clean:
	rm -f main
