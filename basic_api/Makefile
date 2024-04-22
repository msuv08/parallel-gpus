all: main

main: main.cpp MegaGPU.cpp addArrays.cu
	nvcc -o main main.cpp MegaGPU.cpp addArrays.cu -I. -std=c++11

clean:
	rm -f main

