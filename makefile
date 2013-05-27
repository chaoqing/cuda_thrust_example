
cpu:cpu.o
	g++  -Wall -std=c++0x ./cpu.o  -o cpu.out -lncurses -lrt
cpu.o:main.cpp
	g++  -Wall -std=c++0x -c main.cpp -o cpu.o

gpu:gpu.o
	nvcc -m64 ./gpu.o -o gpu.out -lncurses -lrt -L/usr/local/cuda-5.0/lib64 -lcudart -lcurand
gpu.o:main.cu
	nvcc -m64 -gencode arch=compute_20,code=sm_20  -I/usr/local/cuda-5.0/include -I. -I/usr/local/cuda-5.0/samples/0_Simple -I/usr/local/cuda-5.0/samples/common/inc  -D NDEBUG -c main.cu -o gpu.o

clean:
	rm *.o *.out
