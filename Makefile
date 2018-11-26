# First ssh into air205
# HCUDA_ext.cu is HCUDA.cu with 'extern "C"'
# need -arch=sm_53 to use half precision from cufa_fp16.h

all:
	@gcc -I/usr/local/cuda/include/ -c *.c
	@nvcc -c *.cu -Wno-deprecated-gpu-targets -arch=sm_53 --std=c++11
	@gcc -L/usr/local/cuda/lib64 -o test *.o -lcuda -lcudart -lcublas -lm
	@rm *.o
