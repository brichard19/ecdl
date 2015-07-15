INCLUDE += -I$(shell pwd)/include
LIBDIR=$(shell pwd)/lib
LIBS+=-L$(LIBDIR)


CXX=g++
CXXFLAGS=-O2 -Wall
NASM=nasm
COMPUTE_CAP=30
NVCC=nvcc
NVCCFLAGS=-O2 -gencode=arch=compute_${COMPUTE_CAP},code=\"sm_${COMPUTE_CAP}\" -Xcompiler "${CXXFLAGS}"
CUDA_LIB=/usr/local/cuda/lib
CUDA_INCLUDE=/usr/local/cuda/include
X86_ASM=0

export INCLUDE
export LIBDIR
export NVCC
export NVCCFLAGS
export LIBS
export CXX
export CXXFLAGS
export NASM
export CUDA_LIB
export CUDA_INCLUDE
export X86_ASM

bigint:	client_bigint
ecc:    client_ecc
cuda:   client_cuda
logger: client_logger

client_bigint:
	make --directory bigint

client_ecc: client_bigint
	make --directory ecc

client_logger:
	make --directory logger

client_threads:
	make --directory threads

client_util:
	make --directory util

client_cuda:    client_bigint client_ecc client_logger client_threads client_util
	make --directory client client_cuda

client_cpu:    client_bigint client_ecc client_logger client_threads client_util
	make --directory client client_cpu

clean:
	make --directory client clean
	make --directory bigint clean
	make --directory ecc clean
	make --directory logger clean
	make --directory util clean
	make --directory threads clean
	rm -rf ${LIBDIR}
