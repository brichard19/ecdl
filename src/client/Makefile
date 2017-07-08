INCLUDE += -I$(shell pwd)/include
LIBDIR=$(shell pwd)/lib
LIBS+=-L$(LIBDIR)

# C++ options
CXX=g++
CXXFLAGS=-O2

# CUDA variables
COMPUTE_CAP=30
NVCC=nvcc -O3
NVCCFLAGS=-gencode=arch=compute_${COMPUTE_CAP},code=\"sm_${COMPUTE_CAP}\" -Xptxas="-v" -Xcompiler "${CXXFLAGS}"
CUDA_HOME=/usr/local/cuda-7.5
CUDA_LIB=${CUDA_HOME}/lib64
CUDA_INCLUDE=${CUDA_HOME}/include

# Google test variables
GTEST_DIR=$(shell pwd)/gtest
GTEST_SRC=${GTEST_DIR}/src/*.cc
GTEST_INCLUDE=${GTEST_DIR}/include

#GTEST_INCLUDE=${GTEST_DIR}/include
#GTEST_HEADERS=$(GTEST_DIR)/include/gtest/*.h $(GTEST_DIR)/include/gtest/internal/*.h
#GTEST_SRC=$(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
#GTEST_INCLUDE=-I${GTEST_DIR}/include
#GTEST_SRC=${GTEST_DIR}/src/*.cc
#GTEST_SRC=$(wildcard ${GTEST_DIR}/src/*.cc})

# ASM variables
NASM=nasm
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
export GTEST_DIR
export GTEST_SRC
export GTEST_INCLUDE

bigint:	client_bigint
ecc:    client_ecc
cuda:   client_cuda
logger: client_logger
tle:	client_tle
sha256:	client_sha256
util:	client_util
ecdb:	client_ecdb
tests:	client_tests
config: client_config

client_bigint:
	make --directory bigint

client_ecc: client_bigint
	make --directory ecc

client_tle:	client_bigint
	make --directory tle

client_logger:
	make --directory logger

client_threads:
	make --directory threads

client_util:
	make --directory util

client_cuda:    client_bigint client_ecc client_logger client_threads client_util client_tle client_sha256 client_config
	make --directory client client_cuda

client_cpu:    client_bigint client_ecc client_logger client_threads client_util client_tle client_sha256
	make --directory client client_cpu ecc_test

client_sha256:
	make --directory sha256

client_config:
	make --directory config

client_ecdb: client_bigint
	make --directory ecdb

client_tests: client_sha256
	make --directory test

clean:
	make --directory client clean
	make --directory bigint clean
	make --directory ecc clean
	make --directory logger clean
	make --directory util clean
	make --directory threads clean
	make --directory config clean
	rm -rf ${LIBDIR}
