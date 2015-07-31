CPPSRC:=$(wildcard *.cpp)
CUSRC+=$(wildcard *.cu)
CUSRC:=$(filter-out Fp.cu, $(CUSRC))

all:	cuda

cuda:
	for file in ${CPPSRC} ; do\
		${CXX} -c $$file ${INCLUDE} -I../ -I${CUDA_INCLUDE} ${CXXFLAGS};\
	done
	for file in ${CUSRC} ; do\
		${NVCC} -c ${NVCCFLAGS} ${INCLUDE} -I${CUDA_INCLUDE} ${CUSRC};\
	done
	ar rvs cuda.a *.o

clean:
	rm -f *.o
	rm -f *.a
	rm -f client-cuda
