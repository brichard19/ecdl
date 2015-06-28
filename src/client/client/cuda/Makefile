CPPSRC:=$(wildcard *.cpp)
CUSRC+=$(wildcard *.cu)
CUSRC:=$(filter-out fp131.cu, $(CUSRC))

all:    cuda

cuda:
	for file in ${CPPSRC} ; do\
		${CXX} -c $$file ${INCLUDE} -I../ -I/usr/local/cuda/include ${CXXFLAGS};\
	done
	for file in ${CUSRC} ; do\
		${NVCC} -c ${NVCCFLAGS} ${INCLUDE} -I/usr/local/cuda/include ${CUSRC};\
	done
	ar rvs cuda.a *.o

clean:
	rm -f *.o
	rm -f *.a
	rm -f client-cuda
