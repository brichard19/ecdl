#SRC:=$(wildcard *.cpp)
#SRC+=$(wildcard *.cu)
#SRC:=$(filter-out fp131.cu, $(SRC))

CPPSRC:=$(wildcard *.cpp)
#CUSRC+=$(wildcard *.cu)
#CUSRC:=$(filter-out fp131.cu, $(CUSRC))

CURL=./curl64

all:    client_cuda client_cpu

cuda_lib:
	make --directory cuda

client_cuda: cuda_lib
	g++ -o client-cuda ${CPPSRC} ${INCLUDE} ${LIBS} ${CXXFLAGS} -D_CUDA -I./ -Icuda cuda/cuda.a -L${CURL} -L${CUDA_LIB} -I${CUDA_INCLUDE} -lbigint -lutil -lecc -lgmp -llogger -lthread -lpthread -lcudart -lcurl

cpu_lib:
	make --directory cpu

client_cpu: cpu_lib
	g++ -o client-cpu ${CPPSRC} ${INCLUDE} ${LIBS} ${CXXFLAGS} -D_CPU -I./ -Icpu cpu/cpu.a -lbigint -lutil -lecc -lgmp -llogger -lthread -lpthread -lcurl

clean:
	rm -f *.o
	rm -f client-cuda
	rm -f client-cpu
	make --directory cuda clean
	make --directory cpu clean
