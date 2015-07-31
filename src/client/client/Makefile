CPPSRC:=$(wildcard *.cpp)

all:    client_cuda client_cpu

cuda_lib:
	make --directory cuda

client_cuda: cuda_lib
	g++ -o client-cuda ${CPPSRC} ${INCLUDE} ${LIBS} ${CXXFLAGS} -D_CUDA -I./ -Icuda cuda/cuda.a -L${CUDA_LIB} -I${CUDA_INCLUDE} -lbigint -lutil -lecc -lgmp -llogger -lthread -lpthread -lcudart -lcurl

cpu_lib:
	make --directory cpu

client_cpu: cpu_lib
	g++ -o client-cpu ${CPPSRC} ${INCLUDE} ${LIBS} ${CXXFLAGS} -D_CPU -I./ -Icpu cpu/cpu.a cpu/math/math.a -lbigint -lutil -lecc -lgmp -llogger -lthread -lpthread -lcurl

clean:
	rm -f *.o
	rm -f client-cuda
	rm -f client-cpu
	make --directory cuda clean
	make --directory cpu clean
