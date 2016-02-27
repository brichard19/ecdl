CPPSRC:=$(wildcard *.cpp)
CPPSRC:=$(filter-out fp_test.cpp, $(CPPSRC))
CPPSRC:=$(filter-out jsoncpp.cpp, $(CPPSRC))

all:    client_cuda client_cpu

cuda_lib:
	make --directory cuda

json_lib:
	${CXX} -c jsoncpp.cpp -o jsoncpp.o -O2 -I./

client_cuda: cuda_lib json_lib
	${CXX} -o client-cuda ${CPPSRC} jsoncpp.o ${INCLUDE} ${LIBS} ${CXXFLAGS} -D_CUDA -I./ -Icuda cuda/cuda.a -L${CUDA_LIB} -I${CUDA_INCLUDE} -lbigint -lutil -lecc -lgmp -llogger -lthread -lpthread -lcudart -lcurl

cpu_lib:
	make --directory cpu

client_cpu: cpu_lib	json_lib fp_test
	${CXX} -o client-cpu ${CPPSRC} jsoncpp.o ${INCLUDE} ${LIBS} ${CXXFLAGS} -D_CPU -I./ -Icpu cpu/cpu.a -lbigint -lutil -lecc -lgmp -llogger -lthread -lpthread -lcurl

fp_test:	cpu_lib
	${CXX} -o fp_test.bin fp_test.cpp ${INCLUDE} ${LIBS} -Icpu/math -Icpu/math/gmp cpu/cpu.a -lbigint -lgmp -lutil

clean:
	rm -f *.o
	rm -f client-cuda
	rm -f client-cpu
	rm -f fp_test.bin
	make --directory cuda clean
	make --directory cpu clean
