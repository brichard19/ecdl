CPPSRC:=$(wildcard *.cpp)
CPPSRC:=$(filter-out fp_test.cpp, $(CPPSRC))
CPPSRC:=$(filter-out jsoncpp.cpp, $(CPPSRC))
CPPSRC:=$(filter-out ecctest.cpp, $(CPPSRC))
CPPSRC:=$(filter-out tle_test.cpp, $(CPPSRC))

all:    client_cuda client_cpu

cuda_lib:
	make --directory cuda

json_lib:
	${CXX} -c jsoncpp.cpp -o jsoncpp.o -O2 -I./

client_cuda: cuda_lib json_lib
	${CXX} -o client-cuda ${CPPSRC} jsoncpp.o ${INCLUDE} ${LIBS} ${CXXFLAGS} -D_CUDA -I./ -Icuda cuda/cuda.a -L${CUDA_LIB} -I${CUDA_INCLUDE} -lbigint -lutil -lecc -lgmp -llogger -lsha256 -lthread -lpthread -lcudart -lcurl -ltle -lconfigfile

cpu_lib:
	make --directory cpu

client_cpu: cpu_lib	json_lib fp_test
	${CXX} -o client-cpu ${CPPSRC} jsoncpp.o ${INCLUDE} ${LIBS} ${CXXFLAGS} -D_CPU -I./ -Icpu cpu/cpu.a -lbigint -lsha256 -lutil -lecc -lgmp -llogger -lthread -lpthread -lcurl -ltle

fp_test:	cpu_lib
	${CXX} -o fp_test.bin fp_test.cpp ${INCLUDE} ${LIBS} -Icpu/math -Icpu/math/gmp cpu/cpu.a -lbigint -lgmp -lutil

ecc_test:
	${CXX} -o ecc_test ecctest.cpp ${INCLUDE} ${LIBS} ${CXXFLAGS} -I./ -lbigint -lutil -lecc -lgmp

#tle_test:
#	${CXX} -o tle_test tle_test.cpp ${INCLUDE} ${LIBS} ${CXXFLAGS} -I./ -lbigint -lgmp -ltle -lutil

clean:
	rm -f *.o
	rm -f client-cuda
	rm -f client-cpu
	rm -f fp_test.bin
	rm -f ecc_test
	make --directory cuda clean
	make --directory cpu clean
