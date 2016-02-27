
SRC=$(wildcard *.cpp)
SRC:=$(filter-out workerthread.cpp, $(SRC))
SRC:=$(filter-out RhoCPU.cpp, $(SRC))

OBJS=$(SRC:.cpp=.o)

ifeq ($(X86_ASM),1)
INCLUDE+=-I./math/x86
else
INCLUDE+=-I./math/gmp
endif

all:	fpmath
	for file in ${SRC} ; do\
	    ${CXX} -c $$file ${INCLUDE} -I../ -I./ -I./math ${CXXFLAGS};\
	done
	ar rvs cpu.a ${OBJS} math/obj/*.o

fpmath:
	make --directory math

clean:
	make --directory math clean
	rm -rf *.o
	rm -rf *.a
