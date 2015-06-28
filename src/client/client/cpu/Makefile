
SRC=$(wildcard *.cpp)
SRC:=$(filter-out fp_test.cpp, $(SRC))
OBJS=$(SRC:.cpp=.o)

all:
	#make --directory x86 all
	for file in ${SRC} ; do\
	    ${CXX} -c $$file ${INCLUDE} -I../ -I./ -I./x86 ${CXXFLAGS};\
    done
	ar rvs cpu.a ${OBJS}

clean:
	make --directory x86 clean
	rm -rf *.o
	rm -rf *.a
