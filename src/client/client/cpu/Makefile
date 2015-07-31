
SRC=$(wildcard *.cpp)

OBJS=$(SRC:.cpp=.o)

all:	fpmath
	for file in ${SRC} ; do\
	    ${CXX} -c $$file ${INCLUDE} -I../ -I./ ${CXXFLAGS};\
	done
	ar rvs cpu.a ${OBJS}

fpmath:
	make --directory math

clean:
	make --directory math clean
	rm -rf *.o
	rm -rf *.a
