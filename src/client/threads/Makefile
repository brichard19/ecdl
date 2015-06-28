
all:    ${SRC}
	${CXX} -c threads.cpp -o threads.o ${INCLUDE} ${CXXFLAGS}
	mkdir -p ${LIBDIR}
	ar rvs ${LIBDIR}/libthread.a threads.o

clean:
	rm -rf *.o
