
SRC=$(wildcard *.cpp)
OBJS=$(SRC:.cpp=.o)
TARGETS=
ifeq ($(X86_ASM),1)
OBJS+=x86/x86.o
INCLUDE+=-I./x86
CXXFLAGS+=-D_X86
TARGETS=x86asm
endif

all:	${TARGETS}
	for file in ${SRC} ; do\
	    ${CXX} -c $$file ${INCLUDE} -I../ -I./ ${CXXFLAGS};\
	done
	ar rvs cpu.a ${OBJS}

x86asm:
	make --directory x86 all
	
clean:
	make --directory x86 clean
	rm -rf *.o
	rm -rf *.a
