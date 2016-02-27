
SRC=$(wildcard *.cpp)
SRC:=$(filter-out Fp_templates.cpp, $(SRC))
OBJS=$(SRC:.cpp=.o)
TARGETS=

ifeq ($(X86_ASM),1)
OBJS+=x86/x86.o
INCLUDE+=-I./x86
TARGETS=x86asm
else
INCLUDE+=-I./gmp
endif

all:	${TARGETS}
	for file in ${SRC} ; do\
	    ${CXX} -c $$file ${INCLUDE} -I../ -I./ ${CXXFLAGS};\
	done
	mkdir -p obj
	cp ${OBJS} obj

x86asm:
	make --directory x86 all
	
clean:
	make --directory x86 clean
	rm -rf obj
	rm -rf *.o
	rm -rf *.a
