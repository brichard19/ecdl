#include <stdio.h>
#include <string.h>
#include "BigInteger.h"

#include "Fp.h"

/**
 * Prints a big integer in hex format to stdout
 */
void printInt(const unsigned long *x, int len)
{
    for(int i = len - 1; i >= 0; i--) {
#ifdef _X86
        printf("%.8x", x[i]);
#else
        printf("%.16lx", x[i]);
#endif
    }
    printf("\n");
}

FpBase *getFp(BigInteger &p)
{
    int pLen = p.getWordLength();

    switch(pLen) {
        case 1:
        return new Fp<1>(p);
        case 2:
        return new Fp<2>(p);
        case 3:
        return new Fp<3>(p);
        case 4:
        return new Fp<4>(p);
        case 5:
        return new Fp<5>(p);
        case 6:
        return new Fp<6>(p);
        case 7:
        return new Fp<7>(p);
        case 8:
        return new Fp<8>(p);
    }

    throw "Compile for larger integers";
}