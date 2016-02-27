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