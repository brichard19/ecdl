#include "Fp.h"

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

int main(int argc, char **argv)
{
    for(int i = 0; i < 100000000; i++) {
        //BigInteger a("1c0d76b18c305fbd771", 16);
        //BigInteger b("1c0d76b18c305fbd771", 16);
        BigInteger p("8369143598070916186121", 10);
        BigInteger a = randomBigInteger(2, p);
        //BigInteger b = randomBigInteger(2, p);
        BigInteger b = a;

        /*
        printf("%s\n", a.toString().c_str());
        printf("%s\n", b.toString().c_str());
        printf("%s\n", p.toString().c_str());
        */

        unsigned long x[2] = {0};
        unsigned long y[2] = {0};
        unsigned long z[2] = {0};


        a.getWords(x, 2);
        b.getWords(y, 2);

        Fp<2> fp(p);

        //fp.multiplyModP(x, y, z);
        fp.squareModP(x, z);

        BigInteger zBig(z, 2);
        BigInteger product = (a * b) % p;

        if(zBig != product) {
            printf("%s\n", a.toString().c_str());
            printf("%s\n", b.toString().c_str());
            printf("%s\n", p.toString().c_str());
            printf("%s\n", zBig.toString().c_str());
            printf("%s\n", product.toString().c_str());
            printf("Error!\n");
            break;
        }

    }
    return 0;
}