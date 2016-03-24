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
    int pLen = p.getByteLength();

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
        case 9:
        return new Fp<9>(p);
        case 10:
        return new Fp<10>(p);
        case 11:
        return new Fp<11>(p);
        case 12:
        return new Fp<12>(p);
        case 13:
        return new Fp<13>(p);
        case 14:
        return new Fp<14>(p);
        case 15:
        return new Fp<15>(p);
        case 16:
        return new Fp<16>(p);
        case 17:
        return new Fp<17>(p);
        case 18:
        return new Fp<18>(p);
        case 19:
        return new Fp<19>(p);
        case 20:
        return new Fp<20>(p);
        case 21:
        return new Fp<21>(p);
        case 22:
        return new Fp<22>(p);
        case 23:
        return new Fp<23>(p);
        case 24:
        return new Fp<24>(p);
        case 25:
        return new Fp<25>(p);
        case 26:
        return new Fp<26>(p);
        case 27:
        return new Fp<27>(p);
        case 28:
        return new Fp<28>(p);
        case 29:
        return new Fp<29>(p);
        case 30:
        return new Fp<30>(p);
        case 31:
        return new Fp<31>(p);
        case 32:
        return new Fp<32>(p);
        case 33:
        return new Fp<33>(p);
        case 34:
        return new Fp<34>(p);
        case 35:
        return new Fp<35>(p);
        case 36:
        return new Fp<36>(p);
        case 37:
        return new Fp<37>(p);
        case 38:
        return new Fp<38>(p);
        case 39:
        return new Fp<39>(p);
        case 40:
        return new Fp<40>(p);
        case 41:
        return new Fp<41>(p);
        case 42:
        return new Fp<42>(p);
        case 43:
        return new Fp<43>(p);
        case 44:
        return new Fp<44>(p);
        case 45:
        return new Fp<45>(p);
        case 46:
        return new Fp<46>(p);
        case 47:
        return new Fp<47>(p);
        case 48:
        return new Fp<48>(p);
        case 49:
        return new Fp<49>(p);
        case 50:
        return new Fp<50>(p);
        case 51:
        return new Fp<51>(p);
        case 52:
        return new Fp<52>(p);
        case 53:
        return new Fp<53>(p);
        case 54:
        return new Fp<54>(p);
        case 55:
        return new Fp<55>(p);
        case 56:
        return new Fp<56>(p);
        case 57:
        return new Fp<57>(p);
        case 58:
        return new Fp<58>(p);
        case 59:
        return new Fp<59>(p);
        case 60:
        return new Fp<60>(p);
        case 61:
        return new Fp<61>(p);
        case 62:
        return new Fp<62>(p);
        case 63:
        return new Fp<63>(p);
        case 64:
        return new Fp<64>(p);
    }

    throw "Compile for larger integers";
}