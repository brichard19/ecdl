#include"util.h"
#ifdef _WIN32
    #include<windows.h>
#else
    #include<sys/time.h>
#endif

namespace util {


unsigned int getSystemTime()
{
#ifdef _WIN32
    return GetTickCount();
#else
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec * 1000 + t.tv_usec / 1000;
#endif
}

BigInteger toMontgomery(BigInteger x, BigInteger p)
{
    int rBits = p.getBitLength();
    BigInteger two(2);
    BigInteger r = two.pow(rBits);
     
    return (x * r) % p;
}

BigInteger toMontgomery(BigInteger x, BigInteger r, BigInteger p)
{
    return (x * r) % p;
}

/**
 * Converts a number from montgomery form
 */
BigInteger fromMontgomery(BigInteger x, BigInteger p)
{
    int rBits = p.getBitLength();
    BigInteger two(2);
    BigInteger r = two.pow(rBits);

    BigInteger rInverse = r.invm(p);

    return (x * rInverse) % p;
}

BigInteger fromMontgomery(BigInteger n, BigInteger rInv, BigInteger p)
{
    return (n * rInv) % p;
}

std::string hexEncode(const unsigned char *bytes, unsigned int len)
{
    char buf[3] = {0};
    std::string hex = "";

    for(unsigned int i = 0; i < len; i++) {
        sprintf(buf, "%.2x", (unsigned int)bytes[i]);
        hex += std::string(buf);
    }

    return hex;
}

void hexDecode(std::string hex, unsigned char *bytes)
{
    char hexByte[3] = {0};
    const char *ptr = hex.c_str();
    unsigned int len = hex.length();

    for(unsigned int i = 0; i < len; i+=2) {
        hexByte[0] = ptr[i];
        hexByte[1] = ptr[i+1];
        hexByte[2] = '\0';

        unsigned int value = 0;
        sscanf(hexByte, "%x", &value);
        bytes[i/2] = (unsigned char)value;
    }
}

void printHex(unsigned long x)
{
    printf("%.*lx", (int)sizeof(unsigned long)*2, x);
}

void printHex(unsigned long *x, int len)
{
    for(int i = len - 1; i >= 0; i--) {
        printf("%.*lx", (int)sizeof(unsigned long)*2, x[i]);
    }
}

}