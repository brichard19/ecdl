#ifndef _UTIL_H
#define _UTIL_H

#include"BigInteger.h"

namespace util {

unsigned int getSystemTime();
BigInteger toMontgomery(const BigInteger n, BigInteger r, BigInteger p);
BigInteger toMontgomery(BigInteger x, BigInteger p);
BigInteger fromMontgomery(BigInteger n, BigInteger rInv, BigInteger p);
BigInteger fromMontgomery(BigInteger x, BigInteger p);
std::string hexEncode(const unsigned char *bytes, unsigned int len);
void hexDecode(std::string hex, unsigned char *bytes);
void printHex(unsigned long x);
void printHex(unsigned long *x, int len);

}
#endif