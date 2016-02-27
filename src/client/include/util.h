#ifndef _UTIL_H
#define _UTIL_H

#include"BigInteger.h"

namespace util {

unsigned int getSystemTime();
std::string hexEncode(const unsigned char *bytes, unsigned int len);
void hexDecode(std::string hex, unsigned char *bytes);
void printHex(unsigned long x);
void printHex(unsigned long *x, int len);
void getRandomBytes(unsigned char *buf, unsigned int count);
}

#endif