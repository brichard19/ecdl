#ifndef _UTIL_H
#define _UTIL_H

#include"BigInteger.h"

namespace util {

class Timer {

private:
    unsigned int _startTime;

public:
    Timer();
    void start();
    unsigned int getTime();
};

unsigned int getSystemTime();
std::string hexEncode(const unsigned char *bytes, unsigned int len);
void hexDecode(std::string hex, unsigned char *bytes);
void printHex(unsigned long x);
void printHex(unsigned long *x, int len);
void getRandomBytes(unsigned char *buf, unsigned int count);
unsigned int toInt(const unsigned char *bytes);
void fromInt(unsigned int x, unsigned char *bytes);

}

#endif