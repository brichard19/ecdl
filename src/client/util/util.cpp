#include"util.h"
#ifdef _WIN32
    #include<windows.h>
#else
    #include<unistd.h>
    #include<sys/stat.h>
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

Timer::Timer()
{
    _startTime = 0;
}

void Timer::start()
{
    _startTime = getSystemTime();
}

unsigned int Timer::getTime()
{
    return getSystemTime() - _startTime;
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

void getRandomBytes(unsigned char *buf, unsigned int count)
{
    for(unsigned int i = 0; i < count; i++) {
        buf[i] = (unsigned char)rand();
    }
}


unsigned int toInt(const unsigned char *bytes)
{
    unsigned int x = 0;
    x |= bytes[0] << 24;
    x |= bytes[1] << 16;
    x |= bytes[2] << 8;
    x |= bytes[3];

    return x;
}

void fromInt(unsigned int x, unsigned char *bytes)
{
    bytes[0] = (unsigned char)(x >> 24);
    bytes[1] = (unsigned char)(x >> 16);
    bytes[2] = (unsigned char)(x >> 8);
    bytes[3] = (unsigned char)x;
}

}