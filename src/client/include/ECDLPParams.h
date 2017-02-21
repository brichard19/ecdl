#ifndef _ECDL_PARAMS_H
#define _ECDL_PARAMS_H

#define NUM_R_POINTS 32

#include <vector>
#include "BigInteger.h"

typedef struct {
    BigInteger p;
    BigInteger a;
    BigInteger b;
    BigInteger n;
    BigInteger gx;
    BigInteger gy;
    BigInteger qx;
    BigInteger qy;
    unsigned int dBits;
    std::vector<BigInteger> rx;
    std::vector<BigInteger> ry;
}ECDLPParams;

#endif