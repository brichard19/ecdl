#ifndef _UINT160_H
#define _UINT160_H

#include "BigInteger.h"

typedef struct {
    unsigned int v[ 5 ];
}uint160;

/**
 * Converts uint160 to BigInteger object
 */
BigInteger toBigInteger(uint160 &n);

/**
 * Converts a BigInteger object to a uint160
 */
uint160 fromBigInteger(BigInteger &n);

uint160 uint160FromMontgomery(uint160 &x);
void printUint160(uint160 &i);

#endif
