#ifndef _PRIMEFIELD_H
#define _PRIME_FIELD_H

#include "BigInteger.h"

// use unsigned long because it's the CPUs natural word length
#define WORD_LENGTH_BITS (sizeof(unsigned long)*8)

#define FP_MAX_SIZE (512/WORD_LENGTH_BITS)

typedef unsigned long FpElement[FP_MAX_SIZE];

/**
 * Subtraction mod P
 */
void subModP(const unsigned long *a, const unsigned long *b, unsigned long *diff);

/**
 * Multiplicative inverse mod P
 */
void inverseModP(const unsigned long *a, unsigned long *inverse);

/**
 * Square mod P
 */
void squareModP(const unsigned long *a, unsigned long *aSquared);

/**
 * Multiplication mod P
 */
void multiplyModP(const unsigned long *a, const unsigned long *b, unsigned long *product);

/**
 * Initializes the prime field library
 */
void initFp(BigInteger &p);
void printInt(const unsigned long *x, int len);

#endif