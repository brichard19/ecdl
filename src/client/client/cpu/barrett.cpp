#include <stdio.h>
#include <string.h>
#include "Fp.h"
#include "x86.h"
#include "BigInteger.h"
#include <gmp.h>
#include "gmpxx.h"

// m value for Barrett reduction.
static unsigned long _m[8] = {0};

// Prime modulus
static unsigned long _p[8] = {0};

// 2 x prime modulus (used in Barrett reduction)
static unsigned long _p2[8] = {0};

// Modulus length in bits
static int _pBits;

// Modulus length in words
static int _pLen;

// Length of p^2 in words
static int _pSquaredLength;


static inline void sub(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    #ifdef _X86
        #error "x86 not supported"
        x86_sub160(a, b, diff);
    #else
        mpn_sub_n((long unsigned int *)diff, (const long unsigned int *)a, (const long unsigned int *)b, _pLen);
    #endif
}

static inline void sub2(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    mpn_sub_n((long unsigned int *)diff, (const long unsigned int *)a, (const long unsigned int *)b, _pLen*2);
}

static inline void add(const unsigned long *a, const unsigned long *b, unsigned long *sum)
{
    #ifdef _X86
        x86_add160(a, b, diff);
    #else
        mpn_add_n((long unsigned int *)sum, (const long unsigned int *)a, (const long unsigned int *)b, _pLen);
    #endif
}

static inline void mul(const unsigned long *a, const unsigned long *b, unsigned long *product)
{
    #ifdef _X86
        x86_mul160(a, b, product);
    #else
        mpn_mul_n((long unsigned int *)product, (const long unsigned int *)a, (const long unsigned int *)b, _pLen);
    #endif
}

static inline void mul_low(const unsigned long *a, const unsigned long *b, unsigned long *product)
{
    #ifdef _X86
        x86_mul160(a, b, product);
    #else
        mpn_mul_n((long unsigned int *)product, (const long unsigned int *)a, (const long unsigned int *)b, _pLen);
    #endif
}

static inline void square(const unsigned long *a, unsigned long *product)
{
    #ifdef _X86
        x86_mul160(a, a, product);
    #else
        mpn_sqr((long unsigned int*)product, (long unsigned int *)a, _pLen);
    #endif
}

static bool equalTo(const unsigned long *a, const unsigned long *b, unsigned int len)
{
    for(int i = len - 1; i >= 0; i--) {
        if(a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

static bool greaterThan(const unsigned long *a, const unsigned long *b)
{
    for(int i = _pLen - 1; i >= 0; i--) {
        if(a[i] > b[i]) {
            return true;
        } else if(a[i] < b[i]) {
            return false;
        }
    }

    return false;
}

/**
 * Returns true if a >= b
 */
static bool greaterThanEqualTo(const unsigned long *a, const unsigned long *b, unsigned int len)
{
    for(int i = len - 1; i >= 0; i--) {
        if(a[i] < b[i]) {
            return false;
        } else if(a[i] > b[i]) {
            return true;
        }
    }

    return true;
}

/**
 * Prints integer to screen in hex format
 */
static void printInt(const unsigned long *x, int len)
{
    for(int i = len - 1; i >= 0; i--) {
        printf("%.0lx", x[i]);
    }
    printf("\n");
    
}

/**
 * Performs reduction mod P using the barrett reduction. It is assumed that
 * the product is no greater than (p-1)^2.
 */
static void reduceModP(const unsigned long *x, unsigned long *c)
{
    unsigned long q[10] =  {0};
    unsigned long xm[10] = {0};

    // Get the high bits of x
    unsigned long xHigh[10] = {0};
   
    int rShift = (_pBits) % WORD_LENGTH_BITS;
    int lShift = WORD_LENGTH_BITS - rShift;
    unsigned long mask = ((unsigned long)~0) >> (WORD_LENGTH_BITS - ((_pBits) % WORD_LENGTH_BITS));

    for(int i = 0; i < _pLen; i++) {
        xHigh[ i ] = (x[ _pLen - 1 + i ] >> rShift) | (x[ _pLen + i ] << lShift);
    }

    // Multiply by m
    mul(xHigh, _m, xm);

    // Get the high bits of x * m. 
    for(int i = 0; i < _pLen; i++) {
        q[ i ] = (xm[ _pLen - 1 + i ] >> rShift) | (xm[ _pLen + i ] << lShift);
    }

    // Multiply by p
    unsigned long qp[10] = {0};
    mul(q, _p, qp);

    // Subtract from x
    unsigned long r[10] = {0};
    sub2(x, qp, r);

    // The trick here is that instead of multiplying xm by p, we multiplied only the top
    // half by p. This still works because the lower bits of the product are discarded anyway.
    // But it could have been the case that there was a carry from the multiplication operation on
    // the lower bits, which will result in r being >= 2p because in that case we would be
    // doing x - (q-1) *p instead of x - q*p. So we need to check for >= 2p and >= p. Its more checks
    // but saves us from doing a multiplication.
    if(greaterThanEqualTo(r, _p2, _pLen)) {
        sub2(r, _p2, c);
    } else if(greaterThanEqualTo(r, _p, _pLen)) {
        sub(r, _p, c);
    } else {
        for(int i = 0; i < _pLen; i++) {
            c[i] = r[i];
        }
    }
}

void initFp(BigInteger &p)
{
    BigInteger p2 = p * 2;

    // Precompute _m
    _pBits = p.getBitLength();

    _pLen = p.getWordLength();
    BigInteger k = BigInteger(2).pow(2 * _pBits);
    BigInteger m = k / p;

    // Convert P and M to words
    p.getWords(_p, _pLen);
    p2.getWords(_p2, p2.getWordLength());

    m.getWords(_m, _pLen);
}

void subModP(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    sub(a, b, diff);

    if(diff[_pLen-1] & ((unsigned long)0x01 << (WORD_LENGTH_BITS-1)) ) {
        add(diff, _p, diff);
    }
}

void multiplyModP(const unsigned long *a, const unsigned long *b, unsigned long *c)
{
    unsigned long product[10];

    mul(a, b, product);
    reduceModP(product, c);
}

void squareModP(const unsigned long *a, unsigned long *aSquared)
{
    unsigned long product[10]={0};

    square(a, product);
    reduceModP(product, aSquared);
}

void inverseModP(const unsigned long *input, unsigned long *inverse)
{
    mpz_t a;
    mpz_t aInv;
    mpz_t p;

    mpz_init(a);
    mpz_init(aInv);
    mpz_init(p);

    mpz_import(a, _pLen, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, input);
    mpz_import(p, _pLen, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, _p);

    mpz_invert(aInv, a, p);

    // Need to zero out the destination
    memset(inverse, 0, sizeof(unsigned long) * _pLen);

    mpz_export(inverse, NULL, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, aInv);

    mpz_clear(a);
    mpz_clear(aInv);
    mpz_clear(p);
}