#include <stdio.h>
#include <string.h>
#include "Fp.h"
#include "BigInteger.h"
#include <gmp.h>
#include "gmpxx.h"

#ifdef _X86
#include "x86.h"
#endif

// m value for Barrett reduction.
static FpElement _m = {0};

// Prime modulus
static FpElement _p = {0};

// 2 x prime modulus (used in Barrett reduction)
static FpElement _p2 = {0};

// Modulus length in bits
static int _pBits;

// Modulus length in words
static int _pWords;

// Length of m in words
static int _mWords;

static int gmp_sub(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    return mpn_sub_n((long unsigned int *)diff, (const long unsigned int *)a, (const long unsigned int *)b, _pWords);
}

static int gmp_sub2(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    return mpn_sub_n((long unsigned int *)diff, (const long unsigned int *)a, (const long unsigned int *)b, _pWords + 1);
}

static void gmp_add(const unsigned long *a, const unsigned long *b, unsigned long *sum)
{
    mpn_add_n((long unsigned int *)sum, (const long unsigned int *)a, (const long unsigned int *)b, _pWords);
}

static void gmp_mul(const unsigned long *a, const unsigned long *b, unsigned long *product)
{
    mpn_mul_n((long unsigned int *)product, (const long unsigned int *)a, (const long unsigned int *)b, _pWords);
}

static void gmp_square(const unsigned long *a, unsigned long *product)
{
    mpn_sqr((long unsigned int*)product, (long unsigned int *)a, _pWords);
}

static int (*sub)(const unsigned long *, const unsigned long *, unsigned long*) = gmp_sub;
static int (*sub2)(const unsigned long *, const unsigned long *, unsigned long*) = gmp_sub2;
static void (*add)(const unsigned long *, const unsigned long *, unsigned long*) = gmp_add;
static void (*mul)(const unsigned long *, const unsigned long *, unsigned long*) = gmp_mul;
static void (*square)(const unsigned long *, unsigned long*) = gmp_square;

void printInt(const unsigned long *x, int len)
{
    for(int i = len - 1; i >= 0; i--) {
        printf("%.16lx", x[i]);
    }
    printf("\n");
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

static void rightShift(const unsigned long *in, unsigned long *out)
{
    int rShift = (_pBits) % WORD_LENGTH_BITS;
    int lShift = WORD_LENGTH_BITS - rShift;

    if(rShift > 0) {
        for(int i = 0; i < _pWords; i++) {
            out[ i ] = (in[ _pWords - 1 + i ] >> rShift) | (in[ _pWords + i ] << lShift);
        }
    } else {
        for(int i = 0; i < _pWords; i++) {
            out[ i ] = in[_pWords + i];
        }
    }
}

/**
 * Performs reduction mod P using the barrett reduction. It is assumed that
 * the product is no greater than (p-1)^2
 */
static void reduceModP(const unsigned long *x, unsigned long *c)
{
    FpElement q;
    FpElement xm;

    // Get the high bits of x
    FpElement xHigh;

    rightShift(x, xHigh);

    // Multiply by m
    mul(xHigh, _m, xm);

    // Get the high bits of xHigh * m. 
    rightShift(xm, q);

    // It is possible that m is 1 bit longer than p. If p ends on a word boundry then m will
    // be 1 word longer than p. To avoid doing an extra multiplication when doing mHigh * m
    // (because the 1 would be in a separate word), add xHigh to the result after shifting
    if(_mWords > _pWords) {
        FpElement tmp;
        add(q, xHigh, tmp);
        memcpy(q, tmp, sizeof(FpElement));
    }

    // Multiply by p
    FpElement qp;
    mul(q, _p, qp);

    // Subtract from x
    FpElement r;
    sub2(x, qp, r);

    // The trick here is that instead of multiplying xm by p, we multiplied only the top
    // half by p. This still works because the lower bits of the product are discarded anyway.
    // But it could have been the case that there was a carry from the multiplication operation on
    // the lower bits, which will result in r being >= 2p because in that case we would be
    // doing x - (q-1) *p instead of x - q*p. So we need to check for >= 2p and >= p. Its more checks
    // but saves us from doing a multiplication.
    if(greaterThanEqualTo(r, _p2, _pWords+1)) {
        sub2(r, _p2, c);
    } else if(greaterThanEqualTo(r, _p, _pWords+1)) {
        sub(r, _p, c);
    } else {
        memcpy(c, r, sizeof(unsigned long)*_pWords);
    }
}

void initFp(BigInteger &p)
{
    BigInteger p2 = p * 2;

    // Precompute _m
    _pBits = p.getBitLength();
    _pWords = p.getWordLength();

    // k = 4^n
    BigInteger k = BigInteger(2).pow(2 * _pBits);
    BigInteger m = k / p;

    // Convert P and M to words
    p.getWords(_p, _pWords);
    p2.getWords(_p2, p2.getWordLength());

    _mWords = m.getWordLength();

    m.getWords(_m, _mWords);

#ifdef _X86
    if(_pBits <= 64) {
        add = x86_add64;
        sub = x86_sub64;
        sub2 = x86_sub96;
        mul = x86_mul64;
        square = x86_square64;
    } else if(_pBits <= 96) {
        add = x86_add96;
        sub = x86_sub96;
        sub2 = x86_sub128;
        mul = x86_mul96;
        square = x86_square96;
    } else if(_pBits <= 128) {
        add = x86_add128;
        sub = x86_sub128;
        sub2 = x86_sub160;
        mul = x86_mul128;
        square = x86_square128;
    } else if(_pBits <= 160) {
        add = x86_add160;
        sub = x86_sub160;
        sub2 = x86_sub192;
        mul = x86_mul160;
        square = x86_square160;
    } else if(_pBits <= 192) {
        add = x86_add192;
        sub = x86_sub192;
        sub2 = x86_sub224;
        mul = x86_mul192;
        square = x86_square192;
    } else if(_pBits <= 224) {
        add = x86_add224;
        sub = x86_sub224;
        sub2 = x86_sub256;
        mul = x86_mul224;
        square = x86_square224;
    }
#endif

}

void subModP(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    int borrow = sub(a, b, diff);

    // Check for negative
    if(borrow) {
        add(diff, _p, diff);
    }
}

void multiplyModP(const unsigned long *a, const unsigned long *b, unsigned long *c)
{
    FpElement product;

    mul(a, b, product);
    reduceModP(product, c);
}

void squareModP(const unsigned long *a, unsigned long *aSquared)
{
    FpElement product;

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

    mpz_import(a, _pWords, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, input);
    mpz_import(p, _pWords, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, _p);

    mpz_invert(aInv, a, p);

    // Need to zero out the destination
    memset(inverse, 0, sizeof(unsigned long) * _pWords);

    mpz_export(inverse, NULL, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, aInv);

    mpz_clear(a);
    mpz_clear(aInv);
    mpz_clear(p);
}