#ifndef _PRIME_FIELD_H
#define _PRIME_FIELD_H

#include "BigInteger.h"

#ifdef _X86
#include "x86.h"
#else
#include "gmp_math.h"
#endif

// use unsigned long because it's the CPUs natural word length
#define WORD_LENGTH_BITS (sizeof(unsigned long)*8)

// Get the length of byte string in words
#define WORDS(b) ( (b) + (sizeof(unsigned long)-1) ) / sizeof(unsigned long)

void printInt(const unsigned long *x, int len);


class FpBase {
public:
    virtual void subModP(const unsigned long *a, const unsigned long *b, unsigned long *diff) = 0;
    virtual void multiplyModP(const unsigned long *a, const unsigned long *b, unsigned long *c) = 0;
    virtual void squareModP(const unsigned long *a, unsigned long *aSquared) = 0;
    virtual void inverseModP(const unsigned long *input, unsigned long *inverse) = 0;
};

FpBase *getFp(BigInteger &p);

template <int N> 
class Fp : public FpBase {

private:
    // m value for Barrett reduction.
    unsigned long _m[WORDS(N)];

    // Prime modulus
    unsigned long _p[WORDS(N)];

    // Modulus length in bits
    int _pBits;

    // Modulus length in words
    int _pWords;

    // Length of m in words
    int _mWords;

    // Length of m in bits
    int _mBits;

    // P in GMP format because GMP does the modular inversion
    mpz_t _gmp_p;

    void getHighBits(const unsigned long *in, unsigned long *out);
    void reduceModP(const unsigned long *x, unsigned long *c);

public:

    Fp() {}

    Fp(const BigInteger &p)
    {
        printf("Initializing Fp for %s %d bytes\n", p.toString().c_str(), N);
        memset(_p, 0, sizeof(_p));
        memset(_m, 0, sizeof(_m));
        _pBits = p.getBitLength();
        _pWords = p.getWordLength();
        p.getWords(_p, _pWords);

        // Precompute m = 4^n / p
        BigInteger k = BigInteger(2).pow(2 * _pBits);
        BigInteger m = k / p;

        _mWords = m.getWordLength();
        _mBits = m.getBitLength();
        m.getWords(_m, _mWords);

        mpz_init(_gmp_p);

        mpz_import(_gmp_p, _pWords, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, _p);
    }


    void subModP(const unsigned long *a, const unsigned long *b, unsigned long *diff);
    void multiplyModP(const unsigned long *a, const unsigned long *b, unsigned long *c);
    void squareModP(const unsigned long *a, unsigned long *aSquared);
    void inverseModP(const unsigned long *input, unsigned long *inverse);
};


/**
 * Given a number 3 * _pBits in length, get the high _pBits bits
 */
template<int N> void Fp<N>::getHighBits(const unsigned long *in, unsigned long *out)
{
    int rShift = (2*_pBits) % WORD_LENGTH_BITS;
    int lShift = WORD_LENGTH_BITS - rShift;
    int index = (2 * _pBits) / WORD_LENGTH_BITS;

    if(rShift > 0) {
        for(int i = 0; i < WORDS(N); i++) {
            out[ i ] = (in[ index + i ] >> rShift) | (in[ index + i + 1 ] << lShift);
        }
    } else {
        for(int i = 0; i < WORDS(N); i++) {
            out[ i ] = in[index + i];
        }
    }
}

/**
 * Performs reduction mod P using the barrett reduction. It is assumed that
 * the product is <= (p-1)^2
 */
template<int N> void Fp<N>::reduceModP(const unsigned long *x, unsigned long *c)
{
    unsigned long xm[WORDS(N) + WORDS(2*N)] = {0};

    // Multiply by m to get a 3k-bit value
    mul<WORDS(N), WORDS(2*N)>(_m, x, xm);

    // It's possible m can be 1 bit longer than P. If P ends on a word boundary then
    // m will be 1 word longer than p, so it's quicker to do an addition on the higher
    // bits of xm than to multiply by 1.
    if(_mWords > WORDS(N)) {
        add<WORDS(2*N)>(x, &xm[WORDS(N)], &xm[WORDS(N)]);
    }

    // Get the high k bits of xm
    unsigned long q[WORDS(N)] = {0};
    getHighBits(xm, q);

    // Multiply by p to get a 2k-bit value
    unsigned long qp[WORDS(2*N)] = {0};
    mul<WORDS(N)>(q, _p, qp);

    // Subtract from x to get a k-bit value
    sub<WORDS(N)>(x, qp, c);

    qp[WORDS(N)-1] &= ((unsigned long)~0 >> (_pBits % WORD_LENGTH_BITS));

    // Subtract again if necessary
    if(greaterThanEqualTo<WORDS(N)>(c, _p)) {
        sub<WORDS(N)>(c, _p, c);
    }
}

/**
 * Subtraction mod P
 */
template<int N> void Fp<N>::subModP(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    int borrow = sub<WORDS(N)>(a, b, diff);

    // Check for negative
    if(borrow) {
        add<WORDS(N)>(diff, _p, diff);
    }
}

/**
 * Multiplication mod P
 */
template<int N> void Fp<N>::multiplyModP(const unsigned long *a, const unsigned long *b, unsigned long *c)
{
    unsigned long product[WORDS(N*2)];
    mul<WORDS(N)>(a, b, product);
    reduceModP(product, c);
}

/**
 * Square mod P
 */
template<int N> void Fp<N>::squareModP(const unsigned long *a, unsigned long *aSquared)
{
    unsigned long product[WORDS(N*2)];

    square<WORDS(N)>(a, product);
    reduceModP(product, aSquared);
}

/**
 * Modular inverse mod P
 */
template<int N> void Fp<N>::inverseModP(const unsigned long *input, unsigned long *inverse)
{
    mpz_t a;
    mpz_t aInv;

    mpz_init(a);
    mpz_init(aInv);

    mpz_import(a, _pWords, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, input);

    mpz_invert(aInv, a, _gmp_p);

    // Need to zero out the destination
    memset(inverse, 0, sizeof(unsigned long) * WORDS(N));

    mpz_export(inverse, NULL, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_LITTLE, 0, aInv);

    mpz_clear(a);
    mpz_clear(aInv);
}
#endif