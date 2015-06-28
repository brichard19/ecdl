#ifndef _BIG_INTEGER_H
#define _BIG_INTEGER_H

#include <string>

#ifdef _WIN32
#include "mpirxx.h"
#else
#include "gmpxx.h"
#endif

#define GMP_BYTE_ORDER_MSB 1
#define GMP_BYTE_ORDER_LSB -1

#define GMP_ENDIAN_BIG 1
#define GMP_ENDIAN_LITTLE -1
#define GMP_ENDIAN_NATIVE 0

class BigInteger {

private:
    mpz_class e;

public:
    BigInteger();
    BigInteger( int i );
    BigInteger( const BigInteger &i );
    BigInteger( std::string s, int base = 0);
    BigInteger( const unsigned char *bytes, size_t len );
    BigInteger( const unsigned long *words, size_t len );

    std::string toString(int base = 10);

    BigInteger pow( unsigned int exponent );
    BigInteger powm( const BigInteger &exponent, const BigInteger &modulus );
    BigInteger powm( unsigned int exponent, const BigInteger &modulus );
    BigInteger invm( const BigInteger &modulus );

    int lsb();
    BigInteger rshift( int n );
    bool isZero();
    bool operator==( const BigInteger &i ) const;
    bool operator!=( const BigInteger &i ) const;
    size_t getBitLength();
    size_t getByteLength();
    size_t getWordLength();
    void getWords( unsigned long *words, size_t size );
    void getBytes( unsigned char *bytes, size_t size );
    bool equals( BigInteger &i );

    BigInteger operator-(const BigInteger &i);
    BigInteger operator+(const BigInteger &i);
    BigInteger operator%(const BigInteger &i);
    BigInteger operator*(const BigInteger &i);
    BigInteger operator*(int &i);
    BigInteger operator/(const BigInteger &i);
    BigInteger operator+=(const BigInteger &a);
};

BigInteger randomBigInteger( BigInteger &max );

#endif
