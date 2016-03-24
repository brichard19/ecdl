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
    BigInteger( const unsigned int *words, size_t len);
    
    ~BigInteger();

    std::string toString(int base = 10) const;

    BigInteger pow( unsigned int exponent ) const;
    BigInteger pow( const BigInteger &exponent, const BigInteger &modulus ) const;
    BigInteger pow( unsigned int exponent, const BigInteger &modulus ) const;
    BigInteger invm( const BigInteger &modulus ) const;

    int lsb() const;
    BigInteger rshift( int n ) const;
    bool isZero() const;
    bool operator==( const BigInteger &i ) const;
    bool operator!=( const BigInteger &i ) const;
    size_t getBitLength() const;
    size_t getByteLength() const;
    size_t getWordLength() const;
    size_t getLength32() const;
    size_t getLength64() const;
    size_t getLengthNative() const;
    void getWords( unsigned long *words, size_t size ) const;
    void getWords( unsigned int *words, size_t size ) const;
    void getWords(unsigned int *words) const;
    void getBytes( unsigned char *bytes, size_t size ) const;
    bool equals( BigInteger &i ) const;

    BigInteger operator-(const BigInteger &i) const;
    BigInteger operator+(const BigInteger &i) const;
    BigInteger operator%(const BigInteger &i) const;
    BigInteger operator*(const BigInteger &i) const;
    BigInteger operator*(int &i) const;
    BigInteger operator/(const BigInteger &i) const;
    BigInteger operator+=(const BigInteger &a) const;
    //BigInteger operator=(const BigInteger &i) const;
};

BigInteger randomBigInteger( const BigInteger &min, const BigInteger &max );

#endif
