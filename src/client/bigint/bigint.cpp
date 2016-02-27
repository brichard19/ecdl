#include <stdlib.h>
#include "BigInteger.h"
#include "util.h"

#ifdef _WIN32
#include "mpirxx.h"
#else
#include "gmpxx.h"
#endif

BigInteger::BigInteger()
{
    mpz_set_si(e.get_mpz_t(), 0);
}

BigInteger::~BigInteger()
{
}

BigInteger::BigInteger(int i)
{
    mpz_set_si(this->e.get_mpz_t(), i);
}

BigInteger::BigInteger(std::string s, int base)
{
    // Try to detect the base
    if(base == 0) {

        // Check for hex
        if(s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            base = 16;
            s = s.substr(2);
        } else {
            for(unsigned int i = 0; i < s.length(); i++) {
                char c = s[i];
                if((c >= 'A' && c <= 'F') || (c >= 'a' && c<= 'f')) {
                    base = 16;
                    break;
                }
            }
        }

        if(base == 0) {
            base = 10;
        }
    }

    if(mpz_set_str(this->e.get_mpz_t(), s.c_str(), base)){
        throw std::string("Error converting string to BigInteger");
    }
}

BigInteger::BigInteger(const BigInteger &i)
{
    this->e = i.e;
}

BigInteger::BigInteger(const unsigned char *bytes, size_t len)
{
    // Import the bytes interprated as an integer with least significant bytes first
    mpz_import(this->e.get_mpz_t(), len, GMP_BYTE_ORDER_LSB, 1, GMP_ENDIAN_NATIVE, 0, bytes);
}

BigInteger::BigInteger(const unsigned long *words, size_t len)
{
    mpz_import(this->e.get_mpz_t(), len, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_NATIVE, 0, words);
}

BigInteger::BigInteger(const unsigned int *words, size_t len)
{
    mpz_import(this->e.get_mpz_t(), len, GMP_BYTE_ORDER_LSB, sizeof(unsigned int), GMP_ENDIAN_NATIVE, 0, words);
}

BigInteger BigInteger::pow(unsigned int exponent)
{
    BigInteger product;

    mpz_pow_ui(product.e.get_mpz_t(), this->e.get_mpz_t(), exponent);

    return product;
}

BigInteger BigInteger::pow(const BigInteger &exponent, const BigInteger &modulus)
{
    BigInteger product;

    mpz_powm( product.e.get_mpz_t(), this->e.get_mpz_t(), exponent.e.get_mpz_t(), modulus.e.get_mpz_t() );

    return product;
}

BigInteger BigInteger::pow(unsigned int exponent, const BigInteger &modulus)
{
    BigInteger product;

    mpz_powm_ui( product.e.get_mpz_t(), this->e.get_mpz_t(), exponent, modulus.e.get_mpz_t() );

    return product;
}

BigInteger BigInteger::invm(const BigInteger &modulus)
{
    BigInteger inverse;

    mpz_invert( inverse.e.get_mpz_t(), this->e.get_mpz_t(), modulus.e.get_mpz_t() );

    return inverse;
}

BigInteger BigInteger::rshift(int n)
{
    BigInteger tmp;

    tmp.e = this->e;
    tmp.e >>= n;

    return tmp;
}

int BigInteger::lsb()
{
    return mpz_tstbit( this->e.get_mpz_t(), 0 );
}

bool BigInteger::isZero()
{
    int zero = mpz_cmp_ui( this->e.get_mpz_t(), 0 );

    if( zero == 0 ) {
        return true;
    } else {
        return false;
    }
}

/*
BigInteger BigInteger::operator=(const BigInteger &i)
{
    this->e = i.e;
}
*/

bool BigInteger::operator==(const BigInteger &i) const
{
    int r = mpz_cmp( this->e.get_mpz_t(), i.e.get_mpz_t() );

    if( r == 0 ) {
        return true;
    }

    return false;
}

bool BigInteger::operator!=(const BigInteger &i) const
{
    int r = mpz_cmp( this->e.get_mpz_t(), i.e.get_mpz_t() );

    if( r != 0 ) {
        return true;
    }

    return false;
}

BigInteger BigInteger::operator%(const BigInteger &m)
{
    BigInteger mod;
    mod.e = this->e % m.e;

    if(mod.e < 0) {
        mod.e = mod.e + m.e;
    }

    return mod;
}

BigInteger BigInteger::operator-(const BigInteger &a) const
{
    BigInteger diff;

    diff.e = this->e - a.e;
    return diff;
}

BigInteger BigInteger::operator+(const BigInteger &a) const
{
    BigInteger sum;
    sum.e = this->e + a.e;

    return sum;
}

BigInteger BigInteger::operator+=(const BigInteger &a)
{
    BigInteger sum;

    sum.e = this->e + a.e;

    return sum;
}

BigInteger BigInteger::operator*(const BigInteger &a)
{
    BigInteger product;
    product.e = this->e * a.e;

    return product;
}

BigInteger BigInteger::operator*(int &i)
{
    BigInteger product;
    product.e = this->e * i;

    return product;
}

BigInteger BigInteger::operator/(const BigInteger &i)
{
    BigInteger quotient;
    quotient.e = this->e / i.e;

    return quotient;
}

std::string BigInteger::toString(int base) const
{
    char *ptr = mpz_get_str(NULL, base, this->e.get_mpz_t());
    std::string s(ptr);
    free( ptr );

    return s;
}

size_t BigInteger::getBitLength()
{
    size_t bits = mpz_sizeinbase( this->e.get_mpz_t(), 2 );

    return bits;
}

size_t BigInteger::getByteLength()
{
    size_t bits = mpz_sizeinbase( this->e.get_mpz_t(), 2 );

    return (bits + 7) / 8;
}

size_t BigInteger::getWordLength()
{
    size_t bits = mpz_sizeinbase( this->e.get_mpz_t(), 2 );
    int wordSize = sizeof(unsigned long)*8;

    return (bits + wordSize - 1) / wordSize;
}

size_t BigInteger::getLengthNative()
{
    size_t bits = mpz_sizeinbase( this->e.get_mpz_t(), 2 );
    int wordSize = sizeof(unsigned long)*8;

    return (bits + wordSize - 1) / wordSize;
}

size_t BigInteger::getLength32()
{
    size_t bits = mpz_sizeinbase( this->e.get_mpz_t(), 2 );

    return (bits + 31) / 32;
}

size_t BigInteger::getLength64()
{
    size_t bits = mpz_sizeinbase( this->e.get_mpz_t(), 2 );

    return (bits + 63) / 64;
}

void BigInteger::getWords(unsigned long *words, size_t size)
{
    memset( words, 0, size * sizeof(unsigned long) );
    mpz_export( words, NULL, GMP_BYTE_ORDER_LSB, sizeof(unsigned long), GMP_ENDIAN_NATIVE, 0, this->e.get_mpz_t() );
}


void BigInteger::getWords(unsigned int *words, size_t size)
{
    memset( words, 0, size * sizeof(unsigned int) );
    mpz_export( words, NULL, GMP_BYTE_ORDER_LSB, sizeof(unsigned int), GMP_ENDIAN_NATIVE, 0, this->e.get_mpz_t() );
}

void BigInteger::getWords(unsigned int *words)
{
    mpz_export( words, NULL, GMP_BYTE_ORDER_LSB, sizeof(unsigned int), GMP_ENDIAN_NATIVE, 0, this->e.get_mpz_t() );
}


void BigInteger::getBytes(unsigned char *bytes, size_t size)
{
    memset( bytes, 0, size );
    mpz_export( bytes, NULL, GMP_BYTE_ORDER_LSB, 1, GMP_ENDIAN_NATIVE, 0, this->e.get_mpz_t() );
}

bool BigInteger::equals(BigInteger &i)
{
    int r = mpz_cmp( this->e.get_mpz_t(), i.e.get_mpz_t() );

    if( r == 0 ) {
        return true;
    } else {
        return false;
    }
}

BigInteger randomBigInteger(const BigInteger &min, const BigInteger &max)
{
    BigInteger range = max - min;

    unsigned int len = range.getByteLength();

    unsigned char bytes[len];

    util::getRandomBytes(bytes, len);

    BigInteger x( bytes, len );
     
    x = x % range;
    BigInteger value = min + x;

    return value;
}
