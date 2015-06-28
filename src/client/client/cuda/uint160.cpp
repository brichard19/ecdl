#include "util.h"
#include "uint160.h"

/**
  * Converts a uint160 object to a BigInteger object
  */
BigInteger toBigInteger(uint160 &n)
{
    return BigInteger(n.v, 5);
}

/**
 * Converts a BigInteger object to a uint160
 */
uint160 fromBigInteger(BigInteger &n)
{
    uint160 x;
    n.getWords(x.v, 5);

    return x;
}

/**
 * Converts a uint160 value into montgomery form
 */
uint160 uint160FromMontgomery(uint160 &x, BigInteger &rInv, BigInteger &p)
{
    BigInteger e(x.v, 5);
    BigInteger g = util::fromMontgomery(e, rInv, p);
    uint160 r;

    g.getWords(r.v, 5);

    return r;
}

void printUint160(uint160 &i)
{
    printf("%.8x%.8x%.8x%.8x%.8x\n", i.v[4], i.v[3], i.v[2], i.v[1], i.v[0]); 
}