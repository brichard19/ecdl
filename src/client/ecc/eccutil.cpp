#include "ecc.h"
#include "BigInteger.h"

/**
 * Generates a set of R points for performing random walk
 */
void generateRPoints(ECCurve &curve, ECPoint &q, BigInteger *aAra, BigInteger *bAra, BigInteger *xAra, BigInteger *yAra, int n)
{
    ECPoint g = curve.getBasepoint();
    BigInteger order = curve.n();
    BigInteger modulus = curve.p();

    // Generate random Ri = aiG + biQ
    for( int i = 0; i < n; i++ ) {
        ECPoint r1;
        ECPoint r2;
        ECPoint r3;

        // Generate random multiplies
        BigInteger a = randomBigInteger(BigInteger(2), order);
        BigInteger b = randomBigInteger(BigInteger(2), order);

        // Multiply G and Q
        r1 = curve.multiply(a, g);
        r2 = curve.multiply(b, q);

        // Add the two points
        r3 = curve.add(r1, r2);

        xAra[ i ] = r3.getX();
        yAra[ i ] = r3.getY();

        if(aAra != NULL) {
            aAra[ i ] = a;
        }
        
        if(bAra != NULL) {
            bAra[ i ] = b;
        }
    }
}

static BigInteger legendreSymbol(const BigInteger &a, const BigInteger &p)
{
    BigInteger ls = a.pow((p-1)/2, p);

    if(ls == p-1) {
        return -1;
    }

    return ls;
}


int squareRootModP(const BigInteger &n, const BigInteger &p, BigInteger *out)
{
    if(n == 0) {
        return 0;
    }

    if(p == 2) {
        *out = n;
        return 1;
    }


    BigInteger ls = legendreSymbol(n, p);

    if(ls != 1) {
        return 0;
    } 

    if(n % 4 == 3) {
        BigInteger r = n.pow((p+1)/4, p);
        out[0] = r;
        out[1] = p - r;
        return 2;
    }

    unsigned int s = 0; 
    BigInteger q = p - 1;

    while(q % 2 == 0) {
        s = s + 1;
        q = q / 2;
    }

    BigInteger z = 2;
    while(legendreSymbol(z, p) != -1) {
        z = z + 1;
    }


    BigInteger c = z.pow(q, p);
    BigInteger r = n.pow((q+1)/2, p); 
    BigInteger t = n.pow(q, p);

    unsigned int m = s;
     
    while(t != 1) {

        //Find lowest i where t^(2^i) = 1
        unsigned int i = 1;
        unsigned int e = 1;
        while(e < m) {
            if(t.pow(BigInteger(2).pow(i), p) == 1) {
                break;
            }
            i = i * 2;
        }

        BigInteger b = c.pow(BigInteger(2).pow(m - i - 1), p);
        r = (r * b) % p;
        t = (t * b * b) % p;
        c = (b * b) % p;
        m = i;
    }

    out[0] = r;
    out[1] = p - r;

    return 2;
}

void compressPoint(ECPoint &point, unsigned char *buf)
{
    unsigned char sign = 0x00;
    BigInteger x = point.getX();
    BigInteger y = point.getY();

    if((y & 0x01) == 0x01) {
        sign = 0x01;
    }

    int len = x.getByteLength();
    x.getBytes(buf + 1, len);
    buf[0] = sign;
}

bool decompressPoint(const ECCurve &curve, const unsigned char *encoded, int encodedLen, ECPoint &out)
{
    unsigned char sign = encoded[0];

    BigInteger x(&encoded[1], encodedLen-1);

    BigInteger a = curve.a();
    BigInteger b = curve.b(); 
    BigInteger p = curve.p();

    BigInteger z = (x.pow(3) + a * x + b) % p;

    BigInteger roots[2];

    int numRoots = squareRootModP(z, p, roots);

    if(numRoots != 2) {
        return false;
    }

    // Make sure roots[0] is even, roots[1] is odd
    if((roots[0] & 1) == 0) {
        BigInteger tmp = roots[0];
        roots[0] = roots[1];
        roots[1] = tmp;
    }

    out.x = x;
    if(sign) {
        out.y = roots[0];
    } else {
        out.y = roots[1];
    }

    return true;
}