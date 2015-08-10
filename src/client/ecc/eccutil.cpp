#include "ecc.h"
#include "BigInteger.h"

/**
 * Generates a set of R points for performing random walk
 */
void generateRPoints(ECCurve curve, ECPoint q, BigInteger *aAra, BigInteger *bAra, BigInteger *xAra, BigInteger *yAra, int n)
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
        BigInteger a = randomBigInteger(order);
        BigInteger b = randomBigInteger(order);

        // Multiply G and Q
        r1 = curve.multiplyPoint(a, g);
        r2 = curve.multiplyPoint(b, q);

        // Add the two points
        r3 = curve.addPoint(r1, r2);

        // Convert coordinates to montgomery form
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