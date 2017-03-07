#ifndef _ECC_H
#define _ECC_H

#include "BigInteger.h"

typedef struct {
    const char *p;
    const char *a;
    const char *b;
    const char *n;
    const char *bpx;
    const char *bpy;
}ECParams;

class ECPoint {

public:

    ECPoint();
    ECPoint(const ECPoint &p);
    ECPoint(const BigInteger &x, const BigInteger &y);

    BigInteger getX() const;
    BigInteger getY() const;

    BigInteger x;
    BigInteger y;

    bool isPointAtInfinity();
    bool operator==(ECPoint &p);
};

class ECPointJacobian {

private:
    BigInteger x;
    BigInteger y;
    BigInteger z;

public:
    ECPointJacobian();
    ECPointJacobian(const ECPointJacobian &p);
    ECPointJacobian(const BigInteger &x, const BigInteger &y);
    ECPointJacobian(const BigInteger &x, const BigInteger &y, const BigInteger &z);

    BigInteger getX();
    BigInteger getY();
    BigInteger getZ();
    bool isPointAtInfinity();
};

class ECCurve {

private:
    BigInteger _a;
    BigInteger _b;
    BigInteger _n;
    BigInteger _p;
    BigInteger _bpx;
    BigInteger _bpy;

public:
    ECCurve();
    ECCurve(ECParams &params);
    ECCurve(BigInteger p, BigInteger n, BigInteger a, BigInteger b, BigInteger bpx, BigInteger bpy);
    ECPoint getBasepoint();

    ECPoint add(ECPoint &p, ECPoint &q);
    ECPoint doubl(ECPoint &p);
    ECPoint multiply(BigInteger &k, ECPoint &p);
    ECPointJacobian toJacobian(ECPoint &p);
    ECPoint toAffine(ECPointJacobian &p);
    ECPointJacobian addJacobian(ECPointJacobian &p, ECPointJacobian &q);
    ECPointJacobian doubleJacobian(ECPointJacobian &p);

    BigInteger a() const { return _a; };
    BigInteger b() const { return _b; };
    BigInteger p() const { return _p; };
    BigInteger n() const { return _n; };

    BigInteger compressPoint(ECPoint &p);
    
    bool pointExists(ECPoint &p);

    ECCurve &operator=(const ECCurve &p);
};

void generateRPoints(ECCurve &curve, ECPoint &q, BigInteger *aAra, BigInteger *bAra, BigInteger *xAra, BigInteger *yAra, int n);
void compressPoint(ECPoint &point, unsigned char *encoded);
bool decompressPoint(const ECCurve &curve, const unsigned char *encoded, int len, ECPoint &out);
int squareRootModP(const BigInteger &n, const BigInteger &p, BigInteger *out);
#endif
