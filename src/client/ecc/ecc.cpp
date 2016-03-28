#include <iostream>
#include "ecc.h"
#include "BigInteger.h"

ECCurve::ECCurve()
{
}

ECCurve::ECCurve( ECParams &params )
{
    _p = BigInteger( params.p, 16 );
    _n = BigInteger( params.n, 16 );
    _a = BigInteger( params.a, 16 );
    _b = BigInteger( params.b, 16 );
    _bpx = BigInteger( params.bpx, 16 );
    _bpy = BigInteger( params.bpy, 16 );
}

ECCurve::ECCurve( BigInteger p, BigInteger n, BigInteger a, BigInteger b, BigInteger bpx, BigInteger bpy )
{
    _p = p;
    _n = n;
    _a = a;
    _b = b;
    _bpx = bpx;
    _bpy = bpy;
}

ECPoint ECCurve::add( ECPoint &p, ECPoint &q )
{
    BigInteger rx;
    BigInteger ry;
    BigInteger px = p.getX();
    BigInteger py = p.getY();
    BigInteger qx = q.getX();
    BigInteger qy = q.getY();

    // Px == Qx && Py == Qy
    if( p == q ) {
        return doubl( p );
    }

    // Px == Qx && Py != Qy
    if( px == qx ) {
        return ECPoint();
    }

    if( p.isPointAtInfinity() ) {
        return q;
    }

    if( q.isPointAtInfinity() ) {
        return p;
    }

    // s = (py - qy)/(px - qx)
    BigInteger rise = (py - qy) % _p;
    BigInteger run = (px - qx) % _p;
    BigInteger s = (run.invm(_p) * rise) % _p;

    // rx = s^2 - px - qx
    rx = (s*s - px - qx) % _p;

    // ry = -py + s(px - rx)
    ry = (s * (px - rx) - py) % _p;

    return ECPoint( rx, ry );
}

ECPoint ECCurve::doubl( ECPoint &p )
{
    if( p.isPointAtInfinity() ) {
        return p;
    }

    BigInteger px = p.getX();
    BigInteger py = p.getY();
    BigInteger rx;
    BigInteger ry;

    // 1 / 2py
    BigInteger yInv = (py + py) % _p;
    yInv = yInv.invm(_p);

    // 3 * px^2 + a
    BigInteger s = ((((px * px) * 3) + _a) * yInv) % _p;

    // rx = s^2 - 2px
    rx = (s * s - px - px) % _p;

    // ry = -py + s(px - rx)
    ry = (s * (px - rx) - py) % _p;

    return ECPoint( rx, ry );
}

ECPoint ECCurve::multiply( BigInteger &k, ECPoint &p )
{
    BigInteger m = k;
    ECPointJacobian q = toJacobian( p );

    m = m % _n;

    ECPointJacobian r;

    while( !m.isZero() ) {
        if( m.lsb() ) {
            r = addJacobian( r, q );
        }
        m = m.rshift( 1 );
        q = doubleJacobian( q );
    }

    return toAffine( r );
}

ECPointJacobian ECCurve::toJacobian( ECPoint &p )
{
    return ECPointJacobian( p.getX(), p.getY() );
}

ECPoint ECCurve::toAffine( ECPointJacobian &p )
{
    BigInteger z = p.getZ();

    BigInteger zInv = z.invm( _p );
    BigInteger z2Inv = (zInv * zInv) % _p;
    BigInteger z3Inv = (z2Inv * zInv) % _p;

    BigInteger x = p.getX();
    BigInteger y = p.getY();

    return ECPoint( (x * z2Inv) % _p, (y * z3Inv) % _p );
}

ECPoint ECCurve::getBasepoint()
{
    return ECPoint( _bpx, _bpy );
}

bool ECCurve::pointExists( ECPoint &p )
{
    BigInteger x = p.getX();
    BigInteger y = p.getY();

    BigInteger leftSide = (y * y) % _p;
    BigInteger rightSide = ((x*x*x) + (_a*x) + _b) % _p;

    return leftSide == rightSide;
}

ECPointJacobian ECCurve::doubleJacobian( ECPointJacobian &p )
{
    BigInteger x = p.getX();
    BigInteger y = p.getY();
    BigInteger z = p.getZ();

    if( p.isPointAtInfinity() ) {
        return p;
    }

    // S = 4XY^2
    BigInteger s = ((x * y * y) * 4) % _p;

    // M = 3X^2 + AZ^4
    BigInteger z4 = (z * z) % _p;
    z4 = (z4 * z4) % _p;

    BigInteger m = (((x * x) * 3) + (_a * z4)) % _p;

    // X' = M^2 - 2S
    BigInteger x2 = (m * m - s - s) % _p;

    // Y' = M(S - X') - 8Y^4
    BigInteger y4 = (y * y) % _p;
    y4 = (y4 * y4) % _p;

    BigInteger y2 = ((m * (s - x2)) - (y4 * 8)) % _p;

    // Z' = 2YZ
    BigInteger z2 = ((y * z) * 2) % _p;

    return ECPointJacobian( x2, y2, z2 );
}

ECPointJacobian ECCurve::addJacobian( ECPointJacobian &p1, ECPointJacobian &p2 )
{
    BigInteger x1 = p1.getX();
    BigInteger y1 = p1.getY();
    BigInteger z1 = p1.getZ();

    BigInteger x2 = p2.getX();
    BigInteger y2 = p2.getY();
    BigInteger z2 = p2.getZ();

    if( p1.isPointAtInfinity() ) {
        return p2;
    } else if( p2.isPointAtInfinity() ) {
        return p1;
    }

    // U1 = X1*Z2^2
    BigInteger u1 = (x1 * z2 * z2) % _p;

    // U2 = X2*Z1^2
    BigInteger u2 = (x2 * z1 * z1) % _p;

    // S1 = Y1*Z2^3
    BigInteger s1 = (y1 * z2 * z2 * z2) % _p;

    // S2 = Y2*Z1^3
    BigInteger s2 = (y2 * z1 * z1 *z1) % _p;

    if( u1 == u2 ) {
        if( s1 != s2 ) {
            return ECPointJacobian();
        } else {
            return doubleJacobian( p1 );
        }
    }

    BigInteger h = (u2 - u1) % _p;
    BigInteger h2 = (h * h) % _p;
    BigInteger h3 = (h2 * h) % _p;
    BigInteger r = (s2 - s1) % _p;
    BigInteger t = ((u1 * h2) * 2) % _p;

    // X' = R^2 - H^3 - 2*U1*H^2
    BigInteger newX = ((r * r) - h3 - t) % _p;

    // Y' = R*(U1*H^2 - X') - S1*H^3
    BigInteger newY = ((r * (u1*h2 - newX)) - (s1 * h3)) % _p;

    // Z' = H*Z1*Z2
    BigInteger newZ = (h * z1 * z2) % _p;

    return ECPointJacobian(newX, newY, newZ);
}
