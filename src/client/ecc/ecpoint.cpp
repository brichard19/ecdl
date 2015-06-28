#include "ecc.h"
#include "BigInteger.h"

ECPoint::ECPoint()
{
}

ECPoint::ECPoint( const BigInteger &x, const BigInteger &y )
{
    this->x = x;
    this->y = y;
}

ECPoint::ECPoint( const ECPoint &p )
{
    this->x = p.x;
    this->y = p.y;
}

bool ECPoint::isPointAtInfinity()
{
    return this->x.isZero() && this->y.isZero();
}

BigInteger ECPoint::getX()
{
    return this->x;
}

BigInteger ECPoint::getY()
{
    return this->y;
}

bool ECPoint::operator==( ECPoint &p )
{
    if( this->x == p.x && this->y == p.y ) {
        return true;
    } else {
        return false;
    }
}
