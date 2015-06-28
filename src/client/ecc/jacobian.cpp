#include "BigInteger.h"
#include "ecc.h"

ECPointJacobian::ECPointJacobian()
{
}

ECPointJacobian::ECPointJacobian( const BigInteger &x, const BigInteger &y )
{
    this->x = x;
    this->y = y;
    this->z = BigInteger( 1 );
}

ECPointJacobian::ECPointJacobian( const BigInteger &x, const BigInteger &y, const BigInteger &z )
{
    this->x = x;
    this->y = y;
    this->z = z;
}

ECPointJacobian::ECPointJacobian( const ECPointJacobian &p )
{
    this->x = p.x;
    this->y = p.y;
    this->z = p.z;
}

BigInteger ECPointJacobian::getX()
{
    return this->x;
}

BigInteger ECPointJacobian::getY()
{
    return this->y;
}

BigInteger ECPointJacobian::getZ()
{
    return this->z;
}

bool ECPointJacobian::isPointAtInfinity()
{
    if( this->x.isZero() && this->y.isZero() ) {
        return true;
    } else {
        return false;
    }
}
