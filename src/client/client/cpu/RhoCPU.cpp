#include "RhoCPU.h"

static void copyWords(const unsigned long *src, unsigned long *dest, int n)
{
    memcpy(dest, src, sizeof(unsigned long) * n);
}

/**
 * Generates a random point on the curve
 */
void RhoCPU::generateStartingPoint(BigInteger &x, BigInteger &y, BigInteger &a, BigInteger &b)
{
    unsigned long buf[FP_MAX];

    do {
        // 1 < a,b < n
        a = randomBigInteger(2, _params.n);
        b = randomBigInteger(2, _params.n);

        // aG, bQ, aG + bQ
        ECPoint p1 = _curve.multiplyPoint(a, _g);
        ECPoint p2 = _curve.multiplyPoint(b, _q);
        ECPoint p3 = _curve.addPoint(p1, p2);

        x = p3.getX();
        y = p3.getY();

        // Check that we don't start on a distinguished point
        x.getWords(buf, _pLen);
    }while((buf[0] & _dBitsMask) == 0);

}

bool RhoCPU::checkDistinguishedBits(const unsigned long *x)
{
    if((x[ 0 ] & _dBitsMask) == 0) {
        return true;
    } else {
        return false;
    }
}

RhoCPU::RhoCPU(const ECDLPParams *params,
                        const BigInteger *rx,
                        const BigInteger *ry,
                        int numRPoints,
                        int pointsInParallel,
                        void (*callback)(struct CallbackParameters *)
                        )
{
    // Copy parameters
    _params = *params;
    /*
    typedef struct {
    BigInteger p;
    BigInteger a;
    BigInteger b;
    BigInteger n;
    BigInteger gx;
    BigInteger gy;
    BigInteger qx;
    BigInteger qy;
    unsigned int dBits;
}ECDLPParams;
*/
    _fp = getFp(_params.p);
    _pLen = _params.p.getWordLength();

    // Create curve
    _curve = ECCurve(_params.p, _params.n, _params.a, _params.b, _params.gx, _params.gy);

    // Create points
    _g = ECPoint(_params.gx, _params.gy);
    _q = ECPoint(_params.qx, _params.qy);

    _pointsInParallel = pointsInParallel;

    // Initialize Barrett code for modulus P
    BigInteger p = _curve.p();
  
    // Pointer to the coefficients of the starting points
    _a = new BigInteger[pointsInParallel];
    _b = new BigInteger[pointsInParallel];

    for(int i = 0; i < pointsInParallel; i++) {
        _a[i] = BigInteger(0);
        _b[i] = BigInteger(0);
    }

    // (x,y) of the current points
    _x = new unsigned long[pointsInParallel * _pLen];
    _y = new unsigned long[pointsInParallel * _pLen];
    _rx = new unsigned long[32 * _pLen];
    _ry = new unsigned long[32 * _pLen];
    _diffBuf = new unsigned long[pointsInParallel * _pLen];
    _chainBuf = new unsigned long[pointsInParallel * _pLen];
    _lengthBuf = new unsigned long long [pointsInParallel];

    // Initialize length to 1 (starting point counts as 1 point)
    memset(_lengthBuf, 0, sizeof(unsigned int) * pointsInParallel);
    for(int i = 0; i < pointsInParallel; i++) {
        _lengthBuf[i] = 1;
    }

    // Copy R points
    for(int i = 0; i < 32; i++) {
        int index = i * _pLen;
        rx[i].getWords(&_rx[index], _pLen);
        ry[i].getWords(&_ry[index], _pLen);
    }

    // Set mask for detecting distinguished points
    _dBitsMask = ~0;
    _dBitsMask >>= WORD_LENGTH_BITS - params->dBits;

    // Mask for selecting R point
    _rPointMask = numRPoints - 1;

    // Gets called when distinguished point is found
    _callback = callback;

    // Generate starting points and exponents
    for(int i = 0; i < _pointsInParallel; i++) {
        BigInteger a;
        BigInteger b;
        BigInteger x;
        BigInteger y;

        generateStartingPoint(x, y, a, b);

        _a[i] = a;
        _b[i] = b;

        x.getWords((unsigned long *)&_x[i * _pLen], _pLen);
        y.getWords((unsigned long *)&_y[i * _pLen], _pLen);
    }
}

RhoCPU::~RhoCPU()
{
    delete[] _x;
    delete[] _y;
    delete[] _rx;
    delete[] _ry;
    delete[] _diffBuf;
    delete[] _chainBuf;
    delete[] _lengthBuf;
}

void RhoCPU::doStepSingle()
{
    unsigned long px[FP_MAX] = {0};
    unsigned long py[FP_MAX] = {0};

    // Copy onto stack
    copyWords(_x, px, _pLen);
    copyWords(_y, py, _pLen);

    int idx = px[0] & _rPointMask;

    unsigned long run[FP_MAX] = {0};
    _fp->subModP(px, &_rx[idx * _pLen], run);

    unsigned long runInv[FP_MAX] = {0};
    _fp->inverseModP(run, runInv);

    // Calculate (Py - Qy)/(Px - Qx)
    unsigned long rise[FP_MAX] = {0};
    _fp->subModP(py, &_ry[idx * _pLen], rise);

    unsigned long s[FP_MAX] = {0};
    
    _fp->multiplyModP(runInv, rise, s);

    // calculate s^2
    unsigned long s2[FP_MAX] = {0};
    _fp->squareModP(s, s2);

    // Rx = s^2 - Px - Qx
    unsigned long newX[FP_MAX] = {0};

    _fp->subModP(s2, px, newX);
    _fp->subModP(newX, &_rx[ idx * _pLen], newX);

    // Ry = s(Px - Rx) - Py
    unsigned long k[FP_MAX] = {0};
    _fp->subModP(px, newX, k);
   
    _fp->multiplyModP(k, s, k);
    unsigned long newY[FP_MAX] = {0};
    _fp->subModP(k, py, newY);

    // Increment walk length
    (*_lengthBuf)++;

    bool isDistinguishedPoint = checkDistinguishedBits(newX);

    bool isFruitlessCycle = false;

    if ( *_lengthBuf >= (unsigned long long)1 << (_params.dBits + 2)) {
        isFruitlessCycle = true;
    }
    
    //Check for distinguished point
    if(isDistinguishedPoint || isFruitlessCycle) {
        
        if(isDistinguishedPoint) {
            printf("Found distinguished point!\n");
            // Call callback function
            if(_callback != NULL) {
                struct CallbackParameters cp;

                cp.aStart = *_a;
                cp.bStart = *_b;
                cp.x = BigInteger(newX, _pLen);
                cp.y = BigInteger(newY, _pLen);
                cp.length = *_lengthBuf;

                _callback(&cp);
            }
        } else {
            printf("Possible cycle found (%lld iterations), rejecting\n", *_lengthBuf);
        }

        // Generate new starting point
        BigInteger aNew;
        BigInteger bNew;
        BigInteger xNew;
        BigInteger yNew;

        // Generate new starting point
        generateStartingPoint(xNew, yNew, aNew, bNew);

        // Copy new point to memory
        xNew.getWords(_x, _pLen);
        yNew.getWords(_y, _pLen);
        *_a = aNew;
        *_b = bNew;

        *_lengthBuf = 1;

    } else {
        // Write result to memory
        copyWords(newX, _x, _pLen);
        copyWords(newY, _y, _pLen);
    }
}


void RhoCPU::doStepMulti()
{
    unsigned long *chainBuf = _chainBuf;
    unsigned long *diffBuf = _diffBuf;
    unsigned long long *lengthBuf = _lengthBuf;

    // Initialize to 1
    unsigned long product[FP_MAX] = {0};
    product[0] = 1;

    for(unsigned int i = 0; i < _pointsInParallel; i++) {
        unsigned int index = i * _pLen;

        unsigned int idx = _x[ index ] & _rPointMask;
        unsigned long diff[FP_MAX];

        _fp->subModP(&_x[index], &_rx[idx * _pLen], diff);
        copyWords(diff, &diffBuf[ index ], _pLen);
        
        _fp->multiplyModP(product, diff, product);
        copyWords(product, &chainBuf[ index ], _pLen);
    }
    

    unsigned long inverse[FP_MAX];
    _fp->inverseModP(product, inverse);

    // Extract inverse of the differences
    for(int i = _pointsInParallel - 1; i >= 0; i--) {
        int index = i * _pLen;

        // Get the inverse of the last difference by multiplying the inverse
        // of the product of all the differences with the product of all but
        // the last difference
        unsigned long invDiff[FP_MAX];
        
        if(i >= 1) {
            _fp->multiplyModP(inverse, &chainBuf[(i - 1) * _pLen], invDiff);
            _fp->multiplyModP(inverse, &diffBuf[ index ], inverse);
        } else {
            copyWords(inverse, invDiff, _pLen);
        }
        

        unsigned int idx = _x[ index ] & _rPointMask;

        unsigned long px[FP_MAX];
        unsigned long py[FP_MAX];

        // Copy onto stack
        copyWords(&_x[ index ], px, _pLen);
        copyWords(&_y[ index ], py, _pLen);
     
        // Calculate slope (Py - Qy)/(Px - Qx)
        unsigned long rise[FP_MAX];
        _fp->subModP(py, &_ry[ idx * _pLen], rise);
        unsigned long s[FP_MAX];
        _fp->multiplyModP(invDiff, rise, s);

        // calculate s^2
        unsigned long s2[FP_MAX];
        _fp->squareModP(s, s2);

        // Rx = s^2 - Px - Qx
        unsigned long newX[FP_MAX];
        _fp->subModP(s2, px, newX);
        _fp->subModP(newX, &_rx[ idx * _pLen], newX);

        // Ry = s(Px - Rx) - Py
        unsigned long k[FP_MAX];
        _fp->subModP(px, newX, k);
   
        _fp->multiplyModP(k, s, k);
        unsigned long newY[FP_MAX];
        _fp->subModP(k, py, newY);

        // Increment walk length
        lengthBuf[i]++;

        bool isDistinguishedPoint = checkDistinguishedBits(newX);

        bool isFruitlessCycle = false;

        if ( lengthBuf[i] >= (unsigned long long)1 << (_params.dBits + 2)) {
            isFruitlessCycle = true;
        }
        
        //Check for distinguished point
        if(isDistinguishedPoint || isFruitlessCycle) {
            
            if(isDistinguishedPoint) {

                // Call callback function
                if(_callback != NULL) {
                    struct CallbackParameters cp;
                    cp.aStart = _a[i];
                    cp.bStart = _b[i];
                    cp.x = BigInteger(newX, _pLen);
                    cp.y = BigInteger(newY, _pLen);
                    cp.length = lengthBuf[i];
                    _callback(&cp);
                }
            } else {
                //printf("Possible cycle found (%lld iterations), rejecting\n", lengthBuf[i]);
            }

            // Generate new starting point
            BigInteger aNew;
            BigInteger bNew;
            BigInteger xNew;
            BigInteger yNew;

            // Generate new starting point
            generateStartingPoint(xNew, yNew, aNew, bNew);

            // Copy new point to memory
            xNew.getWords(&_x[index], _pLen);
            yNew.getWords(&_y[index], _pLen);
            _a[i] = aNew;
            _b[i] = bNew;
            lengthBuf[i] = 0;
            
        } else {
            // Write result to memory
            copyWords(newX, &_x[ index ], _pLen);
            copyWords(newY, &_y[ index ], _pLen);
        }
    }
}

void RhoCPU::doStep()
{
    //doStepMulti();
    if(_pointsInParallel > 1) {
        doStepMulti();
    } else {
        doStepSingle();
    }
}