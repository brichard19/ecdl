#ifndef _RHO_CPU_H
#define _RHO_CPU_H

#include "ecc.h"
#include "Fp.h"
#include "ECDLContext.h"

class RhoBase {

public:
    virtual void doStep() = 0;
};

template<int N> void copyWords(const unsigned long *src, unsigned long *dest)
{
    memcpy(dest, src, sizeof(unsigned long) * N);
}

template<int N> class RhoCPU : public RhoBase {

private:

    ECPoint _g;
    ECPoint _q;
    ECDLPParams _params;
    ECCurve _curve;

    // Starting G and Q coefficients
    BigInteger *_a;
    BigInteger *_b;

    // Current X and Y coordinates
    unsigned long *_x;
    unsigned long *_y;

    // R points
    unsigned long *_rx;
    unsigned long *_ry;

    // Buffers for simultaneous inversion
    unsigned long *_diffBuf;
    unsigned long *_chainBuf;

    // Length of each walk
    unsigned long long *_lengthBuf;

    unsigned int _numPoints;
    unsigned int _rPointMask;
    unsigned long _dBitsMask;
    //unsigned int _pLen;

    Fp<N> _fp;

    void (*_callback)(struct CallbackParameters *);

    void generateStartingPoint(BigInteger &x, BigInteger &y, BigInteger &a, BigInteger &b);
    bool checkDistinguishedBits(const unsigned long *x);

    void doStepSingle();
    void doStepMulti();

public:
    RhoCPU(ECDLPParams *params,
                    BigInteger *rx,
                    BigInteger *ry,
                    int numRPoints,
                    int numPoints,
                    void (*callback)(struct CallbackParameters *)
                    );

    virtual void doStep();
};

/**
 * Generates a random point on the curve
 */
template <int N> void RhoCPU<N>::generateStartingPoint(BigInteger &x, BigInteger &y, BigInteger &a, BigInteger &b)
{
    unsigned long buf[N];

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
        x.getWords(buf, N);
    }while((buf[0] & _dBitsMask) == 0);

}

template <int N> bool RhoCPU<N>::checkDistinguishedBits(const unsigned long *x)
{
    if((x[ 0 ] & _dBitsMask) == 0) {
        return true;
    } else {
        return false;
    }
}

template <int N> RhoCPU<N>::RhoCPU(ECDLPParams *params,
                        BigInteger *rx,
                        BigInteger *ry,
                        int numRPoints,
                        int numPoints,
                        void (*callback)(struct CallbackParameters *)
                        )
{
    // Copy parameters
    _params = *params;

    _fp = Fp<N>(_params.p);

    // Create curve
    _curve = ECCurve(_params.p, _params.n, _params.a, _params.b, _params.gx, _params.gy);

    // Create points
    _g = ECPoint(_params.gx, _params.gy);
    _q = ECPoint(_params.qx, _params.qy);

    _numPoints = numPoints;

    // Initialize Barrett code for modulus P
    BigInteger p = _curve.p();
  
    // Pointer to the coefficients of the starting points
    _a = new BigInteger[numPoints];
    _b = new BigInteger[numPoints];

    for(int i = 0; i < numPoints; i++) {
        _a[i] = BigInteger(0);
        _b[i] = BigInteger(0);
    }

    // (x,y) of the current points
    _x = new unsigned long[numPoints * N];
    _y = new unsigned long[numPoints * N];
    _rx = new unsigned long[32 * N];
    _ry = new unsigned long[32 * N];
    _diffBuf = new unsigned long[numPoints * N];
    _chainBuf = new unsigned long[numPoints * N];
    _lengthBuf = new unsigned long long [numPoints];

    // Initialize length to 1 (starting point counts as 1 point)
    memset(_lengthBuf, 0, sizeof(unsigned int) * numPoints);
    for(int i = 0; i < numPoints; i++) {
        _lengthBuf[i] = 1;
    }

    // Copy R points
    for(int i = 0; i < 32; i++) {
        int index = i * N;
        rx[i].getWords(&_rx[index], N);
        ry[i].getWords(&_ry[index], N);
    }

    // Set mask for detecting distinguished points
    _dBitsMask = ~0;
    _dBitsMask >>= WORD_LENGTH_BITS - params->dBits;

    // Mask for selecting R point
    _rPointMask = numRPoints - 1;

    // Gets called when distinguished point is found
    _callback = callback;

    for(int i = 0; i < _numPoints; i++) {
        BigInteger a;
        BigInteger b;
        BigInteger x;
        BigInteger y;

        generateStartingPoint(x, y, a, b);

        //a.getWords((unsigned long *)&_a[i *N], N);
        //b.getWords((unsigned long *)&_b[i *N], N);
        _a[i] = a;
        _b[i] = b; 
        x.getWords((unsigned long *)&_x[i *N], N);
        y.getWords((unsigned long *)&_y[i *N], N);
    }
}

template<int N> void RhoCPU<N>::doStepSingle()
{
    /*
    const int N1 = WORDLEN(N);
    const int N2 = WORDLEN2(N);
    const int N3 = WORDLEN3(N);
    */

    unsigned long px[N] = {0};
    unsigned long py[N] = {0};

    // Copy onto stack
    copyWords<N>(_x, px);
    copyWords<N>(_y, py);

    int idx = px[0] & _rPointMask;

    unsigned long run[N] = {0};
    _fp.subModP(px, &_rx[idx * N], run);

    unsigned long runInv[N] = {0};
    _fp.inverseModP(run, runInv);

    // Calculate (Py - Qy)/(Px - Qx)
    unsigned long rise[N] = {0};
    _fp.subModP(py, &_ry[idx * N], rise);

    unsigned long s[N];
    _fp.multiplyModP(runInv, rise, s);

    // calculate s^2
    unsigned long s2[N] = {0};
    _fp.squareModP(s, s2);

    // Rx = s^2 - Px - Qx
    unsigned long newX[N] = {0};

    _fp.subModP(s2, px, newX);
    _fp.subModP(newX, &_rx[ idx * N], newX);

    // Ry = s(Px - Rx) - Py
    unsigned long k[N] = {0};
    _fp.subModP(px, newX, k);
   
    _fp.multiplyModP(k, s, k);
    unsigned long newY[N] = {0};
    _fp.subModP(k, py, newY);

    // Increment walk length
    (*_lengthBuf)++;

    bool isDistinguishedPoint = checkDistinguishedBits(newX);

    bool isFruitlessCycle = false;

    if ( *_lengthBuf >= (unsigned long long)1 << (_params.dBits + 2)) {
        isFruitlessCycle = true;
    }
   
    /* 
    BigInteger xTmp(newX, N);
    BigInteger yTmp(newY, N);
    ECPoint pointTmp(xTmp, yTmp);

    if(!_curve.pointExists(pointTmp)) {
        printf("Error: Invalid point\n");
        printInt(newX, N);
        printInt(newY, N);
        printf("%d\n", *_lengthBuf);
        printf("\n");
        printInt(px, N);
        printInt(py, N);
        printInt(s, N);
        printInt(s2, N);
        printInt(k, N);
        getchar();
    }
    */

    //Check for distinguished point
    if(isDistinguishedPoint || isFruitlessCycle) {
        
        if(isDistinguishedPoint) {
            printf("Found distinguished point!\n");
            // Call callback function
            if(_callback != NULL) {
                struct CallbackParameters cp;

                cp.aStart = *_a;
                cp.bStart = *_b;
                cp.x = BigInteger(newX, N);
                cp.y = BigInteger(newY, N);
                cp.length = *_lengthBuf;

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
        xNew.getWords(_x, N);
        yNew.getWords(_y, N);
        *_a = aNew;
        *_b = bNew;

        *_lengthBuf = 1;

    } else {
        // Write result to memory
        copyWords<N>(newX, _x);
        copyWords<N>(newY, _y);
    }
}


template<int N> void RhoCPU<N>::doStepMulti()
{
    unsigned long *chainBuf = _chainBuf;
    unsigned long *diffBuf = _diffBuf;
    unsigned long long *lengthBuf = _lengthBuf;

    // Initialize to 1
    unsigned long product[N] = {0};
    product[0] = 1;

    for(unsigned int i = 0; i < _numPoints; i++) {
        unsigned int index = i * N;

        unsigned int idx = _x[ index ] & _rPointMask;
        unsigned long diff[N];

        _fp.subModP(&_x[index], &_rx[idx * N], diff);
        copyWords<N>(diff, &diffBuf[ index ]);
        
        _fp.multiplyModP(product, diff, product);
        copyWords<N>(product, &chainBuf[ index ]);
    }
    

    unsigned long inverse[N];
    _fp.inverseModP(product, inverse);

    // Extract inverse of the differences
    for(int i = _numPoints - 1; i >= 0; i--) {
        int index = i * N;

        // Get the inverse of the last difference by multiplying the inverse
        // of the product of all the differences with the product of all but
        // the last difference
        unsigned long invDiff[N];
        
        if(i >= 1) {
            _fp.multiplyModP(inverse, &chainBuf[(i - 1) * N], invDiff);
            _fp.multiplyModP(inverse, &diffBuf[ index ], inverse);
        } else {
            copyWords<N>(inverse, invDiff);
        }
        

        unsigned int idx = _x[ index ] & _rPointMask;

        unsigned long px[N] = {0};
        unsigned long py[N] = {0};

        // Copy onto stack
        copyWords<N>(&_x[ index ], px);
        copyWords<N>(&_y[ index ], py);
     
        // Calculate slope (Py - Qy)/(Px - Qx)
        unsigned long rise[N];
        _fp.subModP(py, &_ry[ idx * N], rise);
        unsigned long s[N] = {0};
        _fp.multiplyModP(invDiff, rise, s);

        // calculate s^2
        unsigned long s2[N] = {0};
        _fp.squareModP(s, s2);

        // Rx = s^2 - Px - Qx
        unsigned long newX[N] = {0};
        _fp.subModP(s2, px, newX);
        _fp.subModP(newX, &_rx[ idx * N], newX);

        // Ry = s(Px - Rx) - Py
        unsigned long k[N];
        _fp.subModP(px, newX, k);
   
        _fp.multiplyModP(k, s, k);
        unsigned long newY[N];
        _fp.subModP(k, py, newY);

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
                    cp.x = BigInteger(newX, N);
                    cp.y = BigInteger(newY, N);
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
            xNew.getWords(&_x[index], N);
            yNew.getWords(&_y[index], N);
            _a[i] = aNew;
            _b[i] = bNew;
            lengthBuf[i] = 0;
            
        } else {
            // Write result to memory
            copyWords<N>(newX, &_x[ index ]);
            copyWords<N>(newY, &_y[ index ]);
        }
    }
}

template<int N> void RhoCPU<N>::doStep()
{
    //doStepMulti();
    if(_numPoints > 1) {
        doStepMulti();
    } else {
        doStepSingle();
    }
}
#endif