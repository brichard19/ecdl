#include "BigInteger.h"
#include "ECDLCPU.h"

#include "Fp.h"
#include "util.h"

static ECPoint _g;
static ECPoint _q;
static ECDLPParams _params;
static ECCurve _curve;
static BigInteger *_a;
static BigInteger *_b;
static unsigned long *_x;
static unsigned long *_y;
static unsigned long *_rx;
static unsigned long *_ry;
static unsigned long *_diffBuf;
static unsigned long *_chainBuf;
static unsigned int _numPoints;
static unsigned int _rPointMask;
static unsigned long _mask;
static unsigned int _pLen;

static void (*_callback)(struct CallbackParameters *);

static inline bool checkMask(const unsigned long *x)
{
    if((x[ 0 ] & _mask) == 0) {
        return true;
    } else {
        return false;
    }
}

/**
 * Generates a random point on the curve
 */
static void generateNewPoint(BigInteger *x, BigInteger *y, BigInteger *a, BigInteger *b)
{
    unsigned long buf[_pLen];

    do {
        *a = randomBigInteger(_params.n);
        *b = randomBigInteger(_params.n);

        ECPoint p1 = _curve.multiplyPoint(*a, _g);
        ECPoint p2 = _curve.multiplyPoint(*b, _q);
        ECPoint p3 = _curve.addPoint(p1, p2);

        *x = p3.getX();
        *y = p3.getY();

        // Check that we don't start on a distinguished point
        x->getWords(buf, _pLen);
    }while(checkMask(buf));

}

void initThreadGlobals(ECDLPParams *params,
                        BigInteger *rx,
                        BigInteger *ry,
                        int numRPoints,
                        BigInteger *a,
                        BigInteger *b,
                        BigInteger *x,
                        BigInteger *y,
                        int numThreads,
                        int numPoints,
                        int dBits,
                        void (*callback)(struct CallbackParameters *)
                        )
{
    // Copy parameters
    _params = *params;

    // Create curve
    _curve = ECCurve(_params.p, _params.n, _params.a, _params.b, _params.gx, _params.gy);

    // Create points
    _g = ECPoint(_params.gx, _params.gy);
    _q = ECPoint(_params.qx, _params.qy);

    // Convert paramters from BigInteger
    _numPoints = numPoints;

    // Initialize Barrett code
    BigInteger p = _curve.p();
    initFp(p);
    unsigned int pLen = p.getWordLength();
    _pLen = pLen;

    printf("P is %d words\n", _pLen);

    // Pointer to the coefficients of the starting points
    _a = a;
    _b = b;

    printf("Allocating room for %d points (%lld bytes)\n", numPoints, (unsigned long long)(4 * numPoints * sizeof(unsigned long) * pLen));
    
    // (x,y) of the current points
    _x = new unsigned long[numPoints * numThreads * pLen];
    _y = new unsigned long[numPoints * numThreads * pLen];
    _rx = new unsigned long[32 * pLen];
    _ry = new unsigned long[32 * pLen];
    _diffBuf = new unsigned long[numPoints * numThreads * pLen];
    _chainBuf = new unsigned long[numPoints * numThreads * pLen];

    // Copy points
    for(int i = 0; i < numPoints * numThreads; i++) {
        int index = i * pLen;
        x[i].getWords(&_x[index], pLen);
        y[i].getWords(&_y[index], pLen);
    }

    // Copy R points
    for(int i = 0; i < 32; i++) {
        int index = i * pLen;
        rx[i].getWords(&_rx[index], pLen);
        ry[i].getWords(&_ry[index], pLen);
    }

    // Set mask for detecting distinguished points
    _mask = ~0;
    _mask >>= WORD_LENGTH_BITS - dBits;

    printf("Mask: ");
    util::printHex(_mask);
    printf("\n");

    // Mask for selecting R point
    _rPointMask = numRPoints - 1;
    printf("R Mask: %.8x\n", _rPointMask);

    // Gets called when distinguished point is found
    _callback = callback;
    if(_callback == NULL) {
        printf("Callback is NULL\n");
    }
}

void cleanupThreadGlobals()
{
    delete[] _x;
    delete[] _y;
    delete[] _rx;
    delete[] _ry;
    delete[] _diffBuf;
    delete[] _chainBuf;
}

void copyWords(unsigned long *src, unsigned long *dest, int len)
{
    for(int i = 0; i < len; i++) {
        dest[i] = src[i];
    }
}

void doStep(int threadId)
{
    unsigned long *x = &_x[threadId * _numPoints * _pLen];
    unsigned long *y = &_y[threadId * _numPoints * _pLen];
    unsigned long *chainBuf = &_chainBuf[threadId * _numPoints * _pLen];
    unsigned long *diffBuf = &_diffBuf[threadId * _numPoints * _pLen];
    BigInteger *a = &_a[threadId * _numPoints];
    BigInteger *b = &_b[threadId * _numPoints];

    // Initialize to 1
    FpElement product = {0};

    product[0] = 1;

    for(unsigned int i = 0; i < _numPoints; i++) {
        unsigned int index = i * _pLen;

        unsigned int idx = x[ index ] & _rPointMask;
        FpElement diff;

        subModP(&x[index], &_rx[idx * _pLen], diff);
        copyWords(diff, &diffBuf[ index ], _pLen);
        
        multiplyModP(product, diff, product);
        copyWords(product, &chainBuf[ index ], _pLen);
    }
   
    FpElement inverse = {0};
    inverseModP(product, inverse);

    // Extract inverse of the differences
    for(int i = _numPoints - 1; i >= 0; i--) {
        int index = i * _pLen;

        // Get the inverse of the last difference by multiplying the inverse
        // of the product of all the differences with the product of all but
        // the last difference
        FpElement invDiff;
        if(i >= 1) {
            multiplyModP(inverse, &chainBuf[(i - 1) * _pLen], invDiff);
            multiplyModP(inverse, &diffBuf[ index ], inverse);
        } else {
            copyWords(inverse, invDiff, _pLen);
        }
        unsigned int idx = x[ index ] & _rPointMask;

        FpElement px;
        FpElement py;
        copyWords(&x[ index ], px, _pLen);
        copyWords(&y[ index ], py, _pLen);
      
        // s = (Py - Qy)/(Px - Qx)
        FpElement rise;
        subModP(py, &_ry[ idx * _pLen ], rise);
        FpElement s;
        multiplyModP(invDiff, rise, s);
        FpElement s2;
        squareModP(s, s2);

        // Rx = s^2 - Px - Qx
        FpElement newX;
        subModP(s2, px, newX);
        subModP(newX, &_rx[ idx * _pLen ], newX);

        // Ry = s(Px - Rx) - Py
        FpElement k;
        subModP(px, newX, k);
     
        multiplyModP(k, s, k);
        FpElement newY;
        subModP(k, py, newY);

        //Check for distinguished point, call callback function if found
        if(checkMask(newX)) {

            if(_callback != NULL) {
                struct CallbackParameters cp;
                cp.aStart = a[i];
                cp.bStart = b[i];
                cp.x = BigInteger(newX, _pLen);
                cp.y = BigInteger(newY, _pLen);
                _callback(&cp);
            }

            // Generate new starting point
            BigInteger aNew;
            BigInteger bNew;
            BigInteger xNew;
            BigInteger yNew;

            // Generate new starting point
            generateNewPoint(&xNew, &yNew, &aNew, &bNew);

            // Copy new point to memory
            xNew.getWords(&x[index], _pLen);
            yNew.getWords(&y[index], _pLen);
            a[i] = aNew;
            b[i] = bNew;
        } else {
            // Write result to memory
            copyWords(newX, &x[ index ], _pLen);
            copyWords(newY, &y[ index ], _pLen);
        }
    }
}

void *benchmarkThreadFunction(void *p)
{
    BenchmarkThreadParams *params = (BenchmarkThreadParams *)p;

    unsigned int threadId = params->threadId;

    printf("Starting benchmark thread %d\n", threadId);
    unsigned int t0 = util::getSystemTime();
    for(unsigned int i = 0; i < params->iterations; i++) {
        doStep(threadId);
    }
    params->t = util::getSystemTime() - t0;

    printf("Exiting benchmark thread %d\n", threadId);

    return NULL;
}

void *workerThreadFunction(void *p)
{
    WorkerThreadParams *params = (WorkerThreadParams *)p;

    printf("Thread %d started\n", params->threadId);
    unsigned int threadId = params->threadId;

    while(params->running) {
        doStep(threadId);
    }
    printf("Thread %d exiting\n", params->threadId);

    return NULL;
}

