#include "BigInteger.h"
#include "ECDLCPU.h"

#include "math/Fp.h"
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
static unsigned int *_lengthBuf;

static unsigned int _numPoints;
static unsigned int _rPointMask;
static unsigned long _dBitsMask;
static unsigned int _pLen;

static void (*_callback)(struct CallbackParameters *);

static inline bool checkDistinguishedBits(const unsigned long *x)
{
    if((x[ 0 ] & _dBitsMask) == 0) {
        return true;
    } else {
        return false;
    }
}

/**
 * Generates a random point on the curve
 */
static void generateStartingPoint(BigInteger &x, BigInteger &y, BigInteger &a, BigInteger &b)
{
    unsigned long buf[_pLen];

    do {
        a = randomBigInteger(2, _params.n);
        b = randomBigInteger(2, _params.n);

        ECPoint p1 = _curve.multiplyPoint(a, _g);
        ECPoint p2 = _curve.multiplyPoint(b, _q);
        ECPoint p3 = _curve.addPoint(p1, p2);

        x = p3.getX();
        y = p3.getY();

        // Check that we don't start on a distinguished point
        x.getWords(buf, _pLen);
    }while(checkDistinguishedBits(buf));

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

    _numPoints = numPoints;

    // Initialize Barrett code for modulus P
    BigInteger p = _curve.p();
    initFp(p);
    unsigned int pLen = p.getWordLength();
    _pLen = pLen;

    // Pointer to the coefficients of the starting points
    _a = a;
    _b = b;

    // (x,y) of the current points
    _x = new unsigned long[numPoints * numThreads * pLen];
    _y = new unsigned long[numPoints * numThreads * pLen];
    _rx = new unsigned long[32 * pLen];
    _ry = new unsigned long[32 * pLen];
    _diffBuf = new unsigned long[numPoints * numThreads * pLen];
    _chainBuf = new unsigned long[numPoints * numThreads * pLen];
    _lengthBuf = new unsigned int[numPoints * numThreads];

    // Initialize length to 1 (starting point counts as 1 point)
    memset(_lengthBuf, 0, sizeof(unsigned int) * numPoints * numThreads);
    for(int i = 0; i < numPoints * numThreads; i++) {
        _lengthBuf[i] = 1;
    }

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
    _dBitsMask = ~0;
    _dBitsMask >>= WORD_LENGTH_BITS - dBits;

    // Mask for selecting R point
    _rPointMask = numRPoints - 1;

    // Gets called when distinguished point is found
    _callback = callback;
}

void cleanupThreadGlobals()
{
    delete[] _x;
    delete[] _y;
    delete[] _rx;
    delete[] _ry;
    delete[] _diffBuf;
    delete[] _chainBuf;
    delete[] _lengthBuf;
}

template<int N> void copyWords(const unsigned long *src, unsigned long *dest)
{
    memcpy(dest, src, sizeof(unsigned long) * N);
}


template<int N> void doStepInternal(int threadId)
{
    unsigned long *x = &_x[threadId * _numPoints * N];
    unsigned long *y = &_y[threadId * _numPoints * N];
    unsigned long *chainBuf = &_chainBuf[threadId * _numPoints * N];
    unsigned long *diffBuf = &_diffBuf[threadId * _numPoints * N];
    unsigned int *lengthBuf = &_lengthBuf[threadId * _numPoints];

    // Initialize to 1
    unsigned long product[N] = {0};
    product[0] = 1;

    for(unsigned int i = 0; i < _numPoints; i++) {
        unsigned int index = i * N;

        unsigned int idx = x[ index ] & _rPointMask;
        unsigned long diff[N];

        subModP<N>(&x[index], &_rx[idx * N], diff);
        copyWords<N>(diff, &diffBuf[ index ]);
        
        multiplyModP<N>(product, diff, product);
        copyWords<N>(product, &chainBuf[ index ]);
    }
    

    unsigned long inverse[N];
    inverseModP<N>(product, inverse);

    // Extract inverse of the differences
    for(int i = _numPoints - 1; i >= 0; i--) {
        int index = i * N;

        // Get the inverse of the last difference by multiplying the inverse
        // of the product of all the differences with the product of all but
        // the last difference
        unsigned long invDiff[N];
        
        if(i >= 1) {
            multiplyModP<N>(inverse, &chainBuf[(i - 1) * N], invDiff);
            multiplyModP<N>(inverse, &diffBuf[ index ], inverse);
        } else {
            copyWords<N>(inverse, invDiff);
        }
        

        unsigned int idx = x[ index ] & _rPointMask;

        unsigned long px[N];
        unsigned long py[N];

        // Copy onto stack
        copyWords<N>(&x[ index ], px);
        copyWords<N>(&y[ index ], py);
     
        // Calculate slope (Py - Qy)/(Px - Qx)
        unsigned long rise[N];
        subModP<N>(py, &_ry[ idx * _pLen ], rise);
        unsigned long s[N];
        multiplyModP<N>(invDiff, rise, s);

        // calculate s^2
        unsigned long s2[N];
        squareModP<N>(s, s2);

        // Rx = s^2 - Px - Qx
        unsigned long newX[N];
        subModP<N>(s2, px, newX);
        subModP<N>(newX, &_rx[ idx * _pLen ], newX);

        // Ry = s(Px - Rx) - Py
        unsigned long k[N];
        subModP<N>(px, newX, k);
     
        multiplyModP<N>(k, s, k);
        unsigned long newY[N];
        subModP<N>(k, py, newY);

        // Increment walk length
        lengthBuf[i]++;

        bool isDistinguishedPoint = checkDistinguishedBits(newX);

        bool isFruitlessCycle = false;

        if ( lengthBuf[i] >= (unsigned long long)1 << (_params.dBits + 2)) {
            isFruitlessCycle = true;
        }
       
        //Check for distinguished point
        if(isDistinguishedPoint || isFruitlessCycle) {
            
            BigInteger *a = &_a[threadId * _numPoints];
            BigInteger *b = &_b[threadId * _numPoints];
            
            if(isDistinguishedPoint) {

                // Call callback function
                if(_callback != NULL) {
                    struct CallbackParameters cp;
                    cp.aStart = a[i];
                    cp.bStart = b[i];
                    cp.x = BigInteger(newX, N);
                    cp.y = BigInteger(newY, N);
                    cp.length = lengthBuf[i];
                    _callback(&cp);
                }
            } else {
                Logger::logInfo("Thread %d: Possible cycle found (%d iterations), rejecting\n", threadId, lengthBuf[i]);
            }

            // Generate new starting point
            BigInteger aNew;
            BigInteger bNew;
            BigInteger xNew;
            BigInteger yNew;

            // Generate new starting point
            generateStartingPoint(xNew, yNew, aNew, bNew);

            // Copy new point to memory
            xNew.getWords(&x[index], N);
            yNew.getWords(&y[index], N);
            a[i] = aNew;
            b[i] = bNew;
            lengthBuf[i] = 0;
        } else {
            // Write result to memory
            copyWords<N>(newX, &x[ index ]);
            copyWords<N>(newY, &y[ index ]);
        }
        
    }
}

static void doStep(int threadId) {
    switch(_pLen) {
        case 1:
            doStepInternal<1>(threadId);
            break;
        case 2:
            doStepInternal<2>(threadId);
            break;
        case 3:
            doStepInternal<3>(threadId);
            break;
        case 4:
            doStepInternal<4>(threadId);
            break;
        case 5:
            doStepInternal<5>(threadId);
            break;
        case 6:
            doStepInternal<6>(threadId);
            break;
        case 7:
            doStepInternal<7>(threadId);
            break;
        default:
            logger::logError("ERROR: COMPILE SUPPORT FOR LARGER INTEGERS\n");
            exit(1);
            break;
    }
}

void *benchmarkThreadFunction(void *p)
{
    BenchmarkThreadParams *params = (BenchmarkThreadParams *)p;

    unsigned int threadId = params->threadId;

    unsigned int t0 = util::getSystemTime();
    for(unsigned int i = 0; i < params->iterations; i++) {
        doStep(threadId);
    }
    params->t = util::getSystemTime() - t0;

    return NULL;
}

void *workerThreadFunction(void *p)
{
    WorkerThreadParams *params = (WorkerThreadParams *)p;

    unsigned int threadId = params->threadId;

    while(params->running) {
        doStep(threadId);
    }

    return NULL;
}
