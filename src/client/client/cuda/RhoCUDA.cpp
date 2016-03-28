#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "logger.h"
#include "RhoCUDA.h"
#include "kernels.h"
#include "ecc.h"
#include "util.h"
#include "cudapp.h"

void printBigInt(unsigned int *x, int len)
{
    for(int i = len - 1; i>= 0; i--) {
        printf("%.8x", x[i]);
    }
    printf("\n");
}

/**
 * Checks if the kernel should still be running
 */
bool RhoCUDA::isRunning()
{
    //TODO: Protect access with a mutex
    return _runFlag;
}

void RhoCUDA::setRunFlag(bool flag)
{
    // TODO: Protect access with mutex
    _runFlag = flag;
}

/**
 * If the kernl is currently running, stop it
 */
bool RhoCUDA::stop()
{
    // TODO: Protect access to this with a mutex
    _runFlag = false;

    return true;
}

/**
 * Takes an integer array and 'splats' its contents into coalesced form that the GPU uses
 */
void RhoCUDA::splatBigInt(const unsigned int *x, unsigned int *ara, int index)
{
    unsigned int numPoints = _totalPoints;

    for(unsigned int i = 0; i < _pWords; i++) {
        ara[ numPoints * i + index ] = x[i];
    }
}

/**
 * Performs the opposite of splatBigInt. Reads an integer from coalesced form
 * into an array.
 */
void RhoCUDA::extractBigInt(const unsigned int *ara, int index, unsigned int *x)
{
    unsigned int numPoints = _totalPoints;

    for(unsigned int i = 0; i < _pWords; i++) {
        x[ i ] = ara[ numPoints * i + index ];
    }
}

void RhoCUDA::setRPoints()
{
    unsigned int rxAra[ _numRPoints * _pWords ];
    unsigned int ryAra[ _numRPoints * _pWords ];

    memset(rxAra, 0, sizeof(unsigned int) * _numRPoints * _pWords);
    memset(ryAra, 0, sizeof(unsigned int) * _numRPoints * _pWords);

    for(unsigned int i = 0; i < _numRPoints; i++) {
        _rx[i].getWords(&rxAra[ i * _pWords]);
        _ry[i].getWords(&ryAra[ i * _pWords]);
    }

    cudaError_t cudaError = copyRPointsToDevice(rxAra, ryAra, _pWords, _numRPoints);

    if( cudaError != cudaSuccess ) {
        throw cudaError;
    }
}

/**
 * Generates required constants and copies them to the GPU constant memory
 */
void RhoCUDA::setupDeviceConstants()
{
    cudaError_t cudaError = cudaSuccess;

    unsigned int pAra[_pWords];
    _params.p.getWords(pAra);

    unsigned int mAra[_mWords];
    _m.getWords(mAra);

    cudaError = initDeviceParams(pAra, _pBits, mAra, _mBits, _params.dBits);

    if(cudaError != cudaSuccess) {
        throw cudaError;
    }

    cudaError = initDeviceConstants(_totalPoints);

    if(cudaError != cudaSuccess) {
        throw cudaError;
    }

}

/**
 * Generates a random point in the form aG + bQ. 
 */
void RhoCUDA::getRandomPoint(unsigned int *x, unsigned int *y, unsigned int *a, unsigned int *b)
{
    unsigned int mask = (0x01 << _params.dBits) - 1;
    do {
        memset(x, 0, sizeof(unsigned int) * _pWords);
        memset(y, 0, sizeof(unsigned int) * _pWords);
        memset(a, 0, sizeof(unsigned int) * _pWords);
        memset(b, 0, sizeof(unsigned int) * _pWords);

        // points G and Q
        ECPoint g(_params.gx, _params.gy);
        ECPoint q(_params.qx, _params.qy);

        // Random a and b
        BigInteger m1 = randomBigInteger(2, _params.n);
        BigInteger m2 = randomBigInteger(2, _params.n);

        // aG, bQ
        ECPoint aG = _curve.multiply(m1, g);
        ECPoint bQ = _curve.multiply(m2, q);

        // aG + bQ
        ECPoint sum = _curve.add(aG, bQ);

        // Convert to uint160 type
        BigInteger sumX = sum.getX();
        BigInteger sumY = sum.getY();

        sumX.getWords(x);
        sumY.getWords(y);
        m1.getWords(a);
        m2.getWords(b);
    }while((x[0] & mask) == 0);
}

/**
 * Generates random 'a' and 'b' values
 */
void RhoCUDA::generateExponentsHost()
{
    unsigned int numPoints = _totalPoints;

    for(unsigned int index = 0; index < numPoints; index++) {
        // TODO: Use better RNG here
        BigInteger a = randomBigInteger(1, _params.n);
        BigInteger b = randomBigInteger(1, _params.n);
        unsigned int aWords[_pWords];
        unsigned int bWords[_pWords];

        memset(aWords, 0, _pWords * sizeof(unsigned int));
        memset(bWords, 0, _pWords * sizeof(unsigned int));

        a.getWords(aWords);
        b.getWords(bWords);
        splatBigInt(aWords, _aStart, index);
        splatBigInt(bWords, _bStart, index);
    }
}

/**
 * For points G and Q, create threee lookup tables containing:
            G, 2G, 4G, 8G, ... (2^n)G,
            Q, 2Q, 4G, 8Q, ... (2^n)Q,
            G+Q, 2(G+Q), 4(G+Q), 8(G+Q), ... (2^n)(G+Q)
 */
void RhoCUDA::generateLookupTable(unsigned int *gx, unsigned int *gy, unsigned int *qx, unsigned int *qy, unsigned int *gqx, unsigned int *gqy)
{
    // Clear memory
    memset(gx, 0, sizeof(unsigned int) * _pBits * _pWords);
    memset(gy, 0, sizeof(unsigned int) * _pBits * _pWords);
    memset(qx, 0, sizeof(unsigned int) * _pBits * _pWords);
    memset(qy, 0, sizeof(unsigned int) * _pBits * _pWords);
    memset(gqx, 0, sizeof(unsigned int) * _pBits * _pWords);
    memset(gqy, 0, sizeof(unsigned int) * _pBits * _pWords);

    // G, Q, G+Q
    ECPoint g(_params.gx, _params.gy);
    ECPoint q(_params.qx, _params.qy);
    ECPoint sum = _curve.add(g, q);

    if(!_curve.pointExists(g)) {
        printf("Error: G is not on the curve!\n");
    }

    if(!_curve.pointExists(q)) {
        printf("Error: Q is not on the curve!\n");
    }

    // Convert Gx, Gy to words
    BigInteger x = g.getX();
    BigInteger y = g.getY();
    x.getWords(gx);
    y.getWords(gy);

    // Convert Qx, Qy to words
    x = q.getX();
    y = q.getY();
    x.getWords(qx);
    y.getWords(qy); 

    // Convert (G+Q)x, (G+Q)y to words
    x = sum.getX();
    y = sum.getY();
    x.getWords(gqx);
    y.getWords(gqy);

    for(unsigned int i = 1; i < _pBits; i++) {
        g = _curve.doubl(g);
        q = _curve.doubl(q);
        sum = _curve.add(g, q);

        if(!_curve.pointExists(g)) {
            printf("Error: G is not on the curve!\n");
        }
        if(!_curve.pointExists(q)) {
            printf("Error: Q is not on the curve!\n");
        }
    
        if(!_curve.pointExists(sum)) {
            printf("Error: G + Q is not on the curve!\n");
        }

        x = g.getX(); 
        y = g.getY(); 
        x.getWords(&gx[i * _pWords]);
        y.getWords(&gy[i * _pWords]);
      
        x = q.getX();
        y = q.getY();
        x.getWords(&qx[i * _pWords]);
        y.getWords(&qy[i * _pWords]);

        x = sum.getX();
        y = sum.getY();
        x.getWords(&gqx[i * _pWords]);
        y.getWords(&gqy[i * _pWords]);
    }
}

/**
 * Generates a random set of starting points for the random walks. Each
 * GPU thread has a different starting point
 */
void RhoCUDA::generateStartingPoints()
{   
    cudaError_t cudaError = cudaSuccess;

    unsigned int gx[ _pBits * _pWords ];
    unsigned int gy[ _pBits * _pWords ];
    unsigned int qx[ _pBits * _pWords ];
    unsigned int qy[ _pBits * _pWords ];
    unsigned int gqx[ _pBits * _pWords ];
    unsigned int gqy[ _pBits * _pWords ];

    // Generate the lookup table to make point generation faster
    generateLookupTable(gx, gy, qx, qy, gqx, gqy);

    unsigned int *devGx = NULL;
    unsigned int *devGy = NULL;
    unsigned int *devQx = NULL;
    unsigned int *devQy = NULL;
    unsigned int *devGQx = NULL;
    unsigned int *devGQy = NULL;

    // Allocate tables in device memory and copy them over
    try {
        unsigned int size = _pBits * _pWords * sizeof(unsigned int);
        devGx = (unsigned int *)CUDA::malloc(size);
        devGy = (unsigned int *)CUDA::malloc(size);
        devQx = (unsigned int *)CUDA::malloc(size);
        devQy = (unsigned int *)CUDA::malloc(size);
        devGQx = (unsigned int *)CUDA::malloc(size);
        devGQy = (unsigned int *)CUDA::malloc(size);

        CUDA::memcpy(devGx, gx, size, cudaMemcpyHostToDevice);
        CUDA::memcpy(devGy, gy, size, cudaMemcpyHostToDevice);
        CUDA::memcpy(devQx, qx, size, cudaMemcpyHostToDevice);
        CUDA::memcpy(devQy, qy, size, cudaMemcpyHostToDevice);
        CUDA::memcpy(devGQx, gqx, size, cudaMemcpyHostToDevice);
        CUDA::memcpy(devGQy, gqy, size, cudaMemcpyHostToDevice);
    } catch(cudaError_t e) {
        cudaError = e;
        goto end;
    }

    // Generate 'a' and 'b'
    generateExponentsHost();

    // Reset points to point at infinity
    resetPoints(_blocks, _threadsPerBlock, _devX, _devY, _totalPoints);

    for(unsigned int i = 0; i < _pBits; i++) {
        cudaError = multiplyAddG( _blocks, _threadsPerBlock,
                                  _devAStart, _devBStart,
                                  devGx, devGy,
                                  devQx, devQy,
                                  devGQx, devGQy,
                                  _devX, _devY,
                                  _devDiffBuf, _devChainBuf,
                                  _totalPoints, _pointsInParallel, i);
        if( cudaError != cudaSuccess ) {
            goto end;
        }
    }


    for(unsigned int i = 0; i < _totalPoints; i++) {
        unsigned int xTest[_pWords];
        unsigned int yTest[_pWords];
        unsigned int aTest[_pWords];
        unsigned int bTest[_pWords];

        memset(xTest, 0, sizeof(unsigned int) * _pWords);
        memset(yTest, 0, sizeof(unsigned int) * _pWords);
        memset(aTest, 0, sizeof(unsigned int) * _pWords);
        memset(bTest, 0, sizeof(unsigned int) * _pWords);

        readXFromDevice(i, xTest);
        readYFromDevice(i, yTest);
        extractBigInt(_aStart, i, aTest);
        extractBigInt(_bStart, i, bTest);

        BigInteger xBig(xTest, _pWords);
        BigInteger yBig(yTest, _pWords);
        BigInteger aBig(aTest, _pWords);
        BigInteger bBig(bTest, _pWords);

        ECPoint g(_params.gx, _params.gy);
        ECPoint q(_params.qx, _params.qy);
         
        ECPoint p(xBig, yBig);

        ECPoint p1 = _curve.multiply(aBig, g);
        ECPoint p2 = _curve.multiply(bBig, q);
        ECPoint sum = _curve.add(p1, p2);

        if(!_curve.pointExists(p)) {
            printf("Error: Point is NOT on curve!\n");
            printf("%s\n%s\n", xBig.toString(16).c_str(), yBig.toString(16).c_str());

            unsigned int a[_pWords];
            unsigned int b[_pWords];

            extractBigInt(_aStart, 0, a);
            extractBigInt(_bStart, 0, b);
            printf("a: ");
            printBigInt(a, _pWords);
            printf("b: ");
            printBigInt(b, _pWords);

        }

        if(sum.getX() != xBig || sum.getY() != yBig) {
            printf("INVALID STARTING POINT\n");
            printf("a: %s\n", aBig.toString(16).c_str());
            printf("b: %s\n", bBig.toString(16).c_str());

            printf("Expected:\n");
            printf("[ %s, %s ]\n", sum.getX().toString(16).c_str(), sum.getY().toString(16).c_str());
            printf("Actual:\n");
            printf("[ %s, %s ]\n", xBig.toString(16).c_str(), yBig.toString(16).c_str());
            throw "ERROR";
        }

    }
end:
    CUDA::free(devGx);
    CUDA::free(devGy);
    CUDA::free(devQx);
    CUDA::free(devQy);
    CUDA::free(devGQx);
    CUDA::free(devGQy);

    if(cudaError != cudaSuccess) {
        throw cudaError;
    }
}

void RhoCUDA::writeBigIntToDevice(const unsigned int *x, unsigned int *dest, unsigned int index )
{
    unsigned int numPoints = _totalPoints;

    for(unsigned int i = 0; i < _pWords; i++) {
        CUDA::memcpy(&dest[ numPoints * i + index ], &x[ i ], 4, cudaMemcpyHostToDevice);
    }
}


void RhoCUDA::readBigIntFromDevice(const unsigned int *src, unsigned int index, unsigned int *x)
{
    unsigned int numPoints = _totalPoints;

    for(unsigned int i = 0; i < _pWords; i++) {
        CUDA::memcpy(&x[ i ], &src[ numPoints * i + index ], 4, cudaMemcpyDeviceToHost);
    }
}

void RhoCUDA::readXFromDevice(unsigned int index, unsigned int *x)
{
    readBigIntFromDevice(_devX, index, x);
}

void RhoCUDA::readYFromDevice(unsigned int index, unsigned int *y)
{
    readBigIntFromDevice(_devY, index, y);
}

void RhoCUDA::writeXToDevice(unsigned int *x, unsigned int index)
{
    writeBigIntToDevice(x, _devX, index);
}

void RhoCUDA::writeYToDevice(unsigned int *y, unsigned int index)
{
    writeBigIntToDevice(y, _devY, index);
}

/**
 * Allocates memory on the host and device according to the number of threads, and
 * number of simultanous computations per thread
 */
void RhoCUDA::allocateBuffers()
{
    size_t numPoints = _totalPoints;
    size_t arraySize = sizeof(unsigned int) * _pWords * numPoints;

    Logger::logInfo("Allocating %ld bytes on device", arraySize * 4);
    Logger::logInfo("Allocating %ld bytes on host", arraySize * 2);
    Logger::logInfo("%d blocks", _blocks);
    Logger::logInfo("%d threads (%d threads per block)", totalThreads(), _threadsPerBlock);
    Logger::logInfo("%d points", numPoints);

    _counters = new unsigned long long[numPoints];
    memset(_counters, 0, sizeof(unsigned long long) * numPoints);

    // Allocate 'a' values in host memory
    _aStart = (unsigned int *)CUDA::hostAlloc(arraySize, cudaHostAllocMapped);
    memset( _aStart, 0, arraySize );

    // Map host memory to device address space
    _devAStart = (unsigned int *)CUDA::getDevicePointer(_aStart, 0);

    // Allocate 'b' values in host memory
    _bStart = (unsigned int *)CUDA::hostAlloc(arraySize, cudaHostAllocMapped );
    memset( _bStart, 0, arraySize );

    // Map host memory to device address space
    _devBStart = (unsigned int *)CUDA::getDevicePointer(_bStart, 0);

    // Each thread gets a flag to notify that it has found a distinguished point
    _pointFoundFlags = (unsigned int *)CUDA::hostAlloc(numPoints * sizeof(unsigned int), cudaHostAllocMapped);
    memset(_pointFoundFlags, 0, numPoints * sizeof(unsigned int));

    // Each section gets a flag to notify if any threads in that block found a distinguished point
    _sectionFlags = (unsigned int *)CUDA::hostAlloc(totalSections() * sizeof(unsigned int), cudaHostAllocMapped);
    memset(_sectionFlags, 0, totalSections() * sizeof(unsigned int));

    // Allocate 'x' values in device memory
    _devX = (unsigned int *)CUDA::malloc(arraySize);

    // Allocate 'y' values in device memory
    _devY = (unsigned int*)CUDA::malloc(arraySize);

    // Allocate buffer to hold difference when computing point addition
    _devDiffBuf = (unsigned int *)CUDA::malloc(arraySize);

    // Allocate buffer to hold the multiplication chain when computing batch inverse
    _devChainBuf = (unsigned int *)CUDA::malloc(arraySize);

    // Allocate integer to be used as a flag if a distinguished point is found
    _devPointFoundFlag = (unsigned int *)CUDA::malloc(4);
}

/**
 * Frees the memory allocated by allocateBuffers
 */
void RhoCUDA::freeBuffers()
{
    cudaFree(_aStart);
    cudaFree(_bStart);
    cudaFree(_devX);
    cudaFree(_devY);
    cudaFree(_devDiffBuf);
    cudaFree(_devChainBuf);
    cudaFree(_devPointFoundFlag);
    cudaFree(_pointFoundFlags);
    cudaFree(_sectionFlags);

    delete[] _counters;
}

/**
 * Verifies that aG + bQ = (x,y)
 */
bool RhoCUDA::verifyPoint(BigInteger &x, BigInteger &y)
{
    ECPoint p = ECPoint(x, y);

    if(!_curve.pointExists(p)) {
        Logger::logError("x: %s\n", x.toString(16).c_str());
        Logger::logError("y: %s\n", y.toString(16).c_str());
        Logger::logError("Point is not on the curve\n");
        return false;
    }

    return true;
}

bool RhoCUDA::initializeDevice()
{
    // Set current device
    CUDA::setDevice(_device);
    
    // Poll while kernel is running, but yield the thread if necessary
    //CUDA::setDeviceFlags(cudaDeviceScheduleYield);
    CUDA::setDeviceFlags(cudaDeviceScheduleBlockingSync);
    //cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferL1);

    try {
        allocateBuffers();

        setupDeviceConstants();
        setRPoints();
    }catch(cudaError_t err) {
        Logger::logError("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }

    _initialized = true;
    
    return true;
}

void RhoCUDA::uninitializeDevice()
{
    freeBuffers();
    cudaDeviceReset();
    _initialized = false;
}

/**
 * Creates a new context
 */
RhoCUDA::RhoCUDA( int device, unsigned int blocks,
                                      unsigned int threads,
                                      unsigned int totalPoints,
                                      unsigned int pointsInParallel,
                                      const ECDLPParams *params,
                                      const BigInteger *rx,
                                      const BigInteger *ry,
                                      int numRPoints,
                                      void (*callback)(struct CallbackParameters *) )
{
    _mainCounter = 1;
    _blocks = blocks;
    _threadsPerBlock = threads;
    _totalPoints = totalPoints;
    _pointsInParallel = pointsInParallel;
    _device = device;
    _callback = callback;
    _runFlag = true;
    _params = *params;
    _numRPoints = numRPoints; 
    _params.p = params->p;


    // Initialize barrett reduction bits
    _pBits = _params.p.getBitLength();
    _pWords = (_pBits + 31) / 32;

    // m is 4^k / p where k is the number of bits in p
    _m = BigInteger(4).pow(_pBits);
    _m = _m / _params.p;
    _mBits = _m.getBitLength();
    _mWords = (_mBits + 31) / 32;

    // Copy random walk points
    for(unsigned int i = 0; i < _numRPoints; i++) {
        _rx[ i ] = rx[ i ];
        _ry[ i ] = ry[ i ];
    }

    _curve = ECCurve(_params.p, _params.n, _params.a, _params.b, _params.gx, _params.gy);
}

/**
 * Initializes the context with random points
 */
bool RhoCUDA::init()
{
    try { 
        initializeDevice();
        generateStartingPoints();
    } catch(cudaError_t err) {
        Logger::logError("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

void RhoCUDA::reset()
{

}

/**
 * Frees all resources used by the context
 */
RhoCUDA::~RhoCUDA()
{
    if(_initialized) {
        uninitializeDevice();
    }
}

/**
 * Reads the flag from the GPU indicating if a point was found
 */
bool RhoCUDA::pointFound()
{
    unsigned int flag = 0;
    CUDA::memcpy(&flag, _devPointFoundFlag, 4, cudaMemcpyDeviceToHost);

    if(flag) {
        return true; 
    }

    return false;
}

bool RhoCUDA::doStep()
{
    //do {
    cudaError_t cudaError = cudaSuccess;

    cudaError = cudaDoStep(_blocks,
        _threadsPerBlock,
        _devX,
        _devY,
        _devDiffBuf,
        _devChainBuf,
        _devPointFoundFlag,
        _sectionFlags,
        _pointFoundFlags,
        _totalPoints,
        _pointsInParallel);
    
    if(cudaError != cudaSuccess) {
        Logger::logError("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return false;
    }

    _mainCounter++;

    if(pointFound()) {
        for(unsigned int section = 0; section < totalSections(); section++) {
            if(!_sectionFlags[section]) {
                continue;
            }

            _sectionFlags[section] = 0;

            for(unsigned int i = 0; i < 32; i++) {

                unsigned int index = (section * 32) + i;

                // Check if a point was found
                if(!_pointFoundFlags[index]) {
                    continue;
                }
                _pointFoundFlags[index] = 0;

                unsigned int x[_pWords];
                unsigned int y[_pWords];
                unsigned int a[_pWords];
                unsigned int b[_pWords];

                memset(x, 0, _pWords * sizeof(unsigned int));
                memset(y, 0, _pWords * sizeof(unsigned int));
                memset(a, 0, _pWords * sizeof(unsigned int));
                memset(b, 0, _pWords * sizeof(unsigned int));

                readXFromDevice(index, x);
                readYFromDevice(index, y);
                extractBigInt(_aStart, index, a);
                extractBigInt(_bStart, index, b);

                BigInteger xBig(x, _pWords);
                BigInteger yBig(y, _pWords);
                BigInteger aBig(a, _pWords);
                BigInteger bBig(b, _pWords);
                if(!verifyPoint(xBig, yBig)) {
                    Logger::logError( "==== INVALID POINT ====\n" );
                    printf("Index: %d\n", index);
                    Logger::logError("%s %s\n", aBig.toString(16).c_str(), bBig.toString(16).c_str());
                    Logger::logError("[%s, %s]\n", xBig.toString(16).c_str(), yBig.toString(16).c_str());
                    printf("a: ");
                    printBigInt(a, _pWords);
                    printf("b: ");
                    printBigInt(b, _pWords);
                    printf("x: ");
                    printBigInt(x, _pWords);
                    printf("y: ");
                    printBigInt(y, _pWords);
                    return false;
                }
             
                struct CallbackParameters p;
                p.aStart = aBig;
                p.bStart = bBig;
                p.x = xBig;
                p.y = yBig;
                p.length = (unsigned int)(_mainCounter - _counters[index]);

                _callback(&p);
                
                unsigned int newX[_pWords];
                unsigned int newY[_pWords];
                unsigned int newA[_pWords];
                unsigned int newB[_pWords]; 
                getRandomPoint(newX, newY, newA, newB);
                
                // Write a, b to host memory
                splatBigInt(newA, _aStart, index);
                splatBigInt(newB, _bStart, index);

                // Write x, y to device memory
                writeXToDevice( newX, index );
                writeYToDevice( newY, index );

                _counters[index] = _mainCounter;
            }
        }
    }

    return true;

    //running = isRunning();
    //iterations++;
    //}while( running );
}

/**
 * Runs the context
 */
bool RhoCUDA::run()
{
    setRunFlag(true);

    do {
        if(!doStep()) {
            return false;
        }
    }while(isRunning());

    return true;
}

bool RhoCUDA::benchmark(unsigned long long *pointsPerSecondPtr)
{
    unsigned int t0 = 0;
    unsigned int t1 = 0;
    unsigned int count = 25000;
    bool success = true;
    float seconds = 0;
    unsigned int iterationsPerSecond = 0;
    unsigned long long pointsPerSecond = 0;

    //count = 0;
    /*
    while(1) {
        cudaError_t cudaError = cudaSuccess;
        
        cudaError = doStep(_blocks,
            _threadsPerBlock,
            _devX,
            _devY,
            _devDiffBuf,
            _devChainBuf,
            _devPointFoundFlag,
            _sectionFlags,
            _pointFoundFlags,
            _totalPoints,
            _pointsInParallel); 

        count++;
        if(cudaError != cudaSuccess) {
            Logger::logError("CUDA error: %s\n", cudaGetErrorString( cudaError ));
            success = false;
            goto end;
        }
        t1 = util::getSystemTime();

        if(t1 - t0 >= 10000) {
            break;
        }
    }*/


    t0 = util::getSystemTime();
    for(unsigned int i = 0; i < count; i++) {
        cudaError_t cudaError = cudaSuccess;
        
        cudaError = cudaDoStep(_blocks,
            _threadsPerBlock,
            _devX,
            _devY,
            _devDiffBuf,
            _devChainBuf,
            _devPointFoundFlag,
            _sectionFlags,
            _pointFoundFlags,
            _totalPoints,
            _pointsInParallel); 

        if(cudaError != cudaSuccess) {
            Logger::logError("CUDA error: %s\n", cudaGetErrorString( cudaError ));
            success = false;
            goto end;
        }

        i++;
        t1 = util::getSystemTime();
    }
    t1 = util::getSystemTime();
  
    // Number of seconds that elapsed 
    seconds = (float)(t1 - t0)/1000;
    iterationsPerSecond = (unsigned int)count / seconds;
    pointsPerSecond = iterationsPerSecond * _totalPoints;

    Logger::logInfo("%d iterations in %dms (%d iterations per second)", count, (t1-t0), iterationsPerSecond);
    Logger::logInfo("%lld points per second", pointsPerSecond);

    if( pointsPerSecondPtr != NULL ) {
        *pointsPerSecondPtr = pointsPerSecond;
    }

end:

    return success;
}
