#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "logger.h"
#include "ECDLCuda.h"
#include "kernels.h"
#include "ecc.h"
#include "BigInteger.h"
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
bool ECDLCudaContext::isRunning()
{
    //TODO: Protect access with a mutex
    return runFlag;
}

void ECDLCudaContext::setRunFlag(bool flag)
{
    // TODO: Protect access with mutex
    runFlag = flag;
}

/**
 * If the kernl is currently running, stop it
 */
bool ECDLCudaContext::stop()
{
    // TODO: Protect access to this with a mutex
    runFlag = false;

    return true;
}


void ECDLCudaContext::splatBigInt(const unsigned int *x, unsigned int *ara, int block, int thread, int index)
{
    splatBigInt(x, ara, threadsPerBlock * block + thread, index);
}

/**
 * Takes an integer array and 'splats' its contents into coalesced form that the GPU uses
 */
void ECDLCudaContext::splatBigInt(const unsigned int *x, unsigned int *ara, int thread, int index)
{
    unsigned int numThreads = threadsPerBlock * blocks;

    for(int i = 0; i < this->pLen; i++) {
        ara[ this->pLen * numThreads * index + numThreads * i + thread ] = x[i];
    }
}

void ECDLCudaContext::extractBigInt(const unsigned int *ara, int block, int thread, int index, unsigned int *x)
{
    extractBigInt(ara, threadsPerBlock * block + thread, index, x);
}

/**
 * Performs the opposite of splatBigInt. Reads an integer from coalesced form
 * into an array.
 */
void ECDLCudaContext::extractBigInt(const unsigned int *ara, int thread, int index, unsigned int *x)
{
    unsigned int numThreads = threadsPerBlock * blocks;

    for(int i = 0; i < this->pLen; i++) {
        x[ i ] = ara[ this->pLen * numThreads * index + numThreads * i + thread ];
    }
}

void ECDLCudaContext::setRPoints()
{
    unsigned int rxAra[ this->rPoints * this->pLen ];
    unsigned int ryAra[ this->rPoints * this->pLen ];

    memset(rxAra, 0, sizeof(unsigned int) * this->rPoints * this->pLen);
    memset(ryAra, 0, sizeof(unsigned int) * this->rPoints * this->pLen);

    for(int i = 0; i < this->rPoints; i++) {
        this->rx[i].getWords(&rxAra[ i * this->pLen]);
        this->ry[i].getWords(&ryAra[ i * this->pLen]);
    }

    cudaError_t cudaError = copyRPointsToDevice(rxAra, ryAra, this->pLen, this->rPoints);
    if( cudaError != cudaSuccess ) {
        throw cudaError;
    }
}

/**
 * Generates required constants and copies them to the GPU constant memory
 */
void ECDLCudaContext::setupDeviceConstants()
{
    cudaError_t cudaError = cudaSuccess;

    unsigned int pAra[this->pLen];
    this->p.getWords(pAra);

    unsigned int mAra[this->mLen];
    this->m.getWords(mAra);

    BigInteger pMinus2 = p - 2;
    unsigned int pMinus2Ara[this->pLen];
    pMinus2.getWords(pMinus2Ara);

    BigInteger pTimes2 = this->p * 2;
    unsigned int pTimes2Ara[pTimes2.getWordLength()];
    pTimes2.getWords(pTimes2Ara);

    BigInteger pTimes3 = this->p * 3;
    unsigned int pTimes3Ara[pTimes3.getWordLength()];
    pTimes3.getWords(pTimes3Ara);

    cudaError = initDeviceParams(pAra, this->pBits, mAra, this->mBits, pMinus2Ara, pTimes2Ara, pTimes3Ara, params.dBits);
    
    if(cudaError != cudaSuccess) {
        throw cudaError;
    }
}

/**
 * Generates a random point n the form aG + bQ. 
 */
void ECDLCudaContext::getRandomPoint(unsigned int *x, unsigned int *y, unsigned int *a, unsigned int *b)
{
    unsigned int mask = (0x01 << this->params.dBits) - 1;
    do {
        memset(x, 0, sizeof(unsigned int) * this->pLen);
        memset(y, 0, sizeof(unsigned int) * this->pLen);
        memset(a, 0, sizeof(unsigned int) * this->pLen);
        memset(b, 0, sizeof(unsigned int) * this->pLen);

        // points G and Q
        ECPoint g(params.gx, params.gy);
        ECPoint q(params.qx, params.qy);

        // Random a and b
        BigInteger m1 = randomBigInteger(params.n);
        BigInteger m2 = randomBigInteger(params.n);

        // aG, bQ
        ECPoint aG = this->curve.multiplyPoint(m1, g);
        ECPoint bQ = this->curve.multiplyPoint(m2, q);

        // aG + bQ
        ECPoint sum = this->curve.addPoint(aG, bQ);

        // Convert to uint160 type
        BigInteger sumX = sum.getX();
        BigInteger sumY = sum.getY();

        sumX.getWords(x);
        sumY.getWords(y);
        m1.getWords(a);
        m2.getWords(b);
    }while(x[0] & mask == 0);
}

/**
 * Generates random 'a' and 'b' values
 */
void ECDLCudaContext::generateMultipliersHost()
{
    for(unsigned int index = 0; index < this->pointsPerThread; index++) {
        for(unsigned int thread = 0; thread < this->blocks * this->threadsPerBlock; thread++) {
            // TODO: Use better RNG here
            BigInteger m1 = randomBigInteger(params.n);
            BigInteger m2 = randomBigInteger(params.n);
            unsigned int a[this->pLen];
            unsigned int b[this->pLen];

            memset(a, 0, this->pLen * sizeof(unsigned int));
            memset(b, 0, this->pLen * sizeof(unsigned int));

            m1.getWords(a);
            m2.getWords(b);
            splatBigInt(a, AStart, thread, index);
            splatBigInt(b, BStart, thread, index);
        }
    }
}

void ECDLCudaContext::generateLookupTable(unsigned int *gx, unsigned int *gy, unsigned int *qx, unsigned int *qy, unsigned int *gqx, unsigned int *gqy)
{
    memset(gx, 0, sizeof(unsigned int) * this->pBits * this->pLen);
    memset(gy, 0, sizeof(unsigned int) * this->pBits * this->pLen);
    memset(qx, 0, sizeof(unsigned int) * this->pBits * this->pLen);
    memset(qy, 0, sizeof(unsigned int) * this->pBits * this->pLen);
    memset(gqx, 0, sizeof(unsigned int) * this->pBits * this->pLen);
    memset(gqy, 0, sizeof(unsigned int) * this->pBits * this->pLen);

    ECPoint g(params.gx, params.gy);
    ECPoint q(params.qx, params.qy);
    ECPoint sum = this->curve.addPoint(g, q);

    if(!this->curve.pointExists(g)) {
        printf("G is not on the curve!\n");
    }

    if(!this->curve.pointExists(q)) {
        printf("Q is not on the curve!\n");
    }

    // Convert Gx, Gy
    BigInteger x = g.getX();
    BigInteger y = g.getY();
    x.getWords(gx);
    y.getWords(gy);

    // Convert Qx, Qy
    x = q.getX();
    y = q.getY();
    x.getWords(qx);
    y.getWords(qy); 

    // Convert G+Q
    x = sum.getX();
    y = sum.getY();
    x.getWords(gqx);
    y.getWords(gqy);

    // Generate 2G, 4G .. (2^130)G and 2Q, 4Q ... (2^130)Q
    for(unsigned int i = 1; i < this->pBits; i++) {
        g = this->curve.doublePoint(g);
        q = this->curve.doublePoint(q);
        sum = this->curve.addPoint(g, q);

        if(!this->curve.pointExists(g)) {
            printf("G is not on the curve!\n");
        }
        if(!this->curve.pointExists(q)) {
            printf("Q is not on the curve!\n");
        }
    
        if(!this->curve.pointExists(sum)) {
            printf("G + Q is not on the curve!\n");
        }

        x = g.getX(); 
        y = g.getY(); 
        x.getWords(&gx[i * pLen]);
        y.getWords(&gy[i * pLen]);
      
        x = q.getX();
        y = q.getY();
        x.getWords(&qx[i * pLen]);
        y.getWords(&qy[i * pLen]);

        x = sum.getX();
        y = sum.getY();
        x.getWords(&gqx[i * pLen]);
        y.getWords(&gqy[i * pLen]);
    }
}

void ECDLCudaContext::generateStartingPoints()
{    
    cudaError_t cudaError = cudaSuccess;

    unsigned int gx[ this->pBits * this->pLen ];
    unsigned int gy[ this->pBits * this->pLen ];
    unsigned int qx[ this->pBits * this->pLen ];
    unsigned int qy[ this->pBits * this->pLen ];
    unsigned int gqx[ this->pBits * this->pLen ];
    unsigned int gqy[ this->pBits * this->pLen ];

    generateLookupTable(gx, gy, qx, qy, gqx, gqy);

    unsigned int *devGx = NULL;
    unsigned int *devGy = NULL;
    unsigned int *devQx = NULL;
    unsigned int *devQy = NULL;
    unsigned int *devGQx = NULL;
    unsigned int *devGQy = NULL;

    try {
        unsigned int size = this->pBits * this->pLen * sizeof(unsigned int);
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
    generateMultipliersHost();

    // Reset points to point at infinity
    resetPoints(this->blocks, this->threadsPerBlock, this->devX, this->devY, this->pointsPerThread);

    Logger::logInfo("Multiplying points");
    for(unsigned int i = 0; i < this->pBits; i++) {
        cudaError = multiplyAddG( this->blocks, this->threadsPerBlock,
                                  this->devAStart, this->devBStart,
                                  devGx, devGy,
                                  devQx, devQy,
                                  devGQx, devGQy,
                                  this->devX, this->devY,
                                  this->devDiffBuf, this->devChainBuf,
                                  i, this->pointsPerThread );
        if( cudaError != cudaSuccess ) {
            goto end;
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

void ECDLCudaContext::writeBigIntToDevice( const unsigned int *x, unsigned int *dest, unsigned int threadId, unsigned int index )
{
    unsigned int numThreads = this->blocks * this->threadsPerBlock;

    for(int i = 0; i < this->pLen; i++) {
        CUDA::memcpy(&dest[ this->pLen * numThreads * index + numThreads * i + threadId ], &x[ i ], 4, cudaMemcpyHostToDevice);
    }
}


void ECDLCudaContext::readBigIntFromDevice( const unsigned int *src, unsigned int threadId, unsigned int index, unsigned int *x)
{
    unsigned int numThreads = this->threads;

    for(int i = 0; i < this->pLen; i++) {
        CUDA::memcpy(&x[ i ], &src[ this->pLen * numThreads * index + numThreads * i + threadId ], 4, cudaMemcpyDeviceToHost);
    }
}

void ECDLCudaContext::readXFromDevice(unsigned int threadId, unsigned int index, unsigned int *x)
{
    readBigIntFromDevice(this->devX, threadId, index, x);
}

void ECDLCudaContext::readYFromDevice(unsigned int threadId, unsigned int index, unsigned int *y)
{
    readBigIntFromDevice(this->devY, threadId, index, y);
}

void ECDLCudaContext::writeXToDevice(unsigned int *x, unsigned int threadId, unsigned int index)
{
    writeBigIntToDevice(x, this->devX, threadId, index);
}

void ECDLCudaContext::writeYToDevice(unsigned int *y, unsigned int threadId, unsigned int index)
{
    writeBigIntToDevice(y, this->devY, threadId, index);
}

/**
 * Allocates memory on the host and device according to the number of threads, and
 * number of simultanous computations per thread
 */
void ECDLCudaContext::allocateBuffers()
{
    size_t numPoints = this->blocks * this->threadsPerBlock * this->pointsPerThread;
    size_t arraySize = sizeof(unsigned int) * this->pLen * numPoints;

    Logger::logInfo("Allocating %ld bytes on device", arraySize * 4 );
    Logger::logInfo("Allocating %ld bytes on host", arraySize * 2 );
    Logger::logInfo("%d blocks", this->blocks );
    Logger::logInfo("%d threads (%d threads per block)", this->threads, this->threadsPerBlock);
    Logger::logInfo("%d points in parallel (%d points per thread)",
            this->threads * this->pointsPerThread, this->pointsPerThread);

    // Allocate 'a' values in host memory
    this->AStart = (unsigned int *)CUDA::hostAlloc(arraySize, cudaHostAllocMapped);
    memset( this->AStart, 0, arraySize );

    // Map host memory to device address space
    this->devAStart = (unsigned int *)CUDA::getDevicePointer(this->AStart, 0);

    // Allocate 'b' values in host memory
    this->BStart = (unsigned int *)CUDA::hostAlloc(arraySize, cudaHostAllocMapped );
    memset( this->BStart, 0, arraySize );

    // Map host memory to device address space
    this->devBStart = (unsigned int *)CUDA::getDevicePointer(this->BStart, 0);

    // Each thread gets a flag to notify that it has found a distinguished point
    this->pointFoundFlags = (unsigned int *)CUDA::hostAlloc(numPoints * sizeof(unsigned int), cudaHostAllocMapped);
    memset(this->pointFoundFlags, 0, numPoints * sizeof(unsigned int));

    // Each block gets a flag to notify if any threads in that block found a distinguished point
    this->blockFlags = (unsigned int *)CUDA::hostAlloc(this->blocks * sizeof(unsigned int), cudaHostAllocMapped);
    memset(this->blockFlags, 0, this->blocks * sizeof(unsigned int));

    // Allocate 'x' values in device memory
    this->devX = (unsigned int *)CUDA::malloc(arraySize);

    // Allocate 'y' values in device memory
    this->devY = (unsigned int*)CUDA::malloc(arraySize);

    // Allocate buffer to hold difference when computing point addition
    this->devDiffBuf = (unsigned int *)CUDA::malloc(arraySize);

    // Allocate buffer to hold the multiplication chain when computing batch inverse
    this->devChainBuf = (unsigned int *)CUDA::malloc(arraySize);

    // Allocate integer to be used as a flag if a distinguished point is found
    this->devPointFoundFlag = (unsigned int *)CUDA::malloc(4);
}

/**
 * Frees the memory allocated by allocateBuffers
 */
void ECDLCudaContext::freeBuffers()
{
    cudaFree(this->AStart);
    cudaFree(this->BStart);
    cudaFree(this->devX);
    cudaFree(this->devY);
    cudaFree(this->devDiffBuf);
    cudaFree(this->devChainBuf);
    cudaFree(this->devPointFoundFlag);
    cudaFree(this->pointFoundFlags);
    cudaFree(this->blockFlags);
}

/**
 * Verifies that aG + bQ = (x,y)
 */
bool ECDLCudaContext::verifyPoint(BigInteger &x, BigInteger &y)
{
    ECPoint p = ECPoint(x, y);

    if(!this->curve.pointExists(p)) {
        Logger::logError("x: %s\n", x.toString(16).c_str());
        Logger::logError("y: %s\n", y.toString(16).c_str());
        Logger::logError("Point is not on the curve\n");
        return false;
    }

    return true;
}

bool ECDLCudaContext::initializeDevice()
{
    // Set current device
    CUDA::setDevice(this->device);
    
    // Poll while kernel is running, but yield the thread if necessary
    //CUDA::setDeviceFlags(cudaDeviceScheduleYield);
    CUDA::setDeviceFlags(cudaDeviceScheduleBlockingSync);
    
    try {
        allocateBuffers();

        setupDeviceConstants();
        setRPoints();
    }catch(cudaError_t err) {
        Logger::logError("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }

    this->initialized = true;
    
    return true;
}

void ECDLCudaContext::uninitializeDevice()
{
    freeBuffers();
    cudaDeviceReset();
    this->initialized = false;
}

/**
 * Creates a new context
 */
ECDLCudaContext::ECDLCudaContext( int device, unsigned int blocks,
                                      unsigned int threads, unsigned int points,
                                      ECDLPParams *params,
                                      BigInteger *rx,
                                      BigInteger *ry,
                                      int rPoints,
                                      void (*callback)(struct CallbackParameters *) )
{
    this->blocks = blocks;
    this->threadsPerBlock = threads;
    this->pointsPerThread = points;
    this->threads = this->blocks * this->threadsPerBlock;
    this->totalPoints = blocks * threads * points;
    this->device = device;
    this->callback = callback;
    this->runFlag = true;
    this->params = *params;
    this->rPoints = rPoints; 
    this->p = params->p;

    this->pBits = this->p.getBitLength();
    this->pLen = (this->pBits + 31) / 32;

    // m is 4^k / p where k is the number of bits in p
    this->m = BigInteger(4).pow(this->pBits);
    this->m = this->m / this->p;
    this->mBits = this->m.getBitLength();
    this->mLen = (this->mBits + 31) / 32;

    Logger::logInfo("P bits: %d", this->pBits);
    Logger::logInfo("P words: %d", this->pLen);
    Logger::logInfo("P:      %s", this->p.toString().c_str());
    Logger::logInfo("M:    %s", this->m.toString().c_str());
    Logger::logInfo("M bits: %d", this->mBits);
    Logger::logInfo("M words: %d", this->mLen);

    // Copy random walk points
    Logger::logInfo("R points:");
    for(int i = 0; i < rPoints; i++) {
        this->rx[ i ] = rx[ i ];
        this->ry[ i ] = ry[ i ];
        Logger::logInfo("%.2x [%s,%s]", i, this->rx[i].toString().c_str(), this->ry[i].toString().c_str());
    }
    Logger::logInfo(""); 
    this->curve = ECCurve(this->params.p, this->params.n, this->params.a, this->params.b, this->params.gx, this->params.gy);
}

/**
 * Initializes the context with random points
 */
bool ECDLCudaContext::init()
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

/**
 * Frees all resources used by the context
 */
ECDLCudaContext::~ECDLCudaContext()
{
    if(this->initialized) {
        uninitializeDevice();
    }
}

bool ECDLCudaContext::getFlag()
{
    unsigned int flag = 0;
    CUDA::memcpy(&flag, this->devPointFoundFlag, 4, cudaMemcpyDeviceToHost);

    if(flag) {
        return true; 
    }

    return false;
}

/**
 * Runs the context
 */
bool ECDLCudaContext::run()
{
    unsigned int threadId = 0;
    unsigned long long iterations = 0;
    bool running = true;

    Logger::logInfo("Running");
    setRunFlag(true);

    do {
        cudaError_t cudaError = cudaSuccess;
        cudaError = doStep(this->blocks,
            this->threadsPerBlock,
            this->devX,
            this->devY,
            this->devDiffBuf,
            this->devChainBuf,
            this->devPointFoundFlag,
            NULL,
            this->blockFlags,
            this->pointFoundFlags,
            this->pointsPerThread);
        
        if(cudaError != cudaSuccess) {
            Logger::logError("CUDA error: %s\n", cudaGetErrorString(cudaError));
            break;
        }

        if(getFlag()) {

            for(unsigned int block = 0; block < this->blocks; block++) {
                if(!this->blockFlags[block]) {
                    continue;
                }
                this->blockFlags[block] = 0;
                for(unsigned int thread = 0; thread < this->threadsPerBlock; thread++) {
                    for(unsigned int pointIndex = 0; pointIndex < this->pointsPerThread; pointIndex++) {
                        unsigned int index = block * this->threadsPerBlock * this->pointsPerThread + thread * this->pointsPerThread + pointIndex;

                        // Check if a point was found
                        if(!this->pointFoundFlags[index]) {
                            continue;
                        }
                        this->pointFoundFlags[index] = 0;

                        threadId = block * this->threadsPerBlock + thread;

                        unsigned int x[this->pLen];
                        unsigned int y[this->pLen];
                        unsigned int Astart[this->pLen];
                        unsigned int Bstart[this->pLen];

                        memset(x, 0, sizeof(x));
                        memset(y, 0, sizeof(x));
                        memset(Astart, 0, sizeof(x));
                        memset(Bstart, 0, sizeof(x));

                        readXFromDevice(threadId, pointIndex, x);
                        readYFromDevice(threadId, pointIndex, y);
                        extractBigInt(this->AStart, block, thread, pointIndex, Astart);
                        extractBigInt(this->BStart, block, thread, pointIndex, Bstart);

                        BigInteger xBig(x, this->pLen);
                        BigInteger yBig(y, this->pLen);
                        BigInteger aStartBig(Astart, this->pLen);
                        BigInteger bStartBig(BStart, this->pLen);

                        if(!verifyPoint(xBig, yBig)) {
                            Logger::logError( "==== INVALID POINT ====\n" );
                            printf("Index: %d\n", pointIndex);
                            printf("Thread: %d\n", thread);
                            printf("Block: %d\n", block);
                            Logger::logError("%s %s\n", aStartBig.toString(16).c_str(), bStartBig.toString(16).c_str());
                            Logger::logError("[%s, %s]\n", xBig.toString(16).c_str(), yBig.toString(16).c_str());
                            printf("a: ");
                            printBigInt(AStart, this->pLen);
                            printf("b: ");
                            printBigInt(BStart, this->pLen);
                            printf("x: ");
                            printBigInt(x, this->pLen);
                            printf("y: ");
                            printBigInt(y, this->pLen);
                            return false;
                        }
                        
                        struct CallbackParameters p;
                        p.aStart = aStartBig;
                        p.bStart = bStartBig;
                        p.x = xBig;
                        p.y = yBig;

                        callback(&p);
                        
                        unsigned int newX[this->pLen];
                        unsigned int newY[this->pLen];
                        unsigned int newA[this->pLen];
                        unsigned int newB[this->pLen]; 
                        getRandomPoint( newX, newY, newA, newB );
                        
                        // Write a, b to host memory
                        splatBigInt( newA, this->AStart, threadId, pointIndex );
                        splatBigInt( newB, this->BStart, threadId, pointIndex );

                        // Write x, y to device memory
                        writeXToDevice( newX, threadId, pointIndex );
                        writeYToDevice( newY, threadId, pointIndex );
                    }
                }
            }
        }


        running = isRunning();
        iterations++;
    }while( running );

    return true;
}

bool ECDLCudaContext::benchmark( unsigned long long *pointsPerSecond )
{
    unsigned int t0 = 0;
    unsigned int t1 = 0;
    unsigned int count = 10000;
    bool success = true;
    float seconds = 0; 
    t0 = util::getSystemTime();
    for(unsigned int i = 0; i < count; i++) {
        cudaError_t cudaError = cudaSuccess;
        
        cudaError = doStep(this->blocks,
            this->threadsPerBlock,
            this->devX,
            this->devY,
            this->devDiffBuf,
            this->devChainBuf,
            this->devPointFoundFlag,
            NULL,
            this->blockFlags,
            this->pointFoundFlags,
            this->pointsPerThread); 

        if(cudaError != cudaSuccess) {
            Logger::logError("CUDA error: %s\n", cudaGetErrorString( cudaError ));
            success = false;
            goto end;
        }

        i++;
    }
    t1 = util::getSystemTime();
    seconds = (float)(t1 - t0)/1000;
    Logger::logInfo("%d iterations in %dms (%d iterations per second)", count, (t1-t0), (int)((float)(count/seconds)));
    Logger::logInfo("%lld points per second", ((unsigned long long)count * this->threads * this->pointsPerThread*1000)/(t1 - t0));
    if( pointsPerSecond != NULL ) {
        *pointsPerSecond = ((unsigned long long)count * this->threads * this->pointsPerThread*1000)/(t1 - t0);
    }

end:

    return success;
}
