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

/**
 * Takes a uint160 and 'splats' its contents into coalesced form that the GPU uses
 */
void ECDLCudaContext::splatUint160(uint160 &x, unsigned int *ara, int thread, int index)
{
    unsigned int numThreads = threadsPerBlock * blocks;
    ara[ 5 * numThreads * index + thread ] = x.v[ 0 ];
    ara[ 5 * numThreads * index + numThreads + thread ] = x.v[ 1 ];
    ara[ 5 * numThreads * index + numThreads * 2 + thread ] = x.v[ 2 ];
    ara[ 5 * numThreads * index + numThreads * 3 + thread ] = x.v[ 3 ];
    ara[ 5 * numThreads * index + numThreads * 4 + thread ] = x.v[ 4 ];
}

/**
 * Performs the opposite of splatUint160. Reads an integer from coalesced form
 * into a uint160
 */
uint160 ECDLCudaContext::extractUint160(unsigned int *ara, int thread, int index)
{
    uint160 x;
    unsigned int numThreads = threadsPerBlock * blocks;

    x.v[ 0 ] = ara[ 5 * numThreads * index + thread ];
    x.v[ 1 ] = ara[ 5 * numThreads * index + numThreads + thread ];
    x.v[ 2 ] = ara[ 5 * numThreads * index + numThreads * 2 + thread ];
    x.v[ 3 ] = ara[ 5 * numThreads * index + numThreads * 3 + thread ];
    x.v[ 4 ] = ara[ 5 * numThreads * index + numThreads * 4 + thread ];

    return x;
}

void ECDLCudaContext::setRPoints()
{
    uint160 rx[ this->rPoints ];
    uint160 ry[ this->rPoints ];

    // Convert to uint160
    for(int i = 0; i < this->rPoints; i++) {
        BigInteger t = util::toMontgomery(this->rx[ i ], this->r, this->p);
        rx[ i ] = fromBigInteger(t);
        t = util::toMontgomery(this->ry[ i ], this->r, this->p);
        ry[ i ] = fromBigInteger(t);
    }

    cudaError_t cudaError = copyRPointsToDevice(rx, ry, this->rPoints);
    if( cudaError != cudaSuccess ) {
        throw cudaError;
    }
}

/**
 * Generates required constants and copies them to the GPU constant memory
 */
void ECDLCudaContext::setupDeviceConstants()
{
    uint160 gx[ this->pBits ];
    uint160 gy[ this->pBits ];
    uint160 qx[ this->pBits ];
    uint160 qy[ this->pBits ];

    ECPoint g(params.gx, params.gy);
    ECPoint q(params.qx, params.qy);

    if(!this->curve.pointExists(g)) {
        printf("G is not on the curve!\n");
    }
    if(!this->curve.pointExists(q)) {
        printf("Q is not on the curve!\n");
    }

    BigInteger gxMontgomery;
    BigInteger gyMontgomery;
    BigInteger qxMontgomery;
    BigInteger qyMontgomery;

    // Point G in montgomery form
    gxMontgomery = util::toMontgomery(g.getX(), this->r, this->p);
    gx[ 0 ] = fromBigInteger(gxMontgomery);
    gyMontgomery = util::toMontgomery(g.getY(), this->r, this->p);
    gy[ 0 ] = fromBigInteger(gyMontgomery);

    // Point Q in montgomery form
    qxMontgomery = util::toMontgomery(q.getX(), this->r, this->p);
    qx[ 0 ] = fromBigInteger( qxMontgomery );
    qyMontgomery = util::toMontgomery(q.getY(), this->r, this->p);
    qy[ 0 ] = fromBigInteger( qyMontgomery );

    // Generate 2G, 4G .. (2^130)G and 2Q, 4Q ... (2^130)Q
    for(unsigned int i = 1; i < this->pBits; i++) {
        g = this->curve.doublePoint(g);
        q = this->curve.doublePoint(q);

        if(!this->curve.pointExists(g)) {
            printf("G is not on the curve!\n");
        }
        if(!this->curve.pointExists(q)) {
            printf("Q is not on the curve!\n");
        }
      
        BigInteger m = util::toMontgomery(g.getX(), this->r, this->p);
        gx[ i ] = fromBigInteger(m);
        
        m = util::toMontgomery(g.getY(), this->r, this->p);
        gy[ i ] = fromBigInteger(m);

        m = util::toMontgomery(q.getX(), this->r, this->p);
        qx[ i ] = fromBigInteger(m);
        
        m = util::toMontgomery(q.getY(), this->r, this->p);
        qy[ i ] = fromBigInteger(m);
    }

    // Copy points to device
    cudaError_t cudaError = cudaSuccess;

    cudaError = copyMultiplesToDevice(gx, gy, qx, qy);
    if(cudaError != cudaSuccess) {
        throw cudaError;
    }

    cudaError = initDeviceParams(fromBigInteger(this->p),
                                 fromBigInteger(this->pInv),
                                 fromBigInteger(this->pMinus2),
                                 fromBigInteger(this->rModP),
                                 this->r.getBitLength(),
                                 params.dBits);
    
    if(cudaError != cudaSuccess) {
        throw cudaError;
    }
}

/**
 * Generates a random point n the form aG + bQ. 
 */
void ECDLCudaContext::getRandomPoint(uint160 &x, uint160 &y, uint160 &a, uint160 &b)
{
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

    // Convert X and Y to montgomery form
    BigInteger xMontgomery = util::toMontgomery(sum.getX(), this->r, this->p);
    BigInteger yMontgomery = util::toMontgomery(sum.getY(), this->r, this->p);

    // Convert to uint160 type
    x = fromBigInteger(xMontgomery);
    y = fromBigInteger(yMontgomery);
    a = fromBigInteger(m1);
    b = fromBigInteger(m2);
}

/**
 * Generates random 'a' and 'b' values
 */
void ECDLCudaContext::generateMultipliersHost()
{
    for(unsigned int index = 0; index < pointsPerThread; index++) {
        for(unsigned int thread = 0; thread < blocks * threadsPerBlock; thread++) {
            // TODO: Use better RNG here
            BigInteger m1 = randomBigInteger(params.n);
            BigInteger m2 = randomBigInteger(params.n);
            uint160 a = fromBigInteger(m1);
            uint160 b = fromBigInteger(m2);
            splatUint160(a, AStart, thread, index);
            splatUint160(b, BStart, thread, index);
        }
    }
}

void ECDLCudaContext::generateStartingPoints()
{    
    cudaError_t cudaError = cudaSuccess;

    // Generate 'a' and 'b'
    generateMultipliersHost();

    // Reset points to point at infinity
    resetPoints(this->blocks, this->threadsPerBlock, this->devX, this->devY, this->pointsPerThread);

    Logger::logInfo("Multiplying points P");
    for(unsigned int i = 0; i < this->pBits; i++) {
        cudaError = multiplyAddG( this->blocks, this->threadsPerBlock,
                                  this->devAStart,
                                  this->devX, this->devY,
                                  this->devDiffBuf, this->devChainBuf,
                                  i, this->pointsPerThread );
        if( cudaError != cudaSuccess ) {
            throw cudaError;
        }
    }
 

    uint160 xMontgomery = readUint160FromDevice(this->devX, 0, 0);
    uint160 yMontgomery = readUint160FromDevice(this->devY, 0, 0);

    BigInteger x = util::fromMontgomery(toBigInteger(xMontgomery), this->rInv, this->p); 
    BigInteger y = util::fromMontgomery(toBigInteger(yMontgomery), this->rInv, this->p);

    // Compute P = P + bQ. Each kernel call performs 1 point addition
    Logger::logInfo( "Multiplying points Q" );
    for(unsigned int i = 0; i < this->pBits; i++) {
        cudaError = multiplyAddQ( this->blocks, this->threadsPerBlock,
                                  this->devBStart,
                                  this->devX, this->devY,
                                  this->devDiffBuf, this->devChainBuf,
                                  i, this->pointsPerThread );
        if( cudaError != cudaSuccess ) {
            throw cudaError;
        }
    }
}

void ECDLCudaContext::writeUint160ToDevice( uint160 &x, unsigned int *dest, unsigned int threadId, unsigned int index )
{
    unsigned int numThreads = this->blocks * this->threadsPerBlock;

    CUDA::memcpy(&dest[ 5 * numThreads * index + threadId ], &x.v[ 0 ], 4, cudaMemcpyHostToDevice);
    CUDA::memcpy(&dest[ 5 * numThreads * index + numThreads + threadId ], &x.v[ 1 ], 4, cudaMemcpyHostToDevice);
    CUDA::memcpy(&dest[ 5 * numThreads * index + numThreads * 2 + threadId ], &x.v[ 2 ], 4, cudaMemcpyHostToDevice);
    CUDA::memcpy(&dest[ 5 * numThreads * index + numThreads * 3 + threadId ], &x.v[ 3 ], 4, cudaMemcpyHostToDevice);
    CUDA::memcpy(&dest[ 5 * numThreads * index + numThreads * 4 + threadId ], &x.v[ 4 ], 4, cudaMemcpyHostToDevice);
}


uint160 ECDLCudaContext::readUint160FromDevice( unsigned int *src, unsigned int threadId, unsigned int index )
{
    uint160 x;
    unsigned int numThreads = this->threads;

    CUDA::memcpy(&x.v[ 0 ], &src[ 5 * numThreads * index + threadId ], 4, cudaMemcpyDeviceToHost);
    CUDA::memcpy(&x.v[ 1 ], &src[ 5 * numThreads * index + numThreads + threadId ], 4, cudaMemcpyDeviceToHost);
    CUDA::memcpy(&x.v[ 2 ], &src[ 5 * numThreads * index + numThreads * 2 + threadId ], 4, cudaMemcpyDeviceToHost);
    CUDA::memcpy(&x.v[ 3 ], &src[ 5 * numThreads * index + numThreads * 3 + threadId ], 4, cudaMemcpyDeviceToHost);
    CUDA::memcpy(&x.v[ 4 ], &src[ 5 * numThreads * index + numThreads * 4 + threadId ], 4, cudaMemcpyDeviceToHost);

    return x;
}

uint160 ECDLCudaContext::readXFromDevice(unsigned int threadId, unsigned int index)
{
    return readUint160FromDevice(this->devX, threadId, index);
}

uint160 ECDLCudaContext::readYFromDevice(unsigned int threadId, unsigned int index)
{
    return readUint160FromDevice(this->devY, threadId, index);
}

void ECDLCudaContext::writeXToDevice(uint160 &value, unsigned int threadId, unsigned int index)
{
    writeUint160ToDevice(value, this->devX, threadId, index);
}

void ECDLCudaContext::writeYToDevice(uint160 &value, unsigned int threadId, unsigned int index)
{
    writeUint160ToDevice(value, this->devY, threadId, index);
}

/**
 * Allocates memory on the host and device according to the number of threads, and
 * number of simultanous computations per thread
 */
void ECDLCudaContext::allocateBuffers()
{
    size_t numValues = this->blocks * this->threadsPerBlock * this->pointsPerThread;
    size_t totalSize = sizeof(uint160) * numValues;

    Logger::logInfo("Allocating %ld bytes on device", totalSize * 4 );
    Logger::logInfo("Allocating %ld bytes on host", totalSize * 2 );
    Logger::logInfo("%d blocks", this->blocks );
    Logger::logInfo("%d threads (%d threads per block)", this->threads, this->threadsPerBlock);
    Logger::logInfo("%d points in parallel (%d points per thread)",
            this->threads * this->pointsPerThread, this->pointsPerThread);

    // Allocate 'a' values in host memory
    this->AStart = (unsigned int *)CUDA::hostAlloc(totalSize, cudaHostAllocMapped);
    memset( this->AStart, 0, totalSize );

    // Map host memory to device address space
    this->devAStart = (unsigned int *)CUDA::getDevicePointer(this->AStart, 0);

    // Allocate 'b' values in host memory
    this->BStart = (unsigned int *)CUDA::hostAlloc(totalSize, cudaHostAllocMapped );
    memset( this->BStart, 0, totalSize );

    // Each thread gets a flag to notify that it has found a distinguished point
    this->pointFoundFlags = (unsigned int *)CUDA::hostAlloc(numValues * sizeof(unsigned int), cudaHostAllocMapped);
    memset(this->pointFoundFlags, 0, numValues * sizeof(unsigned int));

    // Each block gets a flag to notify if any threads in that block found a distinguished point
    this->blockFlags = (unsigned int *)CUDA::hostAlloc(this->blocks * sizeof(unsigned int), cudaHostAllocMapped);
    memset(this->blockFlags, 0, this->blocks * sizeof(unsigned int));

    // Map host memory to device address space
    this->devBStart = (unsigned int *)CUDA::getDevicePointer(this->BStart, 0);
    
    // Allocate 'x' values in device memory
    this->devX = (unsigned int *)CUDA::malloc(totalSize);

    // Allocate 'y' values in device memory
    this->devY = (unsigned int*)CUDA::malloc(totalSize);

    // Allocate buffer to hold difference when computing point addition
    this->devDiffBuf = (unsigned int *)CUDA::malloc(totalSize);

    // Allocate buffer to hold the multiplication chain when computing batch inverse
    this->devChainBuf = (unsigned int *)CUDA::malloc(totalSize);

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
        Logger::logError("x: %s\n", x.toString().c_str());
        Logger::logError("y: %s\n", y.toString().c_str());
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
    this->device = device;
    this->callback = callback;
    this->runFlag = true;
    this->params = *params;
    this->rPoints = rPoints; 
    this->p = params->p;
 
    // Compute R and R inverse. R is the smallest power of
    // 2 that is greater than P
    this->pBits = this->p.getBitLength();
    this->r = BigInteger(2).pow(this->pBits);
    this->rInv = this->r.invm(this->p);

    // -(P^-1) mod R
    this->pInv = this->p.invm(this->r);
    this->pInv = (this->r - this->pInv) % this->r;

    // P minus 2 (used for computing inverse mod P)
    this->pMinus2 = this->p - BigInteger(2); 

    // 1 in montgomery form
    this->rModP = this->r % this->p;

    Logger::logInfo("P bits: %d", this->pBits);
    Logger::logInfo("P:      %s", this->p.toString().c_str());
    Logger::logInfo("R:      %s", this->r.toString().c_str());
    Logger::logInfo("R bits: %d", this->r.getBitLength());
    Logger::logInfo("Pinv:   %s", this->pInv.toString().c_str());
    Logger::logInfo("Rinv:   %s", this->rInv.toString().c_str());
    Logger::logInfo("P-2:    %s", this->pMinus2.toString().c_str());
    Logger::logInfo("RmodP:  %s", this->rModP.toString().c_str());

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

                        uint160 x = readXFromDevice(threadId, pointIndex);
                        uint160 y = readYFromDevice(threadId, pointIndex);
                        uint160 Astart = extractUint160(this->AStart, threadId, pointIndex);
                        uint160 Bstart = extractUint160(this->BStart, threadId, pointIndex);

                        BigInteger xm = toBigInteger(x);
                        BigInteger ym = toBigInteger(y);
                        BigInteger xBig = util::fromMontgomery(xm, this->rInv, this->p);
                        BigInteger yBig = util::fromMontgomery(ym, this->rInv, this->p);
                        BigInteger aStartBig = toBigInteger(Astart);
                        BigInteger bStartBig = toBigInteger(Bstart);

                        printf("Found point\n");
                        printf("%lld iterations\n", iterations);
                        if(!verifyPoint(xBig, yBig)) {
                            Logger::logError("%s %s\n", aStartBig.toString().c_str(), bStartBig.toString().c_str());
                            Logger::logError("[%s, %s]\n", xm.toString().c_str(), ym.toString().c_str());
                            Logger::logError("[%s, %s]\n", xBig.toString().c_str(), yBig.toString().c_str());
                            Logger::logError( "==== INVALID POINT ====\n" );
                            return false;
                        }
                        
                        struct CallbackParameters p;
                        p.aStart = aStartBig;
                        p.bStart = bStartBig;
                        p.x = xm;
                        p.y = ym;

                        callback(&p);
                        
                        // Generate new point
                        uint160 newX;
                        uint160 newY;
                        uint160 newA;
                        uint160 newB;
                        
                        getRandomPoint( newX, newY, newA, newB );
                        
                        // Write a, b to host memory
                        splatUint160( newA, this->AStart, threadId, pointIndex );
                        splatUint160( newB, this->BStart, threadId, pointIndex );

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
