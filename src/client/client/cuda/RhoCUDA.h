#ifndef _RHO_CUDA_H
#define _RHH_CUDA_H

#include "ECDLContext.h"
#include "ecc.h"
#include "BigInteger.h"
#include <cuda_runtime.h>

class RhoCUDA {

private:
    unsigned long long _mainCounter;
    volatile bool _runFlag;

    unsigned int _pointsPerThread; 
    unsigned int _blocks;
    unsigned int _threadsPerBlock;
    unsigned int _numThreads;
    unsigned int _numRPoints;

    /**
     * Pointers to host memory
     */
    unsigned int *_aStart;
    unsigned int *_bStart;
    unsigned int *_pointFoundFlags;
    unsigned int *_blockFlags;
    unsigned long long *_counters;

    /**
     * Pointers to host memory from device
     */
    unsigned int *_devAStart;
    unsigned int *_devBStart;
    unsigned int *_devBlockFlags;
    unsigned int *_devPointFoundFlags;

    /**
     * Pointers to device memory
     */
    unsigned int *_devDiffBuf;
    unsigned int *_devChainBuf;
    unsigned int *_devX;
    unsigned int *_devY;

    ECDLPParams _params;
    ECCurve _curve;

    BigInteger _rx[NUM_R_POINTS];
    BigInteger _ry[NUM_R_POINTS];

    int _pBits;
    int _mBits;
    int _pWords;
    int _mWords;
    BigInteger _m;

    // The current device
    int _device;
    bool _initialized;
    
    void (*_callback)(struct CallbackParameters *);

    void readX(unsigned int *x, unsigned int block, unsigned int thread, unsigned int index);
    void readY(unsigned int *y, unsigned int block, unsigned int thread, unsigned int index);
    void readA(unsigned int *a, unsigned int block, unsigned int thread, unsigned int index);
    void readB(unsigned int *b, unsigned int block, unsigned int thread, unsigned int index);

    void writeX(const unsigned int *x, unsigned int block, unsigned int thread, unsigned int index);
    void writeY(const unsigned int *y, unsigned int block, unsigned int thread, unsigned int index);
    void writeA(const unsigned int *a, unsigned int block, unsigned int thread, unsigned int index);
    void writeB(const unsigned int *b, unsigned int block, unsigned int thread, unsigned int index);

    void splatBigInt(unsigned int *ara, const unsigned int *x, unsigned int block, unsigned int thread, unsigned int index );
    void extractBigInt(unsigned int *x, const unsigned int *ara, unsigned int block, unsigned int thread, unsigned int index);

    void writeBigInt(unsigned int *dest, const unsigned int *src, unsigned int block, unsigned int thread, unsigned int index);
    void readBigInt(unsigned int *dest, const unsigned int *src, unsigned int block, unsigned int thread, unsigned int index);

    unsigned int getIndex(unsigned int block, unsigned int thread, unsigned int idx);

    // Initializaton
    void generateLookupTable(unsigned int *gx, unsigned int *gy, unsigned int *qx, unsigned int *qy, unsigned int *gqx, unsigned int *gqy);

    void generateExponentsHost();
    void generateStartingPoints(bool doVerify = false);
    void allocateBuffers();
    void freeBuffers();
    void setupDeviceConstants();

    void uninitializeDevice();
    bool initializeDevice();
    bool getFlag();
    void getRandomPoint(unsigned int *x, unsigned int *y, unsigned int *a, unsigned int *b);
    bool verifyPoint(BigInteger &x, BigInteger &y);
    void setRunFlag(bool flag);
    void setRPoints();
    bool pointFound();

    bool doStep();

    void cudaException(cudaError_t error);

public:
    RhoCUDA( int device,
           unsigned int blocks,
           unsigned int threads,
           unsigned int pointsPerThread,
           const ECDLPParams *params,
           const BigInteger *rx,
           const BigInteger *ry,
           int rPoints,
           void (*callback)(struct CallbackParameters *));

    ~RhoCUDA();
    bool init();
    void reset();
    bool run();
    bool stop();
    bool isRunning();

    // Debug code
    bool benchmark(unsigned long long *pointsPerSecond);
};

#endif