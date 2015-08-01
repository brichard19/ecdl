#ifndef _ECDL_CUDA_H
#define _ECDL_CUDA_H

#include "ecc.h"
#include "BigInteger.h"
#include "ECDLContext.h"
#include "cudapp.h"

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

struct CallbackParameters {
    BigInteger aStart;
    BigInteger bStart;
    BigInteger x;
    BigInteger y;
};

class ECDLCudaContext : public ECDLContext {

private:
    bool runFlag;
    unsigned int pointsPerThread;
    unsigned int blocks;
    unsigned int threads;
    unsigned int threadsPerBlock;
    int rPoints;
    unsigned int totalPoints;

    /**
     * Pointers to host memory
     */
    unsigned int *AStart;
    unsigned int *BStart;
    unsigned int pointFoundFlag;
    unsigned int *pointFoundFlags;

    /**
     * Pointers to host memory from device
     */
    unsigned int *devAStart;
    unsigned int *devBStart;
    unsigned int *devPointFoundFlag;
    unsigned int *devPointFoundFlags;
    unsigned int *blockFlags;

    /**
     * Pointers to device memory
     */
    unsigned int *devDiffBuf;
    unsigned int *devChainBuf;
    unsigned int *devX;
    unsigned int *devY;

    ECDLPParams params;
    ECCurve curve;

    BigInteger rx[ 32 ];
    BigInteger ry[ 32 ];

    BigInteger p;
    unsigned int pBits;
    unsigned int mBits;
    unsigned int pLen;
    unsigned int mLen;
    unsigned int p2Len;
    BigInteger m;

    // The current device
    int device;
    bool initialized;
    
    void (*callback)(struct CallbackParameters *);

    void readXFromDevice(unsigned int threadId, unsigned int index, unsigned int *x);
    void readYFromDevice(unsigned int threadId, unsigned int index, unsigned int *y);
    void readAFromDevice(unsigned int threadId, unsigned int index, unsigned int *a);
    void readBFromDevice(unsigned int threadId, unsigned int index, unsigned int *b);
    void writeXToDevice(unsigned int *x, unsigned int threadId, unsigned int index);
    void writeYToDevice(unsigned int *y, unsigned int threadId, unsigned int index);
    void writeAToDevice(unsigned int *a, unsigned int threadId, unsigned int index);
    void writeBToDevice(unsigned int *b, unsigned int threadId, unsigned int index);
    void generateLookupTable(unsigned int *gx, unsigned int *gy, unsigned int *qx, unsigned int *qy, unsigned int *gqx, unsigned int *gqy);
    void splatBigInt(const unsigned int *x, unsigned int *ara, int thread, int index );
    void splatBigInt(const unsigned int *x, unsigned int *ara, int block, int thread, int index );
    void extractBigInt(const unsigned int *ara, int thread, int index, unsigned int *x);
    void extractBigInt(const unsigned int *ara, int block, int thread, int index, unsigned int *x);

    void generateMultipliersHost();
    void writeBigIntToDevice(const unsigned int *x, unsigned int *dest, unsigned int threadId, unsigned int index );
    void readBigIntFromDevice(const unsigned int *src, unsigned int threadId, unsigned int index, unsigned int *x);
    void generateStartingPoints();
    void allocateBuffers();
    void freeBuffers();
    void uninitializeDevice();
    bool initializeDevice();
    bool getFlag();
    void getRandomPoint(unsigned int *x, unsigned int *y, unsigned int *a, unsigned int *b);
    void setupDeviceConstants();
    bool verifyPoint(BigInteger &x, BigInteger &y);
    void setRunFlag(bool flag);
    void setRPoints();

public:
    ECDLCudaContext( int device,
                       unsigned int blocks,
                       unsigned int threads,
                       unsigned int points,
                       ECDLPParams *params,
                       BigInteger *rx,
                       BigInteger *ry,
                       int rPoints,
                       void (*callback)(struct CallbackParameters *));

    ~ECDLCudaContext();
    virtual bool init();
    virtual bool run();
    virtual bool stop();
    virtual bool isRunning();

    // Debug code
    virtual bool benchmark(unsigned long long *pointsPerSecond);
};

#endif
