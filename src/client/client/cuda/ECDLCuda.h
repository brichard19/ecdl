#ifndef _ECDL_CUDA_H
#define _ECDL_CUDA_H

#include "ecc.h"
#include "uint160.h"
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
    BigInteger pInv;
    BigInteger r;
    BigInteger rInv;
    BigInteger rModP;
    BigInteger pMinus2;

    // The current device
    int device;
    bool initialized;
    
    void (*callback)(struct CallbackParameters *);

    uint160 readXFromDevice(unsigned int threadId, unsigned int index);
    uint160 readYFromDevice(unsigned int threadId, unsigned int index);
    uint160 readAFromDevice(unsigned int threadId, unsigned int index);
    uint160 readBFromDevice(unsigned int threadId, unsigned int index);
    void writeXToDevice(uint160 &value, unsigned int threadId, unsigned int index);
    void writeYToDevice(uint160 &value, unsigned int threadId, unsigned int index);
    void writeAToDevice(uint160 &value, unsigned int threadId, unsigned int index);
    void writeBToDevice(uint160 &value, unsigned int threadId, unsigned int index);
    void splatUint160(uint160 &x, unsigned int *ara, int thread, int index );
    uint160 extractUint160( unsigned int *ara, int thread, int index );
    void generateMultipliersHost();
    void writeUint160ToDevice(uint160 &x, unsigned int *dest, unsigned int threadId, unsigned int index);
    uint160 readUint160FromDevice(unsigned int *src, unsigned int threadId, unsigned int index);
    void generateStartingPoints();
    void allocateBuffers();
    void freeBuffers();
    void uninitializeDevice();
    bool initializeDevice();
    bool getFlag();
    void getRandomPoint(uint160 &x, uint160 &y, uint160 &a, uint160 &b);
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
