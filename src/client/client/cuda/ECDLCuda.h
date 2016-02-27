#ifndef _ECDL_CUDA_H
#define _ECDL_CUDA_H

#include "ecc.h"
#include "BigInteger.h"
#include "ECDLContext.h"
#include "cudapp.h"

class ECDLCudaContext : public ECDLContext {

private:
    unsigned long long _mainCounter;
    bool _runFlag;

    unsigned int _pointsInParallel;
    unsigned int _totalPoints;
    unsigned int _blocks;
    unsigned int _threadsPerBlock;
    unsigned int _numRPoints;

    /**
     * Pointers to host memory
     */
    unsigned int *_aStart;
    unsigned int *_bStart;
    unsigned int *_pointFoundFlags;
    unsigned long long *_counters;

    /**
     * Pointers to host memory from device
     */
    unsigned int *_devAStart;
    unsigned int *_devBStart;
    unsigned int *_devPointFoundFlag;
    unsigned int *_devPointFoundFlags;

    unsigned int *_sectionFlags;

    /**
     * Pointers to device memory
     */
    unsigned int *_devDiffBuf;
    unsigned int *_devChainBuf;
    unsigned int *_devX;
    unsigned int *_devY;

    ECDLPParams _params;
    ECCurve _curve;

    BigInteger _rx[ NUM_R_POINTS ];
    BigInteger _ry[ NUM_R_POINTS ];

    //BigInteger p;
    unsigned int _pBits;
    unsigned int _mBits;
    unsigned int _pWords;
    unsigned int _mWords;
    BigInteger _m;

    // The current device
    int _device;
    bool _initialized;
    
    void (*_callback)(struct CallbackParameters *);

    void readXFromDevice(unsigned int index, unsigned int *x);
    void readYFromDevice(unsigned int index, unsigned int *y);
    void readAFromDevice(unsigned int index, unsigned int *a);
    void readBFromDevice(unsigned int index, unsigned int *b);
    void writeXToDevice(unsigned int *x, unsigned int index);
    void writeYToDevice(unsigned int *y, unsigned int index);
    void writeAToDevice(unsigned int *a, unsigned int index);
    void writeBToDevice(unsigned int *b, unsigned int index);
    void generateLookupTable(unsigned int *gx, unsigned int *gy, unsigned int *qx, unsigned int *qy, unsigned int *gqx, unsigned int *gqy);
    void splatBigInt(const unsigned int *x, unsigned int *ara, int index );
    void extractBigInt(const unsigned int *ara, int index, unsigned int *x);

    void generateExponentsHost();
    void writeBigIntToDevice(const unsigned int *x, unsigned int *dest, unsigned int index );
    void readBigIntFromDevice(const unsigned int *src, unsigned int index, unsigned int *x);
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
    bool pointFound();

    inline unsigned int totalThreads()
    {
        return _threadsPerBlock * _blocks;
    }

    inline unsigned int totalSections()
    {
        return (_totalPoints + 31)/32;
    }

public:
    ECDLCudaContext( int device,
                       unsigned int blocks,
                       unsigned int threads,
                       unsigned int totalPoints,
                       unsigned int pointsInParallel,
                       const ECDLPParams *params,
                       const BigInteger *rx,
                       const BigInteger *ry,
                       int rPoints,
                       void (*callback)(struct CallbackParameters *));

    virtual ~ECDLCudaContext();
    virtual bool init();
    virtual bool run();
    virtual bool stop();
    virtual bool isRunning();

    // Debug code
    virtual bool benchmark(unsigned long long *pointsPerSecond);
};

#endif
