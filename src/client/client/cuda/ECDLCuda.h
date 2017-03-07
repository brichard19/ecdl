#ifndef _ECDL_CUDA_H
#define _ECDL_CUDA_H

#include "ecc.h"
#include "BigInteger.h"
#include "ECDLContext.h"
#include "RhoCUDA.h"
#include "cudapp.h"

class ECDLCudaContext : public ECDLContext {

private:
    RhoCUDA *_rho;
    ECDLPParams _params;
    ECCurve _curve;
    void (*_callback)(struct CallbackParameters *);

    BigInteger _rx[NUM_R_POINTS];
    BigInteger _ry[NUM_R_POINTS];

    unsigned int _device;
    unsigned int _blocks;
    unsigned int _threads;
    unsigned int _totalPoints;
    int _pointsPerThread;
    int _rPoints;

public:

    virtual ~ECDLCudaContext();
    virtual bool init();
    virtual void reset();
    virtual bool run();
    virtual bool stop();
    virtual bool isRunning();

    ECDLCudaContext( int device,
                       unsigned int blocks,
                       unsigned int threads,
                       unsigned int pointsPerThread,
                       const ECDLPParams *params,
                       const BigInteger *rx,
                       const BigInteger *ry,
                       int rPoints,
                       void (*callback)(struct CallbackParameters *));

    virtual bool benchmark(unsigned long long *pointsPerSecond);
};

#endif
