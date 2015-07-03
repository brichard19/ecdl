#ifndef _ECDL_CPU_H
#define _ECDL_CPU_H

#include "threads.h"
#include "BigInteger.h"
#include "ecc.h"

#include "ECDLContext.h"

/**
 Elliptic curve discrete logarithm parameters
 */
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

/**
 Parameters passed to the callback when a distinguished point is found
 */
struct CallbackParameters {
    BigInteger aStart;
    BigInteger bStart;
    BigInteger x;
    BigInteger y;
};

typedef struct {
    volatile bool *running;
    unsigned int threadId;
}WorkerThreadParams;

typedef struct {
    unsigned int threadId;
    unsigned int iterations;
    unsigned int t;
}BenchmarkThreadParams;

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
                        );


void cleanupThreadGlobals();
void *workerThreadFunction(void *p);
void *benchmarkThreadFunction(void *p);


class ECDLCpuContext : public ECDLContext {

private:
    // Number of threads
    int _numThreads;

    // Thread handles
    Thread *_threads;

    // Parameters for each thread
    WorkerThreadParams *_threadParams;

    // Flag to indicate if the threads are running
    bool _running;

    // Callback that gets called when distinguished point is found 
    void (*_callback)(struct CallbackParameters *);

    // Number of points each thread works on in parallel
    unsigned int _pointsPerThread;

    // R-points
    int _rPoints;
    BigInteger _rx[ 32 ];
    BigInteger _ry[ 32 ];

    // Problem parameters
    ECDLPParams _params;
    ECCurve _curve;

    // multipliers for the starting points aG + bQ
    BigInteger *_aStart;
    BigInteger *_bStart;

    // X and Y values for the starting points
    BigInteger *_x;
    BigInteger *_y;

    BigInteger _p;

    void generateStartingPoints();
public:

    virtual bool init();
    virtual bool run();
    virtual bool stop();
    virtual bool isRunning();

    ECDLCpuContext(
                       unsigned int threads,
                       unsigned int pointsPerThread,
                       ECDLPParams *params,
                       BigInteger *rx,
                       BigInteger *ry,
                       int rPoints,
                       void (*callback)(struct CallbackParameters *)
                      );

    virtual bool benchmark(unsigned long long *pointsPerSecond);
    ~ECDLCpuContext();
};

#endif
