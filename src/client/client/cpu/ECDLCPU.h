#ifndef _ECDL_CPU_H
#define _ECDL_CPU_H

#include "threads.h"
#include "BigInteger.h"
#include "ecc.h"

#include "ECDLContext.h"

#include <vector>

class ECDLCpuContext;
class RhoBase;

typedef struct {
    ECDLCpuContext *instance;
    int threadId;
}WorkerThreadParams;

typedef struct {
    ECDLCpuContext *instance;
    int threadId;
    unsigned long long iterationsPerSecond;
}BenchmarkThreadParams;



class ECDLCpuContext : public ECDLContext {

private:

    // Problem parameters
    ECDLPParams _params;
    ECCurve _curve;

    // Number of threads
    int _numThreads;

    // Workers
    std::vector<Thread> _workerThreads;
    std::vector<WorkerThreadParams> _workerThreadParams;
    std::vector<RhoBase *> _workerCtx;

    // Flag to indicate if the threads are running
    volatile bool _running;

    // Callback that gets called when distinguished point is found 
    void (*_callback)(struct CallbackParameters *);

    // Number of points each thread works on in parallel
    unsigned int _pointsPerThread;

    // R-points
    int _rPoints;
    BigInteger _rx[ NUM_R_POINTS ];
    BigInteger _ry[ NUM_R_POINTS ];

    static void *workerThreadEntry(void *ptr);
    static void *benchmarkThreadEntry(void *ptr);

    void workerThreadFunction(int threadId);
    void benchmarkThreadFunction(unsigned long long *iterationsPerSecond);

    RhoBase *getRho(bool callback = true);

public:

    virtual bool init();
    void reset();
    virtual bool run();
    virtual bool stop();
    virtual bool isRunning();
    virtual bool benchmark(unsigned long long *pointsPerSecond);

    ECDLCpuContext(
                   unsigned int threads,
                   unsigned int pointsPerThread,
                   const ECDLPParams *params,
                   const BigInteger *rx,
                   const BigInteger *ry,
                   int rPoints,
                   void (*callback)(struct CallbackParameters *)
                  );

    virtual ~ECDLCpuContext();
};

#endif
