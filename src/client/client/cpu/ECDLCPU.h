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
    ECDLCpuContext *context;
    int threadId;
}WorkerThreadParams;

typedef struct {
    ECDLCpuContext *instance;
    int threadId;
}BenchmarkThreadParams;



class ECDLCpuContext : public ECDLContext {

private:
        // Problem parameters
    ECDLPParams _params;
    ECCurve _curve;

    // Number of threads
    int _numThreads;

    // Thread handles
    std::vector<Thread> _threads;

    std::vector<WorkerThreadParams> _threadParams;

    // Workers for each thread
    std::vector<RhoBase *> _workers;

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

    void workerThreadFunction(int threadId);

    RhoBase *getRho();

public:

    virtual bool init();
    virtual bool run();
    virtual bool stop();
    virtual bool isRunning();

    ECDLCpuContext(
                   unsigned int threads,
                   unsigned int pointsPerThread,
                   const ECDLPParams *params,
                   BigInteger *rx,
                   BigInteger *ry,
                   int rPoints,
                   void (*callback)(struct CallbackParameters *)
                  );

    virtual bool benchmark(unsigned long long *pointsPerSecond);
    virtual ~ECDLCpuContext();
};

#endif
