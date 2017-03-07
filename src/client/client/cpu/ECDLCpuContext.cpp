#include "ECDLCPU.h"
#include "logger.h"
#include "util.h"
#include "logger.h"
#include "RhoCPU.h"

#define BENCHMARK_ITERATIONS 10000000

ECDLCpuContext::ECDLCpuContext( unsigned int numThreads,
                                unsigned int pointsPerThread,
                                const ECDLPParams *params,
                                const BigInteger *rx,
                                const BigInteger *ry,
                                int rPoints,
                                void (*callback)(struct CallbackParameters *)
                                )
{
    _numThreads = numThreads;
    _pointsPerThread = pointsPerThread;
    _callback = callback;
    _params = *params;
    _rPoints = rPoints;
    _running = false;
 
    // Copy random walk points
    for(int i = 0; i < rPoints; i++) {
        _rx[i] = rx[i];
        _ry[i] = ry[i];
    }

    // Set up curve using parameters 
    _curve = ECCurve(_params.p, _params.n, _params.a, _params.b, _params.gx, _params.gy);
}

RhoBase *ECDLCpuContext::getRho(bool callback)
{
    return new RhoCPU(&_params, _rx, _ry, _rPoints, _pointsPerThread, callback ? _callback : NULL);
}

ECDLCpuContext::~ECDLCpuContext()
{
    _workerThreads.clear();
    _workerThreadParams.clear();
}

bool ECDLCpuContext::init()
{
    reset();

    
    for(int i = 0; i < _numThreads; i++) {
        _workerCtx.push_back(getRho());
    }
    
    return true;
}

void ECDLCpuContext::reset()
{
    _workerThreads.clear();
    _workerThreadParams.clear();
}

bool ECDLCpuContext::stop()
{
    // TODO: Protect with mutex
    _running = false;

    // Wait for threads to finish
    for(int i = 0; i < _numThreads; i++) {
        _workerThreads[i].wait();
    }

    return true;
}

bool ECDLCpuContext::run()
{
    _running = true;

    // Run the threads
    for(int i = 0; i < _numThreads; i++) {
        WorkerThreadParams params;

        params.threadId = i;
        params.instance = this;
        _workerThreadParams.push_back(params);

        _workerThreads.push_back(Thread(&ECDLCpuContext::workerThreadEntry, &_workerThreadParams[i]));
    }

    // Wait for threads to finish
    for(int i = 0; i < _numThreads; i++) {
        _workerThreads[i].wait();
    }

    return true;
}

/**
 * Entry point method for the thread
 */
void *ECDLCpuContext::workerThreadEntry(void *ptr)
{
    WorkerThreadParams *params = (WorkerThreadParams *)ptr;

    ((ECDLCpuContext *)params->instance)->workerThreadFunction(params->threadId);

    return NULL;
}

/**
 * The method where all the work is done for the thread
 */
void ECDLCpuContext::workerThreadFunction(int threadId)
{
    RhoBase *r = _workerCtx[threadId];

    while(_running) {
        r->doStep();
    }
}

void *ECDLCpuContext::benchmarkThreadEntry(void *ptr)
{
    BenchmarkThreadParams *params = (BenchmarkThreadParams *)ptr;

    ((ECDLCpuContext *)params->instance)->benchmarkThreadFunction(&params->iterationsPerSecond);

    return NULL;
}

void ECDLCpuContext::benchmarkThreadFunction(unsigned long long *iterationsPerSecond)
{
    util::Timer timer;
    timer.start();

    RhoBase *r = getRho(false);

    for(int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        r->doStep();
    }

    unsigned int t = timer.getTime();

    *iterationsPerSecond = (unsigned long long) ((double)BENCHMARK_ITERATIONS / ((double)t/1000.0));
}

bool ECDLCpuContext::isRunning()
{
    return _running;
}

bool ECDLCpuContext::benchmark(unsigned long long *pointsPerSecondOut)
{
    int numThreads = _numThreads;
    int pointsPerThread = _pointsPerThread;

    std::vector<BenchmarkThreadParams *> params;
    std::vector<Thread> threads;

    // Start threads
    for(int i = 0; i < numThreads; i++) {
        BenchmarkThreadParams *p = new BenchmarkThreadParams;

        p->threadId = i;
        p->instance = this;
        params.push_back(p);

        Thread t(benchmarkThreadEntry, p);

        threads.push_back(t);
    }

    // Wait for all threads to finish
    for(int i = 0; i < numThreads; i++) {
        threads[i].wait();
    }

    unsigned long long iterationsPerSecond = 0;
    unsigned long long pointsPerSecond = 0;
    
    for(int i = 0; i < numThreads; i++) {
        iterationsPerSecond += params[i]->iterationsPerSecond;
        pointsPerSecond += params[i]->iterationsPerSecond * _pointsPerThread;
    }

    Logger::logInfo("%lld iterations per second\n", iterationsPerSecond);
    Logger::logInfo("%lld points per second\n", pointsPerSecond);

    if(pointsPerSecondOut) {
        *pointsPerSecondOut = pointsPerSecond;
    }

    return true;
}