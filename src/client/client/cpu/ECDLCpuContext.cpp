#include "ECDLCPU.h"
#include "logger.h"
#include "util.h"
#include "logger.h"
#include "RhoCPU.h"

ECDLCpuContext::ECDLCpuContext( unsigned int numThreads,
                                unsigned int pointsPerThread,
                                const ECDLPParams *params,
                                BigInteger *rx,
                                BigInteger *ry,
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

RhoBase *ECDLCpuContext::getRho()
{
    int pLen = _params.p.getWordLength();
    //int pLen = _params.p.getByteLength();

    switch(pLen) {
        case 1:
            return new RhoCPU<1>(&_params, _rx, _ry, _rPoints, _pointsPerThread, _callback);
        case 2:
            return new RhoCPU<2>(&_params, _rx, _ry, _rPoints, _pointsPerThread, _callback);
        case 3:
            return new RhoCPU<3>(&_params, _rx, _ry, _rPoints, _pointsPerThread, _callback);
        case 4:
            return new RhoCPU<4>(&_params, _rx, _ry, _rPoints, _pointsPerThread, _callback);
        case 5:
            return new RhoCPU<5>(&_params, _rx, _ry, _rPoints, _pointsPerThread, _callback);
        case 6:
            return new RhoCPU<6>(&_params, _rx, _ry, _rPoints, _pointsPerThread, _callback);
        case 7:
            return new RhoCPU<7>(&_params, _rx, _ry, _rPoints, _pointsPerThread, _callback);
        case 8:
            return new RhoCPU<8>(&_params, _rx, _ry, _rPoints, _pointsPerThread, _callback);
        default:
            throw "COMPILE WITH LARGER INTEGER SUPPORT";
    }
}

ECDLCpuContext::~ECDLCpuContext()
{
}

bool ECDLCpuContext::init()
{
    for(int i = 0; i < _numThreads; i++) {
        _workers.push_back(getRho());
    }
    /*
    // Generate the starting points
    for(unsigned int i = 0; i < _numThreads * _pointsPerThread; i++) {
        BigInteger a = randomBigInteger(2, _params.n);
        BigInteger b = randomBigInteger(2, _params.n);

        ECPoint g = ECPoint(_params.gx, _params.gy);
        ECPoint q = ECPoint(_params.qx, _params.qy);

        ECPoint p1 = _curve.multiplyPoint(a, g);
        ECPoint p2 = _curve.multiplyPoint(b, q);

        ECPoint p3 = _curve.addPoint(p1, p2);

        _aStart[i] = a;
        _bStart[i] = b;
        _x[i] = p3.getX();
        _y[i] = p3.getY();
    }

    initThreadGlobals(&_params, _rx, _ry, 32, _aStart, _bStart, _x, _y, _numThreads, _pointsPerThread, _params.dBits, _callback);

    */
    return true;
}

bool ECDLCpuContext::stop()
{
    // TODO: Protect with mutex
    _running = false;

    // Wait for threads to finish
    for(int i = 0; i < _numThreads; i++) {
        _threads[i].wait();
    }

    _threads.clear();
    _threadParams.clear();

    return true;
}

bool ECDLCpuContext::run()
{
    _running = true;
    // Run the threads
    for(int i = 0; i < _numThreads; i++) {
        printf("Starting thread  %d\n", i);
        WorkerThreadParams params;

        params.threadId = i;
        params.context = this;
        _threadParams.push_back(params);

        _threads.push_back(Thread(&ECDLCpuContext::workerThreadEntry, &_threadParams[i]));
    }

    /*
    // Wait for threads to finish
    for(int i = 0; i < _numThreads; i++) {
        _threads[i].wait();
    }

    delete[] _threads;
    delete[] _threadParams;

    */
    for(int i = 0; i < _numThreads; i++) {
        _threads[i].wait();
    }

    return true;
}

/**
 * Entry point method for the thread
 */
void *ECDLCpuContext::workerThreadEntry(void *ptr)
{
    printf("workerThreadEntry()\n");
    WorkerThreadParams *params = (WorkerThreadParams *)ptr;

    ((ECDLCpuContext *)params->context)->workerThreadFunction(params->threadId);

    return NULL;
}

/**
 * The method where all the work is done for the thread
 */
void ECDLCpuContext::workerThreadFunction(int threadId)
{
    printf("Worker thread function()\n");
    RhoBase *r = _workers[threadId];

    while(_running) {
        r->doStep();
    }
}

bool ECDLCpuContext::isRunning()
{
    return _running;
}

bool ECDLCpuContext::benchmark(unsigned long long *pointsPerSecondOut)
{
    /*
    int numThreads = _numThreads;
    int pointsPerThread = _pointsPerThread;
    int iterations = 1000000;

    BenchmarkThreadParams *params = new BenchmarkThreadParams[numThreads];
    Thread *threads = new Thread[numThreads];

    // Start threads
    for(int i = 0; i < numThreads; i++) {
        params[i].threadId = i;
        params[i].iterations = iterations;
        threads[i] = Thread(benchmarkThreadFunction, &params[i]);
    }

    // Wait for all threads to finish
    for(int i = 0; i < numThreads; i++) {
        threads[i].wait();
    }

    unsigned long long iterationsPerSecond = 0;
    unsigned long long pointsPerSecond = 0;
    
    for(int i = 0; i < numThreads; i++) {
        float seconds = (float)(params[i].t)/1000;
        unsigned long long threadIterations = (unsigned long long)((float)iterations / seconds);
        unsigned long long threadPoints = (iterations * pointsPerThread)/seconds;

        iterationsPerSecond += threadIterations;
        pointsPerSecond += threadPoints;
    }

    printf("%lld iterations per second\n", iterationsPerSecond);
    printf("%lld points per second\n", pointsPerSecond);

    if(pointsPerSecondOut) {
        *pointsPerSecondOut = pointsPerSecond;
    }

    delete[] threads;
    delete[] params;

    return true;
    */

    return true;
}