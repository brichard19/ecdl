#include "ECDLCPU.h"
#include "logger.h"
#include "util.h"
#include "logger.h"

ECDLCpuContext::ECDLCpuContext( unsigned int numThreads,
                                unsigned int pointsPerThread,
                                ECDLPParams *params,
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
    _p = params->p;
    _running = false;
 
    // Copy random walk points
    Logger::logInfo("R points:");
    for(int i = 0; i < rPoints; i++) {
        _rx[i] = rx[i];
        _ry[i] = ry[i];
        Logger::logInfo("%.2x [%s,%s]", i, _rx[i].toString().c_str(), _ry[i].toString().c_str());
    }

    // Set up curve using parameters 
    _curve = ECCurve(_params.p, _params.n, _params.a, _params.b, _params.gx, _params.gy);

    // Allocate room for points
    _x = new BigInteger[_pointsPerThread * _numThreads];
    _y = new BigInteger[_pointsPerThread * _numThreads];

    // Allocate memory for multipliers for starting points
    _aStart = new BigInteger[_pointsPerThread * _numThreads];
    _bStart = new BigInteger[_pointsPerThread * _numThreads];
}

ECDLCpuContext::~ECDLCpuContext()
{
    delete[] _x;
    delete[] _y;
    delete[] _aStart;
    delete[] _bStart;

    cleanupThreadGlobals();
}

bool ECDLCpuContext::init()
{
    // Generate the starting points
    for(unsigned int i = 0; i < _numThreads * _pointsPerThread; i++) {
        BigInteger a = randomBigInteger(_params.n);
        BigInteger b = randomBigInteger(_params.n);

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

    return true;
}

bool ECDLCpuContext::stop()
{
    // TODO: Protect with mutex
    _running = false;

    return true;
}

bool ECDLCpuContext::run()
{
    _threads = new Thread[_numThreads];
    _threadParams = new WorkerThreadParams[_numThreads];

    // Run the threads
    for(int i = 0; i < _numThreads; i++) {
        _threadParams[i].threadId = i;
        _threadParams[i].running = &_running;

        _threads[i] = Thread(workerThreadFunction, &_threadParams[i]);
    }

    _running = true;

    // Wait for threads to finish
    for(int i = 0; i < _numThreads; i++) {
        _threads[i].wait();
    }

    delete[] _threads;
    delete[] _threadParams;

    return true;
}

bool ECDLCpuContext::isRunning()
{
    return _running;
}

bool ECDLCpuContext::benchmark(unsigned long long *pointsPerSecondOut)
{
    int numThreads = _numThreads;
    int pointsPerThread = _pointsPerThread;
    int iterations = 100000;

    BenchmarkThreadParams *params = new BenchmarkThreadParams[numThreads];
    Thread *threads = new Thread[numThreads];

    for(int i = 0; i < numThreads; i++) {
        params[i].threadId = i;
        params[i].iterations = iterations;
        threads[i] = Thread(benchmarkThreadFunction, &params[i]);
    }

    for(int i = 0; i < numThreads; i++) {
        threads[i].wait();
        printf("Thread %d took %d ms\n", i, params[i].t);
    }

    unsigned long long iterationsPerSecond = 0;
    unsigned long long pointsPerSecond = 0;
    for(int i = 0; i < numThreads; i++) {
        float seconds = (float)(params[i].t)/1000;
        unsigned long long threadIterations = (unsigned long long)((float)iterations / seconds);
        unsigned long long threadPoints = (iterations * pointsPerThread)/seconds;

        Logger::logInfo("thread %d: %d iterations in %.2fs (%lld iterations per second)", i, iterations, seconds, threadIterations);
       
        Logger::logInfo("%lld points per second", threadPoints);

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
}