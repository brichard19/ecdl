#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "logger.h"
#include "ECDLCuda.h"
#include "ecc.h"
#include "BigInteger.h"
#include "util.h"


ECDLCudaContext::ECDLCudaContext( int device,
                   unsigned int blocks,
                   unsigned int threads,
                   unsigned int totalPoints,
                   unsigned int pointsInParallel,
                   const ECDLPParams *params,
                   const BigInteger *rx,
                   const BigInteger *ry,
                   int rPoints,
                   void (*callback)(struct CallbackParameters *))
{
    _device = device;
    _blocks = blocks;
    _threads = threads;
    _totalPoints = totalPoints;
    _pointsInParallel = pointsInParallel;
    _params = *params;

    for(int i = 0; i < rPoints; i++) {
        _rx[i] = rx[i];
        _ry[i] = ry[i];
    }

    _rPoints = rPoints;
    _callback = callback;

    _rho = NULL;
}

ECDLCudaContext::~ECDLCudaContext()
{
    delete _rho;
}

bool ECDLCudaContext::init()
{
    _rho = new RhoCUDA(_device, _blocks, _threads, _totalPoints, _pointsInParallel, &_params, _rx, _ry, _rPoints, _callback);
    return true;
}

bool ECDLCudaContext::run()
{
    return _rho->run();
}

void ECDLCudaContext::reset()
{
    _rho->reset();
}

bool ECDLCudaContext::stop()
{
    _rho->stop();

    return true;
}

bool ECDLCudaContext::isRunning()
{
    return _rho->isRunning();
}

bool ECDLCudaContext::benchmark(unsigned long long *pointsPerSecond)
{
    if (_rho != NULL) {
        throw std::string("Cannot run benchmark. GPU is currently busy");
    }

    RhoCUDA *r = new RhoCUDA(_device, _blocks, _threads, _totalPoints, _pointsInParallel, &_params, _rx, _ry, _rPoints, _callback);

    r->init();
    r->benchmark(pointsPerSecond);

    delete r;

    return true;
}