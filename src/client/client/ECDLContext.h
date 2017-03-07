#ifndef _ECDL_CONTEXT_H
#define _ECDL_CONTEXT_H

#define NUM_R_POINTS 32

#include <vector>
#include "BigInteger.h"
#include "ECDLPParams.h"

/**
 Parameters passed to the callback when a distinguished point is found
 */
struct CallbackParameters {
    BigInteger aStart;
    BigInteger bStart;
    BigInteger x;
    BigInteger y;
    unsigned long long length;
};

class ECDLContext {

public:
    virtual bool init() = 0;
    virtual void reset() = 0;
    virtual bool stop() = 0;
    virtual bool run() = 0;
    virtual bool isRunning() = 0;
    virtual bool benchmark(unsigned long long *pointsPerSecond) = 0;
};

#endif