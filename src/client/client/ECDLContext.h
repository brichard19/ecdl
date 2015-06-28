#ifndef _ECDL_CONTEXT_H
#define _ECDL_CONTEXT_H

class ECDLContext {

public:
    virtual bool init() = 0;
    virtual bool stop() = 0;
    virtual bool run() = 0;
    virtual bool isRunning() = 0;
    virtual bool benchmark(unsigned long long *pointsPerSecond) = 0;
};

#endif