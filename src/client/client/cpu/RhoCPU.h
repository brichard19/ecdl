#ifndef _RHO_CPU_H
#define _RHO_CPU_H

#include "ecc.h"
#include "Fp.h"
#include "ECDLContext.h"

//#define N 8
#define FP_MAX 8

class RhoBase {

public:
    virtual void doStep() = 0;
};


class RhoCPU : public RhoBase {

private:

    ECPoint _g;
    ECPoint _q;
    ECDLPParams _params;
    ECCurve _curve;

    // Starting G and Q coefficients
    BigInteger *_a;
    BigInteger *_b;

    // Current X and Y coordinates
    unsigned long *_x;
    unsigned long *_y;

    // R points
    unsigned long *_rx;
    unsigned long *_ry;

    // Buffers for simultaneous inversion
    unsigned long *_diffBuf;
    unsigned long *_chainBuf;

    // Length of each walk
    unsigned long long *_lengthBuf;

    unsigned int _pointsInParallel;
    unsigned int _rPointMask;
    unsigned long _dBitsMask;
    unsigned int _pLen;

    FpBase *_fp;

    void (*_callback)(struct CallbackParameters *);

    void generateStartingPoint(BigInteger &x, BigInteger &y, BigInteger &a, BigInteger &b);
    bool checkDistinguishedBits(const unsigned long *x);

    void doStepSingle();
    void doStepMulti();

public:
    RhoCPU(const ECDLPParams *params,
                    const BigInteger *rx,
                    const BigInteger *ry,
                    int numRPoints,
                    int numPoints,
                    void (*callback)(struct CallbackParameters *)
                    );
    virtual ~RhoCPU();

    virtual void doStep();
};

#endif