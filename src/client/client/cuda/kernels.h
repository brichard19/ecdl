#ifndef _ECDL_CUDA_KERNELS_H
#define _ECDL_CUDA_KERNELS_H

#include <cuda_runtime.h>

#define NUM_R_POINTS 32 // Must be a power of 2
#define FIXED_R_MASK (NUM_R_POINTS-1)

cudaError_t copyMultiplesToDevice( const unsigned int *px,
                                   const unsigned int *py,
                                   const unsigned int *qx,
                                   const unsigned int *qy,
                                   const unsigned int *gqx,
                                   const unsigned int *gqy,
                                   unsigned int len,
                                   unsigned int count );

cudaError_t copyRPointsToDevice(const unsigned int *rx, const unsigned int *ry, int length, int count);

cudaError_t multiplyAddG( int blocks,
                          int threads,
                          const unsigned int *a,
                          const unsigned int *b,
                          const unsigned int *gx,
                          const unsigned int *gy,
                          const unsigned int *qx,
                          const unsigned int *qy,
                          const unsigned int *gqx,
                          const unsigned int *gqy,
                          unsigned int *rx,
                          unsigned int *ry,
                          unsigned int *diffBuf,
                          unsigned int *chainBuf,
                          int step,
                          int count );

cudaError_t resetPoints( int blocks,
                         int threads,
                         unsigned int *rx,
                         unsigned int *ry,
                         int count );

cudaError_t doStep( int blocks,
                    int threads,
                    unsigned int *rx,
                    unsigned int *ry,
                    unsigned int *diffBuf,
                    unsigned int *chainBuf,
                    unsigned int *pointFound,
                    unsigned int *pointThreadId,
                    unsigned int *blockFlags,
                    unsigned int *flags,
                    unsigned int count );

cudaError_t initDeviceParams(const unsigned int *p, unsigned int pBits, const unsigned int *m, unsigned int mBits, const unsigned int *pMinus2, const unsigned int *pTimes2, const unsigned int *pTimes3, unsigned int dBits);
#endif
