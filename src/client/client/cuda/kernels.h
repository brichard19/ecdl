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

cudaError_t multiplyAddG( unsigned int blocks,
                          unsigned int threads,
                          unsigned int pointsPerThread,
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
                          unsigned int step);

cudaError_t resetPoints( unsigned int blocks,
                         unsigned int threads,
                         unsigned int pointsPerThread,
                         unsigned int *rx,
                         unsigned int *ry);

cudaError_t cudaDoStep( int pLen,
                    int blocks,
                    int threads,
                    int pointsPerThread,
                    unsigned int *rx,
                    unsigned int *ry,
                    unsigned int *diffBuf,
                    unsigned int *chainBuf,
                    unsigned int *blockFlags,
                    unsigned int *pointFlags);

cudaError_t initDeviceParams(const unsigned int *p, unsigned int pBits, const unsigned int *m, unsigned int mBits, unsigned int dBits);

cudaError_t initDeviceConstants(unsigned int numPoints);

#endif
