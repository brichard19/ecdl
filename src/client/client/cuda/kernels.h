#ifndef _ECDL_CUDA_KERNELS_H
#define _ECDL_CUDA_KERNELS_H

#include "uint160.h"

#include <cuda_runtime.h>

#define NUM_R_POINTS 32 // Must be a power of 2
#define FIXED_R_MASK (NUM_R_POINTS-1)

cudaError_t copyMultiplesToDevice( const uint160 *px,
                                   const uint160 *py,
                                   const uint160 *qx,
                                   const uint160 *qy );

cudaError_t copyRPointsToDevice(const uint160 *rx, const uint160 *ry, int count);

cudaError_t multiplyAddQ( int blocks,
                          int threads,
                          unsigned int *multiplier,
                          unsigned int *rx,
                          unsigned int *ry,
                          unsigned int *diffBuf,
                          unsigned int *chainBuf,
                          int step,
                          int count );

cudaError_t multiplyAddG( int blocks,
                          int threads,
                          unsigned int *multiplier,
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

cudaError_t multiplicationTest(uint160 *aPtr, uint160 *bPtr, uint160 *rPtr);

cudaError_t initDeviceParams(uint160 p, uint160 pInv, uint160 p2, uint160 one, unsigned int rBits, unsigned int dBits);
cudaError_t setNumDistinguishedBits(unsigned int bits);
cudaError_t setFpParameters(uint160 p, uint160 pInv, uint160 pMinus2, uint160 one, unsigned int rBits);

#endif
