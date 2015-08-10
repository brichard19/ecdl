#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include "Fp.cu"
#include <stdio.h>

#define NUM_R_POINTS 32 // Must be a power of 2
#define R_POINT_MASK (NUM_R_POINTS - 1)

/**
 * Bit mask for identifying distinguished points
 */
__constant__ unsigned int _MASK[ 2 ];

/**
 * The X coordinates of the R points
 */
__constant__ unsigned int _rx[ 10 * NUM_R_POINTS ];

/**
 * The Y coordinates of the Y points
 */
__constant__ unsigned int _ry[ 10 * NUM_R_POINTS ];

/**
 * Shared memory to hold the R points
 */
__shared__ unsigned int _shared_rx[ 10 * NUM_R_POINTS ];
__shared__ unsigned int _shared_ry[ 10 * NUM_R_POINTS ];


/**
 * Point at infinity
 */
__device__ unsigned int _pointAtInfinity[10] = { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                                                 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

template<int N> __device__ void getRX(int index, unsigned int *rx)
{
    for(int i = 0; i < N; i++) {
        rx[i] = _shared_rx[ 32 * i + index ];
    }
}

template<int N> __device__ void getRY(int index, unsigned int *ry)
{
    for(int i = 0; i < N; i++) {
        ry[ i ] = _shared_ry[ 32 * i + index ];
    }
}


/**
 * Reads constants into shared memory
 */
template<int N> __device__ void initSharedMem()
{
    if( threadIdx.x == 0) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < 32; j++) {
                _shared_rx[i * 32 + j] = _rx[ N * j + i];
                _shared_ry[i * 32 + j] = _ry[ N * j + i];
            }
        }
    }
    __syncthreads();
}


template<int N> __device__ void doMultiplication( const unsigned int *aMultiplier, const unsigned int *bMultiplier,
                                  const unsigned int *gx, const unsigned int *gy,
                                  const unsigned int *qx, const unsigned int *qy,
                                  const unsigned int *gqx, const unsigned int *gqy,
                                  unsigned int *rxAra, unsigned int *ryAra,
                                  unsigned int *diffBuf, unsigned int *chainBuf,
                                  int step, int count)
{
    unsigned int product[N] = {0};
    product[0] = 1;
    unsigned int two[N] = {0};
    two[0] = 2;

    unsigned int mask = 1 << (step % 32);
    int word = step / 32;
   
    // To compute (Px - Qx)^-1, we multiply together all the differences and then perfom
    // a single inversion. After each multiplication we need to store the product.
    for(int i = 0; i < count; i++) {
        unsigned int bpx[N];
        unsigned int x[N];
        readBigInt<N>(rxAra, i, x);
        unsigned int diff[N];

        // For point at infinity we set the difference as 2 so the math still
        // works out 
        unsigned int am = readBigIntWord<N>( aMultiplier, i, word );
        unsigned int bm = readBigIntWord<N>( bMultiplier, i, word );

        if( (am | bm) & mask == 0 || step == 0 || equalTo<N>(x, _pointAtInfinity) ) {
            memcpy(diff, two, sizeof(diff));
        } else {
            if( (am & ~bm) & mask ) {
                copy<N>(&gx[step *N], bpx);
            } else if( (~am & bm) & mask) {
                copy<N>(&qx[step *N], bpx);
            } else {
                copy<N>(&gqx[step *N], bpx);
            }
            subModP<N>(x, bpx, diff);
        }

        writeBigInt<N>(diffBuf, i, diff);

        multiplyModP<N>(product, diff, product);
        writeBigInt<N>(chainBuf, i, product);
    }

    // Compute the inverse
    unsigned int inverse[N];
    inverseModP<N>(product, inverse);

    // Multiply by the products stored perviously so that they are canceled out
    for(int i = count - 1; i >= 0; i--) {

        // Get the inverse of the last difference by multiplying the inverse of the product of all the differences
        // with the product of all but the last difference
        unsigned int invDiff[N];
        if( i >= 1 ) {
            unsigned int tmp[N];
            readBigInt<N>(chainBuf, i - 1, tmp);
            multiplyModP<N>(inverse, tmp, invDiff);

            // Cancel out the last difference
            readBigInt<N>(diffBuf, i, tmp);
            multiplyModP<N>(inverse, tmp, inverse);
        } else {
            copy<N>(inverse, invDiff);
        }
       
        unsigned int am = readBigIntWord<N>( aMultiplier, i, word );
        unsigned int bm = readBigIntWord<N>( bMultiplier, i, word );

        if( (am & mask) != 0 || (bm & mask) != 0 ) {
            unsigned int px[N];
            unsigned int py[N];
            unsigned int bpx[N];
            unsigned int bpy[N];
          
            // Select G, Q, or G+Q 
            if( (am & ~bm) & mask ) {
                copy<N>(&gx[step *N], bpx);
                copy<N>(&gy[step *N], bpy);
            } else if( (~am & bm) & mask) {
                copy<N>(&qx[step *N], bpx);
                copy<N>(&qy[step *N], bpy);
            } else {
                copy<N>(&gqx[step *N], bpx);
                copy<N>(&gqy[step *N], bpy);
            }

            // Load the current point
            readBigInt<N>(rxAra, i, px);
            readBigInt<N>(ryAra, i, py);

            if( equalTo<N>( px, _pointAtInfinity ) ) {
                writeBigInt<N>(rxAra, i, bpx);
                writeBigInt<N>(ryAra, i, bpy);
            } else {
                unsigned int s[N];
                unsigned int rx[N];
                unsigned int s2[N];

                // s = Py - Qy / Px - Py
                subModP<N>(py, bpy, s);
                multiplyModP<N>(invDiff, s, s);
                squareModP<N>(s, s2);

                // Rx = s^2 - Px - Qx
                subModP<N>(s2, px, rx);
                subModP<N>(rx, bpx, rx);

                // Ry = -Py + s(Px - Rx)
                unsigned int k[N];
                subModP<N>(px, rx, k);
                multiplyModP<N>(k, s, k);
                unsigned int ry[N];

                subModP<N>(k, py, ry);

                writeBigInt<N>(rxAra, i, rx);
                writeBigInt<N>(ryAra, i, ry);
                
            }
        }
    }
    
}

__global__ void computeProductGKernel( const unsigned int *a, const unsigned int *b,
                                       const unsigned int *gx, const unsigned int *gy,
                                       const unsigned int *qx, const unsigned int *qy,
                                       const unsigned int *gqx, const unsigned int *gqy,
                                       unsigned int *rx, unsigned int *ry,
                                       unsigned int *diffBuf, unsigned int *chainBuf,
                                       int step, int count )
{
    switch(_PWORDS) {
        case 2:
        initFp();
        initSharedMem<2>();
        doMultiplication<2>( a, b, gx, gy, qx, qy, gqx, gqy, rx, ry, diffBuf, chainBuf, step, count ); 
        break; 
        case 3:
        initFp();
        initSharedMem<3>();
        doMultiplication<3>( a, b, gx, gy, qx, qy, gqx, gqy, rx, ry, diffBuf, chainBuf, step, count ); 
        break;
        case 4:
        initFp();
        initSharedMem<4>();
        doMultiplication<4>( a, b, gx, gy, qx, qy, gqx, gqy, rx, ry, diffBuf, chainBuf, step, count ); 
        break;
        case 5:
        initFp();
        initSharedMem<5>();
        doMultiplication<5>( a, b, gx, gy, qx, qy, gqx, gqy, rx, ry, diffBuf, chainBuf, step, count ); 
        break;
        case 6:
        initFp();
        initSharedMem<6>();
        doMultiplication<6>( a, b, gx, gy, qx, qy, gqx, gqy, rx, ry, diffBuf, chainBuf, step, count ); 
        break;
        case 7:
        initFp();
        initSharedMem<7>();
        doMultiplication<7>( a, b, gx, gy, qx, qy, gqx, gqy, rx, ry, diffBuf, chainBuf, step, count ); 
        break;
        case 8:
        initFp();
        initSharedMem<8>();
        doMultiplication<8>( a, b, gx, gy, qx, qy, gqx, gqy, rx, ry, diffBuf, chainBuf, step, count ); 
        break;
    }
}

template<int N> __device__ void resetPointsFunc(unsigned int *rx, unsigned int *ry, int count)
{
    // Reset all points to the identity element
    for(int i = 0; i < count; i++) {
        switch(_PWORDS) {
            case 2:
            writeBigInt<2>( rx, i, _pointAtInfinity );
            writeBigInt<2>( ry, i, _pointAtInfinity );
            break;
            case 3:
            writeBigInt<3>( rx, i, _pointAtInfinity );
            writeBigInt<3>( ry, i, _pointAtInfinity );
            break;
            case 4:
            writeBigInt<4>( rx, i, _pointAtInfinity );
            writeBigInt<4>( ry, i, _pointAtInfinity );
            break;
            case 5:
            writeBigInt<5>( rx, i, _pointAtInfinity );
            writeBigInt<5>( ry, i, _pointAtInfinity );
            break;
            case 6:
            writeBigInt<6>( rx, i, _pointAtInfinity );
            writeBigInt<6>( ry, i, _pointAtInfinity );
            break;
            case 7:
            writeBigInt<7>( rx, i, _pointAtInfinity );
            writeBigInt<7>( ry, i, _pointAtInfinity );
            break;
            case 8:
            writeBigInt<8>( rx, i, _pointAtInfinity );
            writeBigInt<8>( ry, i, _pointAtInfinity );
            break;
        }
    }

}

/**
 * Reset points to point at infinity
 */
__global__ void resetPointsKernel( unsigned int *rx, unsigned int *ry, int count )
{
    switch(_PWORDS) {
        case 2:
        resetPointsFunc<2>(rx, ry, count);
        break;
        case 3:
        resetPointsFunc<3>(rx, ry, count);
        break;
        case 4:
        resetPointsFunc<4>(rx, ry, count);
        break;
        case 5:
        resetPointsFunc<5>(rx, ry, count);
        break;
        case 6:
        resetPointsFunc<6>(rx, ry, count);
        break;
        case 7:
        resetPointsFunc<7>(rx, ry, count);
        break;
        case 8:
        resetPointsFunc<8>(rx, ry, count);
        break;
    }
}


/**
 * Sets the number of distinguished bits to look for
 */
cudaError_t setNumDistinguishedBits(unsigned int dBits)
{
    unsigned int mask[2] = {0xffffffff, 0xffffffff};
    if(dBits > 32) {
        mask[ 1 ] >>= (32 - (dBits - 32));
    } else {
        mask[ 0 ] >>= (32 - dBits);
        mask[ 1 ] = 0;
    }
    return cudaMemcpyToSymbol(_MASK, mask, sizeof(mask), 0, cudaMemcpyHostToDevice);
}

/**
 * Set parameters for the prime field library
 */
cudaError_t setFpParameters(const unsigned int *p, unsigned int pBits, const unsigned int *m, unsigned int mBits, const unsigned int *pMinus2, const unsigned int *pTimes2, const unsigned int *pTimes3)
{
    cudaError_t cudaError = cudaSuccess;
    unsigned int pWords = (pBits + 31) / 32;
    unsigned int mWords = (mBits + 31) / 32;
    unsigned int p2Words = (pBits + 1 + 31) / 32;
    unsigned int p3Words = (pBits + 2 + 31) / 32;

    cudaError = cudaMemcpyToSymbol(_P_CONST, p, sizeof(unsigned int)*pWords, 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }
    
    cudaError = cudaMemcpyToSymbol(_PMINUS2_CONST, pMinus2, sizeof(unsigned int)*pWords, 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }
    
    cudaError = cudaMemcpyToSymbol(_M_CONST, m, sizeof(unsigned int)*mWords, 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol(_PBITS_CONST, &pBits, sizeof(pBits), 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol(_2P_CONST, pTimes2, sizeof(unsigned int) * p2Words, 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol(_3P_CONST, pTimes3, sizeof(unsigned int) * p3Words, 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol(_PWORDS, &pWords, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol(_MWORDS, &mWords, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }


    cudaError = cudaMemcpyToSymbol(_MBITS_CONST, &mBits, sizeof(mBits), 0, cudaMemcpyHostToDevice);
    
end:
    return cudaError;
}


/**
 * Initialize device parameters
 */
cudaError_t initDeviceParams(const unsigned int *p, unsigned int pBits, const unsigned int *m, unsigned int mBits, const unsigned int *pMinus2, const unsigned int *pTimes2, const unsigned int *pTimes3, unsigned int dBits)
{
    cudaError_t cudaError = setFpParameters(p, pBits, m, mBits, pMinus2, pTimes2, pTimes3);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    cudaError = setNumDistinguishedBits(dBits);

end:
    return cudaError;
}

/**
 * Copy a, b, Rx and Ry to global memory
 */
cudaError_t copyRPointsToDevice(const unsigned int *rx, const unsigned int *ry, int length, int count)
{
    cudaError_t cudaError = cudaSuccess;
    size_t size = sizeof(unsigned int) * length * count;

    cudaError = cudaMemcpyToSymbol( _rx, rx, size, 0, cudaMemcpyHostToDevice );
    if( cudaError != cudaSuccess ) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol( _ry, ry, size, 0, cudaMemcpyHostToDevice );
    if( cudaError != cudaSuccess ) {
        goto end;
    }

end:
    return cudaError;
}

cudaError_t multiplyAddG( int blocks, int threads,
                          const unsigned int *a, const unsigned int *b,
                          const unsigned int *gx, const unsigned int *gy,
                          const unsigned int *qx, const unsigned int *qy,
                          const unsigned int *gqx, const unsigned int *gqy,
                          unsigned int *rx, unsigned int *ry,
                          unsigned int *diffBuf, unsigned int *chainBuf,
                          int step, int count )
{
    computeProductGKernel<<<blocks, threads>>>(a, b, gx, gy, qx, qy, gqx, gqy, rx, ry, diffBuf, chainBuf, step, count);
    return cudaDeviceSynchronize();
}

/**
 * Reset all points to point at infinity
 */
cudaError_t resetPoints( int blocks, int threads, unsigned int *rx, unsigned int *ry, int count )
{
    resetPointsKernel<<<blocks, threads>>>( rx, ry, count );
    return cudaDeviceSynchronize();
}

template<int N> __device__ void doStep(
                            unsigned int *xAra,
                            unsigned int *yAra,
                            unsigned int *diffBuf,
                            unsigned int *chainBuf,
                            unsigned int *pointFound,
                            unsigned int *pointThreadId,
                            unsigned int *blockFlags,
                            unsigned int *flags,
                            unsigned int count) {

    // Initalize to 1
    unsigned int product[N] = {0};
    product[0] = 1;

    // Initialize shared memory constants
    initFp();
    initSharedMem<N>();

    // Reset the point found flag
    if( blockIdx.x == 0 && threadIdx.x == 0 ) {
        *pointFound = 0;
    }
     
    // Multiply differences together
    for(int i = 0; i < count; i++) {
        unsigned int x[N];
        readBigInt<N>(xAra, i, x);
        unsigned int idx = x[0] & R_POINT_MASK;

        unsigned int diff[N];
        unsigned int rx[N];
        getRX<N>(idx, rx);
        subModP<N>(x, rx, diff);

        writeBigInt<N>(diffBuf, i, diff);

        multiplyModP<N>(product, diff, product);
        writeBigInt<N>(chainBuf, i, product);
    }

    // Compute inverse
    unsigned int inverse[N];
    inverseModP<N>(product, inverse);

    // Extract inverse of the differences
    for(int i = count - 1; i >= 0; i--) {

        // Get the inverse of the last difference by multiplying the inverse of the product of all the differences
        // with the product of all but the last difference
        unsigned int invDiff[N];

        if(i >= 1) {
            unsigned int tmp[N];
            readBigInt<N>(chainBuf, i - 1, tmp);
            multiplyModP<N>(inverse, tmp, invDiff);

            // Cancel out the last difference
            readBigInt<N>(diffBuf, i, tmp);
            multiplyModP<N>(inverse, tmp, inverse);

        } else {
            copy<N>(inverse, invDiff);
        }
      
        unsigned int px[N];
        unsigned int py[N];

        readBigInt<N>(xAra, i, px);
        readBigInt<N>(yAra, i, py);

        unsigned int idx = px[0] & R_POINT_MASK;
        unsigned int s[N];
        unsigned int s2[N];

        // s^2 = (Py - Ry / Qx - Qx)^2
        unsigned int ry[N];
        getRY<N>(idx, ry);
        subModP<N>(py, ry, s);
        multiplyModP<N>(s, invDiff, s);
        squareModP<N>(s, s2);

        // Rx = s^2 - Px - Qx
        unsigned int newX[N];
        subModP<N>(s2, px, newX);

        unsigned int rx[N];
        getRX<N>(idx, rx);
        subModP<N>(newX, rx, newX);

        // Ry = -Py + s(Px - Rx)
        unsigned int k[N];
        subModP<N>(px, newX, k);
        multiplyModP<N>(k, s, k);
        unsigned int newY[N];
        subModP<N>(k, py, newY);

        // Write resul to memory
        writeBigInt<N>(xAra, i, newX);
        writeBigInt<N>(yAra, i, newY);

        // Check for distinguished point, set flag if found
        if(((newX[ 0 ] & _MASK[ 0 ]) == 0) && ((newX[ 1 ] & _MASK[ 1 ]) == 0)) {
            blockFlags[blockIdx.x] = 1;
            *pointFound = 1;
            flags[blockDim.x * blockIdx.x * count + threadIdx.x * count + i] = 1;
        }
    }
}

__global__ void doStepKernel( unsigned int *xAra,
                              unsigned int *yAra,
                              unsigned int *diffBuf,
                              unsigned int *chainBuf,
                              unsigned int *pointFound,
                              unsigned int *pointThreadId,
                              unsigned int *blockFlags,
                              unsigned int *flags,
                              unsigned int count )
{
    switch(_PWORDS) {
        case 2:
        doStep<2>(xAra, yAra, diffBuf, chainBuf, pointFound, pointThreadId, blockFlags, flags, count);
        break;
        case 3:
        doStep<3>(xAra, yAra, diffBuf, chainBuf, pointFound, pointThreadId, blockFlags, flags, count);
        break;
        case 4:
        doStep<4>(xAra, yAra, diffBuf, chainBuf, pointFound, pointThreadId, blockFlags, flags, count);
        break;
        case 5:
        doStep<5>(xAra, yAra, diffBuf, chainBuf, pointFound, pointThreadId, blockFlags, flags, count);
        break;
        case 6:
        doStep<6>(xAra, yAra, diffBuf, chainBuf, pointFound, pointThreadId, blockFlags, flags, count);
        break;
        case 7:
        doStep<7>(xAra, yAra, diffBuf, chainBuf, pointFound, pointThreadId, blockFlags, flags, count);
        break;
        case 8:
        doStep<8>(xAra, yAra, diffBuf, chainBuf, pointFound, pointThreadId, blockFlags, flags, count);
        break;
    }
}

cudaError_t doStep( int blocks,
                    int threads,
                    unsigned int *rx,
                    unsigned int *ry,
                    unsigned int *diffBuf,
                    unsigned int *chainBuf,
                    unsigned int *pointFound,
                    unsigned int *pointThreadId,
                    unsigned int *pointIndex,
                    unsigned int *flags,
                    unsigned int count )
{
    doStepKernel<<<blocks, threads>>>(rx,
                                      ry,
                                      diffBuf,
                                      chainBuf,
                                      pointFound,
                                      pointThreadId,
                                      pointIndex,
                                      flags,
                                      count );
    return cudaDeviceSynchronize();
}
