#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include "fp131.cu"
#include <stdio.h>

#define NUM_R_POINTS 32 // Must be a power of 2
#define R_POINT_MASK (NUM_R_POINTS - 1)

/**
 * G, 2G, 4G ... (2^130)G
 */
__constant__ uint160 _gMultiplesX[ 131 ];
__constant__ uint160 _gMultiplesY[ 131 ];

/**
 * Q, 2Q, 4Q ... (2^130)Q
 */
__constant__ uint160 _qMultiplesX[ 131 ];
__constant__ uint160 _qMultiplesY[ 131 ];

/**
 * Bit mask for identifying distinguished points
 */
__constant__ unsigned int _mask[ 2 ];

/**
 * The X coordinates of the R points
 */
__constant__ uint160 _rx[ NUM_R_POINTS ];

/**
 * The Y coordinates of the Y points
 */
__constant__ uint160 _ry[ NUM_R_POINTS ];

/**
 * Shared memory to hold the R points
 */
__shared__ unsigned int _shared_rx[ 5 * NUM_R_POINTS ];
__shared__ unsigned int _shared_ry[ 5 * NUM_R_POINTS ];


/**
 * Point at infinity
 */
__device__ uint160 _pointAtInfinity = { { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff } };

__device__ uint160 getRX(int index)
{
    uint160 x;

    x.v[ 0 ] = _shared_rx[ index ];
    x.v[ 1 ] = _shared_rx[ 32 + index ];
    x.v[ 2 ] = _shared_rx[ 64 + index ];
    x.v[ 3 ] = _shared_rx[ 96 + index ];
    x.v[ 4 ] = _shared_rx[ 128 + index ];

    return x;
}

__device__ uint160 getRY(int index)
{
    uint160 y;

    y.v[ 0 ] = _shared_ry[ index ];
    y.v[ 1 ] = _shared_ry[ 32 + index ];
    y.v[ 2 ] = _shared_ry[ 64 + index ];
    y.v[ 3 ] = _shared_ry[ 96 + index ];
    y.v[ 4 ] = _shared_ry[ 128 + index ];

    return y;
}


/**
 * Reads constants into shared memory
 */
__device__ void initSharedMem()
{
    if(threadIdx.x < NUM_R_POINTS) {
        _shared_rx[ threadIdx.x ] = _rx[ threadIdx.x ].v[ 0 ];
        _shared_rx[ 32 + threadIdx.x ] = _rx[ threadIdx.x ].v[ 1 ];
        _shared_rx[ 64 + threadIdx.x ] = _rx[ threadIdx.x ].v[ 2 ];
        _shared_rx[ 96 + threadIdx.x ] = _rx[ threadIdx.x ].v[ 3 ];
        _shared_rx[ 128 + threadIdx.x ] = _rx[ threadIdx.x ].v[ 4 ];

        _shared_ry[ threadIdx.x ] = _ry[ threadIdx.x ].v[ 0 ];
        _shared_ry[ 32 + threadIdx.x ] = _ry[ threadIdx.x ].v[ 1 ];
        _shared_ry[ 64 + threadIdx.x ] = _ry[ threadIdx.x ].v[ 2 ];
        _shared_ry[ 96 + threadIdx.x ] = _ry[ threadIdx.x ].v[ 3 ];
        _shared_ry[ 128 + threadIdx.x ] = _ry[ threadIdx.x ].v[ 4 ];
    }

    __syncthreads();
}

__device__ void doMultiplication( unsigned int *multiplier,
                                  uint160 *bpx, uint160 *bpy,
                                  unsigned int *rxAra, unsigned int *ryAra,
                                  unsigned int *diffBuf, unsigned int *chainBuf,
                                  int step, int count )
{
    // One, in montgomery form
    uint160 product = _ONE;
    uint160 two = { { 2, 0, 0, 0, 0 } };

    unsigned int mask = 1 << (step % 32);
    int word = step / 32;
   
    // To compute (Px - Qx)^-1, we multiply together all the differences and then perfom
    // a single inversion. After each multiplication we need to store the product.
    for(int i = 0; i < count; i++) {
        
        uint160 x = readUint160( rxAra, i );
        uint160 diff;
       
        // For point at infinity we set the difference as 2 so the math still
        // works out 
        unsigned int m = readUint160Word( multiplier, i, word );
        if( (m & mask) == 0 || x.v[ 4 ] == 0xffffffff ) {
            diff = two;
        } else {
            diff = subModP( x, bpx[ step ] );
        }
        diff = subModP( x, bpx[ step ] );

        writeUint160( diffBuf, i, diff );
        product = multiplyMontgomery( product, diff );
        writeUint160( chainBuf, i, product );
    }

    // Compute the inverse
    uint160 inverse = inverseModPMontgomery( product );

    // Multiply by the products stored perviously so that they are canceled out
    for(int i = count - 1; i >= 0; i--) {

        // Get the inverse of the last difference by multiplying the inverse of the product of all the differences
        // with the product of all but the last difference
        uint160 invDiff;
        if( i >= 1 ) {
            invDiff = multiplyMontgomery( inverse, readUint160( chainBuf, i - 1 ) );
            // Cancel out the last difference
            inverse = multiplyMontgomery( inverse, readUint160( diffBuf, i ) );
        } else {
            invDiff = inverse;
        }
       
        unsigned int m = readUint160Word( multiplier, i, word );

        if( (m & mask) != 0 ) {
            uint160 px = readUint160( rxAra, i );
            uint160 py = readUint160( ryAra, i );
            if( equalTo( px, _pointAtInfinity ) ) {
                writeUint160( rxAra, i, bpx[ step ] );
                writeUint160( ryAra, i, bpy[ step ] );
            } else {

                uint160 s = multiplyMontgomery( invDiff, subModP( py, bpy[ step ] ) );
                uint160 s2 = squareMontgomery( s );

                // Rx = s^2 - Px - Qx
                uint160 rx = subModP( s2, px );
                rx = subModP( rx, bpx[ step ] );

                // Ry = -Py + s(Px - Rx)
                uint160 k = subModP( px, rx );
                k = multiplyMontgomery( k, s );
                uint160 ry = subModP( k, py );

                writeUint160( rxAra, i, rx );
                writeUint160( ryAra, i, ry );
            }
        }
    }
}

__global__ void computeProductGKernel( unsigned int *multiplier,
                                       unsigned int *rx, unsigned int *ry,
                                       unsigned int *diffBuf, unsigned int *chainBuf,
                                       int step, int count )
{
    initFP131();
    initSharedMem();
    doMultiplication( multiplier, _gMultiplesX, _gMultiplesY, rx, ry, diffBuf, chainBuf, step, count ); 
}

__global__ void computeProductQKernel( unsigned int *multiplier,
                                       unsigned int *rx, unsigned int *ry,
                                       unsigned int *diffBuf, unsigned int *chainBuf,
                                       int step, int count )
{
    initFP131();
    initSharedMem();
    doMultiplication( multiplier, _qMultiplesX, _qMultiplesY, rx, ry, diffBuf, chainBuf, step, count ); 
}


/**
 * Reset points to point at infinity
 */
__global__ void resetPointsKernel( unsigned int *rx, unsigned int *ry, int count )
{
    // Reset all points to the identity element
    for(int i = 0; i < count; i++) {
        writeUint160( rx, i, _pointAtInfinity );
        writeUint160( ry, i, _pointAtInfinity );
    }
}

/**
 * Copies G, 2G, 4G ... (2^131)P and Q, 2Q, 4Q ... (2^131)Q to constant memory
 */
cudaError_t copyMultiplesToDevice( const uint160 *px, const uint160 *py, const uint160 *qx, const uint160 *qy )
{
    cudaError_t cudaError = cudaSuccess;
    size_t size = sizeof(uint160) * 131;

    cudaError = cudaMemcpyToSymbol( _gMultiplesX, px, size, 0, cudaMemcpyHostToDevice );
    if( cudaError != cudaSuccess ) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol( _gMultiplesY, py, size, 0, cudaMemcpyHostToDevice );
    if( cudaError != cudaSuccess ) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol( _qMultiplesX, qx, size, 0, cudaMemcpyHostToDevice );
    if( cudaError != cudaSuccess ) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol( _qMultiplesY, qy, size, 0, cudaMemcpyHostToDevice );

end:
    return cudaError;
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

    return cudaMemcpyToSymbol(_mask, mask, sizeof(mask), 0, cudaMemcpyHostToDevice);
}

/**
 * Set parameters for the prime field library
 */
cudaError_t setFpParameters(uint160 p, uint160 pInv, uint160 pMinus2, uint160 one, unsigned int rBits)
{
    cudaError_t cudaError = cudaSuccess;

    cudaError = cudaMemcpyToSymbol(_P131, &p, sizeof(p), 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }
    
    cudaError = cudaMemcpyToSymbol(_P131MINUS2, &pMinus2, sizeof(pMinus2), 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }
    
    cudaError = cudaMemcpyToSymbol(_ONE, &one, sizeof(one), 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }

    cudaError = cudaMemcpyToSymbol(_P131INVERSE, &pInv, sizeof(pInv), 0, cudaMemcpyHostToDevice);
    if(cudaError != cudaSuccess) {
        goto end;
    }
    cudaError = cudaMemcpyToSymbol(_RBITS, &rBits, sizeof(rBits), 0, cudaMemcpyHostToDevice);

end:
    return cudaError;
}


/**
 * Initialize device parameters
 */
cudaError_t initDeviceParams(uint160 p, uint160 pInv, uint160 pMinus2, uint160 one, unsigned int rBits, unsigned int dBits)
{
    cudaError_t cudaError = setFpParameters(p, pInv, pMinus2, one, rBits);
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
cudaError_t copyRPointsToDevice(const uint160 *rx, const uint160 *ry, int count)
{
    cudaError_t cudaError = cudaSuccess;
    size_t size = sizeof(uint160) * count;

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
                          unsigned int *multiplier,
                          unsigned int *rx, unsigned int *ry,
                          unsigned int *diffBuf, unsigned int *chainBuf,
                          int step, int count )
{
    computeProductGKernel<<<blocks, threads>>>( multiplier, rx, ry, diffBuf, chainBuf, step, count );
    return cudaDeviceSynchronize();
}

cudaError_t multiplyAddQ( int blocks, int threads,
                          unsigned int *multiplier,
                          unsigned int *rx, unsigned int *ry,
                          unsigned int *diffBuf, unsigned int *chainBuf,
                          int step, int count )
{
    computeProductQKernel<<<blocks, threads>>>( multiplier, rx, ry, diffBuf, chainBuf, step, count );
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
    // One, in montgomery form
    uint160 product = _ONE;

    // Initialize shared memory constants
    initFP131();
    initSharedMem();

    // Reset the point found flag
    if( blockIdx.x == 0 && threadIdx.x == 0 ) {
        *pointFound = 0;
    }
     
    // Multiply differences together
    for(int i = 0; i < count; i++) {
        uint160 x = readUint160(xAra, i);

        unsigned int idx = x.v[ 0 ] & R_POINT_MASK;

        uint160 diff = subModP(x, getRX(idx));
        writeUint160(diffBuf, i, diff);

        product = multiplyMontgomery(product, diff);
        writeUint160(chainBuf, i, product);
    }

    // Compute inverse
    uint160 inverse = inverseModPMontgomery( product );

    // Extract inverse of the differences
    for(int i = count - 1; i >= 0; i--) {

        // Get the inverse of the last difference by multiplying the inverse of the product of all the differences
        // with the product of all but the last difference
        uint160 invDiff;
        if(i >= 1) {
            invDiff = multiplyMontgomery(inverse, readUint160(chainBuf, i - 1));
            // Cancel out the last difference
            inverse = multiplyMontgomery(inverse, readUint160(diffBuf, i));
        } else {
            invDiff = inverse;
        }
        
        uint160 px = readUint160(xAra, i);
        uint160 py = readUint160(yAra, i);

        unsigned int idx = px.v[ 0 ] & R_POINT_MASK;
        uint160 s = multiplyMontgomery(invDiff, subModP(py, getRY(idx)));
        uint160 s2 = squareMontgomery(s);

        // Rx = s^2 - Px - Qx
        uint160 newX = subModP(s2, px);
        newX = subModP(newX, getRX(idx));

        // Ry = -Py + s(Px - Rx)
        uint160 k = subModP(px, newX);
        k = multiplyMontgomery(k, s);
        uint160 newY = subModP(k, py);

        // Write resul tto memory
        writeUint160(xAra, i, newX);
        writeUint160(yAra, i, newY);

        //Check for distinguished X coordinate, set flag if found
        if(((newX.v[ 0 ] & _mask[ 0 ]) == 0) && ((newX.v[ 1 ] & _mask[ 1 ]) == 0)) {
            blockFlags[blockIdx.x] = 1;
            *pointFound = 1;
            flags[blockDim.x * blockIdx.x * count + threadIdx.x * count + i] = 1;
        }
    }
}

__global__ void multiplyTestKernel(uint160 *aPtr, uint160 *bPtr, uint160 *rPtr)
{
    initFP131();
    initSharedMem();

    uint160 a = *aPtr;
    uint160 b = *bPtr;

    uint160 r = multiplyMontgomery(a, b);
    *rPtr = r;
}

cudaError_t multiplicationTest(uint160 *aPtr, uint160 *bPtr, uint160 *rPtr)
{
    multiplyTestKernel<<<1, 1>>>(aPtr, bPtr, rPtr);
    return cudaDeviceSynchronize();
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
