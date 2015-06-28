#ifndef _FP131_CU
#define _FP131_CU

/**
 * Represents a 160-bit integer, little endian
 */
typedef struct {
    unsigned int v[ 5 ];
}uint160;

/**
 * Prime modulus for field
 */
__constant__ uint160 _P131;

/**
 * Multiplicative inverse of P131 mod 2**131
 */
__constant__ uint160 _P131INVERSE;
__constant__ uint160 _P131MINUS2;
__constant__ uint160 _ONE;
__constant__ unsigned int _RBITS;

__shared__ uint160 P131;
__shared__ uint160 P131INVERSE;
__shared__ uint160 P131MINUS2;
__shared__ uint160 ONE;
__shared__ unsigned int RMASK;

/**
 * Read constants into shared memory
 */
__device__ void initFP131()
{
    if( threadIdx.x == 0 ) {
        P131 = _P131;
        P131INVERSE = _P131INVERSE;
        P131MINUS2 = _P131MINUS2;
        ONE = _ONE;
        RMASK = (0x01 << ((_RBITS % 32)-1))-1; // TODO: Is this correct?
    }
    __syncthreads();
}

/**
 * Retrives a single word from a 160-bit integer in global memory
 */
__device__ unsigned int readUint160Word(unsigned int *ara, int i, int word)
{
    unsigned int numThreads = gridDim.x * blockDim.x;
    unsigned int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    return ara[5 * numThreads * i + numThreads * word + threadIndex];
}

/**
 * Retrieves 160-bit integer from global memory
 */
__device__ uint160 readUint160(const unsigned int *ara, int i)
{
    uint160 x;
    unsigned int numThreads = gridDim.x * blockDim.x;
    unsigned int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    
    x.v[ 0 ] = ara[ 5 * numThreads * i + numThreads*0 + threadIndex ];
    x.v[ 1 ] = ara[ 5 * numThreads * i + numThreads*1 + threadIndex ];
    x.v[ 2 ] = ara[ 5 * numThreads * i + numThreads*2 + threadIndex ];
    x.v[ 3 ] = ara[ 5 * numThreads * i + numThreads*3 + threadIndex ];
    x.v[ 4 ] = ara[ 5 * numThreads * i + numThreads*4 + threadIndex ];

    return x;
}

/**
 *Writes a 160-bit intger to global memory
 */
__device__ void writeUint160(unsigned int *ara, int i, uint160 &x)
{
    unsigned int numThreads = gridDim.x * blockDim.x;
    unsigned int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    ara[ 5 * numThreads * i + threadIndex ] = x.v[ 0 ];
    ara[ 5 * numThreads * i + numThreads + threadIndex ] = x.v[ 1 ];
    ara[ 5 * numThreads * i + numThreads*2 + threadIndex ] = x.v[ 2 ];
    ara[ 5 * numThreads * i + numThreads*3 + threadIndex ] = x.v[ 3 ];
    ara[ 5 * numThreads * i + numThreads*4 + threadIndex ] = x.v[ 4 ];
}

/**
 * Computes a * b mod R
 */
__device__ void mulModR(unsigned int *a, unsigned int *b, unsigned int *c)
{
    unsigned int high = 0;
    unsigned int low = 0;

    // b * a[ 0 ]
    c[ 0 ] = a[ 0 ] * b[ 0 ];
    c[ 1 ] = a[ 0 ] * b[ 1 ];
    c[ 2 ] = a[ 0 ] * b[ 2 ];
    c[ 3 ] = a[ 0 ] * b[ 3 ];
    c[ 4 ] = a[ 0 ] * b[ 4 ];

    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 0 ]) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(c[ 1 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 1 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 2 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 3 ]) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );

    // b * a[ 1 ]
    low = a[ 1 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(c[ 1 ]), "r"(low) );
    low = a[ 1 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(low) );
    low = a[ 1 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 1 ] * b[ 3 ];
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );

    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );

    // b * a[ 2 ]
    low = a[ 2 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(low) );
    low = a[ 2 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 2 ] * b[ 2 ];
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );

    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );

    // b * a[ 3 ]
    low = a[ 3 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 3 ] * b[ 1 ];
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );

    // b * a[ 4 ]
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    
    // b * a[ 4 ]
    c[ 4 ] += a[ 4 ] * b[ 0 ];

    // mod R
    //c[ 4 ] &= 0x07;
    //c[ 4 ] &= ((0x01 << ((_RBITS % 32)-1))-1);
    c[4] &= RMASK;
}

/**
 * 160-bit compare equal to
 */
__device__ bool equalTo(const uint160 &a, const uint160 &b)
{
    for( int i = 0; i < 5; i++ ) {
        if( a.v[ i ] != b.v[ i ] ) {
            return false;
        }
    }

    return true;
}

/**
 * 160-bit addition
 */
__device__ uint160 add160(const uint160 &a, const uint160 &b)
{
    uint160 c;

    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 0 ]) : "r"(a.v[ 0 ]), "r"(b.v[ 0 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 1 ]) : "r"(a.v[ 1 ]), "r"(b.v[ 1 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 2 ]) : "r"(a.v[ 2 ]), "r"(b.v[ 2 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 3 ]) : "r"(a.v[ 3 ]), "r"(b.v[ 3 ]) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 4 ]) : "r"(a.v[ 4 ]), "r"(b.v[ 4 ]) );
    
    return c;
}

/**
 * 160-bit addition
 */
__device__ void add160(unsigned int *a, unsigned int *b, unsigned int *c)
{
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 0 ]) : "r"(a[ 0 ]), "r"(b[ 0 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(a[ 1 ]), "r"(b[ 1 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(a[ 2 ]), "r"(b[ 2 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(a[ 3 ]), "r"(b[ 3 ]) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(a[ 4 ]), "r"(b[ 4 ]) );
}


/**
 * 160-bit subtraction
 */
__device__ uint160 sub160(const uint160 &a, const uint160 &b)
{
    uint160 c;
    asm volatile( "sub.cc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 0 ]) : "r"(a.v[ 0 ]), "r"(b.v[ 0 ]) );
    asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 1 ]) : "r"(a.v[ 1 ]), "r"(b.v[ 1 ]) );
    asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 2 ]) : "r"(a.v[ 2 ]), "r"(b.v[ 2 ]) );
    asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 3 ]) : "r"(a.v[ 3 ]), "r"(b.v[ 3 ]) );
    asm volatile( "subc.u32 %0, %1, %2;\n\t" : "=r"(c.v[ 4 ]) : "r"(a.v[ 4 ]), "r"(b.v[ 4 ]) );

    return c;
}

/**
 * 160-bit subtraction
 */
__device__ void sub160(unsigned int *a, unsigned int *b, unsigned int *c)
{
    asm volatile( "sub.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 0 ]) : "r"(a[ 0 ]), "r"(b[ 0 ]) );
    asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(a[ 1 ]), "r"(b[ 1 ]) );
    asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(a[ 2 ]), "r"(b[ 2 ]) );
    asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(a[ 3 ]), "r"(b[ 3 ]) );
    asm volatile( "subc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(a[ 4 ]), "r"(b[ 4 ]) );
}

/*
 * Multiplies two 160-bit integers to get a 320-bit result
 */
__device__ void mul320(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    unsigned int high = 0;
    unsigned int low = 0;

    // add b * a[ 0 ]
    c[ 0 ] = a[ 0 ] * b[ 0 ];
    c[ 1 ] = a[ 0 ] * b[ 1 ];
    c[ 2 ] = a[ 0 ] * b[ 2 ];
    c[ 3 ] = a[ 0 ] * b[ 3 ];
    c[ 4 ] = a[ 0 ] * b[ 4 ];
    c[ 5 ] = 0;
    c[ 6 ] = 0;
    c[ 7 ] = 0;
    c[ 8 ] = 0;
    c[ 9 ] = 0;

    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 0 ]) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(c[ 1 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 1 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 2 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 3 ]) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 4 ]) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );

    // add b * a[ 1 ]
    low = a[ 1 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(c[ 1 ]), "r"(low) );
    low = a[ 1 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(low) );
    low = a[ 1 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 1 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );
    low = a[ 1 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(low) );
    
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(0) );


    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(high) );

    // add b * a[ 2 ]
    low = a[ 2 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(low) );
    low = a[ 2 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 2 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );
    low = a[ 2 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(low) );
    low = a[ 2 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(low) );
    
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(0) );

    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(high) );

    // add b * a[ 3 ]
    low = a[ 3 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 3 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );
    low = a[ 3 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(low) );
    low = a[ 3 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(low) );
    low = a[ 3 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(low) );
    
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(0) );

    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(high) );

    // add b * a[ 4 ]
    low = a[ 4 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );
    low = a[ 4 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(low) );
    low = a[ 4 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(low) );
    low = a[ 4 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(low) );
    low = a[ 4 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(low) );
    
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );

    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(high) );
}

__device__ void mulAdd320( unsigned int *a, unsigned int *b, unsigned int *c )
{
    unsigned int high = 0;
    unsigned int low = 0;

    // add b * a[ 0 ] (low)
    low = a[ 0 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 0 ]) : "r"(c[ 0 ]), "r"(low) );
    low = a[ 0 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(c[ 1 ]), "r"(low) );
    low = a[ 0 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(low) );
    low = a[ 0 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 0 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );

    // extend carry to end
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(0));
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(0));
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(0));
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(0));
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0));


    // add b * a[ 0 ] (high)
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(c[ 1 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
   
    // extend carry to end
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(0) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(0) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(0) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );


    // add b * a[ 1 ] (low)
    low = a[ 1 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(c[ 1 ]), "r"(low) );
    low = a[ 1 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(low) );
    low = a[ 1 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 1 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );
    low = a[ 1 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(low) );
   
    // extend carry
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(0) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(0) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(0) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );


    // add b * a[ 1 ] (high)
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 1 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(high) );
  
    // extend carry to end
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(0) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(0) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );


    // add b * a[ 2 ] (low)
    low = a[ 2 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 2 ]) : "r"(c[ 2 ]), "r"(low) );
    low = a[ 2 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 2 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );
    low = a[ 2 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(low) );
    low = a[ 2 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(low) );
   
    // extend carry
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(0) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(0) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );


    // add b * a[ 3 ] (high)
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 2 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(high) );
   
    // extend carry
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(0) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );


    // add b * a[ 3 ] (low)
    low = a[ 3 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 3 ]) : "r"(c[ 3 ]), "r"(low) );
    low = a[ 3 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );
    low = a[ 3 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(low) );
    low = a[ 3 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(low) );
    low = a[ 3 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(low) );
   
    // extend carry
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(0) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );


    // add b * a[ 3 ] (high)
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 3 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(high) );
   
    // extend carry
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );


    // add b * a[ 4 ] (low)
    low = a[ 4 ] * b[ 0 ];
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 4 ]) : "r"(c[ 4 ]), "r"(low) );
    low = a[ 4 ] * b[ 1 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(low) );
    low = a[ 4 ] * b[ 2 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(low) );
    low = a[ 4 ] * b[ 3 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(low) );
    low = a[ 4 ] * b[ 4 ];
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(low) );
   
    // extend carry
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(0) );


    // b * a[ 4 ] (high)
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 5 ]) : "r"(c[ 5 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 1 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 6 ]) : "r"(c[ 6 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 2 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 7 ]) : "r"(c[ 7 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 3 ] ) );
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 8 ]) : "r"(c[ 8 ]), "r"(high) );
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 4 ]), "r"(b[ 4 ] ) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ 9 ]) : "r"(c[ 9 ]), "r"(high) );
}


/**
 * Multiplication with montgomery reduction
 */
__device__ uint160 multiplyMontgomery(const uint160 &a, const uint160 &b)
{
    unsigned int t[ 10 ] = {0};
    unsigned int tnModR[ 5 ] = {0};
    
    // T = A * B    
    mul320(a.v, b.v, t);

    // s = T * N' mod R
    mulModR(t, P131INVERSE.v, tnModR);

    // Multiply by P
    mulAdd320(tnModR, P131.v, t);

    // Divide by R
    uint160 product;
    int rShift = (_RBITS-1) % 32;
    int lShift = 32 - rShift;
    unsigned int mask = 0xffffffff >> lShift;
    product.v[ 0 ] = (t[ 4 ] >> rShift) | (t[ 5 ] << lShift);
    product.v[ 1 ] = (t[ 5 ] >> rShift) | (t[ 6 ] << lShift);
    product.v[ 2 ] = (t[ 6 ] >> rShift) | (t[ 7 ] << lShift);
    product.v[ 3 ] = (t[ 7 ] >> rShift) | (t[ 8 ] << lShift);
    product.v[ 4 ] = (t[ 8 ] >> rShift) | (t[ 9 ] << lShift);
    product.v[ 4 ] &= mask;

    // Reduce mod P
    uint160 p1;
    p1 = sub160( product, P131 );

    if(p1.v[ 4 ] & 0x80000000) {
        return product;
    } else {
        return p1;
    }
}

/**
 * Square with montgomery reduction
 */
__device__ uint160 squareMontgomery( const uint160 &a )
{
    return multiplyMontgomery(a, a);
}

/**
 * Addition mod P. It is assumed that a < P and b < P
 */
 /*
__device__ uint160 addModP(const uint160 &a, const uint160 &b)
{
    uint160 sum = sub160( a, P131 );
    sum = add160( sum, b );

    if( sum.v[ 4 ] & 0x80000000 ) {
        sum = add160( sum, P131 );
    }

    return sum;
}
*/

/**
 * Subtraction mod P. It is assumed a < P and b < P
 */
__device__ uint160 subModP( const uint160 &a, const uint160 &b )
{
    uint160 diff = sub160( a, b );
    
    // Add P if negative
    if( diff.v[ 4 ] & 0x80000000 ) {
        diff = add160( diff, P131 );
    }

    return diff;
}


/**
 * Computes multiplicative inverse of a mod P using Fermat's little theorem
 */
__device__ uint160 inverseModPMontgomery(const uint160 &a)
{
    // P131 - 2
    unsigned int exponent[ 5 ];
    exponent[ 0 ] = P131MINUS2.v[ 4 ];
    exponent[ 1 ] = P131MINUS2.v[ 3 ];
    exponent[ 2 ] = P131MINUS2.v[ 2 ];
    exponent[ 3 ] = P131MINUS2.v[ 1 ];
    exponent[ 4 ] = P131MINUS2.v[ 0 ];

    uint160 product = ONE;
    uint160 x = a;
    
    for( int i = 0; i < 32; i++ ) {
        if( exponent[ 4 ] & 0x01 ) {
            product = multiplyMontgomery(product, x);
        }

        x = squareMontgomery( x );
        exponent[ 4 ] >>= 1; 
    }
    
    for( int i = 0; i < 32; i++ ) {
        if( exponent[ 3 ] & 1 ) {
            product = multiplyMontgomery(product, x);
        }

        x = squareMontgomery( x );
        exponent[ 3 ] >>= 1;
    }
    
    for( int i = 0; i < 32; i++ ) {
        if( exponent[ 2 ] & 1 ) {
            product = multiplyMontgomery(product, x);
        }

        x = squareMontgomery(x);
        exponent[ 2 ] >>= 1;
    }

    for( int i = 0; i < 32; i++ ) {
        if(exponent[ 1 ] & 1) {
            product = multiplyMontgomery(product, x);
        }

        x = squareMontgomery(x);
        exponent[ 1 ] >>= 1;
    }

    while( exponent[ 0 ] ) {
        if( exponent[ 0 ] & 1 ) {
            product = multiplyMontgomery(product, x);
        }
        x = squareMontgomery(x);
        exponent[ 0 ] >>= 1;
    }

    return product;
}

#endif
