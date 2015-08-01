#ifndef _FP131_CU
#define _FP131_CU

/**
 * Prime modulus for field
 */
__constant__ unsigned int _P_CONST[10];
__constant__ unsigned int _2P_CONST[10];
__constant__ unsigned int _3P_CONST[10];
__constant__ unsigned int _M_CONST[10];
__constant__ unsigned int _PMINUS2_CONST[10];
__constant__ unsigned int _PBITS_CONST;
__constant__ unsigned int _MBITS_CONST;

__constant__ unsigned int _PWORDS;
__constant__ unsigned int _MWORDS;

__shared__ unsigned int _P[10];
__shared__ unsigned int _2P[10];
__shared__ unsigned int _3P[10];
__shared__ unsigned int _M[10];
__shared__ unsigned int _PMINUS2[10];
__shared__ unsigned int _PBITS;
__shared__ unsigned int _MBITS;

template<int N> __device__ void zero(unsigned int *x)
{
    #pragma unroll
    for(int i = 0; i < N; i++) {
        x[i] = 0;
    }
}

template<int N> __device__ void copy(const unsigned int *a, unsigned int *b)
{
    #pragma unroll
    for(int i = 0; i < N; i++) {
        b[i] = a[i];
    }
}

/**
 * Adds two arrays. Returns non-zero if there is a carry
 */
template<int N> __device__ unsigned int add(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 0 ]) : "r"(a[ 0 ]), "r"(b[ 0 ]) );
    #pragma unroll
    for(int i = 1; i < N-1; i++) {
        asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ i ]) : "r"(a[ i ]), "r"(b[ i ]) );
    }
    asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ N-1 ]) : "r"(a[ N-1 ]), "r"(b[ N-1 ]) );

    // Return non-zero on carry
    unsigned int carry = 0;
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(carry) : "r"(0), "r"(0));

    return carry;
}

/**
 * Subtracts two arrays. Returns non-zero if there is a borrow
 */
template<int N> __device__ unsigned int sub(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    asm volatile( "sub.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 0 ]) : "r"(a[ 0 ]), "r"(b[ 0 ]) );
    #pragma unroll
    for(int i = 1; i < N-1; i++) {
        asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ i ]) : "r"(a[ i ]), "r"(b[ i ]) );
    }
    asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ N-1 ]) : "r"(a[ N-1 ]), "r"(b[ N-1 ]) );

    // Return non-zero on borrow
    unsigned int borrow = 0;
    asm volatile( "subc.u32 %0, %1, %2;\n\t" : "=r"(borrow) : "r"(0), "r"(0));

    return borrow;
}

template<int N> __device__ void multiply(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    unsigned int low = 0;
    unsigned int high = 0;

    #pragma unroll
    for(int i = 0; i < N; i++) {
        c[i] = a[0] * b[i];
        c[i+N] = 0;
    }

    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ 0 ] ) );
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 1 ]) : "r"(c[ 1 ]), "r"(high) );
    #pragma unroll
    for(int i = 1; i < N-1; i++) {
        asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ i ]) );
        asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ i + 1 ]) : "r"(c[ i + 1 ]), "r"(high) );
    }
    asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ 0 ]), "r"(b[ N-1 ]) );
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ N ]) : "r"(c[ N ]), "r"(high) );


    #pragma unroll
    for(int i = 1; i < N; i++) {

        low = a[i] * b[0];
        asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ i ]) : "r"(c[ i ]), "r"(low) );
        #pragma unroll
        for(int j = 1; j < N; j++) {
            low = a[i] * b[j];
            asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ i + j ]) : "r"(c[ i + j ]), "r"(low) );
        }
        asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ i + N ]) : "r"(c[ i + N ]), "r"(0) );
      

        asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ i ]), "r"(b[ 0 ] ) );
        asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ i + 1 ]) : "r"(c[ i + 1 ]), "r"(high) );
        #pragma unroll
        for(int j = 1; j < N-1; j++) {
            asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ i ]), "r"(b[ j ] ) );
            asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ j + i + 1 ]) : "r"(c[ j + i + 1 ]), "r"(high) );
        }
        asm volatile( "mul.hi.u32 %0, %1, %2;\n\t" : "=r"(high) : "r"(a[ i ]), "r"(b[ N-1 ] ) );
        asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ i + N ]) : "r"(c[ i + N ]), "r"(high) );
    }
}


/**
 * Read constants into shared memory
 */
__device__ void initFp()
{
    if( threadIdx.x == 0 ) {
        memcpy(_P, _P_CONST, sizeof(_P_CONST));
        memcpy(_M, _M_CONST, sizeof(_M_CONST));
        memcpy(_2P, _2P_CONST, sizeof(_2P_CONST));
        memcpy(_3P, _3P_CONST, sizeof(_2P_CONST));
        memcpy(_PMINUS2, _PMINUS2_CONST, sizeof(_PMINUS2_CONST));
        _PBITS = _PBITS_CONST;
    }
    __syncthreads();
}

template<int N> __device__ void readBigInt(const unsigned int *ara, int idx, unsigned int *x)
{
    unsigned int numThreads = gridDim.x * blockDim.x;
    unsigned int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    #pragma unroll
    for(int i = 0; i < N; i++ ) {
        x[ i ] = ara[ N * numThreads * idx + numThreads * i + threadIndex ];
    }
}

/**
 * Retrives a single word from an integer in global memory
 */
template<int N> __device__ unsigned int readBigIntWord(const unsigned int *ara, int i, int word)
{
    unsigned int numThreads = gridDim.x * blockDim.x;
    unsigned int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    return ara[N * numThreads * i + numThreads * word + threadIndex];
}

template<int N> __device__ void writeBigInt(unsigned int *ara, int idx, const unsigned int *x)
{
    unsigned int numThreads = gridDim.x * blockDim.x;
    unsigned int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    #pragma unroll
    for(int i = 0; i < N; i++) {
        ara[ N * numThreads * idx + numThreads * i + threadIndex ] = x[ i ];
    }
}

template<int N> __device__ unsigned int equalTo(const unsigned int *a, const unsigned int *b)
{
    unsigned int result = 0xffffffff;
    for(int i = 0; i < N; i++) {
        unsigned int eq = 0;
        asm volatile("set.eq.u32.u32 %0, %1, %2;\n\t" : "=r"(eq) : "r"(a[i]), "r"(b[i]));
        result &= eq; 
    }

    return result;
}

template<int N> __device__ void rightShift(const unsigned int *in, unsigned int *out)
{
    int rShift = (_PBITS) % 32;
    int lShift = 32 - rShift;

    if(rShift > 0) {
        for(int i = 0; i < N; i++) {
            out[ i ] = (in[ N - 1 + i ] >> rShift) | (in[ N + i ] << lShift);
        }
    } else {
        for(int i = 0; i < N; i++) {
            out[ i ] = in[ N + i];
        }
    }
}

/**
 * Greater than or equal to comparison. Returns non-zero on true and zero on false
 */
template<int N> __device__ unsigned int greaterThanEqualTo(const unsigned int *a, const unsigned int *b)
{
    unsigned int sum = 0;
    unsigned int mask = 0xffffffff;
    for(int i = N - 1; i >= 0; i--) {
        unsigned int lt = 0;
        unsigned int eq = 0;
        unsigned int x = a[i];
        unsigned int y = b[i];
        asm volatile("set.lo.u32.u32 %0, %1, %2;\n\t" : "=r"(lt) : "r"(x), "r"(y));
        asm volatile("set.eq.u32.u32 %0, %1, %2;\n\t" : "=r"(eq) : "r"(x), "r"(y));

        sum |= lt & mask;
        mask &= eq;
    }

    return ~sum;
}

template<int N> __device__ void subModP(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    unsigned int borrow = sub<N>(a, b, c);

    if(borrow) {
        add<N>(c, _P, c);
    }
}

/**
 * Multiplication with montgomery reduction
 */
template<int N> __device__ void multiplyModP(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    unsigned int x[N*2];
    unsigned int xHigh[N];
    unsigned int xm[N*2];
    unsigned int q[N];
    unsigned int qp[N*2];

    multiply<N>(a, b, x);

    rightShift<N>(x, xHigh);
    
    // Multiply by m
    multiply<N>(xHigh, _M, xm);

    // Get the high bits of xHigh * m. 
    rightShift<N>(xm, q);

    // It is possible that m is 1 bit longer than p. If p ends on a word boundry then m will
    // be 1 word longer than p. To avoid doing an extra multiplication when doing mHigh * m
    // (because the 1 would be in a separate word), add xHigh to the result after shifting
    if(_MWORDS > _PWORDS) {
        add<N>(q, xHigh, q);
    }

    // Multiply by p
    multiply<N>(q, _P, qp);
    
    // Subtract from x
    unsigned int r[N+1];
    sub<N+1>(x, qp, r);

    // The trick here is that instead of multiplying xm by p, we multiplied only the top
    // half by p. This still works because the lower bits of the product are discarded anyway.
    // But it could have been the case that there was a carry from the multiplication operation on
    // the lower bits, which will result in r being >= 2p because in that case we would be
    // doing x - (q-1) * p instead of x - q*p. So we need to check for >= 2p and >= p. Its more checks
    // but saves us from doing a multiplication.
    unsigned int gte3p = greaterThanEqualTo<N+1>(r, _3P);
    unsigned int gte2p = greaterThanEqualTo<N+1>(r, _2P);
    unsigned int gtep = greaterThanEqualTo<N+1>(r, _P);

    if(gte3p) {
        sub<N>(r, _3P, c);
    } else if(gte2p) {
        sub<N>(r, _2P, c);
    } else if(gtep) {
        sub<N>(r, _P, c);
    } else {
        copy<N>(r, c);
    }
}

/**
 * Square with montgomery reduction
 */
template<int N> __device__ void squareModP(const unsigned int *a, unsigned int *c)
{
    multiplyModP<N>(a, a, c);
}


/**
 * Computes multiplicative inverse of a mod P using Fermat's little theorem
 */
template<int N> __device__ void inverseModP(const unsigned int *a, unsigned int *inverse)
{
    unsigned int x[N] = {0};
    copy<N>(a, x);

    unsigned int y[N] = {0};
    y[0] = 1;
   
    for(int j = 0; j < N-1; j++) {
        unsigned int e = _PMINUS2[j];
        for( int i = 0; i < 32; i++ ) {
            if( e & 1) {
                multiplyModP<N>(y, x, y);
            }

            squareModP<N>(x, x);
            e >>= 1;
        }
    }

    unsigned int e = _PMINUS2[N-1];
    while( e ) {
        if( e & 1 ) {
            multiplyModP<N>(y, x, y);
        }
        squareModP<N>(x, x);
        e >>= 1;
    }

    copy<N>(y, inverse);
}

#endif
