#ifndef _FP_CU
#define _FP_CU

#include "util.cu"

// Length of p in words
__constant__ unsigned int _PWORDS;

// Length of m in words
__constant__ unsigned int _MWORDS;

// p, prime modulus
__shared__ unsigned int _P[10];
__constant__ unsigned int _P_CONST[10];

// 2 * p
__shared__ unsigned int _2P[10];
__constant__ unsigned int _2P_CONST[10];

// 3 * p
__shared__ unsigned int _3P[10];
__constant__ unsigned int _3P_CONST[10];

// m = 2^2n / k
__shared__ unsigned int _M[10];
__constant__ unsigned int _M_CONST[10];

// p - 2, used in modular inverse
__shared__ unsigned int _PMINUS2[10];
__constant__ unsigned int _PMINUS2_CONST[10];

// Lenght of p in bits
__shared__ unsigned int _PBITS;
__constant__ unsigned int _PBITS_CONST;

// Length of m in bits
__shared__ unsigned int _MBITS;
__constant__ unsigned int _MBITS_CONST;

__constant__ unsigned int _NUM_POINTS;

template<int N> __device__ void add(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    // No carry in
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 0 ]) : "r"(a[ 0 ]), "r"(b[ 0 ]) );

    // Carry in and carry out    
    #pragma unroll
    for(int i = 1; i < N; i++) {
        asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ i ]) : "r"(a[ i ]), "r"(b[ i ]) );
    }
}

/**
 * Subtracts two arrays. Returns non-zero if there is a borrow
 */
template<int N> __device__ unsigned int sub(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    // No borrow in
    asm volatile( "sub.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ 0 ]) : "r"(a[ 0 ]), "r"(b[ 0 ]) );

    // Borrow in and borrow out
    #pragma unroll
    for(int i = 1; i < N; i++) {
        asm volatile( "subc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[ i ]) : "r"(a[ i ]), "r"(b[ i ]) );
    }

    // Return non-zero on borrow
    unsigned int borrow = 0;
    asm volatile( "subc.u32 %0, %1, %2;\n\t" : "=r"(borrow) : "r"(0), "r"(0));

    return borrow;
}

/**
 * Performs N x N word multiplication
 */
template<int N> __device__ void multiply(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    // Compute low 32-bits of each 64-bit product
    for(int i = 0; i < N; i++) {
        c[i] = a[0] * b[i];
        c[i+N] = 0;
    }

    // Compute high 32-bits of each 64-bit product, perform add + carry
    asm volatile( "mad.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[1]) : "r"(a[ 0 ]), "r"(b[ 0 ] ), "r"(c[1]) );

    for(int i = 1; i < N-1; i++) {
        asm volatile( "madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[i+1]) : "r"(a[ 0 ]), "r"(b[ i ]), "r"(c[i+1]));
    }

    asm volatile( "madc.hi.u32 %0, %1, %2, %3;\n\t" : "=r"(c[N]) : "r"(a[ 0 ]), "r"(b[ N-1 ]), "r"(c[N]));


    for(int i = 1; i < N; i++) {
        unsigned int t = a[i];
        asm volatile( "mad.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[i]) : "r"(t), "r"(b[0]), "r"(c[i]));

        for(int j = 1; j < N; j++) {
            asm volatile( "madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[ i + j ]) : "r"(t), "r"(b[j]),"r"(c[i+j]));
        }
        asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[ i + N ]) : "r"(c[ i + N ]), "r"(0) );
     

        asm volatile( "mad.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[i+1]) : "r"(t), "r"(b[ 0 ] ), "r"(c[i+1]) );

        for(int j = 1; j < N-1; j++) {
            asm volatile( "madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[j+i+1]) : "r"(t), "r"(b[ j ] ), "r"(c[i+j+1]) );
        }
        asm volatile( "madc.hi.u32 %0, %1, %2, %3;\n\t" : "=r"(c[i+N]) : "r"(t), "r"(b[ N-1 ] ), "r"(c[i+N]) );
    }
}

/**
 * Squares an N-word value
 */
template<int N> __device__ void square(const unsigned int *a, unsigned int *c)
{
    multiply<N>(a, a, c);
   
    /*
    
    for(int i = 0; i < 2*N; i++) {
        c[i] = 0;
    }

    unsigned int x;
    unsigned int y;

    for(int i = 0; i < N - 1; i++) {

        x = a[i];
        y = a[i + 1];
        asm volatile( "mad.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2 * i + 0 + 1]) : "r"(x), "r"(y), "r"(c[2 * i + 0 + 1]) );
        asm volatile( "madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2 * i + 0 + 2]) : "r"(x), "r"(y), "r"(c[2 * i + 0 + 2]) );
        //for(int j = i + 1; j < N; j++) {
        //for(int j = 1; j < N - i - 1; j++) {
        for(int j = 1; j < N - i; j++) {
            //y = a[i + j + 1];
            y = a[i + j + 1];
            asm volatile( "madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2 * i + j + 1]) : "r"(x), "r"(y), "r"(c[2 * i + j + 1]) );
            asm volatile( "madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2 * i + j + 2]) : "r"(x), "r"(y), "r"(c[2 * i + j + 2]) );
        }

        y = a[N-1];
        asm volatile( "madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2 * i + N + 1]) : "r"(x), "r"(y), "r"(c[2 * i + N + 1]) );
        asm volatile( "madc.hi.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2 * i + N + 2]) : "r"(x), "r"(y), "r"(c[2 * i + N + 2]) );
    }

    */
    /*
    // Multiply by 2 by adding result to itself
    asm volatile( "add.cc.u32 %0, %1, %2;\n\t" : "=r"(c[0]) : "r"(c[0]), "r"(c[0]) );
    for(int i = 1; i < (2*N)-1; i++) {
        asm volatile( "addc.cc.u32 %0, %1, %2;\n\t" : "=r"(c[i]) : "r"(c[i]), "r"(c[i]) );
    }
    asm volatile( "addc.u32 %0, %1, %2;\n\t" : "=r"(c[2*N-1]) : "r"(c[2*N-1]), "r"(c[2*N-1]) );

    // Add the square of each term
    x = a[0];
    asm volatile( "mad.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[0]) : "r"(x), "r"(x), "r"(c[0]) );
    asm volatile( "madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[1]) : "r"(x), "r"(x), "r"(c[1]) );

    for(int i = 1; i < N-1; i++) {
        x = a[i];
        asm volatile( "madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2*i]) : "r"(x), "r"(x), "r"(c[2 * i]) );
        asm volatile( "madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2*i+1]) : "r"(x), "r"(x), "r"(c[2 * i + 1]) );
    } 
   
    x = a[N-1]; 
    asm volatile( "madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2*N-2]) : "r"(x), "r"(x), "r"(c[2*N-2]) );
    asm volatile( "madc.hi.u32 %0, %1, %2, %3;\n\t" : "=r"(c[2 * N -1]) : "r"(x), "r"(x), "r"(c[2*N - 1]) );
    */
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
    #pragma unroll
    for(int i = 0; i < N; i++ ) {
        x[ i ] = ara[ _NUM_POINTS * i + idx ];
    }
}

/**
 * Retrives a single word from an integer in global memory
 */
template<int N> __device__ unsigned int readBigIntWord(const unsigned int *ara, int idx, int word)
{
    return ara[_NUM_POINTS * word + idx];
}

__device__ void writeBigInt(unsigned int *ara, int idx, const unsigned int *x, int len)
{
    for(int i = 0; i < len; i++) {
        ara[_NUM_POINTS * i + idx ] = x[ i ];
    }
}

template<int N> __device__ void writeBigInt(unsigned int *ara, int idx, const unsigned int *x)
{
    #pragma unroll
    for(int i = 0; i < N; i++) {
        ara[_NUM_POINTS * i + idx] = x[ i ];
    }
}

template<int N> __device__ unsigned int equalTo(const unsigned int *a, const unsigned int *b)
{
    for(int i = 0; i < N; i++) {
        if(a[i] != b[i]) {
            return 0;
        }
    }

    return 1;

    /*
    unsigned int result = 0xffffffff;
    for(int i = 0; i < N; i++) {
        unsigned int eq = 0;
        asm volatile("set.eq.u32.u32 %0, %1, %2;\n\t" : "=r"(eq) : "r"(a[i]), "r"(b[i]));
        result &= eq; 
    }

    return result;
    */
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
 * Branchless greater than or equal to comparison. Returns non-zero on true and zero on false
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

/**
 * Subtraction mod P
 */
template<int N> __device__ void subModP(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    unsigned int borrow = sub<N>(a, b, c);

    if(borrow) {
        add<N>(c, _P, c);
    }
}


/**
 * Barrett reduction
 */
template<int N> __device__ void reduceModP(const unsigned int *x, unsigned int *c)
{
    unsigned int xHigh[N];
    unsigned int xm[N*2];
    unsigned int q[N];
    unsigned int qp[N*2];

    // Get top N bits
    rightShift<N>(x, xHigh);
    
    // Multiply by m
    multiply<N>(xHigh, _M, xm);

    // Get the high bits of xHigh * m. 
    rightShift<N>(xm, q);

    // It is possible that m is 1 bit longer than p. If p ends on a word boundry then m will
    // be 1 word longer than p. To avoid doing an extra multiplication when doing xHigh * m
    // (because the 1 would be in the next word), add xHigh to the result after shifting
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

template<int N> __device__ void multiplyModP(const unsigned int *a, const unsigned int *b, unsigned int *c)
{
    unsigned int x[2*N];
    multiply<N>(a, b, x);
    reduceModP<N>(x, c);
}
/**
 * Square with montgomery reduction
 */
template<int N> __device__ void squareModP(const unsigned int *a, unsigned int *c)
{
    //multiplyModP<N>(a, a, c);
    unsigned int x[2*N];
    square<N>(a, x);
    reduceModP<N>(x, c);
}


/**
 * Computes multiplicative inverse of a mod P using Fermat's little theorem (x^(P-2) mod P)
 */
template<int N> __device__ void inverseModP(const unsigned int *a, unsigned int *inverse)
{
    unsigned int x[N];
    copy<N>(a, x);

    unsigned int y[N] = {0};
    y[0] = 1;
   
    for(int j = 0; j < N-1; j++) {
        unsigned int e = _PMINUS2[j];
        for(int i = 0; i < 32; i++) {
            if(e & 1) {
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