#ifndef _UTIL_CU
#define _UTIL_CU

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

#endif