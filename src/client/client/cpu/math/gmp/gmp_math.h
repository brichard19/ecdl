#ifndef _GMP_MATH_H
#define _GMP_MATH_H
#include <gmp.h>

template<int N> int sub(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    return mpn_sub_n((long unsigned int *)diff, (const long unsigned int *)a, (const long unsigned int *)b, N);
}

template< int N> void add(const unsigned long *a, const unsigned long *b, unsigned long *sum)
{
    mpn_add_n((long unsigned int *)sum, (const long unsigned int *)a, (const long unsigned int *)b, N);
}

template<int N> void mul(const unsigned long *a, const unsigned long *b, unsigned long *product)
{
    mpn_mul_n((long unsigned int *)product, (const long unsigned int *)a, (const long unsigned int *)b, N);
}

template<int N1, int N2> void mul(const unsigned long *a, const unsigned long *b, unsigned long *product)
{
    mpn_mul((long unsigned int *)product, (const long unsigned int *)a, N1, (const long unsigned int *)b, N2);
}

template<int N> void square(const unsigned long *a, unsigned long *product)
{
    mpn_sqr((long unsigned int*)product, (long unsigned int *)a, N);
}

/**
 * Returns true if a >= b
 */
template<int N> bool greaterThanEqualTo(const unsigned long *a, const unsigned long *b)
{
    for(int i = N - 1; i >= 0; i--) {
        if(a[i] < b[i]) {
            return false;
        } else if(a[i] > b[i]) {
            return true;
        }
    }

    return true;
}

#endif