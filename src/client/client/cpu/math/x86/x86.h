#ifndef _FP_X86_H
#define _FP_X86_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 Declarations for x86 assembly routines
 */
int x86_sub64(const unsigned long *a, const unsigned long *b, unsigned long *diff);
int x86_sub96(const unsigned long *a, const unsigned long *b, unsigned long *diff);
int x86_sub128(const unsigned long *a, const unsigned long *b, unsigned long *diff);
int x86_sub160(const unsigned long *a, const unsigned long *b, unsigned long *diff);
int x86_sub192(const unsigned long *a, const unsigned long *b, unsigned long *diff);
int x86_sub224(const unsigned long *a, const unsigned long *b, unsigned long *diff);
int x86_sub256(const unsigned long *a, const unsigned long *b, unsigned long *diff);

void x86_add64(const unsigned long *a, const unsigned long *b, unsigned long *sum);
void x86_add96(const unsigned long *a, const unsigned long *b, unsigned long *sum);
void x86_add128(const unsigned long *a, const unsigned long *b, unsigned long *sum);
void x86_add160(const unsigned long *a, const unsigned long *b, unsigned long *sum);
void x86_add192(const unsigned long *a, const unsigned long *b, unsigned long *sum);
void x86_add224(const unsigned long *a, const unsigned long *b, unsigned long *sum);
void x86_add256(const unsigned long *a, const unsigned long *b, unsigned long *sum);

void x86_mul64(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul96(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul128(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul160(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul192(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul224(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul256(const unsigned long *a, const unsigned long *b, unsigned long *product);

void x86_mul64_128(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul96_192(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul128_256(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul160_320(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul192_384(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul224_448(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul256_512(const unsigned long *a, const unsigned long *b, unsigned long *product);

void x86_square64(const unsigned long *a, unsigned long *product);
void x86_square96(const unsigned long *a, unsigned long *product);
void x86_square128(const unsigned long *a, unsigned long *product);
void x86_square160(const unsigned long *a, unsigned long *product);
void x86_square192(const unsigned long *a, unsigned long *product);
void x86_square224(const unsigned long *a, unsigned long *product);
void x86_square256(const unsigned long *a, unsigned long *product);


#ifdef __cplusplus
}
#endif


template<int N> int sub(const unsigned long *a, const unsigned long *b, unsigned long *diff)
{
    switch(N) {
        case 2:
            return x86_sub64(a, b, diff);
        case 3:
            return x86_sub96(a, b, diff);
        case 4:
            return x86_sub128(a, b, diff);
        case 5:
            return x86_sub160(a, b, diff);
        case 6:
            return x86_sub192(a, b, diff);
        case 7:
            return x86_sub224(a, b, diff);
        case 8:
            return x86_sub256(a, b, diff);
        default:
            fprintf(stderr, "Error: Not compiled for %d-bit integers\n", N*32);
            exit(1);
    }
}

template< int N> void add(const unsigned long *a, const unsigned long *b, unsigned long *sum)
{

    switch(N) {
        case 2:
            x86_add64(a, b, sum);
            break;
        case 3:
            x86_add96(a, b, sum);
            break;
        case 4:
            x86_add128(a, b, sum);
            break;
        case 5:
            x86_add160(a, b, sum);
            break;
        case 6:
            x86_add192(a, b, sum);
            break;
        case 7:
            x86_add224(a, b, sum);
            break;
        case 8:
            x86_add256(a, b, sum);
            break;
        default:
            fprintf(stderr, "Error: Not compiled for %d-bit integers\n", N*32);
            exit(1);
    }
}

template<int N> void mul(const unsigned long *a, const unsigned long *b, unsigned long *product)
{
    switch(N) {
        case 2:
            x86_mul64(a, b, product);
            break;
        case 3:
            x86_mul96(a, b, product);
            break;
        case 4:
            x86_mul128(a, b, product);
            break;
        case 5:
            x86_mul160(a, b, product);
            break;
        case 6:
            x86_mul192(a, b, product);
            break;
        case 7:
            x86_mul224(a, b, product);
            break;
        case 8:
            x86_mul256(a, b, product);
            break;
        default:
            fprintf(stderr, "Error: Not compiled for %d-bit integers\n", N*32);
            exit(1);
    }
}

template<int N1, int N2> void mul(const unsigned long *a, const unsigned long *b, unsigned long *product)
{
    switch(N1) {
        case 2:
            x86_mul64_128(a, b, product);
            break;
        case 3:
            x86_mul96_192(a, b, product);
            break;
        case 4:
            x86_mul128_256(a, b, product);
            break;
        case 5:
            x86_mul160_320(a, b, product);
            break;
        case 6:
            x86_mul192_384(a, b, product);
            break;
        case 7:
            x86_mul224_448(a, b, product);
            break;
        case 8:
            x86_mul256_512(a, b, product);
            break;
        default:
            fprintf(stderr, "Error: Not compiled for %d-bit integers\n", N1*32);
            exit(1);
    }
}

template<int N> void square(const unsigned long *a, unsigned long *product)
{
    switch(N) {
        case 2:
            x86_square64(a, product);
            break;
        case 3:
            x86_square96(a, product);
            break;
        case 4:
            x86_square128(a, product);
            break;
        case 5:
            x86_square160(a, product);
            break;
        case 6:
            x86_square192(a, product);
            break;
        case 7:
            x86_square224(a, product);
            break;
        case 8:
            x86_square256(a, product);
            break;
        default:
            fprintf(stderr, "Error: Not compiled for %d-bit integers\n", N*32);
            exit(1);
    }
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