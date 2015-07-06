#ifndef _FP_X86_H
#define _FP_X86_H

#ifdef __cplusplus
extern "C" {
#endif

void x86_sub64(const unsigned long *a, const unsigned long *b, unsigned long *diff);
void x86_sub96(const unsigned long *a, const unsigned long *b, unsigned long *diff);
void x86_sub128(const unsigned long *a, const unsigned long *b, unsigned long *diff);
void x86_sub160(const unsigned long *a, const unsigned long *b, unsigned long *diff);
void x86_sub192(const unsigned long *a, const unsigned long *b, unsigned long *diff);
void x86_sub224(const unsigned long *a, const unsigned long *b, unsigned long *diff);
void x86_sub256(const unsigned long *a, const unsigned long *b, unsigned long *diff);


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

#endif